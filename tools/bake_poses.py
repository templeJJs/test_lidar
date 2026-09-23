#!/usr/bin/env python
"""Запечь позы скелетных клипов в `assets/_poses/**` + сводный `manifest.json`.

    python tools/bake_poses.py                # запечь всё, что ещё не запечено, и собрать манифест
    python tools/bake_poses.py --reuse        # ничего не печь, только пересобрать манифест
    python tools/bake_poses.py --only rigged_animated_humanoid

Зачем: вьюер умеет ровно один формат меша -- `{"vertices": [[x,y,z]...], "faces": [[i,j,k]...]}`
(`web/track3d.js`, `meshFromArrays`; `objects/to_viewer.py`, докстрока). Поэтому позы
приходят не как GLB с AnimationMixer, а как заранее посчитанные вершины: один момент
времени = один .npz. Их же использует синтетика лидара -- ей нужен меш, а не рендер.

Печёт `tools/bake_pose.js` (three + skinning, вне браузера). Этот скрипт только
решает, КАКИЕ моменты печь, складывает их в каталог клипа, детектирует зацикленность
и пишет манифест.

Раскладка (контракт для интегратора):

    assets/_poses/manifest.json                          -- сводка по всем ассетам
    assets/_poses/<asset>/<clip>/*.npz                   -- по одному моменту на файл
    assets/_poses/<asset>/<clip>/frames.npz              -- тот же клип одним файлом:
                                                            vertices (T,N,3) float32, faces (M,3) int32
    assets/_poses/<asset>/<clip>/clip.json               -- метаданные клипа (без вершин)

Вершины в .npz УЖЕ в метрах (баковка идёт с `--scale <calibrated_scale_to_meters>` из
`assets/manifest.json`), Y-up, начало координат под ногами в стойке. Сцена -- Z-up,
поэтому при подстановке в кадр нужно `(x, y, z) -> (x, -z, y)`; это записано в
манифесте (`placement.to_scene_axes`).

Классы разметки (`CLASS_MAP`) описаны там же: какой клип какого ассета брать, и что
делать, если подходящего клипа нет.
"""

import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
POSE_ROOT = os.path.join(ROOT, 'assets', '_poses')
ASSETS_MANIFEST = os.path.join(ROOT, 'assets', 'manifest.json')
BAKE = os.path.join(HERE, 'bake_pose.js')

# Ассеты, которые печём. Масштаб берётся из assets/manifest.json
# (units.calibrated_scale_to_meters): он уже калиброван по росту 1.75 м, второй
# ручной константы не заводим.
ASSETS = [
    ('rigged_animated_humanoid', 'assets/rigged_animated_humanoid/person_animated.fbx'),
    ('animated_human_quaternius',
     'assets/animated_human_quaternius/Animated Human by @Quaternius/FBX/Animated Human.fbx'),
]

# Сколько моментов на клип. Короткий цикл (бег 0.6 с) должен быть частым -- иначе
# линейная интерполяция между моментами срезает углы на быстрых ногах; длинная
# анимация (Idle 10 с) -- редкой, иначе файлы распухают без пользы.
SAMPLE_TABLE = (
    (0.075, 1),   # <= ~1 кадр при 30 fps: это поза, а не анимация
    (0.75, 24),
    (2.0, 18),
    (5.0, 14),
    (math.inf, 12),
)

# Приор по имени. Проверяется измерением (detect_loop); имя решает только там, где
# измерение неоднозначно (сравниваются почти неподвижные позы).
LOOP_NAMES = ('walk', 'run', 'idle', 'standing', 'swim', 'working')
NONLOOP_NAMES = ('punch', 'shoot', 'swing', 'jump', 'death', 'default', 'armatureaction')

STATIC_MAX_S = 0.075          # клип из одного кадра
LOOP_RATIO = 0.10             # endpoint_delta <= 10% размаха => цикл
NONLOOP_RATIO = 0.15          # endpoint_delta >= 15% размаха => не цикл:
                              # при склейке конца с началом рывок в 15% размаха видно
                              # (напр. Walk у Quaternius: 14.4 см при размахе 65 см)
LOOP_POP_WARN_M = 0.02        # цикл с расхождением > 2 см требует кроссфейда на склейке

SCENE_AXES = ('rotate +90 deg about X: (x, y, z)_mesh -> (x, -z, y)_scene '
              '(mesh is Y-up like the glTF/FBX it came from, the frame is Z-up)')

# --- классы разметки -> клипы --------------------------------------------------
# Числа ниже измерены (height_over_samples_m / extent_over_samples_m / loop_evidence
# соседних клипов этого же манифеста), а не взяты из имени клипа.
# `samples`: 'all' -- все моменты клипа, список индексов, или 'last'.
CLASS_MAP = {
    'standing': {
        'asset': 'rigged_animated_humanoid',
        'clip': 'Armature|Standing',
        'samples': 'all',
        'play': 'loop',
        'why': 'bbox 0.70 x 1.75 x 0.40 m, рост по моментам 1.7492..1.7504 m (пешеход 1.75 m), '
               'стояние на месте; склейка цикла без рывка (|end-start| = 0.0000 m)',
        'alternatives': [
            {'asset': 'animated_human_quaternius', 'clip': 'Human Armature|Idle',
             'why': 'то же стояние, рост 1.7431..1.7525 m, 12 моментов на 10 s; тоже '
                    'бесшовный цикл (|end-start| = 0.0000 m), но шаг 0.83 s -- годится только '
                    'для стоящего объекта'},
        ],
        'avoid': "rigged_animated_humanoid / 'Armature|_Default': это T-поза, ширина bbox "
                 "2.1176 m (руки в стороны), для стоящего человека не годится",
    },
    'walking': {
        'asset': 'rigged_animated_humanoid',
        'clip': 'Armature|Walk',
        'samples': 'all',
        'play': 'loop',
        'why': 'рост 1.7302..1.7772 m (просадка на шаге ~5 cm), длина шага до 1.107 m; '
               '|end-start| = 0.0000 m -- цикл склеивается без рывка; 18 моментов, шаг 55 ms',
        'alternatives': [
            {'asset': 'animated_human_quaternius', 'clip': 'Human Armature|Walk',
             'why': 'рост 1.7534..1.8258 m (на 4% выше 1.75) и |end-start| = 0.1440 m при '
                    'размахе 0.6490 m: повтор даст рывок 14 cm, нужен кроссфейд или один проход',
             'warn': 'loop_endpoint_delta_m = 0.144'},
        ],
    },
    'running': {
        'asset': 'rigged_animated_humanoid',
        'clip': 'Armature|Run',
        'samples': 'all',
        'play': 'loop',
        'why': 'рост 1.6671..1.7519 m (в стойке фазы полёта -- 1.75), вынос ноги до 1.282 m; '
               '|end-start| = 0.0004 m -- цикл бесшовный; 18 моментов, шаг 55 ms',
        'alternatives': [
            {'asset': 'animated_human_quaternius', 'clip': 'Human Armature|Run',
             'why': 'рост 1.7026..1.7745 m, |end-start| = 0.0692 m -- рывок 7 cm на склейке',
             'warn': 'в этом ассете Human Armature|Run и Human Armature|ArmatureAction.002 -- '
                     'ПОБАЙТОВО один и тот же клип (проверено: maxdist 0.000000 по всем 24 '
                     'моментам), так что это не два разных бега'},
        ],
    },
    'lying': {
        'asset': 'animated_human_quaternius',
        'clip': 'Human Armature|Death',
        'samples': [13],
        'play': 'static',
        'why': 'ТОЛЬКО последний момент (t = 3.500 s, индекс 13): высота 0.3740 m, след '
               '1.4215 x 1.9051 m, y_min = -0.0654 m -- тело лежит на земле. Отдельного '
               'клипа "лежит" нет ни в одном ассете',
        'alternatives': [
            {'asset': 'rigged_animated_humanoid', 'clip': 'Armature|SwimMove',
             'why': 'единственный клип-человек с горизонтальным телом: высота 0.7030..0.8372 m, '
                    'длина 2.1413..2.2190 m. Но это плавание (руки/ноги гребут), а не лежание: '
                    'берётся, только если нужен именно этот ассет',
             'warn': 'визуально плавание, не покой'},
        ],
        'avoid': "клипы Crouch* (высота 1.3831 m = присед, а не лежание): "
                 "'Armature|CrouchIdle' / 'Armature|CrouchDefault'",
    },
    'fallen': {
        'asset': 'animated_human_quaternius',
        'clip': 'Human Armature|Death',
        'samples': [8, 9, 10, 11, 12, 13],
        'play': 'one_shot',
        'why': 'фаза падения до касания земли и улёгшись: индекс 8 (t = 2.154 s, высота 0.8536 m) '
               '.. индекс 13 (t = 3.500 s, высота 0.3740 m). Отдельного клипа "упал" нет; '
               'динамика -- те же моменты Death, что и у lying, поэтому для статичного объекта '
               'fallen = lying (индекс 13)',
        'alternatives': [
            {'asset': 'rigged_animated_humanoid', 'clip': 'Armature|CrouchRun',
             'why': 'roст 1.3619..1.4110 m -- это приседание на бегу, а не упавшее тело; '
                    'брать только если нужна поза "низко над землёй", а не "лежит"',
             'warn': 'высота 1.38 m: это присед'},
        ],
    },
    'unknown': {
        'asset': 'rigged_animated_humanoid',
        'clip': 'Armature|Standing',
        'samples': 'all',
        'play': 'loop',
        'why': 'нейтральный пешеход 1.75 m, если класс неизвестен',
        'bbox_height_hint': {
            'note': 'класс можно угадать по высоте bbox объекта из разметки, сверяясь с '
                    'height_over_samples_m клипов',
            'ge_1.55': 'standing',           # Standing 1.7492..1.7504, Walk 1.7302..1.7772
            '1.30_to_1.55': 'crouching (нет класса в разметке)',   # CrouchIdle 1.3831
            '0.70_to_1.30': 'swimming/working (нет класса в разметке)',
            'lt_0.70': 'lying/fallen',       # Death last 0.3740, SwimMove 0.7030..0.8372
        },
    },
}


def slugify(text):
    out = []
    for ch in text:
        out.append(ch if (ch.isalnum() or ch in '._-') else '_')
    return ''.join(out).strip('_')


def load_asset_scales():
    """{asset_id: scale_to_meters} из assets/manifest.json (units.calibrated_scale_to_meters)."""
    with open(ASSETS_MANIFEST, encoding='utf-8') as fh:
        man = json.load(fh)
    scales = {}
    for asset in man['assets']:
        units = asset.get('units') or {}
        scale = units.get('calibrated_scale_to_meters')
        if scale:
            scales[asset['id']] = float(scale)
    return scales


def list_clips(fbx):
    out = subprocess.run(['node', BAKE, fbx, '--list'], cwd=ROOT,
                         capture_output=True, text=True, check=True, encoding='utf-8')
    return json.loads(out.stdout)


def sample_count(duration):
    for limit, count in SAMPLE_TABLE:
        if duration <= limit:
            return count
    raise AssertionError('unreachable')


def times_for(duration, n, loop):
    """Моменты равномерно по клипу.

    Нецикл: отрезок [0, d] включительно -- последний момент и есть конец анимации.
    Цикл: [0, d) без правого конца -- последний момент стоит за шаг до повтора, и
    проигрыватель склеивает конец с началом без дубля кадра.
    """
    if n == 1:
        return [0.0]
    if loop:
        return [duration * i / n for i in range(n)]
    return [duration * i / (n - 1) for i in range(n)]


def baked_name(time_value):
    """Имя файла момента так, как его делает bake_pose.js (`t${t.toFixed(3)}` + обрезка нулей)."""
    text = '%.3f' % time_value
    text = text.rstrip('0').rstrip('.')
    return 't' + text


def assert_distinct_names(clip, times):
    """Столкновение имён: 3.499999 и 3.5 оба дают `t3.5`, и второй перезапишет первый."""
    names = [baked_name(t) for t in times]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise SystemExit('clip %r: bake_pose file names collide at 3 decimals: %s (times %s)'
                         % (clip, dup, times))


def run_bake(fbx, clip, times, outdir, scale):
    # --loop once: без него момент ровно на t == duration отдаёт ПЕРВУЮ позу вместо
    # последней (three сворачивает действие на длительности клипа). Для моментов
    # t < duration режим ни на что не влияет, поэтому печём всегда с once.
    cmd = ['node', BAKE, fbx, '--clip', clip, '--times', ','.join(repr(t) for t in times),
           '--outdir', outdir, '--scale', repr(scale), '--loop', 'once', '--quiet']
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    if out.returncode != 0:
        raise RuntimeError('bake_pose failed for %s / %s:\n%s\n%s'
                           % (fbx, clip, out.stdout[-2000:], out.stderr[-2000:]))
    text = out.stdout
    start = min([i for i in (text.find('['), text.find('{')) if i >= 0] or [0])
    return json.loads(text[start:])


def read_sidecar(npz_path):
    with open(npz_path[:-4] + '.json', encoding='utf-8') as fh:
        return json.load(fh)


def load_npz(path):
    with np.load(path) as data:
        return (np.asarray(data['vertices'], dtype=np.float64),
                np.asarray(data['faces'], dtype=np.int64))


def max_disp(a, b):
    if a.shape != b.shape:
        return float('nan')
    return float(np.sqrt(((a - b) ** 2).sum(axis=1)).max())


def moment_files(clip_dir):
    return sorted([os.path.join(clip_dir, f) for f in os.listdir(clip_dir)
                   if f.endswith('.npz') and f != 'frames.npz'],
                  key=lambda p: read_sidecar(p)['time_seconds'])


def moments_match(clip_dir, times):
    """Есть ли в каталоге ровно эти моменты (по числам из sidecar-ов)."""
    got = [round(read_sidecar(p)['time_seconds'], 5) for p in moment_files(clip_dir)]
    return got == [round(t, 5) for t in times]


def drop_moments(clip_dir):
    for path in moment_files(clip_dir):
        os.remove(path)
        os.remove(path[:-4] + '.json')


def detect_loop(fbx, clip_name, duration, clip_dir, scale):
    """Цикл или не цикл: измеряем, насколько конец расходится с началом.

    Пробные баковки t=0 и t=duration в `.scratch/`. Признак цикла: поза в конце
    совпадает с началом (иначе склейка при повторе даст рывок).
    """
    if duration <= STATIC_MAX_S:
        return False, 'static: duration %.4f s <= %.3f s (pose, not animation)' % (
            duration, STATIC_MAX_S), None, None
    probe_dir = os.path.join(ROOT, '.scratch', 'poses_probe',
                             slugify(os.path.basename(os.path.dirname(clip_dir)) + '_' + clip_name))
    os.makedirs(probe_dir, exist_ok=True)
    run_bake(fbx, clip_name, [0.0, duration], probe_dir, scale)
    probes = moment_files(probe_dir)
    v0, _ = load_npz(probes[0])
    vd, _ = load_npz(probes[-1])
    endpoint = max_disp(v0, vd)

    move_range = 0.0
    for path in moment_files(clip_dir):
        v, _ = load_npz(path)
        move_range = max(move_range, max_disp(v0, v))

    if move_range < 0.005:
        return False, ('measured: motion range %.4f m < 5 mm, clip does not move '
                       '(endpoint delta %.4f m)' % (move_range, endpoint)), endpoint, move_range
    if endpoint <= LOOP_RATIO * move_range:
        return True, ('measured: |end-start| = %.4f m <= 10%% of range %.4f m'
                      % (endpoint, move_range)), endpoint, move_range
    if endpoint >= NONLOOP_RATIO * move_range:
        return False, ('measured: |end-start| = %.4f m >= 15%% of range %.4f m (wrap would pop)'
                       % (endpoint, move_range)), endpoint, move_range
    name = clip_name.lower()
    if any(k in name for k in LOOP_NAMES) and not any(k in name for k in NONLOOP_NAMES):
        return True, ('name prior: measured ambiguous (|end-start| = %.4f m, range %.4f m), '
                      'name reads as a cycle' % (endpoint, move_range)), endpoint, move_range
    return False, ('name prior: measured ambiguous (|end-start| = %.4f m, range %.4f m)'
                   % (endpoint, move_range)), endpoint, move_range


def bundle_clip(clip_dir, moments):
    """frames.npz: весь клип одним файлом -- vertices (T,N,3), faces (M,3), times (T,)."""
    verts, times = [], []
    faces = None
    for path in moments:
        v, f = load_npz(path)
        verts.append(v.astype(np.float32))
        if faces is None:
            faces = f.astype(np.int32)
        elif faces.shape != f.shape:
            raise ValueError('faces differ between moments of %s' % clip_dir)
        times.append(read_sidecar(path)['time_seconds'])
    stacked = np.stack(verts, axis=0)
    out = os.path.join(clip_dir, 'frames.npz')
    np.savez_compressed(out, vertices=stacked, faces=faces,
                        times=np.asarray(times, dtype=np.float32))
    return out, stacked, faces


def bake_asset(asset_id, fbx, scale, reuse=False, bundle=True):
    info = list_clips(fbx)
    print('=== %s: %d clips, scale %.6f m/unit, loader %s'
          % (asset_id, len(info['clips']), scale, info['loader']))
    entry = {
        'id': asset_id,
        'file': info['file'],
        'sha256': info['sha256'],
        'loader': info['loader'],
        'scale_to_meters': scale,
        'clip_count': len(info['clips']),
        'clips': [],
    }
    for clip in info['clips']:
        name, duration = clip['name'], clip['duration']
        slug = slugify(name)
        clip_dir = os.path.join(POSE_ROOT, asset_id, slug)
        os.makedirs(clip_dir, exist_ok=True)
        n = sample_count(duration)
        name_reads_loop = (any(k in name.lower() for k in LOOP_NAMES)
                           and not any(k in name.lower() for k in NONLOOP_NAMES))

        # 1) первый набор моментов -- по приору имени; после измерения цикла,
        #    если приор был уверенно неверен, набор перепекается (иначе у цикла
        #    останется дубль кадра на конце, а у нецикла -- потерянный последний кадр).
        times = times_for(duration, n, name_reads_loop)
        assert_distinct_names(name, times)
        if not (reuse and moments_match(clip_dir, times)):
            drop_moments(clip_dir)
            run_bake(fbx, name, times, clip_dir, scale)

        loop, why, endpoint, move_range = detect_loop(fbx, name, duration, clip_dir, scale)
        times = times_for(duration, n, loop)
        assert_distinct_names(name, times)
        if not moments_match(clip_dir, times):
            drop_moments(clip_dir)
            run_bake(fbx, name, times, clip_dir, scale)

        moments = moment_files(clip_dir)
        metas = [read_sidecar(p) for p in moments]
        real_times = [m['time_seconds'] for m in metas]
        if len(real_times) != len(times):
            raise SystemExit('%s/%s: %d moments baked, expected %d'
                             % (asset_id, slug, len(real_times), len(times)))
        if len(set(real_times)) != len(real_times):
            raise SystemExit('%s/%s: duplicate moment times %s' % (asset_id, slug, real_times))

        verts = [load_npz(p)[0] for p in moments]
        heights = [float(v[:, 1].max() - v[:, 1].min()) for v in verts]
        widths = [float(v[:, 0].max() - v[:, 0].min()) for v in verts]
        lengths = [float(v[:, 2].max() - v[:, 2].min()) for v in verts]
        allv = np.concatenate(verts, axis=0)
        steps = [max_disp(verts[i], verts[i + 1]) for i in range(len(verts) - 1)]

        bundle_rel, bundle_bytes = None, 0
        if bundle:
            path, _stacked, _faces = bundle_clip(clip_dir, moments)
            bundle_rel = os.path.relpath(path, ROOT).replace('\\', '/')
            bundle_bytes = os.path.getsize(path)

        step_s = None
        if len(real_times) >= 2:
            step_s = round(float(np.mean(np.diff(real_times))), 5)

        clip_meta = {
            'name': name,
            'slug': slug,
            'duration_s': duration,
            'kind': 'static_pose' if len(real_times) == 1 else 'animation',
            'loop': bool(loop),
            'loop_evidence': why,
            'loop_endpoint_delta_m': None if endpoint is None else round(endpoint, 5),
            'loop_wrap_pop_m': None if (endpoint is None or not loop) else round(endpoint, 5),
            'loop_needs_crossfade': bool(loop and endpoint is not None
                                         and endpoint > LOOP_POP_WARN_M),
            'motion_range_m': None if move_range is None else round(move_range, 5),
            'samples': len(real_times),
            'step_s': step_s,
            'sample_hz': None if duration <= 0 else round(len(real_times) / duration, 3),
            'vertex_move_between_samples_m': round(max(steps), 5) if steps else 0.0,
            'times_s': real_times,
            'per_sample_heights_m': [round(h, 4) for h in heights],
            'vertices': metas[0]['vertices'],
            'triangles': metas[0]['triangles'],
            'bbox_m': {
                'min': [round(float(allv[:, k].min()), 4) for k in range(3)],
                'max': [round(float(allv[:, k].max()), 4) for k in range(3)],
                'size': [round(float(allv[:, k].max() - allv[:, k].min()), 4) for k in range(3)],
            },
            'height_over_samples_m': [round(min(heights), 4), round(max(heights), 4)],
            'extent_over_samples_m': {
                'width_x': [round(min(widths), 4), round(max(widths), 4)],
                'length_z': [round(min(lengths), 4), round(max(lengths), 4)],
            },
            'dir': os.path.relpath(clip_dir, ROOT).replace('\\', '/'),
            'frames': [os.path.relpath(p, ROOT).replace('\\', '/') for p in moments],
            'bundle_npz': bundle_rel,
            'bundle_bytes': bundle_bytes,
            'bytes_total': sum(os.path.getsize(p) for p in moments),
        }
        with open(os.path.join(clip_dir, 'clip.json'), 'w', encoding='utf-8') as fh:
            json.dump(clip_meta, fh, ensure_ascii=False, indent=1)
            fh.write('\n')
        entry['clips'].append(clip_meta)
        print('  %-42s d=%-8.4f n=%-3d hz=%-6s loop=%-5s h=%s..%s m step<=%s m'
              % (name, duration, len(real_times),
                 clip_meta['sample_hz'], loop,
                 clip_meta['height_over_samples_m'][0], clip_meta['height_over_samples_m'][1],
                 clip_meta['vertex_move_between_samples_m']))
    entry['bytes_total'] = sum(c['bytes_total'] for c in entry['clips'])
    entry['bundle_bytes_total'] = sum(c['bundle_bytes'] for c in entry['clips'])
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', action='append', help='печь только этот asset_id (можно несколько)')
    ap.add_argument('--reuse', action='store_true',
                    help='не вызывать bake_pose там, где моменты уже есть, только пересобрать метаданные')
    ap.add_argument('--no-bundle', action='store_true', help='не писать frames.npz')
    args = ap.parse_args()

    scales = load_asset_scales()
    wanted = set(args.only or [])
    manifest = {
        'schema': 'poses-manifest/1',
        'generated_by': 'python tools/bake_poses.py',
        'baked_by': 'node tools/bake_pose.js (three, skinning outside the browser)',
        'pose_root': 'assets/_poses/',
        'unit_convention': {
            'npz_vertices': 'metres, already multiplied by the per-asset scale_to_meters',
            'do_not_scale_again': 'the per-file .json carries scale_applied = the factor that was '
                                  'ALREADY applied to those vertices; multiplying by it again '
                                  'double-scales the mesh',
            'up_axis': 'Y (as in the source glTF/FBX)',
            'origin': 'standing pose: on the ground under the feet (y_min ~ 0)',
        },
        'placement': {
            'to_scene_axes': SCENE_AXES,
            'note': 'the viewer frame is Z-up: rotate, then translate to the object centre',
        },
        'playback': {
            'loop_true': 'times are [0, d) exclusive; index = floor((t mod d) / d * T) -- wraps '
                         'with no duplicated frame',
            'loop_false': 'times are [0, d] inclusive; index = clamp(round(t / d * (T - 1)), 0, T - 1)',
            'loop_wrap_pop': 'at the wrap of a loop clip the pose jumps by loop_wrap_pop_m; '
                             'if loop_needs_crossfade is true, blend the last step into the first '
                             '(or treat the clip as a one-shot)',
            'interpolation': 'linear between neighbouring stored moments is enough; per clip '
                             'vertex_move_between_samples_m is the largest vertex move between '
                             'two moments, i.e. the interpolation error bound in metres',
            'single_file': 'prefer <clip>/frames.npz (vertices (T,N,3) float32, faces (M,3) int32, '
                           'times (T,) float32)',
        },
        'class_map': CLASS_MAP,
        'assets': [],
    }

    for asset_id, fbx in ASSETS:
        if wanted and asset_id not in wanted:
            continue
        if not os.path.exists(os.path.join(ROOT, fbx)):
            print('SKIP %s: %s not found' % (asset_id, fbx), file=sys.stderr)
            continue
        scale = scales.get(asset_id)
        if not scale:
            raise SystemExit('%s: no units.calibrated_scale_to_meters in assets/manifest.json'
                             % asset_id)
        manifest['assets'].append(
            bake_asset(asset_id, fbx, scale, reuse=args.reuse, bundle=not args.no_bundle))

    os.makedirs(POSE_ROOT, exist_ok=True)
    out = os.path.join(POSE_ROOT, 'manifest.json')
    if wanted and os.path.exists(out):
        # при --only перепекается часть ассетов: остальные записи манифеста сохраняем
        with open(out, encoding='utf-8') as fh:
            old = json.load(fh)
        by_id = {a['id']: a for a in old.get('assets', [])}
        for a in manifest['assets']:
            by_id[a['id']] = a
        manifest['assets'] = [by_id[i] for i, _ in ASSETS if i in by_id]
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
        fh.write('\n')
    print('\nwrote %s (%d assets, %d clips, %d moments, %.1f MiB frames + %.1f MiB bundles)'
          % (os.path.relpath(out, ROOT), len(manifest['assets']),
             sum(len(a['clips']) for a in manifest['assets']),
             sum(c['samples'] for a in manifest['assets'] for c in a['clips']),
             sum(a['bytes_total'] for a in manifest['assets']) / 2**20,
             sum(a['bundle_bytes_total'] for a in manifest['assets']) / 2**20))


if __name__ == '__main__':
    main()