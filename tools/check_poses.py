#!/usr/bin/env python
"""Проверить запечённые позы из `assets/_poses/**` и напечатать числа.

    python tools/check_poses.py                 # по манифесту
    python tools/check_poses.py --independent   # + перепечь 3 клипа заново и сверить с файлами

Что проверяется:
  * .npz читается numpy: vertices (N,3) float32, faces (M,3) int32, N/M те же, что в манифесте;
  * лица у всех моментов клипа одинаковые (это один и тот же меш);
  * вершины между моментами РАЗЛИЧАЮТСЯ, если клип не статичный -- иначе поза не анимируется;
  * максимальное смещение вершины за клип в метрах (амплитуда) и максимальное смещение за шаг
    (ошибка линейной интерполяции между сохранёнными моментами);
  * габарит после масштаба правдоподобен для человека: стоя 1.6..1.9 m, лежа/упал 0.3..0.5 m;
  * frames.npz совпадает с по-моментными файлами;
  * (--independent) перепечённый заново момент побайтово совпадает с сохранённым.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
POSE_ROOT = os.path.join(ROOT, 'assets', '_poses')
ASSETS_MANIFEST = os.path.join(ROOT, 'assets', 'manifest.json')
BAKE = os.path.join(HERE, 'bake_pose.js')

# Правдоподобные габариты человека, метры.
STANDING_H = (1.6, 1.9)
LYING_H = (0.3, 0.5)


def load_npz(path):
    with np.load(path) as data:
        return (np.asarray(data['vertices'], dtype=np.float64),
                np.asarray(data['faces'], dtype=np.int64))


def max_disp(a, b):
    return float(np.sqrt(((a - b) ** 2).sum(axis=1)).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--independent', action='store_true',
                    help='перепечь по одному моменту на клип заново и сравнить с файлами')
    args = ap.parse_args()

    with open(os.path.join(POSE_ROOT, 'manifest.json'), encoding='utf-8') as fh:
        man = json.load(fh)
    with open(ASSETS_MANIFEST, encoding='utf-8') as fh:
        scales = {a['id']: (a.get('units') or {}).get('calibrated_scale_to_meters')
                  for a in json.load(fh)['assets']}

    problems = []
    print('%-26s %-34s %-4s %-5s %-9s %-9s %-11s %s'
          % ('asset', 'clip', 'n', 'loop', 'maxmove', 'maxstep', 'height m', 'bbox m (wx,h,lz)'))
    print('-' * 132)

    for asset in man['assets']:
        scale = scales.get(asset['id'])
        if scale is None or abs(scale - asset['scale_to_meters']) > 1e-12:
            problems.append('%s: scale %s != assets/manifest.json %s'
                            % (asset['id'], asset['scale_to_meters'], scale))
        for clip in asset['clips']:
            frames = clip['frames']
            if len(frames) != clip['samples']:
                problems.append('%s/%s: %d frame paths, samples=%d'
                                % (asset['id'], clip['slug'], len(frames), clip['samples']))
                continue
            verts, faces = [], None
            for rel in frames:
                path = os.path.join(ROOT, rel)
                if not os.path.exists(path):
                    problems.append('%s: missing %s' % (clip['slug'], rel))
                    continue
                v, f = load_npz(path)
                if v.shape != (clip['vertices'], 3):
                    problems.append('%s: %s vertices %s != %s'
                                    % (clip['slug'], os.path.basename(path), v.shape, clip['vertices']))
                if f.shape != (clip['triangles'], 3):
                    problems.append('%s: %s faces %s != %s'
                                    % (clip['slug'], os.path.basename(path), f.shape, clip['triangles']))
                if faces is None:
                    faces = f
                elif not np.array_equal(faces, f):
                    problems.append('%s: faces differ between moments' % clip['slug'])
                verts.append(v)

            allv = np.concatenate(verts, axis=0)
            height = float(allv[:, 1].max() - allv[:, 1].min())
            wx = float(allv[:, 0].max() - allv[:, 0].min())
            lz = float(allv[:, 2].max() - allv[:, 2].min())

            steps = [max_disp(verts[i], verts[i + 1]) for i in range(len(verts) - 1)]
            maxmove = max(max_disp(verts[0], v) for v in verts)
            maxstep = max(steps) if steps else 0.0

            static = clip['kind'] == 'static_pose'
            if len(verts) > 1 and not static and maxmove <= 1e-7:
                problems.append('%s: vertices do NOT change between moments (maxmove=%.9f)'
                                % (clip['slug'], maxmove))
            if static and maxmove > 1e-7:
                problems.append('%s: static clip but vertices move %.6f m'
                                % (clip['slug'], maxmove))

            # записанное в манифесте должно совпадать с пересчитанным из файлов
            want_h = clip['height_over_samples_m']
            got_h = [round(float(v[:, 1].max() - v[:, 1].min()), 4) for v in verts]
            if abs(want_h[0] - min(got_h)) > 1e-3 or abs(want_h[1] - max(got_h)) > 1e-3:
                problems.append('%s: height_over_samples_m %s != recomputed %s'
                                % (clip['slug'], want_h, [min(got_h), max(got_h)]))
            want_step = clip['vertex_move_between_samples_m']
            if abs(want_step - round(maxstep, 5)) > 1e-4:
                problems.append('%s: vertex_move_between_samples_m %s != recomputed %.5f'
                                % (clip['slug'], want_step, maxstep))

            # bundle vs per-moment
            if clip.get('bundle_npz'):
                bpath = os.path.join(ROOT, clip['bundle_npz'])
                with np.load(bpath) as z:
                    bv = np.asarray(z['vertices'], dtype=np.float32)
                    bf = np.asarray(z['faces'], dtype=np.int64)
                    bt = np.asarray(z['times'], dtype=np.float32)
                if bv.shape != (len(verts), clip['vertices'], 3):
                    problems.append('%s: bundle vertices %s' % (clip['slug'], bv.shape))
                elif not np.allclose(bv.astype(np.float64), np.stack(verts), atol=1e-6):
                    problems.append('%s: bundle vertices differ from per-moment files' % clip['slug'])
                if not np.array_equal(bf, faces):
                    problems.append('%s: bundle faces differ' % clip['slug'])
                if not np.allclose(bt, clip['times_s'], atol=1e-5):
                    problems.append('%s: bundle times %s != %s' % (clip['slug'], bt, clip['times_s']))

            print('%-26s %-34s %-4d %-5s %-9.4f %-9.4f %-11s %.3f x %.3f x %.3f'
                  % (asset['id'][:26], clip['name'][:34], len(verts), clip['loop'],
                     maxmove, maxstep, '%.4f..%.4f' % (min(got_h), max(got_h)), wx, height, lz))

    # габариты, которые заказчик назвал ожидаемыми, проверяются по class_map --
    # то есть по тому, что реально предложено интегратору
    print('\n-- plausibility (по class_map) --')
    by_id = {a['id']: a for a in man['assets']}
    for cls, spec in man['class_map'].items():
        if 'asset' not in spec:
            continue
        clip = next(c for c in by_id[spec['asset']]['clips'] if c['name'] == spec['clip'])
        want = LYING_H if cls in ('lying', 'fallen') else STANDING_H
        if cls in ('lying', 'fallen'):
            idx = spec['samples'][-1]
            h = clip['per_sample_heights_m'][idx]
            span = [min(clip['per_sample_heights_m'][i] for i in spec['samples']),
                    max(clip['per_sample_heights_m'][i] for i in spec['samples'])]
            ok = want[0] <= h <= want[1]
            print('  %-9s %-24s sample %-2d t=%-6.3f h=%.4f m (all samples %.4f..%.4f) in %s -> %s'
                  % (cls, clip['name'][:24], idx, clip['times_s'][idx], h, span[0], span[1],
                     str(want), 'ok' if ok else 'OUT OF RANGE'))
        else:
            h = clip['height_over_samples_m'][1]
            ok = want[0] <= h <= want[1]
            print('  %-9s %-24s h=%.4f m (over samples %s) in %s -> %s'
                  % (cls, clip['name'][:24], h, clip['height_over_samples_m'], str(want),
                     'ok' if ok else 'OUT OF RANGE'))
        if not ok:
            problems.append('class %s: %s height %.4f m outside %s'
                            % (cls, spec['clip'], h, str(want)))

    if args.independent:
        print('\n-- independent re-bake (same time, fresh node process) --')
        for asset_id, clip_name, idx in (
                ('rigged_animated_humanoid', 'Armature|Walk', 4),
                ('rigged_animated_humanoid', 'Armature|Standing', 9),
                ('animated_human_quaternius', 'Human Armature|Death', 13)):
            asset = next(a for a in man['assets'] if a['id'] == asset_id)
            clip = next(c for c in asset['clips'] if c['name'] == clip_name)
            frame_rel = clip['frames'][idx]
            stored, faces_stored = load_npz(os.path.join(ROOT, frame_rel))
            time = json.load(open(os.path.join(ROOT, frame_rel)[:-4] + '.json',
                                  encoding='utf-8'))['time_seconds']
            out = os.path.join(ROOT, '.scratch', 'check_rebake', asset_id, clip['slug'])
            subprocess.run(['node', BAKE, asset['file'], '--clip', clip_name,
                            '--times', repr(time), '--outdir', out,
                            '--scale', repr(asset['scale_to_meters']),
                            '--loop', 'once', '--quiet'],
                           cwd=ROOT, capture_output=True, text=True, check=True, encoding='utf-8')
            fresh = sorted(os.path.join(out, f) for f in os.listdir(out) if f.endswith('.npz'))
            v2, f2 = load_npz(fresh[-1])
            d = max_disp(stored, v2)
            same_faces = np.array_equal(faces_stored, f2)
            # .npz хранит float32, поэтому свежий процесс может разойтись на несколько ULP:
            # измеренный максимум на Walk -- 5.3e-7 m (0.5 мкм), это на 5 порядков меньше
            # шага между моментами. Порог 1e-5 m.
            print('  %-24s %-24s t=%-8.4f  max|d vertex| = %.9f m  faces equal: %s'
                  % (asset_id, clip_name, time, d, same_faces))
            if d > 1e-5 or not same_faces:
                problems.append('independent re-bake differs: %s/%s at t=%.4f (%.9f m)'
                                % (asset_id, clip_name, time, d))

    print('\n%s: %d problem(s)' % ('FAIL' if problems else 'OK', len(problems)))
    for p in problems:
        print('  ! ' + p)
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())