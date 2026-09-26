"""Recall/precision детектора по размеченным кадрам (`labels/`).

Прогон `detector.core.detect` на кадрах РАЗМЕТКИ: по каждой паре (запись, кадр)
из `labels/` собирается кадр с объектами (облако из .db3, тень снята, точки
объектов вставлены -- как их видел бы лидар, см. `labels_io`), детектор ищет
препятствия, они сопоставляются с объектами карточек. До этого модуля формат
`labels/` никто не читал и детектор по разметке не гонялся.

Сопоставление (критерий тот же, что «детектор подтвердил» у `labeling`):
обнаружение и объект считаются одной целью, если их путевые координаты
(y -- вдоль, u -- поперёк, в системе кадра: `labeling.path_frame`) сходятся в
допусках `max(1.5, длина/2 + 1.0)` м по y и `max(1.0, ширина/2 + 0.6)` м по u.
Сопоставление жадное: обнаружения по убыванию confidence, каждое -- к ещё не
сопоставленному объекту. Дальность объекта -- |центр| в плане (гипотенуза x, y).

Строки отчёта -- группа x ведро дальности (0-10, 10-20, 20-40, 40+ м):
  gt        кадров с объектом этой группы в ведре;
  matched   из них найдены детектором (recall = matched/gt);
  det       препятствий детектора в ведре (на размеченных кадрах);
  det_ok    из них сопоставлены с объектом (precision = det_ok/det).

Группа выбирается `--by`: `class` (по умолчанию, как было), `pose` (поза
карточки), `trajectory` (объекты с траекторией против стоящих), `distance`
(дальности -- при этой группировке вёдра мельче: 0-7.5/7.5-15/15-30/30+,
чтобы дальности 5/10/20/40 м попали в отдельные строки).

Режим `--temporal`: кадры каждой записи гоняются ПОДРЯД отрезками смежных
номеров через `detector.temporal.TemporalDetector` (накопление K кадров +
подтверждение серией M из N + коастинг), а не изолированно. Для продольного
переноса кадров считается профиль хода записи (`labeling.motion_profile`,
кэш -- `.scratch/motion_cache/<запись>.json`). Задержка подтверждения трека
(M кадров) обрезала бы начало серии, поэтому попадания трека до подтверждения
засчитываются ЗАДНИМ ЧИСЛОМ (`TemporalResult.retro`, их число -- `meta['retro']`
в сводке): это оффлайн-оценка записи, а не онлайн-аларм.

Запуск (из корня проекта):

    python -m validation.dataset_eval                       # labels/ (демо-объект)
    python -m validation.dataset_eval --labels-dir .scratch/labels_synth
    python -m validation.dataset_eval --labels-dir dataset/labels \
        --records doubleT_obstacle roundT_doubleT --by pose
    python -m validation.dataset_eval --labels-dir dataset/anchor_person \
        --temporal                                          # якорь с трекером

Пишет `validation/out/dataset_eval.csv` (строки отчёта; при группировке не
по классам имя файла -- `dataset_eval_<группа>.csv`, во временном режиме --
`dataset_eval_temporal[_<группа>].csv`). Прогон экономный:
кадры -- только из `frames` карточек, запись открывается один раз.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

# Вёдра дальности, м: (имя, lo, hi) -- полуинтервал (lo, hi], последний открыт.
DIST_BUCKETS = (('0-10', 0.0, 10.0), ('10-20', 10.0, 20.0),
                ('20-40', 20.0, 40.0), ('40+', 40.0, float('inf')))
# Вёдра при --by distance: мельче, чтобы дальности раскладки 5/10/20/40 м
# попали в РАЗНЫЕ строки (5 -> 0-7.5, 10 -> 7.5-15, 20 -> 15-30, 40 -> 30+).
DIST_BUCKETS_FINE = (('0-7.5', 0.0, 7.5), ('7.5-15', 7.5, 15.0),
                     ('15-30', 15.0, 30.0), ('30+', 30.0, float('inf')))
GROUPS = ('class', 'pose', 'trajectory', 'distance')
# Допуски сопоставления -- как CONFIRM_TOL у labeling.propose/preview.
TOL_Y_BASE_M, TOL_U_BASE_M = 1.5, 1.0


def project_root() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def bucket_of(distance_m, buckets=DIST_BUCKETS):
    """Имя ведра дальности (последнее, куда попадает; None, если <= 0)."""
    if distance_m is None or distance_m <= 0.0:
        return None
    for name, lo, hi in buckets:
        if lo < distance_m <= hi:
            return name
    return buckets[-1][0]


def group_key(row, group_by):
    """Имя группы строки отчёта: класс/поза/траектория (по строке индекса)."""
    if group_by == 'pose':
        return str(row.get('pose') or 'unknown')
    if group_by == 'trajectory':
        return ('traj' if row.get('trajectory') else 'static')
    return str(row.get('class') or 'other')


def build_scene_cloud(scene):
    """Кадр с объектами, как его видел бы лидар: фон без тени + точки объектов.

    Точки, закрытые объектами, из фона уходят (тень), вместо них -- точки
    объектов; только объекты, размеченные на ЭТОМ кадре (`frame_labeled`).
    """
    xyz = np.asarray(scene['xyz'], dtype=np.float64)
    labeled = [o for o in scene['objects'] if o['frame_labeled']]
    shadow = scene['shadow_mask']
    bg = xyz[~shadow] if shadow is not None else xyz
    parts = [bg] + [np.asarray(o['points'], dtype=np.float64) for o in labeled]
    return np.vstack(parts) if len(parts) > 1 else bg


def detect_on_scene(scene, model, cfg):
    """Детектор на кадре с объектами: (препятствия, размер дополненного облака)."""
    from detector.core import detect  # noqa: PLC0415

    aug = build_scene_cloud(scene)
    result = detect(aug, model=model, cfg=cfg)
    obstacles = [o.to_dict() if hasattr(o, 'to_dict') else dict(o)
                 for o in (result.obstacles or [])]
    return obstacles, int(aug.shape[0])


def motion_fn_cached(db_path, cache_dir=None):
    """`motion_between(f_from, f_to)` записи: профиль хода с кэшем на диске.

    Профиль (`labeling.motion_profile`) считается по всей записи за десятки
    секунд, поэтому кладётся в `.scratch/motion_cache/<запись>.json` и
    перечитывается между прогонами.
    """
    import json  # noqa: PLC0415

    import labeling  # noqa: PLC0415
    from detector.temporal import motion_between_from_profile  # noqa: PLC0415

    record = os.path.basename(os.path.dirname(os.path.abspath(db_path)))
    if cache_dir is None:
        cache_dir = os.path.join(project_root(), '.scratch', 'motion_cache')
    path = os.path.join(cache_dir, record + '.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as handle:
            profile = json.load(handle)
    else:
        profile = labeling.motion_profile(db_path)
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(profile, handle)
    return motion_between_from_profile(profile)


def match_objects(scene, obstacles, frame):
    """Сопоставить препятствия с объектами кадра: (список пар, unmatched-детекции).

    `frame` -- путевая система КАДРА (`labeling.PathFrame`): (y, u) объекта
    считаются в той же системе, в которой детектор ищет препятствия. Каждое
    обнаружение (по убыванию confidence) идёт к ближайшему ещё свободному
    объекту в допуске.
    """
    labeled = [o for o in scene['objects'] if o['frame_labeled']]
    targets = []
    for obj in labeled:
        center = obj['center']
        y_c = float(center[1])
        u_c = float(center[0]) - float(frame.axis_x(np.asarray([y_c]))[0])
        dist = float(np.hypot(center[0], center[1]))
        size = (obj.get('instance') or {}).get('size') or [0.6, 1.0, 1.0]
        tol_y = max(TOL_Y_BASE_M, 0.5 * float(size[2]) + 1.0)
        tol_u = max(TOL_U_BASE_M, 0.5 * float(size[0]) + 0.6)
        targets.append({'obj': obj, 'y': y_c, 'u': u_c, 'dist': dist,
                        'tol_y': tol_y, 'tol_u': tol_u, 'matched': None})
    order = sorted(obstacles, key=lambda d: -float(d.get('confidence') or 0.0))
    pairs, unmatched = [], []
    for det in order:
        y_d, u_d = float(det.get('y_m')), float(det.get('u_m'))
        best = None
        for t in targets:
            if t['matched'] is not None:
                continue
            if abs(y_d - t['y']) <= t['tol_y'] and abs(u_d - t['u']) <= t['tol_u']:
                d2 = (y_d - t['y']) ** 2 + (u_d - t['u']) ** 2
                if best is None or d2 < best[0]:
                    best = (d2, t)
        if best is None:
            unmatched.append(det)
        else:
            best[1]['matched'] = det
            pairs.append((best[1], det))
    return pairs, unmatched


def _frame_info(scene, frame, model):
    """Лёгкий слепок кадра для сопоставления: цели и путевая система кадра.

    Облако не хранится (сотни кадров по 8 МБ в памяти не нужны): сопоставлению
    нужны центры/размеры объектов и `PathFrame` (модель + поза кадра).
    """
    import labeling  # noqa: PLC0415
    from detector.core import axis_pose  # noqa: PLC0415

    objects = [{'id': o['id'], 'frame_labeled': o['frame_labeled'],
                'center': o['center'], 'instance': o['instance']}
               for o in scene['objects']]
    pose = axis_pose(np.asarray(scene['xyz'], dtype=np.float64), model)
    return {'frame': int(scene['frame']), 'record': scene['record'],
            'objects': objects,
            'path_frame': labeling.PathFrame(model, pose)}


def eval_labels(labels_dir, bags_dir, ids=None, cfg=None, verbose=True,
                records=None, group_by='class', temporal=False):
    """Прогон по всем объектам labels/: список строк отчёта + счётчики.

    Строка отчёта: dict(group, bucket, gt, matched, det, det_ok, recall,
    precision) -- по группе объекта (`group_by`: класс/поза/траектория/даль-
    ность) и ведру дальности. Recall -- по группам (объекты карточек группы
    имеют); precision -- только по строке `all`: детектор группы не выдаёт, и
    нераспределённое по группам препятствие честнее считать одним знаменателем.
    `records` -- прогнать только эти записи (сплит); `ids` -- только эти объекты.

    `temporal=True` -- режим последовательностей: кадры записи гоняются отрезками
    смежных номеров через `detector.temporal.TemporalDetector` (накопление +
    подтверждение серией M из N). Попадания трека до его подтверждения
    засчитываются задним числом (`retro`): оффлайн-оценка записи, без задержки
    онлайн-алгоритма. Их число возвращается в `meta['retro']`.
    """
    if group_by not in GROUPS:
        raise ValueError('group_by %r: доступны %s' % (group_by, ', '.join(GROUPS)))
    buckets = (DIST_BUCKETS_FINE if group_by == 'distance' else DIST_BUCKETS)
    import labels_io  # noqa: PLC0415
    from detector.profiles import model_for_db  # noqa: PLC0415

    reader = labels_io.LabelsReader(labels_dir, bags_dir)
    id_set = None if ids is None else set(ids)
    rec_set = None if records is None else set(records)
    rows = [r for r in reader.rows
            if (id_set is None or r['id'] in id_set)
            and (rec_set is None or r.get('record') in rec_set)]
    if not rows:
        print('нет объектов в %s (index.jsonl пуст, ids или records не найдены)'
              % labels_dir)
        return [], {}
    if cfg is None:
        from detector import DetectorConfig  # noqa: PLC0415

        cfg = DetectorConfig()

    # Кадры -- UNION по записи: каждый кадр гоняется ОДИН раз, со всеми
    # объектами записи, размеченными на нём.
    by_record = {}
    for row in rows:
        obj = reader.object(row['id'], row.get('record'))
        for f in obj.frames:
            by_record.setdefault(row['record'], {}).setdefault(int(f), []).append(row)

    models, stats = {}, {}
    scenes = 0
    retro_applied = 0
    rows_by_id = {r['id']: r for r in rows}
    try:
        for record in sorted(by_record):
            if record not in models:
                models[record] = model_for_db(reader.db_path(record))
            model = models[record]
            # Проход 1: детекции по кадрам (в temporal-режиме -- отрезками
            # смежных кадров через TemporalDetector, со свежим трекером на
            # каждый отрезок).
            infos = []
            retro_by_frame = {}
            tracker = None
            motion_fn = None
            prev_frame = None
            for f in sorted(by_record[record]):
                scene = reader.frame(record, f, include_unlabeled_frames=False)
                scenes += 1
                info = _frame_info(scene, f, model)
                if not temporal:
                    info['obstacles'], info['n_aug'] = detect_on_scene(
                        scene, model, cfg)
                else:
                    from detector.temporal import TemporalDetector  # noqa: PLC0415

                    if tracker is None or prev_frame is None or f != prev_frame + 1:
                        if motion_fn is None:
                            motion_fn = motion_fn_cached(reader.db_path(record))
                        tracker = TemporalDetector(model, cfg=cfg,
                                                   motion_between=motion_fn)
                    aug = build_scene_cloud(scene)
                    res = tracker.update(aug, frame_index=f)
                    info['obstacles'] = [o.to_dict() for o in res.obstacles]
                    info['n_aug'] = int(aug.shape[0])
                    for rf, robs in res.retro.items():
                        retro_by_frame.setdefault(int(rf), []).extend(
                            o.to_dict() if hasattr(o, 'to_dict') else dict(o)
                            for o in robs)
                infos.append(info)
                prev_frame = f
            if temporal:
                for info in infos:
                    extra = retro_by_frame.get(info['frame'])
                    if extra:
                        # Ретро-детекция, дублирующая уже выданную на кадре,
                        # вторым препятствием не считается (иначе precision
                        # занижается на ровном месте).
                        extra = [o for o in extra
                                 if not any(abs(o['y_m'] - e['y_m']) <= 1.5
                                            and abs(o['u_m'] - e['u_m']) <= 0.9
                                            for e in info['obstacles'])]
                        retro_applied += len(extra)
                        info['obstacles'] = info['obstacles'] + extra
            # Проход 2: сопоставление и счётчики (одинаково для обоих режимов).
            for info in infos:
                obstacles = info['obstacles']
                pairs, unmatched = match_objects(
                    {'objects': info['objects']}, obstacles, info['path_frame'])
                if verbose:
                    matched_ids = [t['obj']['id'] for t, _d in pairs]
                    print('  %-12s кадр %4d: объектов %2d, препятствий %2d, '
                          'сопоставлено %d%s, точек в кадре %d'
                          % (record, info['frame'], len(info['objects']),
                             len(obstacles),
                             len(pairs),
                             (' (%s)' % ', '.join(matched_ids) if matched_ids
                              else ' -- ПРОПУСК' if info['objects'] else ''),
                             info['n_aug']))
                for t, det in pairs:
                    row = rows_by_id.get(t['obj']['id'], {})
                    _bump(stats, group_key(row, group_by),
                          bucket_of(t['dist'], buckets), gt=1, matched=1)
                    _bump(stats, 'all', bucket_of(t['dist'], buckets),
                          gt=1, matched=1, det=1, det_ok=1)
                for obj in (o for o in info['objects'] if o['frame_labeled']):
                    if _pair_of(obj, pairs) is None:
                        dist = float(np.hypot(obj['center'][0], obj['center'][1]))
                        _bump(stats, group_key(rows_by_id.get(obj['id'], {}), group_by),
                              bucket_of(dist, buckets), gt=1)
                        _bump(stats, 'all', bucket_of(dist, buckets), gt=1)
                for det in unmatched:
                    _bump(stats, 'all', bucket_of(det.get('distance_m'), buckets), det=1)
    finally:
        reader.close()
    meta = {'objects': len(rows), 'scenes': scenes}
    if temporal:
        meta['retro'] = retro_applied
    return _report_rows(stats), meta


def _pair_of(obj, pairs):
    for t, _d in pairs:
        if t['obj'] is obj:
            return t
    return None


def _bump(stats, cls, bucket, gt=0, matched=0, det=0, det_ok=0):
    if bucket is None:
        return
    key = (str(cls), str(bucket))
    row = stats.setdefault(key, {'class': str(cls), 'bucket': str(bucket),
                                 'gt': 0, 'matched': 0, 'det': 0, 'det_ok': 0})
    row['gt'] += gt
    row['matched'] += matched
    row['det'] += det
    row['det_ok'] += det_ok


def _report_rows(stats):
    def rates(row):
        recall = (row['matched'] / float(row['gt'])) if row['gt'] else None
        precision = (row['det_ok'] / float(row['det'])) if row['det'] else None
        row['recall'] = recall
        row['precision'] = precision
        return row

    order = {name: i for i, (name, _lo, _hi) in enumerate(DIST_BUCKETS)}
    order.update({name: i for i, (name, _lo, _hi) in enumerate(DIST_BUCKETS_FINE)})
    return [rates(dict(stats[k])) for k in sorted(
        stats, key=lambda k: (stats[k]['class'] != 'all',
                              k[0] if stats[k]['class'] != 'all' else '',
                              order.get(k[1], 99)))]


GROUP_LABELS = {'class': 'класс', 'pose': 'поза', 'trajectory': 'траектория',
                'distance': 'дальность'}


def render_report(rows, meta, labels_dir, group_by='class', temporal=False):
    lines = []
    lines.append('Recall/precision детектора по разметке: %s%s'
                 % (labels_dir,
                    ' (ВРЕМЕННОЙ РЕЖИМ: накопление кадров + трекер M из N, '
                    'попадания до подтверждения засчитаны задним числом)'
                    if temporal else ''))
    lines.append('кадры -- из frames карточек; кадр = фон без тени + точки объектов '
                 '(labels_io); пороги детектора -- DetectorConfig() по умолчанию')
    lines.append('сопоставление: |dy| <= max(%.1f, длина/2+1.0) м и |du| <= max(%.1f, '
                 'ширина/2+0.6) м в путевых координатах кадра; жадно по confidence'
                 % (TOL_Y_BASE_M, TOL_U_BASE_M))
    lines.append('recall -- по %s объектов; precision -- только строка all '
                 '(детектор группы не выдаёт, знаменатель один на все)'
                 % GROUP_LABELS.get(group_by, group_by))
    if meta:
        extra = ('; попаданий задним числом (retro): %d' % meta['retro']
                 if meta.get('retro') is not None else '')
        lines.append('объектов %d, кадров %d%s' % (meta.get('objects', 0),
                                                   meta.get('scenes', 0), extra))
    lines.append('')
    lines.append(' %-9s | дальн., м | gt  | найдено | recall | det  | det_ok | precision'
                 % GROUP_LABELS.get(group_by, 'класс'))
    lines.append('-' * 84)
    for row in rows:
        lines.append(' %-9s | %-8s | %3d | %7d | %6s | %4d | %6d | %s'
                     % (row['class'], row['bucket'], row['gt'], row['matched'],
                        _fmt(row['recall']), row['det'], row['det_ok'],
                        _fmt(row['precision'])))
    return '\n'.join(lines)


def _fmt(value):
    if value is None or value != value:
        return '--'
    return '%.3f' % value


def write_csv(rows, out_dir, name='dataset_eval.csv'):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fields = ['class', 'bucket', 'gt', 'matched', 'det', 'det_ok', 'recall', 'precision']
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description='recall/precision детектора по labels/')
    parser.add_argument('--labels-dir', default='labels',
                        help='каталог разметки (index.jsonl + карточки)')
    parser.add_argument('--bags', default='for_hackathon',
                        help='корень записей с .db3')
    parser.add_argument('--ids', nargs='*', default=None,
                        help='id объектов (по умолчанию все из index.jsonl)')
    parser.add_argument('--records', nargs='*', default=None,
                        help='прогнать только эти записи (например, val-сплит)')
    parser.add_argument('--by', default='class', choices=GROUPS,
                        help='группа строк отчёта (класс/поза/траектория/дальность)')
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--temporal', action='store_true',
                        help='кадры подряд через detector.temporal.TemporalDetector '
                             '(накопление + трекер M из N); профиль хода -- '
                             'labeling.motion_profile, кэш .scratch/motion_cache')
    parser.add_argument('--quiet', action='store_true', help='без строк по кадрам')
    args = parser.parse_args(argv)

    root = project_root()
    labels_dir = (args.labels_dir if os.path.isabs(args.labels_dir)
                  else os.path.join(root, args.labels_dir))
    bags_dir = (args.bags if os.path.isabs(args.bags) else os.path.join(root, args.bags))
    out_dir = args.out_dir or os.path.join(root, 'validation', 'out')

    rows, meta = eval_labels(labels_dir, bags_dir, ids=args.ids, verbose=not args.quiet,
                             records=args.records, group_by=args.by,
                             temporal=args.temporal)
    if not rows:
        return 2
    print()
    print(render_report(rows, meta, args.labels_dir, group_by=args.by,
                        temporal=args.temporal))
    stem = 'dataset_eval'
    if args.temporal:
        stem += '_temporal'
    if args.by != 'class':
        stem += '_%s' % args.by
    path = write_csv(rows, out_dir, name=stem + '.csv')
    print()
    print('файл: %s' % path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
