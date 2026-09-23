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

Строки отчёта -- класс x ведро дальности (0-10, 10-20, 20-40, 40+ м):
  gt        кадров с объектом этого класса в ведре;
  matched   из них найдены детектором (recall = matched/gt);
  det       препятствий детектора в ведре (на размеченных кадрах);
  det_ok    из них сопоставлены с объектом (precision = det_ok/det).

Запуск (из корня проекта):

    python -m validation.dataset_eval                       # labels/ (демо-объект)
    python -m validation.dataset_eval --labels-dir .scratch/labels_synth

Пишет `validation/out/dataset_eval.csv` (строки отчёта). Прогон экономный:
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
# Допуски сопоставления -- как CONFIRM_TOL у labeling.propose/preview.
TOL_Y_BASE_M, TOL_U_BASE_M = 1.5, 1.0


def project_root() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def bucket_of(distance_m):
    """Имя ведра дальности (последнее, куда попадает; None, если <= 0)."""
    if distance_m is None or distance_m <= 0.0:
        return None
    for name, lo, hi in DIST_BUCKETS:
        if lo < distance_m <= hi:
            return name
    return DIST_BUCKETS[-1][0]


def detect_on_scene(scene, model, cfg):
    """Детектор на кадре с объектами: (препятствия, размер дополненного облака).

    Кадр собирается как у `labeling.preview`: фон без ТЕНИ объектов (точки,
    закрытые объектами, из фона уходят) плюс точки объектов; только объекты,
    размеченные на ЭТОМ кадре (`frame_labeled`).
    """
    from detector.core import detect  # noqa: PLC0415

    xyz = np.asarray(scene['xyz'], dtype=np.float64)
    labeled = [o for o in scene['objects'] if o['frame_labeled']]
    shadow = scene['shadow_mask']
    bg = xyz[~shadow] if shadow is not None else xyz
    parts = [bg] + [np.asarray(o['points'], dtype=np.float64) for o in labeled]
    aug = np.vstack(parts) if len(parts) > 1 else bg
    result = detect(aug, model=model, cfg=cfg)
    obstacles = [o.to_dict() if hasattr(o, 'to_dict') else dict(o)
                 for o in (result.obstacles or [])]
    return obstacles, int(aug.shape[0])


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


def eval_labels(labels_dir, bags_dir, ids=None, cfg=None, verbose=True):
    """Прогон по всем объектам labels/: список строк отчёта + счётчики.

    Строка отчёта: dict(class, bucket, gt, matched, det, det_ok, recall,
    precision) -- по классу объекта и ведру дальности. Recall -- по классам
    (объекты карточек классы имеют); precision -- только по строке `all`:
    детектор классы не выдаёт, и нераспределённое по классам препятствие
    честнее считать одним знаменателем.
    """
    import labeling  # noqa: PLC0415
    import labels_io  # noqa: PLC0415
    from detector.profiles import model_for_db  # noqa: PLC0415

    reader = labels_io.LabelsReader(labels_dir, bags_dir)
    rows = [r for r in reader.rows if ids is None or r['id'] in set(ids)]
    if not rows:
        print('нет объектов в %s (index.jsonl пуст или ids не найдены)' % labels_dir)
        return [], {}
    if cfg is None:
        from detector import DetectorConfig  # noqa: PLC0415

        cfg = DetectorConfig()

    # Кадры -- UNION по записи: каждый кадр гоняется ОДИН раз, со всеми
    # объектами записи, размеченными на нём.
    by_record = {}
    for row in rows:
        obj = reader.object(row['id'])
        for f in obj.frames:
            by_record.setdefault(row['record'], {}).setdefault(int(f), []).append(row)

    models, stats = {}, {}
    scenes = 0
    try:
        for record in sorted(by_record):
            if record not in models:
                models[record] = model_for_db(reader.db_path(record))
            model = models[record]
            for f in sorted(by_record[record]):
                scene = reader.frame(record, f, include_unlabeled_frames=False)
                scenes += 1
                obstacles, n_aug = detect_on_scene(scene, model, cfg)
                frame_pf = labeling.path_frame(np.asarray(scene['xyz']), model)
                pairs, unmatched = match_objects(scene, obstacles, frame_pf)
                if verbose:
                    matched_ids = [t['obj']['id'] for t, _d in pairs]
                    print('  %-12s кадр %4d: объектов %2d, препятствий %2d, '
                          'сопоставлено %d%s, точек в кадре %d'
                          % (record, f, len(scene['objects']), len(obstacles),
                             len(pairs),
                             (' (%s)' % ', '.join(matched_ids) if matched_ids
                              else ' -- ПРОПУСК' if scene['objects'] else ''),
                             n_aug))
                for t, det in pairs:
                    _bump(stats, t['obj']['class'], bucket_of(t['dist']),
                          gt=1, matched=1)
                    _bump(stats, 'all', bucket_of(t['dist']), gt=1, matched=1,
                          det=1, det_ok=1)
                for obj in (o for o in scene['objects'] if o['frame_labeled']):
                    if _pair_of(obj, pairs) is None:
                        dist = float(np.hypot(obj['center'][0], obj['center'][1]))
                        _bump(stats, obj['class'], bucket_of(dist), gt=1)
                        _bump(stats, 'all', bucket_of(dist), gt=1)
                for det in unmatched:
                    _bump(stats, 'all', bucket_of(det.get('distance_m')), det=1)
    finally:
        reader.close()
    return _report_rows(stats), {'objects': len(rows), 'scenes': scenes}


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
    return [rates(dict(stats[k])) for k in sorted(
        stats, key=lambda k: (stats[k]['class'] != 'all',
                              k[0] if stats[k]['class'] != 'all' else '',
                              order.get(k[1], 99)))]


def render_report(rows, meta, labels_dir):
    lines = []
    lines.append('Recall/precision детектора по разметке: %s' % labels_dir)
    lines.append('кадры -- из frames карточек; кадр = фон без тени + точки объектов '
                 '(labels_io); пороги детектора -- DetectorConfig() по умолчанию')
    lines.append('сопоставление: |dy| <= max(%.1f, длина/2+1.0) м и |du| <= max(%.1f, '
                 'ширина/2+0.6) м в путевых координатах кадра; жадно по confidence'
                 % (TOL_Y_BASE_M, TOL_U_BASE_M))
    lines.append('recall -- по классам объектов; precision -- только строка all '
                 '(детектор классы не выдаёт, знаменатель один на все классы)')
    if meta:
        lines.append('объектов %d, кадров %d' % (meta.get('objects', 0),
                                                 meta.get('scenes', 0)))
    lines.append('')
    lines.append(' класс     | дальн., м | gt  | найдено | recall | det  | det_ok | precision')
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


def write_csv(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'dataset_eval.csv')
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
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--quiet', action='store_true', help='без строк по кадрам')
    args = parser.parse_args(argv)

    root = project_root()
    labels_dir = (args.labels_dir if os.path.isabs(args.labels_dir)
                  else os.path.join(root, args.labels_dir))
    bags_dir = (args.bags if os.path.isabs(args.bags) else os.path.join(root, args.bags))
    out_dir = args.out_dir or os.path.join(root, 'validation', 'out')

    rows, meta = eval_labels(labels_dir, bags_dir, ids=args.ids, verbose=not args.quiet)
    if not rows:
        return 2
    print()
    print(render_report(rows, meta, args.labels_dir))
    path = write_csv(rows, out_dir)
    print()
    print('файл: %s' % path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
