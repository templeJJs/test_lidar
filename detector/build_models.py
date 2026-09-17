"""Регенератор снимка моделей пути `detector/track_models.json`.

Запуск (open3d и `track_geometry` нужны только здесь, в кадровом цикле -- нет):

    python detector/build_models.py                 # все записи for_hackathon
    python detector/build_models.py doubleT_obstacle

Что делает по каждой записи:

  1. зовёт живой `track_geometry.build_track(db, frame)` на НЕСКОЛЬКИХ кадрах
     (`FRAMES`), а не только на нулевом;
  2. берёт ось как середину пары нитей (`profiles.mid_polyline`) -- нити
     измеряются независимо и имеют РАЗНЫЕ наборы узлов (69 и 77 в
     `doubleT_obstacle`), поэтому прежний код снимка с равными длинами молча
     падал на таблицу;
  3. усредняет узлы по кадрам МЕДИАНОЙ и меряет разброс оси между кадрами: на
     `roundT_doubleT` он доходит до 3.3 м (нити на разных кадрах оказываются в
     разных местах -- датчик доворачивался, либо подхватывалась другая нить), а
     на `doubleT_obstacle` -- 0.04 м. Модель по одному кадру (frame=0) на
     неустойчивой записи уводит габарит на метры, поэтому берётся медиана;
  4. продлевает ось до `profiles.SNAPSHOT_Y_FAR_M` (110 м) по наклону хвоста
     (МНК) и кривизне из стен/свода (`rail_mesh.tunnel_axis`);
  5. выбирает `valid_far_m` -- дальний конец участка, на котором габарит можно
     считать: продление `EXT_STABLE_M` (устойчивая запись) или `EXT_UNSTABLE_M`
     (неустойчивая), и дальше сокращает шагами `EXT_STEP_M`, пока на кадрах
     записи есть ложные срабатывания. Кадры проверки берутся РАВНОМЕРНО по всей
     длине записи (`check_frames`, не меньше `CHECK_N`): фиксированный набор
     0/50/100/150/200 пропускал ложные -- замерено на `roundT_doubleT`, где они
     приходятся на 51.5-54.6 м (кадры 68/70/244). Проверка идёт по записям, про
     которые известно, что объектов в них нет (`KNOWN_OCCUPIED` -- исключение);
     `doubleT_obstacle` держит в габарите человека, на нём эта проверка не
     имеет смысла, и он получает продление по устойчивости;
  6. пишет таблицу вместе с блоком `_meta`: чем сгенерировано, до какой
     дальности измерена ось, каким наклоном продлена, какой разброс по кадрам
     (по участку ДАННЫХ; расхождение хвоста формы -- отдельным числом), какая
     проверка дальности пройдена.

Если живой `build_track` недоступен или падает, запись остаётся из прежнего
снимка, и в `_meta.records[name].source` пишется 'snapshot' -- то есть в самой
модели видно, что ось не переизмерена.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from detector.core import DetectorConfig, TrackModel, detect      # noqa: E402
from detector import profiles                                     # noqa: E402

DATA = os.path.join(_ROOT, 'for_hackathon')
TABLES = os.path.join(_HERE, 'track_models.json')

# Кадры, по которым усредняется ось (9 штук через 25: ловят и доворот датчика).
FRAMES = (0, 25, 50, 75, 100, 125, 150, 175, 200)
# Сколько кадров проверяется на ложные срабатывания (равномерно по длине записи).
# Прежний набор (0/50/100/150/200) -- 5 кадров из 201..877: замерено, что ложные
# приходятся НА ДРУГИЕ кадры (roundT_doubleT -- 51.5-54.6 м на кадрах 68/70/244),
# и проверка «мимо набора» их пропускала. Кадры берутся равномерно по ВСЕЙ длине
# записи, не меньше CHECK_N штук.
CHECK_N = 24
# Записи, в которых заведомо ЕСТЬ объект в габарите (человек, кадры 10..65).
# Их нельзя проверять на «ноль ложных срабатываний».
KNOWN_OCCUPIED = ('doubleT_obstacle',)

STABLE_SPREAD_M = 0.15   # разброс оси между кадрами, при котором запись «устойчивая»
EXT_STABLE_M = 25.0      # продление участка детекции на устойчивой записи
EXT_UNSTABLE_M = 15.0    # ... и на неустойчивой (ось врёт на метры)
EXT_STEP_M = 5.0         # шаг сокращения, пока на кадрах записи есть ложные
EXT_MIN_M = -10.0        # предел сокращения: дальше режем уже измеренную ось

NOTE = ('nodes = [[y, x_axis, z_ugr], ...] по возрастанию y; axis_x -- середина '
        'пары нитей, медиана по кадрам FRAMES; ugr_z -- средний верх головок; '
        'ось продлена в −Y до SNAPSHOT_Y_FAR_M по наклону хвоста (МНК) и '
        'кривизне стен/свода; valid_far_m -- докуда ось достоверна для габарита')


SEARCH = True            # сокращать участок по проверке на кадрах (main может снять)


def records() -> list:
    out = []
    for name in sorted(os.listdir(DATA)):
        db = os.path.join(DATA, name, f'{name}_0.db3')
        if os.path.exists(db):
            out.append((name, db))
    return out


def check_frames(db_path: str, count: int = CHECK_N) -> tuple:
    """Кадры проверки на ложные: ВСЕ кадры записи (не меньше `count` равномерных).

    Почему не фиксированный набор 0/50/100/150/200: замерено, что ложные
    срабатывания приходятся на другие кадры (roundT_doubleT -- 51.5-54.6 м на
    кадрах 68/70/244, ещё одно -- 50.7 м на кадре 36), и проверка «мимо набора»
    их пропускала, а участок оставался длинным. Минимум `count` кадров берётся
    равномерно по длине записи (требование «>= 20»), а сама проверка идёт по ВСЕМ
    кадрам: цена -- один прогон детектора по записи (замерено 97 с на все 2488
    кадров корпуса), зато «ноль ложных» доказуем на всей записи, а не на выборке.
    """
    import bag_reader

    bag = bag_reader.BagFrames(db_path, prefetch_ahead=0)
    try:
        n = len(bag)
    finally:
        bag.close()
    return tuple(range(n))


def _frame_polyline(track) -> tuple:
    """(y, x_axis, z_ugr) одной пары нитей из выдачи `build_track`."""
    mesh = (track.get('meta') or {}).get('rail_mesh') or {}
    pair = mesh.get('rails') or {}
    left = pair.get('left', {}).get('nodes')
    right = pair.get('right', {}).get('nodes')
    if left is None or right is None:
        return None
    return profiles.mid_polyline(left, right)


def _origin_far_y(track, kinds=('measured',)) -> float:
    """Ближний к сенсору конец дальнего участка, где узлы определены ИЗМЕРЕНИЕМ.

    Узлы помечены источником (`node_origin`): `measured` -- рельс измерен в бине,
    `sparse` -- доезд за редкими точками, `walls`/`tangent` -- продление по форме
    тоннеля. Возвращает min(y) по узлам указанных источников для ХУДШЕЙ из двух
    нитей (обе нити обязаны быть такими) или None.
    """
    rails = ((track.get('meta') or {}).get('rail_mesh') or {}).get('rails') or {}
    ends = []
    for side in ('left', 'right'):
        r = rails.get(side) or {}
        nodes = r.get('nodes')
        origin = r.get('node_origin')
        if nodes is None:
            return None
        if origin is None:
            ends.append(float(min(v[0] for v in nodes)))
            continue
        ys = [float(v[0]) for v, o in zip(nodes, origin) if str(o) in kinds]
        if not ys:
            return None
        ends.append(min(ys))
    return max(ends) if ends else None


def median_model(name: str, db_path: str) -> tuple:
    """(модель-медиана по кадрам, отчёт); (None, None) -- живой недоступен."""
    import numpy as np

    from track_geometry import build_track

    lines = []
    curvatures = []
    gauges = []
    data_ends = []
    sparse_ends = []
    straight = (0.0, 0.0)
    for frame in FRAMES:
        try:
            track = build_track(str(db_path), frame)
        except Exception:                   # noqa: BLE001 -- чужой модуль/данные
            continue
        try:
            line = _frame_polyline(track)
        except ValueError:
            line = None
        if line is None:
            continue
        lines.append(line)
        curvatures.append(profiles.tunnel_curvature(track))
        end = _origin_far_y(track)
        if end is not None:
            data_ends.append(end)
        end2 = _origin_far_y(track, kinds=('measured', 'sparse'))
        if end2 is not None:
            sparse_ends.append(end2)
        meta = track.get('meta') or {}
        if meta.get('gauge_axis_mm'):
            gauges.append(float(meta['gauge_axis_mm']))
        axis = track.get('axis') or {}
        if frame == FRAMES[0]:
            straight = (float(axis.get('s', 0.0)), float(axis.get('i', 0.0)))
    if not lines:
        return None, None

    lo = max(line[0][0] for line in lines)
    hi = min(line[0][-1] for line in lines)
    if hi - lo < profiles.MIN_MEASURED_M:
        return None, None
    n = max(2, int(round((hi - lo) / profiles.NODE_STEP_M)) + 1)
    grid = np.linspace(lo, hi, n)
    xs = np.vstack([np.interp(grid, line[0], line[1]) for line in lines])
    zs = np.vstack([np.interp(grid, line[0], line[2]) for line in lines])
    spread = xs.max(axis=0) - xs.min(axis=0)
    # РАЗБРОС ОСИ -- ПО ИЗМЕРЕННОМУ УЧАСТКУ РЕЛЬСА (см. _origin_far_y). Дальше идут
    # доезд за редкими точками (sparse) и продление по форме тоннеля (walls/tangent):
    # у разных кадров они расходятся (замерено на doubleT_obstacle: 0.16 м на −46 м
    # в разреженной полосе и 0.32 м на −85 м в форме) -- это свойство ТРЕКИНГА и
    # МОДЕЛИ продолжения, а не измеренной оси, и в «устойчивость записи» оно не
    # входит. Расхождение полосы и формы отдаётся отдельными числами.
    y_data = max(data_ends) if data_ends else hi
    if not (lo < y_data < hi - 1.0):
        y_data = hi
    m_data = grid >= y_data - 1e-9
    spread_data = spread[m_data] if bool(np.any(m_data)) else spread
    y_sparse = max(sparse_ends) if sparse_ends else lo
    m_sparse = grid >= y_sparse - 1e-9
    model = TrackModel(
        name=name,
        gauge_m=(sum(gauges) / len(gauges) / 1000.0) if gauges else 1.593,
        y_nodes=tuple(float(v) for v in grid),
        axis_x_m=tuple(float(v) for v in np.median(xs, axis=0)),
        ugr_z_m=tuple(float(v) for v in np.median(zs, axis=0)),
        axis_straight_s=straight[0],
        axis_straight_i=straight[1],
        source='live-median',
    )
    report = {
        'frames_used': len(lines),
        'measured_y_m': [round(float(hi), 2), round(float(lo), 2)],
        'data_far_y_m': round(float(y_data), 2),
        'spread_max_m': round(float(spread_data.max()), 3),
        'spread_p95_m': round(float(np.percentile(spread_data, 95)), 3),
        'spread_form_max_m': round(float(spread[~m_data].max()), 3)
        if bool(np.any(~m_data)) else 0.0,
        'curvature': round(float(np.median(curvatures)), 6) if curvatures else 0.0,
        'gauge_m': round(float(model.gauge_m), 4),
    }
    return model, report


def check_range(model: TrackModel, db_path: str, frames) -> dict:
    """Сколько занятых бинов даёт модель на кадрах записи (для «пустых» -- 0)."""
    import bag_reader

    cfg = DetectorConfig()
    bag = bag_reader.BagFrames(db_path, prefetch_ahead=0)
    bins = []
    obstacles = []
    event_frames = []
    nearest = None
    try:
        for frame in frames:
            result = detect(bag[frame].xyz, model, cfg)
            bins.append(result.bins_blocked)
            if result.obstacles:
                event_frames.append(int(frame))
                if nearest is None or result.obstacles[0].distance_m < nearest:
                    nearest = float(result.obstacles[0].distance_m)
            obstacles += [(round(o.distance_m, 1), round(o.u_m, 2)) for o in result.obstacles]
    finally:
        bag.close()
    return {'n_frames': len(bins), 'bins': bins, 'event_frames': event_frames,
            'nearest_fp_m': nearest, 'bins_total': int(sum(bins)),
            'obstacles': obstacles}


def build_one(name: str, db_path: str, old: dict, search: bool = SEARCH) -> tuple:
    """(запись таблицы, отчёт) для одной записи."""
    report = {'source': 'snapshot', 'axis_far_m': None, 'slope_x': None,
              'slope_z': None, 'curvature': None, 'gauge_m': None, 'spread_max_m': None,
              'straight': None, 'valid_far_m': None, 'extension_m': None,
              'check': None, 'note': ''}
    measured = None
    try:
        measured, live_report = median_model(name, db_path)
        if measured is not None:
            report.update(live_report)
            report['source'] = f'build_track(frame={FRAMES[0]}..{FRAMES[-1]}, медиана)'
    except Exception as exc:               # noqa: BLE001 -- чужой модуль/данные
        report['note'] = f'живой build_track недоступен ({exc.__class__.__name__}: {exc})'

    if measured is None:
        if not old:
            raise RuntimeError(f'{name}: нет ни живой модели, ни прежнего снимка')
        measured = TrackModel.from_dict(old)
        report['source'] = 'snapshot'
        report['curvature'] = 0.0
        report['straight'] = (measured.axis_straight_s, measured.axis_straight_i)
        report['gauge_m'] = measured.gauge_m
    report['straight'] = (measured.axis_straight_s, measured.axis_straight_i)

    y_meas_far = float(min(measured.y_nodes))
    sx = profiles.tail_slope(measured.y_nodes, measured.axis_x_m, profiles.TAIL_M)
    sz = profiles.tail_slope(measured.y_nodes, measured.ugr_z_m, profiles.Z_TAIL_M)
    report['axis_far_m'] = round(y_meas_far, 3)
    report['slope_x'] = round(sx, 6)
    report['slope_z'] = round(sz, 6)

    extended = profiles.extend_polyline(measured, profiles.SNAPSHOT_Y_FAR_M,
                                       curvature=report['curvature'] or 0.0)

    stable = (report['spread_max_m'] is not None
              and report['spread_max_m'] <= STABLE_SPREAD_M)
    extension = EXT_STABLE_M if stable else EXT_UNSTABLE_M
    occupied = name in KNOWN_OCCUPIED
    frames = check_frames(db_path)
    report['check_frames_total'] = len(frames)
    report['check_frames_uniform'] = (list(frames[::max(1, len(frames) // CHECK_N)][:CHECK_N])
                                      if frames else [])
    info = None
    if search:
        while True:
            candidate = dataclasses.replace(extended, valid_far_m=y_meas_far - extension)
            info = check_range(candidate, db_path, frames)
            if occupied or info['bins_total'] == 0 or extension <= EXT_MIN_M:
                break
            # СОКРАЩЕНИЕ УЧАСТКА: сразу ЗА ближайшее ложное (запас NODE_STEP_M),
            # кратно шагу EXT_STEP_M, а не на один шаг за проход -- ложное на
            # 50.7 м при участке 50.75 м иначе заставляло бы отыгрывать по 5 м за
            # проход; участок обязан быть максимальным из чистых
            need = (y_meas_far + float(info['nearest_fp_m'] or 0.0)
                    - profiles.NODE_STEP_M)
            cut = math.floor(need / EXT_STEP_M) * EXT_STEP_M
            extension = max(EXT_MIN_M, min(extension - EXT_STEP_M, cut))
    else:
        info = check_range(dataclasses.replace(extended,
                                               valid_far_m=y_meas_far - extension),
                           db_path, frames)
    if not occupied and info['bins_total']:
        info['note'] = ('участок сокращён до предела, ложные на кадрах '
                        'проверки остались')
    report['check'] = info
    report['extension_m'] = extension
    report['valid_far_m'] = round(y_meas_far - extension, 2)

    entry = {
        'gauge_m': round(float(extended.gauge_m), 5),
        'axis_straight_s': round(float(extended.axis_straight_s), 9),
        'axis_straight_i': round(float(extended.axis_straight_i), 9),
        'axis_extension_m': 0.0,
        'valid_far_m': report['valid_far_m'],
        'y_nodes': [round(float(v), 4) for v in extended.y_nodes],
        'axis_x_m': [round(float(v), 4) for v in extended.axis_x_m],
        'ugr_z_m': [round(float(v), 4) for v in extended.ugr_z_m],
    }
    return entry, report


def main(argv: list) -> int:
    search = True
    only = set(argv) or None
    if '--no-search' in argv:
        search = False
        only = only - {'--no-search'} or None
    table = profiles._table()
    out = {'_source': 'track_geometry.build_track(db, frame) -> meta.rail_mesh.rails[side].nodes',
           '_note': NOTE,
           '_meta': {'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                     'generator': 'detector/build_models.py',
                     'frames': list(FRAMES),
                     'check_n': CHECK_N,
                     'snapshot_y_far_m': profiles.SNAPSHOT_Y_FAR_M,
                     'tail_m': profiles.TAIL_M,
                     'z_tail_m': profiles.Z_TAIL_M,
                     'stable_spread_m': STABLE_SPREAD_M,
                     'records': {}}}
    for name, db in records():
        if only is not None and name not in only:
            entry = table.get(name) or {}
            if entry:
                out[name] = entry
                if name in (table.get('_meta', {}).get('records') or {}):
                    out['_meta']['records'][name] = table['_meta']['records'][name]
            continue
        entry, report = build_one(name, db, dict(table.get(name) or {}), search=search)
        out[name] = entry
        out['_meta']['records'][name] = report
        check = report['check'] or {}
        print('%-42s %-34s измерена до %7.2f  продлена до %7.2f  s=%.4f/%.4f  '
              'разброс %.3f м (хвост формы %.3f)  участок до %7.2f  '
              'проверено кадров %d, ложных кадров %d%s'
              % (name, report['source'], report['axis_far_m'],
                 entry['y_nodes'][0], report['slope_x'], report['slope_z'],
                 report['spread_max_m'] if report['spread_max_m'] is not None else float('nan'),
                 report.get('spread_form_max_m') or 0.0,
                 report['valid_far_m'], check.get('n_frames', 0),
                 len(check.get('event_frames') or []),
                 '' if name in KNOWN_OCCUPIED else
                 (', ЛОЖНЫЕ %s' % check.get('obstacles') if check.get('bins_total')
                  else ' (чисто)')))
        if check.get('note'):
            print('    %s' % check['note'])
        if report['note']:
            print('    %s' % report['note'])
    with open(TABLES, 'w', encoding='utf-8') as handle:
        json.dump(out, handle, ensure_ascii=False, indent=1)
        handle.write('\n')
    print('записано: %s' % TABLES)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))