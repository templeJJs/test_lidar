"""FP/час: ложные срабатывания детектора габарита на ВСЁМ корпусе записей.

Прошлые проверки (`docs/detector-results.md`, раздел 1) смотрели 30 кадров из
2488 и говорили «0 занятых бинов». Ноль событий на маленькой выборке сам по себе
ничего не доказывает: важна не абсолютная цифра, а ЧАСТОТА, нормированная на
время наблюдения. Этот модуль прогоняет `detector.core.detect` по каждому кадру
всех шести записей `for_hackathon/*/*.db3`, считает события и переводит их в
ложные срабатывания в час (FP/ч).

Нормы взяты из Tagiew, arXiv 2307.02586:
  <= 1e-2 FP/ч -- предупреждение, <= 1e-4 FP/ч -- экстренное торможение.

Событие = кадр, в котором детектор выдал хотя бы одно препятствие (тревога).
Дополнительно считаются объекты и занятые бины -- чтобы цифра не зависела от
того, как считать событие.

НЕЗАВИСИМОСТЬ СОБЫТИЙ. Кадры записи идут подряд (в этих bag-ах ~35 тыс. кадр/ч),
поэтому 46 событий человека в `doubleT_obstacle` -- это ДВА непрерывных эпизода
(кадры 13..31 и 37..63), а не 46 независимых наблюдений: делить их на часы и
называть частотой нельзя. Поэтому в отчёте рядом с «кадров с событием» стоит
число НЕПРЕРЫВНЫХ ЭПИЗОДОВ (события, разнесённые более чем на `EPISODE_GAP`
кадров, -- разные эпизоды), и FP/ч считается по обоим: по кадрам (как было) и по
эпизодам (честная частота тревог).

При нуле событий частота не измерена, а ОГРАНИЧЕНА СВЕРХУ: для пуассоновского
потока 0 событий за T часов даёт 95 %-ю верхнюю границу lambda < -ln(0.05)/T ~
3/T («правило трёх»). Отсюда и ответ на вопрос «что доказуемо»: чтобы верхняя
граница опустилась до 1e-2/ч, нужно >= 300 ч наблюдений, до 1e-4/ч -- >= 30000 ч.

Запуск (из корня проекта):

    python -m validation.fp_per_hour                 # все 6 записей, все кадры
    python -m validation.fp_per_hour --limit 50      # быстрый прогон по 50 кадрам

Пишет `validation/out/fp_per_hour_summary.csv` (по записям) и
`validation/out/fp_per_hour_events.csv` (каждое найденное препятствие).

`detector/**` правится параллельно, поэтому прогон держит ЕДИНЫЙ снимок: модели
всех записей загружаются до первого кадра, а в отчёте печатаются md5 файлов
`detector/` на начало прогона (после -- проверка, не менялись ли они). Прогон
сам по себе ~100 с и никого не останавливает.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import sys
import time

# Нормы статьи и статистика «правила трёх».
TARGET_WARNING_FP_H = 1e-2
TARGET_BRAKE_FP_H = 1e-4
ALPHA = 0.05                       # значимость верхней границы (95 %)
Z_ONE_SIDED = -math.log(ALPHA)     # 2.9957 -- сколько событий «не видно» в нуле
EPISODE_GAP_FRAMES = 1             # разрыв кадров, рвущий эпизод: > 1 = не подряд

RECORDS = (
    'doubleT_obstacle',
    'doubleT_platform',
    'roundT_doubleT',
    'roundT_pressureGate_roundT',
    'roundT_squareT_pressureGate_squareT',
    'squareT_platform_squareT_switch',
)

DURATION_RE = re.compile(r'^  duration:\s*\n\s+nanoseconds:\s*(\d+)', re.M)

# Файлы детектора, хеши которых печатаются в отчёте: детектор правится
# параллельно, и без ревизии числа отчёта невозможно привязать к коду.
REVISION_FILES = ('core.py', 'profiles.py', 'track_models.json')


def _md5(path: str) -> str:
    import hashlib  # noqa: PLC0415

    try:
        with open(path, 'rb') as handle:
            return hashlib.md5(handle.read()).hexdigest()
    except OSError:
        return '?'


def detector_revision() -> dict:
    """md5 файлов `detector/` на текущий момент -- паспорт отчёта."""
    import detector  # noqa: PLC0415

    base = os.path.dirname(os.path.abspath(detector.__file__))
    return {name: _md5(os.path.join(base, name)) for name in REVISION_FILES}


def project_root() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def bag_db_paths(root=None, names=RECORDS):
    """[(имя записи, путь к .db3)] по каталогам `for_hackathon/<имя>/`."""
    root = project_root() if root is None else root
    out = []
    for name in names:
        found = sorted(glob.glob(os.path.join(root, 'for_hackathon', name, '*.db3')))
        if found:
            out.append((name, found[0]))
    return out


def bag_duration_h(db_path: str):
    """Длительность записи из `metadata.yaml`, часы (None, если файла нет).

    Берётся ВЕРХНИЙ `duration` -- суммарное время бага; пер-файловый `files[].duration`
    лежит глубже по отступу и в регулярку не попадает.
    """
    meta = os.path.join(os.path.dirname(db_path), 'metadata.yaml')
    try:
        with open(meta, encoding='utf-8') as handle:
            text = handle.read()
    except OSError:
        return None
    found = DURATION_RE.search(text)
    if not found:
        return None
    return int(found.group(1)) / 1e9 / 3600.0


def episodes_from_frames(frame_indices, max_gap: int = EPISODE_GAP_FRAMES) -> list:
    """Непрерывные эпизоды событий: [(первый кадр, последний кадр, кадров), ...].

    Кадры записи идут подряд, поэтому несколько подряд идущих событий -- ОДИН
    эпизод (одно препятствие в сцене), а не столько независимых наблюдений,
    сколько кадров. События, разнесённые более чем на `max_gap` кадров,
    считаются разными эпизодами.
    """
    frames = sorted({int(v) for v in frame_indices})
    if not frames:
        return []
    out = [[frames[0], frames[0], 1]]
    for value in frames[1:]:
        if value - out[-1][1] <= int(max_gap):
            out[-1][1] = value
            out[-1][2] += 1
        else:
            out.append([value, value, 1])
    return [tuple(v) for v in out]


def upper_bound_95(hours: float, events: int = 0, alpha: float = ALPHA) -> float:
    """95 %-я верхняя граница частоты событий (событий/час).

    Для нуля событий -- классическое правило трёх (`-ln(alpha)/T`). Общий случай
    (события есть) -- точная граница Пуассона по хи-квадрат; реализована через
    ряд, чтобы не тянуть scipy.
    """
    if hours <= 0:
        return float('inf')
    if events <= 0:
        return -math.log(alpha) / hours
    # Граница Пуассона: lambda_ub = chi2(1-alpha, 2*(n+1)) / (2*T).
    return _chi2_ppf(1.0 - alpha, 2 * (events + 1)) / (2.0 * hours)


def _chi2_ppf(p: float, df: int) -> float:
    """Квантиль хи-квадрат для целых df -- метод Ньютона по гамма-функции."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return float('inf')
    # Начальное приближение Вилсона-Хилферти, дальше -- бисекция по CDF.
    lo, hi = 0.0, max(10.0, float(df) * 5.0 + 50.0)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _chi2_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _chi2_cdf(x: float, df: int) -> float:
    """CDF хи-квадрат с целыми df: P = 1 - sum_{k=0}^{df/2-1} e^-z z^k / k!, z=x/2."""
    if x <= 0.0:
        return 0.0
    z = 0.5 * x
    k_max = df // 2
    term = math.exp(-z)
    total = term
    for k in range(1, k_max):
        term *= z / k
        total += term
    return max(0.0, min(1.0, 1.0 - total))


def hours_to_prove(target_fp_h: float, alpha: float = ALPHA) -> float:
    """Сколько часов наблюдений нужно, чтобы при нуле событий доказать норму."""
    return -math.log(alpha) / target_fp_h


def scan_bag(name, db_path, model=None, cfg=None, limit=None, verbose=True,
             progress_seconds=15.0):
    """Прогнать детектор по всем (или первым `limit`) кадрам одной записи.

    Возвращает dict со счётчиками; падение отдельного кадра не останавливает
    прогон -- оно попадает в `errors` (счётчик), а первое сообщение об ошибке
    печатается сразу, чтобы диагностика не терялась.
    """
    import bag_reader  # noqa: PLC0415

    from detector.core import detect  # noqa: PLC0415
    from detector.profiles import model_for_db  # noqa: PLC0415

    if model is None:
        model = model_for_db(db_path)
    if cfg is None:
        from detector import DetectorConfig  # noqa: PLC0415

        cfg = DetectorConfig()

    frames = bag_reader.BagFrames(db_path, prefetch_ahead=8, cache_size=24)
    n_total = len(frames)
    n_scan = n_total if limit is None else min(n_total, int(limit))

    entry = {
        'bag': name, 'db_path': db_path, 'frames': 0, 'frames_total': int(n_total),
        'frames_with_event': 0, 'episodes': 0, 'objects': 0, 'bins_blocked': 0,
        'duration_h': bag_duration_h(db_path), 'span_h': None,
        'errors': 0, 'seconds': 0.0,
        'model_name': getattr(model, 'name', ''), 'model_source': getattr(model, 'source', ''),
        'axis_range_m': None, 'events': [], 'event_frames': [],
    }
    if n_total:
        entry['span_h'] = (frames.timestamp(n_total - 1) - frames.timestamp(0)) / 1e9 / 3600.0

    t_start = time.time()
    last_print = 0.0
    try:
        for i in range(n_scan):
            frames.want(i)
            try:
                cloud = frames[i].xyz
            except Exception as exc:                    # noqa: BLE001 -- битые кадры не роняют прогон
                entry['errors'] += 1
                if entry['errors'] == 1:
                    print('   !! %s: чтение кадра %d: %s: %s'
                          % (name, i, type(exc).__name__, exc), flush=True)
                continue
            entry['frames'] += 1
            try:
                result = detect(cloud, model=model, cfg=cfg)
            except Exception as exc:                    # noqa: BLE001 -- API/данные меняются параллельно
                entry['errors'] += 1
                if entry['errors'] == 1:
                    print('   !! %s: detect кадра %d: %s: %s'
                          % (name, i, type(exc).__name__, exc), flush=True)
                continue
            entry['axis_range_m'] = tuple(round(float(v), 2) for v in result.axis_range_m)
            obstacles = result.obstacles or []
            entry['bins_blocked'] += int(getattr(result, 'bins_blocked', 0))
            if obstacles:
                entry['frames_with_event'] += 1
                entry['event_frames'].append(int(i))
                entry['objects'] += len(obstacles)
                for ob in obstacles:
                    row = ob.to_dict() if hasattr(ob, 'to_dict') else dict(ob)
                    row['bag'] = name
                    row['frame'] = int(i)
                    row['frame_ts'] = int(frames.timestamp(i))
                    entry['events'].append(row)

            now = time.time()
            if verbose and (now - last_print >= progress_seconds or i == n_scan - 1):
                last_print = now
                done = i + 1
                frac = done / float(max(n_scan, 1))
                elapsed = now - t_start
                eta = elapsed / frac - elapsed if frac > 0 else 0.0
                print('   %-40s кадр %4d/%-4d (%3.0f %%) событий %d, объектов %d, '
                      '%5.1f с, осталось ~%5.1f с'
                      % (name, done, n_scan, 100.0 * frac, entry['frames_with_event'],
                         entry['objects'], elapsed, eta), flush=True)
    finally:
        frames.close()
    entry['seconds'] = time.time() - t_start
    entry['episodes'] = len(episodes_from_frames(entry['event_frames']))
    return entry


def summarize(entries):
    """Строка на запись + итог: часов, событий-кадров, эпизодов, FP/ч обоих видов."""
    rows = []
    for e in entries:
        hours = e['duration_h'] if e['duration_h'] else e['span_h']
        row = dict(e)
        row['hours'] = float(hours) if hours else float('nan')
        row['fp_per_hour'] = (e['frames_with_event'] / row['hours']) if row['hours'] else float('nan')
        row['episodes_per_hour'] = (e['episodes'] / row['hours']) if row['hours'] else float('nan')
        row['objects_per_hour'] = (e['objects'] / row['hours']) if row['hours'] else float('nan')
        row['bins_per_hour'] = (e['bins_blocked'] / row['hours']) if row['hours'] else float('nan')
        rows.append(row)

    hours_total = sum(r['hours'] for r in rows if r['hours'] == r['hours'])
    frames = sum(r['frames'] for r in rows)
    events = sum(r['frames_with_event'] for r in rows)
    episodes = sum(r['episodes'] for r in rows)
    objects = sum(r['objects'] for r in rows)
    bins = sum(r['bins_blocked'] for r in rows)
    total = {
        'bag': 'ИТОГО', 'frames': frames, 'frames_total': sum(r['frames_total'] for r in rows),
        'frames_with_event': events, 'episodes': episodes, 'objects': objects,
        'bins_blocked': bins,
        'hours': hours_total,
        'fp_per_hour': events / hours_total if hours_total else float('nan'),
        'episodes_per_hour': episodes / hours_total if hours_total else float('nan'),
        'objects_per_hour': objects / hours_total if hours_total else float('nan'),
        'bins_per_hour': bins / hours_total if hours_total else float('nan'),
        'errors': sum(r['errors'] for r in rows),
        'seconds': sum(r['seconds'] for r in rows),
    }
    return rows, total


def render_report(rows, total, events, revision=None):
    """Текст: таблица по записям, верхняя граница, вывод о доказуемом.

    `revision` -- хеши `detector/`, снятые в НАЧАЛЕ прогона. Прогон держит единый
    снимок: код детектора импортируется один раз, а модели всех записей
    загружаются до первого кадра, поэтому правки `detector/**` во время прогона
    на результат не влияют, и в отчёте стоит именно тот код, что измерен.
    """
    lines = []
    lines.append('FP/час: ложные срабатывания детектора габарита по всем кадрам 6 записей')
    lines.append('событие = кадр хотя бы с одним препятствием; пороги -- DetectorConfig() '
                 'по умолчанию')
    lines.append('непрерывный эпизод = события подряд (разрыв > %d кадра рвёт эпизод): '
                 'кадры записи идут подряд (~%.0f кадр/ч), поэтому несколько событий '
                 'подряд -- ОДНО препятствие, а не столько независимых наблюдений'
                 % (EPISODE_GAP_FRAMES, _frames_per_hour(rows)))
    lines.append('ось -- сохранённая полилиния (detector/track_models.json, source=%s); '
                 'T -- metadata.yaml'
                 % (rows[0]['model_source'] if rows and rows[0].get('model_source') else '?'))
    rev = revision or detector_revision()
    lines.append('ревизия detector/ на начало прогона: core.py %s, profiles.py %s, '
                 'track_models.json %s'
                 % tuple(rev[name][:12] for name in REVISION_FILES))
    lines.append('')
    lines.append(' запись                        | кадр  | событий | эпизод | объект | бин  | '
                 'T, ч    | FP/ч (кадры) | FP/ч (эпизоды) | FP/ч (бины)')
    lines.append('-' * 150)
    for r in rows:
        lines.append(' %-29s | %5d | %7d | %6d | %6d | %4d | %7.4f | %12s | %14s | %11s'
                     % (r['bag'], r['frames'], r['frames_with_event'], r['episodes'],
                        r['objects'], r['bins_blocked'], r['hours'],
                        _fmt(r['fp_per_hour']), _fmt(r['episodes_per_hour']),
                        _fmt(r['bins_per_hour'])))
    lines.append('-' * 150)
    lines.append(' %-29s | %5d | %7d | %6d | %6d | %4d | %7.4f | %12s | %14s | %11s'
                 % (total['bag'], total['frames'], total['frames_with_event'],
                    total['episodes'], total['objects'], total['bins_blocked'],
                    total['hours'], _fmt(total['fp_per_hour']),
                    _fmt(total['episodes_per_hour']), _fmt(total['bins_per_hour'])))
    if total['errors']:
        lines.append('')
        lines.append('ошибок при прогоне: %d (прогон не остановлен, первая ошибка -- в логе '
                     'выше)' % total['errors'])
    limited = [r for r in rows if r['frames'] < r['frames_total']]
    if limited:
        lines.append('')
        lines.append('ВНИМАНИЕ: прогон частичный (--limit), просканировано %d из %d кадров: '
                     'T и FP/ч НЕ показательны, это только проверка работоспособности'
                     % (total['frames'], total['frames_total']))
    lines.append('')
    episodes = [(r['bag'], episodes_from_frames(r['event_frames'])) for r in rows]
    if any(spans for _n, spans in episodes):
        lines.append('непрерывные эпизоды событий по записям (кадры записи):')
        for name, spans in episodes:
            if not spans:
                continue
            lines.append('  %-29s %s' % (name, ', '.join(
                '%d..%d (%d кадров)' % (a, b, n) for a, b, n in spans)))
        lines.append('')

    # --- верхняя граница и доказуемость -----------------------------------
    hours = total['hours']
    ub_frames = upper_bound_95(hours, total['frames_with_event'])
    ub_episodes = upper_bound_95(hours, total['episodes'])
    ub_bins = upper_bound_95(hours, total['bins_blocked'])
    need_warn = hours_to_prove(TARGET_WARNING_FP_H)
    need_brake = hours_to_prove(TARGET_BRAKE_FP_H)
    lines.append('верхняя граница частоты (95 %%, Пуассон):')
    lines.append('  событий-кадров %d, эпизодов %d за T = %.4f ч --> FP/ч <= %.1f (кадры), '
                 '%.1f (эпизоды), %.1f (бины)'
                 % (total['frames_with_event'], total['episodes'], hours, ub_frames,
                    ub_episodes, ub_bins))
    if total['frames_with_event'] == 0:
        lines.append('  при нуле событий это правило трёх: FP/ч <= -ln(0.05)/T = %.3f/T = %.1f'
                     % (Z_ONE_SIDED, Z_ONE_SIDED / hours))
        lines.append('  смысл: 0 событий за %.4f ч совместим с истинной частотой вплоть до '
                     '~%.0f/ч; нормы 1e-2 и 1e-4/ч таким прогоном не подтверждаются.'
                     % (hours, ub_frames))
    else:
        lines.append('  события ЕСТЬ, поэтому частота измерена, а не ограничена сверху; '
                     'правило трёх (при нуле оно дало бы %.1f/ч) не применяется.'
                     % (Z_ONE_SIDED / hours))
        lines.append('  по ЭПИЗОДАМ частота ниже, чем по кадрам: %.1f против %.1f/ч -- '
                     'события идут сериями, и кадровая частота завышает её в %.1f раза.'
                     % (total['episodes_per_hour'], total['fp_per_hour'],
                        total['fp_per_hour'] / total['episodes_per_hour']
                        if total['episodes_per_hour'] else float('nan')))
    lines.append('')
    lines.append('сколько часов нужно, чтобы ДОКАЗАТЬ норму (нулём событий, 95 %%):')
    lines.append('  предупреждение  1e-2 FP/ч: T >= %.3f/1e-2 = %.0f ч (%.1f суток); '
                 'у нас %.4f ч -- %.4f %% нужного'
                 % (Z_ONE_SIDED, need_warn, need_warn / 24.0, hours, 100.0 * hours / need_warn))
    lines.append('  торможение      1e-4 FP/ч: T >= %.3f/1e-4 = %.0f ч (%.1f года); '
                 'у нас %.4f ч -- %.5f %% нужного'
                 % (Z_ONE_SIDED, need_brake, need_brake / 24.0 / 365.0, hours,
                    100.0 * hours / need_brake))
    lines.append('  то есть текущие %.4f ч не могут доказать ни одну из норм В ПРИНЦИПЕ: '
                 'нужно в %.0f раз больше времени для 1e-2/ч и в %.0f раз -- для 1e-4/ч.'
                 % (hours, need_warn / hours, need_brake / hours))
    lines.append('  (если при этом ещё и целевая частота 1e-2/ч, то %.1f/ч в измерении -- '
                 'в ~%.0f раз выше неё.)'
                 % (total['fp_per_hour'], total['fp_per_hour'] / TARGET_WARNING_FP_H))
    lines.append('')

    # --- разбор событий ---------------------------------------------------
    lines.append('найденные препятствия: %d' % len(events))
    if events:
        lines.append(' фрейм | запись                     | дальн., м | y, м   | u, м  | '
                     'h_min..h_max, м | span, м | точек | conf')
        for ev in sorted(events, key=lambda v: (v['bag'], v['frame'])):
            lines.append(' %5d | %-27s | %9.2f | %6.2f | %5.2f | %5.2f..%5.2f | %7.2f | %5d | %s'
                         % (ev['frame'], ev['bag'], ev.get('distance_m', float('nan')),
                            ev.get('y_m', float('nan')), ev.get('u_m', float('nan')),
                            ev.get('h_min_m', float('nan')), ev.get('h_max_m', float('nan')),
                            ev.get('span_m', float('nan')), ev.get('points', 0),
                            ev.get('confidence', '')))
    else:
        lines.append('  ни одного -- по всем записям детектор молчит.')
    return '\n'.join(lines)


def _fmt(value):
    if value != value:
        return '--'
    if value == 0:
        return '0'
    return '%.3g' % value


def _frames_per_hour(rows):
    """Частота кадров записей: сумма кадров / сумма часов (по metadata.yaml)."""
    hours = sum(r['hours'] for r in rows if r['hours'] == r['hours'])
    frames = sum(r['frames'] for r in rows)
    return frames / hours if hours else float('nan')


def write_outputs(rows, total, events, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    summ_path = os.path.join(out_dir, 'fp_per_hour_summary.csv')
    fields = ['bag', 'frames', 'frames_total', 'frames_with_event', 'episodes', 'objects',
              'bins_blocked', 'hours', 'fp_per_hour', 'episodes_per_hour',
              'objects_per_hour', 'bins_per_hour',
              'errors', 'seconds', 'model_name', 'model_source', 'axis_range_m']
    with open(summ_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in list(rows) + [total]:
            writer.writerow(row)
    ev_path = os.path.join(out_dir, 'fp_per_hour_events.csv')
    ev_fields = ['bag', 'frame', 'frame_ts', 'distance_m', 'y_m', 'u_m', 'x_m',
                 'h_min_m', 'h_max_m', 'span_m', 'points', 'cells', 'confidence',
                 'bin_index', 'u_lo_m', 'u_hi_m']
    with open(ev_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=ev_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(events)
    return summ_path, ev_path


def main(argv=None):
    parser = argparse.ArgumentParser(description='FP/час по всем кадрам 6 записей')
    parser.add_argument('--bags', nargs='*', default=None, help='имена записей (по умолчанию все)')
    parser.add_argument('--limit', type=int, default=None, help='кадров на запись (для быстрого прогона)')
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--quiet', action='store_true', help='только итоговая таблица')
    args = parser.parse_args(argv)

    root = project_root()
    out_dir = args.out_dir or os.path.join(root, 'validation', 'out')
    bags = bag_db_paths(root, args.bags or RECORDS)
    if not bags:
        print('нет записей for_hackathon/*/*.db3')
        return 2

    verbose = not args.quiet
    if verbose:
        print('FP/час: прогон детектора по всем кадрам (%d записей)' % len(bags))

    # Единый снимок: хеши детектора и модели ВСЕХ записей берутся до первого
    # кадра. Код детектора Python кэширует сам, а `track_models.json` без этого
    # перечитывался бы на каждой записи и прогон склеился бы из двух ревизий оси
    # (замерено: правка снимка во время прогона обнуляла события в roundT_doubleT).
    revision = detector_revision()
    from detector.profiles import model_for_db  # noqa: PLC0415

    models = {}
    for name, db in bags:
        try:
            models[name] = model_for_db(db)
        except Exception as exc:                        # noqa: BLE001 -- модель чужого модуля
            print('  !! модель %s не построена: %s -- откат на общую внутри detect'
                  % (name, type(exc).__name__), flush=True)
            models[name] = None
    if verbose:
        for name, model in models.items():
            print('  модель %-40s %s / %s'
                  % (name, getattr(model, 'name', '?'), getattr(model, 'source', '?')),
                  flush=True)

    entries = []
    t_start = time.time()
    for name, db in bags:
        if verbose:
            print('  %s: %s' % (name, os.path.basename(db)), flush=True)
        try:
            entries.append(scan_bag(name, db, model=models.get(name), limit=args.limit,
                                    verbose=verbose))
        except Exception as exc:                        # noqa: BLE001 -- одну запись терять нельзя
            print('  !! %s упала целиком: %s: %s -- продолжаю' % (name, type(exc).__name__, exc),
                  flush=True)
            entries.append({'bag': name, 'db_path': db, 'frames': 0, 'frames_total': 0,
                            'frames_with_event': 0, 'episodes': 0, 'objects': 0,
                            'bins_blocked': 0,
                            'duration_h': bag_duration_h(db), 'span_h': None,
                            'errors': 1, 'seconds': 0.0, 'model_name': '',
                            'model_source': '', 'axis_range_m': None, 'events': [],
                            'event_frames': []})
    elapsed = time.time() - t_start

    rows, total = summarize(entries)
    events = [ev for e in entries for ev in e['events']]
    text = render_report(rows, total, events, revision=revision)
    paths = write_outputs(rows, total, events, out_dir)

    print()
    print(text)
    print()
    print('кадров %d, событий %d, эпизодов %d, T = %.4f ч, FP/ч = %s (кадры), %s '
          '(эпизоды); прогон %.1f с'
          % (total['frames'], total['frames_with_event'], total['episodes'],
             total['hours'], _fmt(total['fp_per_hour']),
             _fmt(total['episodes_per_hour']), elapsed))
    print('файлы: %s' % ', '.join(os.path.basename(p) for p in paths))
    after = detector_revision()
    changed = [name for name in REVISION_FILES if after[name] != revision[name]]
    print('detector/** после прогона %s (снимок отчёта -- хеши начала прогона)'
          % ('изменён: ' + ', '.join(changed) if changed else 'не менялся'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())