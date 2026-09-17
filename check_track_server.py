"""Проверка работы СЕРВЕРА вьюера (не модуля): то, что реально видит браузер.

  python check_track_server.py [порт]

Три части:
 1) /track по всем шести записям: дальность ленты, доля предсказанных узлов, меш пола
    против ленты, превышение верха над полом ПОД ОСЬЮ рельса (|Δx| <= 0.15, максимум z
    меша в полосе — замер кольцом попадает в дно лотка и врёт);
 2) метка рельса на ХОЛОДНОМ кадре: сначала /frame (mode=zones), потом /labels; в
    полосе рельса должно быть 100 % метки rail(0) и 0 % middle(4). Формат заголовка —
    HEADER = '<4sBBHI': magic, version, kind, axis_frame, n;
 3) время /track: холодный, не из кэша, из кэша.

Почему именно так: сервер молча игнорирует `?record=` (нужен `?bag=`), а проба
«превышение над полом» кольцом ±1 м попадает в дно лотка (даёт 0.26-0.65 м вместо
0.12-0.17) — см. docs/track-geometry.md.
"""
from __future__ import annotations

import json
import struct
import sys
import time
import urllib.request

import numpy as np

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
BASE = 'http://127.0.0.1:%d' % PORT
RECS = ['doubleT_obstacle', 'doubleT_platform', 'roundT_doubleT',
        'roundT_pressureGate_roundT', 'roundT_squareT_pressureGate_squareT',
        'squareT_platform_squareT_switch']
RAIL_LABEL, MIDDLE_LABEL = 0, 4


def get(path, raw=False, timeout=900):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        body = r.read()
    return body if raw else json.loads(body)


def parse(buf):
    magic, ver, kind, axis_frame, n = struct.unpack_from('<4sBBHI', buf, 0)
    pos = np.frombuffer(buf[12:12 + n * 12], dtype='<f4').reshape(n, 3)
    lab = np.frombuffer(buf[12 + n * 12:12 + n * 12 + n], dtype=np.uint8)
    return (magic, ver, kind, axis_frame, n), pos, lab


def above_floor(fm, xa, yv, half_y=0.6, half_x=0.15):
    """Максимум z меша пола под осью нити (полка), а не дно лотка."""
    band = np.abs(fm[:, 1] - yv) <= half_y
    if not band.any():
        return None
    sub = fm[band]
    ring = np.abs(sub[:, 0] - xa) <= half_x
    if ring.sum() < 3:
        return None
    return float(np.max(sub[ring][:, 2]))


print('=== 1) /track по записям ===')
print('%-34s %5s | %8s | %-9s | %-15s | %-13s | %s'
      % ('запись', 'кадр', 'лента', 'predicted', 'пол: недобор, м', 'верх-пол, м', 'air_gap form'))
for rec in RECS:
    for f in (0, 60):
        d = get('/track?bag=%s&frame=%d' % (rec, f))
        rm = d['meta']['rail_mesh']
        fm = np.asarray(d['floor_mesh']['vertices'], float)
        far_l = min(v[0] for v in rm['rails']['left']['nodes'])
        far_r = min(v[0] for v in rm['rails']['right']['nodes'])
        npred = sum(int((np.asarray(rm['rails'][s]['node_origin'], object)
                         == 'predicted').sum()) for s in ('left', 'right'))
        ntot = sum(len(rm['rails'][s]['node_origin']) for s in ('left', 'right'))
        gaps = []
        for s in ('left', 'right'):
            n = np.asarray(rm['rails'][s]['nodes'], float)
            n = n[np.argsort(n[:, 0])]
            for i in range(n.shape[0]):
                zf = above_floor(fm, n[i, 1], n[i, 0])
                if zf is not None:
                    gaps.append(n[i, 2] - zf)
        print('%-34s %5d | %8.1f | %4d/%-4d | %15.1f | %s | %s'
              % (rec, f, min(far_l, far_r), npred, ntot,
                 min(far_l, far_r) - fm[:, 1].min(),
                 ('%.3f..%.3f' % (min(gaps), max(gaps))) if gaps else '   нет   ',
                 ' / '.join(str(rm['rails'][s]['air_gap']['max_form_m'])[:6]
                            for s in ('left', 'right'))))

print()
print('=== 2) цвет рельса на ХОЛОДНОМ кадре (сначала /frame, потом /labels) ===')
for rec, f in (('roundT_doubleT', 150), ('doubleT_obstacle', 200)):
    hdr, pos, lab = parse(get('/frame?bag=%s&idx=%d&mode=zones' % (rec, f), raw=True))
    d = get('/track?bag=%s&frame=%d' % (rec, f))
    n = np.asarray(d['meta']['rail_mesh']['rails']['left']['nodes'], float)
    n = n[np.argsort(n[:, 0])]
    zt = np.interp(pos[:, 1], n[:, 0], n[:, 2])
    xa = np.interp(pos[:, 1], n[:, 0], n[:, 1])
    band = ((np.abs(pos[:, 0] - xa) <= 0.06) & (pos[:, 2] >= zt - 0.04)
            & (pos[:, 2] <= zt + 0.02))
    k = int(band.sum())
    c = {int(v): int((lab[band] == v).sum()) for v in np.unique(lab[band])}
    ok = (k > 0 and c.get(RAIL_LABEL, 0) == k)
    print('  %-22s f%-4d ось в заголовке = %-4d точек в полосе %4d  rail %d  middle %d  %s'
          % (rec, f, hdr[3], k, c.get(RAIL_LABEL, 0), c.get(MIDDLE_LABEL, 0),
             'ОК' if ok else 'ПЛОХО'))

print()
print('=== 3) время /track, с ===')
t0 = time.time()
get('/track?bag=doubleT_obstacle&frame=6')
t_cold = time.time() - t0
t_new = []
for f in (7, 8, 9):
    t0 = time.time()
    get('/track?bag=doubleT_obstacle&frame=%d' % f)
    t_new.append(time.time() - t0)
t0 = time.time()
get('/track?bag=doubleT_obstacle&frame=6')
t_cache = time.time() - t0
print('  первый запрос %.3f | не из кэша медиана %.3f (макс %.3f) | из кэша %.3f'
      % (t_cold, float(np.median(t_new)), max(t_new), t_cache))