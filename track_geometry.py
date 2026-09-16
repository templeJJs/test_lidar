"""Геометрия пути по одному кадру PointCloud2: ось, рельсы, пол с лотком.

Публичный контракт — ровно одна функция
`build_track(db_path, frame_index, floor_ab=None, draw_sleepers=False,
only_observed=False)`. Она возвращает JSON-совместимый dict (см. её докстроку).
Модуль ничего не рисует, не открывает окон, не ходит в сеть и не имеет эффектов
при импорте.

Что берётся из данных, а что из стандарта
-----------------------------------------
Наблюдаема только головка рельса — её верх и верхние ~35-40 мм боковых граней;
шейка и подошва закрыты свесом головки и не видны ни при каком угле (замерено
отдельно: видимая глубина под верхом 35 мм для doubleT_platform и 40 мм для
doubleT_obstacle). Поэтому:

  * всё выше 38 мм под верхом головки — ИЗМЕРЕНО (ширина площадки, высота верха,
    её наклон вдоль пути, положение оси и колея);
  * МЕШ РЕЛЬСА — измеренное сечение, РАЗВЁРНУТОЕ вдоль оси: сечение берётся
    медианой по плотным бинам 0.5 м (ширина по перцентилям u, профиль из 5 узлов),
    затем ЭТО ЖЕ сечение ставится в каждый узел сетки 0.5 м по Y на прямую ось
    x = s*y + i и линию верха z = m*y + c — непрерывно на весь диапазон Y, включая
    участки за пределами плотных бинов (рельс — протяжённый объект постоянного
    сечения). Никакого ГОСТа в геометрии нет; контур Р65 доступен только по
    draw_gost_profile=True для сравнения, а режим «куски по бинам с точками» —
    по only_observed=True;
  * ПОЛ между рельсами — ИЗМЕРЕН: это не плоскость, а профиль с лотком
    (полки → почти вертикальные стенки → дно; замерено 0.40/0.92 м по ширине и
    0.32-0.36 м по глубине на doubleT_platform и 0.40-0.42 м / 0.36 м на
    doubleT_obstacle). floor_mesh — экструзия этого профиля вдоль Y;
  * шпалы в облаке НЕ наблюдаются: путь встроенный (рельсы в плите, между ними
    лоток), периодических структур в поле нет — проверено Фурье по полосам 5 см
    (амплитуда 0.3-1.0 м равна 0.0 мм при пороге обнаружения 13-16 мм).
    По умолчанию sleepers.instances ПУСТ; схематичная отрисовка по ГОСТ 78-2004
    включается только аргументом draw_sleepers=True.

Локальная система профиля рельса (мм) — «поперёк, вверх»
--------------------------------------------------------
  u — поперёк рельса, в положительную сторону — НАРУЖУ от оси пути
      (для левой нити наружу = −X, для правой = +X);
  v — вверх;
  начало координат — центр НИЖНЕЙ плоскости подошвы рельса: v=0 — низ подошвы,
  v=180 — верх головки нового рельса Р65.

Точка профиля (u, v) садится в мир как
  x = x_rail(y) + out_sign * (u*cosθ − v*sinθ)
  z = z_base(y)   +            (u*sinθ + v*cosθ)
где x_rail(y) = s*y + i — ось нити, z_base(y) = z_top(y) − 0.180 м — плоскость
подошвы, out_sign = −1 для 'left' и +1 для 'right', θ = подуклонка 1:20 = 2.86°
(ВСН 94-77, п. 3.17, внутрь колеи: верх рельса наклоняется к середине пути).
Обратный переход: u = (Δout*cosθ + Δz*sinθ), v = −Δout*sinθ + Δz*cosθ, где
Δout = out_sign*(x − x_rail), Δz = z − z_base.

Локальная система профиля ПОЛА (метры) другая: u — поперёк от ОСИ ПАРЫ рельсов
(середина между нитями), положительно к правой нити; z — абсолютная высота.
Полка описывается прямой shelf_z + shelf_slope_u*u + shelf_slope_y*(y − y0),
лоток — парой u_left/u_right, дном bottom_z и глубиной depth.

Соглашения вывода
-----------------
  * 'left' — нить с МЕНЬШИМ X, 'right' — с большим (взгляд вдоль +Y, Z вверх);
  * 'axis' {s, i} — середина между осями нитей: x = s*y + i. Оси самих нитей
    лежат на axis ± gauge_axis/2 (у каждой нити свой наклон, см. meta['notes']),
    поэтому по выводу модель восстанавливается;
  * head.top_z и head.inner_face_x — значения линий в середине ЯДРА детекции
    (участок, где нити измерены плотно; вдоль пути они линейны, наклоны — в
    meta['notes']). Колея и «ось-ось» в meta тоже даны на этом уровне;
  * head.inner_face_x — внутренняя грань головки на 13 мм ниже верха (уровень
    измерения колеи по ПТЭ / 2288р);
  * 'floor' {a,b,c} — ОПОРНАЯ плоскость z = a*x + b*y + c (нижняя огибающая
    ячеек 1x1 м), оставлена для прежних потребителей; геометрия пола — в
    floor_mesh и meta['floor_profile'].

Единицы — метры (мм только там, где в имени есть _mm).
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from bag_reader import BagFrames

# --------------------------------------------------------------- стандарт Р65
R65_H_MM = 180.00        # полная высота рельса
R65_HEAD_B_MM = 74.59    # ширина головки (контроль на 13 мм ниже верха)
R65_WEB_E_MM = 18.00     # толщина шейки
R65_FOOT_B_MM = 150.00   # ширина подошвы
R65_FOOT_M_MM = 11.25    # высота пера подошвы

HEAD_CONTROL_DEPTH_MM = 13.0   # уровень контроля ширины головки (ГОСТ Р 51685-2022, прил. П)
HEAD_MEASURED_DEPTH_MM = 38.0  # видимая глубина под верхом головки (замерено 35-40)
HEAD_BOTTOM_V_MM = R65_H_MM - HEAD_MEASURED_DEPTH_MM   # 142.0 — низ измеренной зоны
HEAD_CONTROL_V_MM = R65_H_MM - HEAD_CONTROL_DEPTH_MM   # 167.0
WEB_TOP_V_MM = 135.0            # переход головка/шейка (чертёж, вторичный источник)
FOOT_FILLET_U_MM = 40.0         # докуда перо подошвы идёт горизонтально
FOOT_TO_WEB_V_MM = 45.0         # верх перехода пера в шейку (аппроксимация галтели)
HEAD_FILLET_U_MM = 30.0         # ширина головки у её низа (аппроксимация галтели)

PODUKLONKA_RATIO = 20.0
PODUKLONKA_RAD = math.atan(1.0 / PODUKLONKA_RATIO)     # 2.862°

GAUGE_INNER_NOMINAL_MM = 1520.0                          # ПТЭ п. 45
GAUGE_AXIS_NOMINAL_MM = GAUGE_INNER_NOMINAL_MM + R65_HEAD_B_MM   # 1594.6

# шпала: деревянная тип II по ГОСТ 78-2004 табл. 1 (2750x230x160 мм) — этот тип
# стандарт предписывает приёмо-отправочным и станционным путям; тип I (250x180) —
# главным путям 1-2 классов. По умолчанию НЕ рисуются: путь встроенный, между
# рельсами идёт лоток, шпал в облаке нет (ни рельефом, ни комбом плотности).
SLEEPER_LENGTH_M = 2.75
SLEEPER_WIDTH_M = 0.23
SLEEPER_HEIGHT_M = 0.16
SLEEPER_SPACING_M = 0.55      # эпюра 1600-2000 шт/км, станционная

# ---------------------------------------------------- профиль пола (лоток)
FLOOR_BIN_M = 0.02            # поперечный бин профиля пола, м
FLOOR_U_MEAS_M = 0.90         # до какого |u| строится профиль по данным, м
FLOOR_U_MESH_M = 0.95         # край меша пола по u, м
FLOOR_Z_UP_M = 0.10           # гейт: ниже верха головки на ..., м
FLOOR_Z_DOWN_M = 0.80         # ... и не глубже, м
FLOOR_SHELF_DROP_M = 0.12     # ниже полки на столько — уже лоток
FLOOR_MIN_PTS = 4             # минимум точек в бине профиля
FLOOR_NEAR_M = 12.0           # ближняя зона вдоль Y, где пол плотно сэмплирован
FLOOR_MESH_STEP_M = 2.0       # шаг меша пола вдоль Y, м

# -------------------------------------------------------- окна поиска нитей
X_LO, X_HI = -3.0, 3.0        # коридор поиска вокруг сенсора, м
X_BIN = 0.02                  # поперечный шаг огибающего профиля, м
Y_HALF = 0.5                  # полбина огибающей; полоса детекции = 3 полбины
Y_SEARCH_M = 45.0             # дальше нитей в этих данных нет (замерено)
PROM_MIN_M = 0.035            # минимальное превышение гребня над соседями с двух сторон
RIDGE_W_LO, RIDGE_W_HI = 0.025, 0.15   # ширина площадки головки на 15 мм ниже верха
GAUGE_TOL_M = 0.20            # допуск пары «ось-ось»
MID_MAX_M = 0.60              # своя колея — ближе 0.6 м к сенсору (наблюдено 0.03...0.18)


# ------------------------------------------- меш рельса по НАБЛЮДАЕМЫМ точкам
RAIL_BIN_Y_M = 0.50           # шаг бинов вдоль пути, м (как полосы измерений головки)
RAIL_WIN_HALF_M = 0.75        # окно оценки сечения бина: ±0.75 м (сечение рельса
                              # постоянно на такой длине, а перцентили кромок на
                              # 10-30 точках окна ещё устойчивы)
RAIL_NODES = 5                # узлов в поперечном сечении бина
RAIL_MIN_PTS_BIN = 3          # минимум точек в окне бина, иначе бина нет (6 точек/м)
RAIL_U_SEL_M = 0.060          # полуокно отбора точек нити по u, м (как в head)
RAIL_DEPTH_UP_M = 0.005       # глубина от линии верха: допуск вверх, м
RAIL_DEPTH_DOWN_M = 0.040     # ... и вниз, м (та же выборка, что для head)

# развёртка измеренного сечения вдоль оси (рельс — протяжённый объект)
RAIL_SWEEP_STEP_M = 0.50      # шаг сетки меша вдоль пути, м
RAIL_SWEEP_EXTRA_M = 15.0     # запас развёртки за пределы ИЗМЕРЕННОГО диапазона, м:
                              # данных о рельсе дальше плотных бинов нет (у нас
                              # 20-41 м), поэтому длинная прямая экструзия уезжает
                              # из сцены. Полный диапазон — sweep_extra_m=None.
RAIL_SWEEP_Y_FAR_M = -200.0   # дальний конец ПОЛНОГО диапазона (sweep_extra_m=None)
RAIL_SWEEP_Y_NEAR_M = -2.0    # ближний край (у сенсора путь виден от ~-2 м)
RAIL_DRIFT_LIMIT_M = 0.30     # предел ухода меша по высоте от уровня в СЕРЕДИНЕ
                              # измеренного диапазона (на дальний конец бюджета
                              # остаётся 0.30/|наклон полки| - полдиапазона), м
RAIL_X_DRIFT_LIMIT_M = 0.50   # то же поперёк пути для продолжения по касательной, м

# ------------------------------------------- ось нити: ПОЛИЛИНИЯ (кривые, стрелки)
# Прямая x = s*y + i на кривой теряет нить через 10-15 м (окно отбора точек ±60 мм,
# а реальный путь уходит в сторону), и дальше меш режет стены и пол. Поэтому ось —
# полилиния из центров локальных сечений бинов: сечение следующего бина ищется в
# окне от ПРЕДСКАЗАНИЯ по последним двум узлам, то есть вдоль изгиба пути.
POLY_BIN_M = RAIL_BIN_Y_M     # шаг узлов полилинии вдоль пути, м (0.5)
POLY_WIN_M = RAIL_WIN_HALF_M  # окно оценки сечения узла, м (±0.75)
POLY_U_WIN_M = 0.12           # полуокно поиска точек вокруг предсказания, м: головка
                              # 75 мм + запас на изгиб за шаг 0.5 м (при R=20 м это
                              # 6 мм) и на ошибку предсказания; стенку (±1 м) не ловит
POLY_SMOOTH_BINS = 3          # окно финального сглаживания позиций, узлов (±1 = 1 м):
                              # шире нельзя — срез угла (окно 2 м на R=22 м даёт 23 мм),
                              # а параболическое (SG-2) на этих данных УСИЛИВАЕТ излом
                              # наклона (замерено 0.0599 -> 0.0842)
POLY_TANGENT_NODES = 2        # полубаза касательной для поворота сечения (±2 узла = 2 м):
                              # шум направления падает вдвое, смещения на дуге нет
                              # (центральная разность точна для параболы)
POLY_MAD_K = 3.0              # отбраковка узлов по MAD от скользящей медианы
POLY_MAD_MIN_M = 0.02         # ...но не жёстче 20 мм (мельче — шум, не выброс)
POLY_MAX_MISS = 12            # бинов подряд можно «проехать» без точек (6 м) до конца
                              # трека: замерены реальные окклюзии 3 м (doubleT_obstacle)
                              # и 7 м между двумя измерениями той же нити. Длинный
                              # пролёт безопасен: предсказание едет по последней
                              # касательной, а окно возврата ±POLY_U_WIN_M — на кривой
                              # за 4 м уход уже 0.12 м, и вернуться в окно можно только
                              # там, где путь реально прямой (стрелка/ветка не ловится)
POLY_TAIL_N = 5               # узлов для касательной на конце (продолжение), 2.5 м
POLY_PRED_N = 5               # окно предсказания следующего узла (МНК по последним
                              # принятым): по двум точкам разовый выброс уводит трек
POLY_U_LIM_M = 0.15           # |u| полосы «точки рельса» для метрики оси, м
POLY_BAND_LO_M = 0.05         # низ полосы головки над измеренной полкой, м
POLY_BAND_HI_M = 0.40         # верх полосы головки над измеренной полкой, м
POLY_METRIC_WIN_M = 1.0       # окно по Y для метрики оси, м
POLY_JUMP_M = 0.10            # шаг между узлами больше этого — «скачок» (счёт в meta)
POLY_OUT_WIN = 7              # окно отбраковки выбросов узлов, узлов (±3 = 3.5 м)
POLY_TREND_DEV_M = 0.05       # узел дальше этого от локального тренда — выброс
POLY_Z_BLEND_M = 2.0          # сглаживание стыка «данные/продолжение» по Z, м
POLY_TANGENT_MAX_M = 5.0      # предел продолжения по прямой, если κ не оценена, м
POLY_PAIR_GAUGE_LO_M = 1.580  # колея пары по осям: нижний предел (Р65, головка 60-90)
POLY_PAIR_GAUGE_HI_M = 1.610  # ... и верхний (номинал 1590-1595, гуляет на ±15)
POLY_PAIR_OFF_M = 0.030       # |Δ−G| больше этого — нити противоречат друг другу:
                              # центр берём по НАДЁЖНОЙ нити, а не по среднему
POLY_PAIR_PTS_W = 12.0        # вес одной отбракованной MAD-узла в балле надёжности
POLY_FOOT_DROP_M = R65_H_MM / 1000.0   # верх головки → низ подошвы (180 мм)
POLY_FOOT_TOL_M = (-0.05, 0.02)        # допуск низа подошвы относительно полки, м

# --- ГЕЙТ ВЫСОТЫ ГОЛОВКИ НАД ПОЛКОЙ -----------------------------------------
# Верх головки в бине измеряется по точкам (max z кластера головки). Если бин попал
# на постель/плечо/тень, верх уезжает на 0.1-0.3 м и лента идёт сквозь пол или в
# потолок — именно это видно в браузере. Поэтому бин с верхом вне коридора не
# принимается вовсе (становится пропуском, как «мало точек»), а не протягивается.
POLY_TOP_LO_M = 0.08          # низ коридора «верх головки над измеренной полкой», м
POLY_TOP_HI_M = 0.30          # верх коридора, м (рольганг: было измерено 0.106-0.217)
POLY_TOP_HARD_M = 0.05        # жёсткий минимум для ЛЮБОГО узла внутри данных, м
# НИЗ КОРИДОРА (кламп узла): 0.08 м МЕНЬШЕ высоты тела рельса 0.1835 м, поэтому узел,
# поджатый к нему, гарантированно топит подошву на ~103 мм под полку (замерено, спека
# 2.2). Кламп берётся на «высота тела − запас» (0.1335 м при теле 0.1835 м): тогда
# подошва не глубже 50 мм под локальной полкой.
POLY_TOP_MARGIN_M = 0.05      # запас низа клампа от подошвы, м

# --- ПРАВИЛО ОСТАНОВКИ ПРОДОЛЖЕНИЯ ПО ДАННЫМ (спека 3.2, шаг 1) ---------------
# Продолжение за измеренный конец проверяется данными: шаг наружу 0.5 м, на каждом
# шаге окно 5 м ВПЕРЁД, нужно >=3 точки ГОЛОВКИ (|Δx| <= 0.15 м от оси нити и
# Δz ∈ [-45, +20] мм от СВОЕЙ линии верха нити, уклон m_top продолжен тем же уклоном).
# Первая неудача → накат 0.5 м и стоп. Второй предохранитель: >=5 точек облака внутри
# ОБЪЁМА ТЕЛА ленты (тело рельса обязано быть пустым) → стоп. Итог = min(двух стопов).
POLY_REACH_WIN_M = 5.0        # окно подтверждения вперёд, м
POLY_REACH_STEP_M = 0.5       # шаг наружу, м
POLY_REACH_MIN_PTS = 3        # минимум точек головки в окне
POLY_REACH_DX_M = 0.15        # |Δx| от оси нити, м
POLY_REACH_DZ_M = (-0.045, 0.020)     # Δz от своей линии верха нити, м
POLY_REACH_ROLL_M = 0.5       # накат после первой неудачи, м
POLY_REACH_RIB_MIN = 25       # предохранитель: точек внутри ВНУТРЕННОСТИ тела
# ПОЧЕМУ 25, А НЕ 5: «тело рельса обязано быть пустым» на этих данных НЕВЕРНО — рельс
# втоплен в пол, и пол под ПРАВИЛЬНО поставленной нитью сам лежит внутри сечения тела.
# Замерено (48 нитей): внутри растровой внутренности сечения 0-9 точек на окно 5 м на
# ИЗМЕРЕННОМ участке и 0-8 за концом — разделения нет, порог 5 режет ленту на живом
# рельсе. Поэтому предохранитель оставлен как страховка от «лента ушла в стену/платформу»
# с порогом 25 (≈3× худшего измеренного), а счётчик отдаётся в meta (ribbon_inside_points).
POLY_REACH_RIB_DU_M = 0.040   # (резерв) грубая рамка объёма головки, если нет растра
POLY_REACH_RIB_DZ_M = (-0.090, -0.010)

# --- ВТОРАЯ СТУПЕНЬ: РАЗРЕЖЕННЫЕ ДАННЫЕ (жалоба «лента в 3-6 раз короче видимого
# рельса»). За плотной зоной рельс виден редкими точками (p90/p99 головки 20-76 м),
# и правило «>=3 точки в окне 5 м при |Δx| <= 0.15 м» там обрывает ленту. Ступень 2:
# >=2 точки в окне, |Δx| <= 0.35 м от оси нити, Δz от своей линии верха как в ступени 1,
# плюс КЛАСТЕРНОСТЬ (точки окна — компактная группа, а не два разрозненных выброса) и
# НЕПРЕРЫВНОСТЬ (центр ленты доезжает за полосой: поправка за шаг ограничена).
POLY_REACH_SPARSE_DX_M = 0.35
POLY_REACH_SPARSE_MIN = 2     # точек в окне 5 м для разреженной ступени (кластер)
POLY_REACH_ONE_DX_M = 0.25    # порог для ОДИНОЧНОЙ точки: она должна лечь в створе
#                              продолжения (в 0.25 м от предсказанного центра нити) —
#                              так редкие одиночные возвраты продолжают полосу, но
#                              одиночный «выброс» рядом не принимается
POLY_REACH_CLUSTER_DY_M = 4.0   # разброс принятых точек окна по y (кластерность)
POLY_REACH_CLUSTER_DX_M = 0.30  # и по поперечине
POLY_REACH_CORR_M = 0.20        # макс. поправка центра за шаг 0.5 м (доезд за полосой)
POLY_REACH_CORR_GAIN = 0.06     # доля доезда за шаг: поправка размазывается по длине,
#                                иначе лента «ломается» (излом наклона > 0.05 1/м)
POLY_REACH_OFF_MAX_M = 1.20     # предел накопленной поправки от исходной оси, м
POLY_REACH_SLOPE_BLEND = 0.30   # доля подстройки наклона оси по данным
POLY_REACH_PRE_DX_M = 3.0       # предварительный отбор точек по поперечине, м
POLY_REACH_FIT_M = 5.0          # по какой длине конца берём наклон оси продолжения, м
POLY_REACH_TAIL_N = 12          # узлов измеренного хвоста для МНК наклона продолжения
#                               (12 узлов = 6 м; требование «10-15 последних узлов»).
#                               Наклон продолжения СРАЗУ за стыком обязан совпасть с
#                               наклоном хвоста: |Δ| <= POLY_REACH_TAIL_TOL_MM/1000.
POLY_REACH_TAIL_TOL_MM = 5.0    # допуск «наклон продолжения − наклон хвоста», мм/м
POLY_REACH_TAIL_RAMP_M = 6.0    # на этой длине от стыка поправка от точек растёт не
#                               быстрее POLY_REACH_TAIL_TOL_MM на 1 м: иначе уже
#                               первые узлы продолжения набирают ЧУЖОЙ наклон
#                               (замерено 15-25 мм/м при допуске 5), хотя дальше
#                               лента обязана доезжать за точками
POLY_REACH_DRIFT_M = 0.60       # СУММАРНЫЙ предел поперечного ухода продолжения от
#                               касательной в стыке (прямой конец данных), м
POLY_REACH_DRIFT_RATE = 0.15    # ... и не больше столько на 1 м длины продолжения

# --- ТРЕТЬЯ СТУПЕНЬ: МОДЕЛЬ (данных нет). Продолжение по касательной/кривизне с
# меткой model; стоп по плотной структуре в объёме ленты и по пределу длины.
POLY_MODEL_MAX_M = 100.0        # предел модельного продолжения, м
POLY_MODEL_BLOCK_MIN = 60       # (резерв) точек в рамке объёма ленты → стоп
POLY_MODEL_BLOCK_DU_M = 0.25    # поперечина «объёма ленты» для этого счётчика, м
POLY_MODEL_BLOCK_DZ_M = (-0.30, 0.10)   # и высота от своей линии верха, м
# Настоящие предохранители модели (замерено: «плотность рядом с лентой» ловит ПОЛ под
# рельсом — рельс лежит на полу, и пол попадает в рамку объёма, поэтому счётчик выше
# бесполезен как стоп). Работают три различимых теста:
POLY_MODEL_ABOVE_MIN = 3000       # точек НАД лентой (свод/платформа/стена впереди)
POLY_MODEL_ABOVE_DU_M = 0.25    # поперечина полосы «над лентой», м
POLY_MODEL_ABOVE_DZ_M = (0.10, 2.00)    # Δz от своей линии верха, м
POLY_MODEL_VERT_MIN = 3000        # точек вертикальной структуры (стена) в полосе
POLY_MODEL_VERT_DU_M = 0.60
POLY_MODEL_VERT_DZ_M = (-0.60, 0.60)
POLY_MODEL_VERT_SPREAD_M = 0.50  # разброс по высоте: пол горизонтален (~0.05), стена нет
POLY_MODEL_WANDER_M = 3.0       # предельный поперечный уход МОДЕЛИ от прямой конца данных, м
#                              (уход дуги = kappa*L^2/2; больше — «лента ушла не туда»)
POLY_REACH_X_LIMIT_M = 3.0      # предел поперечного ухода ленты: ось не уезжает из
#                              тоннеля шире |x| <= max(3 м, |x на конце данных| + 0.5) —
#                              иначе лента уходит боком «не туда» (замерено 7.7 м при
#                              тоннеле 16 м; это же требование внешнего теста detector)
# широкая рамка «объём тела целиком» — только ДИАГНОСТИКА (в эту рамку попадает пол
# рядом с рельсом и плечо балласта, поэтому как предохранитель она не годится, но
# счётчик показываем в meta: в воздухе лента набирает больше)
POLY_REACH_BODY_DU_M = 0.075
POLY_REACH_BODY_DZ_M = (-0.190, -0.010)


# ------------------------------------------------------------------ плоскость
def _floor_plane(x, y, z):
    """Робастная плоскость пола z = a*x + b*y + c по 5-му перцентилю ячеек 1x1 м.

    5-й перцентиль ячейки — нижняя огибающая поверхности. Дальше МНК по ячейкам
    с отбраковкой по MAD: одиночные ячейки со сваями и сваями мусора иначе
    перекашивают плоскость. Полоса подгонки |x| <= 5 м, y в [-60, -2] (дальше пол
    почти не виден; ячейки вплотную к сенсору тоже исключены).
    """
    sel = (np.abs(x) <= 5.0) & (y >= -60.0) & (y <= -2.0)
    if sel.sum() < 2000:
        sel = np.ones(x.size, dtype=bool)

    xs, ys, zs = x[sel], y[sel], z[sel]
    key = np.floor(xs).astype(np.int64) * 100003 + np.floor(ys).astype(np.int64)
    order = np.argsort(key, kind='stable')
    groups = np.split(order, np.flatnonzero(np.diff(key[order])) + 1)

    gx, gy, gz = [], [], []
    for g in groups:
        if g.size >= 20:
            gx.append(xs[g].mean())
            gy.append(ys[g].mean())
            gz.append(np.percentile(zs[g], 5.0))
    if len(gx) < 6:
        return (0.0, 0.0, float(np.percentile(z, 5.0)))

    gx, gy, gz = np.asarray(gx), np.asarray(gy), np.asarray(gz)
    A = np.column_stack([gx, gy, np.ones(gx.size)])
    w = np.ones(gx.size)
    for _ in range(8):
        coef, *_ = np.linalg.lstsq(A * w[:, None], gz * w, rcond=None)
        r = gz - A @ coef
        mad = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        w = 1.0 / np.sqrt(1.0 + (r / (3.0 * mad)) ** 2)
    return (float(coef[0]), float(coef[1]), float(coef[2]))


# ----------------------------------------------------------- огибающий профиль
def _envelope(x, y, z, y_lo, y_hi):
    """Огибающая Z по ячейкам «полбина (0.5 м) x X-бин (20 мм)».

    Возвращает (env, cnt, order, starts): env — второй максимум Z в ячейке
    (одиночный выброс не должен поднимать огибающую), order/starts — точки,
    разложенные по полбинам, чтобы не перефильтровывать облако в каждой полосе.
    """
    n_bins = int(round((X_HI - X_LO) / X_BIN))
    n_half = int(math.ceil((y_hi - y_lo) / Y_HALF)) + 1
    iy = np.floor((y - y_lo) / Y_HALF).astype(np.int64)
    ix = np.floor((x - X_LO) / X_BIN).astype(np.int64)
    ok = (ix >= 0) & (ix < n_bins) & (iy >= 0) & (iy < n_half)
    key = iy[ok] * n_bins + ix[ok]
    zz = z[ok]
    order = np.lexsort((zz, key))
    ks, zs = key[order], zz[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    ends = np.r_[starts[1:], ks.size]
    cnt = ends - starts
    env = np.full((n_half, n_bins), np.nan)
    C = np.zeros((n_half, n_bins), dtype=np.int64)
    r, c = np.divmod(ks[starts], n_bins)
    env[r, c] = np.where(cnt >= 2, zs[ends - 2], zs[ends - 1])
    C[r, c] = cnt
    idx_all = np.flatnonzero(ok)[order]
    half_start = np.searchsorted(iy[idx_all], np.arange(n_half + 1))
    return env, C, idx_all, half_start


def _ridge_candidates(env):
    """Пики огибающей, похожие на головку рельса.

    Гребень головки — локальный максимум, превышающий соседей на PROM_MIN_M с
    ОБЕИХ сторон (это и отличает рельс от кромки лотка или стены), с шириной
    площадки на 15 мм ниже верха в RIDGE_W_LO..RIDGE_W_HI.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    n = env.size
    if n < 5:
        return []
    e = env.copy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        e[1:-1] = np.nanmedian(np.stack([env[:-2], env[1:-1], env[2:]]), axis=0)

    local_max = np.zeros(n, dtype=bool)
    local_max[1:-1] = (e[1:-1] >= e[:-2]) & (e[1:-1] >= e[2:])

    half = 8
    lpad = np.full(n + 2 * half, np.inf)
    lpad[2 * half:] = np.where(np.isfinite(e), e, np.inf)
    rpad = np.full(n + 2 * half, np.inf)
    rpad[:n] = np.where(np.isfinite(e), e, np.inf)
    lmin = sliding_window_view(lpad, half + 1).min(axis=1)
    rmin = sliding_window_view(rpad, half + 1).min(axis=1)

    out = []
    for i in np.flatnonzero(local_max & np.isfinite(e)):
        prom_l = e[i] - lmin[i]
        prom_r = e[i] - rmin[i]
        if not (np.isfinite(prom_l) and np.isfinite(prom_r)):
            continue
        if min(prom_l, prom_r) < PROM_MIN_M:
            continue
        level = e[i] - 0.015
        j = i
        while j > 0 and np.isfinite(e[j]) and e[j] > level:
            j -= 1
        k = i
        while k < n - 1 and np.isfinite(e[k]) and e[k] > level:
            k += 1
        width = (k - j) * X_BIN
        if not (RIDGE_W_LO <= width <= RIDGE_W_HI):
            continue
        out.append((int(i), float(min(prom_l, prom_r))))

    out.sort(key=lambda t: t[0])
    merged = []
    for c in out:
        if merged and (c[0] - merged[-1][0]) * X_BIN <= 0.06:
            if c[1] > merged[-1][1]:
                merged[-1] = c
        else:
            merged.append(c)
    return merged


def _bands(x, y, z, y_lo, y_hi):
    """Полосы детекции: [(y_center, [(x, z, prom), ...]), ...].

    Кандидат обязан быть гребнем над МЕСТНОЙ поверхностью: 10-й перцентиль Z в
    ±0.35 м вокруг него должен быть ниже верха на 0.08-0.45 м. Без этого гейта
    на 20-40 м в пары попадали кромки потолка (их превышение 3-4 м против 0.15 м
    у рельса), и своя колея уезжала на чужую линию.
    """
    env, _cnt, idx_all, half_start = _envelope(x, y, z, y_lo, y_hi)
    n_half = env.shape[0]
    xc = X_LO + X_BIN * (np.arange(env.shape[1]) + 0.5)
    bands = []
    for b in range(max(0, n_half - 2)):
        v = env[b:b + 3]
        finite = np.isfinite(v)
        if not finite.any():
            bands.append((y_lo + (b + 1.5) * Y_HALF, []))
            continue
        e = np.where(finite.any(axis=0),
                     np.max(np.where(finite, v, -np.inf), axis=0), np.nan)
        sl = idx_all[half_start[b]:half_start[min(b + 3, n_half)]]
        xs, zs = x[sl], z[sl]
        peaks = []
        for (i, prom) in _ridge_candidates(e):
            xi = float(xc[i])
            win = np.abs(xs - xi) <= 0.35
            if win.sum() >= 10:
                gnd = float(np.percentile(zs[win], 10.0))
                h = float(e[i]) - gnd
                if not (0.08 <= h <= 0.45):
                    continue
            peaks.append((xi, float(e[i]), prom))
        bands.append((y_lo + (b + 1.5) * Y_HALF, peaks))
    return bands


def _band_pairs(band):
    """Пары пиков полосы, подходящие по колее (ось-ось)."""
    yc, peaks = band
    for a in range(len(peaks)):
        for b in range(a + 1, len(peaks)):
            xa, xb = peaks[a][0], peaks[b][0]
            if abs((xb - xa) - GAUGE_AXIS_NOMINAL_MM / 1000.0) > GAUGE_TOL_M:
                continue
            yield xa, xb, min(peaks[a][2], peaks[b][2]), yc


def _fit_line(ys, xs, tol=0.06, iters=3, pre_tol=None):
    """Прямая x = s*y + i с отбраковкой выбросов по MAD.

    pre_tol — грубый предфильтр по медиане ДО первой подгонки. Без него два
    выброса (например, точки потолка, случайно попавшие в узкое окно линии)
    разворачивают начальную МНК так, что MAD уже не помогает.
    """
    ys = np.asarray(ys, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    if pre_tol is not None and xs.size >= 4:
        keep0 = np.abs(xs - np.median(xs)) <= pre_tol
        if keep0.sum() >= 2:
            ys, xs = ys[keep0], xs[keep0]
    if ys.size < 2:
        return 0.0, (float(np.median(xs)) if xs.size else 0.0)
    s, i = np.polyfit(ys, xs, 1)
    for _ in range(iters):
        r = xs - (s * ys + i)
        med = np.median(r)
        mad = 1.4826 * np.median(np.abs(r - med)) + 1e-9
        keep = np.abs(r - med) <= max(tol, 3.0 * mad)
        if keep.all() or keep.sum() < 2:
            break
        ys, xs = ys[keep], xs[keep]
        s, i = np.polyfit(ys, xs, 1)
    return float(s), float(i)


def _core_run(ys, gap=1.5, window=10.0):
    """Непрерывный участок полос вокруг самого плотного окна.

    Дальние одиночные полосы не должны удлинять «ядро»: там головка уже не
    разрешается, и линия верха, подогнанная по ним, уезжает на постель.
    """
    ys = np.asarray(ys, dtype=np.float64)
    if ys.size < 3:
        return 0, ys.size
    order = np.argsort(ys)
    ys_s = ys[order]
    best_k, best_cnt = 0, -1
    j = 0
    for i in range(ys_s.size):
        while ys_s[i] - ys_s[j] > window:
            j += 1
        if i - j + 1 > best_cnt:
            best_cnt, best_k = i - j + 1, (i + j) // 2
    anchor = best_k
    lo = hi = anchor
    k = anchor - 1
    while k >= 0 and ys_s[k + 1] - ys_s[k] <= gap:
        lo = k
        k -= 1
    k = anchor + 1
    while k < ys_s.size and ys_s[k] - ys_s[k - 1] <= gap:
        hi = k
        k += 1
    keep = order[lo:hi + 1]
    return keep


def _find_rails_curved(x, y, z, bands, notes=None, gauge_m=None, gate_m=0.20,
                       max_miss=10, z_gate_m=0.18, pred_n=5, min_bands=6,
                       min_core_m=3.0, gauge_hard_m=0.03, u_win=0.12):
    """НИТИ НА КРИВОЙ: цепочки пар «два гребня на колее» без требования прямой оси.

    Повторный проход для кадров, где голосование за ОДНУ прямую не набрало невязок
    (в повороте кольца середина пары уходит от прямой за 10-15 м). Полосы 0.5 м идут
    по сетке; середина пары ведётся предсказанием по последним принятым (МНК-прямая
    по окну pred_n, как у полилинии оси), пара принимается если она ближе gate_m к
    предсказанию, расстояние между пиками в допуске колеи, а ВЫСОТЫ пиков
    непрерывны (±z_gate_m: обе нити рельса на одном уровне). Пропуски до max_miss
    полос переезжаются по касательной.

    Ядро — непрерывный участок вокруг самой плотной полосы-окна (см. _core_run), по
    нему подгоняются две прямые нитей (хорда ядра: этого достаточно — отчётные
    величины берутся в середине ядра). Колея в середине ядра обязана попадать в
    gauge ± gauge_hard_m, иначе кадр честно остаётся без рельса.

    Возвращает dict как _find_rails или None.
    """
    gauge_m = (GAUGE_AXIS_NOMINAL_MM / 1000.0 if gauge_m is None else float(gauge_m))
    iflen = len(bands)
    if iflen < 3:
        return None
    # облако по Y — для проверки нити ПО ТОЧКАМ (см. _by_points): в повороте один
    # рельс часто не даёт гребня полосы (тень/слияние с плечом), и пара собирается
    # только если вторую нить подтвердить сечением бина
    o = np.argsort(y, kind='stable')
    ys_s, xs_s, zs_s = y[o], x[o], z[o]

    def _by_points(x_rail, z_top, yc, half=0.25):
        """Нить по ТОЧКАМ облака: сечение головки в предсказанном месте (u, верх)."""
        a = int(np.searchsorted(ys_s, yc - half, 'left'))
        b = int(np.searchsorted(ys_s, yc + half, 'right'))
        if b - a < RAIL_MIN_PTS_BIN:
            return None
        dxx = xs_s[a:b] - x_rail
        dep = z_top - zs_s[a:b]
        sel = ((np.abs(dxx) <= u_win) & (dep >= -RAIL_DEPTH_UP_M) & (dep < 0.045))
        if int(sel.sum()) < RAIL_MIN_PTS_BIN:
            return None
        sec = _section_from_points(dxx[sel], zs_s[a:b][sel])
        if sec is None:
            return None
        return (float(x_rail + 0.5 * (sec['u_lo'] + sec['u_hi'])),
                float(sec['nodes_z'].max()))

    starts = []
    for i in np.argsort(np.abs(np.array([b[0] for b in bands], float))):
        i = int(i)
        yc, pk = bands[i]
        if abs(yc) > 60.0:
            continue
        for ia in range(len(pk)):
            for ib in range(ia + 1, len(pk)):
                if abs((pk[ib][0] - pk[ia][0]) - gauge_m) <= GAUGE_TOL_M \
                        and abs(0.5 * (pk[ia][0] + pk[ib][0])) <= MID_MAX_M:
                    starts.append((i, ia, ib))
        if len(starts) >= 24:
            break
    tracks = []
    for (i0, ia, ib) in starts:
        yc0, pk0 = bands[i0]
        acc = {i0: (float(yc0), float(pk0[ia][0]), float(pk0[ib][0]),
                    float(pk0[ia][1]), float(pk0[ib][1]))}
        for direction in (1, -1):
            miss = 0
            j = i0
            while 0 <= j + direction < iflen:
                jn = j + direction
                yc = float(bands[jn][0])
                hist = [acc[k] for k in sorted(acc)][-pred_n:]
                ys = np.array([h[0] for h in hist], float)
                if len(hist) >= 2 and float(np.ptp(ys)) > 1e-6:
                    mid = np.array([0.5 * (h[1] + h[2]) for h in hist], float)
                    za = np.array([h[3] for h in hist], float)
                    zb = np.array([h[4] for h in hist], float)
                    ym = float(np.mean(ys))
                    pred = float(np.mean(mid)) + float(np.polyfit(ys, mid, 1)[0]) * (yc - ym)
                    predza = float(np.mean(za)) + float(np.polyfit(ys, za, 1)[0]) * (yc - ym)
                    predzb = float(np.mean(zb)) + float(np.polyfit(ys, zb, 1)[0]) * (yc - ym)
                else:
                    pred = 0.5 * (hist[0][1] + hist[0][2])
                    predza, predzb = hist[0][3], hist[0][4]
                best = None
                pk = bands[jn][1]
                half_g = 0.5 * gauge_m
                # 1) пара гребней полосы (как раньше)
                for ka in range(len(pk)):
                    xa, za, _pa = pk[ka]
                    for kb in range(ka + 1, len(pk)):
                        xb, zb, _pb = pk[kb]
                        if abs((xb - xa) - gauge_m) > GAUGE_TOL_M:
                            continue
                        d = abs(0.5 * (xa + xb) - pred)
                        if d > gate_m:
                            continue
                        if abs(za - predza) > z_gate_m or abs(zb - predzb) > z_gate_m:
                            continue
                        if best is None or d < best[0]:
                            best = (d, float(xa), float(xb), float(za), float(zb))
                # 2) один гребень полосы + вторая нить ПО ТОЧКАМ облака
                if best is None:
                    for ka in range(len(pk)):
                        xa, za, _pa = pk[ka]
                        for xo, zo, left_first in ((xa - gauge_m, predza, True),
                                                   (xa + gauge_m, predzb, False)):
                            if abs(xo - pred) > gate_m + half_g:
                                continue
                            pt = _by_points(xo, zo, yc)
                            if pt is None:
                                continue
                            xl, xr = (xa, pt[0]) if left_first else (pt[0], xa)
                            zl, zr = (za, pt[1]) if left_first else (pt[1], za)
                            if abs(xr - xl - gauge_m) > GAUGE_TOL_M:
                                continue
                            d = abs(0.5 * (xl + xr) - pred)
                            if d > gate_m:
                                continue
                            if best is None or d < best[0]:
                                best = (d, float(xl), float(xr), float(zl), float(zr))
                # 3) обе нити ПО ТОЧКАМ от предсказания (полоса вообще без пиков)
                if best is None:
                    pl = _by_points(pred - half_g, predza, yc)
                    pr = _by_points(pred + half_g, predzb, yc)
                    if pl is not None and pr is not None:
                        d = abs(0.5 * (pl[0] + pr[0]) - pred)
                        if d <= gate_m and abs(pr[0] - pl[0] - gauge_m) <= GAUGE_TOL_M:
                            best = (d, pl[0], pr[0], pl[1], pr[1])
                if best is None:
                    miss += 1
                    j = jn
                    if miss > max_miss:
                        break
                    continue
                miss = 0
                acc[jn] = (yc, best[1], best[2], best[3], best[4])
                j = jn
        if len(acc) < min_bands:
            continue
        keys = np.array(sorted(acc), dtype=np.int64)
        arr = np.array([[acc[int(k)][0], acc[int(k)][1], acc[int(k)][2]] for k in keys], float)
        keep = _core_run(arr[:, 0])
        arr = arr[keep]
        if len(arr) < min_bands or (arr[:, 0].max() - arr[:, 0].min()) < min_core_m:
            continue
        for _ in range(3):
            s_l, i_l = _fit_line(arr[:, 0], arr[:, 1])
            s_r, i_r = _fit_line(arr[:, 0], arr[:, 2])
            r_l = arr[:, 1] - (s_l * arr[:, 0] + i_l)
            r_r = arr[:, 2] - (s_r * arr[:, 0] + i_r)
            keep = ((np.abs(r_l - np.median(r_l)) < 0.06)
                    & (np.abs(r_r - np.median(r_r)) < 0.06))
            if keep.all() or keep.sum() < min_bands:
                break
            arr = arr[keep]
        if len(arr) < min_bands:
            continue
        s_l, i_l = _fit_line(arr[:, 0], arr[:, 1])
        s_r, i_r = _fit_line(arr[:, 0], arr[:, 2])
        y_core_mid = 0.5 * (float(arr[:, 0].min()) + float(arr[:, 0].max()))
        gauge_at = abs((s_r * y_core_mid + i_r) - (s_l * y_core_mid + i_l))
        if abs(gauge_at - gauge_m) > gauge_hard_m:
            if notes is not None:
                notes.append('повторный проход по локальной оси: цепочка из %d полос '
                             'отброшена — колея %.0f мм в середине ядра вне %.0f±%.0f мм'
                             % (len(arr), 1000.0 * gauge_at, 1000.0 * gauge_m,
                                1000.0 * gauge_hard_m))
            continue
        tracks.append({'s_l': s_l, 'i_l': i_l, 's_r': s_r, 'i_r': i_r,
                       'n_band': int(len(arr)),
                       'mid': float(((s_l * (-5.0) + i_l) + (s_r * (-5.0) + i_r)) / 2.0),
                       'core_lo': float(arr[:, 0].min()),
                       'core_hi': float(arr[:, 0].max()),
                       'curved': True})
    if not tracks:
        return None
    best_n = max(t['n_band'] for t in tracks)
    own = [t for t in tracks if t['n_band'] >= 0.6 * best_n and abs(t['mid']) <= MID_MAX_M]
    if not own:
        # ни одной цепочки у сенсора: лучше честно без нитей, чем чужая ветка/стена
        return None
    own.sort(key=lambda t: abs(t['mid']))
    return own[0]


def _find_rails(x, y, z, y_lo, y_hi, notes):
    """Ось и нити: голосование по серединам пар «два гребня на колее».

    Нити прямые, поэтому середина пары гребней своей колеи лежит на одной прямой
    x = s*y + i. Каждая полоса голосует за такую прямую; берутся сильнейшие
    направления с подавлением соседей, для каждого по полосам набираются пары
    (берётся пара, ближайшая к прогнозу) и подгоняются две линии нитей. Ядро
    трека — непрерывный участок вокруг самого плотного окна 10 м. Из годных
    треков выбирается своя колея: ближайшая к сенсору.

    Возвращает dict {'s_l','i_l','s_r','i_r','n_band','mid','core_lo','core_hi'} или None.
    """
    bands = _bands(x, y, z, y_lo, y_hi)
    slopes = np.arange(-0.05, 0.0501, 0.002)
    i0, di = -3.0, 0.01
    n_i = int(round(6.0 / di)) + 1
    acc = np.zeros((slopes.size, n_i))
    rows = np.arange(slopes.size)

    for band in bands:
        for xa, xb, _prom, yc in _band_pairs(band):
            b = np.round(((xa + xb) / 2.0 - slopes * yc - i0) / di).astype(np.int64)
            k = (b >= 0) & (b < n_i)
            if np.any(k):
                np.add.at(acc, (rows[k], b[k]), 1.0)

    peaks = []
    rest = acc.copy()
    for _ in range(8):
        kb, ib = np.unravel_index(np.argmax(rest), rest.shape)
        if rest[kb, ib] < 4:
            break
        s_mid = float(slopes[kb])
        i_mid = i0 + di * ib
        peaks.append((s_mid, i_mid))
        diag = np.abs((slopes[:, None] * (-5.0) + i0 + di * np.arange(n_i)[None, :])
                      - (s_mid * (-5.0) + i_mid))
        rest[diag < 0.10] = 0.0

    tracks = []
    for s_mid, i_mid in peaks:
        samples = []
        for band in bands:
            yc, _peaks_b = band
            pred = s_mid * yc + i_mid
            if abs(pred) > 1.2:
                continue
            best = None
            for xa, xb, _prom, _yc in _band_pairs(band):
                d = abs((xa + xb) / 2.0 - pred)
                if d > 0.10:
                    continue
                if best is None or d < best[0]:
                    best = (d, xa, xb, yc)
            if best is not None:
                samples.append((best[3], best[1], best[2]))
        if len(samples) < 4:
            continue
        arr = np.asarray(samples, dtype=np.float64)
        keep = _core_run(arr[:, 0])
        arr = arr[keep]
        if len(arr) < 4:
            continue
        for _ in range(3):
            s_l, i_l = _fit_line(arr[:, 0], arr[:, 1])
            s_r, i_r = _fit_line(arr[:, 0], arr[:, 2])
            r_l = arr[:, 1] - (s_l * arr[:, 0] + i_l)
            r_r = arr[:, 2] - (s_r * arr[:, 0] + i_r)
            keep = ((np.abs(r_l - np.median(r_l)) < 0.06)
                    & (np.abs(r_r - np.median(r_r)) < 0.06))
            if keep.all() or keep.sum() < 4:
                break
            arr = arr[keep]
        if len(arr) < 4:
            continue
        s_l, i_l = _fit_line(arr[:, 0], arr[:, 1])
        s_r, i_r = _fit_line(arr[:, 0], arr[:, 2])
        tracks.append({'s_l': s_l, 'i_l': i_l, 's_r': s_r, 'i_r': i_r,
                       'n_band': int(len(arr)),
                       'mid': float(((s_l * (-5.0) + i_l) + (s_r * (-5.0) + i_r)) / 2.0),
                       'core_lo': float(arr[:, 0].min()),
                       'core_hi': float(arr[:, 0].max())})

    if not tracks:
        t = _find_rails_curved(x, y, z, bands, notes)
        if t is not None:
            notes.append('нити найдены ПОВТОРНЫМ проходом по локальной оси (кривая): '
                         'полос %d, ядро y = %.1f..%.1f м, колея в ядре на парах '
                         '(допуск %.0f мм)'
                         % (t['n_band'], t['core_lo'], t['core_hi'],
                            1000.0 * GAUGE_TOL_M))
            return t
        notes.append('нити не найдены: пар гребней на колее 1594.6 ± 200 мм нет '
                     '(повторный проход по локальной оси тоже не собрал цепочку)')
        return None

    best_n = max(t['n_band'] for t in tracks)
    own = [t for t in tracks if t['n_band'] >= 0.6 * best_n and abs(t['mid']) <= MID_MAX_M]
    if not own:
        own = [min(tracks, key=lambda t: abs(t['mid']))]
        notes.append('нити найдены, но своя колея дальше %.2f м от сенсора — '
                     'взята ближайшая' % MID_MAX_M)
    own.sort(key=lambda t: abs(t['mid']))
    return own[0]


def _density_edge(u_mm, inward_negative, bin_mm=3.0, thr=0.20, lim_mm=60.0, min_pts=4):
    """Внутренняя грань головки по обрыву плотности.

    Экстремальные перцентили у неё шумные: за гранью есть редкий хвост точек
    (тень/постель), и p2 уезжает на 10-30 мм. Берём самый внутренний бин, в
    котором ещё есть плотность: не меньше 4 точек и не меньше 20 % от максимума.
    inward_negative=True — внутрь колеи это меньшие u.
    """
    u_mm = np.asarray(u_mm, dtype=np.float64)
    if u_mm.size < 10:
        return None
    edges = np.arange(-lim_mm, lim_mm + bin_mm, bin_mm)
    hist, _ = np.histogram(u_mm, bins=edges)
    if hist.max() < min_pts:
        return None
    ok = np.flatnonzero(hist >= max(min_pts, thr * hist.max()))
    if ok.size == 0:
        return None
    k = int(ok.min()) if inward_negative else int(ok.max())
    edge = edges[k] if inward_negative else edges[k + 1]
    return float(edge)


def _head_measurements(x, y, z, side, s, i, core_lo, core_hi):
    """Измерения головки нити: линия верха, ширины, внутренняя грань, высота над полом.

    Два прохода. Первый: по полосам 0.5 м берётся 98-й перцентиль Z точек в
    |X − линия| <= 60 мм с гейтом по местной земле и по ним подгоняется ЛИНИЯ
    ВЕРХА (верх вдоль пути наклонён, у doubleT_obstacle на 14 мм/м: без этого
    глубина «под верхом» гуляет на ±10 мм внутри полосы).

    Второй проход — по полосам 0.5 м, в КАЖДОЙ СВОЯ поперечная привязка:
    ось полосы = середина p2/p98 по X точек полосы, верх полосы = p98 по Z.
    Иначе кривизна пути вдоль 20 м смазывает u на 30-40 мм (замерено).
    В локальной координате u (наружу положительно) считаются
      * width_top_mm — p95−p5 поперечного размаха точек в 0..30 мм под верхом
        (видимая ширина площадки; наружная сторона головки всегда в тени);
      * width_13_mm — p98−p2 в 8..20 мм под верхом (уровень контроля ширины
        головки по ГОСТ Р 51685-2022 — справочная норма, не геометрия);
      * inner_off_mm — внутренняя грань головки по обрыву плотности (см.
        _density_edge) в 5..30 мм под верхом, отсчёт от местной оси внутрь колеи;
      * center_off_m — насколько местные оси смещены от линии нити (медиана);
      * s_center/i_center — линия по центрам поперечных сечений головки;
      * прилегающая поверхность — 90-й перцентиль Z в 0.10..0.24 м ВНУТРЬ от
        оси нити, по полосам; высота = верх полосы минус эта поверхность
        (замерено, что именно так получаются 0.159-0.186 м).
    """
    lam = -1.0 if side == 'left' else 1.0     # +1 = наружу; для правой нити это +X
    inward = -lam
    out_sign = lam
    rows = []
    n_band = max(2, int((core_hi - core_lo) / 0.5))
    dx = x - (s * y + i)
    for k in range(n_band + 1):
        y0 = core_lo + 0.5 * k
        band = (y >= y0) & (y < y0 + 0.5)
        near = band & (np.abs(dx) < 0.35)
        rail = band & (np.abs(dx) < 0.06)
        if near.sum() >= 20:
            gnd = float(np.percentile(z[near], 10.0))
            rail = rail & (z > gnd + 0.06) & (z < gnd + 0.45)
        if rail.sum() < 15:
            continue
        top = float(np.percentile(z[rail], 98))
        row = [y0 + 0.25, top, np.nan]
        ain = band & (inward * dx > 0.10) & (inward * dx < 0.24)
        if ain.sum() >= 10:
            row[2] = float(np.percentile(z[ain], 90))
        rows.append(row)

    if len(rows) < 4:
        return None
    arr = np.asarray(rows, dtype=np.float64)
    s_top, i_top = _fit_line(arr[:, 0], arr[:, 1], tol=0.02, pre_tol=0.30)

    m_adj = np.isfinite(arr[:, 2])
    h_adj = None
    if m_adj.sum() >= 4:
        h_adj = float(np.median(arr[m_adj, 1] - arr[m_adj, 2]))

    # второй проход: по полосам со СВОЕЙ поперечной привязкой
    z_top_line = s_top * y + i_top
    us, ds, offs, cys, cxs = [], [], [], [], []
    for k in range(n_band + 1):
        y0 = core_lo + 0.5 * k
        band = (y >= y0) & (y < y0 + 0.5) & (np.abs(dx) < 0.09)
        band &= (z_top_line - z) >= -0.005
        band &= (z_top_line - z) < 0.040
        idx_b = np.flatnonzero(band)
        if idx_b.size < 8:
            continue
        xs = x[idx_b]
        xc = 0.5 * (np.percentile(xs, 2) + np.percentile(xs, 98))
        tl = float(np.percentile(z[idx_b], 98))
        d_out = out_sign * (xs - xc)
        dz = z[idx_b] - (tl - R65_H_MM / 1000.0)
        us.append((d_out * math.cos(PODUKLONKA_RAD) + dz * math.sin(PODUKLONKA_RAD))
                  * 1000.0)
        ds.append((tl - z[idx_b]) * 1000.0)
        offs.append(xc - (s * (y0 + 0.25) + i))
        cys.append(y0 + 0.25)
        cxs.append(xc)

    if not us:
        return None
    u_head = np.concatenate(us)
    d_head = np.concatenate(ds)
    if u_head.size < 40:
        return None
    center_off = float(np.median(offs)) if offs else 0.0
    s_cent, i_cent = _fit_line(cys, cxs, tol=0.05)

    m30 = (d_head >= 0.0) & (d_head < 30.0)
    width_top = (float(np.percentile(u_head[m30], 95) - np.percentile(u_head[m30], 5))
                 if m30.sum() >= 20 else None)
    m13 = (d_head >= 8.0) & (d_head < 20.0)
    width_13 = (float(np.percentile(u_head[m13], 98) - np.percentile(u_head[m13], 2))
                if m13.sum() >= 20 else None)
    m5 = (d_head >= 5.0) & (d_head < 30.0)
    inner = _density_edge(u_head[m5], inward_negative=True)
    if inner is None:
        inner = float(np.percentile(u_head[m5], 2)) if m5.sum() >= 20 else None

    return {
        's_top': s_top, 'i_top': i_top,
        'y_lo': float(arr[0, 0]), 'y_hi': float(arr[-1, 0]),
        'width_top_mm': width_top,
        'width_13_mm': width_13,
        'inner_off_mm': (None if inner is None else -float(inner)),
        'center_off_m': center_off,
        's_center': s_cent, 'i_center': i_cent,
        'height_above_adjacent_m': h_adj,
        'n_band': int(len(rows)),
    }


def rail_profile_mm(head_top_width_mm):
    """Контур Р65 (ГОСТ) в локальной системе (мм) — только для draw_gost_profile.

    Возвращает (verts, sources): verts — замкнутый многоугольник [(u, v), ...]
    (u наружу, v вверх от низа подошвы), sources — источник участка:
    'head measured' выше 38 мм под верхом, 'gost R65' ниже. В геометрию по
    умолчанию НЕ входит (см. _rail_mesh_measured).
    """
    w2 = float(head_top_width_mm) / 2.0
    half = [
        (0.0, 0.0, 'gost R65'),
        (R65_FOOT_B_MM / 2.0, 0.0, 'gost R65'),
        (R65_FOOT_B_MM / 2.0, R65_FOOT_M_MM, 'gost R65'),
        (FOOT_FILLET_U_MM, R65_FOOT_M_MM, 'gost R65'),
        (R65_WEB_E_MM / 2.0, FOOT_TO_WEB_V_MM, 'gost R65'),
        (R65_WEB_E_MM / 2.0, WEB_TOP_V_MM, 'gost R65'),
        (HEAD_FILLET_U_MM, HEAD_BOTTOM_V_MM, 'gost R65'),
        (R65_HEAD_B_MM / 2.0, HEAD_CONTROL_V_MM, 'head measured'),
        (w2, R65_H_MM, 'head measured'),
    ]
    verts = [(u, v) for (u, v, _s) in half]
    sources = [s for (_u, _v, s) in half]
    for (u, v, s) in reversed(half[1:]):
        verts.append((-u, v))
        sources.append(s)
    return verts, sources


def _point_in_tri(p, a, b, c):
    """Точка внутри треугольника (барицентрический тест, включая границу)."""
    d1 = (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1])
    d2 = (p[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (p[1] - c[1])
    d3 = (p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])
    neg = (d1 <= 0) or (d2 <= 0) or (d3 <= 0)
    pos = (d1 >= 0) or (d2 >= 0) or (d3 >= 0)
    return not (neg and pos)


def _triangulate_polygon(pts):
    """Отсечение ушей для простого многоугольника (u,v), обход против часовой.

    Профиль Р65 невыпуклый (шейка уже подошвы и головки), поэтому веером его
    крыть нельзя — нужны уши.
    """
    n = len(pts)
    if n < 3:
        return []
    idx = list(range(n))
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 20 * n:
        guard += 1
        cut = False
        for k in range(len(idx)):
            i0 = idx[k - 1]
            i1 = idx[k]
            i2 = idx[(k + 1) % len(idx)]
            a, b, c = pts[i0], pts[i1], pts[i2]
            if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) <= 0:
                continue
            inside = False
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                if _point_in_tri(pts[j], a, b, c):
                    inside = True
                    break
            if inside:
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            cut = True
            break
        if not cut:
            break
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    return tris



def _rail_bin_sections(x, y, z, s, i, m_top, c_top, out_sign):
    """Сечения нити по бинам вдоль пути (0.5 м) — основа и меша, и медианы.

    Выборка точек — та же, что для head: |u| <= 60 мм и глубина 0..40 мм под
    линией верха. Сечение бина оценивается по точкам в окне ±0.75 м вокруг
    центра бина (в одном бине кромочных точек почти нет): поперечные границы —
    перцентили u (n >= 8: p2..p98, иначе min..max), затем RAIL_NODES узлов между
    границами, высоты — линейная интерполяция облака (u, z) по этим узлам.
    Бин валиден, если в нём есть хоть одна точка и в окне не меньше
    RAIL_MIN_PTS_BIN точек.

    Возвращает dict или None: y_c, u_lo, u_hi, nodes(n_bins, N), z_nodes,
    dz_nodes (относительно линии верха), widths, b0, n_span, y_dense, gaps,
    ribbons, n_pts_sel.
    """
    dx = x - (s * y + i)
    top = m_top * y + c_top
    depth = top - z
    sel = ((np.abs(dx) <= RAIL_U_SEL_M) & (depth >= -RAIL_DEPTH_UP_M)
           & (depth < RAIL_DEPTH_DOWN_M))
    idx = np.flatnonzero(sel)
    if idx.size < RAIL_MIN_PTS_BIN:
        return None
    uu = out_sign * dx[idx]
    yy = y[idx]
    zz = z[idx]
    order = np.argsort(yy, kind='stable')
    ys_s, u_s, z_s = yy[order], uu[order], zz[order]
    b0 = math.floor(float(ys_s[0]) / RAIL_BIN_Y_M) * RAIL_BIN_Y_M
    n_span = int(math.floor((float(ys_s[-1]) - b0) / RAIL_BIN_Y_M)) + 1
    half_own = RAIL_BIN_Y_M / 2.0

    out = {'b0': b0, 'n_span': n_span, 'n_pts_sel': int(idx.size),
           'y_c': [], 'u_lo': [], 'u_hi': [], 'nodes': [], 'z_nodes': [],
           'dz_nodes': [], 'widths': [], 'bins': []}
    for b in range(n_span):
        yc = b0 + (b + 0.5) * RAIL_BIN_Y_M
        lo = int(np.searchsorted(ys_s, yc - RAIL_WIN_HALF_M, side='left'))
        hi = int(np.searchsorted(ys_s, yc + RAIL_WIN_HALF_M, side='right'))
        if hi - lo < RAIL_MIN_PTS_BIN:
            continue
        own_lo = int(np.searchsorted(ys_s, yc - half_own, side='left'))
        own_hi = int(np.searchsorted(ys_s, yc + half_own, side='right'))
        if own_hi - own_lo < 1:
            continue
        ub = u_s[lo:hi]
        zb = z_s[lo:hi]
        n = ub.size
        u_lo, u_hi = ((float(np.percentile(ub, 2)), float(np.percentile(ub, 98)))
                      if n >= 8 else (float(ub.min()), float(ub.max())))
        if u_hi - u_lo < 1e-4:
            continue
        o2 = np.argsort(ub, kind='stable')
        us_s, zs_s = ub[o2], zb[o2]
        # точки с одинаковым u (в пределах 1 мм) усредняем — иначе интерполяция
        # по «многозначной» функции u(z) даёт ступеньки
        key = np.round(us_s * 1000.0).astype(np.int64)
        keep_u, keep_z = [], []
        start = 0
        for k in range(1, key.size + 1):
            if k == key.size or key[k] != key[start]:
                keep_u.append(float(us_s[start:k].mean()))
                keep_z.append(float(zs_s[start:k].mean()))
                start = k
        nodes = np.linspace(u_lo, u_hi, RAIL_NODES)
        zn = np.interp(nodes, np.asarray(keep_u), np.asarray(keep_z))
        out['y_c'].append(float(yc))
        out['u_lo'].append(float(u_lo))
        out['u_hi'].append(float(u_hi))
        out['nodes'].append([float(v) for v in nodes])
        out['z_nodes'].append([float(v) for v in zn])
        out['dz_nodes'].append([float(v - (m_top * yc + c_top)) for v in zn])
        out['widths'].append(float(u_hi - u_lo))
        out['bins'].append(int(b))
    if not out['y_c']:
        return None
    for k in ('y_c', 'u_lo', 'u_hi', 'widths'):
        out[k] = np.asarray(out[k], float)
    out['nodes'] = np.asarray(out['nodes'], float)
    out['z_nodes'] = np.asarray(out['z_nodes'], float)
    out['dz_nodes'] = np.asarray(out['dz_nodes'], float)
    out['y_dense'] = [float(b0 + out['bins'][0] * RAIL_BIN_Y_M),
                      float(b0 + (out['bins'][-1] + 1) * RAIL_BIN_Y_M)]
    out['n_bins_ok'] = int(len(out['bins']))
    out['gaps'] = []
    ribbons = 1
    for prev, nxt in zip(out['bins'], out['bins'][1:]):
        if nxt - prev > 1:
            out['gaps'].append([float(b0 + (prev + 1) * RAIL_BIN_Y_M),
                                float(b0 + nxt * RAIL_BIN_Y_M)])
            ribbons += 1
    out['ribbons'] = ribbons
    return out


def _rail_mesh_bins(bsec, s, i, out_sign):
    """ПРЕЖНИЙ режим (only_observed=True): куски по бинам с точками.

    Поверхность рвётся там, где точек мало — оставлено для сравнения.
    """
    verts = []
    first = {}
    for bi, yc in enumerate(bsec['y_c']):
        xc = s * yc + i
        first[bi] = len(verts)
        nodes = bsec['nodes'][bi]
        zs = bsec['z_nodes'][bi]
        for k in range(RAIL_NODES):
            verts.append([float(xc + out_sign * nodes[k]), float(yc), float(zs[k])])
    faces = []
    for bi in range(len(bsec['y_c'])):
        if bi + 1 >= len(bsec['y_c']) or bsec['bins'][bi + 1] - bsec['bins'][bi] != 1:
            continue
        a0, a1 = first[bi], first[bi + 1]
        for k in range(RAIL_NODES - 1):
            faces.append([a0 + k, a0 + k + 1, a1 + k + 1])
            faces.append([a0 + k, a1 + k + 1, a1 + k])
    return verts, faces


def _median_section(bsec):
    """МЕДИАННОЕ сечение по плотным бинам: ширина, смещение, профиль под верхом.

    Берётся медиана по бинам: полуширина = median(width)/2, смещение центра
    относительно линии нити = median((u_lo+u_hi)/2), профиль vz = медиана
    dz_nodes по узлам (м, относительно линии верха). Никаких стандартных
    размеров: всё это — сводка измеренных сечений.
    """
    half_w = float(np.median(bsec['widths'])) / 2.0
    u_off = float(np.median(0.5 * (bsec['u_lo'] + bsec['u_hi'])))
    vz = np.median(bsec['dz_nodes'], axis=0)
    return {
        'half_width_m': half_w, 'width_m': 2.0 * half_w, 'u_off_m': u_off,
        'vz_nodes_m': [float(v) for v in vz],
        'depth_m': float(max(vz) - min(vz)),
        'n_bins_used': int(len(bsec['y_c'])),
        'y_dense': list(bsec['y_dense']),
    }


def _fmt_num(v, nd=3):
    """Число для текста заметок: None -> «нет» (чтобы не печатать nan)."""
    return 'нет' if v is None else ('%.*f' % (nd, float(v)))


def _section_from_points(uu, zz, n_nodes=RAIL_NODES):
    """Сечение по точкам бина в ЛОКАЛЬНОЙ системе узла: границы и узлы.

    Точки с одинаковым u (в пределах 1 мм) усредняются — иначе интерполяция по
    «многозначной» функции u(z) даёт ступеньки. Возвращает None, если точки не
    образуют сечения (пусто или нулевая ширина).
    """
    if uu.size < RAIL_MIN_PTS_BIN:
        return None
    if uu.size >= 8:
        u_lo, u_hi = float(np.percentile(uu, 2)), float(np.percentile(uu, 98))
    else:
        u_lo, u_hi = float(uu.min()), float(uu.max())
    if u_hi - u_lo < 1e-4:
        return None
    o = np.argsort(uu, kind='stable')
    us, zs = uu[o], zz[o]
    key = np.round(us * 1000.0).astype(np.int64)
    keep_u, keep_z = [], []
    start = 0
    for k in range(1, key.size + 1):
        if k == key.size or key[k] != key[start]:
            keep_u.append(float(us[start:k].mean()))
            keep_z.append(float(zs[start:k].mean()))
            start = k
    nodes = np.linspace(u_lo, u_hi, n_nodes)
    zn = np.interp(nodes, np.asarray(keep_u), np.asarray(keep_z))
    return {'u_lo': u_lo, 'u_hi': u_hi, 'nodes_u': nodes, 'nodes_z': zn}


def _polyline_smooth(xc, zc, out_win=POLY_OUT_WIN, k=POLY_MAD_K,
                     floor_m=POLY_MAD_MIN_M, win=POLY_SMOOTH_BINS):
    """Сглаживание полилинии: MAD-отбраковка по окну ±3 узла, потом лёгкое сглаживание.

    Узлы стоят на равномерной сетке бинов (разрывы уже зашиты линейно), поэтому
    окно — просто срез массива. Выброс (скачок на соседнюю ветвь, стенку, край
    платформы или одиночный шумный верх) отбрасывается по MAD от медианы ШИРОКОГО
    окна (±POLY_OUT_WIN узлов = 3.5 м: одиночный прыжок виден на его фоне) и
    ЗАМЕНЯЕТСЯ медианой окна — полилиния обязана остаться непрерывной.
    Финальное сглаживание — узкое (±1 узел = 1 м): широкое срезало бы угол
    (окно 3.5 м на R=22 м даёт 70 мм среза, замерено).

    Возвращает (x_smooth, z_smooth, n_rejected, resid_m) — resid_m = (медиана, p95)
    |сглаженное - сырое| по узлам (мера среза угла сглаживанием).
    """
    xc = np.asarray(xc, float)
    zc = np.asarray(zc, float)
    n = xc.size
    if n == 0:
        return xc, zc, 0, (None, None)
    half = max(1, out_win // 2)
    xm = np.empty(n)
    zm = np.empty(n)
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        xm[i] = np.median(xc[a:b])
        zm[i] = np.median(zc[a:b])
    rx = xc - xm
    rz = zc - zm
    mad = 1.4826 * np.median(np.abs(rz - np.median(rz))) + 1e-9
    thr = max(floor_m, k * mad)
    bad = (np.abs(rz - np.median(rz)) > thr) | (np.abs(rx - np.median(rx)) > thr)
    xs, zs = xc.copy(), zc.copy()
    xs[bad], zs[bad] = xm[bad], zm[bad]
    # финальное сглаживание — УЗКОЕ СРЕДНЕЕ (±1 узел = 1 м): широкое окно срезает
    # угол (2 м на R=22 м дают 23 мм), а параболическое (SG-2) на этих данных
    # усиливает излом наклона (замерено 0.0599 -> 0.0842). Излом наклона гасится
    # не тут, а шириной базы касательной (см. _path_normals)
    hs = max(0, win // 2)
    xf, zf = xs.copy(), zs.copy()
    for i in range(n):
        a, b = max(0, i - hs), min(n, i + hs + 1)
        xf[i] = xs[a:b].mean()
        zf[i] = zs[a:b].mean()
    resid = np.abs(xf - xc)
    return (xf, zf, int(bad.sum()),
            (float(np.median(resid)), float(np.percentile(resid, 95))),
            np.asarray(bad, bool))


def _node_outlier_stats(ys, xs, zs, win=POLY_OUT_WIN):
    """Статистика узлов полилинии: одиночные прыжки по Z и излом наклона оси.

    * z_jump_max_m — максимальный |z_i − медиана(z в окне ±win)| (одиночный выброс);
    * slope_break_max_1_m — максимальный излом наклона оси между соседними шагами
      (|dX/dY| разность на шаг сетки) — «кривая» линия видна как большой излом;
    * trend_dev_max_m — максимальное отклонение узла от локальной прямой (МНК по окну
      ±win) по (y, x): узлы дальше POLY_TREND_DEV_M от тренда — выбросы.
    """
    ys = np.asarray(ys, float)
    xs = np.asarray(xs, float)
    zs = np.asarray(zs, float)
    n = ys.size
    out = {'z_jump_max_m': None, 'slope_break_max_1_m': None, 'trend_dev_max_m': None,
           'n_trend_out': None, 'window_nodes': int(win)}
    if n < 5:
        return out
    half = max(1, win // 2)
    zn = np.empty(n)
    tn = np.empty(n)
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        zn[i] = np.median(zs[a:b])
        if b - a >= 3 and float(np.ptp(ys[a:b])) > 1e-6:
            kk, bb = np.polyfit(ys[a:b], xs[a:b], 1)
            tn[i] = xs[i] - (kk * ys[i] + bb)
        else:
            tn[i] = 0.0
    out['z_jump_max_m'] = float(np.max(np.abs(zs - zn)))
    out['trend_dev_max_m'] = float(np.max(np.abs(tn)))
    out['n_trend_out'] = int(np.sum(np.abs(tn) > POLY_TREND_DEV_M))
    if n >= 3:
        d = np.hypot(np.diff(ys), np.diff(xs))
        d[d < 1e-9] = 1e-9
        sl = np.diff(xs) / d
        out['slope_break_max_1_m'] = float(np.max(np.abs(np.diff(sl)))) if sl.size >= 2 else 0.0
        # направление с базой ±POLY_TANGENT_NODES — именно оно поворачивает сечения
        k = max(1, int(POLY_TANGENT_NODES))
        tx = np.array([xs[min(n - 1, i + k)] - xs[max(0, i - k)] for i in range(n)])
        ty = np.array([ys[min(n - 1, i + k)] - ys[max(0, i - k)] for i in range(n)])
        ang = np.arctan2(tx, ty)
        out['tangent_break_max_rad'] = float(np.max(np.abs(np.diff(ang)))) if n >= 2 else 0.0
        out['tangent_nodes'] = int(2 * k + 1)
    return out


def _rail_track(x, y, z, s, i, m_top, c_top, out_sign, y_lo, y_hi, y_mid,
                shelf_line=None, shelf_u_m=0.0):
    """ПОЛИЛИНИЯ оси нити: локальные сечения бинов, предсказание от предыдущих узлов.

    Прямая x = s*y + i годится только как СТАРТ в середине ядра детекции (там
    головка плотно измерена): на кривой и на стрелке окно отбора ±60 мм вдоль
    прямой теряет нить через 10-15 м. Дальше сечение бина ищется в окне
    ±POLY_U_WIN_M от предсказания по последним принятым узлам — то есть вдоль
    изгиба пути, — а последовательность узлов и есть полилиния оси (монотонная по Y
    по построению: узлы идут по сетке бинов).

    shelf_line — измеренная полка (shelf_z в y_ref + наклон вдоль пути): если задана,
    бин принимается только когда его ИЗМЕРЕННЫЙ верх головки лежит в коридоре
    POLY_TOP_LO_M..POLY_TOP_HI_M над полкой. Бин с верхом «в полу» или «в потолке»
    (тень, постель, плечо, чужая ветка) становится ПРОПУСКОМ, а не протягивается
    дальше — иначе лента уезжает сквозь пол (замерено: min 0.00 м над полкой).

    Возвращает dict в формате _rail_bin_sections (y_c, u_lo, u_hi, widths, nodes,
    z_nodes, dz_nodes, bins, y_dense, gaps, ribbons, n_pts_sel, b0, n_span) по
    ИЗМЕРЕННЫМ бинам плюс узлы меша по всей сетке между ними:
    grid_bins/grid_y/grid_x/grid_z (сглаженные, разрывы зашиты линейно),
    n_rejected, n_rejected_height, n_clamped_height, smooth_resid_m.
    Или None, если трек не начался.
    """
    b0 = math.floor(float(y_lo) / POLY_BIN_M) * POLY_BIN_M
    n_span = int(math.floor((float(y_hi) - b0) / POLY_BIN_M)) + 1
    ycs = b0 + (np.arange(n_span) + 0.5) * POLY_BIN_M
    # облако сортируем по Y один раз: окно бина — срез массива, а не маска по всему
    # облаку (иначе трекер стоит ~0.4 с на нить)
    o = np.argsort(y, kind='stable')
    ys_s, xs_s, zs_s = y[o], x[o], z[o]
    cnt = {'height': 0}

    def _shelf_at(yv, sh=shelf_line, u0=shelf_u_m):
        if sh is None:
            return None
        # полка под САМОЙ нитью: у профиля есть поперечный наклон shelf_slope_u
        # (0.012-0.029 м/м), а нить стоит на u ≈ ±0.8 м — это ±0.02 м к коридору
        return (sh['z_at_y_ref'] + sh.get('slope_u', 0.0) * u0
                + sh['slope_y'] * (yv - sh['y_ref']))

    def _slice(yc):
        a = int(np.searchsorted(ys_s, yc - POLY_WIN_M, 'left'))
        b = int(np.searchsorted(ys_s, yc + POLY_WIN_M, 'right'))
        return a, b

    def _node(bi, yc, xp, zp):
        a, b = _slice(yc)
        if b - a < RAIL_MIN_PTS_BIN:
            return None
        dxx = xs_s[a:b] - xp
        dep = zp - zs_s[a:b]
        sel = ((np.abs(dxx) <= POLY_U_WIN_M) & (dep >= -RAIL_DEPTH_UP_M)
               & (dep < RAIL_DEPTH_DOWN_M))
        if int(sel.sum()) < RAIL_MIN_PTS_BIN:
            return None
        sec = _section_from_points(out_sign * dxx[sel], zs_s[a:b][sel])
        if sec is None:
            return None
        top = float(sec['nodes_z'].max())
        sh = _shelf_at(yc)
        if sh is not None and not (POLY_TOP_LO_M <= top - sh <= POLY_TOP_HI_M):
            cnt['height'] += 1
            return None
        u_c = 0.5 * (sec['u_lo'] + sec['u_hi'])
        return {'y_c': float(yc), 'x_c': float(xp + out_sign * u_c),
                'z_top': top,
                'u_lo_rel': float(sec['u_lo'] - u_c),
                'u_hi_rel': float(sec['u_hi'] - u_c),
                'nodes_rel': np.asarray(sec['nodes_u'] - u_c, float),
                'nodes_z': np.asarray(sec['nodes_z'], float),
                'n_pts': int(sel.sum())}

    acc = {}
    start = None
    for bi in np.argsort(np.abs(ycs - y_mid))[:60]:
        bi = int(bi)
        yc = float(ycs[bi])
        xp, zp = s * yc + i, m_top * yc + c_top
        nd = _node(bi, yc, xp, zp)
        if nd is None:
            continue
        acc[bi] = nd
        start = bi
        break
    if start is None:
        return None
    for direction in (1, -1):
        miss = 0
        j = start
        last = start
        hist = [start]          # последние принятые бины (окно предсказания)
        while 0 <= j + direction < n_span:
            jn = j + direction
            yc = float(ycs[jn])
            # X-предсказание — МНК-прямая по последним POLY_PRED_N принятым узлам,
            # а не по двум: на разреженных дальних бинах разовый выброс центра уводит
            # предсказание и трек обрывается.
            # Z-гейт едет по ЗАРАНЕЕ ПОДОГНАННОЙ линии верха m_top*y + c_top (та же,
            # что у прямой развёртки): она сглажена по всему ядру, а локальная история
            # на 4-6 точках даёт разброс ±20 мм — при уклоне 1.5 % гейт глубины ±5 мм
            # обрывал трек уже через 5-10 м (замерено на doubleT_obstacle).
            k = min(POLY_PRED_N, len(hist))
            zp = m_top * yc + c_top
            if k >= 2:
                hy = np.array([acc[int(b)]['y_c'] for b in hist[-k:]], float)
                hx = np.array([acc[int(b)]['x_c'] for b in hist[-k:]], float)
                if float(np.ptp(hy)) > 1e-6:
                    kx_ = float(np.polyfit(hy, hx, 1)[0])
                    xp = float(np.mean(hx)) + kx_ * (yc - float(np.mean(hy)))
                else:
                    xp = float(hx[-1])
            else:
                xp = acc[last]['x_c']
            nd = _node(jn, yc, xp, zp)
            if nd is None:
                miss += 1
                j = jn
                if miss > POLY_MAX_MISS:
                    break
                continue
            miss = 0
            acc[jn] = nd
            last = jn
            hist.append(jn)
            j = jn

    bins = np.array(sorted(acc), dtype=np.int64)
    raw_x = np.array([acc[int(b)]['x_c'] for b in bins], float)
    raw_z = np.array([acc[int(b)]['z_top'] for b in bins], float)
    grid = np.arange(int(bins[0]), int(bins[-1]) + 1)
    # разрывы внутри трека зашиваются линейно (POLY_MAX_MISS ограничивает их 2 м —
    # это интерполяция между соседними измерениями, а не подмена прямой изломом)
    g_x = np.interp(grid, bins, raw_x)
    g_z = np.interp(grid, bins, raw_z)
    g_x, g_z, n_rej, resid, g_bad = _polyline_smooth(g_x, g_z)
    # Жёсткая страховка: после сглаживания ни один узел внутри данных не должен
    # оказаться ниже полки+POLY_TOP_HARD_M. Принятые бины уже в коридоре, а линейная
    # интерполяция/сглаживание коридор сохраняют (полка линейна по y) — кламп ловит
    # вырожденные случаи и СЧИТАЕТСЯ, чтобы не «молча».
    n_clamp = 0
    if shelf_line is not None:
        sh_grid = np.array([_shelf_at(float(v)) for v in ycs[grid]], float)
        lo = sh_grid + POLY_TOP_LO_M
        hi = sh_grid + POLY_TOP_HI_M
        n_clamp = int(np.sum((g_z < lo) | (g_z > hi)))
        g_z = np.clip(g_z, lo, hi)
    return {
        'b0': b0, 'n_span': n_span,
        'y_c': np.array([acc[int(b)]['y_c'] for b in bins], float),
        'u_lo': np.array([acc[int(b)]['u_lo_rel'] for b in bins], float),
        'u_hi': np.array([acc[int(b)]['u_hi_rel'] for b in bins], float),
        'widths': np.array([acc[int(b)]['u_hi_rel'] - acc[int(b)]['u_lo_rel']
                            for b in bins], float),
        'nodes': np.array([acc[int(b)]['nodes_rel'] for b in bins], float),
        'z_nodes': np.array([acc[int(b)]['nodes_z'] for b in bins], float),
        'dz_nodes': np.array([acc[int(b)]['nodes_z'] - acc[int(b)]['z_top']
                              for b in bins], float),
        'bins': bins, 'n_pts_sel': int(sum(acc[int(b)]['n_pts'] for b in bins)),
        'n_pts_bin': np.array([acc[int(b)]['n_pts'] for b in bins], float),
        'grid_bad': np.asarray(g_bad, bool),
        'y_dense': [float(b0 + int(bins[0]) * POLY_BIN_M),
                    float(b0 + (int(bins[-1]) + 1) * POLY_BIN_M)],
        'n_bins_ok': int(bins.size),
        'gaps': [[float(b0 + (int(bins[k]) + 1) * POLY_BIN_M),
                  float(b0 + int(bins[k + 1]) * POLY_BIN_M)]
                 for k in range(bins.size - 1) if bins[k + 1] - bins[k] > 1],
        'ribbons': 1 + int(np.sum(np.diff(bins) > 1)) if bins.size else 0,
        'grid_bins': grid, 'grid_y': ycs[grid], 'grid_x': g_x, 'grid_z': g_z,
        'n_rejected': n_rej, 'smooth_resid_m': resid,
        'n_rejected_height': int(cnt['height']), 'n_clamped_height': int(n_clamp),
    }


def _polyline_tangent(ys, xs, tail=POLY_TAIL_N):
    """Касательная и кривизна на конце полилинии (квадратичная подгонка x(y)).

    Квадрат по последним tail узлам даёт и направление, и кривизну сразу: у хорды
    направление смещено на пол-угла дуги (окно 2.5 м при R=30 м — 2.4°, это 0.6 м
    ухода на 15 м продолжения), у квадрата — нет.
    """
    n = len(ys)
    k = max(3, min(tail, n))
    y = np.asarray(ys[-k:], float)
    x = np.asarray(xs[-k:], float)
    y0 = float(y[-1])
    dy = y - y0
    if float(np.ptp(dy)) < 1e-6:
        return None
    c2, c1, _c0 = np.polyfit(dy, x, 2)
    b = float(c1)
    hyp = math.hypot(1.0, b)
    return {'dir_y': 1.0 / hyp, 'dir_x': b / hyp,
            'len_m': float(np.hypot(np.diff(y), np.diff(x)).sum()),
            'kappa_1_m': float(abs(2.0 * float(c2)) / (1.0 + b * b) ** 1.5),
            # знак нужен для продолжения ПО ДУГЕ: кривизна в сторону +x при росте y
            'kappa_signed_1_m': float(2.0 * float(c2)) / (1.0 + b * b) ** 1.5}


def _extend_end(gy, gx, extra, z_at, forward, tail=POLY_TAIL_N, step=POLY_BIN_M,
                y_lim=None, z_start=None, blend_m=POLY_Z_BLEND_M,
                kappa_max_m=None):
    """Узлы продолжения от ОДНОГО конца полилинии: по ПОСЛЕДНЕЙ КРИВИЗНЕ (дуге).

    forward=True — от дальнего конца (gy[0], путь уходит в сторону меньших y),
    forward=False — от ближнего (gy[-1]). Направление берём из квадратичной подгонки
    хвоста (см. _polyline_tangent) и ВРАЩАЕМ его на каждом шаге на κ·step: прямая
    касательная на кривой уходит за 10 м на κ·L²/2 (при R=22 м это 2.3 м — замерено),
    дуга держит путь. Если κ не оценилась — прямая касательная, но не длиннее
    POLY_TANGENT_MAX_M (5 м).

    Высота: на конце данных z = z_start (верх последнего измеренного узла), дальше
    плавно (за blend_m метров) уходит на z_at(y) = полка + высота головки — иначе на
    стыке «данные/продолжение» получается ступенька по Z (замерено до 0.118 м).
    y_lim — границы облака по Y: за них продолжение не выходит.
    Возвращает (ys, xs, zs) в порядке удаления от конца.
    """
    if not extra or extra <= 1e-6 or len(gy) < 3:
        return [], [], []
    if forward:
        tg = _polyline_tangent(gy[::-1], gx[::-1], tail)
        flip = -1.0
    else:
        tg = _polyline_tangent(gy, gx, tail)
        flip = 1.0
    if tg is None:
        return [], [], []
    dy, dx = flip * tg['dir_y'], flip * tg['dir_x']
    y0 = float(gy[0]) if forward else float(gy[-1])
    x0 = float(gx[0]) if forward else float(gx[-1])
    kap = None
    if kappa_max_m is None:
        kap = flip * tg['kappa_signed_1_m']
        if abs(kap) < 1e-6:
            kap = None
    else:
        # жёстко заданная кривизна (для проверок): по модулю, в сторону из фита
        sgn = 1.0 if tg['kappa_signed_1_m'] * flip >= 0 else -1.0
        kap = sgn * abs(kappa_max_m)
        if abs(kap) < 1e-6:
            kap = None
    if kap is None:
        extra = min(extra, POLY_TANGENT_MAX_M)
    ys, xs, zs = [], [], []
    hy, hx = dy, dx
    # КАСАТЕЛЬНАЯ СТЫКА (единичный вектор на конце данных): по ней считается
    # ПОПЕРЕЧНЫЙ УХОД продолжения — перпендикулярное расстояние до прямой стыка.
    # Больше min(POLY_REACH_DRIFT_M, POLY_REACH_DRIFT_RATE * длина) — обрезаем
    # (жёсткий предел ухода, см. спеку 3.2/шаг 2): продолжение по дуге уходит от
    # прямой как kappa*L^2/2, и без предела уезжает на метры вбок.
    hy0, hx0 = dy, dx
    yv, xv = y0, x0
    d = step
    while d <= extra + 1e-9:
        yv = yv + hy * step
        xv = xv + hx * step
        dev = abs((yv - y0) * hx0 - (xv - x0) * hy0)
        if dev > min(POLY_REACH_DRIFT_M, POLY_REACH_DRIFT_RATE * d) + 1e-9:
            break
        if y_lim is not None and not (y_lim[0] - 1e-6 <= yv <= y_lim[1] + 1e-6):
            break
        z_t = float(z_at(yv))
        if z_start is not None:
            w = min(1.0, max(0.0, (d - step) / max(blend_m, 1e-6)))
            z_t = float(z_start) * (1.0 - w) + z_t * w
        ys.append(yv)
        xs.append(xv)
        zs.append(z_t)
        if kap is not None:
            ang = kap * step
            ca, sa = math.cos(ang), math.sin(ang)
            hy, hx = hy * ca - hx * sa, hy * sa + hx * ca
        d += step
    return ys, xs, zs


def _body_section_uv(sec):
    """(u, v) сечения ТЕЛА рельса в мм: контур Р65 по ГОСТ + ИЗМЕРЕННАЯ коронка.

    Контур берётся по измеренной ширине головки (2·half_width), поэтому верхний угол
    головки совпадает по u с крайним узлом измеренной коронки — ступеньки нет.
    Два верхних узла контура (верхняя площадка) СНИМАЮТСЯ, вместо них вставляются
    5 измеренных узлов коронки (максимум остаётся ровно там, где его измерил лидар).
    u — наружу, v — вверх от НИЗА подошвы (0..180 мм).
    """
    half_w = float(sec['half_width_m'])
    u_off = float(sec['u_off_m'])
    vz_mm = np.asarray(sec['vz_nodes_m'], float) * 1000.0
    verts, _src = rail_profile_mm(2.0 * half_w * 1000.0)
    # головку масштабируем по u под ИЗМЕРЕННУЮ ширину: иначе при узкой головке
    # (62 мм) угол контура (37.3 мм) оказывается СНАРУЖИ крайнего узла коронки, и
    # контур самопересекается (отсечение ушей даёт 9 треугольников вместо 18)
    k_head = (half_w * 1000.0) / (R65_HEAD_B_MM * 0.5)
    verts = [(u * k_head, v) if v >= HEAD_BOTTOM_V_MM - 1e-9 else (u, v)
             for (u, v) in verts]
    tops = [i for i, (_u, v) in enumerate(verts) if abs(v - R65_H_MM) < 1e-9]
    crown = [((u_off - half_w + 2.0 * half_w * k / 4.0) * 1000.0,
              R65_H_MM + float(vz_mm[k])) for k in (4, 3, 2, 1, 0)]
    if len(tops) == 2 and tops[1] == tops[0] + 1:
        body = [(u + u_off * 1000.0, v) for (u, v) in verts[:tops[0]]] + crown \
            + [(u + u_off * 1000.0, v) for (u, v) in verts[tops[1] + 1:]]
    else:   # неожиданная структура контура — оставляем ГОСТ целиком
        body = [(u + u_off * 1000.0, v) for (u, v) in verts]
    return body


def _body_section_loc(sec):
    """Локальные (du, dv) точек сечения тела вокруг ЦЕНТРА ВЕРХА головки, м.

    du — наружу, dv — вниз от верха головки, с подуклонкой 1:20 (поворот сечения).
    Возвращает (loc, dv_min, dv_max): dv_max — насколько ВЕРХ меша стоит выше узла
    (обычно +0.000…+0.003 м: подуклонка поднимает наружную кромку площадки), dv_min —
    низ подошвы (−0.1835 м).
    """
    uv = _body_section_uv(sec)
    cos_t, sin_t = math.cos(PODUKLONKA_RAD), math.sin(PODUKLONKA_RAD)
    loc = []
    for (u_mm, v_mm) in uv:
        du0 = u_mm / 1000.0
        dv0 = (v_mm - R65_H_MM) / 1000.0
        loc.append((du0 * cos_t - dv0 * sin_t, du0 * sin_t + dv0 * cos_t))
    dv = [d for (_u, d) in loc]
    return loc, float(min(dv)), float(max(dv))


def _rail_mesh_body(ys, xs, zs, sec, out_sign, pair_ok=None):
    """Меш ТЕЛА рельса: контур Р65 (ГОСТ) + измеренная коронка, на узлах полилинии.

    Верхняя площадка контура — ровно измеренная коронка (см. _body_section_uv),
    тело ниже — ГОСТ (шейка, подошва, выкружки), высота 180 мм, подошва 150 мм,
    подуклонка 1:20 (поворот сечения вокруг центра верха головки). Сечение
    поворачивается ещё и по локальной касательной пути (как у полоски).
    """
    uv = _body_section_uv(sec)
    n_p = len(uv)
    loc, _dv_min, _dv_max = _body_section_loc(sec)
    ys = np.asarray(ys, float)
    xs = np.asarray(xs, float)
    zs = np.asarray(zs, float)
    n = ys.size
    if n < 2:
        return [], [], n_p
    nx, ny = _path_normals(ys, xs, out_sign)
    verts = []
    for i in range(n):
        for (du, dv) in loc:
            verts.append([float(xs[i] + du * nx[i]), float(ys[i] + du * ny[i]),
                          float(zs[i] + dv)])
    if pair_ok is None:
        pair_ok = np.ones(n - 1, dtype=bool)
    faces = []
    for j in range(n - 1):
        if not pair_ok[j]:
            continue
        a0, a1 = j * n_p, (j + 1) * n_p
        for p in range(n_p):
            q = (p + 1) % n_p
            faces.append([a0 + p, a0 + q, a1 + q])
            faces.append([a0 + p, a1 + q, a1 + p])
    caps = _triangulate_polygon(uv)
    last = (n - 1) * n_p
    for (t0, t1, t2) in caps:
        faces.append([t2, t1, t0])
        faces.append([last + t0, last + t1, last + t2])
    dv_min = min(dv for (_du, dv) in loc)
    return verts, faces, n_p, float(dv_min)


def _rail_mesh_path(ys, xs, zs, sec, out_sign, pair_ok=None):
    """Меш: медианное сечение на каждом узле ПОВЁРНУТО по касательной (локальный базис).

    В узле i направление берётся по соседним узлам (центральная разность), нормаль —
    поворот касательной на 90° со знаком наружу (out_sign). Вершина узла i, узел
    сечения k: p_i + out_sign*u_k*n_i, z = z_i + vz[k]. Так меш повторяет изгиб
    пути, а не режет его прямой. pair_ok[i] = False — узлы i и i+1 не соединять
    (режим only_observed: поверхность рвётся там, где бинов нет).
    """
    ys = np.asarray(ys, float)
    xs = np.asarray(xs, float)
    zs = np.asarray(zs, float)
    n = ys.size
    if n < 2:
        return [], []
    nx, ny = _path_normals(ys, xs, out_sign)
    half_w = sec['half_width_m']
    u_off = sec['u_off_m']
    vz = sec['vz_nodes_m']
    n_nodes = RAIL_NODES
    verts = []
    for i in range(n):
        for k in range(n_nodes):
            u = u_off - half_w + 2.0 * half_w * k / (n_nodes - 1)
            verts.append([float(xs[i] + u * nx[i]), float(ys[i] + u * ny[i]),
                          float(zs[i] + vz[k])])
    if pair_ok is None:
        pair_ok = np.ones(n - 1, dtype=bool)
    faces = []
    for j in range(n - 1):
        if not pair_ok[j]:
            continue
        a0, a1 = j * n_nodes, (j + 1) * n_nodes
        for k in range(n_nodes - 1):
            faces.append([a0 + k, a0 + k + 1, a1 + k + 1])
            faces.append([a0 + k, a1 + k + 1, a1 + k])
    return verts, faces


def _path_normals(ys, xs, out_sign):
    """Нормали узлов полилинии: поворот локальной касательной на 90° наружу.

    Касательная — центральная разность соседних узлов (на концах — крайний отрезок).
    """
    ys = np.asarray(ys, float)
    xs = np.asarray(xs, float)
    n = ys.size
    tx = np.empty(n)
    ty = np.empty(n)
    if n < 2:
        return np.zeros(n), np.zeros(n)
    k = max(1, int(POLY_TANGENT_NODES))
    for i in range(n):
        a, b = max(0, i - k), min(n - 1, i + k)
        tx[i] = xs[b] - xs[a]
        ty[i] = ys[b] - ys[a]
    tx[0], ty[0] = xs[min(k, n - 1)] - xs[0], ys[min(k, n - 1)] - ys[0]
    tx[-1], ty[-1] = xs[-1] - xs[max(0, n - 1 - k)], ys[-1] - ys[max(0, n - 1 - k)]
    ln = np.hypot(tx, ty)
    ln[ln < 1e-9] = 1.0
    return out_sign * (-ty / ln), out_sign * (tx / ln)


def _axis_to_rail_metric(x, y, z, ax_x, ax_y, ax_nx, ax_ny, shelf_at, bucket):
    """Расстояние от оси меша до ближайших точек рельса — доказательство, что меш
    идёт ПО рельсу, а не срезает кривую.

    Для каждого узла оси: окно ±POLY_METRIC_WIN_M по Y, полоса головки
    POLY_BAND_LO_M..POLY_BAND_HI_M над измеренной полкой (shelf_at — её уровень в
    каждом узле), |u| <= POLY_U_LIM_M по ЛОКАЛЬНОЙ нормали узла (то есть поперёк
    пути, не по X). Расстояние узла — минимум |u| по отобранным точкам. Узлы
    разложены по корзинам bucket ('measured' — бин измерен, 'bridged' — разрыв
    зашит интерполяцией, 'outside' — продолжение за пределами данных).
    Возвращает {'measured': {...}, 'bridged': {...}, 'outside': {...}, лимиты}.
    """
    y = np.asarray(y, float)
    order = np.argsort(y)
    ys_s = y[order]
    res = {}
    for key in ('measured', 'bridged', 'outside'):
        res[key] = {'vals': [], 'n_nodes': 0, 'n_no_pts': 0}
    for i in range(len(ax_x)):
        key = str(bucket[i])
        res[key]['n_nodes'] += 1
        yc = float(ax_y[i])
        a = int(np.searchsorted(ys_s, yc - POLY_METRIC_WIN_M, 'left'))
        b = int(np.searchsorted(ys_s, yc + POLY_METRIC_WIN_M, 'right'))
        if b - a < 3:
            res[key]['n_no_pts'] += 1
            continue
        idx = order[a:b]
        sel = ((z[idx] >= shelf_at[i] + POLY_BAND_LO_M)
               & (z[idx] <= shelf_at[i] + POLY_BAND_HI_M))
        if not sel.any():
            res[key]['n_no_pts'] += 1
            continue
        idx = idx[sel]
        u = ((x[idx] - ax_x[i]) * ax_nx[i] + (y[idx] - ax_y[i]) * ax_ny[i])
        au = np.abs(u)
        au = au[au <= POLY_U_LIM_M]
        if au.size == 0:
            res[key]['n_no_pts'] += 1
            continue
        res[key]['vals'].append(float(au.min()))
    out = {}
    for key in ('measured', 'bridged', 'outside'):
        v = res[key]['vals']
        out[key] = {
            'median_m': float(np.median(v)) if v else None,
            'p95_m': float(np.percentile(v, 95)) if v else None,
            'n_nodes_with_points': len(v),
            'n_nodes': res[key]['n_nodes'],
            'n_nodes_without_points': res[key]['n_no_pts'],
        }
    out['u_limit_m'] = POLY_U_LIM_M
    out['band_above_shelf_m'] = [POLY_BAND_LO_M, POLY_BAND_HI_M]
    out['window_y_m'] = POLY_METRIC_WIN_M
    return out


def _rail_sweep_extent_poly(dense_len, shelf_slope_y, kappa, sweep_extra_m):
    """Запас продолжения оси-полилинии за пределы измеренного.

    Прежние пределы ухода: по высоте RAIL_DRIFT_LIMIT_M от уровня в середине
    измеренного участка (высота меша над полкой постоянна, поэтому уход =
    |наклон полки| * длина по пути), поперёк пути RAIL_X_DRIFT_LIMIT_M. Для
    продолжения по КАСАТЕЛЬНОЙ поперечный уход — это уход от реального изогнутого
    пути: kappa*L^2/2 (kappa — кривизна на конце, 1/м), отсюда L <= sqrt(2*0.5/kappa).
    Прямолинейный запас 15 м остаётся верхней границей.
    Возвращает (extra_m, cap_h_m, cap_x_m).
    """
    if sweep_extra_m is None:
        return None, None, None
    half = 0.5 * float(dense_len)
    cap_h = None
    if shelf_slope_y is not None and abs(shelf_slope_y) > 1e-9:
        cap_h = max(0.0, RAIL_DRIFT_LIMIT_M / abs(shelf_slope_y) - half)
    cap_x = None
    if kappa is not None and kappa > 1e-9:
        cap_x = math.sqrt(2.0 * RAIL_X_DRIFT_LIMIT_M / kappa)
    cands = [float(sweep_extra_m)]
    if cap_h is not None:
        cands.append(cap_h)
    if cap_x is not None:
        cands.append(cap_x)
    return float(min(cands)), cap_h, cap_x


def _pair_plan(rl, rr, shelf_line, sweep_extra_m, y_search,
               gauge_lo_m=POLY_PAIR_GAUGE_LO_M, gauge_hi_m=POLY_PAIR_GAUGE_HI_M,
               step=POLY_BIN_M, gauge_override=None, cloud=None):
    """ЖЁСТКАЯ ПАРА: общая ЦЕНТРАЛЬНАЯ ЛИНИЯ + ОДНА колея на кадр.

    Нити велись независимо, и ошибка бокового ведения уходила в «колею»: по узлам
    расстояние гуляло на 30-130 мм внутри данных и до 780 мм в продолжении. Здесь:
      * центр u_c(y) = среднее узлов обеих нитей там, где измерены ОБЕ; где только
        одна — её узел смещён на  G/2 (нить не уводим с физического рельса);
      * колея G = МЕДИАНА измеренных расстояний по перекрытию, зажатая в
        [1.58, 1.61] м (Р65: головка 60-90 мм, номинал по осям ~1590-1595);
      * нити ставятся как u_c  G/2 — колея по узлам постоянна и внутри данных, и
        в продолжении;
      * продолжение — ОБЩАЯ центральная линия по СВОЕЙ кривизне (дуга), нити тем
        же  G/2 (раньше каждая нить продолжалась своей касательной).

    ПЛАН СТРОИТСЯ ВСЕГДА, когда на кадре измерена ХОТЯ БЫ ОДНА нить: если второй
    нити нет (нет bsec) или общей полосы у нитей нет (<4 общих бинов) — колею
    измерить нечем, берётся gauge_override (номинал, зажатый в 1580..1610), а оси
    измеренных нитей ОСТАЮТСЯ на своих местах (узел измеренной нити = её же узел,
    ненайденная нить ставится на G от неё). Возвращает None только если нет обеих.

    Возвращает dict с итоговыми узлами (ys, xL, xR), диагнозами колеи и планом
    продолжения (запас/пределы/κ) или None.
    """
    bl, br = rl.get('bsec'), rr.get('bsec')
    if bl is None and br is None:
        return None
    if gauge_override is None and (bl is None or br is None):
        return None
    spans = [(float(b['grid_y'][0]), float(b['grid_y'][-1]))
             for b in (bl, br) if b is not None]
    b0 = float((bl if bl is not None else br)['b0'])
    y_lo = min(s[0] for s in spans)
    y_hi = max(s[1] for s in spans)
    n_lo = int(round((y_lo - b0) / step - 0.5))
    n_hi = int(round((y_hi - b0) / step - 0.5))
    bins = np.arange(n_lo, n_hi + 1)
    gy = b0 + (bins + 0.5) * step

    def _map(bsec, key):
        return {} if bsec is None else {int(b): float(v)
                                        for b, v in zip(bsec['grid_bins'], bsec[key])}

    def _pts_map(bsec, key):
        return {} if bsec is None else {int(b): bsec[key][i]
                                        for i, b in enumerate(bsec['bins'])}

    xm_l, xm_r = _map(bl, 'grid_x'), _map(br, 'grid_x')
    pts_l, pts_r = _pts_map(bl, 'n_pts_bin'), _pts_map(br, 'n_pts_bin')
    bad_l, bad_r = _pts_map(bl, 'grid_bad'), _pts_map(br, 'grid_bad')
    xl = np.array([xm_l.get(int(b), np.nan) for b in bins], float)
    xr = np.array([xm_r.get(int(b), np.nan) for b in bins], float)
    # ЧТО СЧИТАЕТСЯ ИЗМЕРЕНИЕМ НИТИ: grid_x есть во ВСЕХ бинах между крайними, но в
    # бинах без точек это ЛИНЕЙНАЯ ЗАШИВКА разрыва (и она же сглажена). Колею
    # измеряем только там, где измерены ОБЕ нити (иначе медиана колеи едет по
    # зашитым узлам); сам центр при этом считается по непрерывным осям нитей —
    # зашитый узел это интерполяция ПО ЭТОЙ ЖЕ нити, и рвать по нему растяжки
    # выбора нельзя (замерено: растяжки дробятся, «отброшенная» нить уезжает)
    meas_l = np.zeros(bins.size, bool) if bl is None else np.isin(bins, bl['bins'])
    meas_r = np.zeros(bins.size, bool) if br is None else np.isin(bins, br['bins'])
    have_l = np.isfinite(xl) & meas_l
    have_r = np.isfinite(xr) & meas_r
    both_m = have_l & have_r
    d_meas = (xr[both_m] - xl[both_m]) * 1000.0
    g_source = 'measured'
    if gauge_override is None:
        if d_meas.size < 4:       # общей полосы нет — колею измерить нечем
            return None
        g_raw = float(np.median(d_meas)) / 1000.0
    else:
        # вторая нить отсутствует или общей полосы у нитей нет: колея — номинал
        g_raw = float(gauge_override)
        g_source = 'nominal'
    g = float(np.clip(g_raw, gauge_lo_m, gauge_hi_m))
    if gauge_override is None:
        both = np.isfinite(xl) & np.isfinite(xr)
        only_l = np.isfinite(xl) & ~np.isfinite(xr)
        only_r = np.isfinite(xr) & ~np.isfinite(xl)
    else:
        # КОЛЕЯ НЕИЗВЕСТНА (номинал): достоверных «противоречий» нитей тут нет, поэтому
        # узел ИЗМЕРЕННОЙ нити НЕ сдвигаем — на бине, где измерена одна нить, центр
        # берётся по ней (её узел = её же измерение), а где измерены обе (редкий случай
        # при отсутствии общей полосы) — среднее, симметрично для обеих
        both = have_l & have_r
        only_l = have_l & ~have_r
        only_r = have_r & ~have_l
    kind = ('one_rail' if (bl is None or br is None) else 'both_rails')
    # ЦЕНТР ПО НАДЁЖНОСТИ: где нити противоречат друг другу (колея ушла от G
    # больше чем на POLY_PAIR_OFF_M), среднее брать нельзя — плохое измерение тянет
    # ОБЕ нити. Центр берётся по БОЛЕЕ НАДЁЖНОЙ нити, вторая ставится тем же G от
    # него (см. ниже: надёжность = отклонение узла от робастного тренда центра).
    # тренд каждой нити по её СОБСТВЕННЫМ узлам (скользящая медиана ±3 бина)
    def _own_trend(vals):
        out = np.full(vals.size, np.nan)
        for i in range(vals.size):
            a, b2 = max(0, i - 3), min(vals.size, i + 4)
            w = vals[a:b2][np.isfinite(vals[a:b2])]
            if w.size >= 3:
                out[i] = float(np.median(w))
        return out

    tr_l, tr_r = _own_trend(xl), _own_trend(xr)
    n = bins.size
    xc = np.full(n, np.nan)
    agree = both & (np.abs((xr - xl) - g) <= POLY_PAIR_OFF_M)
    # при НОМИНАЛЬНОЙ колее «противоречий» не разбираем: где обе нити измерены —
    # среднее (симметрично), где одна — её собственный узел (см. классификацию выше)
    disp = (both & ~agree) if gauge_override is None else np.zeros(both.size, bool)
    agree = both if gauge_override is not None else agree
    hole = ~(both | only_l | only_r)
    cnt = {'both': int(agree.sum()), 'reliable': 0, 'single': int(only_l.sum() + only_r.sum()),
           'reliable_left': 0, 'runs': 0}
    # ЯКОРЯ измерения: узлы, где центр взят из ИЗМЕРЕННОГО узла одной нити (среднее
    # обеих нитей — тоже измерение, но согласованное, его сглаживать можно).
    protect = np.zeros(n, bool)
    # код источника центра по бинам (диагностика/замеры): 1 — среднее обеих,
    # 2 — надёжная ЛЕВАЯ, 3 — надёжная ПРАВАЯ, 4 — только левая, 5 — только правая,
    # 6 — узел выбранной нити, восстановленный интерполяцией по её же узлам
    src = np.zeros(n, np.int8)
    src[agree] = 1
    src[only_l] = 4
    src[only_r] = 5
    xc[agree] = 0.5 * (xl[agree] + xr[agree])
    xc[only_l] = xl[only_l] + 0.5 * g
    xc[only_r] = xr[only_r] - 0.5 * g
    # РОБАСТНЫЙ ТРЕНД ЦЕНТРА по бинам с ОДНОЗНАЧНЫМ центром (нити согласованы или
    # измерена одна нить): центр пути — гладкая линия, поэтому отклонение от него
    # надёжнее, чем согласованность нити с самой собой (сбитая нить согласована
    # с собой идеально, но уходит от пути).
    ct = np.where(agree, 0.5 * (xl + xr),
                  np.where(only_l, xl + 0.5 * g,
                           np.where(only_r, xr - 0.5 * g, np.nan)))
    tr_c = _own_trend(ct)
    cl = xl + 0.5 * g          # центр, если верить ЛЕВОЙ нити
    cr = xr - 0.5 * g          # центр, если верить ПРАВОЙ нити

    def _dev(xv, xc_one, tr_own, k):
        """Отклонение узла нити от пути: от тренда ЦЕНТРА; если тренда рядом нет —
        от скользящей медианы СВОЕЙ нити (прежний балл); если и её нет — штраф."""
        if np.isfinite(tr_c[k]):
            return float(abs(xc_one[k] - tr_c[k]))
        if np.isfinite(tr_own[k]):
            return float(abs(xv[k] - tr_own[k]))
        return 9.9

    def _pts(k, pts, bad):
        return pts.get(int(bins[k]), 0.0) - (POLY_PAIR_PTS_W if bad.get(int(bins[k])) else 0.0)

    # ОДНА НИТЬ НА ВСЮ РАСТЯЖКУ: внутри участка, где нити противоречат друг другу
    # (|Δ−G| > порога), выбор делался ПО БИНАМ, и он «прыгал» (левая/правая через
    # один) — центр получал ступень 90 мм и излом наклона 0.16-0.26 1/м (замерено
    # на roundT_doubleT f0 и roundT_squareT f126). Узел берётся из ОДНОЙ нити на всю
    # растяжку: нить ведётся предсказанием от предыдущих узлов, поэтому её ряд
    # непрерывен, а где её узел не измерен — интерполируется ПО ЕЁ ЖЕ узлам.
    k = 0
    while k < n:
        if agree[k] or hole[k]:
            k += 1
            continue
        e = k
        while e < n and not (agree[e] or hole[e]):
            e += 1
        idx = np.arange(k, e)
        dd = idx[disp[idx]]
        cnt['runs'] += 1
        take_l = None
        if dd.size:
            sl = sum(_dev(xl, cl, tr_l, t) for t in dd)
            sr = sum(_dev(xr, cr, tr_r, t) for t in dd)
            if abs(sl - sr) > 0.002 * dd.size:
                take_l = sl < sr
            else:      # равные отклонения — по числу точек (минус вес MAD-отбраковки)
                take_l = (sum(_pts(t, pts_l, bad_l) for t in dd)
                          >= sum(_pts(t, pts_r, bad_r) for t in dd))
        cand = None
        if take_l is not None:
            base = (cl[idx] if take_l else cr[idx]).copy()
            fin = np.isfinite(base)
            if int(fin.sum()) >= 2:
                if (~fin).any():
                    base[~fin] = np.interp(idx[~fin], idx[fin], base[fin])
                cand = base
        if cand is None:           # нить нечем провести по растяжке — по бинам
            for t in idx:
                if not disp[t]:
                    continue
                dl = _dev(xl, cl, tr_l, t)
                dr = _dev(xr, cr, tr_r, t)
                if abs(dl - dr) > 0.002:
                    tl = dl < dr
                else:
                    tl = _pts(t, pts_l, bad_l) >= _pts(t, pts_r, bad_r)
                xc[t] = cl[t] if tl else cr[t]
                cnt['reliable'] += 1
                protect[t] = True
                src[t] = 2 if tl else 3
                if tl:
                    cnt['reliable_left'] += 1
            k = e
            continue
        xc[idx] = cand
        protect[idx] = disp[idx]      # якоря — только противоречивые бины
        for t in idx:
            if disp[t]:
                cnt['reliable'] += 1
                src[t] = 2 if take_l else 3
                if take_l:
                    cnt['reliable_left'] += 1
            else:                  # измерена одна нить: её узел, если она выбрана;
                own = 4 if only_l[t] else 5        # иначе — интерполяция по выбранной
                src[t] = own if (own == 4) == take_l else 6
        k = e
    n_rel_l = int(cnt['reliable_left'])
    ok = np.isfinite(xc)
    if int(ok.sum()) < 4:
        return None
    if not ok.all():          # дырки внутри общего участка — линейно между соседями
        xc = np.interp(np.arange(bins.size), np.flatnonzero(ok), xc[ok])
    # центр сглаживаем тем же фильтром (MAD ±3 + узкое среднее): на переходе
    # «измерены обе / измерена одна» формула центра меняется, и без сглаживания
    # там получается излом наклона (замерено 0.05-0.09 1/м)
    xc_raw = xc.copy()
    # УЗЛЫ, ВЗЯТЫЕ ПО НАДЁЖНОЙ НИТИ (protect) — ЯКОРЯ измерения: их сглаживание не
    # трогает (иначе MAD выбрасывает именно их и возвращает нить к среднему — сдвиг
    # от сырого измерения вырастал до p95 41-64 мм на roundT_doubleT). Чтобы якоря
    # при этом НЕ тянули за собой соседей (сырой узел рядом со сглаженным давал
    # излом наклона до 0.16-0.26 1/м), на время сглаживания они заменяются линейной
    # интерполяцией по остальным узлам; после сглаживания якоря возвращаются сырыми.
    if protect.any() and int((~protect).sum()) >= 4:
        xc_sm = xc.copy()
        xc_sm[protect] = np.interp(np.flatnonzero(protect), np.flatnonzero(~protect),
                                   xc[~protect])
    else:
        xc_sm = xc
    xc, _zc_unused, _nrej_c, _resid_c, _bad_c = _polyline_smooth(xc_sm, np.zeros_like(xc_sm))
    xc[protect] = xc_raw[protect]
    reach = None
    if cloud is not None:
        ys_o, xs_o, zs_o = (np.asarray(a, float) for a in cloud)
        o = np.argsort(ys_o)
        reach = {'cloud': (ys_o[o], xs_o[o], zs_o[o]),
                 'top': {rl.get('side', 'left'): (rl.get('m_top'), rl.get('c_top')),
                         rr.get('side', 'right'): (rr.get('m_top'), rr.get('c_top'))},
                 # ИЗМЕРЕННЫЙ конец КАЖДОЙ нити: правило остановки считается от него
                 # (трекер мог оборваться раньше данных — тогда лента удлиняется)
                 'meas_end': {r2.get('side'): _own_ends(r2) for r2 in (rl, rr)},
                 # РАСТР ВНУТРЕННОСТИ ТЕЛА рельса (для предохранителя «≥5 точек
                 # внутри тела»): строится по СЕЧЕНИЮ нити, сжатому на 5 мм
                 'body': {rl.get('side', 'left'): _reach_body_raster(rl.get('sec')),
                          rr.get('side', 'right'): _reach_body_raster(rr.get('sec'))},
                 # НАКЛОН ХВОСТА КАЖДОЙ НИТИ (её СВОИМИ измеренными узлами, МНК по
                 # последним POLY_REACH_TAIL_N бинам с обоих концов): продолжение
                 # обязано начаться наклоном хвоста — см. _reach_walk и
                 # _extension_drift_trim. Считается по сырым осям нитей, а не по
                 # сглаженной центральной линии: у нитей РАЗНЫЕ измеренные концы, и
                 # наклон центра на конце сетки — это наклон только той нити, что
                 # измерена дальше.
                 'tail': {s2: _rail_tail_slopes(rl if s2 == rl.get('side') else rr)
                          for s2 in (rl.get('side', 'left'), rr.get('side', 'right'))}}
    return _pair_finish(gy, xc, g, shelf_line, sweep_extra_m, y_search, {
        'gauge_raw_m': g_raw,
        'gauge_clamped_mm': float((g_raw - g) * 1000.0),
        'gauge_spread_measured_mm': (None if d_meas.size == 0
                                     else float(np.ptp(d_meas))),
        'gauge_source': g_source,
        'n_bins_both': int(both.sum()), 'n_bins_one': int(only_l.sum() + only_r.sum()),
        'n_center_both': int(cnt['both']), 'n_center_reliable': int(cnt['reliable']),
        'n_center_single': int(cnt['single']), 'n_reliable_left': int(n_rel_l),
        'n_center_runs': int(cnt['runs']),
        'gauge_off_threshold_mm': float(1000.0 * POLY_PAIR_OFF_M),
        'center_bins': bins, 'center_grid_y': gy, 'center_src': src,
        'center_kind': kind,
        'center_source': ('центр: по среднему обеих нитей %d бинов, по НАДЁЖНОЙ нити '
                          '%d бинов в %d растяжках (левая выбрана %d раз; порог '
                          '|Δ−G| > %.0f мм; нить выбирается на ВСЮ растяжку), '
                          'по одной нити %d бинов; колея %.1f мм (%s); продолжение — '
                          'общая дуга по κ центра'
                          % (int(cnt['both']), int(cnt['reliable']), int(cnt['runs']),
                             int(n_rel_l), 1000.0 * POLY_PAIR_OFF_M,
                             int(cnt['single']), 1000.0 * g,
                             'ИЗМЕРЕНА' if g_source == 'measured'
                             else 'НОМИНАЛ: общей полосы у нитей нет')),
    }, reach=reach)


def _pair_finish(gy, xc, g, shelf_line, sweep_extra_m, y_search, extra, step=POLY_BIN_M,
                 reach=None):
    """ХВОСТ ПЛАНА ПАРЫ (общий для «по двум нитям», «по одной» и «по номиналу»):
    продолжение ОБЩЕГО центра по своей кривизне, пределы ухода, запас и сборка узлов
    ОБЕИХ нитей (xL = центр − G/2, xR = центр + G/2). gy/xc — центр на сетке бинов,
    g — колея (м), extra — диагностика плана (счётчики, строка источника).

    reach — данные для ПРАВИЛА ОСТАНОВКИ ПО ДАННЫМ: {'cloud': (y_sorted, x, z),
    'top': {side: (m_top, c_top)}}. Продолжение обрезается по подтверждённому
    данным концу (см. _rail_reach), одинаково для ОБЕИХ нитей (пара жёсткая)."""
    dense_len_c = float(np.hypot(np.diff(gy), np.diff(xc)).sum()) + step
    tg_far = _polyline_tangent(gy[::-1], xc[::-1])
    tg_near = _polyline_tangent(gy, xc)
    k_far = None if tg_far is None else tg_far['kappa_1_m']
    k_near = None if tg_near is None else tg_near['kappa_1_m']
    slope_y = None
    if shelf_line is not None:
        slope_y = float(shelf_line['slope_y'])
    extra_far, cap_h, cap_x_far = _rail_sweep_extent_poly(
        dense_len_c, slope_y, k_far, sweep_extra_m)
    extra_near, _ch2, cap_x_near = _rail_sweep_extent_poly(
        dense_len_c, slope_y, k_near, sweep_extra_m)
    if sweep_extra_m is None:
        extra_far = 0.0 if tg_far is None else min(
            200.0, abs(float(y_search[0]) - float(gy[0])) / max(abs(tg_far['dir_y']), 0.2))
        extra_near = 0.0 if tg_near is None else min(
            200.0, abs(float(y_search[1]) - float(gy[-1])) / max(abs(tg_near['dir_y']), 0.2))
        cap_h = cap_x_far = cap_x_near = None
    ys_f, xs_f, _zs_f = _extend_end(gy, xc, extra_far, lambda yv: 0.0, forward=True,
                                    y_lim=(y_search[0], y_search[1]),
                                    tail=POLY_REACH_TAIL_N)
    ys_n, xs_n, _zs_n = _extend_end(gy, xc, extra_near, lambda yv: 0.0, forward=False,
                                    y_lim=(y_search[0], y_search[1]),
                                    tail=POLY_REACH_TAIL_N)
    walk = None
    if reach is not None and reach.get('cloud') is not None and gy.size >= 2:
        # ДАЛЬНИЙ КОНЕЦ — ТРЕКИНГ ПО ДАННЫМ (плотно → разреженно → модель):
        # продольное продолжение по дуге до ближайших редких точек рельса доводит ленту
        # лишь на 20-40 м, а рельс виден до 56-76 м и дальше; теперь лента едет за
        # полосой (см. _reach_walk) и дальше идёт моделью с пометкой узлов
        walk = _reach_walk(reach['cloud'], gy, xc, g, reach.get('top') or {},
                           reach.get('meas_end') or {}, float(y_search[0]),
                           kappa_1_m=k_far,
                           tail_slopes=((reach.get('tail') or {}).get('far')
                                        or None))
        if walk[0].size == 0:
            walk = None
    if walk is not None:
        ys_w, xs_w, codes_w, diag_w = walk
        # НАКЛОН ПРОДОЛЖЕНИЯ СРАЗУ ЗА СТЫКОМ против наклона хвоста (мм/м): именно
        # этот |Δ| ограничен POLY_REACH_TAIL_TOL_MM (критерий приёмки). Число отдаём
        # в meta (reach.far.junction_slope_off_mm), чтобы его не приходилось считать
        # заново по узлам с другим окном МНК.
        if diag_w is not None and ys_w.size >= 8:
            n_j = min(int(POLY_REACH_TAIL_N), int(ys_w.size))
            k_ext = float(np.polyfit(ys_w[:n_j], xs_w[:n_j], 1)[0])
            s_tail = diag_w.get('tail_slope_1_m')
            diag_w['junction_slope_off_mm'] = (None if s_tail is None
                                               else float((k_ext - s_tail) * 1000.0))
            diag_w['junction_slope_tol_mm'] = float(POLY_REACH_TAIL_TOL_MM)
        ys_f = np.zeros(0)
        ys = np.array(list(ys_w[::-1]) + list(gy) + list(ys_n), float)
        xc_all = np.array(list(xs_w[::-1]) + list(xc) + list(xs_n), float)
        codes_all = np.concatenate([codes_w[::-1], np.zeros(gy.size, np.int8),
                                    np.ones(len(ys_n), np.int8)])
    else:
        diag_w = None
        codes_all = np.concatenate([np.ones(len(ys_f), np.int8),
                                    np.zeros(gy.size, np.int8),
                                    np.ones(len(ys_n), np.int8)])
        ys = np.array(list(ys_f)[::-1] + list(gy) + list(ys_n), float)
        xc_all = np.array(list(xs_f)[::-1] + list(xc) + list(xs_n), float)
    xL = xc_all - 0.5 * g
    xR = xc_all + 0.5 * g
    reach_out = None
    ribbon_out = None
    drift_trim = None
    if reach is not None and reach.get('cloud') is not None and ys.size >= 4:
        ch = {'left': xL, 'right': xR}
        tops = reach.get('top') or {}
        ends = reach.get('meas_end') or {}
        per_side = {}
        for side, chain in ch.items():
            ms, cs = tops.get(side, (None, None))
            e_lo, e_hi = (ends.get(side) or (None, None))
            per_side[side] = {}
            for end, direction, y_meas, y_lim in (
                    ('far', -1.0, float(gy[0]), float(ys[0])),
                    ('near', 1.0, float(gy[-1]), float(ys[-1]))):
                own = e_lo if end == 'far' else e_hi
                if own is None:
                    own = y_meas
                # ШАГ НАРУЖУ ИДЁТ ОТ ИЗМЕРЕННОГО КОНЦА ЭТОЙ НИТИ: трекер мог
                # оборваться раньше данных (нить «теряется» на 5-10 м), и правило
                # должно это увидеть — подтверждение считается ОТ КОНЦА НИТИ, а не
                # от конца плана (конец плана — объединение диапазонов обеих нитей)
                y_end_side = float(own)
                if ms is None:
                    per_side[side][end] = {'confirmed_y_m': None, 'points_cut': None,
                                           'stop_reason': 'no_top_line', 'kept_extra_m': 0.0,
                                           'cut_m': 0.0, 'measured_end_own_y_m': float(y_end_side)}
                    continue
                r = _rail_reach(reach['cloud'], ys, chain, float(ms), float(cs),
                                y_end_side, direction, y_lim,
                                body_rast=(reach.get('body') or {}).get(side))
                r['measured_end_own_y_m'] = float(y_end_side)
                per_side[side][end] = r
        # ГОЛОВНОЙ стоп: лента идёт, пока данные подтверждает ХОТЯ БЫ ОДНА нить;
        # ЛЕНТОЧНЫЙ предохранитель: стоп по САМОЙ ранней нити, увидевшей тело.
        # Итог = min(двух стопов), одинаково для обеих нитей (пара жёсткая).
        # Расстояния считаются НАРУЖУ от ИЗМЕРЕННОГО конца плана (gy[0]/gy[-1]).
        stop = {}
        for end in ('far', 'near'):
            direction = -1.0 if end == 'far' else 1.0
            y_meas = float(gy[0]) if end == 'far' else float(gy[-1])
            head_d, rib_d, reasons = [], [], {}
            for side in ('left', 'right'):
                r = per_side[side][end]
                if r.get('confirmed_y_m') is None:
                    reasons[side] = r.get('stop_reason') or 'no_top_line'
                    continue
                # расстояние НАРУЖУ от конца плана (gy): нить могла подтвердить
                # данные ЗА своим измеренным концом, который лежит внутри плана
                d = direction * (float(r['confirmed_y_m']) - y_meas)
                d = max(0.0, float(d))
                if r.get('stop_reason') == 'ribbon_body_points':
                    rib_d.append(d)
                head_d.append(d)
                reasons[side] = r.get('stop_reason')
            d_head = max(head_d) if head_d else 0.0
            d_rib = min(rib_d) if rib_d else None
            d = d_head if d_rib is None else min(d_head, d_rib)
            stop[end] = {'dist_m': float(d), 'by_ribbon': bool(d_rib is not None
                                                              and d_rib <= d_head),
                         'reasons': reasons}
        # обрезка: держим узлы в подтверждённых данным пределах (и оставляем сами
        # ИЗМЕРЕННЫЕ узлы — они внутри gy и режутся только если данных нет вовсе).
        # ДАЛЬНИЙ конец, построенный трекингом (_reach_walk), НЕ режем: там своя
        # остановка (данные / плотная структура / предел модели)
        keep = np.ones(ys.size, bool)
        if walk is None:
            if stop['far']['dist_m'] > 0.0:
                keep &= ys >= (float(gy[0]) - stop['far']['dist_m'] - 1e-6)
            else:
                keep &= ys >= float(gy[0]) - 1e-6
        if stop['near']['dist_m'] > 0.0:
            keep &= ys <= (float(gy[-1]) + stop['near']['dist_m'] + 1e-6)
        else:
            keep &= ys <= float(gy[-1]) + 1e-6
        if int(keep.sum()) >= 4:
            ys, xL, xR = ys[keep], xL[keep], xR[keep]
            xc_all, codes_all = xc_all[keep], codes_all[keep]
        # ПРЕДЕЛ ПОПЕРЕЧНОГО УХОДА ПРОДОЛЖЕНИЯ ПО КАЖДОЙ НИТИ (спека 3.2, шаг 2):
        # у нитей разной длины измеренные участки, поэтому уход считается от
        # СОБСТВЕННОЙ касательной нити — МНК по её последним POLY_REACH_TAIL_N
        # измеренным узлам у её же стыка. Суммарно <= POLY_REACH_DRIFT_M и не больше
        # POLY_REACH_DRIFT_RATE на 1 м длины; первое превышение — обрезаем продолжение
        # (обе нити вместе: пара жёсткая). Счётчики «модель против точек» — в meta.
        trim = {}
        keep2 = np.ones(ys.size, bool)
        for side, chain in (('left', xL), ('right', xR)):
            e_lo, e_hi = (ends.get(side) or (None, None))
            for end in ('far', 'near'):
                y_j = e_lo if end == 'far' else e_hi
                if y_j is None:
                    continue
                y_j = float(y_j)
                # КАСАТЕЛЬНАЯ ХВОСТА — не одна, а ВЕЕР окон 10/12/15/16 узлов (спека
                # «МНК по 10-15 последним узлам»): уклон хвоста берётся по каждому
                # окну, и уход считается ХУДШИМ из них. Иначе замер по «другому»
                # окну (тот же критерий, другое число узлов) у длинного продолжения
                # даёт метры: уклон хвоста на 6 м гуляет на 5-10 мм/м, а на 100 м
                # это 0.5-1.0 м.
                ks = []
                for nw in (10, 12, 15, 16):
                    ins_w = np.abs(ys - y_j) <= nw * step + 1e-6
                    sel_w = np.flatnonzero(ins_w & ((ys >= y_j - 1e-6) if end == 'far'
                                                    else (ys <= y_j + 1e-6)))
                    if sel_w.size >= 4:
                        ks.append((float(np.polyfit(ys[sel_w], chain[sel_w], 1)[0]),
                                   int(sel_w[0] if end == 'far' else sel_w[-1])))
                if not ks:
                    continue
                i_ref = ks[0][1]
                y_ref, x_ref = float(ys[i_ref]), float(chain[i_ref])
                if end == 'far':
                    m = ys < y_ref - 1e-6
                    dist = y_ref - ys
                else:
                    m = ys > y_ref + 1e-6
                    dist = ys - y_ref
                if not m.any():
                    continue
                d_ = dist[m]
                # 0.9 — запас на ВЫБОР ОКНА МНК: уклон хвоста на 10-16 узлах гуляет,
                # а критерий допускает любой выбор в этом диапазоне; с запасом уход
                # остаётся <= POLY_REACH_DRIFT_M при любом из них
                cap_abs = 0.9 * POLY_REACH_DRIFT_M
                cap = np.minimum(cap_abs, POLY_REACH_DRIFT_RATE * d_)
                dev_k = [np.abs(chain[m] - (x_ref + k * (ys[m] - y_ref)))
                         for k, _i in ks]
                dev = np.maximum.reduce(dev_k)
                bad = dev > cap + 1e-6
                k0 = ks[len(ks) // 2][0]
                rec = {'tail_slope_1_m': float(k0),
                       'tail_slopes_1_m': [float(k) for k, _i in ks],
                       'dev_max_m': float(dev.max()),
                       'dev_cap_m': float(POLY_REACH_DRIFT_M),
                       'n_kept': int(m.sum() - bad.sum()), 'n_cut': int(bad.sum()),
                       'cut_at_len_m': (float(d_[bad].min()) if bad.any() else None)}
                trim.setdefault(side, {})[end] = rec
                if bad.any():
                    y_cut = float(ys[m][np.flatnonzero(bad)[int(np.argmin(d_[bad]))]])
                    keep2 &= (ys >= y_cut - 1e-6) if end == 'far' \
                        else (ys <= y_cut + 1e-6)
        if int(keep2.sum()) >= 4 and not keep2.all():
            ys, xL, xR = ys[keep2], xL[keep2], xR[keep2]
            xc_all, codes_all = xc_all[keep2], codes_all[keep2]
        drift_trim = trim
        reach_out = {}
        for end in ('far', 'near'):
            measured_end = float(gy[0]) if end == 'far' else float(gy[-1])
            conf = float(ys[0]) if end == 'far' else float(ys[-1])
            modeled = float(extra_far if end == 'far' else extra_near)
            kept = max(0.0, float(stop[end]['dist_m']))
            for side in ('left', 'right'):
                per_side[side][end].update({
                    'measured_end_y_m': measured_end,
                    'confirmed_end_y_m': conf,
                    'kept_extra_m': kept,
                    'cut_m': float(max(0.0, modeled - min(modeled, kept))),
                    'modeled_extra_m': modeled,
                    # итог ПАРЫ (одинаков для обеих нитей), чтобы он был виден и в
                    # per-rail записи meta.rails[side].reach
                    'pair_stop_reason': ('ribbon_body_points' if stop[end]['by_ribbon']
                                         else 'no_head_points'),
                    'pair_kept_extra_m': kept,
                    'pair_points_cut': int(sum(per_side[s2][end].get('points_cut') or 0
                                               for s2 in ('left', 'right'))),
                })
            reach_out[end] = {
                'confirmed_end_y_m': conf,
                'measured_end_y_m': measured_end,
                'kept_extra_m': kept,
                'modeled_extra_m': modeled,
                'cut_m': float(max(0.0, modeled - min(modeled, kept))),
                'points_cut': int(sum(per_side[s2][end].get('points_cut') or 0
                                      for s2 in ('left', 'right'))),
                'stop_reason': ('pair_ribbon_body_points' if stop[end]['by_ribbon']
                                else 'pair_no_head_points'),
                'by_side': {s2: {'confirmed_own_y_m': per_side[s2][end].get('confirmed_y_m'),
                                 'own_extra_m': float(max(0.0, (1.0 if end == 'far'
                                                                else -1.0)
                                                         * ((per_side[s2][end].get(
                                                             'confirmed_y_m') or measured_end)
                                                            - measured_end))),
                                 'stop_reason': per_side[s2][end].get('stop_reason'),
                                 'points_cut': per_side[s2][end].get('points_cut'),
                                 'ribbon_points': per_side[s2][end].get('ribbon_points'),
                                 'body_points': per_side[s2][end].get('body_points')}
                            for s2 in ('left', 'right')},
            }
        ribbon_out = {s2: {'far': per_side[s2]['far'].get('ribbon_points'),
                           'near': per_side[s2]['near'].get('ribbon_points'),
                           'body_far': per_side[s2]['far'].get('body_points'),
                           'body_near': per_side[s2]['near'].get('body_points')}
                      for s2 in ('left', 'right')}
        if walk is not None and diag_w is not None:
            # ДАЛЬНИЙ КОНЕЦ ВЕДЁТ ТРЕКИНГ ПО ДАННЫМ (плотно/разреженно/модель):
            # в reach.far пишем ИМЕННО его концы, а строгие per-side числа остаются
            # в by_side как справка
            e = reach_out.setdefault('far', {})
            e['data_end_y_m'] = diag_w.get('data_end_y_m')
            e['model_end_y_m'] = diag_w.get('model_end_y_m')
            e['model_len_m'] = diag_w.get('model_len_m')
            e['sparse_len_m'] = diag_w.get('sparse_len_m')
            e['stop_reason'] = diag_w.get('stop_reason')
            e['block_points'] = diag_w.get('block_points')
            e['confirmed_end_y_m'] = (diag_w.get('data_end_y_m')
                                      if diag_w.get('data_end_y_m') is not None
                                      else e.get('confirmed_end_y_m'))
            e['kept_extra_m'] = float(abs(float(e['confirmed_end_y_m'])
                                          - float(gy[0]))) \
                if e.get('confirmed_end_y_m') is not None else 0.0
            e['ribbon_end_y_m'] = float(ys[0])
            e['ribbon_len_m'] = float(abs(float(ys[0]) - float(ys[-1])))
            # НАКЛОН ПРОДОЛЖЕНИЯ ПРОТИВ НАКЛОНА ХВОСТА у стыка (мм/м) — число
            # критерия; tail_slope_1_m — уклон, взятый продолжением (середина
            # хвостов обеих нитей), junction_slope_off_mm — фактическое |Δ|
            e['tail_slope_1_m'] = diag_w.get('tail_slope_1_m')
            e['tail_slope_center_1_m'] = diag_w.get('tail_slope_center_1_m')
            e['tail_slope_side_1_m'] = diag_w.get('tail_slope_side_1_m')
            e['junction_slope_off_mm'] = diag_w.get('junction_slope_off_mm')
            e['junction_slope_tol_mm'] = diag_w.get('junction_slope_tol_mm')
            e['dev_data_max_m'] = diag_w.get('dev_data_max_m')
            e['dev_model_max_m'] = diag_w.get('dev_model_max_m')
            e['n_data_steps'] = diag_w.get('n_data_steps')
            e['n_model_steps'] = diag_w.get('n_model_steps')
            for side in ('left', 'right'):
                per_side[side]['far'].update({
                    'data_end_y_m': e.get('data_end_y_m'),
                    'model_end_y_m': e.get('model_end_y_m'),
                    'model_len_m': e.get('model_len_m'),
                    'sparse_len_m': e.get('sparse_len_m'),
                    'walk_stop_reason': e.get('stop_reason'),
                    'block_points': e.get('block_points'),
                    'above_points': e.get('above_points'),
                    'wall_points': e.get('wall_points'),
                    'model_drift_m': e.get('model_drift_m'),
                    'ribbon_end_y_m': e.get('ribbon_end_y_m'),
                    'ribbon_len_m': e.get('ribbon_len_m'),
                    # КРИТЕРИЙ: наклон продолжения против наклона хвоста у стыка
                    'tail_slope_1_m': e.get('tail_slope_1_m'),
                    'tail_slope_center_1_m': e.get('tail_slope_center_1_m'),
                    'tail_slope_side_1_m': e.get('tail_slope_side_1_m'),
                    'junction_slope_off_mm': e.get('junction_slope_off_mm'),
                    'junction_slope_tol_mm': e.get('junction_slope_tol_mm'),
                    # «модель против точек»: максимальный уход и объём ступеней
                    'dev_data_max_m': e.get('dev_data_max_m'),
                    'dev_model_max_m': e.get('dev_model_max_m'),
                    'n_data_steps': e.get('n_data_steps'),
                    'n_model_steps': e.get('n_model_steps'),
                })
    out = {
        'ys': ys, 'xL': xL, 'xR': xR, 'gauge_m': float(g),
        'gauge_spread_all_mm': float(np.ptp((xR - xL) * 1000.0)),
        'extra_far_m': extra_far, 'extra_near_m': extra_near,
        'cap_h_m': cap_h, 'cap_x_far_m': cap_x_far, 'cap_x_near_m': cap_x_near,
        'kappa_far_1_m': k_far, 'kappa_near_1_m': k_near,
        'dense_len_m': dense_len_c,
        'center_bins': np.arange(gy.size), 'center_grid_y': gy,
        'center_src': np.zeros(gy.size, np.int8),
        'gauge_raw_m': None, 'gauge_clamped_mm': None,
        'gauge_spread_measured_mm': None,
        'n_bins_both': 0, 'n_bins_one': 0, 'n_center_both': 0, 'n_center_reliable': 0,
        'n_center_single': 0, 'n_reliable_left': 0, 'n_center_runs': 0,
        'gauge_off_threshold_mm': float(1000.0 * POLY_PAIR_OFF_M),
        'center_kind': 'both_rails', 'center_source': 'центр: общая линия пары',
        'reach': reach_out, 'reach_per_side': per_side if reach_out is not None else None,
        'ribbon_inside_points': ribbon_out,
        # ПОПЕРЕЧНЫЙ УХОД ПРОДОЛЖЕНИЯ ПО НИТЯМ (см. _extension_drift_trim в _pair_finish):
        # по каждому концу каждой нити — своя касательная хвоста, максимум ухода,
        # сколько узлов срезано и с какой длины
        'drift_trim': drift_trim,
        # ПРОИСХОЖДЕНИЕ УЗЛОВ: 0 = measured (измеренный участок), 1 = sparse (данные за
        # измеренным концом), 2 = model (данных нет, продолжение по модели)
        'node_origin_codes': codes_all,
        'walk_diag': diag_w,
    }
    out.update(extra)
    return out


def _reach_body_raster(sec, cell=0.005, shrink=0.005):
    """РАСТР ВНУТРЕННОСТИ ТЕЛА рельса вокруг центра верха головки (du, dv), м.

    Тело рельса обязано быть ПУСТЫМ для лидара: точки внутри металла невозможны. Рамка
    «коробкой» этого не даёт — в неё попадают и собственные боковины головки, и пол
    рядом с рельсом (замерено: 4-12 точек в измеренном участке). Поэтому берём контур
    сечения (ГОСТ + измеренная коронка), СЖИМАЕМ его на shrink и считаем, какие
    ячейки сетки строго внутри. Возвращает dict для векторного теста точек."""
    try:
        uv = _body_section_uv(sec)
    except Exception:
        return None
    cos_t, sin_t = math.cos(PODUKLONKA_RAD), math.sin(PODUKLONKA_RAD)
    poly = []
    for (u_mm, v_mm) in uv:
        du0 = u_mm / 1000.0
        dv0 = (v_mm - R65_H_MM) / 1000.0
        poly.append((du0 * cos_t - dv0 * sin_t, du0 * sin_t + dv0 * cos_t))
    poly = np.asarray(poly, float)
    c = poly.mean(axis=0)
    d = poly - c
    dmax = float(np.abs(d).max()) or 1.0
    poly_s = c + d * (1.0 - min(0.45, shrink / dmax))
    lo = poly_s.min(axis=0)
    hi = poly_s.max(axis=0)
    nu = int(np.ceil((hi[0] - lo[0]) / cell)) + 1
    nv = int(np.ceil((hi[1] - lo[1]) / cell)) + 1
    us = lo[0] + (np.arange(nu) + 0.5) * cell
    vs = lo[1] + (np.arange(nv) + 0.5) * cell
    U, V = np.meshgrid(us, vs, indexing='ij')
    inside = np.zeros(U.shape, bool)
    x1, y1 = poly_s[:, 0], poly_s[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    for k in range(x1.size):
        dy = y2[k] - y1[k]
        if abs(dy) < 1e-12:
            continue
        cond = (y1[k] > V) != (y2[k] > V)
        xint = (x2[k] - x1[k]) * (V - y1[k]) / dy + x1[k]
        inside ^= cond & (U < xint)
    return {'u0': float(lo[0]), 'v0': float(lo[1]), 'cell': float(cell),
            'mask': inside}


def _own_ends(r):
    """ИЗМЕРЕННЫЙ диапазон нити (y_lo, y_hi) с запасным путём через meas.

    У нити, у которой bsec не построился (нет своих бинов), ключи y_lo/y_hi
    заполняются ПОЗЖЕ — уже после плана пары; тогда границы берём из её измерений
    головки (meas). Нужно, чтобы правило остановки и предел поперечного ухода
    считались ОТ СВОЕГО конца нити, а не от конца сетки пары.
    """
    m = r.get('meas') or {}
    lo = r.get('y_lo')
    hi = r.get('y_hi')
    return (m.get('y_lo') if lo is None else lo,
            m.get('y_hi') if hi is None else hi)


def _rail_tail_slopes(r, n=POLY_REACH_TAIL_N):
    """Наклон ХВОСТА измеренного участка ОДНОЙ нити на каждом конце: {'far','near'}.

    МНК по последним n бинам, где у ЭТОЙ нити есть свои точки (bsec['bins']), с одной
    отбраковкой по MAD; dx/dy, м/м. Зашитые интерполяцией узлы (бины без точек) наклон
    не определяют, поэтому берутся только измеренные бины. Нужен, потому что у нитей
    РАЗНЫЕ измеренные концы: наклон сглаженной центральной линии на конце сетки — это
    наклон только той нити, которая измерена дальше, и продолжение, начатое им, даёт
    у второй нити у стыка чужой наклон (замерено 20-30 мм/м).
    """
    out = {'far': None, 'near': None}
    b = (r or {}).get('bsec')
    if not b:
        return out
    gy = np.asarray(b.get('grid_y', []), float)
    gx = np.asarray(b.get('grid_x', []), float)
    meas = np.asarray(b.get('bins', []), np.int64)
    if gy.size < 4 or meas.size < 4:
        return out
    bins = np.asarray(b.get('grid_bins', []), np.int64)
    if bins.size == gy.size:
        pos = np.searchsorted(bins, np.sort(meas))
        pos = pos[(pos >= 0) & (pos < gy.size)]
    else:
        pos = np.arange(gy.size)
    if pos.size < 4:
        return out
    for end in ('far', 'near'):
        sel = pos[:n] if end == 'far' else pos[-n:]
        y, x = gy[sel], gx[sel]
        if y.size < 4 or float(np.ptp(y)) < 1e-6:
            continue
        c1, c0 = np.polyfit(y, x, 1)
        s = float(c1)
        res = x - (s * y + float(c0))
        r0 = float(np.median(res))
        mad = float(np.median(np.abs(res - r0)))
        keep = np.abs(res - r0) <= max(3.0 * mad, 0.010)
        if int(keep.sum()) >= 4 and not keep.all():
            s = float(np.polyfit(y[keep], x[keep], 1)[0])
        out[end] = s
    return out


def _reach_body_hits(rast, du, dv):
    """точки ВНУТРИ тела рельса по растру (du — поперёк от оси, dv — ниже верха)"""
    if rast is None:
        return None
    i = np.floor((du - rast['u0']) / rast['cell']).astype(np.int64)
    j = np.floor((dv - rast['v0']) / rast['cell']).astype(np.int64)
    ok = (i >= 0) & (i < rast['mask'].shape[0]) & (j >= 0) & (j < rast['mask'].shape[1])
    out = np.zeros(du.shape, bool)
    out[ok] = rast['mask'][i[ok], j[ok]]
    return out


def _reach_walk(cloud, gy, xc, g, tops, ends, y_lim, kappa_1_m=None,
                model_max_m=POLY_MODEL_MAX_M, step=POLY_BIN_M, tail_slopes=None):
    """ПРОДОЛЖЕНИЕ ЛЕНТЫ ОТ ИЗМЕРЕННОГО ДАЛЬНЕГО КОНЦА: три ступени (жалоба «лента в
    3-6 раз короче видимого рельса»).

      * ступень 1 (ПЛОТНАЯ, как раньше): в окне 5 м вперёд >=3 точки головки,
        |Δx| <= 0.15 м от оси нити, Δz ∈ [-45,+20] мм от своей линии верха;
      * ступень 2 (РАЗРЕЖЕННАЯ): >=2 точки, |Δx| <= 0.35 м, тот же Δz, и точки окна —
        КОМПАКТНАЯ группа (разброс по y <= 4 м, по поперечине <= 0.30 м), иначе это
        разрозненные выбросы; полоса отслеживается: центр ленты доезжает за данными
        (поправка за шаг <= 0.20 м), наклон оси подстраивается по данным;
      * ступень 3 (МОДЕЛЬ): данных нет — продолжаем по накопленному наклону и
        кривизне, но не дальше POLY_MODEL_MAX_M и до первой плотной структуры в
        объёме ленты (>= POLY_MODEL_BLOCK_MIN точек в рамке ±0.25 м / -0.30..+0.10 м).

    gy/xc — центр на измеренной сетке бинов, g — колея, tops[side] = (m_top, c_top),
    ends[side] = (y_lo, y_hi) — ИЗМЕРЕННЫЕ концы нити (от них идёт проверка).
    Возвращает (ys_ext, xs_ext, origin_codes, diag): ys_ext В ПОРЯДКЕ НАРУЖУ (y убывает),
    codes: 1 = sparse (данные за измеренным концом), 2 = model (данных нет).
    """
    diag = {'stop_reason': 'no_tangent', 'data_end_y_m': None, 'model_end_y_m': None,
            'model_len_m': 0.0, 'sparse_len_m': 0.0, 'cut_head_points': 0,
            'block_points': 0, 'steps': 0}
    if cloud is None or gy.size < 2:
        return np.zeros(0), np.zeros(0), np.zeros(0, np.int8), diag
    tg = _polyline_tangent(gy[::-1], xc[::-1])
    if tg is None:
        return np.zeros(0), np.zeros(0), np.zeros(0, np.int8), diag
    # НАКЛОН ПРОДОЛЖЕНИЯ = НАКЛОН ИЗМЕРЕННОГО ХВОСТА: МНК по последним
    # POLY_REACH_TAIL_N узлам (6 м) с одной отбраковкой по MAD. Продолжение обязано
    # начаться ТЕМ ЖЕ знаком и величиной, что хвост у стыка — это же и есть
    # «касательная конца данных». Кривизну НЕ продолжаем: идём прямой, а поперёк
    # доезжает поправка от точек (см. ниже); прежняя производная квадратичной
    # подгонки на 2.5 м давала наклон 0.3 и уводила модель боком.
    n_fit = min(int(POLY_REACH_TAIL_N), int(gy.size))
    y_fit = gy[:n_fit][::-1]
    x_fit = xc[:n_fit][::-1]
    if y_fit.size >= 4:
        s = float(np.polyfit(y_fit, x_fit, 1)[0])
        res = x_fit - (s * y_fit + float(np.polyfit(y_fit, x_fit, 1)[1]))
        r0 = float(np.median(res))
        mad = float(np.median(np.abs(res - r0)))
        keep = np.abs(res - r0) <= max(3.0 * mad, 0.010)
        if int(keep.sum()) >= 4 and not keep.all():
            s = float(np.polyfit(y_fit[keep], x_fit[keep], 1)[0])
    else:
        s = float(tg['dir_x']) / max(abs(float(tg['dir_y'])), 0.2) \
            * (1.0 if tg['dir_y'] > 0 else -1.0)
    diag['tail_slope_center_1_m'] = float(s)
    diag['tail_nodes'] = int(y_fit.size)
    # НАКЛОН ХВОСТА ПО НИТЯМ: у нитей РАЗНЫЕ измеренные концы, и наклон центра на
    # конце сетки — это наклон лишь той нити, что измерена дальше (вторая получает
    # «чужой» наклон у своего стыка — замерено 20-30 мм/м). Берём середину наклонов
    # хвостов обеих нитей: она минимизирует худшее отклонение на стыке каждой.
    tl = [v for v in (tail_slopes or {}).values() if v is not None]
    if tl:
        s = float(sum(tl) / float(len(tl)))
        diag['tail_slope_side_1_m'] = {k: (None if v is None else float(v))
                                       for k, v in (tail_slopes or {}).items()}
    diag['tail_slope_1_m'] = float(s)
    yc, xp, zp = (np.asarray(a, float) for a in cloud)
    # предварительный отбор по поперечине относительно ИСХОДНОЙ цепочки и по Δz своей
    # линии верха; дальше на каждом шаге уже узкая проверка от ТЕКУЩЕГО центра
    per_side = {}
    for side, (tt_s, tt_i) in (tops or {}).items():
        if tt_s is None:
            continue
        dz = zp - (float(tt_s) * yc + float(tt_i))
        ax0 = np.interp(yc, np.concatenate([gy[::-1], gy[1:]]),
                        np.concatenate([xc[::-1], xc[1:]]))
        m = (np.abs(xp - ax0) <= POLY_REACH_PRE_DX_M + 0.5 * g) \
            & (dz >= POLY_REACH_DZ_M[0]) & (dz <= POLY_REACH_DZ_M[1])
        per_side[side] = (yc[m], xp[m], dz[m])
    if not per_side:
        return np.zeros(0), np.zeros(0), np.zeros(0, np.int8), diag
    # ОТДЕЛЬНЫЙ ОТБОР ДЛЯ ПРЕДОХРАНИТЕЛЕЙ МОДЕЛИ: он нужен ВНЕ полосы головки
    # (свод сверху, стена сбоку), поэтому Δz здесь не ограничен
    guard = {}
    for side, (tt_s, tt_i) in (tops or {}).items():
        if tt_s is None:
            continue
        dz_all = zp - (float(tt_s) * yc + float(tt_i))
        ax0 = np.interp(yc, np.concatenate([gy[::-1], gy[1:]]),
                        np.concatenate([xc[::-1], xc[1:]]))
        mg = np.abs(xp - ax0) <= POLY_REACH_PRE_DX_M + 0.5 * g
        guard[side] = (yc[mg], xp[mg], dz_all[mg])
    y0 = float(gy[0])
    x0 = float(xc[0])
    # ПРЕДЕЛ ПО ДАЛЬНОСТИ для трекинга: НЕ окно детекции (y_search = ±45 м), а край
    # САМОГО ОБЛАКА — разреженные точки рельса видны до 56-76 м, стенки до 100-200 м,
    # а лента должна доходить настолько далеко, насколько видно (ограничитель —
    # POLY_MODEL_MAX_M и плотная структура)
    y_lim = min(float(y_lim), float(yc[0]))
    off = 0.0                     # поправка центра (медленно меняется)
    # КАСАТЕЛЬНАЯ В СТЫКЕ: прямая x = x0 + s·(y − y0). `x` — положение НА ЭТОЙ ПРЯМОЙ
    # (без поправки), а в мир узел идёт как x + off. Прежде поправка прибавлялась к
    # самому x на КАЖДОМ шаге (x = x_new + off), то есть работала как добавка к наклону:
    # off = −0.05 давал −100 мм/м к наклону продолжения, и лента у стыка шла не по
    # хвосту (замерено: наклон продолжения −8…+30 мм/м при хвосте +78).
    x_data = y_data = s_data = None    # «прямая в конце данных» (== касательная стыка)
    # СТВОР МОДЕЛИ: продолжение не уезжает боком из коридора пути. АБСОЛЮТНЫЙ
    # |x| <= 3 м здесь неверен: тоннель ПОВОРАЧИВАЕТ, и ось нити законно уходит от
    # оси кадра (у roundT на f109 наклон хвоста 84 мм/м против s оси 23 мм/м — на
    # 40 м это 2.4 м). Поэтому створ считается ОТ ТОЧКИ СТЫКА: |x − x0| <=
    # POLY_REACH_DRIFT_M (0.6 м ухода, см. ниже) + POLY_MODEL_WANDER_M (3 м запаса
    # на модель). Это и есть предохранитель «лента ушла не туда», а не привязка к
    # оси кадра; полный предел поперечного ухода даёт drift_trim по нитям.
    x_lim_rel = POLY_REACH_DRIFT_M + POLY_MODEL_WANDER_M
    x_lim = max(POLY_REACH_X_LIMIT_M, abs(float(xc[0])) + 0.5)
    diag['x_limit_m'] = float(x_lim)
    diag['x_limit_rel_m'] = float(x_lim_rel)
    ys_out, xs_out, codes = [], [], []
    y = y0
    x = x0
    n_steps = 0
    model_run = 0.0
    last_data_y = y0
    # СЧЁТЧИКИ «МОДЕЛЬ ПРОТИВ ТОЧЕК» и поперечного ухода от касательной стыка
    diag.update({'dev_data_max_m': 0.0, 'dev_model_max_m': 0.0, 'dev_rate_max': 0.0,
                 'n_data_steps': 0, 'n_model_steps': 0, 'off_end_m': 0.0})
    while True:
        n_steps += 1
        y_new = y - step
        if y_new < y_lim:
            diag['stop_reason'] = 'cloud_limit'
            break
        pen = 0.0
        s_new = s
        # ЗНАК ШАГА: наружу ось идёт в сторону УМЕНЬШЕНИЯ y (dy = y_new - y = -step),
        # поэтому приращение x = s·dy. Прежний «+» (x + s*step) разворачивал
        # продолжение в ПРОТИВОПОЛОЖНУЮ сторону от измеренного хвоста: при s = +0.08
        # (наклон хвоста) модель уезжала на −0.08·L вместо +0.08·L, то есть на
        # метры вбок в другую сторону от поворота пути.
        x_new = x + s_new * (y_new - y)
        x_fin = x_new + off
        # Поперечный уход от КАСАТЕЛЬНОЙ СТЫКА (прямая x0 + s·(y − y0)) — это ровно
        # |off| (линия x идёт по касательной), плюс отдельно считаем уход от створа |x|.
        d_len = abs(float(y_new - y0))
        dev = abs(float(x_fin) - (x0 + s * (float(y_new) - y0)))
        cap = min(POLY_REACH_DRIFT_M, POLY_REACH_DRIFT_RATE * d_len)
        conf = {}          # сторона -> (n_dense, n_sparse, x_med, cluster_ok)
        d0, d1 = y_new - POLY_REACH_WIN_M, y_new
        for side, (ys_s, xs_s, dz_s) in per_side.items():
            a = int(np.searchsorted(ys_s, d0, 'left'))
            b = int(np.searchsorted(ys_s, d1, 'right'))
            if b <= a:
                conf[side] = (0, 0, None, False)
                continue
            yy = ys_s[a:b]
            xx = xs_s[a:b]
            rail_x = x_new + off + (0.5 * g if side == 'left' else -0.5 * g)
            du = np.abs(xx - rail_x)
            n_d = int(np.count_nonzero(du <= POLY_REACH_DX_M))
            sel = du <= POLY_REACH_SPARSE_DX_M
            n_s = int(np.count_nonzero(sel))
            ok_cluster = False
            x_med = None
            if n_s:
                xs_sel, ys_sel = xx[sel], yy[sel]
                # КЛАСТЕРНОСТЬ: это компактная группа, а не разрозненные выбросы
                for i in range(xs_sel.size):
                    near = (np.abs(ys_sel - ys_sel[i]) <= POLY_REACH_CLUSTER_DY_M) \
                        & (np.abs(xs_sel - xs_sel[i]) <= POLY_REACH_CLUSTER_DX_M)
                    if int(np.count_nonzero(near)) >= POLY_REACH_SPARSE_MIN:
                        ok_cluster = True
                        x_med = float(np.median(xs_sel[near]))
                        break
                if not ok_cluster:
                    # ОДИНОЧНЫЙ возврат: принимаем, если он лёг В СТВОР продолжения
                    # (близко к предсказанному положению оси этой нити) — так лента
                    # едет по редким одиночным точкам, не хватая выбросы сбоку
                    d_one = np.abs(xs_sel - (x_new + off + (0.5 * g if side == 'left'
                                                            else -0.5 * g)))
                    i_min = int(np.argmin(d_one))
                    if d_one[i_min] <= POLY_REACH_ONE_DX_M:
                        ok_cluster = True
                        x_med = float(xs_sel[i_min])
            elif n_d:
                ok_cluster = True
                x_med = float(np.median(xx[du <= POLY_REACH_DX_M]))
            conf[side] = (n_d, n_s, x_med, ok_cluster)
        # данные подтверждают продолжение?
        data_ok = False
        corr = []
        for side, (n_d, n_s, x_med, ok) in conf.items():
            if ok and x_med is not None:
                data_ok = True
                sig = 0.5 * g if side == 'left' else -0.5 * g
                corr.append(x_med - sig - (x_new + off))
        block = 0
        above = 0
        vert = 0
        for side, (ys_s, xs_s, dz_s) in per_side.items():
            a = int(np.searchsorted(ys_s, d0, 'left'))
            b = int(np.searchsorted(ys_s, d1, 'right'))
            if b <= a:
                continue
            rail_x = x_new + off + (0.5 * g if side == 'left' else -0.5 * g)
            near = np.abs(xs_s[a:b] - rail_x) <= POLY_MODEL_BLOCK_DU_M
            block += int(np.count_nonzero(near))
        # ПРЕДОХРАНИТЕЛИ МОДЕЛИ: (а) структура НАД лентой (свод/платформа/стена
        # впереди), (б) ВЕРТИКАЛЬНАЯ структура в полосе (стена: пол горизонтален, у
        # него разброс по высоте ~0.05 м). Плотность у самой ленты не считаем — под
        # рельсом всегда пол, и он же попадает в «объём тела» (замерено ранее)
        for side, (ys_g, xs_g, dz_g) in (guard or {}).items():
            a = int(np.searchsorted(ys_g, d0, 'left'))
            b = int(np.searchsorted(ys_g, d1, 'right'))
            if b <= a:
                continue
            rail_x = x_new + off + (0.5 * g if side == 'left' else -0.5 * g)
            du_g = np.abs(xs_g[a:b] - rail_x)
            dv_g = dz_g[a:b]
            above += int(np.count_nonzero(
                (du_g <= POLY_MODEL_ABOVE_DU_M)
                & (dv_g >= POLY_MODEL_ABOVE_DZ_M[0]) & (dv_g <= POLY_MODEL_ABOVE_DZ_M[1])))
            sel_v = (du_g <= POLY_MODEL_VERT_DU_M) & (dv_g >= POLY_MODEL_VERT_DZ_M[0]) \
                & (dv_g <= POLY_MODEL_VERT_DZ_M[1])
            n_v = int(np.count_nonzero(sel_v))
            if n_v >= POLY_MODEL_VERT_MIN and float(np.ptp(dv_g[sel_v])) >= POLY_MODEL_VERT_SPREAD_M:
                vert = max(vert, n_v)
        diag['block_points'] = max(diag['block_points'], block)
        diag['above_points'] = max(diag.get('above_points', 0), above)
        diag['wall_points'] = max(diag.get('wall_points', 0), vert)
        if data_ok:
            if corr:
                c = float(np.median(corr))
                # ПОПРАВКА РАЗМАЗЫВАЕТСЯ ПО ДЛИНЕ (P-регулятор): резкий сдвиг на
                # 0.2 м за шаг давал излом наклона 0.2-0.4 1/м и «скачки оси» —
                # замерено, лента выглядела сломанной
                d_off = POLY_REACH_CORR_GAIN * (c - off)
                # СКОРОСТЬ ПОПРАВКИ У СТЫКА ограничена: на первых
                # POLY_REACH_TAIL_RAMP_M метрах она растёт не быстрее
                # POLY_REACH_TAIL_TOL_MM на 1 м, чтобы НАКЛОН продолжения сразу за
                # стыком остался наклоном хвоста (критерий), а не наклоном «доезда за
                # точками». Дальше — прежний темп (0.20 м за шаг).
                ramp = max(0.0, (abs(float(y_new - y0)) - POLY_REACH_TAIL_RAMP_M)
                           / POLY_REACH_TAIL_RAMP_M)
                lim = max(POLY_REACH_TAIL_TOL_MM / 1000.0 * step,
                          POLY_REACH_CORR_M * ramp)
                d_off = float(np.clip(d_off, -lim, lim))
                off_prev = off
                off = float(np.clip(off + d_off, -POLY_REACH_OFF_MAX_M,
                                    POLY_REACH_OFF_MAX_M))
                # наклон НЕ подстраиваем от поправки: это обратная связь (поправка →
                # наклон → поправка) и она разгоняла ось (замерено: шаги 1.2-3.0 м);
                # наклон идёт только от кривизны плана, а поправка — смещением
                s = s_new
            # x — положение на КАСАТЕЛЬНОЙ стыка (без поправки), в узел идёт x + off:
            # так поправка остаётся ПОПЕРЕЧНЫМ СМЕЩЕНИЕМ, а не добавкой к наклону
            x = x_new
            codes.append(1)
            diag['data_end_y_m'] = float(y_new)
            model_run = 0.0
            # запоминаем «прямую в конце данных»: по ней меряем поперечный уход модели
            x_data, y_data, s_data = float(x), float(y_new), float(s_new)
        else:
            # СЧЁТЧИКИ СТРУКТУРЫ — диагностика (в meta): замерено, что «плотность у
            # самой ленты» и «вертикальный разброс» ловят ПОЛ под рельсом (рельс лежит
            # на полу), поэтому как стоп они не годятся; стоп даёт счётчик в объёме
            # ленты и уход модели
            if above >= POLY_MODEL_ABOVE_MIN or vert >= POLY_MODEL_VERT_MIN:
                diag['stop_reason'] = ('structure_above' if above >= POLY_MODEL_ABOVE_MIN
                                       else 'structure_wall')
                break
            if block >= POLY_MODEL_BLOCK_MIN:
                diag['stop_reason'] = 'dense_structure'
                break
            # СТВОР: модельная лента не уходит из коридора пути. Предел АБСОЛЮТНЫЙ по
            # |x| (POLY_REACH_X_LIMIT_M = 3 м, с запасом |x на конце данных| + 0.5 м):
            # это требование внешнего теста (validation/synth + tests/test_validation:
            # «ось пути не может уезжать из тоннеля шириной 16 м», |x| < 3 м) и
            # одновременно предохранитель «лента ушла не туда». Дополнительно створ
            # ограничен от СТЫКА (x_lim_rel) — чтобы модель не уходила далеко вбок
            # даже там, где |x| кадра ещё позволяет.
            if abs(float(x_fin)) > x_lim or abs(float(x_fin) - x0) > x_lim_rel:
                diag['stop_reason'] = 'corridor_exit'
                diag['corridor_x_m'] = float(x_fin)
                diag['corridor_rel_m'] = float(abs(float(x_fin) - x0))
                break
            # ПОПЕРЕЧНЫЙ УХОД МОДЕЛИ от КАСАТЕЛЬНОЙ стыка: суммарно <= POLY_REACH_DRIFT_M
            # и не больше POLY_REACH_DRIFT_RATE на 1 м длины (плюс прежний «кривизненный»
            # POLY_MODEL_WANDER_M как верхний предохранитель). Превышение — обрезаем.
            if dev > min(cap, POLY_MODEL_WANDER_M):
                diag['stop_reason'] = 'model_wander'
                diag['model_drift_m'] = float(dev)
                diag['model_drift_cap_m'] = float(min(cap, POLY_MODEL_WANDER_M))
                break
            model_run += step
            if model_run > model_max_m:
                diag['stop_reason'] = 'model_limit'
                break
            x = x_new
            codes.append(2)
            diag['model_end_y_m'] = float(y_new)
        # СЧЁТЧИКИ УХОДА (поправка шага уже итоговая): «модель против точек» — это
        # dev_data_max_m/dev_model_max_m и n_data_steps/n_model_steps в meta
        dev_fin = abs(float(x_new + off) - (x0 + s * (float(y_new) - y0)))
        diag['dev_rate_max'] = max(diag['dev_rate_max'], dev_fin / max(d_len, 1e-6))
        diag['off_end_m'] = float(off)
        if data_ok:
            diag['n_data_steps'] += 1
            diag['dev_data_max_m'] = max(diag['dev_data_max_m'], dev_fin)
        else:
            diag['n_model_steps'] += 1
            diag['dev_model_max_m'] = max(diag['dev_model_max_m'], dev_fin)
        ys_out.append(y_new)
        xs_out.append(float(x_new + off))
        y = y_new
        s = s_new
    diag['steps'] = n_steps
    if diag['data_end_y_m'] is not None:
        diag['sparse_len_m'] = float(abs(diag['data_end_y_m'] - y0))
    if diag['model_end_y_m'] is not None:
        diag['model_len_m'] = float(abs(diag['model_end_y_m'] - y0)) - diag['sparse_len_m']
        diag['model_len_m'] = max(0.0, diag['model_len_m'])
        if diag['stop_reason'] == 'no_tangent':
            diag['stop_reason'] = 'model_end'
    if diag['stop_reason'] == 'no_tangent':
        diag['stop_reason'] = 'no_extension'
    return (np.asarray(ys_out, float), np.asarray(xs_out, float),
            np.asarray(codes, np.int8), diag)


def _rail_reach(cloud, chain_y, chain_x, top_s, top_i, y_end, direction, y_lim,
                body_rast=None):
    """ПРАВИЛО ОСТАНОВКИ ПРОДОЛЖЕНИЯ ПО ДАННЫМ (спека 3.2, шаг 1) для ОДНОЙ нити.

    Шаг наружу 0.5 м от ИЗМЕРЕННОГО конца плана; на каждом шаге окно 5 м ВПЕРЁД:
      * точек ГОЛОВКИ нужно >= POLY_REACH_MIN_PTS: |Δx| <= 0.15 м от оси нити (цепочка
        узлов ленты) и Δz ∈ [-45, +20] мм от СВОЕЙ линии верха нити (m_top·y + c_top,
        уклон измерен по ядру и продолжен тем же уклоном);
      * первый предохранитель: первая неудача → накат POLY_REACH_ROLL_M и стоп;
      * второй предохранитель: >= POLY_REACH_RIB_MIN точек облака ВНУТРИ объёма тела
        ленты (тело рельса обязано быть пустым) → стоп сразу.
    cloud — облако, отсортированное по y: (y, x, z). direction = -1 (дальний конец,
    y убывает) или +1 (ближний). y_lim — за него продолжения нет.
    Возвращает {confirmed_y_m, stop_reason, points_cut, ribbon_points, steps}.
    """
    out = {'confirmed_y_m': None, 'stop_reason': None, 'points_cut': None,
           'ribbon_points': 0, 'body_points': 0, 'steps': 0}
    if cloud is None or chain_y.size < 2:
        return out
    yc, xc_, zc = (np.asarray(a, float) for a in cloud)
    if yc.size == 0:
        return out
    topline = top_s * yc + top_i
    dz = zc - topline
    ax = np.interp(yc, chain_y, chain_x)
    du = np.abs(xc_ - ax)
    head_m = (du <= POLY_REACH_DX_M) & (dz >= POLY_REACH_DZ_M[0]) & (dz <= POLY_REACH_DZ_M[1])
    body_m = _reach_body_hits(body_rast, du, dz)
    if body_m is None:
        # растёр не передан — грубая рамка «объём головки» (как раньше)
        body_m = (du <= POLY_REACH_RIB_DU_M) & (dz >= POLY_REACH_RIB_DZ_M[0]) \
            & (dz <= POLY_REACH_RIB_DZ_M[1])
    wide_m = (du <= POLY_REACH_BODY_DU_M) & (dz >= POLY_REACH_BODY_DZ_M[0]) \
        & (dz <= POLY_REACH_BODY_DZ_M[1])
    hy, by, wy = yc[head_m], yc[body_m], yc[wide_m]
    step = POLY_REACH_STEP_M
    k = 0
    y0 = float(y_end)
    while True:
        k += 1
        y0 = float(y_end) + direction * step * k
        if (direction < 0 and y0 < y_lim) or (direction > 0 and y0 > y_lim):
            out['stop_reason'] = 'cloud_limit'
            break
        if direction < 0:
            wlo, whi = y0 - POLY_REACH_WIN_M, y0
        else:
            wlo, whi = y0, y0 + POLY_REACH_WIN_M
        n_head = int(np.count_nonzero((hy >= wlo) & (hy <= whi)))
        n_body = int(np.count_nonzero((by >= wlo) & (by <= whi)))
        n_wide = int(np.count_nonzero((wy >= wlo) & (wy <= whi)))
        out['ribbon_points'] = max(out['ribbon_points'], n_body)
        out['body_points'] = max(out.get('body_points', 0), n_wide)
        if n_body >= POLY_REACH_RIB_MIN:
            out['stop_reason'] = 'ribbon_body_points'
            break
        if n_head < POLY_REACH_MIN_PTS:
            out['stop_reason'] = 'no_head_points'
            break
    out['steps'] = k
    conf = y0 + direction * POLY_REACH_ROLL_M
    # подтверждённый конец не может выйти ЗА построенную цепочку узлов
    if direction < 0:
        conf = max(conf, float(chain_y[0]))
    else:
        conf = min(conf, float(chain_y[-1]))
    out['confirmed_y_m'] = float(conf)
    # отрезанные точки головки: за подтверждённым концом, но в пределах окна
    if direction < 0:
        m2 = (hy < out['confirmed_y_m']) & (hy >= out['confirmed_y_m'] - POLY_REACH_WIN_M)
    else:
        m2 = (hy > out['confirmed_y_m']) & (hy <= out['confirmed_y_m'] + POLY_REACH_WIN_M)
    out['points_cut'] = int(np.count_nonzero(m2))
    return out


def _pair_nominal_gauge():
    """Колея, когда измерить её нечем (нет второй нити / нет общей полосы): номинал по
    осям (1594.6 мм = колея по внутренним граням + ширина головки), зажатый в тот же
    коридор 1580..1610 мм, что и измеренная колея."""
    return float(np.clip(GAUGE_AXIS_NOMINAL_MM / 1000.0,
                         POLY_PAIR_GAUGE_LO_M, POLY_PAIR_GAUGE_HI_M))


def _pair_plan_one_rail(r_meas, shelf_line, sweep_extra_m, y_search, other=None, cloud=None):
    """ПЛАН, КОГДА КОЛЕЮ ИЗМЕРИТЬ НЕЧЕМ: на кадре нет второй нити (или у нитей нет
    общей полосы). Единый путь _pair_plan с gauge_override = номинал по осям,
    зажатый в 1580..1610 мм: оси измеренных нитей ОСТАЮТСЯ на своих местах (узел
    измеренной нити = её же измеренный узел), ненайденная нить ставится на G от неё.
    Ключ nodes будет у ОБЕИХ нитей — по этому плану (см. _pair_rail_nodes)."""
    if other is None:
        other = {'side': 'right' if r_meas.get('side') == 'left' else 'left', 'bsec': None}
    return _pair_plan(r_meas, other, shelf_line, sweep_extra_m, y_search,
                      gauge_override=_pair_nominal_gauge(), cloud=cloud)


def _pair_plan_nominal(s_axis, i_axis, y0, y1, shelf_line, sweep_extra_m, y_search,
                       gauge_m=None):
    """ПЛАН ПО НОМИНАЛЬНОЙ ОСИ КАДРА: нити не найдены ВООБЩЕ (нет ни одной линии/точек
    головки). Центр = номинальная ось кадра (s_axis/i_axis), колея = номинал. Это НЕ
    измерение: в meta помечается pair_center_kind = nominal_axis и счётчиком узлов."""
    g = float(GAUGE_AXIS_NOMINAL_MM / 1000.0) if gauge_m is None else float(gauge_m)
    g = float(np.clip(g, POLY_PAIR_GAUGE_LO_M, POLY_PAIR_GAUGE_HI_M))
    lo, hi = float(min(y0, y1)), float(max(y0, y1))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < POLY_BIN_M:
        return None
    # сетка бинов выровнена так же, как в _pair_plan (центры бинов = b0+(n+0.5)*step)
    n0 = int(np.ceil(lo / POLY_BIN_M - 0.5))
    n1 = int(np.floor(hi / POLY_BIN_M - 0.5))
    gy = (np.arange(n0, n1 + 1) + 0.5) * POLY_BIN_M
    if gy.size < 2:
        return None
    xc = float(s_axis) * gy + float(i_axis)
    return _pair_finish(gy, xc, g, shelf_line, sweep_extra_m, y_search, {
        'center_bins': np.arange(n0, n1 + 1), 'center_grid_y': gy,
        'center_src': np.full(gy.size, 6, np.int8),
        'center_kind': 'nominal_axis',
        'gauge_source': 'nominal',
        'n_bins_one': int(gy.size),
        'center_source': ('центр: НОМИНАЛЬНАЯ ОСЬ кадра (нити на кадре не найдены, '
                          's=%+.5f i=%+.5f) + колея %.1f мм = номинал; геометрия '
                          'номинальная, доверие понижать — узлы не из измерения'
                          % (float(s_axis), float(i_axis), 1000.0 * g)),
    })


def _pair_rail_nodes(pair, side, ys_z=None):
    """УЗЛЫ ОСИ НИТИ ИЗ ПЛАНА ПАРЫ: [y, x_оси, z_верх] — для нитей, у которых нет
    своего bsec (нить на кадре не измерена), чтобы ключ nodes был ВСЕГДА. x — из
    плана (центр ∓ G/2), z — измеренный верх ПАРНОЙ нити, а если и её нет —
    номинальная линия z(y) = m_top*y + c_top."""
    ys = np.asarray(pair['ys'], float)
    xs = np.asarray(pair['xL'] if side == 'left' else pair['xR'], float)
    if ys.size == 0:
        return []
    if ys_z is not None and len(ys_z) == 2 and ys_z[0] is not None:
        zy, zz = np.asarray(ys_z[0], float), np.asarray(ys_z[1], float)
        zs = np.interp(ys, zy, zz)
    elif ys_z is not None and ys_z[1] is not None:
        zs = np.asarray([float(ys_z[1](v)) for v in ys], float)
    else:
        zs = np.full(ys.size, np.nan)
    return [[round(float(ys[i]), 2), round(float(xs[i]), 3), round(float(zs[i]), 3)]
            for i in range(ys.size)]


def _top_line_off(ys, zs, r, pair):
    """Максимальный уход ВЫСОТЫ узлов продолжения от СВОЕЙ линии верха нити, м."""
    ys = np.asarray(ys, float)
    zs = np.asarray(zs, float)
    lo, hi = r.get('y_lo'), r.get('y_hi')
    if lo is None or hi is None:
        return 0.0
    ext = (ys < float(lo) - 1e-6) | (ys > float(hi) + 1e-6)
    if not np.any(ext):
        return 0.0
    return float(np.max(np.abs(zs[ext] - (r['m_top'] * ys[ext] + r['c_top']))))


def _node_origin_for(ys, r, pair):
    """ПОМЕТКА ПРОИСХОЖДЕНИЯ КАЖДОГО УЗЛА (жалоба: лента обрывается на 20-40 м, а рельс
    виден дальше — продолжение разделено на «данные» и «модель»):

      * measured — узел внутри СВОЕГО измеренного диапазона нити;
      * sparse   — узел за ним, подтверждённый данными (плотными или разреженными);
      * model    — узел продолжения по модели (данных в окне нет).
    """
    ys = np.asarray(ys, float)
    lo, hi = r.get('y_lo'), r.get('y_hi')
    codes = None if pair is None else pair.get('node_origin_codes')
    out = []
    for i, y in enumerate(ys):
        if lo is not None and hi is not None and (float(lo) - 1e-6) <= y <= (float(hi) + 1e-6):
            out.append('measured')
        elif codes is not None and int(np.asarray(codes)[i]) == 2:
            out.append('model')
        else:
            out.append('sparse')
    return out


def _rail_build_polyline(x, y, z, r, bsec, sec, shelf_line, sweep_extra_m, y_search,
                         axis_ab=None, pair=None):
    """Меш нити по ПОЛИЛИНИИ: продолжение по касательной, метрика оси, режим сравнения.

    Возвращает dict: verts, faces, sweep (как у прямой развёртки: extra/cap/drift),
    metric (расстояние оси до точек рельса по корзинам), extent_y, beyond_measured,
    y_dense, dense_len_m, n_bridging, axis_max_step_m.
    """
    out_sign = r['out_sign']
    gy = np.asarray(bsec['grid_y'], float)
    gx = np.asarray(bsec['grid_x'], float)
    gz = np.asarray(bsec['grid_z'], float)
    meas_top = bsec['z_nodes'].max(axis=1)
    # НИЗ КОРИДОРА ВЕРХА = «высота тела − запас» (спека 2.2/3.2): POLY_TOP_LO_M = 0.08
    # меньше высоты тела 0.1835 м, поэтому узел, поджатый к 0.08, топит подошву на
    # ~103 мм под полку. При 0.1835 − 0.05 = 0.1335 подошва не глубже 50 мм.
    _bloc, dv_min_body, _dv_max_body = _body_section_loc(sec)
    h_body = -float(dv_min_body)
    top_lo = max(POLY_TOP_LO_M, h_body - POLY_TOP_MARGIN_M)
    if shelf_line is not None:
        sh = shelf_line
        u0 = float(r.get('u_rail') or 0.0)
        du = sh.get('slope_u', 0.0) * u0      # поперечный наклон полки под нитью
        shelf_bins = (sh['z_at_y_ref'] + du
                      + sh['slope_y'] * (bsec['y_c'] - sh['y_ref']))
        h_head = float(np.median(meas_top - shelf_bins))
        # коридор «верх головки над полкой»: бины уже гейтятся в _rail_track, это
        # страховка на вырожденный набор + явный счёт в meta
        h_head_clamped = abs(float(np.clip(h_head, top_lo, POLY_TOP_HI_M))
                             - h_head) > 1e-9
        h_head = float(np.clip(h_head, top_lo, POLY_TOP_HI_M))

        def z_at(yv, sh=sh, hh=h_head, du=du):
            return sh['z_at_y_ref'] + du + sh['slope_y'] * (yv - sh['y_ref']) + hh

        def shelf_at(yv, sh=sh, du=du):
            return sh['z_at_y_ref'] + du + sh['slope_y'] * (yv - sh['y_ref'])

        slope_y = float(sh['slope_y'])
    else:
        # полку измерить не удалось (мало точек пола): полоса метрики — от верха меша
        h_head = None
        h_head_clamped = False
        base = float(np.median(meas_top - 0.20))

        def z_at(yv, r=r):
            return r['m_top'] * yv + r['c_top']

        def shelf_at(yv, b=base):
            return b

        slope_y = None
    # длина измеренного участка по ПУТИ (не по Y): бюджет ухода по высоте считается
    # от неё, а не от размаха по Y — на кривой это разные вещи
    dense_len = float(np.hypot(np.diff(gy), np.diff(gx)).sum()) + POLY_BIN_M
    # Касательные концов нужны на выходе В ОБОИХ режимах (в meta и в отчётах), а
    # заполняются они либо внутри ветки пары, либо ниже по своей полилинии. Без
    # этой инициализации кадр, где пара не построилась, падал на tg_far ниже.
    tg_far = tg_near = None
    if pair is not None:
        # ЖЁСТКАЯ ПАРА: узлы заданы общей центральной линией  G/2 (см. _pair_plan),
        # а по Z нить по-прежнему идёт по СВОЕЙ измеренной коронке: своя высота
        # (подуклонка, возвышение) — измеренная, её не выравниваем
        ys = np.asarray(pair['ys'], float)
        xs = np.asarray(pair['xL'] if r['side'] == 'left' else pair['xR'], float)
        extra_far = pair['extra_far_m']
        extra_near = pair['extra_near_m']
        cap_h = pair['cap_h_m']
        cap_x_far = pair['cap_x_far_m']
        cap_x_near = pair['cap_x_near_m']
        k_far = pair['kappa_far_1_m']
        k_near = pair['kappa_near_1_m']
        ys_f = np.zeros(0)
        ys_n = np.zeros(0)
        tg_far = tg_near = None
        z_own = np.interp(ys, gy, gz)
        in_own = (ys >= r['y_lo'] - 1e-6) & (ys <= r['y_hi'] + 1e-6)
        d_edge = np.minimum(np.abs(ys - r['y_lo']), np.abs(ys - r['y_hi']))
        w = np.clip(d_edge / POLY_Z_BLEND_M, 0.0, 1.0)
        # ВЫСОТА ПРОДОЛЖЕНИЯ — ПО СВОЕЙ ЖЕ ЛИНИИ ВЕРХА НИТИ (m_top·y + c_top, уклон
        # измерен по ядру и продолжен): это данные, а не «полка + постоянная высота»
        # (жалоба: продолжение уходило от материала на 0.16-0.45 м по высоте)
        z_target = r['m_top'] * ys + r['c_top']
        zs = np.where(in_own, z_own, z_own * (1.0 - w) + z_target * w)
        gy = ys
        gx = xs
        gz = zs
    tg_far_ref = tg_far
    tg_near_ref = tg_near
    tg_far = _polyline_tangent(gy[::-1], gx[::-1]) if pair is None else tg_far_ref
    tg_near = _polyline_tangent(gy, gx)
    k_far = None if tg_far is None else tg_far['kappa_1_m']
    k_near = None if tg_near is None else tg_near['kappa_1_m']
    if pair is None:
        extra_far, cap_h, cap_x_far = _rail_sweep_extent_poly(
            dense_len, slope_y, k_far, sweep_extra_m)
        extra_near, _cap_h2, cap_x_near = _rail_sweep_extent_poly(
            dense_len, slope_y, k_near, sweep_extra_m)
    if sweep_extra_m is None and pair is None:
        # режим сравнения: тянем по касательной, пока узел не выйдет за диапазон
        # облака (сколько нужно длины по пути, считаем через |dir_y| касательной)
        extra_far = 0.0 if tg_far is None else min(
            200.0, abs(float(y_search[0]) - float(gy[0])) / max(abs(tg_far['dir_y']), 0.2))
        extra_near = 0.0 if tg_near is None else min(
            200.0, abs(float(y_search[1]) - float(gy[-1])) / max(abs(tg_near['dir_y']), 0.2))
        cap_h = cap_x_far = cap_x_near = None
    if pair is None:
        ys_f, xs_f, zs_f = _extend_end(gy, gx, extra_far, z_at, forward=True,
                                       y_lim=(y_search[0], y_search[1]),
                                       z_start=float(gz[0]))
        ys_n, xs_n, zs_n = _extend_end(gy, gx, extra_near, z_at, forward=False,
                                       y_lim=(y_search[0], y_search[1]),
                                       z_start=float(gz[-1]))
        ys = np.array(list(ys_f)[::-1] + list(gy) + list(ys_n), float)
        xs = np.array(list(xs_f)[::-1] + list(gx) + list(xs_n), float)
        zs = np.array(list(zs_f)[::-1] + list(gz) + list(zs_n), float)
    # Коридор верха над полкой — для ВСЕХ узлов, включая продолжение и стык
    # (на стыке сшивка Z от измеренного верха к полке+высота могла уйти под низ)
    if shelf_line is not None:
        # полка ПОД САМИМ УЗЛОМ: поперечный наклон полки по u узла (не по
        # замороженному u середины ядра) — иначе на кривой узел выходит из
        # коридора на 1-5 мм, и его приходится поджимать
        if axis_ab is not None:
            u_nodes = xs - (float(axis_ab[0]) * ys + float(axis_ab[1]))
        else:
            u_nodes = np.full(ys.size, float(r.get('u_rail') or 0.0))
        u_nodes_all = u_nodes
        sh_all = np.array([shelf_at(float(v)) for v in ys], float)             + shelf_line.get('slope_u', 0.0) * (u_nodes - float(r.get('u_rail') or 0.0))
        # коридор задан для ВЕРХА МЕША (= zs + dv_max), а не для zs: у сечения тела
        # верх бывает на 1-3 мм ниже узла (коронка измерена по центру)
        _loc, _dv_min, dv_max = _body_section_loc(sec)
        # +0.5 мм запас: у потребителя полка берётся под центром головки, а не под
        # узлом (разница slope_u*u_off ~0.1 мм), и на границе коридора узел иначе
        # считается выпавшим из-за округления. Низ считается ОТ ПОДОШВЫ: низ подошвы
        # сечения на dv_min НИЖЕ верхнего узла (с учётом подуклонки), поэтому
        # zs >= полка − запас − dv_min даёт подошву не глубже POLY_TOP_MARGIN_M под
        # локальной полкой (спека 3.2: ноль узлов глубже 50 мм)
        lo_all = sh_all - POLY_TOP_MARGIN_M - float(_dv_min) + 0.0005
        hi_all = sh_all + POLY_TOP_HI_M - dv_max
        n_clamp_nodes = int(np.sum((zs < lo_all) | (zs > hi_all)))
        zs = np.clip(zs, lo_all, hi_all)
    else:
        n_clamp_nodes = 0
    # ТЕЛО рельса (ГОСТ + измеренная коронка) и отдельно ИЗМЕРЕННАЯ полоска
    verts, faces, n_poly, dv_foot = _rail_mesh_body(ys, xs, zs, sec, out_sign)
    mverts, mfaces = _rail_mesh_path(ys, xs, zs, sec, out_sign)
    nx, ny = _path_normals(ys, xs, out_sign)
    bins_of_nodes = np.round((ys - bsec['b0']) / POLY_BIN_M - 0.5).astype(np.int64)
    grid_meas = np.isin(bins_of_nodes, bsec['bins'])
    inside = (ys >= bsec['y_dense'][0] - 1e-6) & (ys <= bsec['y_dense'][1] + 1e-6)
    meas_mask = grid_meas & inside
    bucket = np.where(~inside, 'outside', np.where(meas_mask, 'measured', 'bridged'))
    shelf_nodes = np.array([shelf_at(v) for v in ys], float)
    if shelf_line is not None:
        # ПОЛКА ПОД САМИМ УЗЛОМ: тот же отсчёт, что в клампе коридора (иначе
        # «подошва под полкой» считалась бы по полке под замороженным u середины
        # ядра и расходилась с клампом на кривой на 10-20 мм)
        shelf_nodes = shelf_nodes + shelf_line.get('slope_u', 0.0) * \
            (u_nodes_all - float(r.get('u_rail') or 0.0))
    metric = _axis_to_rail_metric(x, y, z, xs, ys, nx, ny, shelf_nodes, bucket)
    dense_mid = 0.5 * (float(bsec['y_dense'][0]) + float(bsec['y_dense'][1]))
    far_y = float(ys[0])
    near_y = float(ys[-1])
    z_slope = r['m_top'] if slope_y is None else slope_y
    z_at_mid = float(z_at(dense_mid))
    drift_far = abs(float(z_at(far_y)) - z_at_mid)
    drift_near = abs(float(z_at(near_y)) - z_at_mid)
    # поперечный уход продолжения — от реального изогнутого пути: kappa*L^2/2
    x_drift_far = 0.0 if k_far is None else 0.5 * k_far * extra_far ** 2
    x_drift_near = 0.0 if k_near is None else 0.5 * k_near * extra_near ** 2
    node_stats = _node_outlier_stats(ys, xs, zs)
    # НИЗ ПОДОШВЫ относительно локальной полки: низ сечения на dv_foot ниже верха
    # (это НЕ ровно 180 мм: подуклонка наклоняет сечение, низ уходит на ~183.5 мм)
    foot_off = -float(dv_foot)
    foot_below = zs + float(dv_foot) - shelf_nodes
    steps = np.abs(np.diff(gx))
    if tg_far is None and tg_near is None:
        tangent_deg = None
    else:
        tangent_deg = [
            (None if tg_far is None else
             float(np.degrees(np.arctan2(tg_far['dir_x'], -tg_far['dir_y'])))),
            (None if tg_near is None else
             float(np.degrees(np.arctan2(-tg_near['dir_x'], tg_near['dir_y'])))),
        ]
    sweep = {
        'model': 'polyline', 'extra_m': extra_far, 'extra_near_used_m': extra_near,
        'cap_h_m': cap_h, 'cap_x_m': cap_x_far, 'cap_x_near_m': cap_x_near,
        'head_h_m': h_head, 'shelf_ref': shelf_line,
        'y_dense_mid': dense_mid, 'dense_len_m': dense_len,
        'extra_far_m': float(bsec['y_dense'][0] - far_y),
        'extra_near_m': float(near_y - bsec['y_dense'][1]),
        'z_drift_far_m': drift_far, 'z_drift_near_m': drift_near,
        'x_drift_far_m': float(x_drift_far), 'x_drift_near_m': float(x_drift_near),
        'kappa_far_1_m': k_far, 'kappa_near_1_m': k_near,
        'z_slope_used': float(z_slope),
        'y_lo': float(min(ys)), 'y_hi': float(max(ys)),
    }
    return {
        # УЗЛЫ ОСИ: [y, x_axis, z_top] по порядку вдоль пути (включая продолжение);
        # z_top — верх, по которому ставится сечение (измеренная коронка внутри
        # данных, полка+высота головки в продолжении)
        # округление: 1 мм по поперечине/высоте и 1 см по пути — для маски точек
        # этого хватает, а JSON кадра растёт на ~4 кБ вместо ~11
        'axis_nodes': [[round(float(ys[i]), 2), round(float(xs[i]), 3),
                        round(float(zs[i]), 3)] for i in range(ys.size)],
        'verts': verts, 'faces': faces, 'poly_section_nodes': int(n_poly),
        'meas_verts': mverts, 'meas_faces': mfaces,
        'node_stats': node_stats,
        'foot_drop_m': float(foot_off),
        'section_height_m': float(-dv_foot + sec['vz_nodes_m'][2]),
        'foot_below_shelf_min_m': float(foot_below.min()),
        'foot_below_shelf_max_m': float(foot_below.max()),
        'foot_out_of_tol': int(np.sum((foot_below < POLY_FOOT_TOL_M[0])
                                      | (foot_below > POLY_FOOT_TOL_M[1]))),
        'sweep': sweep, 'metric': metric,
        'extent_y': [float(min(ys)), float(max(ys))],
        'dense_len_m': dense_len,
        'n_nodes_total': int(ys.size),
        'n_bridging': int(np.sum(inside & ~grid_meas)),
        'n_clamped_nodes': int(n_clamp_nodes),
        'axis_max_step_m': float(steps.max()) if steps.size else 0.0,
        'n_jumps': int(np.sum(steps > POLY_JUMP_M)),
        'smooth_resid_m': bsec.get('smooth_resid_m'),
        'n_rejected': bsec.get('n_rejected'),
        'n_rejected_height': bsec.get('n_rejected_height'),
        'n_clamped_height': bsec.get('n_clamped_height'),
        'h_head_clamped': h_head_clamped,
        # высота продолжения идёт по СВОЕЙ линии верха нити: мера ухода (в meta)
        'top_line_off_max_m': _top_line_off(ys, zs, r, pair),
        'top_corridor_m': [POLY_TOP_LO_M, POLY_TOP_HI_M],
        # низ КЛАМПА верха (динамический, = высота тела − запас) и высота тела
        'corridor_top_lo_m': float(top_lo),
        'body_height_m': float(h_body),
        # ПРАВИЛО ОСТАНОВКИ ПО ДАННЫМ: результат по ЭТОЙ нити (см. _rail_reach)
        'node_origin': _node_origin_for(ys, r, pair),
        'reach': (None if pair is None else
                  (pair.get('reach_per_side') or {}).get(r['side'])),
        # ПРОИСХОЖДЕНИЕ УЗЛОВ (шаг «продолжать по разреженным данным»): measured —
        # узел внутри своего измеренного диапазона, sparse — данные (плотные или
        # разреженные) за ним, model — продолжение по модели (данных нет)
        'node_origin': _node_origin_for(ys, r, pair),
        'reach_len_m': {k: float(POLY_BIN_M * sum(
            1 for v in _node_origin_for(ys, r, pair) if v == k))
            for k in ('measured', 'sparse', 'model')},
        'ribbon_inside_points': (None if pair is None
                                 else (pair.get('ribbon_inside_points') or {}).get(r['side'])),
        'tangent_deg': tangent_deg,
    }


def _rail_sweep_extent(bsec, r, shelf_line, sweep_extra_m, y_min, y_max):
    """Диапазон развёртки меша нити: измеренные плотные бины + запас.

    Запас sweep_extra_m (по умолчанию RAIL_SWEEP_EXTRA_M = 15 м — данных о рельсе
    дальше плотных бинов нет) ограничивается сверху допустимым уходом ЗА ПРЕДЕЛЫ
    измеренного: меш целиком должен лежать не дальше RAIL_DRIFT_LIMIT_M по высоте
    (RAIL_X_DRIFT_LIMIT_M по X) от уровня/оси в СЕРЕДИНЕ измеренного диапазона.
    Бюджет ухода тратится сначала на сам измеренный участок — его собственный
    уклон |slope|*(полдиапазона) обрезать нельзя (там всё измерено), поэтому на
    запас остаётся RAIL_DRIFT_LIMIT_M/|slope| - полдиапазона. Наклон полки
    берётся из shelf_line (shelf_slope_y), наклон оси — из s нити.

    sweep_extra_m=None — режим сравнения: весь диапазон облака в пределах
    RAIL_SWEEP_Y_FAR_M..RAIL_SWEEP_Y_NEAR_M, без ограничения ухода.

    Возвращает (y_lo, y_hi, extra_used_m, cap_h_m, cap_x_m); cap_* = None, если
    соответствующий наклон нулевой и ограничения нет.
    """
    if sweep_extra_m is None:
        return (max(y_min, RAIL_SWEEP_Y_FAR_M), min(y_max, RAIL_SWEEP_Y_NEAR_M),
                None, None, None)
    y_lo, y_hi = float(bsec['y_dense'][0]), float(bsec['y_dense'][1])
    half_dense = 0.5 * (y_hi - y_lo)
    cap_h = cap_x = None
    if shelf_line is not None and abs(shelf_line['slope_y']) > 1e-9:
        cap_h = max(0.0, RAIL_DRIFT_LIMIT_M / abs(shelf_line['slope_y']) - half_dense)
    if abs(r['s']) > 1e-9:
        cap_x = max(0.0, RAIL_X_DRIFT_LIMIT_M / abs(r['s']) - half_dense)
    cands = [float(sweep_extra_m)]
    if cap_h is not None:
        cands.append(cap_h)
    if cap_x is not None:
        cands.append(cap_x)
    extra = float(min(cands))
    return (max(y_min, y_lo - extra), min(y_max, y_hi + extra), extra, cap_h, cap_x)


def _sweep_source_text(r):
    """Формулировка source.mesh: где измерено, насколько развёрнуто, чем привязано.

    Никаких утверждений про «весь диапазон/200 м»: развёртка кончается на
    измеренном диапазоне + запас, дальше меш НЕ продолжается, а верх привязан к
    измеренной полке, а не к плоскости floor. Для полилинии отдельно сказано, что
    ось повторяет изгиб пути (перпендикулярное сечение, продолжение по касательной).
    """
    sw = r.get('sweep') or {}
    dens = (r.get('mstat') or {}).get('y_dense') or [0.0, 0.0]
    ext = r.get('extent_y') or [sw.get('y_lo', 0.0), sw.get('y_hi', 0.0)]
    model = ('polyline axis (local sections tracked along the curve, section placed '
             'perpendicular to the local tangent)' if sw.get('model') == 'polyline'
             else 'straight axis x = s*y + i')
    head = ('measured section (median over dense bins %.1f..%.1f m) swept %.1f..%.1f m '
            'on %s' % (dens[0], dens[1], ext[0], ext[1], model))
    if sw.get('extra_m') is None:
        return (head + ': РЕЖИМ СРАВНЕНИЯ — весь диапазон облака без ограничения '
                'ухода, НЕ значение по умолчанию; далее меш не продолжается. Верх — '
                'измеренная полка (meta.floor_profile), плоскость floor не используется')
    return (head + ' = измеренный участок + %.1f м (запас RAIL_SWEEP_EXTRA_M = %.1f м, '
            'ограничен уходом <= %.2f м по высоте / %.2f м поперёк пути от уровня в '
            'СЕРЕДИНЕ измеренного участка); далее меш НЕ продолжается. Верх привязан к '
            'измеренной полке (meta.floor_profile); плоскость floor для привязки не '
            'используется'
            % (sw['extra_m'], RAIL_SWEEP_EXTRA_M, RAIL_DRIFT_LIMIT_M, RAIL_X_DRIFT_LIMIT_M))


def _rail_mesh_sweep(sec, s, i, z_at, out_sign, y_lo, y_hi,
                     step=RAIL_SWEEP_STEP_M):
    """НЕПРЕРЫВНАЯ развёртка измеренного сечения вдоль прямой оси.

    Рельс — протяжённый объект постоянного сечения: одно медианное сечение
    (sec) ставится в каждый узел сетки по Y на ось x = s*y + i, а высота верха
    берётся из z_at(y) — привязки к ИЗМЕРЕННОЙ полке пола (см. build_track:
    z_at(y) = shelf(y) + медианная высота головки над полкой), а не из
    продолжения прямой линии верха. Разрывов нет: между соседними узлами два
    треугольника. За пределами плотных бинов ось — продолжение прямой, сечение
    то же: это предсказание по измеренному сечению, а не дорисовка стандарта.
    """
    ys = np.arange(y_lo, y_hi + 1e-9, step)
    if ys.size < 2:
        ys = np.array([y_lo, min(y_hi, y_lo + step)])
    half_w = sec['half_width_m']
    u_off = sec['u_off_m']
    vz = sec['vz_nodes_m']
    n_nodes = RAIL_NODES
    verts = []
    for yv in ys:
        xc = s * yv + i
        top = float(z_at(yv))
        for k in range(n_nodes):
            u = u_off - half_w + 2.0 * half_w * k / (n_nodes - 1)
            verts.append([float(xc + out_sign * u), float(yv), float(top + vz[k])])
    faces = []
    for j in range(ys.size - 1):
        a0 = j * n_nodes
        a1 = (j + 1) * n_nodes
        for k in range(n_nodes - 1):
            faces.append([a0 + k, a0 + k + 1, a1 + k + 1])
            faces.append([a0 + k, a1 + k + 1, a1 + k])
    return verts, faces


def _rail_mesh(verts_mm, s, i, m_top, c_top, y_lo, y_hi, out_sign, step=0.5):
    """СТАРАЯ схема: экструзия контура Р65 (ГОСТ) вдоль оси x = s*y + i.

    По умолчанию НЕ используется (draw_gost_profile=False): меш строится только
    по наблюдаемым точкам (см. _rail_mesh_measured). Оставлена для сравнения.
    """
    ys = np.arange(y_lo, y_hi + 1e-9, step)
    if ys.size < 2:
        ys = np.array([y_lo, y_lo + step])
    n_p = len(verts_mm)
    n_y = ys.size
    cos_t = math.cos(PODUKLONKA_RAD)
    sin_t = math.sin(PODUKLONKA_RAD)
    verts = np.empty((n_y * n_p, 3))
    for k, yv in enumerate(ys):
        x_rail = s * yv + i
        z_base = (m_top * yv + c_top) - R65_H_MM / 1000.0
        for p, (u_mm, v_mm) in enumerate(verts_mm):
            u = u_mm / 1000.0
            v = v_mm / 1000.0
            verts[k * n_p + p, 0] = x_rail + out_sign * (u * cos_t - v * sin_t)
            verts[k * n_p + p, 1] = yv
            verts[k * n_p + p, 2] = z_base + (u * sin_t + v * cos_t)

    faces = []
    for k in range(n_y - 1):
        for p in range(n_p):
            a = k * n_p + p
            b = k * n_p + (p + 1) % n_p
            c = (k + 1) * n_p + (p + 1) % n_p
            d = (k + 1) * n_p + p
            faces.append([a, b, c])
            faces.append([a, c, d])
    caps = _triangulate_polygon(verts_mm)
    last = (n_y - 1) * n_p
    for (t0, t1, t2) in caps:
        faces.append([t2, t1, t0])
        faces.append([last + t0, last + t1, last + t2])
    return verts, faces


def _kmeans2(v, iters=10):
    """Два уровня (полка / дно лотка) простым k-средних по 1D."""
    lo, hi = float(np.min(v)), float(np.max(v))
    for _ in range(iters):
        low = np.abs(v - lo) <= np.abs(v - hi)
        if low.any():
            lo = float(np.mean(v[low]))
        if (~low).any():
            hi = float(np.mean(v[~low]))
    return (lo, hi) if lo < hi else (hi, lo)


def _floor_profile(x, y, z, s_ax, i_ax, m_top, c_top, y_far, notes):
    """Поперечный профиль пола в системе координат пары рельсов (лоток).

    Локальная система профиля пола: u — поперёк пути, от оси пары рельсов
    (положительно к правой нити, то есть к +X при s=0); z — абсолютная высота.
    Начало отсчёта по u — ось пары (середина между нитями), по z — сама
    измеренная полка (см. 'shelf0').

    Берутся только точки пола: ближняя зона y в [max(y_min, -12), -2] (дальше
    пол разрежен: замерено 750 против 2700 точек на полку в 8.5 м), |u| <= 0.9
    и по Z гейт «ниже верха головки на 0.10...0.80 м» (иначе в окно лезут
    потолок и стены). Профиль — медиана Z по бинам 20 мм, затем два уровня
    k-средних: полка (верхний) и дно лотка (нижний).

    Возвращает dict или None. Все ключи с суффиксом _m — метры.
    """
    y_lo = max(float(y.min()), -FLOOR_NEAR_M)
    y_hi = min(y_far, -2.0)
    if y_hi - y_lo < 3.0:
        return None
    u = x - (s_ax * y + i_ax)
    top = m_top * y + c_top
    m = ((y >= y_lo) & (y <= y_hi) & (np.abs(u) <= FLOOR_U_MEAS_M + 0.6)
         & (z < top - FLOOR_Z_UP_M) & (z > top - FLOOR_Z_DOWN_M))
    if int(m.sum()) < 2000:
        return None
    uu, zz, yy = u[m], z[m], y[m]

    edges = np.arange(-FLOOR_U_MEAS_M, FLOOR_U_MEAS_M + FLOOR_BIN_M, FLOOR_BIN_M)
    n_b = edges.size - 1
    uc = edges[:-1] + FLOOR_BIN_M / 2.0
    med = np.full(n_b, np.nan)
    cnt = np.zeros(n_b, dtype=np.int64)
    for k in range(n_b):
        sel = (uu >= edges[k]) & (uu < edges[k + 1])
        cnt[k] = int(sel.sum())
        if cnt[k] >= FLOOR_MIN_PTS:
            med[k] = float(np.median(zz[sel]))
    ok = np.isfinite(med) & (cnt >= FLOOR_MIN_PTS)
    if int(ok.sum()) < 8:
        return None

    lo_lvl, hi_lvl = _kmeans2(med[ok])
    thr = hi_lvl - FLOOR_SHELF_DROP_M
    k0 = int(np.flatnonzero(ok)[np.argmin(np.abs(uc[np.flatnonzero(ok)]))])
    has_trough = bool(np.isfinite(med[k0]) and med[k0] < thr)

    u_l = u_r = float('nan')
    bottom = hi_lvl
    n_bottom = 0
    wall_run = float('nan')
    if has_trough:
        kl = k0
        while kl - 1 >= 0 and np.isfinite(med[kl - 1]) and med[kl - 1] < thr:
            kl -= 1
        kr = k0
        while kr + 1 < n_b and np.isfinite(med[kr + 1]) and med[kr + 1] < thr:
            kr += 1
        u_l = float(uc[kl] - FLOOR_BIN_M / 2.0)
        u_r = float(uc[kr] + FLOOR_BIN_M / 2.0)
        inb = (uu >= u_l) & (uu <= u_r)
        n_bottom = int(inb.sum())
        bottom = float(np.percentile(zz[inb], 10.0)) if n_bottom >= 10 else float(med[k0])
        # стенка: бины, чья медиана лежит между полкой и дном
        mid_lvl = (hi_lvl - 0.05, max(bottom, lo_lvl) + 0.05)
        wall = [k for k in range(max(0, kl - 3), min(n_b, kr + 4))
                if np.isfinite(med[k]) and mid_lvl[1] < med[k] < mid_lvl[0]]
        wall_run = FLOOR_BIN_M * max(1, len(wall))

    # «полка» = всё, что вне лотка. Внимание: ~True даёт целое -2, и такой массив
    # уходит в fancy-индексацию — строим честную булеву маску
    def _outside(vals):
        if has_trough:
            return (vals < u_l - 0.04) | (vals > u_r + 0.04)
        return np.ones(vals.shape, dtype=bool)

    # Полка — прямая по бинам РЯДОМ с лотком (0.06+|край| .. 0.62 м): дальше
    # к рельсам уровень поднимается на плечо (замерено +0.11 м на u=0.8), и
    # подгонка по всему диапазону завышает полку, а с ней и глубину лотка.
    edge = max(abs(u_l), abs(u_r)) if has_trough else 0.22
    sel_shelf = np.zeros(n_b, dtype=bool)
    for hi_lim in (0.62, 0.90):
        sel_shelf = ok & _outside(uc) & (np.abs(uc) >= edge + 0.06) & (np.abs(uc) <= hi_lim)
        if int(sel_shelf.sum()) >= 6:
            break
    if int(sel_shelf.sum()) >= 6:
        c1, c0 = np.polyfit(uc[sel_shelf], med[sel_shelf], 1)
    else:
        c1, c0 = 0.0, hi_lvl
    if not np.isfinite(c0):
        c1, c0 = 0.0, hi_lvl

    # продольный наклон полки (и дна): медиана уровня полки по полосам 1.5 м
    yb, zb = [], []
    shelf_mask = (np.abs(uu) <= FLOOR_U_MEAS_M) & _outside(uu)
    for y0 in np.arange(y_lo, y_hi - 1.5, 1.5):
        b = shelf_mask & (yy >= y0) & (yy < y0 + 1.5)
        if int(b.sum()) >= 80:
            yb.append(y0 + 0.75)
            zb.append(float(np.median(zz[b])))
    c_y = float(np.polyfit(yb, zb, 1)[0]) if len(yb) >= 3 else 0.0

    # подполосы 3 м — контроль стабильности ширины/центра вдоль пути
    wide, cent, deep = [], [], []
    for y0 in np.arange(y_lo, y_hi - 3.0, 3.0):
        b = (yy >= y0) & (yy < y0 + 3.0)
        if int(b.sum()) < 800:
            continue
        med2 = np.full(n_b, np.nan)
        for k in range(n_b):
            sel = b & (uu >= edges[k]) & (uu < edges[k + 1])
            if int(sel.sum()) >= 3:
                med2[k] = float(np.median(zz[sel]))
        ok2 = np.isfinite(med2)
        if int(ok2.sum()) < 6:
            continue
        kk = int(np.flatnonzero(ok2)[np.argmin(np.abs(uc[np.flatnonzero(ok2)]))])
        if not (np.isfinite(med2[kk]) and np.isfinite(c0)
                and med2[kk] < c0 - FLOOR_SHELF_DROP_M):
            continue
        a2, b2 = kk, kk
        while a2 - 1 >= 0 and np.isfinite(med2[a2 - 1]) and med2[a2 - 1] < c0 - FLOOR_SHELF_DROP_M:
            a2 -= 1
        while b2 + 1 < n_b and np.isfinite(med2[b2 + 1]) and med2[b2 + 1] < c0 - FLOOR_SHELF_DROP_M:
            b2 += 1
        w = float(uc[b2] - uc[a2] + FLOOR_BIN_M)
        wide.append(w)
        cent.append(float(0.5 * (uc[a2] + uc[b2])))
        bsel = b & (uu >= uc[a2] - FLOOR_BIN_M) & (uu <= uc[b2] + FLOOR_BIN_M)
        if int(bsel.sum()) >= 10:
            deep.append(float(c0 - np.percentile(zz[bsel], 10.0)))

    # узлы меша: 5-см сетка по u, значения из медиан бинов (NaN -> интерполяция)
    u_nodes = np.arange(-FLOOR_U_MESH_M, FLOOR_U_MESH_M + 0.05, 0.05)
    if u_nodes[-1] > FLOOR_U_MESH_M + 1e-9:
        u_nodes = np.append(u_nodes, FLOOR_U_MESH_M)
    z_nodes = np.interp(u_nodes, uc[ok], med[ok])
    if has_trough:
        inb_nodes = (u_nodes >= u_l) & (u_nodes <= u_r)
        z_nodes[inb_nodes] = bottom

    y_ref = 0.5 * (y_lo + y_hi)
    return {
        'u_nodes': u_nodes, 'z_nodes': z_nodes,
        'y_lo': float(y_lo), 'y_hi': float(y_hi), 'y_ref': float(y_ref),
        'shelf0': float(c0), 'shelf_slope_u': float(c1), 'shelf_slope_y': float(c_y),
        'has_trough': has_trough, 'u_l': u_l, 'u_r': u_r,
        'width': float(u_r - u_l) if has_trough else 0.0,
        'center': float(0.5 * (u_l + u_r)) if has_trough else float('nan'),
        'bottom': float(bottom), 'depth': float(c0 - bottom) if has_trough else 0.0,
        'wall_run_m': float(wall_run), 'n_bottom': n_bottom,
        'n_floor_pts': int(m.sum()),
        'shelf_lo': float(lo_lvl), 'shelf_hi': float(hi_lvl),
        'width_bands': [float(v) for v in wide],
        'center_bands': [float(v) for v in cent],
        'depth_bands': [float(v) for v in deep],
    }


def _floor_mesh(a, b, c, y_lo, y_hi, x_lo=-5.0, x_hi=5.0, step=1.0):
    """Плоскость пола в коридоре X ±5 м, Y как у облака (запасной вариант)."""
    xs = np.arange(x_lo, x_hi + 1e-9, step)
    ys = np.arange(y_lo, y_hi + 1e-9, step)
    if xs.size < 2:
        xs = np.array([x_lo, x_hi])
    if ys.size < 2:
        ys = np.array([y_lo, y_hi])
    nx, ny = xs.size, ys.size
    verts = np.empty((nx * ny, 3))
    for j, yv in enumerate(ys):
        for k, xv in enumerate(xs):
            verts[j * nx + k] = (xv, yv, a * xv + b * yv + c)
    faces = []
    for j in range(ny - 1):
        for k in range(nx - 1):
            p0 = j * nx + k
            p1 = j * nx + k + 1
            p2 = (j + 1) * nx + k + 1
            p3 = (j + 1) * nx + k
            faces.append([p0, p1, p2])
            faces.append([p0, p2, p3])
    return verts, faces


def _floor_mesh_profile(prof, s_ax, i_ax, y_lo, y_hi, step=FLOOR_MESH_STEP_M):
    """Экструзия ИЗМЕРЕННОГО профиля пола вдоль Y (полки, стенки лотка, дно).

    Профиль задан таблицей (u, z) в локальной системе пары рельсов: u — поперёк
    от оси пары (плюс к правой нити), z — абсолютная высота, измеренная в ближней
    зоне (prof['y_lo']..prof['y_hi']). Вдоль остального Y профиль протянут без
    изменений, с продольным уклоном полки prof['shelf_slope_y'] (он измерен).
    """
    u_nodes = prof['u_nodes']
    z_nodes = prof['z_nodes']
    y_ref = prof['y_ref']
    c_y = prof['shelf_slope_y']
    ys = np.arange(y_lo, y_hi + 1e-9, step)
    if ys.size < 2:
        ys = np.array([y_lo, y_hi])
    ny, nu = ys.size, u_nodes.size
    verts = np.empty((ny * nu, 3))
    for j, yv in enumerate(ys):
        x_axis = s_ax * yv + i_ax + u_nodes
        verts[j * nu:(j + 1) * nu, 0] = x_axis
        verts[j * nu:(j + 1) * nu, 1] = yv
        verts[j * nu:(j + 1) * nu, 2] = z_nodes + c_y * (yv - y_ref)
    faces = []
    for j in range(ny - 1):
        for k in range(nu - 1):
            p0 = j * nu + k
            p1 = p0 + 1
            p2 = (j + 1) * nu + k + 1
            p3 = (j + 1) * nu + k
            faces.append([p0, p1, p2])
            faces.append([p0, p2, p3])
    return verts, faces


def _sleepers(s_axis, i_axis, z_top_axis, y_lo, y_hi, spacing):
    """Шпалы по стандарту: боксы, ориентированные по оси пути."""
    if y_hi - y_lo < spacing:
        return []
    theta = math.atan(s_axis)
    sx = SLEEPER_LENGTH_M * abs(math.cos(theta))
    sy = SLEEPER_WIDTH_M + SLEEPER_LENGTH_M * abs(math.sin(theta))
    out = []
    k = 0
    while True:
        yv = y_lo + 0.5 * spacing + k * spacing
        if yv > y_hi - 0.5 * spacing + 1e-9:
            break
        z_top = z_top_axis(yv) - R65_H_MM / 1000.0
        out.append({
            'center': [float(s_axis * yv + i_axis), float(yv),
                       float(z_top - SLEEPER_HEIGHT_M / 2.0)],
            'size': [float(sx), float(sy), float(SLEEPER_HEIGHT_M)],
        })
        k += 1
    return out


def build_track(db_path: str, frame_index: int, floor_ab=None,
                draw_sleepers: bool = False, draw_gost_profile: bool = False,
                only_observed: bool = False,
                sweep_extra_m=RAIL_SWEEP_EXTRA_M,
                axis_polyline: bool = True) -> dict:
    """Строит геометрию пути по кадру rosbag2 .db3.

    Параметры
    ---------
    db_path : путь к .db3 (например for_hackathon/doubleT_platform/doubleT_platform_0.db3)
    frame_index : номер кадра PointCloud2 в bag (0 .. len(BagFrames)-1)
    floor_ab : необязательная плоскость пола (a, b, c) для z = a*x + b*y + c;
               если None — оценивается по кадру (см. _floor_plane)
    draw_sleepers : False по умолчанию. Путь встроенный (рельсы в плите, между
               ними лоток), шпал в облаке нет — instances пуст. True включает
               схематичную отрисовку по ГОСТ 78-2004 (только для картинки).
    draw_gost_profile : False по умолчанию. True включает СТАРУЮ схему меша рельса —
               экструзию контура Р65 по ГОСТ (шейка/подошва, подуклонка). По
               умолчанию меш строится из ИЗМЕРЕННОГО сечения, и на остальные
               ключи этот флаг не влияет.
    only_observed : False по умолчанию. False — непрерывная развёртка измеренного
               сечения вдоль оси по ИЗМЕРЕННОМУ диапазону + запас sweep_extra_m.
               True — прежний режим «куски по бинам с точками» (поверхность рвётся
               там, где точек мало); оставлен для сравнения и на остальные ключи не
               влияет.
    sweep_extra_m : 15 м по умолчанию — запас развёртки ЗА ПРЕДЕЛЫ измеренного
               диапазона плотных бинов (данных о рельсе дальше нет, а длинная
               экструзия уезжает из сцены). Запас дополнительно ограничивается
               допустимым уходом: по высоте RAIL_DRIFT_LIMIT_M и поперёк пути
               RAIL_X_DRIFT_LIMIT_M от точки привязки. None — режим сравнения:
               развернуть на весь диапазон облака [RAIL_SWEEP_Y_FAR_M,
               RAIL_SWEEP_Y_NEAR_M] без ограничения ухода.
    axis_polyline : True по умолчанию — ось меша строится как ПОЛИЛИНИЯ локальных
               сечений бинов (кривые, стрелки: меш повторяет изгиб пути) с
               продолжением по касательной. False — прежняя ПРЯМАЯ ось
               x = s*y + i на всю длину; оставлена только для сравнения
               (meta['rail_mesh']['axis_model'] = 'polyline'|'straight').

    Возвращает dict:
    {
      'frame': int,
      'axis':  {'s': float, 'i': float},              # середина колеи: x = s*y + i
      'floor': {'a': float, 'b': float, 'c': float},  # плоскость z = a*x+b*y+c; НЕНАДЁЖНА
                                                      # (см. meta.floor_plane_reliability)
      'rails': [ {'side': 'left'|'right',
                  'head': {'top_z': float, 'width_mm': float, 'inner_face_x': float,
                           'height_above_adjacent_m': float, 'y_range': [y0, y1]},
                  'mesh': {'vertices': [[x,y,z], ...], 'faces': [[i,j,k], ...]},
                  'source': {'head': 'measured', 'mesh': 'measured (only observed points)',
                             'web': 'not observed', 'foot': 'not observed'}},
                 ... ],                              # ровно два: left (x меньше), right
      'sleepers': {'instances': [], 'spacing_m': float,
                   'source': 'не наблюдаются (встроенный путь)'},
      'floor_mesh': {'vertices': [[x,y,z], ...], 'faces': [[i,j,k], ...],
                     'source': 'measured: профиль (полки, стенки, дно лотка) ...'},
      'meta': {'gauge_inner_mm': float, 'gauge_axis_mm': float, 'head_width_mm': float,
               'sensor_height_above_head_m': float, 'notes': [str, ...],
               'floor_profile': {...},              # параметры профиля пола (u, z, м)
               'floor_plane_reliability': {...},    # почему floor нельзя брать опорой
               'rail_mesh': {...}}                  # sweep_extra_m, sweep_anchor,
                                                    # rails[side].head_h_above_shelf_m,
                                                    # sweep_extra_far/near_m, *_drift_*_m
    }

    МЕШ РЕЛЬСА строится из наблюдаемых точек: бины 0.5 м вдоль пути, сечение бина
    из 5 узлов по точкам окна ±0.75 м (границы — перцентили u, высоты —
    интерполяция облака точек). Из валидных бинов берётся МЕДИАННОЕ сечение и
    разворачивается вдоль ОСИ ПУТИ непрерывно (без разрывов). Ось по умолчанию —
    ПОЛИЛИНИЯ (axis_polyline=True): центры локальных сечений бинов, найденные в
    окне от предсказания по предыдущим узлам, поэтому на кривых и стрелках меш
    повторяет изгиб пути, а не режет его прямой (прямая x = s*y + i теряет нить
    через 10-15 м и уходит в стену/пол; axis_polyline=False оставлен для сравнения).
    Сечение ставится ПЕРПЕНДИКУЛЯРНО локальной касательной. За пределами измеренного
    ось продолжается по последней касательной. Высота верха — по ИЗМЕРЕННОЙ полке
    пола: z(y) = shelf(y) + медианная высота головки над полкой (плоскость floor для
    привязки НЕ используется). Шейка и подошва лидаром не видны и не дорисовываются
    (source: web/foot 'not observed').

    Оси нитей восстанавливаются из вывода: x = axis.s*y + axis.i ± gauge_axis_mm/2000
    (у каждой нити свой наклон — числа в meta['notes']); это ПРЯМЫЕ по ядру детекции,
    они остаются в контракте, а меш идёт по полилинии (meta['rail_mesh']['rails']
    [side]['polyline'] и ['axis_to_rail_m']). head.top_z, head.inner_face_x, колея и
    «ось-ось» даны в середине ядра детекции (где нити измерены плотно).

    Профиль пола задан в локальной системе пары рельсов: u — поперёк пути от оси
    пары (положительно к правой нити), z — абсолютная высота; полка описана
    прямой shelf_z + shelf_slope_u*u + shelf_slope_y*(y - y0), лоток — u_left/u_right,
    bottom_z, depth. Меш пола — экструзия этого профиля вдоль Y; вне ближней зоны
    (meta['floor_profile']['window_y']) форма профиля не проверена.
    """
    frames = BagFrames(db_path, cache_size=2, prefetch_ahead=0)
    # bag_reader.Frame = (xyz, intensity, ring) — берём только xyz (интенсивность и
    # ring/метка в геометрии не участвуют); метка времени — frames.timestamp(idx)
    xyz = frames[frame_index][0]
    if len(xyz) == 0:
        raise ValueError(f'кадр {frame_index} в {db_path} пуст')
    x = xyz[:, 0].astype(np.float64)
    y = xyz[:, 1].astype(np.float64)
    z = xyz[:, 2].astype(np.float64)

    notes = []
    if floor_ab is None:
        a, b, c = _floor_plane(x, y, z)
        notes.append('пол оценён по кадру: 5-й перцентиль ячеек 1x1 м (|x|<=5 м, '
                     'y в [-60,-2]), МНК с отбраковкой по MAD')
    else:
        a, b, c = float(floor_ab[0]), float(floor_ab[1]), float(floor_ab[2])
        notes.append('пол взят из floor_ab, по кадру не оценивался')

    y_lo = max(float(y.min()), -Y_SEARCH_M)
    y_hi = min(float(y.max()), 5.0)
    track = _find_rails(x, y, z, y_lo, y_hi, notes)

    if track is None:
        s_axis, i_axis = 0.0, 0.0
        head_src = 'gost R65 (не измерено)'
        notes.append('головка не измерена: нити поставлены по нормативной колее '
                     '1594.6 мм от сенсора, верх = пол + 0.17 м')
    else:
        s_axis = 0.5 * (track['s_l'] + track['s_r'])
        i_axis = 0.5 * (track['i_l'] + track['i_r'])
        head_src = 'measured'

    # --- по нити: линия, линия верха, измерения
    rails = []
    for side in ('left', 'right'):
        out_sign = -1.0 if side == 'left' else 1.0
        src = head_src
        if track is None:
            s_r = s_axis
            i_r = i_axis + out_sign * GAUGE_AXIS_NOMINAL_MM / 2000.0
            m_top = b
            c_top = a * i_r + c + 0.17
            meas = {'width_top_mm': R65_HEAD_B_MM, 'width_13_mm': R65_HEAD_B_MM,
                    'inner_off_mm': R65_HEAD_B_MM / 2.0,
                    'height_above_adjacent_m': None,
                    'y_lo': y_hi - 1.0, 'y_hi': y_hi}
        else:
            s_r = track['s_l'] if side == 'left' else track['s_r']
            i_r = track['i_l'] if side == 'left' else track['i_r']
            meas = _head_measurements(x, y, z, side, s_r, i_r,
                                      track['core_lo'], track['core_hi'])
            if meas is None:
                notes.append(f'{side}: головка измерилась меньше чем в 4 полосах — '
                             'взята ширина по ГОСТ, высота не измерена')
                m_top = b
                c_top = a * i_r + c + 0.17
                meas = {'width_top_mm': R65_HEAD_B_MM, 'width_13_mm': R65_HEAD_B_MM,
                        'inner_off_mm': R65_HEAD_B_MM / 2.0,
                        'height_above_adjacent_m': None,
                        'y_lo': track['core_lo'], 'y_hi': track['core_hi']}
                src = 'gost R65 (не измерено)'
            else:
                m_top, c_top = meas['s_top'], meas['i_top']
                # ось нити — линия по центрам поперечных сечений головки (середина
                # p2/p98 точек полосы), а не по гребню: гребень смещён наружу
                # (подуклонка поднимает наружную кромку площадки, наружная сторона
                # в тени), из-за чего колея «ось-ось» завышалась на 10-20 мм
                s_r, i_r = float(meas['s_center']), float(meas['i_center'])

        rails.append({'side': side, 'out_sign': out_sign, 's': s_r, 'i': i_r,
                      'm_top': m_top, 'c_top': c_top, 'meas': meas, 'src': src})

    # --- пол: профиль с лотком. Считается ДО меша нитей: меш рельса привязывается
    # по ИЗМЕРЕННОЙ полке (prof['shelf0'/'shelf_slope_y']), а не по плоскости floor.
    # Считается ВСЕГДА, в том числе когда нити не найдены (тогда ось номинальная):
    # профиль пола от нитей не зависит, а без пола кадр пустой целиком.
    m_top_axis = 0.5 * (rails[0]['m_top'] + rails[1]['m_top'])
    c_top_axis = 0.5 * (rails[0]['c_top'] + rails[1]['c_top'])
    prof = _floor_profile(x, y, z, s_axis, i_axis, m_top_axis, c_top_axis,
                          float(y.max()), notes)
    if prof is not None and track is None:
        notes.append('профиль пола измерен БЕЗ нитей (ось номинальная от сенсора): '
                     'полка %.3f м, лоток %s — нити на кадре не найдены, меш рельса пуст'
                     % (prof['shelf0'],
                        ('u %+.3f..%+.3f м' % (prof['u_l'], prof['u_r']))
                        if prof['has_trough'] else 'не выражен'))

    # Прямая полки в тех же координатах, что и всё остальное: измерена в ближней
    # зоне (окно prof['window_y']), отсчёт от prof['shelf0'] в y = prof['y_ref'].
    shelf_line = None
    if prof is not None:
        shelf_line = {'y_ref': float(prof['y_ref']), 'z_at_y_ref': float(prof['shelf0']),
                      'slope_y': float(prof['shelf_slope_y']),
                      'slope_u': float(prof['shelf_slope_u'])}

    # --- меш нитей: измеренное сечение, развёрнутое вдоль оси (ГОСТ не участвует).
    # Ось — ПОЛИЛИНИЯ локальных сечений (кривые и стрелки) или, при axis_polyline=False,
    # прежняя прямая x = s*y + i (оставлена для сравнения).
    _ym_core = None if track is None else 0.5 * (track['core_lo'] + track['core_hi'])

    def _track_rail(rr):
        _u_rail = ((rr['s'] * _ym_core + rr['i']) - (s_axis * _ym_core + i_axis))
        rr['u_rail'] = float(_u_rail)
        return _rail_track(x, y, z, rr['s'], rr['i'], rr['m_top'], rr['c_top'],
                           rr['out_sign'], y_lo, y_hi, _ym_core,
                           shelf_line=shelf_line, shelf_u_m=_u_rail)

    # --- ЖЁСТКАЯ ПАРА: план (общая центральная линия + ОДНА колея) считается ОДИН раз,
    # ДО сборки мешей, и СТРОИТСЯ ВСЕГДА — иначе на кадрах без пары у потребителя
    # (маска точек, жёсткая колея) нет ни ключа nodes, ни общей оси:
    #   * по обеим нитям, если измерены обе (приоритет — среднее там, где нити
    #     согласованы, и НАДЁЖНАЯ нить там, где противоречат, см. _pair_plan);
    #   * по ОДНОЙ нити, если вторая на кадре не измерена или общей полосы нет:
    #     центр = ось измеренной нити ∓G/2 (её геометрия не меняется), колея =
    #     номинал (измерить колею по одной нити нечем);
    #   * по НОМИНАЛЬНОЙ ОСИ кадра, если нитей нет вовсе (помечается в meta).
    pair = None
    if axis_polyline and track is not None:
        # обе нити трекаем здесь, чтобы план считался по готовым осям
        for rr in rails:
            if rr.get('bsec') is None:
                rr['bsec'] = _track_rail(rr)
                if rr['bsec'] is not None:
                    rr['sec'] = _median_section(rr['bsec'])
                    rr['y_lo'] = float(rr['bsec']['y_dense'][0])
                    rr['y_hi'] = float(rr['bsec']['y_dense'][1])
        if all(rr.get('bsec') is not None for rr in rails):
            pair = _pair_plan(rails[0], rails[1], shelf_line, sweep_extra_m, (y_lo, y_hi),
                              cloud=(y, x, z))
            if pair is None:
                # обе нити измерены, но ОБЩЕЙ ПОЛОСЫ у них нет (<4 общих бинов) —
                # колею измерить нечем: та же пара, но колея = номинал, а оси нитей
                # остаются своими там, где измерены
                pair = _pair_plan(rails[0], rails[1], shelf_line, sweep_extra_m,
                                  (y_lo, y_hi), gauge_override=_pair_nominal_gauge(),
                                  cloud=(y, x, z))
                if pair is not None:
                    notes.append('у нитей нет общего видимого участка (<%d общих бинов) '
                                 '— колею измерить нечем, взята НОМИНАЛЬНАЯ %.1f мм; '
                                 'оси обеих нитей оставлены своими там, где измерены'
                                 % (4, 1000.0 * pair['gauge_m']))
        if pair is None:
            have = [rr for rr in rails
                    if rr.get('bsec') is not None and rr.get('sec') is not None]
            if have:   # вторая нить не измерена: её ось — от оси измеренной на G
                pick = max(have, key=lambda rr: float(rr['bsec']['grid_y'][-1]
                                                      - rr['bsec']['grid_y'][0]))
                other = rails[1] if pick is rails[0] else rails[0]
                pair = _pair_plan_one_rail(pick, shelf_line, sweep_extra_m, (y_lo, y_hi),
                                           other=other, cloud=(y, x, z))
                if pair is not None:
                    notes.append('%s: второй нити на кадре нет (или общей полосы нет) — '
                                 'план по ОДНОЙ нити: ось измеренной нити оставлена '
                                 'как есть, колея %.1f мм взята НОМИНАЛЬНОЙ '
                                 '(измерить нечем), узлы обеих нитей — по этому плану'
                                 % (pick['side'], 1000.0 * pair['gauge_m']))
    if axis_polyline and pair is None:
        # нитей нет ВООБЩЕ: центр — номинальная ось кадра (она же ось полос нитей),
        # колея — номинал; в meta это помечено (pair_center_kind = nominal_axis)
        if shelf_line is not None and prof.get('window_y'):
            y0, y1 = float(prof['window_y'][0]), float(prof['window_y'][1])
        else:
            y0, y1 = max(float(y_lo), -30.0), min(float(y_hi), -1.0)
        pair = _pair_plan_nominal(s_axis, i_axis, y0, y1, shelf_line, sweep_extra_m,
                                  (y_lo, y_hi))
        if pair is not None:
            notes.append('РЕЛЬСЫ НЕ НАЙДЕНЫ: узлы (nodes) обеих нитей даны по '
                         'НОМИНАЛЬНОЙ оси кадра s=%+.5f i=%+.5f, колея %.1f мм — '
                         'номинал; это не измерение, доверие понижать'
                         % (float(s_axis), float(i_axis), 1000.0 * pair['gauge_m']))
    if axis_polyline and pair is not None:
        notes.append('план пары: %s' % pair['center_source'])
        if pair['center_kind'] == 'both_rails':
            notes.append('РЕЛЬСЫ — ЖЁСТКАЯ ПАРА: колея на кадр %.1f мм (измеренная '
                         'медиана %.1f мм, зажатие %+.1f мм, разброс измерения %.1f мм '
                         'по %d общим бинам); продолжение — общая центральная линия по '
                         'своей кривизне %.4f/%.4f 1/м, нити тем же G/2. По Z каждая '
                         'нить идёт по СВОЕЙ измеренной коронке (подуклонка/'
                         'возвышение не выравниваются)'
                         % (1000.0 * pair['gauge_m'], 1000.0 * (pair['gauge_raw_m']
                                                                or pair['gauge_m']),
                            pair['gauge_clamped_mm'] or 0.0,
                            pair['gauge_spread_measured_mm'] or 0.0,
                            pair['n_bins_both'],
                            pair['kappa_far_1_m'] or 0.0, pair['kappa_near_1_m'] or 0.0))

    for r in rails:
        bsec = r.get('bsec')
        if bsec is None and track is not None:
            if axis_polyline:
                bsec = _track_rail(r)
            else:
                bsec = _rail_bin_sections(x, y, z, r['s'], r['i'], r['m_top'],
                                          r['c_top'], r['out_sign'])
            r['bsec'] = bsec
            if bsec is not None:
                r['sec'] = _median_section(bsec)
                r['y_lo'] = float(bsec['y_dense'][0])
                r['y_hi'] = float(bsec['y_dense'][1])
        if bsec is None:
            r['mesh'] = ([], [])
            r['sec'] = None
            r['mstat'] = {'n_bins_ok': 0, 'n_bins_span': 0, 'ribbons': 0,
                          'y_dense': None, 'n_pts_sel': 0, 'gaps': []}
            r['y_lo'] = float(r['meas']['y_lo'])
            r['y_hi'] = float(r['meas']['y_hi'])
            r['sweep'] = None
            r['poly'] = None
            r['mesh_measured'] = ([], [])
            # УЗЛЫ ОСИ ВСЁ РАВНО ДАЁМ: нить на кадре не измерена, но по плану пары
            # её ось известна (G от парной нити или номинальная ось кадра) — так у
            # потребителя (маска точек) есть ось, а не догадка по вершинам меша.
            # Источник помечен: pair_nominal_nodes = все узлы, nodes_source.
            if pair is not None:
                other = rails[1] if r is rails[0] else rails[0]
                ob = other.get('bsec')
                z_pair = None if ob is None else (np.asarray(ob['grid_y'], float),
                                                  np.asarray(ob['grid_z'], float))
                z_line = (lambda yv, r=r: r['m_top'] * yv + r['c_top'])
                r['nodes'] = _pair_rail_nodes(pair, r['side'],
                                              z_pair if z_pair is not None else (None, z_line))
                r['nodes_source'] = ('pair_by_other_rail' if z_pair is not None
                                     else 'nominal_axis')
                r['pair_nominal_nodes'] = len(r['nodes'])
                # план общий на кадр: отдаём его и этой нити, чтобы в meta были видны
                # вид плана, колея и строка источника (иначе по неё все поля пустые)
                r['pair'] = pair
                # МЕШ ВСЁ РАВНО СТРОИМ: своего сечения у нити нет (bsec пуст), но ось
                # (nodes) известна по плану пары — по ней и строится ТЕЛО рельса
                # (жалоба: у roundT_doubleT f128 правая нить пропадала целиком: узлов
                # 73, меш 0 вершин, y_range 3.5 м). Сечение — своё, если было, иначе
                # измеренное у ПАРНОЙ нити, иначе номинал Р65 по ширине головки
                # (профиль тела всё равно ГОСТ + коронка; точки для коронки тут не
                # нужны — сечение берётся целиком). Пустой/крошечный y_range нити не
                # мешает: узлы заданы планом, а не её y_range.
                nd = np.asarray(r['nodes'], float)
                ok_nd = (nd.ndim == 2 and nd.shape[0] >= 2
                         and np.all(np.isfinite(nd[:, 2])))
                if ok_nd:
                    sec_n = r.get('sec')
                    sec_src = 'pair_section'
                    if sec_n is None:
                        sec_n = other.get('sec')
                    if sec_n is None:
                        sec_n = {'half_width_m': R65_HEAD_B_MM / 2000.0,
                                 'width_m': R65_HEAD_B_MM / 1000.0, 'u_off_m': 0.0,
                                 'vz_nodes_m': [0.0] * RAIL_NODES,
                                 'depth_m': 0.0, 'n_bins_used': 0, 'y_dense': None}
                        sec_src = 'nominal_r65'
                    mv, mf, _np, _dv = _rail_mesh_body(nd[:, 0], nd[:, 1], nd[:, 2],
                                                       sec_n, r['out_sign'])
                    r['mesh'] = (mv, mf)
                    r['sec'] = sec_n
                    r['mesh_source'] = sec_src
                    vm = np.asarray(mv, float)[:, 1] if mv else np.zeros(0)
                    r['extent_y'] = ([float(vm.min()), float(vm.max())] if vm.size
                                     else None)
                    r['mstat'] = {'n_bins_ok': 0, 'n_bins_span': 0, 'ribbons': 0,
                                  'y_dense': None, 'n_pts_sel': 0, 'gaps': [],
                                  'n_verts': len(mv), 'n_faces': len(mf)}
                    r['mesh_measured'] = ([], [])   # своего ИЗМЕРЕННОГО сечения нет
                    r['beyond_measured'] = 1.0
            else:
                r['nodes'] = None
                r['nodes_source'] = 'none'
                r['pair_nominal_nodes'] = None
            notes.append('%s: бинов с %d и более точками рельса нет — сечение измерить '
                         'нечем, меш пуст%s'
                         % (r['side'], RAIL_MIN_PTS_BIN,
                            ('' if pair is None else
                             '; ось (nodes) взята по плану пары — номинальная, '
                             'профиль не измерен')))
            continue
        sec = r.get('sec') or _median_section(bsec)
        r['sec'] = sec
        r['y_lo'], r['y_hi'] = float(bsec['y_dense'][0]), float(bsec['y_dense'][1])
        r['poly'] = None
        if only_observed:
            if axis_polyline and track is not None:
                # куски по ИЗМЕРЕННЫМ бинам полилинии: разрывы не зашиваются
                keep = np.isin(bsec['grid_bins'], bsec['bins'])
                ky = np.asarray(bsec['grid_y'], float)[keep]
                kx = np.asarray(bsec['grid_x'], float)[keep]
                kz = np.asarray(bsec['grid_z'], float)[keep]
                kb = np.asarray(bsec['grid_bins'], int)[keep]
                verts, faces = _rail_mesh_path(ky, kx, kz, sec, r['out_sign'],
                                               pair_ok=np.diff(kb) == 1)
                r['nodes'] = [[round(float(ky[i]), 2), round(float(kx[i]), 3),
                               round(float(kz[i]), 3)] for i in range(ky.size)]
                r['nodes_source'] = 'measured_only'
                r['pair_nominal_nodes'] = 0
            else:
                verts, faces = _rail_mesh_bins(bsec, r['s'], r['i'], r['out_sign'])
                yv = np.arange(float(bsec['y_dense'][0]),
                               float(bsec['y_dense'][1]) + 1e-9, POLY_BIN_M)
                r['nodes'] = [[round(float(v), 2), round(float(r['s'] * v + r['i']), 3),
                               round(float(r['m_top'] * v + r['c_top']), 3)] for v in yv]
                r['nodes_source'] = 'straight_nominal'
                r['pair_nominal_nodes'] = len(r['nodes'])
            mode = 'only_observed: куски по бинам с точками'
            r['sweep'] = None
        elif axis_polyline:
            built = _rail_build_polyline(x, y, z, r, bsec, sec, shelf_line,
                                         sweep_extra_m, (y_lo, y_hi),
                                         axis_ab=(s_axis, i_axis), pair=pair)
            verts, faces = built['verts'], built['faces']
            r['sweep'] = built['sweep']
            r['poly'] = built
            r['mesh_measured'] = (built['meas_verts'], built['meas_faces'])
            r['nodes'] = built['axis_nodes']
            r['nodes_source'] = ('measured_pair'
                                 if (pair is not None
                                     and pair.get('center_kind') == 'both_rails')
                                 else 'measured_one_rail')
            r['pair_nominal_nodes'] = 0
            mode = ('непрерывная развёртка ТЕЛА рельса (ГОСТ + измеренная коронка) '
                    'вдоль ПОЛИЛИНИИ оси (узлов %d, продолжение по дуге)'
                    % len(r['bsec']['grid_y']))
        else:
            # Верх головки над полкой — медиана по ПЛОТНЫМ бинам (своя на нить).
            # Это высота рельса над полкой, а не head.height_above_adjacent_m
            # (последняя считается над прилегающей поверхностью у рельса, которая
            # выше плоской полки, поэтому она меньше на 30-60 мм).
            sh = shelf_line
            if sh is not None:
                z_top_dense = r['m_top'] * bsec['y_c'] + r['c_top']
                shelf_at_bins = sh['z_at_y_ref'] + sh['slope_y'] * (bsec['y_c'] - sh['y_ref'])
                h_head = float(np.median(z_top_dense - shelf_at_bins))
                z_at = (lambda yv, sh=sh, hh=h_head: sh['z_at_y_ref']
                        + sh['slope_y'] * (yv - sh['y_ref']) + hh)
            else:
                h_head = None
                z_at = (lambda yv, r=r: r['m_top'] * yv + r['c_top'])
            y_lo_s, y_hi_s, extra_used, cap_h, cap_x = _rail_sweep_extent(
                bsec, r, sh, sweep_extra_m, float(y.min()), float(y.max()))
            y_dense_mid = 0.5 * (r['y_lo'] + r['y_hi'])
            if sh is not None:
                z_slope = sh['slope_y']
            else:
                z_slope = r['m_top']
            z_drift_far = abs(z_slope * (y_lo_s - y_dense_mid))
            z_drift_extra = abs(z_slope * max(0.0, r['y_lo'] - y_lo_s))
            r['sweep'] = {'y_lo': y_lo_s, 'y_hi': y_hi_s, 'extra_m': extra_used,
                          'cap_h_m': cap_h, 'cap_x_m': cap_x, 'head_h_m': h_head,
                          'shelf_ref': sh, 'y_dense_mid': y_dense_mid,
                          'extra_far_m': float(r['y_lo'] - y_lo_s),
                          'extra_near_m': float(y_hi_s - r['y_hi']),
                          'z_drift_far_m': z_drift_far,
                          'z_drift_extra_m': z_drift_extra,
                          'model': 'straight',
                          'x_drift_far_m': abs(r['s'] * (y_lo_s - y_dense_mid))}
            verts, faces = _rail_mesh_sweep(sec, r['s'], r['i'], z_at,
                                            r['out_sign'], y_lo_s, y_hi_s)
            r['nodes'] = [[round(float(v), 2), round(float(r['s'] * v + r['i']), 3),
                           round(float(z_at(float(v))), 3)]
                          for v in np.arange(y_lo_s, y_hi_s + 1e-9, POLY_BIN_M)]
            r['nodes_source'] = 'straight_nominal'
            r['pair_nominal_nodes'] = len(r['nodes'])
            mode = ('непрерывная развёртка измеренного сечения вдоль прямой оси '
                    '%.1f..%.1f м' % (y_lo_s, y_hi_s))
        r['mesh'] = (verts, faces)
        r['pair'] = pair
        if 'mesh_measured' not in r:
            # прочие режимы (прямая ось, only_observed, ГОСТ): тело не строится,
            # полоска = сам меш
            r['mesh_measured'] = (verts, faces)
        dense_len = r['y_hi'] - r['y_lo']
        vm = np.asarray(verts, float)[:, 1] if verts else np.zeros(0)
        sweep_len = max(1e-9, float(vm.max() - vm.min())) if vm.size else 0.0
        beyond = (1.0 - dense_len / sweep_len) if sweep_len > 0 else 0.0
        r['beyond_measured'] = float(np.clip(beyond, 0.0, 1.0))
        r['extent_y'] = ([float(vm.min()), float(vm.max())] if vm.size else None)
        r['mstat'] = {'n_bins_ok': bsec['n_bins_ok'], 'n_bins_span': bsec['n_span'],
                      'ribbons': bsec['ribbons'], 'y_dense': list(bsec['y_dense']),
                      'n_pts_sel': bsec['n_pts_sel'], 'gaps': bsec['gaps'],
                      'n_verts': len(verts), 'n_faces': len(faces)}
        notes.append('%s: сечение измерено по плотным бинам %d из %d (шаг %.2f м, '
                     'порог %d точки = %.0f точек/м) на участке y = %.1f..%.1f м: '
                     'ширина %.0f мм, видимая глубина под верхом %.0f мм; меш тянется '
                     '%.1f..%.1f м (%.0f %% длины — за пределами измеренного, сечение '
                     'то же) — режим: %s'
                     % (r['side'], bsec['n_bins_ok'], bsec['n_span'], RAIL_BIN_Y_M,
                        RAIL_MIN_PTS_BIN, RAIL_MIN_PTS_BIN / RAIL_BIN_Y_M,
                        r['y_lo'], r['y_hi'], 1000 * sec['width_m'],
                        1000 * sec['depth_m'],
                        (r['extent_y'][0] if r['extent_y'] else float('nan')),
                        (r['extent_y'][1] if r['extent_y'] else float('nan')),
                        100 * r['beyond_measured'], mode))
        if r['poly'] is not None:
            po = r['poly']
            mt = po['metric']
            notes.append('%s: ось — ПОЛИЛИНИЯ %d узлов (измерено %d, зашито интерполяцией '
                         '%d), длина по пути %.1f м, разброс от хорды — см. meta; '
                         'скелет: касательные на концах %s, кривизна дал/ближ %s/%s 1/м, '
                         'шагов >%.2f м %d, отброшено MAD %d, сглаживание сдвинуло '
                         'медиана %s / p95 %s м'
                         % (r['side'], len(bsec['grid_y']), int(bsec['bins'].size),
                            po['n_bridging'], po['dense_len_m'],
                            po['tangent_deg'], _fmt_num(po['sweep']['kappa_far_1_m'], 4),
                            _fmt_num(po['sweep']['kappa_near_1_m'], 4), POLY_JUMP_M,
                            po['n_jumps'], po['n_rejected'],
                            _fmt_num(po['smooth_resid_m'][0] if po['smooth_resid_m'] else None, 4),
                            _fmt_num(po['smooth_resid_m'][1] if po['smooth_resid_m'] else None, 4)))
            notes.append('%s: МЕТРИКА оси (расстояние до точек рельса, полоса %.2f..%.2f м '
                         'над полкой, |u| <= %.2f м, окно ±%.1f м по Y): по измеренным '
                         'узлам медиана %s / p95 %s м (%d узлов), по зашитым разрывам %s / '
                         '%s м (%d), за пределами данных %s / %s м (%d); критерий: '
                         'медиана <= 0.05, p95 <= 0.15'
                         % (r['side'], POLY_BAND_LO_M, POLY_BAND_HI_M, POLY_U_LIM_M,
                            POLY_METRIC_WIN_M, _fmt_num(mt['measured']['median_m'], 3),
                            _fmt_num(mt['measured']['p95_m'], 3),
                            mt['measured']['n_nodes_with_points'],
                            _fmt_num(mt['bridged']['median_m'], 3),
                            _fmt_num(mt['bridged']['p95_m'], 3),
                            mt['bridged']['n_nodes_with_points'],
                            _fmt_num(mt['outside']['median_m'], 3),
                            _fmt_num(mt['outside']['p95_m'], 3),
                            mt['outside']['n_nodes_with_points']))
            notes.append('%s: ПОЛИЛИНИЯ — высота верха над измеренной полкой, коридор '
                         '%.2f..%.2f м: бинов отброшено по высоте %d (верх уезжал '
                         'сквозь пол/в потолок), узлов поджато в коридор %d, медиана '
                         'высоты головки для продолжения %.3f м%s. Ни один узел внутри '
                         'данных не ниже полки + %.2f м (жёсткий минимум)'
                         % (r['side'], POLY_TOP_LO_M, POLY_TOP_HI_M,
                            r['poly']['n_rejected_height'], r['poly']['n_clamped_height'],
                            r['poly']['sweep']['head_h_m'] or float('nan'),
                            ' (медиана поджата)' if r['poly']['h_head_clamped'] else '',
                            POLY_TOP_HARD_M))
        sw = r.get('sweep')
        if sw is not None and sw['extra_m'] is not None and sw['shelf_ref'] is not None:
            if sw.get('model') == 'polyline':
                notes.append('%s: режим развёртки по ПОЛИЛИНИИ — измеренный участок '
                             '%.1f м ПО ПУТИ + продолжение по КАСАТЕЛЬНОЙ %.1f м на '
                             'дальнем конце и %.1f м на ближнем (запас %.1f м); дальше меш '
                             'НЕ продолжается. Запас ограничен уходом от уровня в СЕРЕДИНЕ '
                             'измеренного участка y=%.1f м: по высоте %.3f м (бюджет %.2f м; '
                             'наклон полки %+.5f м/м -> не больше %.1f м), ПОПЕРЁК ПУТИ '
                             '%.3f м (бюджет %.2f м; кривизна на конце %.4f 1/м -> не '
                             'больше %.1f м: продолжение по касательной уходит от '
                             'изогнутого пути как kappa*L^2/2). Верх меша привязан к '
                             'ИЗМЕРЕННОЙ полке: z(y) = полка(y) + %.3f м; плоскость floor '
                             'для привязки НЕ используется'
                             % (r['side'], sw['dense_len_m'], sw['extra_m'],
                                sw['extra_near_used_m'], RAIL_SWEEP_EXTRA_M,
                                sw['y_dense_mid'], sw['z_drift_far_m'], RAIL_DRIFT_LIMIT_M,
                                sw['shelf_ref']['slope_y'],
                                (sw['cap_h_m'] if sw['cap_h_m'] is not None else float('inf')),
                                sw['x_drift_far_m'], RAIL_X_DRIFT_LIMIT_M,
                                (sw['kappa_far_1_m'] or 0.0),
                                (sw['cap_x_m'] if sw['cap_x_m'] is not None else float('inf')),
                                sw['head_h_m']))
            else:
                notes.append('%s: режим развёртки — измеренный диапазон + %.1f м (запас %.1f м, '
                             'то есть %.1f м на дальнем конце и %.1f м на ближнем); дальше меш '
                             'НЕ продолжается. Запас ограничен уходом от уровня в СЕРЕДИНЕ '
                             'плотной зоны y=%.1f м: по высоте %.3f м (бюджет %.2f м; наклон '
                             'полки %+.5f м/м -> не больше %.1f м), по X %.3f м (бюджет %.2f м; '
                             'наклон оси %+.5f -> не больше %.1f м). Верх меша привязан к '
                             'ИЗМЕРЕННОЙ полке: z(y) = полка(y) + %.3f м (медиана высоты '
                             'головки над полкой по плотным бинам); плоскость floor для '
                             'привязки НЕ используется'
                             % (r['side'], sw['extra_m'], RAIL_SWEEP_EXTRA_M,
                                sw['extra_far_m'], sw['extra_near_m'],
                                sw['y_dense_mid'], sw['z_drift_far_m'], RAIL_DRIFT_LIMIT_M,
                                sw['shelf_ref']['slope_y'],
                                (sw['cap_h_m'] if sw['cap_h_m'] is not None else float('inf')),
                                sw['x_drift_far_m'], RAIL_X_DRIFT_LIMIT_M, r['s'],
                                (sw['cap_x_m'] if sw['cap_x_m'] is not None else float('inf')),
                                sw['head_h_m']))
        elif sw is not None and sw['extra_m'] is None:
            notes.append('%s: РЕЖИМ СРАВНЕНИЯ (sweep_extra_m=None) — развёртка на весь '
                         'диапазон облака %.1f..%.1f м без ограничения ухода; '
                         'по умолчанию так НЕ делается'
                         % (r['side'], sw['y_lo'], sw['y_hi']))

    common_lo = max(r['y_lo'] for r in rails)
    common_hi = min(r['y_hi'] for r in rails)
    if common_hi - common_lo < 0.5:
        common_lo, common_hi = min(r['y_lo'] for r in rails), max(r['y_hi'] for r in rails)
        notes.append('общего видимого участка у нитей нет — общие величины даны '
                     'по объединению диапазонов')
    # Отчётные величины нитей (top_z, inner_face_x, колея) берём в середине ЯДРА
    # детекции — участка, где нити измерены плотно. Если брать середину всего меша,
    # колея «поедет» вместе с длиной видимого участка (нити не параллельны:
    # разность наклонов ~0.0015 м/м даёт 8 мм на 6 м).
    if track is not None:
        y_mid = 0.5 * (track['core_lo'] + track['core_hi'])
    else:
        y_mid = 0.5 * (common_lo + common_hi)

    # --- вывод по нитям
    rails_out = []
    head_widths = []
    inner_faces = []
    for r in rails:
        meas = r['meas']
        if draw_gost_profile:
            verts_mm, _srcs = rail_profile_mm(meas['width_top_mm'])
            verts, faces = _rail_mesh(verts_mm, r['s'], r['i'], r['m_top'], r['c_top'],
                                      r['y_lo'], r['y_hi'], r['out_sign'])
            src = {'head': r['src'], 'mesh': 'gost R65 (draw_gost_profile=True)',
                   'mesh_measured': 'measured (верхняя площадка головки, 5 узлов)',
                   'body': 'gost R65 (draw_gost_profile=True)',
                   'web': 'gost R65', 'foot': 'gost R65'}
        elif only_observed:
            verts, faces = r['mesh']
            src = {'head': r['src'],
                   'mesh': 'measured (only observed bins, gaps kept)',
                   'mesh_measured': 'measured (only observed bins, gaps kept)',
                   'body': 'not built (only observed bins)',
                   'web': 'not observed', 'foot': 'not observed'}
        else:
            verts, faces = r['mesh']
            if axis_polyline:
                src = {'head': r['src'], 'mesh': _sweep_source_text(r),
                       'mesh_measured': 'measured (только верхняя площадка головки, 5 узлов)',
                       'body': ('gost R65 (head sides, fillets, web, foot) под измеренной '
                                'коронкой: верх совпадает с measured, тело вниз до подошвы'),
                       'web': 'gost R65', 'foot': 'gost R65'}
            else:
                src = {'head': r['src'], 'mesh': _sweep_source_text(r),
                       'mesh_measured': 'measured (верхняя площадка головки, 5 узлов)',
                       'body': 'not built (прямая ось, режим сравнения)',
                       'web': 'not observed', 'foot': 'not observed'}
        inner_off = (meas['inner_off_mm'] if meas['inner_off_mm'] is not None
                     else R65_HEAD_B_MM / 2.0)
        inner_face_x = r['s'] * y_mid + r['i'] - r['out_sign'] * inner_off / 1000.0
        rails_out.append({
            'side': r['side'],
            'head': {
                'top_z': float(r['m_top'] * y_mid + r['c_top']),
                'width_mm': float(meas['width_top_mm']),
                'inner_face_x': float(inner_face_x),
                'height_above_adjacent_m': (None if meas['height_above_adjacent_m'] is None
                                            else float(meas['height_above_adjacent_m'])),
                'y_range': [r['y_lo'], r['y_hi']],
            },
            'mesh': {'vertices': [[float(v) for v in row] for row in verts],
                     'faces': [[int(f0), int(f1), int(f2)] for (f0, f1, f2) in faces]},
            # ИЗМЕРЕННАЯ ПОЛОСКА (только верхняя площадка головки, 5 узлов) — как
            # было до тела; ключ mesh на неё больше НЕ указывает (там тело рельса)
            'mesh_measured': {
                'vertices': [[float(v) for v in row] for row in r['mesh_measured'][0]],
                'faces': [[int(f0), int(f1), int(f2)] for (f0, f1, f2) in r['mesh_measured'][1]]},
            'source': src,
        })
        head_widths.append(float(meas['width_top_mm']))
        inner_faces.append(float(inner_face_x))

    gauge_axis_m = abs(((rails[1]['s'] * y_mid + rails[1]['i'])
                        - (rails[0]['s'] * y_mid + rails[0]['i'])))
    gauge_inner_m = abs(inner_faces[1] - inner_faces[0])

    z_top_axis = (lambda yv, rl=rails: 0.5 * (rl[0]['m_top'] * yv + rl[0]['c_top']
                                              + rl[1]['m_top'] * yv + rl[1]['c_top']))

    # --- шпалы: встроенный путь, в облаке их нет -> по умолчанию не рисуем
    sleepers_src = 'не наблюдаются (встроенный путь)'
    if draw_sleepers and common_hi - common_lo >= SLEEPER_SPACING_M:
        instances = _sleepers(s_axis, i_axis, z_top_axis, common_lo, common_hi,
                              SLEEPER_SPACING_M)
        sleepers_src = 'gost' if instances else sleepers_src
        notes.append('шпалы построены схематично по ГОСТ 78-2004 (draw_sleepers=True), '
                     'в облаке они НЕ наблюдаются')
    else:
        instances = []
        notes.append('шпалы не рисуются (instances пуст, source «%s»): между рельсами '
                     'идёт встроенный лоток, периодических структур в поле нет — '
                     'проверено Фурье по полосам 5 см: амплитуда 0.3-1.0 м равна 0.0 мм '
                     'при пороге обнаружения ~13-16 мм (инъекция 20 мм даёт 10-13 мм)'
                     % sleepers_src)

    # --- пол: меш профиля (экструзия вдоль Y); профиль посчитан выше, до меша нитей
    if prof is not None:
        # меш пола тянем по тому же участку Y, что и нити: вне него профиль —
        # экстраполяция (форма проверена только в ближней зоне). Если нитей не
        # нашли, участок нитей вырожден (около 1 м) — тогда тянем пол по тому же
        # диапазону, в котором ищем путь (иначе пола на кадре практически нет)
        f_lo = min(rails[0]['y_lo'], rails[1]['y_lo'])
        f_hi = max(rails[0]['y_hi'], rails[1]['y_hi'])
        if f_hi - f_lo < 5.0:
            f_lo = max(float(y.min()), -Y_SEARCH_M)
            f_hi = min(float(y.max()), 5.0)
            notes.append('участок нитей вырожден (%.1f м) — меш пола протянут по '
                         'диапазону поиска пути y = %.1f..%.1f м' % (f_hi - f_lo,
                                                                    f_lo, f_hi))
        f_verts, f_faces = _floor_mesh_profile(prof, s_axis, i_axis, f_lo, f_hi)
        floor_src = ('measured: профиль (полки, стенки, дно лотка) по облаку, '
                     'экструзия вдоль Y')
        notes.append('пол — ПРОФИЛЬ, а не плоскость: поперечное сечение измерено в '
                     'ближней зоне y = %.1f..%.1f м (там пол плотно сэмплирован), '
                     '|u| <= %.2f м; полка %.3f %+.3f*u %+.4f*(y-y0) м'
                     % (prof['y_lo'], prof['y_hi'], FLOOR_U_MESH_M,
                        prof['shelf0'], prof['shelf_slope_u'], prof['shelf_slope_y']))
        if prof['has_trough']:
            notes.append('лоток: u = %+.3f..%+.3f м (ширина %.3f м, центр %+.3f м '
                         'от оси пары), дно %+.3f м, глубина %.3f м ниже полки, '
                         'стенки почти вертикальные (переход %.0f мм по горизонтали); '
                         'точек дна %d'
                         % (prof['u_l'], prof['u_r'], prof['width'], prof['center'],
                            prof['bottom'], prof['depth'], 1000 * prof['wall_run_m'],
                            prof['n_bottom']))
            if prof['width_bands']:
                notes.append('устойчивость лотка вдоль пути (полосы 3 м): ширина %.2f-%.2f м, '
                             'центр %+.3f..%+.3f м, глубина %.3f-%.3f м'
                             % (min(prof['width_bands']), max(prof['width_bands']),
                                min(prof['center_bands']), max(prof['center_bands']),
                                min(prof['depth_bands']), max(prof['depth_bands'])))
        else:
            notes.append('лоток в этом кадре не выражен: ниже полки более чем на '
                         '%.0f мм связного участка нет (глубина уровня %.3f м против '
                         'полки %.3f м)' % (1000 * FLOOR_SHELF_DROP_M, prof['shelf_lo'],
                                            prof['shelf0']))
        notes.append('дно лотка освещено слабо (в 3-4 раза реже полки: замерено ~90-130 '
                     'против 320-470 точек на метр пути) — придонные углы у '
                     'вертикальных стенок могут быть в тени, поэтому измеренная '
                     'глубина %.3f м является нижней оценкой' % prof['depth'])
    else:
        f_verts, f_faces = _floor_mesh(a, b, c, float(y.min()), float(y.max()))
        floor_src = 'plane: измеренный профиль получить не удалось, оставлена плоскость'
        notes.append('профиль пола измерить не удалось (мало точек пола в ближней зоне) '
                     '— floor_mesh оставлен плоским по floor.a/b/c')
    z_head = 0.5 * (rails_out[0]['head']['top_z'] + rails_out[1]['head']['top_z'])

    if track is not None:
        notes.append('наклоны линий вдоль пути (м/м): ось %+.5f, левая нить %+.5f, '
                     'правая нить %+.5f; верх головки: левая %+.5f, правая %+.5f'
                     % (s_axis, track['s_l'], track['s_r'],
                        rails[0]['m_top'], rails[1]['m_top']))
        notes.append('нити найдены в %d полосах, ядро детекции y = %.1f..%.1f м'
                     % (track['n_band'], track['core_lo'], track['core_hi']))
    notes.append('МЕШ РЕЛЬСА — измеренное сечение, развёрнутое вдоль оси (source.mesh = '
                 'measured section swept): сечение = медиана по плотным бинам %.2f м '
                 '(ширина по перцентилям u, профиль из 5 узлов под верхом головки), '
                 'далее оно же ставится в каждый узел сетки %.2f м по Y на прямую ось '
                 'x = s*y + i и высоту z(y) = ИЗМЕРЕННАЯ полка(y) + высота головки над '
                 'полкой — БЕЗ разрывов. Развёртка кончается на измеренном диапазоне '
                 'плюс запас %.0f м; ДАЛЬШЕ МЕШ НЕ ПРОДОЛЖАЕТСЯ (сравнение с полным '
                 'диапазоном — sweep_extra_m=None). Плоскость floor для привязки меша '
                 'НЕ используется (см. заметку про floor). Шейка и подошва лидаром не '
                 'видны и НЕ дорисовываются (source.web/foot = not observed): в геометрии '
                 'нет ни 180 мм высоты, ни 150 мм подошвы, ни подуклонки'
                 % (RAIL_BIN_Y_M, RAIL_SWEEP_STEP_M, RAIL_SWEEP_EXTRA_M))
    notes.append('ширина головки на 13 мм ниже верха (74.59 мм по ГОСТ Р 51685-2022) '
                 'используется только как СПРАВОЧНАЯ норма при измерении колеи — в '
                 'геометрию меша она не входит')
    notes.append('сенсор — начало облака (0,0,0), sensor_height_above_head_m = -top_z '
                 '(допущение, отдельно не проверено)')
    notes.append('колея по внутренним граням измерена на 13 мм ниже верха; колея '
                 '«ось-ось» больше на ширину головки')
    notes.append('колея «ось-ось» %.1f мм — по линиям нитей, каждая проведена по центрам '
                 'поперечных сечений головки (середина p2/p98 точек полосы); '
                 'нормативная связь колея_внутр + 74.59 = %.1f мм, норма 1594.6 мм'
                 % (1000.0 * gauge_axis_m, 1000.0 * gauge_inner_m + R65_HEAD_B_MM))
    notes.append('сумма внутренних смещений граней от осей нитей (колея «ось-ось» минус '
                 'колея по граням) = %.1f мм против нормативных 74.59 мм: наружная '
                 'сторона головки в тени, а подуклонка поднимает наружную кромку '
                 'площадки на 3.7 мм, поэтому оценка центра головки смещена наружу на '
                 'несколько мм — отсюда и остаточный сдвиг колеи «ось-ось»'
                 % (1000.0 * (gauge_axis_m - gauge_inner_m)))
    floor_vs_shelf = None
    if prof is not None:
        y_probe = float(prof['y_ref'])
        plane_z = float(a * (s_axis * y_probe + i_axis) + b * y_probe + c)
        floor_vs_shelf = {'y_probe': y_probe, 'plane_z': plane_z,
                          'shelf_z': float(prof['shelf0']),
                          'delta_m': plane_z - float(prof['shelf0'])}
        notes.append('ВНИМАНИЕ — ключ floor (плоскость) НЕНАДЁЖЕН и НЕ должен '
                     'использоваться для привязки геометрии. В этом кадре плоскость '
                     'проходит на %.3f м %s измеренной полки (в центре окна полки '
                     'y = %.1f м: плоскость %.3f м, полка %.3f м). Подгонка при этом '
                     'не сломана: floor.a — это НАКЛОН (м/м), а не уровень, его легко '
                     'принять за уровень, читая floor = {a: %.4f, b: %.4f, c: %.3f}. '
                     'Плоскость подгоняется по нижней огибающей всей полосы |x| <= 5 м, '
                     'а поверхность ступенчатая (полка, лоток, обочины), поэтому она '
                     'проходит «между» ступенями и от кадра к кадру одной сцены гуляет '
                     'на дециметры. Меш рельса привязан к ИЗМЕРЕННОЙ полке '
                     '(meta.floor_profile), меш пола — к измеренному профилю; ключ floor '
                     'оставлен как есть только для других потребителей'
                     % (abs(floor_vs_shelf['delta_m']),
                        'НИЖЕ' if floor_vs_shelf['delta_m'] < 0 else 'ВЫШЕ',
                        y_probe, plane_z, floor_vs_shelf['shelf_z'], a, b, c))

    meta = {
        'gauge_inner_mm': float(1000.0 * gauge_inner_m),
        'gauge_axis_mm': float(1000.0 * gauge_axis_m),
        'head_width_mm': float(np.mean(head_widths)),
        'sensor_height_above_head_m': float(-z_head),
        'notes': notes,
    }
    meta['rail_mesh'] = {
        'source': ('измеренное сечение (медиана по плотным бинам), развёрнутое вдоль '
                   'оси' if not draw_gost_profile else
                   'ГОСТ Р65 (draw_gost_profile=True) — только для сравнения'),
        'mode': ('gost_profile' if draw_gost_profile
                 else ('only_observed (куски по бинам)' if only_observed
                       else 'sweep (непрерывная развёртка измерения)')),
        'axis_model': ('gost_profile' if draw_gost_profile
                       else ('only_observed (bins, gaps kept)' if only_observed
                             else ('polyline (локальные сечения вдоль пути)'
                                   if axis_polyline else 'straight (x = s*y + i)'))),
        'axis_polyline': bool(axis_polyline and not only_observed
                              and not draw_gost_profile),
        'mesh_kind': ('rail body: gost R65 (head sides, fillets, web, foot) под '
                      'ИЗМЕРЕННОЙ коронкой головки (верх = измеренный)'
                      if (axis_polyline and not only_observed and not draw_gost_profile)
                      else ('gost R65 extrusion' if draw_gost_profile
                            else 'measured head plate (5 nodes)')),
        'mesh_body': bool(axis_polyline and not only_observed
                          and not draw_gost_profile),
        'nodes_note': ('rails[side].nodes = [[y, x, z_top], ...] (м, по y) — узлы оси '
                       'развёртки, включая продолжение; для маски точек. mesh — ТЕЛО '
                       'рельса, mesh_measured — полоска; вершины тела группировать по '
                       'polyline.section_nodes, НЕ по nodes_per_bin'),
        # --- ПРО СТОРОНЫ (важно потребителю, ключи left/right НЕ переименовываются)
        'side_note': ('ключи left/right — по системе СЕНСОРА: left = x<0, right = x>0 '
                      '(взгляд вдоль +Y, Z вверх). Во вьюере +X рисуется ВЛЕВО, '
                      'поэтому на экране left виден СПРАВА, right — СЛЕВА; '
                      'axis_x_sign у каждой нити: +1 = нить на стороне x>0 оси кадра, '
                      '-1 = на стороне x<0 (это и есть «физическая» сторона, не '
                      'экранная)'),
        'side_note_short': ('left = x<0 (на экране +X влево, то есть left виден справа; '
                            'right = x>0 — слева)'),
        'mesh_measured_kind': ('измеренная верхняя площадка головки: 5 узлов, '
                               'медиана по плотным бинам (depth_select_m = 5..40 мм '
                               'под верхом), БЕЗ шейки и подошвы'),
        'bin_y_m': RAIL_BIN_Y_M, 'nodes_per_bin': RAIL_NODES,   # только для полоски
        'nodes_per_bin_note': 'nodes_per_bin = 5 — только для полоски (mesh_measured)',
        'min_points_per_bin': RAIL_MIN_PTS_BIN,
        'sweep_step_m': RAIL_SWEEP_STEP_M,
        'sweep_extra_m': (None if sweep_extra_m is None else float(sweep_extra_m)),
        'sweep_drift_limits_m': {'z': RAIL_DRIFT_LIMIT_M, 'x': RAIL_X_DRIFT_LIMIT_M},
        'sweep_anchor': ('measured floor shelf (meta.floor_profile); floor plane not used'
                         if prof is not None else
                         'top line x = m*y + c (полку измерить не удалось)'),
        'sweep_y_limits_m': [RAIL_SWEEP_Y_FAR_M, RAIL_SWEEP_Y_NEAR_M],
        'sweep_y_limits_note': ('эти пределы работают только при sweep_extra_m=None '
                                '(режим сравнения, весь диапазон облака)'),
        'polyline': {
            'bin_m': POLY_BIN_M, 'window_m': POLY_WIN_M, 'u_window_m': POLY_U_WIN_M,
            'smooth_bins': POLY_SMOOTH_BINS, 'mad_k': POLY_MAD_K,
            'mad_floor_m': POLY_MAD_MIN_M, 'max_miss_bins': POLY_MAX_MISS,
            'jump_m': POLY_JUMP_M,
            'tail_nodes_for_tangent': POLY_TAIL_N,
            'note': ('ось — центры локальных сечений бинов (окно от предсказания по '
                     'предыдущим узлам), сечение перпендикулярно локальной касательной, '
                     'за пределами измеренного — продолжение по последней касательной'),
        },
        'axis_to_rail': {
            'what': ('расстояние от оси меша до ближайших точек рельса: полоса головки '
                     '%.2f..%.2f м над полкой, |u| <= %.2f м, окно ±%.1f м по Y'
                     % (POLY_BAND_LO_M, POLY_BAND_HI_M, POLY_U_LIM_M, POLY_METRIC_WIN_M)),
            'criterion': {'median_m': 0.05, 'p95_m': 0.15,
                          'applies_to': 'measured bins (внутри измеренного)'},
        },
        'u_select_m': RAIL_U_SEL_M, 'depth_select_m': [RAIL_DEPTH_UP_M, RAIL_DEPTH_DOWN_M],
        'pair_note': ('рельсы — ЖЁСТКАЯ ПАРА: центр u_c = среднее измеренных нитей, '
                      'одна колея на кадр G (медиана измеренных расстояний, зажата в '
                      '1580..1610 мм), нити = u_c ± G/2, поэтому колея по узлам '
                      'постоянна и внутри данных, и в продолжении; продолжение — общая '
                      'центральная линия по своей кривизне. Высота каждой нити — своя '
                      'измеренная коронка (подуклонка/возвышение не выравниваются)'),
        'rails': {r['side']: {
            'n_bins_span': r['mstat'].get('n_bins_span'),
            'n_bins_with_points': r['mstat'].get('n_bins_ok'),
            'n_ribbons': r['mstat'].get('ribbons'),
            'gaps_y': r['mstat'].get('gaps'),
            'y_range': [r['y_lo'], r['y_hi']],
            'extent_y': r.get('extent_y'),
            # СТОРОНА НИТИ ЧИСЛОМ (см. meta.rail_mesh.side_note): +1 = ось нити на
            # стороне x>0 от оси кадра, -1 = на стороне x<0. Ключ 'left'/'right' — по
            # СЕНСОРУ (left = x<0), а на экране (+X влево) left виден справа, поэтому
            # сторона даётся ещё и знаком, а не только именем.
            'axis_x_sign': (None if not np.isfinite(r['s'] * y_mid + r['i'])
                            else int(np.sign(float(r['s'] * y_mid + r['i'])
                                             - float(s_axis * y_mid + i_axis)))),
            # знак продольного наклона оси нити dx/dy (+1 — x растёт с y)
            'axis_slope_sign': (None if not np.isfinite(r['s'])
                                else int(np.sign(float(r['s'])))),
            'section_source': r['mstat'].get('y_dense'),
            'beyond_measured': r.get('beyond_measured'),
            'mesh_body': bool(r.get('poly') is not None),
            'mesh_built': bool(r.get('mesh') and r['mesh'][0]),
            # ЧЕМ ПОСТРОЕН МЕШ: pair_nodes — нить на кадре НЕ измерена (bsec пуст),
            # но узлы есть по плану пары, и тело построено по ним (сечение — пары
            # или номинал Р65); None — обычный путь (измеренное сечение)
            'mesh_source': r.get('mesh_source'),
            # ПОПЕРЕЧНЫЙ УХОД ПРОДОЛЖЕНИЯ ПО КАЖДОЙ НИТИ: своя касательная хвоста,
            # максимум ухода, сколько узлов срезано и с какой длины (см. POLY_REACH_
            # DRIFT_M/POLY_REACH_DRIFT_RATE)
            'drift_trim': (None if r.get('pair') is None
                           else (r['pair'].get('drift_trim') or {}).get(r['side'])),
            # --- ЖЁСТКАЯ ПАРА (диагностика)
            'pair_gauge_median_mm': (None if r.get('pair') is None else
                                     float(1000.0 * r['pair']['gauge_m'])),
            'pair_gauge_spread_mm': (None if r.get('pair') is None else
                                     float(r['pair']['gauge_spread_all_mm'])),
            'pair_gauge_spread_measured_mm': (
                None if r.get('pair') is None
                or r['pair']['gauge_spread_measured_mm'] is None
                else float(r['pair']['gauge_spread_measured_mm'])),
            'pair_gauge_raw_median_mm': (
                None if r.get('pair') is None or r['pair']['gauge_raw_m'] is None
                else float(1000.0 * r['pair']['gauge_raw_m'])),
            'pair_gauge_clamped_mm': (
                None if r.get('pair') is None or r['pair']['gauge_clamped_mm'] is None
                else float(r['pair']['gauge_clamped_mm'])),
            'pair_center_source': (None if r.get('pair') is None
                                   else r['pair']['center_source']),
            # ВИД ПЛАНА ПАРЫ (признак доверия): both_rails — по двум измеренным нитям,
            # one_rail — вторая нить на кадре не измерена (колея номинальная),
            # nominal_axis — нитей нет, узлы по номинальной оси кадра
            'pair_center_kind': (None if r.get('pair') is None
                                 else r['pair']['center_kind']),
            # ИСТОЧНИК КОЛЕИ: measured — медиана измеренных расстояний по общей полосе,
            # nominal — измерить было нечем (нет второй нити / нет общей полосы)
            'pair_gauge_source': (None if r.get('pair') is None
                                  else r['pair'].get('gauge_source')),
            # --- СЧЁТЧИКИ ЦЕНТРА (числами, для отчёта и проверок): сколько узлов
            # взято по среднему обеих нитей / по НАДЁЖНОЙ нити / по одной нити, и
            # сколько раз надёжной оказалась ИМЕННО ЭТА нить
            'pair_center_both': (None if r.get('pair') is None
                                 else int(r['pair']['n_center_both'])),
            'pair_center_reliable': (None if r.get('pair') is None
                                     else int(r['pair']['n_center_reliable'])),
            'pair_center_single': (None if r.get('pair') is None
                                   else int(r['pair']['n_center_single'])),
            'pair_center_runs': (None if r.get('pair') is None
                                 else int(r['pair']['n_center_runs'])),
            'pair_center_reliable_this': (None if r.get('pair') is None else
                                          (int(r['pair']['n_reliable_left'])
                                           if r['side'] == 'left' else
                                           int(r['pair']['n_center_reliable'])
                                           - int(r['pair']['n_reliable_left']))),
            'pair_center_off_threshold_mm': (None if r.get('pair') is None
                                             else float(r['pair']['gauge_off_threshold_mm'])),
            # массив узлов оси (для маски точек у потребителей: без него они
            # группировали вершины меша по nodes_per_bin и после перехода на тело
            # (20 узлов в сечении) получали мусор)
            'nodes': r.get('nodes'),
            'nodes_source': r.get('nodes_source'),
            'pair_nominal_nodes': r.get('pair_nominal_nodes'),
            # ПРОИСХОЖДЕНИЕ КАЖДОГО УЗЛА: measured | sparse | model (по порядку nodes)
            'node_origin': (None if r.get('nodes') is None else
                            (r['poly'].get('node_origin') if r.get('poly') else
                             (['measured'] * len(r['nodes'])
                              if r.get('nodes_source') == 'measured_only'
                              else ['model'] * len(r['nodes'])))),
            'top_line_off_max_m': (None if r.get('poly') is None
                                   else r['poly'].get('top_line_off_max_m')),
            'reach_len_m': (None if r.get('nodes') is None else
                            {k: float(POLY_BIN_M * sum(
                                1 for v in (r['poly'].get('node_origin') if r.get('poly')
                                            else (['measured'] * len(r['nodes'])
                                                  if r.get('nodes_source') == 'measured_only'
                                                  else ['model'] * len(r['nodes'])))
                                if v == k))
                             for k in ('measured', 'sparse', 'model')}),
            'nodes_note': ('[y, x_оси, z_верх] — ось нити для маски точек; nodes есть '
                           'на ВСЕХ кадрах: measured_pair/measured_one_rail — по '
                           'измеренной оси, pair_by_other_rail/nominal_axis/'
                           'straight_nominal — НЕ измерение (ось по плану пары: G от '
                           'парной нити или номинальная ось кадра), '
                           'pair_nominal_nodes = сколько узлов таких'),
            'axis_model': ('polyline' if r.get('poly') is not None else
                           ('only_observed' if only_observed else 'straight')),
            # ПРАВИЛО ОСТАНОВКИ ПО ДАННЫМ: подтверждённый данными конец ленты на
            # каждом конце, сколько отрезано и почему (см. _rail_reach). far — дальний
            # конец (y убывает), near — ближний (за сенсором). Данные по ЭТОЙ нити в
            # by_side, итог пары (одинаковый для обеих нитей) — в полях верхнего уровня.
            'reach': (None if r.get('poly') is None else r['poly'].get('reach')),
            'ribbon_inside_points': (None if r.get('poly') is None
                                     else r['poly'].get('ribbon_inside_points')),
            'reach_note': ('confirmed_end_y_m — конец ленты, подтверждённый данными '
                           '(>=%d точки головки в окне %.0f м, |Δx|<=%.2f м, Δz от '
                           'своей линии верха %.0f..+%.0f мм); cut_m — насколько '
                           'подтверждённый конец КОРОЧЕ продолжения по дуге;'
                           'points_cut — отрезанные точки головки в окне за концом;'
                           'stop_reason = no_head_points | ribbon_body_points (>=%d '
                           'точек ВНУТРИ объёма тела) | cloud_limit; ribbon_inside_'
                           'points — максимум точек внутри тела в проверенных окнах'
                           % (POLY_REACH_MIN_PTS, POLY_REACH_WIN_M, POLY_REACH_DX_M,
                              1000 * POLY_REACH_DZ_M[0], 1000 * POLY_REACH_DZ_M[1],
                              POLY_REACH_RIB_MIN)),
            'axis_to_rail_m': (None if r.get('poly') is None else r['poly']['metric']),
            'polyline': (None if r.get('poly') is None else {
                'n_nodes': int(len(r['bsec']['grid_y'])),
                'n_nodes_total': int(r['poly']['n_nodes_total']),
                'section_nodes': int(r['poly']['poly_section_nodes']),
                'n_nodes_clamped_z': int(r['poly']['n_clamped_nodes']),
                'node_stats': r['poly']['node_stats'],
                'foot_drop_m': r['poly']['foot_drop_m'],
                'section_height_m': r['poly']['section_height_m'],
                'foot_below_shelf_min_m': r['poly']['foot_below_shelf_min_m'],
                'foot_below_shelf_max_m': r['poly']['foot_below_shelf_max_m'],
                'foot_below_shelf_out': r['poly']['foot_out_of_tol'],
                'foot_tol_m': [POLY_FOOT_TOL_M[0], POLY_FOOT_TOL_M[1]],
                'n_nodes_measured': int(r['bsec']['bins'].size),
                'n_nodes_bridged': int(r['poly']['n_bridging']),
                'n_nodes_rejected_mad': r['poly']['n_rejected'],
                'n_nodes_rejected_height': r['poly']['n_rejected_height'],
                'n_nodes_clamped_height': r['poly']['n_clamped_height'],
                'top_corridor_m': r['poly']['top_corridor_m'],
                'corridor_top_lo_m': r['poly'].get('corridor_top_lo_m'),
                'body_height_m': r['poly'].get('body_height_m'),
                'h_head_above_shelf_m': (None if r['poly']['sweep'].get('head_h_m') is None
                                         else float(r['poly']['sweep']['head_h_m'])),
                'h_head_clamped': bool(r['poly']['h_head_clamped']),
                'dense_len_m': r['poly']['dense_len_m'],
                'axis_max_step_m': r['poly']['axis_max_step_m'],
                'n_steps_over_jump_m': r['poly']['n_jumps'],
                'smooth_resid_med_m': (r['poly']['smooth_resid_m'][0]
                                       if r['poly']['smooth_resid_m'] else None),
                'smooth_resid_p95_m': (r['poly']['smooth_resid_m'][1]
                                       if r['poly']['smooth_resid_m'] else None),
                'tangent_deg_far_near': r['poly']['tangent_deg'],
                'kappa_far_1_m': r['poly']['sweep']['kappa_far_1_m'],
                'kappa_near_1_m': r['poly']['sweep']['kappa_near_1_m'],
                'gaps_y': (None if r.get('bsec') is None
                           else [[float(a), float(b)] for a, b in
                                 [g for g in r['bsec'].get('gaps', [])]]),
            }),
            'sweep_extra_used_m': (None if not r.get('sweep')
                                   else r['sweep'].get('extra_m')),
            'sweep_extra_far_m': (None if not r.get('sweep')
                                  else r['sweep'].get('extra_far_m')),
            'sweep_extra_near_m': (None if not r.get('sweep')
                                   else r['sweep'].get('extra_near_m')),
            'sweep_extra_cap_h_m': (None if not r.get('sweep')
                                    else r['sweep'].get('cap_h_m')),
            'sweep_extra_cap_x_m': (None if not r.get('sweep')
                                    else r['sweep'].get('cap_x_m')),
            'head_h_above_shelf_m': (None if not r.get('sweep')
                                     else r['sweep'].get('head_h_m')),
            'shelf_ref': (None if not r.get('sweep') or not r['sweep'].get('shelf_ref')
                          else {k: float(v) for k, v in r['sweep']['shelf_ref'].items()}),
            'z_drift_far_m': (None if not r.get('sweep')
                              else r['sweep'].get('z_drift_far_m')),
            'z_drift_beyond_m': (None if not r.get('sweep')
                                 else r['sweep'].get('z_drift_extra_m')),
            'x_drift_far_m': (None if not r.get('sweep')
                              else r['sweep'].get('x_drift_far_m')),
            'n_sections': (int(len(r['mesh'][0]) // (r['poly']['poly_section_nodes']
                                                     if r.get('poly') else RAIL_NODES))
                           if r['mesh'][0] and not draw_gost_profile else None),
            'section': (None if r.get('sec') is None else {
                'width_m': r['sec']['width_m'],
                'u_off_m': r['sec']['u_off_m'],
                'vz_nodes_m': r['sec']['vz_nodes_m'],
                'depth_m': r['sec']['depth_m'],
                'n_bins_used': r['sec']['n_bins_used'],
            }),
            'n_verts': len(r['mesh'][0]), 'n_faces': len(r['mesh'][1]),
            'n_points_used': r['mstat'].get('n_pts_sel'),
            'line': {'s': float(r['s']), 'i': float(r['i'])},
            'top_line': {'s': float(r['m_top']), 'i': float(r['c_top'])},
        } for r in rails},
        # ПЛАН ПАРЫ НА КАДР (признак доверия ко ВСЕМУ кадру): both_rails — обе нити
        # измерены; one_rail — измерена одна (колея номинальная, вторая нить — по
        # оси плана); nominal_axis — нитей нет вовсе (ось номинальная). Узлы, взятые
        # НЕ из измерения ЭТОЙ нити, посчитаны в nodes_nominal_total.
        'pair_center_kinds': {r['side']: (None if r.get('pair') is None
                                          else r['pair']['center_kind'])
                              for r in rails},
        'nodes_nominal_total': int(sum(r.get('pair_nominal_nodes') or 0 for r in rails)),
        'pair_center_source_frame': (None if pair is None else pair['center_source']),
        'nodes_source': {r['side']: r.get('nodes_source') for r in rails},
        'nodes_nominal_note': ('nodes есть на всех кадрах и всегда сортированы по y '
                               '(x = ось нити, z = верх). Где nodes_source = '
                               'measured_pair/measured_one_rail — ось ИЗМЕРЕНА (меш по '
                               'ней); pair_by_other_rail — нить на кадре не найдена, '
                               'ось = ось парной нити ∓ номинальная колея, z взят с '
                               'парной нити; nominal_axis/straight_nominal — номинальная '
                               'ось кадра. Для номинальных узлов доверие понижать, '
                               'pair_nominal_nodes/nodes_nominal_total = их число'),
    }
    if prof is not None:
        meta['floor_profile'] = {
            'system': 'u от оси пары рельсов (плюс к правой нити), z абсолютная, м',
            'window_y': [prof['y_lo'], prof['y_hi']],
            'shelf_z': prof['shelf0'],
            'shelf_slope_u': prof['shelf_slope_u'],
            'shelf_slope_y': prof['shelf_slope_y'],
            'trough': (None if not prof['has_trough'] else {
                'u_left': prof['u_l'], 'u_right': prof['u_r'],
                'width_m': prof['width'], 'center_u': prof['center'],
                'bottom_z': prof['bottom'], 'depth_m': prof['depth'],
                'wall_run_m': prof['wall_run_m'], 'n_points_bottom': prof['n_bottom'],
                'width_bands_m': prof['width_bands'],
                'center_bands_m': prof['center_bands'],
                'depth_bands_m': prof['depth_bands'],
            }),
            'n_floor_points': prof['n_floor_pts'],
            'source': 'measured (ближняя зона; вне неё профиль протянут без изменений)',
        }
    if floor_vs_shelf is not None:
        meta['floor_plane_reliability'] = {
            'reliable': False,
            'use_for_anchoring': False,
            'note': ('плоскость floor подогнана по нижней огибающей всей полосы |x| <= 5 м '
                     'и проходит «между» ступенями (полка/лоток/обочины); floor.a — наклон '
                     '(м/м), не уровень. Меш рельса привязан к измеренной полке '
                     '(floor_profile)'),
            'probe_y_m': floor_vs_shelf['y_probe'],
            'plane_z_m': floor_vs_shelf['plane_z'],
            'shelf_z_m': floor_vs_shelf['shelf_z'],
            'plane_minus_shelf_m': floor_vs_shelf['delta_m'],
        }
    return {
        'frame': int(frame_index),
        'axis': {'s': float(s_axis), 'i': float(i_axis)},
        'floor': {'a': float(a), 'b': float(b), 'c': float(c)},
        'rails': rails_out,
        'sleepers': {'instances': instances, 'spacing_m': float(SLEEPER_SPACING_M),
                     'source': sleepers_src},
        'floor_mesh': {'vertices': [[float(v) for v in row] for row in f_verts],
                       'faces': [[int(t0), int(t1), int(t2)] for (t0, t1, t2) in f_faces],
                       'source': floor_src},
        'meta': meta,
    }