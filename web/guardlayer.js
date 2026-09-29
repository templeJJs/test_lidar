// Слой guard-тракта (pavel/guard/run.py) для основного вьюера web_viewer.py.
//
// Модуль самодостаточный: createGuardLayer({ scene }) грузит маршрут /guard
// (его отдаёт web_viewer.py из ПРЕДРАССЧИТАННЫХ файлов pavel/guard_web/<запись>/
// frame_NNNN.json -- их пишет `cd pavel && python -m guard.export_web <запись>`;
// analyze ~0.4 с/кадр, на лету сервер его не считает) и рисует поверх облака:
//
//   * коридор габарита -- каркас сечений по бинам: зелёный, если обе стены
//     бина измерены ('ok'), жёлтый, если измерена одна сторона ('gauge'),
//     серый -- дыра по предсказанию ('hole');
//   * подсветка РЕЖИМОВ ведения оси (bins.mode из export_web, блок 1в):
//     при включённой подсветке (по умолчанию вкл, тумблер в панели) ось и
//     коридор красятся по режиму бина: rails -- синий, center -- зелёный,
//     lead -- оранжевый, hold -- серый. Старый JSON без mode -- прежняя
//     раскраска по статусу, ничего не ломается;
//   * стены -- продольные линии по измеренным кромкам (в бине 'gauge' сторона
//     без замера -- кламп габарита, её НЕ рисуем: это была бы ложная стена);
//   * ось пути -- линия ведомой оси бинов плюс далёкая ось (axis_far), где она
//     ведёт ('seed'/'straight'/'wall');
//   * препятствия -- каркасные боксы находок блока 3. Уровень находки
//     (obstacles[i].track_confirmed из export_web, блок 3.5 tracker.py):
//     подтверждённая треком в габарите -- красный бокс, метка «⚠ N м», луч
//     и строка тревоги; однокадровая НЕподтверждённая в габарите -- янтарный
//     приглушённый бокс, метка «? ~N м», янтарные точки, БЕЗ красной тревоги
//     (транзиент живёт ровно 1 кадр, настоящий объект -- каждый, замер в
//     tracker.py); подтверждённая треком, но ЗА пределами подтверждённого
//     коридора (obstacles[i].beyond_reach: бин не измерен или дальше reach)
//     -- янтарный бокс полной яркости, метка «~N м (за reach)», янтарная
//     строка: трекер подтверждает чисто по счётчику кадров и обходит
//     правило reach (export_web.py), поэтому это раннее предупреждение, а
//     НЕ красная тревога; вне габарита -- оранжевый, как раньше. Старый JSON
//     без полей track_*/beyond_reach -- прежнее поведение (тревога по
//     in_gauge). Если в JSON
//     есть obstacles[i].point_idx (индексы точек находки в кадре /frame, см.
//     pavel/guard/export_web.py), сами ТОЧКИ препятствия красятся поверх
//     палитры облака: подтверждённые -- ярко-красный (HIT_RGB из hitpoints.js),
//     однокадровые -- янтарь (HIT_RGB_UNC ниже); при смене кадра/выключении
//     слоя цвета восстанавливаются. Старые JSON без point_idx -- только
//     боксы, как раньше.
//   * метка дальности над каждым боксом находки -- текстовый спрайт
//     (canvas-текстура на THREE.Sprite), цвет по уровню находки; плюс
//     луч-указатель от лидара (0,0,0) до центра бокса -- тонкий пунктир
//     (LineDashedMaterial). Спрайты НЕ пересоздаются на кадр: кэш по
//     (текст, цвет), см. spriteCache;
//   * строка тревоги в панели (#guardAlert): по ПОДТВЕРЖДЁННОЙ треком находке
//     в габарите В пределах reach -- яркая «⚠ ПРЕПЯТСТВИЕ: N м (lat ±X.X)»
//     по ближайшей; подтверждённая треком, но за reach -- янтарная «дальняя
//     находка за reach: N м»; только однокадровые -- нейтральная янтарная
//     «неподтверждённая находка: N м»; иначе нежная «свободно до reach м»;
//     слой выключен -- скрыта.
//
// Перформанс: пересборка на кадр не создаёт НИЧЕГО -- постоянная корневая
// группа и пулы буферов с DynamicDrawUsage; заполнение префиксом массива +
// setDrawRange (тот же приём, что у облака в app.js:394-407). Ёмкости пулов
// -- из норм тракта: MAX_RANGE=130 м, BIN=5 м (walls.py:38-39) -> бинов <= 26;
// крышки на число находок нет -- пулы боксов и лучей растут удвоением по
// требованию. Спрайты меток кэшированы (текст «⚠ N м», N -- целые метры
// 0..130 -> множество вариантов ограничено нормативной дальностью тракта,
// кэш не растёт бесконечно; паттерн labellayer.js:513-517).
//
// Сеть: /guard качается через ОБЩИЙ кэш промисов window.__guardPromises по
// (bag, frame) -- тот же промис использует topview.js, второй запрос и
// второй JSON.parse на кадр не уходят.
//
// Точки и их цвета слой берёт через accessor cloud() (posArr/colArr кадра в
// app.js, тот же приём, что у safetylayer.markPoints) и после перекраски
// зовёт touchCloud() (colAttr.needsUpdate). Совпадение системы индексов
// проверяется по frame_npts (hitpoints.hitIndices).
//
// Координаты: JSON приходит в системе guard (fwd=-y, lat=x, up=z -- система
// лидара кадра, см. pavel/guard/geom.py). Вьюер рисует в осях лидара
// (X поперёк, Y вдоль, вперёд -- -Y, Z вверх, см. app.js), поэтому перевод
// один на весь файл: viewer = (lat, -fwd, up) -- функция P() ниже.
//
// Ловушки, пройденные в safetylayer.js/track3d.js, учтены так же:
//   * пересборка -- только по приходу нового кадра; пулы переиспользуются,
//     drawRange обнуляется на пустом кадре (хвосты прошлого кадра не видны,
//     та же ловушка, что app.js:3719);
//   * загрузки сериализованы и всегда применяют ПОСЛЕДНИЙ запрошенный кадр:
//     ответ старого кадра не перезапишет новый (та же гонка, что в /objects);
//   * 404 (кадр не посчитан) -- не ошибка: слой пуст, в панели подсказка
//     команды предрасчёта. Ошибки из общего кэша промисов выкидываются:
//     после пересчёта export_web повторный запрос уйдёт в сеть.

import * as THREE from 'three';

import { paintCloud, restoreCloud } from './hitpoints.js';

const ORDER = 8;                           // поверх облака точек (renderOrder 1)
const COL_OK = 0x35d07f;                   // бин измерен с двух сторон
const COL_GAUGE = 0xffd166;                // измерена одна сторона
const COL_HOLE = 0x8a97b0;                 // дыра, коридор по предсказанию
const COL_WALL = 0x8fc4ff;                 // измеренные кромки стен
const COL_AXIS = 0x59d8ff;                 // ведомая ось бинов
const COL_AXIS_FAR = 0xd98cff;             // далёкая ось (ведение стеной)
const COL_OBST = 0xff4d4d;                 // препятствие в габарите, подтверждено
const COL_OBST_OUT = 0xffa04d;             // препятствие вне габарита
const COL_OBST_UNC = 0xffc233;             // однокадровая неподтверждённая в
                                           // габарите -- янтарь, НЕ тревога
const COL_OBST_FAR = 0xffc233;             // подтверждённая треком, но ЗА
                                           // reach (beyond_reach) -- тот же
                                           // янтарь без приглушения: трек
                                           // держится, раннее предупреждение
const UNC_DIM = 0.6;                       // приглушение янтарного бокса (доля
                                           // яркости): уровень ниже тревоги
// Янтарь для ТОЧЕК однокадровых находок поверх палитры кадра (подтверждённые
// красятся HIT_RGB из hitpoints.js).
const HIT_RGB_UNC = [255, 194, 51];
// Режимы ведения оси (блок 1в, wall_rules): подсветка оси и коридора.
export const MODE_COLORS = {
  rails: 0x4d8dff,                         // ось от рельсов -- синий
  center: 0x35d07f,                        // середина двух стен -- зелёный
  lead: 0xffa04d,                          // ведение по стене -- оранжевый
  hold: 0x8a97b0,                          // продолжение формы в дыре -- серый
};
const CORRIDOR_Z_LO = 0.05;                // низ сечения коридора над полом, м
const CORRIDOR_Z_HI = 2.30;                // верх сечения (визуальный), м
const WALL_Z_HI = 3.00;                    // высота вертикалей стен, м
// Габарит ПОЕЗДА (проверка in_gauge, obstacles.py): коридор рисуется шире
// (кламп ±GAUGE_HALF=1.50 от walls.py), а препятствия проверяются по поезду.
// Чтобы нарисованное совпадало с логикой, габарит поезда рисуем отдельными
// линиями поверх коридора.
const TRAIN_HALF = 1.05;                   // GAUGE_HALF_CHECK (walls.py:68)
const TRAIN_Z_LO = 0.10;                   // GAUGE_H_LO (walls.py)
const TRAIN_Z_HI = 3.00;                   // GAUGE_H_HI (walls.py)
const COL_TRAIN = 0xff66ff;                // габарит поезда -- маджента
const OBST_DEPTH_M = 2.0;                  // глубина бокса препятствия, м
const LABEL_LIFT_M = 0.6;                  // подъём метки над верхом бокса, м
// Ёмкости пулов -- из норм тракта (walls.py:38-39: BIN=5.0, MAX_RANGE=130.0):
const BIN_CAP = 26;                        // бинов коридора <= MAX_RANGE/BIN
const OBST_CAP0 = 8;                       // стартовая ёмкость пулов находок,
                                           // рост удвоением (крышки нет)

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
const $ = (id) => document.getElementById(id);

// guard (fwd, lat, up) -> вьюер (x, y, z). up здесь -- абсолютная z лидара.
const P = (fwd, lat, up) => [lat, -fwd, up];

// Уровень находки: 'hit' -- в габарите и подтверждена треком в пределах reach
// (или старый JSON без полей track_*/beyond_reach -- прежнее поведение,
// тревога по in_gauge), 'far' -- подтверждена треком, но за пределами
// подтверждённого коридора (beyond_reach === true: трекер подтверждает по
// счётчику кадров и обходит правило reach, export_web.py) -- янтарное раннее
// предупреждение, НЕ красная тревога, 'unc' -- в габарите, но однокадровая
// неподтверждённая (track_confirmed === false), 'out' -- вне габарита.
export function obstLevel(h) {
  if (!h.in_gauge) return 'out';
  if (h.track_confirmed === false) return 'unc';
  if (h.beyond_reach === true) return 'far';
  return 'hit';
}

// Общий кэш загрузок /guard между слоями: guardlayer и topview ходят на один
// URL одного кадра -- один fetch + один JSON.parse на (bag, frame). Промис
// разделяется: оба слоя читают payload только на чтение. Ошибки (в т.ч. 404
// «нет предрасчёта») не кэшируются: после пересчёта export_web повторный
// запрос должен уйти в сеть. Ёмкость ограничена FIFO, чтобы долгое листание
// не копило распарсенные JSON.
const guardPromises = (typeof window !== 'undefined')
  ? (window.__guardPromises = window.__guardPromises || new Map())
  : new Map();
const GUARD_CACHE_MAX = 64;                // кадров; запрос микроскопичен
                                           // (~10 КБ), 64 -- с запасом над
                                           // радиусом prefetch frameq (±2)

export function fetchGuardShared(bag, idx) {
  const key = `${bag}|${idx}`;
  let p = guardPromises.get(key);
  if (!p) {
    const url = `/guard?bag=${encodeURIComponent(bag)}&frame=${idx}`;
    p = (async () => {
      const res = await fetch(url, { cache: 'no-store' });
      if (!res.ok) {
        let detail = '';
        try { detail = (await res.json()).error || ''; } catch (e) { /* не JSON */ }
        const err = new Error(`GET ${url} -> ${res.status}`);
        err.status = res.status;
        err.detail = detail;
        throw err;
      }
      return res.json();
    })();
    p.catch(() => guardPromises.delete(key));   // ошибки не кэшируем
    guardPromises.set(key, p);
    if (guardPromises.size > GUARD_CACHE_MAX) {
      guardPromises.delete(guardPromises.keys().next().value);
    }
  }
  return p;
}

function disposeObject(obj) {
  obj.traverse((node) => {
    if (node.geometry) node.geometry.dispose();
    const mat = node.material;
    const disposeMat = (m) => {
      if (m.map) m.map.dispose();   // canvas-текстуры меток -- тоже память
      m.dispose();
    };
    if (Array.isArray(mat)) mat.forEach(disposeMat);
    else if (mat) disposeMat(mat);
  });
}

// Пул сегментов: постоянная LineSegments с буферами ёмкости capVerts и
// DynamicDrawUsage. На кадр ничего не создаётся -- fillSeg перезаписывает
// префикс и ставит setDrawRange. dashed=true добавляет атрибут lineDistance
// (лучи-пунктиры). frustumCulled=false: буфер переиспользуется, сфера охвата
// устарела бы (та же причина, что у облака в app.js).
function makeSegMesh(capVerts, material, dashed) {
  const geom = new THREE.BufferGeometry();
  const mk = (items) => {
    const a = new THREE.Float32BufferAttribute(new Float32Array(capVerts * items), items);
    a.setUsage(THREE.DynamicDrawUsage);
    return a;
  };
  geom.setAttribute('position', mk(3));
  geom.setAttribute('color', mk(3));
  if (dashed) geom.setAttribute('lineDistance', mk(1));
  geom.setDrawRange(0, 0);
  const mesh = new THREE.LineSegments(geom, material);
  mesh.renderOrder = ORDER;
  mesh.frustumCulled = false;
  return mesh;
}

function lineMat(opacity, dashed) {
  const M = dashed ? THREE.LineDashedMaterial : THREE.LineBasicMaterial;
  const opts = {
    vertexColors: true, transparent: true, opacity, depthWrite: false,
  };
  if (dashed) { opts.dashSize = 0.6; opts.gapSize = 0.35; }
  return new M(opts);
}

// Заполнение пула префиксом массивов (pos/col -- по 3 числа на вершину,
// dist -- по 1 для dashed) + drawRange. При нехватке ёмкости -- рост
// удвоением: нормативной крышки на число находок в тракте нет; геометрия при
// этом переиспускается, GPU-буфер перезаливается один раз на рост.
function fillSeg(mesh, pos, col, nVerts, dist) {
  const geom = mesh.geometry;
  let pa = geom.getAttribute('position');
  if (nVerts > pa.count) {
    let cap = Math.max(1, pa.count);
    while (cap < nVerts) cap *= 2;
    geom.dispose();              // освобождает GPU-буферы старых атрибутов
    const dashed = Boolean(geom.getAttribute('lineDistance'));
    const mk = (items) => {
      const a = new THREE.Float32BufferAttribute(new Float32Array(cap * items), items);
      a.setUsage(THREE.DynamicDrawUsage);
      return a;
    };
    pa = mk(3);
    geom.setAttribute('position', pa);
    geom.setAttribute('color', mk(3));
    if (dashed) geom.setAttribute('lineDistance', mk(1));
  }
  pa.array.set(pos, 0);
  pa.needsUpdate = true;
  const ca = geom.getAttribute('color');
  ca.array.set(col, 0);
  ca.needsUpdate = true;
  const da = geom.getAttribute('lineDistance');
  if (da && dist) {
    da.array.set(dist, 0);
    da.needsUpdate = true;
  }
  geom.setDrawRange(0, nVerts);
}

// Текст метки дальности над боксом находки. Экспортирован: node-тест проверяет
// текст напрямую, без canvas.
export function obstLabelText(h) {
  return `⚠ ${Math.round(h.d)} м`;
}

// Текстовый спрайт «⚠ N м»: canvas -> текстура -> Sprite. sizeAttenuation
// выключен -- размер на экране постоянный, читается с любой дистанции;
// depthTest выключен -- не тонет в облаке точек. Без DOM (node-тест) -- null,
// бокс и луч при этом строятся как раньше. Создание дорогое (2D-рендер
// текста + аплоад текстуры), поэтому вызывается только промахом кэша
// spriteCache в createGuardLayer.
function makeLabelSprite(text, colorHex) {
  if (typeof document === 'undefined' || !document.createElement) return null;
  const cv = document.createElement('canvas');
  if (!cv) return null;
  cv.width = 256;
  cv.height = 72;
  const c2 = cv.getContext && cv.getContext('2d');
  if (!c2) return null;
  const col = '#' + colorHex.toString(16).padStart(6, '0');
  c2.fillStyle = 'rgba(8, 10, 16, 0.78)';
  c2.fillRect(0, 0, cv.width, cv.height);
  c2.strokeStyle = col;
  c2.lineWidth = 5;
  c2.strokeRect(2.5, 2.5, cv.width - 5, cv.height - 5);
  c2.font = 'bold 40px "Segoe UI", system-ui, sans-serif';
  c2.textAlign = 'center';
  c2.textBaseline = 'middle';
  c2.fillStyle = col;
  c2.fillText(text, cv.width / 2, cv.height / 2 + 2);
  const tex = new THREE.CanvasTexture(cv);
  tex.minFilter = THREE.LinearFilter;
  const spr = new THREE.Sprite(new THREE.SpriteMaterial({
    map: tex, transparent: true, depthTest: false, depthWrite: false,
    sizeAttenuation: false,
  }));
  const hFrac = 0.055;                       // доля высоты экрана
  spr.scale.set(hFrac * cv.width / cv.height, hFrac, 1);
  spr.renderOrder = ORDER + 1;
  return spr;
}

// Перекраска точек однокадровых (неподтверждённых) находок: тот же приём,
// что paintCloud в hitpoints.js, но янтарным -- уровень ниже тревоги.
// Формат бэкапа общий ({ idx, prev }), восстанавливает restoreCloud.
function paintCloudRgb(colors, indices, rgb) {
  if (!colors || !indices || !indices.length) return null;
  const seen = new Set();
  const idx = [];
  for (const i of indices) {
    if (seen.has(i)) continue;
    seen.add(i);
    idx.push(i);
  }
  if (!idx.length) return null;
  const prev = new Uint8Array(idx.length * 3);
  for (let k = 0; k < idx.length; k++) {
    const j = idx[k] * 3;
    prev[k * 3] = colors[j];
    prev[k * 3 + 1] = colors[j + 1];
    prev[k * 3 + 2] = colors[j + 2];
    colors[j] = rgb[0];
    colors[j + 1] = rgb[1];
    colors[j + 2] = rgb[2];
  }
  return { idx, prev };
}

// Слой. Методы: load(bag, idx) -- кадр из /guard, setVisible(bool), reset(),
// dispose(), data(), isVisible().
// cloud() (необязательный) -- accessor облака кадра { positions, colors,
// count, idx } из app.js; touchCloud() -- пометка colAttr.needsUpdate после
// перекраски. Без них работает всё, кроме подсветки точек препятствий.
export function createGuardLayer({ scene, cloud, touchCloud } = {}) {
  if (!scene) throw new Error('createGuardLayer: нужен scene');

  const state = {
    visible: false,      // по умолчанию слой выключен: панель спокойна
    modeColors: true,    // подсветка режимов ведения оси (bins.mode)
    showUnc: true,       // однокадровые неподтверждённые находки (янтарные)
    frame: null,         // payload последнего применённого кадра
    bag: null,
    idx: -1,
    done: false,         // позиция (bag, idx) обработана (200 или 404): без неё
                         // 404 зацикливал бы загрузку на одном кадре
  };
  let painted = null;    // бэкап paintCloud для восстановления цветов облака
  let inflight = null;   // идущая загрузка (сериализация, см. заголовок)
  let want = null;       // последняя ЗАПРОШЕННАЯ позиция {bag, idx}
  let note404 = '';      // последний текст «нет предрасчёта» (не спамим панель)

  // Постоянная корневая группа и пулы буферов (см. заголовок «Перформанс»).
  const root = new THREE.Group();
  root.name = 'guard-corridor';
  root.visible = state.visible;
  scene.add(root);
  const poolCorridor = makeSegMesh(BIN_CAP * 8 + (BIN_CAP - 1) * 8,
                                   lineMat(0.55), false);
  const poolWallLine = makeSegMesh((BIN_CAP - 1) * 2 * 2, lineMat(0.85), false);
  const poolWallVert = makeSegMesh(BIN_CAP * 2 * 2, lineMat(0.35), false);
  const poolAxis = makeSegMesh((BIN_CAP - 1) * 2, lineMat(0.95), false);
  const poolAxisFar = makeSegMesh((BIN_CAP - 1) * 2, lineMat(0.9), false);
  const poolBoxes = makeSegMesh(OBST_CAP0 * 24, lineMat(0.9), false);
  const poolBeams = makeSegMesh(OBST_CAP0 * 2, lineMat(0.9, true), true);
  const poolTrain = makeSegMesh(BIN_CAP * 8 + (BIN_CAP - 1) * 8,
                                lineMat(0.9), false);
  const pools = [poolCorridor, poolWallLine, poolWallVert, poolAxis,
                 poolAxisFar, poolBoxes, poolBeams, poolTrain];
  for (const m of pools) root.add(m);

  // Кэш спрайтов меток по (текст, цвет): до перф-фикса спрайт (2D-рендер
  // текста + canvas-текстура ~295 КБ) пересоздавался на каждую находку каждого
  // кадра -- до ~8 мс CPU при 16 препятствиях. Множество вариантов ограничено
  // нормативной дальностью тракта (текст «⚠ N м»/«? ~N м»/«~N м (за reach)»,
  // N = целые метры
  // 0..130), кэш не растёт бесконечно. Диспоуз -- в api.dispose().
  const spriteCache = new Map();
  let activeSprites = [];

  function labelSprite(text, colorHex) {
    if (typeof document === 'undefined' || !document.createElement) return null;
    const key = `${text}|${colorHex}`;
    let ent = spriteCache.get(key);
    if (ent === undefined) {
      const proto = makeLabelSprite(text, colorHex);  // null, если canvas нет
      ent = proto ? { material: proto.material, scale: proto.scale.clone(),
                      renderOrder: proto.renderOrder } : null;
      spriteCache.set(key, ent);
    }
    if (!ent) return null;
    // Спрайт — на находку, общие только материал/текстура (дорогая часть в
    // кэше): возвращённый из кэша ОДИН экземпляр Sprite получал position.set
    // каждой находкой с одинаковым текстом, и у второй находки метки не было
    // (замер f737: 4 бокса, 2 метки).
    const spr = new THREE.Sprite(ent.material);
    spr.scale.copy(ent.scale);
    spr.renderOrder = ent.renderOrder;
    return spr;
  }

  // Кадр очищен: пулы в ноль (хвосты прошлого кадра не видны), спрайты сняты.
  // Ничего не диспоузит -- буферы и текстуры переиспользуются.
  function clearFrame() {
    for (const m of pools) m.geometry.setDrawRange(0, 0);
    for (const s of activeSprites) root.remove(s);
    activeSprites = [];
  }

  // Возврат цветов облака: точки препятствий -- не собственность слоя,
  // а временная перекраска чужого colArr.
  function restorePoints() {
    if (!painted) return;
    const c = typeof cloud === 'function' ? cloud() : null;
    // applyFrame при смене кадра уже перезаписал colArr целиком: восстанавливать
    // по бэкапу чужого кадра нельзя (легли бы старые цвета на новые точки).
    if (c && c.colors && c.idx === painted.frameIdx) {
      restoreCloud(c.colors, painted);
      if (typeof touchCloud === 'function') touchCloud();
    }
    painted = null;
  }

  // Индексы точек находок кадра, разбитые по уровню (hit -- подтверждённые
  // треком в пределах reach или вне габарита, unc -- однокадровые в габарите
  // и дальние за reach: оба янтарные, не тревога). Проверка
  // frame_npts -- как в hitpoints.hitIndices: при чужой нумерации не красим.
  function hitIndicesSplit(fr, count) {
    const hit = [];
    const unc = [];
    if (!fr || !Array.isArray(fr.obstacles)) return { hit, unc };
    if (typeof fr.frame_npts === 'number' && fr.frame_npts !== count) {
      return { hit, unc };
    }
    for (const ob of fr.obstacles) {
      const pi = ob && ob.point_idx;
      if (!Array.isArray(pi)) continue;
      const lvl = obstLevel(ob);
      if (lvl === 'unc' && !state.showUnc) continue;
      const dst = (lvl === 'unc' || lvl === 'far') ? unc : hit;
      for (const i of pi) {
        if (Number.isInteger(i) && i >= 0 && i < count) dst.push(i);
      }
    }
    return { hit, unc };
  }

  // Перекраска точек находок текущего guard-кадра: подтверждённые -- красным
  // (HIT_RGB), однокадровые -- янтарным. Кадр вьюера мог уйти вперёд, пока
  // летел ответ /guard: индексы чужого кадра не красим (проверка по idx;
  // дальше стоит проверка по frame_npts).
  function paintPoints(fr) {
    restorePoints();
    if (!fr || typeof cloud !== 'function') return;
    const c = cloud();
    if (!c || !c.colors) return;
    if (typeof c.idx === 'number' && typeof fr.frame === 'number'
        && c.idx !== fr.frame) return;
    const { hit, unc } = hitIndicesSplit(fr, c.count);
    // Порядок важен для точек, попавших в обе находки: янтарь красим первым
    // (его бэкап держит исходный цвет), красный вторым; в объединённом бэкапе
    // красный идёт первым, и restoreCloud вернёт сначала янтарь, потом
    // исходный -- точка не останется подсвеченной.
    const bUnc = paintCloudRgb(c.colors, unc, HIT_RGB_UNC);
    const bHit = paintCloud(c.colors, hit);
    if (bHit && bUnc) {
      const prev = new Uint8Array(bHit.prev.length + bUnc.prev.length);
      prev.set(bHit.prev);
      prev.set(bUnc.prev, bHit.prev.length);
      painted = { idx: bHit.idx.concat(bUnc.idx), prev };
    } else {
      painted = bHit || bUnc;
    }
    if (painted) {
      painted.frameIdx = c.idx;
      if (typeof touchCloud === 'function') touchCloud();
    }
  }

  function drop() {
    restorePoints();
    clearFrame();
  }

  function build(fr) {
    drop();
    const bins = fr.bins || {};
    const fm = bins.fwd_mid || [];
    const status = bins.status || [];
    const lead = bins.lead || [];
    if (!fm.length || !isNum(fr.floor)) return;
    const floor = fr.floor;

    const cOk = new THREE.Color(COL_OK).toArray();
    const cGauge = new THREE.Color(COL_GAUGE).toArray();
    const cHole = new THREE.Color(COL_HOLE).toArray();
    const cWall = new THREE.Color(COL_WALL).toArray();
    const cAxis = new THREE.Color(COL_AXIS).toArray();
    const cAxisFar = new THREE.Color(COL_AXIS_FAR).toArray();
    const cObst = new THREE.Color(COL_OBST).toArray();
    const cObstOut = new THREE.Color(COL_OBST_OUT).toArray();
    const cObstUnc = new THREE.Color(COL_OBST_UNC).multiplyScalar(UNC_DIM).toArray();
    const cObstFar = new THREE.Color(COL_OBST_FAR).toArray();
    // Подсветка режимов: bins.mode есть только у нового предрасчёта; в старом
    // JSON поля нет, и всё остаётся по статусу бина, как раньше.
    const mode = bins.mode || [];
    const cMode = {};
    for (const m of Object.keys(MODE_COLORS)) {
      cMode[m] = new THREE.Color(MODE_COLORS[m]).toArray();
    }
    const modeCol = (i) => (state.modeColors && cMode[mode[i]]) || null;
    const binColor = (i) => (modeCol(i)
      || (status[i] === 'ok' ? cOk : status[i] === 'gauge' ? cGauge : cHole));

    // ---- коридор: сечения бинов + продольные рёбра, цвет по статусу ----
    {
      const pos = [];
      const col = [];
      const push = (p, c) => { pos.push(p[0], p[1], p[2]); col.push(c[0], c[1], c[2]); };
      // Полуширины коридора бина -> углы сечения (z над полом).
      const corner = (i, side, top) => {
        const ax = isNum(bins.axis[i]) ? bins.axis[i] : 0;
        const half = side < 0 ? bins.half_left[i] : bins.half_right[i];
        return P(fm[i], ax + side * half, floor + (top ? CORRIDOR_Z_HI : CORRIDOR_Z_LO));
      };
      const drawn = [];
      for (let i = 0; i < fm.length; i++) {
        if (status[i] === 'stop') continue;
        if (!isNum(bins.half_left[i]) || !isNum(bins.half_right[i])) continue;
        drawn.push(i);
        const c = binColor(i);
        const r = [corner(i, -1, false), corner(i, 1, false),
                   corner(i, 1, true), corner(i, -1, true)];
        for (let k = 0; k < 4; k++) { push(r[k], c); push(r[(k + 1) % 4], c); }
      }
      for (let s = 0; s + 1 < drawn.length; s++) {
        const i = drawn[s];
        const j = drawn[s + 1];
        if (j - i > 1) continue;      // разрыв по 'stop' не зашиваем
        const c = modeCol(j) || (status[i] === 'ok' && status[j] === 'ok' ? cOk : binColor(j));
        for (const [side, top] of [[-1, false], [1, false], [1, true], [-1, true]]) {
          push(corner(i, side, top), c);
          push(corner(j, side, top), c);
        }
      }
      fillSeg(poolCorridor, pos, col, pos.length / 3);
    }

    // ---- габарит поезда (±TRAIN_HALF от оси бина, TRAIN_Z_LO..TRAIN_Z_HI
    // над полом) -- пунктирные маджентовые сечения и рёбра поверх коридора.
    // Это те рамки, по которым считается in_gauge; коридор может быть шире.
    {
      const pos = [];
      const col = [];
      const cT = new THREE.Color(COL_TRAIN).toArray();
      const push = (p) => { pos.push(p[0], p[1], p[2]); col.push(cT[0], cT[1], cT[2]); };
      const corner = (i, side, top) => {
        const ax = isNum(bins.axis[i]) ? bins.axis[i] : 0;
        return P(fm[i], ax + side * TRAIN_HALF,
                 floor + (top ? TRAIN_Z_HI : TRAIN_Z_LO));
      };
      const drawn = [];
      for (let i = 0; i < fm.length; i++) {
        if (status[i] === 'stop') continue;
        if (!isNum(bins.half_left[i]) || !isNum(bins.half_right[i])) continue;
        drawn.push(i);
        const r = [corner(i, -1, false), corner(i, 1, false),
                   corner(i, 1, true), corner(i, -1, true)];
        for (let k = 0; k < 4; k++) { push(r[k]); push(r[(k + 1) % 4]); }
      }
      for (let s = 0; s + 1 < drawn.length; s++) {
        const i = drawn[s];
        const j = drawn[s + 1];
        if (j - i > 1) continue;
        for (const [side, top] of [[-1, false], [1, false], [1, true], [-1, true]]) {
          push(corner(i, side, top));
          push(corner(j, side, top));
        }
      }
      fillSeg(poolTrain, pos, col, pos.length / 3);
    }

    // ---- стены: продольная линия по кромке + вертикали, только измеренные ----
    // В бине 'ok' измерены обе стороны, в 'gauge' -- только ведущая (lead);
    // кламп габарита стеной не является, и рисовать его как стену нельзя.
    // Продольные кромки -- сегменты в poolWallLine (та же картинка, что у
    // полилинии: линии тонкие, стыки не читаются), вертикали -- poolWallVert.
    {
      const posL = [];
      const colL = [];
      const posV = [];
      const colV = [];
      const push = (pos, col, p) => {
        pos.push(p[0], p[1], p[2]);
        col.push(cWall[0], cWall[1], cWall[2]);
      };
      for (const side of [-1, 1]) {
        let prev = null;
        for (let i = 0; i < fm.length; i++) {
          const measured = status[i] === 'ok'
            || (status[i] === 'gauge'
                && (lead[i] === (side < 0 ? 'left' : 'right') || lead[i] === 'both'));
          const edge = side < 0 ? bins.left[i] : bins.right[i];
          if (!measured || !isNum(edge) || !isNum(bins.axis[i])) {
            prev = null;
            continue;
          }
          const lat = bins.axis[i] + edge;
          const top = P(fm[i], lat, floor + WALL_Z_HI);
          if (prev && i === prev.i + 1) {
            push(posL, colL, prev.p);
            push(posL, colL, top);
          }
          prev = { i, p: top };
          // Вертикаль от пола до верха стены -- читается как «стена», не нить.
          push(posV, colV, P(fm[i], lat, floor));
          push(posV, colV, top);
        }
      }
      fillSeg(poolWallLine, posL, colL, posL.length / 3);
      fillSeg(poolWallVert, posV, colV, posV.length / 3);
    }

    // ---- ось: ведомая линия бинов + далёкая ось, где она ведёт ----
    // Ось красится по режиму бина (bins.mode); без mode -- одним COL_AXIS.
    {
      const pos = [];
      const col = [];
      const push = (p, c) => { pos.push(p[0], p[1], p[2]); col.push(c[0], c[1], c[2]); };
      let prev = null;
      for (let i = 0; i < fm.length; i++) {
        if (status[i] === 'stop' || !isNum(bins.axis[i])) { prev = null; continue; }
        const p = P(fm[i], bins.axis[i], floor + 0.15);
        const c = modeCol(i) || cAxis;
        if (prev && i === prev.i + 1) { push(prev.p, prev.c); push(p, c); }
        prev = { i, p, c };
      }
      fillSeg(poolAxis, pos, col, pos.length / 3);
      const far = fr.axis_far;
      if (far && Array.isArray(far.fwd_mid) && Array.isArray(far.lat)) {
        const posF = [];
        const colF = [];
        const pushF = (p) => {
          posF.push(p[0], p[1], p[2]);
          colF.push(cAxisFar[0], cAxisFar[1], cAxisFar[2]);
        };
        let prevF = null;
        for (let i = 0; i < far.fwd_mid.length; i++) {
          const st = (far.status || [])[i];
          if (st !== 'seed' && st !== 'straight' && st !== 'wall') continue;
          if (!isNum(far.lat[i])) continue;
          const p = P(far.fwd_mid[i], far.lat[i], floor + 0.25);
          // Разрывы по статусу не зашиваем -- как у прежней полилинии
          // (она соединяла все валидные точки подряд, разрыв i не проверяла).
          if (prevF) { pushF(prevF); pushF(p); }
          prevF = p;
        }
        fillSeg(poolAxisFar, posF, colF, posF.length / 3);
      }
    }

    // ---- препятствия: каркасный бокс находки по уровню трека ----
    {
      const pos = [];
      const col = [];
      const posB = [];
      const colB = [];
      const distB = [];
      const push = (p, c) => { pos.push(p[0], p[1], p[2]); col.push(c[0], c[1], c[2]); };
      const usedSprites = [];
      for (const h of fr.obstacles || []) {
        if (!isNum(h.d) || !isNum(h.x_min) || !isNum(h.x_max) || !isNum(h.height)) continue;
        const lvl = obstLevel(h);
        if (lvl === 'unc' && !state.showUnc) continue;   // фантомы погашены галочкой
        const c = lvl === 'hit' ? cObst
          : lvl === 'unc' ? cObstUnc
          : lvl === 'far' ? cObstFar : cObstOut;
        const z0 = floor;
        const z1 = floor + Math.max(h.height, 0.2);
        // Углы бокса: [x, fwd, z] в системе guard, перевод через P.
        const g = [];
        for (const z of [z0, z1]) {
          for (const x of [h.x_min, h.x_max]) {
            for (const f of [h.d - OBST_DEPTH_M / 2, h.d + OBST_DEPTH_M / 2]) {
              g.push(P(f, x, z));
            }
          }
        }
        // g: [z0: (x0,f0),(x0,f1),(x1,f0),(x1,f1), z1: то же] -- 12 рёбер.
        const E = [[0, 1], [2, 3], [0, 2], [1, 3],
                   [4, 5], [6, 7], [4, 6], [5, 7],
                   [0, 4], [1, 5], [2, 6], [3, 7]];
        for (const [a, b] of E) { push(g[a], c); push(g[b], c); }

        // Метка дальности над боксом + луч-указатель от лидара до центра
        // бокса: находку видно мгновенно, без поиска красных точек глазами.
        // Красная «⚠ N м» -- подтверждённая треком в габарите в пределах
        // reach; однокадровая -- янтарная «? ~N м»; дальняя за reach --
        // янтарная «~N м (за reach)»; вне габарита -- оранжевая, как раньше.
        const cx = (h.x_min + h.x_max) / 2;
        const colHex = (lvl === 'unc' || lvl === 'far') ? COL_OBST_UNC
          : lvl === 'out' ? COL_OBST_OUT
          : (h.confirmed !== false || h.track_confirmed === true)
            ? COL_OBST : COL_OBST_OUT;
        const text = lvl === 'unc' ? `? ~${Math.round(h.d)} м`
          : lvl === 'far' ? `~${Math.round(h.d)} м (за reach)`
          : obstLabelText(h);
        const label = labelSprite(text, colHex);
        if (label) {
          label.position.set(...P(h.d, cx, z1 + LABEL_LIFT_M));
          root.add(label);
          usedSprites.push(label);
        }
        // Луч-пунктир: пул сегментов, lineDistance [0, len] на луч -- как
        // computeLineDistances у прежней отдельной Line на находку.
        const to = P(h.d, cx, (z0 + z1) / 2);
        const cb = lvl === 'unc' ? cObstUnc : lvl === 'far' ? cObstFar : cObst;
        posB.push(0, 0, 0, to[0], to[1], to[2]);
        colB.push(cb[0], cb[1], cb[2], cb[0], cb[1], cb[2]);
        const len = Math.hypot(to[0], to[1], to[2]);
        distB.push(0, len);
      }
      for (const s of activeSprites) {
        if (!usedSprites.includes(s)) root.remove(s);
      }
      activeSprites = usedSprites;
      fillSeg(poolBoxes, pos, col, pos.length / 3);
      fillSeg(poolBeams, posB, colB, posB.length / 3, distB);
    }

    root.visible = state.visible;
  }

  // Строка тревоги в guard-группе панели: по ПОДТВЕРЖДЁННОЙ треком находке
  // в габарите В ПРЕДЕЛАХ reach -- яркая «⚠ ПРЕПЯТСТВИЕ: N м (lat ±X.X)» по
  // БЛИЖАЙШЕЙ такой находке; подтверждённая треком, но ЗА reach
  // (beyond_reach) -- янтарная «дальняя находка за reach: N м» (трекер
  // подтверждает по счётчику кадров, обходя правило reach -- это раннее
  // предупреждение, а не тревога); только однокадровые -- нейтральная
  // янтарная строка без красного фона (это НЕ тревога); без находок --
  // нежная «свободно до reach м»; слой выключен или кадр не загружен --
  // скрыта. Старый JSON без полей track_*/beyond_reach -- прежнее поведение
  // (тревога по in_gauge).
  function updateAlert() {
    const el = $('guardAlert');
    if (!el) return;
    const fr = state.frame;
    let worst = null;
    let worstFar = null;
    let worstUnc = null;
    if (state.visible && fr) {
      for (const h of fr.obstacles || []) {
        if (!h || !h.in_gauge || !isNum(h.d)) continue;
        if (h.track_confirmed === false) {
          if (state.showUnc && (!worstUnc || h.d < worstUnc.d)) worstUnc = h;
        } else if (h.beyond_reach === true) {
          // Красной тревоге (worst) -- только track_confirmed && !beyond_reach;
          // у старого JSON beyond_reach нет (undefined !== true) -- как раньше.
          if (!worstFar || h.d < worstFar.d) worstFar = h;
        } else if (!worst || h.d < worst.d) {
          worst = h;
        }
      }
    }
    const latOf = (h) => (isNum(h.x_min) && isNum(h.x_max)
      ? (h.x_min + h.x_max) / 2 : 0);
    if (worst) {
      const lat = latOf(worst);
      el.style.display = '';
      el.style.background = 'rgba(255, 60, 60, 0.20)';
      el.style.border = '1px solid rgba(255, 90, 90, 0.7)';
      el.style.color = '#ffc4c4';
      el.style.fontWeight = '600';
      const sign = lat >= 0 ? '+' : '−';
      el.textContent = `⚠ ПРЕПЯТСТВИЕ: ${Math.round(worst.d)} м`
        + ` (lat ${sign}${Math.abs(lat).toFixed(1)})`;
    } else if (worstFar) {
      const lat = latOf(worstFar);
      el.style.display = '';
      el.style.background = 'transparent';
      el.style.border = '1px solid rgba(255, 194, 51, 0.45)';
      el.style.color = '#ffd98a';
      el.style.fontWeight = '400';
      const sign = lat >= 0 ? '+' : '−';
      el.textContent = `дальняя находка за reach: ${Math.round(worstFar.d)} м`
        + ` (lat ${sign}${Math.abs(lat).toFixed(1)})`;
    } else if (worstUnc) {
      const lat = latOf(worstUnc);
      el.style.display = '';
      el.style.background = 'transparent';
      el.style.border = '1px solid rgba(255, 194, 51, 0.45)';
      el.style.color = '#ffd98a';
      el.style.fontWeight = '400';
      const sign = lat >= 0 ? '+' : '−';
      el.textContent = `неподтверждённая находка: ${Math.round(worstUnc.d)} м`
        + ` (lat ${sign}${Math.abs(lat).toFixed(1)})`;
    } else if (state.visible && fr && isNum(fr.reach)) {
      el.style.display = '';
      el.style.background = 'transparent';
      el.style.border = '1px solid transparent';
      el.style.color = '#6f7b90';
      el.style.fontWeight = '400';
      el.textContent = `свободно до ${Math.round(fr.reach)} м`;
    } else {
      el.style.display = 'none';
    }
  }

  function updateSummary() {
    const el = $('guardSummary');
    if (!el) return;
    const fr = state.frame;
    if (!fr) {
      el.textContent = note404 || 'кадр не загружен';
      return;
    }
    const nIn = (fr.obstacles || []).filter((h) => h.in_gauge).length;
    const nAll = (fr.obstacles || []).length;
    // Поля track_* пишет новый export_web; в старом кэше их нет -- не показываем.
    const hasTrk = (fr.obstacles || []).some((h) => h.track_confirmed !== undefined);
    const nTrk = (fr.obstacles || []).filter((h) => h.track_confirmed).length;
    el.textContent = `кадр ${fr.frame}: подтверждено до ${Math.round(fr.reach)} м`
      + ` (с мостами ${Math.round(fr.reach_bridged)} м), бинов ок ${fr.n_ok}`
      + (nAll ? `, препятствий ${nAll} (в габарите ${nIn})` : ', препятствий нет')
      + (hasTrk ? `, треком подтверждено ${nTrk}` : '');
    const note = $('guardNote');
    if (note) note.textContent = '';
  }

  // Загрузка с сериализацией и «последний запрошенный выигрывает»: ответ
  // старого кадра не применяется, если запрошен новый (гонка /objects).
  function load(bag, idx) {
    want = { bag, idx };
    if (!state.visible) return Promise.resolve();   // выключен -- не ходим в сеть
    if (inflight) return inflight;
    const run = (async () => {
      try {
        while (want) {
          const cur = want;
          const { bag: b, idx: i } = cur;
          if (state.bag === b && state.idx === i && state.done) {
            // Выход только после уступки цикла: load(), вызванный СИНХРОННО
            // следом, пока inflight ещё висит, только обновит want -- без
            // уступки его запрос потерялся бы вместе с этим выходом.
            await Promise.resolve();
            if (want === cur || (want.bag === b && want.idx === i)) {
              // guard-кадр тот же, но applyFrame перезаписывает colArr при
              // КАЖДОМ showFrame (повторный шаг на тот же кадр, опоздавший
              // ответ соседнего showFrame) -- подсветку точек надо вернуть.
              if (state.visible && state.frame) paintPoints(state.frame);
              return;
            }
            continue;
          }
          try {
            const payload = await fetchGuardShared(b, i);
            if (want !== cur && want && (want.bag !== b || want.idx !== i)) continue;
            state.bag = b;
            state.idx = i;
            state.done = true;
            state.frame = payload;
            note404 = '';
            build(payload);
            // Точки находок -- поверх палитры облака (drop() в build уже
            // вернул цвета прежнего кадра). Вне build(): его ранний выход на
            // вырожденном payload не должен снимать подсветку точек.
            if (state.visible) paintPoints(payload);
          } catch (e) {
            if (e && e.status === 404) {
              // Кадр просто не посчитан: слой пуст, подсказка -- в панели.
              if (want !== cur && want && (want.bag !== b || want.idx !== i)) continue;
              state.bag = b;
              state.idx = i;
              state.done = true;
              state.frame = null;
              note404 = e.detail || 'нет предрасчёта';
              drop();
            } else {
              throw e;
            }
          }
        }
      } finally {
        inflight = null;
        updateSummary();
        updateAlert();
      }
    })();
    inflight = run;
    return run;
  }

  const api = {
    load,
    data: () => state.frame,
    // Общий fetch /guard (кэш window.__guardPromises): его же использует
    // topview (app.js передаёт как guardFetch) -- второй запрос и второй
    // JSON.parse на кадр не уходят.
    guardFetch: fetchGuardShared,
    setVisible(on) {
      state.visible = Boolean(on);
      root.visible = state.visible;
      if (!state.visible) {
        restorePoints();
      } else if (state.frame) {
        // load() на уже применённом кадре выходит по быстрому пути без
        // build -- подсветку точек возвращаем сами.
        paintPoints(state.frame);
      }
      updateAlert();      // выключен -- строка тревоги скрыта
      return state.visible;
    },
    isVisible: () => state.visible,
    // Тумблер «подсветка режимов» из панели: перестраивает каркас текущего
    // кадра другой раскраской (режимы вкл/выкл), в сеть не ходит.
    setModeColors(on) {
      state.modeColors = on !== false;
      if (state.frame) {
        build(state.frame);
        if (state.visible) paintPoints(state.frame);
      }
      return state.modeColors;
    },
    // Показ однокадровых неподтверждённых находок (янтарных): фантомы-транзиенты
    // живут 1-3 кадра; кто не хочет их видеть — гасит галочкой.
    setShowUnc(on) {
      state.showUnc = on !== false;
      if (state.frame) {
        build(state.frame);
        if (state.visible) paintPoints(state.frame);
        updateAlert(state.frame);
      }
      return state.showUnc;
    },
    // Перекрасить точки находок после внешней перекраски облака того же кадра
    // (paintZoneColors/markPoints перезаписывают colArr целиком и сносят
    // подсветку): app.js зовёт это после paintZoneColors.
    repaint() {
      if (state.visible && state.frame) paintPoints(state.frame);
    },
    // Смена записи: от прежней не остаются ни каркас, ни сводка.
    reset() {
      drop();
      state.frame = null;
      state.bag = null;
      state.idx = -1;
      state.done = false;
      want = null;
      note404 = '';
      updateSummary();
      updateAlert();
    },
    dispose() {
      drop();
      state.frame = null;
      scene.remove(root);
      disposeObject(root);              // пулы: геометрии и материалы
      for (const ent of spriteCache.values()) {
        if (ent && ent.material) {        // кэшированные материалы и текстуры
          if (ent.material.map) ent.material.map.dispose();
          ent.material.dispose();
        }
      }
      spriteCache.clear();
    },
  };
  // Отладка из консоли страницы (тот же приём, что window.__objectsLayer).
  if (typeof window !== 'undefined') window.__guardLayer = api;
  return api;
}
