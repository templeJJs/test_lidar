import * as THREE from 'three';
import { OrbitControls } from './OrbitControls.js';

// Слой объектов пути (track3d.js) подключаем МЯГКО. Файл отдаётся отдельным
// маршрутом /track3d.js, и если он не отдан (маршрут убрали, файла нет, статика
// закрыта) -- страница обязана жить: облако, раскраски, пресеты, разрез и
// «лента рельсов» работают без объектов пути. Статический import в этом случае
// уронил бы весь модуль app.js -- то есть всю страницу, а не только объекты.
let createTrackLayer = null;
let createTrackLayerError = null;
try {
  ({ createTrackLayer } = await import('./track3d.js'));
} catch (e) {
  createTrackLayerError = e;
  console.warn('track3d: слой объектов пути недоступен —', e);
}

// Слой объектов из замеров (objects.js) -- тот же приём: свой файл, мягкий
// импорт, страница живёт без него.
let createObjectsLayer = null;
try {
  ({ createObjectsLayer } = await import('./objects.js'));
} catch (e) {
  console.warn('objects: слой объектов из замеров недоступен —', e);
}

// Слой коридора безопасности (safetylayer.js) подключаем так же мягко и по той же
// причине: без файла или маршрута страница обязана жить целиком -- облако,
// раскраски, объекты пути, разрез.
let createSafetyLayer = null;
let createSafetyLayerError = null;
try {
  ({ createSafetyLayer } = await import('./safetylayer.js'));
} catch (e) {
  createSafetyLayerError = e;
  console.warn('safetylayer: слой коридора безопасности недоступен —', e);
}

// Слой разметки (labellayer.js) -- тот же мягкий импорт: без файла страница живёт
// без режима разметки, а панель говорит, почему его нет.
let createLabelLayer = null;
let createLabelLayerError = null;
// Геометрия позы и траектории -- чистые функции того же модуля: их считает и клиент,
// и сервер (`labeling.py`), одной формулой, иначе рисунок разошёлся бы со счётчиком
// закрытых точек. Модуля нет -- разметка недоступна целиком (объекты некуда ставить).
let labelGeom = null;
try {
  const mod = await import('./labellayer.js');
  createLabelLayer = mod.createLabelLayer;
  labelGeom = {
    poseMoment: mod.poseMoment,
    poseVerticesAt: mod.poseVerticesAt,
    placePose: mod.placePose,
    poseInScene: mod.poseInScene,
    trajectoryAt: mod.trajectoryAt,
    trajectoryBaseAtFrame: mod.trajectoryBaseAtFrame,
    trajectorySpeed: mod.trajectorySpeed,
    trajectoryDistance: mod.trajectoryDistance,
    trajectoryLabel: mod.trajectoryLabel,
  };
} catch (e) {
  createLabelLayerError = e;
  console.warn('labellayer: слой разметки недоступен —', e);
}

const $ = (id) => document.getElementById(id);
const errBox = $('err');

// Навесить обработчик, если элемент есть: панель могут править параллельно, и
// отсутствие новой группы не должно ломать всю страницу.
function wire(id, event, handler) {
  const node = $(id);
  if (node) node.addEventListener(event, handler);
}

function fail(e) {
  errBox.style.display = 'block';
  errBox.textContent = String(e && e.stack ? e.stack : e);
  console.error(e);
}

// Данные приходят в осях лидара: X -- поперёк, Y -- вдоль тоннеля (вперёд это -Y),
// Z -- вверх. Поэтому "верх" мира задаём по Z, иначе OrbitControls крутит вокруг Y.
THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

const RF = 0;  // kind: rgb8
const LF = 1;  // kind: label8

const HUD_KEYS = {
  rail: 'рельс', bed: 'низ', ceiling: 'потолок', wall: 'стены', middle: 'средний',
};

let meta = null;
let renderer, scene, camera, controls, points, gridMesh, axesMesh;
let posAttr, colAttr;
let hideAttr;
let capacity = 0;
let posArr, colArr, hideArr;
// Точки, закрытые объектом разметки: индекс точки кадра -> погашена. Маска живёт
// в атрибуте облака, поэтому она не может «накопить» мусор между кадрами --
// applyHiddenPoints гасит прежние и ставит новые.
let hiddenIdx = [];
const cache = new Map();
// О неудаче перекраски по /labels пишем в консоль ОДИН раз за сессию: маршрута
// может не быть у сервера, поднятого до этой правки, а на каждом кадре спамить
// незачем (перекраска -- уточнение раскраски, а не обязанность страницы).
let labelsWarned = false;
function warnLabelsOnce(...args) {
  if (labelsWarned) return;
  labelsWarned = true;
  console.warn(...args);
}
const state = {
  idx: 0, playing: true, loading: false, fps: 10, realtime: false, mode: 'zones',
  maxDist: 40, numPoints: 0, fpsShown: 0, lastFrameMs: 0, nextTick: 0, counts: null,
  labels: null,       // метки зон текущего кадра (вью в буфер) -- для перекраски на месте
  labelFits: 0,       // сколько точек этих меток реально влезло
  bag: null,          // текущая запись (датасет); сервер в режиме корня держит несколько
  profile: null,      // профиль вида этой записи (web/view_profiles.json на сервере)
  switching: false,   // идёт переключение записи: старые данные уже сброшены
};

const FRAME_MARK = new Uint8Array([0x4c, 0x5a, 0x46, 0x52]); // 'LZFR'

// 3D-объекты пути (рельсы/шпалы/пол) из /track. Сам слой живёт в track3d.js:
// здесь только подключение -- создание, вызов load при смене кадра и кнопки.
let trackLayer = null;
let objectsLayer = null;
let pointsOn = true;   // тумблер «облако точек» (в «чистом поле» скрыто всегда)

// Коридор безопасности из /meta (блок `tunnel`). Сам слой живёт в safetylayer.js:
// здесь только передача метаданных, подсветка точек после кадра и тумблер блока.
let safetyLayer = null;

// Слой разметки (labellayer.js): боксы объектов и голубые точки трассировки.
// Панель, объекты и запросы к серверу -- здесь, отрисовка -- в слое.
let labelLayer = null;

function applyPointsVisible() {
  points.visible = pointsOn && !(trackLayer && trackLayer.isCleanField());
  syncRailRibbon();      // виден ли наш меш рельсов -> гасить зелёную полосу middle
}

/**
 * Погасить точки кадра, закрытые объектом разметки (`/label/preview` ->
 * `removed_indices`). Без этого за поставленным объектом остаётся «ряд точек
 * вдаль»: те самые лучи кадра, которые объект перекрывает, -- а он их перекрывает
 * ТОЛЬКО в трассировке, пока объект не «сожжён» в облако.
 *
 * Индексы относятся к кадру, в котором считан предпросмотр (опорному), поэтому
 * гасить их можно лишь когда этот кадр и показан: на другом кадре облако --
 * другое, и те же номера означали бы чужие точки.
 */
function applyHiddenPoints(indices) {
  if (!hideAttr) return;
  const next = Array.isArray(indices) ? indices : [];
  if (next.length === hiddenIdx.length) {
    let same = true;
    for (let i = 0; i < next.length; i++) {
      if (next[i] !== hiddenIdx[i]) { same = false; break; }
    }
    if (same) return;
  }
  let lo = Infinity;
  let hi = -Infinity;
  for (const i of hiddenIdx) {
    if (i >= 0 && i < capacity) { hideArr[i] = 0; lo = Math.min(lo, i); hi = Math.max(hi, i); }
  }
  for (const i of next) {
    if (i >= 0 && i < capacity) { hideArr[i] = 1; lo = Math.min(lo, i); hi = Math.max(hi, i); }
  }
  hiddenIdx = next.slice();
  if (hi >= lo) {
    hideAttr.needsUpdate = true;
    hideAttr.updateRange.offset = lo;
    hideAttr.updateRange.count = hi - lo + 1;
  }
}

// Слой линии хода/туннеля (web/safetylayer.js) -- гость в этом клиенте. Любой его
// сбой не должен ронять страницу: раньше исключение из safetyLayer.update(meta) в
// main() обрывало main() на полпути -- вставали и кадры, и панель, и пресеты.
// Теперь ловим, один раз пишем причину в панель и продолжаем без их слоя.
let safetyBroken = null;

function safeSafety(what, fn) {
  if (!safetyLayer) return null;
  try {
    return fn(safetyLayer);
  } catch (e) {
    if (!safetyBroken) {
      const reason = (e && e.message) ? e.message : String(e);
      safetyBroken = `${what}: ${reason}`;
      console.warn('safetylayer: слой линии хода/туннеля не работает —', e);
      const note = $('safetyNote');
      if (note) {
        note.textContent = `слой линии хода/туннеля не работает (${safetyBroken}). `
          + 'Остальная страница работает: облако, кадры, объекты пути, разрез.';
      }
    }
    return null;
  }
}

// Слой не подключился: причину пишем в компактный HUD и в свёрнутый блок, но
// остальную страницу не трогаем -- облако, раскраски, пресеты, разрез и «лента
// рельсов» живут своей жизнью. Кнопки группы не «мертвим»: они просто ничего не
// делают, а состояние видно в панели.
function showTrackUnavailable(error) {
  const reason = (error && (error.message || String(error))) || 'слой track3d.js не загрузился';
  const line = `объекты пути недоступны: ${reason}`;
  const hud = $('hudTrack');
  if (hud) hud.textContent = line;
  noteCamera(line);        // то же рядом с кнопками группы
  const box = $('trackSrc');
  if (box) {
    box.textContent = '';
    const note = document.createElement('div');
    note.textContent = `${line}. Страница работает: облако точек, раскраски, `
      + 'разрез, пресеты и «лента рельсов» — на месте.';
    box.appendChild(note);
  }
}

// Разрез X-Z в правом нижнем углу: рисуем на 2D-canvas из того же кадра
// (posArr/colArr), без дополнительных запросов к серверу.
const SECTION_W = 380, SECTION_H = 240;
// Разрез -- СТАТИЧНАЯ рамка из метаданных записи: ни одна координата не берётся
// из точек кадра, поэтому срез не перекадрируется при движении. Координаты --
// ВЫПРЯМЛЕННЫЕ, иначе уклон записи размазывает картинку по вертикали:
//   u = x - ось(y)  -- поперёк пути, 0 на оси (ось = meta.path.variants.axis,
//                      иначе константа meta.axis_x);
//   w = z - пол(y)  -- высота над ЛОКАЛЬНОЙ плоскостью пола (meta.floor.a + b·y).
// Поэтому пол идёт строго по линии w = 0, рельсы -- на своей высоте, а не
// диагональными полосами. Рамка:
//   u: ±2.6 м        -- обе нити своей колеи гарантированно в кадре;
//   w: -0.6 .. +3.5 м -- от 0.6 м ниже плоскости пола до 3.5 м над ней;
//   точки: фиксированная плита SECTION_Y_NEAR..SECTION_Y_FAR м от сенсора.
const SECTION_U_HALF_M = 2.6;
const SECTION_TOP_M = 3.5;
const SECTION_BOTTOM_M = -0.6;
const SECTION_Y_NEAR = -2;      // ближняя граница плиты точек, м
const SECTION_Y_FAR = -30;      // дальняя граница плиты точек, м
let sectionCanvas, sectionCtx, sectionDpr = 1, sectionLastIdx = -1;

async function loadMeta(bag = state.bag) {
  // `?bag=` понимают все маршруты сервера; без параметра он отдаёт запись по
  // умолчанию, поэтому одиночный запуск работает как раньше.
  const url = bag ? `/meta?bag=${encodeURIComponent(bag)}` : '/meta';
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  meta = await res.json();
  state.bag = meta.bag;          // сервер подтверждает, чью запись отдал
  // Коридор безопасности -- часть метаданных: слой перерисовывается сразу, как
  // только пришёл /meta (и на старте страницы, и при переключении записи).
  safeSafety('update', (l) => l.update(meta));
  return meta;
}

// Список записей: в режиме корня сервер кладёт в /meta ключ `bags`, в одиночном
// запуске его нет -- тогда запись ровно одна (совместимость со старым сервером).
function bagList() {
  if (Array.isArray(meta.bags) && meta.bags.length) return meta.bags;
  return [{ name: meta.bag, frames: meta.frames, has_profile: false }];
}

function makePoints() {
  capacity = Math.max(1000, Number(meta.max_points_hint) || 400000);
  posArr = new Float32Array(capacity * 3);
  colArr = new Uint8Array(capacity * 3);
  hideArr = new Uint8Array(capacity);
  const geom = new THREE.BufferGeometry();
  posAttr = new THREE.BufferAttribute(posArr, 3);
  posAttr.setUsage(THREE.DynamicDrawUsage);
  colAttr = new THREE.BufferAttribute(colArr, 3, true); // normalized uint8
  colAttr.setUsage(THREE.DynamicDrawUsage);
  hideAttr = new THREE.BufferAttribute(hideArr, 1);
  hideAttr.setUsage(THREE.DynamicDrawUsage);
  geom.setAttribute('position', posAttr);
  geom.setAttribute('aColor', colAttr);
  geom.setAttribute('aHide', hideAttr);
  geom.setDrawRange(0, 0);

  const uniforms = {
    uSize: { value: meta.point_size },
    uPixelRatio: { value: Math.min(window.devicePixelRatio || 1, 2) },
    uMaxDist: { value: state.maxDist },
  };
  const mat = new THREE.ShaderMaterial({
    uniforms,
    vertexShader: `
      attribute vec3 aColor;
      attribute float aHide;
      uniform float uSize;
      uniform float uPixelRatio;
      uniform float uMaxDist;
      varying vec3 vColor;
      varying float vKeep;
      void main() {
        vColor = aColor;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mv;
        vKeep = (aHide < 0.5 && -position.y <= uMaxDist) ? 1.0 : 0.0;
        float d = max(-mv.z, 0.5);
        gl_PointSize = clamp(uSize * uPixelRatio * (14.0 / d), 1.0, 90.0);
      }`,
    fragmentShader: `
      precision mediump float;
      varying vec3 vColor;
      varying float vKeep;
      void main() {
        if (vKeep < 0.5) discard;
        gl_FragColor = vec4(vColor, 1.0);
      }`,
    depthTest: true,
    depthWrite: true,
  });
  points = new THREE.Points(geom, mat);
  points.frustumCulled = false;
  points.renderOrder = 1;
  scene.add(points);
  return mat;
}

function lineMat(color) {
  return new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.55 });
}

function buildStaticOverlays() {
  // сетка на плоскости Z = 0 (шаг 5 м, ±40 по X, 0..-length по Y)
  const g = meta.grid;
  const pts = [];
  for (let x = -g.half_width; x <= g.half_width + 1e-6; x += g.spacing) {
    pts.push(x, -g.length, g.z, x, 0, g.z);
  }
  for (let y = -g.length; y <= 1e-6; y += g.spacing) {
    pts.push(-g.half_width, y, g.z, g.half_width, y, g.z);
  }
  const gg = new THREE.BufferGeometry();
  gg.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
  gridMesh = new THREE.LineSegments(gg, lineMat(0x4a5568));
  gridMesh.visible = $('grid').checked;
  gridMesh.renderOrder = 0;
  scene.add(gridMesh);

  // оси X/Y/Z из начала координат
  const a = new THREE.BufferGeometry();
  a.setAttribute('position', new THREE.Float32BufferAttribute(
    [0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 6], 3));
  a.setAttribute('color', new THREE.Float32BufferAttribute(
    [1, .25, .25, 1, .25, .25, .3, 1, .35, .3, 1, .35, .4, .6, 1, .4, .6, 1], 3));
  axesMesh = new THREE.LineSegments(a, new THREE.LineBasicMaterial({ vertexColors: true }));
  axesMesh.visible = $('axes').checked;
  scene.add(axesMesh);
}

// Рельсы (модель) из панели убраны -- см. комментарий выше: /set отсюда не зовём.

// Ручной «модели рельсов» в панели больше нет: ось и колея приходят в /meta из
// детектора (track_geometry) через профиль записи, коридор безопасности сервер
// заметает вдоль измеренной линии рельсов, а «лента рельсов» -- это раскраска зон.
// Поэтому из нашей страницы /set не вызывается вовсе; маршрут на сервере остался
// (им пользуется React-клиент и CLI: axis/gauge/contact/tunnel_*/path_variant).
// Если в панели появится управление коридором или линией хода -- сюда вернётся
// вызов `/set?bag=<запись>&tunnel_half=...&path_variant=...`, сервер его понимает.

// ------------------------------------------------- одна модель рельсов ------
// «Лента рельсов» вьюера -- это НЕ отдельный 3D-объект: в раскраске `zones` сервер
// помечает точки головок рельсов зоной `rail` (label 0), а клиент красит их
// ярко-оранжевым (ZONE_COLORS.rail). Те же рельсы видны засечками в разрезе X-Z.
//
// Меш рельсов в панели есть, но по умолчанию ВЫКЛЮЧЕН (галочка `t-rails`): рельсы
// пользователь хочет видеть ТОЧКАМИ. Перекраски точек зоны `rail` больше нет ни при
// каком положении галочки -- точки с меткой `rail` всегда цвета рельса. Полоса
// меша -- отдельная сущность, и гасить ленту точками ей не нужно.

function railMeshOn() {
  return Boolean(trackLayer && typeof trackLayer.railsVisible === 'function'
    && trackLayer.railsVisible());
}

let railBandHidden = null;     // последнее применённое состояние гашения полосы

// Нити рельсов (x = s*y + i): нужны, чтобы покрасить полосу вокруг рельса локально,
// не заводя второй источник геометрии и не спрашивая сервер.
//
// Берём их из данных /track, которые слой объектов пути уже скачал: это тот же
// источник, по которому сервер ставит метку rail (ось своего кадра, иначе --
// ближайшего разобранного). Слой держит последний собранный кадр, поэтому у края
// полосы пара точек может остаться зелёной -- замер: 0.7 % зелёной middle в полосе
// остаётся у границы ±0.5 м.
//
// `/meta` отдаёт тот же ключ (`rail_zone.lines_xy`), но только когда сервер уже
// посчитал геометрию какого-нибудь кадра этой записи -- при первом открытии
// страницы и при переходе на запись, которую сервер ещё не строил, там пусто.
// Поэтому /meta -- только запасной источник (старый сервер, слой пути не подключился).
function railLinesXy() {
  const data = trackLayer && typeof trackLayer.data === 'function' ? trackLayer.data() : null;
  const rails = ((data || {}).meta || {}).rail_mesh;
  const bySide = rails && rails.rails;
  if (bySide && typeof bySide === 'object') {
    const out = [];
    for (const side of Object.keys(bySide).sort()) {
      const line = (bySide[side] || {}).line;
      const s = line ? Number(line.s) : NaN;
      const i = line ? Number(line.i) : NaN;
      if (Number.isFinite(s) && Number.isFinite(i)) out.push([s, i]);
    }
    if (out.length) return out;
  }
  const rz = meta && meta.rail_zone;
  const lines = rz && Array.isArray(rz.lines_xy) ? rz.lines_xy : null;
  if (!lines || !lines.length) return null;
  return lines.filter((l) => Array.isArray(l) && l.length >= 2
    && Number.isFinite(l[0]) && Number.isFinite(l[1]));
}

// Полуширина полосы вокруг нити, где зелёная middle читается как «второй рельс».
// Замер по for_hackathon (кадры 0 и середина): в ±0.5 м лежит 81-13836 точек
// middle (0.5-30 % зоны), в ±0.25 м -- 41-4775.
const RAIL_BAND_M = 0.5;

function paintZoneColors() {
  const labels = state.labels;
  if (!labels) return;
  const pal = meta.zone_colors;
  const fits = state.labelFits;
  const meshOn = railMeshOn();
  const lines = meshOn ? railLinesXy() : null;
  const midIdx = meta.zone_names.indexOf('middle');
  const bedIdx = meta.zone_names.indexOf('bed');
  const band = lines && bedIdx >= 0 && midIdx >= 0;
  for (let i = 0, j = 0; i < fits; i++, j += 3) {
    let c = pal[labels[i]] || [255, 0, 255];
    if (band && labels[i] === midIdx) {
      // Зелёная полоса вдоль рельса гасится в цвет «низа» -- и только она:
      // остальная зона middle остаётся как была.
      const x = posArr[j], y = posArr[j + 1];
      for (let k = 0; k < lines.length; k++) {
        if (Math.abs(x - (lines[k][0] * y + lines[k][1])) <= RAIL_BAND_M) {
          c = pal[bedIdx];
          break;
        }
      }
    }
    colArr[j] = c[0]; colArr[j + 1] = c[1]; colArr[j + 2] = c[2];
  }
  colAttr.needsUpdate = true;
  // Подсветку туннеля их слой тоже делает по colArr -- повторяем её, иначе
  // перекраска рельсов её снесёт.
  safeSafety('markPoints', (l) => l.markPoints(posArr, colArr, state.numPoints));
  colAttr.needsUpdate = true;
  sectionLastIdx = -1;      // разрез берёт цвета из colArr -- перерисуем и его
}

// Зовётся из applyPointsVisible (слой сообщает о смене видимости/данных) и при
// переключении тумблера рельсов в панели: от меша зависит гашение зелёной полосы
// `middle` вдоль нитей (см. paintZoneColors), а не цвет точек зоны `rail`.
function syncRailRibbon() {
  const hidden = railMeshOn();
  if (hidden === railBandHidden) return;
  railBandHidden = hidden;
  paintZoneColors();
  buildLegend();
}

function buildLegend() {
  const box = $('legend');
  box.innerHTML = '';
  const add = (cssColor, text) => {
    const d = document.createElement('div');
    const i = document.createElement('i');
    i.style.background = cssColor;
    const s = document.createElement('span');
    s.textContent = text;
    d.appendChild(i); d.appendChild(s); box.appendChild(d);
  };
  const rgb = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;
  const pal = meta.zone_colors;
  meta.zone_names.forEach((name, i) => {
    add(rgb(pal[i]), `${name} — ${HUD_KEYS[name] || name}`);
  });
  add(rgb(meta.blocked_color), 'разрыв — коридор занят препятствием');
}

function applyFrame(buffer, idx) {
  const u8 = new Uint8Array(buffer);
  for (let i = 0; i < 4; i++) {
    if (u8[i] !== FRAME_MARK[i]) throw new Error('bad frame magic');
  }
  const dv = new DataView(buffer);
  const kind = dv.getUint8(5);
  const n = dv.getUint32(8, true);
  const fits = Math.min(n, capacity);
  if (n > capacity) {
    console.warn(`frame has ${n} points > capacity ${capacity}; drawing first ${fits}`);
  }
  const positions = new Float32Array(buffer, 12, fits * 3);
  posArr.set(positions);
  if (kind === RF) {
    colArr.set(new Uint8Array(buffer, 12 + n * 12, fits * 3));
    state.labels = null;      // раскраска вшита в кадр -- перекрашивать нечего
  } else if (kind === LF) {
    // Метки зон держим как вью в буфер кадра: по ним можно мгновенно перекрасить
    // кадр без похода на сервер (нужно при переключении «одной модели рельсов»).
    const labels = new Uint8Array(buffer, 12 + n * 12, n);
    state.labels = labels;
    state.labelFits = fits;
    // `reserved` заголовка -- номер кадра, по оси которого сервер поставил метку
    // rail (axis_frame, см. RailZone). Он равен индексу кадра всегда, кроме
    // запасного пути: геометрию кадра собрать не удалось вовсе (нет
    // track_geometry) -- тогда рельс может читаться зелёным, и об этом стоит
    // сказать в консоль один раз за сессию. На экране это уже не исправить:
    // своей оси кадра у сервера нет, значит и полосы рельса нет.
    const axisFrame = dv.getUint16(6, true);
    if (Number.isFinite(Number(idx)) && axisFrame !== Number(idx)) {
      warnLabelsOnce(`frame: метка rail посчитана по оси кадра ${axisFrame}, `
        + `а показан кадр ${idx} — геометрия кадра не собралась`);
    }
    paintZoneColors();
    const counts = new Array(meta.zone_colors.length).fill(0);
    for (let i = 0; i < n; i++) counts[labels[i]]++;
    state.counts = counts;
  } else {
    throw new Error('unknown colour kind ' + kind);
  }
  posAttr.needsUpdate = true;
  colAttr.needsUpdate = true;
  points.geometry.setDrawRange(0, fits);
  state.numPoints = fits;
  sectionLastIdx = -1;   // разрез перерисуем на новом кадре
}

function fetchFrame(idx) {
  const q = new URLSearchParams({ idx, mode: state.mode });
  if (state.bag) q.set('bag', state.bag);
  const url = `/frame?${q}`;
  return fetch(url).then((r) => {
    if (!r.ok) throw new Error(`GET ${url} -> ${r.status}`);
    return r.arrayBuffer();
  });
}

// «Запись · fNN» — то, что копируется по клику на номер кадра в панели.
function frameStamp() {
  return `${state.bag || '—'} · f${state.idx}`;
}

// Номер кадра в строке «кадр (клик = скопировать)»: обновляем на каждой смене кадра,
// а на время показа «скопировано» подменяем текст статусом.
function updateFrameCopy(flash) {
  const el = $('frameCopy');
  if (!el) return;
  const total = Math.max(0, meta.frames - 1);
  el.textContent = flash || `${state.idx} / ${total}`;
  el.title = `Скопировать «${frameStamp()}»`;
}

// Метки зон одного кадра лёгким маршрутом: 12 байт заголовка + n байт uint8, без
// позиций (у /frame на кадре 190 тыс. точек одни позиции -- 2.3 МБ). Нужен для
// перекраски УЖЕ нарисованного кадра, когда геометрия пути этого кадра посчитана.
function fetchLabels(bag, idx) {
  const q = new URLSearchParams({ idx });
  if (bag) q.set('bag', bag);
  const url = `/labels?${q}`;
  return fetch(url, { cache: 'no-store' }).then((r) => {
    if (!r.ok) throw new Error(`GET ${url} -> ${r.status}`);
    return r.arrayBuffer();
  });
}

// Перекраска после готовой геометрии кадра. Метку rail сервер ставит по оси
// СВОЕГО кадра и сам собирает геометрию, если её нет (/frame?mode=zones несёт
// axis_frame в заголовке), поэтому перекраска -- уточнение на запасной путь:
// геометрию собрать не удалось вовсе (нет track_geometry) -- тогда /labels может
// дать другую полосу, чем кадр. Порядок: /track кадра -> /labels кадра ->
// перекраска. Обычно /labels отвечает из кэша (сборка уже сделана кадром).
// Ничего не роняем: нет слоя, нет геометрии этого кадра, уехали на другой кадр,
// /labels не ответил -- молча оставляем кадр как есть.
function relabelAfterTrack(promise, bag, idx) {
  Promise.resolve(promise).then(() => {
    if (!trackLayer || state.idx !== idx || state.bag !== bag) return;
    const data = trackLayer.data();
    // Геометрия могла приехать для ДРУГОГО кадра: слой считает только последний
    // запрошенный. Просить тогда /labels своего кадра -- это сборка на сервере
    // (1-2.6 с) в тот момент, когда пользователь уже смотрит другой кадр.
    if (!data || (Number.isFinite(Number(data.frame)) && Number(data.frame) !== idx)) {
      return;
    }
    if (!state.labels) return;      // кадр отдан цветами: перекрашивать нечего
    return fetchLabels(bag, idx).then((buf) => {
      if (state.idx !== idx || state.bag !== bag) return;
      applyLabels(buf);
    });
  }).catch((e) => {
    // Перекраска -- уточнение раскраски, а не обязанность страницы: молча
    // оставляем кадр как есть (в консоль -- одна строка за сессию).
    warnLabelsOnce('labels: перекраска кадра не сделана —', e);
  });
}

// Буфер /labels: тот же заголовок, что у /frame, но kind всегда 1, а в `reserved`
// лежит номер кадра, по оси которого посчитана метка rail (axis_frame).
function applyLabels(buffer) {
  const u8 = new Uint8Array(buffer);
  for (let i = 0; i < 4; i++) {
    if (u8[i] !== FRAME_MARK[i]) throw new Error('bad labels magic');
  }
  const dv = new DataView(buffer);
  if (dv.getUint8(5) !== LF) throw new Error('bad labels kind');
  const axisFrame = dv.getUint16(6, true);
  const n = dv.getUint32(8, true);
  if (n !== state.numPoints) {
    // Разошлось число точек: позиции в posArr от другого набора -- метки не наши.
    warnLabelsOnce(`labels: точек ${n}, в кадре ${state.numPoints} — пропускаю`);
    return;
  }
  if (axisFrame !== state.idx) {
    // Ожидаемо только в вырожденном случае (геометрии кадра нет вовсе: /labels
    // ушёл в запасной путь), поэтому сообщаем один раз за сессию.
    warnLabelsOnce(`labels: метка rail посчитана по оси кадра ${axisFrame}, `
      + `а показан кадр ${state.idx}`);
  }
  const fits = Math.min(n, capacity);
  state.labels = new Uint8Array(buffer, 12, fits);
  state.labelFits = fits;
  const all = new Uint8Array(buffer, 12, n);
  const counts = new Array(meta.zone_colors.length).fill(0);
  for (let i = 0; i < n; i++) counts[all[i]]++;
  state.counts = counts;
  paintZoneColors();     // внутри и colAttr.needsUpdate, и подсветка туннеля
}

function prefetch(idx) {
  if (idx < 0 || idx >= meta.frames) return;
  // Геометрию кадра греем заранее -- тогда первый же /frame этого кадра получит
  // метку rail по своей оси (см. track3d.js: prefetch). С /frame, который сам
  // собирает ось кадра, прогрев стал ещё и способом не ждать сборку в момент
  // показа: prefetch(idx) качает /frame кадра, а тот собирает его геометрию, пока
  // показывается текущий -- к показу кадр отдаётся из кэша сервера.
  // Раньше при непрерывном воспроизведении прогрев выключался (`!state.loading`):
  // считалось, что он мешает текущему кадру. На практике наоборот: сборка идёт
  // ~0.4-0.7 с на кадр, воспроизведение требует 10 кадров/с, и без прогрева
  // геометрия КАЖДОГО нового кадра считается с нуля -- лента отстаёт на 5-10 кадров.
  // Со прогревом сервер считает следующий кадр, пока показывается текущий; в слое
  // при этом не больше одной предзагрузки в полёте (PREFETCH_MAX_INFLIGHT), так что
  // очередь не растёт.
  if (trackLayer && trackLayer.prefetch) {
    trackLayer.prefetch(meta.bag, idx);
  }
  if (cache.has(idx)) return;
  const p = fetchFrame(idx);
  p.catch(() => cache.delete(idx));   // битый запрос не оставляем в кэше
  cache.set(idx, p);
}

async function showFrame(idx) {
  idx = Math.max(0, Math.min(idx, meta.frames - 1));
  prefetch(idx + 1);
  prefetch(idx + 2);
  let p = cache.get(idx);
  if (!p) { p = fetchFrame(idx); }
  const buf = await p;
  cache.delete(idx);
  applyFrame(buf, idx);
  // Маска «заслонено» принадлежит кадру, в котором считан предпросмотр: без
  // разметки её быть не должно вовсе (иначе точки остались бы погашенными).
  if (!label.on && hiddenIdx.length) applyHiddenPoints([]);
  // Подсветка точек внутри туннеля -- сразу после applyFrame: цвета кадра только
  // что перезаписаны, и красить нужно уже их (порядок важен, не косметика).
  safeSafety('markPoints', (l) => l.markPoints(posArr, colArr, state.numPoints));
  colAttr.needsUpdate = true;
  state.idx = idx;
  $('frame').value = String(idx);
  updateFrameCopy();
  // Разметка живёт в путевой системе: на новом кадре объекты смещаются вдоль пути
  // на ход записи (или едут по траектории), поэтому и боксы, и меши поз, и числа в
  // панели пересчитываются на каждый кадр. Закрытые точки считаны по СВОЕМУ кадру,
  // поэтому записи других кадров уходят, а для объектов этого кадра предпросмотр
  // досчитывается -- но только когда воспроизведение на паузе: во время потока
  // кадров трассировка (десятки миллисекунд на объект) шла бы в каждом кадре.
  if (label.on) {
    labelDraw();
    renderLabel();
    ensurePreviews();
  }
  // Геометрия пути тоже зависит от кадра; слой сам решает, когда пересчитывать
  // (кэш по кадру и «считаем только последний запрошенный» -- build_track 1-2 с).
  if (trackLayer) {
    const loaded = trackLayer.load(meta.bag, idx);
    loaded.catch(fail);
    // Кадр пришёл раньше геометрии, значит его метка rail могла быть посчитана по
    // оси другого кадра. Как только геометрия ЭТОГО кадра готова -- просим метки
    // заново (лёгким маршрутом) и перекрашиваем точки.
    relabelAfterTrack(loaded, meta.bag, idx);
  }
  // Объекты из замеров -- отдельная ручка: сбой слоя не должен ронять кадр.
  if (objectsLayer) {
    objectsLayer.load(meta.bag, idx).catch(fail);
  }
}

// Одна короткая строка «что камера сделала и почему». Пишем в один элемент и
// здесь, и в слое объектов (вид по объектам, профиль, «чистое поле») -- чтобы
// перемещение камеры не выглядело самовольным.
function noteCamera(text) {
  const el = $('cameraNote');
  if (el) el.textContent = text;
}

// Вид по объектам пути -- это отдельный пресет «Объекты пути»: он же отмечается
// активным, когда слой ставит стартовую камеру (а не молча снимает подсветку).
// Ключ 'track' продублирован в web/track3d.js (TRACK_PRESET).
const PRESETS = [
  ['lidar', 'От лидара'], ['top', 'Сверху'], ['side', 'Сбоку'],
  ['iso', 'Изометрия'], ['behind', 'Сзади'], ['track', 'Объекты пути'], ['fit', 'Fit'],
];

function presetLabel(name) {
  const found = PRESETS.find(([key]) => key === name);
  return found ? found[1] : name;
}

// Ползунок «Дальность, м»: последний тик -- БЕЗ ОГРАНИЧЕНИЯ. Значение тика
// DIST_INF_TICK означает «∞»: в шейдер уходит большое число (точки приходят все,
// обрезания по дальности сервер в кадре не делает), а виду камеры оставляем
// разумный размах (DIST_PRESET_SPAN_M), иначе предустановки улетают на тысячи метров.
const DIST_INF_TICK = 205;
const DIST_PRESET_SPAN_M = 200;
const DIST_DEFAULT_M = 40;
const distIsInf = (v) => Number(v) >= DIST_INF_TICK;
const distValue = (v) => (distIsInf(v) ? 1e6 : Number(v));

function cameraPreset(name) {
  const cx = meta.axis_x;
  const ymid = -Math.min(state.maxDist, DIST_PRESET_SPAN_M) / 2;
  const P = {
    lidar: [[cx, 2.0, 1.6], [cx, -25, -0.5]],
    top: [[cx, ymid, 70], [cx, ymid, 0]],
    side: [[38, ymid, 0], [cx, ymid, 0]],
    iso: [[30, 22, 22], [cx, ymid, 0]],
    behind: [[cx, 12, 4], [cx, -25, -0.5]],
  };
  if (name === 'track') {
    // Вписывание по габариту объектов (рельсы + шпалы, окно 12 м). Подсветку и
    // строку «что с камерой» ставит сам слой.
    if (trackLayer && trackLayer.data()) {
      trackLayer.fitToTrack();
      for (const b of $('presets').children) {
        b.classList.toggle('active', b.dataset.preset === name);
      }
      return;
    }
    showTrackUnavailable(trackLayer ? new Error('меши пути ещё не пришли от /track')
      : createTrackLayerError);
    return;
  }
  if (name === 'fit') {
    // Вписывание по ПОПЕРЕЧНЫМ размерам габарита (X и Z), а не по его диагонали:
    // диагональ тянет камеру на десятки метров (пол -- плита ~199 м вдоль пути,
    // 40 м вперёд по Y), и рельс 0.156 м занимает единицы пикселей. Если объекты
    // пути уже загружены -- берём их габарит (рельсы + шпалы, без пола).
    if (trackLayer && trackLayer.data()) {
      trackLayer.fitToTrack(undefined, { mark: false });
      controls.update();
      for (const b of $('presets').children) {
        b.classList.toggle('active', b.dataset.preset === name);
      }
      noteCamera('камера: Fit по габариту объектов пути (рельсы + шпалы, окно 12 м)');
      return;
    }
    const n = state.numPoints || 1;
    const box = new THREE.Box3();
    const v = new THREE.Vector3();
    for (let i = 0; i < n; i += Math.max(1, Math.floor(n / 4000))) {
      v.set(posArr[i * 3], posArr[i * 3 + 1], posArr[i * 3 + 2]);
      box.expandByPoint(v);
    }
    if (box.isEmpty()) return;
    const c = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const fov = THREE.MathUtils.degToRad(camera.fov);
    const dist = Math.max(Math.max(size.x, size.z) / (2 * Math.tan(fov / 2) * 0.24), 3);
    camera.position.copy(c).addScaledVector(new THREE.Vector3(0.55, 0.60, 0.58).normalize(), dist);
    controls.target.copy(c);
    noteCamera('камера: Fit по габариту облака точек (объекты пути не загружены)');
  } else {
    const [pos, tgt] = P[name];
    camera.position.set(pos[0], pos[1], pos[2]);
    controls.target.set(tgt[0], tgt[1], tgt[2]);
    noteCamera(`камера: пресет «${presetLabel(name)}»`);
  }
  controls.update();
  for (const b of $('presets').children) b.classList.toggle('active', b.dataset.preset === name);
}

function updateHud() {
  const t = state.idx / Math.max(meta.hz, 0.01);
  // Компактный HUD вверху панели -- 1-2 строки; подробности ушли в свёрнутый
  // блок «Подробности кадра» (см. #hud ниже), чтобы не висеть простыней перед
  // настройками. Вторую строку (колея/головка/счётчики) пишет track3d.js.
  const hudFrame = $('hudFrame');
  if (hudFrame) {
    hudFrame.textContent =
      `${meta.bag} · кадр ${state.idx + 1}/${meta.frames} · точек ${state.numPoints}`
      + ` · ${state.lastFrameMs.toFixed(1)} мс`;
  }
  let s = `кадр ${state.idx + 1}/${meta.frames}   t = ${t.toFixed(1)} с\n`;
  s += `точек ${state.numPoints}   ${state.lastFrameMs.toFixed(1)} мс   ${state.fpsShown.toFixed(1)} fps\n`;
  s += `пол z = ${meta.floor.a} + ${meta.floor.b}·y  (${meta.floor.grade_pct} %)\n`;
  s += `ось x = ${meta.axis_x} м   колея ${meta.gauge_mm} мм   (из детектора рельсов)\n`;
  s += `стены x = ${meta.walls.join(', ')}\n`;
  s += `рельсов в модели ${Array.isArray(meta.rails && meta.rails.x) ? meta.rails.x.length : 0}`
    + `   x = ${(meta.rails && meta.rails.x) ? meta.rails.x.join(', ') : '—'}\n`;
  const bl = meta.blocked_spans;
  s += `разрывов ${bl.length}${bl.length ? ': ' + bl.map((p) => `${p[0]}…${p[1]} м`).join(', ') : ''}`;
  if (state.counts) {
    const parts = state.counts
      .map((c, i) => `${meta.zone_names[i]} ${c}`)
      .filter((_, i) => state.counts[i] > 0);
    s += '\n' + parts.join('   ');
  }
  $('hud').textContent = s;
}

// ------------------------------------------------------------------ разметка --
// Режим разметки: тот же поток, что в React-клиенте (app/client/src/api/label.ts,
// state/useLabeling.ts, panel/LabelPanel.tsx) -- клик по облаку -> /label/propose
// (кластер: под кликом, иначе ближайший рядом; плюс проверка детектором) -> правка
// чисел и ручек в панели ->
// /label/preview (точки трассировки) -> /label/save (карточка, .npz, index.jsonl).
// Предпросмотр запускается СРАЗУ после постановки -- и на автоматической ветке, и
// на ручной: число закрытых точек в панели и погашенные точки кадра появляются без
// «переключи класс, чтобы посчиталось». Предпросмотр считается по ВСЕМ объектам
// кадра (`label.previews`), а не по выбранному: закрытые точки складываются.
// Клиент ничего не считает по геометрии: все числа приходят с сервера, здесь они
// только показываются и правятся. Отрисовка -- в web/labellayer.js.
//
// Четыре вещи, которые правятся ЗДЕСЬ, а не в слое:
//   * основание объекта -- на поверхности клика (`z_ref` от сервера), поэтому
//     объект не «в пол»: центр = основание + высота/2, и все правки высоты идут
//     от опоры, а не от центра;
//   * точки кадра, заслонённые объектами (`removed_indices`), гасятся в облаке
//     (атрибут aHide): иначе за объектом остаётся «ряд точек вдаль»;
//   * ПОЗА: у человека рисуется запечённый меш (`assets/_poses`) -- момент клипа
//     выбирается по времени кадра (`labelPoseTime` + `poseMoment` из
//     labellayer.js, та же формула, что в `labeling.pose_moment`), а какая поза
//     играет, решает ПРАВИЛО (`labelScenePose`): с траекторией -- ходьба, без неё --
//     стояние;
//   * ТРАЕКТОРИЯ: две точки (начало и конец) и секунды, отсюда скорость; позиция на
//     кадре интерполируется по времени (`labelBaseAt`), а не по номеру кадра.

const LABEL_CLASSES = [['person', 'человек'], ['equipment', 'оборудование'], ['other', 'прочее']];
const LABEL_POSES = [['standing', 'стоит'], ['walking', 'идёт'], ['running', 'бежит'],
  ['lying', 'лежит'], ['fallen', 'упал'], ['unknown', 'неизвестно']];
const DEFAULT_POSE = { person: 'standing', equipment: 'unknown', other: 'unknown' };
const AUTOMATION_TEXT = {
  detector: 'детектор подтвердил', cluster: 'предложено кластером',
  manual: 'вручную (не предложено)',
};
const AUTOMATION_COLOR = { detector: 0x4ade80, cluster: 0xfacc15, manual: 0xf87171 };
const AUTOMATION_CSS = {
  detector: '#4ade80', cluster: '#facc15', manual: '#f87171',
};
const PREVIEW_CSS = '#38bdf8';

// Объект разметки в панели: геометрия в системе ОПОРНОГО кадра + трек + статус
// автоматики. Опорный кадр -- тот, в котором объект поставлен: правки чисел и
// предпросмотр считаются в нём, а на других кадрах объект проецируется ходом.
const label = {
  on: false,               // режим разметки (тумблер панели)
  motion: null,            // ход записи из /motion (м/кадр, км/ч)
  motionError: null,
  objects: [],
  selected: null,          // ключ выбранного объекта
  // Предпросмотр -- ПО ВСЕМ объектам кадра, не только по выбранному: закрытые точки
  // складываются, и при постановке второго объекта набор первого не теряется
  // (`web/labellayer.js:setPreviews`). Ключ -- ключ объекта.
  previews: new Map(),     // key -> {key, bag, frame, points, removed, counts, pose}
  previewError: null,
  previewSeq: 0,
  busy: false,
  dirty: false,           // есть несохранённые правки (смена записи/закрытие -- потеря)
  status: null,
  report: null,            // ответ /label/save
  seq: 0,                  // нумератор ключей объектов
  timer: null,             // дебаунс предпросмотра
  framesDraft: null,       // {key, text}: что человек набрал в поле «кадры»
  undo: [],                // снимки списка объектов для Ctrl+Z
  panelAll: false,         // в разметке показать всю панель, а не только разметку
  clips: new Map(),        // 'person|walking' -> {state, clip|error} -- клипы поз
  trajArm: null,           // 'start'|'end': следующий клик по облаку -- точка траектории
};

const labelHz = () => (meta && meta.hz > 0 ? meta.hz : 10);
const num3 = (v) => Array.isArray(v) ? [Number(v[0]), Number(v[1]), Number(v[2])] : [0, 0, 0];

function labelSelected() {
  return label.objects.find((o) => o.key === label.selected) || null;
}

/**
 * Выбрать объект. Набранный текст поля «кадры» живёт только со своим объектом:
 * при переходе на другой он сбрасывается (иначе у нового объекта показывался бы
 * текст, набранный для прежнего).
 */
function labelSelect(key) {
  label.selected = key;
  if (label.framesDraft && label.framesDraft.key !== key) label.framesDraft = null;
}

function labelSpeedKmh(o) {
  if (o && o.speedSource === 'manual' && Number.isFinite(o.speedKmh)) return o.speedKmh;
  return label.motion ? label.motion.kmh : 0;
}

/**
 * Ход трека, м/кадр: ЗАМЕРЕННЫЙ ход записи берём у сервера КАК ЕСТЬ
 * (`/motion.m_per_frame`): перевод через км/ч и частоту записи туда-обратно даёт
 * 0.6752 вместо замеренных 0.675, а в карточку и в панель должен уйти ровно тот ход,
 * который измерил сервер. Число из панели (км/ч) переводим по частоте записи, как
 * это делает `labeling`: `speed_m_per_frame = км/ч / 3.6 / hz`.
 *
 * Это ход СЕНСОРА, а не скорость объекта: у объекта с траекторией своя скорость
 * (`labelGeom.trajectorySpeed`), и она показывается отдельно -- смешивать их нельзя,
 * иначе позиция на кадре получила бы удвоенный ход.
 */
function labelSpeedM(o) {
  if (o && o.speedSource === 'manual') {
    return (Number.isFinite(o.speedKmh) ? o.speedKmh : 0) / 3.6 / labelHz();
  }
  if (label.motion && Number.isFinite(label.motion.m_per_frame)) {
    return label.motion.m_per_frame;
  }
  return 0;
}

/**
 * Ход записи на КАДРЕ из профиля `/motion.profile`, м/кадр (`null` -- профиля нет
 * или кадру нет значения: остаток совмещения плохой / алиас без проверки).
 *
 * Профиль -- это ход на каждый кадр записи, а не одно число: поезд разгоняется и
 * тормозит, на `squareT_platform_squareT_switch` он долго стоит. Одно число на
 * запись давало вдвое неверный ход у трёх записей (0.750 вместо 1.500 и т.п.).
 */
function labelMotionAt(o, frame) {
  if (o && o.speedSource === 'manual') return labelSpeedM(o);
  const prof = (label.motion && Array.isArray(label.motion.profile))
    ? label.motion.profile : null;
  if (!prof) return null;
  const f = Math.round(Number(frame));
  const v = (Number.isFinite(f) && f >= 0 && f < prof.length) ? prof[f] : null;
  return Number.isFinite(v) ? Number(v) : null;
}

/**
 * Смещение объекта вдоль пути между кадрами: ИНТЕГРАЛ профиля, а не `v * df`.
 *
 * Ровно то же, что делает сервер перед записью (`labeling.motion_between`):
 * сумма хода всех кадров между `fromFrame` и `toFrame`. Кадр без значения
 * (недостоверный остаток) берётся по медиане записи -- иначе разрыв профиля
 * обнулял бы ход и объект дёргался бы.
 */
function labelMotionBetween(o, fromFrame, toFrame) {
  const a = Math.round(Number(fromFrame));
  const b = Math.round(Number(toFrame));
  const fallback = labelSpeedM(o);
  if (!Number.isFinite(a) || !Number.isFinite(b) || a === b) return 0;
  const prof = (!o || o.speedSource !== 'manual')
    && label.motion && Array.isArray(label.motion.profile) ? label.motion.profile : null;
  if (!prof) return fallback * (b - a);
  const step = Math.max(1, Math.round(Number(label.motion.profile_step) || 1));
  const value = (f) => {
    const v = (f >= 0 && f < prof.length) ? prof[f] : null;
    return Number.isFinite(v) ? Number(v) : fallback;
  };
  let dy = 0;
  if (b >= a) {
    for (let f = a; f < b; f += step) dy += value(f);
  } else {
    for (let f = a - step; f >= b; f -= step) dy -= value(f);
  }
  return dy;
}

/**
 * Профиль хода для панели: сводка и сколько кадров достоверны. Одно число на
 * запись скрывало и разгон, и стоянку, поэтому панель показывает и профиль.
 */
function labelMotionSummary() {
  const m = label.motion;
  if (!m) return null;
  const prof = Array.isArray(m.profile) ? m.profile : null;
  const known = prof ? prof.filter((v) => Number.isFinite(v)) : [];
  return {
    median: Number.isFinite(m.m_per_frame) ? m.m_per_frame : null,
    kmh: Number.isFinite(m.kmh) ? m.kmh : null,
    residual: Number.isFinite(m.residual_m) ? m.residual_m : null,
    frames: prof ? prof.length : 0,
    known: known.length,
    reliable: Number.isFinite(m.reliable_frac) ? m.reliable_frac : null,
    here: prof ? labelMotionAt(null, state.idx) : null,
    flags: Array.isArray(m.profile_flags) ? m.profile_flags : null,
  };
}

/** Скорость объекта для показа, км/ч: у объекта с траекторией -- своя (расстояние
 * / секунды), иначе -- ход сенсора (панель показывает её в поле «Скорость трека»). */
function labelShownKmh(o) {
  if (o && o.traj && labelGeom) return labelGeom.trajectorySpeed(o.traj) * 3.6;
  return labelSpeedKmh(o);
}

/**
 * Центр объекта на кадре: y(f) = y_ref + v·(f − ref), положительный ход -- сенсор
 * приближается к объекту (`labeling._instance_at`). Поперёк и по высоте объект не
 * ползёт: он стоит на месте пути, а не на месте экрана.
 *
 * У объекта с ТРАЕКТОРИЕЙ позиция другая: он ЕДЕТ по ней, и точка берётся из
 * времени кадра (`labelBaseAt`), а не сдвигом на ход записи.
 */
function labelCenterAt(o, frame) {
  const base = labelBaseAt(o, frame);
  return [base[0], base[1], base[2] + o.size[1] / 2];
}

/**
 * Основание объекта (низ, опора) на кадре: траектория, если она есть, плюс ход
 * сенсора. Ровно та же формула, что `labeling._instance_center` при сохранении --
 * иначе карточка разошлась бы с тем, что видно в сцене.
 */
function labelBaseAt(o, frame) {
  const ref = Number.isFinite(o.refFrame) ? o.refFrame : frame;
  const zRef = Number.isFinite(o.zRef) ? o.zRef : o.center[2] - o.size[1] / 2;
  const dy = labelMotionBetween(o, ref, frame);
  if (o.traj && labelGeom) {
    return labelGeom.trajectoryBaseAtFrame(o.traj, ref, frame, labelHz(), labelSpeedM(o), dy);
  }
  return [o.center[0], o.center[1] + dy, zRef];
}

/** Время позы от опорного кадра объекта, с: индекс момента берётся по нему. */
function labelPoseTime(o, frame) {
  const ref = Number.isFinite(o.refFrame) ? o.refFrame : frame;
  return (frame - ref) / labelHz();
}

/**
 * Какая поза играет в СЦЕНЕ -- правило, а не ручная настройка на каждый кадр:
 * у объекта с траекторией движение выглядит идущим (`walking`), без неё -- стоящим
 * (`standing`); явные `lying`/`fallen`/`running` правилом не переписываются.
 * Та же функция в `labeling.pose_rule` -- сцена и карточка говорят одно и то же.
 */
function labelScenePose(o) {
  if (!o || !labelGeom) return o ? o.pose : 'unknown';
  return labelGeom.poseInScene(o.pose, Boolean(o.traj));
}

/** Концы траектории в системе ТЕКУЩЕГО кадра (ход записи учтён). */
function labelTrajectoryScene(o) {
  if (!o.traj || !labelGeom) return null;
  const dy = labelMotionBetween(o, o.refFrame, state.idx);
  const a = o.traj.start;
  const b = o.traj.end;
  return {
    start: [a[0], a[1] + dy, a[2]],
    end: [b[0], b[1] + dy, b[2]],
    seconds: o.traj.seconds,
  };
}

// ------------------------------------------------------------------ клипы поз
//
// Клип берётся у сервера ОДИН раз на (класс, позу) и живёт в кэше: 18 моментов
// ходьбы -- 580 КБ бинаря, дальше по кадру едет только время, и воспроизведение
// не ходит на сервер за каждым кадром позы. Выбор момента -- `poseMoment` из
// labellayer.js, та же формула, что в `labeling.pose_moment`.

/** base64 -> Float32Array/Uint32Array: ровно по своему байтовому буферу. */
function labelB64Floats(text) {
  const bin = atob(String(text || ''));
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

function labelB64Uints(text) {
  const bin = atob(String(text || ''));
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Uint32Array(bytes.buffer);
}

const labelClipKey = (cls, pose) => `${cls}|${pose}`;

/**
 * Клип позы класса и позы из кэша; нет -- запрашиваем у сервера (`/label/poses`).
 * Возвращает {state:'ok'|'loading'|'error', clip|error} -- панель по этому состоянию
 * говорит, что происходит, а слой рисует примитив/каркас, пока клипа нет.
 */
function labelClipFor(cls, pose) {
  if (!labelGeom) return { state: 'error', error: 'нет labellayer.js' };
  const key = labelClipKey(cls, pose);
  const hit = label.clips.get(key);
  if (hit) return hit;
  const slot = { state: 'loading' };
  label.clips.set(key, slot);
  const bag = state.bag;
  const url = `/label/poses?class=${encodeURIComponent(cls)}&pose=${encodeURIComponent(pose)}`
    + (bag ? `&bag=${encodeURIComponent(bag)}` : '');
  fetch(url, { cache: 'no-store' })
    .then(async (res) => {
      const body = await res.json();
      if (!res.ok || !body.ok) {
        throw new Error(body && body.error ? body.error : `${url} -> ${res.status}`);
      }
      const vertices = labelB64Floats(body.vertices_b64);
      const faces = labelB64Uints(body.faces_b64);
      if (!vertices.length || !faces.length) throw new Error('пустой клип позы');
      slot.clip = {
        vertices, faces,
        times: Float32Array.from(body.times_s || []),
        periodS: Number(body.period_s) || 0,
        play: body.play || 'one_shot',
        count: Number(body.vertex_count) || 0,
        meta: body,
      };
      slot.state = 'ok';
    })
    .catch((e) => {
      slot.state = 'error';
      slot.error = String((e && e.message) || e);
    })
    .then(() => {
      // Клип пришёл -- перерисовать сцену и панель (кадр при этом не менялся).
      if (label.on) {
        labelDraw();
        renderLabel();
      }
    });
  return slot;
}

/** Меш позы объекта на кадре: {positions, faces, clip, moment, blend, label}. */
function labelPoseMesh(o, frame) {
  if (!labelGeom || o.cls !== 'person') return { pose: null, info: null };
  const scenePose = labelScenePose(o);
  const slot = labelClipFor('person', scenePose);
  if (slot.state !== 'ok') return { pose: null, info: { scenePose, state: slot.state,
                                                         error: slot.error || null } };
  const clip = slot.clip;
  const t = labelPoseTime(o, frame);
  const verts = labelGeom.poseVerticesAt(clip, t);
  const base = labelBaseAt(o, frame);
  const placed = labelGeom.placePose(verts, {
    center: [base[0], base[1]], baseZ: base[2],
    yaw: o.yaw, pitch: o.pitch, roll: o.roll,
  });
  const moment = labelGeom.poseMoment(clip.times, clip.periodS, clip.play, t);
  const samples = clip.times.length;
  return {
    pose: {
      positions: placed.positions, faces: clip.faces, source: 'poses',
      clip: clip.meta.clip, asset: clip.meta.asset, moment: moment.index,
      blend: moment.blend, timeInClip: moment.timeInClip, samples,
      lift: placed.lift,
      label: `${clip.meta.clip} · момент ${moment.index + 1}/${samples}`
        + ` · t=${moment.timeInClip.toFixed(2)} с`,
    },
    info: {
      scenePose, state: 'ok', clip: clip.meta.clip, asset: clip.meta.asset,
      play: clip.play, samples, moment: moment.index, blend: moment.blend,
      time: t, timeInClip: moment.timeInClip, heightM: clip.meta.height_m,
      vertices: clip.count, faces: clip.meta.face_count, lift: placed.lift,
      bbox: poseBbox(placed.positions),
    },
  };
}

function poseBbox(positions) {
  let xlo = Infinity; let xhi = -Infinity; let ylo = Infinity; let yhi = -Infinity;
  let zlo = Infinity; let zhi = -Infinity;
  for (let k = 0; k < positions.length; k += 3) {
    const x = positions[k]; const y = positions[k + 1]; const z = positions[k + 2];
    if (x < xlo) xlo = x;
    if (x > xhi) xhi = x;
    if (y < ylo) ylo = y;
    if (y > yhi) yhi = y;
    if (z < zlo) zlo = z;
    if (z > zhi) zhi = z;
  }
  return { min: [xlo, ylo, zlo], max: [xhi, yhi, zhi],
           size: [xhi - xlo, yhi - ylo, zhi - zlo] };
}

/**
 * Объект для слоя: бокс на текущем кадре + меш позы + траектория. Слой только
 * рисует: всю геометрию (место, момент клипа, концы траектории) считает эта
 * функция, и теми же формулами, что сервер при сохранении.
 */
function labelLayerItem(o) {
  const got = labelPoseMesh(o, state.idx);
  const base = labelBaseAt(o, state.idx);
  return {
    key: o.key,
    center: [base[0], base[1], base[2] + o.size[1] / 2],
    size: o.size,
    yaw: o.yaw,
    pitch: o.pitch,
    roll: o.roll,
    color: AUTOMATION_COLOR[o.automation] !== undefined
      ? AUTOMATION_COLOR[o.automation] : AUTOMATION_COLOR.manual,
    selected: o.key === label.selected,
    pose: got.pose,
    poseInfo: got.info,
    traj: labelTrajectoryScene(o),
  };
}

/** Дальность объекта (вдоль пути, м) на кадре: вдоль пути растёт вперёд, это -y. */
function labelRangeAt(o, frame) {
  return -labelCenterAt(o, frame)[1];
}

async function labelPost(path, body) {
  const url = state.bag ? `${path}?bag=${encodeURIComponent(state.bag)}` : path;
  const res = await fetch(url, {
    method: 'POST',
    cache: 'no-store',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let text = `${path} -> ${res.status}`;
    try {
      const body2 = await res.json();
      if (body2 && body2.error) text = body2.error;
    } catch (e) { /* тело не JSON -- оставляем код */ }
    throw new Error(text);
  }
  return res.json();
}

function setLabelStatus(text) {
  label.status = text || null;
  renderLabelStatus();
}

// Ход записи: спрашиваем на каждую запись (замер на сервере кэширован). Ответ
// держим вместе с именем записи: иначе после переключения старый ход ещё жил бы
// в состоянии и трек считался бы по чужому замеру.
async function labelLoadMotion() {
  const bag = state.bag;
  const url = bag ? `/motion?bag=${encodeURIComponent(bag)}` : '/motion';
  try {
    const res = await fetch(url, { cache: 'no-store' });
    const body = await res.json();
    if (state.bag !== bag) return;
    if (!res.ok) throw new Error(body && body.error ? body.error : `${url} -> ${res.status}`);
    label.motion = body.motion;
    label.motionError = null;
  } catch (e) {
    if (state.bag !== bag) return;
    label.motion = null;
    label.motionError = String((e && e.message) || e);
  }
  renderLabel();
}

// Смена записи: объекты принадлежат записи, в которой поставлены, поэтому их не
// переносим -- начинаем с пустого списка, а ход записи спрашиваем заново.
function labelOnBag() {
  label.objects = [];
  label.selected = null;
  label.previews.clear();
  label.previewError = null;
  label.report = null;
  label.status = null;
  label.framesDraft = null;
  label.motion = null;
  label.motionError = null;
  label.undo = [];
  label.trajArm = null;
  label.previewSeq += 1;
  label.busy = false;     // летевший предпросмотр уже никому не ответит
  applyHiddenPoints([]);
  labelDraw();
  labelLoadMotion();
}

// ------------------------------------------------------------ объекты и правка

// Отмена (Ctrl+Z): снимки списка объектов ДО правки. Правки мелкие и частые
// (перетаскивание -- это десятки вызовов), поэтому копия делается по объекту, а
// не по ссылке, а глубина ограничена: разметка -- инструмент, а не редактор.
const LABEL_UNDO_MAX = 60;

function labelSnapshot() {
  return { objects: label.objects.map((o) => ({ ...o, center: o.center.slice(),
                                                size: o.size.slice(),
                                                frames: o.frames.slice() })),
           selected: label.selected };
}

function labelPushUndo() {
  label.dirty = true;      // любая правка делает разметку несохранённой
  label.undo.push(labelSnapshot());
  if (label.undo.length > LABEL_UNDO_MAX) label.undo.shift();
}

function labelUndoLast() {
  const snap = label.undo.pop();
  if (!snap) {
    setLabelStatus('отменять нечего');
    return false;
  }
  label.objects = snap.objects;
  label.selected = snap.selected && snap.objects.some((o) => o.key === snap.selected)
    ? snap.selected : (snap.objects.length ? snap.objects[snap.objects.length - 1].key : null);
  label.framesDraft = null;
  labelDraw();
  renderLabel();
  schedulePreview();
  setLabelStatus('отменено (осталось шагов: ' + label.undo.length + ')');
  return true;
}

/** Есть ли фокус в поле ввода: тогда горячие клавиши не наши. */
function typingInField() {
  const el = document.activeElement;
  if (!el) return false;
  const tag = String(el.tagName || '').toLowerCase();
  return tag === 'input' || tag === 'select' || tag === 'textarea';
}

/**
 * Габарит объекта, поставленного РУКАМИ: ОСНОВАНИЕ -- в точке клика (на той
 * поверхности, куда попали), центр -- на полвысоты выше. Раньше центр клался
 * РОВНО в точку клика, и объект уходил в пол на полвысоты -- ровно то, на что
 * жаловался заказчик. `size` и `zRef` приходят от сервера вместе с предложением
 * (там же и пол под кликом): одна геометрия на обе ветки -- предложенную и ручную.
 *
 * `size_source` сервер отдаёт вместе с габаритом: `cluster` -- по найденному
 * кластеру, `band` -- по точкам рядом с кликом, `default` -- мерить было нечего
 * (жёсткий запас). Панель показывает это словами: «габарит по умолчанию» не
 * должен выглядеть измеренным.
 */
function labelManualObject(point, got) {
  // Габарит и опору берём у сервера, только если он их действительно прислал:
  // у старого сервера (до этой правки) `click`/`z_ref` нет, а `size` -- минимумы
  // (0.3×0.25×0.2), то есть бокс-крошка вместо объекта. Признак нового ответа --
  // точка клика в нём.
  const fromServer = Boolean(got && got.click && Array.isArray(got.size)
    && got.size.length === 3);
  const size = fromServer ? num3(got.size) : [0.6, 1.0, 1.0];
  const sizeSource = got && typeof got.size_source === 'string'
    ? got.size_source : (fromServer ? 'band' : 'default');
  const zRef = fromServer && Number.isFinite(Number(got.z_ref))
    ? Number(got.z_ref) : Number(point[2]);
  const floorZ = got && got.surface && Number.isFinite(Number(got.surface.floor_z_m))
    ? Number(got.surface.floor_z_m) : null;
  return {
    cls: 'equipment',
    pose: 'unknown',
    center: [Number(point[0]), Number(point[1]), zRef + size[1] / 2],
    size,
    sizeSource,
    zRef,
    floorZ,
    yaw: got && Number.isFinite(Number(got.yaw)) ? Number(got.yaw) : 0,
    pitch: 0,
    roll: 0,
    automation: 'manual',
    reason: got && got.reason
      ? `автоматика не предложила: ${got.reason}`
      : 'поставлено вручную — автоматика объект не предложила',
  };
}

function labelAddManual(point, got) {
  labelPushUndo();
  const base = labelManualObject(point, got);
  const item = {
    key: `lbl-${++label.seq}`,
    id: null,
    refFrame: state.idx,
    frames: [state.idx],
    speedKmh: null,
    speedSource: 'measured',
    ...base,
  };
  label.objects.push(item);
  labelSelect(item.key);
  setLabelStatus(`объект поставлен вручную: основание на поверхности `
    + `z=${fmt3(item.zRef)} м, центр z=${fmt3(item.center[2])} м · `
    + (item.sizeSource === 'default'
      ? 'габарит не измерен (задайте руками или «Подогнать по точкам»)'
      : 'габарит по точкам рядом')
    + ' · число закрытых точек — в строке «предпросмотр»');
  labelDraw();
  renderLabel();
  // Предпросмотр трассировки -- СРАЗУ, как и на автоматической ветке: иначе точки
  // объекта считались бы только после смены класса, и панель молчала бы о том,
  // сколько лучей кадра попало в бокс (ровно эта жалоба и была).
  schedulePreview();
}

// Клик по облаку: габарит предлагает сервер. Автоматика не предложила (кластера нет
// ни под кликом, ни рядом) -- объект встаёт руками в ту же точку, красным, и его
// габарит сервер тоже считает по точкам рядом (или честно помечает как неизмеренный).
//
// Если в панели взведена кнопка «точка начала/конца» траектории -- тот же клик
// ставит точку траектории, а объект не появляется: траектория ставится по облаку
// ровно так же, как объект, и кликать по тому же месту дважды незачем.
function labelPick(point) {
  if (label.trajArm) {
    if (!labelSelected()) {
      label.trajArm = null;
      setLabelStatus('точка траектории не поставлена: сначала выберите объект');
      renderLabel();
      return;
    }
    labelSetTrajPoint(label.trajArm, point);
    return;
  }
  labelProposeAt(point).catch(fail);
}

async function labelProposeAt(point) {
  const bag = state.bag;
  setLabelStatus('предложение по клику…');
  try {
    const got = await labelPost('/label/propose', { frame: state.idx, click: point });
    // Запись могла смениться, пока шёл запрос: объект чужой записи не ставим --
    // его координаты в системе лидара другой записи, и он бы «висел» в воздухе.
    if (state.bag !== bag) return;
    if (!got.ok || got.automation === 'manual') {
      labelAddManual(point, got);
      return;
    }
    labelPushUndo();
    const item = {
      key: `lbl-${++label.seq}`,
      id: null,
      cls: got.pose_suggest || 'equipment',
      pose: got.pose_suggest === 'person' ? 'standing' : 'unknown',
      refFrame: Math.round(Number(got.frame)) || state.idx,
      frames: [Math.round(Number(got.frame)) || state.idx],
      center: num3(got.center),
      size: num3(got.size),
      sizeSource: typeof got.size_source === 'string' ? got.size_source : 'cluster',
      // Опорная поверхность: её вернул сервер вместе с точкой клика. Панель
      // показывает её как «низ», и по ней же считается высота при правках.
      // Запасной путь -- низ предложенного бокса, а не его центр.
      zRef: Number.isFinite(Number(got.z_ref)) ? Number(got.z_ref)
        : num3(got.center)[2] - num3(got.size)[1] / 2,
      // Пол под объектом: низ ручкой ниже него не опускается (сервер присылает
      // его вместе с опорой).
      floorZ: got.surface && Number.isFinite(Number(got.surface.floor_z_m))
        ? Number(got.surface.floor_z_m) : null,
      yaw: Number(got.yaw) || 0,
      pitch: Number(got.pitch) || 0,
      roll: Number(got.roll) || 0,
      automation: got.automation,
      reason: got.reason,
      speedKmh: null,
      speedSource: 'measured',
    };
    label.objects.push(item);
    labelSelect(item.key);
    setLabelStatus(`${got.reason} · основание z=${fmt3(item.zRef)} м `
      + `(клик z=${fmt3(point[2])} м)`);
    labelDraw();
    renderLabel();
    schedulePreview();
  } catch (e) {
    setLabelStatus(String((e && e.message) || e));
  }
}

/**
 * «Подогнать по точкам»: переспросить габарит у сервера, целясь в ЦЕНТР самого
 * бокса. Ручная постановка перестаёт быть слепой: если объект рядом есть, клик по
 * его центру находит кластер, и бокс обтягивается по точкам. Класс и поза уже
 * выбраны человеком -- их подгонка не переписывает.
 */
async function labelRefit() {
  const o = labelSelected();
  if (!o) {
    setLabelStatus('подгонять нечего: объект не выбран');
    return;
  }
  const bag = state.bag;
  setLabelStatus('подгоняю бокс по точкам рядом…');
  try {
    // Клик -- центр бокса в ЕГО опорном кадре: /label/propose принимает точку в
    // системе лидара того кадра, для которого спрашивают, и центр хранится именно
    // в опорном (на другом кадре объект смещён ходом записи).
    const got = await labelPost('/label/propose',
      { frame: o.refFrame, click: [o.center[0], o.center[1], o.center[2]] });
    if (state.bag !== bag) return;
    if (!got.ok) {
      setLabelStatus(`подогнать не удалось: ${got.reason}`);
      return;
    }
    const size = num3(got.size);
    const zRef = Number.isFinite(Number(got.z_ref)) ? Number(got.z_ref)
      : num3(got.center)[2] - size[1] / 2;
    labelPatch(o.key, {
      // Центр -- как у предложенного бокса: сервер уже поставил его на опору
      // (`z_ref + высота/2`), и пересчитывать его здесь значило бы гадать.
      center: num3(got.center),
      size,
      sizeSource: typeof got.size_source === 'string' ? got.size_source : 'cluster',
      zRef,
      floorZ: got.surface && Number.isFinite(Number(got.surface.floor_z_m))
        ? Number(got.surface.floor_z_m) : o.floorZ,
      yaw: Number(got.yaw) || 0,
      pitch: Number(got.pitch) || 0,
      roll: Number(got.roll) || 0,
      automation: got.automation,
      reason: `подогнано по точкам: ${got.reason}`,
    });
    setLabelStatus(`подогнано по точкам: ${got.reason} · `
      + `габарит ${fmt3(size[0])}×${fmt3(size[1])}×${fmt3(size[2])} м`);
  } catch (e) {
    setLabelStatus(String((e && e.message) || e));
  }
}

function labelPatch(key, next, options) {
  const item = label.objects.find((o) => o.key === key);
  if (!item) return;
  if (!(options && options.silent)) labelPushUndo();
  Object.assign(item, next);
  // Бокс перерисовывается СРАЗУ: иначе правка видна только после ответа сервера
  // (предпросмотр трассировки -- сотни миллисекунд), и «тяну за грань, а ничего не
  // движется» -- это именно оно.
  labelDraw();
  renderLabel();
  schedulePreview();
}

function labelRemove(key) {
  const item = label.objects.find((o) => o.key === key);
  if (!item) return;
  labelPushUndo();
  label.objects = label.objects.filter((o) => o.key !== key);
  // Предпросмотр удалённого объекта убираем сразу: его закрытые точки больше не
  // закрыты (иначе набор остался бы в маске до следующего пересчёта).
  label.previews.delete(key);
  if (label.selected === key) {
    label.selected = null;
    label.trajArm = null;
  }
  labelDraw();
  renderLabel();
  // Остальные объекты кадра пересчитываем: наборы закрытых точек складываются
  // по всем, и удаление одного меняет общую картину.
  schedulePreview();
  setLabelStatus(`объект удалён (Ctrl+Z вернёт)`);
}

// ------------------------------------------------------- правка мышью (ручки)

/**
 * Точка траектории: начало или конец. Точку даёт клик по облаку (как постановка
 * объекта) или перетаскивание конца мышью (`{kind:'traj'}` от слоя).
 *
 * Точка приходит в системе ТЕКУЩЕГО кадра, а траектория хранится в системе ОПОРНОГО
 * кадра объекта -- поэтому y переводится тем же ходом, каким объект проецируется на
 * кадр. Высота: клик по облаку берёт опору не выше основания объекта (клик по полу
 * даёт пол, клик по телу -- саму опору); при перетаскивании высота конца остаётся
 * своей (её задали, когда ставили точку).
 *
 * Ставя НАЧАЛО, объект переносится в него: у объекта с траекторией позиция на
 * опорном кадре -- это начало траектории (`labeling._instance_center`, t = 0).
 */
function labelSetTrajPoint(which, point, options) {
  const o = labelSelected();
  if (!o || !labelGeom) return;
  const fromDrag = Boolean(options && options.fromDrag);
  const dy = labelMotionBetween(o, state.idx, o.refFrame);
  const size = o.size;
  const zRef = Number.isFinite(o.zRef) ? o.zRef : o.center[2] - size[1] / 2;
  const base = [Number(point[0]), Number(point[1]) + dy,
                fromDrag ? Number(point[2]) : Math.min(Number(point[2]), zRef)];
  const own = [o.center[0], o.center[1], zRef];
  const old = o.traj || { start: own, end: own, seconds: 0 };
  const start = which === 'start' ? base : old.start;
  const end = which === 'end' ? base : old.end;
  const distance = labelGeom.trajectoryDistance({ start, end });
  // Секунды по умолчанию -- по скорости пешехода 1.4 м/с: правдоподобный ход, а
  // человек всё равно правит секунды числом.
  const seconds = o.traj && Number.isFinite(o.traj.seconds) && o.traj.seconds > 0
    ? o.traj.seconds : Math.max(0.5, distance / 1.4);
  const traj = { start, end, seconds, source: 'manual' };
  const next = { traj };
  if (which === 'start') {
    next.center = [start[0], start[1], start[2] + size[1] / 2];
    next.zRef = start[2];
  }
  labelPatch(o.key, next, options);
  label.trajArm = null;
  const speed = labelGeom.trajectorySpeed(traj);
  setLabelStatus(`траектория: ${which === 'start' ? 'начало' : 'конец'} на `
    + `[${base.map((v) => v.toFixed(2)).join(', ')}] · ${distance.toFixed(2)} м за `
    + `${seconds.toFixed(1)} с → ${speed.toFixed(2)} м/с (${(speed * 3.6).toFixed(1)} км/ч)`
    + ' · поза в сцене выбирается правилом: есть траектория → идёт');
}

/** Секунды траектории числом: скорость пересчитывается (расстояние не меняется). */
function labelSetTrajSeconds(value) {
  const o = labelSelected();
  if (!o || !o.traj || !Number.isFinite(value) || value <= 0) return;
  labelPatch(o.key, { traj: { ...o.traj, seconds: value } });
  setLabelStatus(`траектория: ${o.traj.seconds.toFixed(2)} с → `
    + `${labelGeom.trajectorySpeed({ ...o.traj, seconds: value }).toFixed(2)} м/с`);
}

/** Убрать траекторию: объект снова стоит, поза в сцене возвращается к `standing`. */
function labelClearTraj() {
  const o = labelSelected();
  if (!o || !o.traj) {
    setLabelStatus('траектории у объекта нет');
    return;
  }
  labelPushUndo();
  labelPatch(o.key, { traj: null }, { silent: true });
  setLabelStatus('траектория убрана: объект снова стоит на месте');
}

/**
 * Правка ручкой от слоя: размер, высота, поворот, конец траектории. Геометрия
 * считается ЗДЕСЬ (слой знает только, где ручка и куда её потянули), и каждое поле
 * сразу уходит в бокс, предпросмотр и карточку.
 *
 *   {kind:'size', axis:'x'|'y', size_m} -- ширина поперёк / длина вдоль;
 *   {kind:'height', edge:'top'|'bottom', z_m} -- высота: тянем верх -- низ стоит,
 *     тянем низ -- стоит верх (иначе объект «уезжает» с опорной поверхности);
 *   {kind:'yaw', yaw_deg} -- поворот вокруг вертикали;
 *   {kind:'traj', which:'start'|'end', point} -- конец траектории (шаг отмены уже
 *     сделан в onDragStart, поэтому здесь `silent`).
 */
function labelApplyEdit(edit) {
  const o = labelSelected();
  if (!o || !edit) return;
  if (edit.kind === 'traj' && edit.point) {
    labelSetTrajPoint(edit.which === 'start' ? 'start' : 'end', edit.point, { silent: true, fromDrag: true });
    return;
  }
  if (edit.kind === 'size') {
    const value = Math.max(0.05, Number(edit.size_m) || 0);
    if (!Number.isFinite(value)) return;
    const size = o.size.slice();
    if (edit.axis === 'x') size[0] = value;
    else size[2] = value;
    labelPatch(o.key, { size });
    return;
  }
  if (edit.kind === 'height') {
    const size = o.size.slice();
    const base = Number.isFinite(o.zRef) ? o.zRef : o.center[2] - size[1] / 2;
    const top = base + size[1];
    // Низ не опускаем НИЖЕ ПОЛА: «объект в полу» -- ровно то, на что жаловался
    // заказчик. Пол пришёл с сервера вместе с точкой клика (`surface.floor_z_m`).
    const floorZ = Number.isFinite(o.floorZ) ? o.floorZ : -Infinity;
    const bottomNew = edit.edge === 'bottom'
      ? Math.max(floorZ, Math.min(Number(edit.z_m), top - 0.05)) : base;
    const topNew = edit.edge === 'top'
      ? Math.max(Number(edit.z_m), base + 0.05) : top;
    const value = Math.max(0.05, topNew - bottomNew);
    if (!Number.isFinite(value)) return;
    size[1] = value;
    // Верх тянем -- низ остаётся опорой; низ тянем -- верх остаётся, а опора
    // переезжает (объект продолжает стоять на том, что под ним).
    const zRef = edit.edge === 'top' ? base : bottomNew;
    labelPatch(o.key, { size, zRef, center: [o.center[0], o.center[1], zRef + value / 2] });
    return;
  }
  if (edit.kind === 'yaw') {
    labelPatch(o.key, { yaw: Number(edit.yaw_deg) || 0 });
  }
}

function labelNudge(dx, dy) {
  const o = labelSelected();
  if (!o) return;
  labelPatch(o.key, { center: [o.center[0] + dx, o.center[1] + dy, o.center[2]] });
}

function labelRotate(delta) {
  const o = labelSelected();
  if (!o) return;
  let yaw = (Number(o.yaw) || 0) + delta;
  yaw = ((yaw + 180) % 360 + 360) % 360 - 180;
  labelPatch(o.key, { yaw });
  setLabelStatus(`поворот: рыскание ${fmt3(yaw)}°`);
}

function labelResize(delta) {
  const o = labelSelected();
  if (!o) return;
  const size = o.size.slice();
  size[1] = Math.max(0.05, size[1] + delta);
  const zRef = Number.isFinite(o.zRef) ? o.zRef : o.center[2] - o.size[1] / 2;
  labelPatch(o.key, { size, zRef, center: [o.center[0], o.center[1], zRef + size[1] / 2] });
  setLabelStatus(`высота ${fmt3(size[1])} м (низ стоит на z=${fmt3(zRef)} м)`);
}

// ------------------------------------------------------------------ предпросмотр

// Дебаунс: правка чисел и перетаскивание мышью не должны занимать сервер на
// каждый пиксель (трассировка лучей кадра -- самая дорогая ручка разметки).
function schedulePreview() {
  if (label.timer) clearTimeout(label.timer);
  label.timer = setTimeout(() => { label.timer = null; refreshPreviews(); }, 140);
}

/** Объекты ТЕКУЩЕГО кадра: по их кадрам (`frames`) -- то же, что список в панели. */
function labelPreviewTargets() {
  return label.objects.filter((o) => (o.frames || []).includes(state.idx));
}

/** Подпись запроса: по ней видно, что предпросмотр объекта устарел. */
function labelPreviewSignature(o) {
  return JSON.stringify([
    state.bag, state.idx, o.cls, labelScenePose(o), labelPoseTime(o, state.idx),
    o.center, o.size, o.yaw, o.pitch, o.roll, labelBaseAt(o, state.idx),
  ]);
}

/**
 * Предпросмотр по ВСЕМ объектам кадра, а не по одному выбранному.
 *
 * Раньше он хранился для выбранного (`label.preview` + `previewFor`), и постановка
 * второго объекта стирала набор первого -- ровно жалоба «когда я поставил вторую
 * точку, не пересчитались другие точки, которые за ним». Теперь у каждого объекта
 * кадра своя запись (`label.previews`), пересчёт идёт при ЛЮБОМ изменении
 * (постановка, перенос, размер, поза, класс, удаление), а закрытые точки
 * складываются по всем объектам в слое (`setPreviews` -> `onHidden`).
 *
 * Записи принадлежат кадру: индексы закрытых точек считаны по ЭТОМУ кадру, поэтому
 * на другом кадре они не применяются, а сам предпросмотр считается заново --
 * но только для объектов, которые в этом кадре размечены (`frames`).
 */
async function refreshPreviews({ force = false } = {}) {
  if (!label.on) {
    label.previews.clear();
    labelDraw();
    renderLabelPreview();
    return;
  }
  for (const key of [...label.previews.keys()]) {
    const entry = label.previews.get(key);
    const o = label.objects.find((x) => x.key === key);
    if (!o || entry.frame !== state.idx || entry.bag !== state.bag) label.previews.delete(key);
  }
  const targets = labelPreviewTargets();
  const pending = [];
  for (const o of targets) {
    const sig = labelPreviewSignature(o);
    const hit = label.previews.get(o.key);
    if (!force && hit && hit.signature === sig) continue;
    pending.push({ o, sig });
  }
  if (!pending.length) {
    labelDraw();
    renderLabelPreview();
    return;
  }
  const seq = ++label.previewSeq;
  label.busy = true;
  renderLabelPreview();
  let failed = 0;
  for (const { o, sig } of pending) {
    const base = labelBaseAt(o, state.idx);
    try {
      const got = await labelPost('/label/preview', {
        // Кадр -- ТЕКУЩИЙ: и центр, и время позы берутся на нём, поэтому маска
        // закрытых точек принадлежит именно этому кадру (и рисуемым точкам кадра).
        frame: state.idx, class: o.cls, pose: labelScenePose(o),
        center: [base[0], base[1], base[2] + o.size[1] / 2],
        size: o.size, yaw: o.yaw, pitch: o.pitch, roll: o.roll,
        pose_time_s: labelPoseTime(o, state.idx),
        base_z: base[2],
      });
      if (seq !== label.previewSeq) return;
      const pts = Array.isArray(got.points) ? got.points : [];
      const arr = new Float32Array(pts.length * 3);
      for (let i = 0; i < pts.length; i++) {
        arr[i * 3] = pts[i][0];
        arr[i * 3 + 1] = pts[i][1];
        arr[i * 3 + 2] = pts[i][2];
      }
      label.previews.set(o.key, {
        key: o.key, bag: state.bag, frame: state.idx, signature: sig,
        points: arr,
        // Точки считаны в системе ЭТОГО кадра: сдвигать их не нужно (раньше
        // предпросмотр считался в опорном кадре и сдвигался ходом при отрисовке).
        removed: Array.isArray(got.removed_indices) ? got.removed_indices : [],
        counts: got.counts || {},
        n_points: got.n_points || 0,
        pose: got.pose_info || null,
        meshBbox: got.mesh_bbox || null,
        poseTime: Number(got.pose_info && got.pose_info.pose_time_s) || 0,
        path: got.path || null,
        confirmed: Boolean(got.confirmed),
        detector: got.detector || null,
      });
      label.previewError = null;
    } catch (e) {
      if (seq !== label.previewSeq) return;
      failed += 1;
      label.previews.delete(o.key);
      label.previewError = String((e && e.message) || e);
    }
  }
  if (seq !== label.previewSeq) return;
  label.busy = false;
  if (failed) setLabelStatus(`предпросмотр: ошибок ${failed} (${label.previewError})`);
  labelDraw();
  renderLabelPreview();
  renderLabel();
}

/** Совместимость с прежним именем (одна перерисовка «выбранного» предпросмотра). */
function refreshPreview() {
  return refreshPreviews({ force: true });
}

/**
 * Досчитать предпросмотр для объектов показанного кадра, если его нет.
 *
 * Вызывается при смене кадра. Воспроизведение идёт -- пропускаем: трассировка
 * стоит десятки миллисекунд на объект, и на 10 Гц она бы занимала сервер целиком.
 * На паузе (пошаговый просмотр, правка) -- считаем: без этого после перехода на
 * кадр объекта закрытые точки не гасились бы, пока что-нибудь не поправишь.
 */
function ensurePreviews() {
  if (!label.on || state.playing) return;
  const targets = labelPreviewTargets();
  const missing = targets.some((o) => {
    const entry = label.previews.get(o.key);
    return !entry || entry.frame !== state.idx || entry.signature !== labelPreviewSignature(o);
  });
  if (missing) schedulePreview();
}

/** Запись предпросмотра выбранного объекта (панель показывает её числа). */
function labelPreviewOf(o) {
  return o ? label.previews.get(o.key) || null : null;
}

/** Есть ли предпросмотр по всем объектам кадра (для строки «всего закрыто»). */
function labelHiddenTotal() {
  const seen = new Set();
  for (const entry of label.previews.values()) {
    if (entry.frame !== state.idx || entry.bag !== state.bag) continue;
    for (const i of entry.removed || []) seen.add(i);
  }
  return seen.size;
}

// --------------------------------------------------------------------- запись

async function labelSave() {
  if (!label.objects.length) {
    setLabelStatus('сохранять нечего: поставьте хотя бы один объект');
    return;
  }
  setLabelStatus('запись в labels/…');
  try {
    const objects = label.objects.map((o) => {
      // Центр -- как его видит СЦЕНА на опорном кадре: у объекта с траекторией это
      // начало траектории (`labeling._instance_center`, t = 0), у стоящего -- центр
      // бокса. Иначе карточка разошлась бы с тем, что человек нарисовал.
      const centerRef = labelCenterAt(o, o.refFrame);
      return {
        id: o.id || undefined,
        class: o.cls,
        pose: o.pose,
        // Поза СЦЕНЫ -- правило (едет = идёт), сервер считает его тем же
        // `pose_rule` и кладёт в карточку как `pose_scene`.
        pose_scene: labelScenePose(o),
        pose_time_s: labelPoseTime(o, o.refFrame),
        ref_frame: o.refFrame,
        frames: o.frames && o.frames.length ? o.frames : [o.refFrame],
        center: centerRef,
        size: o.size,
        // Опорная поверхность (низ бокса) -- в карточку: по ней видно, что объект
        // поставили на пол/платформу, а не «уронили» в пол.
        z_ref: Number.isFinite(o.zRef) ? o.zRef : undefined,
        yaw: o.yaw,
        pitch: o.pitch,
        roll: o.roll,
        speed_m_per_frame: labelSpeedM(o),
        speed_source: o.speedSource,
        automation: o.automation,
        // Траектория движения: начало, конец, секунды. Скорость сервер считает сам
        // (расстояние / время) и кладёт в карточку и в index.jsonl.
        trajectory: o.traj
          ? { start: o.traj.start, end: o.traj.end, seconds: o.traj.seconds,
              source: o.traj.source || 'manual' }
          : undefined,
      };
    });
    // hz отдаём явно: км/ч -> м/кадр считается по частоте записи, и без этого
    // сервер записал бы трек по своей частоте, а карточка разошлась бы с числом
    // в панели.
    const report = await labelPost('/label/save', { objects, hz: labelHz() });
    label.report = report;
    label.dirty = false;          // сохранено: смена записи больше не опасна
    const errors = report.errors || [];
    const written = report.written || [];
    // id присваивает сервер: подставляем их ПО ИНДЕКСУ исходного объекта, и при
    // ЧАСТИЧНОМ успехе тоже -- written/errors несут index. Иначе у записанных
    // объектов id не появлялся, пока хотя бы один упал, и повторное «Сохранить»
    // плодило дубли. Старый сервер без index -- прежнее поведение (полный успех).
    for (const w of written) {
      if (w && w.id != null && w.index != null && label.objects[w.index]) {
        label.objects[w.index].id = w.id;
      }
    }
    if (!written.some((w) => w && w.index != null) && !errors.length) {
      const ids = written.map((w) => w.id);
      label.objects.forEach((o, i) => { if (ids[i]) o.id = ids[i]; });
    }
    const firstErr = errors[0] || {};
    const errText = (firstErr.index != null ? `объект #${firstErr.index + 1}: ` : '')
      + String(firstErr.error || '');
    setLabelStatus(errors.length
      ? `сохранено ${written.length}, ошибок ${errors.length}: ${errText}`
      : `сохранено объектов: ${written.length} (всего в записи ${report.objects_total})`);
  } catch (e) {
    setLabelStatus(String((e && e.message) || e));
  }
  renderLabel();
}

// -------------------------------------------------------------------- отрисовка

function labelDraw() {
  if (!labelLayer) return;
  // Точки трассировки и закрытые точки -- по ВСЕМ объектам кадра: слой складывает
  // их в один набор, а гасит точки кадра вьюер (onHidden -> applyHiddenPoints).
  const entries = [...label.previews.values()]
    .filter((p) => p.bag === state.bag && p.frame === state.idx);
  labelLayer.setPreviews(entries.map((p) => ({
    key: p.key, points: p.points, hidden: p.removed,
  })));
  labelLayer.setObjects(label.objects.map((o) => labelLayerItem(o)));
}

function fmt3(v) {
  return Number.isFinite(v) ? String(Math.round(v * 1000) / 1000) : '';
}

/** Строка-проводник: что делать сейчас. Ведёт по шагам, а не описывает всё сразу. */
function labelGuideText() {
  if (!label.on) return '';
  if (!labelLayer) {
    return 'режим разметки недоступен: слой labellayer.js не создался — '
      + 'работает только панель. Причина в строке ниже.';
  }
  const o = labelSelected();
  if (!o) {
    return 'ЛКМ по облаку — предложить габарит (объект встанет ОСНОВАНИЕМ на ту '
      + 'поверхность, куда попал клик). Ctrl+Z — отмена, Tab — панель.';
  }
  if (label.busy) return 'считаю лучи кадра: точки объекта и что заслонено…';
  if (label.trajArm) {
    return `кликните по облаку там, где объект ${label.trajArm === 'start'
      ? 'НАЧИНАЕТ' : 'ЗАКАНЧИВАЕТ'} движение — траектория ставится так же, как объект `
      + '(Esc — отменить постановку точки)';
  }
  const steps = [
    'числа в полях перестраивают бокс сразу',
    'тяни ЛКМ внутри бокса — перенос центра; розовые точки на гранях — размер/высота, '
      + 'оранжевая над верхом — поворот; ЛКМ вне бокса — камера',
    'Q/E — поворот ±5° (Shift ±1°), W/A/S/D — сдвиг 0.05 м',
    o.cls === 'person'
      ? 'поза рисуется запечённым мешем: у едущего (с траекторией) играет клип ходьбы, '
        + 'у стоящего — стояния'
      : 'запечённые меши поз есть только у класса «человек»',
    '«точка начала»/«точка конца» + ЛКМ по облаку — траектория; её концы (зелёный и '
      + 'красный) тянутся мышью, секунды — полем справа',
    'Esc — снять выбор (тогда клик по этому месту поставит новый объект)',
    'Del — удалить, Ctrl+Z — отмена',
  ];
  return steps.join(' · ');
}

function renderLabelGuide() {
  const el = $('lbHelp');
  if (!el) return;
  el.textContent = labelGuideText();
  el.style.display = label.on ? 'block' : 'none';
}

/** Режим разметки: панель показывает разметку, а не приборную доску. */
function applyLabelPanelMode() {
  const panel = $('panel');
  if (!panel) return;
  panel.classList.toggle('label-mode', Boolean(label.on));
  panel.classList.toggle('all', Boolean(label.panelAll));
}

/** Значение поля: не трогаем то, которое человек правит прямо сейчас. */
function setField(id, value) {
  const el = $(id);
  if (!el || el === document.activeElement) return;
  el.value = value;
}

function renderLabelStatus() {
  const el = $('lbStatus');
  if (el) el.textContent = label.status || '';
}

function renderLabelPreview() {
  const el = $('labelPreview');
  const total = $('labelHiddenAll');
  if (total) {
    const n = labelHiddenTotal();
    const objects = labelPreviewTargets().length;
    total.textContent = label.on && objects
      ? `закрыто точек по всем объектам кадра: ${n} (объектов ${objects})`
      : '';
  }
  if (!el) return;
  const o = labelSelected();
  if (label.previewError) {
    el.textContent = `предпросмотр: ${label.previewError}`;
    return;
  }
  if (label.busy && !labelPreviewOf(o)) {
    el.textContent = 'считаю лучи кадра…';
    return;
  }
  const p = o ? labelPreviewOf(o) : null;
  if (!p) {
    el.textContent = label.on ? 'предпросмотр не запрошен' : '';
    return;
  }
  const path = p.path || {};
  const counts = p.counts || {};
  const along = Number.isFinite(path.along_m) ? path.along_m.toFixed(3) : '—';
  const lateral = Number.isFinite(path.lateral_m) ? path.lateral_m.toFixed(3) : '—';
  // Габарит МЕША, а не бокса: по нему видно, что поза -- это геометрия (лежащий
  // человек ниже стоящего), а не подпись в карточке.
  const bb = (p.meshBbox && Array.isArray(p.meshBbox.extent) ? p.meshBbox.extent
    : (p.pose && Array.isArray(p.pose.bbox) ? p.pose.bbox : null));
  const mesh = bb ? ` · меш ${bb[0].toFixed(2)}×${bb[1].toFixed(2)}×${bb[2].toFixed(2)} м`
    : '';
  // «Заслонено» показываем числом точек ОБЛАКА, которые реально погашены: часть
  // заслонённых лучей имеет второй возврат вне видимой дальности, таких точек у
  // страницы нет вовсе (counts.points_removed считает их по кадру целиком).
  const hidden = Number.isFinite(counts.points_removed_drawn)
    ? counts.points_removed_drawn : counts.points_removed;
  el.textContent =
    `точек ${p.n_points} (лучей попало ${counts.rays_hit} из ${counts.rays}, `
    + `заслонено ${hidden} — их прячем) · дальность ${along} м, `
    + `поперёк ${lateral} м${mesh}`
    + (label.previews.size > 1
      ? ` · объектов с предпросмотром ${label.previews.size}, `
        + `закрыто всего ${labelHiddenTotal()}` : '')
    + ' · '
    + (p.confirmed ? 'детектор видит объект на дополненном облаке'
      : 'детектор объект не подтвердил');
  el.title = `поза в сцене ${labelScenePose(o)} (карточка: ${o.pose}`
    + (o.traj ? ', есть траектория' : '') + `); меш -- по кадрам ${state.idx}`
    + `: габарит ${bb ? bb.map((v) => v.toFixed(2)).join(' × ') : '—'} м; `
    + `заслонено точек кадра ${counts.points_removed} (тень на лучах объекта)`;
}

/**
 * Строка про МЕШ ПОЗЫ: какой клип, какой момент и по какому правилу он играет.
 * Ради этого строка и нужна: по ней видно, что рисуется настоящая запечённая поза,
 * а не примитив, и что момент идёт за кадром воспроизведения.
 */
function renderLabelMesh(o) {
  const el = $('lbMesh');
  if (!el) return;
  if (!o) {
    el.textContent = '';
    return;
  }
  const scenePose = labelScenePose(o);
  if (o.cls !== 'person') {
    el.textContent = `меш: примитив (класс ${o.cls}) — запечённые позы есть только у человека`;
    el.title = 'У не-человека геометрия -- примитив sim/objects.py: запечённых поз '
      + 'для оборудования нет';
    return;
  }
  const got = labelPoseMesh(o, state.idx);
  const info = got.info || {};
  if (info.state === 'loading') {
    el.textContent = `поза ${scenePose}: клип грузится с сервера…`;
    return;
  }
  if (info.state === 'error') {
    el.textContent = `поза ${scenePose}: клип не получен (${info.error}) — `
      + 'рисуется бокс, трассировка всё равно идёт по мешу позы';
    return;
  }
  const bb = info.bbox ? info.bbox.size : null;
  el.textContent = `поза в сцене ${scenePose} (в карточке ${o.pose}`
    + (o.traj ? ', есть траектория — правило «едет = идёт»' : ', траектории нет')
    + `) · клип ${info.clip} (${info.asset}), момент ${info.moment + 1}/${info.samples}`
    + `, t=${info.time.toFixed(2)} с от опорного кадра`
    + (info.blend > 0.001 ? ` (склейка ${info.blend.toFixed(2)})` : '')
    + ` · ${info.play === 'loop' ? 'цикл' : (info.play === 'static' ? 'статично' : 'один проход')}`
    + ` · меш ${info.vertices} вершин, рост ${bb ? bb[2].toFixed(2) : '—'} м`
    + `, низ на z=${(info.bbox ? info.bbox.min[2] : 0).toFixed(3)} м`;
  el.title = 'Вершины -- из assets/_poses (манифест class_map), момент выбран по '
    + `времени кадра; меш ставится тем же расчётом, что трассировка на сервере `
    + `(labeling.place_pose): низ на опору, середина габарита в центр, поворот `
    + `${o.yaw.toFixed(1)}°/${o.pitch.toFixed(1)}°/${o.roll.toFixed(1)}°`;
}

/** Строка про ТРАЕКТОРИЮ: откуда, куда, сколько секунд, с какой скоростью. */
function renderLabelTraj(o) {
  const el = $('lbTraj');
  if (!el) return;
  if (!o) {
    el.textContent = '';
    return;
  }
  if (!o.traj) {
    el.textContent = 'траектория не задана: «точка начала»/«точка конца» — '
      + 'клик по облаку, потом тяни концы мышью';
    return;
  }
  const d = labelGeom ? labelGeom.trajectoryDistance(o.traj) : 0;
  const v = labelGeom ? labelGeom.trajectorySpeed(o.traj) : 0;
  el.textContent = `траектория: ${d.toFixed(2)} м за ${o.traj.seconds.toFixed(1)} с `
    + `→ ${v.toFixed(2)} м/с (${(v * 3.6).toFixed(1)} км/ч) · `
    + `начало ${o.traj.start.map((x) => x.toFixed(2)).join(', ')} · `
    + `конец ${o.traj.end.map((x) => x.toFixed(2)).join(', ')} · `
    + `на кадре ${state.idx} объект ${labelTrajProgress(o)}`;
  el.title = 'Траектория сохраняется в карточку и в index.jsonl: начало, конец, '
    + 'секунды, скорость. Позиция на кадре интерполируется ПО ВРЕМЕНИ '
    + '(t = (кадр − опорный)/Гц), после конца объект стоит на конечной точке';
}

/** Где объект по траектории: доля пути и пройденная часть, в метрах. */
function labelTrajProgress(o) {
  const t = labelPoseTime(o, state.idx);
  const s = Number(o.traj.seconds) || 0;
  const frac = s > 0 ? Math.min(Math.max(t / s, 0), 1) : 0;
  const d = labelGeom ? labelGeom.trajectoryDistance(o.traj) : 0;
  const state2 = t >= s ? 'на конечной точке' : (t <= 0 ? 'в начале' : 'в пути');
  return `${state2}: ${frac.toFixed(2)} пути (${(frac * d).toFixed(2)} м), t=${t.toFixed(2)} с`;
}

function renderLabelTrack(o) {
  const el = $('labelTrack');
  if (!el) return;
  const v = labelSpeedM(o);
  const here = labelRangeAt(o, state.idx);
  const atRef = labelRangeAt(o, o.refFrame);
  const s = o.speedSource === 'manual' ? 'вручную' : 'замер';
  el.textContent =
    `трек: ход сенсора ${v.toFixed(4)} м/кадр при ${labelHz().toFixed(2)} Гц `
    + `(${(v * labelHz() * 3.6).toFixed(1)} км/ч, ${s}) · `
    + `дальность на кадре ${state.idx}: ${here.toFixed(3)} м `
    + `(опорный кадр ${o.refFrame}: ${atRef.toFixed(3)} м, за кадр ${(-v).toFixed(3)} м)`
    + (state.idx === o.refFrame ? ' — текущий кадр и есть опорный' : '')
    + (o.traj
      ? ` · ЕДЕТ по траектории со своей скоростью ${labelShownKmh(o).toFixed(1)} км/ч `
        + `(${(labelShownKmh(o) / 3.6).toFixed(2)} м/с), см. строку траектории`
      : '');
}

function renderLabelList() {
  const box = $('lbList');
  if (!box) return;
  const here = label.objects.filter((o) => o.frames.includes(state.idx));
  const count = $('lbCount');
  if (count) count.textContent = String(here.length);
  box.textContent = '';
  for (const o of label.objects) {
    const row = document.createElement('div');
    const dot = document.createElement('span');
    dot.className = 'lbdot';
    dot.style.background = AUTOMATION_CSS[o.automation] || AUTOMATION_CSS.manual;
    dot.title = AUTOMATION_TEXT[o.automation] || o.automation;
    const name = document.createElement('button');
    name.className = 'lbname' + (o.key === label.selected ? ' sel' : '');
    // Вдоль пути -- это -y: объект впереди имеет отрицательный y.
    const prev = labelPreviewOf(o);
    const hidden = prev && prev.frame === state.idx ? (prev.removed || []).length : null;
    name.textContent = `${o.cls}${o.pose !== 'unknown' ? '/' + o.pose : ''} · вдоль `
      + `${(-o.center[1]).toFixed(1)} м · кадр ${o.refFrame}`
      + (o.frames.length > 1 ? `+${o.frames.length - 1}` : '')
      + (o.frames.includes(state.idx) ? '' : ' · не в кадре')
      + (o.traj ? ` · траектория ${o.traj.seconds.toFixed(1)} с` : '')
      + (hidden !== null ? ` · закрыто ${hidden}` : '');
    name.title = 'выбрать объект';
    name.onclick = () => {
      labelSelect(o.key);
      renderLabel();
      schedulePreview();
    };
    const del = document.createElement('button');
    del.textContent = '×';
    del.title = 'удалить объект';
    del.onclick = () => labelRemove(o.key);
    row.appendChild(dot);
    row.appendChild(name);
    row.appendChild(del);
    box.appendChild(row);
  }
}

function buildLabelLegend() {
  const box = $('lbLegend');
  if (!box) return;
  box.textContent = '';
  const add = (css, text) => {
    const span = document.createElement('span');
    span.style.display = 'flex';
    span.style.alignItems = 'center';
    span.style.gap = '4px';
    const dot = document.createElement('span');
    dot.className = 'lbdot';
    dot.style.background = css;
    span.appendChild(dot);
    span.appendChild(document.createTextNode(text));
    box.appendChild(span);
  };
  for (const key of ['detector', 'cluster', 'manual']) add(AUTOMATION_CSS[key], AUTOMATION_TEXT[key]);
  add(PREVIEW_CSS, 'точки трассировки');
}

const fillSelect = (sel, pairs, value) => {
  if (!sel) return;
  sel.textContent = '';
  for (const [key, text] of pairs) {
    const o = document.createElement('option');
    o.value = key;
    o.textContent = text;
    sel.appendChild(o);
  }
  if (value !== undefined) sel.value = value;
};

function renderLabelMotion() {
  const el = $('labelMotion');
  if (!el) return;
  if (!labelLayer) {
    el.textContent = `режим разметки недоступен: ${(createLabelLayerError
      && (createLabelLayerError.message || String(createLabelLayerError)))
      || 'слой labellayer.js не загрузился'}. Остальная страница работает как обычно.`;
    return;
  }
  if (label.motionError) {
    el.textContent = `ход записи: ${label.motionError}`;
    return;
  }
  const m = label.motion;
  if (!m) {
    el.textContent = 'ход записи: считается…';
    return;
  }
  const pairs = (m.pairs || []).map((p) => p.frames.join('→')).join(', ');
  const s = labelMotionSummary();
  const here = s && Number.isFinite(s.here) ? `${s.here.toFixed(3)} м/кадр на кадре ${state.idx}` : null;
  el.textContent = `ход записи (профиль по кадрам): ${m.m_per_frame.toFixed(3)} м/кадр · `
    + `${m.kmh.toFixed(1)} км/ч · остаток `
    + `${Number.isFinite(m.residual_m) ? m.residual_m.toFixed(3) + ' м' : '—'}`
    + (s && s.frames ? ` · достоверных кадров ${(s.reliable * 100).toFixed(0)}% из ${s.frames}` : '')
    + (here ? ` · ${here}` : '')
    + (pairs ? ` · пары ${pairs}` : '');
}

/** Панель разметки: числа выбранного объекта, список кадра, предпросмотр, статус. */
function renderLabel() {
  const body = $('labelBody');
  const edit = $('labelEdit');
  const badge = $('src-label');
  if (badge) badge.textContent = label.on ? 'вкл' : 'выкл';
  if (body) body.style.display = label.on ? 'block' : 'none';
  if (!label.on) return;
  renderLabelMotion();
  renderLabelGuide();
  const o = labelSelected();
  if (edit) edit.style.display = o ? 'block' : 'none';
  if (!o) {
    renderLabelStatus();
    renderLabelList();
    return;
  }
  const dot = $('lbDot');
  if (dot) dot.style.background = AUTOMATION_CSS[o.automation] || AUTOMATION_CSS.manual;
  const auto = $('lbAuto');
  if (auto) {
    auto.textContent = AUTOMATION_TEXT[o.automation] || o.automation;
    auto.style.color = AUTOMATION_CSS[o.automation] || '';
  }
  const idBox = $('lbId');
  if (idBox) idBox.textContent = o.id || 'не сохранён';
  const reason = $('lbReason');
  if (reason) reason.textContent = o.reason || '';
  const clsSel = $('lb-class');
  if (clsSel) clsSel.value = o.cls;
  const poseSel = $('lb-pose');
  if (poseSel) {
    poseSel.value = o.pose;
    poseSel.disabled = o.cls !== 'person';
  }
  setField('lb-cx', fmt3(o.center[0]));
  setField('lb-cy', fmt3(o.center[1]));
  setField('lb-cz', fmt3(o.center[2]));
  setField('lb-sx', fmt3(o.size[0]));
  setField('lb-sy', fmt3(o.size[1]));
  setField('lb-sz', fmt3(o.size[2]));
  setField('lb-yaw', fmt3(o.yaw));
  setField('lb-pitch', fmt3(o.pitch));
  setField('lb-roll', fmt3(o.roll));
  setField('lb-ref', String(o.refFrame));
  const base = $('lbBase');
  if (base) {
    const zRef = Number.isFinite(o.zRef) ? o.zRef : o.center[2] - o.size[1] / 2;
    base.textContent = `низ (опора): z=${fmt3(zRef)} м · верх z=${fmt3(zRef + o.size[1])} м`
      + (Number.isFinite(o.zRef) ? '' : ' — опора не пришла с сервера, взята от центра');
  }
  // Откуда габарит: измеренный он или поставлен «на глаз». Сервер помечает это
  // (`size_source`), и панель обязана сказать прямо: иначе размер по умолчанию
  // выглядит как результат замера.
  const sizeSrc = $('lbSizeSrc');
  if (sizeSrc) {
    const src = o.sizeSource || 'cluster';
    sizeSrc.style.display = src === 'cluster' ? 'none' : 'block';
    sizeSrc.textContent = src === 'default'
      ? 'габарит не измерен: рядом с кликом точек не нашлось — задайте размеры '
        + 'руками или нажмите «подогнать по точкам»'
      : (src === 'band'
        ? 'габарит по точкам рядом с кликом (кластера автоматика не нашла) — '
          + 'проверьте «подогнать по точкам»'
        : '');
  }
  // Набранный текст поля «кадры» держим ВМЕСТЕ с объектом: иначе набранное для
  // одного объекта показалось бы у другого (поле в фокусе мы не перезаписываем).
  const draft = label.framesDraft && label.framesDraft.key === o.key
    ? label.framesDraft.text : null;
  setField('lb-frames', draft === null ? o.frames.join(', ') : draft);
  setField('lb-kmh', fmt3(labelSpeedKmh(o)));
  const meas = $('lbMeas');
  if (meas) meas.textContent = `${label.motion ? label.motion.kmh.toFixed(1) : '—'} км/ч`;
  renderLabelMesh(o);
  renderLabelTraj(o);
  setField('lb-traj-sec', o.traj ? fmt3(o.traj.seconds) : '');
  const arm = $('lb-traj-arm');
  if (arm) {
    arm.textContent = label.trajArm === 'start'
      ? 'жду клика: НАЧАЛО траектории — ЛКМ по облаку'
      : (label.trajArm === 'end' ? 'жду клика: КОНЕЦ траектории — ЛКМ по облаку' : '');
    arm.style.display = label.trajArm ? 'block' : 'none';
  }
  for (const [id, which] of [['lb-traj-start', 'start'], ['lb-traj-end', 'end']]) {
    const btn = $(id);
    if (btn) btn.classList.toggle('active', label.trajArm === which);
  }
  renderLabelTrack(o);
  renderLabelPreview();
  renderLabelList();
  renderLabelStatus();
  const rep = $('lbReport');
  if (rep) {
    const r = label.report;
    if (!r) rep.textContent = '';
    else {
      const w = (r.written || [])[0];
      rep.textContent = `сохранено ${(r.written || []).length} объектов`
        + `, всего в записи ${r.objects_total}`
        + (w ? `, точек ${w.n_points}, npz ${w.size_bytes} Б, id ${w.id}` : '')
        + ((r.errors || []).length ? `; ошибки: ${r.errors.map((e) => e.error).join('; ')}` : '');
    }
  }
}

// ------------------------------------------------------ панель и мышь разметки

function labelNumInputs() {
  const patch3 = (ids, key) => {
    ids.forEach((id, i) => wire(id, 'input', () => {
      const o = labelSelected();
      const value = Number($(id).value);
      if (!o || !Number.isFinite(value)) return;
      const next = o[key].slice();
      next[i] = value;
      labelPatch(o.key, { [key]: next });
    }));
  };
  patch3(['lb-cx', 'lb-cy', 'lb-cz'], 'center');
  patch3(['lb-sx', 'lb-sy', 'lb-sz'], 'size');
  const angles = { 'lb-yaw': 'yaw', 'lb-pitch': 'pitch', 'lb-roll': 'roll' };
  for (const id of Object.keys(angles)) {
    wire(id, 'input', () => {
      const o = labelSelected();
      const value = Number($(id).value);
      if (!o || !Number.isFinite(value)) return;
      labelPatch(o.key, { [angles[id]]: value });
    });
  }
  wire('lb-traj-sec', 'input', () => {
    const value = Number($('lb-traj-sec').value);
    if (Number.isFinite(value) && value > 0) labelSetTrajSeconds(value);
  });
  // «точка начала»/«точка конца»: взводим ожидание клика по облаку, повторное
  // нажатие снимает ожидание (человек передумал).
  for (const [id, which] of [['lb-traj-start', 'start'], ['lb-traj-end', 'end']]) {
    wire(id, 'click', () => {
      const o = labelSelected();
      if (!o) {
        setLabelStatus('сначала выберите объект: траектория ставится ему');
        return;
      }
      label.trajArm = label.trajArm === which ? null : which;
      setLabelStatus(label.trajArm
        ? `кликните по облаку там, где объект ${which === 'start' ? 'начинает' : 'заканчивает'} `
          + 'движение (точка ставится так же, как объект)'
        : 'постановка точки траектории отменена');
      renderLabel();
    });
  }
  wire('lb-traj-del', 'click', () => labelClearTraj());
  wire('lb-ref', 'input', () => {
    const o = labelSelected();
    const value = Math.round(Number($('lb-ref').value));
    if (!o || !Number.isFinite(value)) return;
    const frames = o.frames.includes(value) ? o.frames : [value];
    labelPatch(o.key, { refFrame: value, frames });
  });
  wire('lb-frames', 'input', () => {
    const o = labelSelected();
    if (!o) return;
    const text = $('lb-frames').value;
    label.framesDraft = { key: o.key, text };
    const list = text.split(',')
      .map((v) => v.trim())
      .filter((v) => v !== '' && Number.isFinite(Number(v)))
      .map((v) => Math.round(Number(v)));
    if (list.length) labelPatch(o.key, { frames: list });
  });
}

function labelWire() {
  // Панель разметки ждёт слоя, но НЕ молчит без него: раньше весь labelWire
  // выходил на первой строке («if (!labelLayer) return»), и в группе разметки
  // не работало НИЧЕГО -- ни тумблер режима, ни список классов, ни кнопка
  // сохранения, и объяснения причины тоже не было. Теперь обработчики есть
  // всегда, а без слоя органы управления честно объясняют, чего не хватает
  // (как это сделано для объектов пути в showTrackUnavailable).
  const layerless = () => {
    const reason = (createLabelLayerError
      && (createLabelLayerError.message || String(createLabelLayerError)))
      || 'слой labellayer.js не загрузился';
    const box = $('labelNote');
    if (box) {
      box.textContent = `режим разметки недоступен: ${reason}. `
        + 'Панель работает, но поставленный объект не нарисован и мышью не '
        + 'правится: боксы и точки рисует слой разметки. Остальная страница — '
        + 'облако, кадры, пресеты, разрез — работает как обычно.';
    }
    setLabelStatus(`разметка недоступна: ${reason}`);
    renderLabel();
  };
  fillSelect($('lb-class'), LABEL_CLASSES);
  fillSelect($('lb-pose'), LABEL_POSES);
  buildLabelLegend();
  applyLabelPanelMode();
  wire('t-label', 'change', (e) => {
    label.on = e.target.checked;
    // Разметка и воспроизведение не дружат: клик целится в кадр, который через
    // 100 мс уже сменился, а предпросмотр на ходу вообще не считается. Включили
    // разметку -- останавливаем облако и говорим об этом.
    if (label.on && state.playing) {
      state.playing = false;
      const playBtn = $('play');
      if (playBtn) playBtn.textContent = 'Играть';
      setLabelStatus('воспроизведение остановлено: объект ставится в текущий кадр');
    }
    if (labelLayer) labelLayer.setVisible(label.on);
    const note = $('labelNote');
    if (note) note.style.display = label.on ? 'none' : 'block';
    applyLabelPanelMode();
    renderLabel();
    if (!label.on) {
      label.previews.clear();
      label.previewSeq += 1;      // ответы в полёте уже не нужны
      label.busy = false;         // ...и статус «считаю» не должен висеть вечно
      applyHiddenPoints([]);
      labelDraw();
      return;
    }
    // Режим включили: считаем закрытые точки по всем объектам кадра (объекты могли
    // остаться от прошлого включения, а предпросмотр могли сбросить).
    schedulePreview();
    if (labelLayer) {
      setLabelStatus('режим разметки: ЛКМ по облаку — предложение габарита');
    } else {
      layerless();
    }
  });
  wire('lb-panel-all', 'click', (e) => {
    label.panelAll = !label.panelAll;
    e.target.classList.toggle('active', label.panelAll);
    e.target.textContent = label.panelAll ? 'только разметка' : 'вся панель';
    applyLabelPanelMode();
  });
  wire('lb-class', 'change', () => {
    const o = labelSelected();
    if (!o) return;
    const cls = $('lb-class').value;
    labelPatch(o.key, { cls, pose: DEFAULT_POSE[cls] || 'unknown' });
  });
  wire('lb-pose', 'change', () => {
    const o = labelSelected();
    if (o) labelPatch(o.key, { pose: $('lb-pose').value });
  });
  labelNumInputs();
  wire('lb-kmh', 'input', () => {
    const o = labelSelected();
    const value = Number($('lb-kmh').value);
    if (o && Number.isFinite(value)) {
      labelPatch(o.key, { speedKmh: value, speedSource: 'manual' });
    }
  });
  wire('lb-speed-measured', 'click', () => {
    const o = labelSelected();
    if (o) labelPatch(o.key, { speedKmh: null, speedSource: 'measured' });
  });
  wire('lb-speed-50', 'click', () => {
    const o = labelSelected();
    if (o) labelPatch(o.key, { speedKmh: 50, speedSource: 'manual' });
  });
  wire('lb-recount', 'click', () => {
    if (label.timer) { clearTimeout(label.timer); label.timer = null; }
    refreshPreview();
  });
  wire('lb-fit', 'click', () => labelRefit());
  wire('lb-save', 'click', () => labelSave());
  wire('lb-undo', 'click', () => labelUndoLast());
  window.addEventListener('keydown', labelKeyDown);
}

/**
 * Клавиатура разметки: Q/E -- поворот (±5°, Shift -- 1°), W/A/S/D (и Shift со
 * стрелками) -- сдвиг 0.05 м, Del -- удалить, Ctrl+Z -- отмена. Работает только
 * в режиме разметки: стрелки и пробел заняты кадрами и паузой (wireUi), и
 * отбирать их вне разметки нельзя.
 */
function labelKeyDown(ev) {
  if (!label.on || typingInField()) return;
  const mod = ev.ctrlKey || ev.metaKey;
  if (mod && String(ev.key).toLowerCase() === 'z') {
    ev.preventDefault();
    labelUndoLast();
    return;
  }
  if (mod || ev.altKey) return;
  const step = 0.05;             // сдвиг, м
  const big = ev.shiftKey ? 1 : 5;   // поворот: Shift -- мелкий шаг
  switch (ev.code) {
    case 'KeyQ': ev.preventDefault(); labelRotate(big); return;
    case 'KeyE': ev.preventDefault(); labelRotate(-big); return;
    case 'KeyW': ev.preventDefault(); labelNudge(0, -step); return;
    case 'KeyS': ev.preventDefault(); labelNudge(0, step); return;
    case 'KeyA': ev.preventDefault(); labelNudge(-step, 0); return;
    case 'KeyD': ev.preventDefault(); labelNudge(step, 0); return;
    // Снять выбор: тогда ЛКМ внутри поставленного бокса снова ставит НОВЫЙ объект,
    // а не тянет старый (внутри бокса жест -- всегда правка).
    case 'Escape': {
      if (label.trajArm) {
        ev.preventDefault();
        label.trajArm = null;
        renderLabel();
        setLabelStatus('постановка точки траектории отменена');
        return;
      }
      if (!labelSelected()) return;
      ev.preventDefault();
      labelSelect(null);
      // Предпросмотры -- по объектам кадра, а не по выбранному: снятие выбора их не
      // трогает (иначе точки, закрытые ДРУГИМИ объектами, снова открылись бы).
      labelDraw();
      renderLabel();
      setLabelStatus('выбор снят: клик по облаку снова ставит новый объект');
      return;
    }
    case 'Delete':
    case 'Backspace': {
      const o = labelSelected();
      ev.preventDefault();
      if (o) labelRemove(o.key);
      return;
    }
    // Shift + стрелки: стрелки без Shift листают кадры (wireUi), поэтому сдвиг
    // объекта -- именно с ними. Плюс-минус по высоте -- PageUp/PageDown.
    case 'ArrowUp': if (!ev.shiftKey) return; ev.preventDefault(); labelNudge(0, -step); return;
    case 'ArrowDown': if (!ev.shiftKey) return; ev.preventDefault(); labelNudge(0, step); return;
    case 'ArrowLeft': if (!ev.shiftKey) return; ev.preventDefault(); labelNudge(-step, 0); return;
    case 'ArrowRight': if (!ev.shiftKey) return; ev.preventDefault(); labelNudge(step, 0); return;
    case 'PageUp': ev.preventDefault(); labelResize(step); return;
    case 'PageDown': ev.preventDefault(); labelResize(-step); return;
    default:
  }
}

function wireUi() {
  const presetBox = $('presets');
  for (const [key, label] of PRESETS) {
    const b = document.createElement('button');
    b.textContent = label;
    b.dataset.preset = key;
    b.onclick = () => cameraPreset(key);
    presetBox.appendChild(b);
  }

  const modeSel = $('mode');
  // список режимов заполняет fillModeSelect() -- он же вызывается при переключении
  // записи (raskraska общая, но meta этой записи приходит заново)
  modeSel.onchange = () => {
    state.mode = modeSel.value;
    cache.clear();
    showFrame(state.idx).catch(fail);
  };

  // Записи (датасеты): селектор и профиль вида на запись. Кнопки сохранения нет,
  // если сервер её не поддерживает -- тогда просто ничего не подписываем.
  const bagSel = $('bag');
  if (bagSel) bagSel.onchange = () => selectBag(bagSel.value).catch(fail);
  wire('saveView', 'click', () => saveProfile().catch(fail));

  $('play').onclick = () => {
    state.playing = !state.playing;
    $('play').textContent = state.playing ? 'Пауза' : 'Играть';
    state.nextTick = performance.now();
  };
  for (const b of document.querySelectorAll('[data-step]')) {
    b.onclick = () => {
      state.playing = false;
      $('play').textContent = 'Играть';
      showFrame(state.idx + Number(b.dataset.step)).catch(fail);
    };
  }
  const fr = $('frame');
  fr.max = String(meta.frames - 1);
  fr.oninput = () => {
    state.playing = false;
    $('play').textContent = 'Играть';
    showFrame(Number(fr.value)).catch(fail);
  };
  $('fps').oninput = (e) => { state.fps = Number(e.target.value); $('fpsVal').textContent = state.fps; };
  $('realtime').onchange = (e) => { state.realtime = e.target.checked; };
  // Номер текущего кадра рядом с воспроизведением: по клику копируется «запись · fN»,
  // чтобы кадр можно было назвать словами, а не пересылать скриншот целиком.
  const frameCopy = $('frameCopy');
  if (frameCopy) {
    frameCopy.onclick = () => {
      const text = frameStamp();
      const ok = () => {
        frameCopy.textContent = 'скопировано';
        setTimeout(updateFrameCopy, 1200);
      };
      const bad = () => {
        frameCopy.textContent = 'не скопировалось';
        setTimeout(updateFrameCopy, 1600);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(ok, bad);
        return;
      }
      const ta = document.createElement('textarea');   // http без clipboard API
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand('copy'); ok(); } catch (e) { bad(); }
      ta.remove();
    };
  }

  $('psize').oninput = (e) => {
    points.material.uniforms.uSize.value = Number(e.target.value);
    $('psizeVal').textContent = e.target.value;
  };
  $('dist').oninput = (e) => {
    const raw = Number(e.target.value);
    state.maxDist = distValue(raw);
    $('distVal').textContent = distIsInf(raw) ? '∞' : String(raw);
    points.material.uniforms.uMaxDist.value = state.maxDist;
    buildGrid();
  };
  $('grid').onchange = (e) => { gridMesh.visible = e.target.checked; };
  $('axes').onchange = (e) => { axesMesh.visible = e.target.checked; };
  // Ползунков «модели рельсов» (ось/колея/контактный/смещение) в панели больше нет:
  // ось и колея приходят из /meta (детектор + профиль записи), коридор сервер
  // заметает вдоль измеренной линии рельсов.
  $('additive').onchange = (e) => {
    points.material.blending = e.target.checked ? THREE.AdditiveBlending : THREE.NormalBlending;
    points.material.depthWrite = !e.target.checked;
    points.material.needsUpdate = true;
  };

  // Объекты пути: слой из track3d.js -- меши /track, кадрирование по объектам,
  // «чистое поле». Тумблеры независимы от раскраски зон и не трогают пресеты
  // камеры. Тумблер облака работает и без слоя; кнопки объектов подписываем только
  // если слой есть -- иначе они молча ничего не делают, а причина видна в панели
  // (showTrackUnavailable).
  wire('t-points', 'change', (e) => { pointsOn = e.target.checked; applyPointsVisible(); });
  applyPointsVisible();
  if (createObjectsLayer && !objectsLayer) {
    objectsLayer = createObjectsLayer({ scene, camera, controls });
    wire('t-objects', 'change', (e) => objectsLayer.setMaster(e.target.checked));
    // Начальное состояние берём из панели: объекты из замеров выключены по
    // умолчанию (галочка снята), а слой сам показывает всё -- без этой строки
    // 3D-объекты рисовались бы при снятой галочке.
    const objectsBox = $('t-objects');
    objectsLayer.setMaster(Boolean(objectsBox && objectsBox.checked));
    const fit = $('objects-fit');
    if (fit) {
      fit.onclick = () => {
        if (!objectsLayer.fitToObjects()) fail(new Error('нет объектов: вписать нечего'));
      };
    }
    const fitSmall = $('objects-fit-small');
    if (fitSmall) {
      fitSmall.onclick = () => {
        if (!objectsLayer.fitToSmallObjects()) {
          fail(new Error('нет мелких объектов: включите их галочками'));
        }
      };
    }
  }
  if (trackLayer) {
    wire('t-rails', 'change', (e) => trackLayer.setVisible({ rails: e.target.checked }));
    wire('t-sleepers', 'change', (e) => trackLayer.setVisible({ sleepers: e.target.checked }));
    wire('t-floor', 'change', (e) => trackLayer.setVisible({ floor: e.target.checked }));
    wire('t-startview', 'change', (e) => trackLayer.setAutoView(e.target.checked));
    wire('clean', 'click', () => trackLayer.setCleanField(!trackLayer.isCleanField()));
    wire('view-profile', 'click', () => trackLayer.viewProfile());
  } else {
    // Слоя нет: контролы группы не «мертвим», но вместо работы они объясняют,
    // почему её нет (в панели уже висит та же причина) -- чтобы нажатие не
    // выглядело как поломка.
    for (const id of ['t-rails', 't-sleepers', 't-floor', 't-startview']) {
      wire(id, 'change', () => showTrackUnavailable(createTrackLayerError));
    }
    for (const id of ['clean', 'view-profile']) {
      wire(id, 'click', () => showTrackUnavailable(createTrackLayerError));
    }
  }

  // Коридор безопасности: слой живёт в safetylayer.js, серверных параметров у него
// нет (объём заметается вдоль измеренной оси рельсов, размеры нормативные), поэтому
// в панели один тумблер -- показывать каркас или нет.
  wire('tun-on', 'change', (e) => {
    safeSafety('setVisible', (l) => l.setVisible(e.target.checked));
  });
  if (!safetyLayer) {
    // Слоя нет (нет файла или маршрута /safetylayer.js): блок остаётся, но вместо
    // рисунка говорит, почему его нет -- как это сделано для объектов пути.
    const reason = (createSafetyLayerError
      && (createSafetyLayerError.message || String(createSafetyLayerError)))
      || 'слой safetylayer.js не загрузился';
    const box = $('tunSummary');
    if (box) box.textContent = `коридор недоступен: ${reason}`;
  }

  window.addEventListener('keydown', (ev) => {
    // Shift+стрелки в режиме разметки -- сдвиг объекта, а не кадр: кадры листаются
    // стрелками без Shift (обработчик разметки подписан ниже, см. labelKeyDown).
    const labelShift = label.on && ev.shiftKey
      && (ev.code === 'ArrowLeft' || ev.code === 'ArrowRight');
    if (labelShift) return;
    if (ev.code === 'Space') { ev.preventDefault(); $('play').click(); }
    else if (ev.code === 'ArrowLeft') { ev.preventDefault(); showFrame(state.idx - 1).catch(fail); }
    else if (ev.code === 'ArrowRight') { ev.preventDefault(); showFrame(state.idx + 1).catch(fail); }
    else if (ev.code === 'Tab') { ev.preventDefault(); $('panel').classList.toggle('hidden'); }
  });
  window.addEventListener('resize', onResize);

  // Панель разметки -- последней: её обработчики зависят от того, создался ли слой
  // (labellayer.js). Слоя нет -- группа остаётся, но говорит, почему не работает.
  labelWire();
}

function buildGrid() {
  if (!gridMesh) return;
  const g = meta.grid;
  const len = Math.min(state.maxDist, g.length);
  const pts = [];
  const hw = Math.min(g.half_width, 40);
  for (let x = -hw; x <= hw + 1e-6; x += g.spacing) pts.push(x, -len, g.z, x, 0, g.z);
  for (let y = -len; y <= 1e-6; y += g.spacing) pts.push(-hw, y, g.z, hw, y, g.z);
  gridMesh.geometry.dispose();
  const gg = new THREE.BufferGeometry();
  gg.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
  gridMesh.geometry = gg;
}

function onResize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  if (points) points.material.uniforms.uPixelRatio.value = Math.min(window.devicePixelRatio || 1, 2);
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function initSection() {
  sectionCanvas = $('section');
  sectionCtx = sectionCanvas.getContext('2d');
  sectionDpr = Math.min(window.devicePixelRatio || 1, 2);
  sectionCanvas.width = Math.round(SECTION_W * sectionDpr);
  sectionCanvas.height = Math.round(SECTION_H * sectionDpr);
  sectionCanvas.style.width = SECTION_W + 'px';
  sectionCanvas.style.height = SECTION_H + 'px';
  sectionLastIdx = -1;
}

// Линия оси записи x = s·y + i. Берём ту же ось, по которой сервер строит путь
// (meta.path.variants.axis): так u отсчитывается от настоящей оси, а не от нуля
// мира. Варианта нет (старый сервер, одиночная запись) -- ось постоянна и равна
// meta.axis_x. Ось в этих записях почти прямая, но выпрямление обязано работать и
// на кривой: поэтому МНК по точкам варианта, а не первые две точки.
function sectionAxisLine() {
  const v = meta.path && meta.path.variants ? meta.path.variants.axis : null;
  const ys = v && Array.isArray(v.y) ? v.y : null;
  const xs = v && Array.isArray(v.x) ? v.x : null;
  if (!ys || !xs || v.available === false || ys.length < 2 || ys.length !== xs.length) {
    return { s: 0, i: meta.axis_x };
  }
  let my = 0, mx = 0;
  for (let k = 0; k < ys.length; k++) { my += ys[k]; mx += xs[k]; }
  my /= ys.length; mx /= xs.length;
  let cov = 0, varY = 0;
  for (let k = 0; k < ys.length; k++) {
    cov += (ys[k] - my) * (xs[k] - mx);
    varY += (ys[k] - my) ** 2;
  }
  const s = varY > 1e-9 ? cov / varY : 0;
  return { s, i: mx - s * my };
}

// Разрез X-Z по фиксированной плите точек впереди (SECTION_Y_NEAR..FAR метров от
// сенсора), в ВЫПРЯМЛЕННЫХ координатах u/w (см. константы выше). Рамка -- ТОЛЬКО
// из метаданных записи (ось, пол), поэтому кадр в кадр она одна и та же и срез не
// «ходит ходуном». Рисуем только когда сменился кадр (applyFrame сбрасывает
// sectionLastIdx), иначе -- ничего.
function drawSection() {
  if (!sectionCtx || !meta || !posArr) return;
  if (state.idx === sectionLastIdx) return;
  sectionLastIdx = state.idx;
  const t0 = performance.now();

  const W = SECTION_W, H = SECTION_H, dpr = sectionDpr, ctx = sectionCtx;
  const padL = 44, padR = 10, padT = 34, padB = 30;
  const plotX = padL, plotY = padT;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const yNear = SECTION_Y_NEAR, yFar = SECTION_Y_FAR;
  const yc = (yNear + yFar) / 2;                 // середина окна по Y, м
  const rgbOf = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;
  const axis = sectionAxisLine();                // ось записи: x = s·y + i
  const fa = meta.floor.a, fb = meta.floor.b;    // пол записи: z = a + b·y

  // 1) рамка: u = ±SECTION_U_HALF_M, w = SECTION_BOTTOM_M..SECTION_TOP_M.
  // Значения -- из /meta (включая профильные правки оси и пола), не из точек.
  const umin = -SECTION_U_HALF_M, umax = SECTION_U_HALF_M;
  const wmin = SECTION_BOTTOM_M, wmax = SECTION_TOP_M;
  let du = umax - umin, dw = wmax - wmin;
  if (!(du > 0)) du = 2 * SECTION_U_HALF_M;
  if (!(dw > 0)) dw = SECTION_TOP_M - SECTION_BOTTOM_M;
  const sc = Math.min(plotW / du, plotH / dw);   // равный масштаб по u и w
  const uc = (umin + umax) / 2, wc = (wmin + wmax) / 2;
  // +u (мировой +X) -- ВЛЕВО, как в 3D: там camera.up = Z и взгляд почти вдоль -Y,
  // поэтому экранное «вправо» -- это -X. Иначе разрез читается как обратная сторона.
  const mapU = (u) => plotX + plotW / 2 - (u - uc) * sc;
  const mapW = (w) => plotY + plotH / 2 - (w - wc) * sc;

  // 2) точки плиты y ∈ [yFar, yNear]: выпрямляем каждую точку (u, w) и пишем
  // пиксели через ImageData, без строк на каждую точку.
  const iw = Math.round(W * dpr), ih = Math.round(H * dpr);
  const img = ctx.createImageData(iw, ih);
  const buf32 = new Uint32Array(img.data.buffer);
  buf32.fill(0xff180f0c);                        // фон панели (ABGR)
  const n = state.numPoints;
  const axS = axis.s, axI = axis.i;              // локальные копии: цикл по 200k точек
  const inv = -sc;                               // mapU/mapW в виде px = c + inv*coord
  const cx = plotX + plotW / 2 + uc * sc;
  const cy = plotY + plotH / 2 + wc * sc;
  let drawn = 0;
  for (let i = 0; i < n; i++) {
    const j = i * 3;
    const y = posArr[j + 1];
    if (y < yFar || y > yNear) continue;         // плита фиксированная
    const u = posArr[j] - (axS * y + axI);       // поперёк пути от оси
    const w = posArr[j + 2] - (fa + fb * y);     // высота над локальным полом
    const px = Math.round((cx + inv * u) * dpr);
    const py = Math.round((cy + inv * w) * dpr);
    if (px < 0 || py < 0 || px >= iw || py >= ih) continue;
    drawn++;
    buf32[py * iw + px] = 0xff000000
      | (colArr[j + 2] << 16) | (colArr[j + 1] << 8) | colArr[j];
  }
  // Рамка и масштаб -- в state: по ним видно, что рамка не зависит от кадра
  // (меняется только число попавших в плиту точек).
  state.sectionFrame = {
    umin, umax, wmin, wmax, sc, uc, wc, yNear, yFar, drawn, total: n,
    axis, floor: { a: fa, b: fb }, ms: performance.now() - t0,
  };
  ctx.putImageData(img, 0, 0);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  // 3) светлая сетка 1 м и подписи в метрах: по горизонтали u (поперёк пути),
  // по вертикали w (высота над полом)
  const step = sc >= 20 ? 1 : 2;
  ctx.lineWidth = 1;
  ctx.strokeStyle = 'rgba(120,140,170,0.18)';
  ctx.beginPath();
  for (let u = Math.ceil(umin / step) * step; u <= umax; u += step) {
    const px = Math.round(mapU(u)) + 0.5;
    ctx.moveTo(px, plotY); ctx.lineTo(px, plotY + plotH);
  }
  for (let w = Math.ceil(wmin / step) * step; w <= wmax; w += step) {
    const py = Math.round(mapW(w)) + 0.5;
    ctx.moveTo(plotX, py); ctx.lineTo(plotX + plotW, py);
  }
  ctx.stroke();

  ctx.fillStyle = 'rgba(138,151,176,0.9)';
  ctx.font = '9px "Segoe UI", system-ui, sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (let u = Math.ceil(umin / step) * step; u <= umax; u += step) {
    ctx.fillText(String(u), mapU(u), plotY + plotH + 3);
  }
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (let w = Math.ceil(wmin / step) * step; w <= wmax; w += step) {
    ctx.fillText(String(w), plotX - 4, mapW(w));
  }

  // 4) оверлеи
  ctx.setLineDash([]);
  ctx.strokeStyle = 'rgb(130,150,180)';          // пол: в выпрямленных координатах w = 0
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  const fy = mapW(0);
  ctx.moveTo(plotX, fy); ctx.lineTo(plotX + plotW, fy);
  ctx.stroke();

  ctx.strokeStyle = rgbOf(meta.zone_colors[3]);  // стены: их x как u (на середине окна)
  ctx.setLineDash([5, 4]);
  ctx.lineWidth = 1.25;
  ctx.beginPath();
  const axMid = axS * yc + axI;
  for (const x of meta.walls) {
    const px = mapU(x - axMid);
    if (px < plotX || px > plotX + plotW) continue;
    ctx.moveTo(px, plotY); ctx.lineTo(px, plotY + plotH);
  }
  ctx.stroke();

  ctx.strokeStyle = '#7fb2ff';                   // ось пути: u = 0 по построению
  ctx.setLineDash([2, 3]);
  ctx.lineWidth = 1.25;
  ctx.beginPath();
  const axp = mapU(0);
  if (axp >= plotX && axp <= plotX + plotW) {
    ctx.moveTo(axp, plotY); ctx.lineTo(axp, plotY + plotH);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // Рельсы: засечки от линии пола (w = 0) до высоты головки из /meta -- обе нити
  // стоят на ±колея/2 от оси и одной высоты.
  const railX = meta.rails.x || [];
  const railH = meta.rails.heights || [];
  ctx.strokeStyle = rgbOf(meta.rail_color);
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  for (let k = 0; k < railX.length; k++) {
    const px = mapU(railX[k] - axMid);
    if (px < plotX || px > plotX + plotW) continue;
    const h = Number(railH[k]);
    const top = Number.isFinite(h) && h > 0 ? h : 0.16;   // высота головки над полом
    ctx.moveTo(px, mapW(0));
    ctx.lineTo(px, mapW(top));
  }
  ctx.stroke();

  // 5) заголовок и сводка
  ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = '#dde4f0';
  ctx.font = '600 12px "Segoe UI", system-ui, sans-serif';
  const nearM = Math.abs(SECTION_Y_NEAR), farM = Math.abs(SECTION_Y_FAR);
  ctx.fillText(`Разрез X-Z, окно ${nearM}…${farM} м от сенсора; `
    + 'выпрямлено по оси и полу', 10, 15);
  ctx.fillStyle = 'rgba(138,151,176,0.95)';
  ctx.font = '10px "Segoe UI", system-ui, sans-serif';
  ctx.fillText(
    `пол z = ${meta.floor.a} + ${meta.floor.b}·y , ось x = ${meta.axis_x} м , `
    + `колея ${meta.gauge_mm} мм`, 10, 28);

  // Названия осей -- явно: по горизонтали мировой X (влево, как в 3D), по
  // вертикали высота над плоскостью пола.
  ctx.textAlign = 'left';
  ctx.fillStyle = 'rgba(138,151,176,0.95)';
  ctx.fillText('X, м (+X влево, как в 3D)', plotX, H - 3);
  ctx.save();
  ctx.translate(10, plotY + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText('высота над полом, м', 0, 0);
  ctx.restore();

  // 6) мини-легенда внутри панели
  const lx = plotX + plotW - 104, ly = plotY + 6;
  ctx.fillStyle = 'rgba(5,7,12,0.72)';
  ctx.strokeStyle = 'rgba(140,160,200,0.25)';
  ctx.lineWidth = 1;
  roundRect(ctx, lx, ly, 98, 60, 6);
  ctx.fill(); ctx.stroke();
  ctx.font = '10px "Segoe UI", system-ui, sans-serif';
  ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
  const legendRow = (row, color, dash, text) => {
    const y = ly + 12 + row * 12;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.setLineDash(dash);
    ctx.beginPath();
    ctx.moveTo(lx + 8, y); ctx.lineTo(lx + 26, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#dde4f0';
    ctx.fillText(text, lx + 32, y);
  };
  legendRow(0, 'rgb(130,150,180)', [], 'пол');
  legendRow(1, rgbOf(meta.rail_color), [], 'рельсы');
  legendRow(2, rgbOf(meta.zone_colors[3]), [4, 3], 'стены');
  legendRow(3, '#7fb2ff', [2, 3], 'ось');
}

// ------------------------------------------------- записи, метаданные, профиль --
// Переключение записи (датасета) и профиль вида на запись. Сервер в режиме корня
// держит несколько записей, поэтому `?bag=` понимают все маршруты, а профиль вида
// («Сохранить вид для этой записи») лежит в web/view_profiles.json на сервере.

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

// Всё, что зависит от метаданных записи: показ, раскраска, легенда, кадры,
// заголовок. Вызывается и на старте, и при переключении записи, поэтому ничего
// от прежней записи остаться не должно.
function applyMeta() {
  // Последний тик ползунка «Дальность» -- БЕЗ ОГРАНИЧЕНИЯ (∞): точки приходят с
  // сервера все (обрезания по дальности в кадре нет), 200 было только порогом
  // отрисовки в шейдере и жёстким максимумом ползунка.
  const distTick = Math.min(Number(meta.max_dist) || DIST_DEFAULT_M, DIST_INF_TICK);
  state.maxDist = distValue(distTick);
  $('dist').value = String(distTick);
  $('distVal').textContent = distIsInf(distTick) ? '∞' : String(distTick);
  $('psize').value = String(meta.point_size);
  $('psizeVal').textContent = meta.point_size;
  if (points) {
    points.material.uniforms.uMaxDist.value = state.maxDist;
    points.material.uniforms.uSize.value = meta.point_size;
  }
  $('frame').max = String(Math.max(0, meta.frames - 1));
  fillModeSelect();
  buildLegend();
  buildGrid();
  document.title = `LiDAR — ${meta.bag}`;
  // Ось и колея никуда не отправляются: они приходят из /meta (детектор + профиль
  // записи) и видны в HUD и разрезе. Ползунков модели рельсов в панели нет.
}

// Раскраска: список режимов общий для всех записей, поэтому выбор пользователя
// сохраняем, если он ещё доступен.
function fillModeSelect() {
  const sel = $('mode');
  const wanted = meta.modes.includes(state.mode) ? state.mode : meta.default_mode;
  sel.textContent = '';              // в DOM это снимает все прежние <option>
  for (const m of meta.modes) {
    const o = document.createElement('option');
    o.value = m; o.textContent = m;
    sel.appendChild(o);
  }
  sel.value = wanted;
  state.mode = wanted;
}

function fillBagSelect() {
  const sel = $('bag');
  if (!sel) return;
  const list = bagList();
  sel.textContent = '';
  for (const b of list) {
    const o = document.createElement('option');
    o.value = b.name;
    o.textContent = `${b.name} (${b.frames} кадр.)` + (b.has_profile ? ' · вид сохранён' : '');
    sel.appendChild(o);
  }
  sel.value = state.bag;
}

// Профиль вида записи: GET отдаёт файл, POST его перезаписывает. Профиль --
// удобство: если его нет, вид берётся из /meta записи, как раньше.
async function loadProfile(bag) {
  try {
    const r = await fetch(`/profile?bag=${encodeURIComponent(bag)}`, { cache: 'no-store' });
    if (!r.ok) return null;
    const j = await r.json();
    return j.profile || null;
  } catch (e) {
    console.warn('профиль вида не прочитан:', e);
    return null;
  }
}

// Пресет, который сейчас подсвечен в панели (в том числе «Объекты пути» и
// «Профиль рельса»): именно его и запишем в профиль.
function activePreset() {
  for (const b of $('presets').children) {
    if (b.classList.contains('active')) return b.dataset.preset;
  }
  const prof = $('view-profile');
  if (prof && prof.classList.contains('active')) return 'profile';
  return null;
}

function updateProfileNote() {
  const el = $('profileNote');
  if (!el) return;
  const p = state.profile;
  if (!p) {
    el.textContent = 'профиля вида нет — вид берётся из /meta записи';
    return;
  }
  const bits = [];
  if (p.preset) bits.push(`пресет ${p.preset}`);
  if (Array.isArray(p.position)) bits.push('position');
  if (Array.isArray(p.target)) bits.push('target');
  if (isNum(p.axis_x)) bits.push(`ось ${p.axis_x} м`);
  if (isNum(p.gauge_mm)) bits.push(`колея ${p.gauge_mm} мм`);
  if (isNum(p.max_dist)) bits.push(`дальность ${p.max_dist} м`);
  el.textContent = `профиль: ${bits.join(', ') || 'пустой'}`
    + (p.saved_at ? ` (${p.saved_at})` : '');
}

// Параметры профиля, которые видны сразу. Ось и колея сюда не входят: они уже
// применены на сервере при открытии записи (профиль -> аргументы Viewer) и пришли
// в /meta -- клиенту их некуда и незачем отправлять.
function applyProfileParams(profile) {
  if (!profile) return;
  if (isNum(profile.max_dist)) {
    const tick = Math.min(Number(profile.max_dist), DIST_INF_TICK);
    state.maxDist = distValue(tick);
    $('dist').value = String(tick);
    $('distVal').textContent = distIsInf(tick) ? '∞' : String(tick);
    if (points) points.material.uniforms.uMaxDist.value = state.maxDist;
    buildGrid();
  }
}

// Камера из профиля: пресет и/или явные position/target. Пресеты «объекты пути» и
// «профиль рельса» требуют мешей /track, поэтому ждём первую загрузку геометрии.
async function applyProfileView(profile) {
  if (!profile) return;
  const preset = profile.preset;
  if (preset === 'track' || preset === 'profile') {
    if (trackLayer) {
      await trackLayer.load(state.bag, 0).catch(fail);   // меши этой записи
      if (preset === 'track') cameraPreset('track');
      else trackLayer.viewProfile();
    }
  } else if (preset && PRESETS.some(([key]) => key === preset)) {
    cameraPreset(preset);
  }
  if (Array.isArray(profile.position) && profile.position.length === 3) {
    camera.position.set(profile.position[0], profile.position[1], profile.position[2]);
    controls.update();
  }
  if (Array.isArray(profile.target) && profile.target.length === 3) {
    controls.target.set(profile.target[0], profile.target[1], profile.target[2]);
    controls.update();
  }
}

async function saveProfile() {
  const body = {
    preset: activePreset(),
    position: [camera.position.x, camera.position.y, camera.position.z],
    target: [controls.target.x, controls.target.y, controls.target.z],
    axis_x: meta.axis_x,
    gauge_mm: meta.gauge_mm,
    max_dist: (distIsInf(state.maxDist) || state.maxDist > DIST_PRESET_SPAN_M) ? DIST_PRESET_SPAN_M : state.maxDist,
  };
  const url = `/profile?bag=${encodeURIComponent(state.bag)}`;
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`POST ${url} -> ${r.status}`);
  const j = await r.json();
  state.profile = j.profile || null;
  const entry = bagList().find((b) => b.name === state.bag);
  if (entry) entry.has_profile = true;       // пометка в списке записей
  fillBagSelect();
  updateProfileNote();
  noteCamera(`вид сохранён для записи «${state.bag}»`);
}

// Переключение записи: сначала убираем ВСЁ, что осталось от прежней (точки, меши,
// подписи, профиль вида), затем тянем метаданные, кадр, /track и профиль новой.
async function selectBag(name) {
  if (!name || name === state.bag || state.switching) return;
  // Несохранённая разметка привязана к СТАРОЙ записи: смена записи её стирает.
  // Спрашиваем, прежде чем терять десятки минут работы молча.
  if (label.on && label.dirty && label.objects.length
      && typeof window.confirm === 'function'
      && !window.confirm(`Разметка не сохранена: ${label.objects.length} объект(ов).`
        + ' Переключить запись и потерять её?')) {
    const bagSel = $('bag');
    if (bagSel) bagSel.value = state.bag;    // селектор возвращаем на место
    return;
  }
  state.switching = true;
  try {
    state.bag = name;
    state.idx = 0;
    state.counts = null;
    state.numPoints = 0;
    state.profile = null;
    cache.clear();
    $('profileNote').textContent = '';
    if (points) {
      points.geometry.setDrawRange(0, 0);      // ни одной чужой точки в кадре
      points.visible = false;
    }
    if (trackLayer) trackLayer.reset();        // меши, плашки, подписи -- прочь
    safeSafety('reset', (l) => l.reset());   // каркас коридора и подсветка точек
    // Объекты разметки принадлежат той записи, в которой поставлены: чужие боксы
    // на новом облаке были бы ошибкой счисления, поэтому список начинается заново.
    labelOnBag();
    sectionLastIdx = -1;                       // разрез перерисуется по новой записи
    updateHud();
    await loadMeta(name);
    applyMeta();                               // показ/легенда/раскраска/кадры
    const profile = await loadProfile(name);
    state.profile = profile;
    applyProfileParams(profile);               // дальность из профиля (ось/колея -- в /meta)
    await showFrame(0);
    const hasView = profile && (profile.preset || profile.position || profile.target);
    await applyProfileView(profile);           // камера: пресет и/или position/target
    if (!hasView && !(trackLayer && trackLayer.isAutoView())) {
      // Профиля вида у записи нет: вид по умолчанию -- пресет «От лидара» (при
      // включённом автовиде стартовую камеру ставит слой, когда придут меши).
      cameraPreset('lidar');
      $('presets').children[0].classList.add('active');
    }
    applyPointsVisible();
    updateProfileNote();
    updateHud();
  } catch (e) {
    fail(e);
  } finally {
    state.switching = false;
  }
}

async function main() {
  await loadMeta();
  // state.maxDist уже выставлен в applyMeta по тику ползунка (последний тик = ∞)

  const canvas = $('view');
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  renderer.setClearColor(0x05070c, 1);

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 4000);
  camera.up.set(0, 0, 1);
  camera.position.set(meta.axis_x, 2, 1.6);

  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.screenSpacePanning = false;
  controls.maxDistance = 1500;
  controls.target.set(meta.axis_x, -20, -0.5);
  controls.update();

  makePoints();
  buildStaticOverlays();
  // Слой объектов пути: свой свет, меши из /track, вписывание камеры по
  // рельсам и шпалам, режим «чистое поле». Облако точек он только трогает
  // через refreshPoints -- владелец видимости остаётся здесь. Если модуль не
  // подключился (нет /track3d.js) -- страница едет дальше без объектов пути,
  // а в панели написано, что слоя нет.
  if (createTrackLayer) {
    trackLayer = createTrackLayer({
      scene, camera, controls,
      refreshPoints: applyPointsVisible,
      onError: fail,
    });
  } else {
    showTrackUnavailable(createTrackLayerError);
  }
  // Слой коридора безопасности: ни света, ни камеры ему не нужно -- каркас рисуется
  // LineBasicMaterial, а подсветка точек идёт по тумблеру блока. Метаданные передаём
  // сразу, дальше их обновляет loadMeta() (на старте и при смене записи). Нет модуля
  // -- страница живёт без него, блок в панели говорит, почему рисунка нет (wireUi).
  if (createSafetyLayer) {
    // Создание идёт НЕ через safeSafety: у него первая же строка -- «слоя ещё нет,
    // выходим», и фабрика просто не вызывалась (safetyLayer оставался null, а блок
    // в панели писал «слой не загрузился»). Ловим ошибку фабрики здесь, дальше все
    // вызовы -- через safeSafety.
    try {
      safetyLayer = createSafetyLayer({ scene, THREE });
    } catch (e) {
      safetyLayer = null;
      safetyBroken = `создание слоя: ${e && e.message ? e.message : e}`;
      console.warn('safetylayer: слой коридора безопасности не создался —', e);
      const note = $('safetyNote');
      if (note) {
        note.textContent = `слой коридора не создался (${safetyBroken}). `
          + 'Остальная страница работает: облако, кадры, объекты пути, разрез.';
      }
    }
    safeSafety('update', (l) => l.update(meta));
    // По умолчанию коридор выключен (галочка в разметке снята): панель и картинка
    // спокойны, пока пользователь сам не включит объём.
    safeSafety('setVisible', (l) => l.setVisible($('tun-on') ? $('tun-on').checked : false));
  }
  // Слой разметки: свои группы, своё состояние; облако он читает через аксессор
  // (владелец видимости и данных -- по-прежнему эта страница). Создание -- ДО wireUi:
  // обработчики панели разметки зависят от того, создался ли слой.
  if (createLabelLayer) {
    try {
      labelLayer = createLabelLayer({
        scene, camera, controls, canvas,
        cloud: () => ({ positions: posArr, count: state.numPoints, maxDist: state.maxDist }),
        onPick: (point) => labelPick(point),
        // Клик по невыбранному объекту выбирает его (см. labellayer.pickBox):
        // выбрать мышью по сцене, а не только строкой списка в панели.
        onSelect: (key) => {
          labelSelect(key);
          labelDraw();
          renderLabel();
          setLabelStatus('объект выбран — тяните за бокс, ручки на гранях');
        },
        onDragMove: (x, y, z) => {
          const o = labelSelected();
          if (!o) return;
          // Мышью правят В ТЕКУЩЕМ кадре, а центр объекта хранится в опорном:
          // переводим тем же ходом, которым объект проецируется на кадр, иначе
          // на не-опорном кадре (ход ~1 м/кадр) объект прыгал бы на полметра.
          const dy = labelMotionBetween(o, state.idx, o.refFrame);
          labelPatch(o.key, { center: [x, y + dy, z] }, { silent: true });
        },
        onDragStart: () => labelPushUndo(),
        onDragEnd: () => { schedulePreview(); setLabelStatus('правка мышью: переставлено'); },
        onEdit: (edit) => labelApplyEdit(edit),
        onStatus: setLabelStatus,
        // Закрытые точки кадра: слой складывает наборы ВСЕХ объектов и отдаёт один
        // список -- вьюер гасит ровно его (`applyHiddenPoints`).
        onHidden: (indices) => {
          if (label.on) applyHiddenPoints(indices);
        },
      });
    } catch (e) {
      labelLayer = null;
      createLabelLayerError = e;
      console.warn('labellayer: слой разметки не создался —', e);
    }
    // Без проверки на null исключение фабрики обрывало main() ЗДЕСЬ: панель
    // оставалась без обработчиков (галочки «ничего не делают»), а причина не
    // читалась нигде. Теперь причина видна в блоке разметки (labelWire).
    if (labelLayer) labelLayer.setVisible(Boolean($('t-label') && $('t-label').checked));
  }
  buildLegend();
  wireUi();

  // Ползунки, раскраска, легенда, кадры -- всё из метаданных этой записи, и список
  // записей для селектора (в одиночном запуске он из одной строки).
  applyMeta();
  fillBagSelect();
  updateProfileNote();
  // Ход записи нужен разметке сразу, даже пока режим выключен: замер на сервере
  // кэширован, а без него первая же разметка ждала бы секунды.
  labelLoadMotion();

  buildGrid();
  initSection();
  await showFrame(0);
  updateHud();
  state.nextTick = performance.now();

  let last = performance.now();
  function loop(now) {
    requestAnimationFrame(loop);
    const dt = now - last;
    last = now;
    if (dt > 0) state.fpsShown = state.fpsShown ? state.fpsShown * 0.9 + (1000 / dt) * 0.1 : 1000 / dt;

    if (state.playing && !state.loading && now >= state.nextTick) {
      const period = 1000 / Math.max(state.fps, 0.1);
      state.nextTick = Math.max(now + period, state.nextTick + period);
      const t0 = performance.now();
      state.loading = true;
      showFrame(state.idx + 1 >= meta.frames ? 0 : state.idx + 1)
        .then(() => { state.lastFrameMs = performance.now() - t0; updateHud(); })
        .catch(fail)
        .finally(() => { state.loading = false; });
    }
    controls.update();
    renderer.render(scene, camera);
    drawSection();
  }
  requestAnimationFrame(loop);
  // Стартовый вид, по приоритету:
  //   1) профиль вида этой записи (сохранён кнопкой «Сохранить вид для записи»);
  //   2) автовид по объектам (галочка в панели, по умолчанию СНЯТА);
  //   3) пресет «От лидара» -- как раньше.
  // По умолчанию (галочка снята, профиля нет) камеру ставит пресет «От лидара»,
  // и он же подсвечен: слой при этом камеру не трогает.
  const startProfile = await loadProfile(state.bag);
  state.profile = startProfile;
  if (startProfile
      && (startProfile.preset || startProfile.position || startProfile.target)) {
    await applyProfileView(startProfile);
    noteCamera(`вид из профиля записи «${state.bag}» — переставить можно пресетом`);
  } else if (trackLayer && trackLayer.isAutoView()) {
    const note = $('cameraNote');
    if (note) note.textContent = 'стартовая камера: вид по объектам пути (рельсы и шпалы), окно 12 м — ставит слой';
  } else {
    cameraPreset('lidar');
    $('presets').children[0].classList.add('active');
  }
  updateProfileNote();
}

// Автообновление: как только сервер увидит, что файлы фронтенда изменились,
// страница перезагрузится сама -- не нужно просить F5 при каждой правке.
let currentVersion = null;
async function pollVersion() {
  try {
    const r = await fetch('/version', { cache: 'no-store' });
    const j = await r.json();
    if (currentVersion === null) currentVersion = j.version;
    else if (j.version !== currentVersion) {
      // Несохранённая разметка дороже автообновления: перезагрузка отложится,
      // в статусе -- просьба сохранить. После сохранения dirty=false и следующий
      // опрос перезагрузит страницу сам.
      if (label.on && label.dirty && label.objects.length) {
        setLabelStatus('фронтенд обновился — сохраните разметку, страница перезагрузится после');
        return;
      }
      location.reload(); return;
    }
  } catch (e) { /* сервер перезапускается -- просто ждём */ }
  setTimeout(pollVersion, 1000);
}
// Несохранённая разметка не должна пропадать молча при закрытии вкладки:
// браузер спросит, прежде чем уйти.
window.addEventListener('beforeunload', (ev) => {
  if (!(label.on && label.dirty && label.objects.length)) return;
  ev.preventDefault();
  ev.returnValue = '';
});

pollVersion();

main().catch(fail);