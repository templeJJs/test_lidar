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
let capacity = 0;
let posArr, colArr;
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

function applyPointsVisible() {
  points.visible = pointsOn && !(trackLayer && trackLayer.isCleanField());
  syncRailRibbon();      // виден ли наш меш рельсов -> гасить зелёную полосу middle
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
  const geom = new THREE.BufferGeometry();
  posAttr = new THREE.BufferAttribute(posArr, 3);
  posAttr.setUsage(THREE.DynamicDrawUsage);
  colAttr = new THREE.BufferAttribute(colArr, 3, true); // normalized uint8
  colAttr.setUsage(THREE.DynamicDrawUsage);
  geom.setAttribute('position', posAttr);
  geom.setAttribute('aColor', colAttr);
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
      uniform float uSize;
      uniform float uPixelRatio;
      uniform float uMaxDist;
      varying vec3 vColor;
      varying float vKeep;
      void main() {
        vColor = aColor;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mv;
        vKeep = (-position.y <= uMaxDist) ? 1.0 : 0.0;
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
  // Подсветка точек внутри туннеля -- сразу после applyFrame: цвета кадра только
  // что перезаписаны, и красить нужно уже их (порядок важен, не косметика).
  safeSafety('markPoints', (l) => l.markPoints(posArr, colArr, state.numPoints));
  colAttr.needsUpdate = true;
  state.idx = idx;
  $('frame').value = String(idx);
  updateFrameCopy();
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

function cameraPreset(name) {
  const cx = meta.axis_x;
  const ymid = -state.maxDist / 2;
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
    state.maxDist = Number(e.target.value);
    $('distVal').textContent = state.maxDist;
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
    if (ev.code === 'Space') { ev.preventDefault(); $('play').click(); }
    else if (ev.code === 'ArrowLeft') { ev.preventDefault(); showFrame(state.idx - 1).catch(fail); }
    else if (ev.code === 'ArrowRight') { ev.preventDefault(); showFrame(state.idx + 1).catch(fail); }
    else if (ev.code === 'Tab') { ev.preventDefault(); $('panel').classList.toggle('hidden'); }
  });
  window.addEventListener('resize', onResize);
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
  state.maxDist = meta.max_dist;
  $('dist').value = String(state.maxDist);
  $('distVal').textContent = state.maxDist;
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
    state.maxDist = profile.max_dist;
    $('dist').value = String(profile.max_dist);
    $('distVal').textContent = profile.max_dist;
    if (points) points.material.uniforms.uMaxDist.value = profile.max_dist;
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
    max_dist: state.maxDist,
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
  state.maxDist = meta.max_dist;

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
  buildLegend();
  wireUi();

  // Ползунки, раскраска, легенда, кадры -- всё из метаданных этой записи, и список
  // записей для селектора (в одиночном запуске он из одной строки).
  applyMeta();
  fillBagSelect();
  updateProfileNote();

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
    else if (j.version !== currentVersion) { location.reload(); return; }
  } catch (e) { /* сервер перезапускается -- просто ждём */ }
  setTimeout(pollVersion, 1000);
}
pollVersion();

main().catch(fail);