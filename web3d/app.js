// Track3D: 3D-объекты пути (рельсы, шпалы, пол) + опционально облако точек.
//
// Данные лидара идут "как есть": X поперёк, Y вдоль (вперёд -- -Y), Z вверх.
// Меши пути приходят из track_geometry уже в этих осях, поэтому к ним не
// применяется никакой поворот. Меши статичны: собираются при загрузке данных
// (не в цикле анимации), старая геометрия и материал диспоузятся при пересборке.

import * as THREE from 'three';
import { OrbitControls } from './OrbitControls.js';

const $ = (id) => document.getElementById(id);
const errBox = $('err');

function fail(e) {
  errBox.style.display = 'block';
  errBox.textContent = String(e && e.stack ? e.stack : e);
  console.error(e);
}

function clearError() {
  errBox.style.display = 'none';
  errBox.textContent = '';
}

// --------------------------------------------------------------- оси мира ---
// Единственное место, где задаётся "верх": three по умолчанию считает верхом +Y,
// а у наших данных верх -- +Z. Мутация глобальная (Object3D.DEFAULT_UP), поэтому
// делается ровно один раз и до создания камеры с контролами.
function setupWorldUp() {
  THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
}

const DARK_BG = 0x05070c;
const CLEAN_BG = 0xe9ecf1;

// Порядки отрисовки: сплошные объекты не просвечивают друг через друга.
// Все материалы непрозрачные, depthTest/depthWrite включены, пол смещён
// polygonOffset, чтобы не мерцать с нижними гранями шпал.
const ORDER = { floor: 0, points: 1, sleepers: 5, rails: 6 };

const RAIL_COLOR = { left: 0x63c6ff, right: 0xffb454, other: 0xffd479 };
const FLOOR_COLOR = 0x59637a;
const SLEEPER_COLOR = 0x9a7b4f;
const POINT_MODE = 'height';
const CORRIDOR_VIEW_SPAN = 60;   // сколько метров коридора показывать в стартовом виде

const state = {
  meta: null,
  bag: null,
  frame: 0,
  frames: 1,
  data: null,
  clean: false,
  saved: null,
  pointCount: 0,
  visible: { points: true, rails: true, sleepers: true, floor: true },
  loading: false,
};

let renderer, scene, camera, controls;
let pointsObj = null;
const groups = { floor: null, sleepers: null, rails: null };

// --------------------------------------------------------- происхождение ----

// track_geometry отдаёт source как 'measured', 'gost', 'gost R65',
// 'gost R65 (не измерено)' -- приводим к русской пометке для легенды.
const SOURCE_TEXT = {
  measured: 'измерено', lidar: 'измерено (лидар)', observed: 'измерено',
  gost: 'ГОСТ', 'гост': 'ГОСТ', standard: 'ГОСТ',
  meta: 'по метаданным', floor: 'по метаданным',
};
const RAIL_PART = { head: 'головка', web: 'шейка', foot: 'подошва' };

function sourceText(value) {
  if (value === null || value === undefined || value === '') return '—';
  const raw = String(value).trim();
  const known = SOURCE_TEXT[raw.toLowerCase()];
  if (known) return known;
  return raw
    .replace(/^gost/i, 'ГОСТ')
    .replace(/\bgost\b/i, 'ГОСТ')
    .replace(/r65/gi, 'Р65');
}

// CSS-класс пометки: зелёная -- измерено, жёлтая -- стандарт.
function sourceKind(value) {
  const key = String(value === null || value === undefined ? '' : value)
    .trim().toLowerCase();
  if (/gost|гост|standard|r65|р65/.test(key)) return 'gost';
  if (/measur|измер|наблюд|lidar|observed/.test(key)) return 'measured';
  return '';
}

function railSourceParts(rail) {
  const src = (rail && rail.source) || {};
  return ['head', 'web', 'foot']
    .filter((k) => src[k])
    .map((k) => `${RAIL_PART[k]} — ${sourceText(src[k])}`);
}

function meshStats(mesh) {
  const v = mesh && Array.isArray(mesh.vertices) ? mesh.vertices.length : 0;
  const f = mesh && Array.isArray(mesh.faces) ? mesh.faces.length : 0;
  return { vertices: v, faces: f };
}

function isNumber(v) {
  return typeof v === 'number' && Number.isFinite(v);
}

function fmt(v, digits = 4) {
  return isNumber(v) ? v.toFixed(digits) : String(v);
}

// ---------------------------------------------------- построение геометрии ---

function objectMaterial(color) {
  return new THREE.MeshStandardMaterial({
    color, metalness: 0.15, roughness: 0.75,
    flatShading: true, side: THREE.DoubleSide,
  });
}

function railMaterial(side) {
  const key = (side === 'left' || side === -1) ? 'left'
    : (side === 'right' || side === 1) ? 'right' : 'other';
  return objectMaterial(RAIL_COLOR[key]);
}

// faces могут прийти треугольниками (по контракту) или многоугольниками -- режем веером.
function triangulate(faces) {
  const index = [];
  for (const face of faces || []) {
    if (!face || face.length < 3) continue;
    for (let k = 1; k + 1 < face.length; k++) index.push(face[0], face[k], face[k + 1]);
  }
  return index;
}

function meshFromArrays(vertices, faces, material) {
  const positions = new Float32Array(vertices.length * 3);
  for (let i = 0; i < vertices.length; i++) {
    const v = vertices[i] || [0, 0, 0];
    positions[i * 3] = v[0];
    positions[i * 3 + 1] = v[1];
    positions[i * 3 + 2] = v[2];
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const index = triangulate(faces);
  if (index.length) geom.setIndex(index);
  geom.computeVertexNormals();
  return new THREE.Mesh(geom, material);
}

function disposeObject(obj) {
  obj.traverse((node) => {
    if (node.geometry) node.geometry.dispose();
    const mat = node.material;
    if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
    else if (mat) mat.dispose();
  });
}

function dropGroup(key) {
  const group = groups[key];
  if (!group) return;
  scene.remove(group);
  disposeObject(group);
  groups[key] = null;
}

function dropAllObjects() {
  dropGroup('floor');
  dropGroup('sleepers');
  dropGroup('rails');
}

function buildFloor(data) {
  dropGroup('floor');
  const mesh = data.floor_mesh || {};
  const vertices = mesh.vertices || [];
  if (!vertices.length) return;
  const material = objectMaterial(FLOOR_COLOR);
  material.polygonOffset = true;      // не мерцать с нижними гранями шпал
  material.polygonOffsetFactor = 1;
  material.polygonOffsetUnits = 1;
  const group = new THREE.Group();
  group.name = 'floor';
  const m = meshFromArrays(vertices, mesh.faces, material);
  m.renderOrder = ORDER.floor;
  group.add(m);
  group.renderOrder = ORDER.floor;
  groups.floor = group;
  scene.add(group);
}

function buildRails(data) {
  dropGroup('rails');
  const rails = data.rails || [];
  const group = new THREE.Group();
  group.name = 'rails';
  for (const rail of rails) {
    const mesh = rail && rail.mesh;
    if (!mesh || !Array.isArray(mesh.vertices) || !mesh.vertices.length) continue;
    const m = meshFromArrays(mesh.vertices, mesh.faces, railMaterial(rail.side));
    m.renderOrder = ORDER.rails;
    group.add(m);
  }
  if (!group.children.length) return;
  group.renderOrder = ORDER.rails;
  groups.rails = group;
  scene.add(group);
}

function buildSleepers(data) {
  dropGroup('sleepers');
  const sleepers = data.sleepers || {};
  const instances = sleepers.instances || [];
  if (!instances.length) return;

  const group = new THREE.Group();
  group.name = 'sleepers';
  const material = objectMaterial(SLEEPER_COLOR);

  // BoxGeometry(w, h, d) откладывает w по X, h по Y, d по Z локальных осей меша.
  // Меш не поворачиваем, поэтому в осях данных это ровно
  // (размер_x, размер_y, размер_z) -- иначе шпала встанет вертикальной стенкой.
  const sizes = instances.map((i) => (i && i.size) || [0.25, 0.22, 0.18]);
  const uniform = sizes.every((s) => s[0] === sizes[0][0] && s[1] === sizes[0][1]
    && s[2] === sizes[0][2]);

  if (uniform) {
    const geom = new THREE.BoxGeometry(sizes[0][0], sizes[0][1], sizes[0][2]);
    const mesh = new THREE.InstancedMesh(geom, material, instances.length);
    const mtx = new THREE.Matrix4();
    instances.forEach((inst, i) => {
      const c = (inst && inst.center) || [0, 0, 0];
      mesh.setMatrixAt(i, mtx.makeTranslation(c[0], c[1], c[2]));
    });
    mesh.instanceMatrix.needsUpdate = true;
    mesh.renderOrder = ORDER.sleepers;
    group.add(mesh);
  } else {
    instances.forEach((inst) => {
      const c = (inst && inst.center) || [0, 0, 0];
      const s = (inst && inst.size) || [0.25, 0.22, 0.18];
      const m = new THREE.Mesh(new THREE.BoxGeometry(s[0], s[1], s[2]), material);
      m.position.set(c[0], c[1], c[2]);
      m.renderOrder = ORDER.sleepers;
      group.add(m);
    });
  }
  group.renderOrder = ORDER.sleepers;
  groups.sleepers = group;
  scene.add(group);
}

function rebuildTrackObjects() {
  const data = state.data;
  if (!data) { dropAllObjects(); return; }
  buildFloor(data);
  buildSleepers(data);
  buildRails(data);
  applyVisibility();
}

// ------------------------------------------------------------- облако точек --

function buildPoints(buffer) {
  const u8 = new Uint8Array(buffer);
  const magic = String.fromCharCode(u8[0], u8[1], u8[2], u8[3]);
  if (magic !== 'T3DP') throw new Error(`bad points magic ${JSON.stringify(magic)}`);
  const dv = new DataView(buffer);
  const kind = dv.getUint8(5);
  const n = dv.getUint32(8, true);
  if (kind !== 0) throw new Error(`unknown points kind ${kind}`);
  // заголовок 12 байт -- смещение позиций кратно 4, вью без копии
  const positions = new Float32Array(buffer, 12, n * 3);
  const colors8 = new Uint8Array(buffer, 12 + n * 12, n * 3);

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('aColor', new THREE.BufferAttribute(colors8, 3, true));

  const material = new THREE.ShaderMaterial({
    uniforms: {
      uSize: { value: 2.6 },
      uPixelRatio: { value: Math.min(window.devicePixelRatio || 1, 2) },
    },
    vertexShader: `
      attribute vec3 aColor;
      uniform float uSize;
      uniform float uPixelRatio;
      varying vec3 vColor;
      void main() {
        vColor = aColor;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mv;
        float d = max(-mv.z, 0.5);
        gl_PointSize = clamp(uSize * uPixelRatio * (14.0 / d), 1.0, 60.0);
      }`,
    fragmentShader: `
      precision mediump float;
      varying vec3 vColor;
      void main() { gl_FragColor = vec4(vColor, 1.0); }`,
    depthTest: true,
    depthWrite: true,
  });

  const obj = new THREE.Points(geom, material);
  obj.frustumCulled = false;
  obj.renderOrder = ORDER.points;

  if (pointsObj) {
    scene.remove(pointsObj);
    disposeObject(pointsObj);
  }
  pointsObj = obj;
  scene.add(obj);
  state.pointCount = n;
  applyVisibility();
}

async function loadPoints() {
  const url = `/points?bag=${encodeURIComponent(state.bag)}`
    + `&frame=${state.frame}&mode=${POINT_MODE}`;
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) throw new Error(`GET /points -> ${res.status}`);
  buildPoints(await res.arrayBuffer());
}

// ------------------------------------------------------------ видимость -----

function applyVisibility() {
  if (pointsObj) pointsObj.visible = state.visible.points && !state.clean;
  for (const key of ['floor', 'sleepers', 'rails']) {
    if (groups[key]) groups[key].visible = state.visible[key];
  }
}

// ------------------------------------------------- вид "чистое поле" --------

function boundsFromData(data) {
  const box = new THREE.Box3();
  const v = new THREE.Vector3();
  let any = false;
  const eat = (vertices) => {
    for (const p of vertices || []) {
      if (!p || p.length < 3) continue;
      if (!Number.isFinite(p[0]) || !Number.isFinite(p[1]) || !Number.isFinite(p[2])) continue;
      box.expandByPoint(v.set(p[0], p[1], p[2]));
      any = true;
    }
  };
  for (const rail of data.rails || []) eat(rail && rail.mesh && rail.mesh.vertices);
  eat((data.floor_mesh || {}).vertices);
  for (const inst of ((data.sleepers || {}).instances || [])) {
    const c = inst && inst.center;
    const s = inst && inst.size;
    if (!c || !s) continue;
    box.expandByPoint(v.set(c[0] - s[0] / 2, c[1] - s[1] / 2, c[2] - s[2] / 2));
    box.expandByPoint(v.set(c[0] + s[0] / 2, c[1] + s[1] / 2, c[2] + s[2] / 2));
    any = true;
  }
  return any ? box : null;
}

// Габарит считается по мешам (вершины/габариты инстансов из JSON), не по облаку.
function boundsOfObjects() {
  const fromData = state.data ? boundsFromData(state.data) : null;
  if (fromData) return fromData;
  const box = new THREE.Box3();
  let any = false;
  for (const key of ['floor', 'sleepers', 'rails']) {
    if (!groups[key]) continue;
    const b = new THREE.Box3().setFromObject(groups[key]);
    if (b.isEmpty()) continue;
    box.union(b);
    any = true;
  }
  return any ? box : null;
}

// Камера сбоку-сзади-сверху: коридор уходит в -Y, значит смотрим из +Y.
function cameraDirection() {
  return new THREE.Vector3(0.55, 0.60, 0.58).normalize();
}

function fitCameraToBox(box, margin = 1.6) {
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(size.length() * 0.5, 0.5);
  const fov = THREE.MathUtils.degToRad(camera.fov);
  const distance = (radius / Math.sin(fov / 2)) * margin;
  camera.position.copy(center).addScaledVector(cameraDirection(), distance);
  controls.target.copy(center);
  camera.near = Math.max(distance / 2000, 0.05);
  camera.far = Math.max(distance * 12, 900);
  camera.updateProjectionMatrix();
  controls.update();
}

// Стартовый вид: ближние CORRIDOR_VIEW_SPAN метров коридора (у сенсора y ~ 0).
function corridorViewBox(box) {
  const b = box.clone();
  b.min.y = Math.max(b.min.y, b.max.y - CORRIDOR_VIEW_SPAN);
  return b;
}

function setCleanField(on) {
  if (on === state.clean) return;
  state.clean = on;
  if (on) {
    state.saved = {
      position: camera.position.clone(),
      target: controls.target.clone(),
      near: camera.near,
      far: camera.far,
      background: scene.background ? scene.background.clone() : null,
    };
    scene.background = new THREE.Color(CLEAN_BG);
    const box = boundsOfObjects();
    if (box) fitCameraToBox(box);
    else fail(new Error('нет мешей пути: габарит для «чистого поля» не построить'));
  } else if (state.saved) {
    camera.position.copy(state.saved.position);
    controls.target.copy(state.saved.target);
    camera.near = state.saved.near;
    camera.far = state.saved.far;
    camera.updateProjectionMatrix();
    scene.background = state.saved.background;
    controls.update();
    state.saved = null;
  }
  applyVisibility();
  $('clean').classList.toggle('active', on);
  $('clean').textContent = on ? 'Выйти из «чистого поля»' : 'Чистое поле';
  // В «чистом поле» облако не грузится; если его ещё нет, а тумблер включён --
  // догружаем при выходе, иначе точки не появятся до смены кадра.
  if (!on && state.visible.points && !pointsObj) {
    loadPoints().then(updateHud).catch(fail);
  }
  updateHud();
}

// ------------------------------------------------------------ панель --------

function setSourceBadge(id, text, kind) {
  const el = $(id);
  el.textContent = text;
  el.className = 'src' + (kind ? ' ' + kind : '');
}

function updateSourceBadges() {
  const data = state.data;
  setSourceBadge('src-points', 'измерено', 'measured');
  if (!data) {
    for (const id of ['src-rails', 'src-sleepers', 'src-floor']) setSourceBadge(id, '—', '');
    return;
  }
  const rails = (data.rails || []).filter((r) => r && r.source);
  const kinds = new Set();
  for (const rail of rails) for (const k of ['head', 'web', 'foot']) {
    const kind = sourceKind((rail.source || {})[k]);
    if (kind) kinds.add(kind);
  }
  const railText = kinds.has('measured') && kinds.has('gost') ? 'измерено + ГОСТ'
    : kinds.has('measured') ? 'измерено'
      : kinds.has('gost') ? 'ГОСТ' : '—';
  setSourceBadge('src-rails', railText,
    kinds.has('measured') ? 'measured' : (kinds.has('gost') ? 'gost' : ''));

  const sleepers = data.sleepers || {};
  const sleeperSrc = sleepers.source;
  setSourceBadge('src-sleepers', sourceText(sleeperSrc), sourceKind(sleeperSrc));

  const floorSrc = (data.floor_mesh || {}).source;
  setSourceBadge('src-floor', sourceText(floorSrc), sourceKind(floorSrc));
}

function legendRow(swatch, title, subtitle) {
  const row = document.createElement('div');
  const mark = document.createElement('i');
  mark.style.background = swatch;
  const text = document.createElement('div');
  text.textContent = title;
  if (subtitle) {
    const em = document.createElement('em');
    em.textContent = subtitle;
    text.appendChild(em);
  }
  row.appendChild(mark);
  row.appendChild(text);
  $('legend').appendChild(row);
}

function updateLegend() {
  const box = $('legend');
  box.innerHTML = '';
  const data = state.data;
  legendRow('#dde4f0', 'Облако точек — измерено (лидар)',
    'раскраска по высоте (turbo), до 40 м вперёд от сенсора');
  if (!data) {
    legendRow('#6b7280', 'Рельсы — данных нет',
      'track_geometry недоступен: /track не вернул геометрию');
    return;
  }
  const rails = data.rails || [];
  const first = rails.find((r) => r && r.mesh && (r.mesh.vertices || []).length);
  const parts = first ? railSourceParts(first) : [];
  if (rails.length) {
    legendRow('linear-gradient(90deg,#63c6ff,#ffb454)', `Рельсы: ${rails.length} шт`,
      'левая нить (меньший X) — синяя, правая — жёлтая; '
      + (parts.length ? parts.join(' · ') : 'происхождение в source не указано'));
  } else {
    legendRow('#6b7280', 'Рельсы: мешей нет', 'build_track вернул пустой rails');
  }

  const sleepers = data.sleepers || {};
  const instances = sleepers.instances || [];
  const sleepersBox = instances.length
    ? `шаг ${Number(sleepers.spacing_m || 0).toFixed(3)} м, бокс ${(instances[0].size || [])
      .map((v) => Number(v).toFixed(2)).join(' × ')} м (X × Y × Z)`
    : 'инстансов нет';
  legendRow('#' + new THREE.Color(SLEEPER_COLOR).getHexString(),
    `Шпалы: ${instances.length} шт — ${sourceText(sleepers.source)}`, sleepersBox);

  const floor = data.floor_mesh || {};
  const stats = meshStats(floor);
  const plane = data.floor || {};
  const planeText = [plane.a, plane.b, plane.c].every(isNumber)
    ? `z = ${fmt(plane.a, 5)}·x + ${fmt(plane.b, 5)}·y + ${fmt(plane.c, 3)}` : '';
  legendRow('#' + new THREE.Color(FLOOR_COLOR).getHexString(),
    `Пол — ${sourceText(floor.source)}`,
    (stats.vertices ? `${stats.vertices} вершин, ${stats.faces} граней` : 'меша пола нет')
    + (planeText ? `; плоскость ${planeText}` : ''));
}

function updateHud() {
  if (!state.data) {
    const tg = state.meta ? state.meta.track_geometry : null;
    $('hud').textContent = 'объекты пути недоступны'
      + (tg && !tg.available ? '\n' + tg.error : '')
      + `\nкадр ${state.frame + 1} / ${state.frames}`
      + (pointsObj ? `\nточек ${state.pointCount}` : '');
    return;
  }
  const data = state.data;
  const meta = data.meta || {};
  const lines = [];
  lines.push(`bag ${state.bag}   кадр ${state.frame + 1} / ${state.frames}`);
  lines.push(`frame в JSON: ${data.frame !== undefined ? data.frame : '—'}`
    + `   точек ${pointsObj ? state.pointCount : 0}`);
  if (meta.gauge_inner_mm !== undefined && meta.gauge_inner_mm !== null) {
    lines.push(`колея (внутренние грани): ${meta.gauge_inner_mm} мм`
      + (meta.gauge_axis_mm !== undefined && meta.gauge_axis_mm !== null
        ? `   ось—ось: ${meta.gauge_axis_mm} мм` : ''));
  }
  if (meta.head_width_mm !== undefined && meta.head_width_mm !== null) {
    lines.push(`ширина головки: ${meta.head_width_mm} мм`);
  }
  if (meta.sensor_height_above_head_m !== undefined
    && meta.sensor_height_above_head_m !== null) {
    lines.push(`сенсор над верхом головки: ${meta.sensor_height_above_head_m} м`);
  }
  const rails = data.rails || [];
  lines.push(`рельсов ${rails.length}`
    + (rails.length ? `   меш: ${rails.map((r) => meshStats(r && r.mesh).vertices).join('+')} вершин`
      + ` / ${rails.map((r) => meshStats(r && r.mesh).faces).join('+')} граней` : ''));
  for (const rail of rails) {
    const h = rail.head || {};
    lines.push(`  нить ${rail.side}: внутр. грань x = ${fmt(h.inner_face_x, 3)}, `
      + `верх z = ${fmt(h.top_z, 3)}, головка ${fmt(h.width_mm, 1)} мм`
      + (isNumber(h.height_above_adjacent_m)
        ? `, высота над прилегающей ${fmt(h.height_above_adjacent_m, 3)} м` : ''));
    if (rail.source) lines.push(`    source: ${railSourceParts(rail).join(' · ')}`);
  }
  const sleepers = data.sleepers || {};
  lines.push(`шпал ${(sleepers.instances || []).length}`
    + (sleepers.spacing_m ? `   шаг ${Number(sleepers.spacing_m).toFixed(3)} м` : ''));
  const floorStats = meshStats(data.floor_mesh || {});
  lines.push(`пол: ${floorStats.vertices} вершин / ${floorStats.faces} граней`);
  // Семантика из track_geometry: axis {s,i} -> x = s*y + i, floor {a,b,c} -> z = a*x+b*y+c.
  // Если поля не числа (null из-за NaN) -- показываем как есть, без домыслов.
  const ax = data.axis || {};
  const fl = data.floor || {};
  lines.push(isNumber(ax.s) && isNumber(ax.i)
    ? `axis  x = ${fmt(ax.s, 5)}·y + ${fmt(ax.i, 3)}`
    : `axis  ${JSON.stringify(data.axis)}`);
  lines.push(isNumber(fl.a) && isNumber(fl.b) && isNumber(fl.c)
    ? `floor z = ${fmt(fl.a, 5)}·x + ${fmt(fl.b, 5)}·y + ${fmt(fl.c, 3)}`
    : `floor ${JSON.stringify(data.floor)}`);
  // meta.notes приходит и строкой, и списком строк -- показываем по строкам.
  if (meta.notes) {
    const notes = Array.isArray(meta.notes) ? meta.notes : [meta.notes];
    for (const note of notes) lines.push(`notes: ${note}`);
  }
  if (state.clean) lines.push('режим: чистое поле');
  $('hud').textContent = lines.join('\n');
}

function updateFrameLabel() {
  $('frameVal').textContent = `${state.frame + 1} / ${state.frames}`;
}

// ------------------------------------------------------------ загрузка ------

async function loadTrack() {
  const bag = state.bag;
  const frame = state.frame;
  try {
    const res = await fetch(`/track?bag=${encodeURIComponent(bag)}&frame=${frame}`,
      { cache: 'no-store' });
    if (!res.ok) {
      let detail = '';
      try { detail = (await res.json()).error || ''; } catch (e) { /* не JSON */ }
      throw new Error(`GET /track -> ${res.status}${detail ? ': ' + detail : ''}`);
    }
    const data = await res.json();
    if (bag !== state.bag || frame !== state.frame) return;   // устаревший ответ
    state.data = data;
    clearError();
  } catch (e) {
    if (bag !== state.bag || frame !== state.frame) return;
    state.data = null;
    fail(e);
  }
  // Меши собираются здесь, вне цикла анимации.
  rebuildTrackObjects();
  updateLegend();
  updateSourceBadges();
  updateHud();

  if (state.visible.points && !state.clean) {
    try {
      await loadPoints();
    } catch (e) {
      fail(e);   // облако -- опция: без него объекты пути всё равно видны
    }
    updateHud();
  }
  if (state.clean) {
    const box = boundsOfObjects();
    if (box) fitCameraToBox(box);
  }
}

// Кадры грузятся строго по одному (память в притирку): протяжка слайдера
// оставляет только последний запрошенный кадр.
let pendingFrame = null;

async function requestFrame(frame) {
  pendingFrame = frame;
  if (state.loading) return;
  state.loading = true;
  try {
    while (pendingFrame !== null) {
      const next = pendingFrame;
      pendingFrame = null;
      state.frame = next;
      updateFrameLabel();
      await loadTrack();
    }
  } catch (e) {
    fail(e);          // обработчики UI не ждут этот промис -- ошибку показываем сами
  } finally {
    state.loading = false;
  }
}

// --------------------------------------------------------------- инициализация

async function loadMeta() {
  const res = await fetch('/meta', { cache: 'no-store' });
  if (!res.ok) throw new Error(`GET /meta -> ${res.status}`);
  state.meta = await res.json();
}

function wireUi() {
  const bagSel = $('bag');
  for (const bag of state.meta.bags) {
    const opt = document.createElement('option');
    opt.value = bag.name;
    opt.textContent = `${bag.name} (${bag.frames} кадр.)`;
    bagSel.appendChild(opt);
  }
  bagSel.value = state.bag;
  bagSel.onchange = async () => {
    state.bag = bagSel.value;
    const info = state.meta.bags.find((b) => b.name === state.bag);
    state.frames = info ? info.frames : 1;
    const fr = $('frame');
    fr.max = String(Math.max(0, state.frames - 1));
    fr.value = '0';
    state.data = null;
    dropAllObjects();
    rebuildTrackObjects();
    updateLegend();
    updateHud();
    await requestFrame(0);
    const box = boundsOfObjects();
    if (state.clean && box) fitCameraToBox(box);
  };

  const fr = $('frame');
  fr.max = String(Math.max(0, state.frames - 1));
  fr.oninput = () => { requestFrame(Number(fr.value)); };

  $('t-points').onchange = async (e) => {
    state.visible.points = e.target.checked;
    applyVisibility();
    if (state.visible.points && !pointsObj && !state.clean) {
      try { await loadPoints(); updateHud(); } catch (err) { fail(err); }
    }
  };
  $('t-rails').onchange = (e) => { state.visible.rails = e.target.checked; applyVisibility(); };
  $('t-sleepers').onchange = (e) => { state.visible.sleepers = e.target.checked; applyVisibility(); };
  $('t-floor').onchange = (e) => { state.visible.floor = e.target.checked; applyVisibility(); };

  $('clean').onclick = () => setCleanField(!state.clean);
  window.addEventListener('resize', onResize);
}

function onResize() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  if (pointsObj) {
    pointsObj.material.uniforms.uPixelRatio.value = Math.min(window.devicePixelRatio || 1, 2);
  }
}

async function main() {
  setupWorldUp();
  await loadMeta();

  state.bag = state.meta.default_bag || (state.meta.bags[0] || {}).name || null;
  const info = state.meta.bags.find((b) => b.name === state.bag);
  state.frames = info ? Math.max(1, info.frames) : 1;
  if (!state.bag) throw new Error('нет доступных bag-ов (--root не содержит .db3)');

  const canvas = $('view');
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(window.innerWidth, window.innerHeight, false);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(DARK_BG);

  camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.05, 4000);
  camera.up.set(0, 0, 1);              // явно, а не "по умолчанию": данные Z-вверх
  camera.position.set(4, 10, 6);

  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.screenSpacePanning = false;
  controls.maxDistance = 2000;
  controls.target.set(0, -20, 0);
  controls.update();

  // Свет: без него MeshStandardMaterial чёрный.
  scene.add(new THREE.HemisphereLight(0xffffff, 0x2a3040, 1.0));
  const sun = new THREE.DirectionalLight(0xffffff, 1.7);
  sun.position.set(80, 60, 120);
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0x88aaff, 0.5);
  fill.position.set(-60, -40, 60);
  scene.add(fill);

  document.title = `Track3D — ${state.bag}`;
  wireUi();
  updateLegend();
  updateSourceBadges();
  updateFrameLabel();
  await requestFrame(0);

  // Стартовый вид -- по габариту объектов (ближние метры коридора), не по облаку.
  const box = boundsOfObjects();
  if (box) fitCameraToBox(corridorViewBox(box), 1.15);

  // Геометрия статична, поэтому в цикле -- только контролы и рендер.
  function loop() {
    requestAnimationFrame(loop);
    controls.update();
    renderer.render(scene, camera);
  }
  requestAnimationFrame(loop);
}

main().catch(fail);