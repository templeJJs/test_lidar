// Слой разметки для основного вьюера web_viewer.py (страница web/index.html).
//
// По образцу web/track3d.js: модуль самодостаточный, подключается МЯГКИМ импортом
// (нет файла или маршрута -- страница живёт без разметки) и владеет ТОЛЬКО своими
// группами: каркасы боксов и точки предпросмотра. В существующие слои (облако,
// объекты пути, коридор) он не лезет -- ему это и не нужно: облако он читает через
// переданный снаружи аксессор `cloud`.
//
// Геометрия -- ровно та же, что в React-клиенте (app/client/src/viewer/scene.ts),
// иначе каркас разошёлся бы с мешем трассировки на сервере:
//   * оси данных: X поперёк, Y вдоль (вперёд это -Y), Z вверх;
//   * size = (ширина поперёк, ВЫСОТА, длина вдоль) -> куб по осям (X, Z, Y);
//   * поворот каркаса R = Rx(roll)·Ry(pitch)·Rz(yaw) -- матрица `labeling._rotation`;
//   * цвета статуса автоматики: детектор -- зелёный, кластер -- жёлтый, вручную --
//     красный, точки трассировки -- голубые.
//
// Клик: луч из камеры через мышь, из точек кадра берётся БЛИЖАЙШАЯ К ЛУЧУ, но
// только та, что попала в экранное окно курсора. Одной близости к лучу мало:
// на 56 м любая точка попадает «почти на луч» (0.1 м промаха -- это 0.1° при
// поле зрения 60°), поэтому окно в пикселях -- то же, что «под курсором».
//
// Перетаскивание: отдельного режима правки НЕТ (раньше был тумблер «правка мышью»,
// и без него ни ручки, ни перенос не работали -- на это и была жалоба). Правило
// простое: у ВЫБРАННОГО объекта всегда видны ручки, а ЛКМ внутри его проекции
// (или по ручке) начинает жест -- и только тогда вращение камеры глушится, на один
// этот жест (`stopPropagation` в capture-фазе: OrbitControls подписан на канву и
// слушает события ПОСЛЕ нас). ЛКМ вне бокса -- камера, как обычно; тумблеров и
// «заморозки» камеры на весь режим больше нет.
// Центр бокса тянется по горизонтальной плоскости Z = центр бокса, а ручки --
// каждая по своей: боковые по той же плоскости (размер выходит СИММЕТРИЧНЫМ),
// высота -- по вертикальной плоскости через центр, поворот -- по углу в плане
// (см. handleEdit).
//
// Видимость боксов и точек -- depthTest: false: каркас разметки обязан читаться
// поверх облака (340 тыс. точек гасят линии, лежащие на поверхности объекта),
// это инструмент, а не картинка замера.
//
// Меш объекта: у человека рисуется НАСТОЯЩАЯ запечённая поза (`assets/_poses`,
// см. `docs/poses-contract.md`), а не примитив. Расчёт позы (выбор момента, склейка
// моментов, постановка в кадр) живёт ЗДЕСЬ, в чистых функциях `poseMoment` /
// `poseVerticesAt` / `placePose`, и он же повторён в `labeling.py` -- иначе рисунок
// и счётчик закрытых точек разошлись бы. Траектория: линия начала и конца, концы
// тянутся мышью, подпись -- секунды и скорость (спрайт с канвы).

import * as THREE from 'three';

const PICK_RADIUS_PX = 14;          // экранное окно курсора, как в React-клиенте
const CLICK_MOVE_PX = 5;            // больше -- это уже перетаскивание, а не клик
const PREVIEW_COLOR = 0x38bdf8;     // точки трассировки: голубые
// Размер точки -- в ПИКСЕЛЯХ, без затухания по дальности: трассировка объекта на
// 56 м при размере в метрах (как в React-клиенте) даёт доли пикселя, и «точек 104»
// в панели не видно в кадре. Голубое обязано читаться на любой дальности.
const PREVIEW_SIZE_PX = 6;
const MARKER_R_M = 0.14;
// Ручки правки: 6 точек на гранях бокса (размер) + 1 на поворот. Размер -- в
// пикселях, как у точек трассировки: тянуть надо за то, что видно, а на 56 м
// метровая ручка -- доли пикселя.
const HANDLE_R_PX = 11;
const HANDLE_SIZE_PX = 10;
// Курсор у грани бокса (в пикселях): у дальней цели бокс занимает единицы
// пикселей, и попасть точно в его оболочку мышью нельзя -- запас в пикселях и
// окно вокруг центра делают его «хватаемым» и на 56 м.
const BOX_PAD_PX = 6;
const BOX_MIN_PX = 12;
const SIZE_HANDLE_COLOR = 0xf472b6;
const YAW_HANDLE_COLOR = 0xfb923c;
const YAW_HANDLE_UP_M = 0.45;
// Траектория: линия от начала к концу, концы -- крупные точки (зелёная -- начало,
// красная -- конец), у конца стрелка (куда идёт), над серединой -- подпись
// «секунды · скорость».
const TRAJ_COLOR = 0x22d3ee;
const TRAJ_START_COLOR = 0x4ade80;
const TRAJ_END_COLOR = 0xf87171;
const TRAJ_END_R_PX = 13;
const TRAJ_LABEL_UP_M = 0.55;
const TRAJ_ARROW_LEN_M = 0.45;
const TRAJ_ARROW_R_M = 0.12;
const POSE_MESH_OPACITY = 0.55;
const POSE_MESH_SELECTED_OPACITY = 0.75;
const ORDER = { box: 6, pose: 7, traj: 7, preview: 8, marker: 9, handles: 10 };

// --------------------------------------------------- поза: чистые функции ---
//
// Ровно те же формулы, что в `labeling.py` (`pose_moment`, `pose_vertices`,
// `place_pose`): контракт данных -- `assets/_poses/manifest.json`,
// `playback`/`placement`. Проверяются в Node вместе с сервером (сверка вершин).

/**
 * Момент клипа по времени `t` (с): {index, blend, timeInClip}.
 *
 * `loop` -- по контракту: индекс = floor((t mod period)/period · T); последний
 * интервал отдаётся последнему моменту, склейка идёт в первый (`poseVerticesAt`).
 * `one_shot` -- один проход по размаху времен: доля t/span линейно ложится на
 * моменты, в конце -- последний. `static` -- всегда первый момент.
 */
export function poseMoment(times, periodS, play, t) {
  const total = times ? times.length : 0;
  const period = Number(periodS) || 0;
  const time = Number(t) || 0;
  if (play === 'static' || total <= 1 || period <= 0) {
    return { index: 0, blend: 0, timeInClip: 0 };
  }
  if (play === 'loop') {
    let span = time % period;
    if (span < 0) span += period;
    const frac = span / period;
    const index = Math.min(Math.floor(frac * total), total - 1);
    return { index, blend: frac * total - index, timeInClip: span };
  }
  const frac = Math.min(Math.max(time / period, 0), 1);
  const x = frac * (total - 1);
  const index = Math.min(Math.floor(x), total - 2);
  return { index, blend: x - index, timeInClip: Math.min(Math.max(time, 0), period) };
}

/**
 * Вершины позы на времени `t`: линейная склейка соседних моментов.
 *
 * @param clip -- {vertices (Float32Array T·N·3, оси КАДРА), times, periodS, play}
 * @returns Float32Array (N·3) -- НЕ копия исходного массива при blend = 0.
 */
export function poseVerticesAt(clip, t) {
  const total = clip.times.length;
  const n3 = clip.vertices.length / total | 0;
  const got = poseMoment(clip.times, clip.periodS, clip.play, t);
  const i = Math.min(got.index, total - 1);
  if (total <= 1 || got.blend <= 0) {
    return clip.vertices.subarray(i * n3, (i + 1) * n3);
  }
  const j = clip.play === 'loop' ? (i + 1) % total : Math.min(i + 1, total - 1);
  const a = clip.vertices.subarray(i * n3, (i + 1) * n3);
  const b = clip.vertices.subarray(j * n3, (j + 1) * n3);
  const w = got.blend;
  const out = new Float32Array(n3);
  for (let k = 0; k < n3; k++) out[k] = a[k] * (1 - w) + b[k] * w;
  return out;
}

/** R = Rx(roll)·Ry(pitch)·Rz(yaw) -- та же матрица, что `labeling._rotation`. */
export function poseRotation(yaw, pitch, roll) {
  const d = Math.PI / 180;
  const cx = Math.cos((roll || 0) * d); const sx = Math.sin((roll || 0) * d);
  const cy = Math.cos((pitch || 0) * d); const sy = Math.sin((pitch || 0) * d);
  const cz = Math.cos((yaw || 0) * d); const sz = Math.sin((yaw || 0) * d);
  // R = Rx · Ry · Rz (строка за строкой, как в labeling._rotation).
  return [
    cy * cz, -cy * sz, sy,
    cx * sz + sx * sy * cz, cx * cz - sx * sy * sz, -sx * cy,
    sx * sz - cx * sy * cz, sx * cz + cx * sy * sz, cx * cy,
  ];
}

/**
 * Поставить меш позы в кадр: низ -- на опору, середина габарита -- в центр.
 *
 * Три шага, ровно как `labeling.place_pose`: якорь = [середина габарита по X и Y,
 * низ по Z]; поворот вокруг ЯКОРЯ; сдвиг так, что низ меша встаёт РОВНО на `baseZ`
 * (`lift` -- на сколько подняли; у лежащего это +0.065 м вверх, у наклонённого
 * тела -- миллиметры вниз). Возвращает {positions, lift, anchor}.
 */
export function placePose(vertices, opts) {
  const { center, baseZ, yaw = 0, pitch = 0, roll = 0 } = opts || {};
  let xlo = Infinity; let xhi = -Infinity; let ylo = Infinity; let yhi = -Infinity;
  let zlo = Infinity;
  for (let k = 0; k < vertices.length; k += 3) {
    const x = vertices[k]; const y = vertices[k + 1]; const z = vertices[k + 2];
    if (x < xlo) xlo = x;
    if (x > xhi) xhi = x;
    if (y < ylo) ylo = y;
    if (y > yhi) yhi = y;
    if (z < zlo) zlo = z;
  }
  if (!Number.isFinite(xlo)) return { positions: new Float32Array(0), lift: 0, anchor: [0, 0, 0] };
  const anchor = [0.5 * (xlo + xhi), 0.5 * (ylo + yhi), zlo];
  const r = poseRotation(yaw, pitch, roll);
  const out = new Float32Array(vertices.length);
  let rotatedZMin = Infinity;
  for (let k = 0; k < vertices.length; k += 3) {
    const x = vertices[k] - anchor[0];
    const y = vertices[k + 1] - anchor[1];
    const z = vertices[k + 2] - anchor[2];
    const rz = r[6] * x + r[7] * y + r[8] * z;
    if (rz < rotatedZMin) rotatedZMin = rz;
    out[k] = r[0] * x + r[1] * y + r[2] * z;
    out[k + 1] = r[3] * x + r[4] * y + r[5] * z;
    out[k + 2] = rz;
  }
  const lift = Number(baseZ) - (Number(baseZ) + rotatedZMin);
  const cx = Number(center ? center[0] : 0);
  const cy = Number(center ? center[1] : 0);
  const cz = Number(baseZ) + lift;
  for (let k = 0; k < out.length; k += 3) {
    out[k] += cx;
    out[k + 1] += cy;
    out[k + 2] += cz;
  }
  return { positions: out, lift, anchor };
}

/**
 * Какая поза играет в сцене -- ПРАВИЛО, а не ручная настройка на каждый кадр:
 * у объекта с траекторией движение выглядит идущим, у стоящего -- стоящим.
 * `unknown` -- это «не выбрана», читается как стоящая (по умолчанию); явные
 * lying/fallen/running не переписываются. Та же функция в `labeling.pose_rule`.
 */
export function poseInScene(pose, hasTrajectory) {
  const key = String(pose || 'unknown').toLowerCase();
  if (key === 'running') return 'running';
  if (key === 'standing' || key === 'walking' || key === 'unknown') {
    return hasTrajectory ? 'walking' : 'standing';
  }
  return key;
}

// ------------------------------------------------------- траектория: числа ---

/** Скорость траектории, м/с: расстояние между концами, делить на секунды. */
export function trajectorySpeed(traj) {
  const d = trajectoryDistance(traj);
  const s = Number(traj && traj.seconds) || 0;
  return s > 0 ? d / s : 0;
}

/** Длина траектории, м. */
export function trajectoryDistance(traj) {
  if (!traj || !traj.start || !traj.end) return 0;
  const dx = Number(traj.end[0]) - Number(traj.start[0]);
  const dy = Number(traj.end[1]) - Number(traj.start[1]);
  const dz = Number(traj.end[2]) - Number(traj.start[2]);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Точка основания на траектории во время `t`, с (линейная склейка концов).
 * До начала -- начало, после конца -- КОНЕЦ. Та же формула в `labeling.trajectory_at`.
 */
export function trajectoryAt(traj, t) {
  const a = traj.start;
  const b = traj.end;
  const s = Number(traj.seconds) || 0;
  const frac = s <= 0 ? 0 : Math.min(Math.max(Number(t) / s, 0), 1);
  return [a[0] + (b[0] - a[0]) * frac,
          a[1] + (b[1] - a[1]) * frac,
          a[2] + (b[2] - a[2]) * frac];
}

/**
 * Основание объекта на кадре: траектория (если есть) плюс ход сенсора.
 * Траектория записана в системе ОПОРНОГО кадра, а сенсор за кадр сдвигается
 * по Y на `speedMPerFrame` -- ровно как в `labeling._instance_center`.
 */
export function trajectoryBaseAtFrame(traj, refFrame, frame, hz, speedMPerFrame, shiftM) {
  const t = (Number(frame) - Number(refFrame)) / (Number(hz) || 10);
  const base = trajectoryAt(traj, t);
  // Сдвиг сенсора за это время. По умолчанию -- `скорость * разность кадров`, но
  // ход записи меняется по кадрам, и вызывающий (web/app.js) передаёт ИНТЕГРАЛ
  // профиля хода (`labelMotionBetween`) готовым числом: та же формула, что в
  // `labeling._instance_center`.
  const dy = Number.isFinite(Number(shiftM))
    ? Number(shiftM)
    : (Number(speedMPerFrame) || 0) * (Number(frame) - Number(refFrame));
  return [base[0], base[1] + dy, base[2]];
}

/** Подпись траектории: секунды и скорость (её же видно в 3D над линией). */
export function trajectoryLabel(traj) {
  const s = Number(traj && traj.seconds) || 0;
  const v = trajectorySpeed(traj);
  return `${s.toFixed(1)} с · ${v.toFixed(2)} м/с (${(v * 3.6).toFixed(1)} км/ч)`;
}

/** R = Rx(roll)·Ry(pitch)·Rz(yaw) -- матрица поворота из `labeling._rotation`. */
function labelQuaternion(yaw, pitch, roll) {
  const d = Math.PI / 180;
  const q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), (roll || 0) * d);
  q.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), (pitch || 0) * d));
  q.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), (yaw || 0) * d));
  return q;
}

/**
 * Слой разметки.
 *
 * @param scene/camera/controls -- сцена вьюера (controls на время жеста правки
 *   только читается: вращение глушится не выключением controls, а тем, что жест
 *   не доходит до его обработчика).
 * @param canvas -- канва, по которой кликают (события ловим в capture на window:
 *   тогда наш обработчик идёт раньше OrbitControls).
 * @param cloud -- () => ({positions, count, maxDist}): облако текущего кадра.
 * @param onPick(point) -- клик по облаку в режиме разметки (жест не начался).
 * @param onDragMove(x, y, z) -- перенос выбранного бокса по полу.
 * @param onDragStart(kind) / onDragEnd() -- начало и конец жеста мышью: панель
 *   делает один шаг отмены на жест, а не на каждое движение пикселя.
 * @param onEdit(edit) -- правка ручкой: {kind:'size', axis:'x'|'y', size_m} --
 *   размер поперёк/вдоль, {kind:'height', edge:'top'|'bottom', z_m} -- высота
 *   (низ или верх бокса остаётся на месте), {kind:'yaw', yaw_deg} -- поворот,
 *   {kind:'traj', which:'start'|'end', point:[x,y,z]} -- конец траектории.
 * @param onHidden(indices) -- объединённая по всем объектам кадра маска закрытых
 *   точек: облаком владеет вьюер, а «что закрыто» складывает слой (см. setPreviews).
 * @param onStatus(text) -- короткая строка состояния в панель.
 *
 * Объект в `setObjects` может нести, кроме бокса:
 *   * `pose` -- {positions (Float32Array, УЖЕ поставленные в кадр), faces, clip,
 *     moment, blend, source, label}: у человека это настоящий меш запечённой позы;
 *   * `traj` -- {start, end, seconds}: траектория движения (линия, концы, подпись).
 */
export function createLabelLayer({
  scene, camera, controls = null, canvas,
  cloud = null, onPick = null, onSelect = null, onDragMove = null,
  onDragStart = null, onDragEnd = null,
  onEdit = null, onStatus = null, onHidden = null,
} = {}) {
  if (!scene || !camera || !canvas) {
    throw new Error('createLabelLayer: нужны scene, camera и canvas');
  }

  const state = {
    on: false, boxes: [], previewCount: 0, handles: [],
    poseCount: 0, poses: [], trajCount: 0, trajectories: [],
    hidden: [],
  };
  const boxGroup = new THREE.Group();
  boxGroup.name = 'label-boxes';
  const poseGroup = new THREE.Group();
  poseGroup.name = 'label-poses';
  const trajGroup = new THREE.Group();
  trajGroup.name = 'label-trajectories';
  const previewGroup = new THREE.Group();
  previewGroup.name = 'label-preview';
  const handleGroup = new THREE.Group();
  handleGroup.name = 'label-handles';
  boxGroup.visible = false;
  poseGroup.visible = false;
  trajGroup.visible = false;
  previewGroup.visible = false;
  handleGroup.visible = false;
  scene.add(boxGroup);
  scene.add(poseGroup);
  scene.add(trajGroup);
  scene.add(previewGroup);
  scene.add(handleGroup);

  // Каркас -- единичный куб, который масштабируется под `size`: геометрия одна на
  // все боксы, свои у каждого только материал (цвет статуса) и матрица.
  const unitBox = new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1));

  let previewGeometry = null;
  let previewMaterial = null;
  let previewPoints = null;
  let dragZ = 0;
  // Меши позы: геометрия своя у каждого объекта (вершины разные), но пересоздаём
  // её только при смене ЧИСЛА вершин: у ходьбы 18 моментов одной сетки, и кадр за
  // кадром меняются только координаты (`position.needsUpdate`).
  const poseMeshes = new Map();       // key -> {mesh, count, faces}

  // ------------------------------------------------------------- отрисовка ---

  function dropPreview() {
    if (previewPoints) {
      previewGroup.remove(previewPoints);
      previewPoints = null;
    }
    if (previewGeometry) {
      previewGeometry.dispose();
      previewGeometry = null;
    }
    if (previewMaterial) {
      previewMaterial.dispose();
      previewMaterial = null;
    }
    state.previewCount = 0;
  }

  // Ресурсы отрисовки (геометрии, материалы, спрайты) КЭШИРУЮТСЯ и переживают
  // кадр: labelDraw зовётся на каждой смене кадра, а setObjects раньше снимал
  // детей с dispose и пересоздавал всё -- канва 512x96 + CanvasTexture,
  // ConeGeometry и пачка материалов на каждый объект с траекторией, материалы
  // на каждый бокс. Теперь дети снимаются без dispose (clearGroup), а сами кэши
  // живут до dispose слоя; не понадобившееся в текущем кадре вытесняется
  // (evictTrajResources), чтобы кэш не рос на движущихся траекториях.
  function clearGroup(group) {
    group.clear();
  }

  const markerGeometry = new THREE.SphereGeometry(MARKER_R_M, 12, 8);
  const arrowGeometry = new THREE.ConeGeometry(TRAJ_ARROW_R_M, TRAJ_ARROW_LEN_M, 10);
  const materialCache = new Map();     // ключ -> материал (боксы ~4 цветов, метки, линия)
  const trajGeometries = new Map();    // `${start}|${end}` -> {line, a, b, used}
  const trajSprites = new Map();       // `${text}|${color}` -> Sprite (+used)

  function cachedMaterial(key, make) {
    let m = materialCache.get(key);
    if (!m) {
      m = make();
      materialCache.set(key, m);
    }
    return m;
  }

  /** Материал каркаса бокса: цвет статуса автоматики (~4) и выделенность. */
  function boxMaterial(color, selected) {
    return cachedMaterial(`box:${color}:${selected ? 1 : 0}`, () => new THREE.LineBasicMaterial({
      color, transparent: true, opacity: selected ? 0.9 : 0.7, depthTest: false,
    }));
  }

  /** Вытеснение кэшей траектории: не нарисованное в ЭТОМ кадре -- диспоузить. */
  function evictTrajResources() {
    for (const [key, g] of Array.from(trajGeometries)) {
      if (g.used) { g.used = false; continue; }
      g.line.dispose();
      g.a.dispose();
      g.b.dispose();
      trajGeometries.delete(key);
    }
    for (const [key, s] of Array.from(trajSprites)) {
      if (s.used) { s.used = false; continue; }
      if (s.material.map) s.material.map.dispose();
      s.material.dispose();
      trajSprites.delete(key);
    }
  }

  function dropTrajCaches() {
    for (const g of trajGeometries.values()) {
      g.line.dispose();
      g.a.dispose();
      g.b.dispose();
    }
    trajGeometries.clear();
    for (const s of trajSprites.values()) {
      if (s.material.map) s.material.map.dispose();
      s.material.dispose();
    }
    trajSprites.clear();
    for (const m of materialCache.values()) m.dispose();
    materialCache.clear();
  }

  /**
   * Меш позы: готовые вершины в системе кадра (их посчитал `placePose`), лица --
   * из клипа. Геометрия переиспользуется, пока совпадает число вершин.
   */
  function poseMeshFor(item, color) {
    const pose = item.pose;
    const count = pose.positions.length / 3 | 0;
    let slot = poseMeshes.get(item.key);
    if (!slot || slot.count !== count) {
      if (slot) {
        poseGroup.remove(slot.mesh);
        slot.mesh.geometry.dispose();
        slot.mesh.material.dispose();
      }
      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(count * 3), 3));
      geom.setIndex(new THREE.BufferAttribute(
        pose.faces instanceof Uint32Array ? pose.faces : new Uint32Array(pose.faces), 1));
      const material = new THREE.MeshBasicMaterial({
        color, transparent: true, opacity: POSE_MESH_OPACITY,
        depthTest: false, depthWrite: false, side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(geom, material);
      mesh.renderOrder = ORDER.pose;
      mesh.frustumCulled = false;
      slot = { mesh, count, faces: pose.faces };
      poseMeshes.set(item.key, slot);
      poseGroup.add(mesh);
    }
    slot.mesh.geometry.attributes.position.array.set(pose.positions);
    slot.mesh.geometry.attributes.position.needsUpdate = true;
    slot.mesh.material.color.setHex(color);
    slot.mesh.material.opacity = item.selected ? POSE_MESH_SELECTED_OPACITY : POSE_MESH_OPACITY;
    slot.mesh.visible = true;
    return slot.mesh;
  }

  function dropPoseMeshes(keepKeys) {
    for (const [key, slot] of [...poseMeshes]) {
      if (keepKeys.has(key)) continue;
      poseGroup.remove(slot.mesh);
      slot.mesh.geometry.dispose();
      slot.mesh.material.dispose();
      poseMeshes.delete(key);
    }
  }

  /** Подпись траектории -- спрайт с канвы: секунды и скорость. Канвы нет --
   * подпись просто не рисуется (панель всё равно показывает её числом).
   * Спрайт КЭШИРУЕТСЯ по (текст, цвет): содержимое от кадра не зависит, а канва
   * 512x96 с текстурой пересоздавалась на каждую перерисовку. */
  function labelSprite(text, color) {
    try {
      if (typeof document === 'undefined' || !document.createElement) return null;
      const canvas2 = document.createElement('canvas');
      canvas2.width = 512;
      canvas2.height = 96;
      const ctx = canvas2.getContext ? canvas2.getContext('2d') : null;
      if (!ctx) return null;
      ctx.clearRect(0, 0, canvas2.width, canvas2.height);
      ctx.fillStyle = 'rgba(9, 14, 24, 0.72)';
      ctx.fillRect(0, 0, canvas2.width, canvas2.height);
      ctx.font = '600 44px system-ui, sans-serif';
      ctx.fillStyle = '#' + new THREE.Color(color).getHexString();
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(text, canvas2.width / 2, canvas2.height / 2 + 2);
      const texture = new THREE.CanvasTexture(canvas2);
      const material = new THREE.SpriteMaterial({
        map: texture, transparent: true, depthTest: false, depthWrite: false,
      });
      const sprite = new THREE.Sprite(material);
      sprite.scale.set(2.0, 0.375, 1);
      sprite.renderOrder = ORDER.traj;
      sprite.used = false;
      return sprite;
    } catch (e) {
      return null;
    }
  }

  function labelSpriteCached(text, color) {
    const key = `${text}|${color}`;
    let s = trajSprites.get(key);
    if (!s) {
      s = labelSprite(text, color);
      if (!s) return null;
      trajSprites.set(key, s);
    }
    s.used = true;
    return s;
  }

  /**
   * Траектория объекта: линия начала и конца, обе точки, стрелка «куда» и подпись
   * с секундами и скоростью. Концы линии -- ОСНОВАНИЕ объекта (та же точка, куда
   * он встаёт), поэтому линия проходит по полу, а не по центрам.
   *
   * Геометрии берутся из кэша по (начало, конец): линия и обе точки конца -- это
   * одни и те же два вершины, концы -- по одному вершины; ConeGeometry стрелки --
   * одна на слой. Двигается только позиция спрайта.
   */
  function drawTrajectory(item, color) {
    const traj = item.traj;
    const a = new THREE.Vector3(traj.start[0], traj.start[1], traj.start[2]);
    const b = new THREE.Vector3(traj.end[0], traj.end[1], traj.end[2]);
    const gkey = [a.x, a.y, a.z, b.x, b.y, b.z].map((v) => String(Number(v.toFixed(4)))).join(',');
    let g = trajGeometries.get(gkey);
    if (!g) {
      const line = new THREE.BufferGeometry();
      line.setAttribute('position', new THREE.Float32BufferAttribute(
        [a.x, a.y, a.z, b.x, b.y, b.z], 3));
      const pa = new THREE.BufferGeometry();
      pa.setAttribute('position', new THREE.Float32BufferAttribute([a.x, a.y, a.z], 3));
      const pb = new THREE.BufferGeometry();
      pb.setAttribute('position', new THREE.Float32BufferAttribute([b.x, b.y, b.z], 3));
      g = { line, a: pa, b: pb, used: false };
      trajGeometries.set(gkey, g);
    }
    g.used = true;
    const lineMesh = new THREE.Line(g.line, cachedMaterial('traj:line', () => new THREE.LineBasicMaterial({
      color: TRAJ_COLOR, transparent: true, opacity: 0.95, depthTest: false,
    })));
    lineMesh.renderOrder = ORDER.traj;
    lineMesh.frustumCulled = false;
    trajGroup.add(lineMesh);

    const points = new THREE.Points(g.line, cachedMaterial(
      `traj:pts:${item.selected ? TRAJ_END_COLOR : TRAJ_COLOR}`,
      () => new THREE.PointsMaterial({
        color: item.selected ? TRAJ_END_COLOR : TRAJ_COLOR, size: TRAJ_END_R_PX,
        sizeAttenuation: false, depthTest: false, depthWrite: false,
      })));
    points.renderOrder = ORDER.traj;
    points.frustumCulled = false;
    trajGroup.add(points);

    // Отдельные метки начала и конца: начало -- зелёное, конец -- красное (у
    // выбранного объекта они же -- ручки: тянуть можно за любую).
    for (const [geom, col] of [[g.a, TRAJ_START_COLOR], [g.b, TRAJ_END_COLOR]]) {
      const mark = new THREE.Points(geom, cachedMaterial(`traj:mark:${col}`, () => new THREE.PointsMaterial({
        color: col, size: TRAJ_END_R_PX + 2, sizeAttenuation: false,
        depthTest: false, depthWrite: false,
      })));
      mark.renderOrder = ORDER.traj;
      mark.frustumCulled = false;
      trajGroup.add(mark);
    }

    const dir = new THREE.Vector3().subVectors(b, a);
    const len = dir.length();
    if (len > 1e-6) {
      const arrow = new THREE.Mesh(arrowGeometry, cachedMaterial('traj:arrow', () => new THREE.MeshBasicMaterial({
        color: TRAJ_COLOR, depthTest: false, transparent: true, opacity: 0.95,
      })));
      arrow.position.copy(b);
      arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
      arrow.renderOrder = ORDER.traj;
      arrow.frustumCulled = false;
      trajGroup.add(arrow);
    }
    const text = trajectoryLabel(traj);
    const sprite = labelSpriteCached(text, color);
    if (sprite) {
      const mid = new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5);
      sprite.position.set(mid.x, mid.y, mid.z + TRAJ_LABEL_UP_M);
      trajGroup.add(sprite);
    }
    return { key: item.key, start: [a.x, a.y, a.z], end: [b.x, b.y, b.z],
             seconds: Number(traj.seconds) || 0, distance: len,
             speedMps: trajectorySpeed(traj), label: text, selected: Boolean(item.selected) };
  }

  function setObjects(boxes) {
    state.boxes = Array.isArray(boxes) ? boxes : [];
    // Геометрии/материалы/спрайты -- из кэшей слоя, поэтому дети снимаются БЕЗ
    // dispose (см. clearGroup): пересоздание их на каждый кадр давало мусор.
    clearGroup(boxGroup);
    clearGroup(trajGroup);
    state.trajectories = [];
    for (const box of state.boxes) {
      const color = Number.isFinite(box.color) ? box.color : 0xf87171;
      const c = box.center || [0, 0, 0];
      const s = box.size || [1, 1, 1];
      // Меш позы -- вместо каркаса: у человека настоящая геометрия, каркас нужен
      // только выбранному (по нему видно объявленный габарит и ручки правки).
      const hasPose = Boolean(box.pose && box.pose.positions && box.pose.positions.length);
      if (hasPose) {
        poseMeshFor(box, color);
      }
      if (!hasPose || box.selected) {
        const mesh = new THREE.LineSegments(unitBox, boxMaterial(color, box.selected));
        mesh.position.set(c[0], c[1], c[2]);
        // size = (поперёк, высота, вдоль) -> куб по осям (X, Z, Y).
        mesh.scale.set(
          Math.max(s[0], 0.05),
          Math.max(s[2], 0.05),
          Math.max(s[1], 0.05),
        );
        mesh.quaternion.copy(labelQuaternion(box.yaw, box.pitch, box.roll));
        mesh.renderOrder = ORDER.box;
        mesh.frustumCulled = false;
        boxGroup.add(mesh);
      }
      if (box.selected) {
        // Отметка центра: у выбранного объекта сходятся все правки, и его видно
        // даже когда каркас ушёл за край кадра.
        const marker = new THREE.Mesh(
          markerGeometry,
          cachedMaterial(`marker:${color}`, () => new THREE.MeshBasicMaterial({
            color, depthTest: false,
          })));
        marker.position.set(c[0], c[1], c[2]);
        marker.renderOrder = ORDER.marker;
        marker.frustumCulled = false;
        boxGroup.add(marker);
      }
      if (box.traj && box.traj.start && box.traj.end) {
        state.trajectories.push(drawTrajectory(box, color));
      }
    }
    const keep = new Set(state.boxes.filter((b) => b.pose && b.pose.positions
      && b.pose.positions.length).map((b) => b.key));
    dropPoseMeshes(keep);
    state.poseCount = keep.size;
    state.poses = state.boxes.filter((b) => keep.has(b.key)).map((b) => ({
      key: b.key, vertices: b.pose.positions.length / 3 | 0,
      clip: b.pose.clip || null, moment: b.pose.moment, blend: b.pose.blend,
      source: b.pose.source || null, label: b.pose.label || null,
      faces: (b.pose.faces && b.pose.faces.length) ? b.pose.faces.length / 3 : 0,
      bbox: poseBbox(b.pose.positions),
    }));
    state.trajCount = state.trajectories.length;
    evictTrajResources();
    boxGroup.visible = state.on && boxGroup.children.length > 0;
    poseGroup.visible = state.on && poseGroup.children.length > 0;
    trajGroup.visible = state.on && trajGroup.children.length > 0;
    buildHandles();
  }

  function poseBbox(positions) {
    const out = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
    for (let k = 0; k < positions.length; k += 3) {
      for (let a = 0; a < 3; a++) {
        const v = positions[k + a];
        if (v < out.min[a]) out.min[a] = v;
        if (v > out.max[a]) out.max[a] = v;
      }
    }
    return out;
  }

  /**
  * Ручки правки выбранного бокса: шесть точек на его гранях (тянуть -- размер),
  * одна над верхом (поворот вокруг вертикали) и -- если у объекта есть траектория
  * -- её КОНЦЫ (тянуть начало и конец). Показываются ВСЕГДА, когда есть выбранный
  * объект: отдельного тумблера правки нет, и ручки -- единственное, за что видно,
  * что бокс можно тянуть.
  */
  function buildHandles() {
    // Геометрии ручек -- свои на каждый вызов (позиции зависят от бокса), а
    // материалы -- из кэша слоя, поэтому диспоузим только геометрию.
    for (const child of handleGroup.children) {
      if (child.geometry) child.geometry.dispose();
    }
    handleGroup.clear();
    state.handles = [];
    const box = state.boxes.find((b) => b.selected);
    if (!state.on || !box) {
      handleGroup.visible = false;
      return;
    }
    const c = box.center || [0, 0, 0];
    const s = box.size || [1, 1, 1];
    const q = labelQuaternion(box.yaw, box.pitch, box.roll);
    const center = new THREE.Vector3(c[0], c[1], c[2]);
    // size = (поперёк, высота, вдоль) -> локальные оси (X, Z, Y).
    const half = new THREE.Vector3(Math.max(s[0], 0.05) / 2, Math.max(s[2], 0.05) / 2,
                                   Math.max(s[1], 0.05) / 2);
    const parts = [
      { kind: 'size', axis: 'x', sign: 1 }, { kind: 'size', axis: 'x', sign: -1 },
      { kind: 'size', axis: 'y', sign: 1 }, { kind: 'size', axis: 'y', sign: -1 },
      { kind: 'height', edge: 'top' }, { kind: 'height', edge: 'bottom' },
    ];
    const sizePts = [];
    for (const part of parts) {
      const local = new THREE.Vector3(
        part.axis === 'x' ? part.sign * half.x : 0,
        part.axis === 'y' ? part.sign * half.y : 0,
        part.kind === 'height' ? (part.edge === 'top' ? half.z : -half.z) : 0,
      );
      const world = local.clone().applyQuaternion(q).add(center);
      state.handles.push({ ...part, pos: world });
      sizePts.push(world.x, world.y, world.z);
    }
    // Поворот: точка над верхом бокса, на его вертикальной оси.
    const yawPos = new THREE.Vector3(0, 0, half.z + YAW_HANDLE_UP_M)
      .applyQuaternion(q).add(center);
    state.handles.push({ kind: 'yaw', pos: yawPos });
    handleGroup.add(pointsMesh(sizePts, SIZE_HANDLE_COLOR));
    handleGroup.add(pointsMesh([yawPos.x, yawPos.y, yawPos.z], YAW_HANDLE_COLOR));
    const stem = new THREE.BufferGeometry();
    stem.setAttribute('position', new THREE.Float32BufferAttribute(
      [center.x, center.y, center.z + half.z, yawPos.x, yawPos.y, yawPos.z], 3));
    const stemMesh = new THREE.LineSegments(stem, cachedMaterial('hstem', () => new THREE.LineBasicMaterial({
      color: YAW_HANDLE_COLOR, transparent: true, opacity: 0.7, depthTest: false,
    })));
    stemMesh.renderOrder = ORDER.handles;
    stemMesh.frustumCulled = false;
    handleGroup.add(stemMesh);
    // Концы траектории -- тоже ручки: начало зелёное, конец красное, тянутся по
    // горизонтальной плоскости на своей высоте (правка -- {kind:'traj', which}).
    if (box.traj && box.traj.start && box.traj.end) {
      const start = new THREE.Vector3(box.traj.start[0], box.traj.start[1], box.traj.start[2]);
      const end = new THREE.Vector3(box.traj.end[0], box.traj.end[1], box.traj.end[2]);
      state.handles.push({ kind: 'traj', which: 'start', pos: start });
      state.handles.push({ kind: 'traj', which: 'end', pos: end });
      const startMesh = pointsMesh([start.x, start.y, start.z], TRAJ_START_COLOR, TRAJ_END_R_PX + 3);
      const endMesh = pointsMesh([end.x, end.y, end.z], TRAJ_END_COLOR, TRAJ_END_R_PX + 3);
      handleGroup.add(startMesh);
      handleGroup.add(endMesh);
    }
    handleGroup.visible = true;
  }

  function pointsMesh(coords, color, size) {
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.Float32BufferAttribute(coords, 3));
    // Материал ручек -- из кэша (по цвету и размеру): buildHandles зовётся на
    // каждой перерисовке, а материалов тут всего четыре.
    const mat = cachedMaterial(`hpts:${color}:${size || HANDLE_SIZE_PX}`, () => new THREE.PointsMaterial({
      color, size: size || HANDLE_SIZE_PX, sizeAttenuation: false,
      depthTest: false, depthWrite: false,
    }));
    const mesh = new THREE.Points(geom, mat);
    mesh.renderOrder = ORDER.handles;
    mesh.frustumCulled = false;
    return mesh;
  }

  /**
   * Точки трассировки по ВСЕМ объектам кадра и объединённая маска закрытых.
   *
   * `items` -- [{key, points (Float32Array|null), hidden (индексы|null)}]. Голубые
   * точки всех объектов рисуются одним объектом сцены, а ЗАКРЫТЫЕ индексы
   * складываются В ОДИН НАБОР (сортировка + уникальные): раньше маска жила у одного
   * выбранного объекта, и постановка второго объекта стирала набор первого --
   * ровно жалоба «не пересчитались другие точки, которые за ним». Набор уходит в
   * `onHidden(indices)`: облаком владеет вьюер (`applyHiddenPoints`).
   */
  function setPreviews(items) {
    dropPreview();
    const list = Array.isArray(items) ? items : [];
    const arrays = [];
    const hidden = new Set();
    let total = 0;
    for (const item of list) {
      const pts = item && item.points;
      let arr = null;
      if (pts instanceof Float32Array) arr = pts;
      else if (Array.isArray(pts) && pts.length) arr = new Float32Array(pts.flat());
      if (arr && arr.length >= 3) {
        arrays.push(arr);
        total += arr.length / 3;
      }
      for (const i of (item && Array.isArray(item.hidden)) ? item.hidden : []) {
        if (Number.isFinite(i) && i >= 0) hidden.add(i | 0);
      }
    }
    state.hidden = [...hidden].sort((a, b) => a - b);
    if (onHidden) onHidden(state.hidden);
    if (!arrays.length) {
      previewGroup.visible = false;
      return state.previewCount;
    }
    let merged = null;
    if (arrays.length === 1) {
      merged = arrays[0];
    } else {
      merged = new Float32Array(total * 3);
      let off = 0;
      for (const arr of arrays) {
        merged.set(arr, off);
        off += arr.length;
      }
    }
    previewGeometry = new THREE.BufferGeometry();
    previewGeometry.setAttribute('position', new THREE.BufferAttribute(merged, 3));
    previewMaterial = new THREE.PointsMaterial({
      color: PREVIEW_COLOR, size: PREVIEW_SIZE_PX, sizeAttenuation: false,
      depthTest: false, depthWrite: false,
    });
    previewPoints = new THREE.Points(previewGeometry, previewMaterial);
    previewPoints.renderOrder = ORDER.preview;
    previewPoints.frustumCulled = false;
    previewGroup.add(previewPoints);
    state.previewCount = total;
    previewGroup.visible = state.on;
    return state.previewCount;
  }

  /** Совместимость: одна группа точек (используется проверками и вызывающим кодом,
   * которому объединять нечего). */
  function setPreview(points) {
    return setPreviews([{ key: 'preview', points, hidden: [] }]);
  }

  function setVisible(on) {
    state.on = Boolean(on);
    boxGroup.visible = state.on && boxGroup.children.length > 0;
    poseGroup.visible = state.on && poseGroup.children.length > 0;
    trajGroup.visible = state.on && trajGroup.children.length > 0;
    previewGroup.visible = state.on && state.previewCount > 0;
    if (!state.on && state.hidden.length) {
      state.hidden = [];
      if (onHidden) onHidden([]);
    }
    buildHandles();
    return state.on;
  }

  // ---------------------------------------------------------------- ручки ---

  /** Пересечение луча курсора с вертикальной плоскостью через точку (x, y). */
  function pointOnVerticalPlane(clientX, clientY, x, y) {
    const rect = canvas.getBoundingClientRect();
    if (!(rect.width > 0 && rect.height > 0)) return null;
    const ndc = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1,
    );
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, camera);
    // Нормаль -- горизонтальное направление взгляда: плоскость «смотрит» на
    // камеру, и ручка высоты тянется вверх-вниз из любого ракурса, кроме
    // взгляда строго сверху (там высота не читается и мышью).
    const dir = ray.ray.direction;
    const normal = new THREE.Vector3(dir.x, dir.y, 0);
    if (normal.lengthSq() < 1e-9) return null;
    normal.normalize();
    const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(
      normal, new THREE.Vector3(x, y, 0));
    const hit = new THREE.Vector3();
    if (!ray.ray.intersectPlane(plane, hit)) return null;
    return [hit.x, hit.y, hit.z];
  }

  /**
   * Экранные пиксели мировой точки: (x, y) или null, если точка за камерой.
   * Матрица -- та же, что у рендера (`projection·view`), поэтому окно вокруг
   * объекта совпадает с тем, что человек видит.
   */
  function screenOf(p, rect) {
    const r = rect || canvas.getBoundingClientRect();
    if (!(r.width > 0 && r.height > 0)) return null;
    camera.updateMatrixWorld();
    const m = new THREE.Matrix4()
      .multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse).elements;
    const w = m[3] * p.x + m[7] * p.y + m[11] * p.z + m[15];
    if (w <= 1e-6) return null;
    const cx = (m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12]) / w;
    const cy = (m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13]) / w;
    return [r.left + (cx * 0.5 + 0.5) * r.width, r.top + (-cy * 0.5 + 0.5) * r.height];
  }

  /** Восемь углов бокса в экранных пикселях (за камерой -- выброшены). */
  function boxCorners(box, rect) {
    const c = box.center || [0, 0, 0];
    const s = box.size || [1, 1, 1];
    const q = labelQuaternion(box.yaw, box.pitch, box.roll);
    const center = new THREE.Vector3(c[0], c[1], c[2]);
    const h = new THREE.Vector3(Math.max(s[0], 0.05) / 2, Math.max(s[2], 0.05) / 2,
                                Math.max(s[1], 0.05) / 2);
    const out = [];
    for (const sx of [-1, 1]) {
      for (const sy of [-1, 1]) {
        for (const sz of [-1, 1]) {
          const p = new THREE.Vector3(sx * h.x, sy * h.y, sz * h.z)
            .applyQuaternion(q).add(center);
          const px = screenOf(p, rect);
          if (px) out.push(px);
        }
      }
    }
    return out;
  }

  /** Выпуклая оболочка (монотонная цепь) -- проекция бокса это параллелепипед. */
  function convexHull(pts) {
    if (pts.length < 3) return pts.slice();
    const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lower = [];
    for (const q of p) {
      while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop();
      lower.push(q);
    }
    const upper = [];
    for (let i = p.length - 1; i >= 0; i--) {
      const q = p[i];
      while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop();
      upper.push(q);
    }
    lower.pop();
    upper.pop();
    return lower.concat(upper);
  }

  function inConvex(poly, x, y) {
    let pos = 0;
    let neg = 0;
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i];
      const b = poly[(i + 1) % poly.length];
      const c = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);
      if (c > 0) pos += 1;
      else if (c < 0) neg += 1;
    }
    return pos === 0 || neg === 0;
  }

  /** Расстояние от точки до периметра оболочки, px. */
  function distToHull(poly, x, y) {
    let best = Infinity;
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i];
      const b = poly[(i + 1) % poly.length];
      const dx = b[0] - a[0];
      const dy = b[1] - a[1];
      const len2 = dx * dx + dy * dy;
      const t = len2 > 0 ? Math.max(0, Math.min(1, ((x - a[0]) * dx + (y - a[1]) * dy) / len2)) : 0;
      best = Math.min(best, Math.hypot(x - (a[0] + t * dx), y - (a[1] + t * dy)));
    }
    return best;
  }

  /**
   * Курсор внутри проекции бокса (или в запасе `BOX_PAD_PX` вокруг неё).
   */
  function boxHit(box, clientX, clientY, rect) {
    const hull = convexHull(boxCorners(box, rect));
    if (hull.length < 3) {
      // Бокс смотрит ровно вдоль луча: оболочки нет -- берём окно вокруг центра.
      const c = box.center || [0, 0, 0];
      const px = screenOf(new THREE.Vector3(c[0], c[1], c[2]), rect);
      return Boolean(px && Math.hypot(px[0] - clientX, px[1] - clientY) <= BOX_MIN_PX);
    }
    if (inConvex(hull, clientX, clientY)) return true;
    return distToHull(hull, clientX, clientY) <= BOX_PAD_PX;
  }

  /**
   * Бокс ПОД курсором -- любой, не только выбранный. Клик по невыбранному боксу
   * выбирает объект (`onSelect`), а не предлагает новый поверх него: раньше
   * выбрать можно было только строкой списка в панели, и клик по чужому боксу
   * ставил дубль. При перекрытии берём бокс с ближайшим к курсору центром.
   */
  function pickBox(clientX, clientY) {
    if (!state.on) return null;
    const rect = canvas.getBoundingClientRect();
    let best = null;
    let bestD = Infinity;
    for (const box of state.boxes) {
      if (!boxHit(box, clientX, clientY, rect)) continue;
      const c = box.center || [0, 0, 0];
      const px = screenOf(new THREE.Vector3(c[0], c[1], c[2]), rect);
      const d = px ? Math.hypot(px[0] - clientX, px[1] - clientY) : Infinity;
      if (d <= bestD) { bestD = d; best = box; }
    }
    return best;
  }

  /** Ручка под курсором И расстояние до неё в пикселях: нужно, чтобы решить
   * «тянем ручку или двигаем бокс» (см. onPointerDown). */
  function pickHandleNear(clientX, clientY, radiusPx = HANDLE_R_PX) {
    const rect = canvas.getBoundingClientRect();
    if (!(rect.width > 0 && rect.height > 0)) return null;
    const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1;
    const ndcY = -((clientY - rect.top) / rect.height) * 2 + 1;
    camera.updateMatrixWorld();
    const m = new THREE.Matrix4()
      .multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse).elements;
    const halfW = rect.width / 2;
    const halfH = rect.height / 2;
    let best = null;
    let bestD2 = Infinity;
    for (const handle of state.handles) {
      const p = handle.pos;
      const w = m[3] * p.x + m[7] * p.y + m[11] * p.z + m[15];
      if (w <= 1e-6) continue;
      const cx = (m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12]) / w;
      const cy = (m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13]) / w;
      const dx = (cx - ndcX) * halfW;
      const dy = (cy - ndcY) * halfH;
      const d2 = dx * dx + dy * dy;
      if (d2 <= radiusPx * radiusPx && d2 < bestD2) {
        bestD2 = d2;
        best = handle;
      }
    }
    return best ? { handle: best, d: Math.sqrt(bestD2) } : null;
  }

  /** Ручка под курсором: ближайшая в экранном окне, с учётом перекрытия. */
  function pickHandle(clientX, clientY, radiusPx = HANDLE_R_PX) {
    const got = pickHandleNear(clientX, clientY, radiusPx);
    return got ? got.handle : null;
  }

  /** Значение правки по положению курсора: что именно тянет человек. */
  function handleEdit(handle, box, clientX, clientY) {
    const c = box.center;
    if (handle.kind === 'yaw') {
      const hit = pointOnPlane(clientX, clientY, c[2]);
      if (!hit) return null;
      const dx = hit[0] - c[0];
      const dy = hit[1] - c[1];
      if (Math.hypot(dx, dy) < 0.05) return null;
      // Локальная ось Y (длина бокса) после Rz(yaw) -- это (-sin, cos).
      return { kind: 'yaw', yaw_deg: THREE.MathUtils.radToDeg(Math.atan2(-dx, dy)) };
    }
    if (handle.kind === 'height') {
      const hit = pointOnVerticalPlane(clientX, clientY, c[0], c[1]);
      if (!hit) return null;
      return { kind: 'height', edge: handle.edge, z_m: hit[2] };
    }
    if (handle.kind === 'traj') {
      // Конец траектории тянется по горизонтальной плоскости НА СВОЕЙ высоте:
      // траектория -- это «откуда и куда», высота концов (опора) не должна
      // съезжать от случайного движения мыши вверх-вниз.
      const z = handle.pos.z;
      const hit = pointOnPlane(clientX, clientY, z);
      if (!hit) return null;
      return { kind: 'traj', which: handle.which, point: [hit[0], hit[1], z] };
    }
    const hit = pointOnPlane(clientX, clientY, c[2]);
    if (!hit) return null;
    const q = labelQuaternion(box.yaw, box.pitch, box.roll).invert();
    const local = new THREE.Vector3(hit[0] - c[0], hit[1] - c[1], 0).applyQuaternion(q);
    const half = handle.axis === 'x' ? Math.abs(local.x) : Math.abs(local.y);
    return { kind: 'size', axis: handle.axis, size_m: 2 * half };
  }

  // ------------------------------------------------------------------ клик ---

  /** Пересечение луча курсора с горизонтальной плоскостью Z = z (правка мышью). */
  function pointOnPlane(clientX, clientY, z) {
    const rect = canvas.getBoundingClientRect();
    if (!(rect.width > 0 && rect.height > 0)) return null;
    const ndc = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1,
    );
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, camera);
    const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -z);
    const hit = new THREE.Vector3();
    if (!ray.ray.intersectPlane(plane, hit)) return null;
    return [hit.x, hit.y];
  }

  /**
   * Точка облака под курсором: луч из камеры через мышь, из точек кадра -- та,
   * что ближе всех к лучу И попала в экранное окно курсора.
   */
  function pickPoint(clientX, clientY, radiusPx = PICK_RADIUS_PX) {
    const data = cloud ? cloud() : null;
    const positions = data && data.positions;
    const count = data ? Number(data.count) | 0 : 0;
    if (!positions || count <= 0) return null;
    const rect = canvas.getBoundingClientRect();
    if (!(rect.width > 0 && rect.height > 0)) return null;
    const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1;
    const ndcY = -((clientY - rect.top) / rect.height) * 2 + 1;
    const ray = new THREE.Raycaster();
    ray.setFromCamera(new THREE.Vector2(ndcX, ndcY), camera);
    const origin = ray.ray.origin;
    const dir = ray.ray.direction;
    // Проекция точки на экран -- той же матрицей, что и рендер: так окно курсора
    // считается без second-guessing шейдера (размер точки задан в мире).
    camera.updateMatrixWorld();
    const m = new THREE.Matrix4()
      .multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse).elements;
    const halfW = rect.width / 2;
    const halfH = rect.height / 2;
    const maxDist = Number(data.maxDist) || 0;
    const tmp = new THREE.Vector3();
    let best = -1;
    let bestPerp = Infinity;
    for (let i = 0; i < count; i++) {
      const x = positions[i * 3];
      const y = positions[i * 3 + 1];
      const z = positions[i * 3 + 2];
      if (maxDist > 0 && -y > maxDist) continue;   // такие точки шейдер не рисует
      const w = m[3] * x + m[7] * y + m[11] * z + m[15];
      if (w <= 1e-6) continue;
      const cx = (m[0] * x + m[4] * y + m[8] * z + m[12]) / w;
      const cy = (m[1] * x + m[5] * y + m[9] * z + m[13]) / w;
      const dx = (cx - ndcX) * halfW;
      const dy = (cy - ndcY) * halfH;
      if (dx * dx + dy * dy > radiusPx * radiusPx) continue;
      // |v × d| при |d| = 1 -- расстояние от точки до луча, м.
      const perp = tmp.set(x - origin.x, y - origin.y, z - origin.z).cross(dir).length();
      if (perp < bestPerp) {
        bestPerp = perp;
        best = i;
      }
    }
    if (best < 0) return null;
    return [
      positions[best * 3], positions[best * 3 + 1], positions[best * 3 + 2],
    ];
  }

  // События ловим на window в capture-фазе: на самой канве наш обработчик идёт
  // ПОСЛЕ OrbitControls (он подписан раньше), и остановить вращение было бы нечем.
  // Жест правки -- ручка или перенос центра: начинается только по ВЫБРАННОМУ боксу,
  // и с этого момента событие не доходит до OrbitControls -- камера не крутится
  // ровно на этот жест и остаётся рабочей сразу после него.
  const down = { active: false, x: 0, y: 0 };
  const grab = { active: false, moved: false, kind: null, handle: null, box: null,
                 dx: 0, dy: 0 };

  function startGrab(ev, kind, handle, box) {
    grab.active = true;
    grab.moved = false;
    grab.kind = kind;
    grab.handle = handle || null;
    grab.box = box;
    // Ловим событие в capture на window -- значит здесь его ещё никто не видел.
    ev.stopPropagation();
  }

  function onPointerDown(ev) {
    if (!state.on || ev.button !== 0 || ev.target !== canvas) return;
    down.active = true;
    down.x = ev.clientX;
    down.y = ev.clientY;
    const hitAny = pickBox(ev.clientX, ev.clientY);
    // Клик по НЕвыбранному объекту выбирает его, а не предлагает новый поверх:
    // раньше выбрать можно было только строкой в списке панели, и клик по чужому
    // боксу ставил дубль. Выбор не тянет объект -- перенос следующим жестом.
    if (hitAny && !hitAny.selected) {
      down.onBox = true;          // жест по боксу не ставит новый объект
      if (onSelect) onSelect(hitAny.key);
      return;
    }
    const box = state.boxes.find((b) => b.selected);
    if (!box) return;
    // Ручка -- первым делом, но только если курсор к ней БЛИЖЕ, чем к центру
    // бокса: у далёкой цели бокс занимает единицы пикселей, ручки на его гранях
    // стоят вплотную к центру -- и любое нажатие попадало бы в ручку, так что
    // бокс нельзя было бы даже сдвинуть. «Куда ближе -- то и тяну».
    const near = pickHandleNear(ev.clientX, ev.clientY);
    if (near) {
      const c = box.center || [0, 0, 0];
      const cp = screenOf(new THREE.Vector3(c[0], c[1], c[2]));
      const toCenter = cp ? Math.hypot(cp[0] - ev.clientX, cp[1] - ev.clientY) : Infinity;
      if (near.d <= toCenter) {
        startGrab(ev, near.handle.kind || 'handle', near.handle, box);
        return;
      }
    }
    if (pickBox(ev.clientX, ev.clientY) !== box) return;   // вне бокса -- камера
    const hit = pointOnPlane(ev.clientX, ev.clientY, box.center[2]);
    if (!hit) return;
    startGrab(ev, 'move', null, box);
    grab.dx = hit[0] - box.center[0];
    grab.dy = hit[1] - box.center[1];
    dragZ = box.center[2];
  }

  function onPointerMove(ev) {
    if (!grab.active) return;
    // Кнопку отпустили ВНЕ окна: pointerup до нас не дошёл, и без этой проверки
    // жест остался бы висеть -- бокс ехал бы за курсором без нажатой кнопки.
    if (ev.buttons === 0) {
      const didMove = grab.moved;
      grab.active = false;
      grab.moved = false;
      grab.handle = null;
      grab.box = null;
      if (didMove && onDragEnd) onDragEnd();
      return;
    }
    // Мёртвая зона: пока курсор не ушёл от точки нажатия, это ещё клик, а не
    // перетаскивание -- иначе бокс «уезжал» от дрожи руки.
    if (!grab.moved) {
      if (Math.hypot(ev.clientX - down.x, ev.clientY - down.y) <= CLICK_MOVE_PX) {
        ev.stopPropagation();
        return;
      }
      grab.moved = true;
      if (onDragStart) onDragStart(grab.kind);
    }
    ev.stopPropagation();
    if (grab.handle) {
      const edit = handleEdit(grab.handle, grab.box, ev.clientX, ev.clientY);
      if (edit && onEdit) onEdit(edit);
      return;
    }
    const hit = pointOnPlane(ev.clientX, ev.clientY, dragZ);
    if (hit && onDragMove) onDragMove(hit[0] - grab.dx, hit[1] - grab.dy, dragZ);
  }

  function onPointerUp(ev) {
    const wasGrab = grab.active;
    const didMove = grab.moved;
    grab.active = false;
    grab.moved = false;
    grab.handle = null;
    grab.box = null;
    if (wasGrab && didMove) {
      ev.stopPropagation();
      if (onDragEnd) onDragEnd();
    }
    if (!down.active) return;
    down.active = false;
    // Нажатие по боксу (выбор или жест правки) -- не постановка объекта:
    // иначе клик по объекту ставил бы дубль поверх него.
    if (down.onBox) { down.onBox = false; return; }
    if (!state.on || wasGrab || ev.target !== canvas || ev.button !== 0) return;
    if (Math.hypot(ev.clientX - down.x, ev.clientY - down.y) > CLICK_MOVE_PX) return;
    const point = pickPoint(ev.clientX, ev.clientY);
    if (!point) {
      if (onStatus) onStatus('под курсором нет точки облака — кликните по объекту');
      return;
    }
    if (onPick) onPick(point);
  }

  window.addEventListener('pointerdown', onPointerDown, true);
  window.addEventListener('pointermove', onPointerMove, true);
  window.addEventListener('pointerup', onPointerUp, true);

  return {
    setVisible,
    isVisible: () => state.on,
    setObjects,
    setPreview,
    setPreviews,
    previewCount: () => state.previewCount,
    // Объединённая маска закрытых точек (по всем объектам кадра) и что нарисовано.
    hiddenIndices: () => state.hidden.slice(),
    hiddenCount: () => state.hidden.length,
    // Меши поз: сколько, какие клипы и моменты -- проверяется без картинки.
    poses: () => state.poses.map((p) => ({ ...p })),
    poseCount: () => state.poseCount,
    poseMeshVisible: (key) => {
      const slot = poseMeshes.get(key);
      return Boolean(slot && slot.mesh.visible && poseGroup.visible);
    },
    trajectories: () => state.trajectories.map((t) => ({ ...t })),
    trajCount: () => state.trajCount,
    // Клик и плоскость -- публично: так их можно проверить из консоли страницы
    // (и из Node-проверки), не повторяя разбор луча по мыши.
    pick: (clientX, clientY) => pickPoint(clientX, clientY),
    plane: (clientX, clientY, z) => pointOnPlane(clientX, clientY, z),
    // Ручки -- тоже публично: проверка тянет их в пикселях, как мышью. `boxAt`
    // отвечает на вопрос «попал ли курсор в выбранный бокс» (та же проверка, по
    // которой начинается перенос), `isGrabbing` -- идёт ли жест правки сейчас.
    handles: () => state.handles.map((h) => ({
      kind: h.kind, axis: h.axis || null, edge: h.edge || null,
      which: h.which || null, pos: [h.pos.x, h.pos.y, h.pos.z],
    })),
    boxAt: (clientX, clientY) => Boolean(pickBox(clientX, clientY)),
    isGrabbing: () => grab.active,
    handleAt: (clientX, clientY, radiusPx) => {
      const h = pickHandle(clientX, clientY, radiusPx === undefined ? HANDLE_R_PX : radiusPx);
      return h ? { kind: h.kind, axis: h.axis || null, edge: h.edge || null,
                   which: h.which || null } : null;
    },
    editFromHandle: (which, clientX, clientY) => {
      const box = state.boxes.find((b) => b.selected);
      const handle = state.handles.find((h) => h.kind === which)
        || state.handles.find((h) => h.which === which)
        || state.handles.find((h) => h.axis === which || h.edge === which);
      if (!box || !handle) return null;
      return handleEdit(handle, box, clientX, clientY);
    },
    dispose() {
      window.removeEventListener('pointerdown', onPointerDown, true);
      window.removeEventListener('pointermove', onPointerMove, true);
      window.removeEventListener('pointerup', onPointerUp, true);
      grab.active = false;
      setObjects([]);
      dropPreview();
      dropTrajCaches();
      markerGeometry.dispose();
      arrowGeometry.dispose();
      scene.remove(boxGroup);
      scene.remove(poseGroup);
      scene.remove(trajGroup);
      scene.remove(previewGroup);
      scene.remove(handleGroup);
      unitBox.dispose();
      state.on = false;
    },
  };
}