// 3D-объекты пути (рельсы, шпалы, пол) для основного вьюера web_viewer.py.
//
// Модуль самодостаточный: createTrackLayer({ scene, camera, controls, ... })
// рисует меши из JSON /track (его считает track_geometry.build_track), вписывает
// камеру по габариту ОБЪЕКТОВ и умеет режим «чистое поле». Логика перенесена из
// прототипа web3d/ с уже проверенными там решениями по ловушкам:
//
//   * оси данных: X поперёк, Y вдоль (вперёд -- -Y), Z вверх. Меши в JSON уже в
//     этих осях -- никакого поворота к ним не применяется;
//   * «верх» мира задаёт web/app.js одной глобальной мутацией
//     THREE.Object3D.DEFAULT_UP.set(0, 0, 1) плюс camera.up. Здесь она намеренно
//     НЕ повторяется: дубль в двух файлах разъезжается при правках. Объекты слоя
//     создаются в main(), то есть уже после неё;
//   * THREE.ExtrudeGeometry не используется: он выдавливает профиль вдоль +Z и
//     рельс встал бы вертикально. Геометрия берётся готовыми vertices/faces;
//   * BoxGeometry(w, h, d) кладёт высоту в локальный Y, а меш не поворачивается,
//     поэтому порядок размеров шпалы -- ровно (X, Y, Z) из JSON;
//   * меши собираются вне цикла анимации, при пересборке старая геометрия и
//     материал диспоузятся.
//
// Вписывание камеры считается по ПОПЕРЕЧНЫМ размерам габарита (X и Z), а не по
// диагонали: длина коридора вдоль Y не должна отъезжать камеру, а пол (плита
// ~200 м вдоль пути) в габарит кадрирования вообще не входит.

import * as THREE from 'three';

const $ = (id) => document.getElementById(id);

// Порядки отрисовки: сплошные объекты не просвечивают друг через друга.
const ORDER = { floor: 2, sleepers: 5, rails: 6 };

// Ключ пресета камеры «Объекты пути» из ряда пресетов в web/app.js (PRESETS).
// Слой подсвечивает эту кнопку сам: вид по объектам ставится и на старте, и
// кнопкой, и подсветка должна быть одинаковой.
const TRACK_PRESET = 'track';
const START_VIEW_NOTE = 'камера вписана по объектам пути (рельсы и шпалы), окно 12 м';

const CLEAN_BG = 0xe9ecf1;
// Обе нити -- ОДИН нейтральный цвет (сталь): раньше левая была голубой, правая
// оранжевой, и оранжевый рельс читался как «оранжевым непонятно что» -- тот же
// оранжевый, что у зоны rail в раскраске zones. Теперь меш -- один объект.
const RAIL_COLOR = 0xb9c4d4;
const FLOOR_COLOR = 0x59637a;
const SLEEPER_COLOR = 0x9a7b4f;

// --- вписывание камеры --------------------------------------------------------
const START_VIEW_SPAN = 12;         // стартовое окно вдоль пути, м
// «Профиль» -- намеренно ближний вид: профиль рельса должен читаться по форме,
// поэтому ставится фиксированной дистанцией, а не вписыванием.
const PROFILE_DISTANCE_M = 2.8;
// Нижняя граница вписывания: ближе камера упирается в короткий объект.
const MIN_CAMERA_DISTANCE_M = 3;
// distance = max(size_x, size_z) / (2*tg(FOV/2)*fill), то есть fill -- какая доля
// высоты кадра приходится на поперечный размер габарита. 0.24: шпала 2.75 м
// занимает ~24 % высоты -> ~10.4 м дистанции, рельс по ширине ~11 px при 888 px.
const FIT_FILL = 0.24;

// Протяжка слайдера: строим только последний запрошенный кадр. DEBOUNCE_MS -- пауза,
// после которой считаем, что поток запросов прекратился (нужна только чтобы не
// начинать сборку на каждый промежуточный кадр протяжки).
// MAX_DEFER_MS = 0: искусственной задержки больше нет. Запросы, пришедшие пока идёт
// сборка, сходятся в один (собираем последний), а сразу после неё строится уже
// актуальный кадр. Раньше стояло 1500 мс, и при воспроизведении 10 Гц геометрия
// обновлялась раз в ~1.5-2 с -- это читалось как «рельсы отстают».
// Частота обновления теперь ограничена только временем сборки на сервере
// (build_track 0.4-0.7 с); его ускорение -- отдельная задача.
const DEBOUNCE_MS = 180;
const MAX_DEFER_MS = 0;

// --------------------------------------------------------- происхождение ----

// track_geometry отдаёт source как 'measured', 'gost', 'gost R65' -- приводим
// к русской пометке: зелёная плашка «измерено», жёлтая «ГОСТ».
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
  return raw.replace(/^gost/i, 'ГОСТ').replace(/\bgost\b/i, 'ГОСТ').replace(/r65/gi, 'Р65');
}

function sourceKind(value) {
  const key = String(value === null || value === undefined ? '' : value)
    .trim().toLowerCase();
  if (/gost|гост|standard|r65|р65/.test(key)) return 'gost';
  if (/measur|измер|наблюд|lidar|observed/.test(key)) return 'measured';
  return '';
}

// Плашка в строке узкая (34 px по CSS), поэтому длинные пояснения из source
// («measured: профиль (полки, стенки, дно лотка) по облаку, экструзия вдоль Y»)
// сокращаем до пометки, а полный текст показываем в свёрнутом блоке.
function shortSource(value, max = 24) {
  const raw = String(value === null || value === undefined ? '' : value).trim();
  if (!raw) return '—';
  const kind = sourceKind(raw);
  if (kind && raw.length > max) return kind === 'gost' ? 'ГОСТ' : 'измерено';
  const text = sourceText(raw);
  return text.length > max ? text.slice(0, max - 1) + '…' : text;
}

function railSourceParts(rail) {
  const src = (rail && rail.source) || {};
  return ['head', 'web', 'foot'].filter((k) => src[k])
    .map((k) => `${RAIL_PART[k]} — ${sourceText(src[k])}`);
}

function meshStats(mesh) {
  return {
    vertices: mesh && Array.isArray(mesh.vertices) ? mesh.vertices.length : 0,
    faces: mesh && Array.isArray(mesh.faces) ? mesh.faces.length : 0,
  };
}

function isNumber(v) { return typeof v === 'number' && Number.isFinite(v); }

function fmt(v, digits = 3) { return isNumber(v) ? v.toFixed(digits) : String(v); }

// -------------------------------------------------- построение геометрии -----

function objectMaterial(color) {
  return new THREE.MeshStandardMaterial({
    color, metalness: 0.15, roughness: 0.75,
    flatShading: true, side: THREE.DoubleSide,
  });
}

// Один материал на обе нити: side приходит из данных, но цвет от него больше не
// зависит -- иначе две нити выглядели как два разных объекта.
function railMaterial() {
  return objectMaterial(RAIL_COLOR);
}

// faces приходят треугольниками по контракту, но многоугольники тоже принимаем:
// режем веером, иначе грани молча пропадут.
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

// Габарит ТОЛЬКО объектов пути -- рельсы + шпалы. Пол намеренно не входит:
// это плита ~199 м вдоль коридора, для кадрирования она бесполезна (тянет
// вписывание на весь коридор). Рисуется он по-прежнему.
function trackBoundsFromData(data) {
  const box = new THREE.Box3();
  const v = new THREE.Vector3();
  let any = false;
  const eat = (vertices) => {
    for (const p of vertices || []) {
      if (!p || p.length < 3) continue;
      if (!isNumber(p[0]) || !isNumber(p[1]) || !isNumber(p[2])) continue;
      box.expandByPoint(v.set(p[0], p[1], p[2]));
      any = true;
    }
  };
  for (const rail of data.rails || []) eat(rail && rail.mesh && rail.mesh.vertices);
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

// Окно вдоль пути: ближние `span` метров коридора (у сенсора y ~ 0).
function corridorViewBox(box, span) {
  const b = box.clone();
  b.min.y = Math.max(b.min.y, b.max.y - span);
  return b;
}

function cameraDirection() { return new THREE.Vector3(0.55, 0.60, 0.58).normalize(); }
function profileDirection() { return new THREE.Vector3(0.50, 0.72, 0.48).normalize(); }

// Слой. Возвращает объект с методами load / setVisible / setCleanField / dispose
// плюс fitToTrack / viewProfile / isCleanField.
export function createTrackLayer({
  scene, camera, controls,
  refreshPoints = null,     // () => void: переключить видимость облака точек
  onError = null,           // (err) => void: показать ошибку (кроме 503)
  startViewSpan = START_VIEW_SPAN,
} = {}) {
  if (!scene || !camera || !controls) {
    throw new Error('createTrackLayer: нужны scene, camera и controls');
  }

  const state = {
    data: null, bag: null, frame: 0,
    visible: { rails: true, sleepers: true, floor: true },
    clean: false, saved: null, unavailable: null,
    // Автовид по объектам: ставить стартовую камеру по габариту рельсов и шпал.
    // Источник истины -- галочка в панели (браузер восстанавливает её состояние
    // при перезагрузке страницы), иначе -- включено.
    autoView: true,
  };
  const groups = { floor: null, sleepers: null, rails: null };
  const lights = [];
  let request = null;       // последний запрошенный кадр (побеждает последний)
  let drainPromise = null;
  let builtBag = null;
  let builtFrame = null;
  let lastBuildAt = 0;
  let startViewApplied = false;
  let userMovedCamera = false;

  const autoViewBox = $('t-startview');
  if (autoViewBox) state.autoView = Boolean(autoViewBox.checked);

  // Видимость слоёв берём из галочек панели: в разметке меши по умолчанию ВЫКЛЮЧЕНЫ
  // (видна «лента рельсов» вьюера), и слой обязан стартовать в том же состоянии,
  // иначе картинка и панель разъезжаются. Нет галочек -- слой просто видим.
  for (const key of ['rails', 'sleepers', 'floor']) {
    const box = $('t-' + key);
    if (box) state.visible[key] = Boolean(box.checked);
  }

  // Свет: без него MeshStandardMaterial чёрный (в основном вьюере свет не задан
  // -- облако точек рисуется своим шейдером и в лампах не нуждается).
  scene.add(lights[0] = new THREE.HemisphereLight(0xffffff, 0x2a3040, 1.0));
  scene.add(lights[1] = new THREE.DirectionalLight(0xffffff, 1.7));
  lights[1].position.set(80, 60, 120);
  scene.add(lights[2] = new THREE.DirectionalLight(0x88aaff, 0.5));
  lights[2].position.set(-60, -40, 60);

  const onControlsStart = () => { userMovedCamera = true; };
  controls.addEventListener('start', onControlsStart);
  // Клик по пресету камеры основного вьюера -- тоже «пользователь выбрал вид»:
  // стартовый вид по объектам тогда не навязывается, а подсветка вида снимается,
  // чтобы не горели две кнопки сразу. Исключение -- сам пресет «Объекты пути»:
  // его подсвечивает fitToTrack() ниже.
  const presetsBox = $('presets');
  const onPresetClick = (ev) => {
    const preset = ev && ev.target && ev.target.dataset ? ev.target.dataset.preset : null;
    if (preset === TRACK_PRESET) return;
    userMovedCamera = true;
    markViews(null);
  };
  if (presetsBox) presetsBox.addEventListener('click', onPresetClick);

  // ------------------------------------------------------------------ DOM ----

  function setBadge(id, text, kind) {
    const el = $(id);
    if (!el) return;
    el.textContent = text;
    el.className = 'src' + (kind ? ' ' + kind : '');
  }

  function updateBadges() {
    setBadge('src-points', 'измерено', 'measured');   // облако всегда с лидара
    const data = state.data;
    if (!data) {
      for (const id of ['src-rails', 'src-sleepers', 'src-floor']) setBadge(id, '—', '');
      return;
    }
    const kinds = new Set();
    for (const rail of data.rails || []) {
      for (const k of ['head', 'web', 'foot']) {
        const kind = sourceKind((rail && rail.source || {})[k]);
        if (kind) kinds.add(kind);
      }
    }
    setBadge('src-rails',
      kinds.has('measured') && kinds.has('gost') ? 'измерено + ГОСТ'
        : kinds.has('measured') ? 'измерено' : kinds.has('gost') ? 'ГОСТ' : '—',
      kinds.has('measured') ? 'measured' : (kinds.has('gost') ? 'gost' : ''));
    const sleepers = data.sleepers || {};
    const sInst = (sleepers.instances || []).length;
    // Шпал в этих записях НЕТ: путь встроенный (рельсы в плите, между ними лоток),
    // периодики в поле не нашли -- это измеренный результат, а не пропуск замера.
    // Раньше плашка говорила просто «нет», и переключатель выглядел пустышкой;
    // теперь сказано, что именно не наблюдается и где смотреть основание пути.
    setBadge('src-sleepers', sInst ? shortSource(sleepers.source) : 'не наблюдаются',
      sInst ? sourceKind(sleepers.source) : 'gost');
    const sleeperRow = $('t-sleepers');
    if (sleeperRow) {
      sleeperRow.disabled = !sInst;
      const label = sleeperRow.closest('label');
      if (label) {
        label.title = sInst
          ? 'шпалы построены схематично по ГОСТ 78-2004'
          : 'периодических шпал в облаке нет (проверено Фурье по полосам 5 см: '
            + 'амплитуда 0.0 мм при пороге 13-16 мм). Измеренное основание пути -- '
            + 'в секции «Объекты из замеров»: там профиль плиты и лотка по данным.';
      }
    }
    const floor = data.floor_mesh || {};
    const fVerts = (floor.vertices || []).length;
    setBadge('src-floor', fVerts ? shortSource(floor.source) : 'нет',
      fVerts ? sourceKind(floor.source) : '');
  }

  // notes -- «откуда взялись числа»: 11 строк из track_geometry. Свёрнутый блок
  // под настройками, по умолчанию закрыт.
  function updateNotes() {
    const meta = (state.data && state.data.meta) || {};
    const notes = meta.notes ? (Array.isArray(meta.notes) ? meta.notes : [meta.notes]) : [];
    const summary = $('notesSummary');
    if (summary) summary.textContent = `Откуда числа (${notes.length})`;
    const box = $('notes');
    if (box) {
      box.textContent = '';
      for (const note of notes) {
        const line = document.createElement('div');
        line.textContent = String(note);
        box.appendChild(line);
      }
    }
    updateSourceDetails();
  }

  // Пер-нитьевая расшифровка источника и размеров мешей -- в тот же свёрнутый
  // блок: в компактном HUD это не нужно, а в отчёте -- нужно.
  function updateSourceDetails() {
    const box = $('trackSrc');
    if (!box) return;
    box.textContent = '';
    const data = state.data;
    const add = (text) => {
      const line = document.createElement('div');
      line.textContent = text;
      box.appendChild(line);
      return line;
    };
    if (!data) {
      add(state.unavailable ? `объекты пути недоступны: ${state.unavailable}` : 'данных нет');
      return;
    }
    for (const rail of data.rails || []) {
      const st = meshStats(rail && rail.mesh);
      const h = (rail && rail.head) || {};
      add(`нить ${rail.side}: меш ${st.vertices} вершин / ${st.faces} граней; `
        + `внутр. грань x = ${fmt(h.inner_face_x)} м, верх z = ${fmt(h.top_z)} м, `
        + `головка ${fmt(h.width_mm, 1)} мм`
        + (isNumber(h.height_above_adjacent_m)
          ? `, высота над прилегающей ${fmt(h.height_above_adjacent_m)} м` : ''));
      add(`  source: ${railSourceParts(rail).join(' · ') || '—'}`);
    }
    const sl = data.sleepers || {};
    const inst = sl.instances || [];
    add(`шпалы: ${inst.length} шт, шаг ${fmt(sl.spacing_m)} м, `
      + `бокс ${(inst[0] && inst[0].size || []).map((v) => fmt(v, 2)).join(' × ')} м (X × Y × Z), `
      + `source: ${sourceText(sl.source)}`);
    const fs = meshStats(data.floor_mesh || {});
    const fl = data.floor || {};
    add(`пол: меш ${fs.vertices} вершин / ${fs.faces} граней, source: `
      + `${sourceText((data.floor_mesh || {}).source)}; плоскость z = ${fmt(fl.a, 5)}·x + `
      + `${fmt(fl.b, 5)}·y + ${fmt(fl.c)}`);
    const ax = data.axis || {};
    add(`ось: x = ${fmt(ax.s, 5)}·y + ${fmt(ax.i)}`);
  }

  // Компактный HUD (2-я строка): колея по граням / ось-ось, ширина головки,
  // счётчики. Первую строку (запись, кадр, точки) пишет app.js.
  function updateTrackHud() {
    const el = $('hudTrack');
    if (!el) return;
    if (!state.data) {
      el.textContent = state.unavailable
        ? `объекты пути недоступны: ${state.unavailable}`
        : 'объекты пути: загрузка…';
      return;
    }
    const meta = state.data.meta || {};
    const parts = [];
    if (meta.gauge_inner_mm !== undefined && meta.gauge_inner_mm !== null) {
      parts.push(`колея по граням ${Math.round(Number(meta.gauge_inner_mm))} мм`
        + (meta.gauge_axis_mm !== undefined && meta.gauge_axis_mm !== null
          ? ` / ось–ось ${Math.round(Number(meta.gauge_axis_mm))} мм` : ''));
    }
    if (meta.head_width_mm !== undefined && meta.head_width_mm !== null) {
      parts.push(`головка ${Number(meta.head_width_mm).toFixed(1)} мм`);
    }
    const rails = (state.data.rails || []).length;
    const sl = state.data.sleepers || {};
    const inst = (sl.instances || []).length;
    parts.push(`рельсов ${rails}`,
      inst && isNumber(sl.spacing_m) ? `шпал ${inst} (шаг ${sl.spacing_m.toFixed(3)} м)` : `шпал ${inst}`);
    if (isNumber(meta.sensor_height_above_head_m)) {
      parts.push(`сенсор над головкой ${Number(meta.sensor_height_above_head_m).toFixed(3)} м`);
    }
    if (state.clean) parts.push('режим: чистое поле');
    el.textContent = parts.join(' · ');
  }

  // --------------------------------------------------------------- меши -----

  function dropGroup(key) {
    const group = groups[key];
    if (!group) return;
    scene.remove(group);
    disposeObject(group);
    groups[key] = null;
  }

  function buildFloor(data) {
    dropGroup('floor');
    const mesh = data.floor_mesh || {};
    const vertices = mesh.vertices || [];
    if (!vertices.length) return;
    const material = objectMaterial(FLOOR_COLOR);
    material.polygonOffset = true;        // не мерцать с нижними гранями шпал
    material.polygonOffsetFactor = 1;
    material.polygonOffsetUnits = 1;
    const group = new THREE.Group();
    group.name = 'track-floor';
    const m = meshFromArrays(vertices, mesh.faces, material);
    m.renderOrder = ORDER.floor;
    group.add(m);
    group.renderOrder = ORDER.floor;
    groups.floor = group;
    scene.add(group);
  }

  function buildSleepers(data) {
    dropGroup('sleepers');
    const instances = (data.sleepers || {}).instances || [];
    if (!instances.length) return;
    const group = new THREE.Group();
    group.name = 'track-sleepers';
    const material = objectMaterial(SLEEPER_COLOR);
    // BoxGeometry(w, h, d): w по X, h по Y, d по Z. Меш не поворачивается, поэтому
    // размеры берём ровно в порядке данных -- иначе шпала встанет стенкой.
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

  function buildRails(data) {
    dropGroup('rails');
    const group = new THREE.Group();
    group.name = 'track-rails';
    for (const rail of data.rails || []) {
      const mesh = rail && rail.mesh;
      if (!mesh || !Array.isArray(mesh.vertices) || !mesh.vertices.length) continue;
      const m = meshFromArrays(mesh.vertices, mesh.faces, railMaterial());
      m.renderOrder = ORDER.rails;
      group.add(m);
    }
    if (!group.children.length) return;
    group.renderOrder = ORDER.rails;
    groups.rails = group;
    scene.add(group);
  }

  function applyVisibility() {
    for (const key of ['floor', 'sleepers', 'rails']) {
      if (groups[key]) groups[key].visible = state.visible[key];
    }
    if (refreshPoints) refreshPoints();
  }

  // Меши собираются здесь -- вне цикла анимации, по одному разу на кадр.
  function rebuildObjects() {
    const data = state.data;
    if (!data) {
      dropGroup('floor'); dropGroup('sleepers'); dropGroup('rails');
      if (refreshPoints) refreshPoints();
      return;
    }
    buildFloor(data);
    buildSleepers(data);
    buildRails(data);
    applyVisibility();
  }

  // ------------------------------------------------------------- камера -----

  // near/far намеренно не трогаем: в основном вьюере near = 0.1, far = 4000, и
  // этого хватает от профиля (2.8 м) до ползунка «дальность» (200 м) и
  // controls.maxDistance (1500 м). Сужение far обрезало бы облако точек.
  function placeCamera(target, distance, direction) {
    camera.position.copy(target).addScaledVector(direction, distance);
    controls.target.copy(target);
    camera.updateProjectionMatrix();
    controls.update();
  }

  function fitCameraToBox(box, { fill = FIT_FILL, direction = cameraDirection() } = {}) {
    const size = box.getSize(new THREE.Vector3());
    const transverse = Math.max(size.x, size.z);
    const fov = THREE.MathUtils.degToRad(camera.fov);
    const distance = Math.max(
      transverse / (2 * Math.tan(fov / 2) * fill), MIN_CAMERA_DISTANCE_M);
    placeCamera(box.getCenter(new THREE.Vector3()), distance, direction);
    return distance;
  }

  function trackBounds() {
    if (state.data) {
      const fromData = trackBoundsFromData(state.data);
      if (fromData) return fromData;
    }
    const box = new THREE.Box3();
    let any = false;
    for (const key of ['sleepers', 'rails']) {   // пол в габарит не входит
      if (!groups[key]) continue;
      const b = new THREE.Box3().setFromObject(groups[key]);
      if (b.isEmpty()) continue;
      box.union(b);
      any = true;
    }
    return any ? box : null;
  }

  // Вписывание по объектам пути в окне span метров. Дистанция -- по поперечным
  // размерам (X, Z): иначе 35-метровая длина нитей уводит камеру на десятки
  // метров и рельс занимает единицы пикселей.
  function fitToTrack(span = startViewSpan) {
    const box = trackBounds();
    if (!box) return 0;
    return fitCameraToBox(corridorViewBox(box, span));
  }

  function railBoundsOf(rail) {
    const box = new THREE.Box3();
    const v = new THREE.Vector3();
    let any = false;
    for (const p of ((rail && rail.mesh && rail.mesh.vertices) || [])) {
      if (!p || p.length < 3) continue;
      if (!isNumber(p[0]) || !isNumber(p[1]) || !isNumber(p[2])) continue;
      box.expandByPoint(v.set(p[0], p[1], p[2]));
      any = true;
    }
    if (any) return box;
    const head = (rail && rail.head) || {};
    if (!isNumber(head.inner_face_x) || !isNumber(head.top_z)) return null;
    const half = (isNumber(head.width_mm) ? head.width_mm : 70) / 2000;
    const height = isNumber(head.height_above_adjacent_m) ? head.height_above_adjacent_m : 0.17;
    const range = Array.isArray(head.y_range) ? head.y_range : [-1, 0];
    box.expandByPoint(v.set(head.inner_face_x - half, Math.min(range[0], range[1]),
      head.top_z - height));
    box.expandByPoint(v.set(head.inner_face_x + half, Math.max(range[0], range[1]),
      head.top_z));
    return box;
  }

  // Нить для «Профиля»: та, чей меш подходит ближе всего к сенсору (больший Y).
  function profileRail() {
    let best = null;
    let bestY = -Infinity;
    for (const rail of (state.data && state.data.rails) || []) {
      const vertices = (rail && rail.mesh && rail.mesh.vertices) || [];
      if (!vertices.length) continue;
      let maxY = -Infinity;
      for (const p of vertices) {
        if (p && isNumber(p[1]) && p[1] > maxY) maxY = p[1];
      }
      if (maxY > bestY) { bestY = maxY; best = rail; }
    }
    return best;
  }

  // Фиксированная дистанция: профиль рельса должен читаться по форме.
  function viewProfile() {
    const rail = profileRail();
    const box = rail ? railBoundsOf(rail) : null;
    if (!box) {
      // Нет меша: если данных ещё нет, это не ошибка (первый кадр грузится ~2 с)
      // -- просто молчим, кампанию не портим красной плашкой.
      if (state.data && onError) {
        onError(new Error('нет меша рельса: «Профиль» показать нечем'));
      }
      return false;
    }
    const center = box.getCenter(new THREE.Vector3());
    // цель -- 1.5 м вглубь от ближнего конца сегмента, на видимом участке
    const y = Math.min(box.max.y - 0.5, Math.max(box.min.y + 0.5, box.max.y - 1.5));
    placeCamera(new THREE.Vector3(center.x, y, center.z), PROFILE_DISTANCE_M,
      profileDirection());
    return true;
  }

  // Подсветка вида: кнопка «Профиль рельса» в группе и пресет «Объекты пути» в
// ряду пресетов -- это один и тот же вид, поэтому отмечаем обе точки.
  function markViews(id) {
    const groupBtn = $('view-profile');
    if (groupBtn) groupBtn.classList.toggle('active', id === 'profile');
    for (const b of (presetsBox ? presetsBox.children : [])) {
      const isTrack = b.dataset && b.dataset.preset === TRACK_PRESET;
      if (isTrack) b.classList.toggle('active', id === 'objects');
      // Вид по объектам ставит слой -- прочие пресеты в ряду гаснут, иначе
      // подсвеченными окажутся две кнопки сразу.
      else if (id === 'objects') b.classList.remove('active');
    }
  }

  // Короткая строка «что камера сделала и почему» -- чтобы перемещение камеры на
  // старте не выглядело самовольным.
  function noteCamera(text) {
    const el = $('cameraNote');
    if (el) el.textContent = text;
  }

  // ------------------------------------------------------- чистое поле ------

  // «Чистое поле»: облако скрыто, нейтральный фон, камера вписана по объектам
  // (рельсы + шпалы, окно startViewSpan). Повторное нажатие возвращает облако
  // по тумблеру и прежнюю камеру.
  function setCleanField(on) {
    if (on === state.clean) return state.clean;
    state.clean = on;
    if (on) {
      state.saved = {
        position: camera.position.clone(),
        target: controls.target.clone(),
        background: scene.background ? scene.background.clone() : null,
      };
      scene.background = new THREE.Color(CLEAN_BG);
      if (!fitToTrack(startViewSpan) && state.data && onError) {
        onError(new Error('нет мешей пути: габарит для «чистого поля» не построить'));
      }
      markViews(null);
      noteCamera('«чистое поле»: облако скрыто, камера вписана по объектам пути');
    } else if (state.saved) {
      camera.position.copy(state.saved.position);
      controls.target.copy(state.saved.target);
      scene.background = state.saved.background;
      camera.updateProjectionMatrix();
      controls.update();
      state.saved = null;
      noteCamera('выход из «чистого поля»: прежняя камера и облако точек');
    }
    applyVisibility();
    const btn = $('clean');
    if (btn) {
      btn.classList.toggle('active', on);
      btn.textContent = on ? 'Выйти из «чистого поля»' : 'Чистое поле';
    }
    updateTrackHud();
    return state.clean;
  }

  // ------------------------------------------------------------- загрузка ----

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  async function loadOnce(bag, frame) {
    let data = null;
    try {
      const res = await fetch(
        `/track?bag=${encodeURIComponent(bag)}&frame=${frame}`, { cache: 'no-store' });
      if (!res.ok) {
        let detail = '';
        try { detail = (await res.json()).error || ''; } catch (e) { /* не JSON */ }
        if (res.status === 503) {
          // track_geometry нет -- это не ошибка страницы: облако работает, а в
          // компактном HUD видно, почему объектов пути нет.
          state.unavailable = detail || 'track_geometry недоступен';
          state.data = null;
          console.warn('GET /track -> 503:', state.unavailable);
          return false;
        }
        throw new Error(`GET /track -> ${res.status}${detail ? ': ' + detail : ''}`);
      }
      data = await res.json();
    } catch (e) {
      state.data = null;
      state.unavailable = String(e && e.message ? e.message : e);
      if (onError) onError(e);
      return false;
    }
    state.data = data;
    state.bag = bag;
    state.frame = frame;
    state.unavailable = null;
    return true;
  }

  // Протяжка слайдера: пока строится кадр, новые запросы только заменяют
  // «последний» -- пересчитывается ровно он (сервер тратит 1-2.6 с на кадр).
  async function drain() {
    while (request) {
      let { bag, frame } = request;
      request = null;
      const hasData = Boolean(state.data) && builtBag === bag;
      // Собираем запросы, пока они идут потоком (протяжка слайдера), но:
      //  * не дольше MAX_DEFER_MS -- воспроизведение идёт непрерывно (10 Гц),
      //    «тишины» можно не дождаться, и геометрия застыла бы навсегда;
      //  * и не откладываем вовсе, если геометрии ещё нет -- иначе страница,
      //    стартующая сразу с воспроизведения, не показала бы объекты пути.
      const deadline = lastBuildAt + MAX_DEFER_MS;
      while (hasData && request && Date.now() < deadline) {
        await sleep(DEBOUNCE_MS);
        if (request) {
          ({ bag, frame } = request);
          request = null;
        }
      }
      if (bag === builtBag && frame === builtFrame && state.data) continue;
      const ok = await loadOnce(bag, frame);
      if (ok) {
        builtBag = bag;
        builtFrame = frame;
        lastBuildAt = Date.now();
      }
      rebuildObjects();
      updateBadges();
      updateNotes();
      updateTrackHud();
      if (state.clean) fitToTrack(startViewSpan);
      // Стартовый вид по объектам ставим ОДИН раз: дальше камеру не трогаем ни на
      // смене кадра, ни на слайдере, ни в воспроизведении. И только если:
      //   * автовид включён галочкой (выключен -> остаётся пресет пользователя);
      //   * пользователь ещё не трогал камеру сам (иначе кадр «прыгнет» под руками).
      if (!startViewApplied && ok && state.autoView && !userMovedCamera) {
        startViewApplied = true;
        fitToTrack(startViewSpan);
        markViews('objects');
        noteCamera(START_VIEW_NOTE);
      } else if (!startViewApplied && ok) {
        // Автовид выключен (галочка по умолчанию снята) или вид выбрал пользователь:
        // объекты пути камеру НЕ трогают. Подсказок не пишем: состояние видно по
        // самой галочке и по подсвеченному пресету -- лишнего текста в панели быть
        // не должно. Флаг ставим, чтобы не возвращаться к этому решению.
        startViewApplied = true;
      }
    }
  }

  function load(bag, frame) {
    request = { bag, frame: Number(frame) | 0 };
    if (!drainPromise) {
      drainPromise = drain().finally(() => { drainPromise = null; });
    }
    return drainPromise;
  }

  // ------------------------------------------------ предзагрузка геометрии ----

  // Сколько прогретых кадров помним: ключ 'bag:frame', порядок = LRU. Набор нужен
  // только чтобы не просить один и тот же кадр дважды, и держим его близко к
  // серверному кэшу геометрии (TRACK_CACHE_FRAMES = 5 в web_viewer.py): дольше
  // этого срока «прогрет» уже ничего не значит.
  const PREFETCH_KEEP = 8;
  // Сборка кадра на сервере идёт ПО ОДНОЙ (build_lock в TrackGeometry). Очередь из
  // предзагрузок отодвинула бы текущий кадр, поэтому в полёте не больше одной.
  const PREFETCH_MAX_INFLIGHT = 1;
  const prefetched = new Map();
  let prefetchInFlight = 0;

  // Прогреть геометрию кадра заранее. Зачем: метка зоны rail считается по оси
  // ИМЕННО своего кадра, и если геометрии кадра в кэше сервера нет, /frame считает
  // её в момент запроса -- кадр ждёт сборку. Эта предзагрузка (она идёт параллельно
  // с предзагрузкой самого кадра, см. prefetch в app.js) собирает геометрию
  // заранее, чтобы к показу кадра она уже была в кэше сервера, а не в браузере.
  // Ответ не разбираем: греть нужно серверный кэш, а не свой (JSON кадра ~1 МБ,
  // держать его в браузере незачем).
  // Возвращает промис прогрева; null -- греть нечего (уже прогрет или загружен)
  // или нельзя (лимит в полёте). Ошибка прогрева -- не ошибка страницы.
  function prefetch(bag, frame) {
    if (!bag) return null;
    const index = Number(frame) | 0;
    const key = `${bag}:${index}`;
    if (prefetched.has(key)) return null;
    if (state.data && state.bag === bag && builtFrame === index) return null;
    if (prefetchInFlight >= PREFETCH_MAX_INFLIGHT) return null;
    prefetched.set(key, true);
    while (prefetched.size > PREFETCH_KEEP) {
      prefetched.delete(prefetched.keys().next().value);
    }
    prefetchInFlight += 1;
    return fetch(`/track?bag=${encodeURIComponent(bag)}&frame=${index}`,
      { cache: 'no-store' })
      .then(() => true)
      .catch(() => { prefetched.delete(key); return false; })
      .finally(() => { prefetchInFlight -= 1; });
  }

  // ------------------------------------------------------------- публично ---

  updateBadges();
  updateNotes();
  updateTrackHud();
  applyVisibility();

  return {
    load,
    // Прогрев геометрии соседних кадров (idx+1, idx+2) -- зовёт app.js из своего
    // prefetch(); см. комментарий у функции.
    prefetch,
    setVisible(patch = {}) {
      for (const key of ['rails', 'sleepers', 'floor']) {
        if (key in patch) state.visible[key] = Boolean(patch[key]);
      }
      applyVisibility();
    },
    setCleanField,
    isCleanField: () => state.clean,
    // Автовид по объектам: только стартовое поведение. Выключение на ходу камеру
    // не двигает (ничего не «прыгает»), включение -- тоже: вид ставит кнопка.
    setAutoView(on) {
      state.autoView = Boolean(on);
      if (!state.autoView && !startViewApplied) return state.autoView;
      noteCamera(state.autoView
        ? 'автовид по объектам включён: вид поставится на следующей загрузке страницы'
        : 'автовид по объектам выключен: камеру ставит ваш пресет (вид — кнопкой «Объекты пути»)');
      return state.autoView;
    },
    isAutoView: () => state.autoView,
    fitToTrack(span = startViewSpan, { mark = true } = {}) {
      const distance = fitToTrack(span);
      if (mark) markViews('objects');
      if (mark && distance) noteCamera(START_VIEW_NOTE);
      return distance;
    },
    viewProfile() {
      const ok = viewProfile();
      markViews(ok ? 'profile' : null);
      if (ok) noteCamera(`камера: профиль рельса, ${PROFILE_DISTANCE_M} м`);
      return ok;
    },
    data: () => state.data,
    // Виден ли наш меш рельсов прямо сейчас: по этому признаку app.js гасит
    // «ленту рельсов» вьюера (оранжевые точки зоны rail) -- чтобы не было двух
    // моделей рельсов одновременно.
    railsVisible: () => Boolean(state.visible.rails && groups.rails),
    // Полный сброс данных слоя: при смене записи от прежней не должно остаться ни
    // мешей, ни плашек, ни подписей, ни выбранного вида. Сбрасываем и «решённость»
    // стартового вида: у новой записи он свой (галочка/профиль решают заново).
    reset() {
      state.data = null;
      state.bag = null;
      state.frame = 0;
      state.unavailable = null;
      state.saved = null;
      if (state.clean) {
        scene.background = state.saved ? state.saved.background : null;
        state.clean = false;
      }
      builtBag = null;
      builtFrame = null;
      lastBuildAt = 0;
      startViewApplied = false;
      userMovedCamera = false;
      request = null;
      prefetched.clear();     // у прежней записи прогрев больше ничего не значит
      dropGroup('floor');
      dropGroup('sleepers');
      dropGroup('rails');
      for (const id of ['src-rails', 'src-sleepers', 'src-floor']) setBadge(id, '—', '');
      const btn = $('clean');
      if (btn) {
        btn.classList.remove('active');
        btn.textContent = 'Чистое поле';
      }
      markViews(null);
      updateNotes();          // «Откуда числа (0)» и «данных нет» вместо старых чисел
      updateTrackHud();
      applyVisibility();
    },
    dispose() {
      controls.removeEventListener('start', onControlsStart);
      if (presetsBox) presetsBox.removeEventListener('click', onPresetClick);
      for (const key of ['floor', 'sleepers', 'rails']) dropGroup(key);
      for (const light of lights) scene.remove(light);
      lights.length = 0;
      request = null;
      state.data = null;
      if (state.clean) {
        scene.background = state.saved ? state.saved.background : null;
        state.clean = false;
      }
      state.saved = null;
    },
  };
}