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
//   * препятствия -- каркасные боксы находок блока 3: красные в габарите,
//     оранжевые вне его; неподтверждённые (бин-'hole') прозрачнее. Если в
//     JSON есть obstacles[i].point_idx (индексы точек находки в кадре /frame,
//     см. pavel/guard/export_web.py), сами ТОЧКИ препятствия красятся в
//     ярко-красный (HIT_RGB из hitpoints.js) поверх палитры облака; при смене
//     кадра/выключении слоя цвета восстанавливаются. Старые JSON без
//     point_idx -- только боксы, как раньше.
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
//   * пересборка группы -- только по приходу нового кадра, старая геометрия
//     и материал диспоузятся;
//   * загрузки сериализованы и всегда применяют ПОСЛЕДНИЙ запрошенный кадр:
//     ответ старого кадра не перезапишет новый (та же гонка, что в /objects);
//   * 404 (кадр не посчитан) -- не ошибка: слой пуст, в панели подсказка
//     команды предрасчёта.

import * as THREE from 'three';

import { hitIndices, paintCloud, restoreCloud } from './hitpoints.js';

const ORDER = 8;                           // поверх облака точек (renderOrder 1)
const COL_OK = 0x35d07f;                   // бин измерен с двух сторон
const COL_GAUGE = 0xffd166;                // измерена одна сторона
const COL_HOLE = 0x8a97b0;                 // дыра, коридор по предсказанию
const COL_WALL = 0x8fc4ff;                 // измеренные кромки стен
const COL_AXIS = 0x59d8ff;                 // ведомая ось бинов
const COL_AXIS_FAR = 0xd98cff;             // далёкая ось (ведение стеной)
const COL_OBST = 0xff4d4d;                 // препятствие в габарите
const COL_OBST_OUT = 0xffa04d;             // препятствие вне габарита
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
const OBST_DEPTH_M = 2.0;                  // глубина бокса препятствия, м

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
const $ = (id) => document.getElementById(id);

// guard (fwd, lat, up) -> вьюер (x, y, z). up здесь -- абсолютная z лидара.
const P = (fwd, lat, up) => [lat, -fwd, up];

function disposeObject(obj) {
  obj.traverse((node) => {
    if (node.geometry) node.geometry.dispose();
    const mat = node.material;
    if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
    else if (mat) mat.dispose();
  });
}

function lineSegments(pos, col, opacity) {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geom.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  const mesh = new THREE.LineSegments(geom, new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, opacity, depthWrite: false,
  }));
  mesh.renderOrder = ORDER;
  return mesh;
}

function polyline(points, color, opacity) {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(points.flat(), 3));
  const line = new THREE.Line(geom, new THREE.LineBasicMaterial({
    color, transparent: true, opacity, depthWrite: false,
  }));
  line.renderOrder = ORDER;
  return line;
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
    frame: null,         // payload последнего применённого кадра
    bag: null,
    idx: -1,
    done: false,         // позиция (bag, idx) обработана (200 или 404): без неё
                         // 404 зацикливал бы загрузку на одном кадре
  };
  let group = null;
  let painted = null;    // бэкап paintCloud для восстановления цветов облака
  let inflight = null;   // идущая загрузка (сериализация, см. заголовок)
  let want = null;       // последняя ЗАПРОШЕННАЯ позиция {bag, idx}
  let note404 = '';      // последний текст «нет предрасчёта» (не спамим панель)

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

  // Перекраска точек находок текущего guard-кадра. Кадр вьюера мог уйти
  // вперёд, пока летел ответ /guard: индексы чужого кадра не красим
  // (проверка по idx; дальше в hitIndices стоит проверка по frame_npts).
  function paintPoints(fr) {
    restorePoints();
    if (!fr || typeof cloud !== 'function') return;
    const c = cloud();
    if (!c || !c.colors) return;
    if (typeof c.idx === 'number' && typeof fr.frame === 'number'
        && c.idx !== fr.frame) return;
    painted = paintCloud(c.colors, hitIndices(fr, c.count));
    if (painted) {
      painted.frameIdx = c.idx;
      if (typeof touchCloud === 'function') touchCloud();
    }
  }

  function drop() {
    restorePoints();
    if (!group) return;
    scene.remove(group);
    disposeObject(group);
    group = null;
  }

  function build(fr) {
    drop();
    const bins = fr.bins || {};
    const fm = bins.fwd_mid || [];
    const status = bins.status || [];
    const lead = bins.lead || [];
    if (!fm.length || !isNum(fr.floor)) return;
    const floor = fr.floor;
    const step = fm.length > 1 ? fm[1] - fm[0] : 5.0;

    const cOk = new THREE.Color(COL_OK).toArray();
    const cGauge = new THREE.Color(COL_GAUGE).toArray();
    const cHole = new THREE.Color(COL_HOLE).toArray();
    const cWall = new THREE.Color(COL_WALL).toArray();
    const cAxis = new THREE.Color(COL_AXIS).toArray();
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

    group = new THREE.Group();
    group.name = 'guard-corridor';

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
      if (pos.length) group.add(lineSegments(pos, col, 0.55));
    }

    // ---- стены: продольная линия по кромке + вертикали, только измеренные ----
    // В бине 'ok' измерены обе стороны, в 'gauge' -- только ведущая (lead);
    // кламп габарита стеной не является, и рисовать его как стену нельзя.
    {
      const pos = [];
      const col = [];
      const push = (p) => {
        pos.push(p[0], p[1], p[2]);
        col.push(cWall[0], cWall[1], cWall[2]);
      };
      for (const side of [-1, 1]) {
        const linePts = [];
        for (let i = 0; i < fm.length; i++) {
          const measured = status[i] === 'ok'
            || (status[i] === 'gauge'
                && (lead[i] === (side < 0 ? 'left' : 'right') || lead[i] === 'both'));
          const edge = side < 0 ? bins.left[i] : bins.right[i];
          if (!measured || !isNum(edge) || !isNum(bins.axis[i])) {
            if (linePts.length > 1) group.add(polyline(linePts, COL_WALL, 0.85));
            linePts.length = 0;
            continue;
          }
          const lat = bins.axis[i] + edge;
          linePts.push(P(fm[i], lat, floor + WALL_Z_HI));
          // Вертикаль от пола до верха стены -- читается как «стена», а не нить.
          push(P(fm[i], lat, floor));
          push(P(fm[i], lat, floor + WALL_Z_HI));
        }
        if (linePts.length > 1) group.add(polyline(linePts, COL_WALL, 0.85));
      }
      if (pos.length) group.add(lineSegments(pos, col, 0.35));
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
      if (pos.length) group.add(lineSegments(pos, col, 0.95));
      const far = fr.axis_far;
      if (far && Array.isArray(far.fwd_mid) && Array.isArray(far.lat)) {
        const ptsFar = [];
        for (let i = 0; i < far.fwd_mid.length; i++) {
          const st = (far.status || [])[i];
          if (st !== 'seed' && st !== 'straight' && st !== 'wall') continue;
          if (!isNum(far.lat[i])) continue;
          ptsFar.push(P(far.fwd_mid[i], far.lat[i], floor + 0.25));
        }
        if (ptsFar.length > 1) group.add(polyline(ptsFar, COL_AXIS_FAR, 0.9));
      }
    }

    // ---- препятствия: каркасный бокс находки (красный в габарите) ----
    {
      const pos = [];
      const col = [];
      const push = (p, c) => { pos.push(p[0], p[1], p[2]); col.push(c[0], c[1], c[2]); };
      for (const h of fr.obstacles || []) {
        if (!isNum(h.d) || !isNum(h.x_min) || !isNum(h.x_max) || !isNum(h.height)) continue;
        const c = new THREE.Color(h.in_gauge ? COL_OBST : COL_OBST_OUT).toArray();
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
      }
      if (pos.length) {
        const mesh = lineSegments(pos, col, 0.9);
        group.add(mesh);
      }
    }

    group.visible = state.visible;
    scene.add(group);
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
    el.textContent = `кадр ${fr.frame}: подтверждено до ${Math.round(fr.reach)} м`
      + ` (с мостами ${Math.round(fr.reach_bridged)} м), бинов ок ${fr.n_ok}`
      + (nAll ? `, препятствий ${nAll} (в габарите ${nIn})` : ', препятствий нет');
    const note = $('guardNote');
    if (note) note.textContent = '';
  }

  async function fetchFrame(bag, idx) {
    const url = `/guard?bag=${encodeURIComponent(bag)}&frame=${idx}`;
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
            const payload = await fetchFrame(b, i);
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
      }
    })();
    inflight = run;
    return run;
  }

  const api = {
    load,
    data: () => state.frame,
    setVisible(on) {
      state.visible = Boolean(on);
      if (group) group.visible = state.visible;
      if (!state.visible) {
        restorePoints();
      } else if (state.frame) {
        // load() на уже применённом кадре выходит по быстрому пути без
        // build -- подсветку точек возвращаем сами.
        paintPoints(state.frame);
      }
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
    },
    dispose() {
      drop();
      state.frame = null;
    },
  };
  // Отладка из консоли страницы (тот же приём, что window.__objectsLayer).
  if (typeof window !== 'undefined') window.__guardLayer = api;
  return api;
}
