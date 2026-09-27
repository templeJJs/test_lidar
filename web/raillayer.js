// Линии рельсов и оси из карты пути pavel/track_map.py для web_viewer.py.
//
// Модуль самодостаточный: createRailsLayer({ scene }) грузит маршрут /rails
// (web_viewer.py отдаёт ПРЕДРАССЧИТАННЫЕ файлы pavel/rails_web/<запись>/
// frame_NNNN.json -- их пишет `cd pavel && python export_rails_web.py
// <запись>`; нужны track_map.npz и trajectory.npz в записи) и рисует:
//
//   * левую и правую нити рельсов (красные полилинии),
//   * осевую линию пути (голубая).
//
// Это НЕ слой объектов пути track3d.js: тот рисует /track из корневого
// track_geometry.build_track (меши рельсов/шпал/пола), а здесь -- карта пути
// тракта pavel, сшитая из наблюдений всей записи (track_map.npz): она знает
// рельсы и там, где детекция кадра их не видит. Слои независимы, у каждого
// свой тумблер.
//
// Координаты: JSON уже в системе кадра лидара (x поперёк, y назад -- вперёд
// это -y, z вверх), а вьюер рисует облако ровно в этих осях (см. app.js),
// поэтому точки идут в геометрию КАК ЕСТЬ, без перевода (в отличие от
// guardlayer.js, где система guard fwd/lat/up).
//
// Ловушки -- те же, что в guardlayer.js: пересборка только по приходу кадра
// (старая геометрия диспоузится), загрузки сериализованы с «последний
// запрошенный выигрывает», 404 (нет track_map.npz у записи / кадр не
// посчитан) -- не ошибка, а пустой слой и подсказка в панели.

import * as THREE from 'three';

const ORDER = 8;                           // поверх облака точек (renderOrder 1)
const COL_RAIL = 0xff5a5a;                 // нити рельсов
const COL_CENTER = 0x59d8ff;               // осевая линия

const $ = (id) => document.getElementById(id);

function disposeObject(obj) {
  obj.traverse((node) => {
    if (node.geometry) node.geometry.dispose();
    const mat = node.material;
    if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
    else if (mat) mat.dispose();
  });
}

function polyline(points, color, opacity) {
  const flat = [];
  for (const p of points) flat.push(p[0], p[1], p[2]);
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(flat, 3));
  const line = new THREE.Line(geom, new THREE.LineBasicMaterial({
    color, transparent: true, opacity, depthWrite: false,
  }));
  line.renderOrder = ORDER;
  return line;
}

// Слой. Методы: load(bag, idx), setVisible(bool), isVisible(), reset(),
// dispose(), data().
export function createRailsLayer({ scene } = {}) {
  if (!scene) throw new Error('createRailsLayer: нужен scene');

  const state = {
    visible: false,
    frame: null,
    bag: null,
    idx: -1,
    done: false,         // позиция обработана (200 или 404): иначе 404
                         // зацикливал бы загрузку на одном кадре
  };
  let group = null;
  let inflight = null;
  let want = null;
  let note404 = '';

  function drop() {
    if (!group) return;
    scene.remove(group);
    disposeObject(group);
    group = null;
  }

  function build(fr) {
    drop();
    group = new THREE.Group();
    group.name = 'rails-map';
    if ((fr.left || []).length > 1) group.add(polyline(fr.left, COL_RAIL, 0.95));
    if ((fr.right || []).length > 1) group.add(polyline(fr.right, COL_RAIL, 0.95));
    if ((fr.center || []).length > 1) group.add(polyline(fr.center, COL_CENTER, 0.6));
    group.visible = state.visible;
    scene.add(group);
  }

  function updateSummary() {
    const el = $('rails2Summary');
    if (!el) return;
    const fr = state.frame;
    if (!fr) {
      el.textContent = note404 || 'кадр не загружен';
      return;
    }
    el.textContent = `кадр ${fr.frame}: узлов ${fr.n_nodes}`
      + (fr.gauge_med ? `, колея ${(fr.gauge_med * 1000).toFixed(0)} мм` : '')
      + ' (карта track_map.npz)';
    const note = $('rails2Note');
    if (note) note.textContent = '';
  }

  async function fetchFrame(bag, idx) {
    const url = `/rails?bag=${encodeURIComponent(bag)}&frame=${idx}`;
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
  // старого кадра не перезапишет новый (та же гонка, что в /objects).
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
            if (want === cur || (want.bag === b && want.idx === i)) return;
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
          } catch (e) {
            if (e && e.status === 404) {
              // Запись без track_map.npz / кадр не посчитан: слой пуст,
              // подсказка -- в панели.
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
      return state.visible;
    },
    isVisible: () => state.visible,
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
  if (typeof window !== 'undefined') window.__railsLayer = api;
  return api;
}
