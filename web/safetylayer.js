// Коридор безопасности для основного вьюера web_viewer.py.
//
// Модуль самодостаточный: createSafetyLayer({ scene }) рисует по блоку `tunnel`
// из /meta (его считает zones.tunnel_blocked):
//
//   * каркас коридора -- LineSegments: рёбра сечений каждые 2 м плюс продольные
//     рёбра между сечениями; занятые бины рисуются красным;
//   * подсветка точек внутри объёма -- markPoints() вызывается ВЛАДЕЛЬЦЕМ после
//     applyFrame: цвета кадра к этому моменту только что перезаписаны.
//
// Никаких линий хода и наконечников здесь нет: объём заметается вдоль измеренной
// оси рельсов (track_geometry.build_track) на стороне сервера, и клиенту нужен
// только он. Варианты «коридор / рельсы / ось» из панели убраны как лишние.
//
// Оси данных: X поперёк, Y вдоль (вперёд -- -Y), Z вверх. Высоты коридора заданы
// НАД УГР: z_low/z_high -- метры над уровнем головки рельса, поэтому абсолютная
// z = floor(y) + rail_height + z_*. Линия, вокруг которой строится сечение, лежит
// на z головки рельса, то есть низ объёма отстоит от неё на z_low.
//
// Ловушки, уже пройденные в track3d.js, здесь учтены так же:
//   * THREE.Object3D.DEFAULT_UP (Z вверх) ставит web/app.js; слой создаётся
//     в main() после этого и своего «верха» не навязывает;
//   * пересборка группы -- только по update(), не в цикле анимации; старая
//     геометрия и материал при этом диспоузятся;
//   * markPoints считается по разреженной таблице x(y) (шаг 5 см): интерполяция
//     по 46 узлам на каждую из 340 тыс. точек кадра была бы дороже самого кадра.

import * as THREE from 'three';

export const TUNNEL_COLOR = 0x8fc4ff;
export const TUNNEL_BLOCKED_COLOR = 0xff5a5a;

const ORDER = 8;                           // поверх облака точек (у него renderOrder 1)
const TUNNEL_OPACITY = 0.45;
const SECTION_STEP_M = 2.0;                // рёбра сечений каркаса, м
const STEP_FALLBACK_M = 1.0;               // если шаг бинов не выводится из данных
const CONTACT_HALF = 0.15;                 // полоса контактного рельса, м (zones.py)
const CONTACT_BAND = 0.20;                 // +-м вокруг его высоты (zones.py)
const MARK_COLOR = [255, 48, 48];          // точки внутри коридора -- красным
const LOOKUP_STEP_M = 0.05;                // шаг таблицы x(y) для подсветки точек

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
const $ = (id) => document.getElementById(id);

function disposeObject(obj) {
  obj.traverse((node) => {
    if (node.geometry) node.geometry.dispose();
    const mat = node.material;
    if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
    else if (mat) mat.dispose();
  });
}

// Длина ребра бина по массиву узлов (медиана разностей: последний бин бывает
// короче остальных, и среднее по нему «съезжает»).
function binStep(values) {
  const diffs = [];
  for (let i = 1; i < values.length; i++) {
    const d = values[i] - values[i - 1];
    if (isNum(d) && d > 0) diffs.push(d);
  }
  if (!diffs.length) return STEP_FALLBACK_M;
  diffs.sort((a, b) => a - b);
  return diffs[Math.floor(diffs.length / 2)];
}

// Слой коридора безопасности. Методы: update(meta) -- блок `tunnel` из /meta
// (или из ответа /set), setVisible(bool), markPoints(posArr, colArr, count),
// reset(), dispose().
export function createSafetyLayer({ scene, THREE: three = null } = {}) {
  if (!scene) throw new Error('createSafetyLayer: нужен scene');
  const T = three || THREE;

  const state = {
    visible: false,      // по умолчанию каркас не рисуется: панель спокойна
    tunnel: null, floor: null, rails: null, axis: 0,
    mark: null,
  };
  let group = null;
  let marked = 0;

  function drop() {
    if (!group) return;
    scene.remove(group);
    disposeObject(group);
    group = null;
  }

  // Абсолютная z плоскости пола; коридор задан над УГР, то есть над полом.
  function floorZ(y) {
    const f = state.floor;
    if (!f || !isNum(f.a) || !isNum(f.b)) return 0;
    return f.a + f.b * y;
  }

  function buildTunnel() {
    drop();
    const t = state.tunnel;
    if (!t || !Array.isArray(t.y) || t.y.length < 2) return;
    const y = t.y;
    const half = t.half_width;
    const zLow = t.z_low;
    const zHigh = t.z_high;
    const railH = isNum(t.rail_height) ? t.rail_height : 0.16;
    if (!isNum(half) || half <= 0) return;
    if (!isNum(zLow) || !isNum(zHigh) || zHigh <= zLow) return;
    const line = (Array.isArray(t.x) && t.x.length === y.length) ? t.x : null;
    const xAt = (i) => (line ? line[i] : state.axis);
    const blocked = Array.isArray(t.blocked) ? t.blocked : [];
    const normal = new T.Color(TUNNEL_COLOR).toArray();
    const bad = new T.Color(TUNNEL_BLOCKED_COLOR).toArray();

    // Сечения -- каждые SECTION_STEP_M метров; последний узел добавляем всегда,
    // иначе каркас обрывается раньше объёма.
    const step = binStep(y);
    const every = Math.max(1, Math.round(SECTION_STEP_M / (step || STEP_FALLBACK_M)));
    const idx = [];
    for (let i = 0; i < y.length; i += every) idx.push(i);
    if (idx[idx.length - 1] !== y.length - 1) idx.push(y.length - 1);

    const pos = [];
    const col = [];
    const push = (p, c) => { pos.push(p[0], p[1], p[2]); col.push(c[0], c[1], c[2]); };
    // Угол сечения: -1 / +1 -- сторона от оси, top -- верх или низ объёма.
    const corner = (i, side, top) => [
      xAt(i) + side * half, y[i], floorZ(y[i]) + railH + (top ? zHigh : zLow),
    ];
    const ring = (i) => [
      corner(i, -1, false), corner(i, 1, false), corner(i, 1, true), corner(i, -1, true),
    ];

    for (const i of idx) {
      const c = blocked[i] ? bad : normal;
      const r = ring(i);
      for (let k = 0; k < 4; k++) { push(r[k], c); push(r[(k + 1) % 4], c); }
    }
    // Продольные рёбра: занятость берём по любому из двух соседних бинов.
    for (let s = 0; s + 1 < idx.length; s++) {
      const i = idx[s], j = idx[s + 1];
      const c = (blocked[i] || blocked[j]) ? bad : normal;
      for (const [side, top] of [[-1, false], [1, false], [1, true], [-1, true]]) {
        push(corner(i, side, top), c);
        push(corner(j, side, top), c);
      }
    }

    const geom = new T.BufferGeometry();
    geom.setAttribute('position', new T.Float32BufferAttribute(pos, 3));
    geom.setAttribute('color', new T.Float32BufferAttribute(col, 3));
    group = new T.Group();
    group.name = 'safety-tunnel';
    const mesh = new T.LineSegments(geom, new T.LineBasicMaterial({
      vertexColors: true, transparent: true, opacity: TUNNEL_OPACITY, depthWrite: false,
    }));
    mesh.renderOrder = ORDER;
    group.add(mesh);
    group.visible = state.visible;
    scene.add(group);
  }

  // ------------------------------------------------------- подсветка точек ----

  // Таблица x(y) по узлам коридора: подсветка идёт на каждый кадр (340 тыс. точек),
  // и поиск места в 46 узлах на точку заметно дороже самой проверки.
  function buildMark() {
    const t = state.tunnel;
    const floor = state.floor;
    if (!t || !Array.isArray(t.y) || t.y.length < 2) return null;
    if (!floor || !isNum(floor.a) || !isNum(floor.b)) return null;
    if (!isNum(t.half_width) || t.half_width <= 0) return null;
    if (!isNum(t.z_low) || !isNum(t.z_high) || t.z_high <= t.z_low) return null;
    const y = t.y;
    const line = (Array.isArray(t.x) && t.x.length === y.length) ? t.x : null;
    const yMin = y[0], yMax = y[y.length - 1];
    if (!isNum(yMin) || !isNum(yMax) || yMax <= yMin) return null;
    const n = Math.max(2, Math.ceil((yMax - yMin) / LOOKUP_STEP_M) + 1);
    const table = new Float32Array(n);
    if (line) {
      let k = 0;
      for (let i = 0; i < n; i++) {
        const v = Math.min(yMin + i * LOOKUP_STEP_M, yMax);
        while (k + 2 < y.length && y[k + 1] < v) k++;
        const span = y[k + 1] - y[k];
        const f = span > 1e-9 ? Math.min(Math.max((v - y[k]) / span, 0), 1) : 0;
        table[i] = line[k] + (line[k + 1] - line[k]) * f;
      }
    } else {
      table.fill(state.axis);
    }
    // Контактный рельс -- третья нить в meta.rails: сервер исключает его из
    // объёма, и подсветка не должна красить то, что помехой не считается.
    const rails = state.rails || {};
    const cx = Array.isArray(rails.x) ? rails.x[2] : null;
    const ch = Array.isArray(rails.heights) ? rails.heights[2] : null;
    const excludeContact = Boolean(t.exclude_contact) && isNum(cx) && isNum(ch);
    return {
      yMin, yMax, step: LOOKUP_STEP_M, n, table,
      half: t.half_width,
      zLow: t.z_low + t.rail_height,       // в zrel: над полом
      zHigh: t.z_high + t.rail_height,
      floorA: floor.a, floorB: floor.b,
      contactX: excludeContact ? cx : null, contactH: excludeContact ? ch : null,
    };
  }

  // Перекраска точек внутри объёма. Вызывается ПОСЛЕ applyFrame: цвета кадра
  // только что перезаписаны, красные точки ложатся поверх них. При скрытом
  // коридоре не красим ничего -- иначе красное висело бы без объяснения.
  function markPoints(posArr, colArr, count = 0) {
    marked = 0;
    const m = state.mark;
    if (!posArr || !colArr || !m || !state.visible) return 0;
    const inv = 1 / m.step;
    const last = Math.max(0, Math.min(count | 0, posArr.length / 3 | 0));
    for (let i = 0, j = 0; i < last; i++, j += 3) {
      const y = posArr[j + 1];
      if (y < m.yMin || y > m.yMax) continue;
      const x = posArr[j];
      const t = (y - m.yMin) * inv;
      let k = t | 0;
      if (k > m.n - 2) k = m.n - 2;
      const xline = m.table[k] + (m.table[k + 1] - m.table[k]) * (t - k);
      const dx = x - xline;
      if (dx > m.half || dx < -m.half) continue;
      const zrel = posArr[j + 2] - (m.floorA + m.floorB * y);
      if (zrel < m.zLow || zrel > m.zHigh) continue;
      if (m.contactX !== null
          && Math.abs(x - m.contactX) <= CONTACT_HALF
          && Math.abs(zrel - m.contactH) <= CONTACT_BAND) continue;
      colArr[j] = MARK_COLOR[0];
      colArr[j + 1] = MARK_COLOR[1];
      colArr[j + 2] = MARK_COLOR[2];
      marked++;
    }
    return marked;
  }

  // ------------------------------------------------------------------ панель --

  // Сводка «занято N из M бинов» -- из блока tunnel: числа считает сервер, клиент
  // только показывает их (и длину занятых участков по spans).
  function updateSummary() {
    const el = $('tunSummary');
    if (!el) return;
    const t = state.tunnel;
    if (!t || !isNum(t.bins_total)) {
      el.textContent = 'нет данных от сервера';
      return;
    }
    const spans = Array.isArray(t.spans) ? t.spans : [];
    const len = spans.reduce((acc, s) => (Array.isArray(s) ? acc + Math.abs(s[1] - s[0]) : acc), 0);
    el.textContent = `занято ${t.bins_blocked} из ${t.bins_total} бинов`
      + (len > 0 ? `, ${len.toFixed(1)} м` : '');
  }

  // Размеры габарита -- текстом, без ползунков: числа нормативные, но берём их из
  // meta.tunnel, чтобы подпись не разъезжалась с тем объёмом, что нарисован.
  function updateSizeNote() {
    const el = $('tunSize');
    if (!el) return;
    const t = state.tunnel;
    if (!t || !isNum(t.half_width) || !isNum(t.z_low) || !isNum(t.z_high)) return;
    const width = (t.half_width * 2).toFixed(2);
    const half = t.half_width.toFixed(2);
    el.textContent = `ширина ${width} м (±${half} м от оси), высота `
      + `${t.z_low.toFixed(2)}…${t.z_high.toFixed(2)} м над головкой рельса`
      + ' (габарит М, ГОСТ 23961-80); вдоль измеренной оси рельсов.';
  }

  // --------------------------------------------------------------- публично ---

  function update(nextMeta) {
    if (!nextMeta) return;
    state.tunnel = nextMeta.tunnel || null;
    state.floor = nextMeta.floor || null;
    state.rails = nextMeta.rails || null;
    state.axis = isNum(nextMeta.axis_x) ? nextMeta.axis_x : 0;
    state.mark = buildMark();
    buildTunnel();
    updateSummary();
    updateSizeNote();
  }

  function clear() {
    drop();
    state.tunnel = null;
    state.mark = null;
    marked = 0;
  }

  return {
    update,
    // bool -- тумблер «показывать коридор» в панели; объект {tunnel} принимаем
    // для совместимости со стилем track3d.js.
    setVisible(patch = {}) {
      state.visible = typeof patch === 'boolean'
        ? Boolean(patch)
        : ('tunnel' in patch ? Boolean(patch.tunnel) : state.visible);
      if (group) group.visible = state.visible;
      return state.visible;
    },
    isVisible: () => state.visible,
    markPoints,
    markedPoints: () => marked,
    // Смена записи: от прежней не должны остаться ни каркас, ни подсветка.
    reset() {
      clear();
      updateSummary();
      updateSizeNote();
    },
    dispose() {
      clear();
      state.floor = null;
      state.rails = null;
    },
  };
}