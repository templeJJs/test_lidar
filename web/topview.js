// Панель «вид сверху» (2D) для основного вьюера web_viewer.py.
//
// Модуль самодостаточный, по образцу guardlayer.js: createTopView({ cloud })
// рисует на своём <canvas id="topview"> (блок #topviewPanel в index.html)
// плоскую проекцию сцены сверху:
//   * точки кадра -- из posArr/colArr вьюера (accessor cloud(), тот же кадр,
//     что на 3D-экране), повторный fetch /frame НЕ делается; десятки тысяч
//     точек прореживаются до ~6k (decimateStep);
//   * коридор габарита, стены, ось пути и препятствия -- из /guard (тот же
//     JSON и та же сериализованная загрузка «последний запрошенный выигрывает»,
//     что в guardlayer.js); 404 (кадр не предрасчитан) -- не ошибка: рисуются
//     только точки.
//   * ТОЧКИ препятствий (obstacles[i].point_idx из /guard) -- ярко-красные
//     (HIT_RGB из hitpoints.js), 2x2 px, поверх облака и без прореживания.
//     Старые JSON без point_idx -- только боксы, как раньше.
//   * у каждого бокса препятствия -- маленькая подпись дальности «N м» тем же
//     цветом, что бокс (красный в габарите, оранжевый вне).
//   * ось пути красится по режиму ведения бина (bins.mode, блок 1в): rails --
//     синий, center -- зелёный, lead -- оранжевый, hold -- серый. Старый JSON
//     без mode -- ось одним COL_AXIS, как раньше. Тумблер в панели guard.
//
// Координаты: точки кадра лежат в осях лидара (X поперёк, Y вдоль, вперёд это
// -Y, см. app.js), guard JSON -- в системе guard (fwd = -y_lidar, lat = x_lidar,
// см. guardlayer.js). Поэтому для точек: lat = x, fwd = -y.
//
// Экранная рамка фиксированная (не зависит от кадра): поперёк (lat) ±10 м,
// вперёд (fwd) 0..130 м. +lat на экране ВЛЕВО -- как в 3D-виде и в разрезе X-Z
// (там camera.up = Z, взгляд вдоль -Y, поэтому экранное «вправо» -- это -X),
// иначе плоский вид читался бы зеркально относительно остальных панелей.

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
const $ = (id) => document.getElementById(id);

import { hitIndices, hitMarks, HIT_RGB } from './hitpoints.js';

export const TV = {
  W: 240,            // ширина canvas, CSS px
  H: 520,            // высота canvas, CSS px
  LAT_HALF: 10,      // полуширина рамки по поперёку, м
  FWD_MAX: 130,      // дальность рамки вперёд, м
  TARGET: 6000,      // сколько точек максимум рисуем (децимация)
  PAD: { l: 30, r: 8, t: 18, b: 16 },
  OBST_DEPTH_M: 2.0, // глубина бокса препятствия (как OBST_DEPTH_M в guardlayer)
};

// Цвета -- те же, что в guardlayer.js, чтобы 2D и 3D говорили одно и то же.
const COL_OK = '#35d07f';        // бин измерен с двух сторон
const COL_GAUGE = '#ffd166';     // измерена одна сторона
const COL_HOLE = '#8a97b0';      // дыра, коридор по предсказанию
const COL_WALL = '#8fc4ff';      // измеренные кромки стен
const COL_AXIS = '#59d8ff';      // ведомая ось бинов
const COL_AXIS_FAR = '#d98cff';  // далёкая ось
const COL_OBST = '#ff4d4d';      // препятствие в габарите
const COL_OBST_OUT = '#ffa04d';  // препятствие вне габарита
const COL_GRID = 'rgba(120,140,170,0.16)';
const COL_TEXT = 'rgba(138,151,176,0.9)';
const BG_ABGR = 0xff180f0c;      // фон панели (ABGR, как у разреза X-Z)

// Режимы ведения оси (блок 1в, wall_rules) -- те же цвета, что MODE_COLORS
// в guardlayer.js, чтобы 2D и 3D говорили одно и то же.
export const MODE_COLORS = {
  rails: '#4d8dff',    // ось от рельсов -- синий
  center: '#35d07f',   // середина двух стен -- зелёный
  lead: '#ffa04d',     // ведение по стене -- оранжевый
  hold: '#8a97b0',     // продолжение формы в дыре -- серый
};

/** Прямоугольник рисунка внутри canvas (CSS px): поля под подписи шкал. */
export function plotGeom(w = TV.W, h = TV.H) {
  const p = TV.PAD;
  return { x: p.l, y: p.t, w: w - p.l - p.r, h: h - p.t - p.b };
}

/**
 * Проекция guard-точки (lat, fwd) в CSS-пиксели панели.
 * lat +10 м -> левый край, -10 м -> правый (см. заголовок про зеркалирование);
 * fwd 0 м -> нижний край (сенсор), TV.FWD_MAX м -> верхний.
 */
export function projectPoint(lat, fwd, g) {
  return {
    x: g.x + g.w / 2 - lat * (g.w / 2 / TV.LAT_HALF),
    y: g.y + g.h - fwd * (g.h / TV.FWD_MAX),
  };
}

/** Шаг прореживания: из count точек рисуем примерно target. */
export function decimateStep(count, target = TV.TARGET) {
  if (!(count > 0) || !(target > 0)) return 1;
  return Math.max(1, Math.floor(count / target));
}

function polyline(ctx, pts) {
  if (!pts.length) return;
  ctx.beginPath();
  pts.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
  ctx.stroke();
}

/**
 * Полная отрисовка панели в контекст ctx. Вынесена из фабрики и не читает DOM,
 * чтобы node-тест мог гнать её на моке контекста.
 *   opts: { geom, dpr, w, h, points: {positions, colors, hidden, count}|null,
 *           frame: guard-payload|null, modeColors: bool (по умолчанию вкл) }
 */
export function renderTopView(ctx, opts) {
  const g = opts.geom;
  const dpr = opts.dpr || 1;
  const w = opts.w || TV.W;
  const h = opts.h || TV.H;
  const iw = Math.round(w * dpr);
  const ih = Math.round(h * dpr);

  // 1) фон + точки кадра пикселями (ImageData), без строк на каждую точку.
  const img = ctx.createImageData(iw, ih);
  const buf32 = new Uint32Array(img.data.buffer);
  buf32.fill(BG_ABGR);
  const pts = opts.points;
  const fr0 = opts.frame;
  // Точки препятствий: красим после облака, поверх и без децимации. Только
  // если guard-кадр -- про ЭТОТ кадр облака (idx совпал; старый оверлей при
  // прокрутке не должен красить чужие точки). Совпадение системы индексов
  // с /frame проверяет hitIndices по frame_npts.
  const marks = (pts && pts.positions && fr0
                 && (typeof pts.idx !== 'number' || fr0.frame === pts.idx))
    ? hitMarks(pts.count, hitIndices(fr0, pts.count)) : null;
  if (pts && pts.positions && pts.count > 0) {
    const step = decimateStep(pts.count);
    for (let i = 0; i < pts.count; i += step) {
      if (pts.hidden && pts.hidden[i]) continue;
      if (marks && marks[i]) continue;   // точки находок -- ниже, красным
      const j = i * 3;
      const lat = pts.positions[j];
      const fwd = -pts.positions[j + 1];       // вперёд это -Y лидара
      if (fwd < 0 || fwd > TV.FWD_MAX || lat < -TV.LAT_HALF || lat > TV.LAT_HALF) continue;
      const p = projectPoint(lat, fwd, g);
      const px = Math.round(p.x * dpr);
      const py = Math.round(p.y * dpr);
      if (px < 0 || py < 0 || px >= iw || py >= ih) continue;
      const c = pts.colors;
      buf32[py * iw + px] = 0xff000000
        | ((c ? c[j + 2] : 200) << 16) | ((c ? c[j + 1] : 210) << 8) | (c ? c[j] : 230);
    }
    if (marks) {
      const red = 0xff000000 | (HIT_RGB[2] << 16) | (HIT_RGB[1] << 8) | HIT_RGB[0];
      for (let i = 0; i < pts.count; i++) {
        if (!marks[i]) continue;
        if (pts.hidden && pts.hidden[i]) continue;
        const j = i * 3;
        const lat = pts.positions[j];
        const fwd = -pts.positions[j + 1];
        if (fwd < 0 || fwd > TV.FWD_MAX || lat < -TV.LAT_HALF || lat > TV.LAT_HALF) continue;
        const p = projectPoint(lat, fwd, g);
        const px = Math.round(p.x * dpr);
        const py = Math.round(p.y * dpr);
        // 2x2 device-пикселя: точка находки должна читаться над одиночными
        // пикселями прореженного облака.
        for (let dy = 0; dy < 2; dy++) {
          for (let dx = 0; dx < 2; dx++) {
            const x = px + dx;
            const y = py + dy;
            if (x < 0 || y < 0 || x >= iw || y >= ih) continue;
            buf32[y * iw + x] = red;
          }
        }
      }
    }
  }
  ctx.putImageData(img, 0, 0);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  // 2) сетка и подписи шкал: поперёк шаг 5 м, вперёд шаг 10 м.
  ctx.lineWidth = 1;
  ctx.strokeStyle = COL_GRID;
  ctx.beginPath();
  for (let lat = -TV.LAT_HALF; lat <= TV.LAT_HALF; lat += 5) {
    const px = Math.round(projectPoint(lat, 0, g).x) + 0.5;
    ctx.moveTo(px, g.y); ctx.lineTo(px, g.y + g.h);
  }
  for (let fwd = 0; fwd <= TV.FWD_MAX; fwd += 10) {
    const py = Math.round(projectPoint(0, fwd, g).y) + 0.5;
    ctx.moveTo(g.x, py); ctx.lineTo(g.x + g.w, py);
  }
  ctx.stroke();
  ctx.fillStyle = COL_TEXT;
  ctx.font = '9px "Segoe UI", system-ui, sans-serif';
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (let fwd = 0; fwd <= TV.FWD_MAX; fwd += 10) {
    ctx.fillText(String(fwd), g.x - 3, projectPoint(0, fwd, g).y);
  }
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (let lat = -TV.LAT_HALF; lat <= TV.LAT_HALF; lat += 5) {
    ctx.fillText(String(lat), projectPoint(lat, 0, g).x, g.y + g.h + 3);
  }

  // 3) оверлеи guard-кадра, обрезанные рамкой рисунка.
  const fr = opts.frame;
  ctx.save();
  ctx.beginPath();
  ctx.rect(g.x, g.y, g.w, g.h);
  ctx.clip();

  if (fr && fr.bins) {
    const bins = fr.bins;
    const fm = bins.fwd_mid || [];
    const status = bins.status || [];
    const axis = bins.axis || [];
    const lead = bins.lead || [];
    const hl = bins.half_left || [];
    const hr = bins.half_right || [];
    const binColor = (i) => (status[i] === 'ok' ? COL_OK
      : status[i] === 'gauge' ? COL_GAUGE : COL_HOLE);

    // Коридор по бинам: сечение поперёк + продольные кромки, цвет по статусу
    // (зелёный ok / жёлтый gauge / серый hole -- как в guardlayer.js).
    ctx.lineWidth = 1.5;
    let prev = null;
    for (let i = 0; i < fm.length; i++) {
      if (status[i] === 'stop' || !isNum(hl[i]) || !isNum(hr[i])) { prev = null; continue; }
      const ax = isNum(axis[i]) ? axis[i] : 0;
      const l = projectPoint(ax - hl[i], fm[i], g);
      const r = projectPoint(ax + hr[i], fm[i], g);
      ctx.strokeStyle = binColor(i);
      ctx.globalAlpha = 0.8;
      polyline(ctx, [l, r]);
      if (prev && i - prev.i === 1) {
        ctx.globalAlpha = 0.5;
        polyline(ctx, [prev.l, l]);
        polyline(ctx, [prev.r, r]);
      }
      prev = { i, l, r };
    }
    ctx.globalAlpha = 1;

    // Стены: кромка бина абсолютно -- axis + left/right; рисуем только
    // измеренные стороны (в 'gauge' кламп габарита стеной не является).
    ctx.strokeStyle = COL_WALL;
    ctx.lineWidth = 1.25;
    for (const side of [-1, 1]) {
      const edges = side < 0 ? (bins.left || []) : (bins.right || []);
      let line = [];
      for (let i = 0; i < fm.length; i++) {
        const measured = status[i] === 'ok'
          || (status[i] === 'gauge'
              && (lead[i] === (side < 0 ? 'left' : 'right') || lead[i] === 'both'));
        if (!measured || !isNum(edges[i]) || !isNum(axis[i])) {
          polyline(ctx, line);
          line = [];
          continue;
        }
        line.push(projectPoint(axis[i] + edges[i], fm[i], g));
      }
      polyline(ctx, line);
    }

    // Ось пути: ведомая линия бинов + далёкая ось, где она ведёт. Цвет
    // сегмента оси -- режим бина (bins.mode); старый JSON без mode -- одним
    // COL_AXIS, как раньше.
    const modes = bins.mode || [];
    const modeOn = opts.modeColors !== false;
    ctx.lineWidth = 1.25;
    let prevA = null;
    for (let i = 0; i < fm.length; i++) {
      if (status[i] === 'stop' || !isNum(axis[i])) { prevA = null; continue; }
      const p = projectPoint(axis[i], fm[i], g);
      if (prevA && i === prevA.i + 1) {
        ctx.strokeStyle = (modeOn && MODE_COLORS[modes[i]]) || COL_AXIS;
        polyline(ctx, [prevA.p, p]);
      }
      prevA = { i, p };
    }
    const far = fr.axis_far;
    if (far && Array.isArray(far.fwd_mid) && Array.isArray(far.lat)) {
      ctx.strokeStyle = COL_AXIS_FAR;
      polyline(ctx, far.fwd_mid.map((f, i) => ({ f, i }))
        .filter(({ i }) => ['seed', 'straight', 'wall'].includes((far.status || [])[i])
          && isNum(far.lat[i]))
        .map(({ f, i }) => projectPoint(far.lat[i], f, g)));
    }

    // Препятствия: бокс (x_min..x_max, d ± глубина/2), красный в габарите,
    // оранжевый вне; неподтверждённый -- прозрачнее (как в guardlayer).
    for (const ob of fr.obstacles || []) {
      if (!isNum(ob.d) || !isNum(ob.x_min) || !isNum(ob.x_max)) continue;
      const a = projectPoint(ob.x_min, ob.d - TV.OBST_DEPTH_M / 2, g);
      const b = projectPoint(ob.x_max, ob.d + TV.OBST_DEPTH_M / 2, g);
      const rx = Math.min(a.x, b.x), ry = Math.min(a.y, b.y);
      const rw = Math.abs(a.x - b.x), rh = Math.abs(a.y - b.y);
      const col = ob.in_gauge ? COL_OBST : COL_OBST_OUT;
      ctx.globalAlpha = ob.confirmed === false ? 0.16 : 0.3;
      ctx.fillStyle = col;
      ctx.fillRect(rx, ry, rw, rh);
      ctx.globalAlpha = ob.confirmed === false ? 0.45 : 0.95;
      ctx.strokeStyle = col;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(rx, ry, rw, rh);
      // Подпись дальности у бокса: читается без перевода глаз на 3D-вид.
      ctx.globalAlpha = ob.confirmed === false ? 0.6 : 1;
      ctx.fillStyle = col;
      ctx.font = '9px "Segoe UI", system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(`${Math.round(ob.d)} м`, rx + rw / 2, ry - 2);
    }
    ctx.globalAlpha = 1;
  }

  // Сенсор: треугольник в начале рамки (смотрит вперёд, вверх картинки).
  const s0 = projectPoint(0, 0, g);
  ctx.fillStyle = '#e8eef8';
  ctx.beginPath();
  ctx.moveTo(s0.x, s0.y - 5);
  ctx.lineTo(s0.x - 4, s0.y + 3);
  ctx.lineTo(s0.x + 4, s0.y + 3);
  ctx.closePath();
  ctx.fill();

  ctx.restore();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
}

// Слой панели. Методы: load(bag, idx) -- кадр из /guard, draw(idx) -- перерисовка
// по текущему кадру вьюера (вызывается из цикла рендера, внутри защита от
// лишней перерисовки), setVisible(bool), reset(), dispose(), data(), isVisible().
export function createTopView({ cloud } = {}) {
  if (typeof cloud !== 'function') throw new Error('createTopView: нужен cloud()');
  const panel = $('topviewPanel');
  const canvas = $('topview');
  if (!panel || !canvas) throw new Error('в index.html нет блока #topviewPanel/#topview');
  const ctx = canvas.getContext('2d');
  const dpr = Math.min((typeof window !== 'undefined' && window.devicePixelRatio) || 1, 2);
  canvas.width = Math.round(TV.W * dpr);
  canvas.height = Math.round(TV.H * dpr);
  canvas.style.width = TV.W + 'px';
  canvas.style.height = TV.H + 'px';
  const geom = plotGeom();

  const state = {
    visible: true,       // по умолчанию панель включена (галочка в HTML checked)
    folded: false,
    modeColors: true,    // подсветка режимов ведения оси (bins.mode)
    frame: null,         // guard-payload последнего применённого кадра (null при 404)
    bag: null,
    idx: -1,
    done: false,
    seq: 0,              // бампается при смене guard-данных: draw() их подхватит
  };
  let inflight = null;
  let want = null;
  let note404 = '';
  let lastKey = '';

  function updateSummary() {
    const el = $('topviewSummary');
    if (!el) return;
    if (!state.visible) { el.textContent = 'выключен'; return; }
    const fr = state.frame;
    if (!fr) { el.textContent = note404 ? `guard: ${note404}` : 'guard-кадр не загружен'; return; }
    const obs = fr.obstacles || [];
    const nIn = obs.filter((o) => o.in_gauge).length;
    el.textContent = `guard: кадр ${fr.frame}, бинов ок ${fr.n_ok}`
      + (obs.length ? `, препятствий ${obs.length} (в габарите ${nIn})` : '');
  }

  function applyPanelState() {
    // Галочка и сворачивание прячут только картинку: шапка с чекбоксом и
    // кнопкой остаётся, иначе панель нельзя было бы вернуть.
    const show = state.visible && !state.folded;
    canvas.style.display = show ? '' : 'none';
    const sum = $('topviewSummary');
    if (sum) sum.style.display = show ? '' : 'none';
    const fold = $('topviewFold');
    if (fold) fold.textContent = state.folded ? '+' : '–';
    if (show) lastKey = '';          // при возврате картинки -- перерисовать
    updateSummary();
  }

  const foldBtn = $('topviewFold');
  if (foldBtn) {
    foldBtn.addEventListener('click', () => {
      state.folded = !state.folded;
      applyPanelState();
    });
  }

  async function fetchGuard(bag, idx) {
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

  // Та же механика, что в guardlayer.js: загрузки сериализованы и всегда
  // применяют ПОСЛЕДНИЙ запрошенный кадр; 404 -- пустой оверлей, не ошибка.
  function load(bag, idx) {
    want = { bag, idx };
    if (!state.visible) return Promise.resolve();
    if (inflight) return inflight;
    const run = (async () => {
      try {
        while (want) {
          const cur = want;
          const { bag: b, idx: i } = cur;
          if (state.bag === b && state.idx === i && state.done) {
            await Promise.resolve();   // уступка цикла, как в guardlayer.load
            if (want === cur || (want.bag === b && want.idx === i)) return;
            continue;
          }
          try {
            const payload = await fetchGuard(b, i);
            if (want !== cur && want && (want.bag !== b || want.idx !== i)) continue;
            state.bag = b;
            state.idx = i;
            state.done = true;
            state.frame = payload;
            note404 = '';
            state.seq += 1;
          } catch (e) {
            if (e && e.status === 404) {
              if (want !== cur && want && (want.bag !== b || want.idx !== i)) continue;
              state.bag = b;
              state.idx = i;
              state.done = true;
              state.frame = null;
              note404 = e.detail || 'нет предрасчёта';
              state.seq += 1;
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

  // Перерисовка из цикла рендера app.js: ничего не делает, пока ни кадр
  // вьюера, ни guard-данные не изменились (как drawSection по sectionLastIdx).
  function draw(idx) {
    if (!state.visible || state.folded) return;
    const key = `${state.bag}|${idx}|${state.seq}`;
    if (key === lastKey) return;
    lastKey = key;
    renderTopView(ctx, { geom, dpr, w: TV.W, h: TV.H, points: cloud(),
                         frame: state.frame, modeColors: state.modeColors });
  }

  const api = {
    load,
    draw,
    data: () => state.frame,
    setVisible(on) {
      state.visible = Boolean(on);
      applyPanelState();
      return state.visible;
    },
    isVisible: () => state.visible,
    // Тумблер «подсветка режимов» (общий с guard-слоем): бамп seq, чтобы draw()
    // перерисовал панель с другой раскраской оси.
    setModeColors(on) {
      state.modeColors = on !== false;
      state.seq += 1;
      return state.modeColors;
    },
    // Смена записи: guard-данные прежней записи не показываем ни кадра.
    reset() {
      state.frame = null;
      state.bag = null;
      state.idx = -1;
      state.done = false;
      state.seq += 1;
      want = null;
      note404 = '';
      lastKey = '';
      updateSummary();
    },
    dispose() {
      state.frame = null;
      if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
    },
  };
  applyPanelState();
  if (typeof window !== 'undefined') window.__topView = api;   // отладка из консоли
  return api;
}
