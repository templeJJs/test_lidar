import * as THREE from 'three';
import { OrbitControls } from './OrbitControls.js';

const $ = (id) => document.getElementById(id);
const errBox = $('err');

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
const state = {
  idx: 0, playing: true, loading: false, fps: 10, realtime: false, mode: 'zones',
  maxDist: 40, numPoints: 0, fpsShown: 0, lastFrameMs: 0, nextTick: 0, counts: null,
};

const FRAME_MARK = new Uint8Array([0x4c, 0x5a, 0x46, 0x52]); // 'LZFR'

// Разрез X-Z в правом нижнем углу: рисуем на 2D-canvas из того же кадра
// (posArr/colArr), без дополнительных запросов к серверу.
const SECTION_W = 380, SECTION_H = 240, SECTION_WINDOW = 12;
let sectionCanvas, sectionCtx, sectionDpr = 1, sectionLastIdx = -1;

async function loadMeta() {
  const res = await fetch('/meta');
  if (!res.ok) throw new Error('GET /meta -> ' + res.status);
  meta = await res.json();
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

function railQuery() {
  return new URLSearchParams({
    axis: $('axis').value,
    gauge: $('gauge').value,
    contact: $('contact').checked ? '1' : '0',
    offset: $('coff').value,
    side: $('cside').checked ? '1' : '-1',
  }).toString();
}

let railTimer = null;

function onRailChange() {
  $('axisVal').textContent = Number($('axis').value).toFixed(2);
  $('gaugeVal').textContent = $('gauge').value;
  $('coffVal').textContent = Number($('coff').value).toFixed(2);
  clearTimeout(railTimer);
  railTimer = setTimeout(applyRailParams, 140);   // не спамим сервер при протяжке
}

async function applyRailParams() {
  const r = await fetch('/set?' + railQuery(), { cache: 'no-store' });
  if (!r.ok) throw new Error('GET /set -> ' + r.status);
  const j = await r.json();
  meta.rails = j.rails;
  meta.floor = j.floor;
  meta.blocked_spans = j.blocked_spans;
  meta.axis_x = j.axis_x;
  meta.gauge_mm = j.gauge_mm;
  buildGrid();
  cache.clear();
  await showFrame(state.idx);       // перекрасить 'rail' по новой геометрии
  updateHud();
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
  meta.zone_names.forEach((name, i) => {
    add(rgb(meta.zone_colors[i]), `${name} — ${HUD_KEYS[name] || name}`);
  });
  add(rgb(meta.blocked_color), 'разрыв — коридор занят препятствием');
}

function applyFrame(buffer) {
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
  } else if (kind === LF) {
    const labels = new Uint8Array(buffer, 12 + n * 12, n);
    const pal = meta.zone_colors;
    for (let i = 0, j = 0; i < fits; i++) {
      const c = pal[labels[i]] || [255, 0, 255];
      colArr[j++] = c[0]; colArr[j++] = c[1]; colArr[j++] = c[2];
    }
    const counts = new Array(pal.length).fill(0);
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
  const url = `/frame?idx=${idx}&mode=${encodeURIComponent(state.mode)}`;
  return fetch(url).then((r) => {
    if (!r.ok) throw new Error(`GET /frame -> ${r.status}`);
    return r.arrayBuffer();
  });
}

function prefetch(idx) {
  if (idx < 0 || idx >= meta.frames) return;
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
  applyFrame(buf);
  state.idx = idx;
  $('frame').value = String(idx);
}

const PRESETS = [
  ['lidar', 'От лидара'], ['top', 'Сверху'], ['side', 'Сбоку'],
  ['iso', 'Изометрия'], ['behind', 'Сзади'], ['fit', 'Fit'],
];

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
  if (name === 'fit') {
    const n = state.numPoints || 1;
    const box = new THREE.Box3();
    const v = new THREE.Vector3();
    for (let i = 0; i < n; i += Math.max(1, Math.floor(n / 4000))) {
      v.set(posArr[i * 3], posArr[i * 3 + 1], posArr[i * 3 + 2]);
      box.expandByPoint(v);
    }
    if (box.isEmpty()) return;
    const c = box.getCenter(new THREE.Vector3());
    const r = Math.max(box.getSize(new THREE.Vector3()).length() * 0.55, 3);
    camera.position.set(c.x + r, c.y + r * 0.8, c.z + r * 0.7);
    controls.target.copy(c);
  } else {
    const [pos, tgt] = P[name];
    camera.position.set(pos[0], pos[1], pos[2]);
    controls.target.set(tgt[0], tgt[1], tgt[2]);
  }
  controls.update();
  for (const b of $('presets').children) b.classList.toggle('active', b.dataset.preset === name);
}

function updateHud() {
  const t = state.idx / Math.max(meta.hz, 0.01);
  let s = `кадр ${state.idx + 1}/${meta.frames}   t = ${t.toFixed(1)} с\n`;
  s += `точек ${state.numPoints}   ${state.lastFrameMs.toFixed(1)} мс   ${state.fpsShown.toFixed(1)} fps\n`;
  s += `пол z = ${meta.floor.a} + ${meta.floor.b}·y  (${meta.floor.grade_pct} %)\n`;
  s += `ось x = ${meta.axis_x} м   колея ${meta.gauge_mm} мм\n`;
  s += `стены x = ${meta.walls.join(', ')}\n`;
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
  for (const m of meta.modes) {
    const o = document.createElement('option');
    o.value = m; o.textContent = m;
    modeSel.appendChild(o);
  }
  modeSel.value = meta.default_mode;
  state.mode = meta.default_mode;
  modeSel.onchange = () => {
    state.mode = modeSel.value;
    cache.clear();
    showFrame(state.idx).catch(fail);
  };

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
  for (const id of ['axis', 'gauge', 'coff']) {
    $(id).oninput = onRailChange;
  }
  for (const id of ['contact', 'cside']) {
    $(id).onchange = onRailChange;
  }
  $('additive').onchange = (e) => {
    points.material.blending = e.target.checked ? THREE.AdditiveBlending : THREE.NormalBlending;
    points.material.depthWrite = !e.target.checked;
    points.material.needsUpdate = true;
  };

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

// Разрез X-Z по точкам вперёд на SECTION_WINDOW метров. Рисуем только когда
// сменился кадр (applyFrame сбрасывает sectionLastIdx), иначе -- ничего.
function drawSection() {
  if (!sectionCtx || !meta || !posArr) return;
  if (state.idx === sectionLastIdx) return;
  sectionLastIdx = state.idx;

  const W = SECTION_W, H = SECTION_H, dpr = sectionDpr, ctx = sectionCtx;
  const padL = 36, padR = 10, padT = 34, padB = 22;
  const plotX = padL, plotY = padT;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const yWin = -SECTION_WINDOW;
  const yc = -SECTION_WINDOW / 2;
  const rgbOf = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;
  const floorZ = meta.floor.a + meta.floor.b * yc;

  // 1) диапазон X/Z: точки окна + все оверлеи, чтобы ничего не обрезалось
  let xmin = Infinity, xmax = -Infinity, zmin = Infinity, zmax = -Infinity;
  const n = state.numPoints;
  for (let i = 0; i < n; i++) {
    const j = i * 3;
    if (posArr[j + 1] < yWin) continue;
    const x = posArr[j], z = posArr[j + 2];
    if (x < xmin) xmin = x;
    if (x > xmax) xmax = x;
    if (z < zmin) zmin = z;
    if (z > zmax) zmax = z;
  }
  const eatX = (v) => {
    if (!Number.isFinite(v)) return;
    if (v < xmin) xmin = v;
    if (v > xmax) xmax = v;
  };
  eatX(meta.axis_x);
  for (const x of meta.walls) eatX(x);
  for (const x of meta.rails.x) eatX(x);
  if (Number.isFinite(floorZ)) {
    if (floorZ < zmin) zmin = floorZ;
    if (floorZ > zmax) zmax = floorZ;
  }
  if (!(xmin <= xmax)) { xmin = -2; xmax = 2; }
  if (!(zmin <= zmax)) { zmin = -1; zmax = 1; }
  const mrg = 0.6;
  xmin -= mrg; xmax += mrg; zmin -= mrg; zmax += mrg;
  let dx = xmax - xmin, dz = zmax - zmin;
  if (!(dx > 0)) dx = 1;
  if (!(dz > 0)) dz = 1;
  const sc = Math.min(plotW / dx, plotH / dz);   // равный масштаб по X и Z
  const xc = (xmin + xmax) / 2, zc = (zmin + zmax) / 2;
  const mapX = (x) => plotX + plotW / 2 + (x - xc) * sc;
  const mapY = (z) => plotY + plotH / 2 - (z - zc) * sc;

  // 2) точки: пишем пиксели через ImageData, без строк на каждую точку
  const iw = Math.round(W * dpr), ih = Math.round(H * dpr);
  const img = ctx.createImageData(iw, ih);
  const buf32 = new Uint32Array(img.data.buffer);
  buf32.fill(0xff180f0c);                        // фон панели (ABGR)
  for (let i = 0; i < n; i++) {
    const j = i * 3;
    if (posArr[j + 1] < yWin) continue;
    const px = Math.round(mapX(posArr[j]) * dpr);
    const py = Math.round(mapY(posArr[j + 2]) * dpr);
    if (px < 0 || py < 0 || px >= iw || py >= ih) continue;
    buf32[py * iw + px] = 0xff000000
      | (colArr[j + 2] << 16) | (colArr[j + 1] << 8) | colArr[j];
  }
  ctx.putImageData(img, 0, 0);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  // 3) светлая сетка 1 м и подписи в метрах
  const step = sc >= 20 ? 1 : 2;
  ctx.lineWidth = 1;
  ctx.strokeStyle = 'rgba(120,140,170,0.18)';
  ctx.beginPath();
  for (let x = Math.ceil(xmin / step) * step; x <= xmax; x += step) {
    const px = Math.round(mapX(x)) + 0.5;
    ctx.moveTo(px, plotY); ctx.lineTo(px, plotY + plotH);
  }
  for (let z = Math.ceil(zmin / step) * step; z <= zmax; z += step) {
    const py = Math.round(mapY(z)) + 0.5;
    ctx.moveTo(plotX, py); ctx.lineTo(plotX + plotW, py);
  }
  ctx.stroke();

  ctx.fillStyle = 'rgba(138,151,176,0.9)';
  ctx.font = '9px "Segoe UI", system-ui, sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (let x = Math.ceil(xmin / step) * step; x <= xmax; x += step) {
    ctx.fillText(String(x), mapX(x), plotY + plotH + 3);
  }
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (let z = Math.ceil(zmin / step) * step; z <= zmax; z += step) {
    ctx.fillText(String(z), plotX - 4, mapY(z));
  }

  // 4) оверлеи
  const fy = mapY(floorZ);
  ctx.setLineDash([]);
  ctx.strokeStyle = 'rgb(130,150,180)';          // пол: z = a + b*y в центре окна
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(plotX, fy); ctx.lineTo(plotX + plotW, fy);
  ctx.stroke();

  ctx.strokeStyle = rgbOf(meta.zone_colors[3]);  // стены
  ctx.setLineDash([5, 4]);
  ctx.lineWidth = 1.25;
  ctx.beginPath();
  for (const x of meta.walls) {
    const px = mapX(x);
    if (px < plotX || px > plotX + plotW) continue;
    ctx.moveTo(px, plotY); ctx.lineTo(px, plotY + plotH);
  }
  ctx.stroke();

  ctx.strokeStyle = '#7fb2ff';                   // ось пути
  ctx.setLineDash([2, 3]);
  ctx.lineWidth = 1.25;
  ctx.beginPath();
  const axp = mapX(meta.axis_x);
  if (axp >= plotX && axp <= plotX + plotW) {
    ctx.moveTo(axp, plotY); ctx.lineTo(axp, plotY + plotH);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = rgbOf(meta.rail_color);      // рельсы: засечки у линии пола
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  for (const x of meta.rails.x) {
    const px = mapX(x);
    if (px < plotX || px > plotX + plotW) continue;
    ctx.moveTo(px, mapY(floorZ + 0.7));
    ctx.lineTo(px, mapY(floorZ - 0.15));
  }
  ctx.stroke();

  // 5) заголовок и сводка
  ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = '#dde4f0';
  ctx.font = '600 12px "Segoe UI", system-ui, sans-serif';
  ctx.fillText(`Разрез X-Z (первые ${SECTION_WINDOW} м)`, 10, 15);
  ctx.fillStyle = 'rgba(138,151,176,0.95)';
  ctx.font = '10px "Segoe UI", system-ui, sans-serif';
  ctx.fillText(
    `пол z = ${meta.floor.a} + ${meta.floor.b}·y , ось x = ${meta.axis_x} м , колея ${meta.gauge_mm} мм`,
    10, 28);

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
  buildLegend();
  wireUi();

  $('axis').value = String(meta.axis_x);
  $('axis').min = String(Math.round((meta.axis_x - 5) * 100) / 100);
  $('axis').max = String(Math.round((meta.axis_x + 5) * 100) / 100);
  $('gauge').value = String(meta.gauge_mm);
  onRailChange();
  $('dist').value = String(state.maxDist);
  $('distVal').textContent = state.maxDist;
  $('psize').value = String(meta.point_size);
  $('psizeVal').textContent = meta.point_size;
  document.title = `LiDAR — ${meta.bag}`;

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
  cameraPreset('lidar');
  $('presets').children[0].classList.add('active');
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