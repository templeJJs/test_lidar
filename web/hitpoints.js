// Подсветка ТОЧЕК препятствий (obstacles[i].point_idx из /guard) в облаке
// кадра: общий кусок для guardlayer.js (3D, красит colArr) и topview.js
// (2D, пиксели canvas). Чистые функции без THREE и DOM — node-тест гоняет
// их напрямую (без заглушек браузера).
//
// point_idx — индексы в ПОДГОТОВЛЕННОМ кадре вьюера (/frame после trim по
// max_dist/behind и возможного прореживания max_points, см. шапку
// pavel/guard/export_web.py). Совпадение систем индексов проверяется по
// frame_npts: если оно не равно числу точек облака (вьюер запущен с другими
// --max-dist/--max-points), индексы чужие — красить нельзя, остаются боксы.

export const HIT_RGB = [255, 34, 34];    // ярко-красный поверх палитры кадра

/**
 * Индексы точек всех находок кадра.
 * frame — guard-payload (или null), count — число точек облака вьюера.
 * Старые JSON без point_idx (и кадры с чужой нумерацией) дают [] — слои
 * показывают только боксы, как раньше.
 */
export function hitIndices(frame, count) {
  if (!frame || !Array.isArray(frame.obstacles)) return [];
  if (typeof frame.frame_npts === 'number' && frame.frame_npts !== count) return [];
  const out = [];
  for (const ob of frame.obstacles) {
    const pi = ob && ob.point_idx;
    if (!Array.isArray(pi)) continue;
    for (const i of pi) {
      if (Number.isInteger(i) && i >= 0 && i < count) out.push(i);
    }
  }
  return out;
}

/**
 * Красит colors (Uint8Array RGB, 3 байта на точку) по индексам в HIT_RGB.
 * Возвращает бэкап { idx, prev } для restoreCloud или null, если красить
 * нечего. Дубли индексов красятся/сохраняются один раз.
 */
export function paintCloud(colors, indices) {
  if (!colors || !indices || !indices.length) return null;
  const seen = new Set();
  const idx = [];
  for (const i of indices) {
    if (seen.has(i)) continue;
    seen.add(i);
    idx.push(i);
  }
  if (!idx.length) return null;
  const prev = new Uint8Array(idx.length * 3);
  for (let k = 0; k < idx.length; k++) {
    const j = idx[k] * 3;
    prev[k * 3] = colors[j];
    prev[k * 3 + 1] = colors[j + 1];
    prev[k * 3 + 2] = colors[j + 2];
    colors[j] = HIT_RGB[0];
    colors[j + 1] = HIT_RGB[1];
    colors[j + 2] = HIT_RGB[2];
  }
  return { idx, prev };
}

/** Возвращает цвета, сохранённые paintCloud. Пустой/чужой бэкап — no-op. */
export function restoreCloud(colors, backup) {
  if (!colors || !backup) return;
  const { idx, prev } = backup;
  for (let k = 0; k < idx.length; k++) {
    const j = idx[k] * 3;
    colors[j] = prev[k * 3];
    colors[j + 1] = prev[k * 3 + 1];
    colors[j + 2] = prev[k * 3 + 2];
  }
}

/**
 * Метки точек находок для 2D-панели: Uint8Array(count) с 1 на индексах из
 * indices (null, если пусто). Панель рисует помеченные точки красным после
 * облака, без прореживания.
 */
export function hitMarks(count, indices) {
  if (!indices || !indices.length || !(count > 0)) return null;
  const marks = new Uint8Array(count);
  for (const i of indices) {
    if (i >= 0 && i < count) marks[i] = 1;
  }
  return marks;
}
