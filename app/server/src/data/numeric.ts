// Числовые помощники, повторяющие numpy там, где это влияет на совпадение с
// Python-эталоном (`bag_reader.py` / `zones.py`): раскладка значений по бинам
// гистограммы, линейная интерполяция перцентилей, МНК-прямая, arange/linspace
// и np.interp.
//
// Числа считаем в number (float64), как numpy-версия. Отдельно отмечено, где
// numpy работает во float32 (арифметика массива float32 с Python-скаляром):
// там каждый шаг округляем через f32(), иначе младшие биты не совпадут.

/** Округление до float32: numpy считает массив float32 со скаляром во float32. */
export const f32 = Math.fround

/** Значения в порядке возрастания (перцентили и медиана, как в numpy). */
export function sortedCopy(values: ArrayLike<number>): Float64Array {
  const out = Float64Array.from(values)
  out.sort()
  return out
}

/** Индекс первого элемента > v (np.searchsorted side='right'). */
export function searchSortedRight(sorted: ArrayLike<number>, v: number): number {
  let lo = 0
  let hi = sorted.length
  while (lo < hi) {
    const mid = (lo + hi) >>> 1
    if (v < sorted[mid]!) hi = mid
    else lo = mid + 1
  }
  return lo
}

/** Индекс первого элемента >= v (np.searchsorted side='left'). */
export function searchSortedLeft(sorted: ArrayLike<number>, v: number): number {
  let lo = 0
  let hi = sorted.length
  while (lo < hi) {
    const mid = (lo + hi) >>> 1
    if (sorted[mid]! < v) lo = mid + 1
    else hi = mid
  }
  return lo
}

/** Первый индекс максимума (np.argmax). */
export function argMax(values: ArrayLike<number>): number {
  let best = 0
  for (let i = 1; i < values.length; i++) {
    if (values[i]! > values[best]!) best = i
  }
  return best
}

/**
 * Перцентиль методом 'linear' (умолчание numpy) по УЖЕ отсортированному
 * массиву: виртуальный индекс (n-1)*q/100 и интерполяция между соседними
 * порядковыми статистиками. Форма интерполяции -- как в numpy `_lerp`: при
 * t >= 0.5 считаем от правого конца, иначе от левого (разница в младших битах,
 * но она накапливается).
 */
export function percentileFromSorted(sorted: ArrayLike<number>, q: number): number {
  const n = sorted.length
  if (n === 0) throw new RangeError('percentile: пустой массив')
  const idx = (n - 1) * (q / 100.0)
  const lo = Math.floor(idx)
  const hi = lo + 1
  if (hi >= n) return sorted[n - 1]!
  const t = idx - lo
  const d = sorted[hi]! - sorted[lo]!
  return t >= 0.5 ? sorted[hi]! - d * (1.0 - t) : sorted[lo]! + d * t
}

/** Перцентиль методом 'linear' (numpy default) по произвольному массиву. */
export function percentileLinear(values: ArrayLike<number>, q: number): number {
  return percentileFromSorted(sortedCopy(values), q)
}

/** Медиана как np.median: среднее двух центральных значений при чётном n. */
export function median(values: ArrayLike<number>): number {
  const n = values.length
  if (n === 0) throw new RangeError('median: пустой массив')
  const a = sortedCopy(values)
  if (n % 2 === 1) return a[(n - 1) >>> 1]!
  return (a[n / 2 - 1]! + a[n / 2]!) / 2.0
}

/**
 * Копия `np.arange(start, stop, step)` для float64.
 *
 * В numpy массив заливается так: длина = ceil((stop-start)/step), первые два
 * элемента -- start и start+step, остальные -- start + i*delta, где
 * delta = (start+step) - start. Это ни i*step, ни накопление суммы: для
 * кривых шагов (0.05, 0.1) младшие биты получаются свои, и от них зависит,
 * в какой бин гистограммы попадёт точка.
 */
export function arange(start: number, stop: number, step: number): Float64Array {
  const delta = stop - start
  const tmp = delta / step
  let length: number
  if (tmp === 0.0 && delta !== 0.0) {
    // npy_signbit(tmp): 1/tmp == -Infinity для -0.0
    length = 1 / tmp < 0 ? 0 : 1
  } else {
    if (!Number.isFinite(tmp)) throw new RangeError('arange: не удалось вычислить длину')
    length = Math.ceil(tmp)
  }
  if (length <= 0) return new Float64Array(0)
  const out = new Float64Array(length)
  out[0] = start
  if (length === 1) return out
  out[1] = start + step
  const d = out[1]! - start
  for (let i = 2; i < length; i++) out[i] = start + i * d
  return out
}

/** Копия `np.linspace(start, stop, num)`: i*delta + start, последний = stop. */
export function linspace(start: number, stop: number, num: number): Float64Array {
  if (num <= 0) return new Float64Array(0)
  const out = new Float64Array(num)
  if (num === 1) {
    out[0] = start
    return out
  }
  const delta = (stop - start) / (num - 1)
  for (let i = 0; i < num; i++) out[i] = i * delta + start
  out[num - 1] = stop
  return out
}

/**
 * Число попаданий в бины, как `np.histogram(values, bins=edges)`: бины
 * полуинтервальные [e_i, e_{i+1}), последний замкнут справа, значения вне
 * диапазона [e_0, e_n] отбрасываются (NaN тоже не попадает никуда).
 */
export function histogramCounts(values: ArrayLike<number>, edges: ArrayLike<number>): Float64Array {
  const nb = edges.length - 1
  if (nb <= 0) return new Float64Array(0)
  const counts = new Float64Array(nb)
  const last = edges[nb]!
  for (let k = 0; k < values.length; k++) {
    const v = values[k]!
    if (!(v <= last)) continue
    let i = searchSortedRight(edges, v) - 1
    if (i < 0) continue
    if (i >= nb) i = nb - 1
    counts[i] = counts[i]! + 1
  }
  return counts
}

/**
 * Двумерная гистограмма (плоский массив ny*nx, строка -- бин по Y), правила
 * раскладки те же, что у `histogramCounts` и `np.histogram2d`.
 */
export function histogram2dCounts(
  ys: ArrayLike<number>,
  xs: ArrayLike<number>,
  yEdges: ArrayLike<number>,
  xEdges: ArrayLike<number>,
): Float64Array {
  const ny = yEdges.length - 1
  const nx = xEdges.length - 1
  if (ny <= 0 || nx <= 0) return new Float64Array(0)
  const counts = new Float64Array(ny * nx)
  const lastY = yEdges[ny]!
  const lastX = xEdges[nx]!
  for (let k = 0; k < ys.length; k++) {
    const y = ys[k]!
    const x = xs[k]!
    if (!(y <= lastY) || !(x <= lastX)) continue
    let iy = searchSortedRight(yEdges, y) - 1
    let ix = searchSortedRight(xEdges, x) - 1
    if (iy < 0 || ix < 0) continue
    if (iy >= ny) iy = ny - 1
    if (ix >= nx) ix = nx - 1
    counts[iy * nx + ix] = counts[iy * nx + ix]! + 1
  }
  return counts
}

/**
 * Прямая МНК `np.polyfit(x, y, 1)` -> [наклон, сдвиг]. Считаем через центр
 * тяжести: numpy решает через SVD, расхождение ~1e-15 (проверено на данных).
 */
export function polyfit1(xs: ArrayLike<number>, ys: ArrayLike<number>): [number, number] {
  const n = xs.length
  let sx = 0.0
  let sy = 0.0
  for (let i = 0; i < n; i++) {
    sx += xs[i]!
    sy += ys[i]!
  }
  const mx = sx / n
  const my = sy / n
  let sxx = 0.0
  let sxy = 0.0
  for (let i = 0; i < n; i++) {
    const dx = xs[i]! - mx
    sxx += dx * dx
    sxy += dx * (ys[i]! - my)
  }
  const slope = sxy / sxx
  return [slope, my - slope * mx]
}

/** Линейная интерполяция `np.interp(x, xp, fp)` для одного значения. */
export function interp(x: number, xp: ArrayLike<number>, fp: ArrayLike<number>): number {
  const n = xp.length
  if (x <= xp[0]!) return fp[0]!
  if (x >= xp[n - 1]!) return fp[n - 1]!
  const i = searchSortedLeft(xp, x)
  const slope = (fp[i]! - fp[i - 1]!) / (xp[i]! - xp[i - 1]!)
  return slope * (x - xp[i - 1]!) + fp[i - 1]!
}

/**
 * Норма вектора так, как её считает np.linalg.norm по float32: сумма квадратов
 * накапливается во float32 (порядок x, y, z), корень тоже float32.
 */
export function normF32(x: number, y: number, z: number): number {
  const s = f32(f32(f32(x * x) + f32(y * y)) + f32(z * z))
  return Math.sqrt(s)
}