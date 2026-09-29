// Очередь кадров вьюера: навигация showFrame без гонок и лишних запросов.
//
// Что было плохо (web/app.js до этой правки):
//   * showFrame(idx) ждал СВОЙ ответ /frame и применял его без проверки
//     актуальности: при быстрой серии (стрелки, протяжка слайдера, 1→2→3)
//     ответы приходили вразнобой, и на экран попадал кадр, чей ответ пришёл
//     ПОСЛЕДНИМ, а не тот, что запрошен последним, -- облако «отбрасывало
//     то вперёд, то назад»;
//   * prefetch грел только кадры ВПЕРЁД (idx+1, idx+2): шаг назад каждый раз
//     ждал сервер (первый показ кадра -- сборка геометрии, десятые доли секунды);
//   * прямой fetchFrame из showFrame не попадал в кэш: повторный showFrame того
//     же кадра (или дубль из prefetch) качал /frame заново, а протяжка слайдера
//     ставила в полёт по запросу на каждый промежуточный кадр;
//   * устаревшие запросы не отменялись: ответы по ~2.3 МБ докачивались и
//     занимали браузерные слоты (6 на хост) -- запрос ТЕКУЩЕГО кадра ждал в
//     очереди за пятью устаревшими;
//   * prefetch /frame обходил лимит «1 в полёте», который держит прогрев
//     геометрии (track3d.js): /frame при холодной геометрии собирает её сам,
//     а сборка на сервере глобально сериализована (build_lock) -- параллельные
//     холодные запросы не дают throughput, только отодвигают текущий кадр.
//
// Механика здесь, в чистом виде (без DOM и three.js -- модуль проверяется
// node-тестом на моках fetch, .scratch/_tmp_frameq_test.mjs):
//   * токен запроса: применяется ответ только ПОСЛЕДНЕГО запрошенного кадра
//     («последний запрошенный выигрывает»), опоздавшие ответы и их ошибки
//     молча отбрасываются (show() возвращает false);
//   * prefetch соседей В ОБЕ стороны (±radius): назад-навигация тоже идёт
//     по прогретому кэшу; в полёте не больше PREFETCH_MAX_INFLIGHT prefetch-
//     запросов, остальные ждут очереди и пересматриваются на каждый show();
//   * отмена устаревшего: на каждый show() запросы в полёте и очередь prefetch
//     вне окна ±radius (и чужих tag) отменяются через AbortController --
//     браузерный слот освобождается сразу (уже начатая серверная сборка
//     доработает и ляжет в серверный кэш -- не страшно);
//   * дедупликация: один кадр -- один запрос (show делит промис с prefetch);
//     повторный show того же кадра (та же запись и режим раскраски) в сеть
//     не ходит вовсе;
//   * кэш ограничен по размеру: LRU с move-to-end при попадании (как в
//     серверном _cached, web_viewer.py) -- свежепоказанный кадр не вытесняется
//     раньше нетронутых prefetch'ей, шаг «назад-вперёд» мгновенный.
//
// Ключ кэша и дедупа -- (tag, idx): tag -- строка личности потока кадров
// (в app.js это «запись·режим»), поэтому смена записи/режима сама делает
// старые записи недостижимыми, а clear() их просто выкидывает из памяти.

// Сколько prefetch-запросов /frame держим в полёте одновременно. Зеркало
// PREFETCH_MAX_INFLIGHT из track3d.js: сборка геометрии на сервере
// сериализована build_lock, поэтому больше одного холодного /frame в полёте
// throughput не даёт, а браузерные слоты съедает. Запрос ТЕКУЩЕГО кадра этим
// лимитом не ограничен -- его пользователь ждёт.
const PREFETCH_MAX_INFLIGHT = 1;

export function createFrameQueue({
  fetchFrame,          // (idx, signal) -> Promise<buffer> -- загрузка кадра (GET /frame)
  onFrame,             // (buffer, idx) -> void -- применить кадр; может кидать
  prefetchFrame = null, // (idx) -> void -- прогрев серверного кэша (track3d.prefetch)
  radius = 2,          // сколько соседей греть в каждую сторону
  cacheMax = 16,       // сколько промисов/буферов держать в кэше
} = {}) {
  if (typeof fetchFrame !== 'function') throw new Error('frameq: нужен fetchFrame');
  if (typeof onFrame !== 'function') throw new Error('frameq: нужен onFrame');

  const cache = new Map();     // `${tag}:${idx}` -> Promise<buffer>
  const inflight = new Map();  // key -> { tag, idx, ctl, isPrefetch }
  const waitQueue = [];        // ключи prefetch, ждущие свободный слот
  const waitMeta = new Map();  // key -> { tag, idx }
  let prefetchInFlight = 0;
  let reqSeq = 0;            // номер последнего запрошенного кадра (токен гонки)
  let shownKey = null;       // ключ последнего ПРИМЕНЁННОГО кадра (дедуп повторов)

  const keyOf = (tag, idx) => `${tag}:${idx}`;

  function remember(key, p) {
    // Битый запрос не оставляем в кэше; сравнение с p -- чтобы поздняя ошибка
    // старого промиса не снесла НОВЫЙ запрос под тем же ключом (перезапрос
    // после abort/вытеснения).
    p.catch(() => { if (cache.get(key) === p) cache.delete(key); });
    cache.set(key, p);
    // LRU по порядку Map: выталкиваем самые старые записи, свежие (и недавно
    // показанные -- show() делает move-to-end) живут дольше. Вытесненный
    // запрос в полёте не отменяем: он дольет и осядет ниоткуда, а устаревшие
    // отменяет show() через abort.
    while (cache.size > cacheMax) cache.delete(cache.keys().next().value);
  }

  function startRequest(tag, idx, isPrefetch) {
    const key = keyOf(tag, idx);
    const ctl = new AbortController();
    inflight.set(key, { tag, idx, ctl, isPrefetch });
    const p = Promise.resolve().then(() => fetchFrame(idx, ctl.signal));
    const settle = () => {
      // Отменённый/очищенный запрос уже снят с учёта (dropRequest/clear) --
      // второй раз счётчики не трогаем.
      if (!inflight.delete(key)) return;
      if (isPrefetch) {
        prefetchInFlight -= 1;
        drainPrefetch();
      }
    };
    p.then(settle, settle);
    remember(key, p);
    return p;
  }

  // Отмена запроса в полёте: освобождает браузерный слот и слот prefetch.
  function dropRequest(key) {
    const rec = inflight.get(key);
    if (!rec) return;
    inflight.delete(key);
    rec.ctl.abort();
    cache.delete(key);
    if (rec.isPrefetch) prefetchInFlight -= 1;
  }

  // Свободный prefetch-слот -- следующему из очереди ожидания.
  function drainPrefetch() {
    while (prefetchInFlight < PREFETCH_MAX_INFLIGHT && waitQueue.length) {
      const key = waitQueue.shift();
      const m = waitMeta.get(key);
      waitMeta.delete(key);
      if (!m || cache.has(key)) continue;
      prefetchInFlight += 1;
      startRequest(m.tag, m.idx, true);
    }
  }

  // Прогреть соседний кадр: серверный кэш геометрии (prefetchFrame -- он же
  // собирает ось кадра, чтобы /frame не ждал сборку в момент показа) и сам
  // /frame в клиентский кэш. Без сети, если кадр уже в кэше или ждёт очереди.
  // Сам запрос /frame встаёт в очередь prefetch (лимит PREFETCH_MAX_INFLIGHT).
  function prefetch(tag, idx, frames) {
    if (!(idx >= 0) || !(idx < frames)) return;
    if (prefetchFrame) prefetchFrame(idx);
    const key = keyOf(tag, idx);
    if (cache.has(key) || waitMeta.has(key)) return;
    waitMeta.set(key, { tag, idx });
    waitQueue.push(key);
    drainPrefetch();
  }

  /**
   * Показать кадр. Возвращает true, если кадр применён (onFrame отработал),
   * false -- если кадр уже показан (дедуп) или запрос опоздал (пока летел
   * ответ, запросили другой кадр). Ошибка загрузки ПОСЛЕДНЕГО запрошенного
   * кадра пробрасывается; ошибка опоздавшего -- нет (она никому не нужна).
   */
  async function show(tag, idx, frames) {
    const myReq = ++reqSeq;
    const key = keyOf(tag, idx);
    if (shownKey === key) return false;        // тот же кадр: сеть не трогаем
    // Отменяем устаревшее: запросы в полёте и очередь prefetch вне окна
    // ±radius этой же личности (tag) -- устаревший /frame держит браузерный
    // слот и отодвигает текущий кадр.
    for (const k of [...inflight.keys()]) {
      const rec = inflight.get(k);
      if (rec.tag !== tag || Math.abs(rec.idx - idx) > radius) dropRequest(k);
    }
    for (let i = waitQueue.length - 1; i >= 0; i--) {
      const m = waitMeta.get(waitQueue[i]);
      if (!m || m.tag !== tag || Math.abs(m.idx - idx) > radius) {
        waitMeta.delete(waitQueue[i]);
        waitQueue.splice(i, 1);
      }
    }
    drainPrefetch();   // освободившийся слот -- следующему из очереди
    for (let d = 1; d <= radius; d++) {
      prefetch(tag, idx + d, frames);          // вперёд
      prefetch(tag, idx - d, frames);          // и назад
    }
    let p = cache.get(key);
    if (p) {
      // Настоящий LRU: попадание освежает запись (move-to-end).
      cache.delete(key);
      cache.set(key, p);
    } else {
      // Кладём запрос в кэш: повторный show того же кадра (или поздний
      // prefetch) разделит этот промис, а не поставит второй запрос.
      p = startRequest(tag, idx, false);
    }
    let buf;
    try {
      buf = await p;
    } catch (e) {
      if (myReq !== reqSeq) return false;      // опоздавший: чужая ошибка молчит
      throw e;
    }
    if (myReq !== reqSeq) return false;        // опоздавший ответ -- отбрасываем
    onFrame(buf, idx);                         // кинул -- кадр НЕ считается показанным
    shownKey = key;
    // Применённый буфер ИЗ кэша не убираем: шаг «назад-вперёд» по только что
    // показанным кадрам обязан быть мгновенным, без повторного /frame. Память
    // ограничивает cacheMax (LRU-вытеснение), чужие tag не пересекаются.
    return true;
  }

  return {
    show,
    prefetch,
    // Смена записи/режима: старые ключи и так недостижимы (другой tag), здесь
    // освобождаем память, отменяем запросы в полёте и забываем «показанный»
    // кадр. reqSeq НЕ сбрасываем: все вызовы clear() в app.js сразу ведут в
    // showFrame (он инкрементит токен), а сброс под ноль позволил бы
    // опоздавшему ответу со старым номером совпасть с новым.
    clear() {
      for (const rec of inflight.values()) rec.ctl.abort();
      inflight.clear();
      prefetchInFlight = 0;
      waitQueue.length = 0;
      waitMeta.clear();
      cache.clear();
      shownKey = null;
    },
    // Для тестов и отладки.
    get size() { return cache.size; },
    get shown() { return shownKey; },
    get inflightCount() { return inflight.size; },
  };
}
