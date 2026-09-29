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
//     ставила в полёт по запросу на каждый промежуточный кадр.
//
// Механика здесь, в чистом виде (без DOM и three.js -- модуль проверяется
// node-тестом на моках fetch, .scratch/_tmp_frameq_test.mjs):
//   * токен запроса: применяется ответ только ПОСЛЕДНЕГО запрошенного кадра
//     («последний запрошенный выигрывает»), опоздавшие ответы и их ошибки
//     молча отбрасываются (show() возвращает false);
//   * prefetch соседей В ОБЕ стороны (±radius): назад-навигация тоже идёт
//     по прогретому кэшу;
//   * дедупликация: один кадр -- один запрос (show делит промис с prefetch);
//     повторный show того же кадра (та же запись и режим раскраски) в сеть
//     не ходит вовсе;
//   * кэш ограничен по размеру (выталкивание самых старых записей), чтобы
//     протяжка слайдера не копила мегабайты невостребованных буферов.
//
// Ключ кэша и дедупа -- (tag, idx): tag -- строка личности потока кадров
// (в app.js это «запись·режим»), поэтому смена записи/режима сама делает
// старые записи недостижимыми, а clear() их просто выкидывает из памяти.

export function createFrameQueue({
  fetchFrame,          // (idx) -> Promise<buffer> -- загрузка кадра (GET /frame)
  onFrame,             // (buffer, idx) -> void -- применить кадр; может кидать
  prefetchFrame = null, // (idx) -> void -- прогрев серверного кэша (track3d.prefetch)
  radius = 2,          // сколько соседей греть в каждую сторону
  cacheMax = 16,       // сколько промисов/буферов держать в кэше
} = {}) {
  if (typeof fetchFrame !== 'function') throw new Error('frameq: нужен fetchFrame');
  if (typeof onFrame !== 'function') throw new Error('frameq: нужен onFrame');

  const cache = new Map();   // `${tag}:${idx}` -> Promise<buffer>
  let reqSeq = 0;            // номер последнего запрошенного кадра (токен гонки)
  let shownKey = null;       // ключ последнего ПРИМЕНЁННОГО кадра (дедуп повторов)

  const keyOf = (tag, idx) => `${tag}:${idx}`;

  function remember(key, p) {
    p.catch(() => cache.delete(key));   // битый запрос не оставляем в кэше
    cache.set(key, p);
    // LRU по порядку добавления: старые записи выталкиваем, свежие живут.
    while (cache.size > cacheMax) cache.delete(cache.keys().next().value);
  }

  // Прогреть соседний кадр: серверный кэш геометрии (prefetchFrame -- он же
  // собирает ось кадра, чтобы /frame не ждал сборку в момент показа) и сам
  // /frame в клиентский кэш. Без сети, если кадр уже в кэше.
  function prefetch(tag, idx, frames) {
    if (!(idx >= 0) || !(idx < frames)) return;
    if (prefetchFrame) prefetchFrame(idx);
    const key = keyOf(tag, idx);
    if (cache.has(key)) return;
    remember(key, Promise.resolve().then(() => fetchFrame(idx)));
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
    for (let d = 1; d <= radius; d++) {
      prefetch(tag, idx + d, frames);          // вперёд
      prefetch(tag, idx - d, frames);          // и назад
    }
    let p = cache.get(key);
    if (!p) {
      // Кладём запрос в кэш: повторный show того же кадра (или поздний
      // prefetch) разделит этот промис, а не поставит второй запрос.
      p = Promise.resolve().then(() => fetchFrame(idx));
      remember(key, p);
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
    // ограничивает cacheMax (выталкивание старых), чужие tag не пересекаются.
    return true;
  }

  return {
    show,
    prefetch,
    // Смена записи/режима: старые ключи и так недостижимы (другой tag), здесь
    // просто освобождаем память и забываем «показанный» кадр.
    clear() { cache.clear(); shownKey = null; },
    // Для тестов и отладки.
    get size() { return cache.size; },
    get shown() { return shownKey; },
  };
}
