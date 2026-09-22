# Лицензии ассетов

Этот каталог — **не** в git: бинарники ассетов в репозиторий не попадают
(`.gitignore`: `assets/*` с исключениями для `LICENSES.md` и `manifest.json`).
Скачиваются они скриптом:

```bash
python tools/fetch_assets.py            # скачать + распаковать + проверить sha256
python tools/fetch_assets.py --check    # только проверить, ничего не качать
python tools/check_assets.py            # состояние пайплайна: файлы, читатели, лицензии
```

Числа (вершины/треугольники/bbox/единицы) и машинные поля лицензий — в
`manifest.json`; таблица ниже — человекочитаемая сводка.

## Сводка

| id | ассет | лицензия | атрибуция | бинарники в git | источник |
|----|-------|----------|-----------|-----------------|----------|
| `kenney_city_kit_roads` | Kenney City Kit Roads 2.1 | CC0 1.0 | не требуется (просят, но не обязывают) | можно, но не коммитим | https://kenney.nl/assets/city-kit-roads |
| `animated_human_quaternius` | Animated Human by @Quaternius | CC0 1.0 (текст в архиве; на quaternius.com тот же пак — под QAL v1.0) | не требуется | **нет** (см. ниже) | https://opengameart.org/content/animated-human-low-poly |
| `rigged_animated_humanoid` | Rigged and Animated Humanoid (XCVG Systems) | CC0 1.0 | не требуется (courtesy: XCVG / xcvgsystems.com) | можно, но не коммитим | https://opengameart.org/content/rigged-and-animated-humanoid |
| `cesium_man` | CesiumMan | CC-BY 4.0 | **да** | можно, но не коммитим | https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/CesiumMan |
| `rigged_figure` | RiggedFigure | CC-BY 4.0 | **да** | можно, но не коммитим | https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/RiggedFigure |
| `fox` | Fox | CC-BY 4.0 | **да** | можно, но не коммитим | https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Fox |
| `kira` | kira (three.js example) | CC0 1.0 | не требуется | можно, но не коммитим | https://github.com/mrdoob/three.js/tree/dev/examples/models/gltf |
| `mixamo_fbx_manual` | Mixamo (любой персонаж) | Adobe Mixamo terms | по условиям Mixamo | **нет** | https://www.mixamo.com/ (качать вручную) |

Почему «можно, но не коммитим»: решение проекта — держать в git только скрипты,
манифест и этот файл. Даже CC0-бинарники раздувают историю, а `fetch_assets.py`
восстанавливает их одной командой с проверкой sha256.

## Тексты атрибуции (CC-BY, обязательны к отображению)

* **CesiumMan** — CesiumMan by Cesium (Khronos glTF-Sample-Assets), CC-BY-4.0,
  https://creativecommons.org/licenses/by/4.0/
* **RiggedFigure** — RiggedFigure by Cesium (Khronos glTF-Sample-Assets), CC-BY-4.0
* **Fox** — `CC-BY 4.0 Model by PixelMannen https://opengameart.org/content/fox-and-shiba
  and @tomkranis https://sketchfab.com/3d-models/low-poly-fox-by-pixelmannen-animated-371dea88d7e04a76af5763f2a36866bc
  and @AsoboStudio with @scurest` (строка взята из поля `asset.copyright` самого Fox.glb)

## Особые случаи

**`animated_human_quaternius` — бинарники только локально.**
Архив, зеркалированный на OpenGameArt, содержит `License.txt` с CC0 1.0. Сам
Quaternius раздаёт эту же модель на quaternius.com под собственной лицензией
**QAL v1.0**, которая запрещает распространять ассеты *как ассеты*. Чтобы не
зависеть от трактовки, пак помечен `redistribute_binaries: false`: он качается
скриптом (`python tools/fetch_assets.py --only animated_human_quaternius`), в
git не попадает, атрибуция не требуется.

**`mixamo_fbx_manual` — нет URL, качать руками.**
У Mixamo нет прямой ссылки: нужна интерактивная сессия. Порядок действий лежит в
`manifest.json` (`source.manual_instructions`) и печатается `fetch_assets.py`:
залогиниться, выбрать персонажа и клип, Download → FBX Binary, With Skin, 30 fps,
Keyframe Reduction: none, положить файл в `assets/mixamo_fbx_manual/<имя>.fbx`.
Mixamo отдаёт сантиметры: масштаб в метры `0.01` (проверять `tools/inspect_mesh.mjs`).

**`kira` — DRACO.** Файл сжат `KHR_draco_mesh_compression`: в браузере он читается
клиентским DRACOLoader'ом, а офлайн — ни open3d (assimp: *«GLTF: Draco mesh
compression not supported»*), ни three в Node (`Worker`/`importScripts`
недоступны). Анимаций в нём нет вообще. В манифесте помечен
`read_by: {three_node: false, open3d: false}` — это ожидаемое состояние, а не
сбой (проверка в `check_assets.py` сравнивает факт с ожиданием).

## Что подтверждено на диске

`tools/check_assets.py` проверяет по каждому ассету: архив скачан и его sha256
совпадает с закреплённым, файлы распакованы, лицензия записана (и атрибуция там,
где она обязательна), ассет читается ожидаемыми ридерами. Лицензионные файлы
внутри архивов (не коммитятся, но проверены при написании этой страницы):

* `assets/kenney_city_kit_roads/License.txt` — «Creative Commons Zero, CC0»,
  «Support by crediting 'Kenney' or 'www.kenney.nl' (this is not a requirement)»;
* `assets/animated_human_quaternius/Animated Human by @Quaternius/License.txt` —
  «CC0 1.0 Universal (CC0 1.0) Public Domain Dedication»;
* `assets/rigged_animated_humanoid/COPYING.txt` — полный текст CC0 1.0,
  `README.txt` — «This is licensed CC0 Public Domain … Attribution is not required»;
* `cesium_man`, `rigged_figure`, `fox` — лицензия указана в репозитории
  Khronos glTF-Sample-Assets (CC-BY 4.0), внутри GLB атрибуция у Fox лежит в
  `asset.copyright`.