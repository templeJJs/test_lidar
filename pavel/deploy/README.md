# Поставка guard_detector — Docker (Ubuntu 22.04 + ROS 2 Humble)

Нода габаритного контроля: обёртка тракта `pavel/guard`
(ось пути → стены → конструкции → препятствия, `guard.run.analyze`) в ROS 2.
open3d в образе НЕТ и не нужен: разбор PointCloud2 — `guard/cloud.py`
(numpy, без зависимостей).

## Требования

- Docker (проверено на Docker Desktop 29.1.3, Windows; на целевой машине —
  любой Linux с Docker).
- Росbag2-запись с топиком `sensor_msgs/msg/PointCloud2` (например
  `for_hackathon/doubleT_obstacle`, топик `/sensing/lidar/hesai128/pointcloud`).

## Сборка

Из **корня репозитория** (контексту нужны `pavel/` и `ros2_ws/`):

```bash
docker build -f pavel/deploy/Dockerfile -t guard_detector .
```

## Запуск

```bash
# Только нода (лидар/бэг публикует кто-то другой):
docker run --rm --net host guard_detector

# Нода + воспроизведение записи (каталог бэга монтируется в /data):
docker run --rm -v "$PWD/for_hackathon/doubleT_obstacle:/data/doubleT_obstacle" \
    guard_detector ros2 launch guard_detector guard.launch.py \
    topic:=/sensing/lidar/hesai128/pointcloud bag:=/data/doubleT_obstacle

# Часть записи (с 5-й секунды) — для быстрой проверки на больших бэгах:
#   ... bag:=/data/doubleT_obstacle start_offset:=5
```

Параметры ноды (через launch-аргумент `topic` или `-p`):

| параметр | умолчание | что |
|---|---|---|
| `topic` | `/lidar_points` | топик PointCloud2 лидара |
| `frames` | `20` | окно усреднения fps/латентности |
| `publish_corridor` | `true` | публиковать линии габарита для RViz |

## Проверка топиков

В том же контейнере (второй сессией: `docker exec -it <id> bash -c
'source /opt/ros/humble/setup.bash && bash'`):

```bash
ros2 topic echo /guard/status --once -l 5000   # JSON: fps, latency_ms, reach, коридор, находки
ros2 topic hz  /guard/status                   # ~10 Гц при 10 Гц лидара
ros2 topic echo /guard/obstacles --once        # MarkerArray: дальность, lat, размер, in_gauge, confirmed
```

(`ros2 topic echo` по умолчанию обрезает строки до 128 символов — для
`/guard/status` нужен `-l`, иначе JSON виден не целиком.)

Латентность кадра печатается в лог ноды на каждом кадре
(`кадр: 65 мс, точек 346526, reach 115 м, находок 1, в габарите 1, ...`)
и дублируется в `/guard/status` (`latency_ms` — медиана окна, `latency_last_ms`).

## Топики

| топик | тип | что |
|---|---|---|
| `/guard/obstacles` | `visualization_msgs/MarkerArray` | CUBE+TEXT на находку: дальность, lat, ш×в, in_gauge, confirmed. Цвет: красный — в габарите и подтверждено, оранжевый — в габарите без подтверждения (бин-'hole'), серый — вне габарита |
| `/guard/corridor` | `visualization_msgs/MarkerArray` | линии габаритного коридора: зелёный — бин подтверждён измерением, жёлтый — ведётся предсказанием ('hole', «не наблюдается», НЕ «свободно») |
| `/guard/status` | `std_msgs/String` (JSON) | `recv_fps, proc_fps, latency_ms, latency_last_ms, points, axis_measured, reach_m, reach_bridged_m, corridor{half_*_med/min_m}, obstacles{hits, in_gauge, in_gauge_confirmed, detect_ms}` |

**QoS подписки не менять на `SensorDataQoS()`**: в записях ТЗ издатель
объявлен надёжным (RELIABLE), BEST_EFFORT-подписка молча получает ноль
сообщений.

Координаты лидара: x вправо, y НАЗАД, z вверх (guard внутри: fwd=-y, lat=x,
up=z). Маркеры кладутся в `frame_id` сообщения: x = lat_abs находки,
y = −дальность, z = пол + высота/2.

## Латентность

Замер `analyze` внутри контейнера (Docker Desktop на AMD 8C/16T, numpy 1.21,
общие vCPU) по двум полным проигрываниям doubleT_obstacle (396 кадров по
~346 тыс. точек): **медиана 85 мс, p90 106 мс, min 74 мс**; первый кадр после
старта ~0.25-0.4 с (ленивый импорт scipy — разово, на поток не влияет).
На хосте без контейнера (numpy 2.4): медиана 67 мс, max 85 мс.
Бюджет 10 Гц (100 мс/кадр) в контейнере выдержан по медиане; на целевом
i7-9700E (8C, 2.6 ГГц, bare metal, без оверхеда VM) запас сохраняется.

Проверено на записи doubleT_obstacle (201 кадр): нода отработала 199 кадров
(первые 1-2 уходят до конца discovery — это KEEP_LAST depth=1, а не потери),
человек в габарите стабильно найден: находка 52-58 м, lat ±1.6, ~0.6×1.2 м,
in_gauge=true, confirmed=true (красный маркер в /guard/obstacles).

Известная особенность `ros2 bag play` из-под launch: при закрытом stdin
плеер завершался сразу — лечится флагом `--disable-keyboard-controls`
(уже стоит в guard.launch.py). Дефолтный `--read-ahead-queue-size` 1000 при
облаках ~24 МБ — десятки ГБ ОЗУ; в launch выставлено 4.
