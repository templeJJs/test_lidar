# `tunnel_detector` — ROS 2 пакет детектора

Тонкий слой над алгоритмом из `detector/` (лежит в корне репозитория и
монтируется в `/opt/lidar`). Пороги и обоснование — `docs/detector-algorithm.md`,
измерения — `docs/detector-results.md`.

## Три команды

```bash
# 1. собрать (в контейнере ros:humble; детектор уже в PYTHONPATH)
colcon build --packages-select tunnel_detector && source install/setup.bash

# 2. запустить узел модели и детектор (record обязателен для качества)
ros2 launch tunnel_detector detect.launch.py record:=roundT_doubleT

# 3. проиграть запись и посмотреть картинку
ros2 launch tunnel_detector detect.launch.py \
    record:=roundT_doubleT bag:=/data/roundT_doubleT rviz:=true
```

Проверка потока:

```bash
ros2 topic echo /tunnel/status --once     # JSON: fps, latency_ms, bins_blocked, ...
ros2 topic hz /tunnel/obstacles
```

## Топики

| топик | тип | QoS | что |
|---|---|---|---|
| `/lidar_points` | `sensor_msgs/PointCloud2` | подписка RELIABLE, depth 1, VOLATILE | пять записей, frame `hesai_lidar` |
| `/sensing/lidar/hesai128/pointcloud` | `sensor_msgs/PointCloud2` | то же | `doubleT_obstacle`, frame `lidar_livox` |
| `/tunnel/model` | `std_msgs/String` (JSON) | TRANSIENT_LOCAL | полилиния оси + УГР |
| `/tunnel/obstacles` | `visualization_msgs/MarkerArray` | VOLATILE | CUBE + TEXT_VIEW_FACING с дистанцией |
| `/tunnel/obstacle_points` | `sensor_msgs/PointCloud2` | VOLATILE | точки найденных препятствий |
| `/tunnel/status` | `std_msgs/String` (JSON) | VOLATILE | `recv_fps, proc_fps, latency_ms, points, cells, bins_blocked, bins_total, max_data_range_m, detections` |

**QoS подписки не менять на `SensorDataQoS()`.** В записях издатель объявлен
надёжным (`reliability: 1`), и BEST_EFFORT-подписка на надёжного издателя даёт
молча ноль сообщений — это главная ловушка этого датасета.

## Узлы

* `model_node` — один раз считает модель пути (полилиния оси + измеренный УГР) и
  публикует её латч-топиком. Источник: параметр `record` → таблица
  `detector/track_models.json` (снимок `track_geometry.build_track`); иначе
  живой `build_track`; иначе прямая ось с предупреждением. Без `record`
  детектор работает по прямой оси и на поворотах даёт ложные срабатывания
  (замерено: 13 занятых бинов против 0).
* `detector_node` — на каждое облако: `np.frombuffer` → `view(N, point_step)` →
  колонки x, y, z по смещениям из `msg.fields` → `detector.detect` →
  маркеры + статус.

## Запуск без ROS

```bash
python -m unittest discover -s tests -q
python -c "import bag_reader, detector, json; \
  db='for_hackathon/roundT_doubleT/roundT_doubleT_0.db3'; \
  bf=bag_reader.BagFrames(db); print(json.dumps(detector.detect(bf[0].xyz, model=detector.model_for_db(db)).to_dict(), ensure_ascii=False))"
```