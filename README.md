# LiDAR Hackathon

Инструменты для работы с данными лидара: визуализация облаков точек из ROS 2 bag-файлов и LAZ-файлов, проверка габарита железнодорожного пути.

## Структура проекта

```
├── for_hackathon/               # ROS 2 bag-файлы (Hesai LiDAR, PointCloud2)
│   ├── doubleT_obstacle/        # Двойная стрелка + препятствие (~20 сек, 201 кадр)
│   ├── doubleT_platform/        # Двойная стрелка + платформа (~34 сек, 345 кадров)
│   ├── roundT_doubleT/          # Круглая + двойная стрелка (~25 сек, 252 кадра)
│   ├── roundT_pressureGate_roundT/                   # Круглая + ворота + круглая (~27 сек, 268 кадров)
│   ├── roundT_squareT_pressureGate_squareT/          # Круглая + прямоуг. + ворота + прямоуг. (~55 сек, 545 кадров)
│   └── squareT_platform_squareT_switch/              # Прямоуг. + платформа + прямоуг. + стрелка (~88 сек, 877 кадров)
├── visualize_bag.py             # Анимированная визуализация bag-файлов (без ROS)
├── snapshot.py                  # Интерактивный 3D-просмотр одного кадра
├── trim_bag.py                  # Прореживание bag-файла (каждый N-й кадр)
├── view_cloud.py                # Просмотр LAZ/LAS облаков точек
├── gabarit.py                   # Проверка габарита ж/д пути по аннотированным рельсам
├── Dockerfile                   # ROS 2 Humble (для ros2 bag play)
└── docker-compose.yml           # Запуск player + listener в Docker
```

### Обозначения в названиях bag-файлов

| Сокращение | Что это |
|---|---|
| `roundT` | Круглая стрелка (round turnout) |
| `doubleT` | Двойная стрелка (double turnout) |
| `squareT` | Прямоугольная стрелка (square turnout) |
| `platform` | Платформа (перрон) |
| `obstacle` | Препятствие на пути |
| `pressureGate` | Напорные ворота / шлагбаум |
| `switch` | Стрелочный перевод |

## Требования

```bash
pip install numpy open3d laspy scikit-learn
```

## Быстрый старт

```bash
# 1. Установить зависимости
pip install numpy open3d laspy scikit-learn

# 2. Разархивировать данные
tar --zstd -xvf for_hackathon.zst

# 3. Запустить визуализацию любого bag-файла
python3 visualize_bag.py for_hackathon/doubleT_platform
```

## Визуализация bag-файлов (без ROS, без Docker)

Скрипт читает `.db3` (SQLite) напрямую, парсит CDR-сериализованные PointCloud2 и показывает покадровую анимацию в Open3D.

```bash
# По умолчанию — doubleT_platform
python3 visualize_bag.py

# Любой другой bag — просто указать путь к папке
python3 visualize_bag.py for_hackathon/doubleT_obstacle
python3 visualize_bag.py for_hackathon/roundT_doubleT
python3 visualize_bag.py for_hackathon/roundT_pressureGate_roundT
python3 visualize_bag.py for_hackathon/roundT_squareT_pressureGate_squareT
python3 visualize_bag.py for_hackathon/squareT_platform_squareT_switch
```

Управление в окне Open3D: мышь — вращение, scroll — зум, Q — выход.

## Просмотр одного кадра (snapshot)

Открывает первый кадр из bag-файла в интерактивном 3D-окне — удобно для настройки камеры и изучения облака.

```bash
python3 snapshot.py for_hackathon/doubleT_platform
```

## Прореживание bag-файла

Создаёт облегчённую копию bag-файла, оставляя каждый N-й кадр. Полезно для быстрого предпросмотра больших записей.

```bash
# Каждый 5-й кадр (по умолчанию)
python3 trim_bag.py for_hackathon/doubleT_platform

# Каждый 10-й кадр
python3 trim_bag.py for_hackathon/doubleT_platform 10
```

Результат сохраняется в папку `<имя_bag>_trim<N>/`.

## Просмотр LAZ-файлов

```bash
# Все LAZ в текущей папке, окраска по интенсивности
python3 view_cloud.py

# Выбор файла по индексу и режима окраски
python3 view_cloud.py 0 intensity   # по интенсивности
python3 view_cloud.py 0 height      # по высоте (синий→красный)
python3 view_cloud.py 0 rgb         # оригинальные цвета
```

## Проверка габарита

Определяет положение рельсов из аннотации (`*_rail.laz`), строит габаритный коридор 3.2×3.7 м и подсвечивает точки внутри него.

```bash
python3 gabarit.py
```

Цвета: серый — облако, жёлтый — рельсы, зелёный — точки в габарите, красная рамка — границы габарита.

## Запуск через Docker (ROS 2)

Если нужен `ros2 bag play` или ROS-топики:

```bash
# На Mac сначала запустить colima
colima start --memory 4 --disk 40

# Собрать образ
docker-compose build

# Запустить player + listener
docker-compose up

# Или проиграть один bag вручную
docker-compose run ros2 ros2 bag play for_hackathon/doubleT_platform
```

## Данные

- **Bag-файлы**: Hesai LiDAR, топик `/lidar_points`, 307200 точек/кадр, ~10 Hz, поля: x, y, z, intensity, ring, timestamp
- **LAZ-файлы**: статические сканы ж/д путей
