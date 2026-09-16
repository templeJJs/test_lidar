"""Симулятор туннеля и лидарных данных.

Модуль строит синтетические записи метро-туннеля в том же формате, что наши
`for_hackathon/*.db3` (rosbag2 + CDR PointCloud2), чтобы весь существующий
конвейер -- `bag_reader`, `zones`, `web_viewer` -- работал на синтетике без правок.

Зачем: в реальных записях нет размеченных препятствий, обучать и проверять
детектор не на чем. Спека -- docs/superpowers/specs/2026-09-15-tunnel-simulator-design.md,
план -- docs/superpowers/plans/2026-09-15-tunnel-simulator.md.
"""