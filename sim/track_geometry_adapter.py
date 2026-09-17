"""Заглушка адаптера рельсов: Task 3 заменит её настоящими мешами R65 и шпал."""


def add_track(parts: list, p) -> dict:
    from sim.scene import GAUGE_M
    axis = p.track_axis_x
    return {'axis_x': axis, 'gauge': GAUGE_M,
            'rails_x': [axis - GAUGE_M / 2.0, axis + GAUGE_M / 2.0]}