"""Модель лидара, извлечённая из настоящих записей (не из даташита).

Почему не даташит: у нас конкретный сенсор, и воспроизводить надо его. Замеры по
всем шести записям дали одинаковые 128 колец с углами +14.40..-25.12 и медианным
шагом 0.168, но разное число азимутальных колонок на кадр (2709 против ~1400) --
и, что важнее, разное ПОЛЕ ЗРЕНИЯ по азимуту: 247° у широкой записи против 101° у
узкой. Поэтому колонки и поле зрения -- параметры модели, а не константы.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

AZ_BIN_DEG = 0.5        # ячейка азимута для замера поля зрения: мельче шага колонок


def azimuth_span_deg(xyz: np.ndarray, bin_deg: float = AZ_BIN_DEG) -> float:
    """Поле зрения по азимуту: сколько ячеек занято, столько градусов сенсор и видит.

    Прибор не снимает полный оборот: у широкой записи хвостовой сектор 113° пуст
    (точки собственного вагона при записи отфильтрованы), у узкой запись укладывается
    в ±50°. Замер по ЗАНЯТЫМ ячейкам даёт «ширину» этого поля зрения; сами ячейки
    берутся грубее шага колонок, поэтому мелкие пропуски возвратов его не занижают.
    """
    az = np.degrees(np.arctan2(np.asarray(xyz)[:, 0], -np.asarray(xyz)[:, 1]))
    edges = np.arange(-180.0, 180.0 + bin_deg, bin_deg)
    occupied = np.histogram(az, bins=edges)[0] > 0
    return float(bin_deg * int(np.count_nonzero(occupied)))


@dataclass
class SensorModel:
    rings: int = 128
    elevations_deg: np.ndarray = field(default_factory=lambda: np.zeros(0))
    azimuth_columns: int = 2709
    azimuth_span_deg: float = 360.0   # поле зрения по азимуту: у записей 247 / 101
    range_median_m: float = 6.9
    range_p95_m: float = 28.2
    range_max_m: float = 208.7
    range_sigma_m: float = 0.02
    intensity: dict = field(default_factory=dict)     # median, p90, share_gt25
    dropout: float = 0.0

    @classmethod
    def from_bag(cls, db_path: str, frame: int = 0) -> 'SensorModel':
        """Снять модель с одного кадра настоящей записи."""
        import bag_reader

        frames = bag_reader.BagFrames(db_path, cache_size=2, prefetch_ahead=0)
        try:
            f = frames[frame]
            xyz = f.xyz.astype(np.float64)
            ring = f.ring
            inten = f.intensity
        finally:
            frames.close()
        if ring is None:
            raise ValueError('в записи нет поля ring -- модель сенсора не снять')

        r = np.hypot(xyz[:, 0], xyz[:, 1])
        elev = np.degrees(np.arctan2(xyz[:, 2], r))
        rings = int(ring.max()) + 1
        elevations = np.full(rings, np.nan)
        for k in range(rings):
            m = ring == k
            if int(np.count_nonzero(m)) >= 10:
                elevations[k] = float(np.median(elev[m]))
        # пустые кольца заполняем интерполяцией: у сенсора должны быть все лучи
        good = ~np.isnan(elevations)
        if not bool(good.all()) and bool(good.any()):
            idx = np.arange(rings)
            elevations = np.interp(idx, idx[good], elevations[good])

        # шум дальности -- по самой гладкой поверхности в кадре, а не по полу: балласт
        # шероховат на 5-10 см, и его шероховатость забивает оценку (пол по MAD
        # даёт 0.10-0.21 м, то есть меряет камни, а не сенсор). Берём стену:
        # вертикальная плоскость облицовки, разброс по X вокруг её центра.
        import zones
        a, b, _rms = zones.fit_floor(xyz, y_min=-40.0, y_max=-2.0)
        above = (xyz[:, 2] - (a + b * xyz[:, 1])) > 0.7
        if int(np.count_nonzero(above)) >= 1000:
            hist, edges = np.histogram(xyz[above, 0],
                                       bins=np.arange(xyz[above, 0].min(),
                                                      xyz[above, 0].max() + 0.1, 0.1))
            peak = float(0.5 * (edges[int(np.argmax(hist))] + edges[int(np.argmax(hist)) + 1]))
            wall = xyz[above, 0][np.abs(xyz[above, 0] - peak) < 0.5]
            sigma = 1.4826 * float(np.median(np.abs(wall - np.median(wall))))
        else:
            sigma = 0.02

        stats = {}
        if inten is not None:
            stats = {'median': float(np.median(inten)),
                     'p90': float(np.percentile(inten, 90)),
                     'share_gt25': float(np.mean(inten > 25))}
        columns = int(round(len(xyz) / float(rings)))
        return cls(rings=rings, elevations_deg=elevations, azimuth_columns=columns,
                   azimuth_span_deg=azimuth_span_deg(xyz),
                   range_median_m=float(np.median(r)),
                   range_p95_m=float(np.percentile(r, 95)),
                   range_max_m=float(r.max()), range_sigma_m=max(sigma, 0.005),
                   intensity=stats)

    def to_dict(self) -> dict:
        return {'rings': self.rings,
                'elevations_deg': [float(v) for v in self.elevations_deg],
                'azimuth_columns': self.azimuth_columns,
                'azimuth_span_deg': self.azimuth_span_deg,
                'range_median_m': self.range_median_m,
                'range_p95_m': self.range_p95_m,
                'range_max_m': self.range_max_m,
                'range_sigma_m': self.range_sigma_m,
                'intensity': self.intensity, 'dropout': self.dropout}

    @classmethod
    def from_dict(cls, d: dict) -> 'SensorModel':
        d = dict(d)
        d['elevations_deg'] = np.asarray(d['elevations_deg'], dtype=np.float64)
        return cls(**d)

    def save(self, path: str) -> None:
        import json
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str) -> 'SensorModel':
        import json
        with open(path, encoding='utf-8') as fh:
            return cls.from_dict(json.load(fh))

    def azimuths_deg(self) -> np.ndarray:
        """Азимуты кадра: `azimuth_columns` замеров на поле зрения `azimuth_span_deg`.

        Азимут 0° -- вперёд (в наших записях это -Y), отсчёт симметричный: у
        сенсора поле зрения разложено влево-вправо от курса. Полный оборот
        (360°) остаётся умолчанием для макетного сенсора; у записей поле зрения
        уже (247° у широкой, 101° у узкой), и хвостовой сектор не снимается вовсе
        -- поэтому синтетика его и не должна показывать.
        """
        half = 0.5 * float(self.azimuth_span_deg)
        return np.linspace(-half, half, int(self.azimuth_columns), endpoint=False)