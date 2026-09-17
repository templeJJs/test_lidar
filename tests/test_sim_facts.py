"""Замеры по записям (`sim/dataset_facts.json`): форма сечения и её границы.

Артефакт -- это то, из чего генератор берёт числа, поэтому его факты должны быть
проверяемыми: у нас уже был в артефакте мусорный `rail_bands` (локальные пики
гистограммы по X), который выглядел как данные, но не значил ничего. Здесь
закреплено то, что замер действительно даёт: профиль сечения по шести записям,
наклонный свод у двухпутной записи и невидимый в ближней зоне свод у узких.
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACTS = os.path.join(ROOT, 'sim', 'dataset_facts.json')
WIDE = 'doubleT_obstacle'
NARROW = 'roundT_doubleT'
AXIS_HALF_M = 1.0               # полоса «над осью пути» -- та же, что в check_sim


def facts() -> dict:
    with open(FACTS, encoding='utf-8') as fh:
        return json.load(fh)


def section_of(name: str) -> list:
    bags = facts()['bags']
    return (bags.get(name) or {}).get('section') or []


def top_over_axis(section: list) -> float:
    vals = [b['top'] for b in section if abs(b['x']) <= AXIS_HALF_M]
    return float(sum(vals) / len(vals)) if vals else float('nan')


class TestSectionProfile(unittest.TestCase):
    def test_every_measured_bag_has_a_section(self):
        bags = facts()['bags']
        with_section = [n for n, f in bags.items() if f.get('section')]
        self.assertEqual(len(with_section), len(bags),
                         f'профиль сечения есть не у всех записей: {sorted(bags)}')
        self.assertGreaterEqual(len(with_section), 6)

    def test_section_is_a_well_formed_profile(self):
        """Ячейки идут по X по порядку, свод не ниже ложа, точек достаточно."""
        for name, f in facts()['bags'].items():
            section = f.get('section') or []
            xs = [b['x'] for b in section]
            self.assertEqual(xs, sorted(xs), f'{name}: ячейки профиля не по порядку')
            self.assertEqual(len(set(xs)), len(xs), f'{name}: ячейки повторяются')
            for b in section:
                self.assertGreaterEqual(b['top'], b['low'],
                                        f'{name}: свод ниже ложа в x={b["x"]}')
                self.assertGreaterEqual(b['n'], 30, f'{name}: в ячейке мало точек')

    def test_wide_recording_has_a_sloping_vault(self):
        """У двухпутной записи свод НЕ плоский: 4.8 м слева, 4.2 м справа.

        Это и есть расхождение со сценой: генератор строит плоский потолок
        4.10 м, а сверка показывает 4.16 против 4.86 м у записи (Δ 0.70).
        """
        section = section_of(WIDE)
        self.assertTrue(section, f'нет профиля сечения у {WIDE}')
        left = [b['top'] for b in section if -2.2 <= b['x'] <= -1.2]
        right = [b['top'] for b in section if 4.5 <= b['x'] <= 6.0]
        self.assertTrue(left and right)
        drop = max(left) - max(right)
        self.assertGreater(drop, 0.40, f'свод {WIDE} должен падать слева направо')

    def test_narrow_recordings_have_no_vault_in_the_near_zone(self):
        """У узких записей свод в ближней зоне не видно: над осью всего ~0.4 м.

        Сцена же рисует там плоский потолок 4.1 м (проверено `check_sim`:
        4.13 против 0.40 -- Δ 3.73 м). Пока это ИЗВЕСТНОЕ расхождение, и тест
        не даёт ему пропасть незамеченным.
        """
        for name in (NARROW, 'roundT_pressureGate_roundT'):
            section = section_of(name)
            if not section:
                self.skipTest(f'нет профиля у {name}')
            top = top_over_axis(section)
            self.assertLess(top, 1.0, f'{name}: над осью ожидали пусто, а свод {top:.2f} м')

    def test_bogus_rail_bands_are_gone(self):
        """Поле `rail_bands` было мусором (пики гистограммы по X) -- его больше нет."""
        for name, f in facts()['bags'].items():
            self.assertNotIn('rail_bands', f, f'{name}: вернулось мусорное поле')


if __name__ == '__main__':
    unittest.main()