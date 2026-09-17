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

import numpy as np

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


class TestSectionProfile(unittest.TestCase):
    def test_every_measured_bag_has_a_section(self):
        bags = facts()['bags']
        with_section = [n for n, f in bags.items() if f.get('section')]
        self.assertEqual(len(with_section), len(bags),
                         f'профиль сечения есть не у всех записей: {sorted(bags)}')
        self.assertGreaterEqual(len(with_section), 6)

    def test_section_is_a_well_formed_profile(self):
        """Ячейки идут по X по порядку, свод не ниже ложа, точек достаточно.

        Свод может быть НЕ замерен в ячейке (там его просто не видно) -- тогда
        `top` равен None, и это честнее нуля: ноль означал бы «свода нет».
        """
        for name, f in facts()['bags'].items():
            section = f.get('section') or []
            xs = [b['x'] for b in section]
            self.assertEqual(xs, sorted(xs), f'{name}: ячейки профиля не по порядку')
            self.assertEqual(len(set(xs)), len(xs), f'{name}: ячейки повторяются')
            measured = 0
            for b in section:
                self.assertGreaterEqual(b['n'], 30, f'{name}: в ячейке мало точек')
                if b.get('top') is None:
                    continue
                measured += 1
                self.assertGreaterEqual(b['top'], b['low'],
                                        f'{name}: свод ниже ложа в x={b["x"]}')
            self.assertGreater(measured, len(section) // 2,
                               f'{name}: свод замерен меньше чем в половине ячеек')

    def test_wide_recording_has_a_sloping_vault(self):
        """У двухпутной записи свод НЕ плоский: 4.9 м слева, 4.3 м справа.

        Сцена строит плоский свод на верной высоте (это убрало расхождение по
        уровню: 4.75 против 4.69), но наклон по X пока не воспроизводит --
        сырьё для этого и есть профиль сечения.
        """
        section = section_of(WIDE)
        self.assertTrue(section, f'нет профиля сечения у {WIDE}')
        left = [b['top'] for b in section
                if b.get('top') is not None and -2.2 <= b['x'] <= -1.2]
        right = [b['top'] for b in section
                 if b.get('top') is not None and 4.5 <= b['x'] <= 6.0]
        self.assertTrue(left and right)
        drop = max(left) - max(right)
        self.assertGreater(drop, 0.40, f'свод {WIDE} должен падать слева направо')

    def test_every_recording_has_a_measured_vault(self):
        """Свод есть у ВСЕХ шести записей, и он выше 3 м.

        Сначала замер показывал «свода над осью нет» (0.4 м) -- это была ошибка
        оценщика, а не свойство данных: свод попадает в поле зрения прибора только
        на дальности, а перцентиль по смешанному окну показывал ложе (точек ложа
        в ближней зоне на порядок больше). Правильный замер -- перцентиль 99.5 по
        точкам дальше 8 м, и он даёт гребень 4.6…5.1 м, а у записи со станцией
        8.5 м.
        """
        for name, f in facts()['bags'].items():
            section = f.get('section') or []
            tops = [b['top'] for b in section if b.get('top') is not None]
            self.assertTrue(tops, f'{name}: свод не замерен')
            crown = float(np.median([b['top'] for b in section
                                     if b.get('top') is not None
                                     and abs(b['x']) <= AXIS_HALF_M] or tops))
            self.assertGreater(crown, 3.0, f'{name}: гребень свода {crown:.2f} м')
            self.assertLess(crown, 12.0, f'{name}: гребень свода {crown:.2f} м')

    def test_profiles_carry_the_measured_vault(self):
        """Сцена берёт высоту свода из замера, а не из константы 4.1 м."""
        from sim import scene as sc
        for prof in sc.fact_profiles():
            section = prof.get('section') or []
            crown = float(np.median([b['top'] for b in section
                                     if b.get('top') is not None
                                     and abs(b['x']) <= AXIS_HALF_M]))
            self.assertAlmostEqual(prof['vault_m'], crown, places=3,
                                   msg=f'{prof["bag"]}: высота свода не из замера')
            self.assertNotAlmostEqual(prof['vault_m'], sc.TUNNEL_HEIGHT_M, places=2,
                                      msg='у записей свод не на 4.1 м')

    def test_bogus_rail_bands_are_gone(self):
        """Поле `rail_bands` было мусором (пики гистограммы по X) -- его больше нет."""
        for name, f in facts()['bags'].items():
            self.assertNotIn('rail_bands', f, f'{name}: вернулось мусорное поле')


if __name__ == '__main__':
    unittest.main()