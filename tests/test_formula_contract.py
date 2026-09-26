"""Контракт формул клиент/сервер: векторы на диске == выход labeling.py.

`contracts/formula_vectors.json` -- эталонные входы/выходы формул, которые
написаны дважды (сервер `labeling.py`, клиент `web/labellayer.js`/`web/app.js`).
Векторы считает `formula_contract.py` ВЫЗОВОМ серверных функций (источник
истины), клиент сверяется с ними Node-тестом `.scratch/formula_contract_test.mjs`
(его при наличии node гоняет и этот модуль -- TestNodeContract).

Здесь проверяется:
* файл контракта на диске совпадает с регенерацией из ТЕКУЩЕГО labeling.py
  байт-в-байт: правка серверной формулы без `python formula_contract.py`
  роняет батарею сразу, а не оставляет клиент молча расходиться;
* структура: все формулы пары присутствуют, у каждой есть векторы, допуски
  заданы;
* разошедшийся случай из ревью: поза ВНЕ списка на сервере валидируется в
  `standing` (`pose_rule`) -- векторы pose_rule обязаны содержать такие входы.
"""
import json
import os
import shutil
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import formula_contract  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACT_PATH = formula_contract.CONTRACT_PATH
NODE_TEST = os.path.join(ROOT, '.scratch', 'formula_contract_test.mjs')

EXPECTED_FORMULAS = {
    'pose_moment', 'pose_vertices', 'pose_rotation', 'place_pose', 'pose_rule',
    'trajectory_at', 'trajectory_normalize', 'motion_at', 'motion_between',
    'instance_center', 'pose_time',
}


class TestVectorsInSync(unittest.TestCase):
    """Файл контракта соответствует текущему коду сервера."""

    def test_file_exists(self):
        self.assertTrue(os.path.isfile(CONTRACT_PATH),
                        f'нет {CONTRACT_PATH} -- запустите: python formula_contract.py')

    def test_file_matches_server(self):
        with open(CONTRACT_PATH, encoding='utf-8') as fh:
            on_disk = fh.read()
        regenerated = formula_contract.dumps(formula_contract.vectors())
        self.assertEqual(
            on_disk, regenerated,
            'contracts/formula_vectors.json расходится с labeling.py: '
            'после правки серверной формулы пересчитайте векторы '
            '(python formula_contract.py) и проверьте клиент '
            '(node .scratch/formula_contract_test.mjs)')

    def test_check_cli(self):
        rc = formula_contract.main(['--check'])
        self.assertEqual(rc, 0)


class TestCoverage(unittest.TestCase):
    """Структура контракта: все пары формул покрыты, допуски заданы."""

    @classmethod
    def setUpClass(cls):
        cls.data = formula_contract.vectors()

    def test_all_formula_pairs_present(self):
        names = set(self.data['formulas'])
        self.assertEqual(names, EXPECTED_FORMULAS)

    def test_every_formula_has_cases_and_sides(self):
        for name, entry in self.data['formulas'].items():
            self.assertTrue(entry.get('server'), name)
            self.assertTrue(entry.get('client'), name)
            self.assertGreaterEqual(len(entry.get('cases') or []), 4, name)

    def test_tolerances_present(self):
        tol = self.data['tolerance']
        for key in ('position_m', 'float32_m', 'blend', 'time_s', 'matrix'):
            self.assertIn(key, tol)
            self.assertGreater(tol[key], 0.0)
        # Порог позиции не слабее микрона: замеры хода в проекте -- 1e-3 м.
        self.assertLessEqual(tol['position_m'], 1e-6)

    def test_pose_rule_vectors_cover_unknown_pose(self):
        """Разошедшийся случай ревью: поза вне списка -> standing (сервер)."""
        cases = self.data['formulas']['pose_rule']['cases']
        garbage = [c for c in cases
                   if c['pose'] not in (None, '', 'standing', 'walking', 'running',
                                        'lying', 'fallen', 'unknown',
                                        'WALKING', 'Standing')]
        self.assertTrue(garbage, 'нет векторов с позой вне списка')
        for c in garbage:
            self.assertEqual(c['expect'], 'standing',
                             f"поза {c['pose']!r} обязана валидироваться в standing")

    def test_motion_vectors_cover_holes_and_backwards(self):
        cases = self.data['formulas']['motion_between']['cases']
        self.assertTrue(any(c['to'] < c['from'] for c in cases),
                        'нет векторов обратного хода')
        names = {c['motion'] for c in cases}
        self.assertIn('holes', names)       # дыры профиля -> медиана
        self.assertIn('scalar', names)      # без профиля -> default * df
        self.assertIn('step2', names)       # шаг профиля > 1


class TestNodeContract(unittest.TestCase):
    """Клиент против векторов: node .scratch/formula_contract_test.mjs."""

    def test_client_matches_vectors(self):
        node = shutil.which('node')
        if node is None:
            raise unittest.SkipTest('node не найден')
        if not os.path.isfile(NODE_TEST):
            raise unittest.SkipTest(f'нет {NODE_TEST}')
        got = subprocess.run([node, NODE_TEST], cwd=ROOT, capture_output=True,
                             text=True, timeout=120)
        if got.returncode != 0:
            tail = '\n'.join((got.stdout + '\n' + got.stderr).splitlines()[-25:])
            self.fail(f'клиент расходится с векторами контракта:\n{tail}')


if __name__ == '__main__':
    unittest.main()
