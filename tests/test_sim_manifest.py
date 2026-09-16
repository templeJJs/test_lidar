"""Тесты манифеста датасета: сцена, объекты с разметкой, запись рядом с датасетом."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestManifest(unittest.TestCase):
    def test_manifest_roundtrip_and_required_fields(self):
        from sim.manifest import Manifest, SceneInfo, ObjectInfo
        m = Manifest(seed=42, frames=5, hz=10.0,
                     scene=SceneInfo(kind='straight', section='wide', length_m=60.0,
                                     grade_pct=2.34, tracks=2,
                                     sensor_pose={'x': 0.0, 'y': 0.0, 'z': 1.82}),
                     objects=[ObjectInfo(id=1, cls='person',
                                         bbox_xyz=[[-0.2, -20.3, -1.6], [0.2, -19.7, 0.15]],
                                         y_along_m=-20.0, offset_from_axis_m=0.0,
                                         in_gabarit=True, frames_visible=[0, 1])])
        d = m.to_dict()
        for key in ('seed', 'frames', 'hz', 'scene', 'objects'):
            self.assertIn(key, d)
        self.assertEqual(d['objects'][0]['class'], 'person')
        self.assertTrue(d['objects'][0]['in_gabarit'])
        again = Manifest.from_dict(d)
        self.assertEqual(again.objects[0].y_along_m, -20.0)

    def test_manifest_is_written_next_to_the_bag(self):
        import json
        import tempfile
        from sim.manifest import Manifest, SceneInfo
        d = tempfile.mkdtemp(prefix='simtest_')
        path = Manifest(seed=1, frames=1, hz=10.0,
                        scene=SceneInfo(kind='straight', section='narrow', length_m=20.0,
                                        grade_pct=1.0, tracks=1, sensor_pose={'z': 1.32}),
                        objects=[]).save(d)
        with open(path, encoding='utf-8') as fh:
            data = json.load(fh)
        self.assertEqual(data['scene']['section'], 'narrow')


if __name__ == '__main__':
    unittest.main()