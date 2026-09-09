import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/reconstruction'))
from cleanup_mesh import clean, read_ply, write_mesh, component_mask


class CleanupTests(unittest.TestCase):
    def test_supported_surface_retained_and_artifacts_removed(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                             [10, 10, 10], [11, 10, 10], [10, 11, 10]], dtype=float)
        faces = np.array([[0, 1, 2], [2, 1, 0], [0, 0, 1], [3, 4, 5]])
        keep, report = clean(vertices, faces, vertices[:3], dict(
            support_voxel_fraction=0.003, minimum_component_faces=1,
            maximum_removed_fraction=0.9))
        np.testing.assert_array_equal(keep, [True, False, False, False])
        self.assertEqual(report['removal_reasons']['duplicate_faces'], 1)
        self.assertEqual(report['removal_reasons']['degenerate_faces'], 1)
        self.assertEqual(report['removal_reasons']['unsupported_faces'], 1)
        self.assertTrue(report['ready_to_write'])

    def test_component_filter_keeps_large_component(self):
        faces = np.array([[0, 1, 2], [2, 1, 3], [4, 5, 6]])
        keep, counts = component_mask(faces, 7, 2)
        np.testing.assert_array_equal(keep, [True, True, False])
        self.assertEqual(counts, [2, 1])

    def test_removal_limit_blocks_candidate(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        keep, report = clean(vertices, np.array([[0, 1, 2], [2, 0, 1]]), vertices,
                             dict(support_voxel_fraction=0.003,
                                  minimum_component_faces=1, maximum_removed_fraction=0.1))
        self.assertFalse(report['ready_to_write'])

    def test_roundtrip_preserves_positions_winding_and_refuses_overwrite(self):
        vertices = np.array([[9, 9, 9], [1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
        faces = np.array([[3, 1, 2]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mesh.ply'
            write_mesh(path, vertices, faces)
            output, triangles = read_ply(path)
            np.testing.assert_array_equal(output[triangles], vertices[faces])
            with self.assertRaises(FileExistsError):
                write_mesh(path, vertices, faces)


if __name__ == '__main__':
    unittest.main()
