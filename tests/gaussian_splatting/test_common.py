from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.gaussian_splatting.core import common  # noqa: E402


class GaussianSplattingCommonTests(unittest.TestCase):
    def test_expected_mask_preserves_complete_image_filename(self) -> None:
        image_root = Path("images")
        mask_root = Path("masks")
        image_path = image_root / "nested" / "frame_000000.jpg"

        self.assertEqual(
            common.expected_mask_path(image_path, image_root, mask_root),
            mask_root / "nested" / "frame_000000.jpg.png",
        )

    def test_output_child_cannot_escape_workspace(self) -> None:
        workspace = common.PROJECT_ROOT / "data" / "processed" / "3dgs"
        with self.assertRaises(common.PipelineError):
            common.resolve_child(workspace, "../outside", "output.cache")

    def test_sparse_component_validation_accepts_binary_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sparse_model = Path(directory)
            for component in common.SPARSE_MODEL_COMPONENTS:
                (sparse_model / f"{component}.bin").write_bytes(b"model")

            self.assertEqual(common.validate_sparse_components(sparse_model), "bin")

    def test_sparse_component_validation_rejects_mixed_formats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sparse_model = Path(directory)
            (sparse_model / "cameras.bin").write_bytes(b"model")
            (sparse_model / "images.bin").write_bytes(b"model")
            (sparse_model / "points3D.txt").write_text("# model", encoding="utf-8")

            with self.assertRaisesRegex(common.PipelineError, "mix"):
                common.validate_sparse_components(sparse_model)

    def test_reads_registered_names_from_text_model(self) -> None:
        contents = """# Image list with two lines of data per image:
1 1 0 0 0 0 0 0 1 frame_000000.jpg
10.0 20.0 -1
2 1 0 0 0 0 0 0 1 nested/frame_000006.jpg

"""
        with tempfile.TemporaryDirectory() as directory:
            images_txt = Path(directory) / "images.txt"
            images_txt.write_text(contents, encoding="utf-8")

            self.assertEqual(
                common.read_text_model_image_names(images_txt),
                ["frame_000000.jpg", "nested/frame_000006.jpg"],
            )


if __name__ == "__main__":
    unittest.main()
