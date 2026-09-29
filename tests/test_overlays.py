import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from overlays import OverlayEngine


class OverlayEngineTests(unittest.TestCase):
    def test_loads_rgba_and_blends_with_global_opacity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgba.png"
            image = np.zeros((10, 20, 4), dtype=np.uint8)
            image[:, :, 2] = 200
            image[:, :, 3] = 128
            self.assertTrue(cv2.imwrite(str(path), image))

            engine = OverlayEngine()
            self.assertTrue(engine.load_asset("rgba", path))
            frame = np.zeros((30, 30, 3), dtype=np.uint8)
            result = engine.draw(
                frame,
                "rgba",
                center=(15, 15),
                target_width=20,
                opacity=0.5,
            )

            self.assertIs(result, frame)
            self.assertGreater(frame[15, 15, 2], 45)
            self.assertLess(frame[15, 15, 2], 55)

    def test_supports_three_channel_images_and_frame_clipping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plain.jpg"
            image = np.zeros((10, 10, 3), dtype=np.uint8)
            image[:, :, 1] = 200
            self.assertTrue(cv2.imwrite(str(path), image))

            engine = OverlayEngine()
            self.assertTrue(engine.load_asset("plain", path))
            frame = np.zeros((12, 12, 3), dtype=np.uint8)
            engine.draw(
                frame,
                "plain",
                center=(0, 0),
                target_width=20,
                opacity=1.0,
                flip_horizontal=True,
            )

            self.assertGreater(frame[0, 0, 1], 150)

    def test_missing_asset_is_reported_once(self) -> None:
        engine = OverlayEngine()
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            self.assertFalse(engine.load_asset("missing", "/does/not/exist.png"))
            self.assertFalse(engine.load_asset("missing", "/does/not/exist.png"))

        self.assertEqual(output.getvalue().count("Overlay-Asset fehlt"), 1)

    def test_dissolve_breaks_alpha_into_irregular_islands(self) -> None:
        image = np.full((80, 120, 4), 255, dtype=np.uint8)

        middle = OverlayEngine._dissolve_overlay(image, 0.55, seed=12)
        end = OverlayEngine._dissolve_overlay(image, 1.0, seed=12)

        original_alpha = int(np.count_nonzero(image[:, :, 3]))
        middle_alpha = int(np.count_nonzero(middle[:, :, 3]))
        end_alpha = int(np.count_nonzero(end[:, :, 3]))
        self.assertGreater(middle_alpha, 0)
        self.assertLess(middle_alpha, original_alpha)
        self.assertLess(end_alpha, middle_alpha)

    def test_visibility_mask_limits_overlay_to_person_silhouette(self) -> None:
        engine = OverlayEngine()
        engine.assets["shape"] = np.full((20, 20, 4), 255, dtype=np.uint8)
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.float32)
        mask[:, :10] = 1.0

        engine.draw(
            frame,
            "shape",
            center=(10, 10),
            target_width=20,
            opacity=1.0,
            visibility_mask=mask,
        )

        self.assertGreater(int(frame[:, :10].sum()), 0)
        self.assertEqual(int(frame[:, 10:].sum()), 0)


if __name__ == "__main__":
    unittest.main()
