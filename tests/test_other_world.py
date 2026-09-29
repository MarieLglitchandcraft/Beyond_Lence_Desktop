import random
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from other_world import LayeredOtherWorld


class LayeredOtherWorldTests(unittest.TestCase):
    def test_requested_far_layer_movements_are_configured(self) -> None:
        behaviors = LayeredOtherWorld.FAR_BEHAVIORS

        self.assertGreater(behaviors["scene_01"]["rise_rate"], 0.0)
        self.assertGreater(behaviors["scene_01"]["zoom_rate"], 0.0)
        self.assertEqual(behaviors["scene_02"]["rise_rate"], 0.0)
        self.assertGreater(behaviors["scene_02"]["zoom_rate"], 0.0)
        self.assertGreater(behaviors["scene_03"]["rise_rate"], 0.0)
        self.assertGreater(behaviors["scene_04"]["rise_rate"], 0.0)

    def test_camera_translation_keeps_layers_anchored_to_the_room(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scene_directory = Path(directory) / "scene_01"
            scene_directory.mkdir()

            background = np.zeros((60, 100, 4), dtype=np.uint8)
            background[:, :, 3] = 255
            foreground = np.zeros_like(background)
            foreground[22:38, 36:52] = (240, 240, 240, 255)
            cv2.imwrite(
                str(scene_directory / "00_background.png"),
                background,
            )
            cv2.imwrite(
                str(scene_directory / "01_far.png"),
                foreground,
            )

            library = LayeredOtherWorld(directory, random.Random(2))
            identity = np.eye(3, dtype=np.float32)
            translated = identity.copy()
            translated[0, 2] = 12.0
            still = library.render(0, 100, 60, 0.0, identity, 0.0, 7)
            moved = library.render(0, 100, 60, 0.0, translated, 0.0, 7)

        still_x = float(np.mean(np.where(still[:, :, 0] > 100)[1]))
        moved_x = float(np.mean(np.where(moved[:, :, 0] > 100)[1]))
        self.assertGreater(moved_x - still_x, 8.0)

    def test_four_layers_are_discovered_in_depth_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scene_directory = Path(directory) / "scene_01"
            scene_directory.mkdir()
            layer = np.zeros((20, 30, 4), dtype=np.uint8)
            for name in (
                "03_foreground.png",
                "01_far.png",
                "00_background.png",
                "02_midground.png",
            ):
                cv2.imwrite(str(scene_directory / name), layer)

            library = LayeredOtherWorld(directory, random.Random(2))

        names = [path.name for path in library.scenes[0].layers]
        self.assertEqual(
            names,
            [
                "00_background.png",
                "01_far.png",
                "02_midground.png",
                "03_foreground.png",
            ],
        )


if __name__ == "__main__":
    unittest.main()
