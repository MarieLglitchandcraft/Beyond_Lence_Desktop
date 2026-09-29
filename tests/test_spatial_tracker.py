import unittest

import cv2
import numpy as np

from spatial_tracker import SpatialTracker


class SpatialTrackerTests(unittest.TestCase):
    def test_tracks_background_translation(self) -> None:
        random = np.random.default_rng(42)
        first = np.zeros((360, 640), dtype=np.uint8)
        for x, y in random.integers((20, 20), (620, 340), size=(130, 2)):
            cv2.circle(first, (int(x), int(y)), 3, 255, -1)

        expected_x = 13.0
        expected_y = -7.0
        second = cv2.warpAffine(
            first,
            np.array([[1.0, 0.0, expected_x], [0.0, 1.0, expected_y]]),
            (640, 360),
        )
        tracker = SpatialTracker()

        initial = tracker.update(first)
        tracked = tracker.update(second)

        self.assertEqual(initial["tracking_confidence"], 0.0)
        self.assertGreater(tracked["tracking_confidence"], 0.5)
        self.assertGreater(tracked["tracking_points"], 20)
        self.assertAlmostEqual(tracked["camera_motion"][0], expected_x, delta=1.2)
        self.assertAlmostEqual(tracked["camera_motion"][1], expected_y, delta=1.2)

    def test_person_area_is_excluded_from_scene_features(self) -> None:
        tracker = SpatialTracker()
        mask = np.zeros((100, 160), dtype=np.float32)
        mask[20:90, 50:120] = 1.0

        exclusion = tracker._prepare_exclusion_mask(mask, (100, 160))

        self.assertIsNotNone(exclusion)
        self.assertEqual(exclusion[50, 80], 255)
        self.assertEqual(exclusion[0, 0], 0)


if __name__ == "__main__":
    unittest.main()
