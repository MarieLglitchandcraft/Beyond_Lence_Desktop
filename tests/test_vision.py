import unittest

import cv2
import numpy as np

from vision import VisionEngine


class NoPersonVision:
    status = "test"

    def analyze(self, frame):
        return {
            "faces": [],
            "bodies": [],
            "pose_landmarks": [],
            "person_mask": None,
            "person_detection_available": False,
        }

    def close(self):
        pass


class VisionEngineTests(unittest.TestCase):
    def test_selfie_debug_geometry_is_mirrored_but_text_can_be_redrawn(self):
        mask = np.zeros((4, 10), dtype=np.float32)
        mask[:, 1:3] = 1.0
        analysis = {
            "person_mask": mask,
            "overlay_regions": [{"box": (1, 2, 3, 4), "center": (2, 4)}],
            "dark_portals": [{"center": (3.0, 5.0), "angle": 12.0}],
            "bodies": [(1, 2, 3, 4)],
            "faces": [(4, 5, 2, 3)],
            "camera_motion": (2.5, -1.5),
        }

        mirrored = VisionEngine.mirror_analysis_for_display(analysis, 10)

        self.assertEqual(mirrored["overlay_regions"][0]["box"], (6, 2, 3, 4))
        self.assertEqual(mirrored["overlay_regions"][0]["center"], (7, 4))
        self.assertEqual(mirrored["dark_portals"][0]["center"], (6.0, 5.0))
        self.assertEqual(mirrored["dark_portals"][0]["angle"], -12.0)
        self.assertEqual(mirrored["bodies"], [(6, 2, 3, 4)])
        self.assertEqual(mirrored["faces"], [(4, 5, 2, 3)])
        self.assertEqual(mirrored["camera_motion"], (-2.5, -1.5))
        np.testing.assert_array_equal(
            mirrored["person_mask"],
            cv2.flip(mask, 1),
        )

    def test_dark_bright_and_neutral_areas_become_candidates(self) -> None:
        frame = np.full((400, 600, 3), 125, dtype=np.uint8)
        frame[:, :200] = 55
        frame[:, 400:] = 225

        analysis = VisionEngine(person_vision=NoPersonVision()).analyze(frame)
        sources = {item["source"] for item in analysis["overlay_regions"]}

        self.assertTrue(analysis["dark_regions"])
        self.assertTrue(analysis["bright_regions"])
        self.assertIn("dark", sources)
        self.assertIn("bright", sources)
        self.assertGreaterEqual(len(analysis["overlay_regions"]), 20)

    def test_person_mask_moves_overlay_candidates_away_from_people(self) -> None:
        frame = np.full((400, 600, 3), 80, dtype=np.uint8)
        person_mask = np.zeros((400, 600), dtype=np.float32)
        person_mask[:, :200] = 1.0

        class MaskedPersonVision(NoPersonVision):
            def analyze(self, frame):
                result = super().analyze(frame)
                result["faces"] = [(0, 0, 200, 400)]
                result["person_mask"] = person_mask
                result["person_detection_available"] = True
                return result

        analysis = VisionEngine(person_vision=MaskedPersonVision()).analyze(frame)
        best = analysis["overlay_regions"][0]

        self.assertGreaterEqual(best["box"][0], 200)
        self.assertEqual(best["person_coverage"], 0.0)

    def test_dark_oval_becomes_portal_only_after_temporal_confirmation(self) -> None:
        frame = np.full((480, 640, 3), 170, dtype=np.uint8)
        cv2.ellipse(frame, (180, 220), (62, 88), 12, 0, 360, (12, 12, 12), -1)
        engine = VisionEngine(person_vision=NoPersonVision())

        first = engine.analyze(frame)
        self.assertTrue(first["dark_oval_candidates"])
        self.assertFalse(first["dark_portals"])

        for _ in range(3):
            confirmed = engine.analyze(frame)

        self.assertEqual(len(confirmed["dark_portals"]), 1)
        center_x, center_y = confirmed["dark_portals"][0]["center"]
        self.assertAlmostEqual(center_x, 180, delta=3)
        self.assertAlmostEqual(center_y, 220, delta=3)

    def test_dark_oval_inside_person_mask_is_not_a_portal(self) -> None:
        frame = np.full((360, 480, 3), 170, dtype=np.uint8)
        cv2.ellipse(frame, (240, 180), (55, 72), 0, 0, 360, (8, 8, 8), -1)
        person_mask = np.zeros((360, 480), dtype=np.float32)
        person_mask[90:280, 160:320] = 1.0

        engine = VisionEngine(person_vision=NoPersonVision())
        candidates = engine.detect_dark_ovals(frame[:, :, 0], person_mask=person_mask)

        self.assertFalse(candidates)


if __name__ == "__main__":
    unittest.main()
