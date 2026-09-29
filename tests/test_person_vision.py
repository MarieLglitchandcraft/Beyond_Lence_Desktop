import unittest

import numpy as np

from person_vision import PersonVisionEngine


class FakeMediaPipeMask:
    def __init__(self, data):
        self.data = data

    def numpy_view(self):
        return self.data


class FakeSegmentationResult:
    def __init__(self, mask):
        self.confidence_masks = [FakeMediaPipeMask(mask)]


class PersonVisionEngineTests(unittest.TestCase):
    def test_face_fallback_does_not_expand_an_existing_person_mask(self) -> None:
        mask = np.zeros((120, 160), dtype=np.float32)
        mask[25:90, 50:110] = 0.9
        original = mask.copy()

        PersonVisionEngine._add_detection_fallbacks(
            mask,
            faces=[(55, 30, 50, 50)],
            bodies=[],
        )

        np.testing.assert_array_equal(mask, original)

    def test_selfie_segmentation_creates_body_box(self) -> None:
        confidence = np.zeros((8, 8), dtype=np.float32)
        confidence[1:7, 2:6] = 0.95
        result = FakeSegmentationResult(confidence)

        mask = PersonVisionEngine._extract_person_segmentation(
            result,
            frame_size=(80, 80),
        )
        boxes = PersonVisionEngine._boxes_from_mask(mask)

        self.assertEqual(mask.shape, (80, 80))
        self.assertGreater(float(np.max(mask)), 0.9)
        self.assertEqual(len(boxes), 1)
        self.assertGreater(boxes[0][2] * boxes[0][3], 1000)

    def test_weak_background_uncertainty_is_rejected(self) -> None:
        confidence = np.full((8, 8), 0.2, dtype=np.float32)
        result = FakeSegmentationResult(confidence)

        mask = PersonVisionEngine._extract_person_segmentation(
            result,
            frame_size=(80, 80),
        )

        self.assertFalse(np.any(mask))
        self.assertEqual(PersonVisionEngine._boxes_from_mask(mask), [])


if __name__ == "__main__":
    unittest.main()
