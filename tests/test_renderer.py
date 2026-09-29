import unittest

import cv2
import numpy as np

from renderer import Renderer


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += float(seconds)


class RendererCaptureTests(unittest.TestCase):
    def test_selfie_camera_is_mirrored_before_interface_is_drawn(self):
        frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)

        normal = Renderer.apply_selfie_camera_mirror(
            frame,
            selfie_mode=False,
        )
        selfie = Renderer.apply_selfie_camera_mirror(
            frame,
            selfie_mode=True,
        )

        self.assertIs(normal, frame)
        np.testing.assert_array_equal(selfie, cv2.flip(frame, 1))

    def test_selfie_display_rotation_turns_composed_frame_without_mirroring(self):
        frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)

        normal = Renderer.apply_selfie_display_rotation(
            frame,
            selfie_mode=False,
        )
        selfie = Renderer.apply_selfie_display_rotation(
            frame,
            selfie_mode=True,
        )

        self.assertIs(normal, frame)
        np.testing.assert_array_equal(
            selfie,
            cv2.rotate(frame, cv2.ROTATE_180),
        )

    def test_capture_feedback_flashes_freezes_photo_and_then_returns_live(self):
        clock = FakeClock()
        renderer = Renderer(monotonic_clock=clock)
        photo = np.full((300, 512, 3), 40, dtype=np.uint8)
        live = np.full((300, 512, 3), 9, dtype=np.uint8)

        renderer.show_capture_confirmation(
            photo,
            duration=2.0,
            flash_duration=0.12,
        )

        flash = renderer.apply_capture_feedback(live)
        self.assertGreater(float(flash.mean()), 200.0)

        clock.advance(0.20)
        preview = renderer.apply_capture_feedback(live)
        np.testing.assert_array_equal(preview[10, 10], photo[10, 10])
        self.assertFalse(np.array_equal(preview[150, 256], live[150, 256]))

        clock.advance(1.81)
        held_preview = renderer.apply_capture_feedback(live)
        self.assertFalse(np.array_equal(held_preview, live))

        clock.advance(0.12)
        result = renderer.apply_capture_feedback(live)
        np.testing.assert_array_equal(result, live)

    def test_capture_feedback_keeps_its_own_copy(self):
        clock = FakeClock()
        renderer = Renderer(monotonic_clock=clock)
        photo = np.full((120, 200, 3), 70, dtype=np.uint8)

        renderer.show_capture_confirmation(photo)
        photo[:] = 0
        clock.advance(0.20)
        preview = renderer.apply_capture_feedback(np.zeros_like(photo))

        np.testing.assert_array_equal(preview[5, 5], (70, 70, 70))


if __name__ == "__main__":
    unittest.main()
