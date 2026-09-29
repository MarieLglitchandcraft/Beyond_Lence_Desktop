import unittest
from unittest.mock import patch
import sys
from types import SimpleNamespace

import cv2
import numpy as np

from display_output import (
    DisplayInfo,
    calculate_windowed_size,
    create_error_frame,
    create_display_window,
    detect_display_info,
    fit_frame_to_canvas,
    OpenCVDisplayWindow,
    PygameDisplayWindow,
    prepare_display_frame,
)


class DisplayOutputTests(unittest.TestCase):
    def test_emergency_switch_produces_a_completely_black_frame(self):
        frame = np.full((8, 12, 3), 217, dtype=np.uint8)

        result = prepare_display_frame(
            frame,
            emergency_off=True,
        )

        self.assertEqual(result.shape, frame.shape)
        self.assertFalse(np.any(result))

    def test_xrandr_primary_output_is_detected(self):
        outputs = {
            "xrandr": (
                "DSI-1 connected 800x480+1920+0\n"
                "HDMI-1 connected primary 1920x1080+0+0\n"
            ),
        }

        def command_runner(command, environment):
            return outputs.get(command[0], "")

        display = detect_display_info(
            environ={"DISPLAY": ":0"},
            command_runner=command_runner,
            sysfs_root="/does-not-exist",
            use_tkinter=False,
        )

        self.assertEqual(display.size, (1920, 1080))
        self.assertEqual(display.source, "xrandr")
        self.assertEqual(display.name, "HDMI-1")
        self.assertEqual((display.x, display.y), (0, 0))

    def test_wayland_current_output_is_detected(self):
        output = (
            "HDMI-A-1 \"Unknown display\"\n"
            "  Enabled: yes\n"
            "  Position: 30,20\n"
            "  Modes:\n"
            "    1280x800 px, 59.910000 Hz (current, preferred)\n"
        )

        def command_runner(command, environment):
            return output if command[0] == "wlr-randr" else ""

        display = detect_display_info(
            environ={"WAYLAND_DISPLAY": "wayland-0"},
            command_runner=command_runner,
            sysfs_root="/does-not-exist",
            use_tkinter=False,
        )

        self.assertEqual(display.size, (1280, 800))
        self.assertEqual(display.source, "wlr-randr")
        self.assertEqual(display.name, "HDMI-A-1")
        self.assertEqual((display.x, display.y), (30, 20))

    def test_explicit_display_override_is_supported_for_diagnosis(self):
        display = detect_display_info(
            environ={"BEYOND_LENCE_DISPLAY_SIZE": "1366x768"},
            sysfs_root="/does-not-exist",
            use_tkinter=False,
        )

        self.assertEqual(display.size, (1366, 768))
        self.assertEqual(display.source, "environment override")

    def test_detection_uses_safe_fallback(self):
        display = detect_display_info(
            environ={},
            command_runner=lambda command, environment: "",
            sysfs_root="/does-not-exist",
            use_tkinter=False,
        )

        self.assertEqual(display.size, (1024, 600))
        self.assertEqual(display.source, "fallback")

    def test_portrait_display_is_reported_but_not_rotated(self):
        display = DisplayInfo(600, 1024, "test")

        self.assertTrue(display.is_portrait)
        self.assertEqual(display.orientation, "hochkant")

    def test_frame_is_fitted_without_stretching_or_cutting(self):
        frame = np.full((600, 1024, 3), 180, dtype=np.uint8)

        result = fit_frame_to_canvas(frame, (1920, 1080))

        self.assertEqual(result.shape, (1080, 1920, 3))
        self.assertFalse(np.any(result[0, 0]))
        self.assertTrue(np.all(result[540, 960] == 180))
        visible_rows = np.flatnonzero(np.any(result, axis=(1, 2)))
        visible_width = int(round(1024 * (1080 / 600)))
        expected_left = (1920 - visible_width) // 2
        self.assertEqual(visible_rows[0], 0)
        self.assertEqual(visible_rows[-1], 1079)
        self.assertFalse(np.any(result[:, :expected_left]))

    def test_target_size_is_used_for_frame_and_emergency_frame(self):
        frame = np.full((60, 100, 3), 51, dtype=np.uint8)

        normal = prepare_display_frame(
            frame,
            target_size=(500, 300),
        )
        emergency = prepare_display_frame(
            frame,
            emergency_off=True,
            target_size=(500, 300),
        )

        self.assertEqual(normal.shape, (300, 500, 3))
        self.assertEqual(emergency.shape, (300, 500, 3))
        self.assertFalse(np.any(emergency))

    def test_windowed_size_preserves_camera_aspect_ratio(self):
        size = calculate_windowed_size((1920, 1080))

        self.assertEqual(size, (1512, 886))
        self.assertAlmostEqual(size[0] / size[1], 1024 / 600, places=2)

    def test_error_frame_is_visible_and_has_requested_size(self):
        frame = create_error_frame(
            (800, 480),
            "KEIN KAMERABILD",
            ("Kamera pruefen.",),
        )

        self.assertEqual(frame.shape, (480, 800, 3))
        self.assertTrue(np.any(frame))

    @patch("display_output.cv2.setWindowProperty")
    @patch("display_output.cv2.resizeWindow")
    @patch("display_output.cv2.namedWindow")
    def test_fullscreen_window_uses_plain_gui_and_detected_size(
        self,
        named_window,
        resize_window,
        set_window_property,
    ):
        display = DisplayInfo(1280, 800, "test")
        window = OpenCVDisplayWindow("Beyond Lence", display, fullscreen=True)

        window.open()

        flags = named_window.call_args.args[1]
        self.assertEqual(
            flags,
            cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_GUI_NORMAL", 0x10),
        )
        resize_window.assert_called_once_with("Beyond Lence", 1280, 800)
        set_window_property.assert_called_once()

    def test_linux_uses_pygame_instead_of_opencv_highgui(self):
        fake_pygame = SimpleNamespace()
        with (
            patch("display_output.sys.platform", "linux"),
            patch.dict(sys.modules, {"pygame": fake_pygame}),
        ):
            window = create_display_window(
                "Beyond Lence",
                DisplayInfo(1280, 800, "test"),
                environ={},
            )

        self.assertIsInstance(window, PygameDisplayWindow)
        self.assertIs(window.pygame, fake_pygame)

    def test_opencv_backend_can_still_be_selected_for_diagnosis(self):
        with patch("display_output.sys.platform", "linux"):
            window = create_display_window(
                "Beyond Lence",
                DisplayInfo(1280, 800, "test"),
                environ={"BEYOND_LENCE_DISPLAY_BACKEND": "opencv"},
            )

        self.assertIsInstance(window, OpenCVDisplayWindow)


if __name__ == "__main__":
    unittest.main()
