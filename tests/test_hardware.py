from datetime import datetime
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from camera import PiCamera, validate_camera_frame
from config import MODE_BEYOND, MODE_NORMAL, MODE_SOFT
from controls import DebouncedBoolean, GPIOControls
from photo_capture import PhotoCapture
from platform_utils import is_raspberry_pi


class FakeButton:
    instances = {}

    def __init__(self, pin, **kwargs):
        self.pin = pin
        self.kwargs = kwargs
        self.is_pressed = False
        self.when_pressed = None
        self.closed = False
        self.instances[pin] = self

    def press(self):
        self.is_pressed = True
        if self.when_pressed is not None:
            self.when_pressed()

    def close(self):
        self.closed = True


class FakeLED:
    instances = {}

    def __init__(self, pin):
        self.pin = pin
        self.value = False
        self.closed = False
        self.instances[pin] = self

    def on(self):
        self.value = True

    def off(self):
        self.value = False

    def close(self):
        self.closed = True


class FakeEncoder:
    def __init__(self, a, b, **kwargs):
        self.a = a
        self.b = b
        self.when_rotated_clockwise = None
        self.when_rotated_counter_clockwise = None
        self.closed = False

    def clockwise(self):
        self.when_rotated_clockwise()

    def counter_clockwise(self):
        self.when_rotated_counter_clockwise()

    def close(self):
        self.closed = True


class FakePicamera2:
    def __init__(self):
        self.configuration = None
        self.started = False
        self.stopped = False
        self.closed = False

    def create_video_configuration(self, **kwargs):
        return kwargs

    def configure(self, configuration):
        self.configuration = configuration

    def start(self):
        self.started = True

    def capture_array(self, stream):
        self.stream = stream
        return np.full((600, 1024, 3), 73, dtype=np.uint8)

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class FakeClock:
    def __init__(self, value=0.0):
        self.value = float(value)

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += float(seconds)


class HardwareTests(unittest.TestCase):
    def setUp(self):
        FakeButton.instances = {}
        FakeLED.instances = {}

    def make_controls(self, **kwargs):
        return GPIOControls(
            button_factory=FakeButton,
            led_factory=FakeLED,
            encoder_factory=FakeEncoder,
            **kwargs,
        )

    def test_gpio_pin_mapping_modes_and_beyond_exclusivity(self):
        controls = self.make_controls()
        self.assertEqual(controls.get_active_modes(), [MODE_NORMAL])

        FakeButton.instances[17].is_pressed = True
        self.assertEqual(controls.get_active_modes(), [MODE_SOFT])

        FakeButton.instances[23].is_pressed = True
        self.assertEqual(controls.get_active_modes(), [MODE_SOFT])

        FakeButton.instances[17].is_pressed = False
        self.assertEqual(controls.get_active_modes(), [MODE_BEYOND])
        controls.close()

    def test_encoder_changes_frequency_without_driving_detector_leds(self):
        controls = self.make_controls()
        self.assertFalse(any(led.value for led in controls.detector_leds))

        controls.encoder.clockwise()
        controls.encoder.clockwise()
        controls.encoder.clockwise()
        self.assertEqual(controls.get_frequency(), "III")
        self.assertFalse(any(led.value for led in controls.detector_leds))

        FakeButton.instances[13].press()
        self.assertEqual(controls.get_frequency(), "0")
        self.assertFalse(any(led.value for led in controls.detector_leds))
        controls.close()

    def test_detector_leds_show_click_strength_and_turn_off(self):
        controls = self.make_controls()

        controls.set_detector_level(0.15)
        self.assertEqual(
            [led.value for led in controls.detector_leds],
            [True, False, False, False, False],
        )

        controls.set_detector_level(0.55)
        self.assertEqual(
            [led.value for led in controls.detector_leds],
            [True, True, True, False, False],
        )

        controls.set_detector_level(1.0)
        self.assertTrue(all(led.value for led in controls.detector_leds))

        controls.set_detector_level(0.0)
        self.assertFalse(any(led.value for led in controls.detector_leds))
        controls.close()

    def test_photo_button_emits_one_debounced_request(self):
        controls = self.make_controls()
        FakeButton.instances[24].press()

        self.assertTrue(controls.consume_photo_request())
        self.assertFalse(controls.consume_photo_request())
        controls.close()

    def test_reed_sensor_enables_selfie_mode_only_when_unfolded(self):
        controls = self.make_controls()

        FakeButton.instances[18].is_pressed = True
        self.assertFalse(controls.is_selfie_mode())

        FakeButton.instances[18].is_pressed = False
        self.assertTrue(controls.is_selfie_mode())
        controls.close()

    def test_aircraft_switch_blacks_out_display(self):
        clock = FakeClock()
        controls = self.make_controls(clock=clock)

        self.assertEqual(controls.DISPLAY_KILL_PIN, 19)
        self.assertFalse(FakeButton.instances[19].kwargs["pull_up"])
        self.assertEqual(FakeButton.instances[19].kwargs["bounce_time"], 0.12)

        FakeButton.instances[19].is_pressed = False
        self.assertFalse(controls.is_display_emergency_off())

        FakeButton.instances[19].is_pressed = True
        self.assertFalse(controls.is_display_emergency_off())
        clock.advance(0.24)
        self.assertFalse(controls.is_display_emergency_off())
        clock.advance(0.01)
        self.assertTrue(controls.is_display_emergency_off())
        controls.close()

    def test_aircraft_switch_ignores_rapid_contact_bounce(self):
        clock = FakeClock()
        state = DebouncedBoolean(False, settle_time=0.25, clock=clock)

        self.assertFalse(state.update(True)[0])
        clock.advance(0.08)
        self.assertFalse(state.update(False)[0])
        clock.advance(0.08)
        self.assertFalse(state.update(True)[0])
        clock.advance(0.24)
        self.assertFalse(state.update(True)[0])
        clock.advance(0.01)
        stable, _, changed = state.update(True)

        self.assertTrue(stable)
        self.assertTrue(changed)

    def test_picamera2_backend_returns_opencv_frame_and_closes(self):
        camera = PiCamera(camera_factory=FakePicamera2)

        frame = camera.read()

        self.assertEqual(frame.shape, (600, 1024, 3))
        self.assertEqual(camera.last_raw_shape, (600, 1024, 3))
        self.assertEqual(camera.last_output_shape, (600, 1024, 3))
        self.assertTrue(camera.camera.started)
        self.assertEqual(
            camera.camera.configuration["main"]["format"],
            "RGB888",
        )
        camera.release()
        self.assertTrue(camera.camera.stopped)
        self.assertTrue(camera.camera.closed)

    def test_camera_frame_validation_rejects_missing_or_malformed_frames(self):
        self.assertFalse(validate_camera_frame(None)[0])
        self.assertFalse(validate_camera_frame(np.zeros((10, 10)))[0])
        self.assertFalse(validate_camera_frame(np.zeros((0, 0, 3)))[0])
        self.assertTrue(validate_camera_frame(np.zeros((10, 10, 3)))[0])

    def test_photo_capture_uses_timestamped_png(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = PhotoCapture(
                directory,
                clock=lambda: datetime(2026, 7, 15, 9, 8, 7, 654321),
            )
            frame = np.full((30, 40, 3), 120, dtype=np.uint8)

            path = capture.save(frame)

            self.assertEqual(
                path.name,
                "beyond_lence_2026-07-15_09-08-07_654321.png",
            )
            self.assertTrue(path.is_file())
            loaded = cv2.imread(str(path))
            np.testing.assert_array_equal(loaded, frame)

    def test_pi_detection_reads_device_model(self):
        with tempfile.TemporaryDirectory() as directory:
            model_file = Path(directory) / "model"
            model_file.write_text("Raspberry Pi 5 Model B Rev 1.0")
            self.assertTrue(is_raspberry_pi(model_file))


if __name__ == "__main__":
    unittest.main()
