import random
import unittest

from config import MODE_BEYOND, MODE_ENTITIES, MODE_GLITCH, MODE_NORMAL, MODE_SOFT
from sound import PIEZO_PIN, GeigerBuzzer, calculate_anomaly_activity


class FakePWMDevice:
    def __init__(self):
        self.frequency = 0
        self.value = 0.0
        self.off_calls = 0
        self.closed = False

    def off(self):
        self.value = 0.0
        self.off_calls += 1

    def close(self):
        self.closed = True


class SoundTests(unittest.TestCase):
    def test_piezo_uses_final_gpio_pin(self):
        self.assertEqual(PIEZO_PIN, 14)

    def test_frequency_zero_and_normal_mode_are_silent(self):
        self.assertEqual(
            calculate_anomaly_activity("0", [MODE_BEYOND], entity_count=8),
            0.0,
        )
        self.assertEqual(
            calculate_anomaly_activity("V", [MODE_NORMAL], entity_count=8),
            0.0,
        )

    def test_activity_rises_with_frequency_and_visible_entities(self):
        low = calculate_anomaly_activity(
            "I",
            [MODE_ENTITIES],
            entity_count=0,
        )
        high = calculate_anomaly_activity(
            "V",
            [MODE_ENTITIES],
            entity_count=6,
        )
        beyond = calculate_anomaly_activity(
            "V",
            [MODE_BEYOND],
            entity_count=8,
            portal_count=2,
        )

        self.assertGreater(high, low)
        self.assertGreater(beyond, high)
        self.assertLessEqual(beyond, 1.0)

    def test_combined_modes_are_more_active(self):
        one_mode = calculate_anomaly_activity("III", [MODE_SOFT])
        combined = calculate_anomaly_activity(
            "III",
            [MODE_SOFT, MODE_GLITCH],
        )
        self.assertGreater(combined, one_mode)

    def test_buzzer_update_and_click_use_pwm_device(self):
        device = FakePWMDevice()
        detector_levels = []
        buzzer = GeigerBuzzer(
            device=device,
            detector_callback=detector_levels.append,
            rng=random.Random(7),
            start_worker=False,
        )

        buzzer.update("V", [MODE_BEYOND], entity_count=8, portal_count=2)
        self.assertEqual(buzzer.activity, 1.0)
        buzzer._emit_click(buzzer.activity)

        self.assertGreaterEqual(device.frequency, 1800)
        self.assertLessEqual(device.frequency, 2900)
        self.assertGreaterEqual(device.off_calls, 1)
        self.assertEqual(device.value, 0.0)
        self.assertEqual(detector_levels[0], 1.0)
        self.assertEqual(detector_levels[-1], 0.0)
        self.assertEqual(len(detector_levels) % 2, 0)

        buzzer.close()
        self.assertTrue(device.closed)


if __name__ == "__main__":
    unittest.main()
