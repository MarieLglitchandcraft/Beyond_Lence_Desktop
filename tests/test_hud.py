import unittest

import numpy as np

from config import MODE_BEYOND, MODE_ENTITIES, MODE_GLITCH, MODE_NORMAL, MODE_SOFT
from hud import HUD


class HUDTests(unittest.TestCase):
    def test_formats_normal_and_single_modes(self):
        self.assertEqual(HUD.format_active_modes([MODE_NORMAL]), "MODUS NORMAL")
        self.assertEqual(HUD.format_active_modes([MODE_SOFT]), "MODUS HEXENHAUS")
        self.assertEqual(HUD.format_active_modes([MODE_BEYOND]), "MODUS FROM BEYOND")

    def test_formats_combined_modes_in_switch_order(self):
        self.assertEqual(
            HUD.format_active_modes([MODE_SOFT, MODE_ENTITIES, MODE_GLITCH]),
            "MODI HEXENHAUS + ENTITAETEN + GLITCH",
        )

    def test_active_hud_draws_mode_information(self):
        frame = np.zeros((120, 1024, 3), dtype=np.uint8)
        result = HUD().draw_active_hud(
            frame,
            runtime=12,
            active_modes=[MODE_ENTITIES],
        )

        self.assertGreater(np.count_nonzero(result), 0)

    def test_capture_confirmation_is_time_limited(self):
        now = [10.0]
        hud = HUD(clock=lambda: now[0])
        hud.show_capture_confirmation(duration=1.0)
        active = np.zeros((120, 1024, 3), dtype=np.uint8)
        expired = active.copy()

        hud.draw_active_hud(active, runtime=12, active_modes=[MODE_NORMAL])
        now[0] = 11.1
        hud.draw_active_hud(expired, runtime=12, active_modes=[MODE_NORMAL])

        self.assertGreater(np.count_nonzero(active), np.count_nonzero(expired))


if __name__ == "__main__":
    unittest.main()
