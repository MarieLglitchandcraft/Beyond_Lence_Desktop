import unittest
import random
import tempfile
from pathlib import Path

import cv2
import numpy as np

from config import MODE_BEYOND, MODE_GLITCH, MODE_NORMAL, MODE_SOFT
from effects import EffectEngine


class EffectEngineTests(unittest.TestCase):
    def test_frequency_zero_preserves_normal_frame(self) -> None:
        frame = np.full((80, 120, 3), 100, dtype=np.uint8)
        result = EffectEngine().apply(frame, [MODE_NORMAL], "0")
        np.testing.assert_array_equal(result, frame)

    def test_frequency_alone_does_not_glitch_normal_camera(self) -> None:
        frame = np.full((120, 180, 3), 100, dtype=np.uint8)
        result = EffectEngine().apply(frame, [MODE_NORMAL], "V")

        np.testing.assert_array_equal(result, frame)

    def test_from_beyond_is_silent_at_frequency_zero(self) -> None:
        frame = np.full((120, 180, 3), 100, dtype=np.uint8)
        engine = EffectEngine(clock=lambda: 7.0)

        result = engine.apply(frame, [MODE_BEYOND], "0")

        np.testing.assert_array_equal(result, frame)

    def test_from_beyond_spectral_air_escalates_with_frequency(self) -> None:
        height, width = 120, 180
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
        frame[:, :, 1] = 110
        frame[:, :, 2] = np.linspace(255, 0, width, dtype=np.uint8)[None, :]
        engine = EffectEngine(clock=lambda: 9.0)

        low = engine.apply(frame, [MODE_BEYOND], "I")
        high = engine.apply(frame, [MODE_BEYOND], "V")
        low_difference = float(np.mean(np.abs(low.astype(int) - frame.astype(int))))
        high_difference = float(np.mean(np.abs(high.astype(int) - frame.astype(int))))

        self.assertGreater(low_difference, 0.1)
        self.assertGreater(high_difference, low_difference * 1.5)
        self.assertGreater(float(np.std(high)), 20.0)

    def test_glitch_is_owned_by_mode_three(self) -> None:
        frame = np.full((120, 180, 3), 100, dtype=np.uint8)
        engine = EffectEngine(clock=lambda: 0.0, noise_seed=4)
        normal = engine.apply_display_glitch(frame, [MODE_NORMAL], "V", frame)
        result = engine.apply_display_glitch(frame, [MODE_GLITCH], "V", frame)

        np.testing.assert_array_equal(normal, frame)
        self.assertEqual(result.shape, frame.shape)
        self.assertGreater(float(np.mean(np.abs(result.astype(int) - frame))), 0.5)

    def test_glitch_frequency_zero_has_no_visible_effect(self) -> None:
        frame = np.full((120, 180, 3), 100, dtype=np.uint8)
        engine = EffectEngine(clock=lambda: 12.0, noise_seed=5)

        result = engine.apply_display_glitch(
            frame,
            [MODE_GLITCH],
            "0",
            frame,
        )

        np.testing.assert_array_equal(result, frame)
        self.assertIsNone(engine._rift_frame)
        self.assertIsNone(engine._next_rift_time)

    def test_glitch_opens_an_intermittent_parallel_world(self) -> None:
        now = [0.0]
        frame = np.zeros((140, 220, 3), dtype=np.uint8)
        frame[:, ::11] = (30, 180, 70)
        frame[::11, :] = (80, 220, 120)
        engine = EffectEngine(
            clock=lambda: now[0],
            random_source=random.Random(8),
            noise_seed=8,
        )

        ambient = engine.apply_display_glitch(
            frame,
            [MODE_GLITCH],
            "V",
            frame,
        )
        now[0] = float(engine._next_rift_time) + 0.001
        rift = engine.apply_display_glitch(
            frame,
            [MODE_GLITCH],
            "V",
            frame,
        )

        self.assertIsNotNone(engine._rift_frame)
        self.assertGreater(
            float(np.mean(np.abs(rift.astype(int) - ambient.astype(int)))),
            8.0,
        )

        engine.apply_display_glitch(frame, [MODE_NORMAL], "V", frame)
        self.assertIsNone(engine._rift_frame)
        self.assertIsNone(engine._next_rift_time)

    def test_parallel_world_uses_scene_images_when_available(self) -> None:
        source = np.zeros((90, 160, 3), dtype=np.uint8)
        source[:] = (0, 210, 0)
        other_world = np.zeros((90, 160, 4), dtype=np.uint8)
        other_world[:] = (12, 20, 225, 255)

        with tempfile.TemporaryDirectory() as directory:
            scene_directory = Path(directory) / "scene_01"
            scene_directory.mkdir()
            self.assertTrue(
                cv2.imwrite(
                    str(scene_directory / "00_background.png"),
                    other_world,
                ),
            )
            now = [0.0]
            engine = EffectEngine(
                clock=lambda: now[0],
                random_source=random.Random(3),
                noise_seed=3,
                other_world_directory=directory,
            )
            engine.apply_display_glitch(
                source,
                [MODE_GLITCH],
                "V",
                source,
                {},
            )
            now[0] = float(engine._next_rift_time) + 0.001
            engine.apply_display_glitch(
                source,
                [MODE_GLITCH],
                "V",
                source,
                {},
            )
            result = engine._rift_frame

        self.assertEqual(engine._rift_scene_index, 0)
        self.assertGreater(
            float(np.mean(result[:, :, 2])),
            float(np.mean(result[:, :, 1])) * 3.0,
        )

    def test_parallel_world_scene_persists_between_repeated_flashes(self) -> None:
        now = [0.0]
        frame = np.full((90, 160, 3), 80, dtype=np.uint8)
        engine = EffectEngine(
            clock=lambda: now[0],
            random_source=random.Random(11),
            noise_seed=11,
        )

        engine.apply_display_glitch(frame, [MODE_GLITCH], "V", frame, {})
        selected_scene = engine._rift_scene_index
        now[0] = float(engine._next_rift_time) + 0.001
        engine.apply_display_glitch(frame, [MODE_GLITCH], "V", frame, {})
        now[0] = float(engine._rift_end_time) + 0.001
        engine.apply_display_glitch(frame, [MODE_GLITCH], "V", frame, {})
        now[0] = float(engine._next_rift_time) + 0.001
        engine.apply_display_glitch(frame, [MODE_GLITCH], "V", frame, {})

        self.assertIsNotNone(selected_scene)
        self.assertEqual(engine._rift_scene_index, selected_scene)

    def test_witch_house_warps_room_but_preserves_person(self) -> None:
        frame = np.zeros((140, 220, 3), dtype=np.uint8)
        frame[:, ::20] = 220
        frame[::20, :] = 220
        person_mask = np.zeros((140, 220), dtype=np.float32)
        person_mask[35:115, 80:145] = 1.0
        analysis = {"person_mask": person_mask}

        result = EffectEngine(clock=lambda: 8.0).apply(
            frame,
            [MODE_SOFT],
            "III",
            analysis,
        )

        np.testing.assert_array_equal(
            result[45:105, 90:135],
            frame[45:105, 90:135],
        )
        background_difference = np.mean(
            np.abs(result[:, :60].astype(int) - frame[:, :60].astype(int)),
        )
        self.assertGreater(float(background_difference), 1.0)

    def test_witch_house_iv_and_v_escalate_clearly(self) -> None:
        height, width = 180, 280
        horizontal = np.linspace(0, 255, width, dtype=np.uint8)
        vertical = np.linspace(0, 255, height, dtype=np.uint8)
        frame = np.empty((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = horizontal[None, :]
        frame[:, :, 1] = vertical[:, None]
        frame[:, :, 2] = (
            (
                horizontal[None, :].astype(np.uint16)
                + vertical[:, None].astype(np.uint16)
            )
            // 2
        ).astype(np.uint8)
        engine = EffectEngine(clock=lambda: 11.0)

        differences = {}
        for frequency in ("III", "IV", "V"):
            result = engine.apply(frame, [MODE_SOFT], frequency)
            differences[frequency] = float(
                np.mean(np.abs(result.astype(int) - frame.astype(int))),
            )

        self.assertGreater(differences["IV"], differences["III"] * 1.20)
        self.assertGreater(differences["V"], differences["IV"] * 1.05)

    def test_witch_house_ignores_the_soft_outer_mask_halo(self) -> None:
        original = np.full((80, 100, 3), 180, dtype=np.uint8)
        effect = np.full((80, 100, 3), 40, dtype=np.uint8)
        person_mask = np.zeros((80, 100), dtype=np.float32)
        person_mask[20:60, 25:75] = 0.20
        person_mask[28:52, 35:65] = 1.0

        restored = EffectEngine._restore_person(
            effect,
            original,
            {"person_mask": person_mask},
        )

        np.testing.assert_array_equal(restored[30:50, 40:60], original[30:50, 40:60])
        np.testing.assert_array_equal(restored[22:26, 28:34], effect[22:26, 28:34])


if __name__ == "__main__":
    unittest.main()
