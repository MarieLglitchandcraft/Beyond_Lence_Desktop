import unittest
from unittest.mock import patch

import numpy as np

from config import MODE_BEYOND, MODE_ENTITIES, MODE_NORMAL, MODE_SOFT
from entity_manager import (
    BEHAVIOR_BRUSH,
    BEHAVIOR_CROSS,
    BEHAVIOR_DRIFT,
    BEHAVIOR_EMERGE,
    BEHAVIOR_FLOAT,
    BEHAVIOR_GLIMPSE,
    BEHAVIOR_LURK,
    BEHAVIOR_PORTAL,
    BEHAVIOR_RIFT,
    BEHAVIOR_SWIM,
    BEHAVIOR_WATCH,
    INTERACTION_CONSUMED,
    INTERACTION_DEVOURER,
    ENTITY_BEYOND_CREATURE,
    ENTITY_EYE,
    ENTITY_SHADOW,
    ENTITY_TENTACLE,
    Entity,
    EntityManager,
)
from scene_manager import SceneManager


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def quiet_scene() -> dict:
    return {
        "motion_count": 0,
        "scene_stability": 1.0,
        "is_quiet": True,
        "has_good_overlay_spot": True,
        "frame_size": (640, 480),
        "best_overlay_regions": [
            {"box": (10, 20, 120, 100), "score": 3, "center": (70, 70)},
        ],
    }


class SceneManagerTests(unittest.TestCase):
    def test_stability_and_person_counts(self) -> None:
        clock = FakeClock()
        manager = SceneManager(clock=clock)
        analysis = {
            "dark_regions": [(0, 0, 20, 20)],
            "motion_regions": [],
            "overlay_regions": [
                {"box": (index, 0, 10, 10), "score": 3, "center": (0, 0)}
                for index in range(4)
            ],
            "faces": [(1, 2, 3, 4)],
            "bodies": [],
        }

        initial = manager.evaluate(analysis)
        self.assertEqual(initial["scene_stability"], 0.0)
        self.assertFalse(initial["is_quiet"])
        self.assertTrue(initial["person_present"])
        self.assertEqual(initial["face_count"], 1)
        self.assertEqual(initial["body_count"], 0)
        self.assertEqual(len(initial["best_overlay_regions"]), 4)

        clock.advance(4.3)
        quiet = manager.evaluate(analysis)
        self.assertTrue(quiet["is_quiet"])

        analysis["motion_regions"] = [(0, 0, 30, 30)]
        moving = manager.evaluate(analysis)
        self.assertEqual(moving["scene_stability"], 0.0)
        self.assertFalse(moving["is_quiet"])


class EntityManagerTests(unittest.TestCase):
    def test_witch_house_and_entities_increase_spawn_chance(self) -> None:
        profile = {"spawn_chance": 0.4}
        scene = {"motion_count": 0}

        normal = EntityManager.get_spawn_chance(
            profile,
            scene,
            [MODE_ENTITIES],
        )
        witch_house = EntityManager.get_spawn_chance(
            profile,
            scene,
            [MODE_SOFT, MODE_ENTITIES],
        )

        self.assertEqual(normal, 0.4)
        self.assertGreater(witch_house, normal)
    def test_entity_types_receive_distinct_behaviors(self) -> None:
        self.assertIn(
            EntityManager.choose_behavior(ENTITY_EYE, "III", [MODE_ENTITIES]),
            (BEHAVIOR_WATCH, BEHAVIOR_GLIMPSE),
        )
        self.assertIn(
            EntityManager.choose_behavior(ENTITY_SHADOW, "III", [MODE_ENTITIES]),
            (BEHAVIOR_CROSS, BEHAVIOR_LURK),
        )
        self.assertEqual(
            EntityManager.choose_behavior(ENTITY_TENTACLE, "III", [MODE_ENTITIES]),
            BEHAVIOR_EMERGE,
        )

    def test_crossing_entity_moves_on_a_path_without_resizing(self) -> None:
        class RecordingOverlay:
            def __init__(self):
                self.calls = []

            def load_assets(self, assets):
                pass

            def draw(self, frame, asset_name, center, target_width, opacity, **kwargs):
                self.calls.append((center, target_width, opacity))

        clock = FakeClock()
        overlay = RecordingOverlay()
        manager = EntityManager(overlay_engine=overlay, clock=clock)
        manager.entities = [
            Entity(
                entity_type=ENTITY_SHADOW,
                asset_name="normal_shadow",
                x=100,
                y=120,
                size=180,
                opacity=0.5,
                lifetime=10.0,
                intensity=0.5,
                phase=0.0,
                behavior=BEHAVIOR_CROSS,
                motion_x=120.0,
                motion_y=0.0,
                start_time=0.0,
            ),
        ]
        frame = np.zeros((300, 500, 3), dtype=np.uint8)

        clock.now = 2.0
        manager.draw(frame)
        clock.now = 6.0
        manager.draw(frame)

        first_center, first_width, _ = overlay.calls[0]
        second_center, second_width, _ = overlay.calls[1]
        self.assertGreater(second_center[0], first_center[0])
        self.assertEqual(first_center[1], second_center[1])
        self.assertEqual((first_width, second_width), (180, 180))

    def test_tentacle_emerges_and_retracts_once(self) -> None:
        self.assertEqual(EntityManager._emerge_reveal(0.0), 0.0)
        self.assertEqual(EntityManager._emerge_reveal(0.5), 1.0)
        self.assertEqual(EntityManager._emerge_reveal(1.0), 0.0)

    def test_camera_transform_moves_existing_entity_with_the_room(self) -> None:
        clock = FakeClock()
        manager = EntityManager(clock=clock)
        manager.entities = [
            Entity(
                entity_type=ENTITY_EYE,
                asset_name="normal_eye_dark",
                x=100,
                y=120,
                size=80,
                opacity=0.5,
                lifetime=10.0,
                intensity=0.5,
                phase=0.0,
                start_time=0.0,
            ),
        ]
        scene = quiet_scene()
        scene["has_good_overlay_spot"] = False
        scene["tracking_confidence"] = 0.9
        scene["camera_transform"] = np.array(
            [[1.0, 0.0, -14.0], [0.0, 1.0, 8.0]],
            dtype=np.float32,
        )

        manager.update([MODE_ENTITIES], "III", scene)

        self.assertEqual((manager.entities[0].x, manager.entities[0].y), (86, 128))

    def test_entities_keep_a_constant_size_instead_of_pulsing(self) -> None:
        class RecordingOverlay:
            def __init__(self):
                self.widths = []

            def load_assets(self, assets):
                pass

            def draw(self, frame, asset_name, center, target_width, opacity, **kwargs):
                self.widths.append(target_width)

        clock = FakeClock()
        overlay = RecordingOverlay()
        manager = EntityManager(overlay_engine=overlay, clock=clock)
        manager.entities = [
            Entity(
                entity_type=ENTITY_EYE,
                asset_name="normal_eye_dark",
                x=100,
                y=100,
                size=180,
                opacity=0.5,
                lifetime=10.0,
                intensity=1.0,
                phase=0.0,
                start_time=0.0,
            ),
        ]
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        clock.now = 2.0
        manager.draw(frame)
        clock.now = 5.0
        manager.draw(frame)

        self.assertEqual(overlay.widths, [180, 180])

    def test_person_mask_keeps_entities_behind_people(self) -> None:
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        protected = frame.copy()
        frame[:] = 220
        mask = np.zeros((4, 6), dtype=np.float32)
        mask[:, :3] = 1.0

        EntityManager._restore_person_layer(frame, protected, mask)

        self.assertTrue(np.all(frame[:, :3] == 0))
        self.assertTrue(np.all(frame[:, 3:] == 220))

    def test_beyond_creature_crosses_from_front_through_person_to_back(self) -> None:
        class RecordingOverlay:
            def __init__(self):
                self.calls = []

            def load_assets(self, assets):
                pass

            def draw(self, frame, asset_name, **kwargs):
                self.calls.append(kwargs.get("visibility_mask"))

        clock = FakeClock()
        overlay = RecordingOverlay()
        manager = EntityManager(overlay_engine=overlay, clock=clock)
        manager._beyond_active = True
        manager.entities = [Entity(
            entity_type=ENTITY_BEYOND_CREATURE,
            asset_name="beyond_jellyfish",
            x=100,
            y=100,
            size=120,
            opacity=0.5,
            lifetime=10.0,
            intensity=0.8,
            phase=0.0,
            penetrates_person=True,
            person_pass_start=0.2,
            person_pass_duration=0.5,
            start_time=0.0,
        )]
        frame = np.zeros((220, 220, 3), dtype=np.uint8)
        person_mask = np.ones((220, 220), dtype=np.float32)

        clock.now = 1.0
        manager.draw(frame.copy(), person_mask)
        self.assertEqual(overlay.calls, [None])

        overlay.calls.clear()
        clock.now = 4.5
        manager.draw(frame.copy(), person_mask)
        self.assertEqual(len(overlay.calls), 2)
        self.assertIsNone(overlay.calls[0])
        self.assertIs(overlay.calls[1], person_mask)

        overlay.calls.clear()
        clock.now = 8.0
        manager.draw(frame.copy(), person_mask)
        self.assertEqual(overlay.calls, [None])

    def test_normal_and_beyond_asset_pools_are_separate(self) -> None:
        normal_asset = EntityManager.choose_asset("eye", [MODE_ENTITIES])
        beyond_asset = EntityManager.choose_asset("beyond_creature", [MODE_BEYOND])

        self.assertIsNotNone(normal_asset)
        self.assertIsNotNone(beyond_asset)
        self.assertTrue(normal_asset.startswith("normal_"))
        self.assertTrue(beyond_asset.startswith("beyond_"))

    def test_from_beyond_frequency_controls_ecology_density(self) -> None:
        zero = EntityManager.get_profile("0", [MODE_BEYOND])
        low = EntityManager.get_profile("I", [MODE_BEYOND])
        high = EntityManager.get_profile("V", [MODE_BEYOND])

        self.assertEqual(zero["max_entities"], 0)
        self.assertGreater(low["max_entities"], 0)
        self.assertGreater(high["max_entities"], low["max_entities"])
        self.assertGreater(high["intensity"], low["intensity"])

    def test_from_beyond_can_spawn_anywhere_without_detected_overlay_spots(self) -> None:
        clock = FakeClock()
        manager = EntityManager(clock=clock)
        scene = quiet_scene()
        scene["has_good_overlay_spot"] = False
        scene["best_overlay_regions"] = []

        with (
            patch("entity_manager.random.random", return_value=0.0),
            patch("entity_manager.random.choice", side_effect=lambda items: items[0]),
            patch("entity_manager.random.randint", side_effect=lambda low, high: low),
            patch("entity_manager.random.uniform", side_effect=lambda low, high: high),
        ):
            manager.update([MODE_BEYOND], "I", scene)

        self.assertEqual(len(manager.entities), 1)
        self.assertTrue(manager.entities[0].asset_name.startswith("beyond_bubble"))

    def test_from_beyond_assets_receive_distinct_organic_behaviors(self) -> None:
        self.assertEqual(
            EntityManager.choose_beyond_behavior("beyond_bubble_1"),
            BEHAVIOR_FLOAT,
        )
        self.assertEqual(
            EntityManager.choose_beyond_behavior("beyond_centipede_1"),
            BEHAVIOR_SWIM,
        )
        self.assertEqual(
            EntityManager.choose_beyond_behavior("beyond_jellyfish"),
            BEHAVIOR_DRIFT,
        )
        self.assertEqual(
            EntityManager.choose_beyond_behavior("beyond_mouth"),
            BEHAVIOR_RIFT,
        )

    def test_detected_portal_places_creature_at_opening_without_resizing_path(self) -> None:
        entity = Entity(
            entity_type=ENTITY_BEYOND_CREATURE,
            asset_name="beyond_polyp",
            x=0,
            y=0,
            size=80,
            opacity=0.5,
            lifetime=10.0,
            intensity=0.8,
            phase=0.0,
        )
        portal = {
            "center": (240.0, 180.0),
            "axes": (120.0, 90.0),
        }

        with patch("entity_manager.random.uniform", side_effect=lambda low, high: high):
            EntityManager._configure_portal_entity(entity, portal, (640, 480))

        self.assertEqual((entity.x, entity.y), (240, 180))
        self.assertEqual(entity.behavior, BEHAVIOR_PORTAL)
        self.assertTrue(entity.penetrates_person)
        self.assertNotEqual((entity.motion_x, entity.motion_y), (0.0, 0.0))

    def test_mouth_is_not_part_of_regular_beyond_spawns(self) -> None:
        with patch("entity_manager.random.choice", side_effect=lambda items: items[-1]):
            for frequency in ("IV", "V"):
                entity_type = EntityManager().choose_entity_type(
                    frequency,
                    [MODE_BEYOND],
                )
                self.assertNotEqual(entity_type, "beyond_aperture")

    def test_devour_event_moves_predator_and_dissolves_victim(self) -> None:
        manager = EntityManager(clock=lambda: 4.0)
        predator = Entity(
            entity_type=ENTITY_BEYOND_CREATURE,
            asset_name="beyond_jellyfish",
            x=80,
            y=100,
            size=140,
            opacity=0.5,
            lifetime=20.0,
            intensity=1.0,
            phase=0.2,
            start_time=0.0,
        )
        victim = Entity(
            entity_type=ENTITY_BEYOND_CREATURE,
            asset_name="beyond_polyp",
            x=330,
            y=210,
            size=120,
            opacity=0.5,
            lifetime=20.0,
            intensity=1.0,
            phase=1.2,
            start_time=0.0,
        )
        manager.entities = [predator, victim]

        with (
            patch("entity_manager.random.sample", return_value=[predator, victim]),
            patch("entity_manager.random.uniform", return_value=3.0),
        ):
            self.assertTrue(manager._start_devour_event(4.0))

        self.assertEqual(predator.interaction_role, INTERACTION_DEVOURER)
        self.assertEqual(victim.interaction_role, INTERACTION_CONSUMED)
        self.assertEqual(predator.interaction_target_x, victim.x)
        self.assertEqual(predator.interaction_target_y, victim.y)

    def test_camera_motion_can_start_an_entity_reaction(self) -> None:
        manager = EntityManager(clock=lambda: 2.0)
        entity = Entity(
            entity_type=ENTITY_BEYOND_CREATURE,
            asset_name="beyond_jellyfish",
            x=200,
            y=180,
            size=140,
            opacity=0.5,
            lifetime=20.0,
            intensity=1.0,
            phase=0.0,
            start_time=0.0,
        )
        manager.entities = [entity]
        scene = quiet_scene()
        scene["camera_motion"] = (8.0, -3.0)

        with (
            patch("entity_manager.random.random", return_value=0.0),
            patch("entity_manager.random.uniform", side_effect=lambda low, high: high),
        ):
            manager._react_to_camera(scene, 2.0, 1.0)

        self.assertGreater(entity.reaction_duration, 0.0)
        self.assertLess(entity.reaction_x, 0.0)
        self.assertGreater(entity.reaction_y, 0.0)

    def test_brush_and_mouth_are_separate_rare_event_entities(self) -> None:
        manager = EntityManager(clock=lambda: 5.0)

        with patch("entity_manager.random.uniform", side_effect=lambda low, high: high):
            self.assertTrue(manager._start_brush_event((1000, 600), 5.0, 1.0))
            self.assertTrue(manager._start_mouth_event((1000, 600), 5.0, 1.0))

        brush = next(
            entity for entity in manager.entities if entity.behavior == BEHAVIOR_BRUSH
        )
        mouth = next(
            entity for entity in manager.entities if entity.asset_name == "beyond_mouth"
        )
        self.assertTrue(brush.event_entity)
        self.assertEqual(brush.depth_layer, 2)
        self.assertGreaterEqual(brush.size, 720)
        self.assertTrue(mouth.event_entity)
        self.assertEqual(mouth.depth_layer, 1)

    def test_spawn_attempts_are_time_limited(self) -> None:
        clock = FakeClock()
        manager = EntityManager(clock=clock)

        with patch("entity_manager.random.random", return_value=1.0) as chance:
            manager.update([MODE_ENTITIES], "III", quiet_scene())
            manager.update([MODE_ENTITIES], "III", quiet_scene())
            clock.advance(0.4)
            manager.update([MODE_ENTITIES], "III", quiet_scene())
            self.assertEqual(chance.call_count, 1)

            clock.advance(0.2)
            manager.update([MODE_ENTITIES], "III", quiet_scene())
            self.assertEqual(chance.call_count, 2)

    def test_scanning_motion_allows_spawn_and_disabling_clears_entities(self) -> None:
        clock = FakeClock()
        manager = EntityManager(clock=clock)
        moving_scene = quiet_scene()
        moving_scene["motion_count"] = 1
        moving_scene["is_quiet"] = False

        with (
            patch("entity_manager.random.random", return_value=0.0),
            patch("entity_manager.random.choice", side_effect=lambda items: items[0]),
            patch("entity_manager.random.randint", side_effect=lambda low, high: low),
            patch("entity_manager.random.uniform", side_effect=lambda low, high: high),
        ):
            manager.update([MODE_ENTITIES], "III", moving_scene)

        self.assertEqual(len(manager.entities), 1)
        manager.update([MODE_NORMAL], "V", quiet_scene())
        self.assertEqual(manager.entities, [])

    def test_frequency_v_spawns_large_entities_even_with_motion(self) -> None:
        clock = FakeClock()
        manager = EntityManager(clock=clock)
        scene = quiet_scene()
        scene["frame_size"] = (1920, 1080)
        scene["motion_count"] = 2
        scene["is_quiet"] = False
        scene["scene_stability"] = 0.0

        with (
            patch("entity_manager.random.random", return_value=0.0),
            patch("entity_manager.random.choice", side_effect=lambda items: items[0]),
            patch("entity_manager.random.randint", side_effect=lambda low, high: low),
            patch("entity_manager.random.uniform", side_effect=lambda low, high: high),
        ):
            manager.update([MODE_ENTITIES], "V", scene)

        self.assertEqual(len(manager.entities), 1)
        self.assertGreaterEqual(manager.entities[0].size, int(1920 * 0.32))


if __name__ == "__main__":
    unittest.main()
