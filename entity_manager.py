import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import (
    BEYOND_PROFILES,
    FREQUENCY_PROFILES,
    MODE_BEYOND,
    MODE_ENTITIES,
    MODE_SOFT,
)
from overlays import OverlayEngine


ENTITY_EYE = "eye"
ENTITY_TENTACLE = "tentacle"
ENTITY_SHADOW = "shadow"
ENTITY_ATMOSPHERE = "atmosphere"
ENTITY_BEYOND_CREATURE = "beyond_creature"
ENTITY_BEYOND_APERTURE = "beyond_aperture"

BEHAVIOR_WATCH = "watch"
BEHAVIOR_LURK = "lurk"
BEHAVIOR_CROSS = "cross"
BEHAVIOR_GLIMPSE = "glimpse"
BEHAVIOR_EMERGE = "emerge"
BEHAVIOR_DRIFT = "drift"
BEHAVIOR_FLOAT = "float"
BEHAVIOR_SWIM = "swim"
BEHAVIOR_FLOW = "flow"
BEHAVIOR_RIFT = "rift"
BEHAVIOR_BRUSH = "brush"
BEHAVIOR_PORTAL = "portal_emerge"

INTERACTION_DEVOURER = "devourer"
INTERACTION_CONSUMED = "consumed"

BEYOND_EVENT_DELAYS = {
    "III": (18.0, 28.0),
    "IV": (12.0, 20.0),
    "V": (7.0, 13.0),
}

ASSET_ROOT = Path(__file__).resolve().parent / "assets"

ASSET_PATHS = {
    "normal_eye_dark": ASSET_ROOT / "normal" / "Auge_sw.png",
    "normal_eye_blue": ASSET_ROOT / "normal" / "auge_blau.png",
    "normal_eye_orange": ASSET_ROOT / "normal" / "auge_orange.png",
    "normal_tentacle_1": ASSET_ROOT / "normal" / "tentakel1.png",
    "normal_tentacle_2": ASSET_ROOT / "normal" / "tentakel2.png",
    "normal_tentacle_single": ASSET_ROOT / "normal" / "tentakel_einer.png",
    "normal_tentacle_wide": ASSET_ROOT / "normal" / "tentakel_overall.png",
    "normal_shadow": ASSET_ROOT / "normal" / "schattenwesen.png",
    "normal_polyp": ASSET_ROOT / "normal" / "polyp2.png",
    "normal_hands": ASSET_ROOT / "normal" / "haende.png",
    "normal_blood": ASSET_ROOT / "normal" / "blut.png",
    "normal_jelly": ASSET_ROOT / "normal" / "quell2.png",
    "beyond_centipede_1": ASSET_ROOT / "from_beyond" / "1000fuss.png",
    "beyond_centipede_2": ASSET_ROOT / "from_beyond" / "1000fuss2.png",
    "beyond_jellyfish": ASSET_ROOT / "from_beyond" / "Qualle.png",
    "beyond_polyp": ASSET_ROOT / "from_beyond" / "polyp.png",
    "beyond_tentacle": ASSET_ROOT / "from_beyond" / "tentakel_blau.png",
    "beyond_bubble_1": ASSET_ROOT / "from_beyond" / "blase1.png",
    "beyond_bubble_2": ASSET_ROOT / "from_beyond" / "blase2.png",
    "beyond_bubbles": ASSET_ROOT / "from_beyond" / "blasen.png",
    "beyond_mouth": ASSET_ROOT / "from_beyond" / "Mund.png",
}

NORMAL_ASSET_POOLS = {
    ENTITY_EYE: (
        "normal_eye_dark",
        "normal_eye_blue",
        "normal_eye_orange",
    ),
    ENTITY_TENTACLE: (
        "normal_tentacle_1",
        "normal_tentacle_2",
        "normal_tentacle_single",
        "normal_tentacle_wide",
    ),
    ENTITY_SHADOW: (
        "normal_shadow",
        "normal_polyp",
    ),
    ENTITY_ATMOSPHERE: (
        "normal_hands",
        "normal_blood",
        "normal_jelly",
    ),
}

BEYOND_ASSET_POOLS = {
    ENTITY_TENTACLE: ("beyond_tentacle",),
    ENTITY_ATMOSPHERE: (
        "beyond_bubble_1",
        "beyond_bubble_2",
        "beyond_bubbles",
    ),
    ENTITY_BEYOND_CREATURE: (
        "beyond_centipede_1",
        "beyond_centipede_2",
        "beyond_jellyfish",
        "beyond_polyp",
    ),
    ENTITY_BEYOND_APERTURE: ("beyond_mouth",),
}


@dataclass
class Entity:
    entity_type: str
    asset_name: str
    x: int
    y: int
    size: int
    opacity: float
    lifetime: float
    intensity: float
    phase: float
    behavior: str = BEHAVIOR_LURK
    motion_x: float = 0.0
    motion_y: float = 0.0
    flip_horizontal: bool = False
    edge: str | None = None
    depth_layer: int = 1
    wobble_speed: float = 1.0
    wobble_amplitude: float = 0.0
    spectral_strength: float = 0.0
    refract_background: bool = False
    event_entity: bool = False
    interaction_role: str | None = None
    interaction_start: float = 0.0
    interaction_duration: float = 0.0
    interaction_target_x: float = 0.0
    interaction_target_y: float = 0.0
    reaction_start: float = 0.0
    reaction_duration: float = 0.0
    reaction_x: float = 0.0
    reaction_y: float = 0.0
    penetrates_person: bool = False
    person_pass_start: float = 0.28
    person_pass_duration: float = 0.42
    start_time: float = field(default_factory=time.monotonic)

    def is_alive(self, now: float | None = None) -> bool:
        current_time = time.monotonic() if now is None else now
        return (current_time - self.start_time) < self.lifetime


class EntityManager:
    def __init__(
        self,
        overlay_engine: OverlayEngine | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.clock = clock
        self.overlay_engine = overlay_engine or OverlayEngine()
        self.overlay_engine.load_assets(ASSET_PATHS)
        self._prepare_beyond_assets()
        self.entities: list[Entity] = []
        self.last_spawn_time = float("-inf")
        self.last_spawn_attempt_time = float("-inf")
        self._beyond_active = False
        self._frequency = "0"
        self._beyond_asset_queues: dict[str, list[str]] = {}
        self._next_beyond_event_time: float | None = None
        self._last_reaction_check_time = float("-inf")

    def _prepare_beyond_assets(self) -> None:
        """Turns the rectangular mouth image into a feathered spatial rift."""
        assets = getattr(self.overlay_engine, "assets", None)
        if not isinstance(assets, dict):
            return

        mouth = assets.get("beyond_mouth")
        if mouth is None or mouth.size == 0:
            return

        height, width = mouth.shape[:2]
        x, y = np.meshgrid(
            np.linspace(-1.0, 1.0, width, dtype=np.float32),
            np.linspace(-1.0, 1.0, height, dtype=np.float32),
        )
        radius = np.sqrt((x / 1.02) ** 2 + (y / 0.92) ** 2)
        edge = np.clip((1.0 - radius) / 0.16, 0.0, 1.0)
        brightness = np.max(mouth[:, :, :3], axis=2).astype(np.float32) / 255.0
        structure = np.clip((brightness - 0.015) / 0.16, 0.0, 1.0)
        dark_core = np.exp(-((x / 0.22) ** 2 + (y / 0.30) ** 2))
        alpha = np.maximum(structure, dark_core * 0.92) * edge
        mouth[:, :, 3] = np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8)

    def update(
        self,
        active_modes: list[int],
        frequency: str,
        scene_state: dict[str, Any],
    ) -> None:
        now = self.clock()
        self.entities = [entity for entity in self.entities if entity.is_alive(now)]
        self._finish_interactions(now)
        previous_frequency = self._frequency
        beyond_active = MODE_BEYOND in active_modes
        if beyond_active != self._beyond_active:
            self.entities.clear()
            self.last_spawn_time = float("-inf")
            self.last_spawn_attempt_time = float("-inf")
            self._next_beyond_event_time = None
            self._last_reaction_check_time = float("-inf")
        self._beyond_active = beyond_active
        self._frequency = frequency

        if self._beyond_active and (
            self._next_beyond_event_time is None
            or frequency != previous_frequency
        ):
            self._schedule_next_beyond_event(now, frequency)

        entity_mode_active = (
            MODE_ENTITIES in active_modes or MODE_BEYOND in active_modes
        )
        if not entity_mode_active:
            self.entities.clear()
            self._next_beyond_event_time = None
            return

        self._anchor_entities_to_scene(scene_state)

        profile = self.get_profile(frequency, active_modes)
        if profile["max_entities"] <= 0:
            self.entities.clear()
            self._next_beyond_event_time = None
            return

        max_entities = int(profile["max_entities"])
        regular_entities = [
            entity for entity in self.entities if not entity.event_entity
        ]
        event_entities = [
            entity for entity in self.entities if entity.event_entity
        ]
        if len(regular_entities) > max_entities:
            regular_entities = regular_entities[:max_entities]
            self.entities = regular_entities + event_entities

        if self._beyond_active:
            self._react_to_camera(scene_state, now, float(profile["intensity"]))
            self._maybe_trigger_beyond_event(
                scene_state,
                now,
                float(profile["intensity"]),
            )

        regular_count = sum(
            not entity.event_entity for entity in self.entities
        )
        if regular_count >= max_entities:
            return

        spawn_interval = float(profile["spawn_interval"])
        if MODE_SOFT in active_modes and MODE_ENTITIES in active_modes:
            spawn_interval *= 0.72

        if now - self.last_spawn_time < spawn_interval:
            return

        if (
            not self._beyond_active
            and not scene_state.get("has_good_overlay_spot", False)
        ):
            return

        overlay_regions = scene_state.get("best_overlay_regions", [])
        if self._beyond_active:
            overlay_regions = self._beyond_regions(
                scene_state.get("frame_size"),
            )
        elif not overlay_regions:
            return

        if now - self.last_spawn_attempt_time < profile["attempt_interval"]:
            return

        self.last_spawn_attempt_time = now
        spawn_chance = self.get_spawn_chance(
            profile,
            scene_state,
            active_modes,
        )
        if random.random() > spawn_chance:
            return

        region_limit = int(profile["region_limit"])
        portal = None
        if self._beyond_active:
            portals = scene_state.get("dark_portals", [])
            portal_chance = 0.16 + float(profile["intensity"]) * 0.42
            if portals and random.random() < portal_chance:
                portal = random.choice(portals[:3])

        if portal is not None:
            px, py = portal["center"]
            pw, ph = portal.get("axes", (80, 80))
            region = {
                "box": (
                    int(px - pw * 0.45),
                    int(py - ph * 0.45),
                    max(1, int(pw * 0.90)),
                    max(1, int(ph * 0.90)),
                ),
                "center": (int(px), int(py)),
                "score": portal.get("score", 1.0),
            }
            entity_type = random.choice((
                ENTITY_BEYOND_CREATURE,
                ENTITY_BEYOND_CREATURE,
                ENTITY_TENTACLE,
            ))
        else:
            region = random.choice(overlay_regions[:region_limit])
            entity_type = self.choose_entity_type(frequency, active_modes)
        asset_name = (
            self._choose_beyond_asset(entity_type)
            if self._beyond_active
            else self.choose_asset(entity_type, active_modes)
        )
        if asset_name is None:
            return
        behavior = (
            BEHAVIOR_PORTAL
            if portal is not None
            else (
                self.choose_beyond_behavior(asset_name)
                if self._beyond_active
                else self.choose_behavior(entity_type, frequency, active_modes)
            )
        )

        entity = self._create_entity(
            entity_type=entity_type,
            asset_name=asset_name,
            behavior=behavior,
            region=region,
            frame_size=scene_state.get("frame_size"),
            profile=profile,
            now=now,
        )
        if portal is not None:
            self._configure_portal_entity(
                entity,
                portal,
                scene_state.get("frame_size") or (1280, 720),
            )
        self.entities.append(entity)
        self.last_spawn_time = now

    @staticmethod
    def _configure_portal_entity(
        entity: Entity,
        portal: dict[str, Any],
        frame_size: tuple[int, int],
    ) -> None:
        """Starts a creature inside a detected opening and lets it drift out."""
        frame_width, frame_height = frame_size
        center_x, center_y = portal["center"]
        axis_width, axis_height = portal.get("axes", (80, 80))
        opening_size = max(float(axis_width), float(axis_height))
        entity.x = int(center_x)
        entity.y = int(center_y)
        entity.size = int(np.clip(
            max(entity.size, opening_size * 1.18),
            frame_width * 0.11,
            frame_width * 0.42,
        ))
        angle = random.uniform(-math.pi, math.pi)
        distance = random.uniform(frame_width * 0.20, frame_width * 0.38)
        entity.motion_x = math.cos(angle) * distance
        entity.motion_y = math.sin(angle) * distance * (
            frame_height / max(1.0, frame_width)
        )
        entity.flip_horizontal = entity.motion_x < 0
        entity.edge = "right" if entity.motion_x < 0 else "left"
        entity.behavior = BEHAVIOR_PORTAL
        entity.lifetime = random.uniform(7.0, 12.0)
        entity.depth_layer = 1
        entity.penetrates_person = True
        entity.person_pass_start = random.uniform(0.22, 0.36)
        entity.person_pass_duration = random.uniform(0.34, 0.52)

    @staticmethod
    def _beyond_regions(
        frame_size: tuple[int, int] | None,
    ) -> list[dict[str, Any]]:
        frame_width, frame_height = frame_size or (1280, 720)
        regions = []
        columns = 3
        rows = 2
        cell_width = max(1, frame_width // columns)
        cell_height = max(1, frame_height // rows)
        for row in range(rows):
            for column in range(columns):
                left = column * cell_width
                top = row * cell_height
                width = frame_width - left if column == columns - 1 else cell_width
                height = frame_height - top if row == rows - 1 else cell_height
                regions.append({
                    "box": (left, top, width, height),
                    "score": 1.0,
                    "center": (left + width // 2, top + height // 2),
                })
        return regions

    def _choose_beyond_asset(self, entity_type: str) -> str | None:
        pool = BEYOND_ASSET_POOLS.get(entity_type, ())
        if not pool:
            return None

        queue = self._beyond_asset_queues.get(entity_type, [])
        if not queue:
            queue = list(pool)
            random.shuffle(queue)
            self._beyond_asset_queues[entity_type] = queue
        return queue.pop()

    def _schedule_next_beyond_event(self, now: float, frequency: str) -> None:
        delay_range = BEYOND_EVENT_DELAYS.get(frequency)
        if delay_range is None:
            self._next_beyond_event_time = None
            return
        self._next_beyond_event_time = now + random.uniform(*delay_range)

    def _maybe_trigger_beyond_event(
        self,
        scene_state: dict[str, Any],
        now: float,
        intensity: float,
    ) -> None:
        if self._next_beyond_event_time is None:
            return
        if now < self._next_beyond_event_time:
            return

        if any(
            entity.event_entity or entity.interaction_role is not None
            for entity in self.entities
        ):
            self._schedule_next_beyond_event(now, self._frequency)
            return

        if self._frequency == "III":
            event_pool = (
                "brush",
                "brush",
                "devour",
                "devour",
                "devour",
            )
        else:
            # The mouth occupies only one slot: even at V it should feel like
            # an exceptional discovery rather than a recurring screensaver.
            event_pool = (
                "brush",
                "brush",
                "brush",
                "devour",
                "devour",
                "devour",
                "devour",
                "mouth",
            )

        event = random.choice(event_pool)
        frame_size = scene_state.get("frame_size") or (1280, 720)
        started = False
        if event == "devour":
            started = self._start_devour_event(now)
        elif event == "mouth":
            started = self._start_mouth_event(frame_size, now, intensity)
        if not started:
            self._start_brush_event(frame_size, now, intensity)

        self._schedule_next_beyond_event(now, self._frequency)

    def _start_devour_event(self, now: float) -> bool:
        eligible = [
            entity
            for entity in self.entities
            if (
                entity.entity_type == ENTITY_BEYOND_CREATURE
                and not entity.event_entity
                and entity.interaction_role is None
            )
        ]
        if len(eligible) < 2:
            return False

        predator, victim = random.sample(eligible, 2)
        duration = random.uniform(2.6, 3.8)
        victim_age = max(0.0, now - victim.start_time)
        predator_age = max(0.0, now - predator.start_time)
        victim_progress = self._smoothstep(
            float(np.clip(victim_age / max(victim.lifetime, 0.001), 0.0, 1.0)),
        )
        target_x = victim.x + victim.motion_x * victim_progress
        target_y = victim.y + victim.motion_y * victim_progress

        predator.interaction_role = INTERACTION_DEVOURER
        predator.interaction_start = now
        predator.interaction_duration = duration
        predator.interaction_target_x = target_x
        predator.interaction_target_y = target_y
        predator.depth_layer = victim.depth_layer
        predator.lifetime = max(predator.lifetime, predator_age + duration + 1.0)

        victim.interaction_role = INTERACTION_CONSUMED
        victim.interaction_start = now
        victim.interaction_duration = duration
        victim.lifetime = max(victim.lifetime, victim_age + duration)
        return True

    def _start_brush_event(
        self,
        frame_size: tuple[int, int],
        now: float,
        intensity: float,
    ) -> bool:
        if any(entity.behavior == BEHAVIOR_BRUSH for entity in self.entities):
            return False

        frame_width, frame_height = frame_size
        asset_name = random.choice((
            "beyond_centipede_1",
            "beyond_centipede_2",
            "beyond_jellyfish",
            "beyond_tentacle",
        ))
        direction = -1.0 if random.random() < 0.5 else 1.0
        size = int(frame_width * random.uniform(0.72, 1.04))
        center_x = frame_width + size // 2 if direction < 0 else -size // 2
        center_y = int(frame_height * random.uniform(0.22, 0.78))
        entity_type = (
            ENTITY_TENTACLE
            if asset_name == "beyond_tentacle"
            else ENTITY_BEYOND_CREATURE
        )
        self.entities.append(Entity(
            entity_type=entity_type,
            asset_name=asset_name,
            x=center_x,
            y=center_y,
            size=size,
            opacity=random.uniform(0.28, 0.40),
            lifetime=random.uniform(4.8, 6.6),
            intensity=intensity,
            phase=random.uniform(0.0, math.tau),
            behavior=BEHAVIOR_BRUSH,
            motion_x=direction * (frame_width + size * 1.35),
            motion_y=random.uniform(-frame_height * 0.08, frame_height * 0.08),
            flip_horizontal=direction < 0,
            edge="right" if direction < 0 else "left",
            depth_layer=2,
            wobble_speed=random.uniform(0.75, 1.20),
            wobble_amplitude=size * random.uniform(0.025, 0.045),
            spectral_strength=0.92,
            event_entity=True,
            penetrates_person=True,
            person_pass_start=random.uniform(0.18, 0.28),
            person_pass_duration=random.uniform(0.42, 0.58),
            start_time=now,
        ))
        return True

    def _start_mouth_event(
        self,
        frame_size: tuple[int, int],
        now: float,
        intensity: float,
    ) -> bool:
        if any(entity.asset_name == "beyond_mouth" for entity in self.entities):
            return False

        frame_width, frame_height = frame_size
        side = random.choice((0.22, 0.78))
        size = int(frame_width * random.uniform(0.52, 0.72))
        self.entities.append(Entity(
            entity_type=ENTITY_BEYOND_APERTURE,
            asset_name="beyond_mouth",
            x=int(frame_width * side),
            y=int(frame_height * random.uniform(0.32, 0.68)),
            size=size,
            opacity=random.uniform(0.24, 0.36),
            lifetime=random.uniform(9.0, 13.0),
            intensity=intensity,
            phase=random.uniform(0.0, math.tau),
            behavior=BEHAVIOR_RIFT,
            motion_x=random.uniform(-frame_width * 0.035, frame_width * 0.035),
            motion_y=random.uniform(-frame_height * 0.025, frame_height * 0.025),
            depth_layer=1,
            wobble_speed=random.uniform(0.32, 0.58),
            wobble_amplitude=size * random.uniform(0.014, 0.026),
            spectral_strength=0.96,
            event_entity=True,
            start_time=now,
        ))
        return True

    def _react_to_camera(
        self,
        scene_state: dict[str, Any],
        now: float,
        intensity: float,
    ) -> None:
        if now - self._last_reaction_check_time < 0.40:
            return
        self._last_reaction_check_time = now

        motion_x, motion_y = scene_state.get("camera_motion", (0.0, 0.0))
        motion_x = float(motion_x)
        motion_y = float(motion_y)
        motion_strength = math.hypot(motion_x, motion_y)
        moving_regions = int(scene_state.get("motion_count", 0))
        if motion_strength < 1.2 and moving_regions <= 0:
            return

        frame_width, frame_height = scene_state.get("frame_size") or (1280, 720)
        response_chance = 0.035 + intensity * 0.105
        for entity in self.entities:
            if (
                entity.entity_type not in (ENTITY_BEYOND_CREATURE, ENTITY_TENTACLE)
                or entity.event_entity
                or entity.interaction_role is not None
            ):
                continue
            if random.random() > response_chance:
                continue

            if motion_strength >= 1.2:
                response_x = -motion_x * random.uniform(4.0, 8.5)
                response_y = -motion_y * random.uniform(4.0, 8.5)
            else:
                response_x = random.uniform(-frame_width * 0.06, frame_width * 0.06)
                response_y = random.uniform(-frame_height * 0.05, frame_height * 0.05)
            entity.reaction_x = float(np.clip(
                response_x,
                -frame_width * 0.14,
                frame_width * 0.14,
            ))
            entity.reaction_y = float(np.clip(
                response_y,
                -frame_height * 0.12,
                frame_height * 0.12,
            ))
            entity.reaction_start = now
            entity.reaction_duration = random.uniform(0.9, 1.7)

    def _finish_interactions(self, now: float) -> None:
        for entity in self.entities:
            if entity.interaction_role != INTERACTION_DEVOURER:
                continue
            if now < entity.interaction_start + entity.interaction_duration:
                continue
            entity.x = int(round(entity.interaction_target_x))
            entity.y = int(round(entity.interaction_target_y))
            entity.motion_x *= 0.25
            entity.motion_y *= 0.25
            entity.interaction_role = None
            entity.interaction_duration = 0.0

    def _anchor_entities_to_scene(self, scene_state: dict[str, Any]) -> None:
        if not self.entities:
            return

        confidence = float(scene_state.get("tracking_confidence", 0.0))
        transform = scene_state.get("camera_transform")
        if confidence < 0.20 or transform is None:
            return

        matrix = np.asarray(transform, dtype=np.float32)
        if matrix.shape != (2, 3) or not np.all(np.isfinite(matrix)):
            return

        for entity in self.entities:
            parallax = (0.58, 1.0, 1.32)[
                int(np.clip(entity.depth_layer, 0, 2))
            ]
            adjusted = matrix.copy()
            adjusted[:, :2] = (
                np.eye(2, dtype=np.float32)
                + (adjusted[:, :2] - np.eye(2, dtype=np.float32)) * parallax
            )
            adjusted[:, 2] *= parallax
            position = adjusted @ np.array(
                [entity.x, entity.y, 1.0],
                dtype=np.float32,
            )
            entity.x = int(round(float(position[0])))
            entity.y = int(round(float(position[1])))
            if entity.interaction_role == INTERACTION_DEVOURER:
                target = adjusted @ np.array(
                    [
                        entity.interaction_target_x,
                        entity.interaction_target_y,
                        1.0,
                    ],
                    dtype=np.float32,
                )
                entity.interaction_target_x = float(target[0])
                entity.interaction_target_y = float(target[1])

        frame_size = scene_state.get("frame_size")
        if frame_size:
            frame_width, frame_height = frame_size
            self.entities = [
                entity
                for entity in self.entities
                if (
                    -entity.size <= entity.x <= frame_width + entity.size
                    and -entity.size <= entity.y <= frame_height + entity.size
                )
            ]

    def _create_entity(
        self,
        entity_type: str,
        asset_name: str,
        behavior: str,
        region: dict[str, Any],
        frame_size: tuple[int, int] | None,
        profile: dict[str, Any],
        now: float,
    ) -> Entity:
        x, y, width, height = region["box"]
        frame_width, frame_height = frame_size or (1280, 720)
        is_beyond_asset = asset_name.startswith("beyond_")
        intensity = float(profile["intensity"])

        if is_beyond_asset:
            if intensity < 0.25:
                depth_layer = 0
            elif intensity < 0.50:
                depth_layer = random.choice((0, 0, 1))
            elif intensity < 0.75:
                depth_layer = random.choice((0, 1, 1, 2))
            else:
                depth_layer = random.choice((0, 1, 1, 2, 2))
        else:
            depth_layer = 1

        min_size = max(32, int(frame_width * profile["size_range"][0]))
        max_size = max(min_size, int(frame_width * profile["size_range"][1]))
        size = random.randint(min_size, max_size)
        if is_beyond_asset:
            size = max(24, int(round(size * (0.62, 1.0, 1.30)[depth_layer])))

        center_x = x + random.randint(0, max(1, width - 1))
        center_y = y + random.randint(0, max(1, height - 1))
        edge = None
        flip_horizontal = random.random() < 0.5
        motion_x = 0.0
        motion_y = 0.0
        lifetime = random.uniform(*profile["lifetime_range"])
        refract_background = asset_name in {
            "beyond_bubble_1",
            "beyond_bubble_2",
            "beyond_bubbles",
        }

        if is_beyond_asset and behavior in (BEHAVIOR_SWIM, BEHAVIOR_FLOW):
            direction = -1.0 if random.random() < 0.5 else 1.0
            edge = "right" if direction < 0 else "left"
            center_x = (
                frame_width + size // 2
                if direction < 0
                else -size // 2
            )
            center_y = y + random.randint(0, max(1, height - 1))
            motion_x = direction * (frame_width + size * 1.35)
            motion_y = random.uniform(-frame_height * 0.18, frame_height * 0.18)
            flip_horizontal = direction < 0
            lifetime = random.uniform(8.0, 15.0)
        elif is_beyond_asset and behavior == BEHAVIOR_FLOAT:
            center_y = frame_height + size // 3
            motion_x = random.uniform(-frame_width * 0.16, frame_width * 0.16)
            motion_y = -random.uniform(frame_height * 0.72, frame_height * 1.20)
            lifetime = random.uniform(10.0, 18.0)
        elif is_beyond_asset and behavior == BEHAVIOR_DRIFT:
            motion_x = random.uniform(-frame_width * 0.30, frame_width * 0.30)
            motion_y = random.uniform(-frame_height * 0.24, frame_height * 0.24)
            lifetime = random.uniform(9.0, 17.0)
        elif is_beyond_asset and behavior == BEHAVIOR_RIFT:
            size = max(size, int(frame_width * (0.30 + intensity * 0.22)))
            motion_x = random.uniform(-frame_width * 0.06, frame_width * 0.06)
            motion_y = random.uniform(-frame_height * 0.04, frame_height * 0.04)
            lifetime = random.uniform(7.0, 12.0)
            depth_layer = 2
        elif entity_type == ENTITY_TENTACLE and frame_size:
            edge = "left" if center_x < frame_width // 2 else "right"
            center_x = size // 2 if edge == "left" else frame_width - size // 2
            flip_horizontal = edge == "right"
        elif entity_type == ENTITY_ATMOSPHERE:
            center_x = frame_width // 2
            center_y = frame_height // 2
            size = frame_width
            flip_horizontal = random.random() < 0.5

        if not is_beyond_asset and behavior == BEHAVIOR_CROSS:
            direction = -1.0 if random.random() < 0.5 else 1.0
            distance = random.uniform(frame_width * 0.12, frame_width * 0.34)
            motion_x = direction * distance
            motion_y = random.uniform(-frame_height * 0.025, frame_height * 0.025)
            lifetime = random.uniform(2.8, 6.5)
            flip_horizontal = direction < 0
        elif not is_beyond_asset and behavior == BEHAVIOR_GLIMPSE:
            lifetime = random.uniform(0.28, 0.75)
        elif not is_beyond_asset and behavior == BEHAVIOR_EMERGE:
            lifetime = random.uniform(1.6, 4.2)
        elif not is_beyond_asset and behavior in (BEHAVIOR_WATCH, BEHAVIOR_LURK):
            lifetime = max(lifetime, random.uniform(1.8, 4.8))

        opacity_range = {
            ENTITY_EYE: (0.20, 0.42),
            ENTITY_TENTACLE: (0.24, 0.48),
            ENTITY_SHADOW: (0.12, 0.28),
            ENTITY_ATMOSPHERE: (0.10, 0.25),
            ENTITY_BEYOND_CREATURE: (0.30, 0.58),
            ENTITY_BEYOND_APERTURE: (0.16, 0.34),
        }
        opacity_min, opacity_max = opacity_range[entity_type]
        opacity_boost = 0.8 + intensity * 0.55
        if is_beyond_asset and depth_layer == 0:
            opacity_boost *= 0.66
        elif is_beyond_asset and depth_layer == 2:
            opacity_boost *= 0.90

        is_bubble = refract_background
        if is_bubble:
            # Soap bubbles should remain airy and iridescent instead of
            # becoming dark, eye-like spectral discs.
            opacity_boost *= 0.72
            wobble_speed = random.uniform(0.58, 0.96)
            wobble_amplitude = size * random.uniform(0.045, 0.080)
            spectral_strength = 0.10 + intensity * 0.16
        else:
            wobble_speed = random.uniform(0.46, 1.05) * (
                0.8 + intensity * 0.55
            )
            wobble_amplitude = size * random.uniform(0.018, 0.045)
            spectral_strength = 0.42 + intensity * 0.46 if is_beyond_asset else 0.0

        penetrates_person = (
            is_beyond_asset
            and entity_type in (ENTITY_BEYOND_CREATURE, ENTITY_TENTACLE)
            and behavior != BEHAVIOR_RIFT
            and random.random() < 0.28 + intensity * 0.48
        )

        return Entity(
            entity_type=entity_type,
            asset_name=asset_name,
            x=center_x,
            y=center_y,
            size=size,
            opacity=min(0.9, random.uniform(opacity_min, opacity_max) * opacity_boost),
            lifetime=lifetime,
            intensity=intensity,
            phase=random.uniform(0.0, math.tau),
            behavior=behavior,
            motion_x=motion_x,
            motion_y=motion_y,
            flip_horizontal=flip_horizontal,
            edge=edge,
            depth_layer=depth_layer,
            wobble_speed=wobble_speed,
            wobble_amplitude=wobble_amplitude,
            spectral_strength=spectral_strength,
            refract_background=refract_background,
            penetrates_person=penetrates_person,
            person_pass_start=random.uniform(0.20, 0.38),
            person_pass_duration=random.uniform(0.34, 0.54),
            start_time=now,
        )

    @staticmethod
    def get_profile(frequency: str, active_modes: list[int]) -> dict[str, Any]:
        if MODE_BEYOND in active_modes:
            return BEYOND_PROFILES.get(frequency, BEYOND_PROFILES["0"])
        return FREQUENCY_PROFILES.get(frequency, FREQUENCY_PROFILES["0"])

    @staticmethod
    def get_spawn_chance(
        profile: dict[str, Any],
        scene_state: dict[str, Any],
        active_modes: list[int] | tuple[int, ...] = (),
    ) -> float:
        motion_count = int(scene_state.get("motion_count", 0))
        motion_boost = 1.0 + min(motion_count, 5) * 0.05
        witch_house_boost = (
            1.35
            if MODE_SOFT in active_modes and MODE_ENTITIES in active_modes
            else 1.0
        )
        return min(
            float(profile["spawn_chance"]) * motion_boost * witch_house_boost,
            1.0,
        )

    def choose_entity_type(self, frequency: str, active_modes: list[int]) -> str:
        if MODE_BEYOND in active_modes:
            if frequency == "I":
                return ENTITY_ATMOSPHERE
            if frequency == "II":
                return random.choice((
                    ENTITY_ATMOSPHERE,
                    ENTITY_ATMOSPHERE,
                    ENTITY_BEYOND_CREATURE,
                ))
            if frequency == "III":
                return random.choice((
                    ENTITY_ATMOSPHERE,
                    ENTITY_BEYOND_CREATURE,
                    ENTITY_BEYOND_CREATURE,
                    ENTITY_TENTACLE,
                ))
            if frequency == "IV":
                return random.choice((
                    ENTITY_ATMOSPHERE,
                    ENTITY_BEYOND_CREATURE,
                    ENTITY_BEYOND_CREATURE,
                    ENTITY_TENTACLE,
                ))
            if frequency == "V":
                return random.choice((
                    ENTITY_ATMOSPHERE,
                    ENTITY_BEYOND_CREATURE,
                    ENTITY_BEYOND_CREATURE,
                    ENTITY_TENTACLE,
                    ENTITY_TENTACLE,
                ))
            return ENTITY_ATMOSPHERE

        if frequency == "V":
            return random.choice((
                ENTITY_EYE,
                ENTITY_EYE,
                ENTITY_TENTACLE,
                ENTITY_SHADOW,
                ENTITY_SHADOW,
            ))

        if frequency == "IV":
            return random.choice((ENTITY_EYE, ENTITY_TENTACLE, ENTITY_SHADOW))

        if frequency in ("II", "III"):
            return random.choice((ENTITY_EYE, ENTITY_EYE, ENTITY_SHADOW))

        return ENTITY_EYE

    @staticmethod
    def choose_behavior(
        entity_type: str,
        frequency: str,
        active_modes: list[int],
    ) -> str:
        if entity_type == ENTITY_TENTACLE:
            return BEHAVIOR_EMERGE

        if entity_type == ENTITY_EYE:
            choices = (
                BEHAVIOR_WATCH,
                BEHAVIOR_WATCH,
                BEHAVIOR_WATCH,
                BEHAVIOR_GLIMPSE,
            )
            return random.choice(choices)

        if entity_type == ENTITY_SHADOW:
            choices = (
                BEHAVIOR_CROSS,
                BEHAVIOR_CROSS,
                BEHAVIOR_LURK,
            )
            return random.choice(choices)

        if entity_type == ENTITY_BEYOND_CREATURE:
            if MODE_BEYOND in active_modes:
                choices = (
                    BEHAVIOR_CROSS,
                    BEHAVIOR_LURK,
                    BEHAVIOR_GLIMPSE,
                )
            else:
                choices = (BEHAVIOR_GLIMPSE, BEHAVIOR_LURK)
            return random.choice(choices)

        return BEHAVIOR_GLIMPSE if frequency in ("IV", "V") else BEHAVIOR_LURK

    @staticmethod
    def choose_beyond_behavior(asset_name: str) -> str:
        if asset_name.startswith("beyond_bubble"):
            return BEHAVIOR_FLOAT
        if asset_name.startswith("beyond_centipede"):
            return BEHAVIOR_SWIM
        if asset_name == "beyond_tentacle":
            return BEHAVIOR_FLOW
        if asset_name == "beyond_mouth":
            return BEHAVIOR_RIFT
        return BEHAVIOR_DRIFT

    @staticmethod
    def choose_asset(entity_type: str, active_modes: list[int]) -> str | None:
        pools = (
            BEYOND_ASSET_POOLS
            if MODE_BEYOND in active_modes
            else NORMAL_ASSET_POOLS
        )
        assets = pools.get(entity_type, ())
        return random.choice(assets) if assets else None

    def draw(
        self,
        frame: np.ndarray,
        person_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        protected_frame = frame.copy() if person_mask is not None else None
        now = self.clock()
        entities = sorted(self.entities, key=lambda entity: entity.depth_layer)
        if protected_frame is None:
            for entity in entities:
                self._draw_entity(frame, entity, now)
            return frame

        if not self._beyond_active:
            for entity in entities:
                self._draw_entity(frame, entity, now)
            self._restore_person_layer(frame, protected_frame, person_mask)
            return frame

        behind_entities: list[Entity] = []
        front_entities: list[Entity] = []
        passing_entities: list[tuple[Entity, float]] = []
        for entity in entities:
            if entity.penetrates_person:
                pass_progress = self._person_pass_progress(entity, now)
                if pass_progress < 0.0:
                    front_entities.append(entity)
                else:
                    behind_entities.append(entity)
                    if pass_progress <= 1.0:
                        passing_entities.append((entity, pass_progress))
            elif entity.depth_layer >= 2:
                front_entities.append(entity)
            else:
                behind_entities.append(entity)

        for entity in behind_entities:
            self._draw_entity(frame, entity, now)

        self._restore_person_layer(frame, protected_frame, person_mask)

        for entity in front_entities:
            self._draw_entity(frame, entity, now)

        # During the crossing only the part inside the detected silhouette is
        # visible on top. The rest already lies behind the person.
        for entity, pass_progress in passing_entities:
            exit_fade = self._smoothstep(min(
                1.0,
                max(0.0, (1.0 - pass_progress) / 0.18),
            ))
            inside_opacity = (
                0.56 + math.sin(pass_progress * math.pi) * 0.34
            ) * exit_fade
            self._draw_entity(
                frame,
                entity,
                now,
                opacity_scale=inside_opacity,
                visibility_mask=person_mask,
            )

        return frame

    def _draw_entity(
        self,
        frame: np.ndarray,
        entity: Entity,
        now: float,
        opacity_scale: float = 1.0,
        visibility_mask: np.ndarray | None = None,
    ) -> None:
        age = max(0.0, now - entity.start_time)
        progress = float(np.clip(age / entity.lifetime, 0.0, 1.0))
        opacity = entity.opacity * self._behavior_opacity(
            progress,
            entity.behavior,
            entity.intensity,
        ) * max(0.0, opacity_scale)
        reveal = 1.0

        if entity.behavior == BEHAVIOR_EMERGE:
            reveal = self._emerge_reveal(progress)
        elif entity.behavior == BEHAVIOR_PORTAL:
            reveal = self._smoothstep(min(1.0, progress / 0.18))

        travel = self._smoothstep(progress)
        wobble_phase = entity.phase + age * entity.wobble_speed
        reaction_x = 0.0
        reaction_y = 0.0
        reaction_wobble = 1.0
        if entity.reaction_duration > 0.0:
            reaction_progress = (now - entity.reaction_start) / entity.reaction_duration
            if 0.0 <= reaction_progress <= 1.0:
                reaction_envelope = math.sin(reaction_progress * math.pi)
                reaction_x = entity.reaction_x * reaction_envelope
                reaction_y = entity.reaction_y * reaction_envelope
                reaction_wobble += reaction_envelope * 1.35

        wobble_x = (
            math.sin(wobble_phase)
            * entity.wobble_amplitude
            * reaction_wobble
        )
        wobble_y = (
            math.sin(wobble_phase * 0.73 + 1.4)
            * entity.wobble_amplitude
            * 0.62
            * reaction_wobble
        )
        if entity.behavior == BEHAVIOR_FLOAT:
            # Two asynchronous waves make the path meander like a soap bubble
            # instead of following a single mechanical diagonal.
            wobble_x = (
                math.sin(wobble_phase) * entity.wobble_amplitude * 1.65
                + math.sin(wobble_phase * 0.37 + entity.phase * 1.7)
                * entity.wobble_amplitude
                * 1.05
            )
            wobble_y = (
                math.sin(wobble_phase * 0.71 + 1.4)
                * entity.wobble_amplitude
                * 1.08
                + math.cos(wobble_phase * 0.29 - 0.4)
                * entity.wobble_amplitude
                * 0.38
            )
        center_x = entity.x + entity.motion_x * travel + wobble_x + reaction_x
        center_y = entity.y + entity.motion_y * travel + wobble_y + reaction_y

        dissolve = 0.0
        warp_multiplier = reaction_wobble
        if entity.interaction_role is not None and entity.interaction_duration > 0.0:
            interaction_progress = float(np.clip(
                (now - entity.interaction_start) / entity.interaction_duration,
                0.0,
                1.0,
            ))
            if entity.interaction_role == INTERACTION_DEVOURER:
                approach = self._smoothstep(min(1.0, interaction_progress / 0.82))
                center_x = (
                    center_x * (1.0 - approach)
                    + entity.interaction_target_x * approach
                )
                center_y = (
                    center_y * (1.0 - approach)
                    + entity.interaction_target_y * approach
                )
                warp_multiplier *= 1.0 + interaction_progress * 0.85
            elif entity.interaction_role == INTERACTION_CONSUMED:
                dissolve = self._smoothstep(max(
                    0.0,
                    (interaction_progress - 0.32) / 0.68,
                ))
                shake = entity.size * 0.018 * dissolve
                center_x += math.sin(wobble_phase * 5.2) * shake
                center_y += math.cos(wobble_phase * 4.4) * shake
                opacity *= 1.0 - dissolve * 0.22
                warp_multiplier *= 1.0 + dissolve * 2.4

        center = (int(round(center_x)), int(round(center_y)))

        is_beyond = entity.asset_name.startswith("beyond_")
        warp_amount = 0.0
        warp_phase = None
        if is_beyond:
            warp_amount = max(
                1.0,
                entity.size * (0.006 + entity.intensity * 0.010),
            ) * warp_multiplier
            warp_phase = wobble_phase
            if entity.behavior == BEHAVIOR_FLOAT:
                warp_amount *= 2.15

        self.overlay_engine.draw(
            frame,
            entity.asset_name,
            center=center,
            target_width=max(1, entity.size),
            opacity=opacity,
            flip_horizontal=entity.flip_horizontal,
            height_scale=1.0,
            reveal=reveal,
            reveal_from_right=entity.edge == "right",
            warp_phase=warp_phase,
            warp_amount=warp_amount,
            spectral_strength=entity.spectral_strength,
            refract_background=entity.refract_background,
            dissolve=dissolve,
            dissolve_seed=int(entity.phase * 10000.0),
            visibility_mask=visibility_mask,
        )

    @staticmethod
    def _person_pass_progress(entity: Entity, now: float) -> float:
        normalized_age = max(0.0, now - entity.start_time) / max(
            0.001,
            entity.lifetime,
        )
        if normalized_age < entity.person_pass_start:
            return -1.0
        return (
            normalized_age - entity.person_pass_start
        ) / max(0.05, entity.person_pass_duration)

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        return value * value * (3.0 - 2.0 * value)

    @staticmethod
    def _emerge_reveal(progress: float) -> float:
        if progress < 0.20:
            return EntityManager._smoothstep(progress / 0.20)
        if progress > 0.76:
            return EntityManager._smoothstep((1.0 - progress) / 0.24)
        return 1.0

    @staticmethod
    def _behavior_opacity(
        progress: float,
        behavior: str,
        intensity: float,
    ) -> float:
        if behavior == BEHAVIOR_GLIMPSE:
            if progress < 0.10:
                return EntityManager._smoothstep(progress / 0.10)
            if progress > 0.68:
                return EntityManager._smoothstep((1.0 - progress) / 0.32)
            return 1.0

        if behavior == BEHAVIOR_BRUSH:
            if progress < 0.08:
                return EntityManager._smoothstep(progress / 0.08)
            if progress > 0.88:
                return EntityManager._smoothstep((1.0 - progress) / 0.12)
            return 1.0

        if behavior == BEHAVIOR_RIFT:
            if progress < 0.30:
                return EntityManager._smoothstep(progress / 0.30)
            if progress > 0.76:
                return EntityManager._smoothstep((1.0 - progress) / 0.24)
            return 1.0

        return EntityManager._fade(progress, intensity)

    @staticmethod
    def _restore_person_layer(
        frame: np.ndarray,
        protected_frame: np.ndarray,
        person_mask: np.ndarray,
    ) -> None:
        if person_mask.size == 0:
            return

        frame_height, frame_width = frame.shape[:2]
        if person_mask.shape[:2] != (frame_height, frame_width):
            person_mask = cv2.resize(
                person_mask,
                (frame_width, frame_height),
                interpolation=cv2.INTER_LINEAR,
            )

        alpha = np.clip(person_mask, 0.0, 1.0).astype(np.float32)[:, :, None]
        restored = (
            protected_frame.astype(np.float32) * alpha
            + frame.astype(np.float32) * (1.0 - alpha)
        )
        frame[:] = np.clip(restored, 0, 255).astype(np.uint8)

    @staticmethod
    def _fade(progress: float, intensity: float) -> float:
        fade_in_end = max(0.025, 0.18 - intensity * 0.14)
        fade_out_start = min(0.94, 0.74 + intensity * 0.16)

        if progress < fade_in_end:
            return progress / fade_in_end
        if progress > fade_out_start:
            return max(0.0, (1.0 - progress) / (1.0 - fade_out_start))
        return 1.0
