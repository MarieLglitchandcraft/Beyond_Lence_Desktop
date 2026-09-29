import cv2
import numpy as np
from pathlib import Path
import random
import time

from config import (
    MODE_SOFT,
    MODE_ENTITIES,
    MODE_GLITCH,
    MODE_BEYOND,
)
from other_world import LayeredOtherWorld


OTHER_WORLD_DIRECTORY = Path(__file__).resolve().parent / "assets" / "other_world"


class EffectEngine:
    FREQUENCY_INTENSITY = {
        "0": 0.0,
        "I": 0.12,
        "II": 0.25,
        "III": 0.45,
        "IV": 0.72,
        "V": 1.0,
    }
    WITCH_HOUSE_INTENSITY = {
        "0": 0.0,
        "I": 0.10,
        "II": 0.28,
        "III": 0.58,
        "IV": 1.12,
        "V": 1.80,
    }
    GLITCH_PROFILES = {
        "0": {
            "strength": 0.0,
            "first_delay": (1.2, 2.3),
            "interval": (7.0, 11.0),
            "duration": (0.14, 0.28),
            "coverage": 0.0,
            "world_opacity": 0.0,
        },
        "I": {
            "strength": 0.20,
            "first_delay": (0.9, 1.9),
            "interval": (5.5, 9.0),
            "duration": (0.16, 0.34),
            "coverage": 0.22,
            "world_opacity": 0.12,
        },
        "II": {
            "strength": 0.34,
            "first_delay": (0.7, 1.5),
            "interval": (4.2, 7.0),
            "duration": (0.20, 0.42),
            "coverage": 0.30,
            "world_opacity": 0.20,
        },
        "III": {
            "strength": 0.52,
            "first_delay": (0.55, 1.2),
            "interval": (2.8, 5.0),
            "duration": (0.24, 0.55),
            "coverage": 0.44,
            "world_opacity": 0.30,
        },
        "IV": {
            "strength": 0.75,
            "first_delay": (0.35, 0.85),
            "interval": (1.6, 3.1),
            "duration": (0.32, 0.72),
            "coverage": 0.65,
            "world_opacity": 0.45,
        },
        "V": {
            "strength": 1.0,
            "first_delay": (0.18, 0.48),
            "interval": (0.45, 1.0),
            "duration": (0.72, 1.40),
            "coverage": 0.94,
            "world_opacity": 0.66,
        },
    }

    def __init__(
        self,
        clock=time.monotonic,
        random_source=None,
        noise_seed=None,
        other_world_directory=OTHER_WORLD_DIRECTORY,
    ):
        self.clock = clock
        self.random = random_source or random.Random()
        self.noise_random = np.random.default_rng(noise_seed)
        self._coordinate_cache = {}
        self.other_world = LayeredOtherWorld(
            other_world_directory,
            self.random,
        )
        self._glitch_active = False
        self._next_rift_time = None
        self._rift_start_time = 0.0
        self._rift_end_time = 0.0
        self._rift_frame = None
        self._rift_seed = 0
        self._rift_scene_index = None
        self._rift_anchor = np.eye(3, dtype=np.float32)
        self._world_start_time = 0.0
        self._world_seed = 0

    def apply(self, frame, active_modes, frequency="0", analysis=None):
        result = frame.copy()

        if MODE_BEYOND in active_modes:
            return self.beyond(result, frequency)

        if MODE_SOFT in active_modes:
            result = self.soft(result, frequency, analysis)

        if MODE_ENTITIES in active_modes:
            result = self.entities(result)

        return result

    def soft(self, frame, frequency="0", analysis=None):
        """Dreams in the Witch House: bent geometry and slow local fog."""
        original = frame.copy()
        intensity = self.WITCH_HOUSE_INTENSITY.get(frequency, 0.0)
        time_value = float(self.clock())
        height, width = frame.shape[:2]
        grid_x, grid_y, normalized_x, normalized_y = self._coordinates(
            (height, width),
        )

        phase = time_value * (0.22 + intensity * 0.12)
        amplitude = 1.2 + intensity * 8.0
        edge_weight = 0.45 + 0.75 * np.power(
            np.abs(normalized_x - 0.5) * 2.0,
            1.35,
        )

        wave_x = np.sin(normalized_y * np.pi * 2.7 + phase)
        wave_x += 0.46 * np.sin(
            (normalized_x + normalized_y) * np.pi * 3.2 - phase * 0.7,
        )
        wave_y = np.sin(normalized_x * np.pi * 2.25 - phase * 0.83)
        wave_y += 0.32 * np.sin(
            (normalized_x - normalized_y) * np.pi * 3.6 + phase,
        )

        focus_x = 0.30 + 0.10 * np.sin(time_value * 0.09)
        focus_y = 0.48 + 0.08 * np.cos(time_value * 0.07)
        delta_x = normalized_x - focus_x
        delta_y = normalized_y - focus_y
        local_field = np.exp(
            -((delta_x * delta_x) / 0.085 + (delta_y * delta_y) / 0.14),
        )

        displacement_x = amplitude * (
            wave_x * edge_weight + delta_y * local_field * 2.8
        )
        displacement_y = amplitude * 0.62 * (
            wave_y * (0.55 + local_field) - delta_x * local_field * 2.2
        )
        displacement_x = cv2.resize(
            displacement_x,
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )
        displacement_y = cv2.resize(
            displacement_y,
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )
        map_x = grid_x + displacement_x
        map_y = grid_y + displacement_y
        warped = cv2.remap(
            frame,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        atmosphere_intensity = min(intensity, 1.25)
        darkened = cv2.convertScaleAbs(
            warped,
            alpha=0.94 - atmosphere_intensity * 0.035,
            beta=-4 - int(atmosphere_intensity * 5),
        )
        result = self._add_wandering_fog(
            darkened,
            intensity=atmosphere_intensity,
            time_value=time_value,
        )
        return self._restore_person(result, original, analysis)

    def _coordinates(self, shape):
        cached = self._coordinate_cache.get(shape)
        if cached is not None:
            return cached

        height, width = shape
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        field_width = max(64, width // 4)
        field_height = max(40, height // 4)
        normalized_x, normalized_y = np.meshgrid(
            np.linspace(0.0, 1.0, field_width, dtype=np.float32),
            np.linspace(0.0, 1.0, field_height, dtype=np.float32),
        )
        cached = (grid_x, grid_y, normalized_x, normalized_y)
        self._coordinate_cache[shape] = cached
        return cached

    @staticmethod
    def _add_wandering_fog(frame, intensity, time_value):
        height, width = frame.shape[:2]
        fog_width = max(24, width // 20)
        fog_height = max(16, height // 20)
        x, y = np.meshgrid(
            np.linspace(0.0, 1.0, fog_width, dtype=np.float32),
            np.linspace(0.0, 1.0, fog_height, dtype=np.float32),
        )
        fog = np.sin(x * 8.2 + time_value * 0.11)
        fog += 0.72 * np.sin(y * 6.7 - time_value * 0.08)
        fog += 0.48 * np.sin((x + y) * 11.5 + time_value * 0.045)
        fog = (fog - fog.min()) / max(float(fog.max() - fog.min()), 1e-6)
        fog = cv2.GaussianBlur(fog, (0, 0), 1.25)
        fog = cv2.resize(fog, (width, height), interpolation=cv2.INTER_CUBIC)
        fog = np.clip((fog - 0.46) * 2.25, 0.0, 1.0)

        max_opacity = 0.035 + intensity * 0.095
        alpha = (fog * max_opacity).astype(np.float32)
        fog_color = np.empty_like(frame)
        fog_color[:] = (112, 122, 116)
        return cv2.blendLinear(fog_color, frame, alpha, 1.0 - alpha)

    @staticmethod
    def _restore_person(effect, original, analysis):
        if not analysis:
            return effect

        person_mask = analysis.get("person_mask")
        if person_mask is None or person_mask.size == 0:
            return effect

        height, width = effect.shape[:2]
        if person_mask.shape[:2] != (height, width):
            person_mask = cv2.resize(
                person_mask,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )
        mask = np.clip(person_mask, 0.0, 1.0).astype(np.float32)
        # The detector mask is intentionally generous for keeping entities
        # behind a person.  Effects need a tighter edge, otherwise restoring
        # the brighter original image reveals a pale oval around the head.
        mask = np.clip((mask - 0.34) / 0.46, 0.0, 1.0)
        if float(mask.max()) <= 0.001:
            return effect
        return cv2.blendLinear(original, effect, mask, 1.0 - mask)

    def entities(self, frame):
        dark = cv2.convertScaleAbs(frame, alpha=0.78, beta=-18)

        tint = np.zeros_like(dark)
        tint[:, :, 1] = 25
        tint[:, :, 2] = 8

        result = cv2.addWeighted(dark, 0.9, tint, 0.1, 0)

        return result

    def apply_display_glitch(
        self,
        frame,
        active_modes,
        frequency="0",
        source_frame=None,
        scene_state=None,
    ):
        """Break the finished camera signal and reveal a parallel room."""
        if MODE_GLITCH not in active_modes:
            self._reset_glitch_state()
            return frame

        now = float(self.clock())
        profile = self.GLITCH_PROFILES.get(
            frequency,
            self.GLITCH_PROFILES["0"],
        )
        if float(profile["strength"]) <= 0.0:
            self._reset_glitch_state()
            return frame
        result = self._ambient_glitch(frame, profile, now)

        if not self._glitch_active:
            self._glitch_active = True
            self._rift_scene_index = self.other_world.choose_scene()
            self._rift_anchor = np.eye(3, dtype=np.float32)
            self._world_start_time = now
            self._world_seed = self.random.randrange(0, 2**31 - 1)
            self._next_rift_time = now + self.random.uniform(
                *profile["first_delay"],
            )
        elif self._rift_scene_index is not None:
            self._update_rift_anchor(scene_state)

        if self._rift_frame is not None and now >= self._rift_end_time:
            self._rift_frame = None
            self._next_rift_time = now + self.random.uniform(
                *profile["interval"],
            )

        if (
            self._rift_frame is None
            and self._next_rift_time is not None
            and now >= self._next_rift_time
        ):
            source = frame if source_frame is None else source_frame
            if source.shape[:2] != frame.shape[:2]:
                source = cv2.resize(
                    source,
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            self._start_parallel_rift(source, profile, now)

        if self._rift_frame is not None:
            result = self._composite_parallel_rift(
                result,
                profile,
                now,
                scene_state,
            )

        return result

    def glitch(self, frame, frequency="0"):
        """Compatibility helper for the permanent part of switch 3."""
        profile = self.GLITCH_PROFILES.get(
            frequency,
            self.GLITCH_PROFILES["0"],
        )
        return self._ambient_glitch(frame, profile, float(self.clock()))

    def _ambient_glitch(self, frame, profile, now):
        result = frame.copy()
        height, width = result.shape[:2]
        if height <= 8 or width <= 8:
            return result

        strength = float(profile["strength"])
        channel_shift = max(1, int(round(width * (0.0015 + strength * 0.006))))
        blue = np.roll(result[:, :, 0], -channel_shift, axis=1)
        red = np.roll(result[:, :, 2], channel_shift, axis=1)
        result[:, :, 0] = cv2.addWeighted(
            result[:, :, 0],
            0.48,
            blue,
            0.52,
            0,
        )
        result[:, :, 2] = cv2.addWeighted(
            result[:, :, 2],
            0.48,
            red,
            0.52,
            0,
        )

        noise_limit = max(1, int(round(1.0 + strength * 5.0)))
        noise = self.noise_random.integers(
            -noise_limit,
            noise_limit + 1,
            size=result.shape,
            dtype=np.int16,
        )
        result = np.clip(
            result.astype(np.int16) + noise,
            0,
            255,
        ).astype(np.uint8)

        rolling_y = int((now * (48.0 + strength * 105.0)) % height)
        rolling_height = max(2, int(round(2 + strength * height * 0.012)))
        rolling_bottom = min(height, rolling_y + rolling_height)
        result[rolling_y:rolling_bottom] = cv2.convertScaleAbs(
            result[rolling_y:rolling_bottom],
            alpha=0.72 + strength * 0.10,
            beta=-3,
        )

        tear_tick = int(now * (1.8 + strength * 4.2))
        tear_period = max(3, int(round(8 - strength * 4)))
        if tear_tick % tear_period == 0:
            tear_height = max(2, int(height * (0.006 + strength * 0.018)))
            available = max(1, height - tear_height)
            tear_y = int(
                abs(np.sin(tear_tick * 1.731 + 0.4)) * available,
            )
            tear_shift = max(2, int(width * (0.003 + strength * 0.014)))
            direction = -1 if tear_tick % 2 else 1
            result[tear_y : tear_y + tear_height] = np.roll(
                result[tear_y : tear_y + tear_height],
                direction * tear_shift,
                axis=1,
            )

        return result

    def _start_parallel_rift(self, source, profile, now):
        duration = self.random.uniform(*profile["duration"])
        self._rift_start_time = now
        self._rift_end_time = now + duration
        self._rift_seed = self.random.randrange(0, 2**31 - 1)
        if self._rift_scene_index is not None:
            self._rift_frame = self.other_world.render(
                self._rift_scene_index,
                source.shape[1],
                source.shape[0],
                max(0.0, now - self._world_start_time),
                self._rift_anchor,
                float(profile["strength"]),
                self._world_seed,
            )
        if self._rift_frame is None:
            self._rift_frame = self._build_parallel_world(
                source,
                float(profile["strength"]),
                self._world_seed,
            )
        self._next_rift_time = None

    def _build_parallel_world(self, frame, strength, seed):
        height, width = frame.shape[:2]
        seeded = np.random.default_rng(seed)
        source = frame
        source_corners = np.float32(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        )
        offset_x = width * (0.022 + strength * 0.038)
        offset_y = height * (0.018 + strength * 0.042)
        destination_corners = source_corners + np.float32([
            [seeded.uniform(-offset_x, offset_x), seeded.uniform(-offset_y, offset_y)],
            [seeded.uniform(-offset_x, offset_x), seeded.uniform(-offset_y, offset_y)],
            [seeded.uniform(-offset_x, offset_x), seeded.uniform(-offset_y, offset_y)],
            [seeded.uniform(-offset_x, offset_x), seeded.uniform(-offset_y, offset_y)],
        ])
        perspective = cv2.getPerspectiveTransform(
            source_corners,
            destination_corners,
        )
        warped = cv2.warpPerspective(
            source,
            perspective,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        grid_x, grid_y, _, _ = self._coordinates((height, width))
        vertical_phase = np.linspace(0.0, np.pi * 5.0, height, dtype=np.float32)
        horizontal_phase = np.linspace(0.0, np.pi * 4.0, width, dtype=np.float32)
        displacement = 3.0 + strength * 14.0
        map_x = grid_x + np.sin(vertical_phase + seed * 0.0001)[:, None] * displacement
        map_y = grid_y + np.sin(horizontal_phase - seed * 0.00013)[None, :] * displacement * 0.55
        warped = cv2.remap(
            warped,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        contrast = cv2.createCLAHE(
            clipLimit=1.7,
            tileGridSize=(8, 8),
        ).apply(gray)
        low_frequency = cv2.GaussianBlur(contrast, (0, 0), 2.2)
        detail = cv2.addWeighted(contrast, 1.30, low_frequency, -0.30, -26)
        edges = cv2.Canny(detail, 58, 142)
        parallel = np.empty_like(warped)
        parallel[:, :, 0] = np.clip(
            detail.astype(np.float32) * 0.58
            + edges.astype(np.float32) * 0.28
            + 6,
            0,
            255,
        ).astype(np.uint8)
        parallel[:, :, 1] = np.clip(
            detail.astype(np.float32) * 0.10
            + edges.astype(np.float32) * 0.08,
            0,
            255,
        ).astype(np.uint8)
        parallel[:, :, 2] = np.clip(
            detail.astype(np.float32) * 0.32
            + edges.astype(np.float32) * 0.36
            + 3,
            0,
            255,
        ).astype(np.uint8)
        parallel[detail < 34] = (3, 0, 8)
        return parallel

    def prepare_other_world_frames(self, width, height):
        self.other_world.prepare(width, height)

    def _composite_parallel_rift(self, frame, profile, now, scene_state=None):
        height, width = frame.shape[:2]
        strength = float(profile["strength"])
        age = max(0.0, now - self._rift_start_time)
        world_age = max(0.0, now - self._world_start_time)
        if self._rift_scene_index is not None:
            rendered = self.other_world.render(
                self._rift_scene_index,
                width,
                height,
                world_age,
                self._rift_anchor,
                strength,
                self._world_seed,
            )
            if rendered is not None:
                self._rift_frame = rendered

        if self._rift_frame is None:
            return frame

        step = int(age * (13.0 + strength * 17.0))
        seeded = np.random.default_rng(self._rift_seed + step)

        parallel = self._rift_frame.copy()
        tear_count = 1 + int(round(strength * 5))
        for _ in range(tear_count):
            band_height = int(seeded.integers(3, max(4, int(height * 0.055))))
            band_y = int(seeded.integers(0, max(1, height - band_height)))
            max_shift = max(3, int(width * (0.008 + strength * 0.040)))
            shift = int(seeded.integers(-max_shift, max_shift + 1))
            parallel[band_y : band_y + band_height] = np.roll(
                parallel[band_y : band_y + band_height],
                shift,
                axis=1,
            )

        mask_height = max(18, height // 20)
        mask_width = max(14, width // 48)
        blocks = seeded.random((mask_height, mask_width), dtype=np.float32)
        blocks = cv2.GaussianBlur(blocks, (0, 0), 1.15)
        threshold = float(np.quantile(blocks, 1.0 - profile["coverage"]))
        softness = 0.055 + (1.0 - strength) * 0.035
        alpha_small = np.clip(
            (blocks - threshold + softness) / (softness * 2.0),
            0.0,
            1.0,
        )
        alpha_small = alpha_small * alpha_small * (3.0 - 2.0 * alpha_small)
        alpha = cv2.resize(
            alpha_small,
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )
        alpha = cv2.GaussianBlur(alpha, (0, 0), 1.1)
        flicker = 0.88 + 0.12 * abs(np.sin(age * (19.0 + strength * 11.0)))
        opacity = float(profile["world_opacity"]) * flicker
        alpha = np.clip(alpha * opacity, 0.0, 0.96).astype(np.float32)
        return cv2.blendLinear(parallel, frame, alpha, 1.0 - alpha)

    def _update_rift_anchor(self, scene_state):
        if not scene_state:
            return

        confidence = float(scene_state.get("tracking_confidence", 0.0))
        transform = scene_state.get("camera_transform")
        if confidence < 0.20 or transform is None:
            return

        matrix = np.asarray(transform, dtype=np.float32)
        if matrix.shape != (2, 3) or not np.all(np.isfinite(matrix)):
            return

        step = np.eye(3, dtype=np.float32)
        step[:2] = matrix
        self._rift_anchor = step @ self._rift_anchor

    def _reset_glitch_state(self):
        self._glitch_active = False
        self._next_rift_time = None
        self._rift_start_time = 0.0
        self._rift_end_time = 0.0
        self._rift_frame = None
        self._rift_scene_index = None
        self._rift_anchor = np.eye(3, dtype=np.float32)
        self._world_start_time = 0.0
        self._world_seed = 0

    def beyond(self, frame, frequency="0"):
        """Reveals a translucent ultraviolet ecology inside the real room."""
        intensity = self.FREQUENCY_INTENSITY.get(frequency, 0.0)
        if intensity <= 0.0:
            return frame.copy()

        time_value = float(self.clock())
        height, width = frame.shape[:2]
        grid_x, grid_y, normalized_x, normalized_y = self._coordinates(
            (height, width),
        )

        phase = time_value * (0.18 + intensity * 0.08)
        displacement = 0.45 + intensity * 3.8
        wave_x = np.sin(normalized_y * np.pi * 3.0 + phase)
        wave_x += 0.36 * np.sin(
            (normalized_x + normalized_y) * np.pi * 4.4 - phase * 0.71,
        )
        wave_y = np.sin(normalized_x * np.pi * 2.6 - phase * 0.81)
        wave_y += 0.28 * np.sin(
            (normalized_x - normalized_y) * np.pi * 4.8 + phase,
        )
        wave_x = cv2.resize(wave_x, (width, height), interpolation=cv2.INTER_CUBIC)
        wave_y = cv2.resize(wave_y, (width, height), interpolation=cv2.INTER_CUBIC)
        warped = cv2.remap(
            frame,
            (grid_x + wave_x * displacement).astype(np.float32),
            (grid_y + wave_y * displacement * 0.58).astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        channel_shift = max(1, int(round(width * (0.001 + intensity * 0.004))))
        spectral = warped.copy()
        spectral[:, :, 0] = np.roll(warped[:, :, 0], -channel_shift, axis=1)
        spectral[:, :, 2] = np.roll(warped[:, :, 2], channel_shift, axis=1)
        spectral[:, :, 1] = cv2.convertScaleAbs(
            spectral[:, :, 1],
            alpha=0.78,
            beta=-4,
        )

        violet = np.empty_like(frame)
        violet[:] = (74, 12, 88)
        tint_opacity = 0.055 + intensity * 0.13
        spectral = cv2.addWeighted(
            spectral,
            1.0 - tint_opacity,
            violet,
            tint_opacity,
            0,
        )

        # A slow, irregular veil makes the air itself appear occupied while
        # leaving the room readable underneath.
        field_width = max(18, width // 28)
        field_height = max(12, height // 28)
        x, y = np.meshgrid(
            np.linspace(0.0, 1.0, field_width, dtype=np.float32),
            np.linspace(0.0, 1.0, field_height, dtype=np.float32),
        )
        veil = np.sin(x * 8.7 + phase * 0.8)
        veil += 0.68 * np.sin(y * 7.1 - phase * 0.6)
        veil += 0.42 * np.sin((x + y) * 11.3 + phase * 0.35)
        veil = (veil - veil.min()) / max(float(veil.max() - veil.min()), 1e-6)
        veil = cv2.resize(veil, (width, height), interpolation=cv2.INTER_CUBIC)
        veil = cv2.GaussianBlur(veil, (0, 0), 2.4)
        veil_alpha = np.clip(
            (veil - 0.38) * (0.045 + intensity * 0.10),
            0.0,
            0.16,
        ).astype(np.float32)
        spectral = cv2.blendLinear(
            violet,
            spectral,
            veil_alpha,
            1.0 - veil_alpha,
        )

        reality_weight = 0.94 - intensity * 0.24
        return cv2.addWeighted(
            frame,
            reality_weight,
            spectral,
            1.0 - reality_weight,
            0,
        )
