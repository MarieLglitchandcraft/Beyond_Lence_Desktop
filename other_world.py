from dataclasses import dataclass
from pathlib import Path
import math

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class OtherWorldScene:
    name: str
    layers: tuple[Path, ...]


class LayeredOtherWorld:
    """Loads aligned scene layers and renders anchored depth movement."""

    DEPTH_FACTORS = (0.68, 0.84, 1.0, 1.16)
    MOTION_FACTORS = (0.12, 0.34, 0.68, 1.0)
    FAR_BEHAVIORS = {
        "scene_01": {
            "rise_rate": 0.0025,
            "zoom_rate": 0.0025,
            "max_zoom": 1.08,
        },
        "scene_02": {
            "rise_rate": 0.0,
            "zoom_rate": 0.0060,
            "max_zoom": 1.15,
        },
        "scene_03": {
            "rise_rate": 0.0032,
            "zoom_rate": 0.0,
            "max_zoom": 1.0,
        },
        "scene_04": {
            "rise_rate": 0.0030,
            "zoom_rate": 0.0,
            "max_zoom": 1.0,
        },
    }

    def __init__(self, directory, random_source):
        self.directory = Path(directory) if directory is not None else None
        self.random = random_source
        self.scenes = self._discover_scenes()
        self._layer_cache = {}
        self._prepared_sizes = set()
        self._last_scene_index = -1

    def _discover_scenes(self):
        if self.directory is None or not self.directory.is_dir():
            return []

        scenes = []
        for scene_directory in sorted(
            path for path in self.directory.iterdir() if path.is_dir()
        ):
            layers = tuple(
                sorted(
                    path
                    for path in scene_directory.iterdir()
                    if path.is_file()
                    and path.suffix.lower() in IMAGE_EXTENSIONS
                    and len(path.stem) >= 2
                    and path.stem[:2].isdigit()
                )
            )
            if layers:
                scenes.append(OtherWorldScene(scene_directory.name, layers))

        # A single flat image remains a useful fallback during asset work.
        for path in sorted(
            path
            for path in self.directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            scenes.append(OtherWorldScene(path.stem, (path,)))

        return scenes

    def prepare(self, width, height):
        size = (int(width), int(height))
        if size in self._prepared_sizes:
            return

        render_width, render_height, _ = self._render_size(*size)
        for scene_index in range(len(self.scenes)):
            self._get_layers(scene_index, render_width, render_height)
        self._prepared_sizes.add(size)

    def choose_scene(self):
        count = len(self.scenes)
        if count == 0:
            return None

        if count == 1:
            index = 0
        elif self._last_scene_index < 0:
            index = self.random.randrange(count)
        else:
            offset = 1 + self.random.randrange(count - 1)
            index = (self._last_scene_index + offset) % count

        self._last_scene_index = index
        return index

    def render(
        self,
        scene_index,
        width,
        height,
        elapsed,
        anchor_matrix,
        strength,
        seed,
    ):
        output_width = int(width)
        output_height = int(height)
        width, height, render_scale = self._render_size(
            output_width,
            output_height,
        )
        layers = self._get_layers(scene_index, width, height)
        if not layers:
            return None

        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        anchor = np.asarray(anchor_matrix, dtype=np.float32)
        if anchor.shape != (3, 3):
            anchor = np.eye(3, dtype=np.float32)

        for layer_index, layer in enumerate(layers):
            depth = self._factor(self.DEPTH_FACTORS, layer_index)
            motion = self._factor(self.MOTION_FACTORS, layer_index)
            transform = anchor[:2].copy()
            transform[0, 2] *= depth * render_scale
            transform[1, 2] *= depth * render_scale

            phase = seed * 0.00031 + layer_index * 1.73
            amplitude = (0.45 + strength * 4.8) * motion * render_scale
            transform[0, 2] += math.sin(
                elapsed * (0.24 + layer_index * 0.035) + phase,
            ) * amplitude
            transform[1, 2] += math.cos(
                elapsed * (0.18 + layer_index * 0.028) - phase,
            ) * amplitude * 0.62

            if layer_index == 1:
                behavior = self.FAR_BEHAVIORS.get(
                    self.scenes[scene_index].name.lower(),
                )
                if behavior:
                    zoom = min(
                        1.0 + elapsed * behavior["zoom_rate"],
                        behavior["max_zoom"],
                    )
                    rise = min(
                        elapsed * behavior["rise_rate"] * height,
                        height * 0.12,
                    )
                    transform[:, :2] *= zoom
                    transform[0, 2] += (1.0 - zoom) * width * 0.5
                    transform[1, 2] += (1.0 - zoom) * height * 0.5 - rise

            border_mode = cv2.BORDER_REFLECT_101 if layer_index == 0 else cv2.BORDER_CONSTANT
            warped = cv2.warpAffine(
                layer,
                transform,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=border_mode,
                borderValue=(0, 0, 0, 0),
            )
            alpha = warped[:, :, 3].astype(np.float32) / 255.0
            if layer_index == 0:
                # Background layers fill the scene even when their exported
                # alpha is slightly below 255.
                canvas = warped[:, :, :3].copy()
                continue

            if float(alpha.max()) <= 0.001:
                continue
            canvas = cv2.blendLinear(
                warped[:, :, :3],
                canvas,
                alpha,
                1.0 - alpha,
            )

        if (width, height) != (output_width, output_height):
            canvas = cv2.resize(
                canvas,
                (output_width, output_height),
                interpolation=cv2.INTER_LINEAR,
            )
        return canvas

    def _get_layers(self, scene_index, width, height):
        key = (int(scene_index), int(width), int(height))
        cached = self._layer_cache.get(key)
        if cached is not None:
            return cached

        if not 0 <= scene_index < len(self.scenes):
            return ()

        fitted = []
        for path in self.scenes[scene_index].layers:
            layer = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if layer is None or layer.size == 0:
                continue
            layer = self._ensure_rgba(layer)
            fitted.append(self._cover_fit(layer, width, height))

        result = tuple(fitted)
        self._layer_cache[key] = result
        return result

    @staticmethod
    def _ensure_rgba(image):
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        return image

    @staticmethod
    def _cover_fit(image, width, height):
        source_height, source_width = image.shape[:2]
        scale = max(width / source_width, height / source_height)
        resized_width = max(width, int(round(source_width * scale)))
        resized_height = max(height, int(round(source_height * scale)))
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
        )
        left = max(0, (resized_width - width) // 2)
        top = max(0, (resized_height - height) // 2)
        return resized[top : top + height, left : left + width].copy()

    @staticmethod
    def _factor(values, layer_index):
        if layer_index < len(values):
            return values[layer_index]
        return values[-1] + (layer_index - len(values) + 1) * 0.12

    @staticmethod
    def _render_size(width, height):
        width = int(width)
        height = int(height)
        if width < 640:
            return width, height, 1.0

        scale = 0.72
        return (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
            scale,
        )
