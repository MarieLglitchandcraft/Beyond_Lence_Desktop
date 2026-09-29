from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np


class OverlayEngine:
    """Loads overlay assets once and blends them safely into camera frames."""

    def __init__(self) -> None:
        self.assets: dict[str, np.ndarray] = {}
        self.failed_assets: set[str] = set()
        self.scaled_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self.scaled_cache_limit = 14
        self.warp_coordinate_cache: OrderedDict[tuple[int, int], tuple] = OrderedDict()
        self.warp_coordinate_cache_limit = 14
        self.warped_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self.warped_cache_limit = 30

    def load_asset(self, name: str, path: str | Path) -> bool:
        asset_path = Path(path)

        if name in self.failed_assets:
            return False

        if not asset_path.is_file():
            print(f"Overlay-Asset fehlt oder ist nicht lesbar: {asset_path}")
            self.failed_assets.add(name)
            return False

        image = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"Overlay-Asset fehlt oder ist nicht lesbar: {asset_path}")
            self.failed_assets.add(name)
            return False

        rgba = self._ensure_bgra(image)
        rgba = self._crop_transparent_margin(rgba)

        if rgba.size == 0:
            if name not in self.failed_assets:
                print(f"Overlay-Asset ist vollständig transparent: {asset_path}")
                self.failed_assets.add(name)
            return False

        self.assets[name] = rgba
        for key in tuple(self.scaled_cache):
            if key[0] == name:
                del self.scaled_cache[key]
        for key in tuple(self.warped_cache):
            if key[0] == name:
                del self.warped_cache[key]
        self.failed_assets.discard(name)
        return True

    def load_assets(self, assets: dict[str, str | Path]) -> None:
        for name, path in assets.items():
            self.load_asset(name, path)

    def draw(
        self,
        frame: np.ndarray,
        asset_name: str,
        center: tuple[int, int],
        target_width: int,
        opacity: float,
        *,
        flip_horizontal: bool = False,
        height_scale: float = 1.0,
        reveal: float = 1.0,
        reveal_from_right: bool = False,
        warp_phase: float | None = None,
        warp_amount: float = 0.0,
        spectral_strength: float = 0.0,
        refract_background: bool = False,
        dissolve: float = 0.0,
        dissolve_seed: int = 0,
        visibility_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        asset = self.assets.get(asset_name)
        if asset is None or frame.size == 0 or target_width <= 0 or opacity <= 0:
            return frame

        resized = self._get_scaled_asset(
            asset_name,
            asset,
            target_width,
            height_scale,
            flip_horizontal,
            spectral_strength,
        )

        reveal = float(np.clip(reveal, 0.0, 1.0))
        if reveal < 1.0:
            resized = resized.copy()
            positions = np.linspace(
                0.0,
                1.0,
                resized.shape[1],
                dtype=np.float32,
            )
            if reveal_from_right:
                positions = positions[::-1]

            feather = min(0.12, max(0.03, reveal))
            horizontal_mask = np.clip(
                (reveal - positions) / feather,
                0.0,
                1.0,
            )
            reveal_mask = np.broadcast_to(
                horizontal_mask,
                resized.shape[:2],
            )
            resized[:, :, 3] = (
                resized[:, :, 3].astype(np.float32) * reveal_mask
            ).astype(np.uint8)

        if warp_phase is not None and warp_amount > 0.0:
            phase_steps = 36
            phase_index = int(round(
                (float(warp_phase) % (np.pi * 2.0))
                / (np.pi * 2.0)
                * phase_steps,
            )) % phase_steps
            quantized_phase = phase_index / phase_steps * np.pi * 2.0
            warp_key = (
                asset_name,
                resized.shape[:2],
                flip_horizontal,
                round(float(spectral_strength), 2),
                round(float(warp_amount), 1),
                phase_index,
            )
            resized = self._organic_warp(
                resized,
                phase=quantized_phase,
                amount=float(warp_amount),
                cache_key=warp_key,
            )

        if dissolve > 0.0:
            resized = self._dissolve_overlay(
                resized,
                float(np.clip(dissolve, 0.0, 1.0)),
                int(dissolve_seed),
            )

        self._blend_clipped(
            frame,
            resized,
            center=center,
            opacity=float(np.clip(opacity, 0.0, 1.0)),
            refract_background=refract_background,
            visibility_mask=visibility_mask,
        )
        return frame

    @staticmethod
    def _dissolve_overlay(
        image: np.ndarray,
        progress: float,
        seed: int,
    ) -> np.ndarray:
        """Breaks a consumed creature into stable, irregular spectral islands."""
        result = image.copy()
        height, width = result.shape[:2]
        noise_height = max(5, height // 28)
        noise_width = max(5, width // 28)
        noise = np.random.default_rng(seed).random(
            (noise_height, noise_width),
            dtype=np.float32,
        )
        noise = cv2.GaussianBlur(noise, (0, 0), 0.85)
        noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
        threshold = progress * 1.12 - 0.08
        survival = np.clip((noise - threshold) / 0.16, 0.0, 1.0)
        result[:, :, 3] = (
            result[:, :, 3].astype(np.float32) * survival
        ).astype(np.uint8)
        return result

    def _organic_warp(
        self,
        image: np.ndarray,
        phase: float,
        amount: float,
        cache_key: tuple | None = None,
    ) -> np.ndarray:
        """Bends the object itself without making it pulse in size."""
        height, width = image.shape[:2]
        if height < 3 or width < 3 or amount <= 0.0:
            return image.copy()

        if cache_key is not None:
            cached = self.warped_cache.get(cache_key)
            if cached is not None:
                self.warped_cache.move_to_end(cache_key)
                return cached

        processing_scale = min(1.0, 480.0 / max(1.0, float(width)))
        if processing_scale < 1.0:
            processing_width = max(3, int(round(width * processing_scale)))
            processing_height = max(3, int(round(height * processing_scale)))
            working = cv2.resize(
                image,
                (processing_width, processing_height),
                interpolation=cv2.INTER_AREA,
            )
            working_amount = amount * processing_scale
        else:
            working = image
            working_amount = amount

        working_height, working_width = working.shape[:2]

        grid_x, grid_y, normalized_x, normalized_y = self._warp_coordinates(
            working_height,
            working_width,
        )

        displacement_x = np.sin(
            normalized_y * np.pi * 3.2 + phase,
        )
        displacement_x += 0.42 * np.sin(
            (normalized_x + normalized_y) * np.pi * 4.6 - phase * 0.63,
        )
        displacement_y = 0.48 * np.sin(
            normalized_x * np.pi * 2.8 - phase * 0.74,
        )
        displacement_y += 0.20 * np.sin(
            (normalized_x - normalized_y) * np.pi * 5.1 + phase,
        )

        map_x = grid_x + displacement_x * working_amount
        map_y = grid_y + displacement_y * working_amount
        warped = cv2.remap(
            working,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        if processing_scale < 1.0:
            warped = cv2.resize(
                warped,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

        if cache_key is not None:
            self.warped_cache[cache_key] = warped
            self.warped_cache.move_to_end(cache_key)
            while len(self.warped_cache) > self.warped_cache_limit:
                self.warped_cache.popitem(last=False)
        return warped

    def _warp_coordinates(self, height: int, width: int) -> tuple:
        key = (height, width)
        cached = self.warp_coordinate_cache.get(key)
        if cached is not None:
            self.warp_coordinate_cache.move_to_end(key)
            return cached

        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        normalized_x = grid_x / max(1.0, float(width - 1))
        normalized_y = grid_y / max(1.0, float(height - 1))
        cached = (grid_x, grid_y, normalized_x, normalized_y)
        self.warp_coordinate_cache[key] = cached
        self.warp_coordinate_cache.move_to_end(key)
        while len(self.warp_coordinate_cache) > self.warp_coordinate_cache_limit:
            self.warp_coordinate_cache.popitem(last=False)
        return cached

    @staticmethod
    def _spectral_colour(image: np.ndarray, strength: float) -> np.ndarray:
        result = image.copy()
        source = result[:, :, :3].astype(np.float32)
        luminance = cv2.cvtColor(result[:, :, :3], cv2.COLOR_BGR2GRAY).astype(
            np.float32,
        )

        spectral = np.empty_like(source)
        spectral[:, :, 0] = np.clip(luminance * 1.08 + 18.0, 0.0, 255.0)
        spectral[:, :, 1] = np.clip(luminance * 0.30 + 3.0, 0.0, 255.0)
        spectral[:, :, 2] = np.clip(luminance * 0.88 + 12.0, 0.0, 255.0)
        mixed = source * (1.0 - strength) + spectral * strength
        result[:, :, :3] = np.clip(mixed, 0.0, 255.0).astype(np.uint8)
        return result

    def _get_scaled_asset(
        self,
        asset_name: str,
        asset: np.ndarray,
        target_width: int,
        height_scale: float,
        flip_horizontal: bool,
        spectral_strength: float = 0.0,
    ) -> np.ndarray:
        width_step = 24
        quantized_width = max(
            width_step,
            int(round(target_width / width_step) * width_step),
        )
        quantized_height_scale = max(0.05, round(height_scale / 0.05) * 0.05)
        quantized_spectral_strength = max(
            0.0,
            min(1.0, round(float(spectral_strength) / 0.05) * 0.05),
        )
        key = (
            asset_name,
            quantized_width,
            quantized_height_scale,
            flip_horizontal,
            quantized_spectral_strength,
        )

        cached = self.scaled_cache.get(key)
        if cached is not None:
            self.scaled_cache.move_to_end(key)
            return cached

        scale = quantized_width / asset.shape[1]
        target_height = max(
            1,
            int(round(asset.shape[0] * scale * quantized_height_scale)),
        )
        resized = cv2.resize(
            asset,
            (quantized_width, target_height),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        if flip_horizontal:
            resized = cv2.flip(resized, 1)
        if quantized_spectral_strength > 0.0:
            resized = self._spectral_colour(
                resized,
                quantized_spectral_strength,
            )

        self.scaled_cache[key] = resized
        self.scaled_cache.move_to_end(key)
        while len(self.scaled_cache) > self.scaled_cache_limit:
            self.scaled_cache.popitem(last=False)

        return resized

    @staticmethod
    def _ensure_bgra(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            alpha = np.full(image.shape, 255, dtype=np.uint8)
            return np.dstack((bgr, alpha))

        if image.shape[2] == 4:
            return image

        if image.shape[2] == 3:
            alpha = np.full(image.shape[:2], 255, dtype=np.uint8)
            return np.dstack((image, alpha))

        raise ValueError(f"Nicht unterstützte Kanalzahl: {image.shape[2]}")

    @staticmethod
    def _crop_transparent_margin(image: np.ndarray) -> np.ndarray:
        alpha = image[:, :, 3]
        points = cv2.findNonZero((alpha > 0).astype(np.uint8))
        if points is None:
            return image[0:0, 0:0]

        x, y, width, height = cv2.boundingRect(points)
        return image[y : y + height, x : x + width]

    @staticmethod
    def _blend_clipped(
        frame: np.ndarray,
        overlay: np.ndarray,
        center: tuple[int, int],
        opacity: float,
        refract_background: bool = False,
        visibility_mask: np.ndarray | None = None,
    ) -> None:
        frame_height, frame_width = frame.shape[:2]
        overlay_height, overlay_width = overlay.shape[:2]
        left = int(center[0] - overlay_width // 2)
        top = int(center[1] - overlay_height // 2)
        right = left + overlay_width
        bottom = top + overlay_height

        frame_left = max(0, left)
        frame_top = max(0, top)
        frame_right = min(frame_width, right)
        frame_bottom = min(frame_height, bottom)

        if frame_left >= frame_right or frame_top >= frame_bottom:
            return

        overlay_left = frame_left - left
        overlay_top = frame_top - top
        overlay_right = overlay_left + (frame_right - frame_left)
        overlay_bottom = overlay_top + (frame_bottom - frame_top)

        source = overlay[overlay_top:overlay_bottom, overlay_left:overlay_right]
        destination = frame[frame_top:frame_bottom, frame_left:frame_right]
        alpha = source[:, :, 3:4].astype(np.float32) / 255.0
        alpha *= opacity
        if visibility_mask is not None and visibility_mask.size:
            if visibility_mask.shape[:2] != (frame_height, frame_width):
                visibility_mask = cv2.resize(
                    visibility_mask,
                    (frame_width, frame_height),
                    interpolation=cv2.INTER_LINEAR,
                )
            clipped_mask = visibility_mask[
                frame_top:frame_bottom,
                frame_left:frame_right,
            ]
            alpha *= np.clip(
                clipped_mask.astype(np.float32),
                0.0,
                1.0,
            )[:, :, None]

        source_colour = source[:, :, :3].astype(np.float32)
        if refract_background and destination.shape[0] >= 3 and destination.shape[1] >= 3:
            refracted = cv2.GaussianBlur(destination, (0, 0), 2.2)
            shift = max(1, int(round(destination.shape[1] * 0.012)))
            refracted = np.roll(refracted, shift, axis=1)
            source_colour = (
                source_colour * 0.46
                + refracted.astype(np.float32) * 0.54
            )

        blended = source_colour * alpha
        blended += destination.astype(np.float32) * (1.0 - alpha)
        destination[:] = np.clip(blended, 0, 255).astype(np.uint8)
