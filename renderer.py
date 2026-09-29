import cv2
import numpy as np
import time

from config import MODE_BEYOND
from hud import HUD


class Renderer:
    CAPTURE_MESSAGE = "SPECTRAL IMAGE CAPTURED"

    def __init__(self, monotonic_clock=time.monotonic):
        self.start_time = time.time()
        self.monotonic_clock = monotonic_clock

        self.start_screen_duration = 3.0
        self.calibration_duration = 5.0
        self.capture_preview_duration = 2.0
        self.capture_flash_duration = 0.12
        self._capture_preview_frame = None
        self._capture_preview_started = 0.0

        self.hud = HUD()
        self._scanline_masks = {}
        self._vignette_masks = {}
        self._screen_masks = {}

    def render(self, frame, active_modes):
        frame = self.render_camera(frame, active_modes)
        return self.render_interface(frame, active_modes)

    def render_camera(self, frame, active_modes):
        """Apply only the camera look; UI text is deliberately excluded."""
        return self.apply_camera_style(frame, active_modes)

    def render_interface(self, frame, active_modes):
        """Draw start/calibration/HUD after any selfie mirroring."""
        elapsed = time.time() - self.start_time

        if elapsed < self.start_screen_duration:
            frame = self.hud.draw_start_screen(
                frame,
                elapsed,
                self.start_screen_duration,
            )
            return self.apply_screen_style(frame)

        calibration_elapsed = elapsed - self.start_screen_duration

        if calibration_elapsed < self.calibration_duration:
            frame = self.hud.draw_calibration(
                frame,
                calibration_elapsed,
                self.calibration_duration,
            )
            return frame

        runtime = calibration_elapsed - self.calibration_duration

        frame = self.hud.draw_active_hud(frame, runtime, active_modes)

        return frame

    def set_frequency(self, frequency):
        self.hud.set_frequency(frequency)

    @staticmethod
    def apply_selfie_camera_mirror(frame, selfie_mode=False):
        """Mirror camera content before unmirrored HUD text is drawn."""
        if not selfie_mode:
            return frame
        return cv2.flip(frame, 1)

    @staticmethod
    def apply_selfie_display_rotation(frame, selfie_mode=False):
        """Turn the composed app frame for the mechanically flipped panel."""
        if not selfie_mode:
            return frame
        return cv2.rotate(frame, cv2.ROTATE_180)

    def show_capture_confirmation(
        self,
        frame,
        *,
        duration=2.0,
        flash_duration=0.12,
    ):
        """Freeze the saved image after a short simulated camera flash."""
        self._capture_preview_frame = frame.copy()
        self._capture_preview_started = self.monotonic_clock()
        self.capture_preview_duration = max(0.25, float(duration))
        self.capture_flash_duration = max(
            0.02,
            min(float(flash_duration), self.capture_preview_duration),
        )

    def apply_capture_feedback(self, live_frame):
        if self._capture_preview_frame is None:
            return live_frame

        elapsed = self.monotonic_clock() - self._capture_preview_started
        total_duration = (
            self.capture_flash_duration + self.capture_preview_duration
        )
        if elapsed >= total_duration:
            self._capture_preview_frame = None
            return live_frame

        preview = self._capture_preview_frame.copy()
        if elapsed < self.capture_flash_duration:
            # An almost white first frame is clearly readable as a camera flash
            # but still leaves a trace of the captured scene underneath.
            flash = np.full_like(preview, 255)
            strength = 0.90 * (
                1.0 - elapsed / max(self.capture_flash_duration, 0.001)
            )
            return cv2.addWeighted(
                preview,
                1.0 - strength,
                flash,
                strength,
                0,
            )

        height, width = preview.shape[:2]
        font_scale = max(0.48, min(0.95, width / 1024.0 * 0.78))
        thickness = max(1, int(round(font_scale * 2)))
        text_size, baseline = cv2.getTextSize(
            self.CAPTURE_MESSAGE,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        text_width, text_height = text_size
        center_x = width // 2
        center_y = height // 2
        padding_x = max(16, int(round(width * 0.025)))
        padding_y = max(12, int(round(height * 0.025)))

        panel = preview.copy()
        cv2.rectangle(
            panel,
            (
                center_x - text_width // 2 - padding_x,
                center_y - text_height // 2 - padding_y,
            ),
            (
                center_x + text_width // 2 + padding_x,
                center_y + text_height // 2 + padding_y + baseline,
            ),
            (3, 18, 3),
            -1,
        )
        preview = cv2.addWeighted(panel, 0.72, preview, 0.28, 0)
        cv2.putText(
            preview,
            self.CAPTURE_MESSAGE,
            (
                center_x - text_width // 2,
                center_y + text_height // 2,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (160, 255, 160),
            thickness,
            cv2.LINE_AA,
        )
        return preview

    def apply_camera_style(self, frame, active_modes=()):
        green = self.apply_green_monitor_look(frame)
        if MODE_BEYOND in active_modes and self.hud.frequency != "0":
            intensity = {
                "I": 0.12,
                "II": 0.25,
                "III": 0.45,
                "IV": 0.72,
                "V": 1.0,
            }.get(self.hud.frequency, 0.0)
            spectral_visibility = 0.16 + intensity * 0.46
            frame = cv2.addWeighted(
                green,
                1.0 - spectral_visibility,
                frame,
                spectral_visibility,
                0,
            )
        else:
            frame = green
        return self.apply_screen_style(frame)

    def apply_screen_style(self, frame):
        rows, cols = frame.shape[:2]
        shape = (rows, cols)
        mask = self._screen_masks.get(shape)

        if mask is None:
            scanlines = np.ones((rows, cols), dtype=np.float32)
            scanlines[::3, :] = 0.45
            kernel_x = cv2.getGaussianKernel(cols, cols / 2)
            kernel_y = cv2.getGaussianKernel(rows, rows / 2)
            vignette = kernel_y @ kernel_x.T
            vignette = 0.45 + 0.55 * (vignette / vignette.max())
            mask = (scanlines * vignette)[:, :, np.newaxis]
            self._screen_masks[shape] = mask

        return (frame.astype(np.float32) * mask).astype(np.uint8)

    def apply_green_monitor_look(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        result = np.zeros_like(frame)
        result[:, :, 1] = gray
        result[:, :, 0] = (gray * 0.15).astype(np.uint8)
        result[:, :, 2] = (gray * 0.08).astype(np.uint8)

        result = cv2.convertScaleAbs(result, alpha=1.25, beta=-18)

        return result

    def apply_scanlines(self, frame):
        shape = frame.shape[:2]
        mask = self._scanline_masks.get(shape)
        if mask is None:
            mask = np.ones((*shape, 1), dtype=np.float32)
            mask[::3, :, 0] = 0.45
            self._scanline_masks[shape] = mask

        return (frame.astype(np.float32) * mask).astype(np.uint8)

    def apply_vignette(self, frame):
        rows, cols = frame.shape[:2]

        shape = (rows, cols)
        mask = self._vignette_masks.get(shape)
        if mask is None:
            kernel_x = cv2.getGaussianKernel(cols, cols / 2)
            kernel_y = cv2.getGaussianKernel(rows, rows / 2)
            kernel = kernel_y @ kernel_x.T
            mask = (0.45 + 0.55 * (kernel / kernel.max()))[:, :, np.newaxis]
            self._vignette_masks[shape] = mask

        return (frame.astype(np.float32) * mask).astype(np.uint8)
