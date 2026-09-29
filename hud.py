import cv2
import numpy as np
import time

from config import (
    MODE_BEYOND,
    MODE_ENTITIES,
    MODE_GLITCH,
    MODE_NORMAL,
    MODE_SOFT,
)


MODE_LABELS = {
    MODE_NORMAL: "NORMAL",
    MODE_SOFT: "HEXENHAUS",
    MODE_ENTITIES: "ENTITAETEN",
    MODE_GLITCH: "GLITCH",
    MODE_BEYOND: "FROM BEYOND",
}


class HUD:
    def __init__(self, clock=time.monotonic):
        self.frequency = "0"
        self.clock = clock
        self.capture_message_until = 0.0

    def set_frequency(self, frequency):
        self.frequency = frequency

    def show_capture_confirmation(self, duration=1.25):
        self.capture_message_until = self.clock() + max(0.1, float(duration))

    def draw_start_screen(self, frame, elapsed, duration):
        height, width = frame.shape[:2]
        screen = np.zeros_like(frame)

        progress = min(elapsed / duration, 1.0)

        center_x = width // 2
        center_y = height // 2

        cv2.putText(screen, "BEYOND LENCE", (center_x - 190, center_y - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (120, 255, 120), 2, cv2.LINE_AA)

        cv2.putText(screen, "SPECTRAL FIELD CAMERA", (center_x - 175, center_y - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 200, 90), 1, cv2.LINE_AA)

        bar_total = 420
        bar_width = int(bar_total * progress)

        cv2.rectangle(screen, (center_x - bar_total // 2, center_y + 10),
                      (center_x + bar_total // 2, center_y + 32), (70, 160, 70), 1)

        cv2.rectangle(screen, (center_x - bar_total // 2, center_y + 10),
                      (center_x - bar_total // 2 + bar_width, center_y + 32), (80, 255, 80), -1)

        cv2.putText(screen, f"SYSTEM BOOT {int(progress * 100)}%",
                    (center_x - 110, center_y + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 255, 120), 1, cv2.LINE_AA)

        return screen

    def draw_calibration(self, frame, elapsed, duration):
        height, width = frame.shape[:2]

        progress = min(elapsed / duration, 1.0)
        percent = int(progress * 100)

        center_x = width // 2
        center_y = height // 2

        messages = [
            "CALIBRATION UNIT",
            "ALIGNING SPECTRAL FIELD",
            "MAPPING DARK REGIONS",
            "STABILIZING FREQUENCY",
        ]

        current_index = min(int(progress * len(messages)), len(messages) - 1)

        cv2.rectangle(frame, (center_x - 300, center_y - 110),
                      (center_x + 300, center_y + 120), (20, 80, 20), -1)

        cv2.rectangle(frame, (center_x - 300, center_y - 110),
                      (center_x + 300, center_y + 120), (120, 255, 120), 1)

        cv2.putText(frame, messages[current_index],
                    (center_x - 230, center_y - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 255, 120), 1, cv2.LINE_AA)

        bar_total = 460
        bar_width = int(bar_total * progress)

        cv2.rectangle(frame, (center_x - bar_total // 2, center_y),
                      (center_x + bar_total // 2, center_y + 22), (70, 160, 70), 1)

        cv2.rectangle(frame, (center_x - bar_total // 2, center_y),
                      (center_x - bar_total // 2 + bar_width, center_y + 22), (80, 255, 80), -1)

        cv2.putText(frame, f"{percent}%",
                    (center_x - 25, center_y + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 255, 120), 1, cv2.LINE_AA)

        return frame

    @staticmethod
    def format_active_modes(active_modes):
        labels = [
            MODE_LABELS[mode]
            for mode in active_modes
            if mode in MODE_LABELS and mode != MODE_NORMAL
        ]

        if not labels:
            return "MODUS NORMAL"

        prefix = "MODUS" if len(labels) == 1 else "MODI"
        return f"{prefix} {' + '.join(labels)}"

    @staticmethod
    def right_aligned_position(frame, text, y, font_scale, thickness=1, margin=35):
        text_width = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )[0][0]
        return max(20, frame.shape[1] - margin - text_width), y

    def draw_active_hud(self, frame, runtime, active_modes=()):
        runtime = int(runtime)

        hours = runtime // 3600
        minutes = (runtime % 3600) // 60
        seconds = runtime % 60

        timestamp = f"{hours:02}:{minutes:02}:{seconds:02}"

        cv2.putText(frame, "BEYOND LENCE // DETECTION ACTIVE", (35, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 255, 120), 1, cv2.LINE_AA)

        frequency_text = f"FREQUENCY {self.frequency}"
        frequency_position = self.right_aligned_position(
            frame,
            frequency_text,
            40,
            0.75,
        )
        cv2.putText(frame, frequency_text, frequency_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 255, 120), 1, cv2.LINE_AA)

        mode_text = self.format_active_modes(active_modes)
        mode_position = self.right_aligned_position(
            frame,
            mode_text,
            68,
            0.48,
        )
        cv2.putText(frame, mode_text, mode_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (90, 220, 90), 1, cv2.LINE_AA)

        cv2.putText(frame, f"TIME {timestamp}", (35, frame.shape[0] - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 255, 120), 1, cv2.LINE_AA)

        if self.clock() < self.capture_message_until:
            message = "SPECTRAL IMAGE CAPTURED"
            message_position = self.right_aligned_position(
                frame,
                message,
                frame.shape[0] - 35,
                0.58,
            )
            cv2.putText(
                frame,
                message,
                message_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (170, 255, 170),
                2,
                cv2.LINE_AA,
            )

        return frame
