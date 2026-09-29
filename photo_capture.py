from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


DEFAULT_PHOTO_DIRECTORY = Path(__file__).resolve().parent / "photos"


class PhotoCapture:
    def __init__(self, output_directory=DEFAULT_PHOTO_DIRECTORY, clock=None):
        self.output_directory = Path(output_directory)
        self.clock = clock or datetime.now

    def save(self, frame: np.ndarray) -> Path | None:
        if frame is None or frame.size == 0:
            return None

        self.output_directory.mkdir(parents=True, exist_ok=True)
        timestamp = self.clock().strftime("%Y-%m-%d_%H-%M-%S_%f")
        path = self.output_directory / f"beyond_lence_{timestamp}.png"
        if not cv2.imwrite(str(path), frame):
            return None
        return path
