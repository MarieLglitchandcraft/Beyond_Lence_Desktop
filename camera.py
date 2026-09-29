import cv2
import numpy as np

from config import CAMERA_INDEX, FRAME_HEIGHT, FRAME_WIDTH
from platform_utils import is_raspberry_pi


def validate_camera_frame(frame) -> tuple[bool, str]:
    """Return a precise diagnostic instead of passing malformed frames onward."""
    if frame is None:
        return False, "Kamera lieferte None."
    if not isinstance(frame, np.ndarray):
        return False, f"Kamera lieferte {type(frame).__name__} statt ndarray."
    if frame.size == 0:
        return False, "Kamera lieferte ein leeres Frame."
    if frame.ndim != 3:
        return False, f"Unerwartete Frame-Dimension: {frame.ndim}."
    if frame.shape[0] < 2 or frame.shape[1] < 2:
        return False, f"Ungueltige Frame-Groesse: {frame.shape}."
    if frame.shape[2] < 3:
        return False, f"Zu wenige Farbkanaele: {frame.shape}."
    return True, "ok"


class MacCamera:
    def __init__(self):
        self.backend_name = "OpenCV VideoCapture"
        self.requested_size = (FRAME_WIDTH, FRAME_HEIGHT)
        self.last_raw_shape = None
        self.last_output_shape = None
        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not self.camera.isOpened():
            raise RuntimeError("Kamera konnte nicht geöffnet werden.")

    def read(self):
        success, frame = self.camera.read()

        if not success:
            return None

        self.last_raw_shape = tuple(frame.shape)
        self._report_portrait_frame(frame)
        if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
            frame = cv2.resize(
                frame,
                (FRAME_WIDTH, FRAME_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )

        self.last_output_shape = tuple(frame.shape)
        return frame

    @staticmethod
    def _report_portrait_frame(frame):
        if frame.shape[0] > frame.shape[1] and FRAME_WIDTH > FRAME_HEIGHT:
            print(
                "WARNUNG: Kamera liefert Hochformat "
                f"{frame.shape[1]}x{frame.shape[0]}; keine automatische "
                "Drehung vorgenommen.",
                flush=True,
            )

    def release(self):
        self.camera.release()


class PiCamera:
    """CSI camera backend for current Raspberry Pi OS using Picamera2."""

    def __init__(self, camera_factory=None):
        self.backend_name = "Raspberry Pi CSI / Picamera2"
        self.requested_size = (FRAME_WIDTH, FRAME_HEIGHT)
        self.last_raw_shape = None
        self.last_output_shape = None
        if camera_factory is None:
            try:
                from picamera2 import Picamera2
            except (ImportError, ModuleNotFoundError) as error:
                raise RuntimeError(
                    "Picamera2 fehlt. Auf dem Raspberry Pi bitte "
                    "'sudo apt install python3-picamera2' ausfuehren."
                ) from error
            camera_factory = Picamera2

        self.camera = camera_factory()
        configuration = self.camera.create_video_configuration(
            main={
                "size": (FRAME_WIDTH, FRAME_HEIGHT),
                # Picamera2's RGB888 array layout is directly usable by the
                # OpenCV processing pipeline on Raspberry Pi OS.
                "format": "RGB888",
            },
            buffer_count=4,
        )
        self.camera.configure(configuration)
        self.camera.start()

    def read(self):
        frame = self.camera.capture_array("main")
        if frame is None or frame.size == 0:
            return None

        frame = np.asarray(frame)
        self.last_raw_shape = tuple(frame.shape)
        if frame.ndim != 3 or frame.shape[2] < 3:
            return None
        if frame.shape[2] > 3:
            frame = frame[:, :, :3]
        if frame.shape[0] > frame.shape[1] and FRAME_WIDTH > FRAME_HEIGHT:
            print(
                "WARNUNG: Picamera2 liefert Hochformat "
                f"{frame.shape[1]}x{frame.shape[0]}; keine automatische "
                "Drehung vorgenommen.",
                flush=True,
            )
        if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
            frame = cv2.resize(
                frame,
                (FRAME_WIDTH, FRAME_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
        frame = np.ascontiguousarray(frame)
        self.last_output_shape = tuple(frame.shape)
        return frame

    def release(self):
        try:
            self.camera.stop()
        finally:
            close = getattr(self.camera, "close", None)
            if close is not None:
                close()


def create_camera(force_pi: bool | None = None):
    use_pi_camera = is_raspberry_pi() if force_pi is None else bool(force_pi)
    if use_pi_camera:
        print("Kamera: Raspberry Pi CSI / Picamera2")
        return PiCamera()
    print("Kamera: OpenCV")
    return MacCamera()
