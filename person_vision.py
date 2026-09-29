from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2
import numpy as np


MODEL_DIRECTORY = Path(__file__).resolve().parent / "models" / "mediapipe"
FACE_MODEL = MODEL_DIRECTORY / "blaze_face_short_range.tflite"
POSE_MODEL = MODEL_DIRECTORY / "pose_landmarker_lite.task"
PERSON_MODEL = MODEL_DIRECTORY / "selfie_segmenter.tflite"


class PersonVisionEngine:
    """Optional MediaPipe face/pose detection with a soft person mask."""

    def __init__(
        self,
        *,
        analysis_interval: int = 3,
        work_width: int = 512,
        face_model: str | Path = FACE_MODEL,
        pose_model: str | Path = POSE_MODEL,
        person_model: str | Path = PERSON_MODEL,
    ) -> None:
        self.analysis_interval = max(1, int(analysis_interval))
        self.work_width = max(192, int(work_width))
        self.face_model = Path(face_model)
        self.pose_model = Path(pose_model)
        self.person_model = Path(person_model)
        self.frame_number = 0
        self.face_detector = None
        self.pose_landmarker = None
        self.person_segmenter = None
        self.mp = None
        self.status = "Personenerkennung nicht initialisiert"
        self._last_result = self.empty_result()
        self._initialize()

    @property
    def available(self) -> bool:
        return any((
            self.face_detector is not None,
            self.pose_landmarker is not None,
            self.person_segmenter is not None,
        ))

    @staticmethod
    def empty_result() -> dict[str, Any]:
        return {
            "faces": [],
            "bodies": [],
            "pose_landmarks": [],
            "person_mask": None,
            "person_detection_available": False,
            "person_detection_status": "nicht initialisiert",
        }

    def _initialize(self) -> None:
        missing_models = [
            path.name
            for path in (self.face_model, self.pose_model, self.person_model)
            if not path.is_file()
        ]
        if missing_models:
            self.status = (
                "Personenerkennung aus (Modelle fehlen: "
                + ", ".join(missing_models)
                + ")"
            )
            return

        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except (ImportError, ModuleNotFoundError):
            self.status = "Personenerkennung aus (MediaPipe fehlt)"
            return

        if not self._backend_probe_succeeds():
            self.status = (
                "Personenerkennung aus "
                "(MediaPipe-Grafikdienst in dieser Umgebung nicht verfügbar)"
            )
            return

        self.mp = mp
        errors = []

        if self.face_model.is_file():
            try:
                face_options = vision.FaceDetectorOptions(
                    base_options=python.BaseOptions(
                        model_asset_path=str(self.face_model),
                        delegate=python.BaseOptions.Delegate.CPU,
                    ),
                    running_mode=vision.RunningMode.IMAGE,
                    min_detection_confidence=0.45,
                )
                self.face_detector = vision.FaceDetector.create_from_options(
                    face_options,
                )
            except (RuntimeError, ValueError) as error:
                errors.append(f"Gesicht: {error}")
        else:
            errors.append(f"Gesichtsmodell fehlt: {self.face_model.name}")

        if self.pose_model.is_file():
            try:
                pose_options = vision.PoseLandmarkerOptions(
                    base_options=python.BaseOptions(
                        model_asset_path=str(self.pose_model),
                        delegate=python.BaseOptions.Delegate.CPU,
                    ),
                    running_mode=vision.RunningMode.IMAGE,
                    num_poses=2,
                    min_pose_detection_confidence=0.45,
                    min_pose_presence_confidence=0.35,
                    min_tracking_confidence=0.35,
                    output_segmentation_masks=True,
                )
                self.pose_landmarker = vision.PoseLandmarker.create_from_options(
                    pose_options,
                )
            except (RuntimeError, ValueError) as error:
                errors.append(f"Körper: {error}")
        else:
            errors.append(f"Körpermodell fehlt: {self.pose_model.name}")

        if self.person_model.is_file():
            try:
                segmenter_options = vision.ImageSegmenterOptions(
                    base_options=python.BaseOptions(
                        model_asset_path=str(self.person_model),
                        delegate=python.BaseOptions.Delegate.CPU,
                    ),
                    running_mode=vision.RunningMode.IMAGE,
                    output_confidence_masks=True,
                    output_category_mask=False,
                )
                self.person_segmenter = vision.ImageSegmenter.create_from_options(
                    segmenter_options,
                )
            except (RuntimeError, ValueError) as error:
                errors.append(f"Personenmaske: {error}")
        else:
            errors.append(f"Maskenmodell fehlt: {self.person_model.name}")

        if self.available:
            active = []
            if self.face_detector is not None:
                active.append("Gesicht")
            if self.pose_landmarker is not None:
                active.append("Pose")
            if self.person_segmenter is not None:
                active.append("Personenmaske")
            self.status = f"Personenerkennung aktiv ({' + '.join(active)})"
        else:
            self.status = "Personenerkennung aus (" + "; ".join(errors) + ")"

    def _backend_probe_succeeds(self) -> bool:
        probe = Path(__file__).resolve().parent / "mediapipe_probe.py"
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(probe),
                    str(self.face_model),
                    str(self.pose_model),
                    str(self.person_model),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=12.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False

        return result.returncode == 0

    def analyze(self, frame: np.ndarray) -> dict[str, Any]:
        self.frame_number += 1
        if not self.available or frame.size == 0:
            self._last_result["person_detection_status"] = self.status
            return self._last_result

        if self.frame_number > 1 and self.frame_number % self.analysis_interval != 1:
            self._last_result["person_detection_status"] = self.status
            return self._last_result

        frame_height, frame_width = frame.shape[:2]
        scale = min(1.0, self.work_width / max(frame_width, 1))
        work_size = (
            max(1, int(round(frame_width * scale))),
            max(1, int(round(frame_height * scale))),
        )
        work_frame = cv2.resize(
            frame,
            work_size,
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(work_frame, cv2.COLOR_BGR2RGB)
        media_image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )

        faces = []
        bodies = []
        pose_landmarks = []
        person_mask = np.zeros((frame_height, frame_width), dtype=np.float32)

        try:
            if self.face_detector is not None:
                face_result = self.face_detector.detect(media_image)
                faces = self._extract_faces(
                    face_result,
                    frame_size=(frame_width, frame_height),
                    work_size=work_size,
                )

            if self.person_segmenter is not None:
                segment_result = self.person_segmenter.segment(media_image)
                person_mask = self._extract_person_segmentation(
                    segment_result,
                    frame_size=(frame_width, frame_height),
                )

            if self.pose_landmarker is not None:
                pose_result = self.pose_landmarker.detect(media_image)
                pose_landmarks, bodies = self._extract_poses(
                    pose_result,
                    frame_size=(frame_width, frame_height),
                )
                pose_mask = self._extract_segmentation_mask(
                    pose_result,
                    frame_size=(frame_width, frame_height),
                )
                person_mask = np.maximum(person_mask, pose_mask)
        except (RuntimeError, ValueError) as error:
            self.status = f"Personenerkennung pausiert ({error})"
            self._last_result["person_detection_status"] = self.status
            return self._last_result

        segmented_bodies = self._boxes_from_mask(person_mask)
        if segmented_bodies:
            bodies = segmented_bodies

        self._add_detection_fallbacks(person_mask, faces, bodies)
        if np.any(person_mask):
            blur_size = max(9, int(round(min(frame_width, frame_height) * 0.025)))
            if blur_size % 2 == 0:
                blur_size += 1
            person_mask = cv2.GaussianBlur(person_mask, (blur_size, blur_size), 0)
            person_mask = np.clip(person_mask, 0.0, 1.0).astype(np.float32)
        else:
            person_mask = None

        self._last_result = {
            "faces": faces,
            "bodies": bodies,
            "pose_landmarks": pose_landmarks,
            "person_mask": person_mask,
            "person_detection_available": True,
            "person_detection_status": self.status,
        }
        return self._last_result

    @staticmethod
    def _extract_faces(
        result: Any,
        *,
        frame_size: tuple[int, int],
        work_size: tuple[int, int],
    ) -> list[tuple[int, int, int, int]]:
        frame_width, frame_height = frame_size
        work_width, work_height = work_size
        scale_x = frame_width / max(work_width, 1)
        scale_y = frame_height / max(work_height, 1)
        faces = []

        for detection in getattr(result, "detections", []):
            box = detection.bounding_box
            x = int(round(box.origin_x * scale_x))
            y = int(round(box.origin_y * scale_y))
            width = int(round(box.width * scale_x))
            height = int(round(box.height * scale_y))
            faces.append(PersonVisionEngine._clip_box(
                (x, y, width, height),
                frame_size,
            ))

        return [box for box in faces if box[2] > 0 and box[3] > 0]

    @staticmethod
    def _extract_poses(
        result: Any,
        *,
        frame_size: tuple[int, int],
    ) -> tuple[list[list[tuple[float, float]]], list[tuple[int, int, int, int]]]:
        frame_width, frame_height = frame_size
        poses = []
        bodies = []

        for landmarks in getattr(result, "pose_landmarks", []):
            visible = []
            normalized = []
            for landmark in landmarks:
                visibility = float(getattr(landmark, "visibility", 1.0) or 0.0)
                presence = float(getattr(landmark, "presence", 1.0) or 0.0)
                x = float(landmark.x)
                y = float(landmark.y)
                normalized.append((x, y))
                if visibility >= 0.25 and presence >= 0.25:
                    visible.append((x, y))

            poses.append(normalized)
            if not visible:
                continue

            x_values = [point[0] for point in visible]
            y_values = [point[1] for point in visible]
            left = min(x_values)
            right = max(x_values)
            top = min(y_values)
            bottom = max(y_values)
            padding_x = max(0.04, (right - left) * 0.16)
            padding_y = max(0.04, (bottom - top) * 0.10)
            box = (
                int(round((left - padding_x) * frame_width)),
                int(round((top - padding_y) * frame_height)),
                int(round((right - left + padding_x * 2) * frame_width)),
                int(round((bottom - top + padding_y * 2) * frame_height)),
            )
            bodies.append(PersonVisionEngine._clip_box(box, frame_size))

        return poses, [box for box in bodies if box[2] > 0 and box[3] > 0]

    @staticmethod
    def _extract_segmentation_mask(
        result: Any,
        *,
        frame_size: tuple[int, int],
    ) -> np.ndarray:
        frame_width, frame_height = frame_size
        combined = np.zeros((frame_height, frame_width), dtype=np.float32)

        for segmentation in getattr(result, "segmentation_masks", []) or []:
            mask = np.asarray(segmentation.numpy_view(), dtype=np.float32)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = cv2.resize(
                mask,
                (frame_width, frame_height),
                interpolation=cv2.INTER_LINEAR,
            )
            combined = np.maximum(combined, mask)

        return np.clip(combined, 0.0, 1.0)

    @staticmethod
    def _extract_person_segmentation(
        result: Any,
        *,
        frame_size: tuple[int, int],
    ) -> np.ndarray:
        frame_width, frame_height = frame_size
        masks = getattr(result, "confidence_masks", []) or []
        if not masks:
            return np.zeros((frame_height, frame_width), dtype=np.float32)

        mask = np.asarray(masks[0].numpy_view(), dtype=np.float32)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = cv2.resize(
            mask,
            (frame_width, frame_height),
            interpolation=cv2.INTER_LINEAR,
        )

        # Very low maxima are background uncertainty, not a detected person.
        if float(np.max(mask)) < 0.35:
            return np.zeros((frame_height, frame_width), dtype=np.float32)

        mask = np.clip((mask - 0.20) / 0.65, 0.0, 1.0)
        return mask.astype(np.float32)

    @staticmethod
    def _boxes_from_mask(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
        if mask.size == 0 or not np.any(mask):
            return []

        binary = (mask >= 0.32).astype(np.uint8) * 255
        kernel = np.ones((7, 7), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        min_area = mask.shape[0] * mask.shape[1] * 0.012
        boxes = [
            cv2.boundingRect(contour)
            for contour in contours
            if cv2.contourArea(contour) >= min_area
        ]
        boxes.sort(key=lambda box: box[2] * box[3], reverse=True)
        return boxes[:3]

    @staticmethod
    def _add_detection_fallbacks(
        mask: np.ndarray,
        faces: list[tuple[int, int, int, int]],
        bodies: list[tuple[int, int, int, int]],
    ) -> None:
        if not np.any(mask):
            for x, y, width, height in bodies:
                center = (x + width // 2, y + height // 2)
                axes = (max(1, width // 2), max(1, height // 2))
                cv2.ellipse(mask, center, axes, 0, 0, 360, 0.82, -1)

        for x, y, width, height in faces:
            left = max(0, x)
            top = max(0, y)
            right = min(mask.shape[1], x + width)
            bottom = min(mask.shape[0], y + height)
            face_region = mask[top:bottom, left:right]
            if face_region.size:
                covered = float(np.mean(face_region >= 0.32))
                if covered >= 0.55:
                    continue

            padding_x = int(round(width * 0.08))
            padding_y = int(round(height * 0.12))
            center = (x + width // 2, y + height // 2)
            axes = (
                max(1, width // 2 + padding_x),
                max(1, height // 2 + padding_y),
            )
            cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)

    @staticmethod
    def _clip_box(
        box: tuple[int, int, int, int],
        frame_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        x, y, width, height = box
        frame_width, frame_height = frame_size
        left = int(np.clip(x, 0, frame_width))
        top = int(np.clip(y, 0, frame_height))
        right = int(np.clip(x + width, 0, frame_width))
        bottom = int(np.clip(y + height, 0, frame_height))
        return left, top, max(0, right - left), max(0, bottom - top)

    def close(self) -> None:
        for detector in (
            self.face_detector,
            self.pose_landmarker,
            self.person_segmenter,
        ):
            if detector is not None:
                detector.close()
        self.face_detector = None
        self.pose_landmarker = None
        self.person_segmenter = None
