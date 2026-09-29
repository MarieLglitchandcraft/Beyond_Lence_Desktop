from __future__ import annotations

import math

import cv2
import numpy as np


class SpatialTracker:
    """Estimates how stationary scene points move between camera frames."""

    MIN_POINTS = 8
    MIN_CONFIDENCE = 0.20

    def __init__(self) -> None:
        self.previous_gray: np.ndarray | None = None
        self.previous_exclusion_mask: np.ndarray | None = None

    def update(
        self,
        gray: np.ndarray,
        person_mask: np.ndarray | None = None,
    ) -> dict:
        result = self._empty_result()

        if self.previous_gray is not None:
            result = self._estimate_transform(self.previous_gray, gray)

        self.previous_gray = gray.copy()
        self.previous_exclusion_mask = self._prepare_exclusion_mask(
            person_mask,
            gray.shape,
        )
        return result

    def _estimate_transform(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
    ) -> dict:
        feature_mask = np.full(previous_gray.shape, 255, dtype=np.uint8)
        if self.previous_exclusion_mask is not None:
            exclusion = self.previous_exclusion_mask
            if exclusion.shape != previous_gray.shape:
                exclusion = cv2.resize(
                    exclusion,
                    (previous_gray.shape[1], previous_gray.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            feature_mask[exclusion > 0] = 0

        previous_points = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=180,
            qualityLevel=0.012,
            minDistance=11,
            mask=feature_mask,
            blockSize=7,
        )
        if previous_points is None or len(previous_points) < self.MIN_POINTS:
            return self._empty_result()

        current_points, status, error = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            previous_points,
            None,
            winSize=(25, 25),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                24,
                0.01,
            ),
        )
        if current_points is None or status is None:
            return self._empty_result()

        valid = status.reshape(-1) == 1
        if error is not None:
            valid &= error.reshape(-1) < 32.0
        old = previous_points.reshape(-1, 2)[valid]
        new = current_points.reshape(-1, 2)[valid]
        if len(old) < self.MIN_POINTS:
            return self._empty_result()

        transform, inliers = cv2.estimateAffinePartial2D(
            old,
            new,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.8,
            maxIters=1200,
            confidence=0.99,
            refineIters=10,
        )
        if transform is None or inliers is None:
            return self._empty_result()

        transform = transform.astype(np.float32)
        scale = float(math.hypot(transform[0, 0], transform[1, 0]))
        rotation = math.degrees(math.atan2(transform[1, 0], transform[0, 0]))
        if not 0.92 <= scale <= 1.08 or abs(rotation) > 10.0:
            return self._empty_result()

        # Handheld zoom noise accumulates quickly. Keep rotation/translation,
        # but normalize away tiny frame-to-frame scale fluctuations.
        transform[:, :2] /= max(scale, 1e-6)

        inlier_flags = inliers.reshape(-1).astype(bool)
        inlier_count = int(np.count_nonzero(inlier_flags))
        inlier_ratio = inlier_count / max(len(old), 1)
        if inlier_count:
            projected = cv2.transform(old[None, :, :], transform)[0]
            residual = np.linalg.norm(projected - new, axis=1)
            median_error = float(np.median(residual[inlier_flags]))
        else:
            median_error = 99.0

        point_factor = min(inlier_count / 36.0, 1.0)
        error_factor = max(0.0, 1.0 - median_error / 14.0)
        confidence = float(np.clip(
            inlier_ratio * point_factor * error_factor,
            0.0,
            1.0,
        ))
        if confidence < self.MIN_CONFIDENCE:
            return self._empty_result()

        height, width = current_gray.shape[:2]
        center = np.array([width * 0.5, height * 0.5, 1.0], dtype=np.float32)
        moved_center = transform @ center
        camera_motion = (
            float(moved_center[0] - center[0]),
            float(moved_center[1] - center[1]),
        )

        return {
            "camera_transform": transform,
            "tracking_confidence": confidence,
            "tracking_points": inlier_count,
            "camera_motion": camera_motion,
            "camera_rotation": float(rotation),
        }

    @staticmethod
    def _prepare_exclusion_mask(
        person_mask: np.ndarray | None,
        target_shape: tuple[int, int],
    ) -> np.ndarray | None:
        if person_mask is None or person_mask.size == 0:
            return None

        if person_mask.shape[:2] != target_shape:
            person_mask = cv2.resize(
                person_mask,
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        exclusion = (person_mask > 0.18).astype(np.uint8) * 255
        kernel = np.ones((17, 17), dtype=np.uint8)
        return cv2.dilate(exclusion, kernel, iterations=1)

    @staticmethod
    def _empty_result() -> dict:
        return {
            "camera_transform": np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
            "tracking_confidence": 0.0,
            "tracking_points": 0,
            "camera_motion": (0.0, 0.0),
            "camera_rotation": 0.0,
        }

