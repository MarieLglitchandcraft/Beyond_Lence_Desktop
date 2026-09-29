"""Checks MediaPipe model startup in an isolated process.

Some macOS environments terminate the process at native level when no graphics
service is available. Keeping this probe separate prevents that from taking the
camera application down with it.
"""

from __future__ import annotations

import sys


def main() -> int:
    if len(sys.argv) != 4:
        return 2

    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    face_options = vision.FaceDetectorOptions(
        base_options=python.BaseOptions(
            model_asset_path=sys.argv[1],
            delegate=python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.IMAGE,
    )
    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=sys.argv[2],
            delegate=python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.IMAGE,
        output_segmentation_masks=True,
    )
    segmenter_options = vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(
            model_asset_path=sys.argv[3],
            delegate=python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.IMAGE,
        output_confidence_masks=True,
        output_category_mask=False,
    )

    face_detector = vision.FaceDetector.create_from_options(face_options)
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
    person_segmenter = vision.ImageSegmenter.create_from_options(
        segmenter_options,
    )
    face_detector.close()
    pose_landmarker.close()
    person_segmenter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
