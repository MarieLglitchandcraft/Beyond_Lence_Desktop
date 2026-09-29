import time
from collections.abc import Callable
from typing import Any


class SceneManager:
    STABILITY_WINDOW_SECONDS = 6.0
    QUIET_THRESHOLD = 0.7

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self.last_motion_time = self.clock()

    def evaluate(self, analysis: dict[str, Any]) -> dict[str, Any]:
        dark_regions = analysis.get("dark_regions", [])
        bright_regions = analysis.get("bright_regions", [])
        motion_regions = analysis.get("motion_regions", [])
        overlay_regions = analysis.get("overlay_regions", [])
        dark_portals = analysis.get("dark_portals", [])
        faces = analysis.get("faces", [])
        bodies = analysis.get("bodies", [])
        frame_size = analysis.get("frame_size")

        now = self.clock()

        if motion_regions:
            self.last_motion_time = now

        seconds_since_motion = max(0.0, now - self.last_motion_time)
        scene_stability = min(
            seconds_since_motion / self.STABILITY_WINDOW_SECONDS,
            1.0,
        )

        return {
            "dark_count": len(dark_regions),
            "bright_count": len(bright_regions),
            "motion_count": len(motion_regions),
            "overlay_count": len(overlay_regions),
            "scene_stability": scene_stability,
            "is_quiet": not motion_regions and scene_stability >= self.QUIET_THRESHOLD,
            "has_good_overlay_spot": bool(overlay_regions),
            "best_overlay_regions": overlay_regions[:14],
            "dark_portals": dark_portals[:4],
            "frame_size": frame_size,
            "person_present": bool(faces or bodies),
            "face_count": len(faces),
            "body_count": len(bodies),
            "camera_transform": analysis.get("camera_transform"),
            "tracking_confidence": analysis.get("tracking_confidence", 0.0),
            "tracking_points": analysis.get("tracking_points", 0),
            "camera_motion": analysis.get("camera_motion", (0.0, 0.0)),
        }
