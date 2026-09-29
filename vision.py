import cv2
import numpy as np

from person_vision import PersonVisionEngine
from spatial_tracker import SpatialTracker


class VisionEngine:
    def __init__(self, person_vision=None, spatial_tracker=None):
        self.previous_gray = None
        self.person_vision = (
            person_vision
            if person_vision is not None
            else PersonVisionEngine()
        )
        self.spatial_tracker = spatial_tracker or SpatialTracker()
        self.dark_oval_tracks = []

    @property
    def detector_status(self):
        return getattr(self.person_vision, "status", "Personenerkennung unbekannt")

    def analyze(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        person_analysis = self.person_vision.analyze(frame)
        faces = person_analysis.get("faces", [])
        bodies = person_analysis.get("bodies", [])
        person_mask = person_analysis.get("person_mask")
        spatial_analysis = self.spatial_tracker.update(gray, person_mask)

        dark_regions = self.detect_dark_regions(gray)
        dark_oval_candidates = self.detect_dark_ovals(
            gray,
            person_mask=person_mask,
            protected_regions=faces + bodies,
        )
        dark_portals = self.stabilize_dark_ovals(dark_oval_candidates)
        bright_regions = self.detect_bright_regions(gray)
        motion_regions = self.detect_motion(gray)
        overlay_regions = self.find_overlay_regions(
            frame,
            motion_regions,
            person_mask=person_mask,
            protected_regions=faces + bodies,
        )

        self.previous_gray = gray

        return {
            "dark_regions": dark_regions,
            "dark_oval_candidates": dark_oval_candidates,
            "dark_portals": dark_portals,
            "bright_regions": bright_regions,
            "motion_regions": motion_regions,
            "overlay_regions": overlay_regions,
            "frame_size": (frame.shape[1], frame.shape[0]),
            **spatial_analysis,
            **person_analysis,
        }

    def detect_dark_regions(self, gray):
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        threshold_value = int(np.clip(np.percentile(blurred, 35) + 12, 55, 110))
        mask = cv2.inRange(blurred, 0, threshold_value)
        kernel = np.ones((9, 9), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area > gray.size * 0.001:
                x, y, w, h = cv2.boundingRect(contour)

                if w > gray.shape[1] * 0.035 and h > gray.shape[0] * 0.035:
                    regions.append((x, y, w, h))

        return regions

    def detect_dark_ovals(
        self,
        gray,
        person_mask=None,
        protected_regions=None,
    ):
        """Finds dark, compact round openings outside detected people."""
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        threshold_value = int(np.clip(
            np.percentile(blurred, 24) + 10,
            42,
            96,
        ))
        threshold = cv2.inRange(blurred, 0, threshold_value)
        kernel_size = max(5, int(round(min(gray.shape[:2]) * 0.012)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        frame_area = float(gray.size)
        frame_height, frame_width = gray.shape[:2]
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not frame_area * 0.0018 <= area <= frame_area * 0.16:
                continue
            if len(contour) < 5:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            if (
                width < frame_width * 0.035
                or height < frame_height * 0.045
                or x <= 1
                or y <= 1
                or x + width >= frame_width - 1
                or y + height >= frame_height - 1
            ):
                continue
            aspect = width / max(1.0, float(height))
            if not 0.46 <= aspect <= 2.15:
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0.0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
            solidity = area / max(1.0, hull_area)
            rectangle_fill = area / max(1.0, float(width * height))
            if (
                circularity < 0.46
                or solidity < 0.78
                or not 0.45 <= rectangle_fill <= 0.91
            ):
                continue

            region = (x, y, width, height)
            if self.mask_coverage(region, person_mask) > 0.08:
                continue
            if self.overlaps_any(region, protected_regions or []):
                continue

            (center_x, center_y), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
            axis_width = max(1.0, float(axis_a))
            axis_height = max(1.0, float(axis_b))
            axis_ratio = min(axis_width, axis_height) / max(axis_width, axis_height)
            if axis_ratio < 0.46:
                continue

            contour_mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            ring_mask = cv2.dilate(contour_mask, kernel, iterations=2)
            ring_mask = cv2.subtract(ring_mask, contour_mask)
            inside_mean = float(cv2.mean(blurred, mask=contour_mask)[0])
            ring_mean = float(cv2.mean(blurred, mask=ring_mask)[0])
            contrast = ring_mean - inside_mean
            if contrast < 10.0 and inside_mean > 54.0:
                continue

            darkness = 1.0 - min(inside_mean / 110.0, 1.0)
            score = (
                circularity * 1.15
                + solidity * 0.55
                + axis_ratio * 0.50
                + darkness * 0.70
                + min(max(contrast, 0.0) / 80.0, 1.0) * 0.55
            )
            candidates.append({
                "box": region,
                "center": (float(center_x), float(center_y)),
                "axes": (axis_width, axis_height),
                "angle": float(angle),
                "score": round(float(score), 3),
                "darkness": round(float(darkness), 3),
                "circularity": round(float(circularity), 3),
            })

        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:8]

    def stabilize_dark_ovals(self, candidates):
        """Requires the same opening across several frames before using it."""
        unmatched_tracks = set(range(len(self.dark_oval_tracks)))
        updated_tracks = []
        for candidate in candidates:
            center_x, center_y = candidate["center"]
            best_index = None
            best_distance = float("inf")
            for index in unmatched_tracks:
                track = self.dark_oval_tracks[index]
                old_x, old_y = track["center"]
                distance = float(np.hypot(center_x - old_x, center_y - old_y))
                match_radius = max(
                    42.0,
                    max(candidate["axes"] + track["axes"]) * 0.48,
                )
                if distance < match_radius and distance < best_distance:
                    best_index = index
                    best_distance = distance

            if best_index is None:
                track = dict(candidate)
                track["hits"] = 1
                track["misses"] = 0
            else:
                unmatched_tracks.remove(best_index)
                previous = self.dark_oval_tracks[best_index]
                track = dict(candidate)
                track["center"] = (
                    previous["center"][0] * 0.38 + center_x * 0.62,
                    previous["center"][1] * 0.38 + center_y * 0.62,
                )
                track["axes"] = (
                    previous["axes"][0] * 0.42 + candidate["axes"][0] * 0.58,
                    previous["axes"][1] * 0.42 + candidate["axes"][1] * 0.58,
                )
                track["hits"] = min(30, int(previous["hits"]) + 1)
                track["misses"] = 0
            updated_tracks.append(track)

        for index in unmatched_tracks:
            track = dict(self.dark_oval_tracks[index])
            track["misses"] = int(track.get("misses", 0)) + 1
            if track["misses"] <= 4:
                updated_tracks.append(track)

        self.dark_oval_tracks = updated_tracks
        stable = []
        for track in self.dark_oval_tracks:
            if int(track.get("hits", 0)) < 4 or int(track.get("misses", 0)) > 2:
                continue
            item = dict(track)
            item["stability"] = round(
                min(1.0, float(track["hits"]) / 8.0),
                2,
            )
            stable.append(item)
        stable.sort(
            key=lambda item: item["score"] * item["stability"],
            reverse=True,
        )
        return stable[:4]

    def detect_bright_regions(self, gray):
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        threshold_value = int(np.clip(np.percentile(blurred, 72), 145, 215))
        mask = cv2.inRange(blurred, threshold_value, 255)
        kernel = np.ones((9, 9), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        regions = []
        for contour in contours:
            if cv2.contourArea(contour) > gray.size * 0.001:
                x, y, w, h = cv2.boundingRect(contour)
                if w > gray.shape[1] * 0.035 and h > gray.shape[0] * 0.035:
                    regions.append((x, y, w, h))

        return regions

    def detect_motion(self, gray):
        if self.previous_gray is None:
            return []

        diff = cv2.absdiff(self.previous_gray, gray)
        _, threshold = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area > 800:
                x, y, w, h = cv2.boundingRect(contour)
                regions.append((x, y, w, h))

        return regions

    def find_overlay_regions(
        self,
        frame,
        motion_regions,
        person_mask=None,
        protected_regions=None,
    ):
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        overlay_regions = []

        columns = 6
        rows = 4
        cell_width = max(1, width // columns)
        cell_height = max(1, height // rows)

        for row in range(rows):
            for column in range(columns):
                x = column * cell_width
                y = row * cell_height
                cell_right = width if column == columns - 1 else (column + 1) * cell_width
                cell_bottom = height if row == rows - 1 else (row + 1) * cell_height
                margin_x = max(4, int((cell_right - x) * 0.06))
                margin_y = max(4, int((cell_bottom - y) * 0.06))
                region = (
                    x + margin_x,
                    y + margin_y,
                    max(1, cell_right - x - margin_x * 2),
                    max(1, cell_bottom - y - margin_y * 2),
                )
                rx, ry, rw, rh = region
                cell = gray[ry : ry + rh, rx : rx + rw]
                mean_brightness = float(np.mean(cell))
                local_contrast = float(np.std(cell))

                edge_distance = min(
                    column,
                    columns - 1 - column,
                    row,
                    rows - 1 - row,
                )
                edge_bonus = 1.6 if edge_distance == 0 else 0.35

                if mean_brightness < 105:
                    source = "dark"
                    tonal_score = (105 - mean_brightness) / 70.0
                elif mean_brightness > 165:
                    source = "bright"
                    tonal_score = (mean_brightness - 165) / 70.0
                else:
                    source = "neutral"
                    tonal_score = min(local_contrast / 90.0, 0.7)

                score = 1.0 + edge_bonus + tonal_score
                if self.overlaps_any(region, motion_regions):
                    score -= 0.8

                person_coverage = self.mask_coverage(region, person_mask)
                if person_coverage > 0:
                    score -= person_coverage * 4.5
                if self.overlaps_any(region, protected_regions or []):
                    score -= 1.4

                overlay_regions.append({
                    "box": region,
                    "score": round(max(score, 0.1), 2),
                    "center": (rx + rw // 2, ry + rh // 2),
                    "source": source,
                    "brightness": round(mean_brightness, 1),
                    "person_coverage": round(person_coverage, 3),
                })

        overlay_regions.sort(key=lambda item: item["score"], reverse=True)

        return overlay_regions

    @staticmethod
    def mask_coverage(region, mask):
        if mask is None or mask.size == 0:
            return 0.0

        x, y, width, height = region
        sample = mask[y : y + height, x : x + width]
        return float(np.mean(sample)) if sample.size else 0.0

    def overlaps_any(self, region, other_regions):
        for other in other_regions:
            if self.overlaps(region, other):
                return True

        return False

    def overlaps(self, a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b

        return not (
            ax + aw < bx
            or bx + bw < ax
            or ay + ah < by
            or by + bh < ay
        )

    def draw_debug(self, frame, analysis):
        result = frame.copy()

        person_mask = analysis.get("person_mask")
        if person_mask is not None and person_mask.size:
            tint = np.zeros_like(result)
            tint[:, :, 0] = 255
            alpha = (np.clip(person_mask, 0.0, 1.0) * 0.28)[:, :, None]
            result = (
                result.astype(np.float32) * (1.0 - alpha)
                + tint.astype(np.float32) * alpha
            ).astype(np.uint8)

        for rank, item in enumerate(analysis["overlay_regions"][:5], start=1):
            score = item["score"]
            center_x, center_y = item["center"]

            cv2.circle(result, (center_x, center_y), 10, (40, 230, 120), 2)
            cv2.line(
                result,
                (center_x - 15, center_y),
                (center_x + 15, center_y),
                (40, 230, 120),
                1,
            )
            cv2.line(
                result,
                (center_x, center_y - 15),
                (center_x, center_y + 15),
                (40, 230, 120),
                1,
            )
            cv2.putText(
                result,
                f"{rank}: {score}",
                (center_x + 14, center_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (40, 230, 120),
                1,
            )

        for item in analysis.get("dark_portals", []):
            center = tuple(int(round(value)) for value in item["center"])
            axes = tuple(max(1, int(round(value * 0.5))) for value in item["axes"])
            cv2.ellipse(
                result,
                center,
                axes,
                float(item.get("angle", 0.0)),
                0,
                360,
                (255, 0, 255),
                3,
            )
            cv2.putText(
                result,
                f"PORTAL {item.get('stability', 0.0):.2f}",
                (center[0] + 10, max(20, center[1] - axes[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 0, 255),
                2,
            )
        for x, y, w, h in analysis.get("bodies", []):
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 120, 0), 3)
            cv2.putText(
                result,
                "BODY",
                (x, max(18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 120, 0),
                2,
            )

        for x, y, w, h in analysis.get("faces", []):
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(
                result,
                "FACE",
                (x, max(18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

        ai_state = "ON" if analysis.get("person_detection_available") else "OFF"
        mask_state = "ON" if person_mask is not None and person_mask.size else "NO PERSON"
        status = (
            f"DEBUG  AI {ai_state}  MASK {mask_state}  "
            f"FACE {len(analysis.get('faces', []))}  "
            f"BODY {len(analysis.get('bodies', []))}"
        )
        cv2.putText(
            result,
            status,
            (max(20, result.shape[1] - 610), result.shape[0] - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 255, 180),
            2,
        )

        confidence = float(analysis.get("tracking_confidence", 0.0))
        motion_x, motion_y = analysis.get("camera_motion", (0.0, 0.0))
        tracking_status = (
            f"ANCHOR {confidence * 100:02.0f}%  "
            f"MOVE {motion_x:+.1f},{motion_y:+.1f}  "
            f"POINTS {analysis.get('tracking_points', 0)}"
        )
        cv2.putText(
            result,
            tracking_status,
            (20, result.shape[0] - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (210, 170, 255),
            1,
        )

        return result

    @staticmethod
    def mirror_analysis_for_display(analysis, frame_width):
        """Mirror debug geometry so labels can be redrawn left-to-right."""
        width = max(1, int(frame_width))
        mirrored = dict(analysis)

        person_mask = analysis.get("person_mask")
        if person_mask is not None and person_mask.size:
            mirrored["person_mask"] = cv2.flip(person_mask, 1)

        def mirror_box(box):
            x, y, box_width, box_height = box
            return (
                width - int(x) - int(box_width),
                int(y),
                int(box_width),
                int(box_height),
            )

        mirrored_regions = []
        for item in analysis.get("overlay_regions", []):
            mirrored_item = dict(item)
            if "box" in item:
                mirrored_item["box"] = mirror_box(item["box"])
            if "center" in item:
                center_x, center_y = item["center"]
                mirrored_item["center"] = (
                    width - 1 - int(center_x),
                    int(center_y),
                )
            mirrored_regions.append(mirrored_item)
        mirrored["overlay_regions"] = mirrored_regions

        mirrored_portals = []
        for item in analysis.get("dark_portals", []):
            mirrored_item = dict(item)
            center_x, center_y = item["center"]
            mirrored_item["center"] = (
                width - 1 - float(center_x),
                float(center_y),
            )
            mirrored_item["angle"] = -float(item.get("angle", 0.0))
            mirrored_portals.append(mirrored_item)
        mirrored["dark_portals"] = mirrored_portals

        mirrored["bodies"] = [
            mirror_box(box) for box in analysis.get("bodies", [])
        ]
        mirrored["faces"] = [
            mirror_box(box) for box in analysis.get("faces", [])
        ]

        motion_x, motion_y = analysis.get("camera_motion", (0.0, 0.0))
        mirrored["camera_motion"] = (-float(motion_x), float(motion_y))
        return mirrored

    def close(self):
        close = getattr(self.person_vision, "close", None)
        if close is not None:
            close()
