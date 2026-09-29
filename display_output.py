"""Display detection, OpenCV window setup and final frame preparation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np


FALLBACK_DISPLAY_SIZE = (1024, 600)
DISPLAY_ENVIRONMENT_KEYS = (
    "XDG_SESSION_TYPE",
    "XDG_CURRENT_DESKTOP",
    "XDG_SESSION_DESKTOP",
    "DESKTOP_SESSION",
    "XDG_RUNTIME_DIR",
    "WAYLAND_DISPLAY",
    "DISPLAY",
    "SDL_VIDEODRIVER",
    "QT_QPA_PLATFORM",
    "GDK_BACKEND",
)


@dataclass(frozen=True)
class DisplayInfo:
    width: int
    height: int
    source: str
    name: str = "unknown"
    x: int = 0
    y: int = 0

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def is_portrait(self) -> bool:
        return self.height > self.width

    @property
    def orientation(self) -> str:
        if self.width == self.height:
            return "quadratisch"
        return "hochkant" if self.is_portrait else "querformat"


def _parse_size(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None

    match = re.search(
        r"(?P<width>\d{3,5})\s*[x,]\s*(?P<height>\d{3,5})",
        value,
    )

    if match is None:
        return None

    width = int(match.group("width"))
    height = int(match.group("height"))

    if width < 160 or height < 120:
        return None

    return width, height


def _parse_wlr_randr(output: str) -> DisplayInfo | None:
    current_name = "unknown"
    current_position = (0, 0)

    for line in output.splitlines():
        if line and not line[0].isspace():
            current_name = line.split(maxsplit=1)[0]
            current_position = (0, 0)
            continue

        position_match = re.search(
            r"Position:\s*(-?\d+)\s*,\s*(-?\d+)",
            line,
        )

        if position_match:
            current_position = (
                int(position_match.group(1)),
                int(position_match.group(2)),
            )

        mode_match = re.search(
            r"(\d{3,5})x(\d{3,5})\s+px.*\bcurrent\b",
            line,
            re.IGNORECASE,
        )

        if mode_match:
            return DisplayInfo(
                int(mode_match.group(1)),
                int(mode_match.group(2)),
                source="wlr-randr",
                name=current_name,
                x=current_position[0],
                y=current_position[1],
            )

    return None


def _parse_xrandr(output: str) -> DisplayInfo | None:
    candidates: list[tuple[bool, DisplayInfo]] = []

    pattern = re.compile(
        r"^(?P<name>\S+)\s+connected"
        r"(?P<primary>\s+primary)?"
        r".*?\s(?P<width>\d{3,5})x(?P<height>\d{3,5})"
        r"\+(?P<x>-?\d+)\+(?P<y>-?\d+)",
        re.MULTILINE,
    )

    for match in pattern.finditer(output):
        candidates.append(
            (
                bool(match.group("primary")),
                DisplayInfo(
                    int(match.group("width")),
                    int(match.group("height")),
                    source="xrandr",
                    name=match.group("name"),
                    x=int(match.group("x")),
                    y=int(match.group("y")),
                ),
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda candidate: candidate[0],
        reverse=True,
    )

    return candidates[0][1]


def _parse_xdpyinfo(output: str) -> DisplayInfo | None:
    match = re.search(
        r"dimensions:\s*(\d{3,5})x(\d{3,5})\s+pixels",
        output,
    )

    if match is None:
        return None

    return DisplayInfo(
        int(match.group(1)),
        int(match.group(2)),
        source="xdpyinfo",
        name="X11 screen",
    )


def _default_command_runner(
    command: Sequence[str],
    environment: Mapping[str, str],
) -> str:
    if shutil.which(command[0]) is None:
        return ""

    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
            env=dict(environment),
        )
    except (OSError, subprocess.SubprocessError):
        return ""

    if completed.returncode != 0:
        return ""

    return completed.stdout


def _detect_from_drm(sysfs_root: Path) -> DisplayInfo | None:
    drm_root = sysfs_root / "class" / "drm"

    try:
        connectors = sorted(drm_root.glob("card*-*"))
    except OSError:
        return None

    for connector in connectors:
        try:
            status = (
                connector / "status"
            ).read_text(errors="ignore").strip()
        except OSError:
            continue

        if status.lower() != "connected":
            continue

        try:
            modes = (
                connector / "modes"
            ).read_text(errors="ignore").splitlines()
        except OSError:
            modes = []

        for mode in modes:
            size = _parse_size(mode)

            if size is not None:
                return DisplayInfo(
                    *size,
                    source="DRM/EDID",
                    name=connector.name,
                )

    return None


def _detect_from_framebuffer(
    sysfs_root: Path,
) -> DisplayInfo | None:
    path = (
        sysfs_root
        / "class"
        / "graphics"
        / "fb0"
        / "virtual_size"
    )

    try:
        size = _parse_size(
            path.read_text(errors="ignore")
        )
    except OSError:
        return None

    if size is None:
        return None

    return DisplayInfo(
        *size,
        source="framebuffer",
        name="fb0",
    )


def _detect_desktop_display() -> DisplayInfo | None:
    """
    Detect the primary display on macOS and Windows
    without using Tkinter.
    """

    # macOS
    if sys.platform == "darwin":
        try:
            import AppKit

            screen = AppKit.NSScreen.mainScreen()

            if screen is not None:
                frame = screen.frame()

                width = int(frame.size.width)
                height = int(frame.size.height)

                if width >= 160 and height >= 120:
                    return DisplayInfo(
                        width,
                        height,
                        source="macOS",
                        name="primary",
                    )

        except Exception:
            pass

    # Windows
    if sys.platform == "win32":
        try:
            import ctypes

            user32 = ctypes.windll.user32

            # Make the process DPI-aware so Windows reports
            # the actual usable screen dimensions correctly.
            try:
                user32.SetProcessDPIAware()
            except Exception:
                pass

            width = int(
                user32.GetSystemMetrics(0)
            )
            height = int(
                user32.GetSystemMetrics(1)
            )

            if width >= 160 and height >= 120:
                return DisplayInfo(
                    width,
                    height,
                    source="Windows",
                    name="primary",
                )

        except Exception:
            pass

    return None


def _detect_with_tkinter() -> DisplayInfo | None:
    """
    Legacy/fallback display detection.

    On macOS the desktop version should normally use
    _detect_desktop_display instead because destroying
    a temporary Tk instance can cause Tcl crashes.
    """

    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()

        width = int(
            root.winfo_screenwidth()
        )
        height = int(
            root.winfo_screenheight()
        )

        root.destroy()

    except Exception:
        return None

    if width < 160 or height < 120:
        return None

    return DisplayInfo(
        width,
        height,
        source="Tk",
        name="primary",
    )


def detect_display_info(
    *,
    fallback: tuple[int, int] = FALLBACK_DISPLAY_SIZE,
    environ: Mapping[str, str] | None = None,
    command_runner: Callable[
        [Sequence[str], Mapping[str, str]],
        str,
    ]
    | None = None,
    sysfs_root: str | Path = "/sys",
    use_tkinter: bool = True,
) -> DisplayInfo:
    """
    Detect the active screen without assuming
    the old 1024x600 Raspberry Pi panel.
    """

    environment = dict(
        os.environ
        if environ is None
        else environ
    )

    # Manual override, useful for testing.
    override = _parse_size(
        environment.get(
            "BEYOND_LENCE_DISPLAY_SIZE"
        )
    )

    if override is not None:
        return DisplayInfo(
            *override,
            source="environment override",
            name="override",
        )

    runner = (
        command_runner
        or _default_command_runner
    )

    # Native desktop detection first.
    # This avoids the Tcl/Tk crash observed on macOS.
    desktop_display = _detect_desktop_display()

    if desktop_display is not None:
        return desktop_display

    # Linux / Wayland
    if environment.get("WAYLAND_DISPLAY"):
        display = _parse_wlr_randr(
            runner(
                ("wlr-randr",),
                environment,
            )
        )

        if display is not None:
            return display

    # Linux / X11
    if environment.get("DISPLAY"):
        display = _parse_xrandr(
            runner(
                ("xrandr", "--current"),
                environment,
            )
        )

        if display is not None:
            return display

        display = _parse_xdpyinfo(
            runner(
                ("xdpyinfo",),
                environment,
            )
        )

        if display is not None:
            return display

    # Tk is kept only as an optional fallback.
    if use_tkinter:
        display = _detect_with_tkinter()

        if display is not None:
            return display

    # Raspberry Pi / Linux hardware detection
    display = _detect_from_drm(
        Path(sysfs_root)
    )

    if display is not None:
        return display

    display = _detect_from_framebuffer(
        Path(sysfs_root)
    )

    if display is not None:
        return display

    width, height = fallback

    return DisplayInfo(
        width,
        height,
        source="fallback",
        name="fallback",
    )


def detect_display_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(
        os.environ
        if environ is None
        else environ
    )

    session = (
        environment
        .get("XDG_SESSION_TYPE", "")
        .strip()
        .lower()
    )

    if not session:
        if environment.get("WAYLAND_DISPLAY"):
            session = "wayland"

        elif environment.get("DISPLAY"):
            session = "x11"

        else:
            session = "unknown"

    current_ui = getattr(
        cv2,
        "currentUIFramework",
        None,
    )

    try:
        highgui = (
            current_ui()
            if current_ui is not None
            else "unknown"
        )
    except cv2.error:
        highgui = "unknown"

    result = {
        "session": session,
        "opencv_highgui": (
            highgui or "unknown"
        ),
    }

    for key in DISPLAY_ENVIRONMENT_KEYS:
        result[key] = environment.get(
            key,
            "<nicht gesetzt>",
        )

    return result


def calculate_windowed_size(
    display_size: tuple[int, int],
    *,
    content_size: tuple[int, int] = FALLBACK_DISPLAY_SIZE,
    fraction: float = 0.82,
) -> tuple[int, int]:
    screen_width, screen_height = display_size
    content_width, content_height = content_size

    maximum_width = max(
        320,
        int(round(screen_width * fraction)),
    )

    maximum_height = max(
        240,
        int(round(screen_height * fraction)),
    )

    scale = min(
        maximum_width / max(content_width, 1),
        maximum_height / max(content_height, 1),
    )

    return (
        max(
            320,
            int(round(content_width * scale)),
        ),
        max(
            240,
            int(round(content_height * scale)),
        ),
    )


def fit_frame_to_canvas(
    frame: np.ndarray,
    target_size: tuple[int, int],
) -> np.ndarray:
    """
    Scale a landscape camera image without
    stretching or cutting off its HUD.
    """

    target_width, target_height = target_size

    if (
        frame is None
        or not isinstance(frame, np.ndarray)
        or frame.size == 0
        or frame.ndim != 3
        or frame.shape[2] < 3
    ):
        raise ValueError(
            "Das Ausgabeframe ist leer oder ungueltig."
        )

    if target_width < 1 or target_height < 1:
        raise ValueError(
            "Die Zielaufloesung ist ungueltig."
        )

    source_height, source_width = frame.shape[:2]

    scale = min(
        target_width / source_width,
        target_height / source_height,
    )

    resized_width = max(
        1,
        int(round(source_width * scale)),
    )

    resized_height = max(
        1,
        int(round(source_height * scale)),
    )

    interpolation = (
        cv2.INTER_AREA
        if scale < 1.0
        else cv2.INTER_LINEAR
    )

    resized = cv2.resize(
        frame[:, :, :3],
        (resized_width, resized_height),
        interpolation=interpolation,
    )

    canvas = np.zeros(
        (
            target_height,
            target_width,
            3,
        ),
        dtype=np.uint8,
    )

    left = (
        target_width - resized_width
    ) // 2

    top = (
        target_height - resized_height
    ) // 2

    canvas[
        top : top + resized_height,
        left : left + resized_width,
    ] = resized

    return canvas


def prepare_display_frame(
    frame: np.ndarray,
    *,
    emergency_off: bool = False,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Scale the final app frame;
    never rotate the physical display output.
    """

    if emergency_off:
        if target_size is None:
            return np.zeros_like(frame)

        target_width, target_height = target_size

        return np.zeros(
            (
                target_height,
                target_width,
                3,
            ),
            dtype=np.uint8,
        )

    result = frame

    if (
        target_size is not None
        and result.shape[1::-1] != target_size
    ):
        result = fit_frame_to_canvas(
            result,
            target_size,
        )

    return result


def create_error_frame(
    target_size: tuple[int, int],
    title: str,
    messages: Sequence[str],
) -> np.ndarray:
    width, height = target_size

    frame = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    frame[:] = (8, 18, 8)

    scale = max(
        0.55,
        min(
            width / 1024.0,
            height / 600.0,
        ),
    )

    margin = max(
        24,
        int(round(38 * scale)),
    )

    title_scale = 1.0 * scale
    body_scale = 0.55 * scale

    line_height = max(
        28,
        int(round(38 * scale)),
    )

    cv2.putText(
        frame,
        title,
        (
            margin,
            margin + int(round(42 * scale)),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        title_scale,
        (90, 255, 120),
        max(
            1,
            int(round(2 * scale)),
        ),
        cv2.LINE_AA,
    )

    y = margin + int(round(105 * scale))
    maximum_width = width - margin * 2

    for message in messages:
        words = str(message).split()

        lines: list[str] = []
        current = ""

        for word in words:
            candidate = (
                f"{current} {word}".strip()
            )

            text_width = cv2.getTextSize(
                candidate,
                cv2.FONT_HERSHEY_SIMPLEX,
                body_scale,
                1,
            )[0][0]

            if (
                current
                and text_width > maximum_width
            ):
                lines.append(current)
                current = word

            else:
                current = candidate

        if current:
            lines.append(current)

        if not lines:
            lines = [""]

        for line in lines:
            cv2.putText(
                frame,
                line,
                (margin, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                body_scale,
                (170, 235, 170),
                max(
                    1,
                    int(round(scale)),
                ),
                cv2.LINE_AA,
            )

            y += line_height

            if y > height - margin:
                return frame

        y += max(
            6,
            int(round(8 * scale)),
        )

    return frame


class OpenCVDisplayWindow:
    """
    A HighGUI window whose canvas always
    matches the detected display.
    """

    def __init__(
        self,
        name: str,
        display: DisplayInfo,
        *,
        fullscreen: bool = True,
        content_size: tuple[
            int,
            int,
        ] = FALLBACK_DISPLAY_SIZE,
    ) -> None:
        self.name = name
        self.display = display
        self.fullscreen = bool(fullscreen)

        self.render_size = (
            display.size
            if self.fullscreen
            else calculate_windowed_size(
                display.size,
                content_size=content_size,
            )
        )

        self._opened = False
        self._reported_geometry = False

    @property
    def backend_name(self) -> str:
        environment = (
            detect_display_environment()
        )

        return (
            "OpenCV HighGUI "
            f"({environment['opencv_highgui']})"
        )

    @property
    def mode_name(self) -> str:
        return (
            "fullscreen"
            if self.fullscreen
            else "windowed"
        )

    def open(self) -> None:
        flags = (
            cv2.WINDOW_NORMAL
            | getattr(
                cv2,
                "WINDOW_GUI_NORMAL",
                0x10,
            )
        )

        cv2.namedWindow(
            self.name,
            flags,
        )

        cv2.resizeWindow(
            self.name,
            *self.render_size,
        )

        if self.fullscreen:
            cv2.setWindowProperty(
                self.name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )

        else:
            cv2.setWindowProperty(
                self.name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_NORMAL,
            )

            environment = (
                detect_display_environment()
            )

            if (
                environment["session"]
                != "wayland"
            ):
                window_width, window_height = (
                    self.render_size
                )

                left = (
                    self.display.x
                    + max(
                        0,
                        (
                            self.display.width
                            - window_width
                        )
                        // 2,
                    )
                )

                top = (
                    self.display.y
                    + max(
                        0,
                        (
                            self.display.height
                            - window_height
                        )
                        // 2,
                    )
                )

                cv2.moveWindow(
                    self.name,
                    left,
                    top,
                )

        self._opened = True

    def prepare(
        self,
        frame: np.ndarray,
        *,
        emergency_off: bool = False,
    ) -> np.ndarray:
        return prepare_display_frame(
            frame,
            emergency_off=emergency_off,
            target_size=self.render_size,
        )

    def show(
        self,
        frame: np.ndarray,
        *,
        emergency_off: bool = False,
    ) -> np.ndarray:
        prepared = self.prepare(
            frame,
            emergency_off=emergency_off,
        )

        cv2.imshow(
            self.name,
            prepared,
        )

        if not self._reported_geometry:
            cv2.waitKey(1)
            self.report_geometry()
            self._reported_geometry = True

        return prepared

    @staticmethod
    def poll_key(
        delay_ms: int = 1,
    ) -> int:
        return (
            cv2.waitKey(
                max(
                    1,
                    int(delay_ms),
                )
            )
            & 0xFF
        )

    def report_geometry(self) -> None:
        get_rect = getattr(
            cv2,
            "getWindowImageRect",
            None,
        )

        if get_rect is None:
            print(
                "OpenCV-Fenstergeometrie: "
                "nicht verfuegbar",
                flush=True,
            )
            return

        try:
            left, top, width, height = (
                get_rect(self.name)
            )

        except cv2.error as error:
            print(
                "OpenCV-Fenstergeometrie: "
                f"nicht lesbar ({error})",
                flush=True,
            )
            return

        print(
            "OpenCV-Fenstergeometrie: "
            f"{width}x{height} "
            f"bei {left},{top}",
            flush=True,
        )

    def show_error(
        self,
        title: str,
        messages: Sequence[str],
        *,
        duration: float = 15.0,
    ) -> None:
        if not self._opened:
            try:
                self.open()

            except cv2.error as error:
                print(
                    "Fehlerfenster konnte "
                    "nicht geoeffnet werden: "
                    f"{error}",
                    flush=True,
                )
                return

        safe_size = calculate_windowed_size(
            self.display.size
        )

        try:
            cv2.setWindowProperty(
                self.name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_NORMAL,
            )

            cv2.resizeWindow(
                self.name,
                *safe_size,
            )

        except cv2.error as error:
            print(
                "Fehlerfenster konnte "
                "nicht umgestellt werden: "
                f"{error}",
                flush=True,
            )

        error_frame = create_error_frame(
            safe_size,
            title,
            messages,
        )

        cv2.imshow(
            self.name,
            error_frame,
        )

        deadline = (
            time.monotonic()
            + max(
                1.0,
                float(duration),
            )
        )

        while time.monotonic() < deadline:
            key = self.poll_key(100)

            if key in (
                27,
                ord("q"),
                ord("Q"),
                13,
            ):
                break

    def close(self) -> None:
        if not self._opened:
            return

        try:
            cv2.destroyWindow(
                self.name
            )

        except cv2.error:
            pass

        self._opened = False


class PygameDisplayWindow:
    """
    SDL display backend for Pi systems
    affected by Qt HighGUI corruption.
    """

    def __init__(
        self,
        name: str,
        display: DisplayInfo,
        *,
        fullscreen: bool = True,
        content_size: tuple[
            int,
            int,
        ] = FALLBACK_DISPLAY_SIZE,
        pygame_module=None,
    ) -> None:
        if pygame_module is None:
            import pygame as pygame_module

        self.pygame = pygame_module
        self.name = name
        self.display = display
        self.fullscreen = bool(fullscreen)

        self.render_size = (
            display.size
            if self.fullscreen
            else calculate_windowed_size(
                display.size,
                content_size=content_size,
            )
        )

        self._screen = None
        self._clock = None
        self._opened = False
        self._driver = "unbekannt"

    @property
    def mode_name(self) -> str:
        return (
            "fullscreen"
            if self.fullscreen
            else "windowed"
        )

    @property
    def backend_name(self) -> str:
        return (
            f"Pygame/SDL ({self._driver})"
        )

    def open(self) -> None:
        pygame = self.pygame

        pygame.display.init()
        pygame.display.set_caption(
            self.name
        )

        flags = (
            pygame.FULLSCREEN
            if self.fullscreen
            else pygame.RESIZABLE
        )

        self._screen = pygame.display.set_mode(
            self.render_size,
            flags,
        )

        actual_size = tuple(
            int(value)
            for value
            in self._screen.get_size()
        )

        if (
            len(actual_size) == 2
            and min(actual_size) > 0
        ):
            self.render_size = actual_size

        try:
            self._driver = (
                pygame.display.get_driver()
            )

        except Exception:
            self._driver = "unbekannt"

        self._clock = pygame.time.Clock()
        self._opened = True

        print(
            "SDL-Fenstergeometrie: "
            f"{self.render_size[0]}x"
            f"{self.render_size[1]}",
            flush=True,
        )

    def prepare(
        self,
        frame: np.ndarray,
        *,
        emergency_off: bool = False,
    ) -> np.ndarray:
        return prepare_display_frame(
            frame,
            emergency_off=emergency_off,
            target_size=self.render_size,
        )

    def _draw(
        self,
        frame: np.ndarray,
    ) -> None:
        if self._screen is None:
            raise RuntimeError(
                "Das SDL-Anzeigefenster "
                "ist nicht geoeffnet."
            )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        surface = (
            self.pygame.image.frombuffer(
                rgb.data,
                (
                    rgb.shape[1],
                    rgb.shape[0],
                ),
                "RGB",
            )
        )

        self._screen.blit(
            surface,
            (0, 0),
        )

        self.pygame.display.flip()

    def show(
        self,
        frame: np.ndarray,
        *,
        emergency_off: bool = False,
    ) -> np.ndarray:
        prepared = self.prepare(
            frame,
            emergency_off=emergency_off,
        )

        self._draw(prepared)

        return prepared

    def poll_key(
        self,
        delay_ms: int = 1,
    ) -> int:
        pygame = self.pygame
        key = 255

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                key = ord("q")
                continue

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    key = 27

                elif getattr(
                    event,
                    "unicode",
                    "",
                ):
                    key = ord(
                        event.unicode[0]
                    )

            resize_events = {
                getattr(
                    pygame,
                    "VIDEORESIZE",
                    -1,
                ),
                getattr(
                    pygame,
                    "WINDOWRESIZED",
                    -2,
                ),
            }

            if (
                not self.fullscreen
                and event.type
                in resize_events
                and getattr(
                    event,
                    "w",
                    0,
                )
                > 0
                and getattr(
                    event,
                    "h",
                    0,
                )
                > 0
            ):
                self.render_size = (
                    int(event.w),
                    int(event.h),
                )

                self._screen = (
                    pygame.display.set_mode(
                        self.render_size,
                        pygame.RESIZABLE,
                    )
                )

        if self._clock is not None:
            delay_ms = max(
                1,
                int(delay_ms),
            )

            self._clock.tick(
                max(
                    1,
                    int(
                        round(
                            1000
                            / delay_ms
                        )
                    ),
                )
            )

        return key

    def show_error(
        self,
        title: str,
        messages: Sequence[str],
        *,
        duration: float = 15.0,
    ) -> None:
        if not self._opened:
            try:
                self.open()

            except Exception as error:
                print(
                    "SDL-Fehlerfenster "
                    "konnte nicht "
                    "geoeffnet werden: "
                    f"{error}",
                    flush=True,
                )
                return

        error_frame = create_error_frame(
            self.render_size,
            title,
            messages,
        )

        self._draw(error_frame)

        deadline = (
            time.monotonic()
            + max(
                1.0,
                float(duration),
            )
        )

        while time.monotonic() < deadline:
            key = self.poll_key(100)

            if key in (
                27,
                ord("q"),
                ord("Q"),
                13,
            ):
                break

    def close(self) -> None:
        if not self._opened:
            return

        try:
            self.pygame.display.quit()

        finally:
            self._screen = None
            self._clock = None
            self._opened = False


def create_display_window(
    name: str,
    display: DisplayInfo,
    *,
    fullscreen: bool = True,
    content_size: tuple[
        int,
        int,
    ] = FALLBACK_DISPLAY_SIZE,
    environ: Mapping[str, str] | None = None,
):
    """
    Use SDL on Linux so Qt HighGUI cannot
    add or corrupt control panels.
    """

    environment = dict(
        os.environ
        if environ is None
        else environ
    )

    preference = environment.get(
        "BEYOND_LENCE_DISPLAY_BACKEND",
        "auto",
    ).strip().lower()

    if preference not in {
        "auto",
        "pygame",
        "opencv",
    }:
        print(
            "WARNUNG: Unbekanntes "
            "Display-Backend "
            f"'{preference}', "
            "verwende automatische Auswahl.",
            flush=True,
        )

        preference = "auto"

    wants_pygame = (
        preference == "pygame"
        or (
            preference == "auto"
            and sys.platform.startswith(
                "linux"
            )
        )
    )

    if wants_pygame:
        try:
            return PygameDisplayWindow(
                name,
                display,
                fullscreen=fullscreen,
                content_size=content_size,
            )

        except (
            ImportError,
            ModuleNotFoundError,
        ) as error:
            print(
                "WARNUNG: Pygame/SDL "
                "ist nicht installiert. "
                "OpenCV-Fallback wird "
                f"verwendet: {error}",
                flush=True,
            )

    return OpenCVDisplayWindow(
        name,
        display,
        fullscreen=fullscreen,
        content_size=content_size,
    )
