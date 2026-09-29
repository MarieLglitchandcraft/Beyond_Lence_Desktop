"""Reactive piezo sound for the Raspberry Pi installation."""

from __future__ import annotations

import random
import threading

from config import (
    FREQUENCIES,
    MODE_BEYOND,
    MODE_ENTITIES,
    MODE_GLITCH,
    MODE_NORMAL,
    MODE_SOFT,
)
from platform_utils import is_raspberry_pi


PIEZO_PIN = 14  # BCM GPIO 14, physical pin 8


def calculate_anomaly_activity(
    frequency: str,
    active_modes: list[int],
    *,
    entity_count: int = 0,
    portal_count: int = 0,
) -> float:
    """Return a 0..1 value representing what the camera currently perceives."""
    try:
        frequency_level = FREQUENCIES.index(frequency)
    except ValueError:
        frequency_level = 0

    frequency_strength = frequency_level / max(1, len(FREQUENCIES) - 1)
    anomaly_modes = [mode for mode in active_modes if mode != MODE_NORMAL]
    if frequency_strength <= 0.0 or not anomaly_modes:
        return 0.0

    activity = 0.0
    if MODE_SOFT in anomaly_modes:
        activity = max(activity, 0.05 + frequency_strength * 0.20)
    if MODE_GLITCH in anomaly_modes:
        activity = max(activity, 0.08 + frequency_strength * 0.34)
    if MODE_ENTITIES in anomaly_modes:
        entity_strength = min(max(entity_count, 0) / 6.0, 1.0)
        activity = max(
            activity,
            0.02 + frequency_strength * 0.14 + entity_strength * 0.56,
        )
    if MODE_BEYOND in anomaly_modes:
        entity_strength = min(max(entity_count, 0) / 8.0, 1.0)
        portal_strength = min(max(portal_count, 0) / 2.0, 1.0)
        activity = max(
            activity,
            0.10
            + frequency_strength * 0.30
            + entity_strength * 0.50
            + portal_strength * 0.10,
        )

    # Several simultaneous anomaly modes make the detector a little less calm.
    activity += max(0, len(anomaly_modes) - 1) * 0.05
    return max(0.0, min(activity, 1.0))


class SilentGeigerBuzzer:
    """No-op backend for the Mac and for Pi systems without GPIO support."""

    def __init__(self, detector_callback=None) -> None:
        self.detector_callback = detector_callback
        if self.detector_callback is not None:
            self.detector_callback(0.0)

    def update(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        if self.detector_callback is not None:
            self.detector_callback(0.0)


class GeigerBuzzer:
    """Emit short, irregular piezo clicks from a background thread."""

    def __init__(
        self,
        *,
        device=None,
        detector_callback=None,
        rng: random.Random | None = None,
        start_worker: bool = True,
    ) -> None:
        if device is None:
            try:
                from gpiozero import PWMOutputDevice
            except (ImportError, ModuleNotFoundError) as error:
                raise RuntimeError(
                    "gpiozero fehlt; der Piezo kann nicht gestartet werden."
                ) from error
            device = PWMOutputDevice(
                PIEZO_PIN,
                initial_value=0.0,
                frequency=2200,
            )

        self.device = device
        self.detector_callback = detector_callback
        self.rng = rng or random.Random()
        self._activity = 0.0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._activity_changed = threading.Event()
        self._thread = None
        if start_worker:
            self._thread = threading.Thread(
                target=self._run,
                name="beyond-lence-geiger",
                daemon=True,
            )
            self._thread.start()

    @property
    def activity(self) -> float:
        with self._lock:
            return self._activity

    def update(
        self,
        frequency: str,
        active_modes: list[int],
        *,
        entity_count: int = 0,
        portal_count: int = 0,
    ) -> None:
        activity = calculate_anomaly_activity(
            frequency,
            active_modes,
            entity_count=entity_count,
            portal_count=portal_count,
        )
        with self._lock:
            previous_activity = self._activity
            self._activity = activity

        # Wake the worker only for meaningful changes. Updates arrive every
        # video frame and must not perpetually postpone the next click.
        if (
            (previous_activity <= 0.0) != (activity <= 0.0)
            or abs(previous_activity - activity) >= 0.04
        ):
            self._activity_changed.set()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            activity = self.activity
            if activity <= 0.0:
                self._activity_changed.wait(0.5)
                self._activity_changed.clear()
                continue

            # A Poisson-like interval gives a much more believable detector
            # rhythm than a fixed beep. At maximum activity it reaches roughly
            # twelve clicks per second.
            clicks_per_second = 0.35 + 11.65 * (activity ** 1.45)
            delay = max(0.025, min(2.8, self.rng.expovariate(clicks_per_second)))
            if self._activity_changed.wait(delay):
                self._activity_changed.clear()
                continue
            if self._stop_event.is_set() or self.activity <= 0.0:
                continue

            self._emit_click(self.activity)

    def _set_detector_level(self, level: float) -> None:
        if self.detector_callback is None:
            return
        try:
            self.detector_callback(level)
        except Exception as error:
            print(f"Detektor-LEDs nicht verfuegbar: {error}")
            self.detector_callback = None

    def _emit_single_click(
        self,
        activity: float,
        *,
        frequency_range: tuple[int, int],
        volume: float,
        duration_range: tuple[float, float],
    ) -> None:
        self._set_detector_level(activity)
        duration = self.rng.uniform(*duration_range)
        try:
            self.device.frequency = self.rng.randint(*frequency_range)
            self.device.value = volume
            self._stop_event.wait(duration)
        finally:
            self.device.off()
            # The click is very short. Keep the light visible for at least
            # 35 ms while retaining the exact same rising edge as the sound.
            self._stop_event.wait(max(0.0, 0.035 - duration))
            self._set_detector_level(0.0)

    def _emit_click(self, activity: float) -> None:
        self._emit_single_click(
            activity,
            frequency_range=(1800, 2700),
            volume=0.42 + min(activity, 1.0) * 0.18,
            duration_range=(0.006, 0.014),
        )

        # Dense fields sometimes produce a short double hit.
        if activity > 0.68 and self.rng.random() < activity * 0.30:
            if self._stop_event.wait(self.rng.uniform(0.018, 0.038)):
                return
            self._emit_single_click(
                activity,
                frequency_range=(1950, 2900),
                volume=0.48,
                duration_range=(0.005, 0.010),
            )

    def close(self) -> None:
        self._stop_event.set()
        self._activity_changed.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.device.off()
        self._set_detector_level(0.0)
        close = getattr(self.device, "close", None)
        if close is not None:
            close()


def create_geiger_buzzer(force_gpio=None, detector_callback=None):
    use_gpio = is_raspberry_pi() if force_gpio is None else bool(force_gpio)
    if not use_gpio:
        return SilentGeigerBuzzer(detector_callback)

    try:
        print("Ton: Piezo-Geigerzaehler auf GPIO 14 (Pin 8)")
        return GeigerBuzzer(detector_callback=detector_callback)
    except RuntimeError as error:
        print(f"Piezo nicht verfuegbar: {error}")
        return SilentGeigerBuzzer(detector_callback)
