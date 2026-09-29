import math
import threading
import time

from config import (
    MODE_NORMAL,
    MODE_SOFT,
    MODE_ENTITIES,
    MODE_GLITCH,
    MODE_BEYOND,
    FREQUENCIES,
)
from platform_utils import is_raspberry_pi


class DebouncedBoolean:
    """Accept a GPIO state only after it remains unchanged for a while."""

    def __init__(self, initial_state, *, settle_time=0.25, clock=time.monotonic):
        self.clock = clock
        self.settle_time = max(0.0, float(settle_time))
        self.stable_state = bool(initial_state)
        self.candidate_state = self.stable_state
        self.candidate_since = self.clock()

    def update(self, raw_state):
        raw_state = bool(raw_state)
        now = self.clock()
        raw_changed = raw_state != self.candidate_state
        stable_changed = False

        if raw_changed:
            self.candidate_state = raw_state
            self.candidate_since = now
        elif (
            raw_state != self.stable_state
            and now - self.candidate_since >= self.settle_time
        ):
            self.stable_state = raw_state
            stable_changed = True

        return self.stable_state, raw_changed, stable_changed


class KeyboardControls:
    def __init__(self):
        self.soft_active = False
        self.entities_active = False
        self.glitch_active = False
        self.beyond_active = False
        self.debug_active = False
        self.selfie_mode = False
        self.display_emergency_off = False
        self._photo_requested = threading.Event()

        self.frequency_index = 0
        self.frequencies = FREQUENCIES

    def handle_key(self, key):
        should_quit = False


        if key == ord("1"):
            if self.beyond_active:
                print("Soft blockiert: Beyond ist aktiv.")
            else:
                self.soft_active = not self.soft_active
                print(f"Soft: {self.soft_active}")

        elif key == ord("2"):
            if self.beyond_active:
                print("Entities blockiert: Beyond ist aktiv.")
            else:
                self.entities_active = not self.entities_active
                print(f"Entities: {self.entities_active}")

        elif key == ord("3"):
            if self.beyond_active:
                print("Glitch blockiert: Beyond ist aktiv.")
            else:
                self.glitch_active = not self.glitch_active
                print(f"Glitch: {self.glitch_active}")

        elif key == ord("4"):
            if self.beyond_active:
                self.beyond_active = False
                print("Beyond: False")

            elif (
                not self.soft_active
                and not self.entities_active
                and not self.glitch_active
            ):
                self.beyond_active = True
                print("Beyond: True")

            else:
                print("Beyond kann nur aktiviert werden, wenn Soft, Entities und Glitch deaktiviert sind.")

        elif key == ord("+"):
            self.frequency_index = min(
                self.frequency_index + 1,
                len(self.frequencies) - 1
            )
            print(f"Frequency: {self.get_frequency()}")

        elif key == ord("d"):
            self.debug_active = not self.debug_active
            print(f"Debug: {self.debug_active}")

        elif key == ord("-"):
            self.frequency_index = max(
                self.frequency_index - 1,
                0
            )
            print(f"Frequency: {self.get_frequency()}")

        elif key == ord("p"):
            self._photo_requested.set()

        elif key == ord("r"):
            self.selfie_mode = not self.selfie_mode
            print(f"Selfie-Modus simuliert: {self.selfie_mode}")

        elif key == ord("x"):
            self.display_emergency_off = not self.display_emergency_off
            print(f"Display Not-Aus: {self.display_emergency_off}")

        elif key == ord("q"):
            should_quit = True

        return should_quit

    def get_active_modes(self):
        if self.beyond_active:
            return [MODE_BEYOND]

        active_modes = []

        if self.soft_active:
            active_modes.append(MODE_SOFT)

        if self.entities_active:
            active_modes.append(MODE_ENTITIES)

        if self.glitch_active:
            active_modes.append(MODE_GLITCH)

        if not active_modes:
            active_modes.append(MODE_NORMAL)

        return active_modes

    def get_frequency(self):
        return self.frequencies[self.frequency_index]

    def is_debug_active(self):
        return self.debug_active

    def consume_photo_request(self):
        if not self._photo_requested.is_set():
            return False
        self._photo_requested.clear()
        return True

    def is_selfie_mode(self):
        return self.selfie_mode

    def is_display_emergency_off(self):
        return self.display_emergency_off

    def set_detector_level(self, level):
        pass

    def close(self):
        pass


class GPIOControls(KeyboardControls):
    """Physical Raspberry Pi controls using BCM GPIO numbering."""

    MODE_PINS = {
        MODE_SOFT: 17,       # physical pin 11
        MODE_ENTITIES: 27,   # physical pin 13
        MODE_GLITCH: 22,     # physical pin 15
        MODE_BEYOND: 23,     # physical pin 16
    }
    PHOTO_PIN = 24           # physical pin 18
    ENCODER_CLK_PIN = 5      # physical pin 29
    ENCODER_DT_PIN = 6       # physical pin 31
    ENCODER_SWITCH_PIN = 13  # physical pin 33
    DISPLAY_REED_PIN = 18    # KY-021 signal, physical pin 12
    DISPLAY_KILL_PIN = 19    # aircraft switch OUT, physical pin 35
    DISPLAY_KILL_ACTIVE_STATE = True
    STATUS_LED_PIN = 4       # physical pin 7
    DETECTOR_LED_PINS = (
        16,  # LED 1, physical pin 36
        20,  # LED 2, physical pin 38
        21,  # LED 3, physical pin 40
        12,  # LED 4, physical pin 32
        26,  # LED 5, physical pin 37
    )

    def __init__(
        self,
        *,
        button_factory=None,
        led_factory=None,
        encoder_factory=None,
        clock=time.monotonic,
    ):
        super().__init__()
        if button_factory is None or led_factory is None or encoder_factory is None:
            try:
                from gpiozero import Button, LED, RotaryEncoder
            except (ImportError, ModuleNotFoundError) as error:
                raise RuntimeError(
                    "gpiozero fehlt. Auf dem Raspberry Pi bitte "
                    "'sudo apt install python3-gpiozero python3-lgpio' ausfuehren."
                ) from error
            button_factory = button_factory or Button
            led_factory = led_factory or LED
            encoder_factory = encoder_factory or RotaryEncoder

        self._frequency_lock = threading.RLock()
        self._detector_led_lock = threading.Lock()
        self._last_hardware_modes = None
        self._last_selfie_mode = None
        self.mode_switches = {
            mode: button_factory(pin, pull_up=True, bounce_time=0.05)
            for mode, pin in self.MODE_PINS.items()
        }
        self.photo_button = button_factory(
            self.PHOTO_PIN,
            pull_up=True,
            bounce_time=0.16,
        )
        self.encoder_button = button_factory(
            self.ENCODER_SWITCH_PIN,
            pull_up=True,
            bounce_time=0.12,
        )
        self.encoder = encoder_factory(
            self.ENCODER_CLK_PIN,
            self.ENCODER_DT_PIN,
            max_steps=0,
            wrap=False,
        )
        # The KY-021 has its own pull-up. A nearby magnet closes its reed
        # contact and pulls S low. The magnet is mounted so it is near the
        # sensor while the display is folded closed.
        self.display_reed = button_factory(
            self.DISPLAY_REED_PIN,
            pull_up=None,
            active_state=False,
            bounce_time=0.08,
        )
        # HIGH means emergency blackout is active. The internal pull-down keeps
        # the input defined if OUT briefly loses contact.
        self.display_kill_switch = button_factory(
            self.DISPLAY_KILL_PIN,
            pull_up=False,
            bounce_time=0.12,
        )
        initial_kill_state = bool(self.display_kill_switch.is_pressed)
        self._display_kill_filter = DebouncedBoolean(
            initial_kill_state,
            settle_time=0.25,
            clock=clock,
        )
        self.status_led = led_factory(self.STATUS_LED_PIN)
        self.detector_leds = [
            led_factory(pin) for pin in self.DETECTOR_LED_PINS
        ]

        self.photo_button.when_pressed = self._request_photo
        self.encoder_button.when_pressed = self._reset_frequency
        self.encoder.when_rotated_clockwise = self._increase_frequency
        self.encoder.when_rotated_counter_clockwise = self._decrease_frequency
        self.status_led.on()
        self.set_detector_level(0.0)

    def _request_photo(self, *args):
        self._photo_requested.set()

    def _increase_frequency(self, *args):
        with self._frequency_lock:
            self.frequency_index = min(
                self.frequency_index + 1,
                len(self.frequencies) - 1,
            )
            print(f"Frequency: {self.get_frequency()}")

    def _decrease_frequency(self, *args):
        with self._frequency_lock:
            self.frequency_index = max(self.frequency_index - 1, 0)
            print(f"Frequency: {self.get_frequency()}")

    def _reset_frequency(self, *args):
        with self._frequency_lock:
            self.frequency_index = 0
            print("Frequency: 0")

    def set_detector_level(self, level):
        """Light a short strength bar for one Geiger detector pulse."""
        level = max(0.0, min(float(level), 1.0))
        lit_count = 0
        if level > 0.0:
            lit_count = max(
                1,
                min(
                    len(self.detector_leds),
                    math.ceil(level * len(self.detector_leds)),
                ),
            )

        with self._detector_led_lock:
            for index, led in enumerate(self.detector_leds, start=1):
                if index <= lit_count:
                    led.on()
                else:
                    led.off()

    def _turn_off_detector_leds(self):
        with self._detector_led_lock:
            for led in self.detector_leds:
                led.off()

    def get_active_modes(self):
        states = tuple(
            bool(self.mode_switches[mode].is_pressed)
            for mode in (MODE_SOFT, MODE_ENTITIES, MODE_GLITCH, MODE_BEYOND)
        )
        soft, entities, glitch, beyond = states

        # The physical switch can remain on, but From Beyond only becomes
        # effective when the other three mode switches are all off.
        if beyond and not (soft or entities or glitch):
            active_modes = [MODE_BEYOND]
        else:
            active_modes = []
            if soft:
                active_modes.append(MODE_SOFT)
            if entities:
                active_modes.append(MODE_ENTITIES)
            if glitch:
                active_modes.append(MODE_GLITCH)
            if not active_modes:
                active_modes.append(MODE_NORMAL)

        if states != self._last_hardware_modes:
            self._last_hardware_modes = states
            print(f"GPIO-Modi: {active_modes}")
            if beyond and (soft or entities or glitch):
                print("Beyond wartet: Modus 1, 2 und 3 muessen aus sein.")
        return active_modes

    def get_frequency(self):
        with self._frequency_lock:
            return self.frequencies[self.frequency_index]

    def is_selfie_mode(self):
        # Magnet present = folded closed = normal camera orientation.
        # Magnet absent = unfolded = rotate only the finished camera frame.
        selfie_mode = not bool(self.display_reed.is_pressed)
        if selfie_mode != self._last_selfie_mode:
            self._last_selfie_mode = selfie_mode
            print(
                "Selfie-Modus: "
                f"{'AKTIV (180 Grad + gespiegelt)' if selfie_mode else 'AUS'}",
                flush=True,
            )
        return selfie_mode

    def is_display_emergency_off(self):
        raw_state = bool(self.display_kill_switch.is_pressed)
        state, raw_changed, stable_changed = self._display_kill_filter.update(
            raw_state,
        )
        if raw_changed:
            print(
                "Display-Not-Aus Rohsignal: "
                f"{'HIGH' if raw_state else 'LOW'}",
                flush=True,
            )
        if stable_changed:
            print(
                "Display-Not-Aus stabil: "
                f"{'AKTIV' if state else 'AUS'}",
                flush=True,
            )
        return state

    def close(self):
        self.status_led.off()
        self._turn_off_detector_leds()
        devices = [
            *self.mode_switches.values(),
            self.photo_button,
            self.encoder_button,
            self.encoder,
            self.display_reed,
            self.display_kill_switch,
            self.status_led,
            *self.detector_leds,
        ]
        for device in devices:
            close = getattr(device, "close", None)
            if close is not None:
                close()


def create_controls(force_gpio=None):
    use_gpio = is_raspberry_pi() if force_gpio is None else bool(force_gpio)
    if not use_gpio:
        return KeyboardControls()

    try:
        print("Steuerung: Raspberry-Pi-GPIO")
        return GPIOControls()
    except RuntimeError as error:
        print(f"GPIO-Steuerung nicht verfuegbar: {error}")
        print("Tastatursteuerung bleibt als Notbetrieb aktiv.")
        return KeyboardControls()
