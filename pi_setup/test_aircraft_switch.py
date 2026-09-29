#!/usr/bin/env python3
"""Short raw GPIO19 contact test for the three-pin aircraft switch."""

import argparse
import time


def main():
    try:
        from gpiozero import DigitalInputDevice
    except ImportError:
        raise SystemExit(
            "gpiozero fehlt. Dieses Diagnoseprogramm bitte auf dem "
            "Raspberry Pi in der Projektumgebung starten."
        )

    parser = argparse.ArgumentParser(
        description="Prueft den Flugzeugschalter auf schnelle Kontaktwechsel.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=12.0,
        help="Testdauer in Sekunden (Standard: 12).",
    )
    arguments = parser.parse_args()
    duration = max(2.0, arguments.seconds)

    switch = DigitalInputDevice(
        19,
        pull_up=False,
        bounce_time=None,
    )
    start = time.monotonic()
    previous = bool(switch.is_active)
    last_transition = None
    transitions = 0
    rapid_transitions = 0

    print("Flugzeugschalter-Test auf GPIO19 / Pin 35")
    print("Schalter einmal langsam EIN und wieder AUS schalten.")
    print(f"Startzustand: {'HIGH / Not-Aus' if previous else 'LOW / normal'}")

    try:
        while time.monotonic() - start < duration:
            current = bool(switch.is_active)
            now = time.monotonic()
            if current != previous:
                transitions += 1
                elapsed = now - start
                rapid = (
                    last_transition is not None
                    and now - last_transition < 0.15
                )
                if rapid:
                    rapid_transitions += 1
                suffix = "  < SCHNELLER MEHRFACHWECHSEL" if rapid else ""
                print(
                    f"{elapsed:6.3f}s: "
                    f"{'HIGH / Not-Aus' if current else 'LOW / normal'}"
                    f"{suffix}",
                    flush=True,
                )
                previous = current
                last_transition = now
            time.sleep(0.01)
    finally:
        switch.close()

    print()
    print(f"Zustandswechsel insgesamt: {transitions}")
    print(f"Schnelle Mehrfachwechsel: {rapid_transitions}")
    if rapid_transitions:
        print(
            "ERGEBNIS: Kontaktprellen oder Wackelkontakt erkannt. "
            "Stecker, Loetstellen und GND pruefen."
        )
    elif transitions < 2:
        print(
            "ERGEBNIS: Es wurden nicht beide Schalterlagen erkannt. "
            "Verdrahtung und OUT-Anschluss pruefen."
        )
    else:
        print("ERGEBNIS: Kein schneller Wackelkontakt erkannt.")


if __name__ == "__main__":
    main()
