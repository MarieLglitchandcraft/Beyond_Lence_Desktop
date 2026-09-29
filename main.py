import os
from pathlib import Path
import sys
import time
import traceback


PROJECT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ENVIRONMENT = PROJECT_DIRECTORY / "venv"
PROJECT_PYTHON = PROJECT_ENVIRONMENT / "bin" / "python"
PI_ENVIRONMENT = PROJECT_DIRECTORY / ".pi_venv"
PI_PYTHON = PI_ENVIRONMENT / "bin" / "python"


def ensure_project_environment():
    """Relaunch with the project interpreter when main.py is run via python3."""
    current_environment = Path(sys.prefix).resolve()
    if current_environment in (
        PROJECT_ENVIRONMENT.resolve(),
        PI_ENVIRONMENT.resolve(),
    ):
        return

    if (
        sys.platform.startswith("linux")
        and PI_PYTHON.is_file()
        and os.access(PI_PYTHON, os.X_OK)
    ):
        print("Starte die Raspberry-Pi-Umgebung mit MediaPipe ...", flush=True)
        os.execv(
            str(PI_PYTHON),
            [str(PI_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        )

    # Picamera2 and gpiozero are installed by Raspberry Pi OS via apt and must
    # remain visible in the Pi venv through --system-site-packages. The bundled
    # venv stays Mac-only.
    if (
        sys.platform == "darwin"
        and PROJECT_PYTHON.is_file()
        and os.access(PROJECT_PYTHON, os.X_OK)
    ):
        print("Starte automatisch die Beyond-Lence-Projektumgebung ...", flush=True)
        os.execv(
            str(PROJECT_PYTHON),
            [str(PROJECT_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        )


if __name__ == "__main__":
    ensure_project_environment()

import argparse
import cv2

from camera import create_camera, validate_camera_frame
from config import FRAME_HEIGHT, FRAME_WIDTH, MODE_NORMAL, WINDOW_NAME
from controls import create_controls
from display_output import (
    create_display_window,
    detect_display_environment,
    detect_display_info,
)
from effects import EffectEngine
from entity_manager import EntityManager
from overlays import OverlayEngine
from photo_capture import PhotoCapture
from renderer import Renderer
from scene_manager import SceneManager
from sound import create_geiger_buzzer
from vision import VisionEngine


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Beyond Lence Spektralkamera",
    )
    window_group = parser.add_mutually_exclusive_group()
    window_group.add_argument(
        "--windowed",
        action="store_true",
        help="In einem normalen, verschiebbaren Diagnosefenster starten.",
    )
    window_group.add_argument(
        "--fullscreen",
        action="store_true",
        help="Randloses Vollbild verwenden (Standard).",
    )
    return parser.parse_args(argv)


def _print_display_diagnostics(display, window):
    environment = detect_display_environment()
    print("=== Startdiagnose: Display ===", flush=True)
    print(
        "Erkannte Bildschirmaufloesung: "
        f"{display.width}x{display.height} "
        f"({display.orientation}, Quelle: {display.source}, "
        f"Ausgang: {display.name})",
        flush=True,
    )
    print(
        f"Verwendete Renderaufloesung: "
        f"{window.render_size[0]}x{window.render_size[1]}",
        flush=True,
    )
    print(f"Fenstermodus: {window.mode_name}", flush=True)
    print(f"Ausgabe-Backend: {window.backend_name}", flush=True)
    print(f"Display-Umgebung: {environment['session']}", flush=True)
    print(f"OpenCV-HighGUI: {environment['opencv_highgui']}", flush=True)
    for key in (
        "DISPLAY",
        "WAYLAND_DISPLAY",
        "XDG_SESSION_TYPE",
        "XDG_CURRENT_DESKTOP",
        "XDG_SESSION_DESKTOP",
        "DESKTOP_SESSION",
        "XDG_RUNTIME_DIR",
        "SDL_VIDEODRIVER",
        "QT_QPA_PLATFORM",
        "GDK_BACKEND",
    ):
        print(f"{key}={environment[key]}", flush=True)

    if display.is_portrait:
        print(
            "WARNUNG: Das System meldet das Display im Hochformat. "
            "Beyond Lence behaelt sein Kamerabild im Querformat und versieht "
            "es mit schwarzen Raendern. Die Systemrotation wird nicht "
            "automatisch veraendert.",
            flush=True,
        )
    if display.source == "fallback":
        print(
            "WARNUNG: Die Bildschirmgroesse konnte nicht automatisch "
            f"ermittelt werden; Fallback {display.width}x{display.height}.",
            flush=True,
        )


def _print_camera_diagnostics(camera, frame, window):
    frame_height, frame_width = frame.shape[:2]
    target_width, target_height = window.render_size
    uniform_scale = min(
        target_width / max(frame_width, 1),
        target_height / max(frame_height, 1),
    )
    content_width = int(round(frame_width * uniform_scale))
    content_height = int(round(frame_height * uniform_scale))

    print("=== Startdiagnose: Kamera und Skalierung ===", flush=True)
    print(
        f"Kamera-Backend: {getattr(camera, 'backend_name', type(camera).__name__)}",
        flush=True,
    )
    print(
        "Angeforderte Kameraaufloesung: "
        f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
        flush=True,
    )
    raw_shape = getattr(camera, "last_raw_shape", None)
    if raw_shape is not None:
        print(f"Erkannte rohe Kameraform: {raw_shape}", flush=True)
        if len(raw_shape) >= 2:
            print(
                "Erkannte rohe Kameraaufloesung: "
                f"{raw_shape[1]}x{raw_shape[0]}",
                flush=True,
            )
    else:
        print("Erkannte rohe Kameraform: nicht verfuegbar", flush=True)
    print(f"Form des ersten Kameraframes: {tuple(frame.shape)}", flush=True)
    print(
        f"Verarbeitete Kameraaufloesung: {frame_width}x{frame_height}",
        flush=True,
    )
    print(
        "Skalierungsfaktoren Leinwand: "
        f"x={target_width / max(frame_width, 1):.3f}, "
        f"y={target_height / max(frame_height, 1):.3f}",
        flush=True,
    )
    print(
        "Proportionale Ausgabeskalierung: "
        f"{uniform_scale:.3f}; sichtbarer Inhalt "
        f"{content_width}x{content_height}",
        flush=True,
    )

    if frame_height > frame_width:
        print(
            "WARNUNG: Das erste Kameraframe ist hochkant. Breite und Hoehe "
            "werden protokolliert, aber nicht automatisch vertauscht.",
            flush=True,
        )


def _read_first_valid_frame(camera, attempts=5, delay=0.15):
    last_reason = "Kein Leseversuch."
    for attempt in range(1, max(1, int(attempts)) + 1):
        try:
            frame = camera.read()
        except Exception as error:
            last_reason = f"{type(error).__name__}: {error}"
            print(
                f"Kameralesefehler {attempt}/{attempts}: {last_reason}",
                flush=True,
            )
        else:
            valid, reason = validate_camera_frame(frame)
            if valid:
                return frame, "ok"
            last_reason = reason
            print(
                f"Ungueltiges Kameraframe {attempt}/{attempts}: {reason}",
                flush=True,
            )
        if attempt < attempts:
            time.sleep(max(0.0, float(delay)))
    return None, last_reason


def main(argv=None):
    arguments = parse_arguments(argv)
    fullscreen = not arguments.windowed

    print("=== Beyond Lence ===")
    print("1 = Soft")
    print("2 = Entities")
    print("3 = Glitch")
    print("4 = Beyond")
    print("+ = Frequency hoch")
    print("- = Frequency runter")
    print("d = Debug")
    print("p = sichtbares Foto speichern")
    print("s = Rohbild fuer Diagnose speichern")
    print("r = Selfie-Modus simulieren")
    print("x = Display-Not-Aus simulieren")
    print("q = Beenden")
    print()

    display = detect_display_info(use_tkinter=False)
    window = create_display_window(
        WINDOW_NAME,
        display,
        fullscreen=fullscreen,
        content_size=(FRAME_WIDTH, FRAME_HEIGHT),
    )

    camera = None
    controls = None
    geiger_buzzer = None
    vision = None

    try:
        try:
            window.open()
        except Exception as error:
            print(
                "FENSTER-FEHLER: Das Anzeigefenster konnte nicht "
                f"oeffnen: {error}",
                flush=True,
            )
            return 0
        _print_display_diagnostics(display, window)

        try:
            camera = create_camera()
        except Exception as error:
            print(
                f"KAMERA-FEHLER BEIM START: {type(error).__name__}: {error}",
                flush=True,
            )
            traceback.print_exc()
            window.show_error(
                "KAMERA NICHT VERFUEGBAR",
                (
                    str(error),
                    "Kamerakabel, Picamera2 und Berechtigungen pruefen.",
                    "Das Programm wird beendet und bleibt nicht leer haengen.",
                    "Q oder ESC schliesst diese Meldung.",
                ),
            )
            return 0

        first_frame, first_frame_reason = _read_first_valid_frame(camera)
        if first_frame is None:
            message = (
                "Nach mehreren Versuchen wurde kein gueltiges Kameraframe "
                f"empfangen: {first_frame_reason}"
            )
            print(f"KAMERA-FEHLER: {message}", flush=True)
            window.show_error(
                "KEIN KAMERABILD",
                (
                    message,
                    "Kamerakabel und Picamera2-Konfiguration pruefen.",
                    "Das Programm beendet sich nach dieser Meldung sauber.",
                ),
            )
            return 0

        _print_camera_diagnostics(camera, first_frame, window)

        try:
            controls = create_controls()
            geiger_buzzer = create_geiger_buzzer(
                detector_callback=controls.set_detector_level,
            )
            photo_capture = PhotoCapture()
            effects = EffectEngine()
            renderer = Renderer()
            vision = VisionEngine()
            print(vision.detector_status, flush=True)
            overlays = OverlayEngine()
            entities = EntityManager(overlays)
            scene_manager = SceneManager()
        except Exception as error:
            print(
                f"STARTFEHLER: {type(error).__name__}: {error}",
                flush=True,
            )
            traceback.print_exc()
            window.show_error(
                "BEYOND LENCE STARTFEHLER",
                (
                    str(error),
                    "Details stehen im Terminal beziehungsweise im "
                    "Beyond-Lence-Startprotokoll.",
                ),
            )
            return 0

        pending_frame = first_frame
        invalid_frame_count = 0
        while True:
            if pending_frame is not None:
                frame = pending_frame
                pending_frame = None
            else:
                try:
                    frame = camera.read()
                except Exception as error:
                    frame = None
                    frame_reason = f"{type(error).__name__}: {error}"
                else:
                    _, frame_reason = validate_camera_frame(frame)

                valid, validation_reason = validate_camera_frame(frame)
                if not valid:
                    invalid_frame_count += 1
                    reason = frame_reason or validation_reason
                    print(
                        "KAMERA-FEHLER IM BETRIEB "
                        f"{invalid_frame_count}/3: {reason}",
                        flush=True,
                    )
                    if invalid_frame_count >= 3:
                        window.show_error(
                            "KAMERASIGNAL VERLOREN",
                            (
                                reason,
                                "Das Programm beendet sich jetzt sauber.",
                            ),
                        )
                        return 0
                    window.poll_key(100)
                    continue
                invalid_frame_count = 0

            effects.prepare_other_world_frames(frame.shape[1], frame.shape[0])
            analysis = vision.analyze(frame)
            raw_frame = frame.copy()
            scene_state = scene_manager.evaluate(analysis)
            active_modes = controls.get_active_modes()
            frequency = controls.get_frequency()
            display_emergency_off = controls.is_display_emergency_off()
            selfie_mode = controls.is_selfie_mode()

            entities.update(
                active_modes,
                frequency,
                scene_state,
            )

            if display_emergency_off:
                geiger_buzzer.update("0", [MODE_NORMAL])
            else:
                geiger_buzzer.update(
                    frequency,
                    active_modes,
                    entity_count=len(entities.entities),
                    portal_count=len(scene_state.get("dark_portals", [])),
                )

            frame = effects.apply(
                frame,
                active_modes,
                frequency,
                analysis,
            )
            frame = entities.draw(frame, analysis.get("person_mask"))
            parallel_world_source = frame.copy()

            renderer.set_frequency(frequency)
            frame = renderer.render_camera(frame, active_modes)
            frame = effects.apply_display_glitch(
                frame,
                active_modes,
                frequency,
                parallel_world_source,
                scene_state,
            )
            # Mirror only camera imagery and visual phenomena. Interface text
            # is drawn afterwards and therefore remains readable.
            frame = renderer.apply_selfie_camera_mirror(
                frame,
                selfie_mode=selfie_mode,
            )
            frame = renderer.render_interface(frame, active_modes)
            photo_frame = frame.copy()

            if controls.is_debug_active():
                debug_analysis = analysis
                if selfie_mode:
                    debug_analysis = vision.mirror_analysis_for_display(
                        analysis,
                        frame.shape[1],
                    )
                frame = vision.draw_debug(frame, debug_analysis)

            if controls.consume_photo_request():
                if display_emergency_off:
                    print("Foto blockiert: Display-Not-Aus ist aktiv.")
                else:
                    visible_photo_frame = renderer.apply_selfie_display_rotation(
                        photo_frame,
                        selfie_mode=selfie_mode,
                    )
                    photo_path = photo_capture.save(visible_photo_frame)
                    if photo_path is not None:
                        print(f"Foto gespeichert: {photo_path}")
                        # Start immediately before this frame is shown. This
                        # guarantees a visible flash even when MediaPipe makes
                        # the following camera iteration comparatively slow.
                        renderer.show_capture_confirmation(
                            photo_frame,
                            duration=2.0,
                        )
                    else:
                        print("Foto konnte nicht gespeichert werden.")

            frame = renderer.apply_capture_feedback(frame)
            # Only the app frame is turned for the mechanically flipped panel.
            # The compositor, desktop and mouse coordinate system stay normal.
            frame = renderer.apply_selfie_display_rotation(
                frame,
                selfie_mode=selfie_mode,
            )

            window.show(
                frame,
                emergency_off=display_emergency_off,
            )

            key = window.poll_key(30)

            if key == ord("s"):
                raw_path = "debug_raw_frame.png"
                if cv2.imwrite(raw_path, raw_frame):
                    print(f"Rohbild gespeichert: {raw_path}")
                else:
                    print("Rohbild konnte nicht gespeichert werden.")

            should_quit = key != 255 and controls.handle_key(key)

            if should_quit:
                break
    except Exception as error:
        print(
            f"UNERWARTETER LAUFZEITFEHLER: {type(error).__name__}: {error}",
            flush=True,
        )
        traceback.print_exc()
        window.show_error(
            "LAUFZEITFEHLER",
            (
                str(error),
                "Details stehen im Terminal beziehungsweise im Startprotokoll.",
            ),
        )
        return 0
    finally:
        if geiger_buzzer is not None:
            geiger_buzzer.close()
        if controls is not None:
            controls.close()
        if vision is not None:
            vision.close()
        if camera is not None:
            camera.release()
        window.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
