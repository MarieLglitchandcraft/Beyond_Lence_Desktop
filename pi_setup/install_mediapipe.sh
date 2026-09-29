#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PI_ENVIRONMENT="$PROJECT_DIR/.pi_venv"
PI_PYTHON="$PI_ENVIRONMENT/bin/python"
PI_PIP="$PI_ENVIRONMENT/bin/pip"
MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"
MEDIAPIPE_VERSION=0.10.18
ARCHITECTURE=$(uname -m)
PYTHON_MINOR=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
export MPLCONFIGDIR
mkdir -p "$MPLCONFIGDIR"

if [ "$ARCHITECTURE" != "aarch64" ]; then
    echo "MediaPipe benoetigt hier Raspberry Pi OS 64 Bit (aarch64)."
    echo "Gefundene Architektur: $ARCHITECTURE"
    exit 1
fi

case "$PYTHON_MINOR" in
    3.9|3.10|3.11|3.12)
        ;;
    *)
        echo "Kein fertiges MediaPipe-ARM64-Paket fuer Python $PYTHON_MINOR."
        echo "Unterstuetzt werden Python 3.9 bis 3.12."
        echo "Empfohlen: Raspberry Pi OS Bookworm 64 Bit mit Python 3.11."
        exit 1
        ;;
esac

echo "Erzeuge Pi-Python-Umgebung mit Zugriff auf Picamera2 und GPIO ..."
python3 -m venv --system-site-packages "$PI_ENVIRONMENT"
"$PI_PYTHON" -m pip install --upgrade pip wheel

# Version 0.10.18 is the newest release that still supplies pre-built Linux
# aarch64 wheels for all Python versions used by Bullseye/Bookworm up to 3.12.
# Dependencies are installed separately so pip does not replace Raspberry Pi
# OS OpenCV/NumPy with another, potentially incompatible camera stack.
echo "Installiere MediaPipe $MEDIAPIPE_VERSION fuer ARM64 ..."
"$PI_PIP" install \
    --only-binary=:all: \
    --no-deps \
    "mediapipe==$MEDIAPIPE_VERSION"

"$PI_PIP" install \
    "absl-py>=2,<3" \
    "attrs>=19,<26" \
    "certifi" \
    "flatbuffers>=2,<26" \
    "protobuf>=3.20,<5" \
    "sounddevice>=0.4,<1"

echo "Teste Gesicht, Pose und Personenmaske mit den Projektmodellen ..."
# When the installer is launched over SSH while the desktop is already logged
# in, use its XWayland display for MediaPipe's graphics service.
if [ -z "${DISPLAY:-}" ] && [ -S /tmp/.X11-unix/X0 ]; then
    DISPLAY=:0
    export DISPLAY
fi
if ! "$PI_PYTHON" "$PROJECT_DIR/mediapipe_probe.py" \
    "$PROJECT_DIR/models/mediapipe/blaze_face_short_range.tflite" \
    "$PROJECT_DIR/models/mediapipe/pose_landmarker_lite.task" \
    "$PROJECT_DIR/models/mediapipe/selfie_segmenter.tflite"; then
    echo
    echo "MediaPipe wurde installiert, aber die drei Modelle konnten nicht gestartet werden."
    echo "Fuehre die Installation erneut in einem Terminal auf dem sichtbaren Pi-Desktop aus."
    echo "Bei SSH muss der Pi-Desktop bereits angemeldet und sichtbar sein."
    exit 1
fi

"$PI_PYTHON" -c 'import mediapipe as mp; print("MediaPipe aktiv:", mp.__version__)'
echo "MediaPipe und alle drei Erkennungsmodelle sind einsatzbereit."
