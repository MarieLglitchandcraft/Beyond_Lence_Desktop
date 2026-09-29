#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
AUTOSTART_DIRECTORY="$HOME/.config/autostart"
AUTOSTART_FILE="$AUTOSTART_DIRECTORY/beyond-lence.desktop"
TEMPLATE="$PROJECT_DIR/pi_setup/beyond-lence.desktop.template"
LAUNCHER="$PROJECT_DIR/pi_setup/start_beyond_lence.sh"
MEDIAPIPE_INSTALLER="$PROJECT_DIR/pi_setup/install_mediapipe.sh"
SHUTDOWN_OVERLAY="dtoverlay=gpio-shutdown,gpio_pin=25,active_low=1,gpio_pull=up,debounce=300"

if ! grep -qi "raspberry pi" /proc/device-tree/model 2>/dev/null; then
    echo "Dieses Installationsskript muss auf dem Raspberry Pi laufen."
    exit 1
fi

echo "Installiere Raspberry-Pi-Kamera- und GPIO-Pakete ..."
sudo apt update
sudo apt install -y \
    python3-picamera2 \
    python3-opencv \
    python3-pygame \
    python3-numpy \
    python3-pil \
    python3-gpiozero \
    python3-lgpio \
    python3-venv \
    python3-pip \
    python3-matplotlib \
    python3-cffi \
    libportaudio2 \
    libgl1

chmod +x "$MEDIAPIPE_INSTALLER"
"$MEDIAPIPE_INSTALLER"

if [ -f /boot/firmware/config.txt ]; then
    BOOT_CONFIG=/boot/firmware/config.txt
else
    BOOT_CONFIG=/boot/config.txt
fi

if ! grep -Fq "$SHUTDOWN_OVERLAY" "$BOOT_CONFIG"; then
    echo "Aktiviere den sicheren Abschalttaster auf GPIO 25 ..."
    printf '\n# Beyond Lence: unterer Abschalttaster, Pin 22 gegen Pin 20/GND\n%s\n' \
        "$SHUTDOWN_OVERLAY" | sudo tee -a "$BOOT_CONFIG" >/dev/null
fi

chmod +x "$LAUNCHER"
mkdir -p "$AUTOSTART_DIRECTORY"
sed \
    -e "s|@PROJECT_DIR@|$PROJECT_DIR|g" \
    -e "s|@LAUNCHER@|$LAUNCHER|g" \
    "$TEMPLATE" >"$AUTOSTART_FILE"

echo
echo "Beyond Lence ist fuer den automatischen Start eingerichtet."
echo "Autostart-Datei: $AUTOSTART_FILE"
echo "Fotos werden gespeichert in: $PROJECT_DIR/photos"
echo "Abschalttaster: physischer Pin 22 gegen Pin 20 (GND)"
echo "Personenerkennung: MediaPipe mit Gesicht, Pose und Personenmaske"
echo
echo "Wichtig: Raspberry Pi OS muss den Desktop automatisch anmelden."
echo "Das kann bei Bedarf mit 'sudo raspi-config' aktiviert werden."
