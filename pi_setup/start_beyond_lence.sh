#!/bin/sh
set -u

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
LOG_DIRECTORY="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIRECTORY/beyond-lence.log"
PI_PYTHON="$PROJECT_DIR/.pi_venv/bin/python"
MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"
export MPLCONFIGDIR
mkdir -p "$LOG_DIRECTORY"
mkdir -p "$MPLCONFIGDIR"
cd "$PROJECT_DIR" || exit 1

if [ ! -x "$PI_PYTHON" ]; then
    echo "Pi-Python-Umgebung fehlt. Bitte pi_setup/install_on_pi.sh ausfuehren." \
        >>"$LOG_FILE"
    exit 1
fi

# A clean exit (for example via Q) stays stopped. Unexpected failures restart
# after a short pause, which is useful for a closed camera housing.
while true; do
    "$PI_PYTHON" "$PROJECT_DIR/main.py" >>"$LOG_FILE" 2>&1
    STATUS=$?
    if [ "$STATUS" -eq 0 ]; then
        exit 0
    fi
    sleep 3
done
