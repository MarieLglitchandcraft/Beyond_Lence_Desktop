#!/bin/sh
set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Verwendung: $0 PI-BENUTZERNAME [PI-ADRESSE]"
    echo "Beispiel:   $0 marie raspberrypi.local"
    exit 1
fi

PI_USER=$1
PI_HOST=${2:-raspberrypi.local}
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
DESTINATION="$PI_USER@$PI_HOST:beyond_lence_neu/"

echo "Kopiere Beyond Lence nach $DESTINATION ..."
rsync -av \
    --exclude '.git/' \
    --exclude '.DS_Store' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude 'venv/' \
    --exclude '.pi_venv/' \
    --exclude 'photos/' \
    --exclude 'logs/' \
    "$PROJECT_DIR/" \
    "$DESTINATION"

echo
echo "Kopie abgeschlossen. Naechster Schritt:"
echo "ssh $PI_USER@$PI_HOST"
echo "cd ~/beyond_lence_neu"
echo "./pi_setup/install_on_pi.sh"
