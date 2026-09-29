import os
from pathlib import Path


RASPBERRY_PI_MODEL_FILE = Path("/proc/device-tree/model")


def is_raspberry_pi(model_file: str | Path = RASPBERRY_PI_MODEL_FILE) -> bool:
    """Detects Raspberry Pi without importing Pi-only Python packages."""
    forced = os.environ.get("BEYOND_LENCE_FORCE_PI", "").strip().lower()
    if forced in {"1", "true", "yes", "on"}:
        return True
    if forced in {"0", "false", "no", "off"}:
        return False

    try:
        model = Path(model_file).read_text(errors="ignore")
    except OSError:
        return False
    return "raspberry pi" in model.lower()
