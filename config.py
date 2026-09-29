WINDOW_NAME = "Beyond Lence"

CAMERA_INDEX = 0
# Interne Kamera- und Effektauflösung. Die physische Bildschirmauflösung wird
# beim Start separat erkannt und ist nicht mehr auf diese Werte festgelegt.
FRAME_WIDTH = 1024
FRAME_HEIGHT = 600

MODE_NORMAL = 0
MODE_SOFT = 1
MODE_ENTITIES = 2
MODE_GLITCH = 3
MODE_BEYOND = 4


FREQUENCIES = ("0", "I", "II", "III", "IV", "V")

# Die Größen beziehen sich auf die aktuelle Bildbreite, nicht auf feste Pixel.
FREQUENCY_PROFILES = {
    "0": {
        "spawn_chance": 0.0,
        "attempt_interval": 99.0,
        "spawn_interval": 99.0,
        "max_entities": 0,
        "size_range": (0.0, 0.0),
        "lifetime_range": (0.0, 0.0),
        "region_limit": 3,
        "intensity": 0.0,
    },
    "I": {
        "spawn_chance": 0.16,
        "attempt_interval": 1.6,
        "spawn_interval": 5.5,
        "max_entities": 1,
        "size_range": (0.08, 0.16),
        "lifetime_range": (0.45, 1.2),
        "region_limit": 4,
        "intensity": 0.15,
    },
    "II": {
        "spawn_chance": 0.28,
        "attempt_interval": 1.0,
        "spawn_interval": 3.4,
        "max_entities": 1,
        "size_range": (0.12, 0.25),
        "lifetime_range": (0.6, 1.8),
        "region_limit": 6,
        "intensity": 0.30,
    },
    "III": {
        "spawn_chance": 0.50,
        "attempt_interval": 0.5,
        "spawn_interval": 1.4,
        "max_entities": 2,
        "size_range": (0.18, 0.36),
        "lifetime_range": (0.8, 2.8),
        "region_limit": 8,
        "intensity": 0.50,
    },
    "IV": {
        "spawn_chance": 0.72,
        "attempt_interval": 0.30,
        "spawn_interval": 0.8,
        "max_entities": 4,
        "size_range": (0.24, 0.52),
        "lifetime_range": (0.65, 3.8),
        "region_limit": 10,
        "intensity": 0.75,
    },
    "V": {
        "spawn_chance": 1.0,
        "attempt_interval": 0.12,
        "spawn_interval": 0.18,
        "max_entities": 6,
        "size_range": (0.32, 0.85),
        "lifetime_range": (0.45, 4.8),
        "region_limit": 14,
        "intensity": 1.0,
    },
}

# From Beyond ist kein einzelner Monster-Effekt, sondern eine immer dichter
# sichtbar werdende, bereits im Raum vorhandene Ökologie.  Deshalb hat der
# exklusive Modus eine eigene, deutlich langsamere und länger lebende Staffelung.
BEYOND_PROFILES = {
    "0": {
        "spawn_chance": 0.0,
        "attempt_interval": 99.0,
        "spawn_interval": 99.0,
        "max_entities": 0,
        "size_range": (0.0, 0.0),
        "lifetime_range": (0.0, 0.0),
        "region_limit": 0,
        "intensity": 0.0,
    },
    "I": {
        "spawn_chance": 0.38,
        "attempt_interval": 0.70,
        "spawn_interval": 2.2,
        "max_entities": 2,
        "size_range": (0.10, 0.22),
        "lifetime_range": (6.0, 11.0),
        "region_limit": 5,
        "intensity": 0.16,
    },
    "II": {
        "spawn_chance": 0.62,
        "attempt_interval": 0.45,
        "spawn_interval": 1.25,
        "max_entities": 4,
        "size_range": (0.13, 0.32),
        "lifetime_range": (7.0, 13.0),
        "region_limit": 7,
        "intensity": 0.34,
    },
    "III": {
        "spawn_chance": 0.82,
        "attempt_interval": 0.28,
        "spawn_interval": 0.68,
        "max_entities": 6,
        "size_range": (0.17, 0.44),
        "lifetime_range": (8.0, 15.0),
        "region_limit": 10,
        "intensity": 0.56,
    },
    "IV": {
        "spawn_chance": 0.96,
        "attempt_interval": 0.16,
        "spawn_interval": 0.34,
        "max_entities": 8,
        "size_range": (0.22, 0.60),
        "lifetime_range": (8.0, 17.0),
        "region_limit": 14,
        "intensity": 0.78,
    },
    "V": {
        "spawn_chance": 1.0,
        "attempt_interval": 0.08,
        "spawn_interval": 0.18,
        "max_entities": 10,
        "size_range": (0.26, 0.76),
        "lifetime_range": (9.0, 19.0),
        "region_limit": 18,
        "intensity": 1.0,
    },
}

# Alter Name als Kompatibilität für bestehende Importe.
BEYOND_PROFILE = BEYOND_PROFILES["V"]
