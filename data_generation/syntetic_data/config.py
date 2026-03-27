ACTIVITY_PROFILES = {
    "sleeping": {
        "accel":  {"cadence_hz": 0.0,  "amplitude": 0.02, "gravity_val": 9.81, "noise_std": 0.01},
        "gyro":   {"amplitude": 0.01, "noise_std": 0.005},
        "hr":     {"mean": 55,  "std": 3},
        "spo2":   {"mean": 97.5, "std": 0.3},
        "temp":   {"mean": 34.5, "std": 0.1},
        "breathing_rate": 14,
    },
    "walking": {
        "accel":  {"cadence_hz": 1.9,  "amplitude": 0.6,  "gravity_val": 9.81, "noise_std": 0.08},
        "gyro":   {"amplitude": 0.4,  "noise_std": 0.03},
        "hr":     {"mean": 95,  "std": 5},
        "spo2":   {"mean": 98.0, "std": 0.2},
        "temp":   {"mean": 35.2, "std": 0.15},
        "breathing_rate": 18,
    },
    "running": {
        "accel":  {"cadence_hz": 2.8,  "amplitude": 2.0,  "gravity_val": 9.81, "noise_std": 0.2},
        "gyro":   {"amplitude": 1.5,  "noise_std": 0.1},
        "hr":     {"mean": 155, "std": 8},
        "spo2":   {"mean": 97.0, "std": 0.4},
        "temp":   {"mean": 36.8, "std": 0.2},
        "breathing_rate": 35,
    },
}

LABEL_MAP = {
    "normal":       0,
    "pre_syncope":  1,
    "syncope":      2,
    "recovery":     3,
}

FS = 50  # Hz

OUTPUT_PATH = "./data"