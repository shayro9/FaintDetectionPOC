from dataclasses import dataclass
import numpy as np


@dataclass
class UserProfile:
    user_id:       int
    hr_bias:       float   # bpm offset from population mean
    temp_bias:     float   # °C offset
    spo2_bias:     float   # % offset (slightly negative-skewed population)
    cadence_bias:  float   # Hz offset on gait cadence
    fitness:       float   # 0.6–1.4; scales HR elevation during exercise


# Module-level cache so the same user_id always gets the same profile within a run
_profile_cache: dict[int, UserProfile] = {}


def get_user_profile(user_id: int, rng: np.random.Generator) -> UserProfile:
    """
    Return a stable UserProfile for this user_id.
    The profile is generated once per run and cached, so every session for the
    same user shares the same inter-individual biases.
    """
    if user_id not in _profile_cache:
        _profile_cache[user_id] = UserProfile(
            user_id      = user_id,
            hr_bias      = float(rng.normal(0.0,  8.0)),
            temp_bias    = float(rng.normal(0.0,  0.4)),
            spo2_bias    = float(rng.normal(-0.3, 0.8)),   # slight negative skew
            cadence_bias = float(rng.normal(0.0,  0.25)),
            fitness      = float(rng.uniform(0.6, 1.4)),
        )
    return _profile_cache[user_id]


def clear_profile_cache() -> None:
    """Call at the start of each dataset build to avoid stale state."""
    _profile_cache.clear()
