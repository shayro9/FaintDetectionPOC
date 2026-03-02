import numpy as np
from scipy.signal import butter, filtfilt

from config import FS

def lowpass(sig, cutoff=10.0, fs=FS, order=4):
    nyq = fs / 2.0
    if cutoff >= nyq:
        cutoff = nyq * 0.99
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, sig)


def add_artifacts(sig, dropout=0.003, spikes=0.002, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    s = sig.copy()
    s[rng.random(len(s)) < dropout] = np.nan
    mask = rng.random(len(s)) < spikes
    s[mask] += rng.normal(0, float(np.nanstd(sig)) * 5, int(mask.sum()))
    return s

# ─────────────────────────────────────────────────────────────
# SENSOR GENERATORS
# ─────────────────────────────────────────────────────────────
def sim_accel(t, activity, profile, rng):
    p  = profile["accel"]
    n  = len(t)
    br = profile["breathing_rate"] / 60.0

    if activity == "sleeping":
        x = 0.01 * np.sin(2*np.pi*br*t) + rng.normal(0, p["noise_std"], n)
        y = 0.01 * np.cos(2*np.pi*br*t) + rng.normal(0, p["noise_std"], n)
        z = p["gravity_val"] + 0.015*np.sin(2*np.pi*br*t) + rng.normal(0, p["noise_std"], n)
    else:
        f, A = p["cadence_hz"], p["amplitude"]
        x = (A       * np.sin(2*np.pi*f*t)
             + 0.3*A * np.sin(2*np.pi*2*f*t + 0.3)
             + 0.1*A * np.sin(2*np.pi*3*f*t + 0.6)
             + rng.normal(0, p["noise_std"], n))
        y = (0.4*A  * np.sin(2*np.pi*f*t + np.pi/3)
             + 0.15*A * np.sin(2*np.pi*2*f*t + 0.5)
             + rng.normal(0, p["noise_std"], n))
        z = (p["gravity_val"]
             + 0.5*A * np.sin(2*np.pi*f*t + np.pi/2)
             + 0.02  * np.sin(2*np.pi*br*t)
             + rng.normal(0, p["noise_std"], n))

    return lowpass(x, 20), lowpass(y, 20), lowpass(z, 20)


def sim_gyro(t, activity, profile, rng):
    p = profile["gyro"]
    n = len(t)
    if activity == "sleeping":
        return (rng.normal(0, p["noise_std"], n),
                rng.normal(0, p["noise_std"], n),
                rng.normal(0, p["noise_std"], n))
    f, A = profile["accel"]["cadence_hz"], p["amplitude"]
    gx = lowpass(A     * np.cos(2*np.pi*f*t)       + rng.normal(0, p["noise_std"], n), 15)
    gy = lowpass(0.5*A * np.cos(2*np.pi*f*t + 0.4) + rng.normal(0, p["noise_std"], n), 15)
    gz = lowpass(0.3*A * np.sin(2*np.pi*f*t + 0.8) + rng.normal(0, p["noise_std"], n), 15)
    return gx, gy, gz


def sim_hr(t, profile, rng):
    p  = profile["hr"]
    n  = len(t)
    br = profile["breathing_rate"] / 60.0
    slow = 3.0 * np.sin(2*np.pi*0.005*t)
    rsa  = 2.5 * np.sin(2*np.pi*br*t)
    rw   = np.cumsum(rng.normal(0, 0.05, n))
    rw   = np.clip(rw - rw.mean(), -5, 5)
    hr   = p["mean"] + slow + rsa + rw + rng.normal(0, p["std"]*0.3, n)
    return np.clip(hr, p["mean"]-20, p["mean"]+20)


def sim_spo2(t, profile, rng):
    p = profile["spo2"]
    return np.clip(lowpass(p["mean"] + rng.normal(0, p["std"], len(t)), 0.05), 90.0, 100.0)


def sim_temp(t, profile, rng):
    p = profile["temp"]
    return lowpass(p["mean"] + 0.1*np.sin(2*np.pi*0.003*t)
                   + rng.normal(0, p["std"]*0.5, len(t)), 0.02)