import numpy as np
from scipy.signal import butter, filtfilt, lfilter

from config import FS


# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────

def lowpass(sig, cutoff=10.0, fs=FS, order=4):
    nyq = fs / 2.0
    if cutoff >= nyq:
        cutoff = nyq * 0.99
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, sig)


def _thermal_lag(sig, tau_s: float, fs: int = FS) -> np.ndarray:
    """
    Simulate physiological thermal lag with a first-order IIR low-pass.
    tau_s is the time constant in seconds (~120 s for skin during exercise).

    Pre-warmed via lfilter_zi so output starts at the correct temperature
    instead of ramping up from 0°C.
    """
    from scipy.signal import lfilter_zi
    alpha = 1.0 - np.exp(-1.0 / (tau_s * fs))
    b = np.array([alpha])
    a = np.array([1.0, -(1.0 - alpha)])
    zi = lfilter_zi(b, a) * sig[0]   # initialise at first sample value
    out, _ = lfilter(b, a, sig, zi=zi)
    return out


def add_artifacts(sig, dropout=0.003, spikes=0.002, rng=None):
    """Generic dropout + spike artifacts (sensor hardware faults)."""
    if rng is None:
        rng = np.random.default_rng()
    s = sig.copy()
    s[rng.random(len(s)) < dropout] = np.nan
    mask = rng.random(len(s)) < spikes
    s[mask] += rng.normal(0, float(np.nanstd(sig)) * 5, int(mask.sum()))
    return s


def add_motion_artifacts(hr, spo2, ax, ay, az, rng, fs=FS):
    """
    Scale HR and SpO₂ noise by movement intensity (PPG motion-artifact model).

    Real optical sensors are highly sensitive to limb movement. We remove the
    gravity component and use the residual dynamic acceleration as a proxy for
    optical path disruption.

    Noise mapping (empirically motivated):
        0   m/s²  dynamic  → very low noise  (scale 0.10)
        2   m/s²  dynamic  → moderate noise  (scale 1.50)
        10+ m/s²  dynamic  → heavy artifact  (scale 4.00)
    """
    # Fill NaN before computing magnitude — NaN dropouts in accel should not
    # corrupt the motion envelope (treat them as zero dynamic acceleration).
    ax_c = np.nan_to_num(ax, nan=0.0)
    ay_c = np.nan_to_num(ay, nan=0.0)
    az_c = np.nan_to_num(az, nan=9.81)  # gravity direction

    accel_mag  = np.sqrt(ax_c**2 + ay_c**2 + az_c**2)
    motion_raw = np.abs(accel_mag - 9.81)              # remove ~gravity
    motion_env = lowpass(motion_raw, cutoff=2.0, fs=fs) # smooth envelope

    noise_scale = np.interp(motion_env, [0, 2, 10], [0.10, 1.50, 4.00])

    hr_out   = hr   + rng.normal(0, 1.0,  len(hr))   * noise_scale
    # SpO2 dips slightly under heavy motion (optical interference)
    spo2_out = spo2 - np.abs(rng.normal(0, 0.15, len(spo2))) * (noise_scale / 4.0)

    return hr_out, np.clip(spo2_out, 90.0, 100.0)


# ─────────────────────────────────────────────────────────────
# SENSOR GENERATORS
# ─────────────────────────────────────────────────────────────

def sim_accel(t, activity, profile, rng, cadence_bias=0.0):
    """
    Accelerometer.

    Improvement over original: the walking/running waveform now includes a
    rectified harmonic that produces a sharp asymmetric heel-strike peak,
    matching real wrist-worn accelerometer recordings.  Cadence can be
    offset per user via cadence_bias.
    """
    p  = profile["accel"]
    n  = len(t)
    br = profile["breathing_rate"] / 60.0

    if activity == "sleeping":
        x = 0.01 * np.sin(2*np.pi*br*t) + rng.normal(0, p["noise_std"], n)
        y = 0.01 * np.cos(2*np.pi*br*t) + rng.normal(0, p["noise_std"], n)
        z = p["gravity_val"] + 0.015*np.sin(2*np.pi*br*t) + rng.normal(0, p["noise_std"], n)
    else:
        f = p["cadence_hz"] + cadence_bias
        A = p["amplitude"]

        # Primary sine + asymmetric heel-strike transient (|sin|) + 3rd harmonic
        x = (A       * np.sin(2*np.pi*f*t)
             + 0.4*A * np.abs(np.sin(np.pi*f*t))    # ← sharp heel-strike peak
             + 0.1*A * np.sin(2*np.pi*3*f*t + 0.6)
             + rng.normal(0, p["noise_std"], n))
        y = (0.4*A   * np.sin(2*np.pi*f*t + np.pi/3)
             + 0.15*A * np.sin(2*np.pi*2*f*t + 0.5)
             + rng.normal(0, p["noise_std"], n))
        z = (p["gravity_val"]
             + 0.5*A * np.sin(2*np.pi*f*t + np.pi/2)
             + 0.02  * np.sin(2*np.pi*br*t)
             + rng.normal(0, p["noise_std"], n))

    return lowpass(x, 20), lowpass(y, 20), lowpass(z, 20)


def sim_gyro(t, activity, profile, rng, cadence_bias=0.0):
    """
    Gyroscope with integration drift.

    Real MEMS gyros have a slow-wandering bias caused by temperature and
    mechanical stress.  We model this as a cumulative random walk on all axes.
    """
    p = profile["gyro"]
    n = len(t)

    if activity == "sleeping":
        gx = rng.normal(0, p["noise_std"], n)
        gy = rng.normal(0, p["noise_std"], n)
        gz = rng.normal(0, p["noise_std"], n)
    else:
        f  = profile["accel"]["cadence_hz"] + cadence_bias
        A  = p["amplitude"]
        gx = lowpass(A     * np.cos(2*np.pi*f*t)       + rng.normal(0, p["noise_std"], n), 15)
        gy = lowpass(0.5*A * np.cos(2*np.pi*f*t + 0.4) + rng.normal(0, p["noise_std"], n), 15)
        gz = lowpass(0.3*A * np.sin(2*np.pi*f*t + 0.8) + rng.normal(0, p["noise_std"], n), 15)

    # Slow integration drift (random walk) — magnitude calibrated to typical
    # consumer MEMS gyros (~0.05 °/s/√Hz bias instability)
    drift_x = np.cumsum(rng.normal(0, 0.0008, n))
    drift_y = np.cumsum(rng.normal(0, 0.0008, n))
    drift_z = np.cumsum(rng.normal(0, 0.0005, n))

    return gx + drift_x, gy + drift_y, gz + drift_z


def sim_hr(t, profile, rng, fitness=1.0):
    """
    Heart rate with realistic HRV via RR-interval modelling.

    Steps:
      1. Derive mean HR from activity profile, scaled by user fitness.
      2. Generate beat-to-beat RR intervals from a log-normal distribution
         whose sigma decreases with HR (sympathetic dominance at high HR).
      3. Interpolate instantaneous HR back to the sample grid.
      4. Add Respiratory Sinus Arrhythmia (RSA), Mayer waves, and slow drift.

    This produces a signal whose spectral HF/LF power ratio changes with
    activity — a key realism marker that the original smooth model lacked.
    """
    p        = profile["hr"]
    duration = t[-1]
    br       = profile["breathing_rate"] / 60.0

    # Fitness scales the exercise-induced elevation above a resting baseline
    resting = 60.0
    hr_mean = resting + (p["mean"] - resting) * fitness

    # ── RR interval generation ────────────────────────────
    rr_mean = 60.0 / hr_mean
    # HRV sigma: ~6% at rest, ~2% at peak exercise (sympathetic suppression of HRV)
    hrv_sigma = float(np.interp(hr_mean, [50, 90, 160], [0.06, 0.04, 0.02]))

    n_beats = int(duration / rr_mean) + 20
    rr      = rng.lognormal(mean=np.log(rr_mean), sigma=hrv_sigma, size=n_beats)
    rr      = np.clip(rr, rr_mean * 0.4, rr_mean * 2.0)

    beat_t  = np.cumsum(rr)
    beat_t  = beat_t[beat_t <= duration + rr_mean]
    beat_hr = 60.0 / rr[:len(beat_t)]

    # ── Interpolate to sample grid ────────────────────────
    hr = np.interp(t, beat_t, beat_hr)

    # ── Physiological modulations ─────────────────────────
    rsa   = 2.5 * np.sin(2*np.pi*br*t)       # respiratory sinus arrhythmia
    mayer = 3.0 * np.sin(2*np.pi*0.10*t)      # Mayer waves (~0.10 Hz sympathetic)
    slow  = 3.0 * np.sin(2*np.pi*0.005*t)     # very-low-frequency thermoregulatory drift

    hr = hr + rsa + mayer + slow
    return np.clip(hr, p["mean"] - 25, p["mean"] + 25)


def sim_spo2(t, profile, rng):
    """
    SpO₂ with breathing-correlated oscillation and slow wander.

    Real pulse-oximetry shows a ~0.5–1% peak-to-peak oscillation at the
    breathing frequency (venous pulsation modulates the optical path).
    The original model had only white noise, which looked unrealistically flat.
    """
    p  = profile["spo2"]
    br = profile["breathing_rate"] / 60.0
    n  = len(t)

    # Breathing oscillation — inverted phase: SpO₂ dips slightly at peak inspiration
    breathing_osc = 0.5 * np.sin(2*np.pi*br*t + np.pi)
    slow_wander   = rng.normal(0, p["std"] * 0.3, n)
    noise         = rng.normal(0, p["std"] * 0.5, n)

    base = p["mean"] + breathing_osc + slow_wander + noise
    return np.clip(lowpass(base, 0.5), 90.0, 100.0)


def sim_temp(t, profile, rng, activity="walking"):
    """
    Skin temperature with physiologically realistic thermal lag.

    Skin temperature lags metabolic state by ~2–5 minutes because heat must
    conduct through tissue before reaching the sensor.  We model this with a
    first-order IIR filter (time constant = 120 s during exercise, 300 s during
    sleep when perfusion is reduced).
    """
    p = profile["temp"]
    n = len(t)

    # Instantaneous "target" temperature (what a perfect deep-tissue sensor would see)
    target = (p["mean"]
              + 0.1 * np.sin(2*np.pi*0.003*t)
              + rng.normal(0, p["std"] * 0.5, n))
    target = lowpass(target, 0.02)

    tau_s = 120.0 if activity != "sleeping" else 300.0
    return _thermal_lag(target, tau_s)