import argparse
import numpy as np
import pandas as pd

CFG = {
    "voltage_nom": 12.0,
    "current_idle": 0.3,
    "current_max": 2.0,
    "temp_ambient": 25.0,
    "tau_thermal": 300.0,
    "temp_rise_max": 35.0,
    "vib_base": 0.05,
    "vib_gain": 0.25,
    "gas_base": 120.0,
    "noise": {"current": 0.03, "voltage": 0.05, "temp": 0.15, "vib": 0.01, "gas": 3.0},
}

SCENARIOS = ["overheat", "imbalance", "overload", "gas_leak", "sensor_fault", "drift"]


def make_duty(n, rng):
    """Duty par paliers: valeur parmi [0, 30, 50, 70, 100], tenue 60 à 300 s."""
    levels = [0, 30, 50, 70, 100]
    duty = np.zeros(n)
    i = 0
    while i < n:
        d = int(rng.integers(60, 301))
        duty[i:i + d] = rng.choice(levels)
        i += d
    return duty


def simulate_normal(n, seed):
    rng = np.random.default_rng(seed)
    c, nz = CFG, CFG["noise"]
    t = np.arange(n)
    duty = make_duty(n, rng)

    current_clean = c["current_idle"] + (c["current_max"] - c["current_idle"]) * duty / 100
    current = current_clean + rng.normal(0, nz["current"], n)
    voltage = c["voltage_nom"] - 0.3 * current_clean + rng.normal(0, nz["voltage"], n)

    temp_ambient = c["temp_ambient"] + 2 * np.sin(2 * np.pi * t / 3600) + rng.normal(0, 0.05, n)

    T = np.empty(n)
    T[0] = temp_ambient[0]
    for i in range(1, n):
        target = temp_ambient[i] + c["temp_rise_max"] * duty[i] / 100
        T[i] = T[i - 1] + (target - T[i - 1]) / c["tau_thermal"]
    temp_motor = T + rng.normal(0, nz["temp"], n)

    vib_rms = np.clip(c["vib_base"] + c["vib_gain"] * duty / 100 + rng.normal(0, nz["vib"], n), 0, None)
    gas = np.clip(c["gas_base"] + rng.normal(0, nz["gas"], n), 0, None)
    phase = rng.uniform(0, 2 * np.pi)
    humidity = np.clip(50 + 5 * np.sin(2 * np.pi * t / 7200 + phase) + rng.normal(0, 0.3, n), 0, 100)

    return pd.DataFrame({
        "t": t, "duty": duty, "current": current, "voltage": voltage,
        "temp_motor": temp_motor, "temp_ambient": temp_ambient,
        "humidity": humidity, "vib_rms": vib_rms, "gas": gas,
        "label": 0, "scenario": "normal",
    })


def inject(df, scenario, start, length, rng):
    """Modifie df en place sur [start, start+length) et met label=1."""
    s, e = start, start + length
    if e > len(df):
        raise ValueError("anomalie hors du jeu de données")
    ramp = np.arange(length) / length  # 0 -> 1
    seg = df.index[s:e]

    if scenario == "overheat":
        df.loc[seg, "temp_motor"] += 15 * ramp
    elif scenario == "imbalance":
        df.loc[seg, "vib_rms"] *= 1 + 0.8 * ramp
    elif scenario == "overload":
        df.loc[seg, "current"] += 0.25
    elif scenario == "gas_leak":
        df.loc[seg, "gas"] += 150 * ramp
    elif scenario == "sensor_fault":
        col = rng.choice(["temp_motor", "current", "vib_rms"])
        if rng.random() < 0.5:   # flatline
            df.loc[seg, col] = df.loc[seg[0], col]
        else:                    # pics aléatoires
            mask = rng.random(length) < 0.1
            df.loc[seg, col] += mask * rng.normal(0, 3 * df[col].std(), length)
    elif scenario == "drift":
        col = rng.choice(["temp_motor", "current"])
        df.loc[seg, col] = df.loc[seg, col].values * (1 + 0.04 * ramp)

    df.loc[seg, "label"] = 1
    df.loc[seg, "scenario"] = scenario


def simulate_mixed(n, seed):
    df = simulate_normal(n, seed)
    rng = np.random.default_rng(seed + 1000)
    start = 600  # 10 min de normal au début
    order = list(rng.permutation(SCENARIOS))
    for sc in order:
        length = int(rng.integers(60, 301))
        inject(df, sc, start, length, rng)
        start += length + int(rng.integers(300, 601))  # >= 5 min entre anomalies
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--duration", type=int, default=3600)
    p.add_argument("--anomalies", type=int, default=0)
    p.add_argument("--out", default="data/out.csv")
    a = p.parse_args()

    df = simulate_mixed(a.duration, a.seed) if a.anomalies else simulate_normal(a.duration, a.seed)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} lignes écrites dans {a.out}")