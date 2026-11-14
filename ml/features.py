"""Features sur fenêtres glissantes + résidus "mesure - attendu".

Tout ce qui est appris (régressions, constante de temps thermique) l'est sur le TRAIN
uniquement, puis réappliqué tel quel au test.
"""
import numpy as np
import pandas as pd

RAW = ["current", "voltage", "temp_motor", "vib_rms", "gas"]
RESID = ["current_resid", "voltage_resid", "vib_resid", "temp_resid"]


def ema(x, tau):
    """Moyenne exponentielle : modèle d'inertie du premier ordre (comme la température)."""
    a = 1.0 / tau
    out = np.empty(len(x))
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = out[i - 1] + a * (x[i] - out[i - 1])
    return out


class Residuals:
    """Apprend ce qu'on attend pour chaque capteur, sachant duty (et le courant, l'ambiante)."""

    def fit(self, train):
        d = train["duty"].to_numpy()
        self.cur = np.polyfit(d, train["current"], 1)                 # courant ~ duty
        self.vib = np.polyfit(d, train["vib_rms"], 1)                 # vibration ~ duty
        self.vol = np.polyfit(train["current"], train["voltage"], 1)  # tension ~ courant
        # Température : (T_moteur - T_ambiante) ~ a * duty lissé par une inertie tau.
        # On cherche tau qui minimise l'erreur sur le train.
        rise = (train["temp_motor"] - train["temp_ambient"]).to_numpy()
        best = None
        for tau in (60, 100, 150, 200, 300, 400, 600, 900):
            x = ema(d, tau)
            coef = np.polyfit(x, rise, 1)
            err = np.mean((rise - np.polyval(coef, x)) ** 2)
            if best is None or err < best[0]:
                best = (err, tau, coef)
        _, self.tau, self.temp = best
        return self

    def transform(self, df):
        d = df["duty"].to_numpy()
        out = pd.DataFrame(index=df.index)
        out["current_resid"] = df["current"] - np.polyval(self.cur, d)
        out["vib_resid"] = df["vib_rms"] - np.polyval(self.vib, d)
        out["voltage_resid"] = df["voltage"] - np.polyval(self.vol, df["current"])
        rise_exp = np.polyval(self.temp, ema(d, self.tau))
        out["temp_resid"] = (df["temp_motor"] - df["temp_ambient"]) - rise_exp
        return out


def window_features(sig, win=30):
    """Moyenne, écart-type et pente sur fenêtre glissante, pour chaque colonne de sig."""
    feats = {}
    for c in sig.columns:
        r = sig[c].rolling(win)
        feats[f"{c}_mean"] = r.mean()
        feats[f"{c}_std"] = r.std()
        feats[f"{c}_slope"] = (sig[c].rolling(5).mean() - sig[c].rolling(5).mean().shift(win)) / win
    return pd.DataFrame(feats)


def make_features(df, resid, use_resid=True, win=30):
    """Retourne un DataFrame de features, indexé comme df (les premières lignes sont NaN)."""
    sig = df[RAW].copy()
    if use_resid:
        sig = pd.concat([sig, resid.transform(df)], axis=1)
    return window_features(sig, win)