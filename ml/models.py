
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from features import Residuals, make_features
from metrics import evaluate

PCT = 99.9  # seuil = percentile des scores sur le TRAIN (aucune fausse alerte voulue > 0,1 %)


class ZScore:


    def fit(self, X):
        self.mu, self.sd = X.mean(), X.std().replace(0, 1e-9)
        return self

    def score(self, X):
        return ((X - self.mu) / self.sd).abs().max(axis=1)


class IForest:
    def __init__(self, seed=0):
        self.m = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1)

    def fit(self, X):
        self.m.fit(X)
        return self

    def score(self, X):
        return pd.Series(-self.m.score_samples(X), index=X.index)  # grand = anormal


def compact(X, cols):
    """Moyenne et écart-type sur fenêtre des colonnes données (sans les pentes)."""
    keep = [c for c in X.columns if not c.endswith("_slope") and any(c.startswith(k + "_") for k in cols)]
    return X[keep]


RESID_COLS = ["current_resid", "voltage_resid", "vib_resid", "temp_resid", "gas"]
RAW_COLS = ["current", "voltage", "temp_motor", "vib_rms", "gas"]


def run(model, train_df, test_df, resid, use_resid, cols=None, pct=PCT):
    Xtr = make_features(train_df, resid, use_resid).dropna()
    Xte_full = make_features(test_df, resid, use_resid)
    if cols:  # l'Isolation Forest se dilue avec trop de features : on le limite aux utiles
        Xtr, Xte_full = compact(Xtr, cols), compact(Xte_full, cols)
    valid = Xte_full.dropna()
    model.fit(Xtr)
    thr = np.percentile(model.score(Xtr), pct)            # seuil choisi sur le train
    s = model.score(valid).reindex(test_df.index)         # lignes d'échauffement -> NaN
    alert = (s > thr).fillna(False).to_numpy().astype(int)
    return alert, s


def baseline_alert(train, test, k=3):
    cols = ["current", "voltage", "temp_motor", "vib_rms", "gas"]
    mu, sd = train[cols].mean(), train[cols].std()
    out = (test[cols] < mu - k * sd) | (test[cols] > mu + k * sd)
    return out.any(axis=1).to_numpy().astype(int)


if __name__ == "__main__":
    train = pd.read_csv("data/normal_train.csv")
    test = pd.read_csv("data/test_mixed.csv")
    resid = Residuals().fit(train)
    print(f"constante de temps thermique estimée : tau = {resid.tau} s\n")

    runs = {
        "baseline ±3σ": baseline_alert(train, test),
        "z-score (brut)": run(ZScore(), train, test, resid, False)[0],
        "z-score (+résidus)": run(ZScore(), train, test, resid, True)[0],
        "IsoForest brut (99.9)": run(IForest(), train, test, resid, False, RAW_COLS)[0],
        "IsoForest +résidus (99.9)": run(IForest(), train, test, resid, True, RESID_COLS)[0],
        "IsoForest +résidus (99)": run(IForest(), train, test, resid, True, RESID_COLS, 99)[0],
    }

    summ, tables = {}, {}
    for name, alert in runs.items():
        summ[name], tables[name] = evaluate(test, alert)

    print("=== Résumé ===")
    print(pd.DataFrame(summ).T.round(3).to_string(), "\n")

    print("=== Anomalie détectée ? (>= 20 % de lignes en alerte) ; délai en s ===")
    det = pd.DataFrame({n: t.set_index("scenario")["détecté"] for n, t in tables.items()})
    dly = pd.DataFrame({n: t.set_index("scenario")["délai_s"] for n, t in tables.items()})
    print(det.to_string(), "\n")
    print(dly.to_string())