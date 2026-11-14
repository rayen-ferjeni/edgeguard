import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
 
MIN_FRAC = 0.2   # une anomalie est "détectée" si >= 20 % de ses lignes sont en alerte
SUSTAIN = 5      # délai = première fois où 5 alertes consécutives apparaissent
GUARD = 40       # s après la fin d'une anomalie : la fenêtre glissante "traîne", on ne compte pas ces lignes comme fausses alertes
 
 
def segments(df):
    """[(debut, fin_exclue, scenario), ...] pour chaque bloc label == 1."""
    lab = df["label"].to_numpy()
    segs, start = [], None
    for i, v in enumerate(lab):
        if v == 1 and start is None:
            start = i
        elif v == 0 and start is not None:
            segs.append((start, i, df["scenario"].iloc[start]))
            start = None
    if start is not None:
        segs.append((start, len(lab), df["scenario"].iloc[start]))
    return segs
 
 
def first_sustained(seg, k=SUSTAIN):
    """Indice de la première série de k alertes consécutives, ou None."""
    run = 0
    for i, v in enumerate(seg):
        run = run + 1 if v else 0
        if run == k:
            return i - k + 1
    return None
 
 
def evaluate(df, alert, min_frac=MIN_FRAC, guard=GUARD):
    """df : jeu de test (colonnes label, scenario). alert : 0/1 par ligne.
    Retourne (résumé : dict, tableau par scénario : DataFrame)."""
    y = df["label"].to_numpy()
    alert = np.asarray(alert).astype(int)
    near = np.zeros(len(y), dtype=bool)          # lignes normales juste après une anomalie
    for s, e, _ in segments(df):
        near[e:e + guard] = True
    clean = (y == 0) & ~near
    normal_hours = clean.sum() / 3600
    rows = []
    for s, e, name in segments(df):
        seg = alert[s:e]
        frac = seg.mean()
        rows.append({
            "scenario": name,
            "durée_s": e - s,
            "détecté": bool(frac >= min_frac),
            "délai_s": first_sustained(seg),
            "% lignes en alerte": round(100 * frac, 1),
        })
    table = pd.DataFrame(rows)
    summary = {
        "précision": precision_score(y, alert, zero_division=0),
        "rappel": recall_score(y, alert, zero_division=0),
        "F1": f1_score(y, alert, zero_division=0),
        "fausses_alertes_par_h": ((alert == 1) & clean).sum() / normal_hours,
        "anomalies_détectées": f"{int(table['détecté'].sum())}/{len(table)}",
    }
    return summary, table
 