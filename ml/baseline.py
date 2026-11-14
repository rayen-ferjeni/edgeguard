import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from plots import segments

SENSORS = ["current", "voltage", "temp_motor", "vib_rms", "gas"]
K = 3  # nombre d'écarts-types

train = pd.read_csv("data/normal_train.csv")
test = pd.read_csv("data/test_mixed.csv")

# Seuils calculés sur le TRAIN uniquement
mu = train[SENSORS].mean()
sigma = train[SENSORS].std()
low, high = mu - K * sigma, mu + K * sigma

# Une ligne est en alerte si au moins un capteur sort de ses seuils
out_of_range = (test[SENSORS] < low) | (test[SENSORS] > high)
alert = out_of_range.any(axis=1).to_numpy().astype(int)
y = test["label"].to_numpy()

print("Seuils :")
print(pd.DataFrame({"low": low, "high": high}).round(3), "\n")

print(f"précision : {precision_score(y, alert, zero_division=0):.3f}")
print(f"rappel    : {recall_score(y, alert):.3f}")
print(f"F1        : {f1_score(y, alert):.3f}")

normal_hours = (y == 0).sum() / 3600
false_alarms = ((alert == 1) & (y == 0)).sum()
print(f"fausses alertes / heure : {false_alarms / normal_hours:.2f}\n")

# Tableau par scénario : détecté ou non, délai en secondes
rows = []
for s, e, name in segments(test):
    hits = np.where(alert[s:e] == 1)[0]
    rows.append({
        "scenario": name,
        "durée_s": e - s,
        "détecté": len(hits) > 0,
        "délai_s": int(hits[0]) if len(hits) else None,
        "% lignes en alerte": round(100 * alert[s:e].mean(), 1),
    })
print(pd.DataFrame(rows).to_string(index=False))
