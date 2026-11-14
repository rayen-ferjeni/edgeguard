import pandas as pd
import matplotlib.pyplot as plt

COLS = ["duty", "current", "voltage", "temp_motor", "vib_rms", "gas"]


def segments(df):
    """Retourne [(debut, fin_exclue, scenario), ...] pour chaque bloc label == 1."""
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


def plot_dataset(csv_path, out_png):
    df = pd.read_csv(csv_path)
    segs = segments(df)
    fig, axes = plt.subplots(len(COLS), 1, sharex=True, figsize=(14, 12))
    for ax, col in zip(axes, COLS):
        ax.plot(df["t"], df[col], lw=0.6)
        ax.set_ylabel(col)
        for s, e, name in segs:
            ax.axvspan(df["t"].iloc[s], df["t"].iloc[e - 1], color="red", alpha=0.2)
    for s, e, name in segs:
        axes[0].text(df["t"].iloc[s], axes[0].get_ylim()[1], name,
                     fontsize=8, rotation=45, va="bottom")
    axes[-1].set_xlabel("t (s)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    print("écrit :", out_png)


if __name__ == "__main__":
    plot_dataset("data/test_mixed.csv", "../docs/test_mixed.png")