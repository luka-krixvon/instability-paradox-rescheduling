"""
plot_rolling.py — the paradox compounds over a disruption stream (B1 rolling-horizon).
  rolling_horizon.pdf : final overtime under A1 vs A2 at w4=0 and w4=100, rho=0.85, 5-epoch stream.
    A1 = two nearly equal tall bars (w4 inert, ~2x A2);  A2 = a real w4 lever, far lower.
    cd Code && python3 journal_p0/plot_rolling.py
"""
import argparse
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle; figstyle.apply()
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=HERE)
    ap.add_argument("--out-dir", default=HERE)
    args = ap.parse_args(argv)
    df = pd.read_csv(os.path.join(args.input_dir, "rolling_raw.csv"))
    required = {"regime", "w4", "ot_cumulative"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"rolling_raw.csv missing required columns: {missing}")
    if df.empty:
        raise ValueError("rolling_raw.csv is empty")
    g = df.groupby(["regime", "w4"]).ot_cumulative.mean().reset_index()

    fig, ax = plt.subplots(figsize=(figstyle.COL_W, 2.9))
    regimes = ["A1", "A2"]
    w4s = [0.0, 100.0]
    x = np.arange(len(regimes)); width = 0.36
    colors = {0.0: "#9ecae1", 100.0: "#08519c"}
    for k, w in enumerate(w4s):
        vals = []
        for r in regimes:
            hit = g[(g.regime == r) & (g.w4 == w)].ot_cumulative
            if len(hit) != 1:
                raise ValueError(f"expected one aggregate for regime={r}, w4={w}")
            vals.append(hit.iloc[0])
        bars = ax.bar(x + (k - 0.5) * width, vals, width, label=f"$w_4={int(w)}$", color=colors[w])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 15, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(["A1 (assignment-only)", "A2 (displacement)"])
    ax.set_ylabel("cumulative overtime (min)")
    ax.set_ylim(0, g.ot_cumulative.max() * 1.15)   # headroom for the value labels
    ax.legend(title="deviation weight", fontsize=9)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "rolling_horizon.pdf")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
