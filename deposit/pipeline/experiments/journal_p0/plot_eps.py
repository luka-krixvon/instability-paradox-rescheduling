"""
plot_eps.py — scheme-independence figure from the B1 eps-constraint run.
  eps_scheme_independence.pdf : eps-constraint frontier at tight rho=0.85.
    A1 = a single point (deviation constant over the feasible set => eps-constraint can't trade);
    A2 = a real frontier traced by the same eps-constraint method.
    cd Code && python journal_p0/plot_eps.py
"""
import argparse
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle; figstyle.apply()

HERE = os.path.dirname(os.path.abspath(__file__))


def _strict_true(series, label):
    values = series.astype(str).str.strip().str.lower()
    invalid = sorted(set(values) - {"true", "false"})
    if invalid:
        raise ValueError(f"{label} contains non-boolean values: {invalid}")
    return values.eq("true")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=HERE)
    ap.add_argument("--out-dir", default=HERE)
    args = ap.parse_args(argv)
    fr = pd.read_csv(os.path.join(args.input_dir, "eps_frontier.csv"))
    required = {"method", "eps", "eps_grid_lo", "eps_grid_hi", "D", "W", "U",
                "U_common", "proven_optimal"}
    missing = sorted(required - set(fr.columns))
    if missing:
        raise ValueError(f"eps_frontier.csv missing required columns: {missing}")
    if fr.empty or not _strict_true(fr["proven_optimal"], "proven_optimal").all():
        raise ValueError("plot requires a non-empty, proven-optimal R1 frontier")
    if not (fr["U"] == fr["U_common"]).all():
        raise ValueError("frontier contains a row outside U_common")
    span = fr["eps_grid_hi"] - fr["eps_grid_lo"]
    fr = fr.assign(
        eps_position=((fr["eps"] - fr["eps_grid_lo"]) /
                      span.where(span != 0, 1)).clip(0, 1)
    )
    a2 = (
        fr[fr.method == "A2"]
        .groupby("eps_position")
        .agg(dev=("D", "mean"), ot=("W", "mean"))
        .reset_index()
        .sort_values("eps_position")
    )
    a1 = fr[fr.method == "A1"]
    if a1.empty or a2.empty:
        raise ValueError("both A1 and A2 rows are required")
    a1_dev, a1_ot = a1.D.mean(), a1.W.mean()
    ranges = (
        fr.groupby(["rho", "seed", "absent", "method"])
        .agg(dmin=("D", "min"), dmax=("D", "max"))
        .assign(dev_range=lambda d: d.dmax - d.dmin)
        .reset_index()
    )
    r1 = ranges[ranges.method == "A1"].dev_range.mean()
    r2 = ranges[ranges.method == "A2"].dev_range.mean()

    fig, ax = plt.subplots(figsize=(figstyle.COL_W, 2.9))
    ax.plot(a2["dev"], a2["ot"], "o-", color="#1f77b4", label="A2 (displacement)")
    ax.scatter([a1_dev], [a1_ot], color="#d62728", s=70, marker="s", zorder=5,
               label="A1 (assignment-only)")
    ax.set_xlabel("deviation (achieved)")
    ax.set_ylabel("overtime (min)")
    ax.legend(fontsize=8, loc="upper right")
    ax.text(0.40, 0.55, f"achievable-deviation range\nA1 = {r1:.1f},  A2 = {r2:.1f}",
            transform=ax.transAxes, fontsize=8, color="#555")
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "eps_scheme_independence.pdf")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print(f"wrote {out}")
    print(f"A1: point ({a1_dev:.0f}, {a1_ot:.0f}), dev-range {r1:.1f}")
    print(f"A2: frontier dev {a2.dev.min():.0f}->{a2.dev.max():.0f}, ot {a2.ot.min():.0f}->{a2.ot.max():.0f}, dev-range {r2:.1f}")


if __name__ == "__main__":
    main()
