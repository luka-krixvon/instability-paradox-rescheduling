"""
plot_e4.py — figures from the E4 run (reads e4_raw.csv / e4_summary.csv).
  pareto_feasible.pdf : fair A1 vs A2 in the (deviation, overtime) plane at a tight rho,
                        annotated with unserved (A1 and A2 now aligned => fair comparison).
  phase_rho.pdf       : E3 preview — A2 overtime cost-of-stability vs rho, and A1 inert% vs rho.
    cd Code && python journal_p0/plot_e4.py
"""
import argparse
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle; figstyle.apply()

HERE = os.path.dirname(os.path.abspath(__file__))
W4 = [0, 5, 10, 20, 50, 100]


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
    df = pd.read_csv(os.path.join(args.input_dir, "e4_raw.csv"))
    summ = pd.read_csv(os.path.join(args.input_dir, "e4_summary.csv"))
    needed_raw = {"rho", "seed", "absent", "method", "w4", "deviation", "overtime",
                  "unserved", "proven_optimal"}
    needed_summary = {"rho", "A2_cost_of_stability_min", "A2_cos_ci95",
                      "A1_fully_inert_pct"}
    missing = sorted(needed_raw - set(df.columns))
    missing_summary = sorted(needed_summary - set(summ.columns))
    if missing or missing_summary:
        raise ValueError(
            f"E4 schema mismatch: raw_missing={missing}, summary_missing={missing_summary}"
        )
    if df.empty or not _strict_true(df["proven_optimal"], "proven_optimal").all():
        raise ValueError("plots require a non-empty, proven-optimal R1b artifact")
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Fig 1: fair Pareto at rho=0.85 (aligned with Table tab:pareto in the paper) ----
    rho_t = 0.85 if (df.rho == 0.85).any() else df.rho.max()
    sub = df[df.rho == rho_t]
    a2 = sub[sub.method == "A2"].groupby("w4").agg(dev=("deviation", "mean"), ot=("overtime", "mean")).reindex(W4)
    a1_dev = sub[sub.method == "A1"].deviation.mean(); a1_ot = sub[sub.method == "A1"].overtime.mean()
    a1_u = sub[sub.method == "A1"].unserved.mean(); a2_u = sub[sub.method == "A2"].unserved.mean()

    fig, ax = plt.subplots(figsize=(figstyle.COL_W, 2.9))
    ax.plot(a2["dev"], a2["ot"], "o-", color="#1f77b4", label="A2 (displacement)")
    # label only the endpoints; the mid-range w4 points cluster and would overlap
    for w in (0, 100):
        if w in a2.index and a2.loc[w].notna().all():
            r = a2.loc[w]
            ax.annotate(f"$w_4{{=}}{w}$", (r["dev"], r["ot"]), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
    ax.scatter([a1_dev], [a1_ot], color="#d62728", s=70, marker="s", zorder=5,
               label="A1 (assignment-only)")
    ax.set_yscale("log")                         # A1 (~1800) and the A2 frontier (~50-150) both visible
    ax.set_xlabel("deviation"); ax.set_ylabel("overtime (min, log scale)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "pareto_feasible.pdf"))
    plt.close(fig)

    # ---- Fig 2: E3 preview — cost-of-stability & A1 inert% vs rho ----
    fig, ax1 = plt.subplots(figsize=(figstyle.COL_W, 2.9))
    ax1.errorbar(summ.rho, summ.A2_cost_of_stability_min, yerr=summ.A2_cos_ci95,
                 fmt="o-", color="#1f77b4", capsize=3, label="A2 cost of stability")
    ax1.set_xlabel(r"capacity tightness $\rho$")
    ax1.set_ylabel("A2 cost of stability (min)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(summ.rho, summ.A1_fully_inert_pct, "s--", color="#d62728", label="A1 fully inert %")
    ax2.set_ylabel("A1 fully $w_4$-inert (%)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728"); ax2.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "phase_rho.pdf"))
    plt.close(fig)

    print(f"wrote {args.out_dir}/pareto_feasible.pdf, {args.out_dir}/phase_rho.pdf")
    print(f"tight rho={rho_t}: A1=({a1_dev:.0f},{a1_ot:.0f}) unserved {a1_u:.1f} | "
          f"A2 dev {a2['dev'].iloc[0]:.0f}->{a2['dev'].iloc[-1]:.0f}, ot {a2['ot'].iloc[0]:.0f}->{a2['ot'].iloc[-1]:.0f}, unserved {a2_u:.1f}")


if __name__ == "__main__":
    main()
