"""
plot_figures.py -- regenerate the paper's three figures from released raw results.

Self-contained (pandas + matplotlib only). Reads ../results/*.csv and writes
../figures/{fig1_pareto,fig2_baselines,fig3_tnow}.{pdf,png}.

    cd src && python plot_figures.py

Inputs (see ../results/):
  pareto.csv     w4,n,dev_mean,dev_ci,ot_mean,ot_ci
  baselines.csv  method,w4,dev_mean,dev_ci,ot_mean,ot_ci   (method in A1/A2/VNS/ALNS/cold)
  tnow.csv       t_now,n_disrupt,A1_inert_pct,A2_cost_mean,A2_cost_ci,
                 A2_devdrop_mean,A2_devdrop_ci,A2_dev_monotone
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 12.5, "axes.titlesize": 11.5, "axes.labelsize": 12.5,
                     "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10})

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")


def _save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), dpi=150)
    plt.close(fig)
    print(f"wrote {name}.pdf/.png")


def fig1_pareto():
    """A2 (displacement) stability-efficiency Pareto frontier with 95% CI."""
    df = pd.read_csv(os.path.join(RESULTS, "pareto.csv")).sort_values("w4")
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.errorbar(df.ot_mean, df.dev_mean, xerr=df.ot_ci, yerr=df.dev_ci,
                fmt="o-", color="#1f4e79", lw=2, ms=6, capsize=4, ecolor="#888", zorder=3)
    # per-w4 label offsets to de-clutter the tight elbow (w4=5,10,20 cluster);
    # push the rightmost label (w4=100) leftwards so it stays on-canvas.
    off = {0: (10, 2, "left"), 5: (-6, 14, "right"), 10: (8, -16, "left"),
           20: (12, 2, "left"), 50: (9, 4, "left"), 100: (-10, 10, "right")}
    for _, r in df.iterrows():
        dx, dy, ha = off.get(int(r.w4), (8, 4, "left"))
        ax.annotate(f"$w_4$={int(r.w4)}", (r.ot_mean, r.dev_mean),
                    textcoords="offset points", xytext=(dx, dy), fontsize=10, ha=ha)
    ax.margins(x=0.12, y=0.12)
    ax.set_xlabel("Total overtime (min)  ->  efficiency cost")
    ax.set_ylabel("Total deviation  ->  instability")
    n = int(df.n.iloc[0]) if "n" in df else 32
    ax.set_title(f"A2 displacement: stability-efficiency Pareto with 95% CI\n"
                 f"({n} disruptions over 6 instances)")
    ax.grid(alpha=.3)
    fig.tight_layout()
    _save(fig, "fig1_pareto")


def fig2_baselines():
    """Baselines (A1/cold/VNS/ALNS) vs exact A2 in the stability-efficiency plane."""
    summ = pd.read_csv(os.path.join(RESULTS, "baselines.csv"))
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    styles = {"A2": ("#1f4e79", "o-", "A2 displacement (exact)"),
              "VNS": ("#2ca02c", "s--", "VNS"),
              "ALNS": ("#9467bd", "^--", "ALNS"),
              "A1": ("#d62728", "D-", "A1 assignment-only (exact)")}
    for m, (col, st, lab) in styles.items():
        s = summ[summ.method == m].sort_values("w4")
        if len(s):
            ax.errorbar(s.ot_mean, s.dev_mean, xerr=s.ot_ci, yerr=s.dev_ci, fmt=st,
                        color=col, lw=1.8, ms=6, capsize=3, label=lab, alpha=.9)
    cs = summ[summ.method == "cold"]
    if len(cs):
        c = cs.iloc[0]
        ax.scatter([c.ot_mean], [c.dev_mean], marker="*", s=220, color="#ff7f0e",
                   zorder=5, label="cold-start (efficiency-only)")
    ax.set_xlabel("Total overtime (min)  ->  efficiency cost")
    ax.set_ylabel("Total deviation  ->  instability")
    ax.set_title("Baselines vs exact A1/A2 (mean +/- 95% CI, 32 disruptions)")
    ax.legend(fontsize=9.5)
    ax.grid(alpha=.3)
    fig.tight_layout()
    _save(fig, "fig2_baselines")


def fig3_tnow():
    """Generality across disruption epochs t_now: cost-of-stability + A1 inertness."""
    res = pd.read_csv(os.path.join(RESULTS, "tnow.csv")).sort_values("t_now")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.5))
    ax1.errorbar(res.t_now, res.A2_cost_mean, yerr=res.A2_cost_ci, fmt="o-",
                 color="#1f4e79", lw=2, ms=8, capsize=4)
    ax1.set_xlabel("t_now (disruption epoch)")
    ax1.set_ylabel("A2 overtime cost of stability (min)\nmean +/- 95% CI")
    ax1.set_title("Cost of stability vs. epoch")
    ax1.grid(alpha=.3)
    ax2.bar(res.t_now.astype(str), res.A1_inert_pct, color="#d62728", width=0.6)
    for x, v in zip(res.t_now.astype(str), res.A1_inert_pct):
        ax2.annotate(f"{int(v)}%", (x, v), ha="center", va="bottom", fontsize=9.5)
    ax2.set_xlabel("t_now")
    ax2.set_ylabel("A1: % disruptions with w4 fully inert")
    ax2.set_ylim(0, 105)
    ax2.set_title("A1 inertness vs. epoch")
    ax2.grid(alpha=.3, axis="y")
    fig.suptitle("Generality across disruption epochs (t_now)", fontsize=13)
    fig.tight_layout()
    _save(fig, "fig3_tnow")


if __name__ == "__main__":
    fig1_pareto()
    fig2_baselines()
    fig3_tnow()
    print("done -> figures/")
