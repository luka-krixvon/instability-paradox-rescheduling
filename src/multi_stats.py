"""
multi_stats.py — 多實例 × 多擾動統計：把 Table 1(A1 無效) 與 Pareto(A2) 從單點變成 mean±95%CI

設計：生成 G 個隨機緊實例(make_tight_instance --seed)，對每個實例的「每個活躍未來護理師缺席」
當作一個擾動樣本；對 A1(assignment-only,固定既有指派) 與 A2(允許位移) 兩 regime 各掃 w4，逐時段求解彙總。
統計：
  - A1：在多少比例樣本中 w4 完全無效(deviation 與 overtime 跨 w4 range=0)；deviation/overtime 的 mean±std。
  - A2：每個 w4 的 deviation/overtime 的 mean±95%CI(→Pareto 誤差棒)；穩定加班代價 mean±95%CI。
無需 API。MILP 多但每個極小，建議背景執行。
    cd Code && python multi_stats.py [--instances 6] [--max-absent 6]
"""
from __future__ import annotations
import os, sys, subprocess, argparse, math
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

from SubgroupOptimizer import HHCPFullParser
from displacement_milp import solve_slot, load_static, T_NOW

HERE = os.path.dirname(os.path.abspath(__file__)); MI = os.path.join(HERE, "_mi"); os.makedirs(MI, exist_ok=True)
PY = sys.executable
W4 = [0, 5, 10, 20, 50, 100]
TARGET_N = 16


def ci95(a):
    a = np.asarray(a, float)
    if len(a) < 2: return 0.0
    return 1.96 * a.std(ddof=1) / math.sqrt(len(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=6)
    ap.add_argument("--max-absent", type=int, default=6)
    args = ap.parse_args()

    recs = []
    for g in range(1, args.instances + 1):
        H = os.path.join(MI, f"H{g}.csv"); S = os.path.join(MI, f"S{g}.csv")
        subprocess.run([PY, "make_tight_instance.py", "--target-nurses", str(TARGET_N),
                        "--pack", "tight", "--seed", str(g), "--out-hhcp", H, "--out-sol", S],
                       cwd=HERE, capture_output=True, text=True)
        if not os.path.exists(S):
            print(f"instance g={g} gen failed"); continue
        parser = HHCPFullParser(H); Sset = set(parser.nurses.keys())
        slot_assign = load_static(S)
        active = sorted({n for q, m in slot_assign.items() if q >= T_NOW for n in m.values()})
        active = active[:args.max_absent]
        print(f"instance g={g}: {len(Sset)} nurses, {len(active)} absent-disruption samples {active}")
        for a in active:
            for regime in ["A1", "A2"]:
                for w4 in W4:
                    dev = ot = 0.0
                    for q in sorted(slot_assign):
                        if q < T_NOW: continue
                        out = solve_slot(parser, q, slot_assign[q], a, float(w4), Sset, regime=regime)
                        if out: dev += out[0]; ot += out[1]
                    recs.append(dict(g=g, absent=a, regime=regime, w4=w4,
                                     deviation=int(dev), overtime=round(ot, 1)))
    df = pd.DataFrame(recs)
    df.to_csv("multi_stats_raw.csv", index=False)
    print(f"\n總樣本: {df[['g','absent']].drop_duplicates().shape[0]} 個擾動 × 2 regime × {len(W4)} w4")

    # ---- A1 無效性統計 ----
    a1 = df[df.regime == "A1"]
    grp = a1.groupby(["g", "absent"])
    inert = 0; tot = 0; a1_dev = []; a1_ot = []
    for _, gdf in grp:
        tot += 1
        dr = gdf["deviation"].max() - gdf["deviation"].min()
        orr = gdf["overtime"].max() - gdf["overtime"].min()
        if dr == 0 and orr == 0: inert += 1
        a1_dev.append(gdf["deviation"].iloc[0]); a1_ot.append(gdf["overtime"].iloc[0])
    print("\n=== A1 (assignment-only) 無效性 ===")
    print(f"  w4 完全無效(dev&ot 跨 w4 range=0)之擾動比例: {inert}/{tot} = {100*inert/max(tot,1):.0f}%")
    print(f"  deviation: mean={np.mean(a1_dev):.1f} ± {ci95(a1_dev):.1f} (95%CI)")
    print(f"  overtime : mean={np.mean(a1_ot):.1f} ± {ci95(a1_ot):.1f} (95%CI)")

    # ---- A2 Pareto mean±CI ----
    a2 = df[df.regime == "A2"]
    rows = []
    for w4 in W4:
        sub = a2[a2.w4 == w4]
        rows.append(dict(w4=w4, n=len(sub),
                         dev_mean=round(sub.deviation.mean(), 1), dev_ci=round(ci95(sub.deviation), 1),
                         ot_mean=round(sub.overtime.mean(), 1), ot_ci=round(ci95(sub.overtime), 1)))
    a2sum = pd.DataFrame(rows)
    a2sum.to_csv("multi_stats_A2_pareto.csv", index=False)
    print("\n=== A2 (displacement) Pareto: 每個 w4 的 mean±95%CI ===")
    print(a2sum.to_string(index=False))

    # 穩定加班代價(每擾動 A2: max-min overtime over w4) mean±CI
    cost = a2.groupby(["g", "absent"]).apply(
        lambda d: d.overtime.max() - d.overtime.min()).values
    dev_drop = a2.groupby(["g", "absent"]).apply(
        lambda d: d.deviation.max() - d.deviation.min()).values
    print(f"\n穩定加班代價(A2, max-min overtime over w4): mean={np.mean(cost):.1f} ± {ci95(cost):.1f} (95%CI)")
    print(f"偏差降幅(A2, max-min deviation over w4):     mean={np.mean(dev_drop):.1f} ± {ci95(dev_drop):.1f} (95%CI)")

    # ---- 圖：A2 Pareto with CI 誤差棒 ----
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.errorbar(a2sum.ot_mean, a2sum.dev_mean, xerr=a2sum.ot_ci, yerr=a2sum.dev_ci,
                fmt="o-", color="#1f4e79", lw=2, ms=7, capsize=4, ecolor="#888")
    for _, r in a2sum.iterrows():
        ax.annotate(f"w4={int(r.w4)}", (r.ot_mean, r.dev_mean),
                    textcoords="offset points", xytext=(7, 4), fontsize=8)
    ax.set_xlabel("Total overtime (min)  →  efficiency cost")
    ax.set_ylabel("Total deviation  →  instability")
    ax.set_title(f"A2 displacement: stability–efficiency Pareto with 95% CI\n"
                 f"({df[['g','absent']].drop_duplicates().shape[0]} disruptions over {args.instances} instances)")
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig("fig_multi_pareto.png", dpi=150)
    print("\n已輸出: multi_stats_raw.csv, multi_stats_A2_pareto.csv, fig_multi_pareto.png")


if __name__ == "__main__":
    main()
