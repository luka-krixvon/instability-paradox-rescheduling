"""
stats_tests.py -- significance tests reported in the paper (Results section).

Reproduces the Wilcoxon signed-rank and Friedman tests from the per-disruption
raw data (32 disruptions x 6 weights x 2 regimes).

    cd src && python stats_tests.py

Input: ../results/pareto_disruption_level.csv
       columns: g, absent, regime(A1/A2), w4, deviation, overtime
"""
import os
import pandas as pd
from scipy import stats

CSV = os.path.join(os.path.dirname(__file__), "..", "results", "pareto_disruption_level.csv")
d = pd.read_csv(CSV)
a2 = d[d.regime == "A2"]
ot = a2.pivot_table(index=["g", "absent"], columns="w4", values="overtime")
dev = a2.pivot_table(index=["g", "absent"], columns="w4", values="deviation")
print(f"n disruptions (A2) = {len(ot)}")

w1, p1 = stats.wilcoxon(ot[100] - ot[0], alternative="greater")
print(f"Wilcoxon  OT(100)-OT(0)>0     : W={w1:.0f}, p={p1:.2e}  (cost of stability)")

w2, p2 = stats.wilcoxon(dev[0] - dev[100], alternative="greater")
print(f"Wilcoxon  DEV(0)-DEV(100)>0   : W={w2:.0f}, p={p2:.2e}  (deviation reduction)")

chi, pf = stats.friedmanchisquare(*[dev[c].values for c in [0, 5, 10, 20, 50, 100]])
print(f"Friedman  deviation ~ w4(6 lv): chi2={chi:.1f}, df=5, p={pf:.2e}")

a1 = d[d.regime == "A1"].pivot_table(index=["g", "absent"], columns="w4", values="overtime")
idx = ot.index.intersection(a1.index)
w3, p3 = stats.wilcoxon(a1.loc[idx, 100] - ot.loc[idx, 100], alternative="greater")
print(f"Wilcoxon  A1_OT-A2_OT(w4=100) : W={w3:.0f}, p={p3:.2e}, n={len(idx)}")
