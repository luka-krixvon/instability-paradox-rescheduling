"""
highs_presolve_infeasible.py — minimal reproducer for a solver defect found while
building the S2 lexicographic tie-break.

FINDING.  scipy.optimize.milp / HiGHS returns `status = 2` ("The problem is
infeasible") for a model that has an EXPLICIT feasible point, and returns
`status = 0` for the identical model with `options={"presolve": False}`.

It appears when two nearly-parallel rows both carry a small slack:

    row Z :  W2*sum(O) + w4*(dev_coef . x)  + W5*sum(u)  <=  Z*  + 1e-6 - w4*C
    row W :       sum(O)                                 <=  W*  + 1e-6

Row Z contains `W2*sum(O)`, and row W bounds `sum(O)`.  With a 1e-6 slack on both,
presolve concludes the model is infeasible.  At slack 0, 1e-9, 1e-3 and 1e-1 the
same model solves to optimality, so the effect is specific to that magnitude —
which is exactly the failure mode a "just widen the tolerance" reaction would have
hidden.

WHAT WAS DONE ABOUT IT, and what was not.  The tolerance was NOT changed to obtain
a pass.  `run_e4.py` drops row Z from the final lexicographic stage because the row
is provably redundant there (with U fixed, W <= W* and w4 >= 0, minimising D cannot
raise Z above Z*), and keeps `Z <= Z* + LEX_TOL` as a POST-HOC ASSERTION.  In
addition, every lexicographic stage now passes the previous stage's solution as
`known_feasible_x`, so an "infeasible" verdict that is refuted by an explicit
feasible point ABORTS with a diagnostic naming presolve, instead of being retried
into a pass.

This file is evidence, not a workaround.  Run it to see the two verdicts.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

_THIS = os.path.dirname(os.path.abspath(__file__))
_FIXTURES = os.path.dirname(_THIS)
_REPO = os.path.abspath(os.path.join(_FIXTURES, "..", "..", ".."))
_EXPERIMENTS = os.path.join(_REPO, "experiments")
for _p in (os.path.join(_EXPERIMENTS, "journal_p0"), _EXPERIMENTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def build():
    """The exact failing cell: fixture A, absent = nurse 0, regime A1, slot 5,
    w4 = 5, lexicographic stage 4 (min deviation)."""
    import run_e4 as R
    from SubgroupOptimizer import HHCPFullParser
    from displacement_milp import load_static
    parser = HHCPFullParser(os.path.join(_FIXTURES, "H_epse4_A.csv"))
    plan = load_static(os.path.join(_FIXTURES, "S_epse4_A.csv"))
    m = R.WeightedModel(parser, plan, set(parser.nurses), 0, "A1", [5],
                        R.solver_policy())
    w4, z_star, u_star, w_star, slack = 5.0, 1310.0, 1, 60.0, 1e-6
    rows = list(m._rows) + [m._z_row(w4), m._u_row(), m._o_row()]
    lb = list(m._lb) + [-np.inf, float(u_star), -np.inf]
    ub = list(m._ub) + [z_star + slack - w4 * m.dev_const, float(u_star),
                        w_star + slack]
    lo = np.zeros(m.n)
    hi = np.ones(m.n)
    for k in m.oidx.values():
        hi[k] = np.inf
    integ = np.zeros(m.n)
    for k in range(len(m.xpairs)):
        integ[k] = 1
    for k in m.uidx.values():
        integ[k] = 1
    cons = LinearConstraint(np.array(rows), np.array(lb), np.array(ub))
    return m, cons, integ, Bounds(lo, hi), w4, (rows, lb, ub)


def main():
    m, cons, integ, bounds, w4, (rows, lb, ub) = build()
    # an explicit feasible point, obtained from the previous lexicographic stage
    ref, _ = m._solve("W", w4=w4, z_le=1310.0, u_eq=1)
    x = ref["x"]
    A = np.array(rows)
    vals = A @ x
    worst = max(max(vals[i] - ub[i], 0.0) for i in range(len(ub)))
    print(f"explicit point x = {x.tolist()}")
    print(f"max row violation of that point = {worst:.3e}  "
          f"(0 means it is feasible for the model below)")
    out = {}
    for label, opts in (("presolve ON  (default)", {"time_limit": 30}),
                        ("presolve OFF", {"time_limit": 30, "presolve": False})):
        r = milp(c=m.dev_coef, constraints=cons, integrality=integ, bounds=bounds,
                 options=opts)
        out[label] = int(r.status)
        print(f"{label:22s} -> status={r.status}  {str(r.message)[:70]}")
    ok = out["presolve ON  (default)"] == 2 and out["presolve OFF"] == 0
    print("\nREPRODUCED" if ok else "\nNOT REPRODUCED on this build")
    print("scipy", __import__("scipy").__version__, "numpy", np.__version__)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
