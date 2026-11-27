"""
oracle_independent.py — the REQ-4 item 4 independent oracle for the
epsilon-constraint / common-service-level stage.

WHAT INDEPENDENCE MEANS HERE, stated precisely so it cannot be oversold
---------------------------------------------------------------------
This module shares NO constraint builder, NO objective, NO feasibility test, NO
read-back and NO optimiser with `experiments/journal_p0/eps_constraint.py`.  It
imports nothing from that module (and that module never imports this one — this
file lives under tests/ so it cannot become production runtime).  Instead of a
MILP it does complete enumeration per slot, reduces each slot to the minimum
achievable total overtime for every attainable `(unserved, deviation)` pair, and
combines the slots by a Minkowski-style convolution over that pair, exactly the
"horizon DP over per-slot achievable (D, W) sets" the plan names as an admissible
oracle.  Everything is plain Python; numpy and scipy are not used.

What IS shared, deliberately:

  * `HHCPFullParser` — the instance reader.  Re-implementing the HHCP text format
    would test my parser rather than the optimisation model, and a parser
    disagreement would be indistinguishable from a model disagreement.
  * the numeric PARAMETERS `H_STD`, `H_MAX`, `TRAVEL_PER_TASK`, `T_NOW`,
    `SLOTS_PER_DAY`.  These are the problem's constants, not helpers; hard-coding
    copies here would make the oracle silently stale if a constant ever moved.

The committed schedule is parsed by this file's own `load_plan()`, so
`displacement_milp.load_static` is NOT shared.

The MODEL SEMANTICS are of course the same in both implementations — that is the
point of a cross-check.  A1 pins every incumbent task to its incumbent nurse and
leaves orphans free among their candidates; A2 leaves every task free among its
candidates; a task may always be dropped at the cost of one unserved unit;
deviation is the symmetric difference between the realised assignment and the
still-available part of the incumbent assignment; overtime is
`max(0, load - H_STD)` per nurse per slot; `load <= H_MAX` is hard.

This oracle is exponential in the number of tasks per slot and is therefore
usable on TINY SYNTHETIC FIXTURES ONLY.  Cross-validating the real 176-disruption
family is R1 work and needs the monolithic-MILP form of the oracle instead.
"""
from __future__ import annotations

import itertools
import math
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
_EXPERIMENTS = os.path.join(_REPO, "experiments")
if _EXPERIMENTS not in sys.path:
    sys.path.insert(0, _EXPERIMENTS)

from SubgroupOptimizer import H_MAX, H_STD, HHCPFullParser        # noqa: E402
from displacement_milp import SLOTS_PER_DAY, T_NOW, TRAVEL_PER_TASK  # noqa: E402

W_ATOL = 1e-9          # the tolerance REQ-4 test 3 requires on W


def _cap_slack(cap):
    return math.ulp(float(cap)) * 8.0


# --------------------------------------------------------------------------- #
def load_plan(sol_path):
    """Own reader for the committed schedule.  No production helper involved."""
    plan = {}
    with open(sol_path, "r", encoding="utf-8") as fh:
        header = [h.strip() for h in fh.readline().rstrip("\n").split(",")]
        col = {name: header.index(name) for name in header}
        for name in ("NurseID", "TaskIDs"):
            if name not in col:
                raise AssertionError(f"oracle: column {name} missing by name")
        has_gs = "GlobalSlot" in col
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(",")
            nurse = int(float(parts[col["NurseID"]]))
            if has_gs:
                q = int(float(parts[col["GlobalSlot"]]))
            else:
                day = int(float(parts[col["Day"]]))
                shift = int(float(parts[col["Shift"]]))
                q = (day - 1) * SLOTS_PER_DAY + shift
            for tok in parts[col["TaskIDs"]].replace(" ", "").split(";"):
                if tok:
                    plan.setdefault(q, {})[int(tok)] = nurse
    return plan


def _structures(parser, plan, absent, regime):
    slots = sorted(q for q in plan if q >= T_NOW)
    out = []
    for q in slots:
        tasks = sorted(t for t in plan[q] if t in parser.tasks)
        on_duty = sorted(n for n in parser.nurses if q in parser.nurses[n].slots)
        usable = [n for n in on_duty if n != absent]
        opts, xbar, dur = {}, set(), {}
        for t in tasks:
            inc = plan[q][t]
            elig = [n for n in parser.tasks[t].candidate_nurses if n in usable]
            if regime == "A1" and inc != absent:
                opts[t] = [inc]
            else:
                opts[t] = sorted(elig)
            if inc != absent and inc in usable:
                xbar.add((inc, t))
            dur[t] = float(parser.tasks[t].duration) + float(TRAVEL_PER_TASK)
        out.append({"q": q, "tasks": tasks, "usable": usable, "opts": opts,
                    "xbar": xbar, "dur": dur})
    return out


def slot_achievable(spec):
    """Complete enumeration of one slot -> {(unserved, deviation): min overtime}."""
    tasks = spec["tasks"]
    choices = [[None] + list(spec["opts"][t]) for t in tasks]
    best, slack = {}, _cap_slack(H_MAX)
    for combo in itertools.product(*choices):
        load = {}
        chosen = set()
        u = 0
        for t, i in zip(tasks, combo):
            if i is None:
                u += 1
            else:
                load[i] = load.get(i, 0.0) + spec["dur"][t]
                chosen.add((i, t))
        if any(v > float(H_MAX) + slack for v in load.values()):
            continue
        w = 0.0
        for v in load.values():
            if v > float(H_STD):
                w += v - float(H_STD)
        d = len(chosen - spec["xbar"]) + len(spec["xbar"] - chosen)
        k = (u, d)
        if k not in best or w < best[k] - 0.0:
            best[k] = w
    if not best:
        raise AssertionError(f"oracle: slot {spec['q']} has no feasible option")
    return best


def slot_achievable_all(spec):
    """Every feasible configuration of one slot as (unserved, deviation, overtime).

    Unlike :func:`slot_achievable` this does NOT collapse to the minimum overtime
    per (unserved, deviation) pair, so two configurations that share a pair but
    differ in overtime both survive.  That is what makes it possible to show that a
    one-decimal comparison merges distinct overtime levels.
    """
    tasks = spec["tasks"]
    choices = [[None] + list(spec["opts"][t]) for t in tasks]
    out, slack = [], _cap_slack(H_MAX)
    for combo in itertools.product(*choices):
        load, chosen, u = {}, set(), 0
        for t, i in zip(tasks, combo):
            if i is None:
                u += 1
            else:
                load[i] = load.get(i, 0.0) + spec["dur"][t]
                chosen.add((i, t))
        if any(v > float(H_MAX) + slack for v in load.values()):
            continue
        w = sum(v - float(H_STD) for v in load.values() if v > float(H_STD))
        d = len(chosen - spec["xbar"]) + len(spec["xbar"] - chosen)
        out.append((u, d, w))
    return out


def horizon_achievable(parser, plan, absent, regime):
    """Convolve the per-slot sets -> {(total unserved, total deviation): min W}."""
    acc = {(0, 0): 0.0}
    for spec in _structures(parser, plan, absent, regime):
        nxt = {}
        cur = slot_achievable(spec)
        for (u0, d0), w0 in acc.items():
            for (u1, d1), w1 in cur.items():
                k = (u0 + u1, d0 + d1)
                w = w0 + w1
                if k not in nxt or w < nxt[k]:
                    nxt[k] = w
        acc = nxt
    return acc


def min_unserved(ach):
    return min(u for (u, _d) in ach)


def lex_point(ach, u_common, eps=None):
    """min W s.t. (U == u_common, D <= eps), then min D at that W."""
    pool = {(u, d): w for (u, d), w in ach.items()
            if u == u_common and (eps is None or d <= eps)}
    if not pool:
        return None
    w_min = min(pool.values())
    d_min = min(d for (u, d), w in pool.items() if w <= w_min + W_ATOL)
    return {"U": u_common, "W": w_min, "D": d_min}


def oracle_disruption(hhcp_path, sol_path, absent, parser=None):
    """Everything the epsilon stage claims, recomputed independently."""
    parser = parser or HHCPFullParser(hhcp_path)
    plan = load_plan(sol_path)
    ach = {r: horizon_achievable(parser, plan, absent, r) for r in ("A1", "A2")}
    u_min = {r: min_unserved(ach[r]) for r in ("A1", "A2")}
    u_common = u_min["A1"]
    lo, hi = {}, {}
    for r in ("A1", "A2"):
        pool = [(u, d) for (u, d) in ach[r] if u == u_common]
        if not pool:
            raise AssertionError(f"oracle: U_common={u_common} unattainable in {r} "
                                 f"— A1-subset-A2 containment would be violated")
        lo[r] = min(d for (_u, d) in pool)
        hi[r] = lex_point(ach[r], u_common, eps=None)["D"]
    grid_lo, grid_hi = max(lo["A1"], lo["A2"]), max(hi["A1"], hi["A2"])
    grid = list(range(grid_lo, max(grid_hi, grid_lo) + 1))
    points = {}
    for r in ("A1", "A2"):
        for eps in grid:
            p = lex_point(ach[r], u_common, eps=eps)
            if p is None:
                raise AssertionError(f"oracle: ({r}, eps={eps}) infeasible — the "
                                     f"grid derivation is wrong")
            points[(r, eps)] = p
    return {"u_min": u_min, "u_common": u_common, "grid": grid,
            "grid_lo": grid_lo, "grid_hi": grid_hi, "points": points,
            "achievable": ach, "d_floor": lo, "d_at_wfloor": hi}


def oracle_active_absentees(parser, plan, max_absent=6):
    return sorted({n for q, m in plan.items() if q >= T_NOW
                   for n in m.values()})[:max_absent]
