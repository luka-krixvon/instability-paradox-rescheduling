"""
compare_baselines.py (journal_p0) — instrumented cold-start / VNS / ALNS on the per-slot recovery
assignment problem, with REQ-14 run-level provenance.

Relationship to ``experiments/compare_baselines.py``
----------------------------------------------------
That module is the original, plot-producing baseline script. It is **not** modified here. This
module is the ``journal_p0``-local, *instrumented* replacement that the ``journal_p0`` pipelines
import: same objective, same neighbourhoods, same acceptance logic, but

  * every stochastic procedure receives its **own** :class:`s2_core.RngProvenance` — its own RNG
    object built from its own persisted seed. Passing one ``random.Random`` to two procedures
    couples their streams and makes the second one unreplayable; :func:`s2_core.require_distinct_rngs`
    now refuses it;
  * every run returns a :class:`HeuristicRun` record carrying the algorithm-specific seed, the
    repeat index, the iteration **budget** and the iterations **executed**, the objective
    **evaluation count**, a declared ``stop_reason``, the wall time, the objective, and the
    assignment hash — so one run is one persisted row rather than an aggregate;
  * the capacity guard uses ``s2_core.cap_fp_slack(H_MAX) = math.ulp(H_MAX) * 8`` instead of a
    fixed ``1e-6``.

Why the tolerance change is a fix and not cosmetics: the exact MILP bounds each worker's load by
``<= H_MAX`` exactly, while the old heuristic feasibility test accepted ``load <= H_MAX + 1e-6``.
The heuristics were therefore searching a **strictly larger** feasible set than the exact model,
which is one mechanism that can produce a *negative* "optimality gap" against a proven-optimal
exact solve. Such a gap is a defect to be surfaced, never a value to clamp at zero.

Deliberate non-claims: nothing in this module validates anything. Objective values it returns are
comparable to the exact model only under the caveats above, and only for the same ``w4``.

    (module; no ``__main__`` sweep entry point on purpose — see ``scalability.py --selftest``)
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_EXPERIMENTS = os.path.dirname(_HERE)
if _EXPERIMENTS not in sys.path:
    sys.path.insert(0, _EXPERIMENTS)

import s2_core                                                   # noqa: E402
from s2_core import COL, RngProvenance, assignment_hash, require  # noqa: E402
from SubgroupOptimizer import H_STD, H_MAX                        # noqa: E402
from displacement_milp import W2, W5                              # noqa: E402

__all__ = ["HeuristicRun", "evaluate", "greedy_init", "cold_start", "vns", "alns",
           "run_cold_start", "run_vns", "run_alns", "VNS_ITERS", "ALNS_ITERS",
           "VNS_NOIMP_PATIENCE", "CAP_SLACK", "EvalCounter"]

#: Capacity feasibility allowance. NOT 1e-6 — see the module docstring.
CAP_SLACK = s2_core.cap_fp_slack(H_MAX)

VNS_ITERS = 120                 # shaking iterations budget
VNS_NOIMP_PATIENCE = 25         # no-improvement early stop, previously invisible in the output
ALNS_ITERS = 300                # destroy/repair iterations budget
ALNS_RESET_PERIOD = 60          # periodic return to the incumbent best
ALNS_EXPLORE_PROB = 0.05        # acceptance of a non-improving repair


class EvalCounter:
    """Counts objective evaluations so ``evaluations`` is a measured number, not an estimate."""

    __slots__ = ("n",)

    def __init__(self) -> None:
        self.n = 0

    def tick(self) -> None:
        self.n += 1


@dataclass
class HeuristicRun:
    """One executed run of one method. One run == one persisted row (REQ-14)."""

    algorithm: str
    repeat_index: int
    rng_seed: Optional[int]
    iterations_budget: int
    iterations_executed: int
    evaluations: int
    stop_reason: str
    wall_time_s: float
    objective: float
    deviation: float
    overtime: float
    served: int
    unserved: int
    feasible: bool
    assignment: Dict[Any, Any]
    assignment_hash: str
    w4: float
    budget_regime: str = "fixed_iteration_budget"
    stochastic: bool = True

    def __post_init__(self) -> None:
        require(self.stop_reason in s2_core.STOP_REASONS,
                "stop_reason outside the declared enumeration",
                algorithm=self.algorithm, stop_reason=self.stop_reason)
        require(self.iterations_executed <= self.iterations_budget,
                "iterations_executed exceeds the declared budget",
                algorithm=self.algorithm, executed=self.iterations_executed,
                budget=self.iterations_budget)
        require(self.evaluations >= 0, "negative evaluation count",
                algorithm=self.algorithm, evaluations=self.evaluations)
        if self.stochastic:
            require(self.rng_seed is not None,
                    "a stochastic run must persist its own algorithm-specific seed",
                    algorithm=self.algorithm)
        s2_core.require_finite(self.objective, "objective", algorithm=self.algorithm)
        require(self.feasible,
                "heuristic returned an infeasible assignment; a partial/infeasible result may "
                "not be reported as a solution",
                algorithm=self.algorithm, cap=H_MAX, allowance=CAP_SLACK)

    def as_row(self) -> Dict[str, Any]:
        """The canonical run-level block. Column names come from :class:`s2_core.COL`."""
        return {
            COL.ALGORITHM: self.algorithm,
            COL.REPEAT_INDEX: self.repeat_index,
            COL.RNG_SEED: self.rng_seed,
            COL.ITERS_BUDGET: self.iterations_budget,
            COL.ITERS_EXECUTED: self.iterations_executed,
            COL.EVALUATIONS: self.evaluations,
            COL.STOP_REASON: self.stop_reason,
            COL.WALL_TIME: round(self.wall_time_s, 6),
            COL.ASSIGN_HASH: self.assignment_hash,
            COL.BUDGET_REGIME: self.budget_regime,
            "objective": self.objective,
            "deviation": self.deviation,
            "overtime": self.overtime,
            "served": self.served,
            "unserved": self.unserved,
            "w4": self.w4,
        }


# --------------------------------------------------------------------------------------------
# objective — identical algebra to experiments/compare_baselines.evaluate, tightened allowance
# --------------------------------------------------------------------------------------------


def evaluate(assign: Mapping[Any, Any], tasks: Sequence[Any], dur: Mapping[Any, float],
             xbar: Mapping[Any, Any], w4: float,
             counter: Optional[EvalCounter] = None) -> Tuple[float, float, float, int, bool]:
    """``obj(w4) = W2*overtime + w4*deviation + W5*unserved``.

    ``deviation = sum_{i,t} [x(1-xbar) + (1-x)xbar]``: keep=0, move=2, drop a previously assigned
    task=1, place an orphan=1 — the same accounting as the MILP. Returns
    ``(obj, dev, overtime, served, feasible)``. ``feasible`` uses :data:`CAP_SLACK`.
    """
    if counter is not None:
        counter.tick()
    load: Dict[Any, float] = defaultdict(float)
    served = 0
    for t in tasks:
        w = assign.get(t)
        if w is not None:
            load[w] += dur[t]
            served += 1
    ot = sum(max(0.0, l - H_STD) for l in load.values())
    feas = all(l <= H_MAX + CAP_SLACK for l in load.values())
    dev = 0
    for t in tasks:
        w = assign.get(t)
        xb = xbar.get(t)
        if xb is None:
            dev += 1 if w is not None else 0
        else:
            dev += 0 if w == xb else (1 if w is None else 2)
    uns = len(tasks) - served
    obj = W2 * ot + w4 * dev + W5 * uns
    return obj, dev, round(ot, 1), served, feas


def _fits(load: Mapping[Any, float], w: Any, t: Any, dur: Mapping[Any, float]) -> bool:
    return w is None or load[w] + dur[t] <= H_MAX + CAP_SLACK


def _loads(assign: Mapping[Any, Any], tasks: Sequence[Any],
           dur: Mapping[Any, float]) -> Dict[Any, float]:
    load: Dict[Any, float] = defaultdict(float)
    for t in tasks:
        w = assign.get(t)
        if w is not None:
            load[w] += dur[t]
    return load


def greedy_init(tasks: Sequence[Any], cand: Mapping[Any, Sequence[Any]],
                dur: Mapping[Any, float], xbar: Mapping[Any, Any], w4: float,
                ignore_dev: bool = False) -> Dict[Any, Any]:
    """Per-task greedy: pick the feasible worker (or ``None``) with the least marginal cost."""
    assign: Dict[Any, Any] = {}
    load: Dict[Any, float] = defaultdict(float)
    for t in sorted(tasks, key=lambda t: -dur[t]):
        best_w, best_c = None, None
        for w in list(cand[t]) + [None]:
            if not _fits(load, w, t, dur):
                continue
            if w is None:
                dot = 0.0
                dd = 1 if xbar.get(t) is not None else 0
                pen = W5
            else:
                dot = max(0.0, load[w] + dur[t] - H_STD) - max(0.0, load[w] - H_STD)
                xb = xbar.get(t)
                dd = 0 if w == xb else (2 if xb is not None else 1)
                pen = 0.0
            c = W2 * dot + (0.0 if ignore_dev else w4 * dd) + pen
            if best_c is None or c < best_c - 1e-9:
                best_c, best_w = c, w
        assign[t] = best_w
        if best_w is not None:
            load[best_w] += dur[t]
    return assign


def cold_start(tasks: Sequence[Any], cand: Mapping[Any, Sequence[Any]],
               dur: Mapping[Any, float], xbar: Mapping[Any, Any]) -> Dict[Any, Any]:
    """Efficiency-only rebuild: minimize overtime (+unserved), ignore deviation. Deterministic."""
    return greedy_init(tasks, cand, dur, xbar, 0.0, ignore_dev=True)


def _local_search(assign: Mapping[Any, Any], tasks: Sequence[Any],
                  cand: Mapping[Any, Sequence[Any]], dur: Mapping[Any, float],
                  xbar: Mapping[Any, Any], w4: float,
                  counter: EvalCounter) -> Tuple[Dict[Any, Any], float]:
    """relocate + swap best-improvement descent. One :func:`evaluate` call per candidate move."""
    cur = dict(assign)
    curo, _, _, _, feas = evaluate(cur, tasks, dur, xbar, w4, counter)
    require(feas, "local search started from an infeasible assignment",
            cap=H_MAX, allowance=CAP_SLACK)
    improved = True
    while improved:
        improved = False
        for t in tasks:                                                    # relocate
            for w in list(cand[t]) + [None]:
                if w == cur[t]:
                    continue
                trial = dict(cur)
                trial[t] = w
                o, _, _, _, ok = evaluate(trial, tasks, dur, xbar, w4, counter)
                if ok and o < curo - 1e-9:
                    cur, curo, improved = trial, o, True
        for i in range(len(tasks)):                                        # swap
            for j in range(i + 1, len(tasks)):
                t1, t2 = tasks[i], tasks[j]
                w1, w2 = cur[t1], cur[t2]
                if w1 == w2:
                    continue
                if w2 is not None and w2 not in cand[t1]:
                    continue
                if w1 is not None and w1 not in cand[t2]:
                    continue
                trial = dict(cur)
                trial[t1], trial[t2] = w2, w1
                o, _, _, _, ok = evaluate(trial, tasks, dur, xbar, w4, counter)
                if ok and o < curo - 1e-9:
                    cur, curo, improved = trial, o, True
    return cur, curo


# --------------------------------------------------------------------------------------------
# the two stochastic procedures — each takes its OWN RngProvenance, never a shared RNG object
# --------------------------------------------------------------------------------------------


def vns(tasks: Sequence[Any], cand: Mapping[Any, Sequence[Any]], dur: Mapping[Any, float],
        xbar: Mapping[Any, Any], w4: float, prov: RngProvenance,
        iters: int = VNS_ITERS, patience: int = VNS_NOIMP_PATIENCE,
        counter: Optional[EvalCounter] = None) -> Tuple[Dict[Any, Any], float, int, str]:
    """Variable-neighbourhood search. Returns ``(assign, obj, iterations_executed, stop_reason)``."""
    require(isinstance(prov, RngProvenance),
            "vns requires its OWN RngProvenance; a bare shared random.Random is refused "
            "because it makes the run unreplayable (REQ-14)", got=type(prov).__name__)
    counter = counter if counter is not None else EvalCounter()
    if not tasks:
        return {}, 0.0, 0, "empty_instance"
    rng = prov.rng
    best = greedy_init(tasks, cand, dur, xbar, w4)
    best, besto = _local_search(best, tasks, cand, dur, xbar, w4, counter)
    noimp = 0
    executed = 0
    stop_reason = "budget_exhausted"
    for _ in range(iters):
        if noimp >= patience:
            stop_reason = "no_improvement_early_stop"
            break
        executed += 1
        trial = dict(best)
        for _shake in range(rng.randint(1, 2)):
            t = rng.choice(list(tasks))
            # The shake must stay inside the CAPACITY-FEASIBLE region. The original code drew
            # from `cand[t] + [None]` unconditionally, so it could place a task on an already
            # full worker; the descent was then started from an infeasible point and its
            # incumbent objective `curo` was an infeasible-solution value that is not comparable
            # to the MILP objective. Only the outer acceptance test filtered it, so an infeasible
            # point could still be the local-search *basis* for a whole iteration.
            trial[t] = None
            load = _loads(trial, tasks, dur)
            options = [w for w in cand[t] if _fits(load, w, t, dur)] + [None]
            trial[t] = rng.choice(options)
        trial, o = _local_search(trial, tasks, cand, dur, xbar, w4, counter)
        _, _, _, _, feas = evaluate(trial, tasks, dur, xbar, w4, counter)
        if feas and o < besto - 1e-9:
            best, besto, noimp = trial, o, 0
        else:
            noimp += 1
    return best, besto, executed, stop_reason


def alns(tasks: Sequence[Any], cand: Mapping[Any, Sequence[Any]], dur: Mapping[Any, float],
         xbar: Mapping[Any, Any], w4: float, prov: RngProvenance,
         iters: int = ALNS_ITERS,
         counter: Optional[EvalCounter] = None) -> Tuple[Dict[Any, Any], float, int, str]:
    """Adaptive large-neighbourhood search. Returns ``(assign, obj, executed, stop_reason)``."""
    require(isinstance(prov, RngProvenance),
            "alns requires its OWN RngProvenance; a bare shared random.Random is refused "
            "because it makes the run unreplayable (REQ-14)", got=type(prov).__name__)
    counter = counter if counter is not None else EvalCounter()
    if not tasks:
        return {}, 0.0, 0, "empty_instance"
    rng = prov.rng
    best = greedy_init(tasks, cand, dur, xbar, w4)
    best, besto = _local_search(best, tasks, cand, dur, xbar, w4, counter)
    cur, curo = dict(best), besto
    executed = 0
    for it in range(iters):
        executed += 1
        if it % ALNS_RESET_PERIOD == ALNS_RESET_PERIOD - 1:
            cur, curo = dict(best), besto
        k = max(1, min(len(tasks), rng.randint(1, 4)))
        removed = rng.sample(list(tasks), k)
        partial = dict(cur)
        for t in removed:
            partial[t] = None
        load: Dict[Any, float] = defaultdict(float)
        for t in tasks:
            if partial[t] is not None:
                load[partial[t]] += dur[t]
        for t in sorted(removed, key=lambda t: -dur[t]):
            best_w, best_c = None, None
            for w in list(cand[t]) + [None]:
                if not _fits(load, w, t, dur):
                    continue
                if w is None:
                    dot = 0.0
                    dd = 1 if xbar.get(t) is not None else 0
                    pen = W5
                else:
                    dot = max(0.0, load[w] + dur[t] - H_STD) - max(0.0, load[w] - H_STD)
                    xb = xbar.get(t)
                    dd = 0 if w == xb else (2 if xb is not None else 1)
                    pen = 0.0
                c = W2 * dot + w4 * dd + pen
                if best_c is None or c < best_c - 1e-9:
                    best_c, best_w = c, w
            partial[t] = best_w
            if best_w is not None:
                load[best_w] += dur[t]
        partial, o = _local_search(partial, tasks, cand, dur, xbar, w4, counter)
        if o < curo - 1e-9 or rng.random() < ALNS_EXPLORE_PROB:
            cur, curo = partial, o
            if o < besto - 1e-9:
                best, besto = dict(partial), o
    return best, besto, executed, "budget_exhausted"


# --------------------------------------------------------------------------------------------
# run wrappers — these are what the pipelines call; each returns ONE persistable HeuristicRun
# --------------------------------------------------------------------------------------------


def _finish(algorithm: str, prov: Optional[RngProvenance], repeat_index: int, budget: int,
            executed: int, counter: EvalCounter, stop_reason: str, t0: float,
            assign: Mapping[Any, Any], tasks: Sequence[Any], dur: Mapping[Any, float],
            xbar: Mapping[Any, Any], w4: float, stochastic: bool,
            budget_regime: str) -> HeuristicRun:
    wall = time.perf_counter() - t0
    obj, dev, ot, served, feas = evaluate(assign, tasks, dur, xbar, w4, counter)
    return HeuristicRun(
        algorithm=algorithm, repeat_index=int(repeat_index),
        rng_seed=(None if prov is None else prov.seed),
        iterations_budget=int(budget), iterations_executed=int(executed),
        evaluations=counter.n, stop_reason=stop_reason, wall_time_s=wall,
        objective=obj, deviation=dev, overtime=ot, served=served,
        unserved=len(tasks) - served, feasible=feas, assignment=dict(assign),
        assignment_hash=assignment_hash(assign), w4=float(w4), stochastic=stochastic,
        budget_regime=budget_regime)


def run_cold_start(tasks: Sequence[Any], cand: Mapping[Any, Sequence[Any]],
                   dur: Mapping[Any, float], xbar: Mapping[Any, Any], w4: float,
                   repeat_index: int = 1) -> HeuristicRun:
    """Deterministic single pass. ``rng_seed`` is ``None`` because no RNG is consumed."""
    counter = EvalCounter()
    t0 = time.perf_counter()
    if not tasks:
        return _finish("cold", None, repeat_index, 1, 0, counter, "empty_instance", t0,
                       {}, tasks, dur, xbar, w4, False, "single_pass")
    assign = cold_start(tasks, cand, dur, xbar)
    return _finish("cold", None, repeat_index, 1, 1, counter, "deterministic_single_pass", t0,
                   assign, tasks, dur, xbar, w4, False, "single_pass")


def run_vns(tasks: Sequence[Any], cand: Mapping[Any, Sequence[Any]], dur: Mapping[Any, float],
            xbar: Mapping[Any, Any], w4: float, prov: RngProvenance,
            repeat_index: Optional[int] = None, iters: int = VNS_ITERS,
            patience: int = VNS_NOIMP_PATIENCE) -> HeuristicRun:
    counter = EvalCounter()
    t0 = time.perf_counter()
    assign, _obj, executed, stop = vns(tasks, cand, dur, xbar, w4, prov, iters=iters,
                                       patience=patience, counter=counter)
    return _finish("VNS", prov,
                   prov.repeat_index if repeat_index is None else repeat_index,
                   iters, executed, counter, stop, t0, assign, tasks, dur, xbar, w4, True,
                   "fixed_iteration_budget")


def run_alns(tasks: Sequence[Any], cand: Mapping[Any, Sequence[Any]], dur: Mapping[Any, float],
             xbar: Mapping[Any, Any], w4: float, prov: RngProvenance,
             repeat_index: Optional[int] = None, iters: int = ALNS_ITERS) -> HeuristicRun:
    counter = EvalCounter()
    t0 = time.perf_counter()
    assign, _obj, executed, stop = alns(tasks, cand, dur, xbar, w4, prov, iters=iters,
                                        counter=counter)
    return _finish("ALNS", prov,
                   prov.repeat_index if repeat_index is None else repeat_index,
                   iters, executed, counter, stop, t0, assign, tasks, dur, xbar, w4, True,
                   "fixed_iteration_budget")
