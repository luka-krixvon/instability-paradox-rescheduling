"""
external/instance_spec.py (P0 / E1-E2) — format-independent core for running the
assignment-only-vs-displacement (A1/A2) study on EXTERNAL benchmarks.

Design
------
Each external benchmark is parsed by its adapter into a normalized `InstanceSpec`
(tasks with durations + candidate workers + a slot; workers with availability).
`MiniParser` then duck-types just the attributes that `displacement_milp.solve_slot`
and `eps_constraint.solve` read (parser.tasks[t].duration / .candidate_nurses;
parser.nurses[n].slots), so the EXISTING recovery solvers run unchanged on external
data. `build_flat_feasible_incumbent` best-fit packs the tasks into (worker, slot)
cells under the same flat workload the recovery optimizes (d_j = duration +
TRAVEL_PER_TASK, cap = rho*H_max), which removes the A1/A2 confound (Section E4) on
the external instances exactly as the synthetic generator does.

This file is fully functional WITHOUT any external data: run it directly to build a
synthetic InstanceSpec, pack a flat-feasible incumbent, and solve one A1 and one A2
recovery, exercising the whole external pipeline end to end. The self-test is a REGRESSION
check, not a validation of anything (`rerun_plan_v2_1.md` ground rule E1/E2).

===============================================================================================
S2 REVISION (2026-07-29) — fail-fast and evidence preservation only
===============================================================================================
Three changes, none of which alters the packing rule or any registered parameter:

1. **Cap-guard allowance.** The packing and feasibility guards compared against `cap + 1e-6`.
   `1e-6` is orders of magnitude larger than the representation error it was meant to absorb, so
   it silently ADMITTED cells over the cap. The allowance is now `math.ulp(cap) * 8`
   (`cap_allowance()`), which is the intended "a few representation steps" and nothing more.
   This is a TIGHTENING; it can only reject placements the old code accepted, and it may therefore
   change external packing outcomes. No external run was performed here, so what it changes in
   practice is unmeasured and is a stage-R2 observation.

2. **Evidence preservation (REQ-9 item 4).** `build_flat_feasible_incumbent()` now reports the
   unplaced task ids, the cap actually used and the allowance, and `greedy_worker_cover()` reports
   how many workers were rejected and how many tasks it dropped for having no kept candidate — a
   count that previously vanished at the list comprehension.

3. **Fail-fast (REQ-2).** `InstanceSpec.validate()` raises `InstanceSpecError` instead of using
   bare `assert` (asserts vanish under `python -O`), and formal callers use
   `build_flat_feasible_incumbent(strict=True)`, which aborts on an unplaced task or a
   non-flat-feasible incumbent instead of returning a report the caller may ignore.

The absence rule, `max_absent`, `min_coverage`, the slot count and the `cap_rho` grid remain
UNSET design parameters reserved for R0-D pre-registration (REQ-1, REQ-9 item 1).

    cd Code && python3 journal_p0/external/instance_spec.py     # self-test, no data needed
"""
from __future__ import annotations
import math, os, sys
from dataclasses import dataclass, field
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))          # Code/
sys.path.insert(0, os.path.dirname(_HERE))                           # Code/journal_p0
from SubgroupOptimizer import H_STD, H_MAX
from displacement_milp import TRAVEL_PER_TASK, solve_slot, W2, W5

# Shared S2 helpers are owned by another S2 agent and may not exist yet; this module must stay
# runnable without them.
try:                                                                 # pragma: no cover
    import s2_core as _s2_core
except Exception:                                                    # pragma: no cover
    _s2_core = None


class InstanceSpecError(AssertionError, ValueError):
    """An adapter produced something the pipeline cannot use, or a guard failed.

    Inherits from both AssertionError and ValueError so callers written against the pre-S2
    behaviour (bare `assert`) keep working.
    """


def cap_allowance(cap):
    """Representation-error allowance for a capacity comparison: eight ULPs of `cap`.

    This replaces the previous hard-coded `1e-6`, which was large enough to admit cells genuinely
    over the cap. It is a numerical guard only and says nothing about whether `cap` is the right
    capacity.

    If the shared S2 helper `s2_core.cap_fp_slack` is present it is used, and a DISAGREEMENT
    between the two definitions is fatal rather than silently resolved in favour of either: two
    stages guarding the same cap with different allowances is exactly the defect this rule exists
    to remove.
    """
    local = math.ulp(float(cap)) * 8
    shared = getattr(_s2_core, "cap_fp_slack", None) if _s2_core is not None else None
    if callable(shared):
        theirs = float(shared(cap))
        if theirs != local:
            raise InstanceSpecError(
                f"cap allowance disagreement at cap={cap!r}: s2_core.cap_fp_slack gives {theirs!r}, "
                f"this module's math.ulp(cap)*8 gives {local!r}. The two must not diverge.")
        return theirs
    return local


# --------------------------------------------------------------------------- #
#  Normalized instance representation (what every adapter must produce)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TaskSpec:
    id: int
    duration: float                 # SERVICE minutes (travel added inside solve_slot)
    slot: int                       # which time slot this task lives in
    candidates: frozenset           # worker ids qualified/eligible for this task


@dataclass(frozen=True)
class WorkerSpec:
    id: int
    slots: frozenset                # slots in which the worker is available


@dataclass
class InstanceSpec:
    name: str
    tasks: list                     # list[TaskSpec]
    workers: list                   # list[WorkerSpec]
    num_slots: int
    meta: dict = field(default_factory=dict)

    def validate(self):
        """Fail loudly if an adapter produced something the pipeline can't use.

        Raises `InstanceSpecError` rather than using bare `assert`, so the guards survive `-O`.
        """
        wids = {w.id for w in self.workers}
        if len(wids) != len(self.workers):
            raise InstanceSpecError("duplicate worker ids")
        tids = {t.id for t in self.tasks}
        if len(tids) != len(self.tasks):
            raise InstanceSpecError("duplicate task ids")
        if not isinstance(self.num_slots, int) or self.num_slots < 1:
            raise InstanceSpecError(f"num_slots must be a positive integer, got {self.num_slots!r}")
        wslots = {w.id: w.slots for w in self.workers}
        for t in self.tasks:
            if not (t.duration > 0) or not math.isfinite(t.duration):
                raise InstanceSpecError(f"task {t.id} has nonpositive or non-finite duration "
                                        f"{t.duration!r}")
            if not (0 <= t.slot < self.num_slots):
                raise InstanceSpecError(f"task {t.id} has slot {t.slot} outside "
                                        f"[0, {self.num_slots})")
            if not t.candidates:
                raise InstanceSpecError(f"task {t.id} has no candidate workers")
            if not t.candidates <= wids:
                raise InstanceSpecError(f"task {t.id} references unknown workers "
                                        f"{sorted(set(t.candidates) - wids)}")
            # every candidate must actually be available in the task's slot
            if not [w for w in t.candidates if t.slot in wslots[w]]:
                raise InstanceSpecError(f"task {t.id}: no candidate available in slot {t.slot}")
        return self


# --------------------------------------------------------------------------- #
#  Duck-typed parser so the existing solvers accept external instances
# --------------------------------------------------------------------------- #
class _Task:
    __slots__ = ("duration", "candidate_nurses")
    def __init__(self, duration, candidate_nurses):
        self.duration = duration
        self.candidate_nurses = candidate_nurses

class _Worker:
    __slots__ = ("slots",)
    def __init__(self, slots):
        self.slots = slots

class MiniParser:
    """Exposes exactly parser.tasks[t].{duration,candidate_nurses} and
    parser.nurses[n].slots, which is all displacement_milp.solve_slot /
    eps_constraint.solve read from a parser."""
    def __init__(self, spec: InstanceSpec):
        self.tasks = {t.id: _Task(t.duration, list(sorted(t.candidates))) for t in spec.tasks}
        self.nurses = {w.id: _Worker(set(w.slots)) for w in spec.workers}
        self._slot_of = {t.id: t.slot for t in spec.tasks}

    def slot_of(self, tid):
        return self._slot_of[tid]


# --------------------------------------------------------------------------- #
#  Tightness induction: reduce to a covering worker subset (the G1 treatment)
# --------------------------------------------------------------------------- #
def greedy_worker_cover(spec: InstanceSpec, extra=0, min_coverage=1):
    """Released benchmarks are over-provisioned, so capacity never binds and the
    trade-off is trivially absent. As the paper does for its own instance (G1), we
    reduce to a small set of workers that still covers every task, then induce
    capacity tightness by concentrating tasks on them (fewer slots / cap = rho*H_max).

    `min_coverage`: keep enough workers that every task has at least this many
    qualified workers in the kept set. With min_coverage>=2, removing one worker at
    the disruption never orphans a task that has no remaining qualified worker, so
    the residual unserved from skill coverage (not capacity) is eliminated.
    `extra`: keep this many additional high-degree workers to relax tightness."""
    extra_requested = extra
    covers = defaultdict(set)                     # worker -> task ids it can serve
    for t in spec.tasks:
        for w in t.candidates:
            covers[w].add(t.id)
    need = {t.id: min(min_coverage, len(t.candidates)) for t in spec.tasks}  # cap by available skill
    if not covers:
        raise InstanceSpecError("no worker covers any task; cannot build a cover")
    kept = set()
    while any(v > 0 for v in need.values()):
        # worker covering the most still-under-covered tasks
        best = max(covers, key=lambda w: sum(1 for tid in covers[w] if need.get(tid, 0) > 0) if w not in kept else -1)
        gain = [tid for tid in covers[best] if need.get(tid, 0) > 0]
        if not gain or best in kept:
            break
        kept.add(best)
        for tid in gain:
            need[tid] -= 1
    for w in sorted(covers, key=lambda w: -len(covers[w])):     # optional extra workers to loosen
        if extra <= 0:
            break
        if w not in kept:
            kept.add(w); extra -= 1
    tasks2 = [TaskSpec(t.id, t.duration, t.slot, frozenset(t.candidates & kept))
              for t in spec.tasks if (t.candidates & kept)]
    workers2 = [w for w in spec.workers if w.id in kept]
    # REQ-9 item 4: the tasks this reduction drops used to vanish silently at the comprehension
    # above, and the residual under-coverage was never reported. Both are now recorded.
    dropped = [t.id for t in spec.tasks if not (t.candidates & kept)]
    under = [tid for tid, v in need.items() if v > 0]
    return InstanceSpec(spec.name, tasks2, workers2, spec.num_slots,
                        meta={**spec.meta, "cover_workers": len(kept),
                              "orig_workers": len(spec.workers), "min_coverage": min_coverage,
                              "cover_workers_rejected": len(spec.workers) - len(kept),
                              "cover_extra_requested": extra_requested,
                              "cover_tasks_dropped": len(dropped),
                              "cover_tasks_dropped_ids": tuple(dropped),
                              "cover_tasks_under_covered": len(under)}).validate()


# --------------------------------------------------------------------------- #
#  Flat-feasible incumbent (removes the A1/A2 confound on external data)
# --------------------------------------------------------------------------- #
def build_flat_feasible_incumbent(spec: InstanceSpec, cap=H_MAX, quiet=False, strict=False):
    """Best-fit pack tasks into (worker, slot) cells under the FLAT workload
    d_j = duration + TRAVEL_PER_TASK, capping each cell at `cap`. Returns
    (slot_assign: {slot: {task_id: worker_id}}, report: dict). Any task that cannot
    be placed is left unplaced and reported (raise `cap` or add workers).

    The cap comparison uses `cap_allowance(cap)` = 8 ULPs, not the previous `1e-6`.

    `strict=True` makes an unplaced task or a non-flat-feasible incumbent FATAL (REQ-2).
    Formal R2 callers always pass it.  ``strict=False`` remains only for explicitly non-formal
    diagnostic construction that consumes the returned report.
    """
    allow = cap_allowance(cap)
    hmax_allow = cap_allowance(H_MAX)
    cell_load = defaultdict(float)                 # (worker, slot) -> flat load
    slot_assign = defaultdict(dict)                # slot -> {task: worker}
    wslots = {w.id: w.slots for w in spec.workers}
    unplaced = []
    for t in sorted(spec.tasks, key=lambda t: -(t.duration + TRAVEL_PER_TASK)):
        add = t.duration + TRAVEL_PER_TASK
        opts = [(cell_load[(w, t.slot)], w) for w in t.candidates
                if t.slot in wslots[w] and cell_load[(w, t.slot)] + add <= cap + allow]
        if not opts:
            unplaced.append(t.id); continue
        opts.sort(key=lambda x: -x[0])             # best-fit: fullest cell that still fits
        w = opts[0][1]
        slot_assign[t.slot][t.id] = w
        cell_load[(w, t.slot)] += add
    used = [L for L in cell_load.values() if L > 1e-9]
    report = dict(placed=sum(len(m) for m in slot_assign.values()), total=len(spec.tasks),
                  unplaced=len(unplaced), unplaced_ids=tuple(unplaced),
                  cap=float(cap), cap_allowance=allow,
                  max_cell=round(max(used), 1) if used else 0.0,
                  mean_cell=round(sum(used) / len(used), 1) if used else 0.0,
                  rho_used_mean=round((sum(used) / len(used)) / H_MAX, 3) if used else 0.0,
                  feasible=(not unplaced) and (max(used, default=0.0) <= H_MAX + hmax_allow))
    if not quiet:
        print(f"  incumbent: placed {report['placed']}/{report['total']} "
              f"unplaced={report['unplaced']} max_cell={report['max_cell']}/{H_MAX:.0f} "
              f"rho_used={report['rho_used_mean']} flat-feasible={report['feasible']}")
    if strict and (unplaced or not report["feasible"]):
        raise InstanceSpecError(
            f"incumbent is not flat-feasible at cap={cap} (allowance={allow!r}): "
            f"unplaced={report['unplaced']} ids={report['unplaced_ids'][:10]} "
            f"max_cell={report['max_cell']} H_MAX={H_MAX}. REQ-2 forbids continuing, and REQ-9 "
            f"item 2 forbids retuning any parameter to make this pass; record the failure instead.")
    return dict(slot_assign), report


# --------------------------------------------------------------------------- #
#  Synthetic self-test (no external data): prove the whole pipeline runs
# --------------------------------------------------------------------------- #
def _synthetic_spec(n_workers=8, n_slots=4, tasks_per_slot=10, seed=1):
    import random
    rng = random.Random(seed)
    workers = [WorkerSpec(id=w, slots=frozenset(range(n_slots))) for w in range(n_workers)]
    tasks = []; tid = 0
    for s in range(n_slots):
        for _ in range(tasks_per_slot):
            cands = frozenset(rng.sample(range(n_workers), k=rng.randint(3, 6)))
            tasks.append(TaskSpec(id=tid, duration=float(rng.choice([40, 55, 70, 85])), slot=s, candidates=cands))
            tid += 1
    return InstanceSpec("synthetic-selftest", tasks, workers, n_slots,
                        meta=dict(source="synthetic"))


def main():
    spec = _synthetic_spec().validate()
    print(f"[self-test] {spec.name}: {len(spec.tasks)} tasks, {len(spec.workers)} workers, {spec.num_slots} slots")
    slot_assign, rep = build_flat_feasible_incumbent(spec, cap=0.85 * H_MAX, strict=True)
    parser, S = MiniParser(spec), {w.id for w in spec.workers}
    # disrupt the busiest worker over all slots and recover one slot under A1 and A2
    load = defaultdict(int)
    for m in slot_assign.values():
        for w in m.values():
            load[w] += 1
    absent = max(load, key=load.get)
    q = next(iter(slot_assign))
    for regime in ("A1", "A2"):
        out = solve_slot(parser, q, slot_assign[q], absent, w4=50.0, S=S, regime=regime)
        dev, ot, served, total = out
        print(f"  {regime} slot {q}: deviation={dev} overtime={ot} served={served}/{total}")
    print("[self-test] OK — external pipeline runs end to end (W2=%.0f, W5=%.0f)." % (W2, W5))


if __name__ == "__main__":
    main()
