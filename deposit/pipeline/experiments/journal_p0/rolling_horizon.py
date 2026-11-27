"""
rolling_horizon.py (P0 / B1) — rolling-horizon recovery over a DISRUPTION STREAM.

Note on scope: for a SINGLE disruption the model's slots are independent (a task is fixed to
its slot; capacity/overtime are per-slot), so a rolling/limited-lookahead recovery coincides
exactly with the one-shot per-slot exact solve. Rolling-horizon is therefore only a distinct
baseline under a STREAM of disruptions, which is what we simulate here: caregivers drop out at
successive epochs and the remaining horizon is re-recovered against the current committed plan.

Also: "stability-constrained rescheduling" (min overtime s.t. deviation <= eps) is exactly the
epsilon-constraint method already evaluated in eps_constraint.py -- it is not a separate baseline.

Question tested: does the instability paradox COMPOUND under a stream?
  - A1 (assignment-only) each step: overtime piles up, w4 stays inert;
  - A2 (displacement) each step: overtime stays controlled.

===============================================================================================
S2 REVISION (2026-07-29) — what changed, and what it does NOT establish
===============================================================================================
This module was revised under `audit_outputs/rerun_plan_v2_1.md` **REQ-7** and **REQ-8** (and the
fail-fast / validation contract of **REQ-2**, **REQ-3**, **REQ-13**).

Nothing in this file is a result, a validation, or evidence that any published number is correct.
The fixtures shipped alongside it (`tests/fixtures/rollext/`) are **regression evidence only**
(ground rule **E1**/**E2** of the re-run plan). Running this module is stage **R3**, which is a
formal re-run stage: `main()` therefore *refuses to run* unless `--stage R3` is passed explicitly,
so that an accidental `python rolling_horizon.py` can no longer overwrite a canonical artifact.

1. **REQ-7 — the absence sequence is now EXOGENOUS.**
   `absence_sequence()` is computed from `(committed_plan, seed, rule)` ONLY, *before* any recovery
   solve. Its body references neither the recovery policy nor the displacement-cost weight, so the
   FOUR configurations of a seed provably consume the same stream. The sequence is persisted on
   every result row (`absent_seq`) and additionally in a per-`(seed, epoch)` table, so
   cross-configuration identity is checkable **from the artifact** rather than from narration.

   The FOUR configurations already existed (`A1/0`, `A1/100`, `A2/0`, `A2/100`). This revision does
   **not** "add a fourth configuration" — that framing would be wrong. What was defective was that
   the absence stream was read off `cur`, the plan *as already modified by the policy under
   evaluation*, so the four configurations could face four different streams.

   The absence RULE is a stage-R3 design parameter. It has **no default**: `absence_sequence()`
   raises `RollingDesignUnregistered` unless the caller names one, because REQ-7's acceptance test
   requires the rule to be declared and hashed in the R3 design manifest *before* the run.

2. **REQ-8 — four overtime estimands, none privileged.**
   Verbatim definitions (identical strings live in `OT_ESTIMANDS` and must be reproduced in any
   table caption that shows these numbers):

   - `ot_full_plan`:
       "sum over EVERY slot of the final plan of max(0, cell_load(caregiver, slot) - H_std)"
   - `ot_recoverable_horizon`:
       "the same sum restricted to slots q >= min(EPOCHS), i.e. the slots any policy may re-solve"
   - `ot_frozen_prefix`:
       "the same sum restricted to slots q < min(EPOCHS), computed from the COMMITTED plan; no
        policy can re-solve those slots, so this is a policy-invariant additive constant"
   - `ot_cumulative`:
       "sum over epochs e in EPOCHS of the terminal overtime of the then-remaining horizon,
        i.e. sum_e [ sum over slots q >= e of max(0, cell_load(caregiver, q) - H_std) ] evaluated
        on the plan as it stands at the end of epoch e; this is NOT ot_full_plan and NOT
        ot_recoverable_horizon"

   None of these is "the true overtime", "the correct overtime", or "the" overtime. Every ratio
   computed from them must name its estimand and its slot scope in the same sentence. Which
   estimand leads the abstract is decision `P-H1` of the re-run plan and is NOT decided here.

   The identity `ot_full_plan - ot_recoverable_horizon == ot_frozen_prefix` is enforced per row as
   a FATAL guard, in exact rational arithmetic, with `ot_frozen_prefix` computed independently from
   the committed plan. It is also re-checked across the four configurations of a seed.

3. **Three deviation quantities, kept strictly distinct — never conflated.**
   - `D` — SYMMETRIC DIFFERENCE, terminal: `|A_final Δ A_committed|` over assignment triples
     `(slot, caregiver, task)`. Reassigning one task from one caregiver to another contributes
     **2**.
   - `dev_terminal_tasks` — TERMINAL DEVIATION: the number of tasks whose final caregiver differs
     from its committed caregiver. The same reassignment contributes **1**.
   - `churn_total` / `churn_per_step` — STEP CHURN: the per-solve deviation accumulated over every
     epoch and every re-solved slot. `churn_per_step` is `churn_total / n_epochs_executed`; both
     denominators (`n_epochs_executed`, `n_slot_solves`) are persisted so either normalization is
     recomputable. The published heading "Per-step churn" described `churn_total`, which is a
     different quantity.
   Structural relation asserted by the validator: `D >= dev_terminal_tasks`.

   **`churn_total` is NOT a sum of `D`-style symmetric differences, and the two must not be added
   or compared as if they were.** The per-solve deviation is the model's own deviation objective:
   it is the symmetric difference against `xbar`, the RETAINED part of the incumbent, which by
   construction excludes every pair whose caregiver has become absent. Re-placing a task whose
   holder just left therefore contributes **1** to churn (the new pair) while contributing **2** to
   a full symmetric difference (the lost pair *and* the new one). Consequently `churn_total` can be
   smaller than `D` even though churn accumulates over five epochs; that is a definitional fact
   about the objective, not an inconsistency, and it is exactly why REQ-8 requires the three
   quantities to be reported separately rather than reconciled.

   `ot_cumulative` depends on the INTERMEDIATE plans, which the final schedule cannot reconstruct,
   so its five per-epoch terms are persisted verbatim in `ot_cumulative_terms` and the validator
   checks that they sum to `ot_cumulative`. Without that column the number would be an unauditable
   scalar.

4. **Fail-fast (REQ-2 / REQ-13).** `solve_assign()` no longer returns `None`: a solver failure, a
   non-admissible solver status, an empty candidate structure, or a non-finite objective raises.
   The previous `if r is None: continue` left the slot on its PRE-disruption assignment and then
   counted it as recovered — a wrong value, not a missing one. Admissible statuses for R3 are
   declared in `ADMISSIBLE_SOLVER_STATUS_R3`.

5. **Per-slot overtime and schedule evidence are persisted.** Two extra tables make every overtime
   number recomputable from the artifact: a per-`(seed, regime, w4, slot)` overtime table carrying
   the per-`(caregiver, slot)` flat loads, and the full recovered schedule.

6. **temp -> validate -> promote.** Canonical outputs are written to a temporary directory,
   validated, and only then moved into place, with rollback of already-moved files on a CAUGHT
   failure. This is a per-file `os.replace`; **no pair-atomicity or family-atomicity is claimed**
   and none should be inferred.

Formal R3 now receives its hashed design manifest and externally verified SNAP
identifier on the CLI.  The paper's old `w4_response()` "tie-breaking noise"
wording remains deferred until the new result is available; it is not reproduced
by this module.  Solver/schedule overtime closure uses a pre-registered `1e-6`
continuous solver-readback tolerance separately from the hard-cap allowance,
which remains `math.ulp(H_MAX)*8`.

    python journal_p0/rolling_horizon.py --stage R3 --absence-rule busiest_committed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(HERE)
for _p in (_PKG_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from SubgroupOptimizer import HHCPFullParser, H_STD, H_MAX          # noqa: E402
from displacement_milp import load_static, TRAVEL_PER_TASK, W2, W5, T_NOW  # noqa: E402

# Shared S2 helpers are owned by another S2 agent and may not exist yet; this module must remain
# runnable without them, so every use goes through _shared() with a local fallback.
try:                                                                # pragma: no cover
    import s2_core
except Exception:                                                   # pragma: no cover
    s2_core = None


def _shared(name, fallback):
    fn = getattr(s2_core, name, None) if s2_core is not None else None
    return fn if callable(fn) else fallback


# --------------------------------------------------------------------------------------------- #
#  Registered design constants (values UNCHANGED from the pre-S2 module)
# --------------------------------------------------------------------------------------------- #
EPOCHS = [2, 4, 6, 8, 10]
E0 = min(EPOCHS)                    # first re-solvable slot; slots q < E0 are the frozen prefix
RHO_TARGET = 0.85
REGIMES = ("A1", "A2")
W4_LEVELS = (0.0, 100.0)
#: The FOUR configurations. They already existed pre-S2; this is not a new configuration.
CONFIGURATIONS = tuple((r, w) for r in REGIMES for w in W4_LEVELS)
N_CONFIGURATIONS = len(CONFIGURATIONS)

#: Absence rules selectable for stage R3. There is deliberately NO default (REQ-7).
ABSENCE_RULES = ("busiest_committed", "seeded_uniform")

#: REQ-13: the admissible solver-status set for THIS stage. scipy's milp status 0 == optimal.
ADMISSIBLE_SOLVER_STATUS_R3 = frozenset({0})

SOLVER_TIME_LIMIT = 30
SOLVER_READBACK_ABS_TOL = 1e-6

OT_ESTIMANDS = {
    "ot_full_plan":
        "sum over EVERY slot of the final plan of max(0, cell_load(caregiver, slot) - H_std)",
    "ot_recoverable_horizon":
        "the same sum restricted to slots q >= min(EPOCHS), i.e. the slots any policy may re-solve",
    "ot_frozen_prefix":
        "the same sum restricted to slots q < min(EPOCHS), computed from the COMMITTED plan; no "
        "policy can re-solve those slots, so this is a policy-invariant additive constant",
    "ot_cumulative":
        "sum over epochs e in EPOCHS of the terminal overtime of the then-remaining horizon, "
        "i.e. sum_e [ sum over slots q >= e of max(0, cell_load(caregiver, q) - H_std) ] evaluated "
        "on the plan as it stands at the end of epoch e; this is NOT ot_full_plan and NOT "
        "ot_recoverable_horizon",
}
CHURN_PER_STEP_DEFINITION = "churn_total / n_epochs_executed"
#: `W` is not a fifth estimand: it names which of the four it is.
W_ESTIMAND = "ot_recoverable_horizon"
W_SCOPE = "slots q >= min(EPOCHS) of the final plan"

RAW_BASENAME = "rolling_raw.csv"
SLOT_BASENAME = "rolling_slot_overtime.csv"
SCHEDULE_BASENAME = "rolling_schedule.csv"
EPOCH_BASENAME = "rolling_absence_sequence.csv"
SOLVER_BASENAME = "rolling_solver_evidence.csv"

_HSTD = Fraction(H_STD)
_TRAVEL = Fraction(TRAVEL_PER_TASK)


# --------------------------------------------------------------------------------------------- #
#  Fatal errors. Every guard in this module is FATAL; nothing degrades to a warning.
# --------------------------------------------------------------------------------------------- #
class RollingFatal(RuntimeError):
    """Base class: any occurrence must abort the stage without writing a canonical artifact."""


class RollingSolveFailure(RollingFatal):
    """A recovery solve failed, or returned a status not admissible for this stage."""


class RollingDesignUnregistered(RollingFatal):
    """A stage-R3 design parameter was not supplied; it must come from the R3 design manifest."""


class RollingValidationError(RollingFatal):
    """A validator check failed. The canonical artifact must be left untouched."""


def load_r3_design(path, snap_id):
    """Load and close the R3 design contract before any instance or solver is touched.

    The manifest pre-registers the *binding method*, while the runner supplies the externally
    recomputed content-addressed archive SHA256.  The hash literal is never written inside the
    archive it identifies (which would be self-referential).  All formal solver and experiment
    choices are returned from the manifest; the CLI supplies no shadow defaults.
    """
    if not path or not os.path.isfile(path):
        raise RollingDesignUnregistered(f"R3 design manifest not found: {path!r}")
    if not isinstance(snap_id, str) or len(snap_id) != 64 \
            or any(ch not in "0123456789abcdef" for ch in snap_id):
        raise RollingDesignUnregistered(
            "--snap-id must be the 64-lowercase-hex SHA256 of the immutable content-addressed "
            "SNAP archive")
    with open(path, "r", encoding="utf-8") as fh:
        try:
            manifest = json.load(fh)
        except ValueError as exc:
            raise RollingDesignUnregistered(
                f"R3 design manifest is not valid JSON: {path!r}: {exc}") from exc
    if manifest.get("stage") != "R3":
        raise RollingDesignUnregistered(
            f"design manifest stage={manifest.get('stage')!r}, expected 'R3'")
    snap_binding = manifest.get("snap_binding")
    if snap_binding != "cli_content_addressed_archive_sha256":
        raise RollingDesignUnregistered(
            "manifest must pre-register "
            "snap_binding='cli_content_addressed_archive_sha256'; the archive hash itself must "
            "not be written inside the archive (which would be self-referential)")
    if s2_core is None:
        raise RollingDesignUnregistered("s2_core is required for a formal R3 run")
    policy = s2_core.StageSolverPolicy.from_manifest(
        manifest, stage="R3", manifest_path=path,
        manifest_sha256=s2_core.sha256_file(path))
    if policy.admissible_codes != {0} or policy.optimal_codes != {0} \
            or policy.accepts_time_limit_incumbent:
        raise RollingDesignUnregistered(
            "R3 is code-0-only: admissible_statuses and optimal_statuses must both be {0}, "
            "and accepts_time_limit_incumbent must be false")
    design = manifest.get("design")
    if not isinstance(design, dict):
        raise RollingDesignUnregistered("manifest.design must be an object")
    required = ("input_manifest", "rho_target", "absence_rule", "epochs",
                "regimes", "w4_levels", "solver_options", "numeric_tolerances")
    missing = [name for name in required if name not in design]
    if missing:
        raise RollingDesignUnregistered(f"manifest.design missing required keys: {missing}")
    if design["absence_rule"] not in ABSENCE_RULES:
        raise RollingDesignUnregistered(
            f"manifest absence_rule={design['absence_rule']!r}; expected one of {ABSENCE_RULES}")
    epochs = design["epochs"]
    if not isinstance(epochs, list) or not epochs or any(
            isinstance(v, bool) or not isinstance(v, int) for v in epochs):
        raise RollingDesignUnregistered("manifest design.epochs must be a non-empty integer list")
    if epochs != sorted(set(epochs)):
        raise RollingDesignUnregistered(
            f"manifest epochs must be strictly increasing and unique, got {epochs!r}")
    regimes = tuple(design["regimes"]) if isinstance(design["regimes"], list) else ()
    if regimes != ("A1", "A2"):
        raise RollingDesignUnregistered(
            f"manifest regimes must be exactly ['A1','A2'], got {regimes!r}")
    w4_levels = design["w4_levels"]
    if not isinstance(w4_levels, list) or not w4_levels:
        raise RollingDesignUnregistered("manifest w4_levels must be a non-empty list")
    try:
        w4_levels = tuple(float(v) for v in w4_levels)
    except (TypeError, ValueError) as exc:
        raise RollingDesignUnregistered("manifest w4_levels must be numeric") from exc
    if not all(np.isfinite(w4_levels)) or len(set(w4_levels)) != len(w4_levels):
        raise RollingDesignUnregistered(
            f"manifest w4_levels must be finite and unique, got {w4_levels!r}")
    solver_options = design["solver_options"]
    if not isinstance(solver_options, dict):
        raise RollingDesignUnregistered("manifest design.solver_options must be an object")
    if solver_options.get("presolve") is not False:
        raise RollingDesignUnregistered(
            "manifest solver_options.presolve must be false (known HiGHS presolve defect)")
    if set(solver_options) != {"presolve", "time_limit", "mip_rel_gap"}:
        raise RollingDesignUnregistered(
            "manifest solver_options must declare exactly presolve, time_limit and mip_rel_gap; "
            f"got {sorted(solver_options)}")
    if "time_limit" not in solver_options:
        raise RollingDesignUnregistered("manifest solver_options.time_limit is required")
    if float(solver_options["time_limit"]) != float(policy.time_limit_s):
        raise RollingDesignUnregistered(
            f"solver_options.time_limit={solver_options['time_limit']!r} disagrees with "
            f"solver.time_limit_s={policy.time_limit_s!r}")
    mip_rel_gap = float(solver_options["mip_rel_gap"])
    if not np.isfinite(mip_rel_gap) or mip_rel_gap < 0:
        raise RollingDesignUnregistered(
            f"solver_options.mip_rel_gap must be finite and non-negative, got {mip_rel_gap!r}")
    numeric_tolerances = design["numeric_tolerances"]
    if not isinstance(numeric_tolerances, dict):
        raise RollingDesignUnregistered("manifest design.numeric_tolerances must be an object")
    if numeric_tolerances.get("solver_schedule_overtime_abs") != SOLVER_READBACK_ABS_TOL:
        raise RollingDesignUnregistered(
            "manifest solver_schedule_overtime_abs must exactly equal the executable "
            f"solver-readback tolerance {SOLVER_READBACK_ABS_TOL!r}")
    if numeric_tolerances.get("capacity_allowance_rule") != "math.ulp(H_MAX)*8":
        raise RollingDesignUnregistered(
            "manifest capacity allowance must remain math.ulp(H_MAX)*8; the solver "
            "readback tolerance may not loosen feasibility")
    source = design["input_manifest"]
    if not isinstance(source, str) or not source.strip():
        raise RollingDesignUnregistered("manifest design.input_manifest must be non-empty")
    source = source if os.path.isabs(source) else os.path.join(os.path.dirname(path), source)
    source = os.path.abspath(source)
    if not os.path.isfile(source):
        raise RollingDesignUnregistered(f"R3 input manifest not found: {source!r}")
    rho_target = float(design["rho_target"])
    if not np.isfinite(rho_target) or rho_target <= 0:
        raise RollingDesignUnregistered("manifest rho_target must be positive and finite")
    return dict(
        manifest=manifest, policy=policy, manifest_path=os.path.abspath(path),
        manifest_sha256=policy.manifest_sha256, snap_id=snap_id,
        input_manifest=source, rho_target=rho_target,
        absence_rule=design["absence_rule"], epochs=list(epochs),
        configurations=tuple((r, w) for r in regimes for w in w4_levels),
        solver_options=dict(solver_options),
        numeric_tolerances=dict(numeric_tolerances))


# --------------------------------------------------------------------------------------------- #
#  REQ-7 — EXOGENOUS absence sequence
# --------------------------------------------------------------------------------------------- #
def absence_sequence(committed, seed, epochs=None, rule=None):
    """Return the absence sequence as a list aligned with `epochs`, one entry per epoch.

    Inputs are ONLY the committed plan, the integer seed, the epoch grid and the named rule.
    The result is therefore fixed before any recovery solve runs and is shared by every
    configuration of the seed. Nothing about how the plan is later recovered can reach this
    function: it never sees a recovered plan, and it takes no policy argument of any kind.

    `committed` maps slot -> {task: caregiver}. An entry is `None` when the rule finds no eligible
    caregiver for that epoch (rather than silently shortening the sequence).

    Rules:
      busiest_committed -- remove the caregiver holding the most tasks in the COMMITTED plan over
                           the still-open slots (q >= e), ties broken by the seeded RNG over the
                           id-sorted tied set.
      seeded_uniform    -- remove a caregiver drawn uniformly by the seeded RNG from the id-sorted
                           set of caregivers still holding a committed task over q >= e.
    In both rules only caregivers that hold at least one committed task over the still-open slots
    are eligible, and a caregiver already removed is not eligible again.
    """
    if rule is None:
        raise RollingDesignUnregistered(
            "absence rule is UNSET. REQ-7 requires the rule to be declared and hashed in the R3 "
            f"design manifest before the run; pass one of {ABSENCE_RULES}.")
    if rule not in ABSENCE_RULES:
        raise RollingDesignUnregistered(f"unknown absence rule {rule!r}; expected one of {ABSENCE_RULES}")
    grid = list(EPOCHS if epochs is None else epochs)
    if not grid:
        raise RollingDesignUnregistered("epoch grid is empty")
    rng = random.Random("rolling-absence|%s|%d" % (rule, int(seed)))
    removed, seq = set(), []
    for e in grid:
        counts = defaultdict(int)
        for q, m in committed.items():
            if q < e:
                continue
            for _t, who in m.items():
                if who is None or who in removed:
                    continue
                counts[who] += 1
        if not counts:
            seq.append(None)
            continue
        if rule == "busiest_committed":
            top = max(counts.values())
            pool = sorted(w for w, c in counts.items() if c == top)
        else:
            pool = sorted(counts)
        pick = pool[rng.randrange(len(pool))]
        removed.add(pick)
        seq.append(pick)
    return seq


def absence_sequence_key(seq):
    """Stable one-cell rendering of an absence sequence, for the `absent_seq` column."""
    return "|".join("-" if a is None else str(a) for a in seq)


# --------------------------------------------------------------------------------------------- #
#  Exact overtime / deviation bookkeeping (Fraction: the partition identity must be exact)
# --------------------------------------------------------------------------------------------- #
def _flat_duration(parser, t):
    return Fraction(parser.tasks[t].duration) + _TRAVEL


def cell_loads(parser, plan, q_min=None, q_max_excl=None):
    """(caregiver, slot) -> exact flat load, over the slot window [q_min, q_max_excl)."""
    load = defaultdict(Fraction)
    for q, m in plan.items():
        if q_min is not None and q < q_min:
            continue
        if q_max_excl is not None and q >= q_max_excl:
            continue
        for t, who in m.items():
            if who is None or t not in parser.tasks:
                continue
            load[(who, q)] += _flat_duration(parser, t)
    return dict(load)


def overtime_by_cell(loads):
    return {k: (L - _HSTD if L > _HSTD else Fraction(0)) for k, L in loads.items()}


def overtime_total(parser, plan, q_min=None, q_max_excl=None):
    cells = overtime_by_cell(cell_loads(parser, plan, q_min=q_min, q_max_excl=q_max_excl))
    tot = Fraction(0)
    for v in cells.values():
        tot += v
    return tot


def overtime_by_slot(parser, plan):
    per = defaultdict(Fraction)
    for (_who, q), v in overtime_by_cell(cell_loads(parser, plan)).items():
        per[q] += v
    return dict(per)


def assignment_triples(plan, q_min=None):
    return {(q, who, t)
            for q, m in plan.items() if (q_min is None or q >= q_min)
            for t, who in m.items() if who is not None}


def symmetric_difference_D(final, committed, q_min=None):
    """|A_final Δ A_committed| over (slot, caregiver, task) triples. NOT the terminal deviation."""
    return len(assignment_triples(final, q_min) ^ assignment_triples(committed, q_min))


def terminal_deviation_tasks(final, committed, q_min=None):
    """Number of tasks whose FINAL caregiver differs from its COMMITTED caregiver. NOT D."""
    n = 0
    for q, m in committed.items():
        if q_min is not None and q < q_min:
            continue
        got = final.get(q, {})
        for t, who0 in m.items():
            if got.get(t) != who0:
                n += 1
    return n


def unserved_count(plan, q_min=None):
    return sum(1 for q, m in plan.items() if (q_min is None or q >= q_min)
               for _t, who in m.items() if who is None)


# --------------------------------------------------------------------------------------------- #
#  Per-slot recovery MILP (model UNCHANGED; fail-soft returns replaced by FATAL raises)
# --------------------------------------------------------------------------------------------- #
def solve_assign(parser, q, incumbent, absent_set, S, w4, regime,
                 admissible_status=ADMISSIBLE_SOLVER_STATUS_R3, *,
                 solver_policy=None, solver_options=None, evidence_sink=None,
                 evidence_context=None):
    """Recover slot q. `incumbent`: {task: caregiver} current committed plan for slot q.

    Returns `(assign {task: caregiver-or-None}, deviation, overtime_raw, status)`.
    `deviation` is the per-solve symmetric difference against the incumbent pair set (step churn
    contribution). `overtime_raw` is the solver's own overtime, UNROUNDED, kept only as a
    cross-check against the schedule-recomputed value; the authoritative overtime numbers in this
    module are always recomputed from the schedule.

    Raises `RollingSolveFailure` instead of returning None (REQ-2). The previous fail-soft return
    left the slot on its pre-disruption assignment and then counted it as recovered.
    """
    if regime not in REGIMES:
        raise RollingSolveFailure(
            f"slot {q} regime={regime!r} w4={w4}: unknown regime; expected {REGIMES}")
    opts = dict({"time_limit": SOLVER_TIME_LIMIT, "presolve": False}
                if solver_options is None else solver_options)
    if opts.get("presolve") is not False:
        raise RollingSolveFailure(
            f"slot {q} regime={regime} w4={w4}: presolve must be false because the known "
            "HiGHS presolve feasible-fixture defect is unresolved")
    if "time_limit" not in opts or not np.isfinite(float(opts["time_limit"])) \
            or float(opts["time_limit"]) <= 0:
        raise RollingSolveFailure(
            f"slot {q} regime={regime} w4={w4}: time_limit must be positive and finite")
    if solver_policy is not None:
        admissible_status = solver_policy.admissible_codes
        if set(admissible_status) != {0}:
            raise RollingDesignUnregistered(
                f"R3 is code-0-only; manifest admitted {sorted(admissible_status)}")
    tasks = [t for t in incumbent if t in parser.tasks]
    if not tasks:
        return {}, 0, 0.0, 0
    avail = {n for n in S if n not in absent_set and q in parser.nurses[n].slots}
    cand = {}
    for t in tasks:
        full = sorted(n for n in parser.tasks[t].candidate_nurses if n in avail)
        st = incumbent.get(t)
        incumbent_available = st is not None and st not in absent_set and st in avail
        if incumbent_available and st not in full:
            raise RollingSolveFailure(
                f"slot {q} regime={regime} w4={w4}: incumbent edge task={t!r}, "
                f"caregiver={st!r} is available but is not in the task's eligible candidate "
                "list. A1 may pin an incumbent only when it is BOTH available and eligible.")
        if regime == "A1" and incumbent_available:
            cand[t] = [st]
        else:
            cand[t] = full
        if regime == "A1" and not set(cand[t]).issubset(set(full)):
            raise RollingSolveFailure(
                f"slot {q} regime={regime} w4={w4}: A1 candidate set {cand[t]!r} is not a "
                f"subset of A2 candidate set {full!r} for task {t!r}")
    dur = {t: parser.tasks[t].duration + TRAVEL_PER_TASK for t in tasks}
    xbar = {}
    for t in tasks:
        n = incumbent[t]
        if n not in absent_set and n in avail:
            xbar[(n, t)] = 1
    nurses = sorted(avail)
    xpairs = [(i, t) for t in tasks for i in cand[t]]
    if not xpairs:
        raise RollingSolveFailure(
            f"slot {q} regime={regime} w4={w4}: EMPTY candidate structure — no task has any "
            f"available caregiver (|tasks|={len(tasks)}, |absent|={len(absent_set)}, "
            f"|avail|={len(avail)}). The design assumed a non-empty candidate set; REQ-2 forbids "
            f"continuing, and the pre-S2 code silently kept the pre-disruption assignment here.")
    xidx = {p: k for k, p in enumerate(xpairs)}
    nO = {i: len(xpairs) + a for a, i in enumerate(nurses)}
    nu = {t: len(xpairs) + len(nurses) + b for b, t in enumerate(tasks)}
    N = len(xpairs) + len(nurses) + len(tasks)
    c = np.zeros(N)
    for (i, t), k in xidx.items():
        c[k] = w4 * (1.0 - 2.0 * (1.0 if (i, t) in xbar else 0.0))
    for i in nurses:
        c[nO[i]] = W2
    for t in tasks:
        c[nu[t]] = W5
    rows, lb, ub = [], [], []
    for t in tasks:
        r = np.zeros(N)
        for i in cand[t]:
            r[xidx[(i, t)]] = 1.0
        r[nu[t]] = 1.0
        rows.append(r); lb.append(1.0); ub.append(1.0)
    for i in nurses:
        r = np.zeros(N)
        for t in tasks:
            if (i, t) in xidx:
                r[xidx[(i, t)]] = dur[t]
        r[nO[i]] = -1.0
        rows.append(r); lb.append(-np.inf); ub.append(H_STD)
    for i in nurses:
        r = np.zeros(N)
        for t in tasks:
            if (i, t) in xidx:
                r[xidx[(i, t)]] = dur[t]
        rows.append(r); lb.append(-np.inf); ub.append(H_MAX)
    cons = LinearConstraint(np.array(rows), np.array(lb), np.array(ub))
    integ = np.zeros(N)
    for k in range(len(xpairs)):
        integ[k] = 1
    for t in tasks:
        integ[nu[t]] = 1
    lo = np.zeros(N); hi = np.ones(N)
    for i in nurses:
        hi[nO[i]] = np.inf
    started = time.monotonic()
    try:
        res = milp(c=c, constraints=cons, integrality=integ, bounds=Bounds(lo, hi),
                   options=opts)
    except Exception as exc:
        raise RollingSolveFailure(
            f"slot {q} regime={regime} w4={w4}: scipy.optimize.milp raised "
            f"{type(exc).__name__}: {exc}") from exc
    wall = time.monotonic() - started
    status = int(getattr(res, "status", -1))
    if not res.success or res.x is None:
        raise RollingSolveFailure(
            f"slot {q} regime={regime} w4={w4}: solver did not succeed "
            f"(status={status}, message={getattr(res, 'message', None)!r}).")
    if status not in admissible_status:
        raise RollingSolveFailure(
            f"slot {q} regime={regime} w4={w4}: solver status {status} is not in this stage's "
            f"admissible set {sorted(admissible_status)} (REQ-13).")
    x = res.x
    if not np.all(np.isfinite(x)):
        raise RollingSolveFailure(f"slot {q} regime={regime} w4={w4}: non-finite solution vector.")
    assign = {}
    for t in tasks:
        picked = [i for i in cand[t] if x[xidx[(i, t)]] > 0.5]
        is_unserved = x[nu[t]] > 0.5
        if len(picked) + int(is_unserved) != 1:
            raise RollingSolveFailure(
                f"slot {q} regime={regime} w4={w4}: task {t!r} violates service closure "
                f"(picked={picked!r}, unserved_value={x[nu[t]]!r})")
        assign[t] = None if is_unserved else picked[0]
    loads = defaultdict(float)
    for t, who in assign.items():
        if who is None:
            continue
        if who not in cand[t]:
            raise RollingSolveFailure(
                f"slot {q} regime={regime} w4={w4}: task {t!r} assigned to {who!r}, "
                f"outside candidates {cand[t]!r}")
        loads[who] += float(dur[t])
    allowance = math.ulp(float(H_MAX)) * 8
    overloaded = {who: value for who, value in loads.items()
                  if value > H_MAX + allowance}
    if overloaded:
        raise RollingSolveFailure(
            f"slot {q} regime={regime} w4={w4}: schedule feasibility check found "
            f"capacity overloads {overloaded!r}; H_MAX={H_MAX}, allowance={allowance!r}")
    chosen = {(assign[t], t) for t in tasks if assign[t] is not None}
    deviation = len(chosen ^ set(xbar.keys()))
    overtime_raw = float(sum(max(0.0, float(x[nO[i]])) for i in nurses))
    if not np.isfinite(overtime_raw):
        raise RollingSolveFailure(f"slot {q} regime={regime} w4={w4}: non-finite overtime.")
    overtime_schedule = float(sum(max(0.0, value - H_STD) for value in loads.values()))
    if abs(overtime_raw - overtime_schedule) > SOLVER_READBACK_ABS_TOL:
        raise RollingSolveFailure(
            f"slot {q} regime={regime} w4={w4}: solver overtime {overtime_raw!r} does not "
            f"close to schedule-derived overtime {overtime_schedule!r} within the "
            f"pre-registered solver-readback tolerance {SOLVER_READBACK_ABS_TOL!r}")
    primal = float(getattr(res, "fun", float("nan")))
    dual = float(getattr(res, "mip_dual_bound", float("nan")))
    if not np.isfinite(primal) or not np.isfinite(dual):
        raise RollingSolveFailure(
            f"slot {q} regime={regime} w4={w4}: code-0 solve lacks finite primal/dual "
            f"(P={primal!r}, B={dual!r})")
    dev_const = float(w4) * len(xbar)
    if s2_core is not None:
        full_gap, gap_undefined = s2_core.full_objective_gap(
            primal, dual, dev_const,
            where=f"R3 seed={(evidence_context or {}).get('seed')} slot={q} "
                  f"regime={regime} w4={w4}")
        assign_hash = s2_core.assignment_hash(assign)
    else:
        denom = abs(primal + dev_const)
        full_gap = (0.0 if denom < 1e-12 and abs(primal - dual) < 1e-12
                    else (primal - dual) / denom)
        gap_undefined = None
        payload = json.dumps(sorted((repr(t), repr(assign[t])) for t in assign))
        assign_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if full_gap is None and gap_undefined != "zero_full_objective_denominator":
        raise RollingSolveFailure(
            f"slot {q} regime={regime} w4={w4}: full-objective gap undefined/non-finite "
            f"(P={primal!r}, B={dual!r}, C={dev_const!r}, reason={gap_undefined!r})")
    if full_gap is not None and not np.isfinite(full_gap):
        raise RollingSolveFailure(
            f"slot {q} regime={regime} w4={w4}: full-objective gap is non-finite "
            f"(P={primal!r}, B={dual!r}, C={dev_const!r})")
    status_name = (solver_policy.check(status, where=f"R3 slot={q}")
                   if solver_policy is not None else "Optimal")
    if evidence_sink is not None:
        ctx = dict(evidence_context or {})
        evidence_sink.append(dict(
            **ctx, slot=q, regime=regime, w4=float(w4),
            solver_status_code=status, solver_status_name=status_name,
            solver_raw_message=str(getattr(res, "message", "")),
            obj_incumbent=primal, mip_dual_bound=dual,
            deviation_constant_C=dev_const,
            mip_gap_solver_raw=float(getattr(res, "mip_gap", 0.0)),
            gap_full_objective=(None if full_gap is None else float(full_gap)),
            gap_full_objective_undefined_reason=gap_undefined,
            stop_reason="proven_optimal",
            assignment_hash=assign_hash, feasibility_verified=True,
            assignment_json=json.dumps(
                sorted((str(t), None if who is None else int(who))
                       for t, who in assign.items()),
                separators=(",", ":")),
            schedule_overtime=overtime_schedule, solver_overtime=overtime_raw,
            solver_readback_abs_tolerance=SOLVER_READBACK_ABS_TOL,
            wall_time_s=float(wall), time_limit_s=float(opts["time_limit"]),
            solver_presolve=False,
        ))
    return assign, deviation, overtime_raw, status


# --------------------------------------------------------------------------------------------- #
#  One configuration over one seed's stream
# --------------------------------------------------------------------------------------------- #
def run_stream(parser, committed, S, regime, w4, absent_seq, epochs=None, seed=None,
               absence_rule=None, *, solver_policy=None, solver_options=None,
               evidence_sink=None):
    """Run ONE configuration against a PRE-COMPUTED absence sequence.

    `absent_seq` is supplied by the caller and is never derived here, which is what makes the four
    configurations of a seed comparable. Returns `(row, slot_rows, schedule_rows)`.
    """
    grid = list(EPOCHS if epochs is None else epochs)
    if not grid:
        raise RollingDesignUnregistered("epoch grid is empty")
    e0 = min(grid)
    if len(absent_seq) != len(grid):
        raise RollingFatal(f"absence sequence length {len(absent_seq)} != epoch grid {len(grid)}")
    cur = {q: dict(m) for q, m in committed.items()}
    absent = set()
    churn_total = 0
    n_slot_solves = 0
    n_epochs_executed = 0
    n_epochs_with_absence = 0
    ot_cumulative = Fraction(0)
    ot_cum_terms = []
    statuses = set()
    milp_gap = 0.0
    for e, a in zip(grid, absent_seq):
        n_epochs_executed += 1
        if a is not None:
            if a not in S:
                raise RollingFatal(f"epoch {e}: absentee {a!r} is not a known caregiver")
            absent.add(a)
            n_epochs_with_absence += 1
            for q in sorted(cur):
                if q < e:
                    continue
                assign, dev, ot_raw, status = solve_assign(
                    parser, q, cur[q], absent, S, w4, regime,
                    solver_policy=solver_policy, solver_options=solver_options,
                    evidence_sink=evidence_sink,
                    evidence_context=dict(seed=seed, epoch=e,
                                          absentee=("" if a is None else a),
                                          absence_rule=absence_rule,
                                          solve_index=n_slot_solves + 1))
                statuses.add(status)
                churn_total += dev
                n_slot_solves += 1
                for t, n in assign.items():
                    cur[q][t] = n
                recomputed = float(overtime_total(parser, {q: cur[q]}))
                milp_gap = max(milp_gap, abs(recomputed - ot_raw))
        term = overtime_total(parser, cur, q_min=e)
        ot_cum_terms.append(term)
        ot_cumulative += term

    ot_full = overtime_total(parser, cur)
    ot_recov = overtime_total(parser, cur, q_min=e0)
    ot_prefix_committed = overtime_total(parser, committed, q_max_excl=e0)
    ot_prefix_final = overtime_total(parser, cur, q_max_excl=e0)
    # FATAL identity guard, exact rational arithmetic, prefix from the COMMITTED plan (REQ-8).
    if ot_full - ot_recov != ot_prefix_committed:
        raise RollingValidationError(
            f"REQ-8 identity FAILED (regime={regime} w4={w4} seed={seed}): "
            f"ot_full_plan({ot_full}) - ot_recoverable_horizon({ot_recov}) = {ot_full - ot_recov} "
            f"!= ot_frozen_prefix({ot_prefix_committed}) computed from the committed plan. "
            f"(prefix recomputed on the final plan = {ot_prefix_final}.)")
    f_full, f_recov, f_prefix = float(ot_full), float(ot_recov), float(ot_prefix_committed)
    if f_full - f_recov != f_prefix:
        raise RollingValidationError(
            f"REQ-8 identity holds exactly but NOT in the persisted float representation "
            f"(regime={regime} w4={w4} seed={seed}): {f_full} - {f_recov} != {f_prefix}. "
            f"Per-slot overtime and per-cell loads are persisted so the exact values are "
            f"recoverable; the guard is fatal by design and the tolerance is NOT to be widened.")

    row = dict(
        seed=seed, regime=regime, w4=w4,
        absence_rule=absence_rule, absent_seq=absence_sequence_key(absent_seq),
        n_absentees=n_epochs_with_absence, epochs=absence_sequence_key(grid), e0=e0,
        ot_full_plan=f_full,
        ot_recoverable_horizon=f_recov,
        ot_frozen_prefix=f_prefix,
        ot_frozen_prefix_final_plan=float(ot_prefix_final),
        ot_cumulative=float(ot_cumulative),
        ot_cumulative_terms="|".join(f"{float(v):.6f}" for v in ot_cum_terms),
        W=f_recov, W_estimand=W_ESTIMAND, W_scope=W_SCOPE,
        U=unserved_count(cur), U_recoverable_horizon=unserved_count(cur, q_min=e0),
        D=symmetric_difference_D(cur, committed),
        D_recoverable_horizon=symmetric_difference_D(cur, committed, q_min=e0),
        dev_terminal_tasks=terminal_deviation_tasks(cur, committed),
        churn_total=churn_total,
        churn_per_step=(churn_total / n_epochs_executed) if n_epochs_executed else float("nan"),
        churn_per_step_definition=CHURN_PER_STEP_DEFINITION,
        n_epochs_executed=n_epochs_executed,
        n_epochs_with_absence=n_epochs_with_absence,
        n_slot_solves=n_slot_solves,
        solver_status_distinct="|".join(str(s) for s in sorted(statuses)) if statuses else "",
        solver_status_worst=(max(statuses) if statuses else 0),
        milp_vs_recomputed_ot_max_abs_diff=milp_gap,
    )

    ot_slot = overtime_by_slot(parser, cur)
    loads = cell_loads(parser, cur)
    per_slot_tasks = defaultdict(int)
    for q, m in cur.items():
        per_slot_tasks[q] += len(m)
    slot_rows = []
    for q in sorted(cur):
        slot_rows.append(dict(seed=seed, regime=regime, w4=w4, slot=q,
                              in_recoverable_horizon=int(q >= e0),
                              ot_slot=float(ot_slot.get(q, Fraction(0))),
                              n_tasks=per_slot_tasks[q],
                              n_unserved=sum(1 for v in cur[q].values() if v is None),
                              cell_loads="|".join(
                                  f"{who}:{float(L):.6f}"
                                  for (who, qq), L in sorted(loads.items()) if qq == q)))
    schedule_rows = [dict(seed=seed, regime=regime, w4=w4, slot=q, task=t,
                          caregiver=("" if who is None else who),
                          committed_caregiver=committed.get(q, {}).get(t, ""),
                          flat_duration=float(_flat_duration(parser, t)) if t in parser.tasks else float("nan"))
                     for q in sorted(cur) for t, who in sorted(cur[q].items())]
    return row, slot_rows, schedule_rows


# --------------------------------------------------------------------------------------------- #
#  Validator (REQ-3 / §4 global validation contract). Every check is fatal.
# --------------------------------------------------------------------------------------------- #
#: Names REQ-8's acceptance test requires to be present and persisted, plus REQ-7's `absent_seq`
#: and the quantities that keep the three deviation notions distinct.
REQUIRED_RAW_COLUMNS = (
    "ot_full_plan", "ot_recoverable_horizon", "ot_cumulative", "U", "D", "W",
    "churn_total", "churn_per_step", "ot_frozen_prefix", "absent_seq", "ot_cumulative_terms",
    "ot_frozen_prefix_final_plan", "U_recoverable_horizon", "D_recoverable_horizon",
    "dev_terminal_tasks", "n_epochs_executed", "n_epochs_with_absence", "n_slot_solves",
    "milp_vs_recomputed_ot_max_abs_diff", "absence_rule", "solver_status_distinct",
    "W_estimand", "W_scope", "churn_per_step_definition",
)


def validate_rolling(df, slot_df, sched_df, seeds, slots_by_seed,
                     admissible_status=ADMISSIBLE_SOLVER_STATUS_R3, *,
                     configurations=None, solver_df=None, snap_id=None,
                     design_manifest_sha256=None):
    """Run the whole validation contract. Returns the list of failures (empty == all checks ran
    without a violation). Never mutates its inputs. A violation is NOT a statement about
    scientific correctness (ground rule E1) — it is a shape/consistency violation."""
    fail = []
    seeds = list(seeds)
    absent_cols = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if absent_cols:
        fail.append(f"[columns] rolling_raw is missing required columns: {absent_cols}")
        return fail                      # every later check would raise on the missing names
    config_grid = CONFIGURATIONS if configurations is None else tuple(configurations)
    expected_rows = len(config_grid) * len(seeds)
    print(f"[shape]   expected={expected_rows} actual={len(df)}")
    if len(df) != expected_rows:
        fail.append(f"[shape] expected={expected_rows} actual={len(df)}")

    key = ["seed", "regime", "w4"]
    if df.duplicated(subset=key).any():
        fail.append(f"[keys] duplicate rows on {key}: "
                    f"{df[df.duplicated(subset=key, keep=False)][key].to_dict('records')}")

    designed = {(s, r, float(w)) for s in seeds for r, w in config_grid}
    actual = {(r.seed, r.regime, float(r.w4)) for r in df.itertuples()}
    missing = designed - actual
    extra = actual - designed
    if missing:
        fail.append(f"[coverage] {len(missing)} designed cells missing: {sorted(missing)[:8]}")
    if extra:
        fail.append(f"[coverage] {len(extra)} undesigned cells present: {sorted(extra)[:8]}")
    if snap_id is not None:
        for name, frame in (("rolling_raw", df), ("slot_overtime", slot_df),
                            ("schedule", sched_df)):
            if "snap_id" not in frame.columns or set(frame.snap_id.astype(str)) != {str(snap_id)}:
                fail.append(f"[hashes] {name}.snap_id missing or disagrees with {snap_id!r}")
            if "design_manifest_sha256" not in frame.columns or set(
                    frame.design_manifest_sha256.astype(str)) != {
                        str(design_manifest_sha256)}:
                fail.append(
                    f"[hashes] {name}.design_manifest_sha256 missing or disagrees")

    # REQ-7: one absence sequence per seed, shared by all four configurations.
    per_seed = df.groupby("seed").absent_seq.nunique()
    if len(per_seed) and int(per_seed.max()) != 1:
        bad = per_seed[per_seed != 1].to_dict()
        fail.append(f"[absence] absent_seq is not unique within a seed: {bad}")
    if df.absence_rule.nunique() > 1:
        fail.append(f"[absence] more than one absence rule in one artifact: "
                    f"{sorted(df.absence_rule.unique())}")

    # REQ-8: the partition identity, per row, on the PERSISTED values.
    for r in df.itertuples():
        lhs = r.ot_full_plan - r.ot_recoverable_horizon
        if lhs != r.ot_frozen_prefix:
            fail.append(f"[identity] seed={r.seed} {r.regime}/w4={r.w4}: "
                        f"{r.ot_full_plan} - {r.ot_recoverable_horizon} = {lhs} "
                        f"!= ot_frozen_prefix={r.ot_frozen_prefix}")
        if r.ot_frozen_prefix != r.ot_frozen_prefix_final_plan:
            fail.append(f"[identity] seed={r.seed} {r.regime}/w4={r.w4}: frozen prefix differs "
                        f"between committed ({r.ot_frozen_prefix}) and final plan "
                        f"({r.ot_frozen_prefix_final_plan}) — a policy touched q < {E0}")
        if r.D < r.dev_terminal_tasks:
            fail.append(f"[deviation] seed={r.seed} {r.regime}/w4={r.w4}: symmetric-difference "
                        f"D={r.D} < terminal deviation {r.dev_terminal_tasks}; the two quantities "
                        f"have been conflated or one is miscomputed")
        if r.n_epochs_executed and r.churn_per_step != r.churn_total / r.n_epochs_executed:
            fail.append(f"[deviation] seed={r.seed} {r.regime}/w4={r.w4}: churn_per_step "
                        f"inconsistent with {CHURN_PER_STEP_DEFINITION}")
        if r.W != getattr(r, W_ESTIMAND):
            fail.append(f"[estimand] seed={r.seed} {r.regime}/w4={r.w4}: W does not equal its "
                        f"declared estimand {W_ESTIMAND}")
        terms = [float(x) for x in str(r.ot_cumulative_terms).split("|") if x != ""]
        if len(terms) != r.n_epochs_executed:
            fail.append(f"[estimand] seed={r.seed} {r.regime}/w4={r.w4}: "
                        f"{len(terms)} ot_cumulative terms for {r.n_epochs_executed} epochs")
        elif abs(sum(terms) - r.ot_cumulative) > max(abs(r.ot_cumulative), 1.0) * 1e-6:
            fail.append(f"[estimand] seed={r.seed} {r.regime}/w4={r.w4}: ot_cumulative_terms sum "
                        f"to {sum(terms)}, column says {r.ot_cumulative}")
    # frozen prefix is policy-invariant within a seed
    for s, g in df.groupby("seed"):
        if g.ot_frozen_prefix.nunique() != 1:
            fail.append(f"[identity] seed={s}: ot_frozen_prefix differs across the "
                        f"{N_CONFIGURATIONS} configurations: {sorted(g.ot_frozen_prefix.unique())}")

    # REQ-13 solver status
    for r in df.itertuples():
        got = {int(x) for x in str(r.solver_status_distinct).split("|") if x != ""}
        if int(r.n_slot_solves) <= 0:
            fail.append(f"[solver] seed={r.seed} {r.regime}/w4={r.w4}: "
                        "no recovery slot was solved")
        if not got <= set(admissible_status):
            fail.append(f"[solver] seed={r.seed} {r.regime}/w4={r.w4}: statuses {sorted(got)} not "
                        f"within admissible {sorted(admissible_status)}")

    # numeric scan
    for name, frame in (("rolling_raw", df), ("slot_overtime", slot_df), ("schedule", sched_df)):
        for col in frame.columns:
            s = frame[col]
            if not pd.api.types.is_numeric_dtype(s):
                continue
            # An unserved task is deliberately persisted as an empty caregiver
            # cell.  pandas infers this identifier column as float after CSV
            # round-trip, so its NaN is schema data rather than a missing
            # numeric measurement.  The dedicated schedule-closure checks below
            # validate both the nullable identifier and the unserved counts.
            if name == "schedule" and col == "caregiver":
                continue
            if s.isna().any():
                fail.append(f"[numeric] {name}.{col} contains NaN/empty")
                continue
            arr = np.asarray(s.to_numpy(), dtype=float)
            if not np.all(np.isfinite(arr)):
                fail.append(f"[numeric] {name}.{col} contains inf")

    # slots: every committed slot present for every configuration
    for s in seeds:
        want = set(slots_by_seed[s])
        for regime, w4 in config_grid:
            got = set(slot_df[(slot_df.seed == s) & (slot_df.regime == regime)
                              & (slot_df.w4 == w4)].slot.tolist())
            if got != want:
                fail.append(f"[slots] seed={s} {regime}/w4={w4}: slot set mismatch, "
                            f"missing={sorted(want - got)[:8]} extra={sorted(got - want)[:8]}")
    # per-slot overtime must re-sum to the row-level full-plan overtime
    agg = slot_df.groupby(["seed", "regime", "w4"]).ot_slot.sum().to_dict()
    for r in df.itertuples():
        got = agg.get((r.seed, r.regime, float(r.w4)))
        if got is None:
            fail.append(f"[slots] no per-slot rows for seed={r.seed} {r.regime}/w4={r.w4}")
        elif abs(got - r.ot_full_plan) > max(abs(r.ot_full_plan), 1.0) * 1e-9:
            fail.append(f"[slots] seed={r.seed} {r.regime}/w4={r.w4}: per-slot overtime sums to "
                        f"{got}, row says ot_full_plan={r.ot_full_plan}")

    # Rebuild load from the persisted terminal schedule, then overtime from that load, and close
    # it through the per-slot table to rolling_raw.  This is deliberately independent of the
    # in-memory plan used by the solver.
    schedule_required = {"seed", "regime", "w4", "slot", "task", "caregiver",
                         "flat_duration"}
    if not schedule_required <= set(sched_df.columns):
        fail.append(f"[slots] schedule missing closure columns: "
                    f"{sorted(schedule_required - set(sched_df.columns))}")
    else:
        if sched_df.duplicated(subset=["seed", "regime", "w4", "slot", "task"]).any():
            fail.append("[keys] schedule has duplicate (seed,regime,w4,slot,task) rows")
        caregiver_numeric = pd.to_numeric(sched_df["caregiver"], errors="coerce")
        caregiver_text = sched_df["caregiver"].astype(str).str.strip()
        invalid_caregiver = caregiver_numeric.isna() & ~(
            sched_df["caregiver"].isna() | caregiver_text.eq(""))
        if invalid_caregiver.any():
            fail.append(
                f"[numeric] schedule.caregiver has "
                f"{int(invalid_caregiver.sum())} non-numeric non-empty identifier(s)")
        non_integral_caregiver = caregiver_numeric.notna() & (
            (caregiver_numeric - caregiver_numeric.round()).abs() > 1e-9)
        if non_integral_caregiver.any():
            fail.append(
                f"[numeric] schedule.caregiver has "
                f"{int(non_integral_caregiver.sum())} non-integral identifier(s)")
        persisted_unserved = sched_df.assign(
            _unserved=caregiver_numeric.isna().astype(int))
        unserved_by_row = persisted_unserved.groupby(
            ["seed", "regime", "w4"])["_unserved"].sum().to_dict()
        for rec in df.to_dict("records"):
            key_ = (rec["seed"], rec["regime"], float(rec["w4"]))
            if int(unserved_by_row.get(key_, -1)) != int(rec["U"]):
                fail.append(
                    f"[slots] schedule unserved count at {key_} is "
                    f"{unserved_by_row.get(key_)!r}, rolling_raw.U={rec['U']!r}")
        unserved_by_slot = persisted_unserved.groupby(
            ["seed", "regime", "w4", "slot"])["_unserved"].sum().to_dict()
        for rec in slot_df.to_dict("records"):
            key_ = (rec["seed"], rec["regime"], float(rec["w4"]), int(rec["slot"]))
            if int(unserved_by_slot.get(key_, -1)) != int(rec["n_unserved"]):
                fail.append(
                    f"[slots] schedule unserved count at {key_} is "
                    f"{unserved_by_slot.get(key_)!r}, "
                    f"slot_overtime.n_unserved={rec['n_unserved']!r}")
        rebuilt_load = defaultdict(float)
        for rec in sched_df.to_dict("records"):
            who = rec["caregiver"]
            if who == "" or pd.isna(who):
                continue
            rebuilt_load[(rec["seed"], rec["regime"], float(rec["w4"]),
                          int(rec["slot"]), int(float(who)))] += float(rec["flat_duration"])
        rebuilt_ot = defaultdict(float)
        for (seed, regime, w4, slot, _who), value in rebuilt_load.items():
            rebuilt_ot[(seed, regime, w4, slot)] += max(0.0, value - H_STD)
        for rec in slot_df.to_dict("records"):
            key_ = (rec["seed"], rec["regime"], float(rec["w4"]), int(rec["slot"]))
            got = rebuilt_ot.get(key_, 0.0)
            want = float(rec["ot_slot"])
            if abs(got - want) > max(abs(want), 1.0) * 1e-9:
                fail.append(
                    f"[slots] schedule->load->OT closure failed at {key_}: "
                    f"schedule={got}, slot_table={want}")

    # Every actual scipy solve has a structured sidecar row.  Aggregate status and solve count
    # must close back to rolling_raw, and the persisted assignment hash must replay.
    if solver_df is not None:
        side_required = {
            "seed", "regime", "w4", "epoch", "slot", "solve_index",
            "solver_status_code", "solver_status_name", "solver_raw_message",
            "obj_incumbent", "mip_dual_bound", "gap_full_objective",
            "gap_full_objective_undefined_reason", "stop_reason",
            "assignment_hash", "assignment_json", "feasibility_verified",
            "schedule_overtime", "solver_overtime", "solver_readback_abs_tolerance",
            "time_limit_s", "solver_presolve",
        }
        if not side_required <= set(solver_df.columns):
            fail.append(f"[solver] solver sidecar missing columns: "
                        f"{sorted(side_required - set(solver_df.columns))}")
        else:
            skey = ["seed", "regime", "w4", "epoch", "slot", "solve_index"]
            if solver_df.duplicated(subset=skey).any():
                fail.append(f"[keys] duplicate solver sidecar rows on {skey}")
            expected_solves = int(pd.to_numeric(df.n_slot_solves).sum())
            if len(solver_df) != expected_solves:
                fail.append(f"[solver] sidecar rows={len(solver_df)} != "
                            f"sum(raw.n_slot_solves)={expected_solves}")
            for rec in solver_df.to_dict("records"):
                where = (rec["seed"], rec["regime"], float(rec["w4"]),
                         rec["epoch"], rec["slot"], rec["solve_index"])
                try:
                    status = int(rec["solver_status_code"])
                except (TypeError, ValueError):
                    fail.append(f"[solver] {where}: invalid status code "
                                f"{rec['solver_status_code']!r}")
                    continue
                if status not in set(admissible_status):
                    fail.append(f"[solver] {where}: status={status} outside "
                                f"{sorted(admissible_status)}")
                if not str(rec["solver_status_name"]).strip() \
                        or not str(rec["solver_raw_message"]).strip():
                    fail.append(f"[solver] {where}: empty status name/raw message")
                if str(rec["stop_reason"]) != "proven_optimal":
                    fail.append(f"[solver] {where}: stop_reason={rec['stop_reason']!r}, "
                                "R3 is code-0-only")
                if rec["feasibility_verified"] not in (True, 1, "True", "true"):
                    fail.append(f"[solver] {where}: feasibility_verified is not true")
                if rec["solver_presolve"] not in (False, 0, "False", "false"):
                    fail.append(f"[solver] {where}: solver_presolve is not false")
                try:
                    recorded_readback_tol = float(rec["solver_readback_abs_tolerance"])
                except (TypeError, ValueError):
                    recorded_readback_tol = float("nan")
                if (not np.isfinite(recorded_readback_tol)
                        or recorded_readback_tol != SOLVER_READBACK_ABS_TOL):
                    fail.append(
                        f"[solver] {where}: solver_readback_abs_tolerance="
                        f"{rec['solver_readback_abs_tolerance']!r} does not equal the "
                        f"registered value {SOLVER_READBACK_ABS_TOL!r}")
                for col in ("obj_incumbent", "mip_dual_bound",
                            "schedule_overtime", "solver_overtime", "time_limit_s"):
                    try:
                        value = float(rec[col])
                    except (TypeError, ValueError):
                        value = float("nan")
                    if not np.isfinite(value):
                        fail.append(f"[numeric] solver sidecar {where} {col}={rec[col]!r}")
                gap = rec["gap_full_objective"]
                reason = rec["gap_full_objective_undefined_reason"]
                if pd.isna(gap):
                    if str(reason) != "zero_full_objective_denominator":
                        fail.append(f"[numeric] solver sidecar {where}: full gap is empty "
                                    f"without its zero-denominator reason")
                    elif abs(float(rec["obj_incumbent"]) -
                             float(rec["mip_dual_bound"])) > SOLVER_READBACK_ABS_TOL:
                        fail.append(f"[solver] {where}: zero-denominator gap is undefined but "
                                    "primal and dual differ")
                elif not np.isfinite(float(gap)):
                    fail.append(f"[numeric] solver sidecar {where}: full gap={gap!r}")
                elif not pd.isna(reason) and str(reason).strip():
                    fail.append(f"[solver] {where}: finite full gap also has undefined reason "
                                f"{reason!r}")
                if abs(float(rec["schedule_overtime"]) -
                       float(rec["solver_overtime"])) > SOLVER_READBACK_ABS_TOL:
                    fail.append(f"[solver] {where}: solver overtime does not close to schedule")
                try:
                    pairs = json.loads(rec["assignment_json"])
                    replay = {int(t): (None if who is None else int(who)) for t, who in pairs}
                    replay_hash = (s2_core.assignment_hash(replay) if s2_core is not None
                                   else hashlib.sha256(json.dumps(sorted(
                                       (repr(t), repr(replay[t])) for t in replay)
                                   ).encode("utf-8")).hexdigest())
                    if replay_hash != rec["assignment_hash"]:
                        fail.append(f"[hashes] solver sidecar {where}: assignment hash mismatch")
                except Exception as exc:
                    fail.append(f"[hashes] solver sidecar {where}: assignment replay failed: "
                                f"{type(exc).__name__}: {exc}")
            for r in df.itertuples():
                group = solver_df[
                    (solver_df.seed == r.seed) &
                    (solver_df.regime == r.regime) &
                    (solver_df.w4.astype(float) == float(r.w4))]
                got = {int(v) for v in group.solver_status_code.tolist()}
                declared = {int(v) for v in str(r.solver_status_distinct).split("|") if v != ""}
                if got != declared:
                    fail.append(f"[solver] seed={r.seed} {r.regime}/w4={r.w4}: "
                                f"sidecar statuses={sorted(got)} raw={sorted(declared)}")
    return fail


def _local_sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path):
    """Prefer the shared S2 helper so this stage and the others hash identically."""
    return _shared("sha256_file", _local_sha256_file)(path)


# --------------------------------------------------------------------------------------------- #
#  temp -> validate -> promote (per-file os.replace; NO atomicity is claimed)
# --------------------------------------------------------------------------------------------- #
def promote(staged, out_dir):
    """Move already-validated staged files into `out_dir`, rolling back files already moved if a
    later move raises. This is a sequence of per-file `os.replace` calls: it is NOT atomic across
    files, and no pair-atomicity or family-atomicity may be claimed for it."""
    backups, moved = {}, []
    try:
        for base, tmp in staged.items():
            dest = os.path.join(out_dir, base)
            if os.path.exists(dest):
                bk = dest + ".s2rollback"
                shutil.copy2(dest, bk)
                backups[dest] = bk
            os.replace(tmp, dest)
            moved.append(dest)
    except Exception:
        for dest in reversed(moved):
            bk = backups.get(dest)
            if bk and os.path.exists(bk):
                os.replace(bk, dest)
            elif os.path.exists(dest):
                os.remove(dest)
        raise
    finally:
        for bk in backups.values():
            if os.path.exists(bk):
                os.remove(bk)
    return moved


# --------------------------------------------------------------------------------------------- #
#  Stage driver
# --------------------------------------------------------------------------------------------- #
def frames_for_seed(parser, committed, S, seed, absence_rule, epochs=None, *,
                    configurations=None, solver_policy=None, solver_options=None,
                    evidence_sink=None):
    """Run all FOUR configurations of one seed against ONE absence sequence.

    Returns `(raw_rows, slot_rows, sched_rows, epoch_rows, absent_seq)`. The sequence is computed
    once, here, before any solve — that single fact is what REQ-7 is about, so both the stage driver
    and the fixtures go through this function rather than duplicating the ordering.
    """
    grid = list(EPOCHS if epochs is None else epochs)
    seq = absence_sequence(committed, seed, epochs=grid, rule=absence_rule)
    epoch_rows = [dict(seed=seed, absence_rule=absence_rule, epoch=e,
                       absent=("" if a is None else a), absent_seq=absence_sequence_key(seq))
                  for e, a in zip(grid, seq)]
    raw, slots, sched = [], [], []
    config_grid = CONFIGURATIONS if configurations is None else tuple(configurations)
    for regime, w4 in config_grid:
        r, sl, sc = run_stream(parser, committed, S, regime, w4, seq, epochs=grid,
                               seed=seed, absence_rule=absence_rule,
                               solver_policy=solver_policy, solver_options=solver_options,
                               evidence_sink=evidence_sink)
        raw.append(r); slots.extend(sl); sched.extend(sc)
    return raw, slots, sched, epoch_rows, seq


def build_frames(manifest_rows, absence_rule, epochs=None, quiet=False, *,
                 configurations=None, solver_policy=None, solver_options=None,
                 evidence_sink=None):
    """Run every seed x configuration. Returns (raw_df, slot_df, sched_df, epoch_df, slots_by_seed).
    No file is written here."""
    grid = list(EPOCHS if epochs is None else epochs)
    raw, slots, sched, epoch_rows = [], [], [], []
    slots_by_seed = {}
    for row in manifest_rows:
        seed = row["seed"]
        parser = HHCPFullParser(row["hhcp_path"])
        committed = {q: dict(m) for q, m in load_static(row["sol_path"]).items()}
        S = set(parser.nurses.keys())
        slots_by_seed[seed] = sorted(committed)
        r, sl, sc, ep, seq = frames_for_seed(
            parser, committed, S, seed, absence_rule, epochs=grid,
            configurations=configurations, solver_policy=solver_policy,
            solver_options=solver_options, evidence_sink=evidence_sink)
        raw.extend(r); slots.extend(sl); sched.extend(sc); epoch_rows.extend(ep)
        if not quiet:
            print(f"seed {seed} done (absent_seq={absence_sequence_key(seq)})", flush=True)
    return (pd.DataFrame(raw), pd.DataFrame(slots), pd.DataFrame(sched),
            pd.DataFrame(epoch_rows), slots_by_seed)


def main(argv=None):
    ap = argparse.ArgumentParser(description="rolling-horizon recovery over a disruption stream")
    ap.add_argument("--stage", required=True, choices=["R3"],
                    help="formal stage this invocation belongs to; required so that an accidental "
                         "bare invocation can no longer overwrite a canonical artifact")
    ap.add_argument("--manifest", required=True,
                    help="registered R3 JSON design manifest; no default is permitted")
    ap.add_argument("--snap-id", required=True,
                    help="64-hex SHA256 independently recomputed from the immutable SNAP archive")
    ap.add_argument("--out-dir", default=HERE)
    args = ap.parse_args(argv)
    design = load_r3_design(args.manifest, args.snap_id)
    if not os.path.isdir(args.out_dir):
        raise RollingDesignUnregistered(f"--out-dir does not exist: {args.out_dir!r}")

    print("=== overtime estimand definitions (REQ-8; reproduce verbatim in any caption) ===")
    for k, v in OT_ESTIMANDS.items():
        print(f"  {k}: {v}")
    print(f"  W = {W_ESTIMAND}, scope: {W_SCOPE}")
    print(f"  churn_per_step = {CHURN_PER_STEP_DEFINITION}")
    print("None of the four overtime estimands is 'the true value'; every ratio must name its "
          "estimand and slot scope.\n")

    man = pd.read_csv(design["input_manifest"])
    required = {"seed", "rho_target", "hhcp", "sol"}
    if not required <= set(man.columns):
        raise RollingDesignUnregistered(
            f"input manifest missing columns {sorted(required - set(man.columns))}")
    man = man[np.isclose(pd.to_numeric(man.rho_target), design["rho_target"])]
    input_dir = os.path.dirname(design["input_manifest"])
    rows = [dict(
        seed=r["seed"],
        hhcp_path=(r["hhcp"] if os.path.isabs(str(r["hhcp"]))
                   else os.path.join(input_dir, str(r["hhcp"]))),
        sol_path=(r["sol"] if os.path.isabs(str(r["sol"]))
                  else os.path.join(input_dir, str(r["sol"]))))
            for _, r in man.iterrows()]
    if not rows:
        raise RollingDesignUnregistered(
            f"no input-manifest rows at rho_target={design['rho_target']}")
    seeds = [r["seed"] for r in rows]
    if len(set(seeds)) != len(seeds):
        raise RollingDesignUnregistered(f"duplicate seed rows in R3 input manifest: {seeds!r}")
    for row in rows:
        for key in ("hhcp_path", "sol_path"):
            if not os.path.isfile(row[key]):
                raise RollingDesignUnregistered(
                    f"input file does not exist for seed={row['seed']}: {key}={row[key]!r}")

    solve_rows = []
    raw, slot, sched, epoch, slots_by_seed = build_frames(
        rows, design["absence_rule"], epochs=design["epochs"],
        configurations=design["configurations"], solver_policy=design["policy"],
        solver_options=design["solver_options"], evidence_sink=solve_rows)
    solver = pd.DataFrame(solve_rows)
    metadata = dict(
        stage="R3", snap_id=design["snap_id"],
        design_manifest_sha256=design["manifest_sha256"],
        input_manifest_sha256=sha256_file(design["input_manifest"]),
        solver_options_json=json.dumps(design["solver_options"], sort_keys=True,
                                       separators=(",", ":")),
        numeric_tolerances_json=json.dumps(design["numeric_tolerances"], sort_keys=True,
                                           separators=(",", ":")))
    for frame in (raw, slot, sched, epoch, solver):
        for name, value in metadata.items():
            frame[name] = value
    fails = validate_rolling(
        raw, slot, sched, seeds, slots_by_seed,
        admissible_status=design["policy"].admissible_codes,
        configurations=design["configurations"], solver_df=solver,
        snap_id=design["snap_id"], design_manifest_sha256=design["manifest_sha256"])
    if fails:
        raise RollingValidationError("validator FAILED; canonical outputs left untouched:\n  - "
                                     + "\n  - ".join(fails))

    tmpdir = tempfile.mkdtemp(prefix="rolling_stage_", dir=args.out_dir)
    try:
        staged = {}
        for base, frame in ((RAW_BASENAME, raw), (SLOT_BASENAME, slot),
                            (SCHEDULE_BASENAME, sched), (EPOCH_BASENAME, epoch),
                            (SOLVER_BASENAME, solver)):
            p = os.path.join(tmpdir, base)
            frame.to_csv(p, index=False)
            staged[base] = p
        # Re-read the LANDED bytes and validate THOSE, never the in-memory frames: the landed bytes
        # are the verification authority.
        fails = validate_rolling(pd.read_csv(staged[RAW_BASENAME]),
                                 pd.read_csv(staged[SLOT_BASENAME]),
                                 pd.read_csv(staged[SCHEDULE_BASENAME]),
                                 seeds, slots_by_seed,
                                 admissible_status=design["policy"].admissible_codes,
                                 configurations=design["configurations"],
                                 solver_df=pd.read_csv(staged[SOLVER_BASENAME]),
                                 snap_id=design["snap_id"],
                                 design_manifest_sha256=design["manifest_sha256"])
        if fails:
            raise RollingValidationError("post-write validator FAILED on the staged bytes; "
                                         "canonical outputs left untouched:\n  - "
                                         + "\n  - ".join(fails))
        for base in staged:
            print(f"[hashes] staged {base} sha256={sha256_file(staged[base])}")
        canonical_write = _shared("canonical_write", None)
        if canonical_write is None:
            # Local fallback: a per-file os.replace loop with rollback of files already moved.
            # NOT atomic across files; no pair- or family-atomicity is claimed either way.
            promote(staged, args.out_dir)
        else:
            def _validator(base):
                """Re-run the FULL contract against the temp bytes of `base`, holding the other
                three tables at their staged bytes. A no-op validator here would defeat the point
                of the shared write path."""
                def _v(temp_path):
                    frames = {b: staged[b] for b in staged}
                    frames[base] = temp_path
                    if base == EPOCH_BASENAME:
                        ep = pd.read_csv(temp_path)
                        missing = {"seed", "absence_rule", "epoch", "absent",
                                   "absent_seq", "snap_id",
                                   "design_manifest_sha256"} - set(ep.columns)
                        if missing:
                            raise RollingValidationError(
                                f"{base}: missing columns {sorted(missing)}")
                        if int(ep.groupby("seed").absent_seq.nunique().max()) != 1:
                            raise RollingValidationError(f"{base}: absent_seq not unique per seed")
                        return
                    bad = validate_rolling(pd.read_csv(frames[RAW_BASENAME]),
                                           pd.read_csv(frames[SLOT_BASENAME]),
                                           pd.read_csv(frames[SCHEDULE_BASENAME]),
                                           seeds, slots_by_seed,
                                           admissible_status=design["policy"].admissible_codes,
                                           configurations=design["configurations"],
                                           solver_df=pd.read_csv(frames[SOLVER_BASENAME]),
                                           snap_id=design["snap_id"],
                                           design_manifest_sha256=design["manifest_sha256"])
                    if bad:
                        raise RollingValidationError(
                            f"{base}: validator FAILED on the landed temp bytes:\n  - "
                            + "\n  - ".join(bad))
                return _v

            for base in (RAW_BASENAME, SLOT_BASENAME, SCHEDULE_BASENAME, EPOCH_BASENAME,
                         SOLVER_BASENAME):
                dest = os.path.join(args.out_dir, base)
                with canonical_write(dest, _validator(base), stage="R3",
                                     diagnostics={"artifact": base,
                                                  "absence_rule": design["absence_rule"],
                                                  "snap_id": design["snap_id"],
                                                  "design_manifest_sha256":
                                                      design["manifest_sha256"]}) as tmp:
                    shutil.copyfile(staged[base], tmp)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    for base in (RAW_BASENAME, SLOT_BASENAME, SCHEDULE_BASENAME, EPOCH_BASENAME,
                 SOLVER_BASENAME):
        p = os.path.join(args.out_dir, base)
        print(f"[hashes] promoted {base} sha256={sha256_file(p)}")
    print(f"[hashes] R3 design manifest sha256={design['manifest_sha256']}")
    print(f"[hashes] SNAP id={design['snap_id']}")

    g = raw.groupby(["regime", "w4"]).agg(
        ot_full_plan=("ot_full_plan", "mean"),
        ot_recoverable_horizon=("ot_recoverable_horizon", "mean"),
        ot_frozen_prefix=("ot_frozen_prefix", "mean"),
        ot_cumulative=("ot_cumulative", "mean"),
        D=("D", "mean"), dev_terminal_tasks=("dev_terminal_tasks", "mean"),
        churn_total=("churn_total", "mean"), churn_per_step=("churn_per_step", "mean"),
        U=("U", "mean")).round(3)
    print(f"\n=== rolling horizon, {len(design['epochs'])}-epoch stream, "
          f"rho={design['rho_target']}, absence rule={design['absence_rule']} "
          f"(mean over seeds) ===")
    print(g.to_string())
    print("\nEvery column above names an estimand defined in OT_ESTIMANDS / this module's "
          "docstring. No column is 'the' overtime.")


if __name__ == "__main__":
    main()
