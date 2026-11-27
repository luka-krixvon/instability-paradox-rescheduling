"""
scalability.py (P0 / E5) — runtime and gap vs instance size for the exact recovery MILP, plus an
exact-vs-metaheuristic comparison that locates the exact-tractability frontier.

Reviewer objection answered: "you claim the problem is NP-hard and solve it with an exact MILP,
but show no runtime-vs-size curve and use only ~16 workers."

What is scaled and why
----------------------
The recovery MILP decomposes per slot and the slots are INDEPENDENT (no variable/constraint
couples two slots: overtime, capacity and deviation are all per (nurse, slot), and each task is
fixed to its slot). So whole-instance solve time = sum of per-slot times, and the NP-hard locus is
the SINGLE-SLOT capacitated generalized-assignment problem (GAP). We scale that per-slot object:
``T_q`` tasks packed into ``M`` workers at committed cap ``RHO_COMMIT*H_MAX``, then a fraction
``ABSENT_FRAC`` of workers is removed at the disruption, so the number of orphaned tasks needing
re-assignment grows with ``T_q``. ``T_q*15`` is a per-slot-times-horizon accounting label (valid
only because slots are provably independent), not a jointly-solved instance size.

Note on slots: ``slot`` here is packing DATA, not a decision. Where allowed slots matter, the
semantics is set membership ``q in theta_j`` (lin2025 Eq. 5); ``|theta_j| = 1`` in the data
shipped with this repository, which is a property of that data and not a general equality. This
script synthesizes its own single-slot GAP ladder and reads no ``rho_manifest.csv`` and no
``H_*``/``S_*`` pair.

Two experiments:
  [A] SIZE sweep at an operationally-realistic post-disruption tightness: orphans and ``n_bin``
      grow with ``T_q``.
  [B] TIGHTNESS sweep at fixed ``T_q``, varying realized tightness across saturation: locates
      where exact solving stops proving optimality within the time limit.

--------------------------------------------------------------------------------------------
S2 changes (REQ-13 / REQ-14 / REQ-15 of ``audit_outputs/rerun_plan_v2_1.md``)
--------------------------------------------------------------------------------------------
1. **Base-instance path fixed.** It previously resolved to ``experiments/HHCP-...csv``, which does
   not exist; the real file is ``experiments/data_problems_solutions/HHCP-...csv``. It is now
   resolved from ``__file__`` via :func:`s2_core.paths` and **existence-checked with no fallback**.
2. **Solver status is gated by the stage's design manifest** (``--manifest``), not by a literal in
   this file, and the status is persisted as an integer **code plus the library's own name**.
   There is no default manifest: no manifest, no run.
3. **A time-limit incumbent is never called optimal.** ``proven_optimal`` is a persisted boolean
   derived from the manifest's ``optimal_statuses``; the primal column is named ``obj_incumbent``;
   :func:`s2_core.assert_not_labeled_optimal` refuses to emit an "optimal"-flavoured label for a
   non-proven row.
4. **Full-objective gap.** The linearization drops ``C = w4 * |xbar|``. A relative gap is not
   translation-invariant, so ``res.mip_gap`` is the wrong quantity. ``P``, ``B`` and ``C`` are all
   persisted and the gap is recomputed as ``((P+C) - (B+C)) / |P+C|``.
5. **Run-level rows.** Every exact repeat and every heuristic repeat is one row. Aggregates are
   computed *from* those rows and record which rows they came from. Previously the reported time
   was the median of 3 repeats while the status/gap/nodes came from the last repeat.
6. **VNS and ALNS each own a replayable, persisted seed and RNG object.** Previously one
   ``random.Random`` was passed to both, coupling their streams.
7. **Iterations budget vs executed, evaluation count, stop reason, assignment hash and the SIGNED
   gap are persisted.** The ``max(0, gap)`` clamp is removed: a negative gap against a
   proven-optimal exact solve now **fails** the stage.
8. **REQ-15.** ``rho_avail``, ``rho_commit``, ``rho_commit_target``, ``M`` and ``C`` already
   existed and are KEPT — adding them is not an achievement and is not claimed as one. What is new
   is the **realized** candidate degree (min/mean/max after eligibility filtering, with
   ``cand_degree_max <= M`` asserted) and a declared confound-control interface: the manifest picks
   ``band`` (validator fails rows whose realized ``rho_avail`` leaves the pre-registered band) or
   ``covariate`` (realized covariates persisted for stratification/regression).

Formal R4 execution remains SNAP-gated.  ``--snap-id`` is mandatory for a formal run and must be
the 64-hex SHA256 of the externally verified, content-addressed SNAP archive.  The manifests
pre-register this binding rule without embedding the archive's own hash inside itself.  ``--emit``
routes every write through :func:`s2_core.canonical_write` +
:func:`s2_core.validate_stage_output`; ``--outdir`` selects the isolated output directory rather
than being silently ignored.

Fairness caveats (unchanged, and now machine-readable): the heuristics run under a **fixed
iteration budget** with no wall-clock cap (``budget_regime`` is a persisted column), so at large
``T_q`` they under-search; their rising gap is a budgeted-search artifact, not a matched
wall-clock race. NP-hardness (Prop 2) is worst-case; ``nodes=1`` in the easy regime means HiGHS
closes a real integer gap at the root via presolve and cuts.

Isolation: every solve runs in a fresh subprocess. Exact time wraps the ``milp()`` call only.

    cd experiments && python3 journal_p0/scalability.py --selftest        # tiny fixtures only
    cd experiments && python3 journal_p0/scalability.py --manifest <size.json> \
      --manifest-tightness <tight.json> --snap-id <closed-snap-id> --outdir <isolated-dir> --emit
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS = os.path.dirname(_HERE)
for _p in (_HERE, _EXPERIMENTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import s2_core                                                          # noqa: E402
from s2_core import COL, PipelineFailure, require, require_finite       # noqa: E402
from SubgroupOptimizer import HHCPFullParser, H_MAX, H_STD              # noqa: E402
from displacement_milp import TRAVEL_PER_TASK, W2, W5                   # noqa: E402


def _load_local_module(basename: str, attr_name: str):
    """Import a sibling module by absolute file path derived from ``__file__``.

    ``experiments/compare_baselines.py`` and ``experiments/journal_p0/compare_baselines.py`` share
    a basename, and which one a bare ``import compare_baselines`` resolves to depends on the
    working directory. Loading by path removes that ambiguity: this script always gets the
    instrumented ``journal_p0`` module.
    """
    path = os.path.join(_HERE, basename + ".py")
    require(os.path.isfile(path), "sibling module not found next to this script",
            basename=basename, expected=os.path.relpath(path, _EXPERIMENTS))
    spec = importlib.util.spec_from_file_location(attr_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[attr_name] = module
    spec.loader.exec_module(module)
    return module


cb = _load_local_module("compare_baselines", "journal_p0_compare_baselines")

_BASE_INSTANCE_FOR_MARKER = "HHCP-patient-65-task-256-nurse-59.csv"


def _base_instance_withheld() -> bool:
    """True only if BASE_INSTANCE_WITHHELD at the tree root names the base CSV."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    try:
        with open(os.path.join(root, "BASE_INSTANCE_WITHHELD"),
                  encoding="utf-8") as handle:
            named = {line.strip() for line in handle
                     if line.strip() and not line.lstrip().startswith("#")}
    except OSError:
        return False
    return _BASE_INSTANCE_FOR_MARKER in named


# Public reproducibility release: --selftest builds tiny synthetic fixtures and
# never opens the base instance, so importing this module must not die merely
# because the file is withheld.  Every mode that really reads it is re-checked
# in main() below with the original message, so REQ-11 fault injection is
# unchanged whenever the marker is absent -- and the sealed tree has no marker.
PATHS = s2_core.paths(require_base_instance=not _base_instance_withheld())                 # resolves + existence-checks the base instance CSV
SRC = PATHS.base_instance               # FIX (1): was experiments/HHCP-...csv, which never existed
# Public release: --selftest never reads the base instance; it only declared it
# as provenance, and a file that is not here cannot be hashed.  The full R4
# sweep below still declares it, and main() refuses every non-selftest mode
# without it, so nothing is relaxed outside the synthetic fixture.
_SELFTEST_SRC_INPUTS = [] if _base_instance_withheld() else [SRC]

STAGE_SIZE = "R4-size"
STAGE_TIGHT = "R4-tightness"
STAGE_SELFTEST = "S2-selftest-corelib"   # not an R-stage; emits fixtures only, never canonical

# Scientific model constants; the formal grid, solver options and repeat counts live only in the
# two R4 design manifests and have no executable fallback in this module.
CAND_PER_TASK = 12
W4 = 50.0
HEUR_REPEATS = 3          # stochastic repeats per algorithm — one row each
SOLVER_READBACK_ABS_TOL = 1e-6


# --------------------------------------------------------------------------------------------
# design manifest (REQ-13 A: the admissible status set is declared BEFORE the run)
# --------------------------------------------------------------------------------------------


def load_design(manifest_path: str, stage: str,
                cli_snap_id: Optional[str] = None) -> Dict[str, Any]:
    """Load the stage design manifest and build its solver policy. No manifest, no run."""
    require(manifest_path, "a design manifest is required; there is no default admissible "
                           "solver-status set in this file (REQ-13 A)", stage=stage)
    require(os.path.isfile(manifest_path), "design manifest not found",
            path=manifest_path, stage=stage)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    policy = s2_core.StageSolverPolicy.from_manifest(
        manifest, stage=stage, manifest_path=manifest_path,
        manifest_sha256=s2_core.sha256_file(manifest_path))
    s2_core.require_keys(manifest, ["design"], where="design manifest root")
    design = manifest["design"]
    s2_core.require_keys(design, ["allow_empty_candidate_sets", "confound_control"],
                         where="design manifest ['design']")
    confound = design["confound_control"]
    s2_core.require_keys(confound, ["mode"], where="design manifest ['design']['confound_control']")
    require(confound["mode"] in ("band", "covariate"),
            "confound_control.mode must be declared as 'band' or 'covariate' (REQ-15 test 3)",
            stage=stage, mode=confound["mode"])
    band = None
    if confound["mode"] == "band":
        s2_core.require_keys(confound, ["rho_avail_band"],
                             where="design manifest confound_control(band)")
        raw = confound["rho_avail_band"]
        require(isinstance(raw, list) and len(raw) == 2,
                "rho_avail_band must be [lo, hi]", stage=stage, value=raw)
        lo = require_finite(raw[0], "rho_avail_band[0]", stage=stage)
        hi = require_finite(raw[1], "rho_avail_band[1]", stage=stage)
        require(lo < hi, "rho_avail_band must satisfy lo < hi", stage=stage, lo=lo, hi=hi)
        band = (lo, hi)
    solver = manifest["solver"]
    formal_r4 = stage in (STAGE_SIZE, STAGE_TIGHT)
    presolve = solver.get("presolve", True)
    require(isinstance(presolve, bool), "solver.presolve must be a boolean",
            stage=stage, value=presolve)
    if formal_r4:
        require(presolve is False,
                "formal R4 manifests must pre-register presolve=false; it may not be selected "
                "after observing a solve", stage=stage, presolve=presolve)
        require(policy.time_limit_s is not None,
                "formal R4 time_limit_s must be declared in the stage manifest", stage=stage)
        require(isinstance(cli_snap_id, str) and cli_snap_id.strip(),
                "--snap-id is mandatory for a formal R4 run", stage=stage)
        normalized_snap = cli_snap_id.strip().lower()
        require(len(normalized_snap) == 64
                and all(ch in "0123456789abcdef" for ch in normalized_snap),
                "--snap-id must be the 64-hex SHA256 of the externally verified SNAP archive",
                stage=stage, cli_snap_id=cli_snap_id)
        require(manifest.get("snap_binding") == "cli_content_addressed_archive_sha256",
                "formal R4 manifest does not pre-register the non-self-referential SNAP binding "
                "rule", stage=stage, snap_binding=manifest.get("snap_binding"))
        s2_core.require_keys(
            design, ["grid", "numeric_tolerances"],
            where="formal R4 design manifest ['design']")
        grid = design["grid"]
        require(isinstance(grid, dict), "design.grid must be an object", stage=stage)
        numeric_tolerances = design["numeric_tolerances"]
        require(isinstance(numeric_tolerances, dict)
                and numeric_tolerances.get("solver_overtime_readback_abs")
                == SOLVER_READBACK_ABS_TOL
                and numeric_tolerances.get("capacity_allowance_rule")
                == "math.ulp(H_MAX)*8",
                "formal R4 numeric tolerances must separate solver readback from the "
                "physical H_MAX feasibility allowance",
                stage=stage, declared=numeric_tolerances,
                executable_solver_readback_abs=SOLVER_READBACK_ABS_TOL)
        require(confound["mode"] == "covariate",
                "R4 confound control is pre-registered as covariate adjustment, not a post-hoc "
                "rho_avail band", stage=stage, mode=confound["mode"])
        s2_core.require_keys(confound, ["covariates"],
                             where="formal R4 confound_control(covariate)")
        covariates = confound["covariates"]
        require(isinstance(covariates, list), "confound_control.covariates must be a list",
                stage=stage)
        required_covariates = {
            COL.RHO_AVAIL, COL.CAND_DEG_MIN, COL.CAND_DEG_MEAN, COL.CAND_DEG_MAX,
        }
        require(required_covariates.issubset(set(covariates)),
                "formal R4 covariate plan must include realized rho_avail and candidate-degree "
                "min/mean/max", stage=stage, declared=covariates,
                missing=sorted(required_covariates - set(covariates)))
        if stage == STAGE_SIZE:
            require(policy.admissible_codes == {0} and policy.optimal_codes == {0}
                    and not policy.accepts_time_limit_incumbent,
                    "R4-size admits code 0 only and every accepted row must be proven optimal",
                    admissible=sorted(policy.admissible_codes),
                    optimal=sorted(policy.optimal_codes),
                    accepts_time_limit_incumbent=policy.accepts_time_limit_incumbent)
        else:
            require(policy.admissible_codes == {0, 1} and policy.optimal_codes == {0}
                    and policy.accepts_time_limit_incumbent,
                    "R4-tightness admits only code 0 plus a guarded code-1 time-limit incumbent; "
                    "code 1 is never optimal", admissible=sorted(policy.admissible_codes),
                    optimal=sorted(policy.optimal_codes),
                    accepts_time_limit_incumbent=policy.accepts_time_limit_incumbent)
    else:
        grid = design.get("grid")
        covariates = confound.get("covariates", [])
        numeric_tolerances = {
            "solver_overtime_readback_abs": SOLVER_READBACK_ABS_TOL,
            "capacity_allowance_rule": "math.ulp(H_MAX)*8",
        }
    effective_snap = (cli_snap_id.strip().lower() if formal_r4 else manifest.get("snap_id"))
    return {"manifest": manifest, "policy": policy, "design": design,
            "confound_mode": confound["mode"], "rho_avail_band": band,
            "covariates": covariates, "grid": grid, "presolve": presolve,
            "solver_readback_abs_tol":
                float(numeric_tolerances["solver_overtime_readback_abs"]),
            "manifest_path": manifest_path,
            "manifest_sha256": policy.manifest_sha256,
            "snap_id": effective_snap}


# --------------------------------------------------------------------------------------------
# instance construction
# --------------------------------------------------------------------------------------------


def base_durations(src: str = SRC) -> List[float]:
    """Real task service durations (minutes) from the base instance CSV. Read-only."""
    parser = HHCPFullParser(src)
    durs = [parser.tasks[t].duration for t in sorted(parser.tasks)]
    require(durs, "base instance yielded no task durations", src=os.path.basename(src))
    for i, d in enumerate(durs):
        require_finite(d, "base duration[{}]".format(i), src=os.path.basename(src))
    return durs


def build_instance(T_q: int, rho_commit: float, absent_frac: float, C_cand: int, seed: int,
                   *, durs_pool: Optional[Sequence[float]] = None,
                   allow_empty_candidate_sets: bool = True) -> Dict[str, Any]:
    """Construct a per-slot recovery subproblem of size ``T_q``.

    The committed incumbent packs ``T_q`` tasks into ``M`` workers at ``cap = rho_commit*H_MAX``;
    then ``max(1, round(absent_frac*M))`` workers are removed, so orphans scale with ``T_q``.
    Returns the
    realized post-disruption tightness ``rho_avail`` **and** the realized candidate degree
    (REQ-15), measured after eligibility filtering.
    """
    import random as _random
    rng = _random.Random(1000 * seed + T_q)          # instance RNG; separate from every algorithm
    pool = list(durs_pool) if durs_pool is not None else base_durations()
    tasks = list(range(T_q))
    dur = {t: pool[rng.randrange(len(pool))] + TRAVEL_PER_TASK for t in tasks}

    cap = rho_commit * H_MAX
    slack = s2_core.cap_fp_slack(cap)                 # NOT 1e-6
    load: List[float] = []
    committed: Dict[int, int] = {}
    for t in sorted(tasks, key=lambda t: -dur[t]):
        placed = False
        for w in sorted(range(len(load)), key=lambda w: -load[w]):     # best-fit: fullest first
            if load[w] + dur[t] <= cap + slack:
                committed[t] = w
                load[w] += dur[t]
                placed = True
                break
        if not placed:
            load.append(dur[t])
            committed[t] = len(load) - 1
    M_committed = len(load)
    s2_core.require_flat_feasible({w: load[w] for w in range(M_committed)}, cap,
                                  where="build_instance committed pack")

    cand: Dict[int, List[int]] = {}
    for t in tasks:                                   # connected candidate graph
        others = [w for w in range(M_committed) if w != committed[t]]
        rng.shuffle(others)
        cand[t] = sorted(set(others[:max(0, C_cand - 1)]) | {committed[t]})

    k_absent = max(1, round(absent_frac * M_committed))
    absent = set(sorted(range(M_committed), key=lambda w: -load[w])[:k_absent])
    avail = [w for w in range(M_committed) if w not in absent]
    cand = {t: [w for w in cand[t] if w not in absent] for t in tasks}   # A2 eligibility filter
    xbar = {t: committed[t] for t in tasks if committed[t] not in absent}
    orphans = sum(1 for t in tasks if committed[t] in absent)
    no_cand = sum(1 for t in tasks if not cand[t])

    if not allow_empty_candidate_sets:
        s2_core.require_nonempty_candidates(cand, where="build_instance T_q={}".format(T_q))

    # ---- REQ-15: REALIZED candidate degree, after eligibility filtering ----
    degrees = [len(cand[t]) for t in tasks]
    deg_min, deg_max = (min(degrees), max(degrees)) if degrees else (0, 0)
    deg_mean = (sum(degrees) / float(len(degrees))) if degrees else 0.0
    require(deg_max <= len(avail),
            "realized candidate degree exceeds the available workforce", T_q=T_q, seed=seed,
            cand_degree_max=deg_max, M=len(avail))

    n_bin = sum(len(cand[t]) for t in tasks) + T_q
    n_cont = len(avail)
    n_cons = T_q + 2 * len(avail)
    demand = float(sum(dur.values()))
    require(avail, "every worker was removed at the disruption; the design assumes a non-empty "
                   "remaining workforce", T_q=T_q, seed=seed, M_committed=M_committed)
    rho_avail = demand / (len(avail) * H_MAX)
    rho_commit_meas = demand / (M_committed * H_MAX)
    require_finite(rho_avail, "rho_avail", T_q=T_q, seed=seed)
    require_finite(rho_commit_meas, "rho_commit", T_q=T_q, seed=seed)

    return dict(tasks=tasks, cand=cand, dur=dur, xbar=xbar, avail=avail, absent=absent,
                M_committed=M_committed, k_absent=k_absent, orphans=orphans, no_cand=no_cand,
                n_bin=n_bin, n_cont=n_cont, n_cons=n_cons,
                cand_degree_min=deg_min, cand_degree_mean=round(deg_mean, 4),
                cand_degree_max=deg_max,
                rho_avail=round(rho_avail, 6), rho_commit=round(rho_commit_meas, 6))


def deviation_constant(xbar: Dict[Any, Any], w4: float) -> float:
    """``C = w4 * |xbar|`` — the constant the linearized deviation term drops (REQ-13 B).

    ``dev = sum_{i,t}[x(1-xb) + (1-x)xb] = sum_{i,t} x(1-2xb) + sum_{i,t} xb``; the model carries
    only the first term, so the full objective is ``model objective + w4*|xbar|``.
    """
    return float(w4) * float(len(xbar))


# --------------------------------------------------------------------------------------------
# exact solve — returns a STRUCTURED result (status code AND name, P, B, C, gap, stop reason)
# --------------------------------------------------------------------------------------------


def _independent_incumbent_assignment(
        raw_x: Any, inst: Dict[str, Any], xidx: Dict[Tuple[Any, Any], int],
        overtime_idx: Dict[Any, int], unserved_idx: Dict[Any, int],
        *, where: str, solver_readback_abs_tol: float) -> Dict[Any, Any]:
    """Independently reconstruct and check a solver incumbent.

    This check does not ask HiGHS whether its own incumbent is feasible.  It rechecks the landed
    vector against task conservation, candidate eligibility, available-worker membership,
    non-negative overtime and the physical ``H_MAX`` capacity.  It is intentionally required for
    a guarded R4-tightness status-1 row before that row can be persisted.
    """
    values = np.asarray(raw_x, dtype=float)
    require(values.ndim == 1 and np.isfinite(values).all(),
            "solver incumbent vector is absent, non-vector, NaN or infinite", where=where)
    tasks, cand, dur, avail = inst["tasks"], inst["cand"], inst["dur"], set(inst["avail"])
    assignment: Dict[Any, Any] = {}
    binary_tol = 1e-7
    for task in tasks:
        selected = []
        for worker in cand[task]:
            value = float(values[xidx[(worker, task)]])
            require(-binary_tol <= value <= 1.0 + binary_tol
                    and min(abs(value), abs(value - 1.0)) <= binary_tol,
                    "incumbent assignment variable is not binary", where=where,
                    task=task, worker=worker, value=value)
            if value > 0.5:
                selected.append(worker)
        u_value = float(values[unserved_idx[task]])
        require(-binary_tol <= u_value <= 1.0 + binary_tol
                and min(abs(u_value), abs(u_value - 1.0)) <= binary_tol,
                "incumbent unserved variable is not binary", where=where,
                task=task, value=u_value)
        require(len(selected) + int(u_value > 0.5) == 1,
                "independent incumbent check failed task conservation",
                where=where, task=task, selected=selected, unserved=u_value)
        assignment[task] = None if u_value > 0.5 else selected[0]
        if assignment[task] is not None:
            require(assignment[task] in avail and assignment[task] in cand[task],
                    "independent incumbent check found an ineligible or unavailable worker",
                    where=where, task=task, worker=assignment[task])

    loads = {worker: 0.0 for worker in avail}
    for task, worker in assignment.items():
        if worker is not None:
            loads[worker] += require_finite(dur[task], "task duration",
                                             where=where, task=task)
    cap_slack = s2_core.cap_fp_slack(H_MAX)
    for worker in sorted(avail):
        load = require_finite(loads[worker], "independently recomputed load",
                              where=where, worker=worker)
        overtime = require_finite(values[overtime_idx[worker]], "incumbent overtime",
                                  where=where, worker=worker)
        require(overtime >= -solver_readback_abs_tol,
                "incumbent carries negative overtime", where=where,
                worker=worker, overtime=overtime,
                solver_readback_abs_tolerance=solver_readback_abs_tol)
        require(load <= H_MAX + cap_slack,
                "independently recomputed incumbent exceeds H_MAX", where=where,
                worker=worker, load=load, cap=H_MAX, allowance=cap_slack)
        require(load - overtime <= H_STD + solver_readback_abs_tol,
                "incumbent overtime does not cover load above H_STD", where=where,
                worker=worker, load=load, overtime=overtime, H_STD=H_STD,
                solver_readback_abs_tolerance=solver_readback_abs_tol)
    return assignment


def solve_exact(inst: Dict[str, Any], w4: float, policy: s2_core.StageSolverPolicy,
                time_limit: float, repeat_index: int,
                presolve: bool, solver_readback_abs_tol: float
                ) -> Tuple[Optional[Dict[Any, Any]], s2_core.SolverResult]:
    """Exact A2 recovery MILP on the raw subproblem. FATAL on an inadmissible solver status."""
    tasks, cand, dur, xbar, avail = (inst["tasks"], inst["cand"], inst["dur"], inst["xbar"],
                                     inst["avail"])
    xpairs = [(i, t) for t in tasks for i in cand[t]]
    xidx = {p: k for k, p in enumerate(xpairs)}
    nO = {i: len(xpairs) + a for a, i in enumerate(avail)}
    nu = {t: len(xpairs) + len(avail) + b for b, t in enumerate(tasks)}
    N = len(xpairs) + len(avail) + len(tasks)

    c = np.zeros(N)
    for (i, t), k in xidx.items():
        xb = 1.0 if xbar.get(t) == i else 0.0
        c[k] = w4 * (1.0 - 2.0 * xb)                  # deviation linear form; constant C dropped
    for i in avail:
        c[nO[i]] = W2
    for t in tasks:
        c[nu[t]] = W5

    rows, lb, ub = [], [], []
    for t in tasks:                                   # serve exactly once or drop
        r = np.zeros(N)
        for i in cand[t]:
            r[xidx[(i, t)]] = 1.0
        r[nu[t]] = 1.0
        rows.append(r); lb.append(1.0); ub.append(1.0)
    for i in avail:                                   # overtime LB: sum dur*x - O <= H_std
        r = np.zeros(N)
        for t in tasks:
            if (i, t) in xidx:
                r[xidx[(i, t)]] = dur[t]
        r[nO[i]] = -1.0
        rows.append(r); lb.append(-np.inf); ub.append(H_STD)
    for i in avail:                                   # capacity: sum dur*x <= H_max
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
    for i in avail:
        hi[nO[i]] = np.inf

    C_const = deviation_constant(xbar, w4)
    t0 = time.perf_counter()
    res = milp(c=c, constraints=cons, integrality=integ, bounds=Bounds(lo, hi),
               options={"time_limit": time_limit, "presolve": presolve})
    dt = time.perf_counter() - t0

    assign: Optional[Dict[Any, Any]] = None
    primal: Optional[float] = None
    status_code = int(getattr(res, "status", -999999))
    raw_message = str(getattr(res, "message", "") or "")
    policy.check(status_code, where="solve_exact repeat={}".format(repeat_index))
    if policy.stage == STAGE_SIZE:
        require(status_code == 0,
                "R4-size accepts solver code 0 only", status_code=status_code,
                raw_message=raw_message, repeat_index=repeat_index)
    if policy.stage == STAGE_TIGHT and status_code == 1:
        require("time limit" in raw_message.lower(),
                "R4-tightness code 1 is admissible only when the raw solver message explicitly "
                "identifies a time limit (not an iteration limit or generic interruption)",
                status_code=status_code, raw_message=raw_message, repeat_index=repeat_index)
    if status_code in (0, 1):
        require(getattr(res, "x", None) is not None,
                "accepted solver status has no incumbent vector", status_code=status_code,
                raw_message=raw_message, repeat_index=repeat_index)
        assign = _independent_incumbent_assignment(
            res.x, inst, xidx, nO, nu,
            where="{} repeat={} status={}".format(policy.stage, repeat_index, status_code),
            solver_readback_abs_tol=solver_readback_abs_tol)
        primal = require_finite(getattr(res, "fun", None), "res.fun", where="solve_exact")
        # model-mismatch guard: the full objective must agree with the shared evaluator
        full_from_model = primal + C_const
        full_from_eval = cb.evaluate(assign, tasks, dur, xbar, w4)[0]
        require(abs(full_from_model - full_from_eval) <= 1e-6 * max(1.0, abs(full_from_eval)),
                "MILP objective and the shared evaluator disagree on the SAME assignment; the "
                "two are not scoring the same object, so no gap computed from them is meaningful",
                obj_from_model_plus_C=full_from_model, obj_from_evaluate=full_from_eval,
                deviation_constant_C=C_const, w4=w4)
    if policy.stage == STAGE_TIGHT and status_code == 1:
        require(assign is not None and primal is not None,
                "R4-tightness code 1 lacks a finite, independently feasible incumbent",
                status_code=status_code, raw_message=raw_message, repeat_index=repeat_index)

    result = s2_core.SolverResult.from_scipy(
        res, policy=policy, deviation_constant_C=C_const, wall_time_s=dt, primal=primal,
        assignment_hash_value=(None if assign is None else s2_core.assignment_hash(assign)),
        repeat_index=repeat_index, where="solve_exact repeat={}".format(repeat_index))
    return assign, result


def _rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024   # mac=bytes, linux=KB


# --------------------------------------------------------------------------------------------
# one isolated measurement -> a list of RUN-LEVEL rows (REQ-14) plus one instance-level row
# --------------------------------------------------------------------------------------------


def measure(stage: str, T_q: int, seed: int, rho_commit: float, absent_frac: float,
            time_limit: float, repeats: int, heur: bool, design: Dict[str, Any],
            *, durs_pool: Optional[Sequence[float]] = None,
            heur_repeats: int = HEUR_REPEATS) -> Dict[str, Any]:
    """Run one configuration. Returns ``{'instance': row, 'runs': [run-level rows]}``."""
    policy: s2_core.StageSolverPolicy = design["policy"]
    require(policy.stage == stage, "policy stage disagrees with the caller",
            policy_stage=policy.stage, stage=stage)
    rss0 = _rss_mb()
    inst = build_instance(
        T_q, rho_commit, absent_frac, CAND_PER_TASK, seed, durs_pool=durs_pool,
        allow_empty_candidate_sets=bool(design["design"]["allow_empty_candidate_sets"]))
    tasks, cand, dur, xbar = inst["tasks"], inst["cand"], inst["dur"], inst["xbar"]

    common = {
        COL.STAGE: stage, "T_q": T_q, "seed": seed,
        COL.RHO_COMMIT_TARGET: rho_commit, "absent_frac": absent_frac,
        COL.M_AVAIL: len(inst["avail"]),
        COL.C_NOMINAL: CAND_PER_TASK,
        COL.CAND_DEG_MIN: inst["cand_degree_min"],
        COL.CAND_DEG_MEAN: inst["cand_degree_mean"],
        COL.CAND_DEG_MAX: inst["cand_degree_max"],
        COL.RHO_AVAIL: inst["rho_avail"],
        COL.RHO_COMMIT: inst["rho_commit"],
        COL.CONFOUND_MODE: design["confound_mode"],
        "snap_id": design.get("snap_id") or "",
        "design_manifest_sha256": design["manifest_sha256"],
        "solver_readback_abs_tolerance": design["solver_readback_abs_tol"],
    }

    # ---- exact: ONE ROW PER REPEAT (previously: median time + last-repeat status) ----
    exact_rows: List[Dict[str, Any]] = []
    reference: Optional[s2_core.SolverResult] = None
    reference_repeat: Optional[int] = None
    reference_assign: Optional[Dict[Any, Any]] = None
    for r in range(1, repeats + 1):
        assign, result = solve_exact(
            inst, W4, policy, time_limit, r, design["presolve"],
            design["solver_readback_abs_tol"])
        row = dict(common)
        row.update(result.as_row())
        row[COL.ALGORITHM] = "EXACT"
        row[COL.ITERS_BUDGET] = 1
        row[COL.ITERS_EXECUTED] = 1
        row[COL.EVALUATIONS] = 1
        row[COL.RNG_SEED] = ""                      # the MILP consumes no RNG of ours
        row[COL.BUDGET_REGIME] = "wall_clock_time_limit"
        row[COL.GAP_SIGNED] = ""                    # a gap OF the reference against itself: n/a
        row[COL.REFERENCE_PROVEN_OPTIMAL] = bool(result.proven_optimal)
        row["solver_message_raw"] = result.message
        row["incumbent_feasible_independent"] = bool(assign is not None)
        row["objective"] = ("" if result.primal is None
                            else result.primal + result.deviation_constant_C)
        exact_rows.append(row)
        # the reference for every heuristic gap is the FIRST proven-optimal repeat, recorded by
        # repeat index so the aggregate is traceable to a row rather than to "the last one"
        if reference is None and result.proven_optimal and assign is not None:
            reference, reference_repeat, reference_assign = result, r, assign
    if reference is None:
        # No repeat proved optimality. There is then no proven-optimal reference, so heuristic
        # gaps are reported against nothing and are left EMPTY with a reason — they are NOT
        # silently computed against a time-limit incumbent and called an optimality gap.
        reference_repeat = None

    rss_ex = _rss_mb()
    ref_full_obj: Optional[float] = None
    if reference is not None and reference.primal is not None:
        ref_full_obj = reference.primal + reference.deviation_constant_C
    ref_proven = bool(reference is not None and reference.proven_optimal)

    # ---- heuristics: one row per algorithm per repeat, each with its OWN seed + RNG ----
    heur_rows: List[Dict[str, Any]] = []
    if heur:
        base_seed = 7 * seed + T_q                   # the historical base; now only a base
        for r in range(1, heur_repeats + 1):
            prov_vns = s2_core.make_rng("VNS", base_seed, r, stage=stage, T_q=T_q)
            prov_alns = s2_core.make_rng("ALNS", base_seed, r, stage=stage, T_q=T_q)
            s2_core.require_distinct_rngs(prov_vns, prov_alns)
            runs = [cb.run_vns(tasks, cand, dur, xbar, W4, prov_vns, repeat_index=r),
                    cb.run_alns(tasks, cand, dur, xbar, W4, prov_alns, repeat_index=r)]
            if r == 1:
                runs.append(cb.run_cold_start(tasks, cand, dur, xbar, W4, repeat_index=1))
            for run in runs:
                row = dict(common)
                row.update(run.as_row())
                row[COL.REFERENCE_PROVEN_OPTIMAL] = ref_proven
                row["solver_message_raw"] = ""
                row["incumbent_feasible_independent"] = ""
                row["reference_repeat_index"] = ("" if reference_repeat is None
                                                 else reference_repeat)
                if ref_full_obj is None:
                    row[COL.GAP_SIGNED] = ""
                    row["gap_undefined_reason"] = "no_exact_incumbent"
                else:
                    where = "{} T_q={} seed={} {} r={}".format(stage, T_q, seed, run.algorithm, r)
                    gap = s2_core.signed_gap_pct(run.objective, ref_full_obj, where=where)
                    # REQ-14 (6): NO max(0, gap) clamp. A negative gap against a PROVEN-optimal
                    # exact solve is a model mismatch and must fail the stage.
                    s2_core.require_gap_sign_admissible(
                        gap, reference_proven_optimal=ref_proven, where=where,
                        algorithm=run.algorithm, heuristic_obj=run.objective,
                        exact_full_obj=ref_full_obj)
                    row[COL.GAP_SIGNED] = round(gap, 6)
                    row["gap_undefined_reason"] = ""
                heur_rows.append(row)

    # ---- instance-level row: aggregates computed FROM the run rows, with provenance ----
    exact_times = [row[COL.WALL_TIME] for row in exact_rows]
    instance_row = dict(common)
    instance_row.update({
        "total_tasks_label": T_q * 15,     # per-slot x horizon accounting label, not an instance
        COL.M_AVAIL: len(inst["avail"]),
        "M_committed": inst["M_committed"],
        "k_absent": inst["k_absent"], "orphans": inst["orphans"], "no_cand": inst["no_cand"],
        COL.C_NOMINAL: CAND_PER_TASK,
        COL.CAND_DEG_MIN: inst["cand_degree_min"],
        COL.CAND_DEG_MEAN: inst["cand_degree_mean"],
        COL.CAND_DEG_MAX: inst["cand_degree_max"],
        COL.RHO_AVAIL: inst["rho_avail"],
        COL.RHO_COMMIT: inst["rho_commit"],
        COL.CONFOUND_MODE: design["confound_mode"],
        COL.RHO_BAND_LO: ("" if design["rho_avail_band"] is None
                          else design["rho_avail_band"][0]),
        COL.RHO_BAND_HI: ("" if design["rho_avail_band"] is None
                          else design["rho_avail_band"][1]),
        "n_bin": inst["n_bin"], "n_cont": inst["n_cont"], "n_cons": inst["n_cons"],
        "w4": W4,
        COL.TIME_LIMIT: time_limit,
        "exact_repeats": len(exact_rows),
        "exact_wall_time_median_s": round(statistics.median(exact_times), 6),
        "exact_wall_time_max_s": round(max(exact_times), 6),
        "exact_rows_summarized": ";".join(
            "repeat={}".format(row[COL.REPEAT_INDEX]) for row in exact_rows),
        "reference_repeat_index": ("" if reference_repeat is None else reference_repeat),
        COL.REFERENCE_PROVEN_OPTIMAL: ref_proven,
        "exact_all_repeats_proven_optimal": all(row[COL.PROVEN_OPTIMAL] for row in exact_rows),
        "obj_incumbent_full_reference": ("" if ref_full_obj is None else ref_full_obj),
        "reference_assignment_hash": ("" if reference_assign is None
                                      else s2_core.assignment_hash(reference_assign)),
        "rss_solve_mb": round(max(0.0, rss_ex - rss0), 1),
        "peak_rss_mb": round(_rss_mb(), 1),
    })
    # REQ-13 test 3, in code: only the two explicitly-boolean provenance columns may carry an
    # optimality word. Any other column name implying optimality would describe a possibly
    # time-limited incumbent as optimal.
    allowed_optimal_columns = {"exact_all_repeats_proven_optimal",
                               COL.REFERENCE_PROVEN_OPTIMAL, COL.PROVEN_OPTIMAL}
    s2_core.assert_not_labeled_optimal(
        [k for k in instance_row if k not in allowed_optimal_columns],
        proven_optimal=False, where="instance-level column names")
    for row in exact_rows + heur_rows:
        s2_core.assert_not_labeled_optimal(
            [k for k in row if k not in allowed_optimal_columns],
            proven_optimal=False, where="run-level column names")
    return {"instance": instance_row, "runs": exact_rows + heur_rows}


# --------------------------------------------------------------------------------------------
# replay demonstration (REQ-14 test 2)
# --------------------------------------------------------------------------------------------


def replay_run_row(row: Dict[str, Any], inst: Dict[str, Any], w4: float) -> Dict[str, Any]:
    """Re-run ONE persisted stochastic row from its persisted seed alone.

    The row is replayable only if ``(algorithm, rng_seed)`` reproduces the objective and the
    assignment hash byte-identically. Returns the comparison; the caller decides fatality.
    """
    algo = row[COL.ALGORITHM]
    seed = int(row[COL.RNG_SEED])
    prov = s2_core.RngProvenance(algorithm=algo, base_seed=-1,
                                 repeat_index=int(row[COL.REPEAT_INDEX]), seed=seed,
                                 rng=__import__("random").Random(seed))
    tasks, cand, dur, xbar = inst["tasks"], inst["cand"], inst["dur"], inst["xbar"]
    if algo == "VNS":
        run = cb.run_vns(tasks, cand, dur, xbar, w4, prov,
                         repeat_index=int(row[COL.REPEAT_INDEX]))
    elif algo == "ALNS":
        run = cb.run_alns(tasks, cand, dur, xbar, w4, prov,
                          repeat_index=int(row[COL.REPEAT_INDEX]))
    else:
        raise PipelineFailure("replay is defined for stochastic algorithms only", algorithm=algo)
    return {"algorithm": algo, "seed": seed,
            "objective_persisted": row["objective"], "objective_replayed": run.objective,
            "assignment_hash_persisted": row[COL.ASSIGN_HASH],
            "assignment_hash_replayed": run.assignment_hash,
            "objective_match": abs(float(row["objective"]) - run.objective) <= 1e-9,
            "assignment_hash_match": row[COL.ASSIGN_HASH] == run.assignment_hash}


# --------------------------------------------------------------------------------------------
# CSV writing — always through canonical_write + validate_stage_output
# --------------------------------------------------------------------------------------------


def write_rows(path: str, rows: Sequence[Dict[str, Any]], *, stage: str,
               spec_builder, diagnostics: Optional[Dict[str, Any]] = None) -> None:
    """temp -> validate the landed temp bytes -> promote. Never a partial canonical file."""
    import csv as _csv
    require(rows, "refusing to write an empty artifact", path=path, stage=stage)
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with s2_core.canonical_write(path, lambda tmp: s2_core.validate_stage_output(
            tmp, spec_builder(tmp)), stage=stage, diagnostics=diagnostics) as tmp:
        with open(tmp, "w", encoding="utf-8", newline="") as handle:
            writer = _csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow({c: row.get(c, "") for c in columns})


def instance_level_spec(artifact: str, stage: str, grid: s2_core.DesignGrid,
                        design: Dict[str, Any], primary_key: Sequence[str],
                        input_paths: Sequence[str]) -> s2_core.StageOutputSpec:
    """Spec for the instance-level table (``tab:scale``'s source).

    REQ-3 note: for the tightness artifact ``(T_q, seed)`` is **not** a primary key — the file also
    varies ``rho_commit_target``. The caller passes the FULL design tuple.
    """
    return s2_core.StageOutputSpec(
        stage=stage, artifact_name=os.path.basename(artifact),
        primary_key=list(primary_key), grid=grid,
        required_columns=[COL.RHO_AVAIL, COL.RHO_COMMIT, COL.RHO_COMMIT_TARGET, COL.M_AVAIL,
                          COL.C_NOMINAL, COL.CAND_DEG_MIN, COL.CAND_DEG_MEAN, COL.CAND_DEG_MAX,
                          COL.CONFOUND_MODE, COL.TIME_LIMIT, "snap_id",
                          "design_manifest_sha256", "solver_readback_abs_tolerance"],
        numeric_columns=[COL.RHO_AVAIL, COL.RHO_COMMIT, COL.RHO_COMMIT_TARGET,
                         COL.M_AVAIL, COL.C_NOMINAL,
                         COL.CAND_DEG_MIN, COL.CAND_DEG_MEAN, COL.CAND_DEG_MAX,
                         "n_bin", "n_cont", "n_cons", "orphans",
                         "exact_wall_time_median_s", "exact_wall_time_max_s",
                         "solver_readback_abs_tolerance"],
        solver_policy=None, run_level=False, rho_avail_band=design["rho_avail_band"],
        input_paths=list(input_paths), snap_id=design.get("snap_id"))


def run_level_spec(artifact: str, stage: str, grid: s2_core.DesignGrid,
                   design: Dict[str, Any], *, primary_key: Sequence[str],
                   repeats_expected: Optional[int],
                   input_paths: Sequence[str]) -> s2_core.StageOutputSpec:
    """Spec for the run-level artifact: the solver block is scoped to the EXACT rows."""
    return s2_core.StageOutputSpec(
        stage=stage, artifact_name=os.path.basename(artifact),
        primary_key=list(primary_key), grid=grid,
        required_columns=[COL.STAGE, "T_q", "seed", COL.ALGORITHM, COL.REPEAT_INDEX,
                          COL.RHO_COMMIT_TARGET, COL.RHO_AVAIL, COL.CAND_DEG_MIN,
                          COL.CAND_DEG_MEAN, COL.CAND_DEG_MAX, COL.CONFOUND_MODE,
                          "solver_message_raw", "incumbent_feasible_independent",
                          "snap_id", "design_manifest_sha256",
                          "solver_readback_abs_tolerance"],
        numeric_columns=[COL.ITERS_BUDGET, COL.ITERS_EXECUTED, COL.EVALUATIONS, COL.WALL_TIME,
                         COL.RHO_COMMIT_TARGET, COL.RHO_AVAIL,
                         COL.CAND_DEG_MIN, COL.CAND_DEG_MEAN,
                         COL.CAND_DEG_MAX, "solver_readback_abs_tolerance"],
        rows_per_cell=1, solver_policy=design["policy"],
        solver_row_filter=(COL.ALGORITHM, ["EXACT"]), run_level=True,
        stochastic_algorithms=["VNS", "ALNS"], repeats_expected=repeats_expected,
        input_paths=list(input_paths), snap_id=design.get("snap_id"),
        nullable_columns=[COL.GAP_SIGNED])


def require_formal_run_contract(rows: Sequence[Dict[str, Any]], stage: str,
                                design: Dict[str, Any]) -> None:
    """Fail before emission if a formal exact row violates the registered R4 status semantics."""
    expected_snap = design["snap_id"]
    for row in rows:
        require(row.get("snap_id") == expected_snap,
                "run row SNAP identifier drifted from the externally supplied registered "
                "archive binding",
                stage=stage, row_snap_id=row.get("snap_id"),
                registered_runtime_snap_id=expected_snap)
        require(float(row.get("solver_readback_abs_tolerance"))
                == float(design["solver_readback_abs_tol"]),
                "run row solver-readback tolerance drifted from the registered R4 manifest",
                stage=stage, row_value=row.get("solver_readback_abs_tolerance"),
                registered=design["solver_readback_abs_tol"])
        if row.get(COL.ALGORITHM) != "EXACT":
            continue
        code = int(row[COL.STATUS_CODE])
        proven = bool(row[COL.PROVEN_OPTIMAL])
        feasible = bool(row.get("incumbent_feasible_independent"))
        if stage == STAGE_SIZE:
            require(code == 0 and proven and feasible,
                    "R4-size requires code 0, proven_optimal=True and an independently feasible "
                    "incumbent on every exact repeat", code=code, proven_optimal=proven,
                    independently_feasible=feasible)
        elif code == 1:
            message = str(row.get("solver_message_raw", ""))
            require("time limit" in message.lower() and feasible and not proven
                    and row.get(COL.STOP_REASON) == "time_limit",
                    "R4-tightness code 1 must be an explicit time-limit row with a finite, "
                    "independently feasible incumbent and must never be labelled optimal",
                    code=code, raw_message=message, independently_feasible=feasible,
                    proven_optimal=proven, stop_reason=row.get(COL.STOP_REASON))
        else:
            require(code == 0 and proven and feasible,
                    "R4-tightness accepts only code 0 optimal rows or guarded code-1 time-limit "
                    "incumbents", code=code, proven_optimal=proven,
                    independently_feasible=feasible)


# --------------------------------------------------------------------------------------------
# sweeps — subprocess-isolated, and gated on a design manifest
# --------------------------------------------------------------------------------------------


def run_sweep(stage: str, configs, tl: float, repeats: int, heur: bool, label: str,
              manifest_path: str, snap_id: str) -> Tuple[List[Dict[str, Any]],
                                                         List[Dict[str, Any]]]:
    """Spawn one isolated child per configuration. A child failure ABORTS the sweep (REQ-2)."""
    instances: List[Dict[str, Any]] = []
    runs: List[Dict[str, Any]] = []
    for (T_q, seed, rc, af) in configs:
        cmd = [sys.executable, os.path.abspath(__file__), "--child",
               "--stage", stage, "--manifest", manifest_path,
               "--snap-id", snap_id,
               "--tq", str(T_q), "--seed", str(seed), "--rho-commit", str(rc),
               "--absent-frac", str(af), "--time-limit", str(tl), "--repeats", str(repeats),
               "--heur", "1" if heur else "0"]
        proc = subprocess.run(cmd, cwd=_EXPERIMENTS, capture_output=True, text=True,
                              timeout=tl * repeats + (400 if heur else 60))
        if proc.returncode != 0:
            raise PipelineFailure(
                "child solve failed; the sweep aborts rather than skipping the cell (REQ-2)",
                stage=stage, T_q=T_q, seed=seed, rho_commit=rc, returncode=proc.returncode,
                stderr_tail=proc.stderr[-1500:])
        line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
        require(line is not None,
                "child produced no RESULT line; the sweep aborts rather than skipping the cell",
                stage=stage, T_q=T_q, seed=seed, stderr_tail=proc.stderr[-800:])
        payload = json.loads(line[len("RESULT "):])
        instances.append(payload["instance"])
        runs.extend(payload["runs"])
        inst_row = payload["instance"]
        print("  {} T_q={:>4} s{} M={:>3} orph={:>3} deg={}/{}/{} rho_av={:.3f} n_bin={:>5} "
              "exact_med={:>7.3f}s all_proven={}".format(
                  label, T_q, seed, inst_row[COL.M_AVAIL], inst_row["orphans"],
                  inst_row[COL.CAND_DEG_MIN], inst_row[COL.CAND_DEG_MEAN],
                  inst_row[COL.CAND_DEG_MAX], inst_row[COL.RHO_AVAIL], inst_row["n_bin"],
                  inst_row["exact_wall_time_median_s"],
                  inst_row["exact_all_repeats_proven_optimal"]), flush=True)
    return instances, runs


def parent(manifest_size: str, manifest_tight: str, emit: bool,
           outdir: str, snap_id: str) -> None:
    """Run the two manifest-declared R4 families into an isolated output directory."""
    require(emit, "parent() without --emit would compute a full sweep and discard it; that is "
                  "not a supported mode. Use --selftest for fixtures.")
    design_size = load_design(manifest_size, STAGE_SIZE, snap_id)
    design_tight = load_design(manifest_tight, STAGE_TIGHT, snap_id)
    output_dir = os.path.abspath(outdir)
    os.makedirs(output_dir, exist_ok=True)

    size_grid = design_size["grid"]
    s2_core.require_keys(
        size_grid,
        ["task_counts", "seeds", "rho_commit_target", "absent_frac", "algorithms",
         "exact_repeats", "heuristic_repeats", "cold_repeats"],
        where="R4-size design.grid")
    size_tq = [int(v) for v in size_grid["task_counts"]]
    size_seeds = [int(v) for v in size_grid["seeds"]]
    size_rho = require_finite(size_grid["rho_commit_target"], "size rho_commit_target")
    size_absent = require_finite(size_grid["absent_frac"], "size absent_frac")
    size_algorithms = list(size_grid["algorithms"])
    size_exact_repeats = int(size_grid["exact_repeats"])
    size_heur_repeats = int(size_grid["heuristic_repeats"])
    size_cold_repeats = int(size_grid["cold_repeats"])
    require(size_algorithms == ["EXACT", "VNS", "ALNS", "cold"],
            "R4-size algorithm order/set differs from the registered design",
            algorithms=size_algorithms)
    require(size_exact_repeats == 3 and size_heur_repeats == 3 and size_cold_repeats == 1,
            "R4-size repeat design must be EXACT/VNS/ALNS=1..3 and cold=1 only",
            exact_repeats=size_exact_repeats, heuristic_repeats=size_heur_repeats,
            cold_repeats=size_cold_repeats)
    size_tl = require_finite(design_size["policy"].time_limit_s,
                             "R4-size manifest time_limit_s")
    print("[A] size sweep: RHO_COMMIT={} ABSENT_FRAC={} seeds={} TL={}s repeats={}".format(
        size_rho, size_absent, len(size_seeds), size_tl, size_exact_repeats))
    cfg_a = [(T_q, seed, size_rho, size_absent)
             for T_q in size_tq for seed in size_seeds]
    inst_a, runs_a = run_sweep(
        STAGE_SIZE, cfg_a, size_tl, size_exact_repeats, True, "size",
        manifest_size, snap_id)
    require_formal_run_contract(runs_a, STAGE_SIZE, design_size)
    grid_a = s2_core.DesignGrid(name="R4-size-runs",
                                axes={COL.STAGE: [STAGE_SIZE], "T_q": size_tq,
                                      "seed": size_seeds, COL.ALGORITHM: size_algorithms,
                                      COL.REPEAT_INDEX: [1, 2, 3]},
                                declared_shortfalls=[
                                    (STAGE_SIZE, T_q, seed, "cold", repeat)
                                    for T_q in size_tq for seed in size_seeds
                                    for repeat in (2, 3)],
                                provenance="R4-size manifest: EXACT/VNS/ALNS repeats 1..3; "
                                           "deterministic cold-start repeat 1 only")
    write_rows(os.path.join(output_dir, "scalability_size_runs.csv"), runs_a, stage=STAGE_SIZE,
               spec_builder=lambda tmp: run_level_spec(tmp, STAGE_SIZE, grid_a, design_size,
                                                       primary_key=[
                                                           COL.STAGE, "T_q", "seed",
                                                           COL.ALGORITHM, COL.REPEAT_INDEX],
                                                       repeats_expected=None,
                                                       input_paths=[SRC, manifest_size]))
    inst_grid_a = s2_core.DesignGrid(name="R4-size-instances",
                                     axes={COL.STAGE: [STAGE_SIZE], "T_q": size_tq,
                                           "seed": size_seeds},
                                     provenance="design_manifest")
    write_rows(os.path.join(output_dir, "scalability_size.csv"), inst_a, stage=STAGE_SIZE,
               spec_builder=lambda tmp: instance_level_spec(
                   tmp, STAGE_SIZE, inst_grid_a, design_size,
                   [COL.STAGE, "T_q", "seed"], [SRC, manifest_size]))

    tight_grid = design_tight["grid"]
    s2_core.require_keys(
        tight_grid,
        ["task_counts", "seeds", "rho_commit_targets", "absent_frac", "algorithms",
         "exact_repeats"],
        where="R4-tightness design.grid")
    tight_tq = [int(v) for v in tight_grid["task_counts"]]
    tight_seeds = [int(v) for v in tight_grid["seeds"]]
    tight_rc = [require_finite(v, "tightness rho_commit_target")
                for v in tight_grid["rho_commit_targets"]]
    tight_absent = require_finite(tight_grid["absent_frac"], "tightness absent_frac")
    tight_algorithms = list(tight_grid["algorithms"])
    tight_repeats = int(tight_grid["exact_repeats"])
    require(len(tight_tq) == 1 and tight_algorithms == ["EXACT"] and tight_repeats == 1,
            "R4-tightness design is one exact repeat over one registered T_q",
            task_counts=tight_tq, algorithms=tight_algorithms, exact_repeats=tight_repeats)
    tight_tl = require_finite(design_tight["policy"].time_limit_s,
                              "R4-tightness manifest time_limit_s")
    print("\n[B] tightness sweep: T_q={} RHO_COMMIT in {} TL={}s".format(
        tight_tq[0], tight_rc, tight_tl))
    cfg_b = [(tight_tq[0], seed, rc, tight_absent)
             for rc in tight_rc for seed in tight_seeds]
    inst_b, runs_b = run_sweep(
        STAGE_TIGHT, cfg_b, tight_tl, tight_repeats, False, "tight",
        manifest_tight, snap_id)
    require_formal_run_contract(runs_b, STAGE_TIGHT, design_tight)
    grid_b = s2_core.DesignGrid(name="R4-tightness-runs",
                                axes={COL.STAGE: [STAGE_TIGHT], "T_q": tight_tq,
                                      COL.RHO_COMMIT_TARGET: tight_rc,
                                      "seed": tight_seeds, COL.ALGORITHM: ["EXACT"],
                                      COL.REPEAT_INDEX: [1]},
                                provenance="design_manifest")
    write_rows(os.path.join(output_dir, "scalability_tightness_runs.csv"),
               runs_b, stage=STAGE_TIGHT,
               spec_builder=lambda tmp: run_level_spec(tmp, STAGE_TIGHT, grid_b, design_tight,
                                                       primary_key=[
                                                           COL.STAGE, "T_q",
                                                           COL.RHO_COMMIT_TARGET, "seed",
                                                           COL.ALGORITHM, COL.REPEAT_INDEX],
                                                       repeats_expected=None,
                                                       input_paths=[SRC, manifest_tight]))
    # REQ-3: the tightness artifact also varies rho_commit_target, so (T_q, seed) is NOT its
    # primary key. The design tuple is (stage, T_q, rho_commit_target, seed).
    inst_grid_b = s2_core.DesignGrid(name="R4-tightness-instances",
                                     axes={COL.STAGE: [STAGE_TIGHT], "T_q": tight_tq,
                                           COL.RHO_COMMIT_TARGET: tight_rc,
                                           "seed": tight_seeds},
                                     provenance="design_manifest")
    write_rows(os.path.join(output_dir, "scalability_tightness.csv"),
               inst_b, stage=STAGE_TIGHT,
               spec_builder=lambda tmp: instance_level_spec(
                   tmp, STAGE_TIGHT, inst_grid_b, design_tight,
                   [COL.STAGE, "T_q", COL.RHO_COMMIT_TARGET, "seed"], [SRC, manifest_tight]))
    print("wrote {} size run rows / {} size instance rows and {} tightness run rows / {} "
          "tightness instance rows under {}".format(
              len(runs_a), len(inst_a), len(runs_b), len(inst_b), output_dir))


# --------------------------------------------------------------------------------------------
# selftest — tiny synthetic fixtures only. Writes under tests/fixtures/corelib/ and nowhere else.
# --------------------------------------------------------------------------------------------

SELFTEST_DURS = [45.0, 60.0, 75.0, 90.0, 105.0, 120.0]
SELFTEST_TQ = [6, 8]
SELFTEST_SEEDS = [1, 2]


def selftest(outdir: str, manifest_path: str) -> int:
    """Tiny synthetic positive-path fixture. Never touches a canonical scalability artifact."""
    s2_core.assert_no_staging_import()
    s2_core.assert_no_absolute_runtime_paths(
        os.path.join(_HERE, "s2_core.py"), os.path.join(_HERE, "compare_baselines.py"),
        os.path.abspath(__file__))
    os.makedirs(outdir, exist_ok=True)
    design = load_design(manifest_path, STAGE_SELFTEST)
    print("manifest          {} sha256={}".format(os.path.basename(manifest_path),
                                                  design["manifest_sha256"]))
    print("base instance     {} (resolved from __file__, exists)".format(
        os.path.relpath(SRC, PATHS.repo_root)))
    print("cap allowance     math.ulp({}) * 8 = {:.6g}".format(H_MAX, cb.CAP_SLACK))

    instances: List[Dict[str, Any]] = []
    runs: List[Dict[str, Any]] = []
    for T_q in SELFTEST_TQ:
        for seed in SELFTEST_SEEDS:
            out = measure(STAGE_SELFTEST, T_q, seed, 0.74, 0.20, 10.0, 2, True, design,
                          durs_pool=SELFTEST_DURS, heur_repeats=2)
            instances.append(out["instance"])
            runs.extend(out["runs"])
            row = out["instance"]
            print("  T_q={} s{} M={} orph={} deg={}/{}/{} rho_av={:.4f} exact_med={:.4f}s "
                  "all_proven={}".format(T_q, seed, row[COL.M_AVAIL], row["orphans"],
                                         row[COL.CAND_DEG_MIN], row[COL.CAND_DEG_MEAN],
                                         row[COL.CAND_DEG_MAX], row[COL.RHO_AVAIL],
                                         row["exact_wall_time_median_s"],
                                         row["exact_all_repeats_proven_optimal"]))

    # a worked full-objective-gap example, printed with P, B, C (REQ-13 test 5)
    example = next((r for r in runs if r[COL.ALGORITHM] == "EXACT"), None)
    require(example is not None, "selftest produced no exact run row")
    print("worked gap example  P={} B={} C={} -> gap_full={} (solver raw gap={})".format(
        example[COL.PRIMAL], example[COL.DUAL], example[COL.DEV_CONST_C],
        example[COL.GAP_FULL], example[COL.GAP_SOLVER_RAW]))
    print("                    full-objective P+C={} B+C={}".format(
        example[COL.PRIMAL_FULL], example[COL.DUAL_FULL]))

    # replay demonstration from the persisted seed alone (REQ-14 test 2)
    vns_row = next((r for r in runs if r[COL.ALGORITHM] == "VNS"), None)
    require(vns_row is not None, "selftest produced no VNS run row")
    inst = build_instance(int(vns_row["T_q"]), 0.74, 0.20, CAND_PER_TASK, int(vns_row["seed"]),
                          durs_pool=SELFTEST_DURS)
    replay = replay_run_row(vns_row, inst, W4)
    print("replay VNS seed={} objective_match={} assignment_hash_match={}".format(
        replay["seed"], replay["objective_match"], replay["assignment_hash_match"]))
    require(replay["objective_match"] and replay["assignment_hash_match"],
            "a persisted stochastic row did not replay from its persisted seed alone",
            **replay)

    grid = s2_core.DesignGrid(
        name="S2-selftest-runs",
        axes={COL.STAGE: [STAGE_SELFTEST], "T_q": SELFTEST_TQ, "seed": SELFTEST_SEEDS,
              COL.ALGORITHM: ["EXACT", "VNS", "ALNS", "cold"], COL.REPEAT_INDEX: [1, 2]},
        declared_shortfalls=[(STAGE_SELFTEST, T_q, s, "cold", 2)
                             for T_q in SELFTEST_TQ for s in SELFTEST_SEEDS],
        provenance="selftest design manifest (cold-start is deterministic: repeat 2 is a "
                   "DECLARED shortfall, not a discovered one)")
    runs_csv = os.path.join(outdir, "fixture_runs.csv")
    write_rows(runs_csv, runs, stage=STAGE_SELFTEST,
               spec_builder=lambda tmp: run_level_spec(tmp, STAGE_SELFTEST, grid, design,
                                                       primary_key=list(s2_core.RUN_LEVEL_KEY),
                                                       repeats_expected=None,
                                                       input_paths=_SELFTEST_SRC_INPUTS + [manifest_path]))
    inst_csv = os.path.join(outdir, "fixture_instances.csv")
    inst_grid = s2_core.DesignGrid(name="S2-selftest-instances",
                                   axes={COL.STAGE: [STAGE_SELFTEST], "T_q": SELFTEST_TQ,
                                         "seed": SELFTEST_SEEDS},
                                   provenance="selftest design manifest")
    write_rows(inst_csv, instances, stage=STAGE_SELFTEST,
               spec_builder=lambda tmp: s2_core.StageOutputSpec(
                   stage=STAGE_SELFTEST, artifact_name=os.path.basename(tmp),
                   primary_key=[COL.STAGE, "T_q", "seed"], grid=inst_grid,
                   required_columns=[COL.RHO_AVAIL, COL.RHO_COMMIT, COL.RHO_COMMIT_TARGET,
                                     COL.M_AVAIL, COL.C_NOMINAL, COL.CAND_DEG_MIN,
                                     COL.CAND_DEG_MEAN, COL.CAND_DEG_MAX, COL.CONFOUND_MODE],
                   numeric_columns=[COL.RHO_AVAIL, COL.RHO_COMMIT, COL.M_AVAIL, COL.C_NOMINAL,
                                    COL.CAND_DEG_MIN, COL.CAND_DEG_MEAN, COL.CAND_DEG_MAX,
                                    "n_bin", "n_cons", "orphans"],
                   solver_policy=None, run_level=False,
                   rho_avail_band=design["rho_avail_band"],
                   input_paths=_SELFTEST_SRC_INPUTS + [manifest_path], snap_id=design.get("snap_id")))
    print("\nselftest fixtures written under {}".format(os.path.relpath(outdir,
                                                                       PATHS.repo_root)))
    print("NOTE: fixtures and a validator PASS are regression/traceability evidence only. They "
          "are not proof, not validation of any published number, and not evidence of an "
          "absence of fabrication.")
    return 0


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--child", action="store_true", help="internal: one isolated measurement")
    ap.add_argument("--stage", default=None)
    ap.add_argument("--manifest", default=None, help="stage design manifest (JSON), required")
    ap.add_argument("--manifest-tightness", default=None)
    ap.add_argument("--snap-id", default=None,
                    help="64-hex SHA256 of the externally verified content-addressed SNAP "
                         "archive; mandatory for formal R4")
    ap.add_argument("--tq", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--rho-commit", type=float, default=None)
    ap.add_argument("--absent-frac", type=float, default=None)
    ap.add_argument("--time-limit", type=float, default=None,
                    help="internal consistency check only; formal limits come from manifests")
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--heur", choices=("0", "1"), default=None)
    ap.add_argument("--selftest", action="store_true",
                    help="tiny synthetic fixtures only; writes under tests/fixtures/corelib/")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--emit", action="store_true",
                    help="emit canonical artifacts; requires a SNAP identifier, so it aborts "
                         "until R0 has produced one")
    args = ap.parse_args(argv)
    # --selftest is the only base-instance-free mode, and it is NOT the first
    # branch dispatched: --child is checked before it, so keying on the flag
    # alone would let `--selftest --child` reach a real measurement.  Key on
    # the mode that will actually run.
    _dispatched_selftest = args.selftest and not args.child
    if not _dispatched_selftest and not os.path.isfile(SRC):
        raise s2_core.PipelineFailure(
            "base instance CSV not found; there is no fallback by design",
            expected_path=os.path.relpath(SRC, PATHS.repo_root),
            journal_p0=os.path.relpath(PATHS.journal_p0, PATHS.repo_root))

    if args.child:
        require(args.stage and args.manifest and args.snap_id
                and args.tq is not None and args.seed is not None
                and args.rho_commit is not None and args.absent_frac is not None
                and args.time_limit is not None and args.repeats is not None
                and args.heur is not None,
                "--child requires --stage --manifest --snap-id --tq --seed --rho-commit "
                "--absent-frac --time-limit --repeats --heur")
        design = load_design(args.manifest, args.stage, args.snap_id)
        manifest_tl = require_finite(design["policy"].time_limit_s,
                                     "manifest solver.time_limit_s", stage=args.stage)
        if args.time_limit is not None:
            require(abs(float(args.time_limit) - manifest_tl) <= 1e-12,
                    "child --time-limit differs from the pre-registered manifest value",
                    stage=args.stage, cli_time_limit=args.time_limit,
                    manifest_time_limit=manifest_tl)
        if args.stage in (STAGE_SIZE, STAGE_TIGHT):
            formal_grid = design["grid"]
            expected_repeats = int(formal_grid["exact_repeats"])
            expected_heur = args.stage == STAGE_SIZE
            require(args.tq in [int(value) for value in formal_grid["task_counts"]]
                    and args.seed in [int(value) for value in formal_grid["seeds"]],
                    "child T_q/seed cell is outside the pre-registered stage grid",
                    stage=args.stage, T_q=args.tq, seed=args.seed)
            registered_rho = (
                [require_finite(formal_grid["rho_commit_target"],
                                "manifest rho_commit_target")]
                if args.stage == STAGE_SIZE else
                [require_finite(value, "manifest rho_commit_target")
                 for value in formal_grid["rho_commit_targets"]])
            require(any(abs(args.rho_commit - value) <= 1e-12
                        for value in registered_rho),
                    "child rho_commit is outside the pre-registered stage grid",
                    stage=args.stage, cli_rho_commit=args.rho_commit,
                    registered=registered_rho)
            registered_absent = require_finite(
                formal_grid["absent_frac"], "manifest absent_frac", stage=args.stage)
            require(abs(args.absent_frac - registered_absent) <= 1e-12,
                    "child absent_frac differs from the pre-registered stage grid",
                    stage=args.stage, cli_absent_frac=args.absent_frac,
                    registered_absent_frac=registered_absent)
            require(args.repeats == expected_repeats,
                    "child exact repeat count differs from the stage manifest",
                    stage=args.stage, cli_repeats=args.repeats,
                    manifest_repeats=expected_repeats)
            require((args.heur == "1") == expected_heur,
                    "child heuristic switch differs from the stage manifest",
                    stage=args.stage, cli_heur=args.heur, expected=expected_heur)
            heur_repeats = int(design["grid"].get("heuristic_repeats", 0))
        else:
            heur_repeats = HEUR_REPEATS
        out = measure(args.stage, args.tq, args.seed, args.rho_commit, args.absent_frac,
                      manifest_tl, args.repeats, args.heur == "1", design,
                      heur_repeats=heur_repeats)
        print("RESULT " + json.dumps(out, default=str))
        return 0
    if args.selftest:
        outdir = args.outdir or os.path.join(PATHS.repo_root, "tests", "fixtures", "corelib")
        manifest = args.manifest or os.path.join(outdir, "selftest_manifest.json")
        return selftest(outdir, manifest)
    require(args.manifest and args.manifest_tightness and args.snap_id and args.outdir,
            "a full sweep requires --manifest (size) and --manifest-tightness; there is no "
            "default solver/grid contract, and --snap-id plus an isolated --outdir are mandatory")
    parent(args.manifest, args.manifest_tightness, args.emit,
           args.outdir, args.snap_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
