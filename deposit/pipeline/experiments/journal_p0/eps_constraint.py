"""
eps_constraint.py — the epsilon-constraint / common-service-level stage driver.

S2 REWRITE (round: S2 CODE + DESIGN FIXES ONLY).  This file is *code*; it produced
NO canonical artifact in this round and it refuses to write one without an explicit
promotion flag (see `--promote-canonical`).  The formal run of this stage is **R1**
and R1 has NOT been executed.

What this file now implements (and what it deliberately does not)
----------------------------------------------------------------
REQ-4 / REQ-5 of `audit_outputs/rerun_plan_v2_1.md`:

  1. COMMON SERVICE LEVEL.  Per disruption we solve `min U` in BOTH regimes to get
     `U_min_A1` and `U_min_A2`, define `U_common := U_min_A1`, and then add
     `sum_q U_q == U_common` as an EQUALITY to every subsequent solve in BOTH
     regimes.  `U_min_A1`, `U_min_A2`, `U_common` and the achieved `U` are columns
     on every output row.

     The claim that `U_common` is attainable in both regimes rests on
     "A1's feasible set is a subset of A2's".  That is NOT assumed here, it is
     CHECKED: `assert_incumbent_eligible()` verifies that every incumbent
     assignment satisfies candidate eligibility and slot availability, and
     `assert_A1_candidates_subset_A2()` verifies the resulting per-task candidate
     sets are nested.  Both are FATAL.  Without them the subset claim — and hence
     the whole `U_common` construction — is unsupported.

  2. ABSOLUTE INTEGER HORIZON BUDGET.  epsilon is one integer budget on the SUM of
     deviation over the whole recovery horizon: `sum_q D_q <= epsilon`.  It is NOT
     a per-slot cap and NOT a fraction.  Because the budget couples the slots, the
     program is built MONOLITHICALLY over all future slots; the old per-slot
     decomposition of this file is gone.

  3. GENUINE SEQUENTIAL LEXICOGRAPHIC SOLVE: `min U` -> `min W s.t. sum D <= eps`
     -> `min sum D`.  Three separate MILPs, each constrained by the *achieved*
     value of the previous one.  No large scalarizing weight is used to emulate
     the order anywhere in this file (`W5 = 1000.0` is NOT imported and NOT used;
     `negative_control_big_weight_is_not_lexicographic()` in the fixture driver
     shows a big-weight scalarization returning a different point).

  4. FULL PRECISION.  Nothing is rounded before it is compared, validated or
     written.  `round(x, 1)` does not appear in this file.  Overtime is
     RECOMPUTED from the integral assignment (`max(0, load - H_STD)` per nurse per
     slot) instead of being read out of the solver's slack variables, which are
     only lower-bounded and are therefore arbitrary in the stages whose objective
     does not price them.

  5. FAIL FAST (REQ-2).  There is no `continue`, no `if r:` and no
     warn-and-proceed.  Every guard raises `S2Fatal`; a failed run promotes
     nothing and writes `*_FAILED_<timestamp>.txt` diagnostics next to the
     intended output.

  6. VALIDATED + HASHED OUTPUT (REQ-3, and section 4's global validation
     contract): temp file -> validator -> atomic promote, with rollback on a
     caught failure.  [shape] [keys] [coverage] [solver] [numeric] [slots]
     [hashes] are implemented; [negative] is demonstrated by the fixture driver
     under `tests/fixtures/epse4/`.  NOTE: the promote is a single-file
     `os.replace`; NO pair-atomicity or family-atomicity is claimed.

  7. PROHIBITION (REQ-4 item 5).  Nothing in this file, and nothing in its test
     suite, uses a previously published endpoint or range as an acceptance
     target.  The epsilon grid is derived from the instance alone.

The tiny-fixture oracle lives in
`tests/fixtures/epse4/oracle_independent.py`. Formal R1 additionally imports
`eps_oracle_independent.py`, a second implementation that imports neither this
module nor `s2_core`. It shares `HHCPFullParser` only as the registered HHCP data
layer, independently parses the incumbent CSV, builds a monolithic MILP, derives
the complete epsilon grid, and compares every `(disruption, regime, epsilon)`
point. Any grid disagreement, `|ΔW| > 1e-9`, unequal D/U, missing point, or
non-code-0 oracle solve aborts R1 and prevents promotion.

Usage (S2 / fixture form; writes nowhere near a canonical path):
    python journal_p0/eps_constraint.py --manifest ../../tests/fixtures/epse4/manifest_epse4_A.csv \
                                        --out-dir ../../tests/fixtures/epse4/out_A

Formal six-rho merge (validates landed shards and does not call a solver):
    python journal_p0/eps_constraint.py --formal --snap-id "$SNAP" \
        --time-limit-s 1800 --manifest journal_p0/rho_manifest.csv \
        --out-dir "$R1_COMBINED" \
        --shard-dir "$R1_045" --shard-dir "$R1_055" \
        --shard-dir "$R1_065" --shard-dir "$R1_075" \
        --shard-dir "$R1_085" --shard-dir "$R1_095"
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

HERE = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS = os.path.dirname(HERE)
_RELEASE_ROOT = os.path.dirname(_EXPERIMENTS)
# Public reproducibility release only.  The base instance from which the 30
# scenarios were derived is not redistributable, so a third party cannot
# recompute its provenance digest.  When BASE_INSTANCE_WITHHELD sits at the
# tree root AND names a manifest-declared source token, that one token
# resolves to None and its recorded digest in rho_manifest.csv stands
# undisputed.  Any other unresolvable token still aborts the run, and no other
# provenance gate is relaxed.  The sealed run carries no such marker.
_WITHHELD_MARKER = os.path.join(_RELEASE_ROOT, "BASE_INSTANCE_WITHHELD")


def _source_is_withheld(token):
    """True only if the release marker names this exact source token."""
    try:
        with open(_WITHHELD_MARKER, "r", encoding="utf-8") as handle:
            named = {line.strip() for line in handle
                     if line.strip() and not line.lstrip().startswith("#")}
    except OSError:
        return False
    return token in named
for _p in (HERE, _EXPERIMENTS):                       # resolved from __file__ only
    if _p not in sys.path:
        sys.path.insert(0, _p)

from SubgroupOptimizer import H_MAX, H_STD, HHCPFullParser   # noqa: E402
from displacement_milp import T_NOW, TRAVEL_PER_TASK, load_static  # noqa: E402
from eps_oracle_independent import oracle_disruption       # noqa: E402

# `W2` / `W5` are deliberately NOT imported: this stage is lexicographic, so a
# scalarizing weight has no role here.  Importing them would invite exactly the
# big-weight emulation REQ-4 item 3 forbids.

STAGE = "R1"
DESIGN_MANIFEST = os.path.join(HERE, "eps_constraint_design_manifest.json")
ORACLE_MODULE = os.path.join(HERE, "eps_oracle_independent.py")
ABSENCE_RULE = "all_active_future_task_holders"
FORMAL_RHOS = (0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
SHARD_ARTIFACTS = (
    "eps_range.csv",
    "eps_frontier.csv",
    "eps_solver_diagnostics.csv",
    "eps_oracle_validation.csv",
)
SHARD_MERGE_RECEIPT = "r1_shard_merge_receipt.json"

# --------------------------------------------------------------------------- #
# shared library.
# `experiments/journal_p0/s2_core.py` is owned by a different S2 agent.  It was
# ABSENT when this rewrite was started, so the first draft carried local
# fallbacks behind the same names; s2_core landed during the round and this file
# now imports it directly rather than keeping a second, never-exercised
# implementation of the same guards alive.  Nothing here re-implements a guard
# s2_core already owns.
# --------------------------------------------------------------------------- #
from s2_core import (COL, DesignGrid, PipelineFailure, SolverResult,  # noqa: E402
                     StageOutputSpec, StageSolverPolicy,
                     assert_no_absolute_runtime_paths, assert_no_staging_import,
                     assignment_hash, canonical_write, cap_fp_slack, fatal,
                     hash_manifest, require, require_finite,
                     require_nonempty_candidates, require_slot_allowed,
                     require_snap_binding, sha256_file, slot_allowed,
                     validate_stage_output)

# --------------------------------------------------------------------------- #
# declared numeric policy (mirrors eps_constraint_design_manifest.json)
# --------------------------------------------------------------------------- #
INT_TOL = 1e-6          # |x - round(x)| tolerance on a variable declared integral
LEX_W_ROW_SLACK = 1e-4  # solver row slack; still strictly below the integral W spacing
LEX_W_TOL = 1e-6        # comparison tolerance on recomputed W values
ORACLE_W_ATOL = 1e-9    # REQ-4 test 3 agreement tolerance on W
FORMAL_SOLVER_TIME_LIMIT_S = 1800.0
OVERTIME_VARIABLE_UPPER_BOUND = float(H_MAX) - float(H_STD)
EPSILON_PARITY_RULE = (
    "D=|xbar|+(n_tasks-U_common)-2|x_intersection_xbar|; "
    "effective_epsilon_is_largest_le_requested_with_that_parity"
)
EPSILON_DIRECT_SOURCE = "direct_effective_budget_solve"
EPSILON_PARITY_REUSE_SOURCE = "parity_equivalent_anchor_reuse"
EPSILON_W_FLOOR_REUSE_SOURCE = "unconstrained_w_floor_anchor_reuse"
EPSILON_DIRECT_PROOF = "code0_optimal_solve_at_effective_epsilon"
EPSILON_PARITY_REUSE_PROOF = (
    "requested_and_anchor_budgets_define_the_same_integer_feasible_set_by_D_parity"
)
EPSILON_W_FLOOR_REUSE_PROOF = (
    "unconstrained_lexicographic_W_then_D_optimum_remains_optimal_for_every_"
    "epsilon_that_contains_its_D_value"
)

EXPECTED_RUNTIME_CONTRACT = {
    "service_level_rule": "U_common=U_min_A1",
    "service_constraint": "sum_u_eq_U_common_in_both_regimes",
    "epsilon_semantics": "absolute_integer_sum_D_over_whole_recovery_horizon",
    "epsilon_grid_rule": (
        "shared_contiguous_integer_grid_from_max_regime_D_floor_"
        "to_max_regime_D_at_Wfloor"
    ),
    "epsilon_effective_budget_rule": EPSILON_PARITY_RULE,
    "epsilon_anchor_reuse": (
        "code0_optimal_anchor_only; parity-equivalent and unconstrained-W-floor "
        "rows retain the complete requested integer grid with explicit proof metadata"
    ),
    "objective_order": [
        "min_U",
        "min_W_given_U_and_epsilon",
        "min_D_given_U_W_and_epsilon",
    ],
    "integrality_tolerance": INT_TOL,
    "lexicographic_W_row_slack": LEX_W_ROW_SLACK,
    "lexicographic_W_comparison_tolerance": LEX_W_TOL,
    "oracle_W_absolute_tolerance": ORACLE_W_ATOL,
    "admissible_solver_status_codes": [0],
    "optimal_solver_status_codes": [0],
    "solver_presolve": False,
    "solver_time_limit_s": FORMAL_SOLVER_TIME_LIMIT_S,
    "overtime_variable_upper_bound_rule": "H_MAX_minus_H_STD",
    "overtime_variable_upper_bound_minutes": OVERTIME_VARIABLE_UPPER_BOUND,
    "formal_canonical_assignment": False,
    "absence_rule": ABSENCE_RULE,
    "formal_max_absent": None,
    "formal_rho_shards": list(FORMAL_RHOS),
    "formal_shard_merge": "validate_and_rebuild_without_resolve",
}

# LEX_W_ROW_SLACK is NOT a cap allowance and must never be confused with one.
# It is slack on an internal lexicographic row, and it is only safe when consecutive
# achievable overtime levels are separated by more than LEX_W_ROW_SLACK.  When every
# task duration is integral, `dur = duration + TRAVEL_PER_TASK` is integral, so
# every achievable overtime total is integral and consecutive levels differ by
# >= 1.0 >> LEX_W_ROW_SLACK.  That is a PROPERTY OF THIS DATA, not a general fact:
# `w_level_separation` is persisted per row so the assumption is auditable, and a
# non-integral instance is refused unless `--allow-fractional-durations` is given.


def find_col(columns, name):
    """Locate a metadata field BY NAME (never by position)."""
    cols = list(columns)
    hits = [c for c in cols if str(c) == str(name)]
    require(len(hits) == 1, "column must appear exactly once, located by name",
            name=name, occurrences=len(hits), available=cols)
    return hits[0]


def optional_manifest_value(row, columns, name, default):
    """Read an optional scenario-provenance field by exact column name."""
    hits = [c for c in list(columns) if str(c) == str(name)]
    require(len(hits) <= 1, "optional manifest column appears more than once",
            name=name, occurrences=len(hits))
    return row[hits[0]] if hits else default


def manifest_bool(value, name):
    """Parse a manifest boolean without Python's unsafe ``bool('False')`` rule."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    token = str(value).strip().lower()
    require(token in ("true", "false", "1", "0"),
            "manifest boolean has an unsupported spelling",
            column=name, value=value)
    return token in ("true", "1")


def solver_policy(cli_time_limit_s=None, require_cli_time_limit=False):
    policy = StageSolverPolicy.from_manifest_file(DESIGN_MANIFEST, stage=STAGE)
    require(policy.admissible_codes == {0} and policy.optimal_codes == {0},
            "R1 is optimal-only: the design manifest must declare code 0 as "
            "the only admissible and optimal status",
            admissible=sorted(policy.admissible_codes),
            optimal=sorted(policy.optimal_codes))
    require(policy.solver_options.get("presolve") is False,
            "R1 must pre-register presolve=false; changing options after a "
            "failure and retrying is forbidden",
            solver_options=dict(policy.solver_options))
    policy.scipy_milp_options(
        cli_time_limit_s=cli_time_limit_s,
        require_cli_time_limit=require_cli_time_limit)
    return policy


def assert_runtime_contract(policy):
    """Bind the executable R1 design to the pre-run manifest exactly.

    The manifest is not merely hashed documentation: every load-bearing
    scientific choice is checked against an executable contract before any
    instance is parsed or any solver is called.
    """
    with open(DESIGN_MANIFEST, "r", encoding="utf-8") as handle:
        try:
            manifest = json.load(handle)
        except ValueError as exc:
            fatal("R1 design manifest is not valid JSON",
                  path=DESIGN_MANIFEST, error=str(exc))
    require(manifest.get("stage") == STAGE,
            "R1 design manifest stage mismatch",
            expected=STAGE, actual=manifest.get("stage"))
    actual = manifest.get("runtime_contract")
    require(actual == EXPECTED_RUNTIME_CONTRACT,
            "R1 executable scientific contract differs from the exact "
            "pre-registered runtime_contract",
            expected=EXPECTED_RUNTIME_CONTRACT, actual=actual)

    service = manifest.get("service_level", {})
    require(service.get("rule") == "U_common := U_min_A1"
            and service.get("applied_as")
            == ("equality constraint sum_q U_q == U_common in BOTH regimes")
            and service.get("exclusions_permitted") == 0,
            "R1 service-level narrative contract disagrees with its executable "
            "runtime_contract", service_level=service)
    epsilon = manifest.get("epsilon", {})
    require(
        epsilon.get("semantics")
        == ("absolute integer budget on the summed deviation over the whole "
            "recovery horizon: sum_q D_q <= epsilon")
        and epsilon.get("is_per_slot_cap") is False
        and epsilon.get("is_fraction") is False
        and epsilon.get("grid_shared_by_regimes") is True
        and epsilon.get("effective_budget_rule") == EPSILON_PARITY_RULE
        and epsilon.get("proof_anchor_policy")
        == ("solve each new effective epsilon to code-0 Optimal; reuse only "
            "fixed-parity equivalents or the already code-0 unconstrained "
            "lexicographic W-floor point, with explicit per-row provenance"),
        "R1 epsilon narrative contract disagrees with its executable "
        "runtime_contract", epsilon=epsilon)
    objective = manifest.get("objective", {})
    require(
        objective.get("order")
        == ["min U", "min W subject to sum_q D_q <= epsilon", "min sum_q D_q"]
        and objective.get("scalarizing_weights_used_to_emulate_the_order") is False
        and objective.get("overtime_variable_upper_bound_rule")
        == "H_MAX_minus_H_STD"
        and objective.get("overtime_variable_upper_bound_minutes")
        == OVERTIME_VARIABLE_UPPER_BOUND,
        "R1 objective narrative contract disagrees with its executable "
        "runtime_contract", objective=objective)
    require(OVERTIME_VARIABLE_UPPER_BOUND >= 0.0,
            "H_MAX-H_STD must be a non-negative overtime-variable upper bound",
            h_max=float(H_MAX), h_std=float(H_STD),
            overtime_variable_upper_bound=OVERTIME_VARIABLE_UPPER_BOUND)
    tolerances = manifest.get("tolerances", {})
    require(
        tolerances.get("integrality_tolerance") == INT_TOL
        and tolerances.get("lexicographic_row_slack") == LEX_W_ROW_SLACK
        and tolerances.get("lexicographic_W_comparison_tolerance") == LEX_W_TOL
        and tolerances.get("oracle_agreement", {}).get("W") == ORACLE_W_ATOL
        and tolerances.get("oracle_agreement", {}).get("D") == "exact"
        and tolerances.get("oracle_agreement", {}).get("U") == "exact",
        "R1 tolerance narrative contract disagrees with executable constants",
        tolerances=tolerances)
    require(0.0 < LEX_W_ROW_SLACK < 1.0,
            "R1 lexicographic W-row slack must remain strictly below the "
            "registered integral W-level spacing",
            lexicographic_W_row_slack=LEX_W_ROW_SLACK)
    oracle_contract = manifest.get("independent_oracle", {})
    require(
        oracle_contract.get("module")
        == "experiments/journal_p0/eps_oracle_independent.py"
        and oracle_contract.get("import_prohibitions")
        == ["eps_constraint", "s2_core"]
        and oracle_contract.get("lexicographic_W_row_slack") == LEX_W_ROW_SLACK
        and oracle_contract.get("solver_evidence_columns")
        == ["oracle_time_limit_s", "oracle_presolve",
            "oracle_effective_epsilon", "oracle_deviation_parity",
            "oracle_epsilon_tightened_by", "oracle_solve_anchor_epsilon",
            "oracle_solver_executed_for_requested_epsilon",
            "oracle_epsilon_solution_source", "oracle_epsilon_reuse_proof",
            "oracle_epsilon_equivalence_rule"]
        and oracle_contract.get("overtime_variable_upper_bound_rule")
        == "H_MAX_ORACLE_minus_H_STD_ORACLE"
        and oracle_contract.get("overtime_variable_upper_bound_minutes")
        == OVERTIME_VARIABLE_UPPER_BOUND,
        "independent-oracle registration differs from the executable contract",
        independent_oracle=oracle_contract)
    with open(ORACLE_MODULE, "r", encoding="utf-8") as handle:
        oracle_tree = ast.parse(handle.read(), filename=ORACLE_MODULE)
    imported_roots = set()
    for node in ast.walk(oracle_tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    prohibited_imports = set(oracle_contract["import_prohibitions"])
    require(not (imported_roots & prohibited_imports),
            "independent oracle imports a prohibited production module",
            oracle_module=ORACLE_MODULE,
            prohibited=sorted(prohibited_imports),
            observed=sorted(imported_roots & prohibited_imports))
    canonical = manifest.get("canonical_assignment", {})
    require(
        canonical.get("formal_default") is False
        and canonical.get("disabled_semantics")
        == ("assignment_hash is empty and may not be used as cross-run "
            "reproducibility evidence"),
        "R1 formal canonical-assignment policy must be explicitly OFF",
        canonical_assignment=canonical)
    disruptions = manifest.get("disruptions", {})
    require(
        disruptions.get("absence_rule") == ABSENCE_RULE
        and disruptions.get("formal_max_absent") is None
        and disruptions.get("formal_coverage")
        == "every active caregiver holding at least one task at q >= T_NOW",
        "R1 disruption-axis narrative contract disagrees with executable policy",
        disruptions=disruptions)
    solver = manifest.get("solver", {})
    require(
        solver.get("admissible_statuses") == [{"code": 0, "name": "Optimal"}]
        and solver.get("optimal_statuses") == [{"code": 0, "name": "Optimal"}]
        and solver.get("accepts_time_limit_incumbent") is False
        and solver.get("time_limit_s") == FORMAL_SOLVER_TIME_LIMIT_S
        and solver.get("options")
        == {"presolve": False, "time_limit": FORMAL_SOLVER_TIME_LIMIT_S},
        "R1 solver narrative contract disagrees with executable solver policy",
        solver=solver)
    require(
        policy.admissible_codes == set(
            EXPECTED_RUNTIME_CONTRACT["admissible_solver_status_codes"])
        and policy.optimal_codes == set(
            EXPECTED_RUNTIME_CONTRACT["optimal_solver_status_codes"])
        and policy.solver_options.get("presolve")
        is EXPECTED_RUNTIME_CONTRACT["solver_presolve"]
        and policy.time_limit_s
        == EXPECTED_RUNTIME_CONTRACT["solver_time_limit_s"],
        "loaded R1 solver policy differs from the exact runtime_contract",
        policy={
            "admissible": sorted(policy.admissible_codes),
            "optimal": sorted(policy.optimal_codes),
            "options": dict(policy.solver_options),
            "time_limit_s": policy.time_limit_s,
        })
    shard_merge = manifest.get("shard_merge", {})
    require(
        shard_merge.get("required_rhos") == list(FORMAL_RHOS)
        and shard_merge.get("required_artifacts_per_shard")
        == list(SHARD_ARTIFACTS)
        and shard_merge.get("receipt")
        == ("r1_shard_merge_receipt.json records each accepted shard rho, "
            "row counts and current four-artifact SHA256 values plus the "
            "complete expected and actual axes; it never contains its own hash"),
        "R1 shard-merge manifest contract differs from executable constants",
        shard_merge=shard_merge)
    return actual


# --------------------------------------------------------------------------- #
# instance-side structure + guards
# --------------------------------------------------------------------------- #
def future_slots(slot_assign):
    return sorted(q for q in slot_assign if q >= T_NOW)


def slot_tasks(parser, slot_assign, q):
    """Tasks of slot q that the parser actually knows, in a deterministic order."""
    return sorted(t for t in slot_assign[q] if t in parser.tasks)


def available_nurses(parser, S, q, absent):
    """Nurses of S on duty in slot q and not absent.

    `q in parser.nurses[n].slots` is set membership in the nurse's allowed-slot
    set (lin2025 Eq. 5 semantics).  In THIS shipped data each task's allowed-slot
    set happens to be a singleton, so the requested slot degenerates to a fixed
    shift; that is a property of the data and is not asserted as a general
    equality anywhere in this file.
    """
    return sorted(n for n in S
                  if n != absent and slot_allowed(q, parser.nurses[n].slots))


def assert_incumbent_eligible(parser, slot_assign, S, absent, label=""):
    """FATAL unless every incumbent assignment is candidate-eligible and on duty.

    This is the load-bearing precondition for `U_common := U_min_A1`.  If some
    incumbent `(n, t)` had `n` outside `t`'s candidate list, then A1 — which pins
    `t` to `n` — would allow an assignment A2 forbids, A1's feasible set would NOT
    be contained in A2's, and neither `U_common`'s attainability in both regimes
    nor REQ-5's comparison would follow.  Returns a small report dict.
    """
    bad_cand, bad_slot, unknown = [], [], []
    n_checked = 0
    for q in future_slots(slot_assign):
        for t in slot_tasks(parser, slot_assign, q):
            n = slot_assign[q][t]
            n_checked += 1
            if n not in parser.nurses:
                unknown.append((q, t, n))
                continue
            if n not in set(parser.tasks[t].candidate_nurses):
                bad_cand.append((q, t, n))
            if not slot_allowed(q, parser.nurses[n].slots):
                bad_slot.append((q, t, n))
    if bad_cand or bad_slot or unknown:
        fatal(
            "incumbent eligibility guard FAILED — A1 feasible-set-subset-A2 may "
            "not be claimed for this instance, so U_common is unsupported",
            label=label, absent=absent,
            incumbent_not_a_candidate=bad_cand[:12],
            incumbent_not_on_duty_in_slot=bad_slot[:12],
            incumbent_nurse_unknown=unknown[:12],
            n_incumbent_pairs_checked=n_checked,
        )
    return {"n_incumbent_pairs_checked": n_checked}


def candidate_sets(parser, slot_assign, S, absent, regime):
    """Per-slot per-task candidate nurse lists for a regime.  Deterministic order."""
    if regime not in ("A1", "A2"):
        fatal("unknown regime", regime=regime)
    cand, empty_after_absence, pre_cand = {}, [], {}
    for q in future_slots(slot_assign):
        avail = set(available_nurses(parser, S, q, absent))
        for t in slot_tasks(parser, slot_assign, q):
            full = sorted(n for n in parser.tasks[t].candidate_nurses if n in avail)
            # The design assumes a non-empty candidate set BEFORE the absence is
            # applied; an empty one there is a data defect and aborts.  A set that
            # becomes empty only because the absent nurse was removed is a
            # legitimate structural outcome (it is how U_min_A1 > 0 arises), so it
            # is counted and persisted rather than treated as a failure.
            pre_cand[(q, t)] = sorted(n for n in parser.tasks[t].candidate_nurses
                                      if n in S)
            st = slot_assign[q][t]
            if regime == "A1" and st != absent:
                cand[(q, t)] = [st]          # incumbent pinned; eligibility guarded
            else:
                cand[(q, t)] = full          # orphan, or A2
            if not cand[(q, t)]:
                empty_after_absence.append((q, t))
    require_nonempty_candidates(
        pre_cand, where=f"pre-absence candidate sets (regime={regime}, "
                        f"absent={absent})")
    return cand, empty_after_absence


def assert_A1_candidates_subset_A2(cand_a1, cand_a2, absent):
    """FATAL unless cand_A1[(q,t)] is a subset of cand_A2[(q,t)] for every task."""
    if set(cand_a1) != set(cand_a2):
        fatal("A1/A2 task keys differ", absent=absent,
              only_a1=sorted(set(cand_a1) - set(cand_a2))[:12],
              only_a2=sorted(set(cand_a2) - set(cand_a1))[:12])
    bad = [(k, cand_a1[k], cand_a2[k]) for k in sorted(cand_a1)
           if not set(cand_a1[k]) <= set(cand_a2[k])]
    if bad:
        fatal("A1 candidate set is NOT contained in A2's — the feasible-set "
              "containment used to justify U_common does not hold",
              absent=absent, offending=bad[:12])
    return {"n_tasks_nested": len(cand_a1)}


def durations(parser, slot_assign):
    """Effective per-task workload = service duration + fixed travel estimate."""
    out = {}
    for q in future_slots(slot_assign):
        for t in slot_tasks(parser, slot_assign, q):
            out[(q, t)] = float(parser.tasks[t].duration) + float(TRAVEL_PER_TASK)
    return out


def incumbent_pairs(parser, slot_assign, S, absent):
    """xbar: the pre-disruption assignment restricted to still-available nurses."""
    xbar = set()
    for q in future_slots(slot_assign):
        avail = set(available_nurses(parser, S, q, absent))
        for t in slot_tasks(parser, slot_assign, q):
            n = slot_assign[q][t]
            if n != absent and n in avail:
                xbar.add((q, n, t))
    return xbar


# --------------------------------------------------------------------------- #
# monolithic horizon model
# --------------------------------------------------------------------------- #
class HorizonModel:
    """One MILP over the WHOLE recovery horizon.

    Slot-local rows: serve-exactly-once-or-drop, overtime lower bound, hard
    capacity.  Horizon-global rows: the unserved equality `sum u == U_common` and
    the absolute integer deviation budget `sum_q D_q <= epsilon`.  The budget is
    what forbids the per-slot decomposition this file used to do.
    """

    def __init__(self, parser, slot_assign, S, absent, regime, policy, label=""):
        self.parser, self.sa, self.S = parser, slot_assign, S
        self.absent, self.regime, self.policy = absent, regime, policy
        self.label = str(label or "")
        self.diagnostics = []
        self.slots = future_slots(slot_assign)
        if not self.slots:
            fatal("no future slot in this instance", absent=absent, t_now=T_NOW)
        self.tasks = {q: slot_tasks(parser, slot_assign, q) for q in self.slots}
        self.avail = {q: available_nurses(parser, S, q, absent) for q in self.slots}
        self.cand, self.empty_after_absence = candidate_sets(
            parser, slot_assign, S, absent, regime)
        self.dur = durations(parser, slot_assign)
        self.xbar = incumbent_pairs(parser, slot_assign, S, absent)

        self.xpairs = [(q, i, t) for q in self.slots for t in self.tasks[q]
                       for i in self.cand[(q, t)]]
        self.xidx = {p: k for k, p in enumerate(self.xpairs)}
        base = len(self.xpairs)
        self.oidx = {}
        for q in self.slots:
            for i in self.avail[q]:
                self.oidx[(q, i)] = base + len(self.oidx)
        base += len(self.oidx)
        self.uidx = {}
        for q in self.slots:
            for t in self.tasks[q]:
                self.uidx[(q, t)] = base + len(self.uidx)
        self.n = base + len(self.uidx)

        self.dev_coef = np.zeros(self.n)
        for (q, i, t), k in self.xidx.items():
            self.dev_coef[k] = 1.0 - 2.0 * (1.0 if (q, i, t) in self.xbar else 0.0)
        self.dev_const = float(len(self.xbar))

        self.n_tasks = sum(len(v) for v in self.tasks.values())
        self.integral_durations = all(float(v).is_integer() for v in self.dur.values())

        self._base_rows, self._base_lb, self._base_ub = self._structural_rows()

    # ---- rows -------------------------------------------------------------
    def _structural_rows(self):
        rows, lb, ub = [], [], []
        for q in self.slots:                                   # serve or drop
            for t in self.tasks[q]:
                r = np.zeros(self.n)
                for i in self.cand[(q, t)]:
                    r[self.xidx[(q, i, t)]] = 1.0
                r[self.uidx[(q, t)]] = 1.0
                rows.append(r); lb.append(1.0); ub.append(1.0)
        for q in self.slots:                                   # overtime LB
            for i in self.avail[q]:
                r = np.zeros(self.n)
                for t in self.tasks[q]:
                    k = self.xidx.get((q, i, t))
                    if k is not None:
                        r[k] = self.dur[(q, t)]
                r[self.oidx[(q, i)]] = -1.0
                rows.append(r); lb.append(-np.inf); ub.append(float(H_STD))
        for q in self.slots:                                   # hard capacity
            for i in self.avail[q]:
                r = np.zeros(self.n)
                for t in self.tasks[q]:
                    k = self.xidx.get((q, i, t))
                    if k is not None:
                        r[k] = self.dur[(q, t)]
                rows.append(r); lb.append(-np.inf); ub.append(float(H_MAX))
        return rows, lb, ub

    def _u_row(self):
        r = np.zeros(self.n)
        for k in self.uidx.values():
            r[k] = 1.0
        return r

    def _w_row(self):
        r = np.zeros(self.n)
        for k in self.oidx.values():
            r[k] = 1.0
        return r

    # ---- objectives (each stage prices exactly one quantity) ---------------
    def _obj(self, what):
        c = np.zeros(self.n)
        if what == "U":
            for k in self.uidx.values():
                c[k] = 1.0
        elif what == "W":
            for k in self.oidx.values():
                c[k] = 1.0
        elif what == "D":
            c = self.dev_coef.copy()
        else:
            fatal("unknown objective stage", what=what)
        return c

    def epsilon_budget(self, u_eq, requested_epsilon):
        """Return the exact parity-tightened epsilon budget.

        With ``sum(u) == U``, every feasible assignment contains exactly
        ``n_tasks-U`` selected x variables.  Therefore

            D = |xbar| + (n_tasks-U) - 2 |x intersect xbar|,

        so all attainable D values have one fixed parity.  Tightening an
        epsilon RHS to the largest integer of that parity does not remove or
        add any integer-feasible assignment; it only strengthens the LP
        relaxation seen by HiGHS.
        """
        require(u_eq is not None and float(u_eq) == int(u_eq),
                "epsilon parity tightening requires an integer U equality",
                u_eq=u_eq, requested_epsilon=requested_epsilon)
        require(float(requested_epsilon) == int(requested_epsilon),
                "epsilon must be an absolute INTEGER budget",
                eps=requested_epsilon)
        requested = int(requested_epsilon)
        selected_pairs = self.n_tasks - int(u_eq)
        require(0 <= selected_pairs <= self.n_tasks,
                "U equality implies an invalid number of selected pairs",
                u_eq=u_eq, n_tasks=self.n_tasks,
                selected_pairs=selected_pairs)
        parity = int((len(self.xbar) + selected_pairs) % 2)
        effective = requested - ((requested - parity) % 2)
        return {
            "requested": requested,
            "effective": int(effective),
            "parity": parity,
            "tightened_by": int(requested - effective),
            "rule": EPSILON_PARITY_RULE,
        }

    def record_epsilon_reuse(self, *, u_eq, requested_epsilon,
                             effective_epsilon, source,
                             solve_anchor_epsilon, w_star):
        """Persist two explicit proof-reuse diagnostics, never fake solver calls."""
        require(source in (EPSILON_PARITY_REUSE_SOURCE,
                           EPSILON_W_FLOOR_REUSE_SOURCE),
                "unknown epsilon proof-reuse source", source=source)
        for lex_stage, objective, w_le in (
                ("lex2_minW", "W", None),
                ("lex3_minD", "D", float(w_star))):
            self.diagnostics.append({
                "stage": STAGE, "regime": self.regime,
                "absent": self.absent, "lex_stage": lex_stage,
                "objective_stage": objective, "u_eq": int(u_eq),
                "epsilon": int(requested_epsilon), "w_le": w_le,
                "d_le": None, "lex_w_row_slack": LEX_W_ROW_SLACK,
                "solver_status_code": 0,
                "solver_message": (
                    "NO SOLVER CALL: reused a code-0 Optimal proof anchor "
                    "under an exactly equivalent integer feasible set"
                ),
                "wall_time_s": 0.0, "nodes": None,
                "time_limit_s": self.policy.time_limit_s,
                "presolve": self.policy.solver_options.get("presolve"),
                "effective_epsilon": int(effective_epsilon),
                "deviation_parity": int(self.epsilon_budget(
                    u_eq, requested_epsilon)["parity"]),
                "epsilon_tightened_by": (
                    int(requested_epsilon) - int(effective_epsilon)),
                "solver_executed": False,
                "epsilon_solution_source": source,
                "solve_anchor_epsilon": solve_anchor_epsilon,
                "epsilon_equivalence_rule": EPSILON_PARITY_RULE,
            })

    # ---- solve ------------------------------------------------------------
    def _solve(self, what, u_eq=None, eps=None, w_le=None, d_le=None,
               fixed=None, allow_infeasible=False, stage_label=""):
        rows = list(self._base_rows); lb = list(self._base_lb); ub = list(self._base_ub)
        if u_eq is not None:
            rows.append(self._u_row()); lb.append(float(u_eq)); ub.append(float(u_eq))
        epsilon_meta = None
        if eps is not None:
            if u_eq is None:
                require(float(eps) == int(eps),
                        "epsilon must be an absolute INTEGER budget", eps=eps)
                epsilon_meta = {
                    "requested": int(eps), "effective": int(eps),
                    "parity": None, "tightened_by": 0,
                    "rule": "not_applied_without_fixed_U",
                }
            else:
                epsilon_meta = self.epsilon_budget(u_eq, eps)
            rows.append(self.dev_coef.copy())
            lb.append(-np.inf)
            ub.append(float(epsilon_meta["effective"]) - self.dev_const)
        if w_le is not None:
            rows.append(self._w_row())
            lb.append(-np.inf); ub.append(float(w_le) + LEX_W_ROW_SLACK)
        if d_le is not None:
            rows.append(self.dev_coef.copy())
            lb.append(-np.inf); ub.append(float(d_le) - self.dev_const)

        lo = np.zeros(self.n); hi = np.ones(self.n)
        for k in self.oidx.values():
            # Exact without big-M inflation: hard capacity gives load <= H_MAX,
            # hence an overtime witness o=max(0, load-H_STD) always lies in this
            # interval.  The finite bound removes scientifically irrelevant slack
            # from U/D stages while preserving every assignment and objective.
            hi[k] = OVERTIME_VARIABLE_UPPER_BOUND
        for k, v in (fixed or {}).items():
            lo[k] = hi[k] = float(v)
        integ = np.zeros(self.n)
        for k in range(len(self.xpairs)):
            integ[k] = 1
        for k in self.uidx.values():
            integ[k] = 1

        t0 = time.perf_counter()
        res = milp(c=self._obj(what),
                   constraints=LinearConstraint(np.array(rows), np.array(lb), np.array(ub)),
                   integrality=integ, bounds=Bounds(lo, hi),
                   options=self.policy.scipy_milp_options())
        wall = time.perf_counter() - t0
        status = int(getattr(res, "status", -1))
        self.diagnostics.append({
            "stage": STAGE, "regime": self.regime, "absent": self.absent,
            "lex_stage": stage_label or what, "objective_stage": what,
            "u_eq": u_eq, "epsilon": eps, "w_le": w_le, "d_le": d_le,
            "lex_w_row_slack": LEX_W_ROW_SLACK,
            "solver_status_code": status,
            "solver_message": str(getattr(res, "message", "")),
            "wall_time_s": wall,
            "nodes": getattr(res, "mip_node_count", None),
            "time_limit_s": self.policy.time_limit_s,
            "presolve": self.policy.solver_options.get("presolve"),
            "effective_epsilon": (
                None if epsilon_meta is None else epsilon_meta["effective"]),
            "deviation_parity": (
                None if epsilon_meta is None else epsilon_meta["parity"]),
            "epsilon_tightened_by": (
                None if epsilon_meta is None else epsilon_meta["tightened_by"]),
            "solver_executed": True,
            "epsilon_solution_source": (
                "direct_non_epsilon_solve" if epsilon_meta is None
                else EPSILON_DIRECT_SOURCE),
            "solve_anchor_epsilon": (
                None if epsilon_meta is None else epsilon_meta["effective"]),
            "epsilon_equivalence_rule": (
                "not_applicable_no_epsilon" if epsilon_meta is None
                else epsilon_meta["rule"]),
        })
        if status == 2 and allow_infeasible:
            return None, {"status": status, "infeasible": True}
        if status not in self.policy.admissible_codes:
            coordinates = {}
            for token in self.label.split():
                key, separator, value = token.partition("=")
                if not separator:
                    continue
                try:
                    if key == "rho":
                        coordinates["rho"] = float(value)
                    elif key == "seed":
                        coordinates["seed"] = int(value)
                except ValueError:
                    # Failure reporting must never mask the original solver status.
                    pass
            failure_record = {
                "event": "R1_SOLVER_STATUS_FAILURE",
                "stage": STAGE,
                "rho": coordinates.get("rho"),
                "seed": coordinates.get("seed"),
                "absent": int(self.absent),
                "regime": str(self.regime),
                "epsilon": None if eps is None else int(eps),
                "effective_epsilon": (
                    None if epsilon_meta is None
                    else int(epsilon_meta["effective"])),
                "deviation_parity": (
                    None if epsilon_meta is None
                    else (None if epsilon_meta["parity"] is None
                          else int(epsilon_meta["parity"]))),
                "epsilon_tightened_by": (
                    None if epsilon_meta is None
                    else int(epsilon_meta["tightened_by"])),
                "lex_stage": stage_label or what,
                "solver_status_code": status,
                "solver_message": str(getattr(res, "message", "")),
                "wall_time_s": wall,
                "time_limit_s": self.policy.time_limit_s,
                "presolve": self.policy.solver_options.get("presolve"),
            }
            print(json.dumps(failure_record, sort_keys=True,
                             separators=(",", ":"), allow_nan=False),
                  file=sys.stderr, flush=True)
            # The record is operational evidence only.  Policy remains code-0
            # only: no incumbent is accepted and no changed-option retry occurs.
            self.policy.check(status, where=f"{STAGE}:{stage_label or what}")
        if res.x is None:
            fatal("solver returned an admissible status but no solution vector",
                  stage=STAGE, lex_stage=stage_label or what, status=status,
                  message=str(getattr(res, "message", "")), regime=self.regime,
                  absent=self.absent)
        sol = self.readback(res.x, stage_label or what)
        # `primal` is the value of the LINEARIZED objective actually minimised, and
        # `deviation_constant_C` is the constant the deviation form drops, so
        # s2_core can rebuild the full-objective gap from the persisted row.
        if what == "D":
            primal, c_const = float(sol["D"]) - self.dev_const, self.dev_const
        elif what == "W":
            primal, c_const = float(sol["W"]), 0.0
        else:
            primal, c_const = float(sol["U"]), 0.0
        sol["solver"] = SolverResult.from_scipy(
            res, policy=self.policy, deviation_constant_C=c_const,
            wall_time_s=wall, primal=primal,
            assignment_hash_value=assignment_hash(sol["assignment"]),
            where=f"{STAGE}:{stage_label or what}")
        sol["status"] = status
        sol["effective_epsilon"] = (
            None if epsilon_meta is None else int(epsilon_meta["effective"]))
        sol["deviation_parity"] = (
            None if epsilon_meta is None or epsilon_meta["parity"] is None
            else int(epsilon_meta["parity"]))
        sol["epsilon_tightened_by"] = (
            None if epsilon_meta is None else int(epsilon_meta["tightened_by"]))
        return sol, {"status": status}

    # ---- exact read-back --------------------------------------------------
    def readback(self, x, stage_label=""):
        """Recompute EVERY reported quantity from the integral assignment.

        The solver's overtime slacks have the exact finite upper bound
        ``H_MAX-H_STD`` but, in the U- and D-stages (whose objectives do not price
        overtime), they can still lie above the true value.  They are therefore
        never read; overtime is recomputed as `max(0, load - H_STD)` per nurse per
        slot at full precision.
        """
        x = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x)):
            fatal("solver returned a non-finite variable", stage=stage_label)
        chosen = set()
        for (q, i, t), k in self.xidx.items():
            v = x[k]
            if abs(v - round(v)) > INT_TOL:
                fatal("variable declared integral came back fractional",
                      stage=stage_label, var=(q, i, t), value=v, tol=INT_TOL)
            if round(v) == 1:
                chosen.add((q, i, t))
        served = {}
        for q in self.slots:
            for t in self.tasks[q]:
                hits = [i for i in self.cand[(q, t)] if (q, i, t) in chosen]
                if len(hits) > 1:
                    fatal("task assigned to more than one nurse",
                          stage=stage_label, slot=q, task=t, nurses=hits)
                served[(q, t)] = hits[0] if hits else None
                uk = self.uidx[(q, t)]
                uv = x[uk]
                if abs(uv - round(uv)) > INT_TOL:
                    fatal("unserved indicator fractional", stage=stage_label,
                          slot=q, task=t, value=uv)
                if bool(round(uv)) != (not hits):
                    fatal("unserved indicator disagrees with the assignment",
                          stage=stage_label, slot=q, task=t, u=uv, nurses=hits)
        loads, W, allow = {}, 0.0, cap_fp_slack(H_MAX)
        for q in self.slots:
            for i in self.avail[q]:
                load = 0.0
                for t in self.tasks[q]:
                    if served[(q, t)] == i:
                        load += self.dur[(q, t)]
                if load > float(H_MAX) + allow:
                    fatal("hard capacity violated", stage=stage_label, slot=q,
                          nurse=i, load=load, h_max=float(H_MAX),
                          allowance=allow, allowance_rule="math.ulp(cap)*8")
                loads[(q, i)] = load
                W += max(0.0, load - float(H_STD))
        U = sum(1 for k, v in served.items() if v is None)
        chosen_pairs = {(q, i, t) for (q, i, t) in chosen}
        D = len(chosen_pairs - self.xbar) + len(self.xbar - chosen_pairs)
        require_finite(W, "recomputed overtime W", stage=stage_label)
        return {"assignment": served, "loads": loads, "U": int(U), "W": float(W),
                "D": int(D), "lex_stage": stage_label}

    # ---- lexicographic driver --------------------------------------------
    def solve_lex(self, u_eq, eps=None, canonical_assignment=False):
        """min W s.t. (U == u_eq, sum D <= eps), then min sum D at that W.

        Genuinely sequential: two MILPs, the second constrained by the achieved
        `W*` of the first.  No scalarizing weight is involved.
        """
        s_w, _ = self._solve("W", u_eq=u_eq, eps=eps, stage_label="lex2_minW")
        w_star = s_w["W"]
        s_d, _ = self._solve("D", u_eq=u_eq, eps=eps, w_le=w_star,
                             stage_label="lex3_minD")
        if s_d["U"] != int(u_eq):
            fatal("U equality not honoured in the D stage",
                  got=s_d["U"], want=int(u_eq))
        if s_d["W"] > w_star + LEX_W_TOL:
            fatal("lexicographic order violated: D stage degraded W",
                  w_star=w_star, w_final=s_d["W"], lex_w_tol=LEX_W_TOL)
        if s_d["W"] < w_star - LEX_W_TOL:
            fatal("D stage found a strictly better W than the W stage — the W "
                  "stage was not optimal", w_star=w_star, w_final=s_d["W"])
        if eps is not None and s_d["D"] > int(eps):
            fatal("epsilon budget violated", eps=int(eps), d=s_d["D"])
        d_status = s_d["status"]
        if canonical_assignment:
            s_d = self._canonical_assignment(
                u_eq=u_eq, eps=eps, w_star=w_star, d_star=s_d["D"])
            s_d["canonical_assignment"] = True
            s_d["solver"].assignment_hash = assignment_hash(s_d["assignment"])
        else:
            s_d["canonical_assignment"] = False
            # An arbitrary optimal assignment is not cross-run evidence.
            s_d["solver"].assignment_hash = None
        s_d["W_star_from_lex2"] = w_star
        if canonical_assignment:
            s_d["status_chain"] = (
                f"{s_w['status']}|{d_status}|{s_d['status']}")
            s_d["lex_stage_chain"] = (
                "lex2_minW|lex3_minD|lex4_canonical_assignment")
        else:
            s_d["status_chain"] = f"{s_w['status']}|{d_status}"
            s_d["lex_stage_chain"] = "lex2_minW|lex3_minD"
        return s_d

    def frontier_point(self, u_eq, requested_epsilon, *, w_floor_point,
                       effective_cache, canonical_assignment=False):
        """Return one requested-grid point from a code-0 proof anchor.

        The complete integer epsilon grid remains in the artifact.  A solver is
        called only for a new effective budget.  Two exact reuse cases are
        permitted and explicitly labelled: the fixed-D-parity duplicate, and
        the top plateau already proved by the no-budget lexicographic W-floor
        solve used during grid derivation.
        """
        meta = self.epsilon_budget(u_eq, requested_epsilon)
        requested = meta["requested"]
        effective = meta["effective"]
        source = None
        proof = None
        anchor_epsilon = None
        solver_executed = False

        if int(w_floor_point["D"]) <= effective:
            anchor = w_floor_point
            source = EPSILON_W_FLOOR_REUSE_SOURCE
            proof = EPSILON_W_FLOOR_REUSE_PROOF
            anchor_epsilon = -1  # explicit sentinel: no-epsilon W-floor anchor
        elif effective in effective_cache:
            anchor = effective_cache[effective]
            source = EPSILON_PARITY_REUSE_SOURCE
            proof = EPSILON_PARITY_REUSE_PROOF
            anchor_epsilon = effective
        else:
            # A grid endpoint is itself attainable D, hence parity-compatible.
            # With ascending integer traversal every parity duplicate must find
            # the immediately preceding effective anchor in the cache.
            require(requested == effective,
                    "parity-equivalent epsilon lacks its prior code-0 anchor",
                    requested_epsilon=requested,
                    effective_epsilon=effective,
                    deviation_parity=meta["parity"],
                    cached_effective_budgets=sorted(effective_cache))
            anchor = self.solve_lex(
                u_eq, eps=effective,
                canonical_assignment=canonical_assignment)
            effective_cache[effective] = anchor
            source = EPSILON_DIRECT_SOURCE
            proof = EPSILON_DIRECT_PROOF
            anchor_epsilon = effective
            solver_executed = True

        point = dict(anchor)
        point.update({
            "requested_epsilon": requested,
            "effective_epsilon": effective,
            "deviation_parity": meta["parity"],
            "epsilon_tightened_by": meta["tightened_by"],
            "solve_anchor_epsilon": int(anchor_epsilon),
            "solver_executed_for_requested_epsilon": bool(solver_executed),
            "epsilon_solution_source": source,
            "epsilon_reuse_proof": proof,
            "epsilon_equivalence_rule": EPSILON_PARITY_RULE,
        })
        require(int(point["D"]) <= effective,
                "epsilon proof anchor is not feasible at the effective budget",
                requested_epsilon=requested,
                effective_epsilon=effective, anchor_D=point["D"],
                source=source)
        require(int(point["D"]) % 2 == int(meta["parity"]),
                "epsilon proof anchor violates the derived D parity",
                requested_epsilon=requested,
                effective_epsilon=effective, anchor_D=point["D"],
                deviation_parity=meta["parity"], source=source)
        if not solver_executed:
            self.record_epsilon_reuse(
                u_eq=u_eq, requested_epsilon=requested,
                effective_epsilon=effective, source=source,
                solve_anchor_epsilon=int(anchor_epsilon),
                w_star=point["W"])
        return point

    def _canonical_assignment(self, u_eq, eps, w_star, d_star):
        """Fix the lexicographically smallest assignment after U/W/D are fixed."""
        order = sorted(range(len(self.xpairs)),
                       key=lambda k: (self.xpairs[k][0], self.xpairs[k][2],
                                      self.xpairs[k][1]))
        fixed = {}
        for k in order:
            trial = dict(fixed)
            trial[k] = 0.0
            sol, _ = self._solve(
                "D", u_eq=u_eq, eps=eps, w_le=w_star, d_le=d_star,
                fixed=trial, allow_infeasible=True,
                stage_label="lex4_canonical_probe")
            if sol is not None and sol["D"] == int(d_star):
                fixed[k] = 0.0
            else:
                fixed[k] = 1.0
        final, _ = self._solve(
            "D", u_eq=u_eq, eps=eps, w_le=w_star, d_le=d_star,
            fixed=fixed, allow_infeasible=True,
            stage_label="lex4_canonical_final")
        if final is None:
            fatal("canonical assignment sweep produced an infeasible final fixing",
                  regime=self.regime, absent=self.absent)
        if final["D"] != int(d_star) or final["U"] != int(u_eq) \
                or abs(final["W"] - float(w_star)) > LEX_W_TOL:
            fatal("canonical assignment sweep changed the lexicographic values",
                  d_star=d_star, d=final["D"], u_star=u_eq, u=final["U"],
                  w_star=w_star, w=final["W"])
        final["canonical_x_key"] = "".join(
            "1" if fixed[k] == 1.0 else "0" for k in order)
        final["canonical_probes"] = len(order)
        return final

    def min_U(self):
        s, _ = self._solve("U", stage_label="lex1_minU")
        return s


# --------------------------------------------------------------------------- #
# one disruption
# --------------------------------------------------------------------------- #
def disruption_family(parser, slot_assign, S, absent, policy, label="",
                      canonical_assignment=False):
    """Everything REQ-4/REQ-5 need for one (instance, absent) disruption.

    Returns (range_row, frontier_rows, diag).  The epsilon grid is derived from
    the instance only:  lo = max over regimes of the minimum attainable total
    deviation at U_common;  hi = max over regimes of the total deviation of that
    regime's overtime-minimising lexicographic point.  `lo` is the largest value
    at which BOTH regimes are guaranteed feasible (A1's feasible set is contained
    in A2's, checked above), so no grid cell can be infeasible and no cell has to
    be skipped — which is what lets REQ-2's fail-fast rule stand.
    """
    print("R1_DISRUPTION_START {}".format(label or f"absent={absent}"),
          flush=True)
    rep = assert_incumbent_eligible(parser, slot_assign, S, absent, label=label)
    cand_a1, empty_a1 = candidate_sets(parser, slot_assign, S, absent, "A1")
    cand_a2, empty_a2 = candidate_sets(parser, slot_assign, S, absent, "A2")
    nest = assert_A1_candidates_subset_A2(cand_a1, cand_a2, absent)

    models = {r: HorizonModel(parser, slot_assign, S, absent, r, policy,
                              label=label)
              for r in ("A1", "A2")}
    u_min = {r: models[r].min_U()["U"] for r in ("A1", "A2")}
    if u_min["A2"] > u_min["A1"]:
        fatal("U_min_A2 > U_min_A1 — impossible if A1's feasible set is contained "
              "in A2's; the containment guard or the model is wrong",
              absent=absent, u_min_a1=u_min["A1"], u_min_a2=u_min["A2"])
    u_common = int(u_min["A1"])

    lo_c, hi_c, w_floor_points = {}, {}, {}
    for r in ("A1", "A2"):
        m = models[r]
        d_floor = m._solve("D", u_eq=u_common, stage_label="grid_minD")[0]["D"]
        w_pt = m.solve_lex(u_common, eps=None)
        w_floor_points[r] = w_pt
        lo_c[r], hi_c[r] = int(d_floor), int(w_pt["D"])
        if hi_c[r] < lo_c[r]:
            fatal("grid derivation inconsistent: D at min-W below the D floor",
                  regime=r, d_floor=lo_c[r], d_at_w_floor=hi_c[r])
    grid_lo = max(lo_c["A1"], lo_c["A2"])
    grid_hi = max(hi_c["A1"], hi_c["A2"])
    if grid_hi < grid_lo:
        grid_hi = grid_lo
    grid = list(range(grid_lo, grid_hi + 1))

    integral = all(models[r].integral_durations for r in ("A1", "A2"))
    sep = "integral_durations" if integral else "unknown"

    rows = []
    for r in ("A1", "A2"):
        m = models[r]
        prev_w = None
        effective_cache = {}
        for eps in grid:
            s = m.frontier_point(
                u_common, eps, w_floor_point=w_floor_points[r],
                effective_cache=effective_cache,
                canonical_assignment=canonical_assignment)
            if s["U"] != u_common:
                fatal("achieved U differs from U_common", regime=r, eps=eps,
                      u=s["U"], u_common=u_common)
            if prev_w is not None and s["W"] > prev_w + LEX_W_TOL:
                fatal("W increased with a looser epsilon — monotonicity violated",
                      regime=r, eps=eps, w=s["W"], w_prev=prev_w)
            prev_w = s["W"]
            row = {
                "absent": absent, "method": r, "eps": int(eps),
                "effective_epsilon": int(s["effective_epsilon"]),
                "deviation_parity": int(s["deviation_parity"]),
                "epsilon_tightened_by": int(s["epsilon_tightened_by"]),
                "solve_anchor_epsilon": int(s["solve_anchor_epsilon"]),
                "solver_executed_for_requested_epsilon": bool(
                    s["solver_executed_for_requested_epsilon"]),
                "epsilon_solution_source": s["epsilon_solution_source"],
                "epsilon_reuse_proof": s["epsilon_reuse_proof"],
                "epsilon_equivalence_rule": s["epsilon_equivalence_rule"],
                "U_min_A1": int(u_min["A1"]), "U_min_A2": int(u_min["A2"]),
                "U_common": u_common, "U": int(s["U"]),
                "W": s["W"], "D": int(s["D"]),
                "eps_grid_lo": grid_lo, "eps_grid_hi": grid_hi,
                "n_future_slots": len(m.slots), "n_tasks_future": m.n_tasks,
                "n_incumbent_pairs_available": len(m.xbar),
                "future_slots_key": "|".join(str(q) for q in m.slots),
                "n_tasks_empty_cand_after_absence": len(m.empty_after_absence),
                "solver_status_chain": s["status_chain"],
                "lex_stage_chain": s["lex_stage_chain"],
                "lex_order": "U>W>D", "eps_semantics": "absolute_integer_sum_D",
                "w_level_separation": sep, "lex_w_tol": LEX_W_TOL,
                "lex_w_row_slack": LEX_W_ROW_SLACK,
                "cap_allowance_rule": "s2_core.cap_fp_slack = math.ulp(cap)*8",
                "canonical_assignment": bool(s["canonical_assignment"]),
                "assignment_hash_evidence": (
                    "canonical_lexicographic_assignment"
                    if s["canonical_assignment"]
                    else "disabled_not_cross_run_evidence"),
            }
            # the s2_core solver block, from the FINAL lexicographic stage
            row.update(s["solver"].as_canonical_row())
            rows.append(row)

    rng_row = {
        "absent": absent,
        "U_min_A1": int(u_min["A1"]), "U_min_A2": int(u_min["A2"]),
        "U_common": u_common,
        "D_floor_A1": lo_c["A1"], "D_at_Wfloor_A1": hi_c["A1"],
        "D_floor_A2": lo_c["A2"], "D_at_Wfloor_A2": hi_c["A2"],
        "eps_grid_lo": grid_lo, "eps_grid_hi": grid_hi, "eps_grid_size": len(grid),
        "n_incumbent_pairs_checked": rep["n_incumbent_pairs_checked"],
        "n_tasks_nested_A1_subset_A2": nest["n_tasks_nested"],
        "n_empty_cand_after_absence_A1": len(empty_a1),
        "n_empty_cand_after_absence_A2": len(empty_a2),
        "w_level_separation": sep,
    }
    diagnostics = []
    for regime, model in models.items():
        for rec in model.diagnostics:
            diagnostics.append(dict(rec, method=regime))
    return rng_row, rows, {
        "u_min": u_min, "grid": grid, "solver_diagnostics": diagnostics}


def active_absentees(parser, slot_assign, max_absent=None):
    """Re-derive the disruption axis from the committed plan.

    Formal R1 enumerates every caregiver who holds at least one future task.
    ``max_absent`` is a fixture/rehearsal convenience only; formal mode rejects
    it before any instance is parsed or solver is called.
    """
    active = sorted({n for q, m in slot_assign.items() if q >= T_NOW
                     for n in m.values()})
    return active if max_absent is None else active[:int(max_absent)]


def compare_independent_oracle(hhcp_path, incumbent_path, absent, range_row,
                               frontier_rows, policy, *, rho, seed):
    """Formal REQ-4 gate over the oracle's independently derived complete grid."""
    oracle = oracle_disruption(
        hhcp_path, incumbent_path, absent,
        rho=rho, seed=seed,
        time_limit_s=policy.time_limit_s,
        presolve=policy.solver_options.get("presolve", True))
    require(int(range_row["U_min_A1"]) == int(oracle["u_min"]["A1"])
            and int(range_row["U_min_A2"]) == int(oracle["u_min"]["A2"])
            and int(range_row["U_common"]) == int(oracle["u_common"]),
            "[oracle] independent U minima/common level disagree",
            absent=absent,
            production=(range_row["U_min_A1"], range_row["U_min_A2"],
                        range_row["U_common"]),
            oracle=(oracle["u_min"]["A1"], oracle["u_min"]["A2"],
                    oracle["u_common"]))

    production_derivation = {
        "D_floor_A1": int(range_row["D_floor_A1"]),
        "D_floor_A2": int(range_row["D_floor_A2"]),
        "D_at_Wfloor_A1": int(range_row["D_at_Wfloor_A1"]),
        "D_at_Wfloor_A2": int(range_row["D_at_Wfloor_A2"]),
        "eps_grid_lo": int(range_row["eps_grid_lo"]),
        "eps_grid_hi": int(range_row["eps_grid_hi"]),
    }
    oracle_derivation = {
        "D_floor_A1": int(oracle["d_floor"]["A1"]),
        "D_floor_A2": int(oracle["d_floor"]["A2"]),
        "D_at_Wfloor_A1": int(oracle["d_at_w_floor"]["A1"]),
        "D_at_Wfloor_A2": int(oracle["d_at_w_floor"]["A2"]),
        "eps_grid_lo": int(oracle["grid_lo"]),
        "eps_grid_hi": int(oracle["grid_hi"]),
    }
    require(production_derivation == oracle_derivation,
            "[oracle-grid] independent D endpoints or shared epsilon bounds "
            "disagree with production",
            absent=absent, production=production_derivation,
            oracle=oracle_derivation)

    production_keys = {
        (str(row["method"]), int(row["eps"])) for row in frontier_rows
    }
    require(len(production_keys) == len(frontier_rows),
            "[oracle-grid] production frontier has duplicate "
            "(regime, epsilon) keys before persistence",
            absent=absent, rows=len(frontier_rows),
            unique_keys=len(production_keys))
    oracle_keys = set(oracle["points"])
    require(production_keys == oracle_keys,
            "[oracle-grid] production frontier key set differs from the "
            "oracle's independently derived complete shared grid",
            absent=absent,
            missing_from_production=sorted(oracle_keys - production_keys),
            extra_in_production=sorted(production_keys - oracle_keys),
            oracle_grid=oracle["epsilon_grid"])
    for regime in ("A1", "A2"):
        production_grid = sorted(
            eps for method, eps in production_keys if method == regime)
        require(production_grid == oracle["epsilon_grid"],
                "[oracle-grid] a regime does not carry the oracle's exact "
                "contiguous shared epsilon grid",
                absent=absent, regime=regime,
                production_grid=production_grid,
                oracle_grid=oracle["epsilon_grid"])

    evidence = []
    for production in frontier_rows:
        key = (str(production["method"]), int(production["eps"]))
        require(key in oracle["points"],
                "[oracle] independent oracle omitted a designed point",
                absent=absent, method=key[0], epsilon=key[1])
        reference = oracle["points"][key]
        delta_w = abs(float(production["W"]) - float(reference["W"]))
        same_d = int(production["D"]) == int(reference["D"])
        same_u = int(production["U"]) == int(reference["U"])
        passed = delta_w <= ORACLE_W_ATOL and same_d and same_u
        evidence.append({
            "absent": int(absent), "method": key[0], "eps": key[1],
            "W_production": float(production["W"]),
            "W_oracle": float(reference["W"]), "abs_W_diff": delta_w,
            "W_atol": ORACLE_W_ATOL,
            "D_production": int(production["D"]),
            "D_oracle": int(reference["D"]),
            "U_production": int(production["U"]),
            "U_oracle": int(reference["U"]),
            "oracle_solver_status_chain": reference["solver_status_chain"],
            "oracle_pass": bool(passed),
            "oracle_module_sha256": sha256_file(ORACLE_MODULE),
            "oracle_eps_grid_lo": int(oracle["grid_lo"]),
            "oracle_eps_grid_hi": int(oracle["grid_hi"]),
            "oracle_D_floor_method": int(oracle["d_floor"][key[0]]),
            "oracle_D_at_Wfloor_method": int(
                oracle["d_at_w_floor"][key[0]]),
            "oracle_independence_scope": oracle["independence_scope"],
            "oracle_time_limit_s": float(policy.time_limit_s),
            "oracle_presolve": bool(policy.solver_options.get("presolve")),
            "oracle_effective_epsilon": int(reference["effective_epsilon"]),
            "oracle_deviation_parity": int(reference["deviation_parity"]),
            "oracle_epsilon_tightened_by": int(
                reference["epsilon_tightened_by"]),
            "oracle_solve_anchor_epsilon": int(
                reference["solve_anchor_epsilon"]),
            "oracle_solver_executed_for_requested_epsilon": bool(
                reference["solver_executed_for_requested_epsilon"]),
            "oracle_epsilon_solution_source": reference[
                "epsilon_solution_source"],
            "oracle_epsilon_reuse_proof": reference["epsilon_reuse_proof"],
            "oracle_epsilon_equivalence_rule": reference[
                "epsilon_equivalence_rule"],
        })
        require(
            int(production["effective_epsilon"])
            == int(reference["effective_epsilon"])
            and int(production["deviation_parity"])
            == int(reference["deviation_parity"])
            and int(production["epsilon_tightened_by"])
            == int(reference["epsilon_tightened_by"]),
            "[oracle] independent epsilon-equivalence derivation disagrees",
            absent=absent, method=key[0], epsilon=key[1],
            production_effective=production["effective_epsilon"],
            oracle_effective=reference["effective_epsilon"],
            production_parity=production["deviation_parity"],
            oracle_parity=reference["deviation_parity"])
        require(passed,
                "[oracle] independent oracle disagrees with production R1 point",
                absent=absent, method=key[0], epsilon=key[1],
                abs_W_diff=delta_w, W_atol=ORACLE_W_ATOL,
                D_production=production["D"], D_oracle=reference["D"],
                U_production=production["U"], U_oracle=reference["U"])
    require(len(evidence) == len(frontier_rows),
            "[oracle] evidence coverage is incomplete",
            absent=absent, expected=len(frontier_rows), actual=len(evidence))
    return evidence


# --------------------------------------------------------------------------- #
# design grids — the INDEPENDENT coverage oracle
# --------------------------------------------------------------------------- #
def _grid_from_cells(name, axes, cells, provenance):
    """Build an s2_core DesignGrid whose designed cells are exactly `cells`.

    Every product cell outside `cells` is DECLARED as a shortfall.  In the
    corrected formal design the disruption axis has no sampling cap; the only
    legitimate product-grid shortfalls here arise because epsilon ranges differ
    by disruption.  They are declared up front, never discovered from output.
    """
    probe = DesignGrid(name=name, axes={k: list(v) for k, v in axes.items()},
                       provenance=provenance)
    full = probe.full_tuples()
    want = {probe._normalize_tuple(c) for c in cells}   # s2_core's own axis norm
    stray = sorted(want - full, key=repr)
    require(not stray, "designed cell outside the declared axes", grid=name,
            offending=stray[:10])
    short = sorted(full - want, key=repr)
    return DesignGrid(name=name, axes={k: list(v) for k, v in axes.items()},
                      declared_shortfalls=short, provenance=provenance)


def build_design(axis, grids, inputs):
    """`axis`  : {(rho, seed): [absent, ...]} re-derived from the committed plans.
       `grids` : {(rho, seed, absent): (lo, hi)} from the grid-derivation solves.

    Independence, stated exactly:
      * the DISRUPTION axis is re-derived from the committed `S_*.csv` with the
        pipeline's own rule, so its coverage check is independent of the artifact;
      * the per-disruption EPSILON RANGE is derived from the grid-derivation
        solves, which run BEFORE the frontier sweep and are not read back from the
        frontier rows.  It is therefore independent of the rows being validated
        but NOT independent of the model — a wrong model would move both.  The
        independent oracles are what check the model itself; this grid only
        checks shape and coverage.
    """
    front_cells, rng_cells = set(), set()
    rhos, seeds, absents, epss = set(), set(), set(), set()
    for (rho, seed), acts in axis.items():
        for a in acts:
            lo, hi = grids[(rho, seed, a)]
            rng_cells.add((rho, seed, a))
            rhos.add(rho); seeds.add(seed); absents.add(a)
            for eps in range(lo, hi + 1):
                epss.add(eps)
                for m in ("A1", "A2"):
                    front_cells.add((rho, seed, a, m, eps))
    prov = ("disruption axis re-derived from the committed S_*.csv; epsilon range "
            "from the pre-sweep grid-derivation solves; NEVER read back from the "
            "artifact under validation")
    front_grid = _grid_from_cells(
        "eps_frontier", {"rho": sorted(rhos), "seed": sorted(seeds),
                         "absent": sorted(absents), "method": ["A1", "A2"],
                         "eps": sorted(epss)}, front_cells, prov)
    rng_grid = _grid_from_cells(
        "eps_range", {"rho": sorted(rhos), "seed": sorted(seeds),
                      "absent": sorted(absents)}, rng_cells, prov)
    return {"axis": axis, "grids": grids, "front_grid": front_grid,
            "rng_grid": rng_grid, "inputs": list(inputs)}


FRONT_NUMERIC = ["eps", "effective_epsilon", "deviation_parity",
                 "epsilon_tightened_by", "solve_anchor_epsilon",
                 "U_min_A1", "U_min_A2", "U_common", "U", "W", "D",
                 "eps_grid_lo", "eps_grid_hi", "n_future_slots", "n_tasks_future",
                 "n_incumbent_pairs_available",
                 "n_tasks_empty_cand_after_absence", "lex_w_tol",
                 "lex_w_row_slack", "eligible_disruptions_in_scenario",
                 "active_caregiver_count", "rho_target", "rho_global",
                 "rho_used_mean", "rho_used_max", "absence_cap"]
RANGE_NUMERIC = ["U_min_A1", "U_min_A2", "U_common", "D_floor_A1",
                 "D_at_Wfloor_A1", "D_floor_A2", "D_at_Wfloor_A2", "eps_grid_lo",
                 "eps_grid_hi", "eps_grid_size", "n_incumbent_pairs_checked",
                 "n_tasks_nested_A1_subset_A2", "n_empty_cand_after_absence_A1",
                 "n_empty_cand_after_absence_A2",
                 "eligible_disruptions_in_scenario", "active_caregiver_count",
                 "rho_target", "rho_global", "rho_used_mean", "rho_used_max",
                 "absence_cap"]

PROVENANCE_COLUMNS = [
    "absence_rule", "absence_cap", "eligible_disruption",
    "eligible_disruptions_in_scenario", "active_caregiver_count",
    "rho_target", "rho_global", "rho_used_mean", "rho_used_max",
    "src_sha256", "incumbent_flat_feasible", "placeable_A1",
    "placeable_A2",
]


def frontier_spec(design, policy, snap_id=None, formal=False):
    return StageOutputSpec(
        stage=STAGE, artifact_name="eps_frontier.csv",
        primary_key=["rho", "seed", "absent", "method", "eps"],
        grid=design["front_grid"],
        required_columns=["U_min_A1", "U_min_A2", "U_common", "U", "W", "D",
                          "lex_order", "eps_semantics", "w_level_separation",
                          "lex_w_row_slack",
                          "effective_epsilon", "deviation_parity",
                          "epsilon_tightened_by", "solve_anchor_epsilon",
                          "solver_executed_for_requested_epsilon",
                          "epsilon_solution_source", "epsilon_reuse_proof",
                          "epsilon_equivalence_rule",
                          "n_incumbent_pairs_available",
                          "solver_status_chain", "lex_stage_chain",
                          "future_slots_key", "cap_allowance_rule",
                          "canonical_assignment", "assignment_hash_evidence",
                          "snap_id"] + PROVENANCE_COLUMNS,
        numeric_columns=FRONT_NUMERIC,
        solver_policy=policy, status_column=COL.STATUS_CODE,
        input_paths=design["inputs"], snap_id=snap_id, require_snap=formal)


def range_spec(design, snap_id=None, formal=False):
    return StageOutputSpec(
        stage=STAGE, artifact_name="eps_range.csv",
        primary_key=["rho", "seed", "absent"], grid=design["rng_grid"],
        required_columns=["U_min_A1", "U_min_A2", "U_common",
                          "w_level_separation", "snap_id"] + PROVENANCE_COLUMNS,
        numeric_columns=RANGE_NUMERIC,
        solver_policy=None,
        input_paths=design["inputs"], snap_id=snap_id, require_snap=formal)


# --------------------------------------------------------------------------- #
# stage-specific checks that the generic contract cannot express
# --------------------------------------------------------------------------- #
def extra_checks(rng_df, front_df, design, verbose=True, formal=False):
    """REQ-4 tests 1/2/5, REQ-5 tests 1/2, plus the independent slot check."""
    log = []

    def ok(tag, msg):
        log.append(f"[{tag}] OK  {msg}")

    rk = [find_col(rng_df.columns, c) for c in ("rho", "seed", "absent")]
    fk = rk + [find_col(front_df.columns, "method"),
               find_col(front_df.columns, "eps")]

    # --- REQ-4 test 2: epsilon absolute + integral, grid shared by both regimes
    e = front_df[find_col(front_df.columns, "eps")].astype(float)
    require(bool(np.all(np.equal(np.mod(e, 1.0), 0.0))),
            "[epsilon] epsilon is not integral", sample=e.head(5).tolist())
    for key, g in front_df.groupby(rk):
        per = {m: sorted(int(v) for v in gg[find_col(gg.columns, "eps")])
               for m, gg in g.groupby(find_col(g.columns, "method"))}
        require(sorted(per) == ["A1", "A2"],
                "[epsilon] a disruption is missing a regime", key=key,
                regimes=sorted(per))
        require(per["A1"] == per["A2"],
                "[epsilon] A1 and A2 do not share the same epsilon grid",
                key=key, a1=per["A1"], a2=per["A2"])
        grid = per["A1"]
        require(grid == list(range(grid[0], grid[-1] + 1)),
                "[epsilon] grid is not a contiguous integer range", key=key,
                grid=grid)
        lo, hi = int(g["eps_grid_lo"].iloc[0]), int(g["eps_grid_hi"].iloc[0])
        require((grid[0], grid[-1]) == (lo, hi),
                "[epsilon] realised grid disagrees with persisted bounds",
                key=key, realised=(grid[0], grid[-1]), persisted=(lo, hi))
    ok("epsilon", "absolute integer budget; contiguous grid shared by both "
                  "regimes on every disruption")

    # Exact formulation-strengthening and proof-anchor audit.  This is not a
    # heuristic cache: the parity identity is recomputed from persisted row
    # quantities, and every reused row must point to a code-0 anchor whose
    # (U,W,D) value is valid for the requested budget.
    range_lookup = {
        (float(row["rho"]), int(row["seed"]), int(row["absent"])): row
        for _, row in rng_df.iterrows()
    }
    frontier_lookup = {
        (float(row["rho"]), int(row["seed"]), int(row["absent"]),
         str(row["method"]), int(row["eps"])): row
        for _, row in front_df.iterrows()
    }
    source_counts = {
        EPSILON_DIRECT_SOURCE: 0,
        EPSILON_PARITY_REUSE_SOURCE: 0,
        EPSILON_W_FLOOR_REUSE_SOURCE: 0,
    }
    for _, row in front_df.iterrows():
        key3 = (float(row["rho"]), int(row["seed"]), int(row["absent"]))
        method = str(row["method"])
        requested = int(row["eps"])
        effective = int(row["effective_epsilon"])
        parity = int(row["deviation_parity"])
        expected_parity = int((
            int(row["n_incumbent_pairs_available"])
            + int(row["n_tasks_future"]) - int(row["U_common"])) % 2)
        expected_effective = requested - ((requested - expected_parity) % 2)
        require(parity == expected_parity
                and effective == expected_effective
                and int(row["epsilon_tightened_by"]) == requested - effective
                and requested - effective in (0, 1)
                and int(row["D"]) <= effective
                and int(row["D"]) % 2 == parity
                and str(row["epsilon_equivalence_rule"]) == EPSILON_PARITY_RULE,
                "[epsilon-proof] persisted parity/effective budget is invalid",
                key=key3 + (method, requested), parity=parity,
                expected_parity=expected_parity, effective=effective,
                expected_effective=expected_effective, D=int(row["D"]))
        source = str(row["epsilon_solution_source"])
        require(source in source_counts,
                "[epsilon-proof] unknown solution source",
                key=key3 + (method, requested), source=source)
        source_counts[source] += 1
        executed = manifest_bool(
            row["solver_executed_for_requested_epsilon"],
            "solver_executed_for_requested_epsilon")
        anchor = int(row["solve_anchor_epsilon"])
        proof = str(row["epsilon_reuse_proof"])
        if source == EPSILON_DIRECT_SOURCE:
            require(executed and requested == effective == anchor
                    and proof == EPSILON_DIRECT_PROOF,
                    "[epsilon-proof] malformed direct anchor",
                    key=key3 + (method, requested), anchor=anchor,
                    effective=effective, executed=executed, proof=proof)
        elif source == EPSILON_PARITY_REUSE_SOURCE:
            anchor_key = key3 + (method, anchor)
            anchor_row = frontier_lookup.get(anchor_key)
            require(not executed and requested == effective + 1
                    and anchor == effective
                    and proof == EPSILON_PARITY_REUSE_PROOF
                    and anchor_row is not None
                    and float(anchor_row["W"]) == float(row["W"])
                    and int(anchor_row["D"]) == int(row["D"]),
                    "[epsilon-proof] parity-reused row lacks an equal prior anchor",
                    key=key3 + (method, requested), anchor_key=anchor_key,
                    effective=effective, executed=executed, proof=proof)
        else:
            rr = range_lookup[key3]
            require(not executed and anchor == -1
                    and proof == EPSILON_W_FLOOR_REUSE_PROOF
                    and int(row["D"]) == int(rr[f"D_at_Wfloor_{method}"])
                    and int(row["D"]) <= effective,
                    "[epsilon-proof] W-floor reuse lacks its endpoint proof",
                    key=key3 + (method, requested), anchor=anchor,
                    endpoint_D=int(rr[f"D_at_Wfloor_{method}"]),
                    row_D=int(row["D"]), effective=effective,
                    executed=executed, proof=proof)
    ok("epsilon-proof", "fixed-D parity and unconstrained-W-floor anchors "
                        f"validated; source counts={source_counts}")

    # --- REQ-4 test 1: U == U_common == U_min_A1 on every row of both regimes
    bad = front_df[front_df[find_col(front_df.columns, "U")]
                   != front_df[find_col(front_df.columns, "U_common")]]
    require(len(bad) == 0, "[service] U != U_common", n=int(len(bad)),
            sample=bad.head(3).to_dict("records"))
    bad = front_df[front_df["U_common"] != front_df["U_min_A1"]]
    require(len(bad) == 0, "[service] U_common != U_min_A1", n=int(len(bad)))
    n_a2_better = int((rng_df["U_min_A2"] < rng_df["U_common"]).sum())
    require(bool((rng_df["U_min_A2"] <= rng_df["U_min_A1"]).all()),
            "[service] U_min_A2 > U_min_A1 somewhere — impossible under the "
            "checked A1-subset-A2 containment")
    ok("service", f"U == U_common == U_min_A1 on all {len(front_df)} rows of both "
                  f"regimes; disruptions with U_min_A2 < U_common: {n_a2_better} "
                  f"of {len(rng_df)} — none excluded")

    # --- REQ-4 test 5: monotonicity
    for key, g in front_df.groupby(rk + [find_col(front_df.columns, "method")]):
        g = g.sort_values(find_col(g.columns, "eps"))
        w = g[find_col(g.columns, "W")].to_numpy(dtype=float)
        require(bool(np.all(np.diff(w) <= LEX_W_TOL)),
                "[monotone] W is not non-increasing in epsilon", key=key,
                w=w.tolist())
    ok("monotone", "W non-increasing in epsilon within every (disruption, regime)")

    # --- REQ-5 tests 1/2: paired comparison at the common service level
    j = front_df.pivot_table(index=rk + ["eps"], columns="method", values="W",
                             aggfunc="first")
    require(not bool(j.isna().to_numpy().any()),
            "[REQ-5] a (disruption, eps) cell is missing a regime; the exclusion "
            "count must be 0", n_missing=int(j.isna().to_numpy().sum()))
    viol = j[j["A2"] > j["A1"] + LEX_W_TOL]
    ok("REQ-5", f"pairs tested={len(j)}  excluded=0  W_A2<=W_A1 on "
                f"{len(j) - len(viol)}  strict violations={len(viol)}")
    if len(viol):
        counterexamples = []
        for k, r in viol.iterrows():
            key = k if isinstance(k, tuple) else (k,)
            rho, seed, absent, eps = key
            mask = (
                np.isclose(front_df["rho"].astype(float), float(rho))
                & (front_df["seed"].astype(int) == int(seed))
                & (front_df["absent"].astype(int) == int(absent))
                & (front_df["eps"].astype(int) == int(eps))
            )
            columns = [
                "rho", "seed", "absent", "eps", "method",
                "U_min_A1", "U_min_A2", "U_common", "U", "W", "D",
                "canonical_assignment", "assignment_hash_evidence",
                COL.STATUS_CODE, COL.STATUS_NAME, COL.PROVEN_OPTIMAL,
            ]
            rows = front_df.loc[mask, columns].sort_values("method")
            counterexamples.append({
                "key": {
                    "rho": float(rho), "seed": int(seed),
                    "absent": int(absent), "eps": int(eps),
                },
                "W_A1": float(r["A1"]),
                "W_A2": float(r["A2"]),
                "A2_minus_A1": float(r["A2"] - r["A1"]),
                "production_rows": rows.to_dict("records"),
                "guards_already_passed": [
                    "incumbent_candidate_and_slot_eligibility",
                    "A1_candidate_sets_subset_of_A2",
                    "same_U_common_equals_U_min_A1",
                    "same_absolute_epsilon",
                ],
            })
        log.append(
            "[REQ-5] strict violations found after the load-bearing subset, "
            "common-U and common-epsilon guards. Formal R1 must abort; rehearsal "
            "output is HUMAN_DECISION_REQUIRED. Evidence: "
            + repr(counterexamples[:10]))
        if formal:
            fatal(
                "[REQ-5] formal R1 found W_A2 > W_A1 at common U and the "
                "same absolute epsilon after A1-subset-A2 guards passed; this is "
                "a candidate counterexample or implementation defect and may not "
                "silently complete",
                n_counterexamples=len(counterexamples),
                counterexamples=counterexamples[:10],
                outcome="HUMAN_DECISION_REQUIRED")

    # --- [slots], independently re-derived
    for key, g in front_df.groupby(rk):
        want = design["future_slots"][tuple(key)]
        for v in set(g["future_slots_key"]):
            require(str(v) == "|".join(str(q) for q in want),
                    "[slots] the horizon slot set in the artifact differs from "
                    "the set re-derived from the committed plan", key=key,
                    in_artifact=v, re_derived=want)
        require(set(int(x) for x in g["n_future_slots"]) == {len(want)},
                "[slots] n_future_slots disagrees with the re-derived horizon",
                key=key, expected=len(want))
    ok("slots", "every row's horizon slot set equals the set re-derived from the "
                "committed plan")

    # The formal disruption/provenance contract is persisted on every row; a
    # later statistics stage must not have to infer it from filenames or row
    # counts.  Legitimate zero-drop failures are labelled, never filtered.
    for frame_name, frame in (("range", rng_df), ("frontier", front_df)):
        expected_rule = str(design["absence_rule"])
        require(set(frame["absence_rule"].astype(str)) == {expected_rule},
                "[coverage] artifact carries the wrong absence rule",
                artifact=frame_name,
                observed=sorted(set(frame["absence_rule"].astype(str))))
        eligible = frame["eligible_disruption"].astype(str).str.lower()
        require(bool(eligible.isin(["true", "1"]).all()),
                "[coverage] an emitted disruption is not marked eligible",
                artifact=frame_name)
        require(bool((frame["rho_target"].astype(float)
                      == frame["rho"].astype(float)).all()),
                "[provenance] rho_target differs from the design rho key",
                artifact=frame_name)
        require(bool((frame["placeable_A1"].astype(str).str.lower().isin(
            ["true", "1"]) == (frame["U_min_A1"].astype(int) == 0)).all()),
            "[placeability] placeable_A1 disagrees with U_min_A1",
            artifact=frame_name)
        require(bool((frame["placeable_A2"].astype(str).str.lower().isin(
            ["true", "1"]) == (frame["U_min_A2"].astype(int) == 0)).all()),
            "[placeability] placeable_A2 disagrees with U_min_A2",
            artifact=frame_name)
        if formal:
            require(expected_rule == ABSENCE_RULE
                    and set(frame["absence_cap"].astype(int)) == {-1},
                    "[coverage] formal artifact carries a debug absence cap",
                    artifact=frame_name, absence_rule=expected_rule,
                    absence_caps=sorted(set(frame["absence_cap"].astype(int))))
            feasible = frame["incumbent_flat_feasible"].astype(str).str.lower()
            require(bool(feasible.isin(["true", "1"]).all()),
                    "[provenance] formal row lacks a flat-feasible incumbent",
                    artifact=frame_name)
    for (rho, seed), expected_absentees in design["axis"].items():
        for frame_name, frame in (("range", rng_df), ("frontier", front_df)):
            sub = frame[(np.isclose(frame["rho"].astype(float), float(rho)))
                        & (frame["seed"].astype(int) == int(seed))]
            require(set(sub["absent"].astype(int))
                    == set(int(v) for v in expected_absentees),
                    "[coverage] scenario absent set differs from re-derived axis",
                    artifact=frame_name, rho=rho, seed=seed)
            require(set(sub["eligible_disruptions_in_scenario"].astype(int))
                    == {len(expected_absentees)},
                    "[coverage] persisted eligible disruption count is wrong",
                    artifact=frame_name, rho=rho, seed=seed,
                    expected=len(expected_absentees))
    ok("provenance", "all-active absence rule, eligibility counts, source/scenario "
                     "metadata and placeability labels are internally consistent")

    # Assignment hashes are reproducibility evidence only when the explicitly
    # declared canonical assignment stage ran.
    for idx, row in front_df.iterrows():
        enabled = str(row["canonical_assignment"]).strip().lower() in (
            "true", "1", "yes")
        raw_hash = row.get(COL.ASSIGN_HASH, "")
        ahash = "" if pd.isna(raw_hash) else str(raw_hash).strip()
        evidence = str(row["assignment_hash_evidence"]).strip()
        if enabled:
            require(bool(ahash) and evidence == "canonical_lexicographic_assignment",
                    "[assignment] canonical assignment row lacks its hash/evidence label",
                    row=int(idx), assignment_hash=ahash, evidence=evidence)
        else:
            require(not ahash,
                    "[assignment] non-canonical assignment hash must be empty; it "
                    "cannot be cross-run evidence",
                    row=int(idx), assignment_hash=ahash)
            require(evidence == "disabled_not_cross_run_evidence",
                    "[assignment] disabled canonicalization has the wrong evidence label",
                    row=int(idx), evidence=evidence)
    ok("assignment", "assignment hashes exist only for canonical lexicographic assignments")

    expected_snap = str(design.get("snap_id") or "")
    observed_snap = {str(v).strip() for v in front_df["snap_id"].fillna("")}
    observed_range_snap = {str(v).strip() for v in rng_df["snap_id"].fillna("")}
    require(observed_snap == {expected_snap}
            and observed_range_snap == {expected_snap},
            "[SNAP] artifact snap_id differs from the run's verified SNAP receipt",
            expected=expected_snap, frontier=sorted(observed_snap),
            range=sorted(observed_range_snap))
    ok("SNAP", "every scientific row carries the verified runtime SNAP identifier"
       if expected_snap else "rehearsal rows carry no SNAP identifier")

    if verbose:
        for line in log:
            print("    " + line)
    return log


# --------------------------------------------------------------------------- #
# write path — s2_core.canonical_write per artifact
# --------------------------------------------------------------------------- #
def write_validated(out_dir, rng_df, front_df, diagnostics_df,
                    oracle_validation_df, design, policy,
                    snap_id=None, formal=False, verbose=True):
    """One `canonical_write` per artifact: temp -> validate landed bytes ->
    promote, with rollback on a caught failure.

    HONEST LIMIT: `canonical_write` promotes exactly ONE file per invocation.
    Scientific files and their diagnostics/oracle sidecars therefore are NOT
    pair-atomic or family-atomic; an interruption can leave a partially promoted
    family. Cross-artifact checks run before promotion and again on landed bytes.
    """
    os.makedirs(out_dir, exist_ok=True)
    extra_checks(rng_df, front_df, design, verbose=False,
                 formal=formal)                                # pre-write gate

    reports = {}
    for name, df, spec in (("eps_range.csv", rng_df,
                            range_spec(design, snap_id, formal)),
                           ("eps_frontier.csv", front_df,
                            frontier_spec(design, policy, snap_id, formal))):
        path = os.path.join(out_dir, name)
        holder = {}

        def _validate(temp_path, _spec=spec, _holder=holder):
            _holder["report"] = validate_stage_output(temp_path, _spec)

        with canonical_write(path, _validate, stage=STAGE,
                             diagnostics={"artifact": name}) as temp_path:
            df.to_csv(temp_path, index=False)
        reports[name] = holder["report"]

    diagnostic_path = os.path.join(out_dir, "eps_solver_diagnostics.csv")

    def _validate_diagnostics(temp_path):
        require(os.path.getsize(temp_path) > 0,
                "epsilon solver diagnostics sidecar is empty",
                artifact="eps_solver_diagnostics.csv")

    with canonical_write(
            diagnostic_path, _validate_diagnostics, stage=STAGE,
            diagnostics={"artifact": "eps_solver_diagnostics.csv",
                         "scientific_result": False}) as temp_path:
        diagnostics_df.to_csv(temp_path, index=False)

    output_names = [
        "eps_range.csv", "eps_frontier.csv", "eps_solver_diagnostics.csv"]
    if formal:
        require(oracle_validation_df is not None
                and len(oracle_validation_df) == len(front_df),
                "formal R1 oracle sidecar must cover every frontier row",
                expected=len(front_df),
                actual=(None if oracle_validation_df is None
                        else len(oracle_validation_df)))
        require(bool(oracle_validation_df["oracle_pass"].astype(bool).all()),
                "formal R1 oracle sidecar contains a failed comparison")
        oracle_path = os.path.join(out_dir, "eps_oracle_validation.csv")

        def _validate_oracle(temp_path):
            landed = pd.read_csv(temp_path)
            key = ["rho", "seed", "absent", "method", "eps"]
            required = set(key + [
                "W_production", "W_oracle", "abs_W_diff", "W_atol",
                "D_production", "D_oracle", "U_production", "U_oracle",
                "oracle_solver_status_chain", "oracle_pass",
                "oracle_module_sha256", "oracle_eps_grid_lo",
                "oracle_eps_grid_hi", "oracle_D_floor_method",
                "oracle_D_at_Wfloor_method", "oracle_independence_scope",
                "oracle_time_limit_s", "oracle_presolve",
            ])
            require(required <= set(landed.columns),
                    "[oracle] landed sidecar lacks grid/model validation fields",
                    missing=sorted(required - set(landed.columns)))
            require(len(landed) == len(front_df),
                    "[oracle] landed sidecar row count differs from frontier",
                    expected=len(front_df), actual=len(landed))
            require(not bool(landed.duplicated(key).any()),
                    "[oracle] duplicate key in landed validation sidecar",
                    key=key)
            expected_keys = {
                tuple(row[c] for c in key)
                for _, row in front_df[key].iterrows()}
            landed_keys = {
                tuple(row[c] for c in key)
                for _, row in landed[key].iterrows()}
            require(landed_keys == expected_keys,
                    "[oracle] landed sidecar keys do not exactly cover the "
                    "frontier design",
                    missing=sorted(expected_keys - landed_keys, key=repr)[:10],
                    extra=sorted(landed_keys - expected_keys, key=repr)[:10])
            bounds = landed.merge(
                front_df[key + ["eps_grid_lo", "eps_grid_hi"]],
                on=key, how="left", validate="one_to_one")
            require(
                bool((bounds["oracle_eps_grid_lo"].astype(int)
                      == bounds["eps_grid_lo"].astype(int)).all())
                and bool((bounds["oracle_eps_grid_hi"].astype(int)
                          == bounds["eps_grid_hi"].astype(int)).all()),
                "[oracle-grid] landed independent grid bounds disagree with "
                "the frontier rows",
                sample=bounds.loc[
                    (bounds["oracle_eps_grid_lo"].astype(int)
                     != bounds["eps_grid_lo"].astype(int))
                    | (bounds["oracle_eps_grid_hi"].astype(int)
                       != bounds["eps_grid_hi"].astype(int)),
                    key + ["oracle_eps_grid_lo", "oracle_eps_grid_hi",
                           "eps_grid_lo", "eps_grid_hi"],
                ].head(10).to_dict("records"))
            require(bool((landed["abs_W_diff"].astype(float)
                          <= landed["W_atol"].astype(float)).all())
                    and bool((landed["D_production"].astype(int)
                              == landed["D_oracle"].astype(int)).all())
                    and bool((landed["U_production"].astype(int)
                              == landed["U_oracle"].astype(int)).all()),
                    "[oracle] landed sidecar violates the registered W/D/U "
                    "agreement rule")
            require(bool(landed["oracle_pass"].astype(bool).all()),
                    "[oracle] landed sidecar contains a failed comparison")
            require(
                set(landed["oracle_independence_scope"].astype(str))
                == {
                    "independent_model_and_incumbent_parser;"
                    "shared_HHCPFullParser_data_layer"
                },
                "[oracle] landed sidecar mislabels the oracle independence scope",
                observed=sorted(set(
                    landed["oracle_independence_scope"].astype(str))))

        with canonical_write(
                oracle_path, _validate_oracle, stage=STAGE,
                diagnostics={"artifact": "eps_oracle_validation.csv",
                             "scientific_validation": True}) as temp_path:
            oracle_validation_df.to_csv(temp_path, index=False)
        output_names.append("eps_oracle_validation.csv")

    # re-read the LANDED bytes and re-run the cross-artifact checks
    landed_rng = pd.read_csv(os.path.join(out_dir, "eps_range.csv"))
    landed_front = pd.read_csv(os.path.join(out_dir, "eps_frontier.csv"))
    log = extra_checks(landed_rng, landed_front, design, verbose=verbose,
                       formal=formal)
    hashes = hash_manifest(
        design["inputs"], [os.path.join(out_dir, n) for n in output_names])
    return hashes, reports, log


# --------------------------------------------------------------------------- #
def _rho_key(value):
    """Stable key for the six decimal rho labels used by the formal design."""
    value = float(value)
    require(math.isfinite(value), "rho is not finite", rho=value)
    return round(value, 8)


def _read_csv_audited(path, why=""):
    """Read CSV only after checking its landed bytes for structural defects."""
    require(os.path.isfile(path), "required R1 shard artifact is missing",
            path=path, why=why)
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    require(bool(rows), "R1 shard artifact is empty", path=path, why=why)
    header = [str(v).strip() for v in rows[0]]
    require(all(header), "R1 shard artifact has an empty column name",
            path=path, header=header, why=why)
    require(len(header) == len(set(header)),
            "R1 shard artifact has duplicate column names",
            path=path, header=header, why=why)
    ragged = [(line, len(row)) for line, row in enumerate(rows[1:], 2)
              if len(row) != len(header)]
    require(not ragged, "R1 shard artifact has ragged CSV rows",
            path=path, expected_fields=len(header), offending=ragged[:10],
            why=why)
    require(len(rows) > 1, "R1 shard artifact contains a header but no data",
            path=path, why=why)
    frame = pd.read_csv(path)
    require(tuple(str(c) for c in frame.columns) == tuple(header),
            "pandas view of R1 shard schema differs from landed CSV header",
            path=path, csv_header=header,
            pandas_header=[str(c) for c in frame.columns], why=why)
    return frame


def _require_columns(frame, required, *, artifact):
    missing = sorted(set(required) - set(frame.columns))
    require(not missing, "formal R1 shard lacks required columns",
            artifact=artifact, missing=missing,
            available=[str(c) for c in frame.columns])


def _strict_bool_series(series, *, artifact, column):
    tokens = series.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0"}
    observed = set(tokens)
    require(observed <= allowed,
            "formal R1 shard boolean column has an unsupported token",
            artifact=artifact, column=column, observed=sorted(observed))
    return tokens.isin(["true", "1"])


def _integer_value(value, *, artifact, column, key=None):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        fatal("formal R1 shard integer field is not numeric",
              artifact=artifact, column=column, value=value, key=key)
    require(math.isfinite(numeric) and numeric == int(numeric),
            "formal R1 shard integer field is non-integral",
            artifact=artifact, column=column, value=value, key=key)
    return int(numeric)


def _resolve_manifest_source_path(value, manifest_dir):
    """Resolve only manifest-declared paths; no machine-specific guess is used."""
    token = str(value).strip()
    require(bool(token), "manifest source path is empty", value=value)
    candidates = ([token] if os.path.isabs(token) else [
        os.path.join(manifest_dir, token),
        os.path.join(HERE, token),
        os.path.join(_EXPERIMENTS, "data_problems_solutions", token),
    ])
    hits = [os.path.abspath(p) for p in candidates if os.path.isfile(p)]
    if not hits and _source_is_withheld(token):
        return None
    require(bool(hits), "manifest-declared source file cannot be resolved",
            value=token, searched=[os.path.abspath(p) for p in candidates])
    return hits[0]


def _derive_shard_merge_authority(manifest_path, snap_id, *,
                                  allow_fixture_manifest=False):
    """Re-derive the full formal axis and provenance without reading a shard.

    Production accepts only the official ``rho_manifest.csv`` beside this
    module.  The fixture escape hatch is a Python-only test seam and is not
    exposed by the command-line entrypoint.
    """
    manifest_path = os.path.abspath(manifest_path)
    require(os.path.isfile(manifest_path), "R1 shard-merge manifest is missing",
            path=manifest_path)
    official = os.path.join(HERE, "rho_manifest.csv")
    if not allow_fixture_manifest:
        require(os.path.samefile(manifest_path, official),
                "formal R1 shard merge accepts only the official rho_manifest.csv",
                requested=manifest_path, official=official)
    man = _read_csv_audited(manifest_path, why="independent formal R1 merge axis")
    required = {
        "rho_target", "seed", "hhcp", "sol", "N", "rho_global",
        "rho_used_mean", "rho_used_max", "src", "src_sha256", "feasible",
        "unplaced", "h_sha256", "h_bytes", "s_sha256", "s_bytes",
        "generator", "generator_sha256", "family_runner",
        "family_runner_sha256",
    }
    _require_columns(man, required, artifact="rho_manifest.csv")
    observed_rhos = {_rho_key(v) for v in man["rho_target"]}
    require(observed_rhos == set(FORMAL_RHOS),
            "formal R1 merge manifest must contain exactly the six registered rho levels",
            expected=list(FORMAL_RHOS), observed=sorted(observed_rhos))
    require(not bool(man.duplicated(["rho_target", "seed"]).any()),
            "formal R1 merge manifest contains duplicate (rho_target,seed) rows")
    seed_counts = {
        _rho_key(rho): sorted(set(int(v) for v in group["seed"]))
        for rho, group in man.groupby("rho_target")
    }
    require(len(man) == 30
            and all(seed_counts.get(rho) == [1, 2, 3, 4, 5]
                    for rho in FORMAL_RHOS),
            "formal R1 merge requires 30 scenarios: seeds 1..5 at each rho",
            rows=len(man), seeds_by_rho=seed_counts)

    axis, future, expected = {}, {}, {}
    inputs = [manifest_path, DESIGN_MANIFEST, ORACLE_MODULE]
    inputs_by_rho = {rho: [manifest_path, DESIGN_MANIFEST, ORACLE_MODULE]
                     for rho in FORMAL_RHOS}
    man_dir = os.path.dirname(manifest_path)
    for _, row in man.iterrows():
        rho, seed = _rho_key(row["rho_target"]), int(row["seed"])
        hp = os.path.abspath(os.path.join(man_dir, str(row["hhcp"])))
        sp = os.path.abspath(os.path.join(man_dir, str(row["sol"])))
        for path, sha_col, byte_col, label in (
                (hp, "h_sha256", "h_bytes", "H instance"),
                (sp, "s_sha256", "s_bytes", "incumbent S plan")):
            require(os.path.isfile(path),
                    "R1 merge manifest points at a missing scientific input",
                    rho=rho, seed=seed, input_role=label, path=path)
            actual_sha = sha256_file(path)
            actual_bytes = os.path.getsize(path)
            require(actual_sha == str(row[sha_col]).strip().lower()
                    and actual_bytes == int(row[byte_col]),
                    "R1 merge H/S bytes disagree with manifest digest/size",
                    rho=rho, seed=seed, input_role=label, path=path,
                    declared_sha256=str(row[sha_col]).strip(),
                    actual_sha256=actual_sha,
                    declared_bytes=int(row[byte_col]), actual_bytes=actual_bytes)
            inputs.append(path); inputs_by_rho[rho].append(path)
        source_paths = {}
        for role, path_col, sha_col in (
                ("source", "src", "src_sha256"),
                ("generator", "generator", "generator_sha256"),
                ("family_runner", "family_runner", "family_runner_sha256")):
            source_path = _resolve_manifest_source_path(row[path_col], man_dir)
            if source_path is None:
                # Release marker names this source as withheld; its recorded
                # digest is the only provenance available here.
                continue
            actual_sha = sha256_file(source_path)
            require(actual_sha == str(row[sha_col]).strip().lower(),
                    "R1 merge manifest source bytes disagree with SHA256",
                    rho=rho, seed=seed, source_role=role, path=source_path,
                    declared_sha256=str(row[sha_col]).strip(),
                    actual_sha256=actual_sha)
            source_paths[role] = source_path
            inputs.append(source_path); inputs_by_rho[rho].append(source_path)
        require(manifest_bool(row["feasible"], "feasible")
                and int(row["unplaced"]) == 0,
                "formal R1 merge requires a feasible incumbent with no unplaced task",
                rho=rho, seed=seed, feasible=row["feasible"],
                unplaced=row["unplaced"])
        parser = HHCPFullParser(hp)
        sa = load_static(sp)
        nurses = set(parser.nurses)
        declared_n = int(row["N"])
        require(declared_n == len(nurses),
                "formal R1 merge manifest N differs from parsed H instance",
                rho=rho, seed=seed, manifest_N=declared_n,
                parsed_nurses=len(nurses))
        actives = active_absentees(parser, sa, None)
        require(bool(actives), "formal R1 merge scenario has no eligible disruption",
                rho=rho, seed=seed)
        axis[(rho, seed)] = list(actives)
        slots = future_slots(sa)
        structural = {}
        for absent in actives:
            eligibility = assert_incumbent_eligible(
                parser, sa, nurses, absent,
                label=f"R1 merge rho={rho} seed={seed} absent={absent}")
            cand_a1, empty_a1 = candidate_sets(parser, sa, nurses, absent, "A1")
            cand_a2, empty_a2 = candidate_sets(parser, sa, nurses, absent, "A2")
            nesting = assert_A1_candidates_subset_A2(cand_a1, cand_a2, absent)
            future[(rho, seed, absent)] = list(slots)
            structural[int(absent)] = {
                "n_incumbent_pairs_checked": int(
                    eligibility["n_incumbent_pairs_checked"]),
                "n_tasks_nested_A1_subset_A2": int(nesting["n_tasks_nested"]),
                "n_empty_cand_after_absence_A1": len(empty_a1),
                "n_empty_cand_after_absence_A2": len(empty_a2),
            }
        expected[(rho, seed)] = {
            "absentees": list(actives),
            "N": declared_n,
            "rho_target": rho,
            "rho_global": float(row["rho_global"]),
            "rho_used_mean": float(row["rho_used_mean"]),
            "rho_used_max": float(row["rho_used_max"]),
            "src_sha256": str(row["src_sha256"]).strip().lower(),
            "h_sha256": str(row["h_sha256"]).strip().lower(),
            "s_sha256": str(row["s_sha256"]).strip().lower(),
            "structural": structural,
        }

    # Keep each path once: repeated source/generator inputs are common to all 30
    # scenarios and hashing them 30 times would add noise without evidence.
    inputs = list(dict.fromkeys(os.path.abspath(p) for p in inputs))
    inputs_by_rho = {
        rho: list(dict.fromkeys(os.path.abspath(p) for p in paths))
        for rho, paths in inputs_by_rho.items()
    }
    return {
        "manifest": man,
        "manifest_path": manifest_path,
        "axis": axis,
        "future_slots": future,
        "scenario_expected": expected,
        "inputs": inputs,
        "inputs_by_rho": inputs_by_rho,
        "snap_id": snap_id,
    }


def _assert_scenario_provenance(frame, authority, *, artifact):
    """Bind every landed scientific row to independently parsed manifest facts."""
    _require_columns(frame, [
        "rho", "seed", "absent", "snap_id", "absence_rule", "absence_cap",
        "eligible_disruption", "eligible_disruptions_in_scenario",
        "active_caregiver_count", "rho_target", "rho_global",
        "rho_used_mean", "rho_used_max", "src_sha256",
        "incumbent_flat_feasible",
    ], artifact=artifact)
    require(set(frame["snap_id"].astype(str)) == {authority["snap_id"]},
            "formal R1 shard carries a mixed or unexpected SNAP",
            artifact=artifact, expected=authority["snap_id"],
            observed=sorted(set(frame["snap_id"].astype(str))))
    require(set(frame["absence_rule"].astype(str)) == {ABSENCE_RULE}
            and set(frame["absence_cap"].astype(int)) == {-1},
            "formal R1 shard carries a debug absence rule/cap",
            artifact=artifact)
    require(bool(_strict_bool_series(
                frame["eligible_disruption"], artifact=artifact,
                column="eligible_disruption").all())
            and bool(_strict_bool_series(
                frame["incumbent_flat_feasible"], artifact=artifact,
                column="incumbent_flat_feasible").all()),
            "formal R1 shard has an ineligible disruption or infeasible incumbent",
            artifact=artifact)
    for key, expected in authority["scenario_expected"].items():
        rho, seed = key
        group = frame[
            np.isclose(frame["rho"].astype(float), rho, rtol=0, atol=1e-12)
            & (frame["seed"].astype(int) == seed)]
        require(not group.empty, "formal R1 shard is missing a manifest scenario",
                artifact=artifact, rho=rho, seed=seed)
        require(set(group["absent"].astype(int)) == set(expected["absentees"]),
                "formal R1 shard disruption axis differs from official S plan",
                artifact=artifact, rho=rho, seed=seed,
                expected=expected["absentees"],
                observed=sorted(set(group["absent"].astype(int))))
        exact = {
            "src_sha256": expected["src_sha256"],
            "eligible_disruptions_in_scenario": len(expected["absentees"]),
            "active_caregiver_count": expected["N"],
        }
        for column, value in exact.items():
            observed = set(group[column].astype(str)) if isinstance(value, str) \
                else set(group[column].astype(int))
            require(observed == {value},
                    "formal R1 shard provenance differs from manifest",
                    artifact=artifact, rho=rho, seed=seed, column=column,
                    expected=value, observed=sorted(observed))
        numeric = {
            "rho_target": expected["rho_target"],
            "rho_global": expected["rho_global"],
            "rho_used_mean": expected["rho_used_mean"],
            "rho_used_max": expected["rho_used_max"],
        }
        for column, value in numeric.items():
            require(bool(np.isclose(group[column].astype(float).to_numpy(),
                                    value, rtol=0, atol=1e-12).all()),
                    "formal R1 shard numeric provenance differs from manifest",
                    artifact=artifact, rho=rho, seed=seed, column=column,
                    expected=value)


def _range_grids_from_authority(rng, authority, *, artifact):
    """Check the independent disruption axis before trusting grid endpoints."""
    required = [
        "rho", "seed", "absent", "U_min_A1", "U_min_A2", "U_common",
        "D_floor_A1", "D_at_Wfloor_A1", "D_floor_A2", "D_at_Wfloor_A2",
        "eps_grid_lo", "eps_grid_hi", "eps_grid_size",
        "n_incumbent_pairs_checked", "n_tasks_nested_A1_subset_A2",
        "n_empty_cand_after_absence_A1", "n_empty_cand_after_absence_A2",
    ]
    _require_columns(rng, required, artifact=artifact)
    observed = [(_rho_key(row["rho"]), int(row["seed"]), int(row["absent"]))
                for _, row in rng.iterrows()]
    expected = {(rho, seed, int(absent))
                for (rho, seed), absentees in authority["axis"].items()
                for absent in absentees}
    require(len(observed) == len(set(observed)),
            "formal R1 range shard has duplicate disruption keys",
            artifact=artifact)
    require(set(observed) == expected,
            "formal R1 range shard does not cover the re-derived disruption axis",
            artifact=artifact,
            missing=sorted(expected - set(observed), key=repr)[:20],
            extra=sorted(set(observed) - expected, key=repr)[:20],
            expected_count=len(expected), actual_count=len(set(observed)))
    grids = {}
    for _, row in rng.iterrows():
        key = (_rho_key(row["rho"]), int(row["seed"]), int(row["absent"]))
        values = {column: _integer_value(row[column], artifact=artifact,
                                          column=column, key=key)
                  for column in required[3:]}
        require(values["U_min_A2"] <= values["U_min_A1"]
                and values["U_common"] == values["U_min_A1"],
                "formal R1 range violates common-service construction",
                artifact=artifact, key=key,
                U_min_A1=values["U_min_A1"],
                U_min_A2=values["U_min_A2"],
                U_common=values["U_common"])
        lo = max(values["D_floor_A1"], values["D_floor_A2"])
        hi = max(values["D_at_Wfloor_A1"], values["D_at_Wfloor_A2"])
        require(values["eps_grid_lo"] == lo
                and values["eps_grid_hi"] == hi
                and hi >= lo
                and values["eps_grid_size"] == hi - lo + 1,
                "formal R1 range violates the registered complete epsilon-grid rule",
                artifact=artifact, key=key, derived=(lo, hi),
                persisted=(values["eps_grid_lo"], values["eps_grid_hi"],
                           values["eps_grid_size"]))
        scenario = authority["scenario_expected"][(key[0], key[1])]
        structural = scenario["structural"][key[2]]
        for column, expected_value in structural.items():
            require(values[column] == int(expected_value),
                    "formal R1 range structural evidence differs from H/S inputs",
                    artifact=artifact, key=key, column=column,
                    expected=expected_value, actual=values[column])
        grids[key] = (lo, hi)
    return grids


def _assert_frontier_formal(front, grids, policy, *, artifact):
    _require_columns(front, [
        "rho", "seed", "absent", "method", "eps", "U_min_A1",
        "U_min_A2", "U_common", "U", "D", "eps_grid_lo",
        "eps_grid_hi", "eps_semantics", "lex_order", "solver_status_chain",
        "lex_stage_chain", "stage", COL.STATUS_CODE, COL.STATUS_NAME,
        COL.PROVEN_OPTIMAL, COL.STOP_REASON, COL.TIME_LIMIT,
        "canonical_assignment", COL.ASSIGN_HASH,
        "effective_epsilon", "deviation_parity", "epsilon_tightened_by",
        "solve_anchor_epsilon", "solver_executed_for_requested_epsilon",
        "epsilon_solution_source", "epsilon_reuse_proof",
        "epsilon_equivalence_rule", "n_tasks_future",
        "n_incumbent_pairs_available",
    ], artifact=artifact)
    expected = set()
    for (rho, seed, absent), (lo, hi) in grids.items():
        for method in ("A1", "A2"):
            for eps in range(lo, hi + 1):
                expected.add((rho, seed, absent, method, eps))
    actual = [(_rho_key(row["rho"]), int(row["seed"]), int(row["absent"]),
               str(row["method"]), int(row["eps"]))
              for _, row in front.iterrows()]
    require(len(actual) == len(set(actual)) and set(actual) == expected,
            "formal R1 frontier is not the complete designed epsilon grid",
            artifact=artifact, expected_count=len(expected), actual_count=len(set(actual)),
            missing=sorted(expected - set(actual), key=repr)[:20],
            extra=sorted(set(actual) - expected, key=repr)[:20])
    require(set(front["method"].astype(str)) == {"A1", "A2"}
            and set(front["eps_semantics"].astype(str))
            == {"absolute_integer_sum_D"}
            and set(front["lex_order"].astype(str)) == {"U>W>D"}
            and set(front["solver_status_chain"].astype(str)) == {"0|0"}
            and set(front["lex_stage_chain"].astype(str))
            == {"lex2_minW|lex3_minD"},
            "formal R1 frontier carries wrong regime/epsilon/lexicographic semantics",
            artifact=artifact)
    require(set(front["stage"].astype(str)) == {STAGE}
            and set(front[COL.STATUS_CODE].astype(int)) == {0}
            and set(front[COL.STATUS_NAME].astype(str)) == {"Optimal"}
            and bool(_strict_bool_series(
                front[COL.PROVEN_OPTIMAL], artifact=artifact,
                column=COL.PROVEN_OPTIMAL).all())
            and set(front[COL.STOP_REASON].astype(str)) == {"proven_optimal"}
            and bool(np.equal(front[COL.TIME_LIMIT].astype(float).to_numpy(),
                              policy.time_limit_s).all()),
            "formal R1 frontier solver policy evidence is inconsistent",
            artifact=artifact)
    canonical = _strict_bool_series(
        front["canonical_assignment"], artifact=artifact,
        column="canonical_assignment")
    hashes = front[COL.ASSIGN_HASH].fillna("").astype(str).str.strip()
    require(not bool(canonical.any()) and bool((hashes == "").all()),
            "formal R1 frontier must disable assignment canonicalization/hash evidence",
            artifact=artifact)
    require(bool((front["U"].astype(int) == front["U_common"].astype(int)).all())
            and bool((front["U_common"].astype(int)
                      == front["U_min_A1"].astype(int)).all())
            and bool((front["U_min_A2"].astype(int)
                      <= front["U_min_A1"].astype(int)).all())
            and bool((front["D"].astype(int)
                      <= front["effective_epsilon"].astype(int)).all())
            and set(front["epsilon_equivalence_rule"].astype(str))
            == {EPSILON_PARITY_RULE}
            and set(front["epsilon_solution_source"].astype(str))
            <= {EPSILON_DIRECT_SOURCE, EPSILON_PARITY_REUSE_SOURCE,
                EPSILON_W_FLOOR_REUSE_SOURCE},
            "formal R1 frontier violates service equality or epsilon budget",
            artifact=artifact)


def _assert_oracle_formal(oracle, front, rng, policy, *, artifact):
    key = ["rho", "seed", "absent", "method", "eps"]
    required = key + [
        "snap_id", "W_production", "W_oracle", "abs_W_diff", "W_atol",
        "D_production", "D_oracle", "U_production", "U_oracle",
        "oracle_solver_status_chain", "oracle_pass", "oracle_module_sha256",
        "oracle_eps_grid_lo", "oracle_eps_grid_hi", "oracle_D_floor_method",
        "oracle_D_at_Wfloor_method", "oracle_independence_scope",
        "oracle_time_limit_s", "oracle_presolve",
        "oracle_effective_epsilon", "oracle_deviation_parity",
        "oracle_epsilon_tightened_by", "oracle_solve_anchor_epsilon",
        "oracle_solver_executed_for_requested_epsilon",
        "oracle_epsilon_solution_source", "oracle_epsilon_reuse_proof",
        "oracle_epsilon_equivalence_rule",
    ]
    _require_columns(oracle, required, artifact=artifact)
    actual_keys = [(_rho_key(row["rho"]), int(row["seed"]), int(row["absent"]),
                    str(row["method"]), int(row["eps"]))
                   for _, row in oracle.iterrows()]
    front_keys = {(_rho_key(row["rho"]), int(row["seed"]), int(row["absent"]),
                   str(row["method"]), int(row["eps"]))
                  for _, row in front.iterrows()}
    require(len(actual_keys) == len(set(actual_keys))
            and set(actual_keys) == front_keys,
            "formal R1 oracle sidecar does not exactly cover the frontier",
            artifact=artifact, expected=len(front_keys), actual=len(set(actual_keys)),
            missing=sorted(front_keys - set(actual_keys), key=repr)[:20],
            extra=sorted(set(actual_keys) - front_keys, key=repr)[:20])
    require(bool(_strict_bool_series(
                oracle["oracle_pass"], artifact=artifact,
                column="oracle_pass").all())
            and not bool(_strict_bool_series(
                oracle["oracle_presolve"], artifact=artifact,
                column="oracle_presolve").any())
            and set(oracle["oracle_solver_status_chain"].astype(str)) == {"0|0"}
            and set(oracle["oracle_module_sha256"].astype(str))
            == {sha256_file(ORACLE_MODULE)}
            and bool(np.equal(
                oracle["oracle_time_limit_s"].astype(float).to_numpy(),
                policy.time_limit_s).all())
            and set(oracle["oracle_independence_scope"].astype(str))
            == {"independent_model_and_incumbent_parser;"
                "shared_HHCPFullParser_data_layer"},
            "formal R1 oracle solver/source/independence evidence is inconsistent",
            artifact=artifact)
    joined = oracle.merge(
        front[key + ["W", "D", "U", "eps_grid_lo", "eps_grid_hi",
                     "effective_epsilon", "deviation_parity",
                     "epsilon_tightened_by"]],
        on=key, how="outer", validate="one_to_one", indicator=True)
    require(set(joined["_merge"].astype(str)) == {"both"}
            and bool(np.isclose(joined["W_production"].astype(float),
                                joined["W"].astype(float),
                                rtol=0, atol=ORACLE_W_ATOL).all())
            and bool(np.isclose(joined["W_oracle"].astype(float),
                                joined["W"].astype(float),
                                rtol=0, atol=ORACLE_W_ATOL).all())
            and bool((joined["abs_W_diff"].astype(float)
                      <= joined["W_atol"].astype(float)).all())
            and bool((joined["W_atol"].astype(float) == ORACLE_W_ATOL).all())
            and bool((joined["D_production"].astype(int)
                      == joined["D"].astype(int)).all())
            and bool((joined["D_oracle"].astype(int)
                      == joined["D"].astype(int)).all())
            and bool((joined["U_production"].astype(int)
                      == joined["U"].astype(int)).all())
            and bool((joined["U_oracle"].astype(int)
                      == joined["U"].astype(int)).all())
            and bool((joined["oracle_eps_grid_lo"].astype(int)
                      == joined["eps_grid_lo"].astype(int)).all())
            and bool((joined["oracle_eps_grid_hi"].astype(int)
                      == joined["eps_grid_hi"].astype(int)).all())
            and bool((joined["oracle_effective_epsilon"].astype(int)
                      == joined["effective_epsilon"].astype(int)).all())
            and bool((joined["oracle_deviation_parity"].astype(int)
                      == joined["deviation_parity"].astype(int)).all())
            and bool((joined["oracle_epsilon_tightened_by"].astype(int)
                      == joined["epsilon_tightened_by"].astype(int)).all())
            and set(oracle["oracle_epsilon_equivalence_rule"].astype(str))
            == {EPSILON_PARITY_RULE},
            "formal R1 oracle values/grid disagree with production frontier",
            artifact=artifact)
    range_lookup = {
        (_rho_key(row["rho"]), int(row["seed"]), int(row["absent"])): row
        for _, row in rng.iterrows()
    }
    for _, row in oracle.iterrows():
        disruption = (_rho_key(row["rho"]), int(row["seed"]), int(row["absent"]))
        rr = range_lookup[disruption]
        method = str(row["method"])
        require(int(row["oracle_D_floor_method"]) == int(rr[f"D_floor_{method}"])
                and int(row["oracle_D_at_Wfloor_method"])
                == int(rr[f"D_at_Wfloor_{method}"]),
                "formal R1 oracle endpoints disagree with eps_range.csv",
                artifact=artifact, key=disruption, method=method)


def _diag_eps_token(value):
    if pd.isna(value) or str(value).strip() == "":
        return None
    return _integer_value(value, artifact="eps_solver_diagnostics.csv",
                          column="epsilon")


def _assert_diagnostics_formal(diagnostics, rng, policy, *, artifact):
    required = [
        "stage", "regime", "method", "rho", "seed", "absent", "lex_stage",
        "objective_stage", "epsilon", "solver_status_code", "solver_message",
        "wall_time_s", "time_limit_s", "presolve", "snap_id",
        "lex_w_row_slack",
        "effective_epsilon", "deviation_parity", "epsilon_tightened_by",
        "solver_executed", "epsilon_solution_source",
        "solve_anchor_epsilon", "epsilon_equivalence_rule",
    ]
    _require_columns(diagnostics, required, artifact=artifact)
    require(set(diagnostics["stage"].astype(str)) == {STAGE}
            and set(diagnostics["solver_status_code"].astype(int)) == {0}
            and not bool(_strict_bool_series(
                diagnostics["presolve"], artifact=artifact,
                column="presolve").any())
            and bool(np.equal(
                diagnostics["time_limit_s"].astype(float).to_numpy(),
                policy.time_limit_s).all())
            and bool(np.isclose(
                diagnostics["lex_w_row_slack"].astype(float).to_numpy(),
                LEX_W_ROW_SLACK, rtol=0, atol=1e-15).all()),
            "formal R1 diagnostics disagree with solver presolve/time/status policy",
            artifact=artifact)
    wall = diagnostics["wall_time_s"].astype(float).to_numpy()
    require(bool(np.isfinite(wall).all()) and bool((wall >= 0).all()),
            "formal R1 diagnostics contain invalid wall-clock evidence",
            artifact=artifact)
    require(bool((diagnostics["regime"].astype(str)
                  == diagnostics["method"].astype(str)).all())
            and set(diagnostics["method"].astype(str)) == {"A1", "A2"},
            "formal R1 diagnostics method/regime labels disagree",
            artifact=artifact)
    objective_by_stage = {
        "lex1_minU": "U", "grid_minD": "D",
        "lex2_minW": "W", "lex3_minD": "D",
    }
    require(set(diagnostics["lex_stage"].astype(str)) == set(objective_by_stage),
            "formal R1 diagnostics contain an unknown/missing lexicographic stage",
            artifact=artifact,
            observed=sorted(set(diagnostics["lex_stage"].astype(str))))
    require(bool(diagnostics.apply(
        lambda row: str(row["objective_stage"])
        == objective_by_stage[str(row["lex_stage"])], axis=1).all()),
        "formal R1 diagnostics objective stage disagrees with lex stage",
        artifact=artifact)

    for _, row in diagnostics.iterrows():
        requested = _diag_eps_token(row["epsilon"])
        executed = manifest_bool(row["solver_executed"], "solver_executed")
        source = str(row["epsilon_solution_source"])
        rule = str(row["epsilon_equivalence_rule"])
        if requested is None:
            require(executed and source == "direct_non_epsilon_solve"
                    and rule == "not_applicable_no_epsilon"
                    and pd.isna(row["effective_epsilon"])
                    and pd.isna(row["solve_anchor_epsilon"]),
                    "formal R1 non-epsilon diagnostic has false reuse metadata",
                    artifact=artifact, row=row.to_dict())
            continue
        effective = _integer_value(
            row["effective_epsilon"], artifact=artifact,
            column="effective_epsilon")
        parity = _integer_value(
            row["deviation_parity"], artifact=artifact,
            column="deviation_parity")
        tightened = _integer_value(
            row["epsilon_tightened_by"], artifact=artifact,
            column="epsilon_tightened_by")
        anchor = _integer_value(
            row["solve_anchor_epsilon"], artifact=artifact,
            column="solve_anchor_epsilon")
        require(parity in (0, 1)
                and effective == requested - ((requested - parity) % 2)
                and tightened == requested - effective in (0, 1)
                and rule == EPSILON_PARITY_RULE,
                "formal R1 diagnostic has invalid parity/effective budget",
                artifact=artifact, row=row.to_dict())
        if source == EPSILON_DIRECT_SOURCE:
            require(executed and requested == effective == anchor,
                    "formal R1 direct diagnostic has invalid anchor metadata",
                    artifact=artifact, row=row.to_dict())
        elif source == EPSILON_PARITY_REUSE_SOURCE:
            require(not executed and requested == effective + 1
                    and anchor == effective and float(row["wall_time_s"]) == 0.0,
                    "formal R1 parity-reuse diagnostic is malformed",
                    artifact=artifact, row=row.to_dict())
        elif source == EPSILON_W_FLOOR_REUSE_SOURCE:
            require(not executed and anchor == -1
                    and float(row["wall_time_s"]) == 0.0,
                    "formal R1 W-floor-reuse diagnostic is malformed",
                    artifact=artifact, row=row.to_dict())
        else:
            fatal("formal R1 diagnostic has unknown epsilon solution source",
                  artifact=artifact, source=source, row=row.to_dict())

    expected = set()
    for _, row in rng.iterrows():
        rho, seed, absent = (_rho_key(row["rho"]), int(row["seed"]),
                             int(row["absent"]))
        lo, hi = int(row["eps_grid_lo"]), int(row["eps_grid_hi"])
        for method in ("A1", "A2"):
            for stage in objective_by_stage:
                expected.add((rho, seed, absent, method, stage, None))
            for eps in range(lo, hi + 1):
                expected.add((rho, seed, absent, method, "lex2_minW", eps))
                expected.add((rho, seed, absent, method, "lex3_minD", eps))
    actual = [(_rho_key(row["rho"]), int(row["seed"]), int(row["absent"]),
               str(row["method"]), str(row["lex_stage"]),
               _diag_eps_token(row["epsilon"]))
              for _, row in diagnostics.iterrows()]
    require(len(actual) == len(set(actual)) and set(actual) == expected,
            "formal R1 diagnostics do not exactly cover every production solve",
            artifact=artifact, expected_count=len(expected),
            actual_count=len(set(actual)),
            missing=sorted(expected - set(actual), key=repr)[:20],
            extra=sorted(set(actual) - expected, key=repr)[:20])


def _axis_records(axis):
    return [
        {"rho": float(rho), "seed": int(seed),
         "absentees": [int(v) for v in absentees],
         "eligible_disruptions": len(absentees)}
        for (rho, seed), absentees in sorted(axis.items())
    ]


def _actual_axis_records(rng):
    records = []
    for (rho, seed), group in rng.groupby(["rho", "seed"], sort=True):
        values = sorted(set(int(v) for v in group["absent"]))
        records.append({"rho": float(_rho_key(rho)), "seed": int(seed),
                        "absentees": values,
                        "eligible_disruptions": len(values)})
    return records


def _epsilon_axis_records(rng):
    return [
        {"rho": float(_rho_key(row["rho"])), "seed": int(row["seed"]),
         "absent": int(row["absent"]),
         "eps_grid_lo": int(row["eps_grid_lo"]),
         "eps_grid_hi": int(row["eps_grid_hi"]),
         "eps_grid_size": int(row["eps_grid_size"])}
        for _, row in rng.sort_values(["rho", "seed", "absent"]).iterrows()
    ]


def merge_formal_shards(manifest_path, out_dir, shard_dirs, *, snap_id,
                        cli_time_limit_s, verbose=True,
                        allow_fixture_manifest=False):
    """Validate six mutually exclusive formal rho shards and rebuild R1.

    No model constructor or solver is called here.  Every axis is re-derived
    from the official H/S inputs, while each epsilon axis is checked three ways:
    range derivation fields, the complete frontier, and the independently
    generated oracle sidecar.
    """
    assert_no_staging_import()
    assert_no_absolute_runtime_paths(__file__)
    policy = solver_policy(cli_time_limit_s=cli_time_limit_s,
                           require_cli_time_limit=True)
    assert_runtime_contract(policy)
    snap_id = require_snap_binding(DESIGN_MANIFEST, snap_id, stage=STAGE)
    dirs = [os.path.abspath(path) for path in shard_dirs]
    require(len(dirs) == len(FORMAL_RHOS),
            "formal R1 merge requires exactly six shard directories",
            expected=len(FORMAL_RHOS), actual=len(dirs))
    require(len(set(dirs)) == len(dirs),
            "formal R1 merge received a duplicate shard directory",
            shard_dirs=dirs)
    out_dir = os.path.abspath(out_dir)
    require(out_dir not in set(dirs),
            "formal R1 merge output directory may not overwrite an input shard",
            out_dir=out_dir, shard_dirs=dirs)
    authority = _derive_shard_merge_authority(
        manifest_path, snap_id,
        allow_fixture_manifest=allow_fixture_manifest)

    range_frames, frontier_frames = [], []
    diagnostics_frames, oracle_frames = [], []
    schemas = {name: None for name in SHARD_ARTIFACTS}
    receipts, verified = [], []
    seen_rhos = {}
    all_grids = {}
    for shard_dir in dirs:
        paths = {name: os.path.join(shard_dir, name)
                 for name in SHARD_ARTIFACTS}
        hashes = {}
        frames = {}
        for name, path in paths.items():
            hashes[name] = sha256_file(path) if os.path.isfile(path) else None
            require(hashes[name] is not None,
                    "formal R1 shard lacks a required landed artifact",
                    shard_dir=shard_dir, artifact=name, path=path)
            frames[name] = _read_csv_audited(
                path, why=f"formal R1 shard {name}")
            schema = tuple(str(c) for c in frames[name].columns)
            if schemas[name] is None:
                schemas[name] = schema
            require(schema == schemas[name],
                    "formal R1 shard artifact schema differs across rho shards",
                    artifact=name, shard_dir=shard_dir,
                    expected=list(schemas[name]), observed=list(schema))
            _require_columns(frames[name], ["rho", "snap_id"], artifact=name)
        rho_sets = {
            name: {_rho_key(v) for v in frame["rho"]}
            for name, frame in frames.items()
        }
        common = set().union(*rho_sets.values())
        require(len(common) == 1
                and all(values == common for values in rho_sets.values()),
                "each formal R1 shard must contain one mutually consistent rho",
                shard_dir=shard_dir,
                observed={k: sorted(v) for k, v in rho_sets.items()})
        rho = next(iter(common))
        require(rho in set(FORMAL_RHOS),
                "formal R1 shard rho is outside the registered six-level design",
                shard_dir=shard_dir, rho=rho)
        require(rho not in seen_rhos,
                "formal R1 shard rho values are not mutually exclusive",
                rho=rho, first=seen_rhos.get(rho), second=shard_dir)
        seen_rhos[rho] = shard_dir
        for name, frame in frames.items():
            observed_snap = set(frame["snap_id"].astype(str))
            require(observed_snap == {snap_id},
                    "formal R1 shard contains a mixed or unexpected SNAP",
                    shard_dir=shard_dir, artifact=name,
                    expected=snap_id, observed=sorted(observed_snap))

        shard_authority = dict(authority)
        shard_authority["axis"] = {
            key: value for key, value in authority["axis"].items()
            if _rho_key(key[0]) == rho}
        shard_authority["future_slots"] = {
            key: value for key, value in authority["future_slots"].items()
            if _rho_key(key[0]) == rho}
        shard_authority["scenario_expected"] = {
            key: value for key, value in authority["scenario_expected"].items()
            if _rho_key(key[0]) == rho}
        rng = frames["eps_range.csv"]
        front = frames["eps_frontier.csv"]
        diagnostics = frames["eps_solver_diagnostics.csv"]
        oracle = frames["eps_oracle_validation.csv"]
        _assert_scenario_provenance(
            rng, shard_authority, artifact="eps_range.csv")
        _assert_scenario_provenance(
            front, shard_authority, artifact="eps_frontier.csv")
        grids = _range_grids_from_authority(
            rng, shard_authority, artifact="eps_range.csv")
        shard_design = build_design(
            shard_authority["axis"], grids,
            authority["inputs_by_rho"][rho])
        shard_design["future_slots"] = shard_authority["future_slots"]
        shard_design["snap_id"] = snap_id
        shard_design["absence_rule"] = ABSENCE_RULE
        validate_stage_output(
            paths["eps_range.csv"],
            range_spec(shard_design, snap_id=snap_id, formal=True))
        validate_stage_output(
            paths["eps_frontier.csv"],
            frontier_spec(shard_design, policy, snap_id=snap_id, formal=True))
        extra_checks(rng, front, shard_design, verbose=False, formal=True)
        _assert_frontier_formal(
            front, grids, policy, artifact="eps_frontier.csv")
        _assert_oracle_formal(
            oracle, front, rng, policy, artifact="eps_oracle_validation.csv")
        _assert_diagnostics_formal(
            diagnostics, rng, policy, artifact="eps_solver_diagnostics.csv")

        all_grids.update(grids)
        range_frames.append(rng); frontier_frames.append(front)
        diagnostics_frames.append(diagnostics); oracle_frames.append(oracle)
        receipt = {
            "rho": float(rho),
            "shard_label": f"rho={rho:.2f}",
            "snap_id": snap_id,
            "row_counts": {name: int(len(frames[name]))
                           for name in SHARD_ARTIFACTS},
            "artifact_sha256": {name: hashes[name]
                                for name in SHARD_ARTIFACTS},
        }
        receipts.append(receipt)
        verified.extend((paths[name], hashes[name]) for name in SHARD_ARTIFACTS)

    require(set(seen_rhos) == set(FORMAL_RHOS),
            "formal R1 shard family has missing or extra rho levels",
            expected=list(FORMAL_RHOS), observed=sorted(seen_rhos))
    for path, expected_sha in verified:
        require(sha256_file(path) == expected_sha,
                "formal R1 shard artifact changed after validation",
                path=path, expected_sha256=expected_sha,
                actual_sha256=sha256_file(path))

    rng = pd.concat(range_frames, ignore_index=True).sort_values(
        ["rho", "seed", "absent"]).reset_index(drop=True)
    front = pd.concat(frontier_frames, ignore_index=True).sort_values(
        ["rho", "seed", "absent", "method", "eps"]).reset_index(drop=True)
    diagnostics = pd.concat(diagnostics_frames, ignore_index=True).sort_values(
        ["rho", "seed", "absent", "method", "lex_stage", "epsilon"],
        na_position="first").reset_index(drop=True)
    oracle = pd.concat(oracle_frames, ignore_index=True).sort_values(
        ["rho", "seed", "absent", "method", "eps"]).reset_index(drop=True)
    full_inputs = list(authority["inputs"])
    full_inputs.extend(path for path, _ in verified)
    full_design = build_design(authority["axis"], all_grids,
                               list(dict.fromkeys(full_inputs)))
    full_design["future_slots"] = authority["future_slots"]
    full_design["snap_id"] = snap_id
    full_design["absence_rule"] = ABSENCE_RULE
    hashes, reports, log = write_validated(
        out_dir, rng, front, diagnostics, oracle, full_design, policy,
        snap_id=snap_id, formal=True, verbose=verbose)

    for path, expected_sha in verified:
        require(sha256_file(path) == expected_sha,
                "formal R1 shard artifact changed while combined family was published",
                path=path, expected_sha256=expected_sha,
                actual_sha256=sha256_file(path))
    merged_paths = {name: os.path.join(out_dir, name) for name in SHARD_ARTIFACTS}
    expected_axis = _axis_records(authority["axis"])
    actual_axis = _actual_axis_records(rng)
    require(actual_axis == expected_axis,
            "combined R1 axis differs from independently re-derived H/S axis",
            expected=expected_axis, actual=actual_axis)
    epsilon_axis = _epsilon_axis_records(rng)
    receipt_object = {
        "stage": STAGE,
        "mode": "formal_six_rho_shard_validation_and_rebuild_without_resolve",
        "snap_id": snap_id,
        "design_manifest_sha256": sha256_file(DESIGN_MANIFEST),
        "rho_manifest_sha256": sha256_file(authority["manifest_path"]),
        "registered_rhos": [float(v) for v in FORMAL_RHOS],
        "scenario_count_expected": len(authority["axis"]),
        "scenario_count_actual": len(actual_axis),
        "eligible_disruptions_expected": sum(
            len(v) for v in authority["axis"].values()),
        "eligible_disruptions_actual": len(rng),
        "frontier_rows_expected": sum(
            2 * (hi - lo + 1) for lo, hi in all_grids.values()),
        "frontier_rows_actual": len(front),
        "full_axis_expected": expected_axis,
        "full_axis_actual": actual_axis,
        "full_epsilon_axis_cross_validated_range_oracle_frontier": epsilon_axis,
        "shards": sorted(receipts, key=lambda item: item["rho"]),
        "merged_artifact_sha256": {
            name: sha256_file(path) for name, path in merged_paths.items()},
        "honest_limit": (
            "canonical_write promotes one file at a time; the five-file merged "
            "family is validated but is not family-atomic"),
    }
    receipt_path = os.path.join(out_dir, SHARD_MERGE_RECEIPT)

    def _validate_receipt(temp_path):
        with open(temp_path, "r", encoding="utf-8") as handle:
            landed = json.load(handle)
        require(landed == receipt_object,
                "landed R1 shard-merge receipt differs from validated object")
        require("receipt_self_hash" not in landed
                and SHARD_MERGE_RECEIPT
                not in landed.get("merged_artifact_sha256", {}),
                "R1 shard-merge receipt may not contain its own hash")
        for name, expected_sha in landed["merged_artifact_sha256"].items():
            require(sha256_file(merged_paths[name]) == expected_sha,
                    "merged R1 artifact changed before receipt promotion",
                    artifact=name, expected_sha256=expected_sha,
                    actual_sha256=sha256_file(merged_paths[name]))
        for path, expected_sha in verified:
            require(sha256_file(path) == expected_sha,
                    "source R1 shard changed before receipt promotion",
                    path=path, expected_sha256=expected_sha,
                    actual_sha256=sha256_file(path))

    with canonical_write(
            receipt_path, _validate_receipt, stage=STAGE,
            diagnostics={"artifact": SHARD_MERGE_RECEIPT,
                         "scientific_result": False,
                         "operation": "six-rho validation-only merge"}) as temp_path:
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(receipt_object, handle, indent=2, sort_keys=True)
            handle.write("\n")
    hashes["outputs"][receipt_path] = sha256_file(receipt_path)
    if verbose:
        print("  [merge] six formal R1 rho shards accepted without re-solving; "
              f"scenarios={len(authority['axis'])} "
              f"disruptions={len(rng)} frontier_rows={len(front)} "
              f"SNAP={snap_id}")
    return (rng, front, diagnostics, oracle, receipt_object,
            full_design, hashes, reports, log)


# --------------------------------------------------------------------------- #
def run(manifest_path, out_dir, max_absent=None, rho_filter=None,
        allow_fractional=False, verbose=True, formal=False, snap_id=None,
        cli_time_limit_s=None, canonical_assignment=None):
    assert_no_staging_import()
    assert_no_absolute_runtime_paths(__file__)
    policy = solver_policy(cli_time_limit_s=cli_time_limit_s,
                           require_cli_time_limit=formal)
    runtime_contract = assert_runtime_contract(policy)
    if formal:
        require(max_absent is None,
                "formal R1 must enumerate every eligible disruption; a finite "
                "max_absent is fixture/rehearsal-only",
                max_absent=max_absent, absence_rule=ABSENCE_RULE)
    if canonical_assignment is None:
        canonical_assignment = (
            bool(runtime_contract["formal_canonical_assignment"])
            if formal else False)
    if formal:
        require(
            bool(canonical_assignment)
            is bool(runtime_contract["formal_canonical_assignment"]),
            "formal R1 canonical-assignment mode differs from the "
            "pre-registered runtime contract",
            requested=bool(canonical_assignment),
            registered=bool(
                runtime_contract["formal_canonical_assignment"]))
        snap_id = require_snap_binding(DESIGN_MANIFEST, snap_id, stage=STAGE)
    else:
        require(not snap_id,
                "a rehearsal may not attach or invent a SNAP identifier",
                stage=STAGE, snap_id=snap_id)

    manifest_path = os.path.abspath(manifest_path)
    man_dir = os.path.dirname(manifest_path)
    man = pd.read_csv(manifest_path)
    c_rho = find_col(man.columns, "rho_target")
    c_seed = find_col(man.columns, "seed")
    c_h = find_col(man.columns, "hhcp")
    c_s = find_col(man.columns, "sol")
    formal_meta_cols = (
        "N", "rho_global", "rho_used_mean", "rho_used_max", "src_sha256",
        "feasible", "unplaced",
    )
    if formal:
        for name in formal_meta_cols:
            find_col(man.columns, name)
    if rho_filter is not None:
        man = man[np.isclose(man[c_rho].astype(float), float(rho_filter))]
        require(not man.empty, "rho filter selected no manifest row",
                rho=rho_filter)

    inputs = [manifest_path, DESIGN_MANIFEST, ORACLE_MODULE]
    run_absence_rule = (
        ABSENCE_RULE if max_absent is None
        else "debug_first_{}_active_future_task_holders".format(int(max_absent)))
    axis, grids, fslots, rng_rows, front_rows, diagnostic_rows, oracle_rows = \
        {}, {}, {}, [], [], [], []
    for _, row in man.iterrows():
        hp = os.path.join(man_dir, str(row[c_h]))
        sp = os.path.join(man_dir, str(row[c_s]))
        for p in (hp, sp):
            require(os.path.isfile(p), "manifest points at a missing file", path=p)
            inputs.append(p)
        parser = HHCPFullParser(hp)
        sa = load_static(sp)
        S = set(parser.nurses.keys())
        actives = active_absentees(parser, sa, max_absent)
        require(bool(actives), "no active caregiver in the future horizon", hhcp=hp)
        rho, seed = float(row[c_rho]), int(row[c_seed])
        axis[(rho, seed)] = actives
        declared_n = int(optional_manifest_value(row, man.columns, "N", len(S)))
        incumbent_feasible = manifest_bool(
            optional_manifest_value(row, man.columns, "feasible", True),
            "feasible") and int(optional_manifest_value(
                row, man.columns, "unplaced", 0)) == 0
        source_digest = str(optional_manifest_value(
            row, man.columns, "src_sha256", "fixture-not-recorded"))
        if formal:
            require(declared_n == len(S),
                    "manifest active-caregiver count differs from parsed instance",
                    rho=rho, seed=seed, manifest_N=declared_n,
                    parsed_nurses=len(S))
            require(incumbent_feasible,
                    "formal R1 requires a generator-certified flat-feasible "
                    "incumbent with no unplaced task",
                    rho=rho, seed=seed)
            require(len(source_digest) == 64
                    and all(ch in "0123456789abcdef" for ch in source_digest.lower()),
                    "formal R1 source digest must be lowercase/uppercase SHA256 hex",
                    rho=rho, seed=seed, src_sha256=source_digest)
        scenario_meta = {
            "absence_rule": run_absence_rule,
            "absence_cap": -1 if max_absent is None else int(max_absent),
            "eligible_disruption": True,
            "eligible_disruptions_in_scenario": len(actives),
            "active_caregiver_count": declared_n,
            "rho_target": rho,
            "rho_global": float(optional_manifest_value(
                row, man.columns, "rho_global", -1.0)),
            "rho_used_mean": float(optional_manifest_value(
                row, man.columns, "rho_used_mean", -1.0)),
            "rho_used_max": float(optional_manifest_value(
                row, man.columns, "rho_used_max", -1.0)),
            "src_sha256": source_digest,
            "incumbent_flat_feasible": incumbent_feasible,
        }
        for a in actives:
            label = f"rho={rho} seed={seed} absent={a}"
            rr, fr, diag = disruption_family(
                parser, sa, S, a, policy, label=label,
                canonical_assignment=canonical_assignment)
            require(allow_fractional or rr["w_level_separation"] == "integral_durations",
                    "non-integral effective durations: consecutive achievable "
                    "overtime levels may be closer than LEX_W_ROW_SLACK, so the "
                    "lexicographic W row cannot be trusted. Re-run with "
                    "--allow-fractional-durations only if that is understood.",
                    label=label, lex_w_row_slack=LEX_W_ROW_SLACK)
            placeability = {
                "placeable_A1": int(rr["U_min_A1"]) == 0,
                "placeable_A2": int(rr["U_min_A2"]) == 0,
            }
            rr.update(rho=rho, seed=seed, snap_id=snap_id or "",
                      **scenario_meta, **placeability)
            rng_rows.append(rr)
            grids[(rho, seed, a)] = (rr["eps_grid_lo"], rr["eps_grid_hi"])
            fslots[(rho, seed, a)] = future_slots(sa)
            for r in fr:
                r.update(rho=rho, seed=seed, snap_id=snap_id or "",
                         **scenario_meta, **placeability)
                front_rows.append(r)
            if formal:
                for o in compare_independent_oracle(
                        hp, sp, a, rr, fr, policy, rho=rho, seed=seed):
                    o.update(rho=rho, seed=seed, snap_id=snap_id)
                    oracle_rows.append(o)
            for d in diag["solver_diagnostics"]:
                d.update(rho=rho, seed=seed, snap_id=snap_id or "")
                diagnostic_rows.append(d)
        if verbose:
            print(f"  rho={rho:.2f} s{seed}: {len(actives)} disruption(s) solved",
                  flush=True)

    lead = ["rho", "seed", "absent"]
    rng_df = pd.DataFrame(rng_rows)
    rng_df = rng_df[lead + [c for c in rng_df.columns if c not in lead]]
    front_df = pd.DataFrame(front_rows)
    front_df = front_df[lead + [c for c in front_df.columns if c not in lead]]
    diagnostics_df = pd.DataFrame(diagnostic_rows)
    oracle_validation_df = (
        pd.DataFrame(oracle_rows) if formal else None)

    design = build_design(axis, grids, inputs)
    design["future_slots"] = fslots
    design["snap_id"] = snap_id or ""
    design["absence_rule"] = run_absence_rule
    hashes, reports, log = write_validated(
        out_dir, rng_df, front_df, diagnostics_df, oracle_validation_df,
        design, policy,
        snap_id=snap_id, formal=formal, verbose=verbose)
    if verbose:
        if formal:
            print(f"  [hashes] SNAP identifier: {snap_id}")
        else:
            print("  [hashes] SNAP identifier: none (rehearsal only; SNAP closes "
                  "before R0 and formal R1)")
    return rng_df, front_df, design, hashes, reports, log


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--manifest", default=os.path.join(HERE, "rho_manifest.csv"))
    ap.add_argument("--out-dir", default=None,
                    help="where eps_range.csv / eps_frontier.csv are written")
    ap.add_argument("--rho", type=float, default=None,
                    help="restrict to one rho_target (default: the whole family, "
                         "which REQ-5's all-disruptions join requires)")
    ap.add_argument(
        "--shard-dir", action="append", default=[], metavar="DIR",
        help="formal merge mode: repeat exactly six times, once for each "
             "already-landed mutually exclusive rho shard; validates and rebuilds "
             "the full R1 family without solving")
    ap.add_argument(
        "--max-absent", type=int, default=None,
        help="fixture/rehearsal cap only; formal R1 enumerates all active "
             "future-task holders and rejects this option")
    ap.add_argument("--allow-fractional-durations", action="store_true")
    ap.add_argument("--snap-id", default=None,
                    help="immutable identifier produced by the SNAP gate; "
                         "required for a formal R1 promotion")
    ap.add_argument("--time-limit-s", type=float, default=None,
                    help="formal R1 must pass the pre-registered value exactly")
    cg = ap.add_mutually_exclusive_group()
    cg.add_argument("--canonical-assignment", dest="canonical_assignment",
                    action="store_true",
                    help="optional rehearsal-only assignment tie-break; formal R1 "
                         "rejects it because assignment identity is not an estimand")
    cg.add_argument("--no-canonical-assignment", dest="canonical_assignment",
                    action="store_false",
                    help="skip expensive assignment canonicalization; hashes are "
                         "then empty and not reproducibility evidence")
    ap.set_defaults(canonical_assignment=None)
    ap.add_argument("--promote-canonical", action="store_true",
                    help="required to write into experiments/journal_p0/; without "
                         "it a canonical target is refused")
    ap.add_argument("--formal", action="store_true",
                    help="formal R1 semantics for any output directory; requires "
                         "--snap-id and --time-limit-s; assignment canonicalization "
                         "is pre-registered OFF and assignment hashes stay empty")
    a = ap.parse_args(argv)
    out_dir = os.path.abspath(a.out_dir) if a.out_dir else HERE
    if os.path.abspath(out_dir) == HERE and not a.promote_canonical:
        fatal("refusing to write the canonical eps pair without "
              "--promote-canonical (canonical regeneration is stage R1, and R1 "
              "has not been authorised)", out_dir=out_dir)
    if a.shard_dir:
        require(a.formal or a.promote_canonical,
                "--shard-dir activates a formal-only R1 merge; pass --formal")
        solve_only = []
        if a.rho is not None:
            solve_only.append("--rho")
        if a.max_absent is not None:
            solve_only.append("--max-absent")
        if a.allow_fractional_durations:
            solve_only.append("--allow-fractional-durations")
        if a.canonical_assignment is not None:
            solve_only.append("--canonical-assignment/--no-canonical-assignment")
        require(not solve_only,
                "formal R1 shard merge refuses solve-only arguments",
                offending=solve_only)
        merge_formal_shards(
            a.manifest, out_dir, a.shard_dir,
            snap_id=a.snap_id, cli_time_limit_s=a.time_limit_s)
    else:
        run(a.manifest, out_dir, max_absent=a.max_absent, rho_filter=a.rho,
            allow_fractional=a.allow_fractional_durations,
            formal=(a.formal or a.promote_canonical), snap_id=a.snap_id,
            cli_time_limit_s=a.time_limit_s,
            canonical_assignment=a.canonical_assignment)
    return 0


if __name__ == "__main__":
    sys.exit(main())
