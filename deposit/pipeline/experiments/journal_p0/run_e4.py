"""
run_e4.py — the weighted-sum (w4-sweep) stage driver.

S2 REWRITE (round: S2 CODE + DESIGN FIXES ONLY).  This file is *code*.  It wrote
NO canonical artifact in this round: `e4_raw.csv` / `e4_summary.csv` were neither
executed nor patched, and this module now REFUSES to write into
`experiments/journal_p0/` without `--promote-canonical`.  Regenerating
`e4_raw.csv` from scratch is stage **R1b**, and R1b has NOT been authorised or run.

REQ-6 of `audit_outputs/rerun_plan_v2_1.md`, as implemented here
--------------------------------------------------------------
1. COMMON SERVICE, THEN A DETERMINISTIC WEIGHTED-SUM TIE-BREAK.
   For every disruption, two independent horizon MILPs obtain `U_min_A1` and
   `U_min_A2`.  The executable containment guard verifies A1 is a subset of A2,
   then fixes `U_common := U_min_A1` in BOTH regimes for every w4 point:

       preliminary   min U in A1 and A2                    -> U_min_A1/U_min_A2
       lex stage 2   min Z = W2*sum(O) + w4*D + W5*U
                     s.t. U = U_common                     -> Z*
       lex stage 3   min W s.t. Z <= Z*, U = U_common      -> W*
       lex stage 4   min D s.t. U = U_common, W <= W*      -> D*
       lex stage 5   optional canonical assignment tie-break

   `W5*U` is constant after the equality is imposed; it is retained in Z only
   so the operational objective remains exactly reconstructible.  It is not
   relied upon to enforce service.  Assignment identity is not an estimand, so
   stage 5 is OFF by default and assignment hashes are empty.

2. NO PATCHING OF THE OLD RAW FILE.  No pre-existing `e4_raw.csv` is consumed
   and there is no in-place edit path.  The raw file is produced whole or not at
   all; only the just-landed bytes are re-read to derive the summary and its
   SHA256-bound provenance.

3. NO CROSS-STAGE TRANSCRIPTION.  Every tabular read goes through
   `_read_csv_audited()`, which is FATAL on any path whose name looks like
   another stage's artifact (`eps_range`, `eps_frontier`, `stability_constrained`,
   `rolling_raw`, `scalability_*`, `external_*`).  The weighted-sum stage may not
   import a number from the epsilon stage, and vice versa.

4. PROVENANCE + NON-DOMINATION.  `e4_summary.csv` is rebuilt by script from the
   raw file only, with a long-format sidecar
   `e4_summary_provenance.csv` giving, per summary cell,
   `(source_file, source_row_key, source_column, agg)`.
   `verify_provenance()` recomputes every cited cell from the cited rows.  The
   validator additionally asserts that no w4 point is dominated by another w4
   point within a (rho, seed, absent, method) group — the structural symptom of an
   unbroken tie.  Both have demonstrated negative controls under
   `tests/fixtures/epse4/`.

5. FAIL FAST (REQ-2).  The old `if r is None: continue` at line 61-62, which left
   a failed disruption with NO ROW AT ALL, is gone.  Every guard raises s2_core's `PipelineFailure`.

6. FULL PRECISION.  Overtime is recomputed from the integral assignment at full
   precision and is never rounded before comparison or validation.  (The
   published file rounded to one decimal; that rounding is not re-introduced.)

7. HORIZON-ONLY FORMAL SOLVES.  `U = U_common` is a horizon-total equality and
   couples slots.  Formal R1b therefore refuses per-slot mode.  The legacy
   `per-slot` spelling is accepted only for non-formal fixture compatibility and
   is routed to the same horizon solve; it can never enter a formal artifact.

Usage (S2 / fixture form; writes nowhere near a canonical path):
    python journal_p0/run_e4.py --manifest ../../tests/fixtures/epse4/manifest_epse4_A.csv \
                                --out-dir ../../tests/fixtures/epse4/out_e4_A
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
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
from displacement_milp import (T_NOW, TRAVEL_PER_TASK, W2, W5,  # noqa: E402
                               load_static)

STAGE = "R1b"
W4_SWEEP = [0, 5, 10, 20, 50, 100]
FORMAL_RHOS = (0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
DESIGN_MANIFEST = os.path.join(HERE, "run_e4_design_manifest.json")
WEIGHTED_OBJECTIVE_FORMULA = "W2 * sum(O) + w4 * D + W5 * U"
LEX_STAGE_ORDER = (
    "min U independently in A1 and A2",
    "min Z at U_common in each regime", "min W", "min D")
EXECUTABLE_MAIN_STAGE_NAMES = (
    "lex2_minZ_at_Ucommon", "lex3_minW", "lex4_minD")
LEX_ORDER_LABEL = "Umin(A1,A2)>Z@Ucommon>W>D"
DECOMPOSITION_MODES = ("per-slot", "horizon")
DEFAULT_DECOMPOSITION = "horizon"
FORMAL_DECOMPOSITION = "horizon"
CANONICAL_ASSIGNMENT_FORMAL_DEFAULT = False
ABSENCE_RULE = "all_active_future_task_holders"
COUNT_SEMANTICS = {
    "active_caregiver_count": "manifest_or_parsed_scenario_N_all_caregivers",
    "eligible_disruptions_in_scenario": "len_active_future_task_holders",
}
CAP_ALLOWANCE_RULE = (
    "math.ulp(cap) * 8 (s2_core.cap_fp_slack); a fixed 1e-6 is NOT an "
    "admissible final pass allowance")
PRIMARY_WEIGHTED_ROLE = "lex2_primary_weighted_Z_at_U_common"
W_REFINE_ROLE = "lex3_minW_at_Zstar_U_common"
FINAL_TIE_BREAK_ROLE = "lex4_final_D_tie_break_not_weighted_certificate"
UMIN_A1_ROLE = "preliminary_independent_min_U_A1"
UMIN_A2_ROLE = "preliminary_independent_min_U_A2"
SOLVER_TIME_LIMIT_S = 1800.0
LATTICE_STRUCTURAL_ULPS = 8
# Historical recovery-05/v3 premise.  It remains named for forensic replay,
# but is not an authority for a formal recovery-06 status-0 certificate.
LATTICE_BOUND_ULPS = 128
LEGACY_LATTICE_METHOD = "objective_lattice_ceiling_v3"
STATUS0_LATTICE_BOUND_ULPS = 0
STATUS0_LATTICE_METHOD = "solver_status0_lattice_consistency_v1"
LATTICE_METHOD = STATUS0_LATTICE_METHOD
STATUS0_LOWER_BOUND_SOURCE = (
    "solver_status0_unique_lattice_cell_no_exact_bound")
STATUS0_JUSTIFICATION = (
    "solver_status0_exact_readback_unique_lattice_cell_not_exact_proof")
STATUS0_CLAIM_WORDING = (
    "Under the registered status-0 Optimal solver contract and "
    "exact-feasible lattice readback, the returned incumbent, live vector "
    "reconstruction, and persisted primal/dual diagnostics are mutually "
    "consistent with the same unique attainable objective level.")
STATUS0_CLAIM_SCOPE = (
    "conditional solver-status and unique-lattice-cell consistency; not an "
    "independent lower-bound proof and not a proof that the floating-point "
    "solver is correct")
STRUCTURAL_LOWER_BOUND_METHOD = "attained_structural_lower_bound_v1"
DEVIATION_CONSTANT_AUTHORITY = (
    "For every re-parsed (rho,seed,absent) H/S instance, "
    "dev_const=len(xbar) is recomputed before validation; U/W roles require "
    "C=0, D requires C=dev_const, weighted Z requires C=w4*dev_const, and P "
    "must equal exact_full_objective-C.")
EXACT_CERTIFICATE_METHODS = (
    LATTICE_METHOD, STRUCTURAL_LOWER_BOUND_METHOD)
RAW_BOUND_METHOD = "raw_bound_equality_fallback_v1"
SOLVER_PRIMAL_SAME_LEVEL_RULE = (
    "directed-up abs(F-P), abs(B-P), and abs(F-B) must each be strictly "
    "smaller than half the declared lattice step; this validates a code-0 "
    "solver-reported optimum but is not an exact mathematical proof")
SOLVER_VECTOR_SAME_LEVEL_RULE = (
    "before adding C, directed-up abs(X-P) and abs(F-X), where "
    "X=c@res.x is reconstructed live, must each be strictly smaller than "
    "half the declared lattice step; the certificate independently checks "
    "F against P")
SOLVER_VECTOR_METHOD = "live_objective_same_lattice_cell_v1"
SOLVER_VECTOR_X_DEFINITION = "raw binary64 c @ res.x before adding C"
SOLVER_VECTOR_NON_LATTICE_RULE = "eight_ulp_F_X_identity"
SOLVER_VECTOR_PROOF_ROLE = (
    "live fail-closed objective-level consistency gate")
SOLVER_VECTOR_SCOPE_NOTE = (
    "X and res.x are not persisted. The final verifier accepts this only as "
    "a SNAP-pinned executable live gate, not as an independently replayable "
    "post-hoc result field.")
SOLVER_DUAL_SAME_LEVEL_RULE = (
    "round-trip-replayed directed-up abs(B-P) and abs(F-B) must each be "
    "strictly smaller than half the declared lattice step")
OBJECTIVE_GAP_CONSISTENCY_RULE = (
    "raw solver gap must replay exactly from immutable F/B and "
    "full-objective gap or its zero-denominator reason must replay exactly "
    "from immutable P/B/C")
STATUS0_REQUIRED_CONDITIONS = (
    "solver_status_code_is_exactly_0",
    "solver_status_name_is_exactly_Optimal",
    "proven_optimal_is_true",
    "timeout_incumbent_is_forbidden",
    "canonical_readback_is_exactly_feasible_for_all_active_rows",
    "P_full_is_an_exact_integer_on_the_registered_objective_lattice",
    "live_directed_up_abs_X_minus_P_is_strictly_less_than_half_step",
    "live_directed_up_abs_F_minus_X_is_strictly_less_than_half_step",
    "persisted_roundtrip_directed_up_abs_F_minus_P_is_strictly_less_than_half_step",
    "persisted_roundtrip_directed_up_abs_B_minus_P_is_strictly_less_than_half_step",
    "persisted_roundtrip_directed_up_abs_F_minus_B_is_strictly_less_than_half_step",
    "raw_solver_gap_is_finite_and_exactly_consistent_with_immutable_F_and_B",
    "full_objective_gap_or_zero_denominator_reason_is_exactly_consistent_with_immutable_P_B_C",
    "raw_P_B_C_F_and_both_gap_fields_are_immutable",
)
STATUS0_LEGACY_SAFE_SEMANTICS = (
    "lattice_safe_dual_bound_full_objective stores raw B_full=B+C as a "
    "diagnostic only; despite the field name it is not a safe bound or proof "
    "authority")
STATUS0_LEGACY_CLOSED_SEMANTICS = (
    "lattice_closed_dual_bound_full_objective stores the exact P lattice "
    "center P_full=P+C; no dual closure is performed")
STATUS0_LEGACY_PRIMAL_SEMANTICS = (
    "lattice_primal_level_full_objective stores the same exact P lattice "
    "center P_full=P+C")
OVERTIME_SLACK_UPPER_BOUND_RULE = "H_MAX-H_STD"
OVERTIME_SLACK_UPPER_BOUND = float(H_MAX) - float(H_STD)
RETRY_POLICY = (
    "none -- a failed solve aborts and is never retried with changed options")

EXPECTED_RUNTIME_CONTRACT = {
    "service_level_rule": "U_common=U_min_A1",
    "service_constraint": "sum_u_eq_U_common_in_both_regimes",
    "objective_order": [
        "independent_min_U_A1_and_A2",
        "min_Z_given_U_common",
        "min_W_given_Z_and_U_common",
        "min_D_given_U_common_and_W",
    ],
    "weighted_objective": WEIGHTED_OBJECTIVE_FORMULA,
    "w4_sweep": list(W4_SWEEP),
    "formal_rhos": list(FORMAL_RHOS),
    "formal_decomposition": FORMAL_DECOMPOSITION,
    "formal_all_active_absentees": True,
    "absence_rule": ABSENCE_RULE,
    "formal_max_absent": None,
    "formal_canonical_assignment": False,
    "solver_presolve": False,
    "solver_time_limit_s": SOLVER_TIME_LIMIT_S,
    "formal_fractional_durations_permitted": False,
    "exact_objective_certificate_method": LATTICE_METHOD,
    "exact_objective_certificate_methods": list(EXACT_CERTIFICATE_METHODS),
    "status0_lattice_bound_ulps": STATUS0_LATTICE_BOUND_ULPS,
    "legacy_objective_lattice_v3_bound_ulps": LATTICE_BOUND_ULPS,
    "objective_lattice_ceiling_v3_formal_authority": False,
    "objective_lattice_structural_ulps": LATTICE_STRUCTURAL_ULPS,
    "objective_solver_primal_same_level_rule": (
        SOLVER_PRIMAL_SAME_LEVEL_RULE),
    "objective_solver_vector_same_level_rule": (
        SOLVER_VECTOR_SAME_LEVEL_RULE),
    "objective_solver_dual_same_level_rule": (
        SOLVER_DUAL_SAME_LEVEL_RULE),
    "objective_gap_consistency_rule": OBJECTIVE_GAP_CONSISTENCY_RULE,
    "standard_capacity_H_STD": float(H_STD),
    "formal_standard_capacity_integral": True,
    "overtime_slack_upper_bound_rule": OVERTIME_SLACK_UPPER_BOUND_RULE,
    "overtime_slack_upper_bound": OVERTIME_SLACK_UPPER_BOUND,
    "disruption_iid_confidence_interval": False,
}

# --------------------------------------------------------------------------- #
# shared library.
# `experiments/journal_p0/s2_core.py` is owned by a different S2 agent.  It was
# ABSENT when this rewrite was started, so the first draft carried local
# fallbacks behind the same names; s2_core landed during the round and this file
# now imports it directly instead of keeping a second, never-exercised copy of
# guards it already owns.
# --------------------------------------------------------------------------- #
from s2_core import (COL, DesignGrid, PipelineFailure, SolverResult,  # noqa: E402
                     StageOutputSpec, StageSolverPolicy,
                     assert_no_absolute_runtime_paths, assert_no_staging_import,
                     assignment_hash, canonical_write, cap_fp_slack, fatal,
                     full_objective_gap, hash_manifest, require, require_finite,
                     require_nonempty_candidates, require_snap_binding,
                     sha256_file, slot_allowed, validate_stage_output)

# --------------------------------------------------------------------------- #
# declared numeric policy (mirrors run_e4_design_manifest.json)
# --------------------------------------------------------------------------- #
INT_TOL = 1e-6      # |x - round(x)| on a variable declared integral
LEX_TOL = 1e-6      # slack on the internal lexicographic rows (Z, W)


def _ulp_margin(*values):
    """Small directed-rounding allowance used only at lattice boundaries.

    This is deliberately unrelated to ``INT_TOL``.  The latter guards integer
    variable readback; using it to move a dual bound between objective levels
    would weaken the optimality certificate.
    """
    finite = [abs(float(v)) for v in values if v is not None
              and math.isfinite(float(v))]
    scale = max(finite + [1.0])
    return float(LATTICE_STRUCTURAL_ULPS) * math.ulp(scale)


def _bound_ulp_margin(*values):
    """Registered numerical envelope subtracted from a code-0 dual bound."""
    finite = [abs(float(v)) for v in values if v is not None
              and math.isfinite(float(v))]
    scale = max(finite + [1.0])
    return float(LATTICE_BOUND_ULPS) * math.ulp(scale)


def assert_live_objective_reconstruction(
        objective_stage, solver_fun_raw, objective_from_x, exact_primal, *,
        w4=None, integral_durations=False, where):
    """Bind the live solver vector to the exact assignment objective.

    For a declared objective lattice, binary64 summation order is not required
    to reproduce the solver's objective bit-for-bit.  Instead, the solver value
    ``F``, the live reconstruction ``X=c@res.x``, and the exact assignment
    readback ``P`` must identify the same unique attainable lattice cell.  The
    checks are performed on the raw linearized values before adding ``C`` so a
    large constant cannot hide disagreement.  The later objective certificate
    still performs its own (and, for lattice-v3, stricter) F/P check.

    Non-lattice historical fixtures have no discrete cell to appeal to, so they
    retain the original eight-ULP F/X identity check.  Formal R1b forbids that
    branch by requiring integral effective durations and capacity.
    """
    f = require_finite(solver_fun_raw, "solver-reported objective fun F",
                       where=where)
    x_objective = require_finite(
        objective_from_x, "live objective reconstruction X=c@res.x",
        where=where)
    p = require_finite(exact_primal, "exact assignment readback P", where=where)
    integral_durations = _cell_bool(
        integral_durations, field="integral_durations",
        where="{}:live-objective-reconstruction".format(where))
    spec = _objective_lattice_spec(
        objective_stage, w4, integral_durations, where=where)
    if spec is None:
        margin = _ulp_margin(f, x_objective)
        require(abs(f - x_objective) <= margin,
                "non-lattice solver objective differs from c @ res.x",
                where=where, objective_stage=objective_stage, w4=w4,
                F=f, X=x_objective, P=p, ulp_identity_margin=margin)
        return {
            "method": "non_lattice_eight_ulp_identity_v1",
            "F": f, "X": x_objective, "P": p,
            "lattice_step": None, "half_lattice_step": None,
            "X_P_delta": math.nextafter(abs(x_objective - p), math.inf),
            "F_X_delta": math.nextafter(abs(f - x_objective), math.inf),
        }

    step, offset, _ = spec
    step = require_finite(step, "objective lattice step", where=where)
    offset = require_finite(offset, "objective lattice offset", where=where)
    require(step > 0.0, "objective lattice step must be positive",
            where=where, objective_stage=objective_stage, w4=w4,
            lattice_step=step, lattice_offset=offset)
    half_step = step / 2.0
    x_p_delta = math.nextafter(abs(x_objective - p), math.inf)
    f_x_delta = math.nextafter(abs(f - x_objective), math.inf)
    require(x_p_delta < half_step,
            "live c @ res.x and exact readback do not identify the same "
            "objective-lattice level",
            where=where, objective_stage=objective_stage, w4=w4,
            F=f, X=x_objective, P=p, X_P_delta=x_p_delta,
            F_X_delta=f_x_delta, half_lattice_step=half_step,
            lattice_step=step, lattice_offset=offset,
            solver_vector_same_level_rule=SOLVER_VECTOR_SAME_LEVEL_RULE)
    require(f_x_delta < half_step,
            "solver fun and live c @ res.x do not identify the same "
            "objective-lattice level",
            where=where, objective_stage=objective_stage, w4=w4,
            F=f, X=x_objective, P=p, X_P_delta=x_p_delta,
            F_X_delta=f_x_delta, half_lattice_step=half_step,
            lattice_step=step, lattice_offset=offset,
            solver_vector_same_level_rule=SOLVER_VECTOR_SAME_LEVEL_RULE)
    return {
        "method": SOLVER_VECTOR_METHOD,
        "F": f, "X": x_objective, "P": p,
        "lattice_step": step, "half_lattice_step": half_step,
        "X_P_delta": x_p_delta, "F_X_delta": f_x_delta,
    }


def _integer_coefficient(value, name, *, where):
    numeric = require_finite(value, name, where=where)
    nearest = int(round(numeric))
    require(abs(numeric - nearest) <= _ulp_margin(numeric, nearest),
            "objective-lattice coefficient is not integral",
            where=where, coefficient=name, value=numeric,
            nearest_integer=nearest,
            ulp_boundary_margin=_ulp_margin(numeric, nearest))
    return abs(nearest)


def _objective_lattice_spec(objective_stage, w4, integral_durations, *, where):
    """Return ``(step, offset, justification)`` for attainable optima.

    U and D are sums/affine sums of binary variables.  With integral effective
    durations and integral H_STD, each assignment has an integral canonical
    overtime value; because W has a positive coefficient whenever it is
    minimised, an optimum W is integral.  Consequently formal weighted Z has
    step gcd(W2, w4, W5)=5 for the registered sweep.
    """
    require(objective_stage in ("U", "Z", "W", "D"),
            "unknown objective stage for lattice certificate",
            where=where, objective_stage=objective_stage)
    if objective_stage in ("U", "D"):
        return 1.0, 0.0, "binary_affine_integer_objective"
    if not integral_durations:
        return None
    if objective_stage == "W":
        return 1.0, 0.0, "integral_effective_duration_canonical_overtime"
    coeffs = [
        _integer_coefficient(W2, "W2", where=where),
        _integer_coefficient(w4, "w4", where=where),
        _integer_coefficient(W5, "W5", where=where),
    ]
    step = 0
    for coefficient in coeffs:
        step = math.gcd(step, coefficient)
    require(step > 0, "weighted objective lattice has zero step",
            where=where, coefficients=coeffs)
    return (float(step), 0.0,
            "gcd_W2_w4_W5_with_integral_canonical_overtime")


def _strict_integer_value(value, name, *, where):
    """Return an integer only when the supplied numeric value is exact."""
    numeric = require_finite(value, name, where=where)
    nearest = int(round(numeric))
    require(numeric == float(nearest),
            "structural certificate input is not an exact integer",
            where=where, field=name, value=numeric, nearest=nearest)
    return nearest


def _structural_certificate_inputs(
        objective_stage, solution, *, w4=None, u_eq=None, where):
    """Rebuild an integer objective and its analytic lower bound.

    This authority is the exact assignment readback, not ``P+C`` and not a
    floating solver field.  Formal durations, capacities, and weights are
    integral, so canonical overtime is integral as well.
    """
    exact_u = _strict_integer_value(solution["U"], "exact U", where=where)
    exact_w = _strict_integer_value(solution["W"], "exact W", where=where)
    exact_d = _strict_integer_value(solution["D"], "exact D", where=where)
    require(exact_u >= 0 and exact_w >= 0 and exact_d >= 0,
            "structural certificate components must be non-negative",
            where=where, U=exact_u, W=exact_w, D=exact_d)

    if objective_stage == "U":
        return exact_u, 0, "nonnegative_unserved_count"
    if objective_stage == "W":
        return exact_w, 0, "nonnegative_canonical_overtime"
    if objective_stage == "D":
        return exact_d, 0, "nonnegative_assignment_deviation"
    require(objective_stage == "Z",
            "unknown stage for structural lower-bound certificate",
            where=where, objective_stage=objective_stage)
    w2_int = _strict_integer_value(W2, "W2", where=where)
    w4_int = _strict_integer_value(w4, "w4", where=where)
    w5_int = _strict_integer_value(W5, "W5", where=where)
    require(w2_int >= 0 and w4_int >= 0 and w5_int >= 0,
            "weighted structural lower bound requires non-negative weights",
            where=where, W2=w2_int, w4=w4_int, W5=w5_int)
    exact_z = w2_int * exact_w + w4_int * exact_d + w5_int * exact_u
    if u_eq is None:
        return exact_z, 0, "nonnegative_weighted_objective"
    fixed_u = _strict_integer_value(u_eq, "fixed U_common", where=where)
    require(fixed_u >= 0 and exact_u == fixed_u,
            "weighted structural lower bound disagrees with fixed U_common",
            where=where, exact_U=exact_u, fixed_U_common=fixed_u)
    return (exact_z, w5_int * fixed_u,
            "fixed_U_common_plus_nonnegative_overtime_and_deviation")


def exact_objective_certificate(objective_stage, primal, dual_bound,
                                deviation_constant_C, solver_fun_raw, *, w4=None,
                                integral_durations=False,
                                exact_full_objective_integer=None,
                                structural_lower_bound_integer=None,
                                structural_lower_bound_source=None, where):
    """Certify a code-0 objective without rewriting the raw solver evidence.

    For minimisation, ``dual_bound`` is the code-0 solver-reported lower-bound
    estimate,
    ``solver_fun_raw`` is the solver's evaluation of its incumbent, and
    ``primal`` is the exact assignment readback.  HiGHS can report a tiny
    ``dual_bound > primal`` roundoff at status 0.  Recovery-06 does not enlarge
    the former data-derived 128-ULP premise after another observed
    counterexample.  If an independently registered analytic lower bound is
    attained, this function returns an exact structural proof.  Otherwise it
    validates only that the code-0 solver estimate, solver objective, and exact
    feasible readback identify one unique lattice cell.  That method is not an
    exact mathematical optimality proof.

    Raw P/B/C/F evidence is never rewritten.  An exact-bound fallback is
    retained only for non-formal fractional-duration fixtures, for which no
    W/Z lattice is asserted.
    """
    p = require_finite(primal, "primal P", where=where)
    b = require_finite(dual_bound, "dual bound B", where=where)
    c = require_finite(deviation_constant_C, "deviation constant C", where=where)
    f = require_finite(
        solver_fun_raw, "solver-reported objective fun F", where=where)
    p_full = p + c
    b_full = b + c
    f_full = f + c
    integral_durations = _cell_bool(
        integral_durations, field="integral_durations",
        where="{}:objective-lattice-premise".format(where))
    spec = _objective_lattice_spec(
        objective_stage, w4, integral_durations, where=where)
    if spec is None:
        require(abs(p_full - b_full) <= INT_TOL,
                "non-integral-duration objective lacks both a lattice proof and "
                "an exact raw dual bound",
                where=where, objective_stage=objective_stage,
                P=p, B=b, C=c, P_full=p_full, B_full=b_full,
                absolute_tolerance=INT_TOL)
        return {
            "method": RAW_BOUND_METHOD,
            "objective_stage": objective_stage,
            "step": 0.0,
            "offset": 0.0,
            "bound_ulps": LATTICE_BOUND_ULPS,
            "boundary_margin": 0.0,
            "safe_dual_bound_full_objective": b_full,
            "closed_dual_bound_full_objective": b_full,
            "primal_level_full_objective": p_full,
            "lower_bound_source": "raw_bound_equality",
            "structural_lower_bound_full_objective": None,
            "exact_full_objective_integer": None,
            "proves_exact_optimality": True,
            "justification": "raw_full_objective_bound_equals_incumbent",
        }

    step, offset, justification = spec
    step = require_finite(step, "objective lattice step", where=where)
    offset = require_finite(offset, "objective lattice offset", where=where)
    require(step > 0.0, "objective lattice step must be positive",
            where=where, objective_stage=objective_stage,
            lattice_step=step, lattice_offset=offset)
    primal_margin = _ulp_margin(p_full, step, offset)
    primal_index = int(round((p_full - offset) / step))
    primal_level = offset + primal_index * step
    require(abs(p_full - primal_level) <= primal_margin,
            "incumbent is not on the declared objective lattice",
            where=where, objective_stage=objective_stage,
            P=p, C=c, P_full=p_full, lattice_step=step,
            lattice_offset=offset, nearest_level=primal_level,
            ulp_boundary_margin=primal_margin)

    structural_values = (
        exact_full_objective_integer,
        structural_lower_bound_integer,
        structural_lower_bound_source,
    )
    supplied = [value is not None for value in structural_values]
    require(all(supplied) or not any(supplied),
            "structural certificate inputs must be supplied together",
            where=where, objective_stage=objective_stage,
            supplied=supplied)
    exact_full_int = lower_bound_int = None
    lower_bound_source = ""
    if all(supplied):
        exact_full_int = _strict_integer_value(
            exact_full_objective_integer,
            "exact full objective integer", where=where)
        lower_bound_int = _strict_integer_value(
            structural_lower_bound_integer,
            "structural lower-bound integer", where=where)
        lower_bound_source = str(structural_lower_bound_source)
        expected_zero_sources = {
            "U": "nonnegative_unserved_count",
            "W": "nonnegative_canonical_overtime",
            "D": "nonnegative_assignment_deviation",
        }
        if objective_stage in expected_zero_sources:
            require(lower_bound_int == 0
                    and lower_bound_source
                    == expected_zero_sources[objective_stage],
                    "structural lower bound does not match its objective stage",
                    where=where, objective_stage=objective_stage,
                    lower_bound=lower_bound_int,
                    lower_bound_source=lower_bound_source,
                    expected_lower_bound=0,
                    expected_source=expected_zero_sources[objective_stage])
        else:
            require(objective_stage == "Z",
                    "unknown structural-certificate objective stage",
                    where=where, objective_stage=objective_stage)
            w4_int = _strict_integer_value(w4, "w4", where=where)
            w5_int = _strict_integer_value(W5, "W5", where=where)
            require(w4_int >= 0 and w5_int > 0,
                    "weighted structural lower bound requires non-negative "
                    "w4 and positive W5",
                    where=where, w4=w4_int, W5=w5_int)
            if lower_bound_source == "nonnegative_weighted_objective":
                require(lower_bound_int == 0,
                        "nonnegative weighted-objective lower bound must be zero",
                        where=where, lower_bound=lower_bound_int,
                        lower_bound_source=lower_bound_source)
            else:
                require(lower_bound_source
                        == "fixed_U_common_plus_nonnegative_overtime_and_deviation"
                        and lower_bound_int >= 0
                        and lower_bound_int % w5_int == 0,
                        "fixed-U weighted lower bound is not a registered W5 "
                        "multiple",
                        where=where, lower_bound=lower_bound_int,
                        lower_bound_source=lower_bound_source, W5=w5_int)
        require(exact_full_int >= lower_bound_int,
                "exact feasible objective is below its structural lower bound",
                where=where, objective_stage=objective_stage,
                exact_full_objective_integer=exact_full_int,
                structural_lower_bound_integer=lower_bound_int)
        require(abs(p_full - float(exact_full_int)) <= primal_margin,
                "raw P+C disagrees with the exact integer objective readback",
                where=where, P=p, C=c, P_full=p_full,
                exact_full_objective_integer=exact_full_int,
                ulp_boundary_margin=primal_margin)

    # If an exact feasible assignment attains a registered analytic lower
    # bound, optimality follows directly: L <= OPT <= P_exact = L.  The raw
    # solver dual remains immutable diagnostic evidence and is not used as the
    # proof.  Half-cell checks catch gross F/B corruption without introducing
    # another tunable ULP envelope.
    if (exact_full_int is not None
            and exact_full_int == lower_bound_int):
        solver_primal_delta = math.nextafter(abs(f - p), math.inf)
        dual_primal_delta = math.nextafter(abs(b - p), math.inf)
        require(solver_primal_delta < step / 2.0,
                "solver objective is outside the exact readback's lattice cell",
                where=where, objective_stage=objective_stage,
                P=p, B=b, C=c, F=f,
                solver_primal_delta=solver_primal_delta,
                half_lattice_step=step / 2.0, lattice_step=step)
        require(dual_primal_delta < step / 2.0,
                "solver dual diagnostic is outside the exact readback's lattice cell",
                where=where, objective_stage=objective_stage,
                P=p, B=b, C=c, F=f,
                dual_primal_delta=dual_primal_delta,
                half_lattice_step=step / 2.0, lattice_step=step)
        return {
            "method": STRUCTURAL_LOWER_BOUND_METHOD,
            "objective_stage": objective_stage,
            "step": step,
            "offset": offset,
            "bound_ulps": LATTICE_BOUND_ULPS,
            "boundary_margin": 0.0,
            "safe_dual_bound_full_objective": float(lower_bound_int),
            "closed_dual_bound_full_objective": float(lower_bound_int),
            "primal_level_full_objective": float(exact_full_int),
            "lower_bound_source": lower_bound_source,
            "structural_lower_bound_full_objective": float(lower_bound_int),
            "exact_full_objective_integer": float(exact_full_int),
            "proves_exact_optimality": True,
            "justification": "exact_feasible_objective_attains_structural_lower_bound",
        }

    # A non-attained analytic floor cannot turn the solver estimate into an
    # exact lower-bound proof.  Bind all three raw linearized scalars to the
    # same unique lattice cell instead.  These are corruption/consistency gates
    # for a status-0 solver result, not a replacement dual-error theorem.
    boundary_margin = 0.0
    half_step = step / 2.0
    solver_primal_delta = math.nextafter(abs(f - p), math.inf)
    dual_primal_delta = math.nextafter(abs(b - p), math.inf)
    solver_dual_delta = math.nextafter(abs(f - b), math.inf)
    require(solver_primal_delta < half_step,
            "solver objective and exact feasible readback do not identify the "
            "same objective-lattice level",
            where=where, objective_stage=objective_stage,
            P=p, B=b, C=c, F=f, P_full=p_full, B_full=b_full,
            F_full=f_full, solver_primal_delta=solver_primal_delta,
            solver_primal_same_level_rule=SOLVER_PRIMAL_SAME_LEVEL_RULE,
            half_lattice_step=half_step,
            lattice_step=step)
    require(dual_primal_delta < half_step,
            "solver dual estimate and exact feasible readback do not identify "
            "the same objective-lattice level",
            where=where, objective_stage=objective_stage,
            P=p, B=b, C=c, F=f,
            dual_primal_delta=dual_primal_delta,
            half_lattice_step=half_step, lattice_step=step,
            solver_primal_same_level_rule=SOLVER_PRIMAL_SAME_LEVEL_RULE)
    require(solver_dual_delta < half_step,
            "solver objective and solver dual estimate do not identify the "
            "same objective-lattice level",
            where=where, objective_stage=objective_stage,
            P=p, B=b, C=c, F=f,
            solver_dual_delta=solver_dual_delta,
            half_lattice_step=half_step, lattice_step=step,
            solver_primal_same_level_rule=SOLVER_PRIMAL_SAME_LEVEL_RULE)

    # Legacy numeric columns remain for schema compatibility only.  For this
    # method B_full is a raw solver diagnostic, not a proven safe lower bound;
    # ``closed`` names the exact-P lattice cell.  Method/source/proof=false
    # prevent either value from being mistaken for the superseded v3 closure.
    safe_bound = b_full
    closed_bound = primal_level
    return {
        "method": LATTICE_METHOD,
        "objective_stage": objective_stage,
        "step": step,
        "offset": offset,
        "bound_ulps": STATUS0_LATTICE_BOUND_ULPS,
        "boundary_margin": boundary_margin,
        "safe_dual_bound_full_objective": safe_bound,
        "closed_dual_bound_full_objective": closed_bound,
        "primal_level_full_objective": primal_level,
        "lower_bound_source": STATUS0_LOWER_BOUND_SOURCE,
        "structural_lower_bound_full_objective": (
            None if lower_bound_int is None else float(lower_bound_int)),
        "exact_full_objective_integer": (
            None if exact_full_int is None else float(exact_full_int)),
        "proves_exact_optimality": False,
        "justification": STATUS0_JUSTIFICATION,
    }


def find_col(columns, name):
    """Locate a metadata field BY NAME (never by position)."""
    cols = list(columns)
    hits = [c for c in cols if str(c) == str(name)]
    require(len(hits) == 1, "column must appear exactly once, located by name",
            name=name, occurrences=len(hits), available=cols)
    return hits[0]


def optional_manifest_value(row, columns, name, default=""):
    """Read optional scenario provenance by exact column name, never position."""
    hits = [c for c in list(columns) if str(c) == str(name)]
    require(len(hits) <= 1, "optional manifest column appears more than once",
            name=name, occurrences=len(hits))
    return row[hits[0]] if hits else default


def manifest_bool(value, name):
    """Parse a manifest boolean without Python's unsafe bool('False') rule."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    token = str(value).strip().lower()
    require(token in ("true", "false", "1", "0"),
            "manifest boolean has an unsupported spelling",
            column=name, value=value)
    return token in ("true", "1")


def resolve_manifest_source_path(value, manifest_dir):
    """Resolve a manifest-declared source without a machine-specific fallback."""
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


def load_design_contract():
    """Load and exact-assert every manifest field that controls executable R1b."""
    require(os.path.isfile(DESIGN_MANIFEST),
            "R1b design manifest is missing", path=DESIGN_MANIFEST)
    with open(DESIGN_MANIFEST, "r", encoding="utf-8") as handle:
        try:
            manifest = json.load(handle)
        except ValueError as exc:
            fatal("R1b design manifest is invalid JSON",
                  path=DESIGN_MANIFEST, error=str(exc))
    require(manifest.get("stage") == STAGE,
            "R1b manifest stage differs from executable stage",
            manifest_stage=manifest.get("stage"), executable_stage=STAGE)
    require(manifest.get("runtime_contract") == EXPECTED_RUNTIME_CONTRACT,
            "R1b executable scientific contract differs from the exact "
            "pre-registered runtime_contract",
            expected=EXPECTED_RUNTIME_CONTRACT,
            actual=manifest.get("runtime_contract"))
    require(np.isfinite(float(H_STD)) and float(H_STD).is_integer()
            and float(manifest["runtime_contract"]["standard_capacity_H_STD"])
            == float(H_STD)
            and manifest["runtime_contract"][
                "formal_standard_capacity_integral"] is True,
            "H_STD must be a manifest-bound integer before W/Z objective "
            "lattices are authorized",
            executable_H_STD=float(H_STD),
            manifest_H_STD=manifest["runtime_contract"].get(
                "standard_capacity_H_STD"),
            manifest_integral_flag=manifest["runtime_contract"].get(
                "formal_standard_capacity_integral"))

    weighted = manifest.get("weighted_objective")
    require(isinstance(weighted, dict),
            "R1b manifest weighted_objective must be an object")
    require(weighted.get("Z") == WEIGHTED_OBJECTIVE_FORMULA
            and float(weighted.get("W2")) == float(W2)
            and float(weighted.get("W5")) == float(W5)
            and list(weighted.get("w4_sweep", [])) == list(W4_SWEEP),
            "R1b manifest weighted objective differs from executable constants",
            manifest=weighted,
            executable={"Z": WEIGHTED_OBJECTIVE_FORMULA, "W2": float(W2),
                        "W5": float(W5), "w4_sweep": list(W4_SWEEP)})

    tie = manifest.get("tie_break")
    require(isinstance(tie, dict)
            and list(tie.get("order", [])) == list(LEX_STAGE_ORDER),
            "R1b manifest lexicographic order differs from executable order",
            manifest_order=(tie or {}).get("order"),
            executable_order=list(LEX_STAGE_ORDER))
    canonical = tie.get("canonical_assignment_stage")
    require(isinstance(canonical, dict)
            and canonical.get("formal_default") == "off"
            and canonical.get("assignment_is_estimand") is False
            and canonical.get("formal_runner_selection")
            == "--no-canonical-assignment"
            and list(canonical.get("probe_admissible_status_codes", []))
            == [0, 2]
            and CANONICAL_ASSIGNMENT_FORMAL_DEFAULT is False,
            "R1b manifest canonical-assignment policy differs from executable policy",
            manifest_policy=canonical,
            executable_formal_default=CANONICAL_ASSIGNMENT_FORMAL_DEFAULT)

    tolerances = manifest.get("tolerances")
    require(isinstance(tolerances, dict)
            and float(tolerances.get("integrality_tolerance")) == INT_TOL
            and float(tolerances.get("lexicographic_row_slack")) == LEX_TOL
            and tolerances.get("cap_allowance_rule") == CAP_ALLOWANCE_RULE,
            "R1b manifest tolerances differ from executable tolerances",
            manifest_tolerances=tolerances,
            executable_integrality_tolerance=INT_TOL,
            executable_lexicographic_row_slack=LEX_TOL)

    variable_bounds = manifest.get("variable_bounds")
    require(
        np.isfinite(OVERTIME_SLACK_UPPER_BOUND)
        and OVERTIME_SLACK_UPPER_BOUND >= 0.0,
        "R1b executable overtime-slack upper bound must be finite and non-negative",
        h_max=float(H_MAX), h_std=float(H_STD),
        overtime_slack_upper_bound=OVERTIME_SLACK_UPPER_BOUND)
    require(
        isinstance(variable_bounds, dict)
        and variable_bounds.get("overtime_slack_variable") == "O[q,i]"
        and float(variable_bounds.get("lower_bound")) == 0.0
        and variable_bounds.get("upper_bound_rule")
        == OVERTIME_SLACK_UPPER_BOUND_RULE
        and float(variable_bounds.get("upper_bound"))
        == OVERTIME_SLACK_UPPER_BOUND,
        "R1b manifest overtime-slack bounds differ from executable bounds",
        manifest_variable_bounds=variable_bounds,
        executable_lower_bound=0.0,
        executable_upper_bound_rule=OVERTIME_SLACK_UPPER_BOUND_RULE,
        executable_upper_bound=OVERTIME_SLACK_UPPER_BOUND)

    decomposition = manifest.get("decomposition")
    require(isinstance(decomposition, dict)
            and tuple(decomposition.get("modes", ())) == DECOMPOSITION_MODES
            and decomposition.get("default") == DEFAULT_DECOMPOSITION
            and decomposition.get("formal_mode") == FORMAL_DECOMPOSITION
            and decomposition.get("formal_per_slot_permitted") is False,
            "R1b manifest decomposition policy differs from executable policy",
            manifest_decomposition=decomposition,
            executable_modes=DECOMPOSITION_MODES,
            executable_default=DEFAULT_DECOMPOSITION,
            executable_formal_mode=FORMAL_DECOMPOSITION)
    grid = manifest.get("design_grid")
    require(isinstance(grid, dict)
            and tuple(grid.get("methods", ())) == ("A1", "A2")
            and list(grid.get("w4", [])) == list(W4_SWEEP)
            and tuple(grid.get("decomposition_modes", ()))
            == DECOMPOSITION_MODES,
            "R1b manifest design grid differs from executable grid",
            manifest_grid=grid,
            executable_methods=["A1", "A2"],
            executable_w4=list(W4_SWEEP),
            executable_modes=DECOMPOSITION_MODES)
    evidence = manifest.get("solver_evidence")
    required_evidence = {
        "status_code", "status_name", "proven_optimal", "solver_fun_raw",
        "P_linearized",
        "B_linearized", "C", "P_full", "B_full", "mip_gap_solver_raw",
        "gap_full_objective", "gap_undefined_reason", "stop_reason",
        "time_limit_s", "exact_certificate_method",
        "exact_certificate_objective_stage", "lattice_step",
        "lattice_offset", "lattice_bound_ulps", "lattice_boundary_margin",
        "lattice_safe_dual_bound_full_objective",
        "lattice_closed_dual_bound_full_objective",
        "lattice_primal_level_full_objective",
        "exact_certificate_lower_bound_source",
        "exact_certificate_structural_lower_bound_full_objective",
        "exact_certificate_exact_full_objective_integer",
        "lattice_proves_exact_optimality", "lattice_justification"}
    require(isinstance(evidence, dict)
            and evidence.get("primary_weighted_role") == PRIMARY_WEIGHTED_ROLE
            and evidence.get("w_refine_role") == W_REFINE_ROLE
            and evidence.get("final_tie_break_role") == FINAL_TIE_BREAK_ROLE
            and evidence.get("u_min_A1_role") == UMIN_A1_ROLE
            and evidence.get("u_min_A2_role") == UMIN_A2_ROLE
            and tuple(evidence.get("main_stage_names", ()))
            == EXECUTABLE_MAIN_STAGE_NAMES
            and set(evidence.get("required_for_each_role", ()))
            == required_evidence,
            "R1b manifest solver-certificate roles differ from executable roles",
            manifest_solver_evidence=evidence,
            executable_roles=[UMIN_A1_ROLE, UMIN_A2_ROLE,
                              PRIMARY_WEIGHTED_ROLE, W_REFINE_ROLE,
                              FINAL_TIE_BREAK_ROLE],
            executable_main_stages=EXECUTABLE_MAIN_STAGE_NAMES,
            executable_required_fields=sorted(required_evidence))
    lineage = manifest.get("objective_lattice_ceiling_v3_lineage")
    require(
        isinstance(lineage, dict)
        and lineage.get("method") == LEGACY_LATTICE_METHOD
        and lineage.get("formal_authority") is False
        and lineage.get("bound_rounding_direction") == "downward"
        and isinstance(lineage.get("bound_rounding_ulps"), int)
        and not isinstance(lineage.get("bound_rounding_ulps"), bool)
        and lineage.get("bound_rounding_ulps") == LATTICE_BOUND_ULPS
        and isinstance(lineage.get("structural_identity_ulps"), int)
        and not isinstance(lineage.get("structural_identity_ulps"), bool)
        and lineage.get("structural_identity_ulps")
        == LATTICE_STRUCTURAL_ULPS
        and lineage.get("solver_vector_same_level_rule")
        == SOLVER_VECTOR_SAME_LEVEL_RULE
        and isinstance(lineage.get("dual_bound_overshoot_guard_ulps"), int)
        and not isinstance(
            lineage.get("dual_bound_overshoot_guard_ulps"), bool)
        and lineage.get("dual_bound_overshoot_guard_ulps")
        == LATTICE_BOUND_ULPS
        and lineage.get("formal_fractional_durations_permitted") is False
        and lineage.get("premise_authority")
        == "rehashed_HS_reparsed_family_wide_before_any_milp_never_artifact_row"
        and lineage.get("H_STD_integral_required") is True
        and lineage.get("effective_duration_rule")
        == "task.duration+TRAVEL_PER_TASK"
        and lineage.get("effective_durations_integral_required") is True
        and lineage.get("raw_P_B_C_F_and_gaps_are_immutable") is True
        and lineage.get("status_1_or_timeout_can_pass") is False
        and lineage.get("steps_full_objective")
        == {"U": 1, "W": 1, "D": 1, "Z": 5}
        and lineage.get("offsets_full_objective")
        == {"U": 0, "W": 0, "D": 0, "Z": 0},
        "R1b v3 lineage differs from the immutable superseded policy",
        manifest_objective_lattice_v3_lineage=lineage,
        executable_method=LEGACY_LATTICE_METHOD,
        executable_bound_rounding_ulps=LATTICE_BOUND_ULPS,
        executable_structural_identity_ulps=LATTICE_STRUCTURAL_ULPS,
        executable_solver_vector_same_level_rule=(
            SOLVER_VECTOR_SAME_LEVEL_RULE))
    status0 = manifest.get("status0_lattice_consistency_certificate")
    require(
        isinstance(status0, dict)
        and status0.get("method") == STATUS0_LATTICE_METHOD
        and status0.get("formal_authority") is True
        and status0.get("claim_wording") == STATUS0_CLAIM_WORDING
        and status0.get("claim_scope") == STATUS0_CLAIM_SCOPE
        and status0.get("proves_exact_optimality") is False
        and tuple(status0.get("required_conditions", ()))
        == STATUS0_REQUIRED_CONDITIONS
        and status0.get("strict_boundary_rule")
        == ("Every registered pairwise distance uses "
            "nextafter(abs(a-b), +infinity) < step/2; equality and "
            "adjacent-cell evidence fail closed.")
        and status0.get("solver_primal_same_level_rule")
        == SOLVER_PRIMAL_SAME_LEVEL_RULE
        and status0.get("solver_vector_same_level_rule")
        == SOLVER_VECTOR_SAME_LEVEL_RULE
        and status0.get("live_checks_before_C") == ["X_vs_P", "F_vs_X"]
        and status0.get("persisted_roundtrip_checks_before_C")
        == ["F_vs_P", "B_vs_P", "F_vs_B"]
        and status0.get("raw_gap_consistency_rule")
        == ("The landed raw solver gap must replay exactly from immutable F "
            "and B under the registered solver-native gap arithmetic; the "
            "landed full-objective gap or zero-denominator reason must replay "
            "exactly from immutable P, B, and C.")
        and status0.get("raw_P_B_C_F_and_gaps_are_immutable") is True
        and status0.get("bound_ulps") == STATUS0_LATTICE_BOUND_ULPS
        and status0.get("boundary_margin") == 0.0
        and status0.get("legacy_safe_field_semantics")
        == STATUS0_LEGACY_SAFE_SEMANTICS
        and status0.get("legacy_closed_field_semantics")
        == STATUS0_LEGACY_CLOSED_SEMANTICS
        and status0.get("legacy_primal_level_field_semantics")
        == STATUS0_LEGACY_PRIMAL_SEMANTICS
        and status0.get("lower_bound_source") == STATUS0_LOWER_BOUND_SOURCE
        and status0.get("X_definition") == SOLVER_VECTOR_X_DEFINITION
        and status0.get("X_persisted") is False
        and status0.get("non_lattice_rule")
        == "unchanged historical exact-raw-bound fallback plus eight_ulp_F_X_identity"
        and status0.get("status_1_or_timeout_can_pass") is False
        and set(status0.get("required_roles", ())) == {
            UMIN_A1_ROLE, UMIN_A2_ROLE, PRIMARY_WEIGHTED_ROLE,
            W_REFINE_ROLE, FINAL_TIE_BREAK_ROLE},
        "R1b status-0 lattice-consistency manifest differs from executable "
        "validation policy",
        manifest_status0_lattice_consistency=status0,
        executable_method=STATUS0_LATTICE_METHOD,
        executable_claim=STATUS0_CLAIM_WORDING,
        executable_required_conditions=STATUS0_REQUIRED_CONDITIONS)
    vector_consistency = manifest.get("solver_vector_consistency")
    require(
        isinstance(vector_consistency, dict)
        and vector_consistency.get("method")
        == SOLVER_VECTOR_METHOD
        and vector_consistency.get("X_definition")
        == SOLVER_VECTOR_X_DEFINITION
        and vector_consistency.get("lattice_rule")
        == SOLVER_VECTOR_SAME_LEVEL_RULE
        and vector_consistency.get("non_lattice_rule")
        == SOLVER_VECTOR_NON_LATTICE_RULE
        and vector_consistency.get("persisted") is False
        and vector_consistency.get("proof_role")
        == SOLVER_VECTOR_PROOF_ROLE
        and vector_consistency.get("scope_note") == SOLVER_VECTOR_SCOPE_NOTE
        and vector_consistency.get("coordinate_wise_proof") is False
        and vector_consistency.get("raw_P_B_C_F_are_rewritten") is False,
        "R1b solver-vector manifest differs from executable live gate",
        manifest_solver_vector_consistency=vector_consistency,
        executable_solver_vector_same_level_rule=(
            SOLVER_VECTOR_SAME_LEVEL_RULE))
    structural = manifest.get("structural_lower_bound_certificate")
    require(
        isinstance(structural, dict)
        and structural.get("method") == STRUCTURAL_LOWER_BOUND_METHOD
        and structural.get("integer_readback_required") is True
        and structural.get("exact_attainment_required") is True
        and structural.get("raw_solver_evidence_is_diagnostic_only") is True
        and structural.get("proves_exact_optimality") is True
        and structural.get("solver_and_dual_same_cell_rule")
        == "up(abs(F-P))<step/2 and up(abs(B-P))<step/2"
        and structural.get("deviation_constant_authority")
        == DEVIATION_CONSTANT_AUTHORITY
        and set(structural.get("registered_lower_bound_sources", ())) == {
            "nonnegative_unserved_count",
            "nonnegative_canonical_overtime",
            "nonnegative_assignment_deviation",
            "nonnegative_weighted_objective",
            "fixed_U_common_plus_nonnegative_overtime_and_deviation",
        },
        "R1b structural lower-bound certificate manifest differs from executable policy",
        manifest_structural_lower_bound_certificate=structural,
        executable_method=STRUCTURAL_LOWER_BOUND_METHOD,
        executable_deviation_constant_authority=(
            DEVIATION_CONSTANT_AUTHORITY))
    service = manifest.get("service_level")
    require(isinstance(service, dict)
            and service.get("rule") == "U_common := U_min_A1"
            and service.get("applied_as")
            == "equality sum_q U_q == U_common in BOTH regimes and every w4"
            and service.get("A1_subset_A2_checked_at_runtime") is True
            and service.get("exclusions_permitted") == 0,
            "R1b common-service narrative differs from executable policy",
            service_level=service)
    disruptions = manifest.get("disruptions")
    require(isinstance(disruptions, dict)
            and disruptions.get("absence_rule") == ABSENCE_RULE
            and disruptions.get("formal_max_absent") is None
            and disruptions.get("formal_coverage")
            == "every active caregiver holding at least one task at q >= T_NOW"
            and disruptions.get("count_semantics") == COUNT_SEMANTICS,
            "R1b formal disruption coverage differs from executable policy",
            disruptions=disruptions, count_semantics=COUNT_SEMANTICS)
    inference = manifest.get("inference")
    require(isinstance(inference, dict)
            and inference.get("disruption_iid_ci_computed") is False
            and inference.get("inferential_stage") == "R5"
            and inference.get("R1b_summary_is_descriptive_only") is True,
            "R1b inference boundary must forbid disruption-IID confidence intervals",
            inference=inference)
    solver = manifest.get("solver")
    require(
        isinstance(solver, dict)
        and solver.get("admissible_statuses")
        == [{"code": 0, "name": "Optimal"}]
        and solver.get("optimal_statuses")
        == [{"code": 0, "name": "Optimal"}]
        and solver.get("accepts_time_limit_incumbent") is False
        and float(solver.get("time_limit_s")) == SOLVER_TIME_LIMIT_S
        and solver.get("options")
        == {"presolve": False, "time_limit": SOLVER_TIME_LIMIT_S}
        and solver.get("retry_policy") == RETRY_POLICY,
        "R1b manifest solver policy differs from the executable recovery contract",
        manifest_solver=solver,
        executable_time_limit_s=SOLVER_TIME_LIMIT_S,
        executable_options={"presolve": False,
                            "time_limit": SOLVER_TIME_LIMIT_S},
        executable_retry_policy=RETRY_POLICY)
    shard_merge = manifest.get("formal_shard_merge")
    require(isinstance(shard_merge, dict)
            and tuple(shard_merge.get("rho_partition", ())) == FORMAL_RHOS
            and shard_merge.get("required_shards") == len(FORMAL_RHOS)
            and shard_merge.get("mutually_exclusive") is True
            and shard_merge.get("merge_owner")
            == ("run_e4.py only; an outer reproducibility harness may launch "
                "shards but may not concatenate them"),
            "R1b formal shard-merge contract differs from executable policy",
            formal_shard_merge=shard_merge)
    dependency = manifest.get("derived_artifact_dependency")
    require(isinstance(dependency, dict)
            and dependency.get("self_reference") is False
            and "source_sha256" in str(dependency.get("receipt", ""))
            and "landed e4_raw.csv" in str(dependency.get("rule", "")),
            "R1b derived-artifact dependency policy must explicitly forbid "
            "self-reference and require a landed-raw SHA256 receipt",
            manifest_dependency=dependency)
    return manifest


def solver_policy(cli_time_limit_s=None, require_cli_time_limit=False,
                  contract=None):
    if contract is None:
        contract = load_design_contract()
    policy = StageSolverPolicy.from_manifest_file(DESIGN_MANIFEST, stage=STAGE)
    require(policy.admissible_codes == {0} and policy.optimal_codes == {0},
            "R1b is optimal-only: the design manifest must declare code 0 as "
            "the only admissible and optimal status",
            admissible=sorted(policy.admissible_codes),
            optimal=sorted(policy.optimal_codes))
    require(policy.solver_options.get("presolve") is False,
            "R1b must pre-register presolve=false; changing options after a "
            "failure and retrying is forbidden",
            solver_options=dict(policy.solver_options))
    require(
        policy.time_limit_s == SOLVER_TIME_LIMIT_S
        and dict(policy.solver_options)
        == {"presolve": False, "time_limit": SOLVER_TIME_LIMIT_S},
        "R1b loaded solver policy differs from the exact executable recovery contract",
        policy_time_limit_s=policy.time_limit_s,
        policy_options=dict(policy.solver_options),
        executable_time_limit_s=SOLVER_TIME_LIMIT_S)
    policy.scipy_milp_options(
        cli_time_limit_s=cli_time_limit_s,
        require_cli_time_limit=require_cli_time_limit)
    return policy

# LEX_TOL is NOT a cap allowance.  It is safe only while consecutive achievable
# values of Z and of W are separated by more than LEX_TOL.  With integral task
# durations `dur = duration + TRAVEL_PER_TASK` is integral, every achievable
# overtime total is integral, and Z = W2*W + w4*D + W5*U with integral W2/w4/W5 is
# integral too, so consecutive levels differ by >= 1.0.  That is a PROPERTY OF
# THIS DATA, recorded per row as `w_level_separation`; a non-integral instance is
# refused unless `--allow-fractional-durations` is passed.

# Reading any of these from this stage would be cross-stage transcription.
_FORBIDDEN_READ_TOKENS = ("eps_range", "eps_frontier", "eps_", "stability_constrained",
                          "rolling_raw", "scalability_", "external_")


def _read_csv_audited(path, why=""):
    """Every tabular read in this module goes through here (REQ-6 item 5)."""
    base = os.path.basename(str(path)).lower()
    for tok in _FORBIDDEN_READ_TOKENS:
        if tok in base:
            fatal("cross-stage transcription refused: the weighted-sum stage may "
                  "not read another stage's artifact", path=str(path),
                  token=tok, why=why)
    # Certificate replay is byte-evidence verification.  The persisted decimal
    # token must round-trip to the same binary64 value; pandas' default parser
    # can choose a neighbouring float and create a false ULP-scale mismatch.
    return pd.read_csv(path, float_precision="round_trip")


# --------------------------------------------------------------------------- #
# instance-side structure + guards  (same rules as the epsilon stage; kept local
# on purpose so the two stages share no artifact and no number)
# --------------------------------------------------------------------------- #
def future_slots(slot_assign):
    return sorted(q for q in slot_assign if q >= T_NOW)


def slot_tasks(parser, slot_assign, q):
    return sorted(t for t in slot_assign[q] if t in parser.tasks)


def available_nurses(parser, S, q, absent):
    """`q in parser.nurses[n].slots` is set membership in the nurse's allowed-slot
    set (lin2025 Eq. 5 semantics).  In THIS shipped data each task's allowed-slot
    set happens to be a singleton, so the requested slot degenerates to a fixed
    shift — a property of the data, not a general equality."""
    return sorted(n for n in S
                  if n != absent and slot_allowed(q, parser.nurses[n].slots))


def assert_incumbent_eligible(parser, slot_assign, S, absent, label=""):
    """FATAL unless every incumbent assignment is candidate-eligible and on duty.

    A1 pins each incumbent task to its incumbent nurse.  If that nurse were not in
    the task's candidate list, A1 would admit an assignment A2 forbids and the
    A1-feasible-set-subset-A2 containment used throughout this project would be
    false.  Guarded here as well as in the epsilon stage."""
    bad_cand, bad_slot, unknown, n = [], [], [], 0
    for q in future_slots(slot_assign):
        for t in slot_tasks(parser, slot_assign, q):
            i = slot_assign[q][t]
            n += 1
            if i not in parser.nurses:
                unknown.append((q, t, i)); continue
            if i not in set(parser.tasks[t].candidate_nurses):
                bad_cand.append((q, t, i))
            if not slot_allowed(q, parser.nurses[i].slots):
                bad_slot.append((q, t, i))
    if bad_cand or bad_slot or unknown:
        fatal("incumbent eligibility guard FAILED — the A1 feasible-set "
              "containment may not be claimed for this instance",
              label=label, absent=absent, incumbent_not_a_candidate=bad_cand[:12],
              incumbent_not_on_duty_in_slot=bad_slot[:12],
              incumbent_nurse_unknown=unknown[:12], n_incumbent_pairs_checked=n)
    return {"n_incumbent_pairs_checked": n}


def assert_incumbent_flat_feasible(parser, slot_assign, label=""):
    """Recompute the incumbent's hard-cap feasibility from the landed plan."""
    allow = cap_fp_slack(H_MAX)
    violations = []
    checked = 0
    for q in future_slots(slot_assign):
        by_nurse = {}
        for t in slot_tasks(parser, slot_assign, q):
            i = slot_assign[q][t]
            by_nurse[i] = by_nurse.get(i, 0.0) + (
                float(parser.tasks[t].duration) + float(TRAVEL_PER_TASK))
        for i, load in by_nurse.items():
            checked += 1
            if load > float(H_MAX) + allow:
                violations.append((q, i, load))
    require(not violations,
            "incumbent plan is not flat-feasible under the executable H_MAX",
            label=label, h_max=float(H_MAX), violations=violations[:12],
            n_violations=len(violations))
    return {"incumbent_flat_feasible": True,
            "n_incumbent_nurse_slot_cells_checked": checked}


def assert_A1_candidates_subset_A2(model_a1, model_a2, absent):
    """Check the load-bearing containment that makes U_common attainable."""
    require(set(model_a1.cand) == set(model_a2.cand),
            "A1/A2 task domains differ", absent=absent,
            only_a1=sorted(set(model_a1.cand) - set(model_a2.cand))[:12],
            only_a2=sorted(set(model_a2.cand) - set(model_a1.cand))[:12])
    bad = []
    for key in sorted(model_a1.cand):
        a1 = set(model_a1.cand[key])
        a2 = set(model_a2.cand[key])
        if not a1.issubset(a2):
            bad.append((key, sorted(a1 - a2)))
    require(not bad,
            "A1 candidate set is not contained in A2; U_common may be "
            "unattainable in A2", absent=absent, violations=bad[:12])
    return {"n_tasks_nested_A1_subset_A2": len(model_a1.cand)}


def active_absentees(parser, slot_assign, max_absent=None):
    """Re-derive disruptions; a cap exists only for explicit rehearsal fixtures."""
    active = sorted({n for q, m in slot_assign.items() if q >= T_NOW
                     for n in m.values()})
    if max_absent is None:
        return active
    require(int(max_absent) > 0, "fixture absence cap must be positive",
            max_absent=max_absent)
    return active[:int(max_absent)]


def scenario_lattice_precondition(parser, slot_assign, *, where):
    """Derive the W/Z integer-lattice premise directly from H/S inputs.

    This helper consumes no artifact row and invokes no solver.  Formal callers
    run it for the complete selected manifest before the first ``milp`` call.
    """
    slots = future_slots(slot_assign)
    require(bool(slots), "no future slot while preflighting lattice inputs",
            where=where, t_now=T_NOW)
    duration_records = [
        (slot, task,
         float(parser.tasks[task].duration) + float(TRAVEL_PER_TASK))
        for slot in slots for task in slot_tasks(parser, slot_assign, slot)]
    require(bool(duration_records),
            "no future task while preflighting lattice inputs", where=where)
    h_std_integral = (
        np.isfinite(float(H_STD)) and float(H_STD).is_integer())
    durations_integral = all(
        np.isfinite(value) and float(value).is_integer()
        for _slot, _task, value in duration_records)
    nonintegral = [
        {"slot": slot, "task": task, "effective_duration": value}
        for slot, task, value in duration_records
        if not (np.isfinite(value) and float(value).is_integer())]
    return {
        "integral_lattice_data": bool(
            h_std_integral and durations_integral),
        "integral_effective_durations": bool(durations_integral),
        "integral_H_STD": bool(h_std_integral),
        "H_STD": float(H_STD),
        "n_effective_durations_checked": len(duration_records),
        "nonintegral_examples": nonintegral[:8],
    }


def _scenario_key(rho, seed):
    return (round(float(rho), 8), int(seed))


def preflight_manifest_lattice(
        man, man_dir, *, rho_column, seed_column, h_column, s_column,
        allow_fractional, formal):
    """Re-hash and parse the complete manifest before any solver call.

    Formal rho sharding does not narrow this preflight: every shard validates
    all selected-family scenarios first, and only then applies ``--rho`` to the
    solve workload.  This prevents a later fractional scenario from being
    discovered after earlier scientific MILPs have already run.
    """
    source_columns = (
        "src", "src_sha256", "generator", "generator_sha256",
        "family_runner", "family_runner_sha256")
    present = [column in man.columns for column in source_columns]
    require(all(present) or not any(present),
            "instance manifest has a partial source-binding family",
            present={name: found for name, found in zip(source_columns, present)})
    if formal:
        require(all(present),
                "formal lattice preflight requires complete source bindings")

    premises, receipts, input_paths = {}, {}, []
    for _, row in man.iterrows():
        rho = float(row[rho_column])
        seed = _exact_int_cell(
            row[seed_column], field=str(seed_column),
            where="R1b lattice manifest preflight")
        key = _scenario_key(rho, seed)
        require(key not in premises,
                "lattice preflight found a duplicate (rho,seed) scenario",
                rho=rho, seed=seed)
        hp = os.path.abspath(os.path.join(man_dir, str(row[h_column])))
        sp = os.path.abspath(os.path.join(man_dir, str(row[s_column])))
        for path in (hp, sp):
            require(os.path.isfile(path),
                    "lattice preflight manifest points at a missing file",
                    rho=rho, seed=seed, path=path)
        h_before, s_before = sha256_file(hp), sha256_file(sp)
        declared_h = str(optional_manifest_value(
            row, man.columns, "h_sha256", h_before)).strip()
        declared_s = str(optional_manifest_value(
            row, man.columns, "s_sha256", s_before)).strip()
        require(declared_h == h_before and declared_s == s_before,
                "lattice preflight H/S bytes disagree with manifest SHA256",
                rho=rho, seed=seed, declared_h_sha256=declared_h,
                actual_h_sha256=h_before, declared_s_sha256=declared_s,
                actual_s_sha256=s_before)
        parser = HHCPFullParser(hp)
        slot_assign = load_static(sp)
        require(sha256_file(hp) == h_before and sha256_file(sp) == s_before,
                "lattice preflight H/S bytes changed while being parsed",
                rho=rho, seed=seed, hhcp=hp, sol=sp)
        premise = scenario_lattice_precondition(
            parser, slot_assign,
            where="R1b lattice preflight rho={} seed={}".format(rho, seed))
        premise.update({"rho": rho, "seed": seed,
                        "h_sha256": h_before, "s_sha256": s_before})
        require(allow_fractional or premise["integral_lattice_data"],
                "non-integral lattice input detected in family-wide preflight "
                "before the first solver call",
                rho=rho, seed=seed, H_STD=premise["H_STD"],
                integral_H_STD=premise["integral_H_STD"],
                integral_effective_durations=(
                    premise["integral_effective_durations"]),
                nonintegral_examples=premise["nonintegral_examples"],
                allow_fractional_durations=allow_fractional)
        premises[key] = premise
        for path, digest in ((hp, h_before), (sp, s_before)):
            receipts[path] = digest
            input_paths.append(path)

        if all(present):
            for label, path_col, hash_col in (
                    ("source", "src", "src_sha256"),
                    ("generator", "generator", "generator_sha256"),
                    ("family_runner", "family_runner", "family_runner_sha256")):
                source_path = resolve_manifest_source_path(
                    row[path_col], man_dir)
                if source_path is None:
                    # Release marker names this source as withheld; its recorded
                    # digest is the only provenance available here.
                    continue
                digest = sha256_file(source_path)
                require(digest == str(row[hash_col]).strip(),
                        "lattice preflight source bytes disagree with SHA256",
                        rho=rho, seed=seed, source_role=label,
                        path=source_path,
                        declared_sha256=str(row[hash_col]).strip(),
                        actual_sha256=digest)
                receipts[source_path] = digest
                input_paths.append(source_path)
        if "feasible" in man.columns:
            require(manifest_bool(row["feasible"], "feasible"),
                    "lattice preflight manifest declares an infeasible incumbent",
                    rho=rho, seed=seed)
    return {
        "premises": premises,
        "receipts": receipts,
        "input_paths": list(dict.fromkeys(input_paths)),
        "formal_family_wide": bool(formal),
    }


def assert_preflight_receipts_unchanged(preflight, *, where):
    changed = []
    for path, expected in preflight["receipts"].items():
        actual = sha256_file(path)
        if actual != expected:
            changed.append({"path": path, "expected_sha256": expected,
                            "actual_sha256": actual})
    require(not changed,
            "lattice-preflight source receipt changed after family-wide parse",
            where=where, changed=changed[:12], n_changed=len(changed))


# --------------------------------------------------------------------------- #
# weighted-sum model over an arbitrary slot subset
# --------------------------------------------------------------------------- #
class WeightedModel:
    """Recovery MILP over a supplied slot set.

    Formal R1b always supplies the complete future horizon because the common
    service equality is horizon-global.  A one-slot model remains available only
    to historical direct-model fixture probes.
    """

    def __init__(self, parser, slot_assign, S, absent, regime, slots, policy, *,
                 rho=None, seed=None):
        if regime not in ("A1", "A2"):
            fatal("unknown regime", regime=regime)
        self.parser, self.sa, self.S = parser, slot_assign, S
        self.absent, self.regime, self.policy = absent, regime, policy
        self.rho = None if rho is None else float(rho)
        self.seed = None if seed is None else int(seed)
        self.diagnostics = []
        self.slots = list(slots)
        self.tasks = {q: slot_tasks(parser, slot_assign, q) for q in self.slots}
        self.avail = {q: available_nurses(parser, S, q, absent) for q in self.slots}

        self.cand, self.empty_after_absence, pre_cand = {}, [], {}
        for q in self.slots:
            av = set(self.avail[q])
            for t in self.tasks[q]:
                # non-empty is required BEFORE the absence is applied; a set that
                # empties only because the absent nurse was removed is a legitimate
                # structural outcome and is counted, not treated as a failure.
                pre_cand[(q, t)] = [n for n in parser.tasks[t].candidate_nurses
                                    if n in S]
                full = sorted(n for n in parser.tasks[t].candidate_nurses if n in av)
                st = slot_assign[q][t]
                self.cand[(q, t)] = [st] if (regime == "A1" and st != absent) else full
                if not self.cand[(q, t)]:
                    self.empty_after_absence.append((q, t))
        require_nonempty_candidates(
            pre_cand, where=f"pre-absence candidate sets (regime={regime}, "
                            f"absent={absent})")

        self.dur = {(q, t): float(parser.tasks[t].duration) + float(TRAVEL_PER_TASK)
                    for q in self.slots for t in self.tasks[q]}
        self.xbar = set()
        for q in self.slots:
            av = set(self.avail[q])
            for t in self.tasks[q]:
                i = slot_assign[q][t]
                if i != absent and i in av:
                    self.xbar.add((q, i, t))

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
        self.integral_effective_durations = all(
            np.isfinite(float(v)) and float(v).is_integer()
            for v in self.dur.values())
        self.integral_standard_capacity = (
            np.isfinite(float(H_STD)) and float(H_STD).is_integer())
        # Historical field name retained in artifacts, but its executable
        # meaning is now the complete W/Z lattice premise, including H_STD.
        self.integral_durations = bool(
            self.integral_effective_durations
            and self.integral_standard_capacity)
        self._rows, self._lb, self._ub = self._structural_rows()

    # ---- rows -------------------------------------------------------------
    def _structural_rows(self):
        rows, lb, ub = [], [], []
        for q in self.slots:
            for t in self.tasks[q]:
                r = np.zeros(self.n)
                for i in self.cand[(q, t)]:
                    r[self.xidx[(q, i, t)]] = 1.0
                r[self.uidx[(q, t)]] = 1.0
                rows.append(r); lb.append(1.0); ub.append(1.0)
        for q in self.slots:
            for i in self.avail[q]:
                r = np.zeros(self.n)
                for t in self.tasks[q]:
                    k = self.xidx.get((q, i, t))
                    if k is not None:
                        r[k] = self.dur[(q, t)]
                r[self.oidx[(q, i)]] = -1.0
                rows.append(r); lb.append(-np.inf); ub.append(float(H_STD))
        for q in self.slots:
            for i in self.avail[q]:
                r = np.zeros(self.n)
                for t in self.tasks[q]:
                    k = self.xidx.get((q, i, t))
                    if k is not None:
                        r[k] = self.dur[(q, t)]
                rows.append(r); lb.append(-np.inf); ub.append(float(H_MAX))
        return rows, lb, ub

    def _o_row(self):
        r = np.zeros(self.n)
        for k in self.oidx.values():
            r[k] = 1.0
        return r

    def _u_row(self):
        r = np.zeros(self.n)
        for k in self.uidx.values():
            r[k] = 1.0
        return r

    def _z_row(self, w4):
        """W2*sum(O) + w4*(dev_coef.x + dev_const) + W5*sum(u): linear part only."""
        return float(W2) * self._o_row() + float(w4) * self.dev_coef \
            + float(W5) * self._u_row()

    def _obj(self, what, w4=None):
        if what == "Z":
            return self._z_row(w4)
        if what == "U":
            return self._u_row()
        if what == "W":
            return self._o_row()
        if what == "D":
            return self.dev_coef.copy()
        fatal("unknown objective stage", what=what)

    def _variable_bounds(self, fixed=None):
        """Return the exact registered variable bounds for every R1b solve.

        Hard capacity gives ``load <= H_MAX`` and the overtime row gives
        ``O >= load - H_STD``.  Hence every assignment admits the canonical
        slack ``O=max(0, load-H_STD) <= H_MAX-H_STD``.  Replacing an oversized
        slack by that canonical value never changes x, U, D, or feasibility and
        cannot worsen any Z/W objective or W upper-bound row.  The finite bound
        therefore removes only scientifically irrelevant slack rays.
        """
        lo = np.zeros(self.n)
        hi = np.ones(self.n)
        for k in self.oidx.values():
            hi[k] = OVERTIME_SLACK_UPPER_BOUND
        for k, v in (fixed or {}).items():
            require(0 <= int(k) < self.n,
                    "fixed variable index is outside the R1b model",
                    index=k, n_variables=self.n)
            value = float(v)
            require(lo[int(k)] <= value <= hi[int(k)],
                    "fixed variable value violates the registered R1b bounds",
                    index=int(k), value=value, lower=float(lo[int(k)]),
                    upper=float(hi[int(k)]))
            lo[int(k)] = hi[int(k)] = value
        return Bounds(lo, hi)

    # ---- solve ------------------------------------------------------------
    def _solve(self, what, w4=None, z_le=None, u_eq=None, w_le=None,
               fixed=None, allow_infeasible=False, stage_label="",
               known_feasible_x=None):
        rows = list(self._rows); lb = list(self._lb); ub = list(self._ub)
        if z_le is not None:
            rows.append(self._z_row(w4)); lb.append(-np.inf)
            ub.append(float(z_le) + LEX_TOL - float(w4) * self.dev_const)
        if u_eq is not None:
            rows.append(self._u_row()); lb.append(float(u_eq)); ub.append(float(u_eq))
        if w_le is not None:
            rows.append(self._o_row()); lb.append(-np.inf)
            ub.append(float(w_le) + LEX_TOL)
        integ = np.zeros(self.n)
        for k in range(len(self.xpairs)):
            integ[k] = 1
        for k in self.uidx.values():
            integ[k] = 1
        t0 = time.perf_counter()
        res = milp(c=self._obj(what, w4),
                   constraints=LinearConstraint(np.array(rows), np.array(lb),
                                                np.array(ub)),
                   integrality=integ, bounds=self._variable_bounds(fixed),
                   options=self.policy.scipy_milp_options())
        wall = time.perf_counter() - t0
        status = int(getattr(res, "status", -1))
        self.diagnostics.append({
            "stage": STAGE, "rho": self.rho, "seed": self.seed,
            "regime": self.regime, "absent": self.absent,
            "slot_set": "|".join(str(q) for q in self.slots),
            "lex_stage": stage_label or what, "objective_stage": what,
            "w4": w4, "z_le": z_le, "u_eq": u_eq, "w_le": w_le,
            "solver_status_code": status,
            "solver_message": str(getattr(res, "message", "")),
            "wall_time_s": wall,
            "nodes": getattr(res, "mip_node_count", None),
            "time_limit_s": self.policy.time_limit_s,
            "presolve": self.policy.solver_options.get("presolve"),
        })
        if status == 2 and known_feasible_x is not None:
            # The previous lexicographic stage returned a point that satisfies
            # every row of THIS stage, so "infeasible" is a demonstrable solver
            # defect, not a property of the data.  It is reported, never retried
            # into a pass.  Observed on scipy/HiGHS with presolve ENABLED; the
            # same model solves to status 0 with options={"presolve": False}.
            # Reproducer: tests/fixtures/epse4/negative/highs_presolve_infeasible.py
            fatal("solver reported INFEASIBLE for a model that the previous "
                  "lexicographic stage already satisfies — the verdict is "
                  "refuted by an explicit feasible point. Do not relax a "
                  "tolerance to make this pass; this is a solver/presolve "
                  "defect and must be reported.",
                  stage=STAGE, lex_stage=stage_label or what, w4=w4,
                  rho=self.rho, seed=self.seed, regime=self.regime,
                  absent=self.absent,
                  rows_violated_by_the_known_point=self._violated_rows(
                      known_feasible_x, z_le, u_eq, w_le, w4),
                  message=str(getattr(res, "message", "")),
                  time_limit_s=self.policy.time_limit_s,
                  presolve=self.policy.solver_options.get("presolve"),
                  retry_attempted=False)
        if status == 2 and allow_infeasible:
            return None, status
        if status not in self.policy.admissible_codes:
            fatal(
                "solver returned a non-admissible status; R1b aborts without retry",
                stage=STAGE, rho=self.rho, seed=self.seed,
                absent=self.absent, regime=self.regime,
                slot_set="|".join(str(q) for q in self.slots),
                lex_stage=stage_label or what, objective_stage=what, w4=w4,
                z_le=z_le, u_eq=u_eq, w_le=w_le,
                status_code=status,
                admissible_status_codes=sorted(self.policy.admissible_codes),
                solver_message=str(getattr(res, "message", "")),
                incumbent_present=getattr(res, "x", None) is not None,
                objective_value=getattr(res, "fun", None),
                mip_dual_bound=getattr(res, "mip_dual_bound", None),
                mip_gap=getattr(res, "mip_gap", None),
                mip_node_count=getattr(res, "mip_node_count", None),
                wall_time_s=wall, time_limit_s=self.policy.time_limit_s,
                presolve=self.policy.solver_options.get("presolve"),
                retry_attempted=False, retry_policy=RETRY_POLICY)
        if res.x is None:
            self.policy.check(status, where=f"{STAGE}:{stage_label or what}")
            fatal("solver returned an admissible status but no solution vector",
                  stage=STAGE, rho=self.rho, seed=self.seed,
                  lex_stage=stage_label or what, objective_stage=what,
                  status=status, w4=w4, regime=self.regime,
                  absent=self.absent,
                  message=str(getattr(res, "message", "")),
                  time_limit_s=self.policy.time_limit_s,
                  presolve=self.policy.solver_options.get("presolve"),
                  retry_attempted=False)
        sol = self.readback(res.x, stage_label or what, w4)
        self._assert_exact_readback_lex_feasible(
            sol, z_le=z_le, u_eq=u_eq, w_le=w_le,
            where=f"{STAGE}:{stage_label or what}")
        # `primal` is the LINEARIZED objective actually minimised;
        # `deviation_constant_C` is the constant the deviation form drops, so
        # s2_core can rebuild the full-objective gap from the persisted row.
        if what == "Z":
            primal, c_const = (float(W2) * sol["W"] + float(w4) * sol["D"]
                               - float(w4) * self.dev_const
                               + float(W5) * sol["U"]), float(w4) * self.dev_const
        elif what == "D":
            primal, c_const = float(sol["D"]) - self.dev_const, self.dev_const
        elif what == "W":
            primal, c_const = float(sol["W"]), 0.0
        else:
            primal, c_const = float(sol["U"]), 0.0
        solver_result = SolverResult.from_scipy(
            res, policy=self.policy, deviation_constant_C=c_const,
            wall_time_s=wall, primal=primal,
            assignment_hash_value=assignment_hash(sol["assignment"]),
            where=f"{STAGE}:{stage_label or what}")
        require(solver_result.status_code == 0 and solver_result.proven_optimal,
                "objective certificate may only consume a code-0 proven result",
                stage=STAGE, rho=self.rho, seed=self.seed,
                absent=self.absent, regime=self.regime,
                lex_stage=stage_label or what,
                status_code=solver_result.status_code,
                proven_optimal=solver_result.proven_optimal)
        solver_fun_raw = require_finite(
            getattr(res, "fun", None), "solver-reported objective fun",
            stage=STAGE, rho=self.rho, seed=self.seed,
            absent=self.absent, regime=self.regime,
            lex_stage=stage_label or what)
        expected_solver_gap = solver_raw_gap_from_reported_fun(
            solver_fun_raw, solver_result.dual_bound,
            where=f"{STAGE}:{stage_label or what}:live-status0-gap")
        observed_solver_gap = require_finite(
            solver_result.gap_solver_raw, "solver-reported mip gap",
            stage=STAGE, rho=self.rho, seed=self.seed,
            absent=self.absent, regime=self.regime,
            lex_stage=stage_label or what)
        require(abs(observed_solver_gap - expected_solver_gap)
                <= _ulp_margin(observed_solver_gap, expected_solver_gap),
                "solver-reported mip gap disagrees with live F/B replay",
                stage=STAGE, rho=self.rho, seed=self.seed,
                absent=self.absent, regime=self.regime,
                lex_stage=stage_label or what,
                solver_fun_raw=solver_fun_raw,
                mip_dual_bound=solver_result.dual_bound,
                observed_mip_gap=observed_solver_gap,
                expected_mip_gap=expected_solver_gap)
        certificate_where = (
            "{}:rho={}:seed={}:absent={}:method={}:w4={}:role={}".format(
                STAGE, self.rho, self.seed, self.absent, self.regime, w4,
                stage_label or what))
        objective_from_x = float(
            self._obj(what, w4) @ np.asarray(res.x, dtype=float))
        assert_live_objective_reconstruction(
            what, solver_fun_raw, objective_from_x, solver_result.primal,
            w4=w4, integral_durations=self.integral_durations,
            where=certificate_where)
        exact_full_int, structural_floor_int, structural_floor_source = (
            _structural_certificate_inputs(
                what, sol, w4=w4, u_eq=u_eq, where=certificate_where))
        try:
            lattice_certificate = exact_objective_certificate(
                what, solver_result.primal, solver_result.dual_bound,
                solver_result.deviation_constant_C, solver_fun_raw, w4=w4,
                integral_durations=self.integral_durations,
                exact_full_objective_integer=exact_full_int,
                structural_lower_bound_integer=structural_floor_int,
                structural_lower_bound_source=structural_floor_source,
                where=certificate_where)
        except PipelineFailure:
            print("R1B_CERTIFICATE_FAILURE " + json.dumps({
                "rho": self.rho, "seed": self.seed,
                "absent": self.absent, "method": self.regime,
                "w4": w4, "role": stage_label or what,
                "objective_stage": what,
                "P": solver_result.primal,
                "B": solver_result.dual_bound,
                "C": solver_result.deviation_constant_C,
                "F": solver_fun_raw,
                "exact_full_objective_integer": exact_full_int,
                "structural_lower_bound_integer": structural_floor_int,
                "structural_lower_bound_source": structural_floor_source,
                "n_tasks": int(self.n_tasks),
                "exact_U": int(sol["U"]),
                "u_eq": (None if u_eq is None else int(u_eq)),
                "assigned_count": int(self.n_tasks) - int(sol["U"]),
                "deviation_constant": int(self.dev_const),
            }, sort_keys=True), file=sys.stderr, flush=True)
            raise
        sol["solver"] = solver_result
        sol["solver_fun_raw"] = solver_fun_raw
        sol["exact_objective_certificate"] = lattice_certificate
        self.diagnostics[-1].update({
            "obj_incumbent": solver_result.primal,
            "solver_fun_raw": solver_fun_raw,
            "mip_dual_bound": solver_result.dual_bound,
            "mip_gap_solver_raw": solver_result.gap_solver_raw,
            "gap_full_objective": solver_result.gap_full_objective,
            "exact_certificate_method": lattice_certificate["method"],
            "exact_certificate_objective_stage": (
                lattice_certificate["objective_stage"]),
            "lattice_step": lattice_certificate["step"],
            "lattice_offset": lattice_certificate["offset"],
            "lattice_bound_ulps": lattice_certificate["bound_ulps"],
            "lattice_boundary_margin": lattice_certificate["boundary_margin"],
            "lattice_safe_dual_bound_full_objective": (
                lattice_certificate["safe_dual_bound_full_objective"]),
            "lattice_closed_dual_bound_full_objective": (
                lattice_certificate["closed_dual_bound_full_objective"]),
            "lattice_primal_level_full_objective": (
                lattice_certificate["primal_level_full_objective"]),
            "exact_certificate_lower_bound_source": (
                lattice_certificate["lower_bound_source"]),
            "exact_certificate_structural_lower_bound_full_objective": (
                lattice_certificate[
                    "structural_lower_bound_full_objective"]),
            "exact_certificate_exact_full_objective_integer": (
                lattice_certificate["exact_full_objective_integer"]),
            "lattice_proves_exact_optimality": (
                lattice_certificate["proves_exact_optimality"]),
        })
        sol["status"] = status
        sol["x"] = np.asarray(res.x, dtype=float)
        return sol, status

    def _violated_rows(self, x, z_le, u_eq, w_le, w4):
        """Which of the lexicographic rows the supposedly-infeasible point breaks.

        An empty list is the whole point: it means the solver's verdict is wrong.
        """
        out = []
        x = np.asarray(x, dtype=float)
        if z_le is not None:
            v = float(self._z_row(w4) @ x)
            ub = float(z_le) + LEX_TOL - float(w4) * self.dev_const
            if v > ub:
                out.append(("Z", v, ub))
        if u_eq is not None:
            v = float(self._u_row() @ x)
            if abs(v - float(u_eq)) > 1e-9:
                out.append(("U", v, float(u_eq)))
        if w_le is not None:
            v = float(self._o_row() @ x)
            if v > float(w_le) + LEX_TOL:
                out.append(("W", v, float(w_le) + LEX_TOL))
        A = np.array(self._rows)
        vals = A @ x
        for i, (lo_, hi_) in enumerate(zip(self._lb, self._ub)):
            if vals[i] < lo_ - 1e-9 or vals[i] > hi_ + 1e-9:
                out.append((f"structural_row_{i}", float(vals[i]), (lo_, hi_)))
        return out

    # ---- exact read-back --------------------------------------------------
    def readback(self, x, stage_label="", w4=None):
        """Recompute every reported quantity from the integral assignment.

        The solver's overtime slacks are only lower-bounded, so in the U- and
        D-stages they are arbitrary above the true value and are never read.
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
                uv = x[self.uidx[(q, t)]]
                if abs(uv - round(uv)) > INT_TOL:
                    fatal("unserved indicator fractional", stage=stage_label,
                          slot=q, task=t, value=uv)
                if bool(round(uv)) != (not hits):
                    fatal("unserved indicator disagrees with the assignment",
                          stage=stage_label, slot=q, task=t, u=uv, nurses=hits)
        W, allow = 0.0, cap_fp_slack(H_MAX)
        for q in self.slots:
            for i in self.avail[q]:
                load = sum(self.dur[(q, t)] for t in self.tasks[q]
                           if served[(q, t)] == i)
                if load > float(H_MAX) + allow:
                    fatal("hard capacity violated", stage=stage_label, slot=q,
                          nurse=i, load=load, h_max=float(H_MAX),
                          allowance=allow, allowance_rule="math.ulp(cap)*8")
                W += max(0.0, load - float(H_STD))
        U = sum(1 for v in served.values() if v is None)
        D = len(chosen - self.xbar) + len(self.xbar - chosen)
        require_finite(W, "recomputed overtime W", stage=stage_label)
        Z = None if w4 is None else float(W2) * W + float(w4) * D + float(W5) * U
        return {"assignment": served, "U": int(U), "W": float(W), "D": int(D),
                "Z": Z, "n_tasks": self.n_tasks, "stage": stage_label}

    def _assert_exact_readback_lex_feasible(
            self, sol, *, z_le=None, u_eq=None, w_le=None, where):
        """Fail unless the canonical exact readback satisfies every lex row.

        ``readback`` rounds only solver-declared integral variables, rebuilds
        service and deviation from that assignment, and replaces numerical
        overtime slacks by ``max(0, load-H_STD)``.  A lattice certificate may
        consume that exact feasible point only after the rebuilt quantities
        have also been checked against the lexicographic rows of this solve.
        """
        if u_eq is not None:
            target = require_finite(u_eq, "u_eq exact-readback target", where=where)
            require(float(target).is_integer(),
                    "canonical exact readback received a non-integral U equality",
                    where=where, u_eq=target)
            require(int(sol["U"]) == int(target),
                    "canonical exact readback violates the U equality",
                    where=where, exact_U=int(sol["U"]), u_eq=int(target))

        if z_le is not None:
            target = require_finite(z_le, "z_le exact-readback target", where=where)
            exact_z = require_finite(
                sol.get("Z"), "canonical exact-readback Z", where=where)
            require(exact_z <= target + LEX_TOL,
                    "canonical exact readback violates the Z refinement row",
                    where=where, exact_Z=exact_z, z_le=target,
                    lexicographic_row_slack=LEX_TOL)

        if w_le is not None:
            target = require_finite(w_le, "w_le exact-readback target", where=where)
            exact_w = require_finite(
                sol.get("W"), "canonical exact-readback W", where=where)
            require(exact_w <= target + LEX_TOL,
                    "canonical exact readback violates the W refinement row",
                    where=where, exact_W=exact_w, w_le=target,
                    lexicographic_row_slack=LEX_TOL)

    # ---- lexicographic driver --------------------------------------------
    def min_U(self, stage_label="preliminary_minU"):
        """Solve service independently of every finite weighted objective."""
        s, _ = self._solve("U", stage_label=stage_label)
        return s

    def solve_lex(self, w4, u_eq=None, canonical_assignment=False):
        """Solve one weighted point, normally at the horizon U_common equality.

        The stage driver always supplies ``u_eq``.  ``None`` retains the old
        direct-model path only for historical non-formal fixture probes; no
        emitted R1b row can use that path.
        """
        if u_eq is not None:
            require(float(u_eq).is_integer() and int(u_eq) >= 0,
                    "common service target must be a non-negative integer",
                    u_eq=u_eq)
            u_target = int(u_eq)
            weighted, _ = self._solve(
                "Z", w4=w4, u_eq=u_target,
                stage_label="lex2_minZ_at_Ucommon")
            z_star = weighted["Z"]
            w_refine, _ = self._solve(
                "W", w4=w4, z_le=z_star, u_eq=u_target,
                stage_label="lex3_minW", known_feasible_x=weighted["x"])
            w_star = w_refine["W"]
            require(float(w4) >= 0.0,
                    "the stage-4 redundancy argument needs w4 >= 0", w4=w4)
            final, _ = self._solve(
                "D", w4=w4, u_eq=u_target, w_le=w_star,
                stage_label="lex4_minD", known_feasible_x=w_refine["x"])
            if final["Z"] > z_star + LEX_TOL:
                fatal("common-service refinement degraded the weighted objective",
                      z_star=z_star, z_final=final["Z"], lex_tol=LEX_TOL)
            if final["U"] != u_target:
                fatal("common-service equality changed during refinement",
                      u_common=u_target, u_final=final["U"])
            if not (w_star - LEX_TOL <= final["W"] <= w_star + LEX_TOL):
                fatal("common-service refinement changed W after the W stage",
                      w_star=w_star, w_final=final["W"], lex_tol=LEX_TOL)
            chain = [weighted["status"], w_refine["status"], final["status"]]
            out = dict(
                U=final["U"], W=final["W"], D=final["D"], Z=final["Z"],
                n_tasks=final["n_tasks"], assignment=dict(final["assignment"]),
                status_chain="|".join(str(c) for c in chain),
                lex_stage_chain="|".join(EXECUTABLE_MAIN_STAGE_NAMES),
                solver=final["solver"],
                primary_weighted_solver=weighted["solver"],
                primary_weighted_solver_fun_raw=weighted["solver_fun_raw"],
                primary_weighted_exact_certificate=(
                    weighted["exact_objective_certificate"]),
                w_refine_solver=w_refine["solver"],
                w_refine_solver_fun_raw=w_refine["solver_fun_raw"],
                w_refine_exact_certificate=(
                    w_refine["exact_objective_certificate"]),
                final_d_tie_break_solver=final["solver"],
                final_d_tie_break_solver_fun_raw=final["solver_fun_raw"],
                final_d_tie_break_exact_certificate=(
                    final["exact_objective_certificate"]),
                canonical_assignment=False)
            if canonical_assignment:
                canonical = self._canonical_assignment(
                    w4, z_star, u_target, w_star, final["D"])
                out.update(canonical)
                out["canonical_assignment"] = True
                h = assignment_hash(out["assignment"])
                out["solver"].assignment_hash = h
                out["final_d_tie_break_solver"].assignment_hash = h
                out["status_chain"] += "|" + "|".join(
                    str(code) for code in canonical["canonical_status_codes"])
                out["lex_stage_chain"] += "|" + "|".join(
                    canonical["canonical_stage_names"])
            else:
                out["solver"].assignment_hash = None
            return out

        # Legacy direct-model fixture path.  The artifact driver never calls it.
        s1, _ = self._solve("Z", w4=w4, stage_label="lex1_minZ")
        z_star = s1["Z"]
        s2, _ = self._solve("U", w4=w4, z_le=z_star, stage_label="lex2_minU",
                            known_feasible_x=s1["x"])
        u_star = s2["U"]
        s3, _ = self._solve("W", w4=w4, z_le=z_star, u_eq=u_star,
                            stage_label="lex3_minW",
                            known_feasible_x=s2["x"])
        w_star = s3["W"]
        # Stage 4 does NOT carry the `Z <= Z*` row, and the omission is a
        # theorem, not a convenience: with `U = U*` fixed, `W <= W*` and
        # `w4 >= 0`, any D-minimal solution has `D <= D(stage 3)`, so
        # `Z = W2*W + w4*D + W5*U <= W2*W* + w4*D(stage 3) + W5*U* = Z*`.  The
        # row is therefore implied.  It is kept as a POST-HOC ASSERTION below, so
        # nothing is taken on trust.  Carrying it as a row is what exposed the
        # HiGHS presolve defect recorded in
        # tests/fixtures/epse4/negative/highs_presolve_infeasible.py: the Z row
        # contains W2*sum(O) and the W row bounds sum(O), and with LEX_TOL on both
        # presolve declared a model INFEASIBLE that has an explicit feasible point.
        require(float(w4) >= 0.0,
                "the stage-4 redundancy argument needs w4 >= 0", w4=w4)
        s4, _ = self._solve("D", w4=w4, u_eq=u_star, w_le=w_star,
                            stage_label="lex4_minD",
                            known_feasible_x=s3["x"])
        if s4["Z"] > z_star + LEX_TOL:
            fatal("lexicographic order violated: a refinement degraded Z",
                  z_star=z_star, z_final=s4["Z"], lex_tol=LEX_TOL)
        if s4["U"] != u_star:
            fatal("lexicographic order violated: U changed after stage 2",
                  u_star=u_star, u_final=s4["U"])
        if not (w_star - LEX_TOL <= s4["W"] <= w_star + LEX_TOL):
            fatal("lexicographic order violated: W changed after stage 3",
                  w_star=w_star, w_final=s4["W"], lex_tol=LEX_TOL)
        chain = [s1["status"], s2["status"], s3["status"], s4["status"]]
        out = dict(U=s4["U"], W=s4["W"], D=s4["D"], Z=s4["Z"], n_tasks=s4["n_tasks"],
                   assignment=dict(s4["assignment"]),
                   status_chain="|".join(str(c) for c in chain),
                   lex_stage_chain="lex1_minZ|lex2_minU|lex3_minW|lex4_minD",
                   solver=s4["solver"],
                   primary_weighted_solver=s1["solver"],
                   primary_weighted_solver_fun_raw=s1["solver_fun_raw"],
                   primary_weighted_exact_certificate=(
                       s1["exact_objective_certificate"]),
                   w_refine_solver=s3["solver"],
                   w_refine_solver_fun_raw=s3["solver_fun_raw"],
                   w_refine_exact_certificate=(
                       s3["exact_objective_certificate"]),
                   final_d_tie_break_solver=s4["solver"],
                   final_d_tie_break_solver_fun_raw=s4["solver_fun_raw"],
                   final_d_tie_break_exact_certificate=(
                       s4["exact_objective_certificate"]),
                   canonical_assignment=False)
        if canonical_assignment:
            canonical = self._canonical_assignment(
                w4, z_star, u_star, w_star, s4["D"])
            out.update(canonical)
            out["canonical_assignment"] = True
            out["solver"].assignment_hash = assignment_hash(out["assignment"])
            out["final_d_tie_break_solver"].assignment_hash = assignment_hash(
                out["assignment"])
            out["status_chain"] += "|" + "|".join(
                str(code) for code in canonical["canonical_status_codes"])
            out["lex_stage_chain"] += "|" + "|".join(
                canonical["canonical_stage_names"])
        else:
            # The numerical (D,W,U) point is lexicographically unique, but the
            # solver-selected assignment is not.  Do not persist its hash as
            # cross-run reproducibility evidence.
            out["solver"].assignment_hash = None
        return out

    def _canonical_assignment(self, w4, z_star, u_star, w_star, d_star):
        """Stage 5: the lexicographically smallest x under the fixed variable
        order `sorted((slot, task, nurse))`, found by a sequential fix-to-0-if-
        feasible sweep.  Exact and deterministic; costs one MILP per x variable.
        It is an explicit opt-in because assignment identity is not an estimand;
        the registered formal policy leaves it off and persists no assignment
        hash."""
        order = sorted(range(len(self.xpairs)),
                       key=lambda k: (self.xpairs[k][0], self.xpairs[k][2],
                                      self.xpairs[k][1]))
        fixed, n_probe = {}, 0
        statuses, stages = [], []
        for k in order:
            trial = dict(fixed); trial[k] = 0.0
            # same redundancy argument as stage 4: no Z row (see solve_lex)
            sol, status = self._solve("D", w4=w4, u_eq=u_star,
                                      w_le=w_star, fixed=trial,
                                      allow_infeasible=True,
                                      stage_label="lex5_canonical")
            n_probe += 1
            statuses.append(int(status))
            stages.append("lex5_canonical_probe[{}]".format(n_probe))
            if sol is not None and sol["D"] <= d_star:
                fixed[k] = 0.0
            else:
                fixed[k] = 1.0
        final, status = self._solve("D", w4=w4, u_eq=u_star,
                                    w_le=w_star, fixed=fixed,
                                    allow_infeasible=True,
                                    stage_label="lex5_canonical_final")
        statuses.append(int(status))
        stages.append("lex5_canonical_final")
        if final is None:
            fatal("canonical assignment sweep produced an infeasible fixing")
        if final["D"] != d_star or final["U"] != u_star:
            fatal("canonical assignment sweep changed the lexicographic values",
                  d_star=d_star, d=final["D"], u_star=u_star, u=final["U"])
        key = "".join("1" if fixed[k] == 1.0 else "0" for k in order)
        return {"assignment": dict(final["assignment"]),
                "canonical_x_key": key, "canonical_probes": n_probe,
                "canonical_status_codes": statuses,
                "canonical_stage_names": stages}


# --------------------------------------------------------------------------- #
# one disruption x regime x w4
# --------------------------------------------------------------------------- #
def _aggregate_certificate(results, policy, where, assignment_hash_value=None):
    """Aggregate separable group certificates for one explicitly named role."""
    require(bool(results), "cannot aggregate an empty solver-certificate family",
            where=where)
    require(len(results) == 1,
            "R1b horizon-only certificate aggregation must be a singleton",
            where=where, groups=len(results))
    require(all(r.status_code == 0 and r.proven_optimal for r in results),
            "R1b aggregate certificate contains a non-code-0/non-optimal group",
            where=where,
            statuses=[(r.status_code, r.status_name, r.proven_optimal)
                      for r in results])
    require(all(r.primal is not None and r.dual_bound is not None for r in results),
            "R1b aggregate certificate needs P and B from every group",
            where=where)
    p = sum(float(r.primal) for r in results)
    b = sum(float(r.dual_bound) for r in results)
    c = sum(float(r.deviation_constant_C) for r in results)
    gap, gap_undef = full_objective_gap(p, b, c, where=where)
    nodes = [r.nodes for r in results]
    return SolverResult(
        stage=STAGE, status_code=0, status_name=policy.status_name(0),
        proven_optimal=True, primal=p, dual_bound=b,
        deviation_constant_C=c, gap_solver_raw=results[0].gap_solver_raw,
        gap_full_objective=gap,
        gap_full_objective_undefined_reason=gap_undef,
        stop_reason="proven_optimal",
        wall_time_s=sum(float(r.wall_time_s) for r in results),
        time_limit_s=policy.time_limit_s,
        nodes=(sum(int(n) for n in nodes)
               if all(n is not None for n in nodes) else None),
        assignment_hash=assignment_hash_value,
        message="aggregate of {} separable group certificates: {}".format(
            len(results), where))


def _certificate_columns(prefix, result, exact_certificate, solver_fun_raw):
    """Stable role-prefixed raw evidence plus replayable validation evidence."""
    method = (exact_certificate.get("method")
              if isinstance(exact_certificate, dict) else None)
    proof = (exact_certificate.get("proves_exact_optimality")
             if isinstance(exact_certificate, dict) else None)
    require(
        isinstance(exact_certificate, dict)
        and method in set(EXACT_CERTIFICATE_METHODS) | {RAW_BOUND_METHOD}
        and ((method == STATUS0_LATTICE_METHOD and proof is False)
             or (method != STATUS0_LATTICE_METHOD and proof is True)),
        "cannot persist an absent, failed, or mislabelled objective validation",
        prefix=prefix, certificate=exact_certificate,
        status0_method=STATUS0_LATTICE_METHOD)
    return {
        prefix + "solver_status_code": result.status_code,
        prefix + "solver_status_name": result.status_name,
        prefix + "proven_optimal": bool(result.proven_optimal),
        prefix + "solver_fun_raw": require_finite(
            solver_fun_raw, prefix + "solver_fun_raw"),
        prefix + "obj_incumbent": result.primal,
        prefix + "mip_dual_bound": result.dual_bound,
        prefix + "obj_incumbent_full_objective": (
            None if result.primal is None
            else result.primal + result.deviation_constant_C),
        prefix + "mip_dual_bound_full_objective": (
            None if result.dual_bound is None
            else result.dual_bound + result.deviation_constant_C),
        prefix + "deviation_constant_C": result.deviation_constant_C,
        prefix + "mip_gap_solver_raw": result.gap_solver_raw,
        prefix + "gap_full_objective": result.gap_full_objective,
        prefix + "gap_full_objective_undefined_reason": (
            result.gap_full_objective_undefined_reason),
        prefix + "stop_reason": result.stop_reason,
        prefix + "time_limit_s": result.time_limit_s,
        prefix + "exact_certificate_method": exact_certificate["method"],
        prefix + "exact_certificate_objective_stage": (
            exact_certificate["objective_stage"]),
        prefix + "lattice_step": exact_certificate["step"],
        prefix + "lattice_offset": exact_certificate["offset"],
        prefix + "lattice_bound_ulps": exact_certificate["bound_ulps"],
        prefix + "lattice_boundary_margin": (
            exact_certificate["boundary_margin"]),
        prefix + "lattice_safe_dual_bound_full_objective": (
            exact_certificate["safe_dual_bound_full_objective"]),
        prefix + "lattice_closed_dual_bound_full_objective": (
            exact_certificate["closed_dual_bound_full_objective"]),
        prefix + "lattice_primal_level_full_objective": (
            exact_certificate["primal_level_full_objective"]),
        prefix + "exact_certificate_lower_bound_source": (
            exact_certificate["lower_bound_source"]),
        prefix + "exact_certificate_structural_lower_bound_full_objective": (
            exact_certificate["structural_lower_bound_full_objective"]),
        prefix + "exact_certificate_exact_full_objective_integer": (
            exact_certificate["exact_full_objective_integer"]),
        prefix + "lattice_proves_exact_optimality": bool(
            exact_certificate["proves_exact_optimality"]),
        prefix + "lattice_justification": exact_certificate["justification"],
    }


def _cell_bool(value, *, field="boolean evidence", where="R1b validator"):
    """Parse persisted booleans without accepting truthy lookalikes.

    CSV round-trips may yield bools, 0/1 numerics, or their exact string
    spellings.  Values such as ``yes`` and ``0.9`` are deliberately rejected.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        require(int(value) in (0, 1),
                "persisted boolean evidence is not exactly 0 or 1",
                where=where, field=field, value=value)
        return bool(int(value))
    if isinstance(value, (float, np.floating)):
        numeric = require_finite(value, field, where=where)
        require(numeric in (0.0, 1.0),
                "persisted boolean evidence is not exactly 0 or 1",
                where=where, field=field, value=value)
        return bool(int(numeric))
    token = str(value).strip().lower()
    require(token in ("true", "false", "1", "0"),
            "persisted boolean evidence has a non-canonical spelling",
            where=where, field=field, value=value,
            accepted=["true", "false", "1", "0"])
    return token in ("true", "1")


def _exact_int_cell(value, *, field, where):
    """Parse an integer-valued persisted scalar without truncation."""
    require(not isinstance(value, (bool, np.bool_)),
            "persisted integer evidence may not be boolean",
            where=where, field=field, value=value)
    token = str(value).strip()
    try:
        numeric = Decimal(token)
    except (InvalidOperation, ValueError):
        fatal("persisted integer evidence is not numeric",
              where=where, field=field, value=value)
    require(numeric.is_finite()
            and numeric == numeric.to_integral_value(),
            "persisted integer evidence is not an exact integer",
            where=where, field=field, value=value)
    return int(numeric)


def validate_strict_scalar_columns(
        frame, *, integer_columns=(), boolean_columns=(), where):
    """Reject lossy integer casts and truthy-looking persisted booleans.

    Pandas deliberately offers permissive conversions such as
    ``astype(int)``.  Those conversions are useful for analysis but are not an
    evidence validator: for example, 8.9 can silently become 8.  Run this gate
    before any downstream grouping or comparison that uses such conversions.
    """
    require(isinstance(frame, pd.DataFrame) and not frame.empty,
            "persisted scalar gate requires a non-empty data frame",
            where=where)
    integer_columns = tuple(integer_columns)
    boolean_columns = tuple(boolean_columns)
    require(not (set(integer_columns) & set(boolean_columns)),
            "a persisted field cannot be both integer and boolean evidence",
            where=where,
            overlap=sorted(set(integer_columns) & set(boolean_columns)))
    checked = 0
    for field in integer_columns:
        if field not in frame.columns:
            continue
        for idx, value in frame[field].items():
            _exact_int_cell(
                value, field=field,
                where="{}:row={}".format(where, idx))
            checked += 1
    for field in boolean_columns:
        if field not in frame.columns:
            continue
        for idx, value in frame[field].items():
            _cell_bool(
                value, field=field,
                where="{}:row={}".format(where, idx))
            checked += 1
    require(checked > 0, "persisted scalar gate found no declared fields",
            where=where)
    return checked


def _require_replayed_float(actual, expected, *, field, where):
    if expected is None:
        require(pd.isna(actual),
                "persisted exact-certificate field should be empty",
                where=where, field=field, actual=actual)
        return
    observed = require_finite(actual, field, where=where)
    margin = _ulp_margin(observed, expected)
    require(abs(observed - float(expected)) <= margin,
            "persisted exact-certificate float differs from independent replay",
            where=where, field=field, actual=observed, expected=expected,
            ulp_boundary_margin=margin)


def solver_raw_gap_from_reported_fun(solver_fun_raw, dual_bound, *, where):
    """Recompute SciPy/HiGHS ``mip_gap`` from its own objective scale."""
    fun = require_finite(solver_fun_raw, "solver_fun_raw", where=where)
    bound = require_finite(dual_bound, "mip_dual_bound", where=where)
    signed_gap, undefined = full_objective_gap(
        fun, bound, 0.0, where=where)
    if signed_gap is None:
        require(undefined == "zero_full_objective_denominator"
                and abs(fun - bound) <= _ulp_margin(fun, bound),
                "zero solver objective has a distinct dual bound, so raw mip "
                "gap cannot be validated",
                where=where, solver_fun_raw=fun, dual_bound=bound,
                undefined_reason=undefined)
        return 0.0
    return abs(float(signed_gap))


def validate_solver_diagnostics_raw_gaps(diagnostics, *, where):
    """Validate every successful diagnostic's solver-native gap evidence."""
    required = {
        "rho", "seed", "absent", "method", "w4", "group_index", "nodes",
        "solver_status_code", "objective_stage", "solver_fun_raw",
        "mip_dual_bound", "mip_gap_solver_raw",
        "exact_certificate_method", "lattice_bound_ulps",
        "exact_certificate_lower_bound_source",
        "exact_certificate_structural_lower_bound_full_objective",
        "exact_certificate_exact_full_objective_integer",
        "lattice_proves_exact_optimality"}
    require(required <= set(diagnostics.columns),
            "solver diagnostics lack raw-gap verification fields",
            where=where, missing=sorted(required - set(diagnostics.columns)))
    validate_strict_scalar_columns(
        diagnostics,
        integer_columns=("seed", "absent", "solver_status_code"),
        boolean_columns=("presolve",),
        where="{}:strict-scalars".format(where))
    checked = 0
    for idx, row in diagnostics.iterrows():
        row_where = "{}:diagnostic_row={}".format(where, int(idx))
        for field in ("w4", "group_index", "nodes"):
            value = row[field]
            if not pd.isna(value) and str(value).strip() != "":
                _exact_int_cell(value, field=field, where=row_where)
        if "common_service_preliminary" in diagnostics.columns:
            value = row["common_service_preliminary"]
            if not pd.isna(value) and str(value).strip() != "":
                _cell_bool(
                    value, field="common_service_preliminary", where=row_where)
        code = _exact_int_cell(
            row["solver_status_code"], field="solver_status_code",
            where=row_where)
        if code != 0:
            continue
        stage = str(row["objective_stage"])
        require(stage in ("U", "Z", "W", "D"),
                "successful diagnostic has an unknown objective stage",
                where=where, row=int(idx), objective_stage=stage)
        require(str(row["exact_certificate_method"])
                in EXACT_CERTIFICATE_METHODS,
                "solver diagnostic certificate method differs from the "
                "executable contract",
                where=row_where,
                observed=row["exact_certificate_method"],
                expected=list(EXACT_CERTIFICATE_METHODS))
        method = str(row["exact_certificate_method"])
        source = str(row["exact_certificate_lower_bound_source"])
        floor_int = _exact_int_cell(
            row["exact_certificate_structural_lower_bound_full_objective"],
            field="exact_certificate_structural_lower_bound_full_objective",
            where=row_where)
        exact_int = _exact_int_cell(
            row["exact_certificate_exact_full_objective_integer"],
            field="exact_certificate_exact_full_objective_integer",
            where=row_where)
        require(floor_int >= 0 and exact_int >= floor_int,
                "solver diagnostic structural integers violate the lower bound",
                where=row_where, floor=floor_int, exact=exact_int)
        zero_sources = {
            "U": "nonnegative_unserved_count",
            "W": "nonnegative_canonical_overtime",
            "D": "nonnegative_assignment_deviation",
        }
        if stage in zero_sources:
            expected_structural_source = zero_sources[stage]
            require(floor_int == 0,
                    "solver diagnostic nonweighted floor must be zero",
                    where=row_where, stage=stage, floor=floor_int)
        else:
            expected_structural_source = (
                "fixed_U_common_plus_nonnegative_overtime_and_deviation")
            require(floor_int % int(W5) == 0,
                    "solver diagnostic weighted floor is not a W5 multiple",
                    where=row_where, floor=floor_int, W5=W5)
        if method == STRUCTURAL_LOWER_BOUND_METHOD:
            require(source == expected_structural_source
                    and exact_int == floor_int,
                    "solver diagnostic structural method/source/floor pairing "
                    "differs from the executable contract",
                    where=row_where, stage=stage, source=source,
                    expected_source=expected_structural_source,
                    exact=exact_int, floor=floor_int)
        else:
            require(method == LATTICE_METHOD
                    and source == STATUS0_LOWER_BOUND_SOURCE
                    and exact_int != floor_int,
                    "solver diagnostic status-0 lattice method/source/floor "
                    "pairing "
                    "differs from the executable contract",
                    where=row_where, stage=stage, method=method,
                    source=source, exact=exact_int, floor=floor_int)
        bound_ulps = _exact_int_cell(
            row["lattice_bound_ulps"], field="lattice_bound_ulps",
            where=row_where)
        expected_bound_ulps = (
            STATUS0_LATTICE_BOUND_ULPS
            if method == STATUS0_LATTICE_METHOD else LATTICE_BOUND_ULPS)
        require(bound_ulps == expected_bound_ulps,
                "solver diagnostic lattice-bound ULP label differs from the "
                "executable method",
                where=row_where, observed=bound_ulps,
                expected=expected_bound_ulps, method=method)
        proof = _cell_bool(
            row["lattice_proves_exact_optimality"],
            field="lattice_proves_exact_optimality", where=row_where)
        require(proof is (method == STRUCTURAL_LOWER_BOUND_METHOD),
                "solver diagnostic exact-proof label differs from its method",
                where=row_where, method=method, observed=proof,
                expected=(method == STRUCTURAL_LOWER_BOUND_METHOD))
        expected = solver_raw_gap_from_reported_fun(
            row["solver_fun_raw"], row["mip_dual_bound"], where=row_where)
        observed = require_finite(
            row["mip_gap_solver_raw"], "mip_gap_solver_raw", where=row_where)
        require(abs(observed - expected) <= _ulp_margin(observed, expected),
                "solver diagnostic raw mip gap differs from solver fun/bound replay",
                where=row_where, observed=observed, expected=expected,
                solver_fun_raw=row["solver_fun_raw"],
                mip_dual_bound=row["mip_dual_bound"])
        checked += 1
    require(checked > 0, "solver diagnostics contain no successful gap evidence",
            where=where)
    return checked


def validate_solver_diagnostics_design(diagnostics, design, *, where):
    """Bind formal diagnostics to the independently derived disruption grid."""
    validate_solver_diagnostics_raw_gaps(diagnostics, where=where)
    if not bool(design.get("formal")):
        return len(diagnostics)
    methods = tuple(design["manifest_authorized_methods"])
    w4_values = tuple(int(value) for value in design["manifest_authorized_w4"])
    expected_disruptions = {
        (round(float(rho), 8), int(seed), int(absent))
        for (rho, seed), absents in design["axis"].items()
        for absent in absents}
    grouped = {}
    for idx, row in diagnostics.iterrows():
        row_where = "{}:diagnostic_row={}".format(where, int(idx))
        key = (
            round(require_finite(row["rho"], "rho", where=row_where), 8),
            _exact_int_cell(row["seed"], field="seed", where=row_where),
            _exact_int_cell(row["absent"], field="absent", where=row_where))
        grouped.setdefault(key, []).append((idx, row))
    require(set(grouped) == expected_disruptions,
            "formal diagnostics disruption axis differs from source-derived design",
            where=where,
            missing=sorted(expected_disruptions - set(grouped)),
            extra=sorted(set(grouped) - expected_disruptions))
    expected_main = {
        (method, w4, stage)
        for method in methods for w4 in w4_values
        for stage in ("Z", "W", "D")}
    for key in sorted(expected_disruptions):
        preliminary, main = [], []
        for idx, row in grouped[key]:
            row_where = "{}:diagnostic_row={}".format(where, int(idx))
            flag = row.get("common_service_preliminary", None)
            is_preliminary = (
                False if flag is None or pd.isna(flag)
                or str(flag).strip() == ""
                else _cell_bool(
                    flag, field="common_service_preliminary", where=row_where))
            (preliminary if is_preliminary else main).append((idx, row))
        require(len(preliminary) == len(methods),
                "formal diagnostics lack exactly one preliminary U solve per method",
                where=where, disruption=key, rows=len(preliminary),
                expected=len(methods))
        preliminary_methods = set()
        for idx, row in preliminary:
            row_where = "{}:diagnostic_row={}".format(where, int(idx))
            method = str(row["method"])
            preliminary_methods.add(method)
            require(method in methods
                    and str(row["objective_stage"]) == "U"
                    and (pd.isna(row["w4"]) or str(row["w4"]).strip() == "")
                    and (pd.isna(row["group_index"])
                         or str(row["group_index"]).strip() == ""),
                    "formal preliminary diagnostic has a wrong method/stage/grid label",
                    where=row_where, disruption=key, method=method,
                    objective_stage=row["objective_stage"], w4=row["w4"],
                    group_index=row["group_index"])
        require(preliminary_methods == set(methods),
                "formal preliminary diagnostics do not cover both methods",
                where=where, disruption=key,
                observed=sorted(preliminary_methods), expected=list(methods))
        observed_main = []
        for idx, row in main:
            row_where = "{}:diagnostic_row={}".format(where, int(idx))
            method = str(row["method"])
            w4 = _exact_int_cell(row["w4"], field="w4", where=row_where)
            stage = str(row["objective_stage"])
            group_index = _exact_int_cell(
                row["group_index"], field="group_index", where=row_where)
            require(method in methods and w4 in w4_values
                    and stage in ("Z", "W", "D") and group_index == 1
                    and str(row.get("decomposition", "")) == "horizon",
                    "formal main diagnostic has a wrong method/stage/grid label",
                    where=row_where, disruption=key, method=method, w4=w4,
                    objective_stage=stage, group_index=group_index,
                    decomposition=row.get("decomposition", ""))
            observed_main.append((method, w4, stage))
        require(len(observed_main) == len(expected_main)
                and len(set(observed_main)) == len(observed_main)
                and set(observed_main) == expected_main,
                "formal main diagnostics do not cover the exact method/w4/stage grid",
                where=where, disruption=key, rows=len(observed_main),
                expected_rows=len(expected_main),
                missing=sorted(expected_main - set(observed_main)),
                extra=sorted(set(observed_main) - expected_main))
    return len(diagnostics)


def replay_persisted_exact_certificate(
        row, prefix, objective_stage, *, integral_durations,
        expected_deviation_constant, where):
    """Recompute a proof from raw P/B/C/F and an input-derived premise.

    ``integral_durations`` must come from independently re-parsed H/S inputs;
    the artifact's ``w_level_separation`` label is evidence to compare, never
    an authority for choosing a lattice.
    """
    p = float(row[prefix + "obj_incumbent"])
    b = float(row[prefix + "mip_dual_bound"])
    c = float(row[prefix + "deviation_constant_C"])
    f = float(row[prefix + "solver_fun_raw"])
    persisted_exact_full_int = _exact_int_cell(
        row[prefix + "exact_certificate_exact_full_objective_integer"],
        field=prefix + "exact_certificate_exact_full_objective_integer",
        where=where)
    persisted_structural_floor_int = _exact_int_cell(
        row[prefix +
            "exact_certificate_structural_lower_bound_full_objective"],
        field=(prefix +
               "exact_certificate_structural_lower_bound_full_objective"),
        where=where)
    persisted_lower_bound_source = str(
        row[prefix + "exact_certificate_lower_bound_source"])

    persisted_tie_c = _exact_int_cell(
        row["tie_break_deviation_constant_C"],
        field="tie_break_deviation_constant_C", where=where)
    n_tasks = _exact_int_cell(row["n_tasks"], field="n_tasks", where=where)
    require(0 <= persisted_tie_c <= n_tasks,
            "persisted deviation constant is outside the source task-count bound",
            where=where, deviation_constant=persisted_tie_c,
            n_tasks=n_tasks)
    source_dev_const = _strict_integer_value(
        expected_deviation_constant,
        "source-derived deviation constant", where=where)
    require(persisted_tie_c == source_dev_const,
            "persisted deviation constant differs from re-parsed H/S authority",
            where=where, persisted=persisted_tie_c,
            source_derived=source_dev_const)

    # Rebuild the exact integer and analytic floor from the scientific row.
    # The three persisted certificate fields are evidence to compare, never
    # authority for selecting the structural branch.
    if objective_stage == "U":
        u_field = {
            "umin_A1_": "U_min_A1",
            "umin_A2_": "U_min_A2",
        }.get(prefix, "U")
        exact_full_int = _exact_int_cell(
            row[u_field], field=u_field, where=where)
        structural_floor_int = 0
        structural_input_source = "nonnegative_unserved_count"
    elif objective_stage == "W":
        exact_full_int = _exact_int_cell(
            row["overtime"], field="overtime", where=where)
        structural_floor_int = 0
        structural_input_source = "nonnegative_canonical_overtime"
    elif objective_stage == "D":
        exact_full_int = _exact_int_cell(
            row["deviation"], field="deviation", where=where)
        structural_floor_int = 0
        structural_input_source = "nonnegative_assignment_deviation"
    else:
        require(objective_stage == "Z",
                "unknown objective stage in certificate replay",
                where=where, objective_stage=objective_stage)
        overtime = _exact_int_cell(
            row["overtime"], field="overtime", where=where)
        deviation = _exact_int_cell(
            row["deviation"], field="deviation", where=where)
        unserved = _exact_int_cell(
            row["unserved"], field="unserved", where=where)
        w4_int = _exact_int_cell(row["w4"], field="w4", where=where)
        u_common = _exact_int_cell(
            row["U_common"], field="U_common", where=where)
        require(min(overtime, deviation, unserved, w4_int, u_common) >= 0
                and unserved == u_common,
                "weighted certificate row violates its nonnegative fixed-U "
                "premise",
                where=where, overtime=overtime, deviation=deviation,
                unserved=unserved, w4=w4_int, U_common=u_common)
        exact_full_int = int(W2) * overtime + w4_int * deviation + int(W5) * unserved
        structural_floor_int = int(W5) * u_common
        structural_input_source = (
            "fixed_U_common_plus_nonnegative_overtime_and_deviation")

    require(persisted_exact_full_int == exact_full_int
            and persisted_structural_floor_int == structural_floor_int,
            "persisted structural certificate integers differ from the "
            "scientific-row recomputation",
            where=where, prefix=prefix,
            persisted_exact=persisted_exact_full_int,
            recomputed_exact=exact_full_int,
            persisted_floor=persisted_structural_floor_int,
            recomputed_floor=structural_floor_int)
    expected_c = {
        "weighted_": _exact_int_cell(row["w4"], field="w4", where=where)
                     * source_dev_const,
        "w_refine_": 0,
        "tie_break_": source_dev_const,
        "umin_A1_": 0,
        "umin_A2_": 0,
    }.get(prefix)
    require(expected_c is not None and c == float(expected_c)
            and p == float(exact_full_int - expected_c),
            "raw P/C do not match the source-derived role objective",
            where=where, prefix=prefix, objective_stage=objective_stage,
            P=p, C=c, exact_full_objective_integer=exact_full_int,
            expected_P=exact_full_int - expected_c,
            expected_C=expected_c, source_deviation_constant=source_dev_const)
    expected = exact_objective_certificate(
        objective_stage, p, b, c, f,
        w4=(float(row["w4"]) if objective_stage == "Z" else None),
        integral_durations=integral_durations,
        exact_full_objective_integer=exact_full_int,
        structural_lower_bound_integer=structural_floor_int,
        structural_lower_bound_source=structural_input_source,
        where=where)
    require(str(row[prefix + "exact_certificate_method"])
            == expected["method"]
            and str(row[prefix + "exact_certificate_objective_stage"])
            == expected["objective_stage"]
            and _exact_int_cell(
                row[prefix + "lattice_bound_ulps"],
                field=prefix + "lattice_bound_ulps", where=where)
            == int(expected["bound_ulps"])
            and _cell_bool(
                row[prefix + "lattice_proves_exact_optimality"],
                field=prefix + "lattice_proves_exact_optimality", where=where)
            is bool(expected["proves_exact_optimality"])
            and str(row[prefix + "lattice_justification"])
            == expected["justification"]
            and persisted_lower_bound_source == expected["lower_bound_source"],
            "persisted exact-certificate labels differ from independent replay",
            where=where, prefix=prefix, expected=expected)
    numeric = {
        "lattice_step": expected["step"],
        "lattice_offset": expected["offset"],
        "lattice_boundary_margin": expected["boundary_margin"],
        "lattice_safe_dual_bound_full_objective": (
            expected["safe_dual_bound_full_objective"]),
        "lattice_closed_dual_bound_full_objective": (
            expected["closed_dual_bound_full_objective"]),
        "lattice_primal_level_full_objective": (
            expected["primal_level_full_objective"]),
        "exact_certificate_structural_lower_bound_full_objective": (
            expected["structural_lower_bound_full_objective"]),
        "exact_certificate_exact_full_objective_integer": (
            expected["exact_full_objective_integer"]),
    }
    for suffix, value in numeric.items():
        _require_replayed_float(
            row[prefix + suffix], value, field=prefix + suffix, where=where)
    return expected


def run_disruption(model, w4, u_common, policy, *,
                   canonical_assignment=False, requested_mode="horizon"):
    """Solve one formal-compatible horizon point at fixed common service."""
    require(model.slots, "no future slot in this instance",
            absent=model.absent, t_now=T_NOW)
    require(requested_mode in DECOMPOSITION_MODES, "unknown requested mode",
            mode=requested_mode, authorized_modes=DECOMPOSITION_MODES)
    s = model.solve_lex(
        w4, u_eq=u_common, canonical_assignment=canonical_assignment)
    stage_names = s["lex_stage_chain"].split("|")
    status_codes = [int(v) for v in s["status_chain"].split("|")]
    require(len(stage_names) == len(status_codes),
            "R1b actual stage/status chain lengths differ",
            slots=list(model.slots), stage_names=stage_names,
            status_codes=status_codes)
    complete_group_chains = [{
        "group_index": 1, "slots": list(model.slots),
        "stage_names": stage_names, "status_codes": status_codes}]
    persisted_assignment_hash = (
        assignment_hash(s["assignment"]) if canonical_assignment else None)
    weighted_certificate = _aggregate_certificate(
        [s["primary_weighted_solver"]], policy,
        where=f"{STAGE}:horizon_lex2_weighted_Z_at_U_common")
    w_refine_certificate = _aggregate_certificate(
        [s["w_refine_solver"]], policy,
        where=f"{STAGE}:horizon_lex3_minW_at_Zstar_U_common")
    tie_break_certificate = _aggregate_certificate(
        [s["final_d_tie_break_solver"]], policy,
        where=f"{STAGE}:horizon_lex4_final_D",
        assignment_hash_value=persisted_assignment_hash)
    out = {
        "deviation": int(s["D"]), "overtime": float(s["W"]),
        "unserved": int(s["U"]), "U": int(s["U"]),
        "n_tasks": int(s["n_tasks"]),
        "weighted_objective_Z": (
            float(W2) * float(s["W"]) + float(w4) * int(s["D"])
            + float(W5) * int(s["U"])),
        "solver_status_chain": s["status_chain"],
        "lex_stage_chain": s["lex_stage_chain"],
        "solver_group_complete_status_chains_json": json.dumps(
            complete_group_chains, sort_keys=True, separators=(",", ":")),
        "decomposition_groups": 1,
        "requested_decomposition": requested_mode,
        "horizon_assignment_hash": persisted_assignment_hash,
        "canonical_assignment": bool(canonical_assignment),
        "assignment_hash_evidence": (
            "canonical_lexicographic_assignment"
            if canonical_assignment else "disabled_not_cross_run_evidence"),
        "solver_certificate_role": FINAL_TIE_BREAK_ROLE,
        "solver_evidence_scope": "single_horizon_lex4_final_D_solve",
        "solver_group_status_codes": str(
            s["final_d_tie_break_solver"].status_code),
        "weighted_solver_certificate_role": PRIMARY_WEIGHTED_ROLE,
        "weighted_solver_evidence_scope": (
            "single_horizon_lex2_primary_weighted_Z_at_U_common_solve"),
        "weighted_group_status_codes": str(
            s["primary_weighted_solver"].status_code),
        "w_refine_solver_certificate_role": W_REFINE_ROLE,
        "w_refine_solver_evidence_scope": (
            "single_horizon_lex3_minW_at_Zstar_U_common_solve"),
        "w_refine_group_status_codes": str(s["w_refine_solver"].status_code),
        "tie_break_solver_certificate_role": FINAL_TIE_BREAK_ROLE,
        "tie_break_solver_evidence_scope": "single_horizon_lex4_final_D_solve",
        "tie_break_group_status_codes": str(
            s["final_d_tie_break_solver"].status_code),
        "w_level_separation": (
            "integral_durations" if model.integral_durations else "unknown"),
        "_diagnostics": [dict(rec, group_index=1, decomposition="horizon")
                         for rec in model.diagnostics],
    }
    # The generic solver block is retained for schema compatibility, but its
    # role is explicitly final-D tie-break only. It is not the weighted-Z proof.
    out.update(tie_break_certificate.as_canonical_row())
    out["solver_fun_raw"] = require_finite(
        s["final_d_tie_break_solver_fun_raw"], "solver_fun_raw")
    out.update(_certificate_columns(
        "weighted_", weighted_certificate,
        s["primary_weighted_exact_certificate"],
        s["primary_weighted_solver_fun_raw"]))
    out.update(_certificate_columns(
        "w_refine_", w_refine_certificate,
        s["w_refine_exact_certificate"],
        s["w_refine_solver_fun_raw"]))
    out.update(_certificate_columns(
        "tie_break_", tie_break_certificate,
        s["final_d_tie_break_exact_certificate"],
        s["final_d_tie_break_solver_fun_raw"]))
    return out


# --------------------------------------------------------------------------- #
# summary + per-cell provenance
# --------------------------------------------------------------------------- #
def _rowkey(r):
    return (f"rho={float(r['rho']):.4f}|seed={int(r['seed'])}|absent={int(r['absent'])}"
            f"|method={r['method']}|w4={int(r['w4'])}")


def _cols(raw):
    return {k: find_col(raw.columns, k) for k in
            ("rho", "seed", "absent", "method", "w4", "deviation", "overtime",
             "unserved")}


def _groups(g, c):
    """The (seed, absent) disruption groups of a per-rho slice, w4-sorted.

    A group that does not carry the full w4 sweep is a REQ-2 violation, not a
    group to be skipped.
    """
    out = []
    for _, gg in g.groupby([c["seed"], c["absent"]]):
        gg = gg.sort_values(c["w4"])
        require(len(gg) == len(W4_SWEEP),
                "a disruption is missing w4 cells — REQ-2 forbids a silently "
                "short group", got=len(gg), want=len(W4_SWEEP),
                keys=sorted(set(gg[c["w4"]])))
        out.append(gg)
    require(bool(out), "no disruption group in this slice")
    return out


def _pct_constant(g, c, columns):
    grps = _groups(g, c)
    hit = sum(1 for gg in grps
              if all(gg[col].nunique() == 1 for col in columns))
    return 100.0 * hit / len(grps)


def _mean_of_group_means(g, c, column):
    return float(np.mean([gg[column].mean() for gg in _groups(g, c)]))


def _cos_values(g, c):
    vals = []
    for gg in _groups(g, c):
        lo = gg[gg[c["w4"]] == min(W4_SWEEP)][c["overtime"]].to_numpy(dtype=float)
        hi = gg[gg[c["w4"]] == max(W4_SWEEP)][c["overtime"]].to_numpy(dtype=float)
        require(len(lo) == 1 and len(hi) == 1,
                "an A2 group lacks the w4 endpoints", n_lo=len(lo), n_hi=len(hi))
        vals.append(float(hi[0] - lo[0]))
    return vals


def _dev_monotone(g, c):
    grps = _groups(g, c)
    mono = 0
    for gg in grps:
        d = gg[c["deviation"]].to_numpy(dtype=float)
        mono += int(all(d[i] >= d[i + 1] - 1e-12 for i in range(len(d) - 1)))
    return f"{mono}/{len(grps)}"


#: One recipe per summary cell.  `method` is the row selector, so the citation set
#: of a cell is exactly the rows its recipe consumes; `agg` and `src_col` are what
#: the provenance sidecar persists.  `build_summary` and `verify_provenance` apply
#: the SAME recipe, so this checks that the cell is derivable from the cited rows —
#: it is a citation-completeness and mutation check, not an independent
#: re-derivation of the statistic.
CELL_RECIPES = {
    "disruptions": dict(
        method=None, agg="ngroups(seed,absent)", src_col="absent",
        fn=lambda g, c: int(g.groupby([c["seed"], c["absent"]]).ngroups)),
    "A1_fully_inert_pct": dict(
        method="A1", agg="pct_groups_with_constant_deviation_and_overtime",
        src_col="deviation,overtime",
        fn=lambda g, c: _pct_constant(g, c, [c["deviation"], c["overtime"]])),
    "A1_dev_inert_pct": dict(
        method="A1", agg="pct_groups_with_constant_deviation",
        src_col="deviation",
        fn=lambda g, c: _pct_constant(g, c, [c["deviation"]])),
    "A1_unserved_mean": dict(
        method="A1", agg="mean_of_group_means", src_col="unserved",
        fn=lambda g, c: _mean_of_group_means(g, c, c["unserved"])),
    "A2_fully_inert_pct": dict(
        method="A2", agg="pct_groups_with_constant_deviation_and_overtime",
        src_col="deviation,overtime",
        fn=lambda g, c: _pct_constant(g, c, [c["deviation"], c["overtime"]])),
    "A2_dev_inert_pct": dict(
        method="A2", agg="pct_groups_with_constant_deviation",
        src_col="deviation",
        fn=lambda g, c: _pct_constant(g, c, [c["deviation"]])),
    "A2_unserved_mean": dict(
        method="A2", agg="mean_of_group_means", src_col="unserved",
        fn=lambda g, c: _mean_of_group_means(g, c, c["unserved"])),
    "A2_dev_monotone": dict(
        method="A2", agg="count_groups_nonincreasing_in_w4", src_col="deviation",
        fn=lambda g, c: _dev_monotone(g, c)),
    "A2_cost_of_stability_min": dict(
        method="A2", agg="mean(overtime@w4max - overtime@w4min)",
        src_col="overtime",
        fn=lambda g, c: float(np.mean(_cos_values(g, c)))),
}

#: Fixed boundary label, not a measured cell.  R1b computes no confidence interval.
CI95_METHOD_NOTE = ("NOT_COMPUTED_IN_R1b — disruption-IID inference removed; "
                    "superseded by R5 cluster-level inference (REQ-10)")


def _slice(raw, c, rho, method):
    g = raw[np.isclose(raw[c["rho"]].astype(float), float(rho))]
    if method is not None:
        g = g[g[c["method"]] == method]
    return g


def build_summary(raw, source_name="e4_raw.csv", source_sha256=None):
    """Rebuild the per-rho summary from the raw rows ONLY, emitting long-format
    per-cell provenance `(source_file, source_row_key, source_column, agg)`."""
    c = _cols(raw)
    rows, prov = [], []
    for rho in sorted(raw[c["rho"]].unique()):
        rec = {"rho": float(rho)}
        for col, spec in CELL_RECIPES.items():
            g = _slice(raw, c, rho, spec["method"])
            require(not g.empty, "no raw row for a summary cell", rho=rho,
                    column=col, method=spec["method"])
            rec[col] = spec["fn"](g, c)
            for _, r in g.iterrows():
                prov.append({"target_artifact": "e4_summary.csv",
                             "target_row_key": f"rho={float(rho):.4f}",
                             "target_column": col, "agg": spec["agg"],
                             "source_file": source_name,
                             "source_sha256": source_sha256 or "",
                             "source_row_key": _rowkey(r),
                             "source_column": spec["src_col"]})
        rec["ci95_method"] = CI95_METHOD_NOTE
        rows.append(rec)
    return pd.DataFrame(rows), pd.DataFrame(prov)


def verify_provenance(summary, raw, prov, source_name="e4_raw.csv",
                      expected_source_sha256=None):
    """Recompute every cited summary cell from EXACTLY the cited raw rows.

    What this does and does not establish, stated precisely: it applies the same
    recipe to the cited subset, so it detects an incomplete or wrong citation set,
    a mutated `source_row_key`, a cell that no longer matches its sources, and a
    foreign source file.  It is NOT an independent re-derivation of the statistic.
    Negative control: mutate one `source_row_key` and this must fail.
    """
    c = _cols(raw)
    raw = raw.copy()
    raw["_key"] = [_rowkey(r) for _, r in raw.iterrows()]
    by_key = {}
    for i, k in enumerate(raw["_key"]):
        require(k not in by_key, "duplicate raw row key", key=k)
        by_key[k] = i
    missing = sorted(set(prov["source_row_key"]) - set(by_key))
    require(not missing,
            "provenance cites a source row key that does not exist in the raw "
            "artifact", n=len(missing), sample=missing[:5], source_file=source_name)
    require(set(prov["source_file"]) == {source_name},
            "provenance cites a foreign source file",
            cited=sorted(set(prov["source_file"])), expected=source_name)
    if expected_source_sha256 is not None:
        require("source_sha256" in prov.columns,
                "provenance lacks source_sha256 for the consumed landed raw bytes")
        require(set(str(v) for v in prov["source_sha256"])
                == {str(expected_source_sha256)},
                "provenance source_sha256 differs from the consumed landed "
                "e4_raw.csv bytes",
                cited=sorted(set(str(v) for v in prov["source_sha256"])),
                expected=expected_source_sha256)
    require(set(prov["target_column"]) == set(CELL_RECIPES),
            "provenance does not cover every summary cell",
            uncited=sorted(set(CELL_RECIPES) - set(prov["target_column"])),
            unknown=sorted(set(prov["target_column"]) - set(CELL_RECIPES)))

    checked = 0
    for (tkey, tcol), g in prov.groupby(["target_row_key", "target_column"]):
        rho = float(str(tkey).split("=", 1)[1])
        cited = raw.iloc[[by_key[k] for k in g["source_row_key"]]]
        target = summary[np.isclose(summary["rho"].astype(float), rho)]
        require(len(target) == 1, "provenance target row not found exactly once",
                target_row_key=tkey, n=len(target))
        spec = CELL_RECIPES[tcol]
        # the citation set must be exactly the rows the recipe consumes
        expect_keys = set(_slice(raw, c, rho, spec["method"])["_key"])
        require(set(g["source_row_key"]) == expect_keys,
                "citation set does not match the rows the recipe consumes",
                target_row_key=tkey, target_column=tcol,
                cited_only=sorted(set(g["source_row_key"]) - expect_keys)[:5],
                missing=sorted(expect_keys - set(g["source_row_key"]))[:5])
        got = spec["fn"](cited, c)
        want = target[tcol].iloc[0]
        same = (str(got) == str(want) if isinstance(got, str)
                else bool(np.isclose(float(got), float(want), rtol=0, atol=1e-12)))
        require(same, "provenance recomputation disagrees with the summary cell",
                target_row_key=tkey, target_column=tcol, recomputed=got,
                in_summary=want, n_cited_rows=len(cited))
        checked += 1
    require(set(summary["ci95_method"].astype(str)) == {CI95_METHOD_NOTE},
            "the ci95_method label was altered; disruption-IID inference is "
            "removed and superseded by R5")
    return {"cells_checked": checked, "provenance_records": len(prov)}



# --------------------------------------------------------------------------- #
# validator  (REQ-3 / section 4 global validation contract, + REQ-6 test 4)
# --------------------------------------------------------------------------- #
def assert_no_dominated_w4_point(raw):
    """No w4 point may be dominated by another w4 point of the same disruption.

    Domination on (deviation, overtime, unserved), all minimised.  This is the
    structural symptom of an unbroken tie at w4 = 0 that REQ-6 was written about.
    It is a FATAL acceptance check, not a repair: if it fires during R1b the run
    must STOP and the group must be reported, never patched into compliance.
    """
    c = {k: find_col(raw.columns, k) for k in
         ("rho", "seed", "absent", "method", "w4", "deviation", "overtime",
          "unserved")}
    viol = []
    for key, g in raw.groupby([c["rho"], c["seed"], c["absent"], c["method"]]):
        pts = [(int(r[c["w4"]]), float(r[c["deviation"]]), float(r[c["overtime"]]),
                float(r[c["unserved"]])) for _, r in g.iterrows()]
        for a in pts:
            for b in pts:
                if a[0] == b[0]:
                    continue
                le = (b[1] <= a[1] + 1e-12 and b[2] <= a[2] + 1e-12
                      and b[3] <= a[3] + 1e-12)
                lt = (b[1] < a[1] - 1e-12 or b[2] < a[2] - 1e-12
                      or b[3] < a[3] - 1e-12)
                if le and lt:
                    viol.append({"group": key, "dominated_w4": a[0],
                                 "dominated_point": a[1:], "by_w4": b[0],
                                 "by_point": b[1:]})
    if viol:
        fatal("a w4 point is dominated by another w4 point of the same "
              "disruption — the tie-break did not remove the degeneracy; STOP "
              "and report, do not patch", n=len(viol), sample=viol[:4])
    return {"groups_checked": int(raw.groupby(
        [c["rho"], c["seed"], c["absent"], c["method"]]).ngroups)}


# --------------------------------------------------------------------------- #
# design grid — the INDEPENDENT coverage oracle
# --------------------------------------------------------------------------- #
def build_design(axis, inputs, contract=None, *, scenario_pool_counts=None,
                 scenario_lattice_preconditions, deviation_constants,
                 formal=False, snap_id=None):
    """`axis`: {(rho, seed): [absent, ...]} re-derived from the committed plans.

    Independence: the disruption axis comes from the committed `S_*.csv` via the
    pipeline's own rule, never from the artifact under validation.  The regime and
    w4 axes are design constants.  Any product cell that does not exist (an
    instance with fewer eligible future-task holders than another) is DECLARED as
    a shortfall up front, which is how the plan requires the 176-vs-180 shortfall
    to be handled.
    """
    if contract is None:
        contract = load_design_contract()
    authorized_methods = list(contract["design_grid"]["methods"])
    authorized_w4 = list(contract["design_grid"]["w4"])
    cells, rhos, seeds, absents = set(), set(), set(), set()
    for (rho, seed), acts in axis.items():
        rhos.add(rho); seeds.add(seed)
        for a in acts:
            absents.add(a)
            for m in authorized_methods:
                for w in authorized_w4:
                    cells.add((rho, seed, a, m, w))
    axes = {"rho": sorted(rhos), "seed": sorted(seeds), "absent": sorted(absents),
            "method": authorized_methods, "w4": authorized_w4}
    prov = ("disruption axis re-derived from the committed S_*.csv with the "
            "pipeline's own active-future-task-holder rule; method and w4 axes are "
            "authorized by the exact-asserted R1b design manifest "
            "(sha256={}); NEVER read back from the artifact under validation"
            .format(sha256_file(DESIGN_MANIFEST)))
    probe = DesignGrid(name="e4_raw", axes={k: list(v) for k, v in axes.items()},
                       provenance=prov)
    full = probe.full_tuples()
    want = {probe._normalize_tuple(c) for c in cells}
    stray = sorted(want - full, key=repr)
    require(not stray, "designed cell outside the declared axes",
            offending=stray[:10])
    grid = DesignGrid(name="e4_raw", axes={k: list(v) for k, v in axes.items()},
                      declared_shortfalls=sorted(full - want, key=repr),
                      provenance=prov)
    pool_counts = dict(scenario_pool_counts or {})
    require(set(pool_counts) == set(axis),
            "scenario caregiver-pool count keys differ from the scenario axis",
            missing=sorted(set(axis) - set(pool_counts), key=repr),
            extra=sorted(set(pool_counts) - set(axis), key=repr))
    for key, value in pool_counts.items():
        require(int(value) > 0,
                "scenario caregiver-pool count must be positive",
                scenario=key, active_caregiver_count=value)
    premise_map = dict(scenario_lattice_preconditions)
    expected_premise_keys = {
        _scenario_key(rho, seed) for rho, seed in axis}
    require(set(premise_map) == expected_premise_keys,
            "source-derived lattice-premise keys differ from the scenario axis",
            missing=sorted(expected_premise_keys - set(premise_map)),
            extra=sorted(set(premise_map) - expected_premise_keys))
    for key, premise in premise_map.items():
        require(isinstance(premise, dict)
                and isinstance(premise.get("integral_lattice_data"), bool),
                "scenario lattice premise is absent or not a strict boolean",
                scenario=key, premise=premise)
        if formal:
            require(premise["integral_lattice_data"] is True,
                    "formal design contains a non-integral W/Z lattice premise",
                    scenario=key, premise=premise)
    deviation_map = dict(deviation_constants)
    expected_disruption_keys = {
        (rho, seed, int(absent))
        for (rho, seed), acts in axis.items() for absent in acts}
    require(set(deviation_map) == expected_disruption_keys,
            "source-derived deviation-constant keys differ from the disruption axis",
            missing=sorted(expected_disruption_keys - set(deviation_map), key=repr),
            extra=sorted(set(deviation_map) - expected_disruption_keys, key=repr))
    for key, value in deviation_map.items():
        require(float(value).is_integer() and int(value) >= 0,
                "source-derived deviation constant is not a non-negative integer",
                disruption=key, deviation_constant=value)
    return {
        "axis": axis, "grid": grid, "inputs": list(inputs),
        "scenario_pool_counts": {
            key: int(value) for key, value in pool_counts.items()},
        "scenario_lattice_preconditions": premise_map,
        "deviation_constants": {
            key: int(value) for key, value in deviation_map.items()},
        "formal": bool(formal), "snap_id": snap_id or "",
        "manifest_sha256": sha256_file(DESIGN_MANIFEST),
        "manifest_authorized_methods": authorized_methods,
        "manifest_authorized_w4": authorized_w4,
    }


CERTIFICATE_NUMERIC_SUFFIXES = [
    "solver_status_code", "solver_fun_raw", "obj_incumbent", "mip_dual_bound",
    "obj_incumbent_full_objective", "mip_dual_bound_full_objective",
    "deviation_constant_C", "mip_gap_solver_raw", "gap_full_objective",
    "time_limit_s",
    "lattice_step", "lattice_offset", "lattice_bound_ulps",
    "lattice_boundary_margin",
    "lattice_safe_dual_bound_full_objective",
    "lattice_closed_dual_bound_full_objective",
    "lattice_primal_level_full_objective",
    "exact_certificate_structural_lower_bound_full_objective",
    "exact_certificate_exact_full_objective_integer"]
EXACT_CERTIFICATE_REQUIRED_SUFFIXES = [
    "exact_certificate_method", "exact_certificate_objective_stage",
    "exact_certificate_lower_bound_source",
    "lattice_proves_exact_optimality", "lattice_justification"]
RAW_NUMERIC = [
    "w4", "deviation", "overtime", "unserved", "U", "U_min_A1",
    "U_min_A2", "U_common", "n_tasks", "lex_tol",
    "decomposition_groups", "weighted_objective_Z",
    "eligible_disruptions_in_scenario", "active_caregiver_count",
    "rho_target", "rho_global", "rho_used_mean", "rho_used_max",
    "scenario_N", "scenario_cap", "n_incumbent_pairs_checked",
    "n_tasks_nested_A1_subset_A2", "solver_fun_raw",
] + [
    prefix + suffix
    for prefix in (
        "weighted_", "w_refine_", "tie_break_", "umin_A1_", "umin_A2_")
    for suffix in CERTIFICATE_NUMERIC_SUFFIXES
]

COMMON_SERVICE_COLUMNS = [
    "U_min_A1", "U_min_A2", "U_common", "U", "service_policy",
    "placeable_A1", "placeable_A2", "u_min_solver_status_chain",
    "umin_A1_solver_certificate_role", "umin_A1_solver_evidence_scope",
    "umin_A2_solver_certificate_role", "umin_A2_solver_evidence_scope",
]

MANIFEST_PROVENANCE_COLUMNS = [
    "incumbent_flat_feasible", "eligible_disruption",
    "eligible_disruptions_in_scenario", "absence_rule",
    "active_caregiver_count", "rho_target", "rho_global", "rho_used_mean",
    "rho_used_max", "scenario_N", "scenario_cap", "source_file",
    "src_sha256", "h_sha256", "s_sha256", "generator_file",
    "generator_sha256", "family_runner_file", "family_runner_sha256",
    "manifest_provenance_mode",
]

SHARD_RECEIPT_COLUMNS = [
    "shard_receipt_id", "shard_raw_sha256", "shard_summary_sha256",
    "shard_provenance_sha256", "shard_diagnostics_sha256",
]


def raw_spec(design, policy, snap_id=None, formal=False):
    return StageOutputSpec(
        stage=STAGE, artifact_name="e4_raw.csv",
        primary_key=["rho", "seed", "absent", "method", "w4"],
        grid=design["grid"],
        required_columns=["lex_order", "lex_stage_chain", "solver_status_chain",
                          "decomposition", "requested_decomposition",
                          "w_level_separation",
                          "cap_allowance_rule", "horizon_assignment_hash",
                          "canonical_assignment", "assignment_hash_evidence",
                          "solver_certificate_role", "solver_evidence_scope",
                          "solver_group_status_codes",
                          "weighted_solver_certificate_role",
                          "weighted_solver_evidence_scope",
                          "weighted_group_status_codes",
                          "w_refine_solver_certificate_role",
                          "w_refine_solver_evidence_scope",
                          "w_refine_group_status_codes",
                          "tie_break_solver_certificate_role",
                          "tie_break_solver_evidence_scope",
                          "tie_break_group_status_codes",
                          "solver_group_complete_status_chains_json",
                          "weighted_solver_status_name",
                          "weighted_proven_optimal",
                          "weighted_gap_full_objective_undefined_reason",
                          "weighted_stop_reason",
                          "w_refine_solver_status_name",
                          "w_refine_proven_optimal",
                          "w_refine_gap_full_objective_undefined_reason",
                          "w_refine_stop_reason",
                          "tie_break_solver_status_name",
                          "tie_break_proven_optimal",
                          "tie_break_gap_full_objective_undefined_reason",
                          "tie_break_stop_reason", "formal_run",
                          "absence_cap_mode", "solver_presolve",
                          "solver_policy_manifest_sha256", "snap_id",
                          "umin_A1_solver_status_name",
                          "umin_A1_proven_optimal",
                          "umin_A1_gap_full_objective_undefined_reason",
                          "umin_A1_stop_reason",
                          "umin_A2_solver_status_name",
                          "umin_A2_proven_optimal",
                          "umin_A2_gap_full_objective_undefined_reason",
                          "umin_A2_stop_reason"]
                         + [
                             prefix + suffix
                             for prefix in (
                                 "weighted_", "w_refine_", "tie_break_",
                                 "umin_A1_", "umin_A2_")
                             for suffix in EXACT_CERTIFICATE_REQUIRED_SUFFIXES
                         ]
                         + COMMON_SERVICE_COLUMNS
                         + MANIFEST_PROVENANCE_COLUMNS
                         + (SHARD_RECEIPT_COLUMNS
                            if design.get("shard_receipts_required") else []),
        numeric_columns=RAW_NUMERIC,
        solver_policy=policy, status_column=COL.STATUS_CODE,
        input_paths=design["inputs"], snap_id=snap_id, require_snap=formal,
        nullable_columns=[
            "weighted_gap_full_objective",
            "w_refine_gap_full_objective",
            "tie_break_gap_full_objective",
            "umin_A1_gap_full_objective",
            "umin_A2_gap_full_objective",
            COL.GAP_FULL])


# --------------------------------------------------------------------------- #
# stage-specific checks the generic contract cannot express
# --------------------------------------------------------------------------- #
def extra_checks(raw, summary, prov, design, verbose=True,
                 landed_raw_sha256=None):
    log = []

    def ok(tag, msg):
        log.append(f"[{tag}] OK  {msg}")

    role_prefixes = (
        "weighted_", "w_refine_", "tie_break_", "umin_A1_", "umin_A2_")
    validate_strict_scalar_columns(
        raw,
        integer_columns=(
            "seed", "absent", "w4", "deviation", "unserved", "U", "U_min_A1",
            "U_min_A2", "U_common", "n_tasks", "decomposition_groups",
            "eligible_disruptions_in_scenario", "active_caregiver_count",
            "scenario_N", "n_incumbent_pairs_checked",
            "n_tasks_nested_A1_subset_A2", COL.STATUS_CODE,
            *(prefix + "solver_status_code" for prefix in role_prefixes),
            *(prefix + "lattice_bound_ulps" for prefix in role_prefixes)),
        boolean_columns=(
            "formal_run", "solver_presolve", "incumbent_flat_feasible",
            "eligible_disruption", "placeable_A1", "placeable_A2",
            "canonical_assignment", COL.PROVEN_OPTIMAL,
            *(prefix + "proven_optimal" for prefix in role_prefixes),
            *(prefix + "lattice_proves_exact_optimality"
              for prefix in role_prefixes)),
        where="R1b raw strict scalar pre-gate")
    ok("strict-scalars", "integer and boolean evidence rejects lossy or "
                         "truthy-looking spellings before any cast")

    key3 = [find_col(raw.columns, c) for c in ("rho", "seed", "absent")]
    require(design.get("manifest_sha256") == sha256_file(DESIGN_MANIFEST),
            "[manifest] design was not built from the currently landed R1b manifest",
            design_manifest_sha256=design.get("manifest_sha256"),
            landed_manifest_sha256=sha256_file(DESIGN_MANIFEST))
    authorized_methods = list(design["manifest_authorized_methods"])
    authorized_w4 = list(design["manifest_authorized_w4"])

    # regime x w4 grid complete on every disruption (the generic [coverage] check
    # already covers this via the product grid; this repeats it per disruption so a
    # failure names the disruption)
    for k, g in raw.groupby(key3):
        cells = {(m, int(w)) for m, w in zip(g[find_col(g.columns, "method")],
                                            g[find_col(g.columns, "w4")])}
        want = {(m, w) for m in authorized_methods for w in authorized_w4}
        require(cells == want, "[coverage] regime x w4 grid incomplete", key=k,
                missing=sorted(want - cells), extra=sorted(cells - want))
    ok("coverage", "regime x w4 grid complete "
                   f"({len(authorized_methods) * len(authorized_w4)} cells) "
                   "on every disruption")

    # [slots]: the horizon task count must be constant across the cells of one
    # disruption and must equal the independently re-derived count
    for k, g in raw.groupby(key3):
        vals = set(int(v) for v in g[find_col(g.columns, "n_tasks")])
        require(len(vals) == 1,
                "[slots] the horizon task count differs across cells of one "
                "disruption — a slot was dropped somewhere", key=k, values=vals)
        want = design["n_tasks_future"][tuple(k)]
        require(vals == {want},
                "[slots] horizon task count disagrees with the count re-derived "
                "from the committed plan", key=k, in_artifact=vals, re_derived=want)
    ok("slots", "every disruption covers exactly the task set re-derived from the "
                "committed plan")

    # Common-service construction and horizon-only execution are load-bearing.
    for k, g in raw.groupby(key3):
        common = set(int(v) for v in g["U_common"])
        u1 = set(int(v) for v in g["U_min_A1"])
        u2 = set(int(v) for v in g["U_min_A2"])
        achieved = set(int(v) for v in g["U"])
        require(len(common) == len(u1) == len(u2) == 1,
                "[service] preliminary/common service values vary across cells",
                key=k, U_common=sorted(common), U_min_A1=sorted(u1),
                U_min_A2=sorted(u2))
        require(common == u1 and achieved == common,
                "[service] every row must satisfy U == U_common == U_min_A1",
                key=k, U=sorted(achieved), U_common=sorted(common),
                U_min_A1=sorted(u1))
        require(next(iter(u2)) <= next(iter(u1)),
                "[service] U_min_A2 > U_min_A1 contradicts checked A1 subset A2",
                key=k, U_min_A1=next(iter(u1)), U_min_A2=next(iter(u2)))
        require(set(g["service_policy"].astype(str))
                == {"U_common=U_min_A1; equality in A1 and A2 for every w4"},
                "[service] service policy label drifted", key=k)
        require(set(g["decomposition"].astype(str)) == {"horizon"}
                and set(g["decomposition_groups"].astype(int)) == {1},
                "[horizon] an emitted R1b row is not a single horizon solve",
                key=k,
                decomposition=sorted(set(g["decomposition"].astype(str))),
                groups=sorted(set(g["decomposition_groups"].astype(int))))
        require(set(g["solver_presolve"].astype(str).str.lower()) == {"false"},
                "[solver] an R1b row does not record presolve=false", key=k)
    ok("service", f"U == U_common == U_min_A1 on all {len(raw)} rows; "
                  "U_min_A2 <= U_min_A1 for every disruption")
    ok("horizon", "every emitted row is one horizon solve at common service")

    formal_expected = bool(design.get("formal"))
    observed_formal = set(raw["formal_run"].astype(str).str.lower())
    require(observed_formal == ({"true"} if formal_expected else {"false"}),
            "[formal] row formal_run flag differs from execution mode",
            expected=formal_expected, observed=sorted(observed_formal))
    if formal_expected:
        require(set(raw["absence_cap_mode"].astype(str)) == {"all_active_no_cap"}
                and set(raw["requested_decomposition"].astype(str))
                == {FORMAL_DECOMPOSITION},
                "[formal] formal R1b must be horizon-only and all-active",
                absence_cap_mode=sorted(set(raw["absence_cap_mode"].astype(str))),
                requested_mode=sorted(set(
                    raw["requested_decomposition"].astype(str))))
    premise_map = design.get("scenario_lattice_preconditions")
    require(isinstance(premise_map, dict),
            "[lattice-premise] design lacks source-derived scenario premises")
    for (rho, seed), acts in design["axis"].items():
        g = raw[(np.isclose(raw["rho"].astype(float), float(rho)))
                & (raw["seed"].astype(int) == int(seed))]
        premise = premise_map.get(_scenario_key(rho, seed))
        require(isinstance(premise, dict)
                and isinstance(premise.get("integral_lattice_data"), bool),
                "[lattice-premise] scenario has no strict source-derived premise",
                rho=rho, seed=seed, premise=premise)
        expected_label = (
            "integral_durations"
            if premise["integral_lattice_data"] else "unknown")
        require(set(g["w_level_separation"].astype(str)) == {expected_label}
                and set(g["h_sha256"].astype(str))
                == {str(premise["h_sha256"])}
                and set(g["s_sha256"].astype(str))
                == {str(premise["s_sha256"])},
                "[lattice-premise] artifact label or H/S receipt differs from "
                "the independently re-parsed source premise",
                rho=rho, seed=seed, expected_label=expected_label,
                observed_labels=sorted(set(
                    g["w_level_separation"].astype(str))),
                premise=premise)
        if formal_expected:
            require(premise["integral_lattice_data"] is True,
                    "[lattice-premise] formal row family lacks an authorized "
                    "integer W/Z lattice",
                    rho=rho, seed=seed, premise=premise)
        pool_count = int(design["scenario_pool_counts"][(rho, seed)])
        require(set(g["absent"].astype(int)) == set(int(a) for a in acts),
                "[provenance] emitted disruption axis differs from independent axis",
                rho=rho, seed=seed)
        require(set(g["eligible_disruptions_in_scenario"].astype(int))
                == {len(acts)},
                "[provenance] eligible_disruptions_in_scenario does not match "
                "the active-future-task-holder axis",
                rho=rho, seed=seed, expected=len(acts))
        require(set(g["active_caregiver_count"].astype(int)) == {pool_count}
                and set(g["scenario_N"].astype(int)) == {pool_count},
                "[provenance] active_caregiver_count/scenario_N does not match "
                "the manifest-or-parsed caregiver pool",
                rho=rho, seed=seed, expected=pool_count,
                eligible_disruptions=len(acts))
    ok("provenance-meta", "raw rows retain the scenario source fields and the "
                          "distinct caregiver-pool and independently re-derived "
                          "eligible-disruption counts")
    ok("lattice-premise", "every artifact label is checked against family-wide "
                           "rehashed and re-parsed H/S inputs")
    if design.get("shard_receipts_required"):
        receipt_by_rho = {
            round(float(r["rho"]), 8): r
            for r in design.get("shard_receipts", [])}
        require(set(receipt_by_rho) == set(FORMAL_RHOS),
                "[shards] merge design lacks the six registered receipts",
                observed=sorted(receipt_by_rho))
        for rho, g in raw.groupby("rho"):
            receipt = receipt_by_rho.get(round(float(rho), 8))
            require(receipt is not None,
                    "[shards] raw rho has no input-shard receipt", rho=rho)
            for column in SHARD_RECEIPT_COLUMNS:
                require(set(g[column].astype(str)) == {str(receipt[column])},
                        "[shards] raw receipt column differs from verified shard",
                        rho=rho, column=column,
                        expected=receipt[column],
                        observed=sorted(set(g[column].astype(str))))
        ok("shards", "every combined raw row retains the verified SHA256 "
                     "receipts of its source shard family")

    # Five separate certificates are mandatory:
    #   umin_A1/A2 = independent preliminary service minima;
    #   weighted_*  = lex2 primary weighted Z at U_common;
    #   w_refine_*  = lex3 minimum W at fixed Zstar and U_common;
    #   tie_break_* = lex4 final D refinement.
    # The generic solver block is explicitly the latter for compatibility and
    # may never be described as the weighted-objective certificate.
    for idx, row in raw.iterrows():
        trusted_integrality = premise_map[
            _scenario_key(row["rho"], row["seed"])][
                "integral_lattice_data"]
        deviation_key = (
            float(row["rho"]),
            _exact_int_cell(row["seed"], field="seed",
                            where="R1b:row={}".format(int(idx))),
            _exact_int_cell(row["absent"], field="absent",
                            where="R1b:row={}".format(int(idx))),
        )
        require(deviation_key in design["deviation_constants"],
                "source-derived deviation constant is absent for raw row",
                row=int(idx), disruption=deviation_key)
        source_deviation_constant = design["deviation_constants"][deviation_key]
        n_groups = _exact_int_cell(
            row["decomposition_groups"], field="decomposition_groups",
            where="R1b:row={}".format(int(idx)))
        def _status_chain(field):
            return [
                _exact_int_cell(
                    token, field=field,
                    where="R1b:row={}:status-chain".format(int(idx)))
                for token in str(row[field]).split("|") if token != ""]

        d_codes = _status_chain("tie_break_group_status_codes")
        weighted_codes = _status_chain("weighted_group_status_codes")
        w_refine_codes = _status_chain("w_refine_group_status_codes")
        legacy_codes = _status_chain("solver_group_status_codes")
        require(len(d_codes) == n_groups and set(d_codes) == {0}
                and weighted_codes == d_codes
                and w_refine_codes == d_codes
                and legacy_codes == d_codes,
                "[solver-aggregate] weighted/tie-break group status evidence is "
                "incomplete, non-optimal, or cross-role inconsistent",
                row=int(idx), groups=n_groups,
                weighted_status_codes=weighted_codes,
                w_refine_status_codes=w_refine_codes,
                tie_break_status_codes=d_codes,
                generic_status_codes=legacy_codes)
        require(row["weighted_solver_certificate_role"] == PRIMARY_WEIGHTED_ROLE
                and row["weighted_solver_evidence_scope"]
                == "single_horizon_lex2_primary_weighted_Z_at_U_common_solve"
                and row["w_refine_solver_certificate_role"] == W_REFINE_ROLE
                and row["w_refine_solver_evidence_scope"]
                == "single_horizon_lex3_minW_at_Zstar_U_common_solve"
                and row["tie_break_solver_certificate_role"]
                == FINAL_TIE_BREAK_ROLE
                and row["tie_break_solver_evidence_scope"]
                == "single_horizon_lex4_final_D_solve"
                and row["solver_certificate_role"] == FINAL_TIE_BREAK_ROLE
                and row["solver_evidence_scope"]
                == "single_horizon_lex4_final_D_solve",
                "[solver-aggregate] certificate role/scope labels are ambiguous",
                row=int(idx),
                weighted_role=row["weighted_solver_certificate_role"],
                w_refine_role=row["w_refine_solver_certificate_role"],
                tie_break_role=row["tie_break_solver_certificate_role"],
                generic_role=row["solver_certificate_role"])

        try:
            group_chains = json.loads(
                str(row["solver_group_complete_status_chains_json"]))
        except (TypeError, ValueError) as exc:
            fatal("[solver-chain] group chain is not valid JSON",
                  row=int(idx), error=str(exc))
        require(isinstance(group_chains, list)
                and len(group_chains) == n_groups,
                "[solver-chain] complete group chain count differs from "
                "decomposition_groups",
                row=int(idx), groups=n_groups, chains=group_chains)
        canonical_enabled = _cell_bool(
            row["canonical_assignment"], field="canonical_assignment",
            where="R1b:row={}:solver-chain".format(int(idx)))
        rebuilt_status_chains, rebuilt_stage_chains = [], []
        for expected_index, chain in enumerate(group_chains, 1):
            require(isinstance(chain, dict),
                    "[solver-chain] group chain entry must be an object",
                    row=int(idx), group_index=expected_index, chain=chain)
            stages = list(chain.get("stage_names", []))
            codes = [
                _exact_int_cell(
                    value, field="solver_group_status_code",
                    where="R1b:row={}:group={}".format(
                        int(idx), expected_index))
                for value in chain.get("status_codes", [])]
            observed_group_index = _exact_int_cell(
                chain.get("group_index"), field="solver_group_index",
                where="R1b:row={}:group={}".format(
                    int(idx), expected_index))
            require(observed_group_index == expected_index
                    and len(stages) == len(codes)
                    and tuple(stages[:3]) == EXECUTABLE_MAIN_STAGE_NAMES
                    and codes[:3] == [0, 0, 0],
                    "[solver-chain] a group does not contain the complete "
                    "code-0 lex2/lex3/lex4 common-service chain",
                    row=int(idx), expected_group_index=expected_index,
                    chain=chain)
            if canonical_enabled:
                require(len(stages) > 3
                        and all(name.startswith("lex5_canonical")
                                for name in stages[3:])
                        and set(codes[3:]) <= {0, 2}
                        and codes[-1] == 0,
                        "[solver-chain] enabled canonical stage chain is "
                        "incomplete or has an undeclared probe status",
                        row=int(idx), chain=chain)
            else:
                require(len(stages) == 3 and len(codes) == 3,
                        "[solver-chain] canonical assignment is off but hidden "
                        "extra solve stages appear in the formal chain",
                        row=int(idx), chain=chain)
            rebuilt_status_chains.append("|".join(str(v) for v in codes))
            rebuilt_stage_chains.append("|".join(stages))
        require(str(row["solver_status_chain"])
                == ";".join(rebuilt_status_chains)
                and str(row["lex_stage_chain"])
                == ";".join(rebuilt_stage_chains),
                "[solver-chain] row-level chains omit or alter an actual group stage",
                row=int(idx), persisted_status=row["solver_status_chain"],
                rebuilt_status=";".join(rebuilt_status_chains),
                persisted_stages=row["lex_stage_chain"],
                rebuilt_stages=";".join(rebuilt_stage_chains))

        def _check_certificate(prefix, target_full, role, objective_stage):
            where = "R1b:row={}:{}".format(int(idx), role)
            code = _exact_int_cell(
                row[prefix + "solver_status_code"],
                field=prefix + "solver_status_code", where=where)
            name = str(row[prefix + "solver_status_name"])
            proven = _cell_bool(
                row[prefix + "proven_optimal"],
                field=prefix + "proven_optimal", where=where)
            stop = str(row[prefix + "stop_reason"])
            solver_fun = require_finite(
                row[prefix + "solver_fun_raw"],
                prefix + "solver_fun_raw", where=where)
            p_full = float(row[prefix + "obj_incumbent_full_objective"])
            b_full = float(row[prefix + "mip_dual_bound_full_objective"])
            p = float(row[prefix + "obj_incumbent"])
            b = float(row[prefix + "mip_dual_bound"])
            c = float(row[prefix + "deviation_constant_C"])
            require(code == 0 and name == "Optimal" and proven
                    and stop == "proven_optimal",
                    "[solver-aggregate] {} certificate is not code-0 Optimal"
                    .format(role), row=int(idx), code=code, name=name,
                    proven=proven, stop_reason=stop)
            expected_solver_gap = solver_raw_gap_from_reported_fun(
                solver_fun, b, where=where)
            persisted_solver_gap = require_finite(
                row[prefix + "mip_gap_solver_raw"],
                prefix + "mip_gap_solver_raw", where=where)
            require(abs(persisted_solver_gap - expected_solver_gap)
                    <= _ulp_margin(persisted_solver_gap, expected_solver_gap),
                    "[solver-aggregate] {} raw solver mip gap differs from "
                    "solver fun/bound replay".format(role),
                    row=int(idx), solver_fun_raw=solver_fun,
                    dual_bound=b, persisted_gap=persisted_solver_gap,
                    expected_gap=expected_solver_gap)
            require(abs((p + c) - p_full) <= _ulp_margin(p, c, p_full)
                    and abs((b + c) - b_full)
                    <= _ulp_margin(b, c, b_full)
                    and abs(p_full - target_full)
                    <= _ulp_margin(p_full, target_full),
                    "[solver-aggregate] {} raw P/B/C identity or incumbent "
                    "target disagrees with its full objective".format(role),
                    row=int(idx), target_full=target_full, P=p, B=b, C=c,
                    P_full=p_full, B_full=b_full,
                    identity_tolerance_rule="8 ulps; no INT_TOL rounding")
            gap_raw = row[prefix + "gap_full_objective"]
            reason_raw = row[
                prefix + "gap_full_objective_undefined_reason"]
            recomputed_gap, recomputed_reason = full_objective_gap(
                p, b, c, where=where)
            if recomputed_gap is None:
                require(pd.isna(gap_raw)
                        and str(reason_raw) == str(recomputed_reason),
                        "[solver-aggregate] {} undefined raw gap differs from "
                        "P/B/C replay".format(role),
                        row=int(idx), gap=gap_raw, reason=reason_raw,
                        recomputed_reason=recomputed_reason)
            else:
                persisted_gap = require_finite(
                    gap_raw, prefix + "gap_full_objective", where=where)
                require(pd.isna(reason_raw)
                        and abs(persisted_gap - recomputed_gap)
                        <= _ulp_margin(persisted_gap, recomputed_gap),
                        "[solver-aggregate] {} raw full-objective gap was "
                        "altered instead of preserved".format(role),
                        row=int(idx), persisted_gap=persisted_gap,
                        recomputed_gap=recomputed_gap, reason=reason_raw)
            replayed = replay_persisted_exact_certificate(
                row, prefix, objective_stage,
                integral_durations=trusted_integrality,
                expected_deviation_constant=source_deviation_constant,
                where=where)
            if _cell_bool(row["formal_run"], field="formal_run", where=where):
                require(replayed["method"] in EXACT_CERTIFICATE_METHODS,
                        "formal R1b row used an unauthorized exact certificate",
                        row=int(idx), role=role, method=replayed["method"])

        weighted_target = (float(W2) * float(row["overtime"])
                           + float(row["w4"]) * float(row["deviation"])
                           + float(W5) * float(row["unserved"]))
        require(abs(float(row["weighted_objective_Z"]) - weighted_target)
                <= INT_TOL,
                "[solver-aggregate] persisted weighted_objective_Z disagrees "
                "with W2*W+w4*D+W5*U",
                row=int(idx), persisted=row["weighted_objective_Z"],
                recomputed=weighted_target)
        _check_certificate(
            "weighted_", weighted_target, PRIMARY_WEIGHTED_ROLE, "Z")
        _check_certificate(
            "w_refine_", float(row["overtime"]), W_REFINE_ROLE, "W")
        _check_certificate(
            "tie_break_", float(row["deviation"]), FINAL_TIE_BREAK_ROLE, "D")
        require(row["umin_A1_solver_certificate_role"] == UMIN_A1_ROLE
                and row["umin_A2_solver_certificate_role"] == UMIN_A2_ROLE
                and row["umin_A1_solver_evidence_scope"]
                == "single_independent_horizon_min_U_solve"
                and row["umin_A2_solver_evidence_scope"]
                == "single_independent_horizon_min_U_solve",
                "[service-certificate] U-min solver roles/scopes are ambiguous",
                row=int(idx))
        _check_certificate(
            "umin_A1_", float(row["U_min_A1"]), UMIN_A1_ROLE, "U")
        _check_certificate(
            "umin_A2_", float(row["U_min_A2"]), UMIN_A2_ROLE, "U")
        require(str(row["u_min_solver_status_chain"]) == "A1:0|A2:0",
                "[service-certificate] preliminary U-min status chain is not "
                "the two independent code-0 solves", row=int(idx),
                chain=row["u_min_solver_status_chain"])

        # Compatibility solver columns must be a byte-level semantic duplicate
        # of the explicitly named tie-break certificate, never of weighted_*.
        generic_gap = row[COL.GAP_FULL]
        tie_gap = row["tie_break_gap_full_objective"]
        gap_same = ((pd.isna(generic_gap) and pd.isna(tie_gap))
                    or (not pd.isna(generic_gap) and not pd.isna(tie_gap)
                        and abs(float(generic_gap) - float(tie_gap))
                        <= _ulp_margin(generic_gap, tie_gap)))
        generic_reason = row[COL.GAP_FULL_UNDEF]
        tie_reason = row["tie_break_gap_full_objective_undefined_reason"]
        reason_same = ((pd.isna(generic_reason) and pd.isna(tie_reason))
                       or str(generic_reason) == str(tie_reason))
        generic_solver_gap = require_finite(
            row[COL.GAP_SOLVER_RAW], COL.GAP_SOLVER_RAW,
            where="R1b:row={}:generic".format(int(idx)))
        generic_solver_fun = require_finite(
            row["solver_fun_raw"], "solver_fun_raw",
            where="R1b:row={}:generic".format(int(idx)))
        expected_generic_solver_gap = solver_raw_gap_from_reported_fun(
            generic_solver_fun, row[COL.DUAL],
            where="R1b:row={}:generic".format(int(idx)))
        require(abs(generic_solver_gap - expected_generic_solver_gap)
                <= _ulp_margin(generic_solver_gap,
                               expected_generic_solver_gap),
                "[solver-aggregate] generic raw solver mip gap differs from "
                "solver fun/bound replay",
                row=int(idx), solver_fun_raw=generic_solver_fun,
                dual_bound=row[COL.DUAL], persisted_gap=generic_solver_gap,
                expected_gap=expected_generic_solver_gap)
        tie_solver_gap = require_finite(
            row["tie_break_mip_gap_solver_raw"],
            "tie_break_mip_gap_solver_raw",
            where="R1b:row={}:generic".format(int(idx)))
        require(_exact_int_cell(
                    row[COL.STATUS_CODE], field=COL.STATUS_CODE,
                    where="R1b:row={}:generic".format(int(idx)))
                == _exact_int_cell(
                    row["tie_break_solver_status_code"],
                    field="tie_break_solver_status_code",
                    where="R1b:row={}:generic".format(int(idx)))
                and str(row[COL.STATUS_NAME])
                == str(row["tie_break_solver_status_name"])
                and _cell_bool(
                    row[COL.PROVEN_OPTIMAL], field=COL.PROVEN_OPTIMAL,
                    where="R1b:row={}:generic".format(int(idx)))
                == _cell_bool(
                    row["tie_break_proven_optimal"],
                    field="tie_break_proven_optimal",
                    where="R1b:row={}:generic".format(int(idx)))
                and abs(generic_solver_fun
                        - float(row["tie_break_solver_fun_raw"]))
                <= _ulp_margin(row["solver_fun_raw"],
                               row["tie_break_solver_fun_raw"])
                and abs(generic_solver_gap - tie_solver_gap)
                <= _ulp_margin(generic_solver_gap, tie_solver_gap)
                and abs(float(row[COL.PRIMAL])
                        - float(row["tie_break_obj_incumbent"]))
                <= _ulp_margin(row[COL.PRIMAL],
                               row["tie_break_obj_incumbent"])
                and abs(float(row[COL.DUAL])
                        - float(row["tie_break_mip_dual_bound"]))
                <= _ulp_margin(row[COL.DUAL],
                               row["tie_break_mip_dual_bound"])
                and abs(float(row[COL.PRIMAL_FULL])
                        - float(row["tie_break_obj_incumbent_full_objective"]))
                <= _ulp_margin(row[COL.PRIMAL_FULL],
                               row["tie_break_obj_incumbent_full_objective"])
                and abs(float(row[COL.DUAL_FULL])
                        - float(row[
                            "tie_break_mip_dual_bound_full_objective"]))
                <= _ulp_margin(
                    row[COL.DUAL_FULL],
                    row["tie_break_mip_dual_bound_full_objective"])
                and abs(float(row[COL.DEV_CONST_C])
                        - float(row["tie_break_deviation_constant_C"]))
                <= _ulp_margin(row[COL.DEV_CONST_C],
                               row["tie_break_deviation_constant_C"])
                and gap_same and reason_same
                and str(row[COL.STOP_REASON])
                == str(row["tie_break_stop_reason"]),
                "[solver-aggregate] generic solver block is not the declared "
                "final-D tie-break certificate",
                row=int(idx))

        d = float(row["deviation"])
        p_full = float(row[COL.PRIMAL_FULL])
        require(abs(p_full - d) <= _ulp_margin(p_full, d),
                "[solver-aggregate] final-D incumbent does not equal the horizon "
                "deviation",
                row=int(idx), deviation=d, primal_full=p_full,
                identity_tolerance_rule="8 ulps; exact lattice proof checked above")
    ok("solver-aggregate", "independent A1/A2 U minima, horizon weighted-Z, "
                           "and final-D certificates are separate code-0 evidence")

    for idx, row in raw.iterrows():
        enabled = _cell_bool(
            row["canonical_assignment"], field="canonical_assignment",
            where="R1b:row={}:assignment".format(int(idx)))
        raw_horizon_hash = row.get("horizon_assignment_hash", "")
        raw_solver_hash = row.get(COL.ASSIGN_HASH, "")
        horizon_hash = ("" if pd.isna(raw_horizon_hash)
                        else str(raw_horizon_hash).strip())
        solver_hash = ("" if pd.isna(raw_solver_hash)
                       else str(raw_solver_hash).strip())
        evidence = str(row["assignment_hash_evidence"]).strip()
        if enabled:
            require(bool(horizon_hash) and bool(solver_hash)
                    and horizon_hash == solver_hash
                    and evidence == "canonical_lexicographic_assignment",
                    "[assignment] canonical horizon assignment lacks matching hashes",
                    row=int(idx), horizon_hash=horizon_hash,
                    solver_hash=solver_hash, evidence=evidence)
        else:
            require(not horizon_hash and not solver_hash,
                    "[assignment] non-canonical assignment hashes must be empty",
                    row=int(idx), horizon_hash=horizon_hash,
                    solver_hash=solver_hash)
            require(evidence == "disabled_not_cross_run_evidence",
                    "[assignment] disabled canonicalization has the wrong evidence label",
                    row=int(idx), evidence=evidence)
    ok("assignment", "assignment hashes exist only for canonical lexicographic assignments")

    expected_snap = str(design.get("snap_id") or "")
    observed_snap = {str(v).strip() for v in raw["snap_id"].fillna("")}
    require(observed_snap == {expected_snap},
            "[SNAP] artifact snap_id differs from the run's verified SNAP receipt",
            expected=expected_snap, observed=sorted(observed_snap))
    ok("SNAP", "every scientific row carries the verified runtime SNAP identifier"
       if expected_snap else "rehearsal rows carry no SNAP identifier")

    # REQ-6 test 4
    dom = assert_no_dominated_w4_point(raw)
    ok("non-domination", f"no dominated w4 point in any of "
                         f"{dom['groups_checked']} (rho,seed,absent,method) groups")

    # REQ-6 test 3
    pv = verify_provenance(
        summary, raw, prov,
        expected_source_sha256=landed_raw_sha256)
    ok("provenance", f"{pv['cells_checked']} summary cell(s) recomputed from "
                     f"{pv['provenance_records']} cited raw rows")
    if landed_raw_sha256 is not None:
        ok("dependency", "every derived provenance row cites consumed landed "
                         f"e4_raw.csv sha256={landed_raw_sha256}")

    # REQ-6 item on cross-stage transcription: the summary may cite e4_raw only
    require(set(prov["source_file"]) == {"e4_raw.csv"},
            "[transcription] the summary cites a source file other than e4_raw.csv",
            cited=sorted(set(prov["source_file"])))
    ok("transcription", "e4_summary.csv cites e4_raw.csv and nothing else")

    if verbose:
        for line in log:
            print("    " + line)
    return log


# --------------------------------------------------------------------------- #
# write path — s2_core.canonical_write per artifact
# --------------------------------------------------------------------------- #
def write_validated(out_dir, raw, summary, prov, diagnostics_df, design, policy,
                    snap_id=None, formal=False, verbose=True,
                    shard_receipts_df=None):
    """One `canonical_write` per artifact: temp -> validate landed bytes ->
    promote, rollback on a caught failure.

    HONEST LIMIT: `canonical_write` promotes exactly ONE file per invocation.
    Scientific files and their diagnostics sidecar therefore are NOT pair-atomic
    or family-atomic; an interruption can leave a partially promoted set.
    Cross-artifact checks run before promotion and again on landed bytes.
    """
    os.makedirs(out_dir, exist_ok=True)
    extra_checks(raw, summary, prov, design, verbose=False)      # pre-write gate
    validate_solver_diagnostics_design(
        diagnostics_df, design, where="R1b diagnostics pre-write gate")

    reports = {}
    raw_path = os.path.join(out_dir, "e4_raw.csv")
    holder = {}

    def _validate_raw(temp_path):
        holder["report"] = validate_stage_output(
            temp_path, raw_spec(design, policy, snap_id, formal))

    # Causal dependency order: raw first, then re-read those exact landed bytes.
    with canonical_write(
            raw_path, _validate_raw, stage=STAGE,
            diagnostics={"artifact": "e4_raw.csv"}) as temp_path:
        raw.to_csv(temp_path, index=False)
    reports["e4_raw.csv"] = holder["report"]
    landed_raw_sha256 = sha256_file(raw_path)
    landed_raw = _read_csv_audited(
        raw_path, why="recompute derived R1b artifacts from landed raw bytes")
    landed_summary, landed_prov = build_summary(
        landed_raw, source_name="e4_raw.csv",
        source_sha256=landed_raw_sha256)
    verify_provenance(
        landed_summary, landed_raw, landed_prov,
        expected_source_sha256=landed_raw_sha256)

    for name, df in (("e4_summary.csv", landed_summary),
                     ("e4_summary_provenance.csv", landed_prov)):
        path = os.path.join(out_dir, name)

        def _validate_derived(temp_path, _name=name):
            require(os.path.getsize(temp_path) > 0,
                    "derived artifact is empty", artifact=_name)
            if _name == "e4_summary_provenance.csv":
                landed_dependency = pd.read_csv(
                    temp_path, float_precision="round_trip")
                require("source_sha256" in landed_dependency.columns
                        and set(str(v) for v in
                                landed_dependency["source_sha256"])
                        == {landed_raw_sha256},
                        "derived provenance does not cite the consumed landed "
                        "e4_raw.csv SHA256",
                        artifact=_name,
                        expected_source_sha256=landed_raw_sha256)
            print(f"[derived]    {_name} recomputed from landed e4_raw.csv "
                  f"sha256={landed_raw_sha256}")

        with canonical_write(
                path, _validate_derived, stage=STAGE,
                diagnostics={"artifact": name,
                             "source_file": "e4_raw.csv",
                             "source_sha256": landed_raw_sha256}) as temp_path:
            df.to_csv(temp_path, index=False)

    diagnostic_path = os.path.join(out_dir, "e4_solver_diagnostics.csv")

    def _validate_diagnostics(temp_path):
        require(os.path.getsize(temp_path) > 0,
                "e4 solver diagnostics sidecar is empty",
                artifact="e4_solver_diagnostics.csv")
        landed_diagnostics = pd.read_csv(
            temp_path, float_precision="round_trip")
        validate_solver_diagnostics_design(
            landed_diagnostics, design,
            where="R1b diagnostics canonical-write byte gate")

    with canonical_write(
            diagnostic_path, _validate_diagnostics, stage=STAGE,
            diagnostics={"artifact": "e4_solver_diagnostics.csv",
                         "scientific_result": False}) as temp_path:
        diagnostics_df.to_csv(temp_path, index=False)

    extra_output_names = []
    if shard_receipts_df is not None:
        receipt_path = os.path.join(out_dir, "e4_shard_receipts.csv")

        def _validate_receipts(temp_path):
            landed_receipts = pd.read_csv(
                temp_path, float_precision="round_trip")
            required = {"shard_receipt_id", "rho", "snap_id"} | set(
                SHARD_RECEIPT_COLUMNS[1:])
            require(required <= set(landed_receipts.columns),
                    "shard receipt sidecar lacks required fields",
                    missing=sorted(required - set(landed_receipts.columns)))
            require(len(landed_receipts) == len(FORMAL_RHOS)
                    and not bool(landed_receipts["shard_receipt_id"].duplicated().any()),
                    "shard receipt sidecar must contain exactly six unique shards",
                    rows=len(landed_receipts))
            require(set(round(float(v), 8) for v in landed_receipts["rho"])
                    == set(FORMAL_RHOS),
                    "shard receipt sidecar rho axis is incomplete or duplicated",
                    observed=sorted(set(float(v) for v in landed_receipts["rho"])))
            require(set(landed_receipts["snap_id"].astype(str)) == {str(snap_id)},
                    "shard receipt sidecar contains a mixed SNAP",
                    expected=snap_id,
                    observed=sorted(set(landed_receipts["snap_id"].astype(str))))
            for col in SHARD_RECEIPT_COLUMNS[1:]:
                for value in landed_receipts[col].astype(str):
                    require(len(value) == 64
                            and all(ch in "0123456789abcdef" for ch in value.lower()),
                            "shard artifact receipt is not a SHA256",
                            column=col, value=value)

        with canonical_write(
                receipt_path, _validate_receipts, stage=STAGE,
                diagnostics={"artifact": "e4_shard_receipts.csv",
                             "scientific_result": False,
                             "source_family": "six formal R1b shards"}) as temp_path:
            shard_receipts_df.to_csv(temp_path, index=False)
        extra_output_names.append("e4_shard_receipts.csv")

    landed = {
        "e4_raw.csv": landed_raw,
        "e4_summary.csv": pd.read_csv(
            os.path.join(out_dir, "e4_summary.csv"),
            float_precision="round_trip"),
        "e4_summary_provenance.csv": pd.read_csv(
            os.path.join(out_dir, "e4_summary_provenance.csv"),
            float_precision="round_trip"),
    } 
    require(sha256_file(raw_path) == landed_raw_sha256,
            "e4_raw.csv changed after derived artifacts consumed it",
            consumed_sha256=landed_raw_sha256,
            final_sha256=sha256_file(raw_path))
    log = extra_checks(
        landed["e4_raw.csv"], landed["e4_summary.csv"],
        landed["e4_summary_provenance.csv"], design, verbose=verbose,
        landed_raw_sha256=landed_raw_sha256)
    hashes = hash_manifest(
        design["inputs"],
        [os.path.join(out_dir, n) for n in
         sorted(list(landed) + ["e4_solver_diagnostics.csv"]
                + extra_output_names)])
    return hashes, reports, log, landed_summary, landed_prov


# --------------------------------------------------------------------------- #
def _derive_merge_design(manifest_path, contract, policy, snap_id, *,
                         allow_fixture_manifest=False):
    """Re-derive the complete merge axis and source metadata from the manifest.

    No shard row participates in this construction.  The production entrypoint
    accepts only the official manifest next to this script; the fixture escape
    hatch is an internal test parameter and is not exposed by the CLI.
    """
    manifest_path = os.path.abspath(manifest_path)
    require(os.path.isfile(manifest_path), "merge manifest is missing",
            path=manifest_path)
    official = os.path.join(HERE, "rho_manifest.csv")
    if not allow_fixture_manifest:
        require(os.path.samefile(manifest_path, official),
                "formal R1b shard merge accepts only the official rho_manifest.csv",
                requested=manifest_path, official=official)
    man = _read_csv_audited(manifest_path, why="independent shard-merge axis")
    required = {
        "rho_target", "seed", "hhcp", "sol", "feasible", "rho_global",
        "rho_used_mean", "rho_used_max", "N", "cap", "src", "src_sha256",
        "h_sha256", "s_sha256", "generator", "generator_sha256",
        "family_runner", "family_runner_sha256"}
    require(required <= set(man.columns),
            "formal merge manifest lacks required scenario provenance fields",
            missing=sorted(required - set(man.columns)))
    observed_rhos = set(round(float(v), 8) for v in man["rho_target"])
    require(observed_rhos == set(FORMAL_RHOS),
            "formal merge manifest must contain exactly the six registered rho levels",
            expected=list(FORMAL_RHOS), observed=sorted(observed_rhos))
    require(not bool(man.duplicated(["rho_target", "seed"]).any()),
            "formal merge manifest contains duplicate (rho_target,seed) rows")
    if not allow_fixture_manifest:
        counts = man.groupby("rho_target")["seed"].nunique().to_dict()
        require(len(man) == 30 and all(int(counts.get(r, 0)) == 5
                                      for r in FORMAL_RHOS),
                "official R1b merge requires 30 scenarios: five seeds at each rho",
                rows=len(man), seed_counts=counts)

    axis, ntasks, expected, scenario_pool_counts = {}, {}, {}, {}
    deviation_constants = {}
    scenario_lattice_preconditions = {}
    inputs = [manifest_path, DESIGN_MANIFEST]
    man_dir = os.path.dirname(manifest_path)
    for _, row in man.iterrows():
        rho = float(row["rho_target"])
        seed = _exact_int_cell(
            row["seed"], field="seed", where="formal R1b merge manifest")
        hp = os.path.join(man_dir, str(row["hhcp"]))
        sp = os.path.join(man_dir, str(row["sol"]))
        for p in (hp, sp):
            require(os.path.isfile(p), "merge manifest points at a missing file",
                    path=p, rho=rho, seed=seed)
            inputs.append(p)
        h_sha, s_sha = sha256_file(hp), sha256_file(sp)
        require(str(row["h_sha256"]).strip() == h_sha
                and str(row["s_sha256"]).strip() == s_sha,
                "merge source H/S bytes disagree with manifest SHA256",
                rho=rho, seed=seed, h_sha256=h_sha, s_sha256=s_sha)
        for label, path_col, hash_col in (
                ("source", "src", "src_sha256"),
                ("generator", "generator", "generator_sha256"),
                ("family_runner", "family_runner", "family_runner_sha256")):
            source_path = resolve_manifest_source_path(row[path_col], man_dir)
            if source_path is None:
                # Release marker names this source as withheld; its recorded
                # digest is the only provenance available here.
                continue
            actual_source_sha = sha256_file(source_path)
            require(actual_source_sha == str(row[hash_col]).strip(),
                    "merge manifest source bytes disagree with SHA256",
                    rho=rho, seed=seed, source_role=label, path=source_path,
                    declared_sha256=str(row[hash_col]).strip(),
                    actual_sha256=actual_source_sha)
            inputs.append(source_path)
        require(manifest_bool(row["feasible"], "feasible"),
                "merge manifest declares an infeasible incumbent",
                rho=rho, seed=seed)
        parser = HHCPFullParser(hp)
        sa = load_static(sp)
        premise = scenario_lattice_precondition(
            parser, sa, where=f"merge rho={rho} seed={seed}")
        premise.update({"rho": rho, "seed": seed,
                        "h_sha256": h_sha, "s_sha256": s_sha})
        require(premise["integral_lattice_data"] is True,
                "formal merge independently found a non-integral W/Z lattice input",
                rho=rho, seed=seed, premise=premise)
        scenario_lattice_preconditions[_scenario_key(rho, seed)] = premise
        nurses = set(parser.nurses)
        declared_n = _exact_int_cell(
            row["N"], field="N",
            where="formal R1b merge manifest rho={} seed={}".format(
                rho, seed))
        require(declared_n == len(nurses),
                "formal merge manifest N differs from the parsed caregiver pool",
                rho=rho, seed=seed, manifest_N=declared_n,
                parsed_caregivers=len(nurses))
        assert_incumbent_flat_feasible(
            parser, sa, label=f"merge rho={rho} seed={seed}")
        actives = active_absentees(parser, sa, None)
        require(bool(actives), "no active future-task holder in merge scenario",
                rho=rho, seed=seed)
        axis[(rho, seed)] = actives
        scenario_pool_counts[(rho, seed)] = declared_n
        nested_counts, incumbent_counts = {}, {}
        slots = future_slots(sa)
        for absent in actives:
            eligibility = assert_incumbent_eligible(
                parser, sa, nurses, absent,
                label=f"merge rho={rho} seed={seed} absent={absent}")
            a1 = WeightedModel(
                parser, sa, nurses, absent, "A1", slots, policy,
                rho=rho, seed=seed)
            a2 = WeightedModel(
                parser, sa, nurses, absent, "A2", slots, policy,
                rho=rho, seed=seed)
            require(a1.dev_const == a2.dev_const
                    and float(a1.dev_const).is_integer()
                    and int(a1.dev_const) >= 0,
                    "source-derived deviation constant differs by method",
                    rho=rho, seed=seed, absent=absent,
                    A1=a1.dev_const, A2=a2.dev_const)
            deviation_constants[(rho, seed, int(absent))] = int(a1.dev_const)
            nesting = assert_A1_candidates_subset_A2(a1, a2, absent)
            ntasks[(rho, seed, absent)] = sum(
                len(slot_tasks(parser, sa, q)) for q in slots)
            incumbent_counts[absent] = int(
                eligibility["n_incumbent_pairs_checked"])
            nested_counts[absent] = int(
                nesting["n_tasks_nested_A1_subset_A2"])
        expected[(rho, seed)] = {
            "absents": list(actives),
            "active_caregiver_count": declared_n,
            "rho_target": rho,
            "rho_global": float(row["rho_global"]),
            "rho_used_mean": float(row["rho_used_mean"]),
            "rho_used_max": float(row["rho_used_max"]),
            "scenario_N": declared_n,
            "scenario_cap": float(row["cap"]),
            "source_file": str(row["src"]),
            "src_sha256": str(row["src_sha256"]),
            "h_sha256": h_sha,
            "s_sha256": s_sha,
            "generator_file": str(row["generator"]),
            "generator_sha256": str(row["generator_sha256"]),
            "family_runner_file": str(row["family_runner"]),
            "family_runner_sha256": str(row["family_runner_sha256"]),
            "n_incumbent_pairs_checked": incumbent_counts,
            "n_tasks_nested_A1_subset_A2": nested_counts,
        }
    design = build_design(
        axis, inputs, contract=contract,
        scenario_pool_counts=scenario_pool_counts,
        scenario_lattice_preconditions=scenario_lattice_preconditions,
        deviation_constants=deviation_constants,
        formal=True, snap_id=snap_id)
    design["n_tasks_future"] = ntasks
    design["scenario_expected"] = expected
    return man, design


def _assert_shard_matches_manifest(raw, rho, design):
    """Compare a landed shard against independently re-derived scenario facts."""
    for (rr, seed), exp in design["scenario_expected"].items():
        if not np.isclose(float(rr), float(rho), rtol=0, atol=1e-12):
            continue
        g = raw[(np.isclose(raw["rho"].astype(float), float(rr), rtol=0,
                            atol=1e-12))
                & (raw["seed"].astype(int) == int(seed))]
        require(not g.empty, "formal shard is missing a manifest scenario",
                rho=rr, seed=seed)
        require(set(g["absent"].astype(int)) == set(exp["absents"]),
                "formal shard disruption axis differs from the input plan",
                rho=rr, seed=seed,
                expected=exp["absents"],
                observed=sorted(set(g["absent"].astype(int))))
        exact_fields = [
            "source_file", "src_sha256", "h_sha256", "s_sha256",
            "generator_file", "generator_sha256", "family_runner_file",
            "family_runner_sha256"]
        numeric_fields = [
            "rho_target", "rho_global", "rho_used_mean", "rho_used_max",
            "scenario_N", "scenario_cap", "active_caregiver_count"]
        for field in exact_fields:
            require(set(g[field].astype(str)) == {str(exp[field])},
                    "formal shard source provenance differs from manifest",
                    rho=rr, seed=seed, field=field,
                    expected=exp[field], observed=sorted(set(g[field].astype(str))))
        for field in numeric_fields:
            values = g[field].astype(float).to_numpy()
            require(bool(np.isclose(values, float(exp[field]), rtol=0,
                                    atol=1e-12).all()),
                    "formal shard numeric provenance differs from manifest",
                    rho=rr, seed=seed, field=field, expected=exp[field])
        require(set(g["eligible_disruptions_in_scenario"].astype(int))
                == {len(exp["absents"])}
                and set(g["absence_rule"].astype(str)) == {ABSENCE_RULE}
                and set(g["absence_cap_mode"].astype(str))
                == {"all_active_no_cap"}
                and set(g["manifest_provenance_mode"].astype(str))
                == {"declared_and_all_sources_rehashed"},
                "formal shard absence provenance is not all-active/no-cap",
                rho=rr, seed=seed)
        for absent, gg in g.groupby("absent"):
            absent = int(absent)
            require(set(gg["n_tasks"].astype(int))
                    == {design["n_tasks_future"][(rr, seed, absent)]}
                    and set(gg["n_incumbent_pairs_checked"].astype(int))
                    == {exp["n_incumbent_pairs_checked"][absent]}
                    and set(gg["n_tasks_nested_A1_subset_A2"].astype(int))
                    == {exp["n_tasks_nested_A1_subset_A2"][absent]},
                    "formal shard structural counts differ from re-derived input",
                    rho=rr, seed=seed, absent=absent)


def merge_formal_shards(manifest_path, out_dir, shard_dirs, *, snap_id,
                        cli_time_limit_s, verbose=True,
                        allow_fixture_manifest=False):
    """Validate and merge the six formal rho shards through the canonical gate."""
    assert_no_staging_import()
    assert_no_absolute_runtime_paths(__file__)
    contract = load_design_contract()
    policy = solver_policy(
        cli_time_limit_s=cli_time_limit_s,
        require_cli_time_limit=True, contract=contract)
    snap_id = require_snap_binding(DESIGN_MANIFEST, snap_id, stage=STAGE)
    dirs = [os.path.abspath(p) for p in shard_dirs]
    require(len(dirs) == len(FORMAL_RHOS),
            "formal R1b merge requires exactly six shard directories",
            expected=len(FORMAL_RHOS), actual=len(dirs))
    require(len(set(dirs)) == len(dirs),
            "formal R1b merge received a duplicate shard directory",
            shard_dirs=dirs)
    _man, design = _derive_merge_design(
        manifest_path, contract, policy, snap_id,
        allow_fixture_manifest=allow_fixture_manifest)

    raw_frames, diagnostic_frames, receipts, verified_artifacts = [], [], [], []
    observed_rho_to_dir = {}
    schemas = {name: None for name in (
        "raw", "summary", "provenance", "diagnostics")}
    artifact_names = {
        "raw": "e4_raw.csv",
        "summary": "e4_summary.csv",
        "provenance": "e4_summary_provenance.csv",
        "diagnostics": "e4_solver_diagnostics.csv",
    }
    for shard_dir in dirs:
        paths = {key: os.path.join(shard_dir, name)
                 for key, name in artifact_names.items()}
        for key, path in paths.items():
            require(os.path.isfile(path),
                    "formal shard lacks a required landed artifact",
                    shard_dir=shard_dir, artifact=key, path=path)
        hashes = {key: sha256_file(path) for key, path in paths.items()}
        verified_artifacts.extend(
            (paths[key], hashes[key]) for key in sorted(paths))
        raw = _read_csv_audited(paths["raw"], why="formal R1b shard raw")
        summary = _read_csv_audited(
            paths["summary"], why="formal R1b shard summary")
        provenance = _read_csv_audited(
            paths["provenance"], why="formal R1b shard provenance")
        diagnostics = _read_csv_audited(
            paths["diagnostics"], why="formal R1b shard diagnostics")
        validate_solver_diagnostics_raw_gaps(
            diagnostics,
            where="formal R1b shard diagnostics {}".format(shard_dir))
        frames = {"raw": raw, "summary": summary,
                  "provenance": provenance, "diagnostics": diagnostics}
        for name, frame in frames.items():
            schema = tuple(str(c) for c in frame.columns)
            if schemas[name] is None:
                schemas[name] = schema
            require(schema == schemas[name],
                    "formal shard artifact schema differs across rho shards",
                    artifact=name, shard_dir=shard_dir,
                    expected=list(schemas[name]), observed=list(schema))
        rhos = sorted(set(round(float(v), 8) for v in raw["rho"]))
        require(len(rhos) == 1 and rhos[0] in set(FORMAL_RHOS),
                "each formal shard must contain exactly one registered rho",
                shard_dir=shard_dir, observed=rhos)
        rho = rhos[0]
        require(rho not in observed_rho_to_dir,
                "formal shard rho values are not mutually exclusive",
                rho=rho, first=observed_rho_to_dir.get(rho), second=shard_dir)
        observed_rho_to_dir[rho] = shard_dir

        shard_axis = {k: v for k, v in design["axis"].items()
                      if np.isclose(float(k[0]), rho, rtol=0, atol=1e-12)}
        shard_pool_counts = {
            k: v for k, v in design["scenario_pool_counts"].items()
            if np.isclose(float(k[0]), rho, rtol=0, atol=1e-12)}
        shard_lattice_preconditions = {
            k: v for k, v in design[
                "scenario_lattice_preconditions"].items()
            if np.isclose(float(k[0]), rho, rtol=0, atol=1e-12)}
        shard_deviation_constants = {
            k: v for k, v in design["deviation_constants"].items()
            if np.isclose(float(k[0]), rho, rtol=0, atol=1e-12)}
        shard_inputs = [p for p in design["inputs"]]
        shard_design = build_design(
            shard_axis, shard_inputs, contract=contract,
            scenario_pool_counts=shard_pool_counts,
            scenario_lattice_preconditions=shard_lattice_preconditions,
            deviation_constants=shard_deviation_constants,
            formal=True, snap_id=snap_id)
        shard_design["n_tasks_future"] = {
            k: v for k, v in design["n_tasks_future"].items()
            if np.isclose(float(k[0]), rho, rtol=0, atol=1e-12)}
        validate_solver_diagnostics_design(
            diagnostics, shard_design,
            where="formal R1b shard diagnostics design gate {}".format(
                shard_dir))
        validate_stage_output(
            paths["raw"], raw_spec(shard_design, policy, snap_id, True))
        verify_provenance(
            summary, raw, provenance,
            expected_source_sha256=hashes["raw"])
        extra_checks(
            raw, summary, provenance, shard_design, verbose=False,
            landed_raw_sha256=hashes["raw"])
        _assert_shard_matches_manifest(raw, rho, design)
        require(set(raw["snap_id"].astype(str)) == {snap_id}
                and set(raw["formal_run"].astype(str).str.lower()) == {"true"}
                and set(raw["solver_policy_manifest_sha256"].astype(str))
                == {sha256_file(DESIGN_MANIFEST)}
                and set(raw["canonical_assignment"].astype(str).str.lower())
                <= {"false", "0"},
                "formal shard runtime/SNAP/canonical policy is inconsistent",
                rho=rho, shard_dir=shard_dir)
        time_columns = [
            COL.TIME_LIMIT, "weighted_time_limit_s", "w_refine_time_limit_s",
            "tie_break_time_limit_s",
            "umin_A1_time_limit_s", "umin_A2_time_limit_s"]
        for col in time_columns:
            require(bool(np.equal(
                        raw[col].astype(float).to_numpy(),
                        policy.time_limit_s).all()),
                    "formal shard time-limit evidence differs from policy",
                    rho=rho, column=col, expected=policy.time_limit_s)
        diag_required = {
            "rho", "seed", "method", "snap_id", "presolve",
            "time_limit_s", "solver_status_code", "objective_stage",
            "solver_fun_raw", "mip_dual_bound", "mip_gap_solver_raw",
            "lattice_proves_exact_optimality"}
        require(diag_required <= set(diagnostics.columns)
                and not diagnostics.empty,
                "formal shard diagnostics are empty or lack solver evidence",
                rho=rho, missing=sorted(diag_required - set(diagnostics.columns)))
        diagnostic_statuses = {
            _exact_int_cell(
                value, field="solver_status_code",
                where="formal R1b shard diagnostics rho={}".format(rho))
            for value in diagnostics["solver_status_code"]}
        diagnostic_presolve = {
            _cell_bool(
                value, field="presolve",
                where="formal R1b shard diagnostics rho={}".format(rho))
            for value in diagnostics["presolve"]}
        require(set(round(float(v), 8) for v in diagnostics["rho"]) == {rho}
                and set(diagnostics["snap_id"].astype(str)) == {snap_id}
                and diagnostic_presolve == {False}
                and diagnostic_statuses == {0}
                and bool(np.equal(
                    diagnostics["time_limit_s"].astype(float).to_numpy(),
                    policy.time_limit_s).all()),
                "formal shard diagnostics disagree with solver policy/SNAP",
                rho=rho)
        receipt = {
            "shard_receipt_id": f"rho={rho:.2f}", "rho": rho,
            "snap_id": snap_id,
            "shard_raw_sha256": hashes["raw"],
            "shard_summary_sha256": hashes["summary"],
            "shard_provenance_sha256": hashes["provenance"],
            "shard_diagnostics_sha256": hashes["diagnostics"],
        }
        for column in SHARD_RECEIPT_COLUMNS:
            raw[column] = receipt[column]
            diagnostics[column] = receipt[column]
        raw_frames.append(raw)
        diagnostic_frames.append(diagnostics)
        receipts.append(receipt)

    require(set(observed_rho_to_dir) == set(FORMAL_RHOS),
            "formal R1b shard family has missing or extra rho levels",
            expected=list(FORMAL_RHOS),
            observed=sorted(observed_rho_to_dir))
    combined_raw = pd.concat(raw_frames, ignore_index=True)
    key = ["rho", "seed", "absent", "method", "w4"]
    combined_raw = combined_raw.sort_values(key).reset_index(drop=True)
    combined_diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    combined_diagnostics = combined_diagnostics.sort_values(
        ["rho", "seed", "method", "absent", "lex_stage"],
        na_position="last").reset_index(drop=True)
    receipt_df = pd.DataFrame(receipts).sort_values("rho").reset_index(drop=True)
    for path, expected_sha in verified_artifacts:
        require(sha256_file(path) == expected_sha,
                "formal shard artifact changed after validation and before merge",
                path=path, expected_sha256=expected_sha,
                actual_sha256=sha256_file(path))
    design["shard_receipts_required"] = True
    design["shard_receipts"] = receipts
    for shard_dir in dirs:
        for name in artifact_names.values():
            design["inputs"].append(os.path.join(shard_dir, name))
    summary, provenance = build_summary(combined_raw)
    hashes, reports, log, summary, provenance = write_validated(
        out_dir, combined_raw, summary, provenance, combined_diagnostics,
        design, policy, snap_id=snap_id, formal=True, verbose=verbose,
        shard_receipts_df=receipt_df)
    for path, expected_sha in verified_artifacts:
        require(sha256_file(path) == expected_sha,
                "formal shard artifact changed while combined artifacts were "
                "being published", path=path, expected_sha256=expected_sha,
                actual_sha256=sha256_file(path))
    if verbose:
        print("  [merge] six mutually exclusive formal rho shards accepted; "
              f"SNAP={snap_id}; rows={len(combined_raw)}")
    return (combined_raw, summary, provenance, combined_diagnostics,
            receipt_df, design, hashes, reports, log)


# --------------------------------------------------------------------------- #
def run(manifest_path, out_dir, mode=DEFAULT_DECOMPOSITION,
        max_absent=None,
        rho_filter=None, canonical_assignment=None, allow_fractional=False,
        verbose=True, formal=False, snap_id=None, cli_time_limit_s=None):
    formal = _cell_bool(formal, field="formal", where="R1b run API")
    allow_fractional = _cell_bool(
        allow_fractional, field="allow_fractional", where="R1b run API")
    assert_no_staging_import()
    assert_no_absolute_runtime_paths(__file__)
    contract = load_design_contract()
    policy = solver_policy(cli_time_limit_s=cli_time_limit_s,
                           require_cli_time_limit=formal, contract=contract)
    authorized_methods = tuple(contract["design_grid"]["methods"])
    authorized_w4 = tuple(contract["design_grid"]["w4"])
    authorized_modes = tuple(contract["design_grid"]["decomposition_modes"])
    require(mode in authorized_modes,
            "requested R1b decomposition mode is not manifest-authorized",
            requested_mode=mode, authorized_modes=authorized_modes)
    if canonical_assignment is None:
        canonical_assignment = CANONICAL_ASSIGNMENT_FORMAL_DEFAULT
    canonical_assignment = _cell_bool(
        canonical_assignment, field="canonical_assignment",
        where="R1b run API")
    if formal:
        require(mode == FORMAL_DECOMPOSITION,
                "formal R1b is horizon-only because U_common is a "
                "horizon-total equality",
                requested_mode=mode, required_mode=FORMAL_DECOMPOSITION)
        require(max_absent is None,
                "formal R1b forbids an absentee cap and must cover every active "
                "future task holder", max_absent=max_absent,
                absence_rule=ABSENCE_RULE)
        require(canonical_assignment is CANONICAL_ASSIGNMENT_FORMAL_DEFAULT,
                "formal R1b canonical-assignment mode differs from the "
                "pre-registered policy", requested=canonical_assignment,
                registered=CANONICAL_ASSIGNMENT_FORMAL_DEFAULT)
        require(not allow_fractional,
                "formal R1b forbids --allow-fractional-durations because its "
                "registered lexicographic and objective-lattice proofs require "
                "integral effective durations",
                allow_fractional_durations=allow_fractional)
        snap_id = require_snap_binding(DESIGN_MANIFEST, snap_id, stage=STAGE)
    else:
        require(not snap_id,
                "a rehearsal may not attach or invent a SNAP identifier",
                stage=STAGE, snap_id=snap_id)

    manifest_path = os.path.abspath(manifest_path)
    man_dir = os.path.dirname(manifest_path)
    man = _read_csv_audited(manifest_path, why="instance manifest")
    c_rho = find_col(man.columns, "rho_target")
    c_seed = find_col(man.columns, "seed")
    c_h = find_col(man.columns, "hhcp")
    c_s = find_col(man.columns, "sol")
    formal_provenance = [
        "feasible", "rho_global", "rho_used_mean", "rho_used_max", "N",
        "cap", "src", "src_sha256", "h_sha256", "s_sha256",
        "generator", "generator_sha256", "family_runner",
        "family_runner_sha256"]
    if formal:
        missing = [c for c in formal_provenance if c not in man.columns]
        require(not missing,
                "formal R1b manifest lacks required scenario provenance fields",
                missing=missing)
    lattice_preflight = preflight_manifest_lattice(
        man, man_dir, rho_column=c_rho, seed_column=c_seed,
        h_column=c_h, s_column=c_s,
        allow_fractional=allow_fractional, formal=formal)
    if rho_filter is not None:
        man = man[np.isclose(man[c_rho].astype(float), float(rho_filter))]
        require(not man.empty, "rho filter selected no manifest row",
                rho=rho_filter)
    selected_lattice_preconditions = {
        _scenario_key(row[c_rho], row[c_seed]):
        lattice_preflight["premises"][_scenario_key(row[c_rho], row[c_seed])]
        for _, row in man.iterrows()}
    assert_preflight_receipts_unchanged(
        lattice_preflight, where="immediately before the first R1b solve")

    inputs, axis, ntasks, scenario_pool_counts, deviation_constants, recs, diagnostic_rows = \
        ([manifest_path, DESIGN_MANIFEST]
         + list(lattice_preflight["input_paths"])), {}, {}, {}, {}, [], []
    for _, row in man.iterrows():
        row_rho = float(row[c_rho])
        row_seed = _exact_int_cell(
            row[c_seed], field=str(c_seed), where="R1b scenario manifest")
        hp = os.path.join(man_dir, str(row[c_h]))
        sp = os.path.join(man_dir, str(row[c_s]))
        for p in (hp, sp):
            require(os.path.isfile(p), "manifest points at a missing file", path=p)
            inputs.append(p)
        actual_h_sha = sha256_file(hp)
        actual_s_sha = sha256_file(sp)
        declared_h_sha = str(optional_manifest_value(
            row, man.columns, "h_sha256", actual_h_sha)).strip()
        declared_s_sha = str(optional_manifest_value(
            row, man.columns, "s_sha256", actual_s_sha)).strip()
        require(declared_h_sha == actual_h_sha and declared_s_sha == actual_s_sha,
                "scenario H/S bytes do not match their manifest SHA256",
                hhcp=hp, sol=sp, declared_h_sha256=declared_h_sha,
                actual_h_sha256=actual_h_sha, declared_s_sha256=declared_s_sha,
                actual_s_sha256=actual_s_sha)
        source_bindings_present = all(
            c in man.columns for c in (
                "src", "src_sha256", "generator", "generator_sha256",
                "family_runner", "family_runner_sha256"))
        if source_bindings_present:
            for label, path_col, hash_col in (
                    ("source", "src", "src_sha256"),
                    ("generator", "generator", "generator_sha256"),
                    ("family_runner", "family_runner", "family_runner_sha256")):
                source_path = resolve_manifest_source_path(
                    row[path_col], man_dir)
                if source_path is None:
                    # Release marker names this source as withheld; its recorded
                    # digest is the only provenance available here.
                    continue
                require(sha256_file(source_path) == str(row[hash_col]).strip(),
                        "manifest-declared source bytes disagree with SHA256",
                        rho=row_rho, seed=row_seed,
                        source_role=label, path=source_path,
                        declared_sha256=str(row[hash_col]).strip(),
                        actual_sha256=sha256_file(source_path))
                inputs.append(source_path)
        parser = HHCPFullParser(hp)
        sa = load_static(sp)
        S = set(parser.nurses.keys())
        scenario_n = _exact_int_cell(
            optional_manifest_value(row, man.columns, "N", len(S)),
            field="N", where="R1b scenario manifest rho={} seed={}".format(
                row_rho, row_seed))
        require(scenario_n > 0,
                "scenario caregiver-pool size must be positive",
                hhcp=hp, manifest_or_parsed_N=scenario_n)
        if formal:
            require(scenario_n == len(S),
                    "formal R1b manifest N differs from the parsed caregiver pool",
                    rho=row_rho, seed=row_seed,
                    manifest_N=scenario_n, parsed_caregivers=len(S))
        actives = active_absentees(parser, sa, max_absent)
        require(bool(actives), "no active future-task holder in the recovery horizon",
                hhcp=hp)
        rho, seed = row_rho, row_seed
        flat = assert_incumbent_flat_feasible(
            parser, sa, label=f"rho={rho} seed={seed}")
        if "feasible" in man.columns:
            require(manifest_bool(row["feasible"], "feasible"),
                    "scenario manifest declares the incumbent infeasible",
                    rho=rho, seed=seed)
        all_prov_present = all(c in man.columns for c in formal_provenance)
        scenario_meta = {
            "incumbent_flat_feasible": bool(flat["incumbent_flat_feasible"]),
            "eligible_disruption": True,
            "eligible_disruptions_in_scenario": len(actives),
            "absence_rule": ABSENCE_RULE,
            "active_caregiver_count": scenario_n,
            "rho_target": rho,
            "rho_global": float(optional_manifest_value(
                row, man.columns, "rho_global", rho)),
            "rho_used_mean": float(optional_manifest_value(
                row, man.columns, "rho_used_mean", rho)),
            "rho_used_max": float(optional_manifest_value(
                row, man.columns, "rho_used_max", rho)),
            "scenario_N": scenario_n,
            "scenario_cap": float(optional_manifest_value(
                row, man.columns, "cap", H_MAX)),
            "source_file": str(optional_manifest_value(
                row, man.columns, "src", os.path.basename(hp))),
            "src_sha256": str(optional_manifest_value(
                row, man.columns, "src_sha256", actual_h_sha)),
            "h_sha256": actual_h_sha,
            "s_sha256": actual_s_sha,
            "generator_file": str(optional_manifest_value(
                row, man.columns, "generator", "fixture_not_declared")),
            "generator_sha256": str(optional_manifest_value(
                row, man.columns, "generator_sha256", "fixture_not_declared")),
            "family_runner_file": str(optional_manifest_value(
                row, man.columns, "family_runner", "fixture_not_declared")),
            "family_runner_sha256": str(optional_manifest_value(
                row, man.columns, "family_runner_sha256", "fixture_not_declared")),
            "manifest_provenance_mode": (
                "declared_and_all_sources_rehashed" if all_prov_present
                else "fixture_optional_defaults_HS_rehashed"),
        }
        axis[(rho, seed)] = actives
        scenario_pool_counts[(rho, seed)] = scenario_n
        for a in actives:
            eligibility = assert_incumbent_eligible(
                parser, sa, S, a,
                label=f"rho={rho} seed={seed} absent={a}")
            ntasks[(rho, seed, a)] = sum(
                len(slot_tasks(parser, sa, q)) for q in future_slots(sa))
            slots = future_slots(sa)
            require(bool(slots), "no future slot in this instance",
                    absent=a, t_now=T_NOW)
            preliminary = {
                regime: WeightedModel(
                    parser, sa, S, a, regime, slots, policy,
                    rho=rho, seed=seed)
                for regime in authorized_methods}
            dev_values = {
                _strict_integer_value(
                    model.dev_const, "source-derived deviation constant",
                    where="R1b:rho={}:seed={}:absent={}:method={}".format(
                        rho, seed, a, regime))
                for regime, model in preliminary.items()}
            require(len(dev_values) == 1 and min(dev_values) >= 0,
                    "source-derived deviation constant differs by method",
                    rho=rho, seed=seed, absent=a,
                    values=sorted(dev_values))
            deviation_constants[(rho, seed, int(a))] = next(iter(dev_values))
            duration_contract = {
                regime: model.integral_durations
                for regime, model in preliminary.items()}
            trusted_premise = selected_lattice_preconditions[
                _scenario_key(rho, seed)]
            require(all(
                        bool(value)
                        == trusted_premise["integral_lattice_data"]
                        for value in duration_contract.values()),
                    "WeightedModel lattice premise differs from the "
                    "family-wide source preflight",
                    rho=rho, seed=seed, absent=a,
                    model_integrality_by_regime=duration_contract,
                    trusted_premise=trusted_premise)
            require(allow_fractional or all(duration_contract.values()),
                    "non-integral effective durations detected before the first "
                    "solver call; the lexicographic rows and W/Z objective "
                    "lattices are not authorized",
                    rho=rho, seed=seed, absent=a,
                    integral_durations_by_regime=duration_contract,
                    allow_fractional_durations=allow_fractional)
            nesting = assert_A1_candidates_subset_A2(
                preliminary["A1"], preliminary["A2"], a)
            umin = {
                "A1": preliminary["A1"].min_U("preliminary_minU_A1"),
                "A2": preliminary["A2"].min_U("preliminary_minU_A2"),
            }
            u_min_a1 = int(umin["A1"]["U"])
            u_min_a2 = int(umin["A2"]["U"])
            require(u_min_a2 <= u_min_a1,
                    "U_min_A2 > U_min_A1 is impossible under checked A1 subset A2",
                    rho=rho, seed=seed, absent=a,
                    U_min_A1=u_min_a1, U_min_A2=u_min_a2)
            u_common = u_min_a1
            common_meta = {
                "U_min_A1": u_min_a1, "U_min_A2": u_min_a2,
                "U_common": u_common,
                "service_policy": (
                    "U_common=U_min_A1; equality in A1 and A2 for every w4"),
                "placeable_A1": bool(u_min_a1 == 0),
                "placeable_A2": bool(u_min_a2 == 0),
                "u_min_solver_status_chain": "A1:{}|A2:{}".format(
                    umin["A1"]["status"], umin["A2"]["status"]),
                "umin_A1_solver_certificate_role": UMIN_A1_ROLE,
                "umin_A1_solver_evidence_scope": (
                    "single_independent_horizon_min_U_solve"),
                "umin_A2_solver_certificate_role": UMIN_A2_ROLE,
                "umin_A2_solver_evidence_scope": (
                    "single_independent_horizon_min_U_solve"),
                "n_incumbent_pairs_checked": int(
                    eligibility["n_incumbent_pairs_checked"]),
                "n_tasks_nested_A1_subset_A2": int(
                    nesting["n_tasks_nested_A1_subset_A2"]),
            }
            common_meta.update(_certificate_columns(
                "umin_A1_", umin["A1"]["solver"],
                umin["A1"]["exact_objective_certificate"],
                umin["A1"]["solver_fun_raw"]))
            common_meta.update(_certificate_columns(
                "umin_A2_", umin["A2"]["solver"],
                umin["A2"]["exact_objective_certificate"],
                umin["A2"]["solver_fun_raw"]))
            for regime, model in preliminary.items():
                for d in model.diagnostics:
                    d.update(rho=rho, seed=seed, method=regime,
                             w4="", common_service_preliminary=True,
                             snap_id=snap_id or "")
                    diagnostic_rows.append(d)
            for regime in authorized_methods:
                for w4 in authorized_w4:
                    model = WeightedModel(
                        parser, sa, S, a, regime, slots, policy,
                        rho=rho, seed=seed)
                    r = run_disruption(
                        model, w4, u_common, policy,
                        requested_mode=mode,
                        canonical_assignment=canonical_assignment)
                    diagnostics = r.pop("_diagnostics")
                    require(allow_fractional or
                            r["w_level_separation"] == "integral_durations",
                            "non-integral effective durations: consecutive "
                            "achievable Z / overtime levels may be closer than "
                            "LEX_TOL, so the lexicographic rows cannot be trusted",
                            absent=a, regime=regime, w4=w4)
                    recs.append(dict(rho=rho, seed=seed, absent=int(a),
                                     method=regime, w4=int(w4),
                                     lex_order=LEX_ORDER_LABEL, lex_tol=LEX_TOL,
                                     cap_allowance_rule=CAP_ALLOWANCE_RULE,
                                     decomposition="horizon",
                                     formal_run=bool(formal),
                                     absence_cap_mode=(
                                         "all_active_no_cap" if max_absent is None
                                         else f"fixture_cap_{int(max_absent)}"),
                                     solver_presolve=False,
                                     solver_policy_manifest_sha256=sha256_file(
                                         DESIGN_MANIFEST),
                                     snap_id=snap_id or "", **scenario_meta,
                                     **common_meta, **r))
                    for d in diagnostics:
                        d.update(rho=rho, seed=seed, method=regime,
                                 snap_id=snap_id or "")
                        diagnostic_rows.append(d)
        if verbose:
            print(f"  rho={rho:.2f} s{seed}: {len(actives)} disruption(s) solved",
                  flush=True)

    assert_preflight_receipts_unchanged(
        lattice_preflight, where="after all R1b solves and before publication")
    raw = pd.DataFrame(recs)
    lead = ["rho", "seed", "absent", "method", "w4"]
    raw = raw[lead + [c for c in raw.columns if c not in lead]]
    diagnostics_df = pd.DataFrame(diagnostic_rows)
    summary, prov = build_summary(raw)
    design = build_design(
        axis, list(dict.fromkeys(inputs)), contract=contract,
        scenario_pool_counts=scenario_pool_counts,
        scenario_lattice_preconditions=selected_lattice_preconditions,
        deviation_constants=deviation_constants,
        formal=formal, snap_id=snap_id)
    design["n_tasks_future"] = ntasks
    hashes, reports, log, summary, prov = write_validated(
        out_dir, raw, summary, prov, diagnostics_df, design, policy,
        snap_id=snap_id, formal=formal, verbose=verbose)
    if verbose:
        if formal:
            print(f"  [hashes] SNAP identifier: {snap_id}")
        else:
            print("  [hashes] SNAP identifier: none (rehearsal only; SNAP closes "
                  "before R0 and formal R1b)")
    return raw, summary, prov, design, hashes, reports, log


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--manifest", default=os.path.join(HERE, "rho_manifest.csv"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--merge-shards", nargs="+", default=None, metavar="SHARD_DIR",
        help="merge exactly six already-landed formal rho shard directories; "
             "the official manifest axis and every shard receipt are revalidated")
    ap.add_argument("--mode", choices=DECOMPOSITION_MODES,
                    default=DEFAULT_DECOMPOSITION)
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument(
        "--max-absent", type=int, default=None,
        help="rehearsal/fixture-only cap; formal R1b rejects this option and "
             "covers every active future task holder")
    ap.add_argument("--snap-id", default=None,
                    help="64-hex SHA256 of the content-addressed SNAP archive; "
                         "required for a formal R1b promotion")
    ap.add_argument("--time-limit-s", type=float, default=None,
                    help="formal R1b must pass the pre-registered value exactly")
    cg = ap.add_mutually_exclusive_group()
    cg.add_argument("--canonical-assignment", dest="canonical_assignment",
                    action="store_true",
                    help="explicitly opt into the costly deterministic assignment "
                         "tie-break and persist its hash; assignment is not an "
                         "estimand")
    cg.add_argument("--no-canonical-assignment", dest="canonical_assignment",
                    action="store_false",
                    help="skip expensive assignment canonicalization; hashes are "
                         "then empty and not reproducibility evidence")
    ap.set_defaults(canonical_assignment=None)
    ap.add_argument("--allow-fractional-durations", action="store_true")
    ap.add_argument("--promote-canonical", action="store_true",
                    help="required to write into experiments/journal_p0/; without "
                         "it a canonical target is refused")
    ap.add_argument("--formal", action="store_true",
                    help="formal R1b semantics for any output directory; requires "
                         "--mode horizon, all active absentees (omit "
                         "--max-absent), --snap-id and --time-limit-s; canonical "
                         "assignment remains off by the registered policy")
    a = ap.parse_args(argv)
    out_dir = os.path.abspath(a.out_dir) if a.out_dir else HERE
    if os.path.abspath(out_dir) == HERE and not a.promote_canonical:
        fatal("refusing to write the canonical e4 artifacts without "
              "--promote-canonical (regeneration of e4_raw.csv is stage R1b, and "
              "R1b has not been authorised)", out_dir=out_dir)
    if a.merge_shards is not None:
        require(a.formal or a.promote_canonical,
                "--merge-shards is a formal-only operation; pass --formal")
        require(a.mode == FORMAL_DECOMPOSITION and a.max_absent is None
                and a.rho is None,
                "formal shard merge requires --mode horizon, no --max-absent, "
                "and no --rho filter",
                mode=a.mode, max_absent=a.max_absent, rho=a.rho)
        require(a.canonical_assignment in (None, False),
                "formal shard merge requires --no-canonical-assignment")
        merge_formal_shards(
            a.manifest, out_dir, a.merge_shards,
            snap_id=a.snap_id, cli_time_limit_s=a.time_limit_s)
        return 0
    run(a.manifest, out_dir, mode=a.mode, max_absent=a.max_absent,
        rho_filter=a.rho, canonical_assignment=a.canonical_assignment,
        allow_fractional=a.allow_fractional_durations,
        formal=(a.formal or a.promote_canonical), snap_id=a.snap_id,
        cli_time_limit_s=a.time_limit_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
