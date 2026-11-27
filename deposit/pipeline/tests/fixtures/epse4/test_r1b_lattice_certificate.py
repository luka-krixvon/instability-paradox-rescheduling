#!/usr/bin/env python3
"""Solver-free hostile tests for the R1b objective-lattice certificate."""
from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
JOURNAL = os.path.join(ROOT, "experiments", "journal_p0")
EXPERIMENTS = os.path.join(ROOT, "experiments")
for path in (JOURNAL, EXPERIMENTS):
    if path not in sys.path:
        sys.path.insert(0, path)

import run_e4 as E4  # noqa: E402
from s2_core import PipelineFailure, full_objective_gap  # noqa: E402


class R1bLatticeCertificateTests(unittest.TestCase):
    ROLE_PREFIXES = (
        "weighted_", "w_refine_", "tie_break_", "umin_A1_", "umin_A2_")

    REAL_FAILURES = (
        # w4, P_linearized, B_linearized, C, P_full
        (100, -17400.0, -17401.315789473683, 18100.0, 700.0),
        (0, 275.0, 274.999998, 0.0, 275.0),
        (100, -12000.0, -12000.626923019945, 17100.0, 5100.0),
        (5, 775.0, 774.9257322311668, 855.0, 1630.0),
        (50, -4835.0, -4835.439648281382, 9100.0, 4265.0),
    )

    RECOVERY04_BOUND_OVERSHOOTS = (
        # stage, w4, P_linearized, B_linearized, C, P_full
        ("Z", 5, -390.0, -389.9999999999995, 815.0, 425.0),
        ("W", None, 0.0, 9.769962616701367e-15, 0.0, 0.0),
        ("W", None, 0.0, 1.8010456216615096e-14, 0.0, 0.0),
        ("Z", 5, 1125.0, 1125.0000000000125, 0.0, 1125.0),
    )

    RECOVERY05_SAME_LEVEL_READBACKS = (
        # Exact recovery-05 failures: (P readback, solver F == B).
        (112.0, 111.99999619999844),
        (173.0, 172.99999619999838),
        (132.0, 131.99999619999818),
    )

    class ExactRowHarness:
        """One-variable harness that exercises production ``_solve`` guards.

        The fake status-0 solver returns an integral point and internally
        consistent F/B evidence.  Only the caller-supplied lexicographic row is
        false, so accepting the point would demonstrate that a floating solver
        row was trusted without an exact readback check.
        """

        _solve = E4.WeightedModel._solve
        _assert_exact_readback_lex_feasible = (
            E4.WeightedModel._assert_exact_readback_lex_feasible)

        def __init__(self, *, U, W, D, w4=0):
            self._solution = {
                "U": int(U), "W": float(W), "D": int(D),
                "Z": (float(E4.W2) * float(W) + float(w4) * int(D)
                      + float(E4.W5) * int(U)),
                "n_tasks": 1, "assignment": {},
            }
            self.n = 1
            self.xpairs = [(0, 0, 0)]
            self.uidx = {}
            self.oidx = {}
            self._rows = [np.array([0.0])]
            self._lb = [-np.inf]
            self._ub = [np.inf]
            self.policy = E4.solver_policy()
            self.rho = 0.95
            self.seed = 901
            self.regime = "A1"
            self.absent = 4
            self.slots = [3]
            self.integral_durations = True
            self.dev_const = 0
            self.diagnostics = []

        def _objective_value(self, what, w4):
            if what == "U":
                return float(self._solution["U"])
            if what == "W":
                return float(self._solution["W"])
            if what == "D":
                return float(self._solution["D"] - self.dev_const)
            if what == "Z":
                return (float(E4.W2) * self._solution["W"]
                        + float(w4) * self._solution["D"]
                        - float(w4) * self.dev_const
                        + float(E4.W5) * self._solution["U"])
            raise AssertionError("unexpected objective stage: {}".format(what))

        def _obj(self, what, w4=None):
            return np.array([self._objective_value(what, w4)])

        def _z_row(self, w4):
            return np.array([self._solution["Z"] - float(w4) * self.dev_const])

        def _u_row(self):
            return np.array([float(self._solution["U"])])

        def _o_row(self):
            return np.array([float(self._solution["W"])])

        def _variable_bounds(self, _fixed=None):
            return E4.Bounds(np.array([0.0]), np.array([1.0]))

        def readback(self, _x, stage_label="", w4=None):
            result = dict(self._solution)
            result["stage"] = stage_label
            result["Z"] = (float(E4.W2) * result["W"]
                           + float(w4 or 0) * result["D"]
                           + float(E4.W5) * result["U"])
            return result

        def status_zero_result(self, what, w4=None):
            objective = self._objective_value(what, w4)
            return SimpleNamespace(
                status=0, x=np.array([1.0]), message="Optimal",
                fun=objective, mip_dual_bound=objective, mip_gap=0.0,
                mip_node_count=0)

    @staticmethod
    def certify(stage, p, b, c, *, solver_fun=None, w4=None, integral=True,
                exact_full=None, structural_floor=None,
                structural_source=None):
        if solver_fun is None:
            solver_fun = p
        if exact_full is None and math.isfinite(float(p + c)):
            nearest = int(round(float(p + c)))
            if float(p + c) == float(nearest):
                exact_full = nearest
                structural_floor = 0
                structural_source = {
                    "U": "nonnegative_unserved_count",
                    "W": "nonnegative_canonical_overtime",
                    "D": "nonnegative_assignment_deviation",
                    "Z": (
                        "fixed_U_common_plus_nonnegative_overtime_and_deviation"),
                }[stage]
        return E4.exact_objective_certificate(
            stage, p, b, c, solver_fun, w4=w4,
            integral_durations=integral,
            exact_full_objective_integer=exact_full,
            structural_lower_bound_integer=structural_floor,
            structural_lower_bound_source=structural_source,
            where="negative-test")

    @staticmethod
    def persisted_row(prefix, certificate, p, b, c, *, w4):
        stage = certificate["objective_stage"]
        exact = int(certificate["exact_full_objective_integer"])
        dev_const = int(c) if stage == "D" else (
            int(c // w4) if stage == "Z" and w4 else 0)
        overtime = 0
        deviation = 0
        if stage == "W":
            overtime = exact
        elif stage == "D":
            deviation = exact
        elif stage == "Z":
            if w4 and exact % int(w4) == 0:
                deviation = exact // int(w4)
            else:
                overtime = exact // int(E4.W2)
        weighted_z = int(E4.W2) * overtime + int(w4) * deviation
        row = {
            "w_level_separation": "integral_durations",
            "w4": w4,
            "overtime": overtime,
            "deviation": deviation,
            "unserved": 0,
            "U": 0,
            "U_common": 0,
            "U_min_A1": exact if prefix == "umin_A1_" else 0,
            "U_min_A2": exact if prefix == "umin_A2_" else 0,
            "weighted_objective_Z": weighted_z,
            "tie_break_deviation_constant_C": dev_const,
            "n_tasks": max(1, dev_const),
            prefix + "obj_incumbent": p,
            prefix + "solver_fun_raw": p,
            prefix + "mip_dual_bound": b,
            prefix + "deviation_constant_C": c,
            prefix + "exact_certificate_method": certificate["method"],
            prefix + "exact_certificate_objective_stage": (
                certificate["objective_stage"]),
            prefix + "lattice_step": certificate["step"],
            prefix + "lattice_offset": certificate["offset"],
            prefix + "lattice_bound_ulps": certificate["bound_ulps"],
            prefix + "lattice_boundary_margin": (
                certificate["boundary_margin"]),
            prefix + "lattice_safe_dual_bound_full_objective": (
                certificate["safe_dual_bound_full_objective"]),
            prefix + "lattice_closed_dual_bound_full_objective": (
                certificate["closed_dual_bound_full_objective"]),
            prefix + "lattice_primal_level_full_objective": (
                certificate["primal_level_full_objective"]),
            prefix + "exact_certificate_lower_bound_source": (
                certificate["lower_bound_source"]),
            prefix + "exact_certificate_structural_lower_bound_full_objective": (
                certificate["structural_lower_bound_full_objective"]),
            prefix + "exact_certificate_exact_full_objective_integer": (
                certificate["exact_full_objective_integer"]),
            prefix + "lattice_proves_exact_optimality": (
                certificate["proves_exact_optimality"]),
            prefix + "lattice_justification": certificate["justification"],
        }
        return row

    @classmethod
    def setUpClass(cls):
        """Build a valid raw family without ever entering scipy.optimize.milp.

        The production fixture's H/S parsing, source-derived lattice premise,
        row assembly and summary/provenance builders are retained.  Only the
        MILP-bearing model and the disk publication step are replaced, so the
        hostile mutations below exercise the real ``extra_checks`` gates.
        """
        class SolverFreeWeightedModel:
            def __init__(self, parser, slot_assign, nurses, absent, regime,
                         slots, policy, **_kwargs):
                self.absent = absent
                self.regime = regime
                self.slots = list(slots)
                self.policy = policy
                self.integral_durations = True
                self.diagnostics = []
                # A1 == A2 is sufficient for the executable subset guard and
                # preserves the independently derived future-task count.
                self.cand = {
                    (slot, task): [0]
                    for slot in self.slots
                    for task in E4.slot_tasks(parser, slot_assign, slot)
                }
                self.n_tasks = len(self.cand)
                self.dev_const = 0.0

            def _certificate(self, objective_stage, *, w4=None):
                result = E4.SolverResult(
                    stage=E4.STAGE,
                    status_code=0,
                    status_name=self.policy.status_name(0),
                    proven_optimal=True,
                    primal=0.0,
                    dual_bound=0.0,
                    deviation_constant_C=0.0,
                    gap_solver_raw=0.0,
                    gap_full_objective=None,
                    gap_full_objective_undefined_reason=(
                        "zero_full_objective_denominator"),
                    stop_reason="proven_optimal",
                    wall_time_s=0.0,
                    time_limit_s=self.policy.time_limit_s,
                    nodes=0,
                    assignment_hash=None,
                )
                certificate = E4.exact_objective_certificate(
                    objective_stage, 0.0, 0.0, 0.0, 0.0,
                    w4=w4, integral_durations=True,
                    exact_full_objective_integer=0,
                    structural_lower_bound_integer=0,
                    structural_lower_bound_source={
                        "U": "nonnegative_unserved_count",
                        "W": "nonnegative_canonical_overtime",
                        "D": "nonnegative_assignment_deviation",
                        "Z": (
                            "fixed_U_common_plus_nonnegative_overtime_and_deviation"),
                    }[objective_stage],
                    where="solver-free-hostile-fixture")
                return result, certificate

            def min_U(self, _stage_label="preliminary_minU"):
                result, certificate = self._certificate("U")
                return {
                    "U": 0,
                    "status": 0,
                    "solver": result,
                    "solver_fun_raw": 0.0,
                    "exact_objective_certificate": certificate,
                }

            def solve_lex(self, w4, u_eq=None, canonical_assignment=False):
                if canonical_assignment:
                    raise AssertionError(
                        "solver-free hostile fixture does not enable canonicalization")
                if u_eq != 0:
                    raise AssertionError("solver-free fixture expects U_common=0")
                weighted, weighted_certificate = self._certificate("Z", w4=w4)
                w_refine, w_refine_certificate = self._certificate("W")
                tie_break, tie_break_certificate = self._certificate("D")
                return {
                    "U": 0,
                    "W": 0.0,
                    "D": 0,
                    "Z": 0.0,
                    "n_tasks": self.n_tasks,
                    "assignment": {},
                    "status_chain": "0|0|0",
                    "lex_stage_chain": "|".join(
                        E4.EXECUTABLE_MAIN_STAGE_NAMES),
                    "primary_weighted_solver": weighted,
                    "primary_weighted_solver_fun_raw": 0.0,
                    "primary_weighted_exact_certificate": weighted_certificate,
                    "w_refine_solver": w_refine,
                    "w_refine_solver_fun_raw": 0.0,
                    "w_refine_exact_certificate": w_refine_certificate,
                    "final_d_tie_break_solver": tie_break,
                    "final_d_tie_break_solver_fun_raw": 0.0,
                    "final_d_tie_break_exact_certificate": tie_break_certificate,
                }

        def skip_publication(_out_dir, _raw, summary, provenance,
                             _diagnostics, _design, _policy, **_kwargs):
            return {}, {}, [], summary, provenance

        manifest = os.path.join(HERE, "manifest_epse4_A.csv")
        with tempfile.TemporaryDirectory(
                prefix="r1b-solver-free-hostile-family-") as out_dir:
            with mock.patch.object(E4, "WeightedModel", SolverFreeWeightedModel), \
                    mock.patch.object(E4, "write_validated", skip_publication), \
                    mock.patch.object(
                        E4, "milp",
                        side_effect=AssertionError(
                            "solver-free hostile tests must not call milp")):
                result = E4.run(
                    manifest, out_dir, mode="horizon", max_absent=1,
                    canonical_assignment=False, verbose=False)
        cls.raw, cls.summary, cls.provenance, cls.design = result[:4]
        source_premises = cls.design["scenario_lattice_preconditions"]
        cls.input_integral_durations = next(iter(source_premises.values()))[
            "integral_lattice_data"]
        if cls.input_integral_durations is not True:
            raise AssertionError(
                "the fixture H/S bytes must independently authorize an integer lattice")
        # Prove that failures below are caused by the mutation, not an invalid
        # synthetic baseline.
        E4.extra_checks(
            cls.raw, cls.summary, cls.provenance, cls.design, verbose=False)

    def test_all_recovery03_failures_close_without_rewriting_raw_gap(self):
        for w4, p, b, c, p_full in self.REAL_FAILURES:
            with self.subTest(w4=w4, p_full=p_full):
                gap_before, reason_before = full_objective_gap(
                    p, b, c, where="negative-test")
                certificate = self.certify("Z", p, b, c, w4=w4)
                gap_after, reason_after = full_objective_gap(
                    p, b, c, where="negative-test")
                self.assertEqual(
                    certificate["method"], E4.STATUS0_LATTICE_METHOD)
                self.assertFalse(certificate["proves_exact_optimality"])
                self.assertEqual(certificate["bound_ulps"], 0)
                self.assertEqual(certificate["step"], 5.0)
                self.assertEqual(
                    certificate["closed_dual_bound_full_objective"], p_full)
                self.assertNotEqual(gap_before, 0.0)
                self.assertEqual((gap_before, reason_before),
                                 (gap_after, reason_after))

    def test_all_recovery04_bound_overshoots_close_without_rewriting_raw_gap(self):
        for stage, w4, p, b, c, p_full in self.RECOVERY04_BOUND_OVERSHOOTS:
            for solver_fun in (p, b):
                with self.subTest(
                        stage=stage, p_full=p_full, solver_fun=solver_fun):
                    gap_before, reason_before = full_objective_gap(
                        p, b, c, where="negative-test")
                    certificate = self.certify(
                        stage, p, b, c, solver_fun=solver_fun, w4=w4)
                    gap_after, reason_after = full_objective_gap(
                        p, b, c, where="negative-test")
                    expected_method = (
                        E4.STRUCTURAL_LOWER_BOUND_METHOD
                        if p_full == 0.0 else E4.STATUS0_LATTICE_METHOD)
                    self.assertEqual(certificate["method"], expected_method)
                    self.assertEqual(
                        certificate["closed_dual_bound_full_objective"],
                        p_full)
                    self.assertEqual(
                        (gap_before, reason_before),
                        (gap_after, reason_after))

    def test_recovery05_solver_readbacks_identify_the_same_W_level(self):
        for primal, solver_fun in self.RECOVERY05_SAME_LEVEL_READBACKS:
            with self.subTest(primal=primal, solver_fun=solver_fun):
                raw_delta = abs(solver_fun - primal)
                certificate = self.certify(
                    "W", primal, solver_fun, 0.0,
                    solver_fun=solver_fun)
                self.assertGreater(raw_delta, 3.8e-6)
                self.assertLess(raw_delta, 3.81e-6)
                self.assertEqual(
                    certificate["method"], E4.STATUS0_LATTICE_METHOD)
                self.assertFalse(certificate["proves_exact_optimality"])
                self.assertEqual(certificate["bound_ulps"], 0)
                self.assertEqual(certificate["step"], 1.0)
                self.assertEqual(
                    certificate["closed_dual_bound_full_objective"], primal)

    def test_recovery06_observed_eleven_ulp_objective_reconstruction(self):
        # F and X are exact values logged by the rejected 221aef66 smoke.
        # P=-100 and w4=10 deliberately provide a synthetic compatible
        # Z-lattice center; the failed exception did not print P, B, or w4.
        p = -100.0
        solver_fun = -100.0000000000013
        objective_from_x = -100.00000000000114
        result = E4.assert_live_objective_reconstruction(
            "Z", solver_fun, objective_from_x, p, w4=10,
            integral_durations=True, where="rejected-221a-smoke-regression")
        self.assertEqual(result["method"],
                         "live_objective_same_lattice_cell_v1")
        self.assertEqual(result["lattice_step"], 5.0)
        self.assertEqual(abs(solver_fun - objective_from_x),
                         11.0 * math.ulp(100.0))
        self.assertGreater(abs(solver_fun - objective_from_x),
                           E4._ulp_margin(solver_fun, objective_from_x))
        self.assertLess(result["F_X_delta"], result["half_lattice_step"])

    def test_live_objective_pairwise_half_cell_guards(self):
        # Small opposite-side rounding remains unambiguously in P's cell.
        passed = E4.assert_live_objective_reconstruction(
            "Z", 699.0, 701.0, 700.0, w4=5,
            integral_durations=True, where="small-opposite-side")
        self.assertLess(passed["F_X_delta"], 2.5)

        cases = (
            # F and X are individually inside P's cell but too far apart.
            (697.55, 702.45, 700.0, "solver fun and live"),
            # Directed-up exact half-step must fail the strict comparison.
            (700.0, 702.5, 700.0, "live c @ res.x"),
            # F/X exact half-step is independently strict as well.
            (702.5, 700.0, 700.0, "solver fun and live"),
            # A jointly corrupted adjacent cell is rejected by the P anchor.
            (705.0, 705.0, 700.0, "live c @ res.x"),
        )
        for solver_fun, objective_from_x, p, message in cases:
            with self.subTest(F=solver_fun, X=objective_from_x, P=p), \
                    self.assertRaisesRegex(PipelineFailure, message):
                E4.assert_live_objective_reconstruction(
                    "Z", solver_fun, objective_from_x, p, w4=5,
                    integral_durations=True, where="hostile-pairwise-cell")

    def test_live_pair_guards_do_not_replace_certificate_F_P_guard(self):
        # X is close to P and F is close to X, but F is outside P's half-cell.
        # The live helper must pass its two edges and the existing certificate
        # must still reject the third edge; neither gate subsumes the other.
        p, objective_from_x, solver_fun = 700.0, 699.8, 697.4
        E4.assert_live_objective_reconstruction(
            "Z", solver_fun, objective_from_x, p, w4=5,
            integral_durations=True, where="third-edge-regression")
        with self.assertRaisesRegex(
                PipelineFailure, "same objective-lattice level"):
            E4.exact_objective_certificate(
                "Z", p, p, 0.0, solver_fun, w4=5,
                integral_durations=True,
                exact_full_objective_integer=700,
                structural_lower_bound_integer=0,
                structural_lower_bound_source="nonnegative_weighted_objective",
                where="third-edge-regression")

    def test_live_gate_runs_inside_production_solve(self):
        harness = self.ExactRowHarness(U=0, W=0, D=0)
        corrupted = SimpleNamespace(
            status=0, x=np.array([1.0]), message="Optimal",
            fun=1.0, mip_dual_bound=1.0, mip_gap=0.0,
            mip_node_count=0)
        with mock.patch.object(E4, "milp", return_value=corrupted) as milp, \
                mock.patch.object(
                    harness, "_obj", return_value=np.array([1.0])), \
                self.assertRaisesRegex(PipelineFailure, "live c @ res.x"):
            harness._solve("W", stage_label="hostile_live_vector_gate")
        milp.assert_called_once()

    def test_large_C_cannot_hide_live_raw_objective_disagreement(self):
        huge_c = float(2 ** 55)
        p, objective_from_x, solver_fun = 0.0, 1.0, 1.0
        self.assertEqual(float(p + huge_c), float(objective_from_x + huge_c))
        with self.assertRaisesRegex(PipelineFailure, "live c @ res.x"):
            E4.assert_live_objective_reconstruction(
                "W", solver_fun, objective_from_x, p,
                integral_durations=True, where="large-C-cancellation")

    def test_live_objective_reconstruction_rejects_nonfinite(self):
        for field, values in (
                ("F", (math.nan, 0.0, 0.0)),
                ("X", (0.0, math.inf, 0.0)),
                ("P", (0.0, 0.0, -math.inf))):
            with self.subTest(field=field), self.assertRaises(PipelineFailure):
                E4.assert_live_objective_reconstruction(
                    "W", values[0], values[1], values[2],
                    integral_durations=True, where="hostile-nonfinite")

    def test_non_lattice_objective_retains_eight_ulp_identity(self):
        inside = 100.0 + 8.0 * math.ulp(100.0)
        result = E4.assert_live_objective_reconstruction(
            "Z", 100.0, inside, 100.0, w4=5,
            integral_durations=False, where="fractional-fixture")
        self.assertEqual(result["method"],
                         "non_lattice_eight_ulp_identity_v1")
        outside = math.nextafter(inside, math.inf)
        with self.assertRaisesRegex(PipelineFailure, "non-lattice solver"):
            E4.assert_live_objective_reconstruction(
                "Z", 100.0, outside, 100.0, w4=5,
                integral_durations=False, where="fractional-fixture")

    def test_recovery06_smoke_uses_attained_structural_lower_bound(self):
        residual = 4.884981308350699e-14
        certificate = self.certify(
            "W", 0.0, residual, 0.0, solver_fun=residual,
            exact_full=0, structural_floor=0,
            structural_source="nonnegative_canonical_overtime")
        self.assertEqual(
            certificate["method"], E4.STRUCTURAL_LOWER_BOUND_METHOD)
        self.assertEqual(
            certificate["lower_bound_source"],
            "nonnegative_canonical_overtime")
        self.assertEqual(
            certificate["structural_lower_bound_full_objective"], 0.0)
        self.assertEqual(certificate["exact_full_objective_integer"], 0.0)
        self.assertTrue(certificate["proves_exact_optimality"])

    def test_recovery06_status0_180_ulp_smoke_tuple_is_consistent_not_a_proof(self):
        # Exact post-49811 smoke evidence.  B == F is 180 ULP above raw P,
        # beyond the obsolete 128-ULP envelope, but every raw pair is still
        # strictly inside D's unique half-step cell.  Raw evidence is immutable.
        p = -54.0
        b = f = -53.999999999994884
        c = 155.0
        raw = (p, b, f)
        # The legacy envelope selected max-operand scale, which is C here;
        # the overshoot is exactly 180 ULP at that registered tuple scale.
        self.assertEqual((b - p) / math.ulp(c), 180.0)
        self.assertGreater(b - p, E4._bound_ulp_margin(
            p, b, c, p + c, b + c, 1.0, 0.0))
        certificate = self.certify(
            "D", p, b, c, solver_fun=f, exact_full=101,
            structural_floor=0,
            structural_source="nonnegative_assignment_deviation")
        self.assertEqual((p, b, f), raw)
        self.assertEqual(certificate["method"], E4.STATUS0_LATTICE_METHOD)
        self.assertFalse(certificate["proves_exact_optimality"])
        self.assertEqual(certificate["bound_ulps"], 0)
        self.assertEqual(certificate["boundary_margin"], 0.0)
        self.assertEqual(
            certificate["lower_bound_source"],
            "solver_status0_unique_lattice_cell_no_exact_bound")
        self.assertEqual(certificate["safe_dual_bound_full_objective"], b + c)
        self.assertEqual(certificate["closed_dual_bound_full_objective"], 101.0)
        self.assertEqual(
            certificate["justification"],
            "solver_status0_exact_readback_unique_lattice_cell_not_exact_proof")

    def test_status0_lattice_requires_all_three_raw_pairs_strictly_inside_half_step(self):
        cases = (
            # Exact equality is forbidden by the strict half-step rule.
            (0.0, 0.0, 0.5, "F/P exact half-step"),
            (0.0, 0.5, 0.0, "B/P exact half-step"),
            (0.0, -0.49, 0.49, "opposite sides make F/B too wide"),
            (0.0, 1.0, 1.0, "joint adjacent lattice cell"),
            (0.0, -1.0, 0.0, "adjacent B lattice cell"),
        )
        for p, b, f, label in cases:
            with self.subTest(label=label), self.assertRaises(PipelineFailure):
                self.certify(
                    "W", p, b, 1.0e16, solver_fun=f,
                    exact_full=10 ** 16, structural_floor=0,
                    structural_source="nonnegative_canonical_overtime")

    def test_status0_magic_ulp_counts_neither_authorize_nor_select_method(self):
        p, c = -54.0, 155.0
        for ulps in (128, 180, 256):
            b = f = p + ulps * math.ulp(p)
            with self.subTest(ulps=ulps):
                certificate = self.certify(
                    "D", p, b, c, solver_fun=f, exact_full=101,
                    structural_floor=0,
                    structural_source="nonnegative_assignment_deviation")
                self.assertEqual(
                    certificate["method"], E4.STATUS0_LATTICE_METHOD)
                self.assertEqual(certificate["bound_ulps"], 0)
        with self.assertRaises(PipelineFailure):
            self.certify(
                "D", p, p + 0.5, c, solver_fun=p + 0.5,
                exact_full=101, structural_floor=0,
                structural_source="nonnegative_assignment_deviation")

    def test_status0_nonfinite_and_large_constant_cancellation_fail_closed(self):
        for field in ("P", "B", "F"):
            values = {"P": 0.0, "B": 0.0, "F": 0.0}
            values[field] = math.nan
            with self.subTest(field=field), self.assertRaises(PipelineFailure):
                self.certify(
                    "W", values["P"], values["B"], float(2 ** 55),
                    solver_fun=values["F"], exact_full=2 ** 55,
                    structural_floor=0,
                    structural_source="nonnegative_canonical_overtime")
        # Adding C erases this adjacent-cell disagreement in binary64; the raw
        # F/P and B/P comparisons must nevertheless reject it.
        huge = float(2 ** 55)
        self.assertEqual(huge + 1.0, huge)
        with self.assertRaises(PipelineFailure):
            self.certify(
                "W", 0.0, 1.0, huge, solver_fun=1.0,
                exact_full=2 ** 55, structural_floor=0,
                structural_source="nonnegative_canonical_overtime")

    def test_forged_D_structural_floor_101_is_not_cardinality_authority(self):
        # Smoke coordinate evidence: n_tasks=186, fixed U=0, and C=155.  Even
        # the basic cardinality arithmetic yields only 186-0-155=31; it cannot
        # authorize a structural D floor of 101.  D's registered analytic floor
        # remains zero, so forging exact==floor==101 must fail closed.
        coordinate = {"n_tasks": 186, "U": 0, "C": 155}
        self.assertEqual(
            coordinate["n_tasks"] - coordinate["U"] - coordinate["C"], 31)
        with self.assertRaisesRegex(
                PipelineFailure, "does not match its objective stage"):
            self.certify(
                "D", -54.0, -54.0, 155.0, solver_fun=-54.0,
                exact_full=101, structural_floor=101,
                structural_source="nonnegative_assignment_deviation")

    def test_nonzero_objective_cannot_borrow_structural_zero_certificate(self):
        p = 1.0
        diagnostic = p + 256.0 * math.ulp(p)
        certificate = self.certify(
            "W", p, diagnostic, 0.0, solver_fun=p,
            exact_full=1, structural_floor=0,
            structural_source="nonnegative_canonical_overtime")
        self.assertEqual(certificate["method"], E4.STATUS0_LATTICE_METHOD)
        self.assertFalse(certificate["proves_exact_optimality"])
        self.assertNotEqual(
            certificate["method"], E4.STRUCTURAL_LOWER_BOUND_METHOD)

    def test_nonzero_W_cannot_forge_matching_nonzero_structural_floor(self):
        with self.assertRaisesRegex(
                PipelineFailure, "does not match its objective stage"):
            self.certify(
                "W", 1.0, 1.0, 0.0, solver_fun=1.0,
                exact_full=1, structural_floor=1,
                structural_source="nonnegative_canonical_overtime")

    def test_coordinated_P_C_cancellation_fails_role_replay(self):
        prefix = "w_refine_"
        certificate = self.certify(
            "W", 0.0, 0.0, 0.0, solver_fun=0.0,
            exact_full=0, structural_floor=0,
            structural_source="nonnegative_canonical_overtime")
        row = self.persisted_row(prefix, certificate, 0.0, 0.0, 0.0, w4=0)
        huge = float(2 ** 55)
        row[prefix + "obj_incumbent"] = huge
        row[prefix + "solver_fun_raw"] = huge
        row[prefix + "mip_dual_bound"] = huge
        row[prefix + "deviation_constant_C"] = -huge
        with self.assertRaisesRegex(PipelineFailure, "raw P/C"):
            E4.replay_persisted_exact_certificate(
                row, prefix, "W", integral_durations=True,
                expected_deviation_constant=0, where="negative-test")

    def test_full_raw_gate_rejects_coordinated_weighted_cancellation(self):
        mutated = self.raw.copy(deep=True)
        idx = mutated.index[0]
        huge = float(2 ** 55)
        for suffix in ("obj_incumbent", "solver_fun_raw", "mip_dual_bound"):
            mutated.loc[idx, "weighted_" + suffix] = huge
        mutated.loc[idx, "weighted_deviation_constant_C"] = -huge
        mutated.loc[idx, "weighted_obj_incumbent_full_objective"] = 0.0
        mutated.loc[idx, "weighted_mip_dual_bound_full_objective"] = 0.0
        mutated.loc[idx, "weighted_mip_gap_solver_raw"] = 0.0
        with self.assertRaisesRegex(PipelineFailure, "raw P/C"):
            E4.extra_checks(
                mutated, self.summary, self.provenance, self.design,
                verbose=False)

    def test_full_raw_gate_rejects_wrong_stage_lower_bound_source(self):
        mutated = self.raw.copy(deep=True)
        mutated.loc[mutated.index[0],
                    "weighted_exact_certificate_lower_bound_source"] = (
                        "nonnegative_unserved_count")
        with self.assertRaises(PipelineFailure):
            E4.extra_checks(
                mutated, self.summary, self.provenance, self.design,
                verbose=False)

    def test_replay_rejects_source_derived_deviation_constant_mismatch(self):
        prefix = "tie_break_"
        p, b, c = -178.0, -178.00000002564101, 183.0
        certificate = self.certify("D", p, b, c, solver_fun=b)
        row = self.persisted_row(prefix, certificate, p, b, c, w4=0)
        with self.assertRaisesRegex(PipelineFailure, "re-parsed H/S authority"):
            E4.replay_persisted_exact_certificate(
                row, prefix, "D", integral_durations=True,
                expected_deviation_constant=182, where="negative-test")

    def test_structural_integer_authority_rejects_cancellation_and_bad_weights(self):
        with self.assertRaisesRegex(
                PipelineFailure, "exact integer objective readback"):
            self.certify(
                "W", float(2 ** 55), 0.0, -float(2 ** 55),
                solver_fun=float(2 ** 55), exact_full=1,
                structural_floor=0,
                structural_source="nonnegative_canonical_overtime")
        solution = {"U": 0, "W": 0, "D": 0}
        with self.assertRaisesRegex(PipelineFailure, "non-negative weights"):
            E4._structural_certificate_inputs(
                "Z", solution, w4=-5, u_eq=0, where="negative-test")

    def test_half_step_and_adjacent_solver_levels_fail_closed(self):
        cases = (
            # Exact half-step is rejected because the registered rule is strict.
            ("W", 112.0, 111.5, None),
            # A solver objective on the adjacent attainable level cannot be
            # presented as the exact readback's objective.
            ("W", 112.0, 111.0, None),
            ("Z", 700.0, 697.5, 5),
            ("Z", 700.0, 695.0, 5),
        )
        for stage, primal, solver_fun, w4 in cases:
            with self.subTest(
                    stage=stage, primal=primal, solver_fun=solver_fun), \
                    self.assertRaisesRegex(
                        PipelineFailure, "same objective-lattice level"):
                self.certify(
                    stage, primal, primal, 0.0,
                    solver_fun=solver_fun, w4=w4)

    def test_exact_same_level_boundary_equality_fails_closed(self):
        # No ULP envelope participates in the status-0 method: exact equality
        # at half a W step is rejected by the registered strict comparison.
        solver_fun = 0.5
        with self.assertRaisesRegex(
                PipelineFailure, "same objective-lattice level"):
            E4.exact_objective_certificate(
                "W", 0.0, 0.0, 0.0, solver_fun,
                integral_durations=True, where="negative-test")

    def test_recovery05_point65_csv_round_trip_replays_exactly(self):
        prefix = "tie_break_"
        p = -178.0
        b = -178.00000002564101
        c = 183.0
        solver_fun = b
        gap, reason = full_objective_gap(p, b, c, where="negative-test")
        self.assertIsNone(reason)
        certificate = self.certify(
            "D", p, b, c, solver_fun=solver_fun)
        row = self.persisted_row(prefix, certificate, p, b, c, w4=0)
        row[prefix + "solver_fun_raw"] = solver_fun
        row[prefix + "gap_full_objective"] = gap
        row[prefix + "gap_full_objective_undefined_reason"] = ""

        with tempfile.TemporaryDirectory(
                prefix="r1b-recovery05-round-trip-") as root:
            path = os.path.join(root, "e4_raw.csv")
            pd.DataFrame([row]).to_csv(path, index=False)
            landed = E4._read_csv_audited(
                path, why="hostile recovery-05 point65 exact replay")
        observed = landed.iloc[0]
        self.assertEqual(float(observed[prefix + "mip_dual_bound"]), b)
        self.assertEqual(float(observed[prefix + "solver_fun_raw"]), solver_fun)
        self.assertEqual(float(observed[prefix + "gap_full_objective"]), gap)
        replayed_gap, replayed_reason = full_objective_gap(
            observed[prefix + "obj_incumbent"],
            observed[prefix + "mip_dual_bound"],
            observed[prefix + "deviation_constant_C"],
            where="negative-test")
        self.assertEqual((replayed_gap, replayed_reason), (gap, None))
        replayed = E4.replay_persisted_exact_certificate(
            observed, prefix, "D",
            integral_durations=self.input_integral_durations,
            expected_deviation_constant=183,
            where="negative-test")
        self.assertEqual(
            replayed["closed_dual_bound_full_objective"], 5.0)

    def test_status_zero_exact_lex_rows_are_rechecked_after_readback(self):
        cases = (
            ("u_eq", self.ExactRowHarness(U=1, W=0, D=0),
             {"what": "U", "u_eq": 0}),
            ("z_le", self.ExactRowHarness(U=0, W=1, D=0, w4=0),
             {"what": "W", "w4": 0, "z_le": 4.0, "u_eq": 0}),
            ("w_le", self.ExactRowHarness(U=0, W=1, D=1, w4=0),
             {"what": "D", "w4": 0, "u_eq": 0, "w_le": 0.0}),
        )
        for label, harness, kwargs in cases:
            solver_result = harness.status_zero_result(
                kwargs["what"], kwargs.get("w4"))
            with self.subTest(row=label), mock.patch.object(
                    E4, "milp", return_value=solver_result) as milp, \
                    self.assertRaises(PipelineFailure):
                harness._solve(
                    stage_label="hostile_exact_{}_violation".format(label),
                    **kwargs)
            milp.assert_called_once()

    def test_status_one_timeout_remains_forbidden_without_retry(self):
        harness = self.ExactRowHarness(U=0, W=0, D=0)
        timeout = SimpleNamespace(
            status=1, x=np.array([1.0]), message="Time limit reached",
            fun=0.0, mip_dual_bound=-1.0, mip_gap=1.0,
            mip_node_count=17)
        with mock.patch.object(E4, "milp", return_value=timeout) as milp, \
                self.assertRaisesRegex(
                    PipelineFailure, "non-admissible status"):
            harness._solve(
                "W", stage_label="hostile_status1_timeout")
        milp.assert_called_once()

    def test_numeric_guards_reject_adjacent_or_common_mode_shift(self):
        for solver_fun, dual in (
                (700.0, 705.0),
                (705.0, 705.0)):
            with self.subTest(solver_fun=solver_fun, dual=dual):
                with self.assertRaises(PipelineFailure):
                    self.certify(
                        "Z", 700.0, dual, 0.0,
                        solver_fun=solver_fun, w4=5)
        # A non-magic sub-half-step perturbation is consistent and must not be
        # rejected merely because it is many ULP from P.
        certificate = self.certify(
            "Z", 700.0, 700.25, 0.0, solver_fun=700.25, w4=5)
        self.assertEqual(certificate["method"], E4.STATUS0_LATTICE_METHOD)

    def test_solver_fun_must_identify_exact_readbacks_lattice_level(self):
        half_cell = 702.5
        with self.assertRaisesRegex(
                PipelineFailure, "same objective-lattice level"):
            self.certify(
                "Z", 700.0, 700.0, 0.0,
                solver_fun=half_cell, w4=5)
        inside = 700.000001
        certificate = self.certify(
            "Z", 700.0, 700.0, 0.0,
            solver_fun=inside, w4=5)
        self.assertEqual(
            certificate["closed_dual_bound_full_objective"], 700.0)

    def test_attained_zero_W_keeps_proof_and_rejects_bad_raw_diagnostics(self):
        certificate = self.certify(
            "W", 0.0, 1.8010456216615096e-14, 0.0,
            solver_fun=1.8010456216615096e-14)
        self.assertEqual(
            certificate["closed_dual_bound_full_objective"], 0.0)
        for solver_fun, dual in ((0.0, -1.0), (1.0, 1.0)):
            with self.subTest(solver_fun=solver_fun, dual=dual):
                with self.assertRaises(PipelineFailure):
                    self.certify(
                        "W", 0.0, dual, 0.0, solver_fun=solver_fun)

    def test_lower_attainable_level_is_not_hidden(self):
        with self.assertRaises(PipelineFailure):
            self.certify("Z", 700.0, 695.0, 0.0, w4=5)

    def test_primal_off_lattice_fails(self):
        with self.assertRaisesRegex(
                PipelineFailure, "incumbent is not on the declared"):
            self.certify("Z", 701.0, 700.0, 0.0, w4=5)

    def test_dual_above_primal_beyond_ulp_fails(self):
        with self.assertRaises(PipelineFailure):
            self.certify("Z", 700.0, 705.0, 0.0, w4=5)

    def test_status0_dual_overshoot_uses_half_step_not_an_ulp_envelope(self):
        p = 700.0
        one_ulp = math.ulp(p)
        for ulps in (64, 128, 256):
            inside = p + ulps * one_ulp
            certificate = self.certify(
                "Z", p, inside, 0.0, solver_fun=p, w4=5)
            self.assertEqual(certificate["method"], E4.STATUS0_LATTICE_METHOD)
            self.assertEqual(certificate["bound_ulps"], 0)
            self.assertEqual(
                certificate["closed_dual_bound_full_objective"], p)
        with self.assertRaises(PipelineFailure):
            self.certify("Z", p, p + 2.5, 0.0, solver_fun=p, w4=5)

    def test_large_constant_does_not_change_a_consistent_raw_status0_cell(self):
        for constant in (1.0e16, -1.0e16):
            with self.subTest(constant=constant):
                certificate = E4.exact_objective_certificate(
                    "W", 0.0, 0.0, constant, 0.0,
                    integral_durations=True, where="large-C-consistent")
                self.assertEqual(
                    certificate["method"], E4.STATUS0_LATTICE_METHOD)
                self.assertEqual(certificate["bound_ulps"], 0)

    def test_raw_solver_and_dual_deltas_cannot_be_hidden_by_cancellation(self):
        shifted_fun_and_bound = 1.0
        for cancellation_constant in (float(2 ** 55), -float(2 ** 55)):
            with self.subTest(constant=cancellation_constant):
                # Adding C first would erase both F-P and B-P.  The executable
                # must instead reject the unsafe raw pairwise evidence.
                self.assertEqual(
                    cancellation_constant + shifted_fun_and_bound,
                    cancellation_constant)
                with self.assertRaises(PipelineFailure):
                    self.certify(
                        "W", 0.0, shifted_fun_and_bound,
                        cancellation_constant,
                        solver_fun=shifted_fun_and_bound)

    def test_nonfinite_operands_and_nonpositive_step_fail_closed(self):
        for field in ("P", "B", "C", "F"):
            for forged in (math.nan, math.inf, -math.inf):
                values = {"P": 0.0, "B": 0.0, "C": 0.0, "F": 0.0}
                values[field] = forged
                with self.subTest(field=field, forged=forged), \
                        self.assertRaises(PipelineFailure):
                    self.certify(
                        "W", values["P"], values["B"], values["C"],
                        solver_fun=values["F"])
        with mock.patch.object(
                E4, "_objective_lattice_spec",
                return_value=(0.0, 0.0, "forged_nonpositive_step")):
            with self.assertRaises(PipelineFailure):
                self.certify("W", 0.0, 0.0, 0.0, solver_fun=0.0)

    def test_adjacent_cell_and_one_ulp_same_cell_are_distinguished(self):
        # One ULP above an adjacent lower level remains an adjacent-cell value.
        just_above_695 = math.nextafter(695.0, math.inf)
        with self.assertRaises(PipelineFailure):
            self.certify("Z", 700.0, just_above_695, 0.0, w4=5)
        just_below_700 = math.nextafter(700.0, -math.inf)
        certificate = self.certify("Z", 700.0, just_below_700, 0.0, w4=5)
        self.assertEqual(
            certificate["closed_dual_bound_full_objective"], 700.0)

    def test_zero_objective_keeps_gap_undefined(self):
        certificate = self.certify("U", 0.0, 0.0, 0.0)
        gap, reason = full_objective_gap(0.0, 0.0, 0.0, where="negative-test")
        self.assertTrue(certificate["proves_exact_optimality"])
        self.assertIsNone(gap)
        self.assertEqual(reason, "zero_full_objective_denominator")

    def test_roles_do_not_borrow_Z_step_five(self):
        for stage in ("U", "W", "D"):
            with self.subTest(stage=stage), self.assertRaises(PipelineFailure):
                self.certify(stage, 700.0, 695.1, 0.0, integral=True)

    def test_fractional_Z_has_no_lattice_escape(self):
        with self.assertRaisesRegex(
                PipelineFailure, "lacks both a lattice proof"):
            self.certify(
                "Z", 700.0, 698.6842105263167, 0.0,
                w4=5, integral=False)

    def test_formal_allow_fractional_fails_before_milp(self):
        with tempfile.TemporaryDirectory(prefix="r1b-fractional-guard-") as out:
            with mock.patch.object(E4, "milp") as milp:
                with self.assertRaisesRegex(
                        PipelineFailure, "forbids --allow-fractional-durations"):
                    E4.run(
                        os.path.join(out, "not-read.csv"), out,
                        formal=True, snap_id="a" * 64,
                        cli_time_limit_s=E4.SOLVER_TIME_LIMIT_S,
                        allow_fractional=True, verbose=False)
                milp.assert_not_called()

    def test_formal_shard_one_ulp_time_limit_tampering_fails_closed(self):
        """No raw role or diagnostics row may round-trip a nearby policy value."""
        policy = E4.solver_policy()
        self.assertEqual(policy.time_limit_s, 1800.0)
        forged_time_limit = math.nextafter(1800.0, math.inf)
        self.assertNotEqual(forged_time_limit, policy.time_limit_s)
        snap_id = "a" * 64
        manifest_sha = "b" * 64
        rho = float(E4.FORMAL_RHOS[0])
        time_columns = (
            E4.COL.TIME_LIMIT,
            *(prefix + "time_limit_s" for prefix in self.ROLE_PREFIXES),
        )

        def assert_rejected(*, raw_column=None, tamper_diagnostics=False):
            raw = self.raw.copy(deep=True)
            raw.loc[:, "rho"] = rho
            raw.loc[:, "formal_run"] = True
            raw.loc[:, "snap_id"] = snap_id
            raw.loc[:, "solver_policy_manifest_sha256"] = manifest_sha
            raw.loc[:, "canonical_assignment"] = False
            for column in time_columns:
                raw.loc[:, column] = policy.time_limit_s
            if raw_column is not None:
                raw.loc[raw.index[0], raw_column] = forged_time_limit

            diagnostics = pd.DataFrame([{
                "rho": rho,
                "seed": 901,
                "method": "A1",
                "snap_id": snap_id,
                "presolve": False,
                "time_limit_s": (
                    forged_time_limit
                    if tamper_diagnostics else policy.time_limit_s),
                "solver_status_code": 0,
                "objective_stage": "D",
                "solver_fun_raw": 0.0,
                "mip_dual_bound": 0.0,
                "mip_gap_solver_raw": 0.0,
                "lattice_proves_exact_optimality": True,
            }])
            frames = {
                "e4_raw.csv": raw,
                "e4_summary.csv": self.summary,
                "e4_summary_provenance.csv": self.provenance,
                "e4_solver_diagnostics.csv": diagnostics,
            }

            def read_frame(path, **_kwargs):
                return frames[os.path.basename(path)].copy(deep=True)

            design = {
                "axis": {(rho, 901): [0]},
                "scenario_pool_counts": {(rho, 901): 1},
                "scenario_lattice_preconditions": {(rho, 901): {}},
                "deviation_constants": {(rho, 901): 0},
                "n_tasks_future": {(rho, 901, 0): 1},
                "inputs": [],
            }
            with tempfile.TemporaryDirectory(
                    prefix="r1b-hostile-formal-time-") as root:
                shard_dirs = [
                    os.path.join(root, "shard-{}".format(index))
                    for index in range(len(E4.FORMAL_RHOS))]
                with mock.patch.object(
                        E4, "assert_no_staging_import"), \
                        mock.patch.object(
                            E4, "assert_no_absolute_runtime_paths"), \
                        mock.patch.object(
                            E4, "load_design_contract", return_value={}), \
                        mock.patch.object(
                            E4, "solver_policy", return_value=policy), \
                        mock.patch.object(
                            E4, "require_snap_binding", return_value=snap_id), \
                        mock.patch.object(
                            E4, "_derive_merge_design",
                            return_value=(pd.DataFrame(), design)), \
                        mock.patch.object(
                            E4.os.path, "isfile", return_value=True), \
                        mock.patch.object(
                            E4, "sha256_file", return_value=manifest_sha), \
                        mock.patch.object(
                            E4, "_read_csv_audited", side_effect=read_frame), \
                        mock.patch.object(
                            E4, "validate_solver_diagnostics_raw_gaps"), \
                        mock.patch.object(
                            E4, "build_design", return_value={}), \
                        mock.patch.object(
                            E4, "validate_solver_diagnostics_design"), \
                        mock.patch.object(
                            E4, "raw_spec", return_value=None), \
                        mock.patch.object(E4, "validate_stage_output"), \
                        mock.patch.object(E4, "verify_provenance"), \
                        mock.patch.object(E4, "extra_checks"), \
                        mock.patch.object(
                            E4, "_assert_shard_matches_manifest"):
                    with self.assertRaisesRegex(
                            PipelineFailure,
                            ("time-limit evidence differs from policy"
                             if raw_column is not None
                             else "diagnostics disagree with solver policy")):
                        E4.merge_formal_shards(
                            os.path.join(root, "manifest.csv"),
                            os.path.join(root, "merged"), shard_dirs,
                            snap_id=snap_id,
                            cli_time_limit_s=policy.time_limit_s,
                            verbose=False, allow_fixture_manifest=True)

        for column in time_columns:
            with self.subTest(artifact="raw", column=column):
                assert_rejected(raw_column=column)
        with self.subTest(artifact="diagnostics", column="time_limit_s"):
            assert_rejected(tamper_diagnostics=True)

    def test_later_fractional_manifest_row_fails_before_first_milp(self):
        """A rho filter must not hide a bad scenario later in the family."""
        with tempfile.TemporaryDirectory(
                prefix="r1b-family-wide-lattice-") as root:
            copies = {}
            for name in (
                    "H_epse4_A.csv", "S_epse4_A.csv",
                    "H_epse4_C_precision.csv", "S_epse4_C_precision.csv"):
                target = os.path.join(root, name)
                shutil.copyfile(os.path.join(HERE, name), target)
                copies[name] = target
            manifest = os.path.join(root, "manifest.csv")
            pd.DataFrame([
                {"rho_target": 0.45, "seed": 901,
                 "hhcp": os.path.basename(copies["H_epse4_A.csv"]),
                 "sol": os.path.basename(copies["S_epse4_A.csv"])},
                {"rho_target": 0.55, "seed": 902,
                 "hhcp": os.path.basename(
                     copies["H_epse4_C_precision.csv"]),
                 "sol": os.path.basename(
                     copies["S_epse4_C_precision.csv"])},
            ]).to_csv(manifest, index=False)
            with mock.patch.object(E4, "milp") as milp:
                with self.assertRaisesRegex(
                        PipelineFailure,
                        "family-wide preflight before the first solver call"):
                    E4.run(
                        manifest, os.path.join(root, "out"),
                        rho_filter=0.45, max_absent=1,
                        canonical_assignment=False, verbose=False)
                milp.assert_not_called()

    def test_fractional_H_STD_fails_before_milp(self):
        with tempfile.TemporaryDirectory(
                prefix="r1b-hstd-integrality-") as out:
            manifest = os.path.join(HERE, "manifest_epse4_A.csv")
            with mock.patch.object(E4, "H_STD", 240.5), \
                    mock.patch.object(E4, "milp") as milp:
                with self.assertRaises(PipelineFailure):
                    E4.run(
                        manifest, out, max_absent=1,
                        canonical_assignment=False, verbose=False)
                milp.assert_not_called()

    def test_preflight_receipt_change_fails_before_publication(self):
        with tempfile.TemporaryDirectory(
                prefix="r1b-preflight-receipt-") as root:
            h_name, s_name = "H.csv", "S.csv"
            h_path, s_path = os.path.join(root, h_name), os.path.join(root, s_name)
            shutil.copyfile(os.path.join(HERE, "H_epse4_A.csv"), h_path)
            shutil.copyfile(os.path.join(HERE, "S_epse4_A.csv"), s_path)
            manifest = pd.DataFrame([{
                "rho_target": 0.45, "seed": 901,
                "hhcp": h_name, "sol": s_name}])
            preflight = E4.preflight_manifest_lattice(
                manifest, root, rho_column="rho_target", seed_column="seed",
                h_column="hhcp", s_column="sol",
                allow_fractional=False, formal=False)
            with open(h_path, "ab") as handle:
                handle.write(b"\n")
            with self.assertRaisesRegex(
                    PipelineFailure, "receipt changed"):
                E4.assert_preflight_receipts_unchanged(
                    preflight, where="hostile TOCTOU test")

    def test_persisted_proof_is_replayed_and_tampering_fails(self):
        prefix = "weighted_"
        p, b, c, w4 = -17400.0, -17401.315789473683, 18100.0, 100
        certificate = self.certify("Z", p, b, c, w4=w4)
        row = self.persisted_row(prefix, certificate, p, b, c, w4=w4)
        replayed = E4.replay_persisted_exact_certificate(
            row, prefix, "Z",
            integral_durations=self.input_integral_durations,
            expected_deviation_constant=181,
            where="negative-test")
        self.assertEqual(replayed["closed_dual_bound_full_objective"], 700.0)

        mutations = {
            prefix + "solver_fun_raw": p + 5.0,
            prefix + "mip_dual_bound": -17405.0,
            prefix + "deviation_constant_C": 18105.0,
            prefix + "exact_certificate_method": "forged_method",
            prefix + "lattice_step": 1.0,
            prefix + "lattice_offset": 5.0,
            prefix + "lattice_boundary_margin": 1.0,
            prefix + "lattice_safe_dual_bound_full_objective": 699.0,
            prefix + "lattice_closed_dual_bound_full_objective": 695.0,
            prefix + "lattice_primal_level_full_objective": 705.0,
            prefix + "lattice_proves_exact_optimality": True,
            prefix + "exact_certificate_objective_stage": "D",
            prefix + "exact_certificate_lower_bound_source": (
                "nonnegative_weighted_objective"),
            prefix + "exact_certificate_structural_lower_bound_full_objective": 5,
            prefix + "exact_certificate_exact_full_objective_integer": 705,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                bad = dict(row)
                bad[field] = value
                with self.assertRaises(PipelineFailure):
                    E4.replay_persisted_exact_certificate(
                        bad, prefix, "Z",
                        integral_durations=self.input_integral_durations,
                        expected_deviation_constant=181,
                        where="negative-test")

    def test_exact_integer_parser_rejects_fractional_bound_ulps(self):
        prefix = "weighted_"
        p, b, c, w4 = -17400.0, -17401.315789473683, 18100.0, 100
        certificate = self.certify("Z", p, b, c, w4=w4)
        row = self.persisted_row(prefix, certificate, p, b, c, w4=w4)
        row[prefix + "lattice_bound_ulps"] = 128.9
        with self.assertRaisesRegex(
                PipelineFailure, "not an exact integer"):
            E4.replay_persisted_exact_certificate(
                row, prefix, "Z",
                integral_durations=self.input_integral_durations,
                expected_deviation_constant=181,
                where="negative-test")

    def test_strict_boolean_parser_rejects_yes_and_fractional_truth(self):
        prefix = "weighted_"
        p, b, c, w4 = -17400.0, -17401.315789473683, 18100.0, 100
        certificate = self.certify("Z", p, b, c, w4=w4)
        baseline = self.persisted_row(prefix, certificate, p, b, c, w4=w4)
        for forged in ("yes", "0.9"):
            with self.subTest(forged=forged), self.assertRaisesRegex(
                    PipelineFailure,
                    "non-canonical spelling|not exactly 0 or 1"):
                row = dict(baseline)
                row[prefix + "lattice_proves_exact_optimality"] = forged
                E4.replay_persisted_exact_certificate(
                    row, prefix, "Z",
                    integral_durations=self.input_integral_durations,
                    expected_deviation_constant=181,
                    where="negative-test")

    def test_api_false_string_is_not_treated_as_true(self):
        self.assertFalse(E4._cell_bool(
            "false", field="canonical_assignment", where="negative-test"))
        with self.assertRaisesRegex(
                PipelineFailure, "lacks both a lattice proof"):
            E4.exact_objective_certificate(
                "Z", 700.0, 698.0, 0.0, 700.0, w4=5,
                integral_durations="false", where="negative-test")

    def test_diagnostics_reject_fractional_integer_coordinates(self):
        baseline = {
            "rho": 0.45, "seed": 901, "absent": 0, "method": "A1",
            "w4": 5, "group_index": 1, "nodes": 0,
            "solver_status_code": 0, "objective_stage": "D",
            "solver_fun_raw": 1.0, "mip_dual_bound": 1.0,
            "mip_gap_solver_raw": 0.0, "lattice_bound_ulps": 0,
            "exact_certificate_method": E4.STATUS0_LATTICE_METHOD,
            "exact_certificate_lower_bound_source": (
                "solver_status0_unique_lattice_cell_no_exact_bound"),
            "exact_certificate_structural_lower_bound_full_objective": 0,
            "exact_certificate_exact_full_objective_integer": 1,
            "lattice_proves_exact_optimality": False, "presolve": False,
        }
        for field, value in (
                ("seed", 901.9), ("absent", 0.9), ("w4", 5.9),
                ("group_index", 1.9), ("nodes", 0.9)):
            with self.subTest(field=field), self.assertRaisesRegex(
                    PipelineFailure, "not an exact integer"):
                row = dict(baseline)
                row[field] = value
                E4.validate_solver_diagnostics_raw_gaps(
                    pd.DataFrame([row]), where="negative-test")

    def test_diagnostics_bind_lattice_bound_ulps_to_executable_contract(self):
        baseline = {
            "rho": 0.45, "seed": 901, "absent": 0, "method": "A1",
            "w4": 5, "group_index": 1, "nodes": 0,
            "solver_status_code": 0, "objective_stage": "D",
            "solver_fun_raw": 1.0, "mip_dual_bound": 1.0,
            "mip_gap_solver_raw": 0.0, "lattice_bound_ulps": 0,
            "exact_certificate_method": E4.STATUS0_LATTICE_METHOD,
            "exact_certificate_lower_bound_source": (
                "solver_status0_unique_lattice_cell_no_exact_bound"),
            "exact_certificate_structural_lower_bound_full_objective": 0,
            "exact_certificate_exact_full_objective_integer": 1,
            "lattice_proves_exact_optimality": False, "presolve": False,
        }
        for forged, message in (
                (0.9, "not an exact integer"),
                (128, "differs from the executable method"),
                (256, "differs from the executable method")):
            with self.subTest(forged=forged), self.assertRaisesRegex(
                    PipelineFailure, message):
                row = dict(baseline)
                row["lattice_bound_ulps"] = forged
                E4.validate_solver_diagnostics_raw_gaps(
                    pd.DataFrame([row]), where="negative-test")
        row = dict(baseline)
        row["exact_certificate_method"] = "objective_lattice_ceiling_v1"
        with self.assertRaisesRegex(
                PipelineFailure, "certificate method differs"):
            E4.validate_solver_diagnostics_raw_gaps(
                pd.DataFrame([row]), where="negative-test")

    def test_forged_row_lattice_label_cannot_authorize_certificate(self):
        prefix = "weighted_"
        p, b, c, w4 = -17400.0, -17401.315789473683, 18100.0, 100
        certificate = self.certify("Z", p, b, c, w4=w4)
        row = self.persisted_row(prefix, certificate, p, b, c, w4=w4)
        row["w_level_separation"] = "integral_durations"
        with self.assertRaisesRegex(
                PipelineFailure, "lacks both a lattice proof"):
            E4.replay_persisted_exact_certificate(
                row, prefix, "Z", integral_durations=False,
                expected_deviation_constant=181,
                where="negative-test")

    def test_landed_lattice_label_must_match_source_premise(self):
        mutated = self.raw.copy(deep=True)
        mutated.loc[mutated.index[0], "w_level_separation"] = "unknown"
        with self.assertRaisesRegex(
                PipelineFailure, "artifact label or H/S receipt differs"):
            E4.extra_checks(
                mutated, self.summary, self.provenance, self.design,
                verbose=False)

    def test_each_role_raw_solver_gap_tamper_fails(self):
        for prefix in self.ROLE_PREFIXES:
            with self.subTest(prefix=prefix):
                mutated = self.raw.copy(deep=True)
                mutated.loc[mutated.index[0], prefix + "mip_gap_solver_raw"] = 0.125
                with self.assertRaisesRegex(
                        PipelineFailure,
                        "raw solver mip gap differs from solver fun/bound replay"):
                    E4.extra_checks(
                        mutated, self.summary, self.provenance, self.design,
                        verbose=False)

    def test_generic_fun_and_gap_coordinated_tamper_fails_tie_duplicate_gate(self):
        mutated = self.raw.copy(deep=True)
        row_index = mutated.index[0]
        # This pair is internally consistent with the unchanged generic B=0:
        # abs(F-B)/abs(F) == 1.  It must still fail because the generic block is
        # required to duplicate the independently named tie-break certificate.
        mutated.loc[row_index, "solver_fun_raw"] = 1.0
        mutated.loc[row_index, "mip_gap_solver_raw"] = 1.0
        with self.assertRaisesRegex(
                PipelineFailure,
                "generic solver block is not the declared final-D tie-break"):
            E4.extra_checks(
                mutated, self.summary, self.provenance, self.design,
                verbose=False)


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False).result
    print("R1B_LATTICE_CERTIFICATE_TESTS_{} tests={}".format(
        "PASS" if result.wasSuccessful() else "FAIL", result.testsRun))
    raise SystemExit(0 if result.wasSuccessful() else 1)
