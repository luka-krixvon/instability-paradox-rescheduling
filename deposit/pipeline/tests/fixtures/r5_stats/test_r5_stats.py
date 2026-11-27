"""Positive, negative and deterministic-reorder tests for formal R5."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


THIS = Path(__file__).resolve().parent
ROOT = THIS.parents[2]
JOURNAL = ROOT / "experiments" / "journal_p0"
for item in (str(THIS), str(JOURNAL), str(ROOT / "experiments")):
    if item not in sys.path:
        sys.path.insert(0, item)

from build_fixture import build, sha256  # noqa: E402
import r5_stats  # noqa: E402
from s2_core import PipelineFailure  # noqa: E402


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class R5FixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp(prefix="r5-fixture-"))
        self.inputs = build(self.temp / "input")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp, ignore_errors=True)

    def run_r5(self, suffix: str = "out"):
        out = self.temp / suffix
        result = r5_stats.run(self.inputs["raw"], self.inputs["manifest"], out)
        return out, result

    def rewrite_raw(self, mutate) -> None:
        frame = pd.read_csv(self.inputs["raw"], keep_default_na=False)
        frame = mutate(frame)
        frame.to_csv(self.inputs["raw"], index=False, lineterminator="\n")

    def expect_failure(self, mutate, needle: str) -> None:
        mutate()
        with self.assertRaises(PipelineFailure) as caught:
            self.run_r5("negative")
        self.assertIn(needle, str(caught.exception))

    def test_positive_and_registered_outputs(self) -> None:
        out, result = self.run_r5()
        self.assertEqual(result["counts"]["dynamic_disruptions"], 60)
        self.assertEqual(result["counts"]["actual_raw_rows"], 720)
        expected = {
            "r5_scenario_effects.csv", "r5_by_rho.csv", "r5_bootstrap.csv",
            "r5_run_manifest.json", "r5_output_hashes.json",
        }
        self.assertEqual({item.name for item in out.iterdir()}, expected)
        by_rho = pd.read_csv(out / "r5_by_rho.csv")
        bootstrap = pd.read_csv(out / "r5_bootstrap.csv")
        self.assertEqual(set(by_rho["per_rho_p_value"]), {"NOT_COMPUTED"})
        self.assertEqual(set(bootstrap["pooled_p_value"]), {"NOT_COMPUTED"})
        self.assertEqual(set(bootstrap["n_resamples"]), {10000})
        self.assertEqual(set(bootstrap["rng_seed"]), {20260730})
        receipt = json.loads((out / "r5_output_hashes.json").read_text())
        self.assertNotIn("r5_output_hashes.json", receipt["outputs"])
        self.assertEqual(receipt["self_hash"], "NOT_COMPUTED_TO_AVOID_SELF_REFERENCE")

    def test_deterministic_after_independent_input_reordering(self) -> None:
        out_a, _ = self.run_r5("out_a")
        scientific = ["r5_scenario_effects.csv", "r5_by_rho.csv", "r5_bootstrap.csv"]
        hashes_a = {name: file_sha(out_a / name) for name in scientific}

        raw = pd.read_csv(self.inputs["raw"], keep_default_na=False)
        raw.sample(frac=1.0, random_state=71).to_csv(
            self.inputs["raw"], index=False, lineterminator="\n")
        manifest = pd.read_csv(self.inputs["manifest"])
        # Reorder each S file independently and update its declared digest.
        for index, row in manifest.iterrows():
            solution = self.inputs["root"] / row["sol"]
            s_frame = pd.read_csv(solution)
            s_frame.sample(frac=1.0, random_state=100 + index).to_csv(
                solution, index=False, lineterminator="\n")
            manifest.loc[index, "s_sha256"] = sha256(solution)
        manifest.sample(frac=1.0, random_state=73).to_csv(
            self.inputs["manifest"], index=False, lineterminator="\n")

        out_b, _ = self.run_r5("out_b")
        hashes_b = {name: file_sha(out_b / name) for name in scientific}
        self.assertEqual(hashes_a, hashes_b)

    def test_exact_floor_and_holm_cumulative_max(self) -> None:
        p_value, floor = r5_stats._exact_two_sided_sign_flip([1, 1, 1, 1, 1])
        self.assertEqual(p_value, 0.0625)
        self.assertEqual(floor, 0.0625)
        adjusted = r5_stats._holm_adjust([0.04, 0.01, 0.03])
        self.assertEqual(adjusted.tolist(), [0.06, 0.03, 0.06])

    def test_optional_preregistered_holm_family(self) -> None:
        registration = self.temp / "holm_registration.json"
        registration.write_text(json.dumps({
            "stage": "R5",
            "family_id": "fixture-overtime-w4-20",
            "estimand": "delta_overtime_A2_minus_A1",
            "w4": 20,
            "rho_levels": [0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
            "procedure": "exact_cluster_sign_flip_then_Holm",
            "registered_before_r5": True,
            "two_sided": True,
        }, sort_keys=True), encoding="utf-8")
        out = self.temp / "holm_out"
        r5_stats.run(self.inputs["raw"], self.inputs["manifest"], out,
                     holm_registration=registration)
        holm = pd.read_csv(out / "r5_holm.csv")
        self.assertEqual(len(holm), 6)
        self.assertTrue((holm["p_holm"] >= holm["p_raw"]).all())
        self.assertEqual(set(holm["minimum_attainable_two_sided_p"]), {0.0625})
        receipt = json.loads((out / "r5_output_hashes.json").read_text())
        self.assertIn("r5_holm.csv", receipt["outputs"])

    def test_formal_positive_propagates_snap(self) -> None:
        snap = "a" * 64
        frame = pd.read_csv(self.inputs["raw"], keep_default_na=False)
        frame["snap_id"] = snap
        frame.to_csv(self.inputs["raw"], index=False, lineterminator="\n")
        out = self.temp / "formal_positive"
        r5_stats.run(self.inputs["raw"], self.inputs["manifest"], out,
                     formal=True, snap_id=snap)
        for name in ("r5_scenario_effects.csv", "r5_by_rho.csv", "r5_bootstrap.csv"):
            self.assertEqual(set(pd.read_csv(out / name)["snap_id"]), {snap})
        receipt = json.loads((out / "r5_output_hashes.json").read_text())
        self.assertEqual(receipt["snap_id"], snap)

    def test_missing_cell_fails(self) -> None:
        self.expect_failure(
            lambda: self.rewrite_raw(lambda frame: frame.iloc[1:].copy()),
            "dynamic all-eligible product")

    def test_duplicate_cell_fails(self) -> None:
        self.expect_failure(
            lambda: self.rewrite_raw(
                lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True)),
            "duplicate primary keys")

    def test_extra_absent_fails(self) -> None:
        def mutate(frame):
            extra = frame.iloc[[0]].copy()
            extra["absent"] = 777
            return pd.concat([frame, extra], ignore_index=True)
        self.expect_failure(lambda: self.rewrite_raw(mutate),
                            "dynamic all-eligible product")

    def test_newly_eligible_absent_in_S_requires_raw_rows(self) -> None:
        def mutate():
            manifest = pd.read_csv(self.inputs["manifest"])
            solution = self.inputs["root"] / manifest.iloc[0]["sol"]
            frame = pd.read_csv(solution)
            frame.loc[len(frame)] = [777, 2, 1, 777, "777;", 30.0]
            frame.to_csv(solution, index=False, lineterminator="\n")
            manifest.loc[0, "s_sha256"] = sha256(solution)
            manifest.to_csv(self.inputs["manifest"], index=False, lineterminator="\n")
        self.expect_failure(mutate, "dynamic all-eligible product")

    def test_common_service_mismatch_fails(self) -> None:
        def mutate(frame):
            frame.loc[0, "U"] = float(frame.loc[0, "U_common"]) + 1
            frame.loc[0, "unserved"] = frame.loc[0, "U"]
            return frame
        self.expect_failure(lambda: self.rewrite_raw(mutate),
                            "did not impose common service")

    def test_legacy_alias_mismatch_fails(self) -> None:
        def mutate(frame):
            frame.loc[0, "unserved"] = float(frame.loc[0, "U"]) + 1
            return frame
        self.expect_failure(lambda: self.rewrite_raw(mutate),
                            "legacy unserved alias")

    def test_nonoptimal_solver_fails(self) -> None:
        def mutate(frame):
            frame.loc[0, "weighted_solver_status_code"] = 1
            frame.loc[0, "weighted_proven_optimal"] = False
            return frame
        self.expect_failure(lambda: self.rewrite_raw(mutate),
                            "non-optimal or malformed solver certificate")

    def test_formal_input_requires_matching_snap_rows(self) -> None:
        with self.assertRaises(PipelineFailure) as caught:
            r5_stats.run(self.inputs["raw"], self.inputs["manifest"],
                         self.temp / "formal", formal=True, snap_id="a" * 64)
        self.assertIn("do not all carry", str(caught.exception))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(R5FixtureTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    evidence = {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "note": "regression evidence only; not proof or scientific validation",
    }
    print("R5_FIXTURE_RESULT " + json.dumps(evidence, sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
