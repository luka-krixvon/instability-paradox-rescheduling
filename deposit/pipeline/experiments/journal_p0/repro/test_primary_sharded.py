#!/usr/bin/env python3
"""Tiny, solver-free tests for primary-sharded orchestration and SNAP routing."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import run_all as RUNNER


SNAP = "a" * 64
RHOS = [0.45, 0.55, 0.65, 0.75, 0.85, 0.95]


class PrimaryShardedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="primary-sharded-test-")
        self.result_root = Path(self.temp.name) / "results"
        self.result_root.mkdir()

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _land_stage(root: Path, name: str) -> None:
        stage = root / name
        stage.mkdir()
        (stage / "stage_hashes.json").write_text("{}\n", encoding="utf-8")

    def test_primary_order_and_merged_stage_gates(self) -> None:
        events = []

        def simple(name):
            def call(root, _snap):
                events.append(name)
                output = "R0_generator" if name == "generator" else name
                self._land_stage(root, output)
            return call

        def r1(root, _snap, rhos, config):
            self.assertEqual(list(rhos), RHOS)
            self.assertEqual(config["jobs"], 3)
            events.append("R1-six-shards-then-owner-merge")
            self._land_stage(root, "R1")

        def r1b(root, _snap, rhos, config):
            self.assertTrue((root / "R1" / "stage_hashes.json").is_file())
            self.assertEqual(list(rhos), RHOS)
            self.assertEqual(config["jobs"], 3)
            events.append("R1b-six-shards-then-owner-merge")
            self._land_stage(root, "R1b")

        with mock.patch.dict(os.environ, {RUNNER.SHARD_JOBS_ENV: "3"}), \
                mock.patch.object(RUNNER, "_registered_rhos", return_value=RHOS), \
                mock.patch.object(RUNNER, "run_environment", simple("environment")), \
                mock.patch.object(RUNNER, "run_preflight", simple("preflight")), \
                mock.patch.object(RUNNER, "run_generator", simple("generator")), \
                mock.patch.object(RUNNER, "run_r1_sharded", r1), \
                mock.patch.object(RUNNER, "run_r1b_sharded", r1b), \
                mock.patch.object(RUNNER, "run_r5", simple("R5")):
            RUNNER.run_primary_sharded(self.result_root, SNAP)

        self.assertEqual(events, [
            "environment", "preflight", "generator",
            "R1-six-shards-then-owner-merge",
            "R1b-six-shards-then-owner-merge", "R5",
        ])

    def test_r1_failure_blocks_r1b_and_r5(self) -> None:
        events = []

        def simple(name):
            def call(root, _snap):
                events.append(name)
                output = "R0_generator" if name == "generator" else name
                self._land_stage(root, output)
            return call

        def fail_r1(_root, _snap, _rhos, _config):
            events.append("R1-failed")
            raise RUNNER.FormalRunError("injected R1 shard failure")

        with mock.patch.object(RUNNER, "_registered_rhos", return_value=RHOS), \
                mock.patch.object(RUNNER, "run_environment", simple("environment")), \
                mock.patch.object(RUNNER, "run_preflight", simple("preflight")), \
                mock.patch.object(RUNNER, "run_generator", simple("generator")), \
                mock.patch.object(RUNNER, "run_r1_sharded", fail_r1), \
                mock.patch.object(RUNNER, "run_r1b_sharded") as r1b, \
                mock.patch.object(RUNNER, "run_r5") as r5:
            with self.assertRaisesRegex(RUNNER.FormalRunError, "injected"):
                RUNNER.run_primary_sharded(self.result_root, SNAP)
        r1b.assert_not_called()
        r5.assert_not_called()
        self.assertEqual(events, ["environment", "preflight", "generator", "R1-failed"])
        self.assertFalse((self.result_root / "R1").exists())

    def test_each_shard_inventory_survives_outer_rename(self) -> None:
        outer_partial = self.result_root / ".R1.partial"
        outer_partial.mkdir()
        required = ("artifact.csv",)

        def fake_run(_command, *, log_path, snap_id, stage, **_kwargs):
            self.assertEqual(snap_id, SNAP)
            log_path.write_text(f"{stage}\n", encoding="utf-8")
            if stage.endswith("0.65"):
                raise RUNNER.FormalRunError("injected shard failure")
            (log_path.parent / "artifact.csv").write_text(
                stage + "\n", encoding="utf-8")

        with mock.patch.object(RUNNER, "_run_logged", fake_run):
            with self.assertRaisesRegex(RUNNER.FormalRunError,
                                        "merge was not started"):
                RUNNER._run_rho_shards(
                    outer_partial,
                    stage="R1",
                    snap_id=SNAP,
                    rhos=RHOS,
                    jobs=3,
                    command_factory=lambda rho, out: ["fake", rho, out],
                    required_files=required,
                )
        failure = json.loads(
            (outer_partial / "r1_shard_failures.json").read_text(encoding="utf-8"))
        self.assertFalse(failure["merge_started"])
        self.assertEqual([row["rho"] for row in failure["failures"]], ["0.65"])
        self.assertFalse((self.result_root / "R1").exists())

        # Repeat without failure, then simulate the outer atomic promotion and
        # prove every persisted relative receipt resolves from final R1.
        shutil.rmtree(outer_partial)
        outer_partial.mkdir()

        def fake_success(_command, *, log_path, snap_id, stage, **_kwargs):
            self.assertEqual(snap_id, SNAP)
            log_path.write_text(f"{stage}\n", encoding="utf-8")
            (log_path.parent / "artifact.csv").write_text(
                stage + "\n", encoding="utf-8")

        with mock.patch.object(RUNNER, "_run_logged", fake_success):
            receipts = RUNNER._run_rho_shards(
                outer_partial,
                stage="R1",
                snap_id=SNAP,
                rhos=RHOS,
                jobs=2,
                command_factory=lambda rho, out: ["fake", rho, out],
                required_files=required,
            )
        RUNNER._record_shard_orchestration(
            outer_partial,
            stage="R1",
            snap_id=SNAP,
            configuration={"jobs": 2},
            rhos=RHOS,
            receipts=receipts,
            merge_command=[
                "fake-merge", "--out-dir", outer_partial,
                "--shard-dir", outer_partial / "shards" / "rho_0p45",
            ],
        )
        final = self.result_root / "R1"
        RUNNER._finish_stage(outer_partial, final, SNAP, "R1")
        outer_inventory = json.loads(
            (final / "stage_hashes.json").read_text(encoding="utf-8"))
        nested_inventories = [
            row["path"] for row in outer_inventory["files"]
            if row["path"].endswith("/stage_hashes.json")
        ]
        self.assertEqual(len(nested_inventories), 6)
        evidence_path = final / "r1_shard_orchestration.json"
        evidence_text = evidence_path.read_text(encoding="utf-8")
        landed = json.loads(evidence_text)
        self.assertNotIn(str(outer_partial), evidence_text)
        self.assertNotIn(".R1.partial", evidence_text)
        self.assertEqual(landed["merge_command"][-3:], [
            "${STAGE_ROOT}",
            "--shard-dir",
            "${STAGE_ROOT}/shards/rho_0p45",
        ])
        for receipt in landed["shards"]:
            inventory = final / receipt["stage_hash_inventory"]
            self.assertFalse(Path(receipt["stage_hash_inventory"]).is_absolute())
            self.assertFalse(Path(receipt["directory"]).is_absolute())
            self.assertTrue(inventory.is_file())
            self.assertEqual(
                RUNNER.sha256_file(inventory),
                receipt["stage_hash_inventory_sha256"],
            )

    def test_cli_contract_and_job_bounds(self) -> None:
        dirs = [Path(f"/results/shards/{index}") for index in range(6)]
        r1_solve = [str(x) for x in RUNNER._r1_shard_command(
            Path("/results/r1"), 0.45, SNAP)]
        r1_merge = [str(x) for x in RUNNER._r1_merge_command(
            Path("/results/R1.partial"), dirs, SNAP)]
        r1b_solve = [str(x) for x in RUNNER._r1b_shard_command(
            Path("/results/r1b"), 0.45, SNAP)]
        r1b_merge = [str(x) for x in RUNNER._r1b_merge_command(
            Path("/results/R1b.partial"), dirs, SNAP)]
        self.assertIn("--rho", r1_solve)
        self.assertNotIn("--max-absent", r1_solve)
        self.assertEqual(r1_merge.count("--shard-dir"), 6)
        self.assertNotIn("--rho", r1_merge)
        self.assertNotIn("--no-canonical-assignment", r1_merge)
        self.assertIn("horizon", r1b_solve)
        self.assertIn("--rho", r1b_solve)
        self.assertNotIn("--max-absent", r1b_solve)
        self.assertIn("--merge-shards", r1b_merge)
        self.assertEqual(r1b_merge[-6:], [str(path) for path in dirs])
        self.assertNotIn("--rho", r1b_merge)

        for valid in ("1", "6"):
            with mock.patch.dict(os.environ, {RUNNER.SHARD_JOBS_ENV: valid}):
                self.assertEqual(RUNNER._shard_configuration()["jobs"], int(valid))
        for invalid in ("0", "7", "not-an-int"):
            with mock.patch.dict(os.environ, {RUNNER.SHARD_JOBS_ENV: invalid}):
                with self.assertRaises(RUNNER.FormalRunError):
                    RUNNER._shard_configuration()

    def test_snapshot_collect_after_generated_fixture_cleanup(self) -> None:
        builder_path = RUNNER.REPRO / "build_snapshot.py"
        spec = importlib.util.spec_from_file_location("snapshot_builder", builder_path)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        probe = (RUNNER.ROOT / "tests" / "fixtures" / "r5_stats" / "out"
                 / "_repro_collect_probe")
        probe.mkdir(parents=True, exist_ok=False)
        generated = probe / "generated_should_not_land.py"
        generated.write_text("raise RuntimeError('must not be collected')\n",
                             encoding="utf-8")
        try:
            first = module.collect_paths(RUNNER.ROOT)
            second = module.collect_paths(RUNNER.ROOT)
            first_rel = [path.relative_to(RUNNER.ROOT).as_posix() for path in first]
            second_rel = [path.relative_to(RUNNER.ROOT).as_posix() for path in second]
            self.assertEqual(first_rel, second_rel)
            self.assertNotIn(generated.relative_to(RUNNER.ROOT).as_posix(), first_rel)
            self.assertIn(
                "tests/fixtures/r5_stats/build_fixture.py", first_rel)
            self.assertFalse(any(
                path.startswith("tests/fixtures/r5_stats/input/")
                for path in first_rel))
        finally:
            shutil.rmtree(probe)
            try:
                probe.parent.rmdir()
            except OSError:
                pass


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False).result
    print("PRIMARY_SHARDED_ORCHESTRATION_TESTS_{} tests={}".format(
        "PASS" if result.wasSuccessful() else "FAIL", result.testsRun))
    raise SystemExit(0 if result.wasSuccessful() else 1)
