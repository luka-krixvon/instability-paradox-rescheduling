#!/usr/bin/env python3
"""Positive and fail-closed regression checks for formal R1 shard merging.

The fixture solves one tiny formal scenario once, then relabels those already
validated landed rows into the registered 6 rho x 5 seed layout.  That keeps
the test about merge integrity rather than solver throughput.  The merge still
re-parses every manifest H/S input and dynamically derives the all-active axis;
the tiny family has 60 disruptions, deliberately not the production total.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
JOURNAL = os.path.join(ROOT, "experiments", "journal_p0")
EXPERIMENTS = os.path.join(ROOT, "experiments")
for path in (JOURNAL, EXPERIMENTS):
    if path not in sys.path:
        sys.path.insert(0, path)

import eps_constraint as R1  # noqa: E402
from s2_core import PipelineFailure, sha256_file  # noqa: E402


SNAP = "a" * 64


def _manifest_row(rho, seed, h_path, s_path):
    source_sha = sha256_file(h_path)
    generator = os.path.join(JOURNAL, "eps_constraint.py")
    family_runner = os.path.abspath(__file__)
    return {
        "rho_target": float(rho), "seed": int(seed),
        "hhcp": h_path, "sol": s_path, "N": 3,
        "rho_global": float(rho), "rho_used_mean": float(rho),
        "rho_used_max": float(rho), "src": h_path,
        "src_sha256": source_sha, "feasible": True, "unplaced": 0,
        "h_sha256": sha256_file(h_path), "h_bytes": os.path.getsize(h_path),
        "s_sha256": sha256_file(s_path), "s_bytes": os.path.getsize(s_path),
        "generator": generator, "generator_sha256": sha256_file(generator),
        "family_runner": family_runner,
        "family_runner_sha256": sha256_file(family_runner),
    }


def _build_family(root):
    h_path = os.path.join(HERE, "H_epse4_A.csv")
    s_path = os.path.join(HERE, "S_epse4_A.csv")
    base_manifest = os.path.join(root, "base_manifest.csv")
    pd.DataFrame([_manifest_row(0.45, 1, h_path, s_path)]).to_csv(
        base_manifest, index=False)
    base_out = os.path.join(root, "base_out")
    R1.run(
        base_manifest, base_out, formal=True, snap_id=SNAP,
        cli_time_limit_s=R1.FORMAL_SOLVER_TIME_LIMIT_S,
        canonical_assignment=False, verbose=False)
    base = {
        name: pd.read_csv(os.path.join(base_out, name))
        for name in R1.SHARD_ARTIFACTS
    }

    manifest_rows, shard_dirs = [], []
    for rho in R1.FORMAL_RHOS:
        shard_dir = os.path.join(root, "shard_{:02d}".format(int(rho * 100)))
        os.makedirs(shard_dir)
        shard_dirs.append(shard_dir)
        expanded = {name: [] for name in R1.SHARD_ARTIFACTS}
        for seed in range(1, 6):
            manifest_rows.append(_manifest_row(rho, seed, h_path, s_path))
            for name, base_frame in base.items():
                frame = base_frame.copy()
                frame["rho"] = float(rho)
                frame["seed"] = int(seed)
                if name in ("eps_range.csv", "eps_frontier.csv"):
                    frame["rho_target"] = float(rho)
                    frame["rho_global"] = float(rho)
                    frame["rho_used_mean"] = float(rho)
                    frame["rho_used_max"] = float(rho)
                    frame["src_sha256"] = sha256_file(h_path)
                expanded[name].append(frame)
        for name, parts in expanded.items():
            pd.concat(parts, ignore_index=True).to_csv(
                os.path.join(shard_dir, name), index=False)
    manifest = os.path.join(root, "rho_manifest_fixture.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest, index=False)
    return manifest, shard_dirs


def _must_fail(label, manifest, shards, mutate=None):
    root = tempfile.mkdtemp(prefix="r1-shard-negative-")
    local = []
    for index, source in enumerate(shards):
        destination = os.path.join(root, f"s{index}")
        shutil.copytree(source, destination)
        local.append(destination)
    if mutate is not None:
        mutate(local)
    out = os.path.join(root, "combined")
    try:
        R1.merge_formal_shards(
            manifest, out, local, snap_id=SNAP,
            cli_time_limit_s=R1.FORMAL_SOLVER_TIME_LIMIT_S,
            verbose=False, allow_fixture_manifest=True)
    except PipelineFailure as exc:
        print(f"R1_SHARD_NEGATIVE_PASS {label}: {str(exc).splitlines()[0]}")
        assert not any(os.path.isfile(os.path.join(out, name))
                       for name in R1.SHARD_ARTIFACTS)
        return
    raise AssertionError(f"negative fixture unexpectedly passed: {label}")


def main():
    root = tempfile.mkdtemp(prefix="r1-shard-positive-")
    manifest, shards = _build_family(root)
    combined = os.path.join(root, "combined")
    rng, front, diagnostics, oracle, receipt, *_ = R1.merge_formal_shards(
        manifest, combined, shards, snap_id=SNAP,
        cli_time_limit_s=R1.FORMAL_SOLVER_TIME_LIMIT_S,
        verbose=False, allow_fixture_manifest=True)
    assert len(rng) == 60                 # dynamic tiny axis: 2 x 30
    assert len(front) == len(oracle) == 360
    assert len(diagnostics) == 1200
    assert receipt["scenario_count_expected"] == 30
    assert receipt["eligible_disruptions_expected"] == 60
    assert receipt["full_axis_expected"] == receipt["full_axis_actual"]
    assert "receipt_self_hash" not in receipt
    assert R1.SHARD_MERGE_RECEIPT not in receipt["merged_artifact_sha256"]
    assert os.path.isfile(os.path.join(combined, R1.SHARD_MERGE_RECEIPT))
    print("R1_SHARD_POSITIVE_PASS scenarios=30 disruptions=60 "
          "frontier=360 diagnostics=1200")

    # Missing rho: five directories cannot masquerade as the full family.
    try:
        R1.merge_formal_shards(
            manifest, os.path.join(root, "missing_out"), shards[:-1],
            snap_id=SNAP,
            cli_time_limit_s=R1.FORMAL_SOLVER_TIME_LIMIT_S, verbose=False,
            allow_fixture_manifest=True)
    except PipelineFailure:
        print("R1_SHARD_NEGATIVE_PASS missing_rho")
    else:
        raise AssertionError("missing-rho family unexpectedly passed")

    # Duplicate rho in a distinct directory (not merely the same path twice).
    duplicate_root = tempfile.mkdtemp(prefix="r1-shard-duplicate-")
    duplicate = os.path.join(duplicate_root, "rho_duplicate")
    shutil.copytree(shards[0], duplicate)
    _must_fail("duplicate_rho", manifest, shards[:-1] + [duplicate])

    def mixed_snap(local):
        for name in R1.SHARD_ARTIFACTS:
            path = os.path.join(local[0], name)
            frame = pd.read_csv(path)
            frame["snap_id"] = "b" * 64
            frame.to_csv(path, index=False)

    _must_fail("mixed_SNAP", manifest, shards, mutate=mixed_snap)

    def missing_frontier_row(local):
        path = os.path.join(local[0], "eps_frontier.csv")
        frame = pd.read_csv(path)
        frame.iloc[1:].to_csv(path, index=False)

    _must_fail("missing_frontier_row", manifest, shards,
               mutate=missing_frontier_row)

    def tamper_service(local):
        path = os.path.join(local[0], "eps_frontier.csv")
        frame = pd.read_csv(path)
        frame.loc[0, "U_common"] = int(frame.loc[0, "U_common"]) + 1
        frame.to_csv(path, index=False)

    _must_fail("tampered_common_service", manifest, shards,
               mutate=tamper_service)
    print("R1_SHARD_MERGE_FIXTURES_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
