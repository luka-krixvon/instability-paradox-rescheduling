#!/usr/bin/env python3
"""Build and verify the deterministic, content-addressed journal SNAP archive."""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[3]
CANONICAL_CONTAINER_ROOT = "/work/snapshot"
BASE_IMAGE_AMD64 = "sha256:70fbdeb75b0d071778ea1df4353ac959118ae8031bc5be33a95d1b8ea1fb8f08"
BASE_INSTANCE = Path(
    "experiments/data_problems_solutions/HHCP-patient-65-task-256-nurse-59.csv"
)
AUDIT_INPUTS = (
    Path("FORMAL_RERUN_RECOVERY_AMENDMENT.md"),
    Path("FORMAL_RERUN_RECOVERY_04_LATTICE_AMENDMENT.md"),
    Path("FORMAL_RERUN_RECOVERY_05_NUMERICAL_ENVELOPE_AMENDMENT.md"),
    Path("FORMAL_RERUN_RECOVERY_06_SAME_LEVEL_AND_ROUNDTRIP_AMENDMENT.md"),
    Path("FORMAL_RERUN_RECOVERY_06_STRUCTURAL_LOWER_BOUND_ADDENDUM.md"),
    Path("FORMAL_RERUN_RECOVERY_06_OBJECTIVE_RECONSTRUCTION_ADDENDUM.md"),
    Path("FORMAL_RERUN_RECOVERY_06_STATUS0_LATTICE_CONSISTENCY_ADDENDUM.md"),
    Path("audit_outputs/generator_design_registration.md"),
    Path("audit_outputs/rerun_plan_v2_1.md"),
    Path("audit_outputs/human_decisions_required_v2_1.md"),
    Path("audit_outputs/s2_historical_pre_s2/official_family/PRES2_rho_manifest.csv"),
)
EPHEMERAL_PARTS = {
    "__pycache__",
    "_nc_tmp",
    "_nc_band",
    "_nc_band_independent",
    "faultinject_out",
    "out",
}
EPHEMERAL_GLOBS = ("*.pyc", "*.pyo", "*.S2TMP.*", "*.FAILED_*.json", ".DS_Store")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ephemeral(path: Path) -> bool:
    return bool(EPHEMERAL_PARTS.intersection(path.parts)) or any(
        fnmatch.fnmatch(path.name, pattern) for pattern in EPHEMERAL_GLOBS
    )


def _withheld_names(root: Path) -> set:
    """Basenames the public-release marker declares as withheld from this tree."""
    marker = root / "BASE_INSTANCE_WITHHELD"
    if not marker.is_file():
        return set()
    return {
        line.strip()
        for line in marker.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def collect_paths(root: Path):
    paths = set()

    # Runtime Python modules directly imported by the journal pipeline.
    for path in (root / "experiments").glob("*.py"):
        paths.add(path)

    # Journal code, manifests, and the reproducibility harness.  Stale result
    # CSV/PDF files are excluded below; the official generated input family is
    # added explicitly.
    journal = root / "experiments/journal_p0"
    for suffix in ("*.py", "*.json", "*.sh"):
        paths.update(
            path for path in journal.rglob(suffix)
            if not _ephemeral(path.relative_to(root))
        )
    paths.add(journal / "repro/Dockerfile")
    paths.add(journal / "repro/requirements.lock")
    paths.add(journal / "repro/test_primary_sharded.py")
    # Explicit presence checks for the new primary statistical stage.  The
    # recursive source collection includes these when present; these additions
    # make an incomplete R5 merge a hard SNAP-build failure.
    paths.add(journal / "r5_stats.py")
    paths.add(journal / "r5_design_manifest.json")
    paths.add(journal / "stats_tests.py")

    # Registered, regenerated input family.
    paths.add(journal / "rho_manifest.csv")
    paths.update(journal.glob("H_rho*.csv"))
    paths.update(journal.glob("S_rho*.csv"))
    if (root / BASE_INSTANCE).is_file():
        paths.add(root / BASE_INSTANCE)
    else:
        # Public reproducibility release: the base instance is not
        # redistributable.  Its absence is permitted only when
        # BASE_INSTANCE_WITHHELD at the tree root names it, which is also what
        # licenses the runtime relaxations in eps_constraint.py, run_e4.py,
        # scalability.py and repro/run_all.py.  Any other missing input is
        # still a hard build failure in the required-input check below.
        if BASE_INSTANCE.name not in _withheld_names(root):
            raise SystemExit(
                "SNAP input missing and not declared withheld: "
                f"{BASE_INSTANCE} (add BASE_INSTANCE_WITHHELD naming it)"
            )
        # The marker must travel inside the archive: the runtime relaxations it
        # licenses are evaluated against the extracted tree, not the host.
        paths.add(root / "BASE_INSTANCE_WITHHELD")

    # Test *sources* needed by the fresh-directory preflight.  Generated
    # fixture outputs and diagnostics are deliberately not SNAP inputs.
    fixtures = root / "tests/fixtures"
    paths.update(path for path in fixtures.rglob("*.py")
                 if not _ephemeral(path.relative_to(root)))
    fixture_inputs = (
        "corelib/selftest_manifest.json",
        "corelib/band_manifest.json",
        "epse4/README.md",
        "epse4/H_epse4_A.csv",
        "epse4/H_epse4_B_ineligible_incumbent.csv",
        "epse4/H_epse4_C_precision.csv",
        "epse4/S_epse4_A.csv",
        "epse4/S_epse4_C_precision.csv",
        "epse4/manifest_epse4_A.csv",
        "epse4/manifest_epse4_A_formal.csv",
        "epse4/manifest_epse4_B.csv",
        "epse4/manifest_epse4_C.csv",
        # Executable presence checks: all fixture Python sources are collected
        # above, but naming these load-bearing scripts explicitly makes an
        # incomplete merge fail SNAP construction instead of silently omitting
        # a required gate.
        "epse4/run_e4_common_service.py",
        "epse4/test_r1_failure_coordinate.py",
        "epse4/test_r1_formal_contract.py",
        "epse4/test_r1_shard_merge.py",
        "epse4/test_r1b_lattice_certificate.py",
        "generator/tiny_hhcp_fixture.csv",
        "r5_stats/README.md",
        "r5_stats/build_fixture.py",
        "r5_stats/test_r5_stats.py",
        "rollext/tiny_hhcrsp_bad_rowwidth.txt",
        "rollext/tiny_hhcrsp_bad_window.txt",
        "rollext/tiny_hhcrsp_dup_key.txt",
        "rollext/tiny_hhcrsp_ok.txt",
        "rollext/tiny_hhcrsp_unbuildable.txt",
        "rollext/tiny_vns_other_layout.txt",
    )
    paths.update(fixtures / rel for rel in fixture_inputs)

    # R5 fixture inputs are generated deterministically by build_fixture.py in
    # a fresh temporary directory.  No generated input/output/expected tree is
    # a SNAP member; only the builder, test source, and explanatory README are.

    # These are executable-run contracts, not optional documentary extras.
    # Adding them unconditionally makes a missing recovery/audit input fail the
    # common missing-file gate below instead of silently changing SNAP scope.
    # Audit provenance documents.  A file the release marker names as withheld
    # may be absent; anything else missing is still a hard failure.
    withheld = _withheld_names(root)
    for rel in AUDIT_INPUTS:
        if (root / rel).is_file():
            paths.add(root / rel)
        elif rel.name not in withheld:
            raise SystemExit(
                f"SNAP audit input missing and not declared withheld: {rel} "
                "(add BASE_INSTANCE_WITHHELD naming it)"
            )

    missing = [str(p) for p in sorted(paths) if not p.is_file()]
    if missing:
        raise SystemExit(f"SNAP input missing: {missing}")
    return sorted(paths, key=lambda p: p.relative_to(root).as_posix())


def manifest_for(root: Path, paths):
    entries = []
    for path in paths:
        rel = path.relative_to(root).as_posix()
        entries.append(
            {
                "path": rel,
                "resolved_container_path": f"{CANONICAL_CONTAINER_ROOT}/{rel}",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema": "llm-dchhr-snap-v1",
        "canonical_container_root": CANONICAL_CONTAINER_ROOT,
        "base_image_linux_amd64_manifest": BASE_IMAGE_AMD64,
        "base_instance": BASE_INSTANCE.as_posix(),
        "files": entries,
    }


def _tar_info(name: str, data: bytes, executable: bool = False):
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o755 if executable else 0o644
    return info


def build(root: Path, output_dir: Path):
    paths = collect_paths(root)
    manifest = manifest_for(root, paths)
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
        + b"\n"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    temp = output_dir / "SNAP_BUILDING.tar"
    if temp.exists():
        raise SystemExit(f"refusing to overwrite stale temporary archive: {temp}")
    with tarfile.open(temp, "w", format=tarfile.PAX_FORMAT) as archive:
        archive.addfile(_tar_info("SNAP_MANIFEST.json", manifest_bytes),
                        io.BytesIO(manifest_bytes))
        for path in paths:
            rel = path.relative_to(root).as_posix()
            data = path.read_bytes()
            executable = path.suffix == ".sh"
            archive.addfile(_tar_info(rel, data, executable), io.BytesIO(data))
    snap_id = sha256_file(temp)
    final = output_dir / f"SNAP_{snap_id}.tar"
    if final.exists():
        if sha256_file(final) != snap_id:
            raise SystemExit(f"existing archive has the wrong digest: {final}")
        temp.unlink()
    else:
        os.replace(temp, final)
    verify(final, snap_id)
    return final, snap_id, manifest


def verify(archive_path: Path, expected_id: str):
    actual_id = sha256_file(archive_path)
    if actual_id != expected_id:
        raise SystemExit(f"archive hash mismatch: expected={expected_id} actual={actual_id}")
    with tarfile.open(archive_path, "r") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if names != sorted(names, key=lambda name: (name != "SNAP_MANIFEST.json", name)):
            raise SystemExit("archive members are not in deterministic order")
        extracted = archive.extractfile("SNAP_MANIFEST.json")
        if extracted is None:
            raise SystemExit("SNAP_MANIFEST.json missing")
        manifest = json.load(extracted)
        for entry in manifest["files"]:
            handle = archive.extractfile(entry["path"])
            if handle is None:
                raise SystemExit(f"manifest member missing: {entry['path']}")
            data = handle.read()
            if len(data) != entry["bytes"] or sha256_bytes(data) != entry["sha256"]:
                raise SystemExit(f"manifest mismatch: {entry['path']}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    archive, snap_id, manifest = build(args.root.resolve(), args.output_dir.resolve())
    print(json.dumps({
        "snap_id": snap_id,
        "archive": str(archive),
        "files": len(manifest["files"]),
        "bytes": archive.stat().st_size,
        "base_image_linux_amd64_manifest": BASE_IMAGE_AMD64,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
