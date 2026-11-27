#!/usr/bin/env python3
"""Verify a read-only SNAP archive, extract it to tmpfs, and execute run_all.py."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile


WORK_ROOT = Path(os.environ.get("DCHHR_WORK_ROOT", "/work/snapshot"))
MARKER = ".dchhr_ephemeral_snapshot"
HEX = frozenset("0123456789abcdef")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message: str) -> "None":
    raise SystemExit(f"SNAP_ENTRYPOINT_ABORTED: {message}")


def safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        fail(f"unsafe archive member path: {name!r}")
    return path


def verify_and_extract(archive_path: Path, snap_id: str) -> None:
    actual = sha256_file(archive_path)
    if actual != snap_id:
        fail(f"archive SHA256 mismatch: expected={snap_id} actual={actual}")
    if WORK_ROOT.exists() and any(WORK_ROOT.iterdir()):
        fail(f"ephemeral extraction root is not empty: {WORK_ROOT}")
    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r") as archive:
        members = archive.getmembers()
        if not members or members[0].name != "SNAP_MANIFEST.json":
            fail("SNAP_MANIFEST.json must be the first archive member")
        for member in members:
            safe_member(member.name)
            if not member.isfile():
                fail(f"only regular files are permitted in SNAP: {member.name}")
        manifest_handle = archive.extractfile("SNAP_MANIFEST.json")
        if manifest_handle is None:
            fail("SNAP_MANIFEST.json cannot be read")
        try:
            manifest = json.load(manifest_handle)
        except (TypeError, ValueError) as exc:
            fail(f"invalid SNAP_MANIFEST.json: {exc}")
        canonical_root = manifest.get("canonical_container_root")
        if canonical_root != WORK_ROOT.as_posix():
            fail(
                "manifest canonical_container_root does not match extraction "
                f"root: manifest={canonical_root!r} runtime={WORK_ROOT.as_posix()!r}"
            )
        entries = manifest.get("files")
        if not isinstance(entries, list) or not entries:
            fail("SNAP manifest has no file inventory")
        expected_names = {"SNAP_MANIFEST.json"}
        for entry in entries:
            if not isinstance(entry, dict):
                fail("SNAP file inventory contains a non-object")
            try:
                name = entry["path"]
                resolved_container_path = entry["resolved_container_path"]
                expected_hash = entry["sha256"]
                expected_bytes = int(entry["bytes"])
            except (KeyError, TypeError, ValueError) as exc:
                fail(f"malformed SNAP file entry: {entry!r}: {exc}")
            safe_member(name)
            expected_resolved_path = WORK_ROOT.joinpath(
                *PurePosixPath(name).parts
            ).as_posix()
            if resolved_container_path != expected_resolved_path:
                fail(
                    "manifest resolved_container_path mismatch: "
                    f"path={name!r} manifest={resolved_container_path!r} "
                    f"runtime={expected_resolved_path!r}"
                )
            expected_names.add(name)
            handle = archive.extractfile(name)
            if handle is None:
                fail(f"manifest member missing: {name}")
            data = handle.read()
            if len(data) != expected_bytes or sha256_bytes(data) != expected_hash:
                fail(f"member hash/size mismatch: {name}")
            target = WORK_ROOT.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(0o755 if name.endswith(".sh") else 0o644)
        actual_names = {member.name for member in members}
        if actual_names != expected_names:
            extra = sorted(actual_names - expected_names)
            missing = sorted(expected_names - actual_names)
            fail(f"archive/manifest member-set mismatch: extra={extra} missing={missing}")

    (WORK_ROOT / MARKER).write_text(
        f"SNAP_ID={snap_id}\n",
        encoding="ascii",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--snap-id", required=True)
    parser.add_argument("--result-root", default="/results")
    parser.add_argument(
        "--stage",
        choices=(
            "all",
            "primary",
            "primary-sharded",
            "environment",
            "preflight",
            "generator",
            "R1",
            "R1b",
            "R5",
            "R3",
            "R4",
            "plots",
        ),
        default="all",
    )
    args = parser.parse_args(argv)
    snap_id = args.snap_id.strip().lower()
    if len(snap_id) != 64 or any(ch not in HEX for ch in snap_id):
        fail("--snap-id must be 64 lowercase hexadecimal characters")
    archive = args.archive.resolve()
    if not archive.is_file():
        fail(f"archive not found: {archive}")
    verify_and_extract(archive, snap_id)
    runner = WORK_ROOT / "experiments" / "journal_p0" / "repro" / "run_all.py"
    if not runner.is_file():
        fail(f"formal runner missing from verified SNAP: {runner}")
    os.chdir(WORK_ROOT)
    os.execv(
        sys.executable,
        [
            sys.executable,
            str(runner),
            "--snap-id",
            snap_id,
            "--result-root",
            args.result_root,
            "--stage",
            args.stage,
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
