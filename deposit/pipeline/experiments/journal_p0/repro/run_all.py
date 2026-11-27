#!/usr/bin/env python3
"""One-command formal internal rerun from an already verified SNAP extraction.

The content-addressed archive is mounted read-only by the host and extracted to
an ephemeral Docker work directory.  The host recomputes its SHA256 and passes
that value as ``--snap-id``.  This runner never invents or rewrites the SNAP id.

Each stage lands first in ``.<stage>.partial`` and is renamed only after its
command exits zero and a file-hash inventory has been written.  Thus a failed
stage cannot be mistaken for a complete result family.

``--stage primary`` is the fresh one-click path for environment, preflight,
generator, R1, horizon/common-service R1b, and synthetic clustered R5.  It
deliberately does not wait for the separately scoped R3/R4 families.
``--stage primary-sharded`` preserves that scientific order while solving the
six registered rho strata concurrently and using only the R1/R1b owners' own
formal merge gates before either stage is atomically promoted.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[3]
JOURNAL = ROOT / "experiments" / "journal_p0"
REPRO = JOURNAL / "repro"
EPHEMERAL_MARKER = ROOT / ".dchhr_ephemeral_snapshot"
BASE_INSTANCE = (
    ROOT
    / "experiments"
    / "data_problems_solutions"
    / "HHCP-patient-65-task-256-nurse-59.csv"
)
PRE_S2_MANIFEST = (
    ROOT
    / "audit_outputs"
    / "s2_historical_pre_s2"
    / "official_family"
    / "PRES2_rho_manifest.csv"
)
HEX = frozenset("0123456789abcdef")
# Public reproducibility release only: names the inputs that are not
# redistributable and are therefore absent from this tree.  See the file
# itself, and the matching relaxations in eps_constraint.py, run_e4.py,
# scalability.py and repro/build_snapshot.py.
WITHHELD_MARKER = ROOT / "BASE_INSTANCE_WITHHELD"
GENERATOR_FILES = ("rho_manifest.csv",)
PRIMARY_STAGES = (
    "environment",
    "preflight",
    "generator",
    "R1",
    "R1b",
    "R5",
)
STAGES = PRIMARY_STAGES + (
    "R3",
    "R4",
    "plots",
)
STAGE_RESULT_NAMES = {"generator": "R0_generator"}
SHARD_JOBS_ENV = "DCHHR_SHARD_JOBS"
DEFAULT_SHARD_JOBS = 6
MAX_SHARD_JOBS = 6
THREAD_LIMIT_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
R1_REQUIRED_FILES = (
    "eps_range.csv",
    "eps_frontier.csv",
    "eps_solver_diagnostics.csv",
    "eps_oracle_validation.csv",
)
R1B_REQUIRED_FILES = (
    "e4_raw.csv",
    "e4_summary.csv",
    "e4_summary_provenance.csv",
    "e4_solver_diagnostics.csv",
)


class FormalRunError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalRunError(message)


def require_snap_id(value: str) -> str:
    normalized = value.strip().lower()
    require(
        len(normalized) == 64 and all(ch in HEX for ch in normalized),
        "--snap-id must be the externally recomputed 64-hex SNAP archive SHA256",
    )
    return normalized


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"CSV has no header: {path}")
        return list(reader)


def _compare_generator_family(candidate_dir: Path) -> None:
    """Require the formal generator to reproduce the SNAP's registered family.

    ``generated_utc`` is the sole intentionally volatile manifest cell.  The
    historical pre-S2 manifest is supplied to both generations, so caregiver
    deltas remain a scientific field and are not excluded from this comparison.
    """
    baseline_manifest = JOURNAL / "rho_manifest.csv"
    candidate_manifest = candidate_dir / "rho_manifest.csv"
    baseline_rows = _read_csv(baseline_manifest)
    candidate_rows = _read_csv(candidate_manifest)
    require(len(baseline_rows) == 30 and len(candidate_rows) == 30,
            "generator family must contain exactly 30 manifest rows")
    by_tag_baseline = {row["tag"]: row for row in baseline_rows}
    by_tag_candidate = {row["tag"]: row for row in candidate_rows}
    require(
        len(by_tag_baseline) == len(baseline_rows)
        and len(by_tag_candidate) == len(candidate_rows)
        and set(by_tag_baseline) == set(by_tag_candidate),
        "generator manifest tag keys differ from the registered SNAP family",
    )
    for tag in sorted(by_tag_baseline):
        left = dict(by_tag_baseline[tag])
        right = dict(by_tag_candidate[tag])
        left.pop("generated_utc", None)
        right.pop("generated_utc", None)
        require(left == right, f"generator semantic manifest drift at tag={tag}")
        for column, hash_column in (("hhcp", "h_sha256"), ("sol", "s_sha256")):
            baseline_file = JOURNAL / by_tag_baseline[tag][column]
            candidate_file = candidate_dir / by_tag_candidate[tag][column]
            require(
                sha256_file(baseline_file) == by_tag_baseline[tag][hash_column],
                f"SNAP family hash mismatch at {tag}/{column}",
            )
            require(
                sha256_file(candidate_file) == by_tag_candidate[tag][hash_column],
                f"formal generator hash mismatch at {tag}/{column}",
            )
            require(
                sha256_file(baseline_file) == sha256_file(candidate_file),
                f"formal generator bytes differ from SNAP family at {tag}/{column}",
            )


def _require_generated_family(result_root: Path, snap_id: str) -> None:
    """Verify R0 reproduced the immutable family used by downstream stages.

    Downstream stages deliberately keep reading the byte-stable manifest already
    inside SNAP.  The fresh generator manifest differs only in its wall-clock
    ``generated_utc`` field; copying it would make otherwise identical stage
    input receipts differ on every run.
    """
    source = result_root / "R0_generator"
    require(source.is_dir(), "R0_generator must pass before this stage")
    receipt = source / "R0_GENERATOR_WITHHELD.json"
    if (BASE_INSTANCE.name in _withheld_sources()
            and not BASE_INSTANCE.is_file()
            and receipt.is_file()):
        # The receipt alone is not the switch.  It lives in the host-writable
        # result root, which nothing digest-verifies, so a stale receipt left
        # by an earlier release run must not disarm the regeneration check in
        # a tree that has the base instance back.  The marker and the absent
        # file are the authority, exactly as in run_generator(); the receipt
        # only has to agree, and to carry this run's snap_id.
        recorded = json.loads(receipt.read_text(encoding="utf-8"))
        require(str(recorded.get("snap_id", "")) == snap_id,
                "R0 withheld receipt belongs to a different run: "
                f"receipt snap_id={recorded.get('snap_id')!r} current={snap_id!r}")
        _require_committed_family()
        return
    rows = _read_csv(source / "rho_manifest.csv")
    require(len(rows) == 30, "R0_generator manifest must contain 30 rows")
    names = {"rho_manifest.csv"}
    for row in rows:
        for column, hash_column in (("hhcp", "h_sha256"), ("sol", "s_sha256")):
            name = row[column]
            require(Path(name).name == name, f"unsafe generated filename: {name!r}")
            path = source / name
            require(path.is_file(), f"generated input missing: {path}")
            require(sha256_file(path) == row[hash_column],
                    f"generated input hash mismatch: {path}")
            names.add(name)
    require(len(names) == 61, "generated family must contain 60 H/S files plus manifest")
    _compare_generator_family(source)


def _manifest_time_limit(path: Path) -> str:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    value = manifest["solver"]["time_limit_s"]
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0,
            f"invalid solver.time_limit_s in {path}")
    return str(float(value))


def _stage_paths(result_root: Path, stage: str):
    final = result_root / stage
    partial = result_root / f".{stage}.partial"
    require(not final.exists(), f"refusing to overwrite completed stage: {final}")
    require(not partial.exists(), f"stale partial stage requires inspection: {partial}")
    partial.mkdir(parents=True)
    return partial, final


def _require_fresh_pipeline(result_root: Path, stages: Iterable[str],
                            label: str) -> None:
    """Reject one-click pipelines before doing work if any target is present."""
    conflicts = []
    for stage in stages:
        output_name = STAGE_RESULT_NAMES.get(stage, stage)
        for candidate in (
                result_root / output_name,
                result_root / f".{output_name}.partial"):
            if candidate.exists():
                conflicts.append(str(candidate))
    require(
        not conflicts,
        f"--stage {label} is a fresh one-click pipeline and requires all of "
        "its final/partial outputs to be absent; run individual stages to "
        f"resume an inspected result root; conflicts={conflicts}",
    )


def _shard_configuration() -> Dict[str, Any]:
    raw = os.environ.get(SHARD_JOBS_ENV)
    token = str(DEFAULT_SHARD_JOBS) if raw is None else raw.strip()
    try:
        jobs = int(token)
    except ValueError as exc:
        raise FormalRunError(
            f"{SHARD_JOBS_ENV} must be an integer from 1 through "
            f"{MAX_SHARD_JOBS}; observed={raw!r}") from exc
    require(
        1 <= jobs <= MAX_SHARD_JOBS,
        f"{SHARD_JOBS_ENV} must be from 1 through {MAX_SHARD_JOBS}; "
        f"observed={jobs}",
    )
    return {
        "jobs": jobs,
        "source": (
            f"environment:{SHARD_JOBS_ENV}"
            if raw is not None else f"default:{DEFAULT_SHARD_JOBS}"
        ),
        "raw_value": raw,
        "thread_environment": dict(sorted(THREAD_LIMIT_ENV.items())),
    }


def _registered_rhos() -> List[float]:
    rows = _read_csv(JOURNAL / "rho_manifest.csv")
    require(rows and all("rho_target" in row for row in rows),
            "rho_manifest.csv lacks rho_target")
    try:
        values = sorted({float(row["rho_target"]) for row in rows})
    except (TypeError, ValueError) as exc:
        raise FormalRunError(
            "rho_manifest.csv contains a malformed rho_target") from exc
    require(
        len(values) == 6 and all(0.0 < value < 1.0 for value in values),
        "primary-sharded requires exactly six distinct rho_target values in "
        f"(0,1); observed={values}",
    )
    return values


def _rho_cli(rho: float) -> str:
    return format(float(rho), ".12g")


def _rho_slug(rho: float) -> str:
    return "rho_" + _rho_cli(rho).replace("-", "m").replace(".", "p")


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _portable_stage_text(value: Any, stage_root: Path,
                         placeholder: str = "${STAGE_ROOT}") -> str:
    """Replace an ephemeral stage-root prefix with a portable placeholder.

    Formal stages are first written below ``.<stage>.partial`` and then
    renamed.  Persisting that pre-promotion absolute path in a command receipt
    or error message would make the evidence stale immediately after success
    and host-specific after export.  Only the persisted representation is
    normalized; subprocesses still receive the real path.
    """
    token = str(value)
    prefix = str(stage_root)
    if token == prefix:
        return placeholder
    if token.startswith(prefix + os.sep):
        return placeholder + "/" + token[len(prefix + os.sep):]
    return token


def _portable_stage_command(command: Iterable[Any],
                            stage_root: Path) -> List[str]:
    return [_portable_stage_text(item, stage_root) for item in command]


def _run_logged(command: Iterable[str], *, log_path: Path, snap_id: str,
                stage: str, cwd: Path = ROOT,
                stream_to_console: bool = True) -> None:
    command = [str(item) for item in command]
    with log_path.open("x", encoding="utf-8", buffering=1) as log:
        first = f"SNAP_ID={snap_id} STAGE={stage}\n"
        log.write(first)
        # The log lands inside the stage directory, so its parent is the
        # correct portable root for both ordinary and nested shard commands.
        log.write(
            "COMMAND="
            + json.dumps(
                _portable_stage_command(command, log_path.parent),
                ensure_ascii=False,
            )
            + "\n"
        )
        log.flush()
        print(first, end="", flush=True)
        child_env = dict(os.environ)
        child_env.update(THREAD_LIMIT_ENV)
        child_env.update({
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONUNBUFFERED": "1",
        })
        if stream_to_console:
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=child_env,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
            code = process.wait()
        else:
            # Direct-to-file output avoids unread PIPE buffers when six shard
            # subprocesses run concurrently.  Every shard has an independent
            # log, which is included in that shard's hash inventory.
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env=child_env,
            )
            code = process.wait()
            print(f"SHARD_COMMAND_EXIT stage={stage} code={code} log={log_path}",
                  flush=True)
    require(code == 0, f"{stage} command failed with exit code {code}; see {log_path}")


def _hash_inventory(directory: Path, snap_id: str, stage: str) -> None:
    files = []
    own_inventory = directory / "stage_hashes.json"
    for path in sorted(directory.rglob("*")):
        # Exclude only this inventory's own bytes.  Nested shard inventories
        # are upstream evidence and must be covered by the promoted outer
        # stage's inventory.
        if path.is_file() and path != own_inventory:
            files.append(
                {
                    "path": path.relative_to(directory).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    require(files, f"{stage} produced no files")
    payload = {
        "schema": "llm-dchhr-formal-stage-hashes-v1",
        "snap_id": snap_id,
        "stage": stage,
        "files": files,
    }
    (directory / "stage_hashes.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _finish_stage(partial: Path, final: Path, snap_id: str, stage: str) -> None:
    _hash_inventory(partial, snap_id, stage)
    os.replace(partial, final)
    print(f"FORMAL_STAGE_COMPLETE stage={stage} output={final}", flush=True)


def _require_stage_files(directory: Path, names: Iterable[str], stage: str) -> None:
    missing = [name for name in names if not (directory / name).is_file()]
    require(not missing, f"{stage} did not emit required files: {missing}")


ShardCommandFactory = Callable[[float, Path], Sequence[Any]]


def _run_rho_shards(
        outer_partial: Path, *, stage: str, snap_id: str,
        rhos: Sequence[float], jobs: int,
        command_factory: ShardCommandFactory,
        required_files: Sequence[str]) -> List[Dict[str, Any]]:
    """Run one isolated subprocess per rho and atomically land each shard.

    A shard is renamed out of its own ``.partial`` directory only after the
    owner command exits zero, its required artifacts exist, and its current
    file-hash inventory has been written.  The outer R1/R1b stage remains
    partial until every shard and the owner-provided merger pass.
    """
    require(len(rhos) == 6 and len(set(rhos)) == 6,
            f"{stage} sharding requires six distinct rho values")
    require(1 <= jobs <= MAX_SHARD_JOBS,
            f"{stage} shard worker count must be 1..{MAX_SHARD_JOBS}; jobs={jobs}")
    shard_root = outer_partial / "shards"
    shard_root.mkdir(parents=True)

    def run_one(rho: float) -> Dict[str, Any]:
        slug = _rho_slug(rho)
        shard_partial, shard_final = _stage_paths(shard_root, slug)
        command = list(command_factory(rho, shard_partial))
        shard_stage = f"{stage}-shard-{_rho_cli(rho)}"
        _run_logged(
            command,
            log_path=shard_partial / f"{stage}_{slug}.log",
            snap_id=snap_id,
            stage=shard_stage,
            stream_to_console=False,
        )
        _require_stage_files(shard_partial, required_files, shard_stage)
        _finish_stage(shard_partial, shard_final, snap_id, shard_stage)
        inventory = shard_final / "stage_hashes.json"
        require(inventory.is_file(),
                f"{shard_stage} lacks its landed hash inventory")
        return {
            "rho": float(rho),
            "rho_cli": _rho_cli(rho),
            # Persist only paths relative to the outer stage.  After the outer
            # ``.R1/.R1b.partial`` directory is renamed, these receipts remain
            # resolvable from the final R1/R1b directory.
            "directory": shard_final.relative_to(outer_partial).as_posix(),
            "stage_hash_inventory": inventory.relative_to(
                outer_partial).as_posix(),
            "stage_hash_inventory_sha256": sha256_file(inventory),
            "required_artifacts": {
                name: sha256_file(shard_final / name)
                for name in sorted(required_files)
            },
        }

    receipts: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=jobs,
                            thread_name_prefix=f"dchhr-{stage}-rho") as pool:
        futures = {pool.submit(run_one, rho): rho for rho in rhos}
        for future in as_completed(futures):
            rho = futures[future]
            try:
                receipts.append(future.result())
            except Exception as exc:  # retained in the unpromoted partial stage
                failures.append({
                    "rho": _rho_cli(rho),
                    "exception_type": type(exc).__name__,
                    "message": _portable_stage_text(exc, outer_partial),
                })
    receipts.sort(key=lambda row: float(row["rho"]))
    failures.sort(key=lambda row: float(row["rho"]))
    if failures:
        _write_json_exclusive(
            outer_partial / f"{stage.lower()}_shard_failures.json",
            {
                "schema": "llm-dchhr-shard-failures-v1",
                "stage": stage,
                "snap_id": snap_id,
                "merge_started": False,
                "completed_shards": receipts,
                "failures": failures,
            },
        )
        raise FormalRunError(
            f"{stage} shard family failed; owner merge was not started; "
            f"failures={failures}")
    require(len(receipts) == 6,
            f"{stage} shard family did not land all six receipts")
    return receipts


def _record_shard_orchestration(
        outer_partial: Path, *, stage: str, snap_id: str,
        configuration: Mapping[str, Any], rhos: Sequence[float],
        receipts: Sequence[Mapping[str, Any]], merge_command: Sequence[Any],
        upstream: Optional[Mapping[str, Any]] = None) -> Path:
    evidence = outer_partial / f"{stage.lower()}_shard_orchestration.json"
    _write_json_exclusive(
        evidence,
        {
            "schema": "llm-dchhr-primary-sharded-orchestration-v1",
            "stage": stage,
            "snap_id": snap_id,
            "configured_concurrency": dict(configuration),
            "rho_values": [float(rho) for rho in rhos],
            "all_six_shards_completed_before_merge": True,
            "merge_command": _portable_stage_command(
                merge_command, outer_partial),
            "command_path_placeholder": {
                "${STAGE_ROOT}": (
                    "directory containing this orchestration receipt"
                ),
            },
            "merge_exit_code": 0,
            "shards": list(receipts),
            "upstream_gate": dict(upstream or {}),
            "publication": "outer partial -> hash inventory -> atomic rename",
        },
    )
    return evidence


def run_environment(result_root: Path, snap_id: str) -> None:
    partial, final = _stage_paths(result_root, "environment")
    _run_logged(
        [sys.executable, REPRO / "environment_probe.py"],
        log_path=partial / "environment.log",
        snap_id=snap_id,
        stage="environment",
    )
    _finish_stage(partial, final, snap_id, "environment")


def run_preflight(result_root: Path, snap_id: str) -> None:
    partial, final = _stage_paths(result_root, "preflight")
    commands = [
        (
            "eps_help",
            [sys.executable, JOURNAL / "eps_constraint.py", "--help"],
        ),
        (
            "e4_help",
            [sys.executable, JOURNAL / "run_e4.py", "--help"],
        ),
        (
            "r5_help",
            [sys.executable, JOURNAL / "r5_stats.py", "--help"],
        ),
        (
            "r5_wrapper_help",
            [sys.executable, JOURNAL / "stats_tests.py", "--help"],
        ),
        (
            "generator_faults",
            [
                sys.executable,
                ROOT / "tests" / "fixtures" / "generator"
                / "faultinject_s2_generator.py",
                "--all",
            ],
        ),
        (
            "generator_manifest",
            [
                sys.executable,
                ROOT / "tests" / "fixtures" / "generator"
                / "negctl_manifest_validator.py",
            ],
        ),
        (
            "eps_e4",
            [
                sys.executable,
                ROOT / "tests" / "fixtures" / "epse4"
                / "run_epse4_fixtures.py",
            ],
        ),
        (
            "r1_formal_contract",
            [
                sys.executable,
                ROOT / "tests" / "fixtures" / "epse4"
                / "test_r1_formal_contract.py",
            ],
        ),
        (
            "r1_failure_coordinate",
            [
                sys.executable,
                ROOT / "tests" / "fixtures" / "epse4"
                / "test_r1_failure_coordinate.py",
            ],
        ),
        (
            "r1_shard_merge",
            [
                sys.executable,
                ROOT / "tests" / "fixtures" / "epse4"
                / "test_r1_shard_merge.py",
            ],
        ),
        (
            "r1b_common_and_shard_merge",
            [
                sys.executable,
                ROOT / "tests" / "fixtures" / "epse4"
                / "run_e4_common_service.py",
                "--run",
            ],
        ),
        (
            "r1b_lattice_certificate",
            [
                sys.executable,
                ROOT / "tests" / "fixtures" / "epse4"
                / "test_r1b_lattice_certificate.py",
            ],
        ),
        (
            "primary_sharded_orchestration",
            [
                sys.executable,
                REPRO / "test_primary_sharded.py",
            ],
        ),
        (
            "r5_cluster_stats",
            [
                sys.executable,
                ROOT / "tests" / "fixtures" / "r5_stats"
                / "test_r5_stats.py",
            ],
        ),
        (
            "rolling_external",
            [
                sys.executable,
                ROOT / "tests" / "fixtures" / "rollext" / "test_rollext.py",
            ],
        ),
        (
            "scalability",
            [
                sys.executable,
                JOURNAL / "scalability.py",
                "--selftest",
            ],
        ),
        (
            "core_negative_controls",
            [
                sys.executable,
                ROOT / "tests" / "fixtures" / "corelib" / "negative_controls.py",
            ],
        ),
    ]
    for index, (label, command) in enumerate(commands, start=1):
        _run_logged(
            command,
            log_path=partial / f"preflight_{index:02d}_{label}.log",
            snap_id=snap_id,
            stage=f"preflight-{index:02d}-{label}",
        )
    _finish_stage(partial, final, snap_id, "preflight")


def _withheld_sources() -> frozenset:
    """Input basenames this tree declares as withheld; empty without a marker."""
    try:
        lines = WITHHELD_MARKER.read_text(encoding="utf-8").splitlines()
    except OSError:
        return frozenset()
    return frozenset(
        line.strip() for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    )


def _declared_src_digests() -> List[str]:
    """The src_sha256 values rho_manifest.csv records for the withheld base."""
    with (JOURNAL / "rho_manifest.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return sorted({str(row.get("src_sha256", "")).strip().lower()
                   for row in rows} - {""})


def _require_committed_family() -> None:
    """Digest-verify the 60 shipped scenarios against the committed manifest."""
    rows = _read_csv(JOURNAL / "rho_manifest.csv")
    require(len(rows) == 30, "committed manifest must contain 30 rows")
    names = set()
    for row in rows:
        for column, hash_column in (("hhcp", "h_sha256"), ("sol", "s_sha256")):
            name = row[column]
            require(Path(name).name == name,
                    f"unsafe committed filename: {name!r}")
            path = JOURNAL / name
            require(path.is_file(), f"committed scenario missing: {path}")
            require(sha256_file(path) == row[hash_column],
                    f"committed scenario hash mismatch: {path}")
            names.add(name)
    require(len(names) == 60,
            f"committed family must contain 60 H/S files, found {len(names)}")
    print("FORMAL_FAMILY_VERIFIED committed_scenarios=60 "
          "source=experiments/journal_p0/rho_manifest.csv", flush=True)


def run_generator(result_root: Path, snap_id: str) -> None:
    partial, final = _stage_paths(result_root, "R0_generator")
    if BASE_INSTANCE.name in _withheld_sources() and not BASE_INSTANCE.is_file():
        # The base instance is not redistributable, so R0 cannot be
        # re-executed from a public release.  The 30 derived scenarios ship
        # pre-generated, and R1/R1b/R5 still verify every one of them against
        # the h_sha256/s_sha256 digests recorded in rho_manifest.csv.  Record
        # the skip as a stage receipt so the result tree states it plainly
        # instead of looking like a completed generation.
        (partial / "R0_GENERATOR_WITHHELD.json").write_text(
            json.dumps(
                {
                    "stage": "R0-generator",
                    "executed": False,
                    "reason": "base instance withheld from the public release",
                    "withheld_input": BASE_INSTANCE.name,
                    "declared_src_sha256": _declared_src_digests(),
                    "scenarios_shipped_pre_generated": True,
                    "snap_id": snap_id,
                },
                indent=2, sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        _finish_stage(partial, final, snap_id, "R0-generator-withheld")
        return
    command = [
        sys.executable,
        JOURNAL / "gen_rho_family.py",
        "--outdir",
        partial,
        "--src",
        BASE_INSTANCE,
        "--baseline-manifest",
        PRE_S2_MANIFEST,
        "--quiet-cells",
    ]
    _run_logged(
        command,
        log_path=partial / "R0_generator.log",
        snap_id=snap_id,
        stage="R0-generator",
    )
    _compare_generator_family(partial)
    _finish_stage(partial, final, snap_id, "R0-generator")


def _r1_shard_command(out_dir: Path, rho: float, snap_id: str) -> List[Any]:
    return [
        sys.executable,
        JOURNAL / "eps_constraint.py",
        "--manifest",
        JOURNAL / "rho_manifest.csv",
        "--out-dir",
        out_dir,
        "--rho",
        _rho_cli(rho),
        "--snap-id",
        snap_id,
        "--time-limit-s",
        _manifest_time_limit(JOURNAL / "eps_constraint_design_manifest.json"),
        "--formal",
        "--no-canonical-assignment",
    ]


def _r1_merge_command(out_dir: Path, shard_dirs: Sequence[Path],
                      snap_id: str) -> List[Any]:
    command: List[Any] = [
        sys.executable,
        JOURNAL / "eps_constraint.py",
        "--manifest",
        JOURNAL / "rho_manifest.csv",
        "--out-dir",
        out_dir,
        "--snap-id",
        snap_id,
        "--time-limit-s",
        _manifest_time_limit(JOURNAL / "eps_constraint_design_manifest.json"),
        "--formal",
    ]
    for shard_dir in shard_dirs:
        command.extend(["--shard-dir", shard_dir])
    return command


def _r1b_shard_command(out_dir: Path, rho: float, snap_id: str) -> List[Any]:
    return [
        sys.executable,
        JOURNAL / "run_e4.py",
        "--manifest",
        JOURNAL / "rho_manifest.csv",
        "--out-dir",
        out_dir,
        "--mode",
        "horizon",
        "--rho",
        _rho_cli(rho),
        "--snap-id",
        snap_id,
        "--time-limit-s",
        _manifest_time_limit(JOURNAL / "run_e4_design_manifest.json"),
        "--formal",
        "--no-canonical-assignment",
    ]


def _r1b_merge_command(out_dir: Path, shard_dirs: Sequence[Path],
                       snap_id: str) -> List[Any]:
    return [
        sys.executable,
        JOURNAL / "run_e4.py",
        "--manifest",
        JOURNAL / "rho_manifest.csv",
        "--out-dir",
        out_dir,
        "--mode",
        "horizon",
        "--snap-id",
        snap_id,
        "--time-limit-s",
        _manifest_time_limit(JOURNAL / "run_e4_design_manifest.json"),
        "--formal",
        "--no-canonical-assignment",
        "--merge-shards",
        *shard_dirs,
    ]


def _run_owner_sharded_stage(
        result_root: Path, *, stage: str, snap_id: str,
        rhos: Sequence[float], configuration: Mapping[str, Any],
        shard_command_factory: ShardCommandFactory,
        merge_command_factory: Callable[[Path, Sequence[Path]], Sequence[Any]],
        shard_required_files: Sequence[str],
        merged_required_files: Sequence[str],
        upstream: Optional[Mapping[str, Any]] = None) -> Path:
    partial, final = _stage_paths(result_root, stage)
    receipts = _run_rho_shards(
        partial,
        stage=stage,
        snap_id=snap_id,
        rhos=rhos,
        jobs=int(configuration["jobs"]),
        command_factory=shard_command_factory,
        required_files=shard_required_files,
    )
    shard_dirs = [partial / str(receipt["directory"]) for receipt in receipts]
    merge_command = list(merge_command_factory(partial, shard_dirs))
    _run_logged(
        merge_command,
        log_path=partial / f"{stage}_merge.log",
        snap_id=snap_id,
        stage=f"{stage}-merge",
    )
    _require_stage_files(partial, merged_required_files, f"{stage}-merge")
    _record_shard_orchestration(
        partial,
        stage=stage,
        snap_id=snap_id,
        configuration=configuration,
        rhos=rhos,
        receipts=receipts,
        merge_command=merge_command,
        upstream=upstream,
    )
    _finish_stage(partial, final, snap_id, stage)
    return final


def run_r1(result_root: Path, snap_id: str) -> None:
    _require_generated_family(result_root, snap_id)
    partial, final = _stage_paths(result_root, "R1")
    design = JOURNAL / "eps_constraint_design_manifest.json"
    command = [
        sys.executable,
        JOURNAL / "eps_constraint.py",
        "--manifest",
        JOURNAL / "rho_manifest.csv",
        "--out-dir",
        partial,
        "--snap-id",
        snap_id,
        "--time-limit-s",
        _manifest_time_limit(design),
        "--formal",
        "--no-canonical-assignment",
    ]
    _run_logged(command, log_path=partial / "R1.log", snap_id=snap_id, stage="R1")
    _require_stage_files(partial, R1_REQUIRED_FILES, "R1")
    _finish_stage(partial, final, snap_id, "R1")


def run_r1b(result_root: Path, snap_id: str) -> None:
    require((result_root / "R1").is_dir(), "R1 must pass before R1b")
    _require_generated_family(result_root, snap_id)
    partial, final = _stage_paths(result_root, "R1b")
    design = JOURNAL / "run_e4_design_manifest.json"
    command = [
        sys.executable,
        JOURNAL / "run_e4.py",
        "--manifest",
        JOURNAL / "rho_manifest.csv",
        "--out-dir",
        partial,
        "--mode",
        "horizon",
        "--snap-id",
        snap_id,
        "--time-limit-s",
        _manifest_time_limit(design),
        "--formal",
        "--no-canonical-assignment",
    ]
    _run_logged(command, log_path=partial / "R1b.log", snap_id=snap_id, stage="R1b")
    _require_stage_files(partial, R1B_REQUIRED_FILES, "R1b")
    _finish_stage(partial, final, snap_id, "R1b")


def run_r1_sharded(result_root: Path, snap_id: str,
                   rhos: Sequence[float],
                   configuration: Mapping[str, Any]) -> None:
    _require_generated_family(result_root, snap_id)
    _run_owner_sharded_stage(
        result_root,
        stage="R1",
        snap_id=snap_id,
        rhos=rhos,
        configuration=configuration,
        shard_command_factory=(
            lambda rho, out_dir: _r1_shard_command(out_dir, rho, snap_id)
        ),
        merge_command_factory=(
            lambda out_dir, shard_dirs: _r1_merge_command(
                out_dir, shard_dirs, snap_id)
        ),
        shard_required_files=R1_REQUIRED_FILES,
        merged_required_files=(
            *R1_REQUIRED_FILES,
            "r1_shard_merge_receipt.json",
        ),
    )


def run_r1b_sharded(result_root: Path, snap_id: str,
                    rhos: Sequence[float],
                    configuration: Mapping[str, Any]) -> None:
    # These three receipts exist only after the complete R1 shard family was
    # owner-merged, hashed, and atomically promoted.  No R1b process can start
    # while R1 is merely partial.
    r1 = result_root / "R1"
    r1_inventory = r1 / "stage_hashes.json"
    r1_orchestration = r1 / "r1_shard_orchestration.json"
    r1_merge_receipt = r1 / "r1_shard_merge_receipt.json"
    for path in (r1_inventory, r1_orchestration, r1_merge_receipt):
        require(path.is_file(),
                f"merged R1 receipt required before R1b sharding: {path}")
    _require_generated_family(result_root, snap_id)
    upstream = {
        "required_stage": "R1",
        "R1_stage_hash_inventory_sha256": sha256_file(r1_inventory),
        "R1_shard_orchestration_sha256": sha256_file(r1_orchestration),
        "R1_owner_merge_receipt_sha256": sha256_file(r1_merge_receipt),
    }
    _run_owner_sharded_stage(
        result_root,
        stage="R1b",
        snap_id=snap_id,
        rhos=rhos,
        configuration=configuration,
        shard_command_factory=(
            lambda rho, out_dir: _r1b_shard_command(out_dir, rho, snap_id)
        ),
        merge_command_factory=(
            lambda out_dir, shard_dirs: _r1b_merge_command(
                out_dir, shard_dirs, snap_id)
        ),
        shard_required_files=R1B_REQUIRED_FILES,
        merged_required_files=(
            *R1B_REQUIRED_FILES,
            "e4_shard_receipts.csv",
        ),
        upstream=upstream,
    )


def run_r3(result_root: Path, snap_id: str) -> None:
    _require_generated_family(result_root, snap_id)
    partial, final = _stage_paths(result_root, "R3")
    command = [
        sys.executable,
        JOURNAL / "rolling_horizon.py",
        "--stage",
        "R3",
        "--manifest",
        JOURNAL / "rolling_horizon_design_manifest.json",
        "--snap-id",
        snap_id,
        "--out-dir",
        partial,
    ]
    _run_logged(command, log_path=partial / "R3.log", snap_id=snap_id, stage="R3")
    _finish_stage(partial, final, snap_id, "R3")


def run_r4(result_root: Path, snap_id: str) -> None:
    partial, final = _stage_paths(result_root, "R4")
    command = [
        sys.executable,
        JOURNAL / "scalability.py",
        "--manifest",
        JOURNAL / "r4_size_design_manifest.json",
        "--manifest-tightness",
        JOURNAL / "r4_tightness_design_manifest.json",
        "--snap-id",
        snap_id,
        "--outdir",
        partial,
        "--emit",
    ]
    _run_logged(command, log_path=partial / "R4.log", snap_id=snap_id, stage="R4")
    _finish_stage(partial, final, snap_id, "R4")


def run_plots(result_root: Path, snap_id: str) -> None:
    for stage in ("R1", "R1b", "R3", "R4"):
        require((result_root / stage).is_dir(), f"{stage} must pass before plots")
    partial, final = _stage_paths(result_root, "plots")
    commands = [
        [
            sys.executable,
            JOURNAL / "stability_constrained.py",
            "--input-dir",
            result_root / "R1",
            "--out-dir",
            partial,
        ],
        [
            sys.executable,
            JOURNAL / "plot_eps.py",
            "--input-dir",
            result_root / "R1",
            "--out-dir",
            partial,
        ],
        [
            sys.executable,
            JOURNAL / "plot_e4.py",
            "--input-dir",
            result_root / "R1b",
            "--out-dir",
            partial,
        ],
        [
            sys.executable,
            JOURNAL / "plot_rolling.py",
            "--input-dir",
            result_root / "R3",
            "--out-dir",
            partial,
        ],
        [
            sys.executable,
            JOURNAL / "plot_scalability.py",
            "--input-dir",
            result_root / "R4",
            "--outdir",
            partial,
        ],
    ]
    for index, command in enumerate(commands, start=1):
        _run_logged(
            command,
            log_path=partial / f"plots_{index}.log",
            snap_id=snap_id,
            stage=f"plots-{index}",
        )
    _finish_stage(partial, final, snap_id, "plots")


def run_r5(result_root: Path, snap_id: str) -> None:
    require((result_root / "R1b").is_dir(), "R1b must pass before R5")
    partial, final = _stage_paths(result_root, "R5")
    command = [
        sys.executable,
        JOURNAL / "r5_stats.py",
        "--e4-input",
        result_root / "R1b" / "e4_raw.csv",
        "--rho-manifest",
        JOURNAL / "rho_manifest.csv",
        "--output-dir",
        partial,
        "--snap-id",
        snap_id,
        "--formal",
    ]
    _run_logged(command, log_path=partial / "R5.log", snap_id=snap_id, stage="R5")
    _require_stage_files(
        partial,
        (
            "r5_scenario_effects.csv",
            "r5_by_rho.csv",
            "r5_bootstrap.csv",
            "r5_run_manifest.json",
            "r5_output_hashes.json",
        ),
        "R5",
    )
    _finish_stage(partial, final, snap_id, "R5")


RUNNERS = {
    "environment": run_environment,
    "preflight": run_preflight,
    "generator": run_generator,
    "R1": run_r1,
    "R1b": run_r1b,
    "R3": run_r3,
    "R4": run_r4,
    "plots": run_plots,
    "R5": run_r5,
}


def run_primary_sharded(result_root: Path, snap_id: str) -> None:
    _require_fresh_pipeline(result_root, PRIMARY_STAGES, "primary-sharded")
    configuration = _shard_configuration()
    rhos = _registered_rhos()
    run_environment(result_root, snap_id)
    run_preflight(result_root, snap_id)
    run_generator(result_root, snap_id)
    run_r1_sharded(result_root, snap_id, rhos, configuration)
    require((result_root / "R1" / "stage_hashes.json").is_file(),
            "R1 did not atomically complete before the R1b gate")
    run_r1b_sharded(result_root, snap_id, rhos, configuration)
    require((result_root / "R1b" / "stage_hashes.json").is_file(),
            "R1b did not atomically complete before the R5 gate")
    run_r5(result_root, snap_id)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snap-id", required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("all", "primary", "primary-sharded") + STAGES,
        default="all")
    args = parser.parse_args(argv)
    snap_id = require_snap_id(args.snap_id)
    require(
        EPHEMERAL_MARKER.is_file(),
        "formal runner refuses the source tree: execute only in the Docker "
        "ephemeral SNAP extraction containing .dchhr_ephemeral_snapshot",
    )
    result_root = args.result_root.resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    if args.stage == "primary-sharded":
        run_primary_sharded(result_root, snap_id)
    elif args.stage == "primary":
        _require_fresh_pipeline(result_root, PRIMARY_STAGES, "primary")
        for stage in PRIMARY_STAGES:
            RUNNERS[stage](result_root, snap_id)
    elif args.stage == "all":
        for stage in STAGES:
            RUNNERS[stage](result_root, snap_id)
    else:
        RUNNERS[args.stage](result_root, snap_id)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FormalRunError as exc:
        print(f"FORMAL_RUN_ABORTED: {exc}", file=sys.stderr)
        raise SystemExit(2)
