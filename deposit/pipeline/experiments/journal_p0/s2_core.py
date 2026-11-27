"""
s2_core.py — shared fail-fast, structured-solver-result and per-stage validation library
for the LLM-DCHHR ``experiments/journal_p0`` pipelines.

Round S2 (code + design fixes). This module is *candidate work for independent review*: it is
mechanism, not evidence. Nothing in here proves anything about any published number.

Scope (REQ-2, REQ-3, and the §4 global validation contract of
``audit_outputs/rerun_plan_v2_1.md``):

  * fail-fast helpers usable by every pipeline — no ``continue``, no ``if r:``, no
    warning-and-proceed. Every guard raises, and every raise is fatal to the run;
  * :class:`SolverResult` — a structured solver outcome carrying the status **code** *and* the
    library's own status **name**, the primal ``P``, the dual bound ``B``, the dropped deviation
    constant ``C``, the raw solver gap, the **full-objective** gap recomputed with ``C``, and a
    declared stop reason;
  * :class:`StageSolverPolicy` — the stage-specific admissible-status gate of REQ-13 §A, built
    **from a design manifest** and never from the observed output;
  * :func:`validate_stage_output` — the per-stage validator: ``[shape] [keys] [coverage]
    [solver] [stoch] [numeric] [slots] [hashes]`` against an **independent** design grid;
  * :func:`canonical_write` — the rule that a partial or failing run may **never** overwrite a
    canonical file: temp -> validate the landed bytes -> promote, with rollback verification of
    the single canonical file on a caught failure.

Deliberate non-claims (E-class epistemics of the audit plan):

  * A fixture, a preflight or a hash agreement is **not** a proof, **not** validation, **not**
    correctness, and **not** evidence of an absence of fabrication.
  * :func:`canonical_write` promotes **one** file with :func:`os.replace`. On a POSIX filesystem
    that single rename is atomic, and on a caught failure this module re-reads the canonical
    bytes and asserts they are unchanged. **No pair-atomicity and no family-atomicity is
    claimed**: a crash between two successive promotions can leave a half-updated family, and an
    uncaught kill (SIGKILL, power loss) runs no rollback at all.
  * ``theta_j`` — the allowed-slot set of lin2025 Eq. (5) — is in general a **set membership**
    test ``q in theta_j``. In the data shipped with this repository ``|theta_j| = 1``, so it
    degenerates to a fixed shift *for this data*. :func:`slot_allowed` keeps the set form; the
    degeneracy is a property of the shipped instance, never a general equality.
  * ``slot`` is packing **data**, not a decision variable.

Path discipline: every path is resolved from ``__file__``. This module contains no absolute
runtime path and must not import anything from ``audit_outputs/`` (see
:func:`assert_no_staging_import`).
"""
from __future__ import annotations

import contextlib
import csv
import datetime as _dt
import hashlib
import itertools
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

__all__ = [
    # exceptions
    "PipelineFailure", "ValidationFailure", "SolverStatusFailure", "CanonicalWriteRefused",
    "ManifestFailure",
    # fail-fast helpers
    "fatal", "require", "require_finite", "require_finite_mapping", "require_keys",
    "field_index_by_name", "value_by_name", "require_no_unplaced", "require_nonempty_candidates",
    "require_flat_feasible", "cap_fp_slack", "slot_allowed", "require_slot_allowed",
    # paths
    "Paths", "paths", "assert_no_staging_import", "assert_no_absolute_runtime_paths",
    # solver
    "StatusSpec", "StageSolverPolicy", "SolverResult", "full_objective_gap", "signed_gap_pct",
    "require_gap_sign_admissible", "assert_not_labeled_optimal",
    # stochastic provenance
    "RngProvenance", "derive_seed", "make_rng", "require_distinct_rngs", "assignment_hash",
    "STOP_REASONS",
    # validation
    "DesignGrid", "StageOutputSpec", "ValidationReport", "validate_stage_output",
    "sha256_file", "sha256_bytes", "hash_manifest", "require_snap_binding",
    # canonical output protection
    "canonical_write",
    # canonical column names
    "COL", "RUN_LEVEL_KEY",
]

# --------------------------------------------------------------------------------------------
# 0. exceptions — every guard in this module is FATAL. Nothing here returns a soft failure.
# --------------------------------------------------------------------------------------------


class PipelineFailure(RuntimeError):
    """Base class for every fatal pipeline abort. Callers must not catch this to continue."""

    def __init__(self, message: str, **context: Any) -> None:
        self.context = dict(context)
        if context:
            rendered = " ".join("{}={!r}".format(k, context[k]) for k in sorted(context))
            message = "{} | {}".format(message, rendered)
        super().__init__(message)


class ValidationFailure(PipelineFailure):
    """A stage validator found at least one violation. Carries every violation, not the first."""

    def __init__(self, message: str, violations: Sequence[str], **context: Any) -> None:
        self.violations = list(violations)
        body = message + "\n" + "\n".join("  [FAIL] " + v for v in self.violations)
        super().__init__(body, **context)


class SolverStatusFailure(PipelineFailure):
    """A solver returned a status outside the stage's declared admissible set (REQ-13 §A)."""


class ManifestFailure(PipelineFailure):
    """The design manifest is missing, malformed, or does not declare a required field."""


class CanonicalWriteRefused(PipelineFailure):
    """A partial / unvalidated result was about to reach a canonical path. It did not."""


def fatal(message: str, **context: Any) -> "None":
    """Abort the run. Named separately so that ``grep -n 'fatal('`` finds every abort site."""
    raise PipelineFailure(message, **context)


def require(condition: Any, message: str, **context: Any) -> None:
    """Fail fast unless ``condition`` is truthy. ``message`` must name the offending key."""
    if not condition:
        raise PipelineFailure(message, **context)


# --------------------------------------------------------------------------------------------
# 1. numeric / structural fail-fast helpers
# --------------------------------------------------------------------------------------------


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def require_finite(value: Any, name: str, **context: Any) -> float:
    """Fail fast on NaN, +-inf, None, or a non-numeric value in a computed quantity (REQ-2)."""
    if value is None:
        raise PipelineFailure("non-finite computed quantity: {} is None".format(name), **context)
    if isinstance(value, bool) or isinstance(value, int):
        return float(value)
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        raise PipelineFailure(
            "non-numeric computed quantity: {}={!r}".format(name, value), **context)
    if not math.isfinite(as_float):
        raise PipelineFailure(
            "non-finite computed quantity: {}={!r}".format(name, value), **context)
    return as_float


def require_finite_mapping(mapping: Mapping[str, Any], *, where: str,
                           skip: Sequence[str] = ()) -> None:
    """NaN/inf scan over an about-to-be-persisted record. ``None`` is allowed only for keys the
    caller lists in ``skip`` — every other key must carry a finite number or a non-empty string."""
    skipset = set(skip)
    for key in sorted(mapping):
        if key in skipset:
            continue
        value = mapping[key]
        if value is None:
            raise PipelineFailure("empty cell in record: {}".format(key), where=where)
        if isinstance(value, str):
            require(value != "", "empty string cell in record: {}".format(key), where=where)
            continue
        if isinstance(value, (list, tuple, dict)):
            continue
        require_finite(value, key, where=where)


def require_keys(mapping: Mapping[str, Any], names: Sequence[str], *, where: str) -> None:
    """Every metadata / column name is located **by name**, never by a hard-coded index."""
    missing = [n for n in names if n not in mapping]
    require(not missing, "missing required field(s) located by name: {}".format(missing),
            where=where, present=sorted(mapping))


def field_index_by_name(header: Sequence[str], name: str, *, where: str) -> int:
    """Resolve a CSV column position from its **name**. Fails fast if absent or duplicated."""
    hits = [i for i, h in enumerate(header) if h == name]
    require(len(hits) == 1,
            "column must appear exactly once, located by name: {!r}".format(name),
            where=where, occurrences=len(hits), header=list(header))
    return hits[0]


def value_by_name(row: Sequence[Any], header: Sequence[str], name: str, *, where: str) -> Any:
    return row[field_index_by_name(header, name, where=where)]


def cap_fp_slack(cap: float) -> float:
    """Floating-point allowance for a capacity guard: ``math.ulp(cap) * 8``.

    A fixed ``1e-6`` is **not** an admissible final pass allowance: it is 13 orders of magnitude
    above the representation error of a 360.0-minute cap and can hide a real overload of a
    micro-minute scale, while also being *looser* than the MILP's own ``<= cap`` row bound, which
    lets a heuristic appear to beat a proven-optimal exact solve.
    """
    return math.ulp(float(cap)) * 8.0


def require_flat_feasible(load: Mapping[Any, float], cap: float, *, where: str) -> None:
    """An incumbent that is not flat-feasible aborts the run (REQ-2)."""
    slack = cap_fp_slack(cap)
    for key in sorted(load, key=repr):
        value = require_finite(load[key], "load[{!r}]".format(key), where=where)
        if value > cap + slack:
            raise PipelineFailure(
                "incumbent not flat-feasible at cell {!r}: load={!r} cap={!r} "
                "allowance={!r}".format(key, value, cap, slack), where=where)


def require_no_unplaced(assign: Mapping[Any, Any], tasks: Iterable[Any], *, where: str) -> None:
    """An unplaced task aborts the run (REQ-2). Use only where the design forbids dropping."""
    unplaced = [t for t in tasks if assign.get(t) is None]
    require(not unplaced,
            "unplaced task(s) where the design assumed full service: {}".format(unplaced[:20]),
            where=where, n_unplaced=len(unplaced))


def require_nonempty_candidates(cand: Mapping[Any, Sequence[Any]], *, where: str) -> None:
    """An empty candidate set where the design assumed a non-empty one aborts (REQ-2)."""
    empty = [t for t in sorted(cand, key=repr) if not cand[t]]
    require(not empty,
            "empty candidate set where the design assumed non-empty: {}".format(empty[:20]),
            where=where, n_empty=len(empty))


def slot_allowed(q: Any, theta_j: Iterable[Any]) -> bool:
    """Allowed-slot semantics of lin2025 Eq. (5): **set membership** ``q in theta_j``.

    Kept in set form on purpose. In the data shipped here ``|theta_j| = 1`` for every patient, so
    this happens to degenerate to a single fixed shift — that is a **property of this data**, not
    a general equality, and callers must not replace the membership test with ``q == theta``.
    """
    return q in set(theta_j)


def require_slot_allowed(q: Any, theta_j: Iterable[Any], *, where: str, key: Any = None) -> None:
    theta = sorted(set(theta_j), key=repr)
    require(theta, "empty allowed-slot set theta_j", where=where, key=key)
    require(slot_allowed(q, theta),
            "slot {!r} not in allowed-slot set theta_j={!r}".format(q, theta),
            where=where, key=key, theta_cardinality=len(theta))


# --------------------------------------------------------------------------------------------
# 2. paths — everything resolved from __file__, no absolute runtime path anywhere
# --------------------------------------------------------------------------------------------

_BASE_INSTANCE_RELATIVE = os.path.join(
    "data_problems_solutions", "HHCP-patient-65-task-256-nurse-59.csv")


@dataclass(frozen=True)
class Paths:
    journal_p0: str
    experiments: str
    repo_root: str
    base_instance: str


def paths(*, require_base_instance: bool = True) -> Paths:
    """Resolve the repository layout from this module's own location.

    ``base_instance`` is the real benchmark CSV. It is resolved **and existence-checked**; there
    is deliberately **no fallback**: renaming the file must make every consumer exit non-zero
    (REQ-11 fault injection), not silently degrade.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    experiments = os.path.dirname(here)
    repo_root = os.path.dirname(experiments)
    base = os.path.join(experiments, _BASE_INSTANCE_RELATIVE)
    if require_base_instance and not os.path.isfile(base):
        raise PipelineFailure(
            "base instance CSV not found; there is no fallback by design",
            expected_path=os.path.relpath(base, repo_root),
            journal_p0=os.path.relpath(here, repo_root))
    return Paths(journal_p0=here, experiments=experiments, repo_root=repo_root,
                 base_instance=base)


def assert_no_staging_import() -> None:
    """Official runtime code must not import from the sealed ``audit_outputs/gate_g*_staging``."""
    offenders = []
    for name, module in list(sys.modules.items()):
        origin = getattr(module, "__file__", None)
        if not origin:
            continue
        norm = origin.replace(os.sep, "/")
        if "/audit_outputs/" in norm:
            offenders.append((name, norm))
    require(not offenders,
            "official runtime code imported a module from audit_outputs/: {}".format(offenders),
            n_offenders=len(offenders))
    bad_path = [p for p in sys.path if "/audit_outputs/" in p.replace(os.sep, "/")]
    require(not bad_path, "sys.path contains an audit_outputs/ entry: {}".format(bad_path))


def assert_no_absolute_runtime_paths(*module_files: str) -> None:
    """Grep gate: no absolute home-directory runtime path literal in official code.

    Official runtime code must resolve every path from ``__file__``. An absolute path rooted at a
    user home directory makes the code machine-specific and silently unreproducible elsewhere.
    """
    # assembled at run time so that this gate does not trip on its own source line
    needles = ("/" + "Users" + "/", "/" + "home" + "/", "C:" + chr(92) + "Users")
    offenders = []
    for module_file in module_files:
        path = os.path.abspath(module_file)
        with open(path, "r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                for needle in needles:
                    if needle in line:
                        offenders.append("{}:{}".format(os.path.basename(path), lineno))
    require(not offenders,
            "absolute runtime path literal found in official code: {}".format(offenders))


# --------------------------------------------------------------------------------------------
# 3. canonical column names — imported by the other pipelines so schemas cannot drift
# --------------------------------------------------------------------------------------------


class COL:
    """Canonical persisted column names. Other pipelines import these instead of retyping."""

    STAGE = "stage"
    # --- solver block (REQ-13) ---
    STATUS_CODE = "solver_status_code"
    STATUS_NAME = "solver_status_name"
    PROVEN_OPTIMAL = "proven_optimal"
    # PRIMAL / DUAL are the solver's own values on the LINEARIZED objective (the deviation
    # constant C is dropped from the model). *_FULL add C back, so the full-objective gap is
    # ((P+C) - (B+C)) / |P+C| and both forms are auditable from the row itself.
    PRIMAL = "obj_incumbent"
    DUAL = "mip_dual_bound"
    PRIMAL_FULL = "obj_incumbent_full_objective"
    DUAL_FULL = "mip_dual_bound_full_objective"
    DEV_CONST_C = "deviation_constant_C"
    GAP_SOLVER_RAW = "mip_gap_solver_raw"
    GAP_FULL = "gap_full_objective"
    GAP_FULL_UNDEF = "gap_full_objective_undefined_reason"
    STOP_REASON = "stop_reason"
    TIME_LIMIT = "time_limit_s"
    WALL_TIME = "wall_time_s"
    NODES = "nodes"
    # --- stochastic-run block (REQ-14) ---
    ALGORITHM = "algorithm"
    REPEAT_INDEX = "repeat_index"
    RNG_SEED = "rng_seed"
    ITERS_BUDGET = "iterations_budget"
    ITERS_EXECUTED = "iterations_executed"
    EVALUATIONS = "evaluations"
    ASSIGN_HASH = "assignment_hash"
    GAP_SIGNED = "gap_signed_pct"
    REFERENCE_PROVEN_OPTIMAL = "reference_proven_optimal"
    BUDGET_REGIME = "budget_regime"
    # --- tightness / candidate-graph block (REQ-15) ---
    RHO_AVAIL = "rho_avail"           # PRE-EXISTING column. Not an addition.
    RHO_COMMIT = "rho_commit"         # PRE-EXISTING column. Not an addition.
    RHO_COMMIT_TARGET = "rho_commit_target"   # PRE-EXISTING column. Not an addition.
    M_AVAIL = "M"                     # PRE-EXISTING column. Not an addition.
    C_NOMINAL = "C"                   # PRE-EXISTING column (design constant). Not an addition.
    CAND_DEG_MIN = "cand_degree_min"
    CAND_DEG_MEAN = "cand_degree_mean"
    CAND_DEG_MAX = "cand_degree_max"
    CONFOUND_MODE = "confound_control_mode"
    RHO_BAND_LO = "rho_avail_band_lo"
    RHO_BAND_HI = "rho_avail_band_hi"


RUN_LEVEL_KEY = (COL.STAGE, "T_q", "seed", COL.ALGORITHM, COL.REPEAT_INDEX)

#: Declared enumeration for :data:`COL.STOP_REASON`. Never null, never outside this set.
STOP_REASONS = (
    "proven_optimal",
    "time_limit",
    "budget_exhausted",
    "no_improvement_early_stop",
    "deterministic_single_pass",
    "empty_instance",
    "solver_error",
)


# --------------------------------------------------------------------------------------------
# 4. solver status policy (REQ-13 §A / §C) — declared by a manifest, before the run
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StatusSpec:
    code: int
    name: str


@dataclass(frozen=True)
class StageSolverPolicy:
    """The admissible-status gate for ONE stage, built from that stage's design manifest.

    ``optimal`` is the subset of ``admissible`` that the manifest declares to be *proven optimal*
    by name and integer code. Any status outside ``admissible`` aborts the stage (REQ-2/REQ-13).
    A status inside ``admissible`` but outside ``optimal`` yields ``proven_optimal = False`` and
    may **never** be described as optimal — see :func:`assert_not_labeled_optimal`.
    """

    stage: str
    admissible: Tuple[StatusSpec, ...]
    optimal: Tuple[StatusSpec, ...]
    accepts_time_limit_incumbent: bool
    time_limit_s: Optional[float]
    solver_options: Mapping[str, Any] = field(default_factory=dict)
    manifest_path: Optional[str] = None
    manifest_sha256: Optional[str] = None

    # -- construction -------------------------------------------------------------------
    @staticmethod
    def from_manifest(manifest: Mapping[str, Any], *, stage: str,
                      manifest_path: Optional[str] = None,
                      manifest_sha256: Optional[str] = None) -> "StageSolverPolicy":
        require_keys(manifest, ["stage", "solver"], where="design manifest root")
        require(manifest["stage"] == stage,
                "manifest declares a different stage than the caller",
                manifest_stage=manifest["stage"], caller_stage=stage)
        solver = manifest["solver"]
        require(isinstance(solver, dict), "manifest['solver'] must be an object", stage=stage)
        require_keys(solver, ["admissible_statuses", "optimal_statuses",
                              "accepts_time_limit_incumbent", "time_limit_s"],
                     where="design manifest ['solver'] for stage {}".format(stage))

        def _specs(raw: Any, label: str) -> Tuple[StatusSpec, ...]:
            require(isinstance(raw, list) and raw,
                    "manifest['solver'][{!r}] must be a non-empty list".format(label),
                    stage=stage)
            out = []
            for entry in raw:
                require(isinstance(entry, dict),
                        "each {} entry must be an object with 'code' and 'name'".format(label),
                        stage=stage, entry=entry)
                require_keys(entry, ["code", "name"],
                             where="manifest['solver'][{!r}] entry".format(label))
                require(isinstance(entry["code"], int) and not isinstance(entry["code"], bool),
                        "status code must be an integer (the library's own code)",
                        stage=stage, entry=entry)
                require(isinstance(entry["name"], str) and entry["name"],
                        "status name must be the library's own non-empty label",
                        stage=stage, entry=entry)
                out.append(StatusSpec(code=int(entry["code"]), name=str(entry["name"])))
            codes = [s.code for s in out]
            require(len(set(codes)) == len(codes),
                    "duplicate status code in manifest['solver'][{!r}]".format(label),
                    stage=stage, codes=codes)
            return tuple(out)

        admissible = _specs(solver["admissible_statuses"], "admissible_statuses")
        optimal = _specs(solver["optimal_statuses"], "optimal_statuses")
        adm_codes = {s.code for s in admissible}
        stray = sorted(s.code for s in optimal if s.code not in adm_codes)
        require(not stray,
                "optimal_statuses must be a subset of admissible_statuses",
                stage=stage, stray_codes=stray)
        for spec in optimal:
            match = [a for a in admissible if a.code == spec.code]
            require(match and match[0].name == spec.name,
                    "status name disagrees between optimal_statuses and admissible_statuses",
                    stage=stage, code=spec.code, optimal_name=spec.name,
                    admissible_name=match[0].name if match else None)
        accepts_tl = solver["accepts_time_limit_incumbent"]
        require(isinstance(accepts_tl, bool),
                "manifest['solver']['accepts_time_limit_incumbent'] must be a boolean",
                stage=stage, value=accepts_tl)
        if not accepts_tl:
            # a stage that does not accept a time-limit incumbent must declare optimal-only
            extra = sorted(s.code for s in admissible if s.code not in {o.code for o in optimal})
            require(not extra,
                    "stage declares accepts_time_limit_incumbent=false but admits non-optimal "
                    "statuses; declare them and the reason, or remove them (REQ-13 A)",
                    stage=stage, non_optimal_admissible_codes=extra)
        tl = solver["time_limit_s"]
        if tl is not None:
            tl = require_finite(tl, "solver.time_limit_s", stage=stage)
            require(tl > 0, "time_limit_s must be positive", stage=stage, value=tl)
        raw_options = solver.get("options", {})
        require(isinstance(raw_options, dict),
                "manifest['solver']['options'] must be an object when present",
                stage=stage, value=raw_options)
        options = dict(raw_options)
        if "time_limit" in options:
            option_tl = require_finite(
                options["time_limit"], "solver.options.time_limit", stage=stage)
            require(tl is not None and option_tl == tl,
                    "solver.options.time_limit must exactly equal solver.time_limit_s",
                    stage=stage, option_time_limit=option_tl, time_limit_s=tl)
            options["time_limit"] = option_tl
        elif tl is not None:
            options["time_limit"] = tl
        if "presolve" in options:
            require(isinstance(options["presolve"], bool),
                    "solver.options.presolve must be a boolean",
                    stage=stage, value=options["presolve"])
        return StageSolverPolicy(
            stage=stage, admissible=admissible, optimal=optimal,
            accepts_time_limit_incumbent=accepts_tl, time_limit_s=tl,
            solver_options=options,
            manifest_path=manifest_path, manifest_sha256=manifest_sha256)

    @staticmethod
    def from_manifest_file(path: str, *, stage: str) -> "StageSolverPolicy":
        require(os.path.isfile(path), "design manifest file not found", path=path)
        digest = sha256_file(path)
        with open(path, "r", encoding="utf-8") as handle:
            try:
                manifest = json.load(handle)
            except ValueError as exc:
                raise ManifestFailure("design manifest is not valid JSON",
                                      path=path, error=str(exc))
        return StageSolverPolicy.from_manifest(manifest, stage=stage, manifest_path=path,
                                               manifest_sha256=digest)

    # -- use ----------------------------------------------------------------------------
    @property
    def admissible_codes(self) -> Set[int]:
        return {s.code for s in self.admissible}

    @property
    def optimal_codes(self) -> Set[int]:
        return {s.code for s in self.optimal}

    def status_name(self, code: int) -> str:
        for spec in self.admissible:
            if spec.code == code:
                return spec.name
        raise SolverStatusFailure(
            "solver status not declared in this stage's manifest, so it has no declared name",
            stage=self.stage, status_code=code,
            declared=sorted((s.code, s.name) for s in self.admissible))

    def is_proven_optimal(self, code: int) -> bool:
        return int(code) in self.optimal_codes

    def check(self, code: int, *, where: str) -> str:
        """Fail fast unless ``code`` is admissible for this stage. Returns the declared name."""
        if int(code) not in self.admissible_codes:
            raise SolverStatusFailure(
                "solver status outside this stage's declared admissible set (REQ-13 A)",
                stage=self.stage, status_code=int(code), where=where,
                admissible=sorted((s.code, s.name) for s in self.admissible))
        return self.status_name(int(code))

    def scipy_milp_options(self, *, cli_time_limit_s: Optional[float] = None,
                           require_cli_time_limit: bool = False) -> Dict[str, Any]:
        """Return the pre-registered SciPy/HiGHS options.

        Formal runners pass ``require_cli_time_limit=True``.  The command-line
        value is then mandatory and must exactly equal the value frozen in the
        design manifest; it is never allowed to silently override that manifest.
        """
        if require_cli_time_limit:
            require(cli_time_limit_s is not None,
                    "formal run requires an explicit --time-limit-s",
                    stage=self.stage, manifest_time_limit_s=self.time_limit_s)
        if cli_time_limit_s is not None:
            cli_value = require_finite(
                cli_time_limit_s, "cli_time_limit_s", stage=self.stage)
            require(self.time_limit_s is not None and cli_value == self.time_limit_s,
                    "CLI time limit must exactly equal the pre-registered manifest value",
                    stage=self.stage, cli_time_limit_s=cli_value,
                    manifest_time_limit_s=self.time_limit_s)
        return dict(self.solver_options)


def assert_not_labeled_optimal(labels: Iterable[str], *, proven_optimal: bool,
                               where: str) -> None:
    """Grep gate in code form: a non-proven row may not be *called* optimal anywhere.

    ``labels`` are the column names, captions, headings or sentences about to be emitted for a
    row whose ``proven_optimal`` is False. A time-limit incumbent is an **incumbent**.
    """
    if proven_optimal:
        return
    banned = ("optimal", "optimum", "proven", "exact-optimal", "最優", "最佳解", "已證明")
    offenders = []
    for label in labels:
        low = str(label).lower()
        for token in banned:
            if token in low:
                offenders.append((label, token))
    require(not offenders,
            "a non-proven (e.g. time-limit incumbent) result must never be described as "
            "optimal/proven: {}".format(offenders), where=where)


# --------------------------------------------------------------------------------------------
# 5. gap arithmetic (REQ-13 §B) and the structured solver result
# --------------------------------------------------------------------------------------------

#: Relative-gap denominators below this magnitude make a *relative* gap undefined, not zero.
GAP_DENOM_FLOOR = 1e-12


def full_objective_gap(primal: float, dual_bound: float, deviation_constant_C: float,
                       *, where: str) -> Tuple[Optional[float], Optional[str]]:
    """Recompute the gap on the **full** objective: ``((P+C) - (B+C)) / |P+C|``.

    The linearized deviation term drops the constant ``C = w4 * |xbar|``. Dropping a constant
    does not move the argmin, but a *relative* gap is not translation-invariant, so the solver's
    own ``mip_gap`` is the wrong quantity to report. Returns ``(gap, undefined_reason)`` with
    exactly one of the two non-None.
    """
    P = require_finite(primal, "primal P", where=where)
    B = require_finite(dual_bound, "dual bound B", where=where)
    C = require_finite(deviation_constant_C, "deviation constant C", where=where)
    numerator = (P + C) - (B + C)          # == P - B, written in full-objective form on purpose
    denominator = abs(P + C)
    if denominator < GAP_DENOM_FLOOR:
        return None, "zero_full_objective_denominator"
    return numerator / denominator, None


def signed_gap_pct(candidate_obj: float, reference_obj: float, *, where: str) -> float:
    """Signed relative gap in percent. **No ``max(0, ...)`` clamp.**

    A clamp would report a heuristic that beats the reference as ``0.0`` and thereby hide a model
    mismatch instead of surfacing it.
    """
    cand = require_finite(candidate_obj, "candidate objective", where=where)
    ref = require_finite(reference_obj, "reference objective", where=where)
    require(abs(ref) >= GAP_DENOM_FLOOR,
            "reference objective is ~0, so a relative gap is undefined (not zero)",
            where=where, reference=ref)
    return (cand - ref) / abs(ref) * 100.0


def require_gap_sign_admissible(gap_pct: float, *, reference_proven_optimal: bool, where: str,
                                tolerance_pct: float = 0.0, **context: Any) -> None:
    """A negative gap against a proven-optimal reference is a defect, not a result (REQ-14 §6).

    It means the two methods are not scoring the same object — different feasibility tolerance,
    different objective, or a stale reference. It must fail the stage.
    """
    value = require_finite(gap_pct, "gap_signed_pct", where=where)
    if reference_proven_optimal and value < -abs(tolerance_pct):
        raise PipelineFailure(
            "heuristic beat a PROVEN-OPTIMAL exact solve: the two are not scoring the same "
            "object (objective, feasibility allowance, or reference is stale)",
            where=where, gap_signed_pct=value, tolerance_pct=tolerance_pct, **context)


@dataclass
class SolverResult:
    """Structured outcome of one exact solve. Carries code AND name, primal, dual, gap, reason."""

    stage: str
    status_code: int
    status_name: str
    proven_optimal: bool
    primal: Optional[float]
    dual_bound: Optional[float]
    deviation_constant_C: float
    gap_solver_raw: Optional[float]
    gap_full_objective: Optional[float]
    gap_full_objective_undefined_reason: Optional[str]
    stop_reason: str
    wall_time_s: float
    time_limit_s: Optional[float]
    nodes: Optional[int]
    assignment_hash: Optional[str]
    repeat_index: int = 1
    message: Optional[str] = None

    def __post_init__(self) -> None:
        require(self.stop_reason in STOP_REASONS,
                "stop_reason outside the declared enumeration",
                stop_reason=self.stop_reason, enumeration=list(STOP_REASONS))
        require_finite(self.wall_time_s, "wall_time_s", stage=self.stage)
        require(int(self.repeat_index) >= 1, "repeat_index is 1-based",
                repeat_index=self.repeat_index)
        if self.proven_optimal:
            require(self.stop_reason == "proven_optimal",
                    "proven_optimal=True requires stop_reason='proven_optimal'",
                    stop_reason=self.stop_reason, stage=self.stage)
        else:
            require(self.stop_reason != "proven_optimal",
                    "stop_reason='proven_optimal' requires proven_optimal=True",
                    stage=self.stage)

    # -- construction from scipy.optimize.milp --------------------------------------------
    @staticmethod
    def from_scipy(res: Any, *, policy: StageSolverPolicy, deviation_constant_C: float,
                   wall_time_s: float, primal: Optional[float],
                   assignment_hash_value: Optional[str], repeat_index: int = 1,
                   where: str = "solve_exact") -> "SolverResult":
        """Build from a SciPy/HiGHS ``milp`` result. Every attribute is read **by name**."""
        code = getattr(res, "status", None)
        require(code is not None, "solver result has no 'status' attribute", where=where)
        code = int(code)
        name = policy.check(code, where=where)          # FATAL if outside the declared set
        proven = policy.is_proven_optimal(code)
        dual = getattr(res, "mip_dual_bound", None)
        raw_gap = getattr(res, "mip_gap", None)
        nodes = getattr(res, "mip_node_count", None)
        message = getattr(res, "message", None)

        gap_full: Optional[float] = None
        gap_undef: Optional[str] = None
        if primal is None:
            gap_undef = "no_incumbent"
        elif dual is None:
            gap_undef = "no_dual_bound_reported"
        else:
            gap_full, gap_undef = full_objective_gap(
                primal, dual, deviation_constant_C, where=where)

        if proven:
            stop_reason = "proven_optimal"
        elif policy.accepts_time_limit_incumbent:
            stop_reason = "time_limit"
        else:                                            # unreachable: policy.check would abort
            stop_reason = "solver_error"

        # A non-proven row may not be labelled optimal anywhere in what we are about to persist.
        assert_not_labeled_optimal(
            [COL.PRIMAL, COL.DUAL, COL.GAP_FULL, COL.STOP_REASON, stop_reason],
            proven_optimal=proven, where=where)

        return SolverResult(
            stage=policy.stage, status_code=code, status_name=name, proven_optimal=proven,
            primal=(None if primal is None else require_finite(primal, "primal", where=where)),
            dual_bound=(None if dual is None else require_finite(dual, "dual", where=where)),
            deviation_constant_C=require_finite(deviation_constant_C, "C", where=where),
            gap_solver_raw=(None if raw_gap is None
                            else require_finite(raw_gap, "mip_gap", where=where)),
            gap_full_objective=gap_full, gap_full_objective_undefined_reason=gap_undef,
            stop_reason=stop_reason, wall_time_s=wall_time_s,
            time_limit_s=policy.time_limit_s,
            nodes=(None if nodes is None else int(nodes)),
            assignment_hash=assignment_hash_value, repeat_index=int(repeat_index),
            message=(None if message is None else str(message)))

    # -- persistence -----------------------------------------------------------------------
    def as_row(self) -> Dict[str, Any]:
        """The canonical solver block of a persisted row. Column names come from :class:`COL`."""
        return {
            COL.STAGE: self.stage,
            COL.STATUS_CODE: self.status_code,
            COL.STATUS_NAME: self.status_name,
            COL.PROVEN_OPTIMAL: bool(self.proven_optimal),
            COL.PRIMAL: self.primal,
            COL.DUAL: self.dual_bound,
            COL.PRIMAL_FULL: (None if self.primal is None
                              else self.primal + self.deviation_constant_C),
            COL.DUAL_FULL: (None if self.dual_bound is None
                            else self.dual_bound + self.deviation_constant_C),
            COL.DEV_CONST_C: self.deviation_constant_C,
            COL.GAP_SOLVER_RAW: self.gap_solver_raw,
            COL.GAP_FULL: self.gap_full_objective,
            COL.GAP_FULL_UNDEF: self.gap_full_objective_undefined_reason,
            COL.STOP_REASON: self.stop_reason,
            COL.TIME_LIMIT: self.time_limit_s,
            COL.WALL_TIME: round(self.wall_time_s, 6),
            COL.NODES: self.nodes,
            COL.ASSIGN_HASH: self.assignment_hash,
            COL.REPEAT_INDEX: self.repeat_index,
        }

    def as_canonical_row(self) -> Dict[str, Any]:
        """Stable scientific fields only.

        Measured wall time, node count and solver message are operational
        diagnostics.  They are intentionally excluded from canonical scientific
        CSVs so two identical solves can reproduce those CSV bytes.  Producers
        persist :meth:`as_diagnostics_row` in a separate diagnostics sidecar.
        """
        row = self.as_row()
        row.pop(COL.WALL_TIME, None)
        row.pop(COL.NODES, None)
        return row

    def as_diagnostics_row(self) -> Dict[str, Any]:
        """Operational fields allowed to vary across otherwise identical runs."""
        return {
            COL.STAGE: self.stage,
            COL.STATUS_CODE: self.status_code,
            COL.STATUS_NAME: self.status_name,
            COL.WALL_TIME: round(self.wall_time_s, 6),
            COL.NODES: self.nodes,
            COL.TIME_LIMIT: self.time_limit_s,
            "solver_message": self.message,
        }

    def describe(self) -> str:
        kind = "proven-optimal" if self.proven_optimal else "INCUMBENT (not proven optimal)"
        return ("stage={} status={}({}) {} P={} B={} C={} gap_full={} stop={}".format(
            self.stage, self.status_code, self.status_name, kind, self.primal, self.dual_bound,
            self.deviation_constant_C, self.gap_full_objective, self.stop_reason))


# --------------------------------------------------------------------------------------------
# 6. stochastic-run provenance (REQ-14) — distinct RNG objects are necessary, not sufficient
# --------------------------------------------------------------------------------------------


def derive_seed(algorithm: str, base_seed: int, repeat_index: int, **extra: Any) -> int:
    """Deterministic, cross-process-stable algorithm-specific seed.

    Uses BLAKE2b over a canonical string, not :func:`hash`, because ``PYTHONHASHSEED``
    randomizes :func:`hash` for strings and would make the seed unreproducible.
    """
    require(isinstance(algorithm, str) and algorithm, "algorithm must be a non-empty string")
    require(int(repeat_index) >= 1, "repeat_index is 1-based", repeat_index=repeat_index)
    payload = "|".join([
        "s2_core.derive_seed.v1", algorithm, str(int(base_seed)), str(int(repeat_index)),
        json.dumps({k: extra[k] for k in sorted(extra)}, sort_keys=True, default=str),
    ])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2 ** 63 - 1)


@dataclass
class RngProvenance:
    """One stochastic run's replayable identity: algorithm, its OWN seed, its OWN RNG object."""

    algorithm: str
    base_seed: int
    repeat_index: int
    seed: int
    rng: random.Random
    extra: Dict[str, Any] = field(default_factory=dict)

    def replay(self) -> random.Random:
        """A fresh RNG from the persisted seed alone — this is what makes the row replayable."""
        return random.Random(self.seed)

    def as_row(self) -> Dict[str, Any]:
        return {COL.ALGORITHM: self.algorithm, COL.REPEAT_INDEX: self.repeat_index,
                COL.RNG_SEED: self.seed}


def make_rng(algorithm: str, base_seed: int, repeat_index: int, **extra: Any) -> RngProvenance:
    """Construct a **fresh** RNG owned by exactly one algorithm-run. Never share the object."""
    seed = derive_seed(algorithm, base_seed, repeat_index, **extra)
    return RngProvenance(algorithm=algorithm, base_seed=int(base_seed),
                         repeat_index=int(repeat_index), seed=seed,
                         rng=random.Random(seed), extra=dict(extra))


def require_distinct_rngs(*provenances: RngProvenance) -> None:
    """Fail fast if two stochastic procedures share an RNG object or a seed."""
    ids = [id(p.rng) for p in provenances]
    require(len(set(ids)) == len(ids),
            "two stochastic procedures share one RNG object; each needs its own",
            algorithms=[p.algorithm for p in provenances])
    seeds = [p.seed for p in provenances]
    require(len(set(seeds)) == len(seeds),
            "two stochastic runs share one seed; each needs its own persisted seed",
            algorithms=[p.algorithm for p in provenances], seeds=seeds)


def assignment_hash(assign: Mapping[Any, Any]) -> str:
    """Order-independent SHA256 of an assignment mapping, for replay comparison."""
    items = sorted(((repr(k), repr(assign[k])) for k in assign))
    payload = json.dumps(items, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------------
# 7. hashing helpers
# --------------------------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    """SHA256 of the bytes **on disk**. Landed bytes are the verification authority."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_manifest(inputs: Sequence[str], outputs: Sequence[str], *,
                  repo_root: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """``[hashes]``: SHA256 of every input and every output, keyed by repo-relative path."""
    def _block(paths_: Sequence[str]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for p in paths_:
            require(os.path.isfile(p), "file to hash does not exist", path=p)
            key = os.path.relpath(p, repo_root) if repo_root else p
            out[key.replace(os.sep, "/")] = sha256_file(p)
        return out
    return {"inputs": _block(inputs), "outputs": _block(outputs)}


def require_snap_binding(design_manifest_path: str, cli_snap_id: str, *,
                         stage: str) -> str:
    """Validate a content-addressed SNAP id without embedding it in its archive.

    The outer runner recomputes the archive SHA256 and supplies that 64-hex
    value on the CLI.  The in-snapshot design manifest pre-registers this
    binding rule; it cannot contain its own content hash without a causal cycle.
    If a non-null literal identifier is nevertheless declared, it must match.
    """
    require(isinstance(cli_snap_id, str)
            and len(cli_snap_id.strip()) == 64
            and all(ch in "0123456789abcdefABCDEF"
                    for ch in cli_snap_id.strip()),
            "formal run requires --snap-id as a 64-hex content-addressed "
            "archive SHA256", stage=stage, value=cli_snap_id)
    require(os.path.isfile(design_manifest_path),
            "stage design manifest does not exist",
            stage=stage, path=design_manifest_path)
    with open(design_manifest_path, "r", encoding="utf-8") as handle:
        try:
            manifest = json.load(handle)
        except ValueError as exc:
            raise ManifestFailure("stage design manifest is not valid JSON",
                                  stage=stage, path=design_manifest_path,
                                  error=str(exc))
    snap = manifest.get("snap")
    require(isinstance(snap, dict),
            "stage design manifest must contain a snap object",
            stage=stage, path=design_manifest_path)
    require(snap.get("snap_binding")
            == "cli_content_addressed_archive_sha256",
            "stage design manifest did not pre-register the required SNAP binding",
            stage=stage, path=design_manifest_path,
            snap_binding=snap.get("snap_binding"))
    literal = snap.get("identifier")
    if literal is not None:
        require(isinstance(literal, str) and literal.strip() == cli_snap_id.strip(),
                "CLI --snap-id differs from the literal identifier declared in "
                "the stage design manifest",
                stage=stage, cli_snap_id=cli_snap_id,
                manifest_snap_id=literal, path=design_manifest_path)
    return cli_snap_id.strip().lower()


# --------------------------------------------------------------------------------------------
# 8. the independent design grid and the per-stage validator (REQ-3 + §4)
# --------------------------------------------------------------------------------------------


def _norm(value: Any) -> Any:
    """Normalize an axis value so that 16 == 16.0 == '16' compare equal across CSV round-trips."""
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PipelineFailure("non-finite axis value in design grid", value=value)
        if float(value).is_integer():
            return int(value)
        return round(float(value), 9)
    if isinstance(value, int):
        return int(value)
    text = str(value).strip()
    try:
        as_float = float(text)
    except ValueError:
        return text
    if as_float.is_integer():
        return int(as_float)
    return round(as_float, 9)


@dataclass
class DesignGrid:
    """An **independent** oracle for the designed cells of one stage.

    Built from the stage's design manifest / from the committed inputs re-derived by the caller —
    **never** read back from the artifact being validated. Reading the grid off the output turns
    the coverage check into a tautology, so :meth:`from_dataframe` refuses to build one.
    """

    name: str
    axes: "Dict[str, List[Any]]"
    declared_shortfalls: "List[Tuple[Any, ...]]" = field(default_factory=list)
    provenance: str = "design_manifest"

    def __post_init__(self) -> None:
        require(self.axes, "design grid has no axes", grid=self.name)
        for axis, values in self.axes.items():
            require(isinstance(values, (list, tuple)) and len(values) > 0,
                    "design axis must be a non-empty list", grid=self.name, axis=axis)
            normed = [_norm(v) for v in values]
            require(len(set(map(repr, normed))) == len(normed),
                    "duplicate value on design axis", grid=self.name, axis=axis, values=values)
            self.axes[axis] = normed
        full = self.full_tuples()
        bad = [t for t in map(self._normalize_tuple, self.declared_shortfalls) if t not in full]
        require(not bad,
                "declared shortfall is not a cell of the design grid",
                grid=self.name, offending=bad[:10])
        self.declared_shortfalls = [self._normalize_tuple(t) for t in self.declared_shortfalls]
        require(len(set(self.declared_shortfalls)) == len(self.declared_shortfalls),
                "duplicate declared shortfall", grid=self.name)

    @staticmethod
    def from_dataframe(*_args: Any, **_kwargs: Any) -> "DesignGrid":
        raise PipelineFailure(
            "a design grid may NOT be derived from the artifact it validates; that makes the "
            "coverage check a tautology. Re-derive it from the design manifest or from the "
            "committed inputs (REQ-3 rationale: the grid must be an independent oracle).")

    @property
    def axis_names(self) -> "List[str]":
        return list(self.axes.keys())

    def _normalize_tuple(self, tup: Sequence[Any]) -> Tuple[Any, ...]:
        require(len(tup) == len(self.axes),
                "grid tuple arity mismatch", grid=self.name,
                expected=len(self.axes), actual=len(tup))
        return tuple(_norm(v) for v in tup)

    def full_tuples(self) -> Set[Tuple[Any, ...]]:
        return set(itertools.product(*[self.axes[a] for a in self.axes]))

    def expected_tuples(self) -> Set[Tuple[Any, ...]]:
        return self.full_tuples() - set(self.declared_shortfalls)

    def expected_rows(self, *, rows_per_cell: int = 1) -> int:
        require(rows_per_cell >= 1, "rows_per_cell must be >= 1", grid=self.name)
        return len(self.expected_tuples()) * rows_per_cell


@dataclass
class StageOutputSpec:
    """Everything the validator needs, all of it declared *before* the artifact is inspected."""

    stage: str
    artifact_name: str
    primary_key: Sequence[str]                 # the FULL design tuple, not a convenient subset
    grid: DesignGrid
    required_columns: Sequence[str]
    numeric_columns: Sequence[str]
    rows_per_cell: int = 1
    solver_policy: Optional[StageSolverPolicy] = None
    status_column: str = COL.STATUS_CODE
    #: ``(column, admissible values)`` selecting the rows the solver block applies to. A run-level
    #: artifact mixes exact rows (which carry a solver status) with heuristic rows (which do not),
    #: so the solver check must be scoped to the exact rows rather than skipped entirely.
    solver_row_filter: Optional[Tuple[str, Sequence[Any]]] = None
    # slots: independent expectation, computed by the caller from the committed plan
    slot_column: Optional[str] = None
    config_columns: Sequence[str] = ()
    expected_slots_fn: Optional[Callable[[Tuple[Any, ...]], Iterable[Any]]] = None
    # stochastic run-level block
    run_level: bool = False
    stochastic_algorithms: Sequence[str] = ()
    repeats_expected: Optional[int] = None
    # REQ-15 confound control
    rho_avail_band: Optional[Tuple[float, float]] = None
    # hashes / SNAP
    input_paths: Sequence[str] = ()
    snap_id: Optional[str] = None
    require_snap: Optional[bool] = None       # default: True for any stage named R*
    nullable_columns: Sequence[str] = ()

    def snap_required(self) -> bool:
        if self.require_snap is not None:
            return bool(self.require_snap)
        return self.stage.upper().startswith("R")


@dataclass
class ValidationReport:
    stage: str
    artifact: str
    expected_rows: int
    actual_rows: int
    checks: Dict[str, str]
    hashes: Dict[str, Dict[str, str]]
    snap_id: Optional[str]
    notes: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, default=str)


def _read_csv_rows(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Re-read the LANDED BYTES. The bytes on disk are the verification authority."""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    require(rows, "artifact is empty (no header row)", path=path)
    header = [h.strip() for h in rows[0]]
    require(len(set(header)) == len(header), "duplicate column name in artifact header",
            path=path, header=header)
    records = []
    for lineno, raw in enumerate(rows[1:], 2):
        require(len(raw) == len(header),
                "ragged CSV row", path=path, line=lineno,
                expected_fields=len(header), actual_fields=len(raw))
        records.append({header[i]: raw[i] for i in range(len(header))})
    return header, records


def _cell(record: Mapping[str, str], column: str, *, path: str, line: int) -> str:
    require(column in record, "column missing from row, located by name",
            column=column, path=path, line=line)
    return record[column]


def validate_stage_output(artifact_path: str, spec: StageOutputSpec) -> ValidationReport:
    """Run the §4 global validation contract against the bytes on disk. FATAL on any violation.

    Performs, and reports by name: ``[shape] [keys] [coverage] [solver] [stoch] [numeric]
    [slots] [hashes]``. Every violation found is collected and raised together, so one run
    surfaces the whole failure set rather than only the first item.
    """
    require(os.path.isfile(artifact_path), "artifact to validate does not exist",
            path=artifact_path, stage=spec.stage)
    header, records = _read_csv_rows(artifact_path)
    violations: List[str] = []
    checks: Dict[str, str] = {}
    notes: List[str] = []
    nullable = set(spec.nullable_columns)

    # ---------------- [shape] ----------------
    expected_rows = spec.grid.expected_rows(rows_per_cell=spec.rows_per_cell)
    actual_rows = len(records)
    shape_line = "expected={} actual={}".format(expected_rows, actual_rows)
    print("[shape]      {} ({})".format(shape_line, spec.artifact_name))
    checks["shape"] = shape_line
    if expected_rows != actual_rows:
        violations.append("[shape] {}".format(shape_line))

    # ---------------- required columns (located BY NAME) ----------------
    needed = list(spec.required_columns) + list(spec.primary_key) + list(spec.numeric_columns)
    if spec.solver_policy is not None:
        needed += [spec.status_column, COL.STATUS_NAME, COL.PROVEN_OPTIMAL, COL.PRIMAL,
                   COL.DUAL, COL.DEV_CONST_C, COL.GAP_FULL, COL.STOP_REASON, COL.TIME_LIMIT]
    if spec.run_level:
        needed += [COL.ALGORITHM, COL.REPEAT_INDEX, COL.RNG_SEED, COL.ITERS_BUDGET,
                   COL.ITERS_EXECUTED, COL.EVALUATIONS, COL.STOP_REASON, COL.GAP_SIGNED,
                   COL.ASSIGN_HASH]
    if spec.slot_column:
        needed.append(spec.slot_column)
    missing = sorted({c for c in needed if c not in header})
    checks["columns"] = "missing={}".format(missing)
    if missing:
        violations.append("[columns] required column(s) absent: {}".format(missing))
        # without the key columns nothing below is meaningful
        raise ValidationFailure(
            "stage {} artifact {} failed validation".format(spec.stage, spec.artifact_name),
            violations, path=artifact_path)

    # ---------------- [keys] ----------------
    seen: Dict[Tuple[Any, ...], int] = {}
    dup: List[str] = []
    key_tuples: List[Tuple[Any, ...]] = []
    for lineno, rec in enumerate(records, 2):
        tup = tuple(_norm(_cell(rec, c, path=artifact_path, line=lineno))
                    for c in spec.primary_key)
        key_tuples.append(tup)
        if tup in seen:
            dup.append("{} first_seen_line={} line={}".format(tup, seen[tup], lineno))
        else:
            seen[tup] = lineno
    checks["keys"] = "primary_key={} duplicates={}".format(list(spec.primary_key), len(dup))
    print("[keys]       primary_key={} duplicates={}".format(list(spec.primary_key), len(dup)))
    if dup:
        violations.append("[keys] duplicate primary key(s): {}".format(dup[:10]))

    # ---------------- [coverage] ----------------
    grid_axes = spec.grid.axis_names
    cov_missing: List[Tuple[Any, ...]] = []
    cov_extra: List[Tuple[Any, ...]] = []
    if all(a in header for a in grid_axes):
        observed = set()
        for lineno, rec in enumerate(records, 2):
            observed.add(tuple(_norm(_cell(rec, a, path=artifact_path, line=lineno))
                               for a in grid_axes))
        expected = spec.grid.expected_tuples()
        cov_missing = sorted(expected - observed, key=repr)
        cov_extra = sorted(observed - expected, key=repr)
        shortfalls = set(spec.grid.declared_shortfalls)
        undeclared = sorted((observed & shortfalls), key=repr)
        if undeclared:
            violations.append(
                "[coverage] row present for a cell declared as a shortfall: {}".format(
                    undeclared[:10]))
    else:
        violations.append(
            "[coverage] design axes {} are not all columns of the artifact, so coverage cannot "
            "be checked against an independent grid".format(
                [a for a in grid_axes if a not in header]))
    checks["coverage"] = "grid={} provenance={} missing={} undesigned={}".format(
        spec.grid.name, spec.grid.provenance, len(cov_missing), len(cov_extra))
    print("[coverage]   grid={} provenance={} missing_cells={} undesigned_cells={}".format(
        spec.grid.name, spec.grid.provenance, len(cov_missing), len(cov_extra)))
    if cov_missing:
        violations.append("[coverage] designed cell(s) absent: {}".format(cov_missing[:10]))
    if cov_extra:
        violations.append("[coverage] row(s) outside the design grid: {}".format(cov_extra[:10]))

    # ---------------- [numeric] ----------------
    nan_hits: List[str] = []
    for lineno, rec in enumerate(records, 2):
        for column in spec.numeric_columns:
            raw = _cell(rec, column, path=artifact_path, line=lineno).strip()
            if raw == "" or raw.lower() in ("nan", "none", "null", "na"):
                if column in nullable and raw == "":
                    continue
                nan_hits.append("line={} {}={!r}".format(lineno, column, raw))
                continue
            try:
                value = float(raw)
            except ValueError:
                nan_hits.append("line={} {}={!r} (not numeric)".format(lineno, column, raw))
                continue
            if not math.isfinite(value):
                nan_hits.append("line={} {}={!r}".format(lineno, column, raw))
    checks["numeric"] = "nan_inf_empty={}".format(len(nan_hits))
    print("[numeric]    NaN/inf/empty cells in numeric columns: {}".format(len(nan_hits)))
    if nan_hits:
        violations.append("[numeric] {}".format(nan_hits[:10]))

    # ---------------- [solver] ----------------
    if spec.solver_policy is None:
        checks["solver"] = "no solver policy declared for this stage"
        print("[solver]     no solver policy declared for this stage")
    else:
        policy = spec.solver_policy
        bad_status: List[str] = []
        bad_name: List[str] = []
        bad_gap: List[str] = []
        bad_label: List[str] = []
        n_solver_rows = 0
        if spec.solver_row_filter is not None:
            filter_col, filter_vals = spec.solver_row_filter
            require(filter_col in header,
                    "solver_row_filter column absent from the artifact",
                    column=filter_col, stage=spec.stage)
            allowed = {_norm(v) for v in filter_vals}
        else:
            filter_col, allowed = None, None
        for lineno, rec in enumerate(records, 2):
            if filter_col is not None:
                if _norm(_cell(rec, filter_col, path=artifact_path, line=lineno)) not in allowed:
                    continue
            n_solver_rows += 1
            raw = _cell(rec, spec.status_column, path=artifact_path, line=lineno).strip()
            try:
                code = int(float(raw))
            except ValueError:
                bad_status.append("line={} status={!r} (not an integer)".format(lineno, raw))
                continue
            if code not in policy.admissible_codes:
                bad_status.append("line={} status={} not in {}".format(
                    lineno, code, sorted(policy.admissible_codes)))
                continue
            declared_name = policy.status_name(code)
            got_name = _cell(rec, COL.STATUS_NAME, path=artifact_path, line=lineno).strip()
            if got_name != declared_name:
                bad_name.append("line={} status_name={!r} expected {!r}".format(
                    lineno, got_name, declared_name))
            proven_raw = _cell(rec, COL.PROVEN_OPTIMAL, path=artifact_path,
                               line=lineno).strip().lower()
            proven = proven_raw in ("true", "1", "yes")
            if proven != policy.is_proven_optimal(code):
                bad_label.append("line={} proven_optimal={!r} but status {} is {}".format(
                    lineno, proven_raw, code,
                    "declared optimal" if policy.is_proven_optimal(code) else "NOT optimal"))
            stop = _cell(rec, COL.STOP_REASON, path=artifact_path, line=lineno).strip()
            if stop not in STOP_REASONS:
                bad_label.append("line={} stop_reason={!r} outside enumeration".format(
                    lineno, stop))
            elif proven and stop != "proven_optimal":
                bad_label.append("line={} proven_optimal=True with stop_reason={!r}".format(
                    lineno, stop))
            elif (not proven) and stop == "proven_optimal":
                bad_label.append("line={} stop_reason='proven_optimal' with "
                                 "proven_optimal=False".format(lineno))
            # recompute the FULL-objective gap from persisted P, B, C
            p_raw = _cell(rec, COL.PRIMAL, path=artifact_path, line=lineno).strip()
            b_raw = _cell(rec, COL.DUAL, path=artifact_path, line=lineno).strip()
            c_raw = _cell(rec, COL.DEV_CONST_C, path=artifact_path, line=lineno).strip()
            g_raw = _cell(rec, COL.GAP_FULL, path=artifact_path, line=lineno).strip()
            if "" in (p_raw, b_raw, c_raw):
                bad_gap.append("line={} P/B/C not all persisted (P={!r} B={!r} C={!r})".format(
                    lineno, p_raw, b_raw, c_raw))
                continue
            try:
                recomputed, undef = full_objective_gap(
                    float(p_raw), float(b_raw), float(c_raw),
                    where="{}:{}".format(spec.artifact_name, lineno))
            except PipelineFailure as exc:
                bad_gap.append("line={} gap recomputation failed: {}".format(lineno, exc))
                continue
            if undef is not None:
                if g_raw != "":
                    bad_gap.append("line={} gap_full persisted as {!r} but is undefined "
                                   "({})".format(lineno, g_raw, undef))
                continue
            if g_raw == "":
                bad_gap.append("line={} gap_full_objective empty but computable "
                               "({:.9g})".format(lineno, recomputed))
                continue
            if abs(float(g_raw) - recomputed) > 1e-9:
                bad_gap.append("line={} gap_full_objective={!r} != recomputed {:.12g} "
                               "from P={} B={} C={}".format(lineno, g_raw, recomputed,
                                                            p_raw, b_raw, c_raw))
        require(n_solver_rows > 0,
                "solver policy declared but no row matched solver_row_filter, so the solver gate "
                "would silently check nothing", stage=spec.stage,
                solver_row_filter=spec.solver_row_filter)
        checks["solver"] = ("stage={} rows_checked={} admissible={} bad_status={} bad_name={} "
                            "bad_gap={} bad_label={}".format(policy.stage, n_solver_rows,
                                                             sorted(policy.admissible_codes),
                                                             len(bad_status), len(bad_name),
                                                             len(bad_gap), len(bad_label)))
        print("[solver]     rows_checked={} admissible={} optimal={} bad_status={} bad_name={} "
              "bad_gap={} bad_label={}".format(n_solver_rows, sorted(policy.admissible_codes),
                                               sorted(policy.optimal_codes), len(bad_status),
                                               len(bad_name), len(bad_gap), len(bad_label)))
        for label, hits in (("status", bad_status), ("status_name", bad_name),
                            ("gap", bad_gap), ("optimality label", bad_label)):
            if hits:
                violations.append("[solver] {}: {}".format(label, hits[:10]))

    # ---------------- [stoch] ----------------
    if not spec.run_level:
        checks["stoch"] = "not a run-level artifact"
        print("[stoch]      not a run-level artifact")
    else:
        iters_bad: List[str] = []
        seed_by_run: Dict[Tuple[str, Any], List[Tuple[int, str]]] = {}
        gap_bad: List[str] = []
        per_algo_counts: Dict[Tuple[Any, ...], int] = {}
        differing = 0
        for lineno, rec in enumerate(records, 2):
            algo = _cell(rec, COL.ALGORITHM, path=artifact_path, line=lineno).strip()
            budget = _cell(rec, COL.ITERS_BUDGET, path=artifact_path, line=lineno).strip()
            executed = _cell(rec, COL.ITERS_EXECUTED, path=artifact_path, line=lineno).strip()
            evals = _cell(rec, COL.EVALUATIONS, path=artifact_path, line=lineno).strip()
            stop = _cell(rec, COL.STOP_REASON, path=artifact_path, line=lineno).strip()
            seed = _cell(rec, COL.RNG_SEED, path=artifact_path, line=lineno).strip()
            ahash = _cell(rec, COL.ASSIGN_HASH, path=artifact_path, line=lineno).strip()
            if stop == "" or stop not in STOP_REASONS:
                iters_bad.append("line={} stop_reason={!r} null or outside enumeration".format(
                    lineno, stop))
            if ahash == "":
                iters_bad.append("line={} assignment_hash empty".format(lineno))
            try:
                b_i, e_i, v_i = int(float(budget)), int(float(executed)), int(float(evals))
            except ValueError:
                iters_bad.append("line={} non-integer iterations/evaluations "
                                 "(budget={!r} executed={!r} evals={!r})".format(
                                     lineno, budget, executed, evals))
            else:
                if e_i > b_i:
                    iters_bad.append("line={} iterations_executed={} > budget={}".format(
                        lineno, e_i, b_i))
                if e_i != b_i:
                    differing += 1
                if v_i < 0:
                    iters_bad.append("line={} evaluations={} < 0".format(lineno, v_i))
            if algo in spec.stochastic_algorithms:
                if seed == "":
                    iters_bad.append("line={} stochastic algorithm {!r} has no persisted "
                                     "rng_seed".format(lineno, algo))
                key = (algo, tuple(_norm(_cell(rec, c, path=artifact_path, line=lineno))
                                   for c in spec.primary_key
                                   if c not in (COL.ALGORITHM, COL.REPEAT_INDEX)))
                seed_by_run.setdefault(key, []).append((lineno, seed))
            gap_raw = _cell(rec, COL.GAP_SIGNED, path=artifact_path, line=lineno).strip()
            if gap_raw == "":
                if COL.GAP_SIGNED not in nullable:
                    gap_bad.append("line={} gap_signed_pct empty".format(lineno))
            else:
                gap_value = float(gap_raw)
                ref_col = COL.REFERENCE_PROVEN_OPTIMAL
                ref_optimal = False
                if ref_col in header:
                    ref_optimal = _cell(rec, ref_col, path=artifact_path,
                                        line=lineno).strip().lower() in ("true", "1", "yes")
                if ref_optimal and gap_value < 0.0:
                    gap_bad.append(
                        "line={} NEGATIVE gap {:.6g}% against a PROVEN-OPTIMAL reference "
                        "(algorithm={})".format(lineno, gap_value, algo))
            cfg = tuple(_norm(_cell(rec, c, path=artifact_path, line=lineno))
                        for c in spec.primary_key if c != COL.REPEAT_INDEX)
            per_algo_counts[cfg] = per_algo_counts.get(cfg, 0) + 1
        if spec.repeats_expected is not None:
            wrong = {k: v for k, v in per_algo_counts.items() if v != spec.repeats_expected}
            if wrong:
                violations.append(
                    "[stoch] repeat count != {} for configuration(s): {}".format(
                        spec.repeats_expected, sorted(wrong.items(), key=repr)[:10]))
        # two repeats of the SAME algorithm on the SAME configuration must not share a seed:
        # equal seeds mean the "repeats" are the same run recorded twice.
        repeated_seed = {}
        for key, pairs in seed_by_run.items():
            seen_seed: Dict[str, int] = {}
            for lineno, seed in pairs:
                if seed in seen_seed:
                    repeated_seed.setdefault(key, []).append(
                        "seed={} at lines {} and {}".format(seed, seen_seed[seed], lineno))
                else:
                    seen_seed[seed] = lineno
        if repeated_seed:
            violations.append("[stoch] two repeats of one algorithm on one configuration share a "
                              "seed, so they are not independent repeats: {}".format(
                                  sorted(repeated_seed.items(), key=repr)[:5]))
        # cross-algorithm seed collision inside one configuration
        by_cfg: Dict[Any, Dict[str, Set[str]]] = {}
        for (algo, cfg), pairs in seed_by_run.items():
            by_cfg.setdefault(cfg, {})[algo] = {s for _l, s in pairs}
        collisions = []
        for cfg, algos in by_cfg.items():
            if len(algos) < 2:
                continue
            for a1, a2 in itertools.combinations(sorted(algos), 2):
                overlap = algos[a1] & algos[a2]
                if overlap:
                    collisions.append((cfg, a1, a2, sorted(overlap)))
        if collisions:
            violations.append("[stoch] two algorithms share a seed within one configuration "
                              "(their streams are then coupled): {}".format(collisions[:5]))
        checks["stoch"] = ("algorithms={} rows_where_executed!=budget={} bad={} gap_bad={}"
                           .format(list(spec.stochastic_algorithms), differing,
                                   len(iters_bad), len(gap_bad)))
        print("[stoch]      algorithms={} rows_where_executed!=budget={} bad={} "
              "gap_bad={}".format(list(spec.stochastic_algorithms), differing,
                                  len(iters_bad), len(gap_bad)))
        if iters_bad:
            violations.append("[stoch] {}".format(iters_bad[:10]))
        if gap_bad:
            violations.append("[stoch] {}".format(gap_bad[:10]))

    # ---------------- [slots] ----------------
    if not spec.slot_column:
        checks["slots"] = "no slot axis declared for this artifact"
        print("[slots]      no slot axis declared for this artifact")
    else:
        require(spec.expected_slots_fn is not None,
                "slot_column declared without an INDEPENDENT expected_slots_fn; reading the "
                "slot set off the output would make the check a tautology",
                stage=spec.stage)
        require(spec.config_columns, "slot check needs config_columns", stage=spec.stage)
        observed_slots: Dict[Tuple[Any, ...], Set[Any]] = {}
        for lineno, rec in enumerate(records, 2):
            cfg = tuple(_norm(_cell(rec, c, path=artifact_path, line=lineno))
                        for c in spec.config_columns)
            slot = _norm(_cell(rec, spec.slot_column, path=artifact_path, line=lineno))
            observed_slots.setdefault(cfg, set()).add(slot)
        slot_bad: List[str] = []
        for cfg in sorted(observed_slots, key=repr):
            expected_slots = {_norm(s) for s in spec.expected_slots_fn(cfg)}
            missing_slots = sorted(expected_slots - observed_slots[cfg], key=repr)
            if missing_slots:
                slot_bad.append("config={} missing slots={}".format(cfg, missing_slots))
        checks["slots"] = "configs={} configs_with_missing_slots={}".format(
            len(observed_slots), len(slot_bad))
        print("[slots]      configs={} configs_with_missing_slots={}".format(
            len(observed_slots), len(slot_bad)))
        if slot_bad:
            violations.append("[slots] {}".format(slot_bad[:10]))

    # ---------------- REQ-15 confound band (only when the design declares one) ----------
    if spec.rho_avail_band is not None:
        lo, hi = spec.rho_avail_band
        require(COL.RHO_AVAIL in header,
                "confound band declared but the pre-existing rho_avail column is absent",
                stage=spec.stage)
        out_of_band = []
        for lineno, rec in enumerate(records, 2):
            raw = _cell(rec, COL.RHO_AVAIL, path=artifact_path, line=lineno).strip()
            if raw == "":
                out_of_band.append("line={} rho_avail empty".format(lineno))
                continue
            value = float(raw)
            if not (lo <= value <= hi):
                out_of_band.append("line={} rho_avail={:.6g} outside band [{}, {}]".format(
                    lineno, value, lo, hi))
        checks["confound"] = "mode=band band=[{}, {}] out_of_band={}".format(
            lo, hi, len(out_of_band))
        print("[confound]   mode=band band=[{}, {}] rows_out_of_band={}".format(
            lo, hi, len(out_of_band)))
        if out_of_band:
            violations.append("[confound] {}".format(out_of_band[:10]))
        if COL.CAND_DEG_MAX in header and COL.M_AVAIL in header:
            deg_bad = []
            for lineno, rec in enumerate(records, 2):
                dmax = _cell(rec, COL.CAND_DEG_MAX, path=artifact_path, line=lineno).strip()
                m = _cell(rec, COL.M_AVAIL, path=artifact_path, line=lineno).strip()
                if dmax == "" or m == "":
                    deg_bad.append("line={} cand_degree_max/M empty".format(lineno))
                elif float(dmax) > float(m):
                    deg_bad.append("line={} cand_degree_max={} > M={}".format(lineno, dmax, m))
            if deg_bad:
                violations.append("[confound] realized candidate degree exceeds the available "
                                  "workforce: {}".format(deg_bad[:10]))

    # ---------------- [hashes] + SNAP ----------------
    hashes = hash_manifest(list(spec.input_paths), [artifact_path])
    for key in sorted(hashes["inputs"]):
        print("[hashes]     input  {}  {}".format(hashes["inputs"][key], key))
    for key in sorted(hashes["outputs"]):
        print("[hashes]     output {}  {}".format(hashes["outputs"][key], key))
    checks["hashes"] = "inputs={} outputs={}".format(len(hashes["inputs"]),
                                                     len(hashes["outputs"]))
    if spec.snap_required():
        if not spec.snap_id:
            violations.append(
                "[hashes] stage {!r} requires the SNAP immutable version identifier and none "
                "exists. A SNAP identifier may NEVER be invented: it is produced by the SNAP "
                "gate before R0. This stage therefore cannot emit a canonical artifact yet."
                .format(spec.stage))
        print("[hashes]     snap_id={!r} (required for stage {})".format(spec.snap_id,
                                                                        spec.stage))
    else:
        notes.append("SNAP identifier not required for stage {!r}; none was recorded and none "
                     "was invented.".format(spec.stage))
        print("[hashes]     snap_id=None (not required for stage {}; none invented)".format(
            spec.stage))

    if violations:
        raise ValidationFailure(
            "stage {} artifact {} FAILED validation ({} violation group(s))".format(
                spec.stage, spec.artifact_name, len(violations)),
            violations, path=artifact_path)

    print("[negative]   reminder: this PASS is regression/traceability evidence only. It is not "
          "proof, not correctness, and not evidence of an absence of fabrication.")
    return ValidationReport(stage=spec.stage, artifact=spec.artifact_name,
                            expected_rows=expected_rows, actual_rows=actual_rows,
                            checks=checks, hashes=hashes, snap_id=spec.snap_id, notes=notes)


# --------------------------------------------------------------------------------------------
# 9. canonical-output protection: partial output may NEVER overwrite a canonical file
# --------------------------------------------------------------------------------------------


def _utc_stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@contextlib.contextmanager
def canonical_write(canonical_path: str, validate: Callable[[str], Any], *, stage: str,
                    diagnostics: Optional[Mapping[str, Any]] = None):
    """temp -> validate the LANDED temp bytes -> promote, with rollback on a CAUGHT failure.

    Yields the temp path. The caller writes there and nowhere else. On success the temp file is
    promoted with :func:`os.replace`. On any caught exception the temp file is removed, the
    pre-existing canonical bytes are re-read and asserted unchanged, a
    ``<canonical>.FAILED_<UTC>.json`` diagnostic is written **beside** the canonical file, and
    the exception is re-raised so the run exits non-zero.

    Honest limits — read these before quoting this function as a guarantee:

    * exactly **one** file is promoted per invocation. This is **not** pair-atomicity and **not**
      family-atomicity: two successive invocations can be interrupted between promotions and
      leave a half-updated family;
    * rollback runs only for a **caught** exception. ``SIGKILL``, an OOM kill or power loss runs
      no rollback; the temp file may survive and the canonical file is simply untouched;
    * :func:`os.replace` is atomic only within one filesystem.
    """
    require(canonical_path, "canonical_path is empty", stage=stage)
    directory = os.path.dirname(os.path.abspath(canonical_path)) or "."
    require(os.path.isdir(directory), "canonical output directory does not exist",
            directory=directory, stage=stage)
    pre_exists = os.path.isfile(canonical_path)
    pre_hash = sha256_file(canonical_path) if pre_exists else None
    temp_path = "{}.S2TMP.{}.{}".format(canonical_path, os.getpid(), _utc_stamp())
    require(not os.path.exists(temp_path), "temp path already exists", temp_path=temp_path)
    try:
        yield temp_path
        if not os.path.isfile(temp_path):
            raise CanonicalWriteRefused(
                "producer wrote no temp file, so there is nothing to validate or promote",
                temp_path=temp_path, stage=stage)
        if os.path.getsize(temp_path) == 0:
            raise CanonicalWriteRefused("producer wrote an empty temp file",
                                        temp_path=temp_path, stage=stage)
        validate(temp_path)                       # re-reads the landed bytes from disk
        temp_hash = sha256_file(temp_path)
        os.replace(temp_path, canonical_path)
        landed_hash = sha256_file(canonical_path)
        require(landed_hash == temp_hash,
                "promoted bytes differ from the validated temp bytes",
                temp_sha256=temp_hash, landed_sha256=landed_hash, stage=stage)
        print("[promote]    {} sha256={} (previous={})".format(
            os.path.basename(canonical_path), landed_hash, pre_hash))
    except BaseException as exc:
        # 1. no partial output survives at the canonical path
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError as unlink_exc:                     # pragma: no cover - reported, fatal
                print("[rollback]   WARNING could not remove temp file {}: {}".format(
                    temp_path, unlink_exc), file=sys.stderr)
        # 2. verify the canonical file is byte-unchanged (single file only)
        rollback_state = "canonical_absent_before_and_after"
        if pre_exists:
            post_hash = sha256_file(canonical_path) if os.path.isfile(canonical_path) else None
            rollback_state = ("canonical_unchanged" if post_hash == pre_hash
                              else "CANONICAL_CHANGED")
            if post_hash != pre_hash:
                print("[rollback]   FATAL canonical file changed during a failed write: "
                      "pre={} post={}".format(pre_hash, post_hash), file=sys.stderr)
        elif os.path.isfile(canonical_path):
            rollback_state = "CANONICAL_CREATED_BY_FAILED_RUN"
        # 3. diagnostics beside, never over, the canonical file
        failed_path = "{}.FAILED_{}.json".format(canonical_path, _utc_stamp())
        payload = {
            "stage": stage,
            "canonical_path": os.path.basename(canonical_path),
            "canonical_sha256_before": pre_hash,
            "canonical_existed_before": pre_exists,
            "rollback_state": rollback_state,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "violations": list(getattr(exc, "violations", []) or []),
            "diagnostics": dict(diagnostics or {}),
            "utc": _utc_stamp(),
            "note": ("Single-file rollback only. No pair-atomicity and no family-atomicity is "
                     "claimed."),
        }
        try:
            with open(failed_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            print("[rollback]   {} | diagnostics -> {}".format(
                rollback_state, os.path.basename(failed_path)), file=sys.stderr)
        except OSError as write_exc:                          # pragma: no cover
            print("[rollback]   WARNING could not write diagnostics {}: {}".format(
                failed_path, write_exc), file=sys.stderr)
        raise
