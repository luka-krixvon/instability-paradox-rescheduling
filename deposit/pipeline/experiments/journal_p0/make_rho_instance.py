"""
make_rho_instance.py — OFFICIAL P0 generator for the capacity-tightness instance
family (H_*/S_* + rho_manifest.csv), carrying the ACCEPTED Gate G.2 semantics.

WHAT THIS FILE IS
-----------------
This is the OFFICIAL generator.  Its public API, CLI and output filenames are the
ones the rest of `experiments/journal_p0/` already depends on
(`H_<tag>.csv`, `S_<tag>.csv`, consumed by `run_e4.py:51-54`,
`eps_constraint.py:155-161`, `rolling_horizon.py:154-159` through
`rho_manifest.csv`).  The GUARD SEMANTICS below were designed, exercised and
accepted under Gate G.2 in `audit_outputs/gate_g2_staging/` and are PORTED here;
they are not re-derived and not re-argued.

**A guard that does not trip is REGRESSION EVIDENCE, never proof.**  A green run
means "no guard tripped on this data at this design point", nothing stronger.
Traceability is not correctness.  Nothing this file writes is a formal result;
nothing here starts, closes or authorises any R-stage.

Dependencies are the repository's own modules, imported through a path derived
from `__file__` (`experiments/` on `sys.path`).  This file imports NOTHING from
`audit_outputs/` and contains no absolute runtime path.  The dependency copies
staged under `audit_outputs/gate_g2_staging/deps_unmodified/` are byte-identical
to the `experiments/` originals imported here, so the ported semantics run
against the same parser bytes they were accepted against.

=============================================================================
SEMANTICS — hard ALLOWED-SLOT constraint
=============================================================================
lin2025 Eq. (5):  x_{i,j,q} = 0 for all q not in theta_j — assignment OUTSIDE the
requested slot SET is FORBIDDEN.  This is an allowed-SET (membership) constraint,
**not an equality in general**.

In THIS shipped benchmark theta_j is a SINGLETON: the format carries exactly one
`requested slot` field per patient, so HERE — and only here — the constraint
degenerates to a fixed shift.  That degeneracy is a property of the SHIPPED DATA,
never a general equality.  Accordingly every check is written as SET MEMBERSHIP
(`slot in allowed_slots(patient)`) and the singleton property is RECORDED as a
measurement (`allowed_slot_set_sizes`), never asserted as a law.

Operationally: **slot is packing DATA, not a decision.**  The caregiver is the
only decision in the packing.

=============================================================================
WHAT THE PORTED GUARDS ENFORCE
=============================================================================
* CORRECTED H METADATA.  `make_tight_instance.emit_hhcp` copies the source
  header block verbatim, so it carried the SOURCE `#(nurse)` (59) into files
  containing only the selected subset.  `emit_hhcp` is NOT modified; it is
  WRAPPED (`emit_hhcp_with_correct_metadata`), and the `#(nurse)` field is
  located BY FIELD NAME — never by a hard-coded column index — and rewritten to
  the number of nurse rows actually emitted.  FATAL chain:
      declared_nurses == parsed_nurse_rows == len(selected_nurses) == final_N
                      == len(landed_parser.nurses)
* LANDED PARSER AS TRUTH.  After the temp pair is written, `HHCPFullParser` is
  constructed **EXACTLY ONCE** on the landed H and EVERY subsequent check runs
  against those landed bytes.  The source parser is used only for the pre-write
  input probe and as the "intended content" side of an explicit landed-H
  equivalence guard.
* EXACT TASK SET, NOT A COUNT.  A `collections.Counter` over the RAW `TaskIDs`
  field of the landed S must equal exactly
  `{task_id: 1 for task_id in landed_parser.tasks}` — no unknown task, no
  missing task, no duplicate.  `served == 256` is NOT a pass condition anywhere
  in this file: `displacement_milp.load_static` builds a `{task: nurse}` dict and
  therefore CANNOT see a duplicate, so no count and no round trip through it may
  substitute for task-SET equality.
* PERSISTED FIELD FIDELITY (added in this round).  Every persisted patient and
  nurse field of the landed H is compared to the parsed object, and the declared
  `#(tasks)` / `#(slot)` / `#(skills)` / `#(nurses)` counts must equal their
  actual list lengths — both the raw token count and the parsed collection size,
  so a duplicated token cannot hide inside a set.
* ownership, allowed-slot membership, availability, candidacy, candidate ORDER
  and candidate SCORE alignment, exact load recomputation, and a cap guard.

=============================================================================
NO EXPECTED VALUE IS A PASS CONDITION
=============================================================================
`cap`, `total_tasks`, the task / patient / nurse ID sets, the slot universe, the
allowed-slot sets, the nurse count and the load bound are ALL derived from the
parsed data at run time.  Two numeric allowances exist and both are declared:

* `PACK_TOL = 1e-6` is carried forward UNCHANGED from the pre-S2 generator and
  is used ONLY inside the packing feasibility predicate, so packing semantics
  are unchanged relative to it.  It is NOT a pass condition.
* the GUARDS compare against a float-REPRESENTATION allowance derived at run
  time from the cap itself, `cap_fp_slack(cap) = math.ulp(cap) * 8` (~2e-13 for
  the caps in this benchmark).  `1e-6` is NOT the final pass allowance anywhere.
  The manifest records `cap_fp_slack` beside the measured minimum per-task load
  increment so a reader can see the allowance is orders of magnitude below
  anything this data can add to a cell.

NEVER adjust rho, cap, seed, starting N, H_MAX or an allowance to obtain a PASS.
If a measurement fails, this module STOPS and reports it.

=============================================================================
PUBLISH SEMANTICS — stated with its real limit
=============================================================================
The temp pair is written under `<outdir>/_tmp/`.  The canonical `H_<tag>.csv` /
`S_<tag>.csv` paths are produced ONLY after EVERY guard passes, by two
`os.replace` calls.

**This is NOT claimed to be pair-atomic, and NOT claimed to be family-atomic.**
What IS claimed, and all that is claimed:
  * on success, both canonical files exist;
  * on a CAUGHT failure — a guard trip, a raise from either `os.replace`, or any
    other exception inside the guarded region — BOTH canonical paths are removed,
    the temps are moved to `<outdir>/_quarantine/` beside a machine-readable
    reason file, and the process exits non-zero;
  * a generic exception goes through the SAME cleanup path as a guard trip.
A power loss or SIGKILL between the two `os.replace` calls is NOT caught and
could leave a single canonical file behind.  That residue is detectable (its
sibling is absent) but it is a real, un-fixed limit of a two-call publish.

OPERATIONAL NOTE.  Stale canonical output for the tag being generated is cleared
INSIDE the guarded region, so a failing run cannot leave an older file behind to
be mistaken for this run's PASS.  Consequently: never point `--outdir` at a
directory whose existing family you are not prepared to lose — generate into a
staging directory and promote after the whole family validates.

DOCSTRING ERRATUM (carried forward, still enforced): whole-block placement of a
patient's tasks into a single (caregiver, slot) cell is only the FIRST ATTEMPT.
The per-task fallback can spread one patient's k tasks over up to k caregivers.
What IS invariant is that every cell a patient occupies has
`slot in allowed_slots(p)` — a patient's cells live in ONE slot column and only
the caregiver row varies.

    cd experiments && python journal_p0/make_rho_instance.py --nurses 16 --seed 1
"""
from __future__ import annotations
import argparse, json, math, os, random, shutil, sys, time
from collections import Counter, defaultdict

# --- pandas is a HARD requirement. Do not substitute the csv module. ---------
try:
    import pandas as pd
except Exception as _e:                                          # pragma: no cover
    sys.stderr.write(
        "FATAL: pandas is unavailable in this environment: %r\n"
        "This module REQUIRES pandas because the disk-truth verification step is\n"
        "specified as pd.read_csv(landed_S). Refusing to substitute the csv module\n"
        "and refusing to fake the requirement. Reporting instead of proceeding.\n" % (_e,))
    raise SystemExit(5)

# --- every path is derived from __file__; no absolute runtime path appears ----
_HERE = os.path.dirname(os.path.abspath(__file__))          # experiments/journal_p0
_EXPERIMENTS = os.path.dirname(_HERE)                       # experiments
if _EXPERIMENTS not in sys.path:
    sys.path.insert(0, _EXPERIMENTS)
from SubgroupOptimizer import HHCPFullParser, H_STD, H_MAX               # noqa: E402,F401
from displacement_milp import TRAVEL_PER_TASK, SLOTS_PER_DAY, load_static  # noqa: E402
from make_tight_instance import greedy_set_cover, emit_hhcp              # noqa: E402

SRC_NAME = "HHCP-patient-65-task-256-nurse-59.csv"
SRC_DEFAULT = os.path.join(_EXPERIMENTS, "data_problems_solutions", SRC_NAME)

SLOT_RULE = "hard_allowed_slot_set_singleton_here"   # lin2025 Eq. (5)
PACK_TOL = 1e-6          # packing predicate ONLY; carried forward unchanged
SEMANTICS_PORTED_FROM = "gate_g2_accepted_semantics"

EXIT_OK = 0
EXIT_GUARD_FAIL = 3          # one or more guards tripped -> nothing published
EXIT_INTERNAL_ERROR = 4      # unexpected exception (cleanup still performed)
EXIT_NO_PANDAS = 5           # see the import guard above

# Guards carried forward from the pre-S2 generator. All fatal.
GUARDS_CARRIED_FROM_PRE_S2 = (
    "cand_slot_violation",
    "slot_filter_removal",
    "requested_slot_violation",
    "availability_candidate_violation",
    "round_trip_slot_violation",
    "unplaced_gt_zero",
    "served_ne_total_tasks",
    "max_cell_gt_cap",
    "rt_max_gt_cap",
)
# Guards ported from the accepted Gate G.2 design. All fatal.
GUARDS_PORTED_FROM_G2 = (
    "source_metadata_inconsistent",
    "landed_metadata_mismatch",
    "landed_h_not_equivalent_to_intended",
    "landed_s_schema_invalid",
    "served_task_multiset_mismatch",
    "row_patient_task_mismatch",
    "landed_max_load_gt_cap",
    "publish_failed",
    "unexpected_exception",
)
# Added in this round. Explicitly ADDITIONS, never substitutions.
GUARDS_ADDED_IN_S2 = (
    "source_persisted_field_mismatch",
    "landed_persisted_field_mismatch",
)
# Extra invariants. Explicitly ADDITIONS, never substitutions for the above.
GUARDS_LABELLED_ADDITIONS = (
    "rt_served_ne_served",
    "landed_max_ne_generator_max",
    "landed_candidate_order_mismatch",
    "landed_score_mismatch",
    "landed_flatload_mismatch",
)
GUARD_NAMES = (GUARDS_CARRIED_FROM_PRE_S2 + GUARDS_PORTED_FROM_G2
               + GUARDS_ADDED_IN_S2 + GUARDS_LABELLED_ADDITIONS)


class GeneratorGuardFailure(Exception):
    """Raised when any fatal guard trips. Carries the full guard report."""

    def __init__(self, failures, rec):
        self.failures = list(failures)
        self.rec = rec
        super().__init__("; ".join(f"{k}={v}" for k, v in self.failures))


class StructuralSurprise(Exception):
    """The file does not have the structure this code was written for. Raised
    instead of guessing. Routed through the SAME cleanup path as a guard trip and
    reported as `unexpected_exception`."""


# ───────────────────────── small derived helpers ─────────────────────────────
def cap_fp_slack(cap):
    """Float-REPRESENTATION allowance for a cap comparison, derived at run time
    from the cap itself: eight units in the last place of `cap`.  NOT a tuned
    tolerance and NOT a pass threshold; many orders of magnitude below the
    smallest load any task in this data can add to a cell (recorded as
    `min_task_increment`)."""
    return math.ulp(float(cap)) * 8.0


def resolve_src(src):
    """Resolve the source instance WITHOUT depending on the current working
    directory and WITHOUT embedding an absolute path: an existing path is used as
    given (so a mutated copy can be passed), otherwise a bare name is resolved
    against `experiments/data_problems_solutions/` relative to `__file__`."""
    if os.path.exists(src):
        return src
    cand = os.path.join(_EXPERIMENTS, "data_problems_solutions", os.path.basename(src))
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError(
        f"source instance {src!r} not found, and neither is {cand!r} -- refusing "
        f"to guess a source file")


def flat(parser, tids):
    return sum(parser.tasks[t].duration + TRAVEL_PER_TASK for t in tids)


def allowed_slots_of_patient(parser, pid):
    """The ALLOWED-SLOT SET of a patient, as a set, so every check is a
    membership test and nothing depends on the set being a singleton.  The
    shipped format carries a single `requested slot` field per patient, so this
    set has size 1 for THIS data — a property of the data, measured and recorded,
    never assumed."""
    if pid not in parser.patients:
        return frozenset()
    return frozenset({parser.patients[pid].req_slot})


def slot_of_row(day, shift):
    return (int(day) - 1) * SLOTS_PER_DAY + int(shift)


def _dups(seq):
    c = Counter(seq)
    return sorted(k for k, v in c.items() if v > 1)


def _gapfree(ids):
    """True iff the sorted ids form a contiguous run. The starting value is taken
    FROM the data, never assumed to be 0."""
    s = sorted(ids)
    if not s:
        return True
    return s == list(range(s[0], s[0] + len(s)))


def parse_tids_raw(field):
    """Parse the RAW TaskIDs field, PRESERVING DUPLICATES and reporting malformed
    tokens. Deliberately not `displacement_milp.parse_tids`-shaped: duplicates
    must survive so the task-multiset Counter can see them."""
    txt = str(field).replace(" ", "")
    good, bad = [], []
    for tok in txt.split(";"):
        tok = tok.strip()
        if tok == "":
            continue
        if tok.lstrip("+-").isdigit():
            good.append(int(tok))
        else:
            bad.append(tok)
    return good, bad


def _isint(tok):
    return str(tok).strip().lstrip("+-").isdigit()


def _name_index(names, key, where):
    """Index of a field located BY NAME, requiring it to be unambiguous. Never a
    hard-coded column index."""
    hits = [i for i, nm in enumerate(names) if nm == key]
    if len(hits) != 1:
        raise StructuralSurprise(
            f"{where}: field {key!r} appears {len(hits)} times in header {names} "
            f"-- ambiguous, refusing to guess")
    return hits[0]


# ── persisted-row field maps: (header name, parsed attribute, caster) ─────────
# Names are LOWER-CASED header labels of the shipped format. They are located by
# name; the maps carry NO column indices.
PATIENT_FIELD_MAP = (
    ("x", "x", float), ("y", "y", float),
    ("requested slot", "req_slot", int),
    ("p1_gender", "pref_gender", int), ("p1_weights", "weight_gender", int),
    ("p2_age", "pref_age", int), ("p2_weights", "weight_age", int),
    ("p3_language", "pref_language", int), ("p3_weights", "weight_language", int),
    ("p4_smoke", "pref_smoke", int), ("p4_weights", "weight_smoke", int),
)
NURSE_FIELD_MAP = (
    ("x", "x", float), ("y", "y", float), ("class", "nurse_class", int),
    ("p1_gender", "gender", int), ("p2_age", "age", int),
    ("p3_language", "language", int), ("p4_smoke", "smoke", int),
)
TASK_FIELD_MAP = (
    ("required skill", "required_skill", int), ("time", "duration", float),
)
# The nurse DATA row is parsed by SubgroupOptimizer._parse_nurse over the row with
# EMPTY FIELDS DROPPED (`clean`), because the shipped nurse header pads the slot
# list with blank columns. The same disposition is used here, and the cleaned
# header order is VERIFIED against this expectation rather than assumed.
NURSE_CLEAN_HEADER_EXPECTED = ("nurse", "x", "y", "class", "p1_gender", "p2_age",
                              "p3_language", "p4_smoke", "#(slot)", "slot list",
                              "#(skills)", "skill list")


# ───────────────────── raw (non-parser) reader for an H file ─────────────────
def scan_h_raw(path):
    """Read an HHCP instance file as TEXT and report its declared metadata, its
    actual row identity AND its raw rows, WITHOUT using HHCPFullParser.

    A deliberately independent second reader: HHCPFullParser ignores the global
    counts line entirely and, being dict-based, cannot represent a DUPLICATE id
    at all. Duplicates and declared-vs-actual disagreements are only visible
    here. Raises StructuralSurprise rather than guessing."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    def find_all(pred):
        return [i for i, l in enumerate(lines) if pred(l.lower())]

    pat_h = find_all(lambda s: s.startswith("patient,"))
    nur_h = find_all(lambda s: s.startswith("nurse,"))
    tsk_h = find_all(lambda s: s.startswith("task,"))
    cnd_h = find_all(lambda s: s.startswith("candidate solution"))
    for name, idxs in (("patient", pat_h), ("nurse", nur_h),
                       ("task", tsk_h), ("candidate solution", cnd_h)):
        if len(idxs) != 1:
            raise StructuralSurprise(
                f"{path}: expected exactly one '{name}' section header, found {len(idxs)} "
                f"at {idxs} -- refusing to guess which one is the real section")
    pi, ni, ti, ci = pat_h[0], nur_h[0], tsk_h[0], cnd_h[0]
    if not (pi < ni < ti < ci):
        raise StructuralSurprise(
            f"{path}: section headers out of the expected order "
            f"patient({pi}) < nurse({ni}) < task({ti}) < candidate({ci})")

    # -- global counts line, located BY FIELD NAME, never by column index ------
    want = {"#(days)", "#(patients)", "#(nurse)", "#(tasks)"}
    meta_hdr = None
    for i, l in enumerate(lines[:pi]):
        names = [f.strip().lower() for f in l.split(",")]
        if want.issubset(set(names)):
            meta_hdr = i
            break
    if meta_hdr is None:
        raise StructuralSurprise(
            f"{path}: no global-counts header line containing all of {sorted(want)} "
            f"was found above the patient section")
    names = [f.strip().lower() for f in lines[meta_hdr].split(",")]
    idx = {key: _name_index(names, key, f"{path} global counts") for key in sorted(want)}
    if meta_hdr + 1 >= len(lines):
        raise StructuralSurprise(f"{path}: global-counts header has no value line")
    meta_vals = lines[meta_hdr + 1].split(",")
    if len(meta_vals) != len(names):
        raise StructuralSurprise(
            f"{path}: global-counts value line has {len(meta_vals)} fields but the header "
            f"has {len(names)}")
    declared = {}
    for key, k in idx.items():
        tok = meta_vals[k].strip()
        if not _isint(tok):
            raise StructuralSurprise(
                f"{path}: global-counts field '{key}' = {tok!r} is not an integer")
        declared[key] = int(tok)

    # -- '#(patients) in each slot' block (present in the shipped format) ------
    slot_block = None
    for i, l in enumerate(lines[:pi]):
        if l.strip().lower().startswith("#(patients) in each slot"):
            if i + 2 >= len(lines):
                raise StructuralSurprise(f"{path}: per-slot patient block is truncated")
            cols = [f.strip() for f in lines[i + 1].split(",") if f.strip() != ""]
            vals = [f.strip() for f in lines[i + 2].split(",") if f.strip() != ""]
            if len(cols) != len(vals) or not all(_isint(v) for v in vals):
                raise StructuralSurprise(
                    f"{path}: per-slot patient block malformed: {len(cols)} columns vs "
                    f"{len(vals)} values")
            slot_block = dict(header_line=i, n_slot_columns=len(cols),
                              counts=[int(v) for v in vals])
            break
    if slot_block is None:
        raise StructuralSurprise(
            f"{path}: the '#(patients) in each slot' block was not found -- this code "
            f"was written for the shipped format and refuses to skip it silently")

    def data_rows(lo, hi):
        out = []
        for l in lines[lo:hi]:
            parts = l.split(",")
            if parts and parts[0].strip().isdigit():
                out.append(parts)
        return out

    pat_rows = data_rows(pi + 1, ni)
    nur_rows = data_rows(ni + 1, ti)
    tsk_rows = data_rows(ti + 1, ci)
    score_rows = data_rows(ci + 1, len(lines))

    patient_ids = [int(r[0]) for r in pat_rows]
    nurse_ids = [int(r[0]) for r in nur_rows]
    task_ids = [int(r[0]) for r in tsk_rows]

    return dict(
        path=path, n_lines=len(lines),
        meta_header_line=meta_hdr, meta_field_index=idx,
        declared_days=declared["#(days)"], declared_patients=declared["#(patients)"],
        declared_nurses=declared["#(nurse)"], declared_tasks=declared["#(tasks)"],
        slot_block=slot_block,
        patient_header=[f.strip().lower() for f in lines[pi].split(",")],
        nurse_header=[f.strip().lower() for f in lines[ni].split(",")],
        task_header=[f.strip().lower() for f in lines[ti].split(",")],
        patient_rows=pat_rows, nurse_rows=nur_rows, task_rows=tsk_rows,
        patient_lines=[l for l in lines[pi + 1:ni]
                       if l.split(",")[0].strip().isdigit()],
        nurse_lines=[l for l in lines[ni + 1:ti]
                     if l.split(",")[0].strip().isdigit()],
        patient_ids=patient_ids, nurse_ids=nurse_ids, task_ids=task_ids,
        n_patient_rows=len(pat_rows), n_nurse_rows=len(nur_rows),
        n_task_rows=len(tsk_rows), n_score_rows=len(score_rows),
    )


# ═══════ persisted patient / nurse / task FIELD fidelity (added in S2) ════════
def check_persisted_row_fields(raw, parser, label):
    """Compare EVERY persisted patient / nurse / task field of a file against the
    object `HHCPFullParser` produced from that SAME file, and require every
    declared list length to equal the actual list length.

    Fields are located BY HEADER NAME.  The nurse row is handled with the
    parser's own disposition — empty fields dropped, because the shipped nurse
    header pads the slot list with blank columns — and the cleaned header order
    is VERIFIED against `NURSE_CLEAN_HEADER_EXPECTED` rather than assumed, so a
    reordered format raises instead of silently mis-reading.

    Both the RAW token count and the PARSED collection size are compared to the
    declared count, so a duplicated slot / skill token cannot hide inside a set.

    Returns (reasons, report)."""
    bad = []

    # ── patients ──────────────────────────────────────────────────────────────
    ph = raw["patient_header"]
    p_idx = {nm: _name_index(ph, nm, f"{label} patient header")
             for nm, _a, _c in PATIENT_FIELD_MAP}
    i_ntasks = _name_index(ph, "#(tasks)", f"{label} patient header")
    i_tasklist = _name_index(ph, "task list", f"{label} patient header")
    if i_tasklist != i_ntasks + 1:
        raise StructuralSurprise(
            f"{label}: patient 'task list' is at column {i_tasklist} but '#(tasks)' is "
            f"at {i_ntasks}; the list is expected to follow its count")
    pat_field_bad, pat_count_bad = [], []
    for r in raw["patient_rows"]:
        pid = int(r[0])
        if pid not in parser.patients:
            pat_field_bad.append(dict(patient=pid, field="<row>", raw="present",
                                      parsed="absent_from_parser"))
            continue
        obj = parser.patients[pid]
        for nm, attr, cast in PATIENT_FIELD_MAP:
            k = p_idx[nm]
            if k >= len(r):
                pat_field_bad.append(dict(patient=pid, field=nm, raw="<missing column>",
                                          parsed=getattr(obj, attr)))
                continue
            try:
                want = cast(r[k].strip())
            except Exception:
                pat_field_bad.append(dict(patient=pid, field=nm, raw=r[k],
                                          parsed=getattr(obj, attr),
                                          note="raw token not castable"))
                continue
            got = getattr(obj, attr)
            if cast is float:
                ok = float(got) == float(want)
            else:
                ok = int(got) == int(want)
            if not ok:
                pat_field_bad.append(dict(patient=pid, field=nm, raw=want, parsed=got))
        # declared #(tasks) == raw token count == parsed list length
        if i_ntasks < len(r) and _isint(r[i_ntasks]):
            declared = int(r[i_ntasks].strip())
        else:
            declared = None
            pat_count_bad.append(dict(patient=pid, kind="declared_ntasks_not_an_integer",
                                      raw=(r[i_ntasks] if i_ntasks < len(r) else None)))
        toks = [t.strip() for t in r[i_tasklist:] if t.strip() != ""]
        if any(not _isint(t) for t in toks):
            pat_count_bad.append(dict(patient=pid, kind="non_integer_task_token",
                                      tokens=toks[:6]))
        raw_ids = [int(t) for t in toks if _isint(t)]
        parsed_ids = list(obj.task_list)
        if declared is not None and declared != len(raw_ids):
            pat_count_bad.append(dict(patient=pid, kind="declared_ntasks_ne_raw_tokens",
                                      declared=declared, raw_tokens=len(raw_ids)))
        if declared is not None and declared != len(parsed_ids):
            pat_count_bad.append(dict(patient=pid, kind="declared_ntasks_ne_parsed_len",
                                      declared=declared, parsed=len(parsed_ids)))
        if raw_ids != parsed_ids:
            pat_count_bad.append(dict(patient=pid, kind="task_list_differs",
                                      raw=raw_ids[:6], parsed=parsed_ids[:6]))
        d = _dups(raw_ids)
        if d:
            pat_count_bad.append(dict(patient=pid, kind="duplicate_task_in_task_list",
                                      duplicates=d[:6]))

    # ── nurses ────────────────────────────────────────────────────────────────
    nh_clean = [f for f in raw["nurse_header"] if f != ""]
    if tuple(nh_clean) != NURSE_CLEAN_HEADER_EXPECTED:
        raise StructuralSurprise(
            f"{label}: nurse header (empty columns dropped) is {nh_clean}, expected "
            f"{list(NURSE_CLEAN_HEADER_EXPECTED)} -- the parser reads the nurse row "
            f"positionally over the cleaned row, so a reordered header would be "
            f"mis-read; refusing to guess")
    n_idx = {nm: _name_index(nh_clean, nm, f"{label} nurse header")
             for nm, _a, _c in NURSE_FIELD_MAP}
    i_nslot = _name_index(nh_clean, "#(slot)", f"{label} nurse header")
    nur_field_bad, nur_count_bad = [], []
    for r in raw["nurse_rows"]:
        clean = [p.strip() for p in r if p.strip()]
        nid = int(clean[0])
        if nid not in parser.nurses:
            nur_field_bad.append(dict(nurse=nid, field="<row>", raw="present",
                                      parsed="absent_from_parser"))
            continue
        obj = parser.nurses[nid]
        for nm, attr, cast in NURSE_FIELD_MAP:
            k = n_idx[nm]
            if k >= len(clean):
                nur_field_bad.append(dict(nurse=nid, field=nm, raw="<missing column>",
                                          parsed=getattr(obj, attr)))
                continue
            try:
                want = cast(clean[k])
            except Exception:
                nur_field_bad.append(dict(nurse=nid, field=nm, raw=clean[k],
                                          parsed=getattr(obj, attr),
                                          note="raw token not castable"))
                continue
            got = getattr(obj, attr)
            ok = (float(got) == float(want)) if cast is float else (int(got) == int(want))
            if not ok:
                nur_field_bad.append(dict(nurse=nid, field=nm, raw=want, parsed=got))
        # declared #(slot) / #(skills) == raw token counts == parsed set sizes
        if i_nslot >= len(clean) or not _isint(clean[i_nslot]):
            nur_count_bad.append(dict(nurse=nid, kind="declared_nslots_not_an_integer",
                                      raw=(clean[i_nslot] if i_nslot < len(clean) else None)))
            continue
        n_slots_dec = int(clean[i_nslot])
        slot_toks = clean[i_nslot + 1:i_nslot + 1 + n_slots_dec]
        k_skill = i_nslot + 1 + n_slots_dec
        if len(slot_toks) != n_slots_dec or any(not _isint(t) for t in slot_toks):
            nur_count_bad.append(dict(nurse=nid, kind="slot_list_shorter_than_declared",
                                      declared=n_slots_dec, tokens=slot_toks[:8]))
            continue
        slot_ids = [int(t) for t in slot_toks]
        if n_slots_dec != len(slot_ids):
            nur_count_bad.append(dict(nurse=nid, kind="declared_nslots_ne_raw_tokens",
                                      declared=n_slots_dec, raw_tokens=len(slot_ids)))
        if n_slots_dec != len(obj.slots):
            nur_count_bad.append(dict(nurse=nid, kind="declared_nslots_ne_parsed_set",
                                      declared=n_slots_dec, parsed=len(obj.slots)))
        if set(slot_ids) != set(obj.slots):
            nur_count_bad.append(dict(nurse=nid, kind="slot_set_differs",
                                      raw=sorted(set(slot_ids))[:8],
                                      parsed=sorted(obj.slots)[:8]))
        d = _dups(slot_ids)
        if d:
            nur_count_bad.append(dict(nurse=nid, kind="duplicate_slot_token",
                                      duplicates=d[:8]))
        if k_skill >= len(clean) or not _isint(clean[k_skill]):
            nur_count_bad.append(dict(nurse=nid, kind="declared_nskills_not_an_integer",
                                      raw=(clean[k_skill] if k_skill < len(clean) else None)))
            continue
        n_sk_dec = int(clean[k_skill])
        sk_toks = clean[k_skill + 1:k_skill + 1 + n_sk_dec]
        if len(sk_toks) != n_sk_dec or any(not _isint(t) for t in sk_toks):
            nur_count_bad.append(dict(nurse=nid, kind="skill_list_shorter_than_declared",
                                      declared=n_sk_dec, tokens=sk_toks[:8]))
            continue
        sk_ids = [int(t) for t in sk_toks]
        if n_sk_dec != len(sk_ids):
            nur_count_bad.append(dict(nurse=nid, kind="declared_nskills_ne_raw_tokens",
                                      declared=n_sk_dec, raw_tokens=len(sk_ids)))
        if n_sk_dec != len(obj.skills):
            nur_count_bad.append(dict(nurse=nid, kind="declared_nskills_ne_parsed_set",
                                      declared=n_sk_dec, parsed=len(obj.skills)))
        if set(sk_ids) != set(obj.skills):
            nur_count_bad.append(dict(nurse=nid, kind="skill_set_differs",
                                      raw=sorted(set(sk_ids))[:8],
                                      parsed=sorted(obj.skills)[:8]))
        d = _dups(sk_ids)
        if d:
            nur_count_bad.append(dict(nurse=nid, kind="duplicate_skill_token",
                                      duplicates=d[:8]))

    # ── tasks ─────────────────────────────────────────────────────────────────
    th = raw["task_header"]
    t_idx = {nm: _name_index(th, nm, f"{label} task header")
             for nm, _a, _c in TASK_FIELD_MAP}
    i_ncand = _name_index(th, "#(nurses)", f"{label} task header")
    tsk_field_bad, tsk_count_bad = [], []
    for r in raw["task_rows"]:
        tid = int(r[0])
        if tid not in parser.tasks:
            tsk_field_bad.append(dict(task=tid, field="<row>", raw="present",
                                      parsed="absent_from_parser"))
            continue
        obj = parser.tasks[tid]
        for nm, attr, cast in TASK_FIELD_MAP:
            k = t_idx[nm]
            if k >= len(r):
                tsk_field_bad.append(dict(task=tid, field=nm, raw="<missing column>",
                                          parsed=getattr(obj, attr)))
                continue
            try:
                want = cast(r[k].strip())
            except Exception:
                tsk_field_bad.append(dict(task=tid, field=nm, raw=r[k],
                                          parsed=getattr(obj, attr),
                                          note="raw token not castable"))
                continue
            got = getattr(obj, attr)
            ok = (float(got) == float(want)) if cast is float else (int(got) == int(want))
            if not ok:
                tsk_field_bad.append(dict(task=tid, field=nm, raw=want, parsed=got))
        if i_ncand >= len(r) or not _isint(r[i_ncand]):
            tsk_count_bad.append(dict(task=tid, kind="declared_ncand_not_an_integer",
                                      raw=(r[i_ncand] if i_ncand < len(r) else None)))
            continue
        n_dec = int(r[i_ncand].strip())
        toks = [t.strip() for t in r[i_ncand + 1:] if t.strip() != ""]
        if any(not _isint(t) for t in toks):
            tsk_count_bad.append(dict(task=tid, kind="non_integer_candidate_token",
                                      tokens=toks[:6]))
        cand_ids = [int(t) for t in toks if _isint(t)]
        parsed_cands = list(obj.candidate_nurses)
        if n_dec != len(cand_ids):
            tsk_count_bad.append(dict(task=tid, kind="declared_ncand_ne_raw_tokens",
                                      declared=n_dec, raw_tokens=len(cand_ids)))
        if n_dec != len(parsed_cands):
            tsk_count_bad.append(dict(task=tid, kind="declared_ncand_ne_parsed_len",
                                      declared=n_dec, parsed=len(parsed_cands)))
        if cand_ids != parsed_cands:
            tsk_count_bad.append(dict(task=tid, kind="candidate_list_differs",
                                      raw=cand_ids[:6], parsed=parsed_cands[:6]))
        d = _dups(cand_ids)
        if d:
            tsk_count_bad.append(dict(task=tid, kind="duplicate_candidate_token",
                                      duplicates=d[:6]))

    for name, lst in (("patient field", pat_field_bad), ("patient count", pat_count_bad),
                      ("nurse field", nur_field_bad), ("nurse count", nur_count_bad),
                      ("task field", tsk_field_bad), ("task count", tsk_count_bad)):
        if lst:
            bad.append(f"{len(lst)} {name} mismatches; first={lst[0]}")

    report = dict(
        patients_field_checked=len(raw["patient_rows"]),
        nurses_field_checked=len(raw["nurse_rows"]),
        tasks_field_checked=len(raw["task_rows"]),
        patient_fields_per_row=len(PATIENT_FIELD_MAP),
        nurse_fields_per_row=len(NURSE_FIELD_MAP),
        task_fields_per_row=len(TASK_FIELD_MAP),
        patient_field_mismatches=len(pat_field_bad),
        patient_count_mismatches=len(pat_count_bad),
        nurse_field_mismatches=len(nur_field_bad),
        nurse_count_mismatches=len(nur_count_bad),
        task_field_mismatches=len(tsk_field_bad),
        task_count_mismatches=len(tsk_count_bad),
    )
    return bad, report


# ────────────────────────── input probe (pre-write) ──────────────────────────
def assert_candidate_slot_consistency(parser):
    """READ-ONLY probe on the INPUT: is every candidate caregiver of every task
    already available in that task's allowed slot set? Returns
    (n_pairs_checked, [violations]). A pass is regression evidence only."""
    viol, pairs = [], 0
    for pid, pat in parser.patients.items():
        allowed = allowed_slots_of_patient(parser, pid)
        for t in pat.task_list:
            if t not in parser.tasks:
                continue
            for n in parser.tasks[t].candidate_nurses:
                pairs += 1
                if n not in parser.nurses or not (allowed & parser.nurses[n].slots):
                    viol.append(dict(patient=pid, allowed_slots=sorted(allowed),
                                     task=t, nurse=n))
    return pairs, viol


def check_source_metadata(parser, raw):
    """Pre-write: the SOURCE file's own declared metadata must agree with what was
    parsed out of it. If the input is already self-contradictory we stop rather
    than propagate it into the official family."""
    bad = []
    if raw["declared_patients"] != len(parser.patients):
        bad.append(f"declared_patients={raw['declared_patients']} != parsed "
                   f"{len(parser.patients)}")
    if raw["declared_tasks"] != len(parser.tasks):
        bad.append(f"declared_tasks={raw['declared_tasks']} != parsed {len(parser.tasks)}")
    if raw["declared_nurses"] != len(parser.nurses):
        bad.append(f"declared_nurses={raw['declared_nurses']} != parsed "
                   f"{len(parser.nurses)}")
    if raw["n_patient_rows"] != len(parser.patients):
        bad.append(f"patient rows={raw['n_patient_rows']} != parsed {len(parser.patients)}")
    if raw["n_task_rows"] != len(parser.tasks):
        bad.append(f"task rows={raw['n_task_rows']} != parsed {len(parser.tasks)}")
    if raw["n_nurse_rows"] != len(parser.nurses):
        bad.append(f"nurse rows={raw['n_nurse_rows']} != parsed {len(parser.nurses)}")
    n_slots = raw["slot_block"]["n_slot_columns"]
    if raw["declared_days"] * SLOTS_PER_DAY != n_slots:
        bad.append(f"declared_days({raw['declared_days']}) * SLOTS_PER_DAY({SLOTS_PER_DAY}) "
                   f"!= per-slot columns({n_slots})")
    hist = Counter()
    for pid in parser.patients:
        for q in allowed_slots_of_patient(parser, pid):
            hist[q] += 1
    declared_hist = raw["slot_block"]["counts"]
    parsed_hist = [hist.get(q, 0) for q in range(n_slots)]
    if declared_hist != parsed_hist:
        bad.append(f"per-slot patient histogram declared={declared_hist} != "
                   f"parsed={parsed_hist}")
    off = sorted(q for q in hist if not (0 <= q < n_slots))
    if off:
        bad.append(f"allowed slots outside the declared universe: {off}")
    return bad, dict(slot_universe=n_slots, per_slot_hist=parsed_hist)


# ─────────── emit_hhcp WRAPPED, metadata corrected on the way out ────────────
def emit_hhcp_with_correct_metadata(src_lines, S, parser):
    """WRAP `make_tight_instance.emit_hhcp` (which is NOT modified) and correct
    the global-counts line it copies verbatim from the source.

    `emit_hhcp` preserves `src_lines[0:nurse_header]` unchanged, which carries
    `#(nurse)` from the SOURCE while emitting only the |S| selected nurse rows.
    Here the `#(nurse)` field is located BY NAME in the counts header (never by a
    hard-coded column index) and rewritten to the number of nurse rows this call
    actually emits, measured from the emitted lines rather than assumed from
    `len(S)`.

    `#(patients)`, `#(tasks)` and `#(days)` are NOT rewritten — the patient
    section is copied verbatim and every task is kept — but they ARE checked
    against the parsed content, and a disagreement is fatal upstream.

    Returns (lines, n_candidate_pairs, report)."""
    base_lines, npairs = emit_hhcp(src_lines, S, parser)

    nur_h = [i for i, l in enumerate(base_lines) if l.lower().startswith("nurse,")]
    tsk_h = [i for i, l in enumerate(base_lines) if l.lower().startswith("task,")]
    if len(nur_h) != 1 or len(tsk_h) != 1:
        raise StructuralSurprise(
            f"emitted lines have {len(nur_h)} nurse headers and {len(tsk_h)} task headers")
    emitted_nurse_rows = [l for l in base_lines[nur_h[0] + 1:tsk_h[0]]
                          if l.split(",")[0].strip().isdigit()]
    emitted_nurse_ids = [int(l.split(",")[0]) for l in emitted_nurse_rows]

    want = {"#(days)", "#(patients)", "#(nurse)", "#(tasks)"}
    meta_hdr = None
    for i, l in enumerate(base_lines[:nur_h[0]]):
        if want.issubset({f.strip().lower() for f in l.split(",")}):
            meta_hdr = i
            break
    if meta_hdr is None:
        raise StructuralSurprise(
            "emitted lines carry no global-counts header containing all of "
            f"{sorted(want)} -- refusing to publish an instance whose metadata "
            "cannot be located, and refusing to invent one")
    names = [f.strip().lower() for f in base_lines[meta_hdr].split(",")]
    k_nurse = _name_index(names, "#(nurse)", "emitted counts header")
    vals = base_lines[meta_hdr + 1].split(",")
    if len(vals) != len(names):
        raise StructuralSurprise(
            f"emitted counts value line has {len(vals)} fields, header has {len(names)}")
    old_val = vals[k_nurse].strip()
    vals[k_nurse] = str(len(emitted_nurse_rows))
    out = list(base_lines)
    out[meta_hdr + 1] = ",".join(vals)

    report = dict(
        meta_header_line=meta_hdr, nurse_field_index=k_nurse,
        declared_nurses_before=old_val, declared_nurses_after=vals[k_nurse],
        emitted_nurse_rows=len(emitted_nurse_rows),
        emitted_nurse_ids_sorted=sorted(emitted_nurse_ids),
        selected_set_size=len(S),
        lines_changed=1 if old_val != vals[k_nurse] else 0,
        candidate_pairs=npairs,
    )
    return out, npairs, report


# ───────────────────────────── packing ───────────────────────────────────────
def flat_pack(parser, S, cap, pack="tight"):
    """Place every task of every patient under the FLAT workload model, capping
    each (caregiver, slot) cell at `cap`.

    SLOT IS DATA, NOT A DECISION. Under the hard allowed-slot constraint every
    task of patient p may only go into a slot in `allowed_slots_of_patient(p)`
    (a singleton in this shipped data). The caregiver is the only decision.

    TRUE INVARIANT: whole-block placement into a single (caregiver, q) cell is
    only the FIRST ATTEMPT. If no single eligible caregiver has room for the
    patient's whole flat load, the per-task fallback distributes that patient's
    tasks over SEVERAL caregivers, so a patient with k tasks may occupy between 1
    and k distinct cells. What IS invariant: every cell a patient occupies has
    `slot in allowed_slots(p)`. Any task the fallback cannot place is returned in
    `unplaced` and is fatal upstream; it is never silently dropped.

    The feasibility predicate uses PACK_TOL, carried forward from the pre-S2
    generator UNCHANGED. Returns (cell_pt, cell_load, unplaced, diag)."""
    nurses = {n: parser.nurses[n] for n in S}
    cell_pt = defaultdict(lambda: defaultdict(list))   # (nid,slot) -> {pid: [tids]}
    cell_load = defaultdict(float)                     # (nid,slot) -> flat load
    unplaced = []
    diag = dict(slot_filter_removals=0, blocks_placed=0, blocks_fell_back=0)
    patients = sorted(parser.patients.values(),
                      key=lambda p: -flat(parser, [t for t in p.task_list
                                                   if t in parser.tasks]))

    def try_place(pid, tids):
        q = parser.patients[pid].req_slot            # slot is DATA
        cand_sets = [set(parser.tasks[t].candidate_nurses) for t in tids]
        full = set.intersection(*cand_sets) & S if cand_sets else set()
        opts = []
        add = flat(parser, tids)
        for nid in full:
            # defensive eligibility filter. Expected to remove NOTHING (see
            # assert_candidate_slot_consistency); counted, never hidden, FATAL.
            if q not in nurses[nid].slots:
                diag["slot_filter_removals"] += 1
                continue
            if cell_load[(nid, q)] + add <= cap + PACK_TOL:
                opts.append((cell_load[(nid, q)], nid, q))
        if opts:
            opts.sort(key=lambda x: (-x[0] if pack == "tight" else x[0]))
            _, nid, slot = opts[0]
            cell_pt[(nid, slot)][pid] += list(tids)
            cell_load[(nid, slot)] += add
            return True
        return False

    for p in patients:
        pid = p.patient_id
        tids = [t for t in p.task_list if t in parser.tasks]
        if not tids:
            continue
        if try_place(pid, tids):
            diag["blocks_placed"] += 1
            continue
        diag["blocks_fell_back"] += 1
        q = p.req_slot
        for t in sorted(tids, key=lambda t: -(parser.tasks[t].duration + TRAVEL_PER_TASK)):
            add = parser.tasks[t].duration + TRAVEL_PER_TASK
            elig = set(parser.tasks[t].candidate_nurses) & S
            elig_slot = set()
            for nn in elig:
                if q in nurses[nn].slots:
                    elig_slot.add(nn)
                else:
                    diag["slot_filter_removals"] += 1
            cand = sorted(elig_slot, key=lambda nn: -cell_load[(nn, q)])
            done = False
            for nid in cand:
                if cell_load[(nid, q)] + add <= cap + PACK_TOL:
                    cell_pt[(nid, q)][pid].append(t)
                    cell_load[(nid, q)] += add
                    done = True
                    break
            if not done:
                unplaced.append(t)
    return cell_pt, cell_load, unplaced, diag


def emit_schedule_flat(parser, cell_pt, cell_load):
    """Write only the columns `load_static` needs (Day, Shift, NurseID, TaskIDs)
    plus PatientID and FlatLoad for reporting. The slot -> (Day, Shift) decode is
    unchanged.

    Kept as a MODULE-LEVEL name, and called through the module global below, so a
    negative fault-injection harness can intercept the intermediate schedule
    WITHOUT any test seam being added to this file."""
    rows = []
    for (nid, slot), pmap in cell_pt.items():
        if not pmap:
            continue
        day, shift = slot // SLOTS_PER_DAY + 1, slot % SLOTS_PER_DAY
        for pid, tids in pmap.items():
            if not tids:
                continue
            rows.append(dict(NurseID=nid, Day=day, Shift=shift, PatientID=pid,
                             TaskIDs=";".join(str(t) for t in tids) + ";",
                             FlatLoad=round(cell_load[(nid, slot)], 1)))
    return pd.DataFrame(rows)


# ══════════════ landed-H equivalence to the intended content ═════════════════
def check_landed_h_equivalence(landed_parser, src_parser, S, landed_raw, src_raw):
    """Every check here compares the LANDED instance (`landed_parser`, built once
    from the file that was actually written) against the INTENDED content derived
    from the source parser restricted to the selected caregiver set S.

    Enforced (all fatal): identical patient ID set; identical allowed-slot set per
    patient; identical task ID set; identical patient_id, duration and required
    skill per task; nurse set == S; availability identical to that nurse in the
    source; per-task candidate set == original candidates INTERSECT S; and — added
    in this round — the patient block and the selected nurse rows are BYTE-FOR-BYTE
    the source lines, because `emit_hhcp` copies them verbatim.

    Returns (fatal_reasons, addition_reasons, report). `addition_reasons` are
    explicitly LABELLED ADDITIONS (candidate ORDER and candidate SCORE
    alignment), not substitutions for any mandated check."""
    bad, adds = [], []

    lp, sp = set(landed_parser.patients), set(src_parser.patients)
    if lp != sp:
        bad.append(f"patient ID set differs: landed-only={sorted(lp - sp)[:8]} "
                   f"source-only={sorted(sp - lp)[:8]}")
    slot_diff = []
    for pid in sorted(lp & sp):
        a = allowed_slots_of_patient(landed_parser, pid)
        b = allowed_slots_of_patient(src_parser, pid)
        if a != b:
            slot_diff.append(dict(patient=pid, landed=sorted(a), source=sorted(b)))
    if slot_diff:
        bad.append(f"allowed-slot set differs for {len(slot_diff)} patients; "
                   f"first={slot_diff[0]}")

    lt, st = set(landed_parser.tasks), set(src_parser.tasks)
    if lt != st:
        bad.append(f"task ID set differs: landed-only={sorted(lt - st)[:8]} "
                   f"source-only={sorted(st - lt)[:8]}")
    fld = []
    for t in sorted(lt & st):
        a, b = landed_parser.tasks[t], src_parser.tasks[t]
        if a.patient_id != b.patient_id:
            fld.append(dict(task=t, field="patient_id", landed=a.patient_id,
                            source=b.patient_id))
        if float(a.duration) != float(b.duration):
            fld.append(dict(task=t, field="duration", landed=a.duration,
                            source=b.duration))
        if a.required_skill != b.required_skill:
            fld.append(dict(task=t, field="required_skill", landed=a.required_skill,
                            source=b.required_skill))
    if fld:
        bad.append(f"{len(fld)} task field mismatches; first={fld[0]}")

    ln = set(landed_parser.nurses)
    if ln != set(S):
        bad.append(f"landed nurse set != selected set: landed-only={sorted(ln - set(S))[:8]} "
                   f"selected-only={sorted(set(S) - ln)[:8]}")
    av = []
    for n in sorted(ln & set(S)):
        if n not in src_parser.nurses:
            av.append(dict(nurse=n, kind="absent_from_source"))
            continue
        if landed_parser.nurses[n].slots != src_parser.nurses[n].slots:
            av.append(dict(nurse=n, kind="availability_differs",
                           landed=sorted(landed_parser.nurses[n].slots),
                           source=sorted(src_parser.nurses[n].slots)))
    if av:
        bad.append(f"{len(av)} nurse availability mismatches; first={av[0]}")

    # -- verbatim-copy blocks, compared as BYTES (added in this round) ---------
    if landed_raw is not None and src_raw is not None:
        if landed_raw["patient_lines"] != src_raw["patient_lines"]:
            n_diff = sum(1 for a, b in zip(landed_raw["patient_lines"],
                                           src_raw["patient_lines"]) if a != b)
            bad.append(f"the patient block is copied verbatim by emit_hhcp but the landed "
                       f"patient lines differ from the source: "
                       f"{len(landed_raw['patient_lines'])} vs "
                       f"{len(src_raw['patient_lines'])} lines, {n_diff} differing "
                       f"positions")
        want_nurse_lines = [l for l in src_raw["nurse_lines"]
                            if int(l.split(",")[0]) in S]
        if landed_raw["nurse_lines"] != want_nurse_lines:
            bad.append(f"the selected nurse rows are copied verbatim by emit_hhcp but the "
                       f"landed nurse lines differ from the source subset: "
                       f"{len(landed_raw['nurse_lines'])} vs {len(want_nurse_lines)} lines")

    cand_set_bad, cand_order_bad, score_bad = [], [], []
    for t in sorted(lt & st):
        want_list = [n for n in src_parser.tasks[t].candidate_nurses if n in S]
        got_list = list(landed_parser.tasks[t].candidate_nurses)
        if set(got_list) != set(want_list):
            cand_set_bad.append(dict(task=t, landed=sorted(set(got_list))[:8],
                                     intended=sorted(set(want_list))[:8]))
        elif got_list != want_list:
            cand_order_bad.append(dict(task=t, landed=got_list[:8], intended=want_list[:8]))
        for n in want_list:
            exp = src_parser.task_nurse_score.get((t, n), 0.5)
            got = landed_parser.task_nurse_score.get((t, n))
            if got is None or float(got) != float(exp):
                score_bad.append(dict(task=t, nurse=n, landed=got, intended=exp))
    if cand_set_bad:
        bad.append(f"{len(cand_set_bad)} tasks whose candidate SET != (source candidates "
                   f"INTERSECT selected set); first={cand_set_bad[0]}")
    if cand_order_bad:
        adds.append(("landed_candidate_order_mismatch",
                     f"{len(cand_order_bad)} tasks whose candidate ORDER differs from the "
                     f"source order; ORDER IS LOAD-BEARING because the candidate-score "
                     f"section is rebuilt by flat index; first={cand_order_bad[0]}"))
    if score_bad:
        adds.append(("landed_score_mismatch",
                     f"{len(score_bad)} (task, nurse) candidate scores differ from the "
                     f"source score (default 0.5 where the source has none); "
                     f"first={score_bad[0]}"))

    report = dict(
        landed_patients=len(lp), landed_tasks=len(lt), landed_nurses=len(ln),
        landed_scores=len(landed_parser.task_nurse_score),
        patients_checked=len(lp & sp), tasks_checked=len(lt & st),
        nurses_checked=len(ln & set(S)),
        allowed_slot_set_sizes=sorted({len(allowed_slots_of_patient(landed_parser, p))
                                       for p in lp}),
        patient_block_verbatim=bool(landed_raw is not None and src_raw is not None
                                    and landed_raw["patient_lines"]
                                    == src_raw["patient_lines"]),
    )
    return bad, adds, report


def check_landed_metadata(landed_parser, landed_raw, S, npairs):
    """Metadata guard on the LANDED H. Every quantity is measured; nothing assumed.

    FATAL chain:  declared_nurses == parsed_nurse_rows == len(selected_nurses)
                  == final_N == len(landed_parser.nurses)
    plus: declared patients == patient rows == parsed patients; declared tasks ==
    task rows == parsed tasks; the nurse ID set is EXACTLY the selected set; no
    duplicate nurse / patient / task ids; patient and task ids have no gaps
    (contiguity judged from the data's own minimum, never from an assumed 0);
    declared_days * SLOTS_PER_DAY == the number of per-slot columns; the declared
    per-slot patient histogram equals the parsed one; and the candidate-score row
    count equals the number of candidate pairs actually emitted.

    Returns (reasons, chain)."""
    bad = []
    final_N = len(S)
    chain = dict(declared_nurses=landed_raw["declared_nurses"],
                 parsed_nurse_rows=landed_raw["n_nurse_rows"],
                 len_selected_nurses=len(S),
                 final_N=final_N,
                 landed_parser_nurses=len(landed_parser.nurses))
    if len(set(chain.values())) != 1:
        bad.append(f"nurse-count chain disagrees: {chain}")
    if landed_raw["declared_patients"] != landed_raw["n_patient_rows"] or \
            landed_raw["declared_patients"] != len(landed_parser.patients):
        bad.append(f"patient count chain disagrees: declared="
                   f"{landed_raw['declared_patients']} rows="
                   f"{landed_raw['n_patient_rows']} parsed="
                   f"{len(landed_parser.patients)}")
    if landed_raw["declared_tasks"] != landed_raw["n_task_rows"] or \
            landed_raw["declared_tasks"] != len(landed_parser.tasks):
        bad.append(f"task count chain disagrees: declared="
                   f"{landed_raw['declared_tasks']} rows="
                   f"{landed_raw['n_task_rows']} parsed="
                   f"{len(landed_parser.tasks)}")
    if set(landed_raw["nurse_ids"]) != set(S):
        bad.append(f"landed nurse ID set != selected set: "
                   f"landed-only={sorted(set(landed_raw['nurse_ids']) - set(S))[:8]} "
                   f"selected-only={sorted(set(S) - set(landed_raw['nurse_ids']))[:8]}")
    for label, ids in (("nurse", landed_raw["nurse_ids"]),
                       ("patient", landed_raw["patient_ids"]),
                       ("task", landed_raw["task_ids"])):
        d = _dups(ids)
        if d:
            bad.append(f"duplicate {label} ids in the landed H: {d[:8]}")
    for label, ids in (("patient", landed_raw["patient_ids"]),
                       ("task", landed_raw["task_ids"])):
        if not _gapfree(ids):
            s = sorted(set(ids))
            holes = [v for v in range(s[0], s[-1] + 1) if v not in set(ids)]
            bad.append(f"{label} ids have gaps in the landed H: {len(holes)} missing, "
                       f"first={holes[:8]}")
    n_slots = landed_raw["slot_block"]["n_slot_columns"]
    if landed_raw["declared_days"] * SLOTS_PER_DAY != n_slots:
        bad.append(f"landed declared_days({landed_raw['declared_days']}) * "
                   f"SLOTS_PER_DAY({SLOTS_PER_DAY}) != per-slot columns({n_slots})")
    hist = Counter()
    for pid in landed_parser.patients:
        for q in allowed_slots_of_patient(landed_parser, pid):
            hist[q] += 1
    parsed_hist = [hist.get(q, 0) for q in range(n_slots)]
    if landed_raw["slot_block"]["counts"] != parsed_hist:
        bad.append(f"landed per-slot patient histogram declared="
                   f"{landed_raw['slot_block']['counts']} != parsed={parsed_hist}")
    if npairs is not None and landed_raw["n_score_rows"] != npairs:
        bad.append(f"landed candidate-score rows={landed_raw['n_score_rows']} != "
                   f"emitted candidate pairs={npairs}")
    return bad, chain


# ══════════ the landed S verified as a task SET, never by a count ════════════
def verify_landed_schedule(landed_parser, landed_sol, cap):
    """Verify the landed schedule against the LANDED instance only.

    A `collections.Counter` is built from the RAW `TaskIDs` field, so duplicates
    survive. `displacement_milp.load_static` builds a `{task: nurse}` dict per
    slot and therefore CANNOT see a duplicate; this function is the only place
    multiplicity is observable.

    Enforced (all fatal): no malformed token; no row without tasks; no unknown
    task; no missing task; no duplicate task; the Counter equals exactly
    {task_id: 1 for task_id in landed_parser.tasks}; each row's PatientID equals
    the patient_id of EVERY task on that row; NurseID exists in the landed H; the
    nurse is a landed candidate for EVERY task on the row; the slot is in the
    landed nurse's availability; the slot is in the patient's allowed-slot set (a
    SINGLETON in this shipped data -- checked as MEMBERSHIP, not equality); and
    max load recomputed from THIS verified exact task set is within cap.

    Returns (schema_bad, multiset_bad, row_bad, slot_viol, avail_viol,
    flatload_bad, load_rep)."""
    df = pd.read_csv(landed_sol)
    required = ["NurseID", "Day", "Shift", "PatientID", "TaskIDs"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return ([f"landed schedule is missing required column(s) {missing}; "
                 f"columns present = {list(df.columns)}"], [], [], [], [], [], {})

    counter = Counter()
    malformed, empty_rows, row_bad, slot_viol, avail_viol = [], [], [], [], []
    load = defaultdict(float)
    cell_rows = defaultdict(list)
    row_keys = []
    rows = 0
    for _, r in df.iterrows():
        rows += 1
        nid = int(r["NurseID"])
        pid = int(r["PatientID"])
        q = slot_of_row(r["Day"], r["Shift"])
        tids, bad_toks = parse_tids_raw(r["TaskIDs"])
        if bad_toks:
            malformed.append(dict(row=rows - 1, NurseID=nid, PatientID=pid,
                                  bad_tokens=bad_toks[:5]))
        if not tids:
            empty_rows.append(dict(row=rows - 1, NurseID=nid, PatientID=pid,
                                   TaskIDs=str(r["TaskIDs"])))
        counter.update(tids)
        row_keys.append((nid, q, pid))
        cell_rows[(nid, q)].append(rows - 1)

        # -- nurse must exist in the LANDED H, and work this slot ---------------
        if nid not in landed_parser.nurses:
            avail_viol.append(dict(kind="nurse_absent_from_landed_H", NurseID=nid, slot=q))
        elif q not in landed_parser.nurses[nid].slots:
            avail_viol.append(dict(kind="nurse_not_available", NurseID=nid, slot=q,
                                   landed_availability=sorted(
                                       landed_parser.nurses[nid].slots)))
        # -- slot must lie in the patient's ALLOWED-SLOT SET (membership) -------
        allowed = allowed_slots_of_patient(landed_parser, pid)
        if not allowed:
            slot_viol.append(dict(kind="patient_absent_from_landed_H", PatientID=pid,
                                  emitted_slot=q))
        elif q not in allowed:
            slot_viol.append(dict(kind="slot_not_in_allowed_set", NurseID=nid,
                                  PatientID=pid, Day=int(r["Day"]), Shift=int(r["Shift"]),
                                  emitted_slot=q, allowed_slots=sorted(allowed)))
        # -- every task on the row: ownership, candidacy, load ------------------
        for t in tids:
            if t not in landed_parser.tasks:
                row_bad.append(dict(kind="task_absent_from_landed_H", row=rows - 1, task=t))
                continue
            if landed_parser.tasks[t].patient_id != pid:
                row_bad.append(dict(kind="row_patient_ne_task_patient", row=rows - 1,
                                    task=t, row_patient=pid,
                                    task_patient=landed_parser.tasks[t].patient_id))
            if nid in landed_parser.nurses and \
                    nid not in landed_parser.tasks[t].candidate_nurses:
                avail_viol.append(dict(kind="nurse_not_candidate_in_landed_H",
                                       NurseID=nid, task=t))
            load[(nid, q)] += landed_parser.tasks[t].duration + TRAVEL_PER_TASK

    dup_keys = _dups(row_keys)
    if dup_keys:
        row_bad.append(dict(kind="duplicate_(NurseID,slot,PatientID)_row_key",
                            n=len(dup_keys), first=list(dup_keys[0])))

    # -- LABELLED ADDITION: persisted FlatLoad vs recomputed cell load ---------
    flatload_bad = []
    if "FlatLoad" in df.columns:
        for (nid, q), idxs in cell_rows.items():
            want = round(load[(nid, q)], 1)
            for i in idxs:
                got = float(df.iloc[i]["FlatLoad"])
                if got != want:
                    flatload_bad.append(dict(row=i, NurseID=nid, slot=q,
                                             persisted=got, recomputed=want))

    # -- the task MULTISET, not a count -------------------------------------
    expected = Counter({t: 1 for t in landed_parser.tasks})
    multiset_bad = []
    if malformed:
        multiset_bad.append(f"{len(malformed)} rows with malformed TaskIDs tokens; "
                            f"first={malformed[0]}")
    unknown = sorted(t for t in counter if t not in landed_parser.tasks)
    missing_t = sorted(t for t in landed_parser.tasks if t not in counter)
    dup = sorted((t, c) for t, c in counter.items() if c > 1)
    if unknown:
        multiset_bad.append(f"{len(unknown)} task ids scheduled that do not exist in the "
                            f"landed H; first={unknown[:8]}")
    if missing_t:
        multiset_bad.append(f"{len(missing_t)} landed tasks never scheduled; "
                            f"first={missing_t[:8]}")
    if dup:
        multiset_bad.append(f"{len(dup)} task ids scheduled more than once; "
                            f"first={dup[:8]}")
    if counter != expected:
        multiset_bad.append(
            "the raw-TaskIDs Counter is NOT equal to "
            "{task_id: 1 for task_id in landed_parser.tasks}: "
            f"|counter|={len(counter)} sum={sum(counter.values())} "
            f"|expected|={len(expected)} -- task-SET equality failed (a count match "
            "would NOT have been sufficient and is not used here)")
    if empty_rows:
        row_bad.append(dict(kind="row_with_no_tasks", n=len(empty_rows),
                            first=empty_rows[0]))

    slack = cap_fp_slack(cap)
    max_load = max(load.values(), default=0.0)
    over = sorted(((f"{k[0]}@{k[1]}", v) for k, v in load.items() if v > cap + slack),
                  key=lambda kv: -kv[1])
    durs = [t.duration for t in landed_parser.tasks.values()]
    load_rep = dict(
        landed_rows=rows,
        landed_cells=len(load),
        landed_max_load=round(max_load, 6),
        landed_max_load_raw=max_load,
        cap=cap, cap_fp_slack=slack,
        cells_over_cap=len(over), cells_over_cap_first=over[:3],
        min_task_increment=(min(durs) + TRAVEL_PER_TASK) if durs else None,
        distinct_tasks_scheduled=len(counter),
        total_task_placements=sum(counter.values()),
        max_task_multiplicity=max(counter.values(), default=0),
        distinct_slots_used=len({k[1] for k in load}),
        task_multiset_equals_landed_task_set=bool(counter == expected),
        duplicate_row_keys=len(dup_keys),
        flatload_column_present=bool("FlatLoad" in df.columns),
        flatload_mismatches=len(flatload_bad),
        s_columns=list(df.columns),
    )
    return [], multiset_bad, row_bad, slot_viol, avail_viol, flatload_bad, load_rep


def round_trip_via_recovery_pipeline(landed_parser, landed_sol):
    """Re-load the landed schedule through the REAL recovery pipeline
    (`displacement_milp.load_static`, itself a `pd.read_csv`) and recompute.

    The landed parser is constructed EXACTLY ONCE elsewhere and is the authority;
    the independent element here is `load_static`'s own re-read of S.

    `load_static` collapses duplicates into a dict, so its `rt_served` is a COUNT
    and can never substitute for the task-set equality above."""
    sa = load_static(landed_sol)
    rt_load = defaultdict(float)
    for q, m in sa.items():
        for t, n in m.items():
            if t in landed_parser.tasks:
                rt_load[(n, q)] += landed_parser.tasks[t].duration + TRAVEL_PER_TASK
    rt_max = max(rt_load.values(), default=0.0)
    rt_served = sum(len(m) for m in sa.values())
    rt_slot_viol = 0
    for q, m in sa.items():
        for t in m:
            if t not in landed_parser.tasks:
                rt_slot_viol += 1
                continue
            pid = landed_parser.tasks[t].patient_id
            if q not in allowed_slots_of_patient(landed_parser, pid):
                rt_slot_viol += 1
    return (rt_max, rt_served, rt_slot_viol,
            len(landed_parser.nurses), len(landed_parser.tasks))


# ──────────────────────────────── generate ───────────────────────────────────
def generate(rho, nurses=24, seed=1, pack="tight", src=SRC_DEFAULT,
             out_hhcp=None, out_sol=None, quiet=False, outdir=None):
    """Build ONE flat-feasible instance at target per-cell tightness `rho` under
    the hard allowed-slot constraint, verify it FROM DISK against the LANDED
    instance, and publish the canonical pair ONLY if every guard passes.

    API note: the pre-S2 signature is preserved, including `out_hhcp` / `out_sol`
    and the `H_<tag>.csv` / `S_<tag>.csv` defaults beside this file. `outdir` is
    an ADDITION that relocates the default pair (used to stage a whole family
    before promoting it). Unlike the pre-S2 version this function NEVER returns
    None: a failure RAISES, because "log the failure and continue" is forbidden.

    PUBLISH SEMANTICS, stated with its limit: two `os.replace` calls are NOT
    claimed to be pair-atomic, and nothing here is family-atomic. On success both
    canonical files exist; on a CAUGHT failure (guard trip, a raise from either
    `os.replace`, or any other exception inside the guarded region) BOTH canonical
    paths are removed, the temps are quarantined with a reason file, and the
    caller exits non-zero. An uncatchable kill between the two replaces is not
    covered.

    Raises GeneratorGuardFailure on a guard trip; re-raises anything else after
    the same cleanup. Returns the manifest dict on success."""
    src = resolve_src(src)
    base = os.path.abspath(outdir) if outdir else _HERE
    tag = f"rho{int(round(rho * 100)):03d}_n{nurses}_s{seed}"
    canon_hhcp = os.path.abspath(out_hhcp) if out_hhcp else os.path.join(base, f"H_{tag}.csv")
    canon_sol = os.path.abspath(out_sol) if out_sol else os.path.join(base, f"S_{tag}.csv")
    outbase = os.path.dirname(canon_hhcp)
    tmpdir = os.path.join(outbase, "_tmp")
    qdir = os.path.join(outbase, "_quarantine")
    for d in (outbase, os.path.dirname(canon_sol), tmpdir, qdir):
        os.makedirs(d, exist_ok=True)

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    tmp_hhcp = os.path.join(tmpdir, f"TMP_H_{tag}.{os.getpid()}.csv")
    tmp_sol = os.path.join(tmpdir, f"TMP_S_{tag}.{os.getpid()}.csv")

    removed_stale = []       # filled inside the guarded region, read by quarantine()
    failures = []            # list of (guard_name, detail)

    def _remove_paths(paths):
        """Remove each path DEFENSIVELY. The cleanup path must never itself raise
        and so hide the original failure: a path that cannot be removed is
        RECORDED and reported, never swallowed."""
        removed, errors = [], []
        for p in paths:
            if not os.path.lexists(p):
                continue
            try:
                if os.path.isdir(p) and not os.path.islink(p):
                    raise IsADirectoryError(f"{p} is a directory, not the expected file")
                os.remove(p)
                removed.append(os.path.basename(p))
            except Exception as e:
                errors.append(f"{os.path.basename(p)}: {type(e).__name__}: {e}")
        return removed, errors

    def remove_canonical_both():
        """Roll-back: NEITHER canonical path may hold this run's output after any
        caught failure. Returns (removed, errors); never raises."""
        return _remove_paths((canon_hhcp, canon_sol))

    def empty_temps():
        return _remove_paths((tmp_hhcp, tmp_sol))

    def quarantine(rec):
        """Move the temps aside, roll the canonical pair back, and ALWAYS write a
        machine-readable reason file. Every step is defensive: a failure inside
        cleanup is recorded rather than replacing the original diagnosis."""
        moved, move_errors = [], []
        for p, kind in ((tmp_hhcp, "H"), (tmp_sol, "S")):
            if os.path.lexists(p):
                dest = os.path.join(qdir, f"FAILED_{kind}_{tag}_{stamp}_{os.getpid()}.csv")
                try:
                    shutil.move(p, dest)
                    moved.append(dest)
                except Exception as e:
                    move_errors.append(f"{os.path.basename(p)}: {type(e).__name__}: {e}")
        rolled_back, rollback_errors = remove_canonical_both()
        residue, residue_errors = empty_temps()
        still_present = [os.path.basename(p) for p in (canon_hhcp, canon_sol)
                         if os.path.lexists(p)]
        reasons = os.path.join(qdir, f"FAILED_REASONS_{tag}_{stamp}_{os.getpid()}.json")
        payload = dict(
            generator=os.path.basename(__file__),
            semantics_ported_from=SEMANTICS_PORTED_FROM,
            tag=tag, rho=rho, seed=seed, nurses_start=nurses,
            stamp_utc=stamp, pid=os.getpid(),
            failures=[dict(guard=k, detail=v) for k, v in failures],
            quarantined=[os.path.basename(m) for m in moved],
            quarantine_move_errors=move_errors,
            canonical_rolled_back=rolled_back,
            canonical_rollback_errors=rollback_errors,
            canonical_absent=[os.path.basename(canon_hhcp), os.path.basename(canon_sol)],
            canonical_absent_verified=[not os.path.lexists(canon_hhcp),
                                       not os.path.lexists(canon_sol)],
            canonical_paths_still_occupied=still_present,
            temp_residue_removed=residue,
            temp_residue_errors=residue_errors,
            removed_stale_canonical=removed_stale,
            publish_atomicity="two os.replace calls; NOT claimed pair-atomic and NOT "
                              "family-atomic; on success both exist, a CAUGHT failure "
                              "is rolled back",
            manifest=rec)
        try:
            with open(reasons, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=1, default=str)
        except Exception as e:                                    # pragma: no cover
            sys.stderr.write(
                f"*** COULD NOT WRITE THE QUARANTINE REASON FILE {reasons}: "
                f"{type(e).__name__}: {e}\n*** payload was: {payload.get('failures')}\n")
        if rollback_errors or move_errors or residue_errors or still_present:
            sys.stderr.write(
                f"*** CLEANUP WAS INCOMPLETE (recorded in {reasons}): "
                f"rollback_errors={rollback_errors} move_errors={move_errors} "
                f"residue_errors={residue_errors} still_occupied={still_present}\n")
        return moved, reasons

    def trip(rec):
        rec["published"] = False
        moved, reasons = quarantine(rec)
        if moved:
            rec["quarantined"] = [os.path.basename(m) for m in moved]
        if not quiet:
            _report(rec, failures, reasons)
        raise GeneratorGuardFailure(failures, rec)

    try:
        # ── clear stale output for this tag, INSIDE the guarded region ────────
        # No stale canonical output for this tag may survive a run of this tag:
        # after a failing run the canonical paths must be provably ABSENT, never
        # left holding an older file that could be mistaken for this run's PASS.
        rs, rs_err = _remove_paths((canon_hhcp, canon_sol))
        removed_stale.extend(rs)
        _, tmp_err = _remove_paths((tmp_hhcp, tmp_sol))
        if rs_err or tmp_err:
            failures.append(("publish_failed",
                             f"an output path for this tag is occupied by something that "
                             f"cannot be cleared before the run: canonical={rs_err} "
                             f"temp={tmp_err} -- refusing to proceed rather than publish "
                             f"beside it"))
            trip(dict(tag=tag, stage="pre_run_path_blocked", published=False))

        # ── read the INPUT twice, two different ways ─────────────────────────
        parser = HHCPFullParser(src)                     # source parser
        src_raw = scan_h_raw(src)                        # independent raw reader
        with open(src, "r", encoding="utf-8", errors="ignore") as f:
            src_lines = f.read().splitlines()
        rng = random.Random(seed)

        cap = float(rho) * H_MAX
        slack = cap_fp_slack(cap)
        total_tasks = len(parser.tasks)                  # derived, never hard-coded

        # ── GUARD (pre-write): input candidate/allowed-slot consistency ──────
        pairs_checked, cand_viol = assert_candidate_slot_consistency(parser)
        if cand_viol:
            failures.append(("cand_slot_violation",
                             f"{len(cand_viol)} of {pairs_checked} candidate pairs; "
                             f"first={cand_viol[0]}"))
        # ── GUARD (pre-write): the source's declared metadata == parsed ───────
        src_meta_bad, src_meta_rep = check_source_metadata(parser, src_raw)
        if src_meta_bad:
            failures.append(("source_metadata_inconsistent", "; ".join(src_meta_bad)))
        # ── GUARD (pre-write, ADDED IN S2): source persisted-field fidelity ──
        src_fld_bad, src_fld_rep = check_persisted_row_fields(src_raw, parser, "source")
        if src_fld_bad:
            failures.append(("source_persisted_field_mismatch", "; ".join(src_fld_bad)))

        if failures:
            trip(dict(tag=tag, slot_rule=SLOT_RULE, rho_target=rho,
                      cap=round(cap, 6), seed=seed, pack=pack, stage="pre_pack",
                      cand_pairs_checked=pairs_checked,
                      cand_slot_violations=len(cand_viol),
                      total_tasks=total_tasks, published=False))

        # ── pack, with the auto-escalation loop ──────────────────────────────
        S = greedy_set_cover(parser.tasks, nurses, rng)
        n_start = len(S)
        escalations = 0
        while True:
            cell_pt, cell_load, unplaced, diag = flat_pack(parser, S, cap=cap, pack=pack)
            if not unplaced or len(S) >= len(parser.nurses):
                break
            need = defaultdict(int)
            for t in unplaced:
                for n in parser.tasks[t].candidate_nurses:
                    if n not in S:
                        need[n] += 1
            if not need:
                break
            S.add(max(need, key=need.get))
            escalations += 1

        cover = min(len(set(t.candidate_nurses) & S) for t in parser.tasks.values())
        served = sum(len(v) for pm in cell_pt.values() for v in pm.values())
        used = [L for L in cell_load.values() if L > 0.0]
        total_slot_cap = sum(len(parser.nurses[n].slots) for n in S) * H_MAX
        total_demand = flat(parser, list(parser.tasks.keys()))
        rho_global = total_demand / total_slot_cap
        rho_used_mean = (sum(used) / len(used) / H_MAX) if used else 0.0
        rho_used_max = (max(used) / H_MAX) if used else 0.0
        max_cell = max(used) if used else 0.0

        cells_per_patient = defaultdict(int)
        for (nid, slot), pmap in cell_pt.items():
            for pid, tids in pmap.items():
                if tids:
                    cells_per_patient[pid] += 1
        max_cells_one_patient = max(cells_per_patient.values(), default=0)
        patients_multi_cell = sum(1 for v in cells_per_patient.values() if v > 1)

        # ── GUARDS (pre-write, carried forward) ──────────────────────────────
        if diag["slot_filter_removals"] > 0:
            failures.append(("slot_filter_removal",
                             f"defensive filter removed {diag['slot_filter_removals']} "
                             f"(caregiver, slot) options -> the candidate-list pre-filter "
                             f"assumption broke"))
        if len(unplaced) > 0:
            failures.append(("unplaced_gt_zero",
                             f"{len(unplaced)} tasks unplaced; first={unplaced[:5]}"))
        if served != total_tasks:
            failures.append(("served_ne_total_tasks",
                             f"served={served} total_tasks={total_tasks} "
                             f"(both derived at run time)"))
        if max_cell > cap + slack:
            failures.append(("max_cell_gt_cap",
                             f"max_cell={max_cell!r} cap={cap!r} fp_slack={slack!r}"))

        rec = dict(
            generator=os.path.basename(__file__),
            semantics_ported_from=SEMANTICS_PORTED_FROM,
            tag=tag, slot_rule=SLOT_RULE, rho_target=rho, cap=round(cap, 6),
            cap_fp_slack=slack, pack_tol_packing_only=PACK_TOL,
            n_start_arg=nurses, N_after_cover=n_start, N=len(S),
            auto_escalations=escalations,
            auto_escalation_required=bool(escalations > 0),
            seed=seed, pack=pack, src=src,
            served=served, total_tasks=total_tasks, unplaced=len(unplaced),
            min_cover_per_task=cover,
            rho_global=round(rho_global, 4), rho_used_mean=round(rho_used_mean, 4),
            rho_used_max=round(rho_used_max, 4),
            max_cell_load=round(max_cell, 1), cap_compared_against=round(cap, 6),
            h_max=H_MAX, used_cells=len(used),
            cand_pairs_checked=pairs_checked, cand_slot_violations=len(cand_viol),
            slot_filter_removals=diag["slot_filter_removals"],
            blocks_placed_whole=diag["blocks_placed"],
            blocks_fell_back=diag["blocks_fell_back"],
            max_cells_one_patient=max_cells_one_patient,
            patients_occupying_multiple_cells=patients_multi_cell,
            src_slot_universe=src_meta_rep["slot_universe"],
            src_persisted_field_report=src_fld_rep,
            stage="post_pack", published=False,
        )
        if failures:
            trip(rec)

        # ── write to TEMPORARY paths ─────────────────────────────────────────
        # both emitters are reached through the module globals, so a negative
        # fault-injection harness can intercept them without a test seam here
        new_lines, npairs, meta_rep = globals()["emit_hhcp_with_correct_metadata"](
            src_lines, S, parser)
        with open(tmp_hhcp, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
        sol = globals()["emit_schedule_flat"](parser, cell_pt, cell_load)
        sol.to_csv(tmp_sol, index=False)
        emitted_rows = int(len(sol))
        del sol            # forbid any later check from using in-memory state
        del cell_pt, cell_load

        # ══ the landed parser, constructed EXACTLY ONCE ══════════════════════
        landed_parser = HHCPFullParser(tmp_hhcp)
        landed_raw = scan_h_raw(tmp_hhcp)      # independent non-parser read of H

        # ── GUARD: landed metadata chain ─────────────────────────────────────
        meta_bad, chain = check_landed_metadata(landed_parser, landed_raw, S, npairs)
        if meta_bad:
            failures.append(("landed_metadata_mismatch", "; ".join(meta_bad)))

        # ── GUARD: landed H == intended generated content ────────────────────
        eq_bad, eq_adds, eq_rep = check_landed_h_equivalence(
            landed_parser, parser, S, landed_raw, src_raw)
        if eq_bad:
            failures.append(("landed_h_not_equivalent_to_intended", "; ".join(eq_bad)))
        for name, detail in eq_adds:
            failures.append((name, detail))

        # ── GUARD (ADDED IN S2): landed persisted-field fidelity ─────────────
        lnd_fld_bad, lnd_fld_rep = check_persisted_row_fields(
            landed_raw, landed_parser, "landed H")
        if lnd_fld_bad:
            failures.append(("landed_persisted_field_mismatch", "; ".join(lnd_fld_bad)))

        # ── GUARD: the landed S, as a task SET ───────────────────────────────
        (schema_bad, multiset_bad, row_bad, slot_viol, avail_viol, flatload_bad,
         load_rep) = verify_landed_schedule(landed_parser, tmp_sol, cap)
        if schema_bad:
            # trip IMMEDIATELY: with a required column absent, nothing downstream of
            # here (including the round trip through load_static) has a defined
            # meaning, and a bare traceback would replace the diagnosis.
            failures.append(("landed_s_schema_invalid", "; ".join(schema_bad)))
            rec.update(stage="landed_s_schema_invalid",
                       landed_s_columns=load_rep.get("s_columns"))
            trip(rec)
        if multiset_bad:
            failures.append(("served_task_multiset_mismatch", "; ".join(multiset_bad)))
        if row_bad:
            failures.append(("row_patient_task_mismatch",
                             f"{len(row_bad)} row/task ownership problems; "
                             f"first={row_bad[0]}"))
        if slot_viol:
            failures.append(("requested_slot_violation",
                             f"{len(slot_viol)} of {load_rep.get('landed_rows')} landed "
                             f"rows place a patient outside its landed allowed-slot SET; "
                             f"first={slot_viol[0]}"))
        if avail_viol:
            failures.append(("availability_candidate_violation",
                             f"{len(avail_viol)} violations against the LANDED H; "
                             f"first={avail_viol[0]}"))
        if flatload_bad:
            failures.append(("landed_flatload_mismatch",
                             f"LABELLED ADDITION: {len(flatload_bad)} rows whose persisted "
                             f"FlatLoad differs from the cell load recomputed from the "
                             f"landed task set; first={flatload_bad[0]}"))
        if load_rep and load_rep["cells_over_cap"]:
            failures.append(("landed_max_load_gt_cap",
                             f"{load_rep['cells_over_cap']} (nurse@slot) cells exceed "
                             f"cap={cap!r} (fp_slack={slack!r}) when the load is "
                             f"recomputed from the VERIFIED exact landed task set; "
                             f"worst={load_rep['cells_over_cap_first']}"))
        if load_rep and abs(load_rep["landed_max_load_raw"] - max_cell) > slack:
            failures.append(("landed_max_ne_generator_max",
                             f"LABELLED ADDITION: independently recomputed landed max load "
                             f"{load_rep['landed_max_load_raw']!r} != generator's "
                             f"{max_cell!r} (fp_slack={slack!r}) -- two measurements of the "
                             f"same quantity disagree"))

        # ── GUARDS carried forward: round trip through the recovery pipeline ──
        rt_max, rt_served, rt_slot_viol, rt_nurses, rt_tasks = \
            round_trip_via_recovery_pipeline(landed_parser, tmp_sol)
        if rt_slot_viol:
            failures.append(("round_trip_slot_violation",
                             f"{rt_slot_viol} task-slot pairs outside the landed "
                             f"allowed-slot set after re-loading through load_static"))
        if rt_max > cap + slack:
            failures.append(("rt_max_gt_cap",
                             f"rt_max={rt_max!r} cap={cap!r} fp_slack={slack!r}"))
        if rt_served != served:
            failures.append(("rt_served_ne_served",
                             f"LABELLED ADDITION (not a substitute for the task-set "
                             f"equality, which load_static cannot see because it collapses "
                             f"duplicates into a dict): rt_served={rt_served} "
                             f"served={served}"))

        rec.update(
            stage="post_write_verify",
            metadata_fix=meta_rep,
            landed_declared_nurses=landed_raw["declared_nurses"],
            landed_nurse_rows=landed_raw["n_nurse_rows"],
            landed_declared_patients=landed_raw["declared_patients"],
            landed_declared_tasks=landed_raw["declared_tasks"],
            landed_score_rows=landed_raw["n_score_rows"],
            landed_nurse_count_chain=chain,
            landed_nurse_count_chain_equal=bool(len(set(chain.values())) == 1),
            landed_equivalence=eq_rep,
            landed_persisted_field_report=lnd_fld_rep,
            allowed_slot_set_sizes=eq_rep["allowed_slot_set_sizes"],
            emitted_rows=emitted_rows,
            landed_rows=load_rep.get("landed_rows"),
            landed_cells=load_rep.get("landed_cells"),
            landed_max_load=load_rep.get("landed_max_load"),
            landed_distinct_tasks_scheduled=load_rep.get("distinct_tasks_scheduled"),
            landed_total_task_placements=load_rep.get("total_task_placements"),
            landed_max_task_multiplicity=load_rep.get("max_task_multiplicity"),
            landed_distinct_slots_used=load_rep.get("distinct_slots_used"),
            landed_duplicate_row_keys=load_rep.get("duplicate_row_keys"),
            landed_flatload_mismatches=load_rep.get("flatload_mismatches"),
            landed_s_columns=load_rep.get("s_columns"),
            task_multiset_equals_landed_task_set=load_rep.get(
                "task_multiset_equals_landed_task_set"),
            min_task_increment=load_rep.get("min_task_increment"),
            requested_slot_violations=len(slot_viol),
            availability_candidate_violations=len(avail_viol),
            row_ownership_violations=len(row_bad),
            cells_over_cap=load_rep.get("cells_over_cap"),
            rt_max=round(rt_max, 1), rt_served=rt_served,
            rt_slot_violations=rt_slot_viol,
            rt_nurses=rt_nurses, rt_tasks=rt_tasks,
            rt_max_le_cap=bool(rt_max <= cap + slack),
            verified_from_disk=True,
            landed_parser_constructed_times=1,
            tmp_hhcp=tmp_hhcp, tmp_sol=tmp_sol,
        )
        if failures:
            trip(rec)

        # ── PUBLISH — only now, and NOT claimed to be pair-atomic ────────────
        publish_steps = []
        try:
            os.replace(tmp_hhcp, canon_hhcp)
            publish_steps.append("H")
            os.replace(tmp_sol, canon_sol)
            publish_steps.append("S")
        except Exception as pe:
            failures.append(("publish_failed",
                             f"publish raised after steps {publish_steps}: "
                             f"{type(pe).__name__}: {pe} -- BOTH canonical paths are "
                             f"removed; two os.replace calls are NOT claimed to be "
                             f"pair-atomic"))
            rec.update(stage="publish_failed", publish_steps=publish_steps)
            trip(rec)
        if not (os.path.exists(canon_hhcp) and os.path.exists(canon_sol)):
            failures.append(("publish_failed",
                             f"after both os.replace calls, H exists="
                             f"{os.path.exists(canon_hhcp)} S exists="
                             f"{os.path.exists(canon_sol)}"))
            rec.update(stage="publish_incomplete", publish_steps=publish_steps)
            trip(rec)

        rec.update(published=True, hhcp=os.path.basename(canon_hhcp),
                   sol=os.path.basename(canon_sol),
                   hhcp_path=canon_hhcp, sol_path=canon_sol,
                   publish_steps=publish_steps,
                   publish_atomicity="two os.replace calls; NOT claimed pair-atomic and "
                                     "NOT family-atomic; on success both exist, a CAUGHT "
                                     "failure is rolled back (both removed)",
                   removed_stale_canonical=removed_stale,
                   feasible=True)
        rec.pop("tmp_hhcp", None)
        rec.pop("tmp_sol", None)

        if not quiet:
            print(f"[slot_rule={SLOT_RULE} rho={rho:.2f} cap={cap:.6g} N={len(S)} "
                  f"(start {n_start}, +{escalations}) seed={seed} pack={pack}]")
            print(f"  H metadata: #(nurse) {meta_rep['declared_nurses_before']} -> "
                  f"{meta_rep['declared_nurses_after']}; chain {chain} "
                  f"(all equal = {len(set(chain.values())) == 1})")
            print(f"  landed parser built ONCE on the landed H; equivalence checked: "
                  f"patients={eq_rep['patients_checked']} tasks={eq_rep['tasks_checked']} "
                  f"nurses={eq_rep['nurses_checked']} "
                  f"allowed_slot_set_sizes={eq_rep['allowed_slot_set_sizes']} "
                  f"patient_block_verbatim={eq_rep['patient_block_verbatim']}")
            print(f"  persisted fields: patients={lnd_fld_rep['patients_field_checked']} "
                  f"nurses={lnd_fld_rep['nurses_field_checked']} "
                  f"tasks={lnd_fld_rep['tasks_field_checked']} mismatches="
                  f"{lnd_fld_rep['patient_field_mismatches']}/"
                  f"{lnd_fld_rep['patient_count_mismatches']}/"
                  f"{lnd_fld_rep['nurse_field_mismatches']}/"
                  f"{lnd_fld_rep['nurse_count_mismatches']}/"
                  f"{lnd_fld_rep['task_field_mismatches']}/"
                  f"{lnd_fld_rep['task_count_mismatches']}")
            print(f"  raw-TaskIDs Counter == {{task: 1 for task in landed tasks}}: "
                  f"{load_rep['task_multiset_equals_landed_task_set']} "
                  f"(distinct={load_rep['distinct_tasks_scheduled']} "
                  f"placements={load_rep['total_task_placements']} "
                  f"max_multiplicity={load_rep['max_task_multiplicity']})")
            print(f"  input cand-slot check: {pairs_checked} pairs, {len(cand_viol)} "
                  f"violations; defensive slot-filter removals="
                  f"{diag['slot_filter_removals']}")
            print(f"  min-cover/task={cover} served={served}/{total_tasks} "
                  f"unplaced={len(unplaced)} whole-block={diag['blocks_placed']} "
                  f"fell-back={diag['blocks_fell_back']}")
            print(f"  landed max load (from the VERIFIED task set)="
                  f"{load_rep['landed_max_load']} vs cap={cap:.6g} "
                  f"fp_slack={slack:.3g} min_task_increment="
                  f"{load_rep['min_task_increment']} cells={load_rep['landed_cells']} "
                  f"slots_used={load_rep['distinct_slots_used']}")
            print(f"  round trip via load_static: nurses={rt_nurses} tasks={rt_tasks} "
                  f"scheduled={rt_served} rt_max={rt_max:.1f} "
                  f"rt_slot_violations={rt_slot_viol}")
            print(f"  PUBLISHED: {canon_hhcp}\n             {canon_sol}")
        return rec

    except GeneratorGuardFailure:
        raise
    except BaseException as e:
        # A generic exception goes through the SAME cleanup path as a guard trip.
        if not failures:
            failures.append(("unexpected_exception", f"{type(e).__name__}: {e}"))
        try:
            quarantine(dict(tag=tag, stage="exception_cleanup", published=False,
                            error=f"{type(e).__name__}: {e}"))
        except Exception as ce:                                   # pragma: no cover
            sys.stderr.write(f"*** CLEANUP ITSELF FAILED: {type(ce).__name__}: {ce}\n")
        sys.stderr.write(
            f"*** ABORTED BY EXCEPTION -- rollback performed: canonical H exists="
            f"{os.path.exists(canon_hhcp)} S exists={os.path.exists(canon_sol)}\n")
        raise


def _report(rec, failures, reasons):
    sys.stderr.write("\n*** GENERATOR GUARD FAILURE -- NOTHING PUBLISHED ***\n")
    for k, v in failures:
        sys.stderr.write(f"  GUARD TRIPPED: {k}: {v}\n")
    sys.stderr.write(f"  stage={rec.get('stage')} tag={rec.get('tag')}\n")
    sys.stderr.write(f"  quarantine reason file: {reasons}\n")
    sys.stderr.flush()


def run_one(rho, nurses=16, seed=1, pack="tight", src=SRC_DEFAULT, outdir=None,
            out_hhcp=None, out_sol=None, quiet=False):
    """Wrap generate() and translate the outcome into a PROCESS EXIT CODE, so a
    family runner and a fault-injection harness share identical failure
    semantics: 0 ok, 3 guard trip, 4 unexpected exception, 5 missing pandas."""
    try:
        rec = generate(rho, nurses=nurses, seed=seed, pack=pack, src=src,
                       out_hhcp=out_hhcp, out_sol=out_sol, quiet=quiet, outdir=outdir)
        print("MANIFEST_JSON " + json.dumps(dict(rec, guards_tripped=[]), default=str))
        return EXIT_OK
    except GeneratorGuardFailure as e:
        print("MANIFEST_JSON " + json.dumps(
            dict(e.rec, guards_tripped=[k for k, _ in e.failures],
                 guard_details=[f"{k}: {v}" for k, v in e.failures]), default=str))
        return EXIT_GUARD_FAIL
    except Exception as e:
        sys.stderr.write(f"*** INTERNAL ERROR (not a guard trip): {type(e).__name__}: {e}\n")
        print("MANIFEST_JSON " + json.dumps(
            dict(stage="internal_error", published=False,
                 guards_tripped=["unexpected_exception"],
                 guard_details=[f"unexpected_exception: {type(e).__name__}: {e}"]),
            default=str))
        return EXIT_INTERNAL_ERROR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.85,
                    help="target per-cell capacity tightness; cell cap C = rho*H_max "
                         "(higher = tighter)")
    ap.add_argument("--nurses", type=int, default=24,
                    help="starting nurse-set size (auto-escalates if tight)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pack", choices=["tight", "spread"], default="tight")
    ap.add_argument("--src", type=str, default=SRC_DEFAULT)
    ap.add_argument("--out-hhcp", type=str, default=None)
    ap.add_argument("--out-sol", type=str, default=None)
    ap.add_argument("--outdir", type=str, default=None,
                    help="directory for the default H_<tag>.csv / S_<tag>.csv pair; "
                         "defaults to this file's directory")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--print-guards", action="store_true",
                    help="print the guard inventory and exit 0 without generating")
    args = ap.parse_args()
    if args.print_guards:
        print(json.dumps(dict(
            generator=os.path.basename(__file__),
            semantics_ported_from=SEMANTICS_PORTED_FROM,
            carried_from_pre_s2=list(GUARDS_CARRIED_FROM_PRE_S2),
            ported_from_g2=list(GUARDS_PORTED_FROM_G2),
            added_in_s2=list(GUARDS_ADDED_IN_S2),
            labelled_additions=list(GUARDS_LABELLED_ADDITIONS),
            total=len(GUARD_NAMES)), indent=1))
        sys.exit(EXIT_OK)
    code = run_one(args.rho, nurses=args.nurses, seed=args.seed, pack=args.pack,
                   src=args.src, outdir=args.outdir, out_hhcp=args.out_hhcp,
                   out_sol=args.out_sol, quiet=args.quiet)
    sys.exit(code)


if __name__ == "__main__":
    main()
