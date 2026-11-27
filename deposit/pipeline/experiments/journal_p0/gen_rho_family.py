"""
gen_rho_family.py — OFFICIAL family runner for the capacity-tightness instance
family, and the writer of `rho_manifest.csv` that E3 (phase diagram), E4 (fair
A1/A2), the epsilon stage and the rolling stage consume.

    cd experiments && python journal_p0/gen_rho_family.py

WHAT CHANGED, AND WHY IT MATTERS
--------------------------------
FAIL FAST, NEVER SKIP.  The pre-S2 version caught an infeasible cell, printed
`SKIP rho=… seed=…` and CONTINUED, so a short family would have been written and
consumed silently.  That is forbidden.  This version has no `continue` and no
`except` around a cell: the first cell that does not publish aborts the whole run
with a non-zero exit code, and `rho_manifest.csv` is NOT touched.

COMPLETE AND UNIQUE.  Before the manifest is promoted it must contain EXACTLY
`|RHOS| x |SEEDS|` rows, one per designed `(rho_target, seed)` cell, with the
full Cartesian grid covered, no duplicate key, no empty required field, no NaN /
inf, every guard count zero, and every recorded H/S SHA256 re-verified against
the bytes on disk.  Any violation aborts before promotion.

TEMP -> VALIDATE -> PROMOTE.  The manifest is written to `_tmp/`, validated, then
moved into place by a single `os.replace`.  Per-pair publication is the
generator's own temp -> validate -> promote (see `make_rho_instance.py`).  NOTHING
here is claimed to be family-atomic: if a late cell fails, the pairs already
published by earlier cells in this run REMAIN on disk (they are reported, and
`rho_manifest.csv` is absent or stale, so the family is detectably incomplete).
They are deliberately NOT auto-deleted, because the output directory may legally
hold a previous family that must not be destroyed by a partial run.

OPERATIONAL CONSEQUENCE.  `make_rho_instance.generate()` clears the canonical
pair for the tag it is generating, inside its guarded region.  Therefore: run this
into a STAGING `--outdir` and promote the family afterwards; do not point it at a
directory whose current family you are not prepared to lose.

WHY `N` IS PERSISTED BESIDE `rho_target`, `cap` AND THE REALIZED UTILIZATION
---------------------------------------------------------------------------
Under the hard allowed-slot constraint a patient's tasks can only be packed into
that patient's requested slot, so the auto-escalation loop needs MORE caregivers
than it did when the packer was free to choose any slot the caregiver worked.  The
caregiver count therefore moves with the semantics, not only with `rho`.  `N`,
`N_after_cover`, `auto_escalations`, `rho_target`, `cap`, `rho_global`,
`rho_used_mean` and `rho_used_max` are all persisted per cell so that a later
analysis CANNOT attribute a caregiver-count effect to `rho` alone.  When
`--baseline-manifest` is given, `pre_s2_N` and `delta_N_vs_pre_s2` are persisted
too, and the delta is summarised on stdout.  NOTHING is retuned to reduce the
delta.

A guard that does not trip is REGRESSION EVIDENCE, never proof. This runner
produces INPUT instances; it is not a formal experiment stage and it closes
nothing.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from make_rho_instance import (                                   # noqa: E402
    SRC_DEFAULT, SLOT_RULE, GeneratorGuardFailure, generate, resolve_src)

# ── the design grid. FIXED. Never widened or narrowed to obtain a pass. ───────
RHOS = [0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
SEEDS = [1, 2, 3, 4, 5]
NURSES_START = 16
OUTDIR_DEFAULT = _HERE
MANIFEST_NAME = "rho_manifest.csv"

EXIT_OK = 0
EXIT_CELL_FAILED = 3
EXIT_MANIFEST_INVALID = 6

# Columns the downstream stages read directly. Presence is validated, not assumed.
DOWNSTREAM_REQUIRED = ("rho_target", "seed", "hhcp", "sol")

MANIFEST_COLUMNS = (
    # ---- pre-S2 columns, order preserved, so existing readers are unaffected --
    "rho_target", "cap", "N", "seed", "served", "unplaced", "rho_global",
    "rho_used_mean", "rho_used_max", "max_cell", "feasible", "rt_max", "hhcp", "sol",
    # ---- identity / provenance ----------------------------------------------
    "tag", "slot_rule", "h_max", "generated_utc",
    "src", "src_sha256", "generator", "generator_sha256",
    "family_runner", "family_runner_sha256",
    "h_sha256", "h_bytes", "s_sha256", "s_bytes",
    # ---- declared vs actual N, and the caregiver-count confound -------------
    "n_start_arg", "N_after_cover", "auto_escalations",
    "landed_declared_nurses", "landed_nurse_rows", "landed_nurse_count_chain_equal",
    "pre_s2_N", "delta_N_vs_pre_s2",
    # ---- served / task-multiset result --------------------------------------
    "total_tasks", "task_multiset_equals_landed_task_set",
    "landed_distinct_tasks_scheduled", "landed_total_task_placements",
    "landed_max_task_multiplicity",
    # ---- violation counts (every one must be 0) -----------------------------
    "cand_slot_violations", "slot_filter_removals", "requested_slot_violations",
    "availability_candidate_violations", "row_ownership_violations",
    "landed_duplicate_row_keys", "landed_flatload_mismatches", "cells_over_cap",
    "rt_slot_violations",
    "patient_field_mismatches", "patient_count_mismatches",
    "nurse_field_mismatches", "nurse_count_mismatches",
    "task_field_mismatches", "task_count_mismatches",
    # ---- cap and the independently recomputed load --------------------------
    "cap_fp_slack", "min_task_increment", "landed_max_load",
    "landed_rows", "landed_cells", "landed_distinct_slots_used",
    "allowed_slot_set_sizes",
)
# Numeric columns scanned for NaN / inf.
NUMERIC_COLUMNS = (
    "rho_target", "cap", "N", "seed", "served", "unplaced", "rho_global",
    "rho_used_mean", "rho_used_max", "max_cell", "rt_max", "h_max",
    "h_bytes", "s_bytes", "n_start_arg", "N_after_cover", "auto_escalations",
    "landed_declared_nurses", "landed_nurse_rows", "total_tasks",
    "landed_distinct_tasks_scheduled", "landed_total_task_placements",
    "landed_max_task_multiplicity", "cap_fp_slack", "min_task_increment",
    "landed_max_load", "landed_rows", "landed_cells", "landed_distinct_slots_used",
)
ZERO_COLUMNS = (
    "unplaced", "cand_slot_violations", "slot_filter_removals",
    "requested_slot_violations", "availability_candidate_violations",
    "row_ownership_violations", "landed_duplicate_row_keys",
    "landed_flatload_mismatches", "cells_over_cap", "rt_slot_violations",
    "patient_field_mismatches", "patient_count_mismatches",
    "nurse_field_mismatches", "nurse_count_mismatches",
    "task_field_mismatches", "task_count_mismatches",
)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_baseline_N(path):
    """Read a PREVIOUS manifest read-only, purely to persist the caregiver-count
    delta. Optional; absent -> the delta columns are empty. Never used as a pass
    condition and never used to retune anything."""
    if not path:
        return {}
    df = pd.read_csv(path)
    for c in ("rho_target", "seed", "N"):
        if c not in df.columns:
            raise SystemExit(f"baseline manifest {path} has no {c!r} column")
    return {(round(float(r["rho_target"]), 6), int(r["seed"])): int(r["N"])
            for _, r in df.iterrows()}


def row_from_manifest(rec, outdir, baseline, provenance):
    """Flatten one generator manifest dict into a manifest ROW. Every value comes
    from the generator's own measurements or from the landed bytes; nothing is
    assumed and no expected value is inserted."""
    h_path = os.path.join(outdir, rec["hhcp"])
    s_path = os.path.join(outdir, rec["sol"])
    fld = rec["landed_persisted_field_report"]
    key = (round(float(rec["rho_target"]), 6), int(rec["seed"]))
    pre = baseline.get(key)
    row = dict(
        rho_target=rec["rho_target"], cap=rec["cap"], N=rec["N"], seed=rec["seed"],
        served=rec["served"], unplaced=rec["unplaced"], rho_global=rec["rho_global"],
        rho_used_mean=rec["rho_used_mean"], rho_used_max=rec["rho_used_max"],
        max_cell=rec["max_cell_load"], feasible=bool(rec["feasible"]),
        rt_max=rec["rt_max"], hhcp=rec["hhcp"], sol=rec["sol"],
        tag=rec["tag"], slot_rule=rec["slot_rule"], h_max=rec["h_max"],
        generated_utc=provenance["generated_utc"],
        src=os.path.basename(rec["src"]), src_sha256=provenance["src_sha256"],
        generator=provenance["generator"],
        generator_sha256=provenance["generator_sha256"],
        family_runner=provenance["family_runner"],
        family_runner_sha256=provenance["family_runner_sha256"],
        h_sha256=sha256_of(h_path), h_bytes=os.path.getsize(h_path),
        s_sha256=sha256_of(s_path), s_bytes=os.path.getsize(s_path),
        n_start_arg=rec["n_start_arg"], N_after_cover=rec["N_after_cover"],
        auto_escalations=rec["auto_escalations"],
        landed_declared_nurses=rec["landed_declared_nurses"],
        landed_nurse_rows=rec["landed_nurse_rows"],
        landed_nurse_count_chain_equal=bool(rec["landed_nurse_count_chain_equal"]),
        pre_s2_N=("" if pre is None else pre),
        delta_N_vs_pre_s2=("" if pre is None else int(rec["N"]) - pre),
        total_tasks=rec["total_tasks"],
        task_multiset_equals_landed_task_set=bool(
            rec["task_multiset_equals_landed_task_set"]),
        landed_distinct_tasks_scheduled=rec["landed_distinct_tasks_scheduled"],
        landed_total_task_placements=rec["landed_total_task_placements"],
        landed_max_task_multiplicity=rec["landed_max_task_multiplicity"],
        cand_slot_violations=rec["cand_slot_violations"],
        slot_filter_removals=rec["slot_filter_removals"],
        requested_slot_violations=rec["requested_slot_violations"],
        availability_candidate_violations=rec["availability_candidate_violations"],
        row_ownership_violations=rec["row_ownership_violations"],
        landed_duplicate_row_keys=rec["landed_duplicate_row_keys"],
        landed_flatload_mismatches=rec["landed_flatload_mismatches"],
        cells_over_cap=rec["cells_over_cap"],
        rt_slot_violations=rec["rt_slot_violations"],
        patient_field_mismatches=fld["patient_field_mismatches"],
        patient_count_mismatches=fld["patient_count_mismatches"],
        nurse_field_mismatches=fld["nurse_field_mismatches"],
        nurse_count_mismatches=fld["nurse_count_mismatches"],
        task_field_mismatches=fld["task_field_mismatches"],
        task_count_mismatches=fld["task_count_mismatches"],
        cap_fp_slack=rec["cap_fp_slack"], min_task_increment=rec["min_task_increment"],
        landed_max_load=rec["landed_max_load"], landed_rows=rec["landed_rows"],
        landed_cells=rec["landed_cells"],
        landed_distinct_slots_used=rec["landed_distinct_slots_used"],
        allowed_slot_set_sizes=";".join(str(v) for v in rec["allowed_slot_set_sizes"]),
    )
    missing = [c for c in MANIFEST_COLUMNS if c not in row]
    extra = [c for c in row if c not in MANIFEST_COLUMNS]
    if missing or extra:
        raise SystemExit(f"manifest row schema drift: missing={missing} extra={extra}")
    return row


def validate_manifest(df, outdir):
    """Independent validation of the manifest ABOUT TO BE PROMOTED. The design
    grid is re-derived here from RHOS x SEEDS, never read back from the file.
    Returns a list of reasons; empty means every check passed."""
    bad = []
    designed = [(round(float(r), 6), int(s)) for r in RHOS for s in SEEDS]
    if len(set(designed)) != len(designed):
        bad.append(f"the DESIGN GRID itself has duplicate cells: "
                   f"{len(designed)} cells, {len(set(designed))} distinct")
    print(f"[shape]    expected={len(designed)} actual={len(df)}")
    if len(df) != len(designed):
        bad.append(f"row count expected={len(designed)} actual={len(df)}")

    missing_cols = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing_cols:
        bad.append(f"missing columns {missing_cols}")
        return bad
    if list(df.columns) != list(MANIFEST_COLUMNS):
        bad.append(f"column order drift: {list(df.columns)}")
    for c in DOWNSTREAM_REQUIRED:
        if c not in df.columns:
            bad.append(f"downstream-required column {c!r} absent")

    # A malformed cell must become a REPORTED reason, never a traceback: a
    # validator that crashes on mutilated input has told the caller nothing.
    keys, keybad = [], []
    for i, (_, r) in enumerate(df.iterrows()):
        try:
            keys.append((round(float(r["rho_target"]), 6), int(r["seed"])))
        except Exception as e:
            keybad.append(f"row {i}: rho_target={r['rho_target']!r} seed={r['seed']!r} "
                          f"({type(e).__name__})")
    if keybad:
        bad.append(f"unusable (rho_target, seed) keys: {keybad[:6]}")
    dup = sorted(k for k in set(keys) if keys.count(k) > 1)
    if dup:
        bad.append(f"duplicate (rho_target, seed) keys: {dup}")
    miss = sorted(set(designed) - set(keys))
    extra = sorted(set(keys) - set(designed))
    print(f"[keys]     unique={len(set(keys))} duplicates={len(dup)}")
    print(f"[coverage] designed={len(designed)} missing={len(miss)} undesigned={len(extra)}")
    if miss:
        bad.append(f"design cells missing from the manifest: {miss}")
    if extra:
        bad.append(f"manifest rows outside the design grid: {extra}")

    nan_hits = []
    for c in NUMERIC_COLUMNS:
        for i, v in enumerate(df[c].tolist()):
            try:
                f = float(v)
            except Exception:
                nan_hits.append(f"{c}[{i}]={v!r} not numeric")
                continue
            if math.isnan(f) or math.isinf(f):
                nan_hits.append(f"{c}[{i}]={v!r}")
    print(f"[numeric]  NaN/inf/non-numeric hits={len(nan_hits)}")
    if nan_hits:
        bad.append(f"NaN / inf / non-numeric cells: {nan_hits[:6]}")

    empty_hits = []
    for c in ("hhcp", "sol", "tag", "slot_rule", "h_sha256", "s_sha256",
              "src_sha256", "generator_sha256", "family_runner_sha256"):
        for i, v in enumerate(df[c].tolist()):
            if v is None or str(v).strip() == "" or str(v) == "nan":
                empty_hits.append(f"{c}[{i}]")
    if empty_hits:
        bad.append(f"empty required cells: {empty_hits[:8]}")

    nonzero = []
    for c in ZERO_COLUMNS:
        for i, v in enumerate(df[c].tolist()):
            try:
                iv = int(v)
            except Exception:
                nonzero.append(f"{c}[{i}]={v!r} not an integer")
                continue
            if iv != 0:
                nonzero.append(f"{c}[{i}]={v}")
    print(f"[guards]   non-zero violation cells={len(nonzero)}")
    if nonzero:
        bad.append(f"non-zero guard counts: {nonzero[:8]}")

    for c, want in (("feasible", True),
                    ("task_multiset_equals_landed_task_set", True),
                    ("landed_nurse_count_chain_equal", True)):
        offend = [i for i, v in enumerate(df[c].tolist()) if bool(v) is not want]
        if offend:
            bad.append(f"{c} is not {want} in rows {offend[:8]}")

    off = []
    for i, (_, r) in enumerate(df.iterrows()):
        try:
            t = (int(r["landed_declared_nurses"]), int(r["landed_nurse_rows"]),
                 int(r["N"]))
        except Exception:
            off.append((i, "unusable nurse-count cells"))
            continue
        if not (t[0] == t[1] == t[2]):
            off.append((i, t))
    if off:
        bad.append(f"declared/rows/N chain disagrees in rows {off[:4]}")

    names = list(df["hhcp"]) + list(df["sol"])
    if len(set(names)) != len(names):
        bad.append("duplicate H/S filenames in the manifest")
    rehash = []
    for _, r in df.iterrows():
        for col, hcol, bcol in (("hhcp", "h_sha256", "h_bytes"),
                                ("sol", "s_sha256", "s_bytes")):
            name = str(r[col]).strip()
            # a blank name must NOT be joined onto outdir: that yields the
            # DIRECTORY, which exists, and hashing it would raise instead of
            # reporting. Treat anything that is not a regular file as absent.
            if name == "" or name.lower() == "nan":
                rehash.append(f"{col} filename is empty")
                continue
            p = os.path.join(outdir, name)
            if not os.path.isfile(p):
                rehash.append(f"{name} is not a regular file under {outdir}")
                continue
            got = sha256_of(p)
            if got != str(r[hcol]):
                rehash.append(f"{name} sha256 recorded={r[hcol]} on-disk={got}")
            try:
                want_bytes = int(r[bcol])
            except Exception:
                rehash.append(f"{name} recorded byte size {r[bcol]!r} is not an integer")
                continue
            if int(os.path.getsize(p)) != want_bytes:
                rehash.append(f"{name} byte size recorded={want_bytes} "
                              f"on-disk={os.path.getsize(p)}")
    print(f"[hashes]   files re-hashed={2 * len(df)} mismatches={len(rehash)}")
    if rehash:
        bad.append(f"H/S hash or size mismatches: {rehash[:6]}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default=None,
                    help="where the H_/S_ pairs and rho_manifest.csv go; defaults to "
                         "this file's directory")
    ap.add_argument("--src", type=str, default=SRC_DEFAULT)
    ap.add_argument("--baseline-manifest", type=str, default=None,
                    help="OPTIONAL previous rho_manifest.csv, read read-only, so the "
                         "caregiver-count delta can be persisted per cell")
    ap.add_argument("--quiet-cells", action="store_true",
                    help="suppress each cell's per-cell report (guard failures are "
                         "always reported)")
    args = ap.parse_args()

    outdir = os.path.abspath(args.outdir) if args.outdir else OUTDIR_DEFAULT
    os.makedirs(outdir, exist_ok=True)
    src = resolve_src(args.src)
    gen_path = os.path.join(_HERE, "make_rho_instance.py")
    provenance = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        src_sha256=sha256_of(src),
        generator="make_rho_instance.py", generator_sha256=sha256_of(gen_path),
        family_runner="gen_rho_family.py",
        family_runner_sha256=sha256_of(os.path.abspath(__file__)))
    baseline = load_baseline_N(args.baseline_manifest)

    designed = [(r, s) for r in RHOS for s in SEEDS]
    print(f"grid: RHOS={RHOS} SEEDS={SEEDS} nurses_start={NURSES_START} "
          f"cells={len(designed)}")
    print(f"outdir={outdir}")
    print(f"src={src} sha256={provenance['src_sha256']}")
    print(f"generator sha256={provenance['generator_sha256']}")
    print(f"slot_rule={SLOT_RULE} (the allowed-slot SET is a singleton in THIS shipped "
          f"data; that is a property of the data, not a general equality)")
    if baseline:
        print(f"baseline manifest: {args.baseline_manifest} ({len(baseline)} cells) — "
              f"persisted for the caregiver-count delta only")

    rows, published = [], []
    for rho, seed in designed:
        try:
            rec = generate(rho, nurses=NURSES_START, seed=seed, src=src,
                           outdir=outdir, quiet=args.quiet_cells)
        except GeneratorGuardFailure as e:
            sys.stderr.write(
                f"\n*** CELL rho={rho} seed={seed} FAILED -- NO PAIR PUBLISHED FOR IT.\n"
                f"*** guards tripped: {[k for k, _ in e.failures]}\n"
                f"*** the family run ABORTS here: no rho_manifest.csv is written, so the "
                f"family is detectably incomplete.\n"
                f"*** pairs already published by THIS run and left on disk "
                f"({len(published)}): {published}\n"
                f"*** they are NOT auto-deleted; nothing here is family-atomic.\n")
            sys.exit(EXIT_CELL_FAILED)
        published.append(rec["tag"])
        rows.append(row_from_manifest(rec, outdir, baseline, provenance))
        r = rows[-1]
        print(f"rho={rho:.2f} s{seed}: N={r['N']} (start {r['N_after_cover']}, "
              f"+{r['auto_escalations']}) declared_nurses={r['landed_declared_nurses']} "
              f"util(mean/max)={r['rho_used_mean']:.3f}/{r['rho_used_max']:.3f} "
              f"served={r['served']}/{r['total_tasks']} "
              f"task_set_exact={r['task_multiset_equals_landed_task_set']} "
              f"landed_max_load={r['landed_max_load']} cap={r['cap']} "
              f"delta_N={r['delta_N_vs_pre_s2']}", flush=True)

    df = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))

    tmpdir = os.path.join(outdir, "_tmp")
    os.makedirs(tmpdir, exist_ok=True)
    tmp_man = os.path.join(tmpdir, f"TMP_{MANIFEST_NAME}.{os.getpid()}")
    man = os.path.join(outdir, MANIFEST_NAME)
    df.to_csv(tmp_man, index=False)

    # validate the BYTES that will be promoted, re-read from disk
    reread = pd.read_csv(tmp_man)
    print("\n--- manifest validation (re-read from the temp bytes) ---")
    bad = validate_manifest(reread, outdir)
    if bad:
        sys.stderr.write("\n*** MANIFEST VALIDATION FAILED -- rho_manifest.csv NOT "
                         "PROMOTED ***\n")
        for b in bad:
            sys.stderr.write(f"  {b}\n")
        try:
            os.remove(tmp_man)
        except Exception as e:                                    # pragma: no cover
            sys.stderr.write(f"  (could not remove {tmp_man}: {e})\n")
        sys.exit(EXIT_MANIFEST_INVALID)

    os.replace(tmp_man, man)
    print(f"\nwrote {man}  ({len(df)} rows, {len(set(df.tag))} distinct tags)")
    print(f"rho_manifest.csv sha256={sha256_of(man)}")

    if baseline:
        d = [(int(r["delta_N_vs_pre_s2"]), float(r["rho_target"]), int(r["seed"]))
             for _, r in df.iterrows() if str(r["delta_N_vs_pre_s2"]) != ""]
        changed = [t for t in d if t[0] != 0]
        worst = max(d) if d else None
        print(f"caregiver-count delta vs the baseline manifest: "
              f"{len(changed)}/{len(d)} cells changed; worst {worst[0]:+d} at "
              f"rho={worst[1]} seed={worst[2]}" if worst else "no delta computed")
        print("  This is a SEMANTICS effect on N, persisted per cell so that it cannot "
              "later be mistaken for a pure rho effect. Nothing was retuned.")
    print(json.dumps(dict(cells=len(df), tags=len(set(df.tag)),
                          all_feasible=bool(df.feasible.all()),
                          all_task_sets_exact=bool(
                              df.task_multiset_equals_landed_task_set.all()),
                          served_values=sorted(set(int(v) for v in df.served)),
                          N_min=int(df.N.min()), N_max=int(df.N.max()))))
    sys.exit(EXIT_OK)


if __name__ == "__main__":
    main()
