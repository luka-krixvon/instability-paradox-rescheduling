"""
negctl_manifest_validator.py — NEGATIVE CONTROLS for the manifest validator in
`experiments/journal_p0/gen_rho_family.py` (`validate_manifest`).

A validator that has never been shown to FAIL is not evidence.  Each case
mutilates an IN-MEMORY copy of the promoted `rho_manifest.csv` and requires
`validate_manifest` to return at least one reason containing an expected
substring.  The `intact` case is the positive control and must return none.

READ-ONLY with respect to every official file: the official manifest and the
official H/S pairs are only READ (the hash re-check needs the real files), and
every mutation happens in a pandas DataFrame that is never written back.

A validator that fires here is evidence the validator is wired, NOT a proof and
NOT a validation of any instance or result.

    cd <repo root> && python tests/fixtures/generator/negctl_manifest_validator.py
"""
from __future__ import annotations
import hashlib, json, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
_JP0 = os.path.join(_ROOT, "experiments", "journal_p0")
sys.path.insert(0, _JP0)
sys.path.insert(0, os.path.join(_ROOT, "experiments"))
import pandas as pd                                              # noqa: E402
from gen_rho_family import validate_manifest                      # noqa: E402

MAN = os.path.join(_JP0, "rho_manifest.csv")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def mut_intact(df):
    return df, None


def mut_drop_row(df):
    return df.drop(index=7).reset_index(drop=True), "missing from the manifest"


def mut_duplicate_key(df):
    d = df.copy()
    d.loc[1, ["rho_target", "seed"]] = [d.loc[0, "rho_target"], d.loc[0, "seed"]]
    return d, "duplicate (rho_target, seed) keys"


def mut_bad_hash(df):
    d = df.copy()
    d.loc[3, "h_sha256"] = "0" * 64
    return d, "hash or size mismatch"


def mut_nonzero_violation(df):
    d = df.copy()
    d.loc[5, "requested_slot_violations"] = 1
    return d, "non-zero guard counts"


def mut_not_feasible(df):
    d = df.copy()
    d.loc[2, "feasible"] = False
    return d, "feasible is not True"


def mut_task_set_false(df):
    d = df.copy()
    d.loc[9, "task_multiset_equals_landed_task_set"] = False
    return d, "task_multiset_equals_landed_task_set is not True"


def mut_blank_filename(df):
    d = df.copy()
    d.loc[4, "hhcp"] = ""
    return d, "empty required cells"


def mut_nan_numeric(df):
    d = df.copy()
    d.loc[6, "landed_max_load"] = float("nan")
    return d, "NaN / inf / non-numeric cells"


def mut_chain_broken(df):
    d = df.copy()
    d.loc[8, "landed_declared_nurses"] = int(d.loc[8, "N"]) + 1
    return d, "chain disagrees"


def mut_drop_column(df):
    return df.drop(columns=["tag"]), "missing columns"


def mut_undesigned_cell(df):
    d = df.copy()
    d.loc[0, "rho_target"] = 0.5
    return d, "outside the design grid"


CASES = [mut_intact, mut_drop_row, mut_duplicate_key, mut_bad_hash,
         mut_nonzero_violation, mut_not_feasible, mut_task_set_false,
         mut_blank_filename, mut_nan_numeric, mut_chain_broken, mut_drop_column,
         mut_undesigned_cell]


def main():
    base = pd.read_csv(MAN)
    recs = []
    for fn in CASES:
        df, expect = fn(base.copy())
        reasons = validate_manifest(df, _JP0)
        if expect is None:
            ok = (reasons == [])
        else:
            ok = any(expect in r for r in reasons)
        recs.append(dict(case=fn.__name__, expected_substring=expect,
                         n_reasons=len(reasons), reasons=reasons,
                         verdict="PASS" if ok else "FAIL"))
        print(f"{recs[-1]['verdict']:4}  {fn.__name__:24} reasons={len(reasons)} "
              f"expected={expect!r}")
    n_pass = sum(1 for r in recs if r["verdict"] == "PASS")
    out = dict(
        harness=os.path.relpath(os.path.abspath(__file__), _ROOT),
        harness_sha256=sha256_of(os.path.abspath(__file__)),
        target="experiments/journal_p0/gen_rho_family.py::validate_manifest",
        target_sha256=sha256_of(os.path.join(_JP0, "gen_rho_family.py")),
        manifest_read=os.path.relpath(MAN, _ROOT), manifest_sha256=sha256_of(MAN),
        read_only_wrt_official_files=True,
        cases=len(recs), passed=n_pass, failed=len(recs) - n_pass,
        epistemic_status="NEGATIVE CONTROLS on the manifest validator. Firing shows "
                         "the validator is wired; it is NOT proof and NOT validation.",
        results=recs)
    dest = os.path.join(_HERE, "negctl_manifest_validator.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"\ncases={len(recs)} pass={n_pass} fail={len(recs) - n_pass}\nwrote {dest}")
    sys.exit(0 if n_pass == len(recs) else 1)


if __name__ == "__main__":
    main()
