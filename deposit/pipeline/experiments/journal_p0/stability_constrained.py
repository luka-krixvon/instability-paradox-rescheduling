"""Build the three-point stability-constrained summary from the R1 artifacts.

The R1 schema uses an absolute integer horizon budget ``eps``.  "tight",
"medium", and "loose" are therefore selected *within each disruption* from its
registered integer grid: the minimum, the grid value nearest the midpoint, and
the maximum.  This avoids reviving the retired ``frac/dev/ot`` schema and avoids
mixing incomparable absolute budgets across disruptions.
"""
import argparse
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = ["rho", "seed", "absent", "method"]
LEVELS = ("tight", "medium", "loose")


def _require_columns(frame, columns, label):
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _strict_true(series, label):
    values = series.astype(str).str.strip().str.lower()
    invalid = sorted(set(values) - {"true", "false"})
    if invalid:
        raise ValueError(f"{label} contains non-boolean values: {invalid}")
    return values.eq("true")


def _choose_levels(group, key_values):
    group = group.sort_values("eps")
    lo = int(group["eps"].min())
    hi = int(group["eps"].max())
    mid = (lo + hi) / 2.0
    targets = {
        "tight": lo,
        "medium": int(group.loc[(group["eps"] - mid).abs().idxmin(), "eps"]),
        "loose": hi,
    }
    rows = []
    for level in LEVELS:
        eps = targets[level]
        hit = group[group["eps"] == eps]
        if len(hit) != 1:
            raise ValueError(
                f"expected one frontier row at {dict(zip(KEY, key_values))}, "
                f"level={level}, eps={eps}; found {len(hit)}"
            )
        row = hit.iloc[0]
        rows.append({
            **dict(zip(KEY, key_values)),
            "eps_level": level,
            "eps": eps,
            "eps_grid_lo": lo,
            "eps_grid_hi": hi,
            "deviation": int(row["D"]),
            "overtime": float(row["W"]),
            "unserved": int(row["U"]),
        })
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=HERE)
    ap.add_argument("--out-dir", default=HERE)
    args = ap.parse_args(argv)
    frontier_path = os.path.join(args.input_dir, "eps_frontier.csv")
    fr = pd.read_csv(frontier_path)
    _require_columns(
        fr,
        KEY + ["eps", "eps_grid_lo", "eps_grid_hi", "U_common", "U", "W", "D",
               "proven_optimal"],
        "eps_frontier.csv",
    )
    if fr.empty:
        raise ValueError("eps_frontier.csv is empty")
    if not _strict_true(fr["proven_optimal"], "proven_optimal").all():
        raise ValueError("stability summary requires every selected R1 row to be proven optimal")
    if not (fr["U"] == fr["U_common"]).all():
        raise ValueError("R1 frontier contains a row outside the common service level")

    pieces = [_choose_levels(g, key) for key, g in fr.groupby(KEY, sort=True)]
    detailed = pd.concat(pieces, ignore_index=True).sort_values(KEY + ["eps_level"])
    if detailed.groupby(KEY).size().ne(3).any():
        raise ValueError("each disruption/regime must contribute exactly three operating points")

    tab = (
        detailed.groupby(["rho", "method", "eps_level"], sort=True)
        .agg(
            disruptions=("absent", "size"),
            deviation=("deviation", "mean"),
            overtime=("overtime", "mean"),
            unserved=("unserved", "mean"),
        )
        .reset_index()
        .rename(columns={"method": "regime"})
    )
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "stability_constrained.csv")
    tab.to_csv(out, index=False, float_format="%.10g", lineterminator="\n")
    print(f"wrote {out} ({len(tab)} rows)")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
