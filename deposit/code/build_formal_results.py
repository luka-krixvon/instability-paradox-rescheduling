#!/usr/bin/env python3
"""Build the paper's recovery-06 summaries, tables, and figures.

The script consumes only the sealed R1, R1b, and R5 CSVs.  It fails closed on
the registered SNAP identifier, row counts, required columns, and cross-stage
key consistency before writing any paper artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SNAP = "d6c9fa4de2c8ca88bd2fa260f8975398ac42ed9ef14c0527213d0865f2980c60"
EXPECTED_ROWS = {
    "R1/eps_range.csv": 518,
    "R1/eps_frontier.csv": 32372,
    "R1/eps_oracle_validation.csv": 32372,
    "R1b/e4_raw.csv": 6216,
    "R1b/e4_summary.csv": 6,
    "R5/r5_scenario_effects.csv": 390,
    "R5/r5_by_rho.csv": 78,
    "R5/r5_bootstrap.csv": 13,
}
EXPECTED_SHA256 = {
    "R1/eps_range.csv": "e3d038174e7bbb547711790abe2acaaddc2602e85a8b96f51424405fee194c53",
    "R1/eps_frontier.csv": "5418d9e995b3dd102884c05918b7ea6500931250ba1b29b03d313ebb93f1f2f1",
    "R1/eps_oracle_validation.csv": "8c2d13ea7e83f8605050e27bccd6d75e085add0cb88e188bb864536f983e0b17",
    "R1b/e4_raw.csv": "532d1ccf826101ff128abc8f8eb18ef8f9d9611b442fd808a23f273f11e8e2c5",
    "R1b/e4_summary.csv": "8cdc1cbdb7e6dfd9ea71c28a72f1b37e2b7f50cbf910bd4602ceae5ac2677e3e",
    "R5/r5_scenario_effects.csv": "9a882377fa697f60bb4289e9d4acc5916f919098be7d6754cc1a27611544a394",
    "R5/r5_by_rho.csv": "df9f4de6ad8b56b9669e8d8f562e291f174390e19181c24973a2920a9f059f9f",
    "R5/r5_bootstrap.csv": "178544b1e02f73650416b7b5e34cc46d80d4118d9328d1c0d093e1f1d5fe2311",
}
EXPECTED_SEAL_SHA256 = {
    "receipts/RECOVERY_06_SEAL_REPORT.json": "aba32eaa93ac6320ca858bcbcb0d46a236275af39fd39ffc41cf990c825e8005",
    "SHA256SUMS_RECOVERY_06.txt": "182e4c9e889df1de8d9f3cba9d2864c25c528542c3da67be8962de2759b9d784",
}
KEYS = ["rho", "seed", "absent"]
RHOS = [0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
WEIGHTS = [0, 5, 10, 20, 50, 100]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_")


def read_inputs(root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    frames: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    for relpath, expected_rows in EXPECTED_ROWS.items():
        path = root / relpath
        require(path.is_file(), f"missing formal input: {path}")
        frame = pd.read_csv(path, low_memory=False)
        require(len(frame) == expected_rows,
                f"row-count mismatch for {relpath}: {len(frame)} != {expected_rows}")
        if relpath == "R1b/e4_summary.csv":
            # The compact owner summary intentionally omits repeated provenance;
            # it is cross-checked against the SNAP-bearing raw family below.
            require("rho" in frame.columns, "rho missing from R1b summary")
        else:
            require("snap_id" in frame.columns, f"snap_id missing from {relpath}")
            snap_values = set(frame["snap_id"].astype(str))
            require(snap_values == {SNAP}, f"SNAP mismatch in {relpath}: {snap_values}")
        frames[relpath] = frame
        observed_hash = sha256(path)
        require(observed_hash == EXPECTED_SHA256[relpath],
                f"SHA-256 mismatch for {relpath}: {observed_hash}")
        hashes[relpath] = observed_hash
    return frames, hashes


def summarize(frames: dict[str, pd.DataFrame], hashes: dict[str, str]) -> dict:
    eps_range = frames["R1/eps_range.csv"].copy()
    eps = frames["R1/eps_frontier.csv"].copy()
    oracle = frames["R1/eps_oracle_validation.csv"].copy()
    raw = frames["R1b/e4_raw.csv"].copy()
    e4_summary = frames["R1b/e4_summary.csv"].copy()
    r5_by_rho = frames["R5/r5_by_rho.csv"].copy()
    bootstrap = frames["R5/r5_bootstrap.csv"].copy()

    require(eps_range[KEYS].duplicated().sum() == 0, "duplicate R1 disruption key")
    require(raw[KEYS + ["method", "w4"]].duplicated().sum() == 0,
            "duplicate R1b method/weight key")
    require(set(eps_range["rho"].unique()) == set(RHOS), "unexpected rho grid")
    require(set(raw["w4"].unique()) == set(WEIGHTS), "unexpected weight grid")
    require(eps_range["incumbent_flat_feasible"].astype(bool).all(),
            "an incumbent-flat-feasibility gate failed")
    require((eps_range["U_common"] == eps_range["U_min_A1"]).all(),
            "common service is not the A1 minimum for every disruption")
    require((eps_range["U_min_A2"] <= eps_range["U_min_A1"]).all(),
            "A2 minimum service loss exceeds A1")
    require((eps_range["placeable_A1"].astype(bool) == (eps_range["U_common"] == 0)).all(),
            "A1 placeability and the zero-drop common-service flag differ")
    require((eps_range["placeable_A2"].astype(int) >=
             eps_range["placeable_A1"].astype(int)).all(),
            "nested A2 is marked less placeable than A1")
    require(oracle["oracle_pass"].astype(bool).all(), "an R1 oracle row failed")

    disruption_keys = eps_range[KEYS].sort_values(KEYS).reset_index(drop=True)
    raw_keys = raw[KEYS].drop_duplicates().sort_values(KEYS).reset_index(drop=True)
    require(disruption_keys.equals(raw_keys), "R1 and R1b disruption keys differ")

    # A method is inert when the complete reported (U, W, D) cost vector is
    # unchanged over all six registered weights.
    inert = (
        raw.groupby(KEYS + ["method"], sort=True)
        .agg(n_w4=("w4", "nunique"), n_U=("U", "nunique"),
             n_W=("overtime", "nunique"), n_D=("deviation", "nunique"))
        .reset_index()
    )
    require((inert["n_w4"] == len(WEIGHTS)).all(), "incomplete R1b weight family")
    inert["fully_inert"] = (
        (inert["n_U"] == 1) & (inert["n_W"] == 1) & (inert["n_D"] == 1)
    )
    placeability = eps_range[KEYS + ["U_common"]].copy()
    placeability["zero_drop_placeable"] = placeability["U_common"] == 0
    placeability = placeability.drop(columns="U_common")
    inert = inert.merge(placeability, on=KEYS, validate="many_to_one")

    # Count distinct efficiency--stability cost vectors actually returned by
    # the registered epsilon trace.  Repeated epsilons/reused anchors collapse.
    frontier = (
        eps.groupby(KEYS + ["method"], sort=True)
        .agg(n_eps=("eps", "size"), n_cost_vectors=("W", "nunique"),
             d_min=("D", "min"), d_max=("D", "max"),
             w_min=("W", "min"), w_max=("W", "max"))
        .reset_index()
    )
    # nunique(W) alone could under-count different D values at the same W.
    exact_cost_counts = (
        eps[KEYS + ["method", "D", "W"]].drop_duplicates()
        .groupby(KEYS + ["method"]).size().rename("n_cost_vectors_exact")
        .reset_index()
    )
    frontier = frontier.drop(columns="n_cost_vectors").merge(
        exact_cost_counts, on=KEYS + ["method"], validate="one_to_one"
    ).rename(columns={"n_cost_vectors_exact": "n_cost_vectors"})
    frontier = frontier.merge(placeability, on=KEYS, validate="many_to_one")
    min_overtime = (
        frontier.pivot(index=KEYS, columns="method", values="w_min")
        .reset_index()
    )
    require({"A1", "A2"}.issubset(min_overtime.columns),
            "missing method in minimum-overtime pivot")
    min_overtime["a2_strict_overtime_improvement"] = (
        min_overtime["A2"] < min_overtime["A1"] - 1e-9
    )
    min_overtime = min_overtime.merge(placeability, on=KEYS, validate="one_to_one")

    qualified = inert[inert["zero_drop_placeable"]]
    qualified_a1 = qualified[qualified["method"] == "A1"]
    require(len(qualified_a1) == int(eps_range["placeable_A1"].sum()),
            "qualified A1 denominator mismatch")
    require(qualified_a1["fully_inert"].all(),
            "A1 is not inert on every theorem-qualified disruption")
    qualified_frontier_a1 = frontier[
        (frontier["method"] == "A1") & frontier["zero_drop_placeable"]
    ]
    require((qualified_frontier_a1["n_cost_vectors"] == 1).all(),
            "A1 epsilon frontier is not singleton on every qualified disruption")

    # Runtime/certificate boundary checks reported in the paper.
    require((eps["solver_status_code"] == 0).all(), "nonzero R1 solver status")
    require((raw["solver_status_code"] == 0).all(), "nonzero R1b final status")
    require((raw["weighted_solver_status_code"] == 0).all(),
            "nonzero R1b weighted status")
    require((raw["w_refine_solver_status_code"] == 0).all(),
            "nonzero R1b W-refinement status")
    require((raw["tie_break_solver_status_code"] == 0).all(),
            "nonzero R1b tie-break status")
    require(not raw["weighted_lattice_proves_exact_optimality"].astype(bool).all(),
            "unexpected blanket exact-optimality flag")

    by_rho = []
    for rho in RHOS:
        er = eps_range[np.isclose(eps_range["rho"], rho)]
        ir = inert[np.isclose(inert["rho"], rho)]
        fr = frontier[np.isclose(frontier["rho"], rho)]
        row = {
            "rho": rho,
            "scenarios": int(er["seed"].nunique()),
            "disruptions": int(len(er)),
            "zero_drop": int(er["placeable_A1"].sum()),
            "a1_inert_all": int(ir[(ir["method"] == "A1") & ir["fully_inert"]].shape[0]),
            "a1_inert_qualified": int(ir[(ir["method"] == "A1") & ir["zero_drop_placeable"] & ir["fully_inert"]].shape[0]),
            "a2_responsive_all": int(ir[(ir["method"] == "A2") & ~ir["fully_inert"]].shape[0]),
            "a2_responsive_qualified": int(ir[(ir["method"] == "A2") & ir["zero_drop_placeable"] & ~ir["fully_inert"]].shape[0]),
            "a2_multicost_eps_all": int(fr[(fr["method"] == "A2") & (fr["n_cost_vectors"] > 1)].shape[0]),
            "a2_multicost_eps_qualified": int(fr[(fr["method"] == "A2") & fr["zero_drop_placeable"] & (fr["n_cost_vectors"] > 1)].shape[0]),
            "a2_strict_overtime_all": int(min_overtime[
                np.isclose(min_overtime["rho"], rho) &
                min_overtime["a2_strict_overtime_improvement"]
            ].shape[0]),
            "a2_strict_overtime_qualified": int(min_overtime[
                np.isclose(min_overtime["rho"], rho) &
                min_overtime["zero_drop_placeable"] &
                min_overtime["a2_strict_overtime_improvement"]
            ].shape[0]),
        }
        by_rho.append(row)

    require(
        (eps_range["placeable_A1"].astype(bool) == (eps_range["U_common"] == 0)).all(),
        "A1 placeability tag and solver-reported zero-drop benchmark disagree",
    )
    total = {
        "scenarios": int(eps_range[["rho", "seed"]].drop_duplicates().shape[0]),
        "disruptions": int(len(eps_range)),
        "incumbent_flat_feasible": int(eps_range["incumbent_flat_feasible"].sum()),
        "zero_drop": int(eps_range["placeable_A1"].sum()),
        "positive_u_reported": int((eps_range["U_common"] > 0).sum()),
        "a1_inert_all": int(inert[(inert["method"] == "A1") & inert["fully_inert"]].shape[0]),
        "a1_inert_qualified": int(qualified_a1["fully_inert"].sum()),
        "a1_singleton_eps_qualified": int((qualified_frontier_a1["n_cost_vectors"] == 1).sum()),
        "a2_responsive_all": int(inert[(inert["method"] == "A2") & ~inert["fully_inert"]].shape[0]),
        "a2_responsive_qualified": int(inert[(inert["method"] == "A2") & inert["zero_drop_placeable"] & ~inert["fully_inert"]].shape[0]),
        "a2_multicost_eps_all": int(frontier[(frontier["method"] == "A2") & (frontier["n_cost_vectors"] > 1)].shape[0]),
        "a2_multicost_eps_qualified": int(frontier[(frontier["method"] == "A2") & frontier["zero_drop_placeable"] & (frontier["n_cost_vectors"] > 1)].shape[0]),
        "a2_strict_overtime_all": int(min_overtime["a2_strict_overtime_improvement"].sum()),
        "a2_strict_overtime_qualified": int(min_overtime[
            min_overtime["zero_drop_placeable"] &
            min_overtime["a2_strict_overtime_improvement"]
        ].shape[0]),
    }

    # Pull the exact R5 summaries used in the table/plot.
    cost_rows = r5_by_rho[r5_by_rho["estimand"] == "A2_cost_of_stability"].copy()
    require(len(cost_rows) == len(RHOS), "missing R5 cost-of-stability rows")
    cost_by_rho = []
    for rho in RHOS:
        r = cost_rows[np.isclose(cost_rows["rho"], rho)].iloc[0]
        cost_by_rho.append({
            "rho": rho,
            "n_scenarios": int(r["n_scenarios"]),
            "n_disruptions": int(r["n_disruptions_total"]),
            "n_zero_drop": int(r["n_zero_drop_disruptions_total"]),
            "median": float(r["median"]),
            "q1": float(r["q1"]),
            "q3": float(r["q3"]),
            "minimum": float(r["minimum"]),
            "maximum": float(r["maximum"]),
        })

    boot = []
    for _, r in bootstrap.iterrows():
        boot.append({
            "estimand": str(r["estimand"]),
            "w4": None if pd.isna(r["w4"]) else int(r["w4"]),
            "w4_lo": None if pd.isna(r["w4_lo"]) else int(r["w4_lo"]),
            "w4_hi": None if pd.isna(r["w4_hi"]) else int(r["w4_hi"]),
            "point": float(r["point_estimate"]),
            "ci_lo": float(r["ci_lo"]),
            "ci_hi": float(r["ci_hi"]),
            "unit": str(r["unit"]),
            "interpretation": str(r["interpretation"]),
        })

    # Preserve the upstream six-row summary as an independent equality check.
    upstream = e4_summary.sort_values("rho").reset_index(drop=True)
    for row in by_rho:
        r = upstream[np.isclose(upstream["rho"], row["rho"])].iloc[0]
        a1_pct = 100.0 * row["a1_inert_all"] / row["disruptions"]
        a2_inert_pct = 100.0 * (row["disruptions"] - row["a2_responsive_all"]) / row["disruptions"]
        require(abs(a1_pct - float(r["A1_fully_inert_pct"])) < 1e-8,
                f"A1 inert summary mismatch at rho={row['rho']}")
        require(abs(a2_inert_pct - float(r["A2_fully_inert_pct"])) < 1e-8,
                f"A2 inert summary mismatch at rho={row['rho']}")

    return {
        "schema": "paper_formal_results_summary_v2",
        "snap_id": SNAP,
        "input_sha256": hashes,
        "expected_rows": EXPECTED_ROWS,
        "totals": total,
        "by_rho": by_rho,
        "cost_of_stability_by_rho": cost_by_rho,
        "bootstrap": boot,
        "interpretive_boundary": {
            "experimental_unit": "derived scenario/generation seed",
            "disruptions_are_nested": True,
            "pooled_bootstrap": "equal-rho-weighted descriptive sensitivity",
            "population_inference": False,
            "causal_rho_effect": False,
            "solver_status0_lattice_consistency_is_exact_optimality_proof": False,
        },
    }


def write_tables(summary: dict, outdir: Path) -> None:
    rows = []
    for r in summary["by_rho"]:
        rows.append(
            f"{r['rho']:.2f} & {r['disruptions']} & {r['zero_drop']} & "
            f"{r['a1_inert_qualified']}/{r['zero_drop']} & "
            f"{r['a2_responsive_qualified']}/{r['zero_drop']} & "
            f"{r['a2_strict_overtime_qualified']}/{r['zero_drop']} \\\\"
        )
    totals = summary["totals"]
    rows.append(
        f"All & {totals['disruptions']} & {totals['zero_drop']} & "
        f"{totals['a1_inert_qualified']}/{totals['zero_drop']} & "
        f"{totals['a2_responsive_qualified']}/{totals['zero_drop']} & "
        f"{totals['a2_strict_overtime_qualified']}/{totals['zero_drop']} \\\\"
    )
    (outdir / "table_structural_by_rho_rows.tex").write_text(
        "\n".join(rows) + "\n\\bottomrule\n"
    )

    cost_rows = []
    for r in summary["cost_of_stability_by_rho"]:
        cost_rows.append(
            f"{r['rho']:.2f} & {r['n_zero_drop']}/{r['n_disruptions']} & "
            f"{r['median']:.1f} & [{r['q1']:.1f}, {r['q3']:.1f}] & "
            f"[{r['minimum']:.1f}, {r['maximum']:.1f}] \\\\"
        )
    (outdir / "table_cost_stability_rows.tex").write_text(
        "\n".join(cost_rows) + "\n\\bottomrule\n"
    )

    macros = {
        "FormalScenarios": totals["scenarios"],
        "FormalDisruptions": totals["disruptions"],
        "FormalZeroDrop": totals["zero_drop"],
        "FormalPositiveUReported": totals["positive_u_reported"],
        "FormalAOneQualifiedInert": totals["a1_inert_qualified"],
        "FormalATwoResponsiveQualified": totals["a2_responsive_qualified"],
        "FormalATwoMultiEpsQualified": totals["a2_multicost_eps_qualified"],
        "FormalATwoStrictOvertimeQualified": totals["a2_strict_overtime_qualified"],
    }
    macro_lines = [f"\\newcommand{{\\{name}}}{{{latex_escape(value)}}}" for name, value in macros.items()]
    (outdir / "formal_result_macros.tex").write_text("\n".join(macro_lines) + "\n")


def plot_regime(summary: dict, figure_dir: Path) -> None:
    by_rho = summary["by_rho"]
    x = np.arange(len(by_rho))
    labels = [f"{r['rho']:.2f}" for r in by_rho]
    zero = np.array([100 * r["zero_drop"] / r["disruptions"] for r in by_rho])
    a1 = np.array([100 * r["a1_inert_qualified"] / r["zero_drop"] for r in by_rho])
    a2 = np.array([100 * r["a2_responsive_qualified"] / r["zero_drop"] for r in by_rho])

    fig, ax = plt.subplots(figsize=(7.15, 3.55))
    width = 0.25
    ax.bar(x - width, zero, width, label="Zero-drop qualified", color="#8DA0CB")
    ax.bar(x, a1, width, label="A1 inert (qualified)", color="#66C2A5")
    ax.bar(x + width, a2, width, label="A2 weight-responsive (qualified)", color="#FC8D62")
    ax.set_xticks(x, labels)
    ax.set_xlabel(r"Target cap ratio $\rho_{target}$")
    ax.set_ylabel("Share of disruptions (%)")
    ax.set_ylim(0, 108)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              fontsize=8, frameon=False)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(figure_dir / f"fig2_scope_responsiveness.{suffix}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(summary: dict, figure_dir: Path) -> None:
    boot = pd.DataFrame(summary["bootstrap"])
    dev = boot[boot["estimand"] == "delta_deviation_A2_minus_A1"].sort_values("w4")
    overtime = boot[boot["estimand"] == "delta_overtime_A2_minus_A1"].sort_values("w4")
    require(list(dev["w4"]) == WEIGHTS, "missing deviation bootstrap weights")
    require(list(overtime["w4"]) == WEIGHTS, "missing overtime bootstrap weights")

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.05), sharex=True)
    panels = [
        (axes[0], dev, "A2 - A1 deviation", "Assignment changes", "#D95F02"),
        (axes[1], overtime, "A2 - A1 overtime", "Minutes", "#1B9E77"),
    ]
    for ax, frame, title, ylabel, color in panels:
        x = frame["w4"].astype(float).to_numpy()
        y = frame["point"].to_numpy()
        lo = frame["ci_lo"].to_numpy()
        hi = frame["ci_hi"].to_numpy()
        ax.fill_between(x, lo, hi, color=color, alpha=0.20, linewidth=0)
        ax.plot(x, y, marker="o", color=color, linewidth=1.8, markersize=4)
        ax.axhline(0, color="#555555", linewidth=0.7)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(r"Deviation weight $w_4$")
        ax.set_ylabel(ylabel)
        ax.grid(color="#E2E2E2", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.set_xticks(WEIGHTS)
    fig.tight_layout(w_pad=1.25)
    for suffix in ("pdf", "png"):
        fig.savefig(figure_dir / f"fig3_weight_tradeoff.{suffix}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("generated"), type=Path)
    parser.add_argument("--figure-dir", default=Path("figures"), type=Path)
    args = parser.parse_args()

    frames, hashes = read_inputs(args.results_root)
    summary = summarize(frames, hashes)
    seal_hashes = {}
    attempt_root = args.results_root.parent
    for relpath, expected_hash in EXPECTED_SEAL_SHA256.items():
        path = attempt_root / relpath
        require(path.is_file(), f"missing formal seal artifact: {path}")
        observed_hash = sha256(path)
        require(observed_hash == expected_hash,
                f"formal seal SHA-256 mismatch for {relpath}: {observed_hash}")
        seal_hashes[relpath] = observed_hash
    summary["formal_seal_sha256"] = seal_hashes
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "formal_results_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_tables(summary, args.output_dir)
    plot_regime(summary, args.figure_dir)
    plot_tradeoff(summary, args.figure_dir)
    print(json.dumps(summary["totals"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
