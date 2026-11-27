"""Plot the R4 instance-level and run-level schemas.

The runner now emits one instance table and one run table for each R4 family.  This consumer does
not read the invalidated legacy ``t_exact``/``obj_vns`` wide schema.  Realized ``rho_avail`` and
candidate-degree covariates are retained in the aggregation and printed beside the figure.
"""
import argparse
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle; figstyle.apply()

HERE = os.path.dirname(os.path.abspath(__file__))


def _require_columns(frame, columns, label):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError("{} is not the registered R4 schema; missing {}".format(label, missing))


def _bool_series(series):
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def _require_family_binding(instances, runs, label):
    """Require one SNAP and one design-manifest digest across both family artifacts."""
    for frame, artifact in ((instances, label + " instances"), (runs, label + " runs")):
        _require_columns(
            frame, ["snap_id", "design_manifest_sha256", "confound_control_mode"], artifact)
        if set(frame["confound_control_mode"].dropna()) != {"covariate"}:
            raise ValueError("{} was not emitted under covariate confound control".format(
                artifact))
        for column in ("snap_id", "design_manifest_sha256"):
            if frame[column].isna().any():
                raise ValueError("{} contains an empty {}".format(artifact, column))
            values = set(frame[column].dropna().astype(str).str.strip().str.lower())
            if len(values) != 1:
                raise ValueError("{} must carry exactly one {}".format(artifact, column))
            value = next(iter(values))
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError("{} carries a malformed {}".format(artifact, column))
    if set(instances["snap_id"].astype(str).str.strip().str.lower()) != set(
            runs["snap_id"].astype(str).str.strip().str.lower()):
        raise ValueError("{} instance/run SNAP identifiers disagree".format(label))
    if set(instances["design_manifest_sha256"].astype(str).str.strip().str.lower()) != set(
            runs["design_manifest_sha256"].astype(str).str.strip().str.lower()):
        raise ValueError("{} instance/run design-manifest hashes disagree".format(label))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=HERE,
                        help="directory containing the four R4 CSV artifacts")
    parser.add_argument("--outdir", default=None,
                        help="figure output directory (default: --input-dir)")
    args = parser.parse_args(argv)
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.outdir or input_dir)
    os.makedirs(output_dir, exist_ok=True)

    A = pd.read_csv(os.path.join(input_dir, "scalability_size.csv"))
    AR = pd.read_csv(os.path.join(input_dir, "scalability_size_runs.csv"))
    B = pd.read_csv(os.path.join(input_dir, "scalability_tightness.csv"))
    BR = pd.read_csv(os.path.join(input_dir, "scalability_tightness_runs.csv"))
    covariates = ["rho_avail", "cand_degree_min", "cand_degree_mean", "cand_degree_max"]
    _require_columns(
        A, ["stage", "T_q", "seed", "confound_control_mode", "orphans",
            "snap_id", "design_manifest_sha256"] + covariates,
        "scalability_size.csv")
    _require_columns(
        AR, ["stage", "T_q", "seed", "algorithm", "repeat_index", "wall_time_s",
             "gap_signed_pct", "solver_status_code", "proven_optimal",
             "incumbent_feasible_independent", "snap_id",
             "design_manifest_sha256", "confound_control_mode"] + covariates,
        "scalability_size_runs.csv")
    _require_columns(
        B, ["stage", "T_q", "rho_commit_target", "seed", "confound_control_mode"]
           + ["snap_id", "design_manifest_sha256"] + covariates,
        "scalability_tightness.csv")
    _require_columns(
        BR, ["stage", "T_q", "rho_commit_target", "seed", "algorithm", "repeat_index",
             "wall_time_s", "time_limit_s", "solver_status_code", "proven_optimal",
             "stop_reason", "gap_full_objective", "solver_message_raw",
             "incumbent_feasible_independent", "snap_id",
             "design_manifest_sha256", "confound_control_mode"] + covariates,
        "scalability_tightness_runs.csv")
    _require_family_binding(A, AR, "R4-size")
    _require_family_binding(B, BR, "R4-tightness")

    size_exact = AR[AR["algorithm"] == "EXACT"]
    if size_exact.empty or not size_exact["solver_status_code"].eq(0).all() \
            or not _bool_series(size_exact["proven_optimal"]).all() \
            or not _bool_series(size_exact["incumbent_feasible_independent"]).all():
        raise ValueError(
            "R4-size plots require code-0, proven, independently feasible EXACT rows only")
    heuristic = AR[AR["algorithm"].isin(["VNS", "ALNS", "cold"])]
    if heuristic["gap_signed_pct"].isna().any():
        raise ValueError("R4-size heuristic gap is missing despite a proven exact reference")

    tight_exact = BR[BR["algorithm"] == "EXACT"].copy()
    if len(tight_exact) != len(B):
        raise ValueError(
            "tightness run/instance cardinality differs; expected one EXACT run per cell")
    if not tight_exact["solver_status_code"].isin([0, 1]).all() \
            or not _bool_series(tight_exact["incumbent_feasible_independent"]).all():
        raise ValueError("R4-tightness contains an inadmissible or independently infeasible row")
    code0 = tight_exact["solver_status_code"].eq(0)
    code1 = tight_exact["solver_status_code"].eq(1)
    proven = _bool_series(tight_exact["proven_optimal"])
    if not proven[code0].all() or proven[code1].any():
        raise ValueError("R4-tightness optimality labels disagree with status codes 0/1")
    if not tight_exact.loc[code1, "stop_reason"].eq("time_limit").all() \
            or not tight_exact.loc[code1, "solver_message_raw"].astype(
                str).str.lower().str.contains("time limit", regex=False).all():
        raise ValueError(
            "R4-tightness code-1 row lacks an explicit raw time-limit message/reason")

    runtime_a = (AR.groupby(["T_q", "algorithm"])["wall_time_s"].median()
                   .unstack("algorithm"))
    gap_a = (AR[AR["algorithm"].isin(["VNS", "ALNS", "cold"])]
             .groupby(["T_q", "algorithm"])["gap_signed_pct"].mean()
             .unstack("algorithm"))
    inst_a = (A.groupby("T_q")
              .agg(rho_avail=("rho_avail", "mean"),
                   rho_avail_min=("rho_avail", "min"),
                   rho_avail_max=("rho_avail", "max"),
                   cand_degree_min=("cand_degree_min", "mean"),
                   cand_degree_mean=("cand_degree_mean", "mean"),
                   cand_degree_max=("cand_degree_max", "mean"),
                   orphans=("orphans", "mean")))
    aggA = inst_a.join(runtime_a.add_prefix("t_")).join(gap_a.add_prefix("gap_")).reset_index()
    required_algorithms = ["t_EXACT", "t_VNS", "t_ALNS", "t_cold",
                           "gap_VNS", "gap_ALNS", "gap_cold"]
    _require_columns(aggA, required_algorithms, "aggregated R4-size run table")

    tight_keys = ["stage", "T_q", "rho_commit_target", "seed"]
    joined_b = B.merge(
        tight_exact[tight_keys + ["wall_time_s", "time_limit_s", "solver_status_code",
                                  "proven_optimal", "stop_reason", "gap_full_objective"]],
        on=tight_keys, how="inner", validate="one_to_one")
    if len(joined_b) != len(B):
        raise ValueError("tightness run/instance key sets disagree")
    joined_b["not_proven"] = ~_bool_series(joined_b["proven_optimal"])
    joined_b["time_limit_hit"] = joined_b["stop_reason"].eq("time_limit")
    limits = joined_b["time_limit_s"].dropna().unique()
    if len(limits) != 1:
        raise ValueError("tightness rows do not share one manifest-declared time limit")
    TL = float(limits[0])
    aggB = (joined_b.groupby("rho_commit_target")
            .agg(rho_avail=("rho_avail", "mean"),
                 cand_degree_mean=("cand_degree_mean", "mean"),
                 t_exact=("wall_time_s", "median"),
                 t_exact_max=("wall_time_s", "max"),
                 gap_full=("gap_full_objective", "max"),
                 time_limit_hit=("time_limit_hit", "any"),
                 not_proven=("not_proven", "any"))
            .reset_index().sort_values("rho_avail"))

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(figstyle.TEXT_W, 2.7))

    # (A) runtime vs size
    axA.plot(aggA.T_q, aggA.t_EXACT, "o-", color="#d62728", label="exact MILP (A2)")
    axA.plot(aggA.T_q, aggA.t_VNS, "s-", color="#1f77b4", label="VNS")
    axA.plot(aggA.T_q, aggA.t_ALNS, "^-", color="#ff7f0e", label="ALNS")
    axA.plot(aggA.T_q, aggA.t_cold, "d-", color="#7f7f7f", label="greedy cold-start")
    axA.set_yscale("log"); axA.set_xlabel("tasks in per-slot recovery $T_q$")
    axA.set_ylabel("median solve time (s, log)")
    axA.set_title("(a) runtime vs size")
    axA.legend(fontsize=8); axA.grid(True, which="both", ls=":", alpha=0.4)

    # (B) heuristic gap vs size
    axB.axhline(0, color="#999", lw=0.8)
    axB.plot(aggA.T_q, aggA.gap_VNS, "s-", color="#1f77b4", label="VNS")
    axB.plot(aggA.T_q, aggA.gap_ALNS, "^-", color="#ff7f0e", label="ALNS")
    axB.plot(aggA.T_q, aggA.gap_cold, "d-", color="#7f7f7f", label="greedy cold-start")
    axB.set_xlabel("tasks in per-slot recovery $T_q$")
    axB.set_ylabel("signed gap vs proven exact (%)")
    axB.set_title("(b) heuristic quality vs size")
    axB.legend(fontsize=8); axB.grid(True, ls=":", alpha=0.4)

    # (C) tightness frontier
    axC.plot(aggB.rho_avail, aggB.t_exact, "o-", color="#d62728", label="exact median")
    axC.plot(aggB.rho_avail, aggB.t_exact_max, "o--", color="#d62728", alpha=0.5, label="exact worst")
    axC.axhline(TL, color="#333", ls=":", lw=1, label=f"time limit {TL:.0f}s")
    axC.axvline(1.0, color="#888", ls="--", lw=1)
    # mark points where exact failed to prove optimality (nonzero gap / TL)
    hard = aggB[aggB.not_proven | aggB.time_limit_hit]
    if not hard.empty:
        axC.scatter(hard.rho_avail, hard.t_exact_max, s=90, facecolors="none",
                    edgecolors="black", linewidths=1.4, zorder=6, label="not proven optimal")
    axC.set_xlabel("post-disruption tightness $\\rho_{avail}$"); axC.set_ylabel("exact solve time (s)")
    axC.set_title(f"(c) tractability frontier ($T_q={int(B.T_q.iloc[0])}$)")
    axC.legend(fontsize=7); axC.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    output_path = os.path.join(output_dir, "scalability.pdf")
    fig.savefig(output_path); plt.close(fig)
    print("wrote {}".format(output_path))
    print("[A] size with realized confound covariates:\n",
          aggA.round(3).to_string(index=False))
    print("[B] tightness with realized confound covariates:\n",
          aggB.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
