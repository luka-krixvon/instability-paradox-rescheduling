# recovery_06 formal-run deposit

This directory lets you **recompute** every number, table and figure in Section VI of

> *Auditing Stability Controls in Dynamic Workforce Reassignment: Assignment-Only versus
> Displacement Recovery in Home Healthcare* — Teng-Ruei Chen, Chun-Cheng Lin.
> XSci preprint, DOI [10.66977/xsci.2611.0042](https://doi.org/10.66977/xsci.2611.0042)

rather than check a digest against a file you cannot obtain.

Run identifier (`snap_id`): `d6c9fa4de2c8ca88bd2fa260f8975398ac42ed9ef14c0527213d0865f2980c60`

---

## What this is, and what it is not

**It is** the eight stage outputs that every reported number is computed from, the two seal
artifacts, the run manifest, and the script that turns them into the paper's tables,
figures and inline numbers. Recomputation is self-contained: no solver, no MILP, no base
instance required.

**If you want to re-solve the MILPs rather than recompute from their results**, that is
[`pipeline/`](pipeline/README.md): the model, the 30 derived scenarios, the pre-registered
design manifests and the containerised runner, which reproduces stages R1, R1b and R5 from
scratch. Read its README first — it states precisely which gate is relaxed there, and why.

**It is not** a re-derivation of the science. Recomputing these artifacts shows the reported
numbers follow from the recorded stage outputs. It does not independently verify the MILP
solutions those stage outputs contain, and it does not validate the modelling choices. The
script's internal gates (`snap_id` equality, `oracle_pass`, `solver_status == 0`) are
consistency checks on sealed outputs, not optimality proofs.

**It does not contain the base instance.** The base instance derives from a co-author's earlier
research; authorisation for its use in this study and in journal submission is not in dispute,
but no data-specific public-redistribution licence has been established for it, so it is not
redistributed here. Stage R0 — which generates the 30 derived scenarios from that instance —
is therefore not reproducible from either this deposit or `pipeline/`. Everything from R1
onward is. `pipeline/` ships the 30 scenarios pre-generated and verifies each against the
digest recorded for it, which is weaker than regenerating them and is the honest limit of
what can be released.

---

## Contents

| Path | Bytes | What it is |
|---|---:|---|
| `code/build_formal_results.py` | 21,878 | the entry point; turns the stage outputs into the paper's tables, figures and `\Formal*` macros |
| `code/figstyle.py` | 9,983 | the figure style it imports; must sit beside it. Needed only for the two figures |
| `run/SHA256SUMS_RECOVERY_06.txt` | 20,075 | run-wide ledger (184 lines); contains all eight digests below |
| `run/receipts/RECOVERY_06_SEAL_REPORT.json` | 26,511 | seal report for the run; `verdict: PASS` |
| `run/results/R1/eps_range.csv` | 144,726 | per-disruption inventory (ρ, seed, absent worker, `U_common`, placeability) |
| `run/results/R1/eps_frontier.csv` | 25,216,336 | the full ε-constraint trace, with solver status columns |
| `run/results/R1/eps_oracle_validation.csv` | 16,985,643 | independent-oracle recheck of every ε row |
| `run/results/R1b/e4_raw.csv` | 21,495,622 | the w₄ × method × disruption cost family, with four solver-status columns |
| `run/results/R1b/e4_summary.csv` | 1,445 | compact per-w₄ summary |
| `run/results/R5/r5_scenario_effects.csv` | 66,190 | per-scenario effects (the bootstrap's input) |
| `run/results/R5/r5_by_rho.csv` | 18,729 | per-ρ stratum summaries |
| `run/results/R5/r5_bootstrap.csv` | 4,911 | bootstrap point estimates and intervals |
| `run/results/R5/r5_run_manifest.json` | 8,606 | R5 design manifest (seed, replicates, quantile method) |
| `run/results/R5/R5.log` | 1,018 | R5 stage log |
| `run/results/environment/environment.log` | 2,050 | the library versions the sealed run used |

61.0 MiB raw; roughly 1.3 MiB compressed. There is no size argument for withholding any of it.

### Which file produces which reported number

| Reported in the paper | Comes from |
|---|---|
| 30 scenarios, 518 disruptions, 488 zero-drop, 30 positive-U | `eps_range.csv` |
| 488 A1 singleton ε-traces; 224 A2 multi-cost; 224 A2 strict-overtime | `eps_frontier.csv` |
| 488 A1 inert; 206 A2 weight-responsive | `e4_raw.csv` |
| Table I (`tab:structural`) | `eps_range.csv` + `eps_frontier.csv` + `e4_raw.csv` |
| Table II (`tab:coststability`) | `r5_by_rho.csv` + `r5_bootstrap.csv` |
| 162.12 min [151.50, 171.10] | `r5_bootstrap.csv` |
| Fig. 2, Fig. 3 | `eps_range.csv`, `e4_raw.csv`, `r5_by_rho.csv` |
| `eps_oracle_validation.csv` | prints nothing — it is a hard gate; the script aborts if any row failed |

---

## Recomputing

The script resolves the two seal artifacts at `--results-root/..`, so `run/` must stay the
parent of `results/`. This layout is not optional.

```sh
cd deposit
python3 -m venv .venv && . .venv/bin/activate
pip install numpy==2.0.2 pandas==2.3.3 scipy==1.13.1 matplotlib==3.9.4
```

**1 — check the eight CSVs against the run-wide ledger** (expect 8 lines, all `OK`):

```sh
cd run && grep -E 'results/(R1|R1b|R5)/(eps_range|eps_frontier|eps_oracle_validation|e4_raw|e4_summary|r5_scenario_effects|r5_by_rho|r5_bootstrap)\.csv$' SHA256SUMS_RECOVERY_06.txt | shasum -a 256 -c && cd ..
```

**2 — check the two seal artifacts** against the digests hard-coded at lines 49–50 of the script:

```sh
shasum -a 256 run/SHA256SUMS_RECOVERY_06.txt run/receipts/RECOVERY_06_SEAL_REPORT.json
```

Expect `182e4c9e889df1de8d9f3cba9d2864c25c528542c3da67be8962de2759b9d784` and
`aba32eaa93ac6320ca858bcbcb0d46a236275af39fd39ffc41cf990c825e8005`.

**3 — recompute every Section VI artifact:**

```sh
mkdir -p out/generated out/figures
python3 code/build_formal_results.py --results-root run/results --output-dir out/generated --figure-dir out/figures
```

**4 — the script prints a totals block** as key-sorted JSON. It must read exactly:

```json
{
  "a1_inert_all": 498,
  "a1_inert_qualified": 488,
  "a1_singleton_eps_qualified": 488,
  "a2_multicost_eps_all": 253,
  "a2_multicost_eps_qualified": 224,
  "a2_responsive_all": 235,
  "a2_responsive_qualified": 206,
  "a2_strict_overtime_all": 253,
  "a2_strict_overtime_qualified": 224,
  "disruptions": 518,
  "incumbent_flat_feasible": 518,
  "positive_u_reported": 30,
  "scenarios": 30,
  "zero_drop": 488
}
```

**5 — compare against the published artifacts.** `out/generated/` holds
`formal_results_summary.json`, `table_structural_by_rho_rows.tex` (Table I),
`table_cost_stability_rows.tex` (Table II) and `formal_result_macros.tex` (the inline
`\Formal*` numbers). `out/figures/` holds `fig2_scope_responsiveness` and
`fig3_weight_tradeoff` as PDF and PNG.

The script fails closed. If a row count, a digest, a `snap_id`, an oracle flag or a solver
status does not match, it stops rather than emitting a number.

---

## Two notes on the reported numbers

**162.12 vs 161.11 is not an arithmetic error.** `A2_cost_of_stability` is 162.115764514 min.
`delta_overtime_A2_minus_A1` is −783.875711355 at w₄ = 0 and −622.773797019 at w₄ = 100; the
difference of those differences is 161.101914336. The residual 1.013850177 min is **A1's own**
overtime change over the same weight range, concentrated in positive-U scenarios at ρ = 0.75,
0.85 and 0.95, with a largest single-scenario contribution of +10.53 min. Re-aggregating
`r5_scenario_effects.csv` with equal weight per ρ stratum reproduces all three figures.

**The bootstrap intervals are reproduced by rerunning a fixed-seed pipeline, not independently
re-derived.** `r5_run_manifest.json` records the seed (20260730), the generator (PCG64), the
replicate count (10,000) and the shared length-5 seed-index vector per replicate; the quantile
method is fixed in `r5_stats.py` rather than recorded in the manifest. Rerunning reproduces the
intervals; it does not constitute an independent statistical check of them.

---

## Relationship to the rest of this repository

This deposit corresponds to the paper above. Material elsewhere in the repository that predates
it reports an earlier, smaller study and **does not** match the paper's numbers. Where the two
disagree, this deposit is authoritative.
