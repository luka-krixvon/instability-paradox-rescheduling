# Auditing Stability Controls in Dynamic Workforce Reassignment

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Solver](https://img.shields.io/badge/MILP-scipy%E2%80%93HiGHS-orange.svg)
![No API key](https://img.shields.io/badge/API%20key-not%20required-brightgreen.svg)
![CPU only](https://img.shields.io/badge/hardware-CPU%20only-lightgrey.svg)
[![Live demo](https://img.shields.io/badge/demo-live%20on%20GitHub%20Pages-14b8a6.svg)](https://luka-krixvon.github.io/instability-paradox-rescheduling/demo/)

> Companion repository for *Auditing Stability Controls in Dynamic Workforce Reassignment:
> Assignment-Only versus Displacement Recovery in Home Healthcare* — Teng-Ruei Chen,
> Chun-Cheng Lin. XSci preprint, DOI
> [10.66977/xsci.2611.0042](https://doi.org/10.66977/xsci.2611.0042).
> **CPU only** — no GPU, no commercial solver, no LLM, no API key.

**The question.** A scheduler lets you put a penalty on how much the roster changes. When a
caregiver calls in sick, does turning that dial up actually buy a more stable roster? It
depends on what the recovery is allowed to do. If recovery may only place the orphaned
visits and must leave every incumbent assignment alone, the number of changed assignments is
fixed by the event itself and the dial multiplies a constant. If recovery may also move
incumbents, and that genuinely lowers attainable overtime at the same service level, then two
or more nondominated cost vectors exist and the dial does choose between them.

---

## Start here

| | |
|---|---|
| **[`deposit/`](deposit/)** | **Recompute** every number, table and figure in Section VI from the sealed stage outputs. One script, no solver, no base instance needed. See [`deposit/README.md`](deposit/README.md). |
| **[Live demo](https://luka-krixvon.github.io/instability-paradox-rescheduling/demo/)** | **Explore** the same results in the browser: the weight-response contrast, a per-capacity view, and a hand-checkable worked example. Reads [`demo/data/results.json`](demo/data/results.json), which mirrors the deposit. |

Headline figures, all recomputable from `deposit/`: 518 disruptions nested in 30 derived
scenarios from one synthetic base instance; 488 met the zero-drop conditions; **488/488** of
those A1 efficient sets hold exactly one cost vector; **224/488** of the A2 traces hold more
than one; raising the deviation weight from 0 to 100 costs A2 **162.12 minutes** of overtime
(descriptive 95% interval [151.50, 171.10], scenario-clustered).

> **Scope, stated up front.** One synthetic base instance. Disruptions are nested inside
> scenarios, so nothing here is a disruption-level statistic. Slots are fixed and travel is a
> workload proxy — no routing, no time-window propagation. The interval is descriptive, from
> scenario-clustered resampling; it is not a significance test.

---

## Repository layout

```
instability-paradox-rescheduling/
├── deposit/                  # ← the paper's evidence. Sealed stage outputs + the
│   ├── code/                 #   script that regenerates its tables and figures.
│   ├── run/results/{R1,R1b,R5}/
│   └── pipeline/             # ← the runnable model: solver code, the 30 derived
│                             #   scenarios, containerised runner for R1/R1b/R5
├── demo/                     # ← GitHub Pages site, built from deposit/
│   └── data/results.json
├── src/                      # EARLIER STUDY's solvers and scripts — see the note below
├── results/                  # EARLIER STUDY's result tables
├── figures/                  # EARLIER STUDY's figures
├── data/                     # empty (the base instance is not redistributed; see Data)
└── requirements.txt
```

⚠️ `src/`, `results/` and `figures/` belong to an earlier, smaller study and **do not**
reproduce the published paper. In particular `src/displacement_milp.py` is labelled as the
core recovery MILP but is **not** the model used in the paper — running it will produce
numbers that do not match. Use `deposit/` instead.

## Reproducing the published results

Everything needed is in [`deposit/`](deposit/) and needs no solver:

```bash
cd deposit
python3 -m venv .venv && . .venv/bin/activate
pip install numpy==2.0.2 pandas==2.3.3 scipy==1.13.1 matplotlib==3.9.4

# 1 — check the eight stage outputs against the run ledger (expect 8 lines, all OK)
cd run && grep -E 'results/(R1|R1b|R5)/.*\.csv$' SHA256SUMS_RECOVERY_06.txt | shasum -a 256 -c && cd ..

# 2 — recompute the tables, figures and inline numbers
mkdir -p out/generated out/figures
python3 code/build_formal_results.py --results-root run/results \
        --output-dir out/generated --figure-dir out/figures
```

The script fails closed: a wrong row count, digest, solver status or oracle flag stops it
rather than emitting a number. Recomputation shows the reported numbers follow from the
recorded stage outputs; it does not independently verify the MILP solutions they contain.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

**The base instance file is not in this repository, but most of its content is.** That is an
awkward sentence and it is the accurate one, so it goes first.

`HHCP-patient-65-task-256-nurse-59.csv` (65 patients, 256 tasks, 59 caregivers) is a
crowdsourced home-healthcare benchmark extending a co-author's earlier work. Authorisation for
its use in this study and in journal submission is not in dispute, but no data-specific
public-redistribution licence has been established for it, so the file itself is not here.

**The 30 scenarios the study solves are released**, in
[`deposit/pipeline/experiments/journal_p0/`](deposit/pipeline/experiments/journal_p0/): 60
`H_rho*.csv` / `S_rho*.csv` files plus `rho_manifest.csv`, which records a SHA-256 for every
one of them. Those scenarios are not an independent derivative of the base instance. Measured
across all 30 files, matching rows as sets:

| Block of the base instance | Recoverable from the released scenarios |
|---|---|
| Patient table, 65 rows | **100 %**, byte-identical |
| Task attributes (`task`, `required skill`, `time`), 256 rows | **100 %** |
| Caregiver table, 59 rows | **81.4 %** (48 rows, each byte-identical) |
| Task candidate-nurse lists, 5 182 entries | **82.4 %** |
| Candidate-solution block, 5 182 rows | **1.2 %** — the released files carry their own, re-scored |

Each `S` file is a generated schedule and is not copied from anywhere. The data carries no
names, no dates, no free text and no direct identifiers; it is synthetic, and
[LICENSE-DATA.md](LICENSE-DATA.md) sets out the evidence for that and its limits.

So the paper's computational claims can be checked two ways. [`deposit/`](deposit/) recomputes
every reported number from the sealed stage outputs without solving anything;
[`deposit/pipeline/`](deposit/pipeline/README.md) re-solves the MILPs of stages R1, R1b and R5
from the released scenarios. What neither can do is regenerate the scenarios from the base
instance — stage R0 — and `pipeline/README.md` sets out each gate that relaxes.

**Licensing.** The MIT licence covers code written by the author. The 30 scenarios are released
for **verification of this paper's results only**, not as a redistributable dataset — see
[LICENSE-DATA.md](LICENSE-DATA.md) for what is and is not granted, and why. Neither licence
reaches the base instance.

---

## Earlier study (superseded)

Before the study above, this repository held a smaller experiment: 32 disruptions across 6
capacity-aware instances, reporting a 29/32 = 91% inertness rate and a 197 ± 33 min cost of
stability, with a monotone frontier and Wilcoxon/Friedman tests.

**None of those numbers were carried into the published paper**, which reports 518
disruptions, 488 zero-drop, 488 singleton A1 traces and 162.12 min [151.50, 171.10]. The
mechanism the earlier study illustrated — A1 inert to the deviation weight, A2 exposing a
tunable frontier — does hold in the paper; the magnitudes do not. Do not cite them.

Its code, tables and figures remain in `src/`, `results/` and `figures/` for the record.

## What is *not* in this repository

An exploratory LLM-driven operator-selection component was part of an abandoned earlier
direction, not of this paper: on these instances routing carried no exploitable signal at the
disruption scale tested (`n <= 3` changed routes, where the LLM selector matched plain VNS).
Keeping it out also means this repository needs **no API key**.

## Citation

Chen, T.-R., & Lin, C.-C. (2026). *Auditing Stability Controls in Dynamic Workforce
Reassignment: Assignment-Only versus Displacement Recovery in Home Healthcare.*
XSci preprint. DOI [10.66977/xsci.2611.0042](https://doi.org/10.66977/xsci.2611.0042)

See also [`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE). © 2026 Teng-Ruei Chen.
