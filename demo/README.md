# Web demo

A zero-dependency, static companion to the paper, deployable as GitHub Pages with no build step.

> **Paper.** Teng-Ruei Chen & Chun-Cheng Lin, *Auditing Stability Controls in Dynamic Workforce
> Reassignment: Assignment-Only versus Displacement Recovery in Home Healthcare*, XSci preprint,
> DOI [10.66977/xsci.2611.0042](https://doi.org/10.66977/xsci.2611.0042).

## The question the demo answers

A scheduler lets you set a penalty on how much the roster changes. When a caregiver calls in sick,
does turning that dial up actually buy a more stable roster?

Under the paper's explicit incumbent-feasibility and orphan-placeability conditions, assignment-only
recovery (**A1**) can only place the orphaned visits, so the number of changed assignments is fixed by
the event: deviation is a **counting invariant**, and the service-maximal efficient set holds exactly
**one cost vector** (assignment vectors may still tie on it). No componentwise nondecreasing
scalarization can reveal a second one. When displacement recovery (**A2**) *strictly* lowers attainable
overtime at the common service level, at least two non-dominated cost vectors exist and a deviation
penalty can select among them.

The practical contribution is a **diagnostic**: before exposing a stability weight, test whether
deviation actually varies over service-equivalent recoveries.

## Pages

| Page | What it shows |
|------|---------------|
| **`index.html`** — audit results | Every figure is read from the sealed stage outputs in [`deposit/run/results/`](../deposit/). The weight-response contrast (A1 flat across the registered `w4` grid, A2 moving), per-stratum structure across the six capacity levels, the distribution of how many cost vectors A2 exposes, four worked disruptions drawn from the sealed epsilon traces, the diagnostic itself, and a file-to-number provenance table. |
| **`mechanism.html`** — worked example | A deliberately tiny home-care instance, enumerated exhaustively in the browser, so the counting can be checked by hand. Two panels differ only in what the recovery is **allowed to move**: A1 (assignment-only) versus A2 (displacement allowed). One instance where both collapse to a single cost vector, one where displacement opens a frontier the weight walks across. |

Both pages are single self-contained documents: inline CSS, inline JS, inline SVG. Nothing is fetched at
runtime — no CDN, no web font, no tracking — so they work offline and from `file://`. Both honour
`prefers-color-scheme` and carry a theme toggle, and both are readable at phone width.

## Numbers on these pages

Headline figures, all from `deposit/run/results/`:

| Quantity | Value | Sealed file |
|---|---|---|
| Derived scenarios / disruptions | 30 / 518 | `R1/eps_range.csv` |
| Qualified (`U_common = 0`) | 488 | `R1/eps_range.csv` |
| Incumbent flat-feasible | 518 of 518 | `R1/eps_range.csv` |
| A1 efficient set is a single cost vector | 488 of 488 qualified | `R1/eps_frontier.csv` |
| A2 exposes more than one cost vector | 224 of 488 qualified (253 overall) | `R1/eps_frontier.csv` |
| A2 strictly lowers attainable overtime | 224 of 488 qualified (253 overall) | `R1/eps_frontier.csv` |
| A1 cost vector unchanged across all six weights | 488 of 488 qualified (498 of 518 overall) | `R1b/e4_raw.csv` |
| A2 weight-responsive | 206 of 488 qualified (235 overall) | `R1b/e4_raw.csv` |
| A2 cost of stability, `w4` 0 → 100 | 162.115764514 min, descriptive 95% interval [151.503817921, 171.103386053] | `R5/r5_bootstrap.csv` |
| A2 − A1 overtime at `w4` 0 and 100 | −783.875711355 and −622.773797019 min | `R5/r5_bootstrap.csv` |

`deposit/code/build_formal_results.py` shows how each one is derived, and
`deposit/README.md` maps file to number. The same figures in machine-readable form are in
[`data/results.json`](data/results.json).

## Scope, stated on the pages as well as here

- One **synthetic base instance**; 30 derived scenarios (six capacity strata × five seeds); 518 disruptions.
- Disruptions are **nested within scenarios**. Every pooled figure is a scenario mean first, then an equal
  weight per stratum — never a disruption-level average.
- Fixed slots; travel enters only as a workload proxy; no routing and no time-window propagation.
- The reported interval is a **descriptive** cluster-bootstrap sensitivity interval over five shared seed
  clusters (10,000 resamples, rng seed 20260730). It is **not** a significance test and not population
  inference; the deposit stores its p-value field as `NOT_COMPUTED` and its own interpretation as
  `descriptive_sensitivity_not_population_inference`.
- The base instance is **not redistributable and is not in this repository**. What is provided is
  `deposit/`: the sealed stage outputs plus the script that derives every reported figure from them.

Not claimed anywhere: a numeric weight threshold or phase boundary; a complexity separation between A1
and A2; any external benchmark validation; any statistical significance. Recomputing these numbers from
the sealed outputs, or matching their digests, is a **consistency check on sealed outputs** — not a proof
that the optimisation behind them is correct.

`mechanism.html` is illustrative only. Its instances are invented to be hand-checkable, its enumerator is
not the solver the paper used, and none of its magnitudes appear in the paper.

## Run

```bash
# from the repo root
python3 -m http.server 8000
# then open http://localhost:8000/demo/
```

Or open `demo/index.html` directly in a browser. To publish: enable GitHub Pages
(Settings → Pages → deploy from `main`, root) and visit `…/demo/`.
