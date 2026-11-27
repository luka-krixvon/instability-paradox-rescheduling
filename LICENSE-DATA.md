# Terms for the 30 released scenarios

This file covers **only** the 60 scenario files
`deposit/pipeline/experiments/journal_p0/H_rho*.csv` and `S_rho*.csv`, and the provenance
record `rho_manifest.csv` beside them. Code in this repository is covered by the MIT
`LICENSE`; this file grants nothing in addition to it and nothing in place of it.

## What you may do

These files are provided **so that the results of one paper can be checked**:

> *Auditing Stability Controls in Dynamic Workforce Reassignment: Assignment-Only versus
> Displacement Recovery in Home Healthcare* — Teng-Ruei Chen, Chun-Cheng Lin.
> DOI [10.66977/xsci.2611.0042](https://doi.org/10.66977/xsci.2611.0042)

You may download them, run the pipeline in `deposit/pipeline/` on them, inspect them, and
publish what you find — including a finding that our numbers are wrong. That is the point of
releasing them.

## What is not granted, and why

No licence to redistribute these files, to publish them as a dataset, or to use them as input
to a different study is granted here.

The reason is specific, and the next section gives the measurements behind it. These
scenarios are not an independent derivative of the base instance: they **reproduce most of it
verbatim**. The base instance is a crowdsourced home-healthcare benchmark extending a
co-author's earlier work, and no public-redistribution licence has been established for it.
The authors therefore cannot grant rights over its content that they do not themselves
clearly hold, and would rather state the position than attach a permissive licence that
over-claims.

If you want to build on this data rather than check this paper with it, write to the
corresponding author. These terms can be widened later; they cannot be narrowed once widened,
which is why they start here.

## Exactly how much of the base instance these files contain

Measured by comparing all 30 released `H_rho*.csv` against the base instance, matching rows
as sets rather than by position:

| Block of the base instance | Present in the released scenarios |
|---|---|
| Patient table, 65 rows | **100 %** — byte-identical in every one of the 30 files |
| Task attributes (`task`, `required skill`, `time`), 256 rows | **100 %** — identical |
| Caregiver table, 59 rows | **81.4 %** — 48 of 59 rows appear across the 30 files, each byte-identical |
| Task candidate-nurse lists, 5 182 entries | **82.4 %** — 910 entries never appear |
| Candidate-solution block, 5 182 rows | **1.2 %** — the released files carry their own, re-scored for the caregivers each retains (1 400–2 731 rows per file); only 61 base rows reappear |

Each individual `H` file carries the full 65-row patient table, the full 256-row task table
with its candidate lists filtered to the caregivers that file retains, a subset of the
caregiver table whose size varies with the capacity-tightness parameter ρ and the seed
(16 to 27 rows, not a fixed count), and a candidate-solution block of its own, re-scored for
that caregiver subset rather than copied. Each `S_rho*.csv` is the generated committed schedule for
that instance — `NurseID, Day, Shift, PatientID, TaskIDs, FlatLoad` — and is not copied from
anywhere.

So: taking the 30 files together recovers the base instance's patient table and task
attributes completely, four fifths of its caregiver rows, and four fifths of its candidate
lists. Releasing them is, in substance, releasing most of that file. We say so here rather
than let a reader infer that the scenarios are synthetic derivatives of something withheld.

## What the data is

The base instance is **synthetic**. An internal provenance audit dated 2026-08-01 records that
it was produced by a C++ generator seeded from the wall clock, that the seed was recovered
(`1769148617`, matching the generator binary's build timestamp), and that replaying the
sequence reproduced all 65 patients, 59 caregivers and 256 tasks with zero differences. We
report that as the audit's finding, not as evidence we can hand you: the replay script and its
output were never archived, and the audit itself calls the seed "plausible exact". The
published work on this instance family states that no real-world case data was used.

The files carry no names, no dates and no free text. Patients, caregivers and tasks appear
only as integer indices, and the four patient attribute columns (`P1_gender`, `P2_age`,
`P3_Language`, `P4_smoke`) are small integer codes that the benchmark does not resolve to any
person.

Copyright (c) 2026 Teng-Ruei Chen, Chun-Cheng Lin.
