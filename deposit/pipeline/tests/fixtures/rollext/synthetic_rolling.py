"""
tests/fixtures/rollext/synthetic_rolling.py — TINY SYNTHETIC fixture for the rolling-horizon
stream (REQ-7 / REQ-8 regression evidence ONLY).

This builds a duck-typed parser and a hand-written committed plan. It reads no benchmark file,
touches no canonical artifact, and is not a model of anything real. Per `rerun_plan_v2_1.md`
ground rules E1/E2, anything demonstrated with it is a REGRESSION TEST, never a proof, never a
validation, and never evidence about whether a published number is correct.

Shape of the fixture, and why:
  * 12 slots (0..11) so that all five epochs of `EPOCHS = [2,4,6,8,10]` have re-solvable slots and
    slots 0..1 form a non-empty frozen prefix;
  * 8 "holder" caregivers 0..7 that hold committed tasks, plus one SPARE caregiver 8 that holds
    nothing. Because both absence rules draw only from caregivers holding a committed task, the
    spare can never be removed, and because it is in every task's candidate set the recovery MILP
    can never hit an empty candidate structure. That keeps the fixture about the bookkeeping under
    test rather than about degenerate infeasibility;
  * three tasks per slot with flat durations 155.0 / 145.5 / 115.0 (service + TRAVEL_PER_TASK), two
    of them committed to the same caregiver so every slot carries a non-zero committed overtime,
    and one non-integer value so the exact-arithmetic path is exercised.
"""
from __future__ import annotations


class _Task:
    __slots__ = ("duration", "candidate_nurses")

    def __init__(self, duration, candidate_nurses):
        self.duration = duration
        self.candidate_nurses = candidate_nurses


class _Nurse:
    __slots__ = ("slots",)

    def __init__(self, slots):
        self.slots = slots


class FakeParser:
    """Exposes exactly what rolling_horizon.solve_assign reads from a parser."""

    def __init__(self, tasks, nurses):
        self.tasks = tasks
        self.nurses = nurses


#: SERVICE durations; TRAVEL_PER_TASK (15.0) is added by the model, giving flat 155.0/145.5/115.0.
SERVICE_DURATIONS = (140.0, 130.5, 100.0)
N_HOLDERS = 8
N_SLOTS = 12
SPARE = N_HOLDERS                    # caregiver id 8: holds nothing, is in every candidate set


def build(n_slots=N_SLOTS, n_holders=N_HOLDERS, durations=SERVICE_DURATIONS):
    """Return (parser, committed, S). `committed` is {slot: {task: caregiver}}."""
    tasks, committed = {}, {}
    for q in range(n_slots):
        base = (q * 3) % n_holders
        ids = [base, (base + 1) % n_holders, (base + 2) % n_holders,
               (base + 3) % n_holders, (base + 4) % n_holders]
        cand = [
            frozenset({ids[0], ids[1], ids[2], SPARE}),
            frozenset({ids[0], ids[1], ids[3], SPARE}),
            frozenset({ids[1], ids[2], ids[4], SPARE}),
        ]
        assign = [ids[0], ids[0], ids[1]]        # two tasks on one caregiver -> committed overtime
        committed[q] = {}
        for k in range(3):
            tid = q * 3 + k
            tasks[tid] = _Task(durations[k], sorted(cand[k]))
            committed[q][tid] = assign[k]
    nurses = {n: _Nurse(frozenset(range(n_slots))) for n in range(n_holders + 1)}
    return FakeParser(tasks, nurses), committed, set(nurses)


def manifest_like(seeds=(1, 2)):
    """Seeds used by the fixture. Deliberately tiny: two seeds x four configurations = 8 rows."""
    return list(seeds)
