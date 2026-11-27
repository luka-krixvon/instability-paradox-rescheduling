"""Build deterministic R5 fixture inputs; never writes production artifacts."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Dict, List


RHOS = [0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
SEEDS = [1, 2, 3, 4, 5]
WEIGHTS = [0, 5, 10, 20, 50, 100]
METHODS = ["A1", "A2"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: List[Dict[str, object]], fields: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build(root: Path) -> Dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    raw_rows = []
    task = 1000
    for rho_index, rho in enumerate(RHOS):
        for seed in SEEDS:
            # Vary the eligible count so equal-scenario weighting is observable.
            n_active = 1 + ((rho_index + seed) % 3)
            solution_name = f"S_rho{int(round(rho * 100)):03d}_fixture_s{seed}.csv"
            solution_path = root / solution_name
            solution_rows = [{
                "NurseID": 99, "Day": 1, "Shift": 3,
                "PatientID": 999, "TaskIDs": f"{task};", "FlatLoad": 1.0,
            }]
            task += 1
            active = []
            for offset in range(n_active):
                nurse = seed * 10 + offset
                active.append(nurse)
                solution_rows.append({
                    "NurseID": nurse, "Day": 2, "Shift": 1,
                    "PatientID": task, "TaskIDs": f"{task};", "FlatLoad": 30.0,
                })
                task += 1
            _write_csv(solution_path, solution_rows,
                       ["NurseID", "Day", "Shift", "PatientID", "TaskIDs", "FlatLoad"])
            manifest_rows.append({
                "rho_target": rho, "seed": seed, "sol": solution_name,
                "s_sha256": sha256(solution_path),
            })

            for absent_index, absent in enumerate(active):
                u_min_a1 = int((rho_index + seed + absent_index) % 4 == 0)
                u_min_a2 = 0
                u_common = max(u_min_a1, u_min_a2)
                base = rho_index * 7 + seed * 2 + absent_index
                for method in METHODS:
                    for weight in WEIGHTS:
                        a1_overtime = 90.0 + base + 0.25 * weight
                        overtime = (a1_overtime if method == "A1"
                                    else a1_overtime - 4.0 + 0.03 * weight)
                        a1_deviation = 12.0 + absent_index
                        deviation = (a1_deviation if method == "A1" else
                                     max(0.0, a1_deviation - 2.0 - weight / 50.0))
                        raw_rows.append({
                            "rho": rho,
                            "seed": seed,
                            "absent": absent,
                            "method": method,
                            "w4": weight,
                            "deviation": deviation,
                            "overtime": overtime,
                            "U_min_A1": u_min_a1,
                            "U_min_A2": u_min_a2,
                            "U_common": u_common,
                            "U": u_common,
                            "unserved": u_common,
                            "eligible_disruption": True,
                            "eligible_disruptions_in_scenario": n_active,
                            "absence_rule": "all_active_future_task_holders",
                            "absence_cap_mode": "all_active_no_cap",
                            "weighted_solver_status_code": 0,
                            "weighted_proven_optimal": True,
                            "tie_break_solver_status_code": 0,
                            "tie_break_proven_optimal": True,
                            "umin_A1_solver_status_code": 0,
                            "umin_A1_proven_optimal": True,
                            "umin_A2_solver_status_code": 0,
                            "umin_A2_proven_optimal": True,
                            "solver_status_code": 0,
                            "proven_optimal": True,
                            "solver_status_chain": "0|0|0",
                            "u_min_solver_status_chain": "A1:0|A2:0",
                            "weighted_group_status_codes": "0",
                            "tie_break_group_status_codes": "0",
                            "solver_group_status_codes": "0",
                            "solver_group_complete_status_chains_json":
                                '[{"group_index":1,"stage_names":["lex2","lex3","lex4"],"status_codes":[0,0,0]}]',
                            "snap_id": "",
                        })

    manifest = root / "rho_manifest.csv"
    raw = root / "e4_raw.csv"
    _write_csv(manifest, manifest_rows, ["rho_target", "seed", "sol", "s_sha256"])
    _write_csv(raw, raw_rows, [
        "rho", "seed", "absent", "method", "w4", "deviation", "overtime",
        "U_min_A1", "U_min_A2", "U_common", "U", "unserved",
        "eligible_disruption", "eligible_disruptions_in_scenario",
        "absence_rule", "absence_cap_mode",
        "weighted_solver_status_code", "weighted_proven_optimal",
        "tie_break_solver_status_code", "tie_break_proven_optimal",
        "umin_A1_solver_status_code", "umin_A1_proven_optimal",
        "umin_A2_solver_status_code", "umin_A2_proven_optimal",
        "solver_status_code", "proven_optimal", "snap_id",
        "solver_status_chain", "u_min_solver_status_chain",
        "weighted_group_status_codes", "tie_break_group_status_codes",
        "solver_group_status_codes", "solver_group_complete_status_chains_json",
    ])
    return {"root": root, "manifest": manifest, "raw": raw}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    built = build(args.output)
    print(built["manifest"])
    print(built["raw"])
