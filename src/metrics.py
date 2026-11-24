"""
metrics.py — 共用度量模組（Phase-2 目標函數與 KPI 計算）

集中定義論文 Phase-2 的目標與 KPI，供 evaluation.py 與 baseline_vns.py 共用，
確保「本方法 / VNS 基線 / Phase-1」三者在**相同度量**下比較（apples-to-apples）。

度量定義（對齊 LLMDrivenReRouter_R03.py 的常數）：
  - 路線目標 Z_route = W3_ROUTING·total_travel + W2_OVERTIME·overtime
  - overtime       = max(0, route_total_time - H_STD)
  - route_total_time 取該 (NurseID, GlobalSlot) 群組中 AccumulatedTime 的最大值
    （AccumulatedTime 已含返回 depot 的時間，是程式自身的時間帳）
  - total_travel   = route_total_time - total_service
  - GlobalSlot     = (Day-1)*SLOTS_PER_DAY + Shift  （solution_schedule.csv 無 GlobalSlot 時用此換算）

注意：Z_deviation 採 (NurseID, PatientID, GlobalSlot) 粒度的對稱差，作為論文 task 級 Hamming 偏差的
      合理代理（proxy）。最終論文若需嚴格 task 級，可改用展開後的 TaskIDs。
"""
from __future__ import annotations
import math
import pandas as pd

# ---- 與 R03 一致的常數 ----
H_STD = 240.0          # 標準班次（overtime 門檻），分鐘
H_MAX = 360.0          # 硬上限，分鐘
W3_ROUTING = 1.0       # 每分鐘旅行成本權重 (Phase-2)
W2_OVERTIME = 5.0      # 每分鐘加班成本權重 (Phase-2)
SLOTS_PER_DAY = 3


def ensure_global_slot(df: pd.DataFrame) -> pd.DataFrame:
    """若無 GlobalSlot 欄位（如 solution_schedule.csv），以 (Day-1)*3+Shift 補上。"""
    if "GlobalSlot" not in df.columns:
        df = df.copy()
        df["GlobalSlot"] = (df["Day"].astype(int) - 1) * SLOTS_PER_DAY + df["Shift"].astype(int)
    return df


def route_metrics(group: pd.DataFrame) -> dict:
    """單一 (NurseID, GlobalSlot) 路線的度量。group 為該路線的所有列。"""
    total_service = float(group["ServiceDuration"].sum())
    route_total_time = float(group["AccumulatedTime"].max())
    total_travel = max(0.0, route_total_time - total_service)
    overtime = max(0.0, route_total_time - H_STD)
    z = W3_ROUTING * total_travel + W2_OVERTIME * overtime
    return {
        "n_stops": int(len(group)),
        "total_service": total_service,
        "total_travel": total_travel,
        "route_total_time": route_total_time,
        "overtime": overtime,
        "feasible": route_total_time <= H_MAX + 1e-6,
        "z": z,
    }


def schedule_kpis(df: pd.DataFrame, t_now: int | None = None) -> dict:
    """整張排班表的彙總 KPI。t_now 給定時只計未來班次 (GlobalSlot >= t_now)。"""
    df = ensure_global_slot(df)
    if t_now is not None:
        df = df[df["GlobalSlot"] >= t_now]
    if len(df) == 0:
        return dict(n_routes=0, n_stops=0, total_travel=0.0, total_overtime=0.0,
                    total_service=0.0, z_total=0.0, n_infeasible=0, max_route_time=0.0)
    z_total = total_travel = total_overtime = total_service = 0.0
    n_routes = n_stops = n_infeasible = 0
    max_route_time = 0.0
    for _, g in df.groupby(["NurseID", "GlobalSlot"]):
        m = route_metrics(g)
        z_total += m["z"]
        total_travel += m["total_travel"]
        total_overtime += m["overtime"]
        total_service += m["total_service"]
        n_routes += 1
        n_stops += m["n_stops"]
        n_infeasible += 0 if m["feasible"] else 1
        max_route_time = max(max_route_time, m["route_total_time"])
    return dict(
        n_routes=n_routes, n_stops=n_stops,
        total_travel=round(total_travel, 3),
        total_overtime=round(total_overtime, 3),
        total_service=round(total_service, 3),
        z_total=round(z_total, 3),
        n_infeasible=n_infeasible,
        max_route_time=round(max_route_time, 3),
    )


def _assignment_set(df: pd.DataFrame, t_now: int) -> set:
    """未來班次的 (NurseID, PatientID, GlobalSlot) 指派集合。"""
    df = ensure_global_slot(df)
    fut = df[df["GlobalSlot"] >= t_now]
    return set(zip(fut["NurseID"].astype(int), fut["PatientID"].astype(int),
                   fut["GlobalSlot"].astype(int)))


def z_deviation(result_df: pd.DataFrame, static_df: pd.DataFrame, t_now: int) -> dict:
    """相對靜態基準 Π_static 的指派偏差（未來班次的對稱差，patient 級代理）。"""
    a = _assignment_set(result_df, t_now)
    b = _assignment_set(static_df, t_now)
    added = a - b           # 結果新增的指派
    removed = b - a         # 相對靜態被移除的指派
    return dict(
        z_deviation=len(added) + len(removed),
        n_added=len(added),
        n_removed=len(removed),
        n_static_future=len(b),
        n_result_future=len(a),
    )
