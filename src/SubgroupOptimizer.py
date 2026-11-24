"""
SubgroupOptimizer.py — D-CHHRP-MD 動態子群集優化器 (純可行解群集版)
====================================================================
設計原則：
  - 護理師-任務媒合分數直接使用 HHCP 檔案 Candidate solution 欄位預計算值
  - 不在程式碼中自行計算任何媒合分數
  - Candidate solution 的 Score 索引對應所有 (task_id, nurse_id) 候選對的序列
  - 不在候選名單中的護理師 → 硬性排除（技能不符已隱含於候選名單）
  - Time Slot 不可出勤 或 H_MAX 工時超限 → 硬性排除
  - 不執行任何優劣比較或最優選擇
  - 輸出為「可行解群集」：每個 (task_id, slot) → 所有可行護理師清單 + HHCP 分數

事件支援：
  1. NURSE_ABSENCE         — 護理師請假：孤兒任務可行護理師群集
  2. PATIENT_RESCHEDULE    — 病患改期：新時段可行護理師群集 + 舊時段釋放記錄
  3. PATIENT_CANCELLATION  — 病患取消：釋放產能記錄
"""

from __future__ import annotations
import copy
import math
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from DynamicEventGenerator import FeasibilityEngine


# ─────────────────────────── 全域物理常數 ────────────────────────────────────
VELOCITY_KMH   = 20.0                          # 移動速度 km/h
VELOCITY_MPM   = (VELOCITY_KMH * 1000) / 60.0 # 移動速度 m/min ≈ 333.33
H_STD          = 240.0   # 標準班次上限 (分鐘)  → 超過此值計加班成本
H_MAX          = 360.0   # 硬性物理上限 (分鐘)  → 絕對不可違反
SLOTS_PER_DAY  = 3       # 每天班次數
TOTAL_SLOTS    = 15      # 5天 × 3班 = 15 slots


# ─────────────────────────── 資料結構定義 ────────────────────────────────────

@dataclass
class StrictEvent:
    """與 DynamicEventGenerator 完全相容的事件物件"""
    event_type: str
    t_now: int
    description: str
    math_constraint: str
    data: Dict = field(default_factory=dict)


@dataclass
class NurseProfile:
    """護理師靜態屬性"""
    nurse_id:   int
    x:          float
    y:          float
    nurse_class: int
    gender:     int           # 0/1
    age:        int           # 實際年齡
    language:   int           # 0/1
    smoke:      int           # 0/1
    slots:      Set[int]      # 可出勤 global slot 集合
    skills:     Set[int]      # 技能等級集合
    candidate_tasks: Set[int] = field(default_factory=set)  # 任務推薦清單


@dataclass
class PatientProfile:
    """病患靜態屬性"""
    patient_id:  int
    x:           float
    y:           float
    req_slot:    int
    # 偏好 & 權重
    pref_gender:    int; weight_gender:   int
    pref_age:       int; weight_age:      int   # age category: 0=young,1=mid,2=senior
    pref_language:  int; weight_language: int
    pref_smoke:     int; weight_smoke:    int
    task_list:   List[int] = field(default_factory=list)


@dataclass
class TaskSpec:
    """任務規格"""
    task_id:        int
    required_skill: int
    duration:       float    # 服務時間 (分鐘)
    patient_id:     Optional[int] = None
    candidate_nurses: List[int] = field(default_factory=list)


@dataclass
class NurseTaskMatch:
    """
    單一護理師-任務-時段的可行媒合記錄。
    前提：此記錄存在即代表：
      (1) 護理師在 HHCP candidate list 中（技能已隱含相符）
      (2) 護理師在該時段可出勤
      (3) 加入此任務後工時 ≤ H_MAX
    """
    task_id:            int
    nurse_id:           int
    slot:               int
    # --- 任務靜態資訊 ---
    required_skill:     int    # 任務所需技能等級
    task_duration:      float  # 任務服務時間 (分鐘)
    # --- 護理師時間資訊 ---
    projected_workload: float  # 加入此任務後護理師預測工時 (分鐘)
    projected_slack:    float  # 加入後剩餘可用時間 (分鐘)，H_MAX - projected_workload
    # --- HHCP 預計算媒合分數 ---
    match_score:        float  # 直接來自 HHCP Candidate solution 欄位，不自行計算



@dataclass
class CandidateSolution:
    """
    一個事件的完整可行解群集輸出。
    不含任何分配決策；由上游排班程式接收後自行選擇。
    """
    event_type:       str
    t_now:            int
    # 核心輸出：{(task_id, slot): [所有可行護理師媒合記錄]}
    # 可行條件：技能符合 + 時段可出勤 + 加入後 workload ≤ H_MAX
    feasible_cluster: Dict[Tuple[int, int], List[NurseTaskMatch]]
    # 無任何可行護理師的任務清單
    unassigned_tasks: List[int]
    # 事件釋放的產能 {(nurse_id, slot): freed_minutes}
    freed_capacity:   Dict[Tuple, float]
    # 對 solution_schedule 的差異記錄（僅含移除操作）
    schedule_delta:   pd.DataFrame
    # 人類可讀摘要
    summary:          str


# ─────────────────────────── HHCP 完整解析器 ─────────────────────────────────

class HHCPFullParser:
    """
    嚴格解析 HHCP 複雜格式，提取：
    - 病患偏好 (4 維) 與任務清單
    - 護理師能力 (技能、班次、屬性)
    - 任務規格 (技能需求、工時、HHCP 候選護理師清單)
    - HHCP Candidate solution 預計算媒合分數 {(task_id, nurse_id): score}
    """

    def __init__(self, hhcp_path: str):
        self.patients:         Dict[int, PatientProfile]       = {}
        self.nurses:           Dict[int, NurseProfile]         = {}
        self.tasks:            Dict[int, TaskSpec]             = {}
        # 預計算媒合分數表：{(task_id, nurse_id): score}
        # 由 Candidate solution 欄位按序列索引建立
        self.task_nurse_score: Dict[Tuple[int, int], float]   = {}
        # 內部：用於建立分數索引的 (task_id, nurse_id) 有序配對清單
        self._task_nurse_pairs: List[Tuple[int, int]]         = []
        self._parse(hhcp_path)
        self._cross_link()

    # ── 主解析流程 ──────────────────────────────────────────────────────────

    def _parse(self, path: str):
        print(f"[HHCPFullParser] Reading: {path}")
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [l.strip() for l in f.readlines()]

        section = None
        for line in lines:
            if not line:
                continue
            lower = line.lower()

            # ── 節點切換（優先判斷，避免誤觸資料行）────────────────────────

            # Candidate solution 標頭行（含 "score"）→ 切入分數解析區
            if lower.startswith("candidate solution"):
                section = "CANDIDATE_SCORE"
                continue  # 跳過標頭行本身

            if line.startswith("#"):
                if "patient," in lower:   section = "PATIENT"
                elif "nurse," in lower:   section = "NURSE"
                elif "task," in lower:    section = "TASK"
                continue

            if lower.startswith("patient,"):  section = "PATIENT"; continue
            if lower.startswith("nurse,"):    section = "NURSE";   continue
            if lower.startswith("task,"):     section = "TASK";    continue

            # ── 資料行路由 ──────────────────────────────────────────────────
            parts = line.split(',')

            if section == "CANDIDATE_SCORE":
                # 分數行格式：<sequential_index>,<score>
                # 例：0,0.440  → sequential index 0 對應 task_nurse_pairs[0]
                self._parse_candidate_score(parts)
                continue

            # 其他區段要求首欄為數字
            if len(parts) < 2 or not parts[0].strip().isdigit():
                continue

            if section == "PATIENT":
                self._parse_patient(parts)
            elif section == "NURSE":
                self._parse_nurse(parts)
            elif section == "TASK":
                self._parse_task(parts)

        print(f"[HHCPFullParser] Loaded {len(self.patients)} patients, "
              f"{len(self.nurses)} nurses, {len(self.tasks)} tasks, "
              f"{len(self.task_nurse_score)} nurse-task scores.")

    def _parse_patient(self, parts: List[str]):
        try:
            pid = int(parts[0])
            self.patients[pid] = PatientProfile(
                patient_id=pid,
                x=float(parts[1]),
                y=float(parts[2]),
                req_slot=int(parts[3]),
                pref_gender=int(parts[4]),   weight_gender=int(parts[5]),
                pref_age=int(parts[6]),       weight_age=int(parts[7]),
                pref_language=int(parts[8]),  weight_language=int(parts[9]),
                pref_smoke=int(parts[10]),    weight_smoke=int(parts[11]),
                task_list=[int(t) for t in parts[13:] if t.strip().isdigit()]
                          if len(parts) > 13 else [],
            )
        except (ValueError, IndexError):
            pass

    def _parse_nurse(self, parts: List[str]):
        try:
            # 去掉空字串方便索引計算
            clean = [p.strip() for p in parts if p.strip()]
            nid   = int(clean[0])
            num_slots = int(clean[8])
            slot_end  = 9 + num_slots            # 在 clean 陣列中
            slots     = {int(x) for x in clean[9:slot_end]}

            num_skills  = int(clean[slot_end])   # blank 已被 clean 過濾
            skills      = {int(x) for x in clean[slot_end + 1: slot_end + 1 + num_skills]}

            self.nurses[nid] = NurseProfile(
                nurse_id=nid,
                x=float(clean[1]),
                y=float(clean[2]),
                nurse_class=int(clean[3]),
                gender=int(clean[4]),
                age=int(clean[5]),
                language=int(clean[6]),
                smoke=int(clean[7]),
                slots=slots,
                skills=skills,
            )
        except (ValueError, IndexError) as e:
            pass

    def _parse_task(self, parts: List[str]):
        try:
            if len(parts) < 3:
                return
            tid       = int(parts[0])
            req_skill = int(parts[1])
            duration  = float(parts[2])
            num_nurses = int(parts[3]) if len(parts) > 3 and parts[3].strip().isdigit() else 0
            cand_nurses = [int(x) for x in parts[4: 4 + num_nurses]
                           if x.strip().isdigit()]
            self.tasks[tid] = TaskSpec(
                task_id=tid,
                required_skill=req_skill,
                duration=duration,
                candidate_nurses=cand_nurses,
            )
            # 按序累積 (task_id, nurse_id) 配對，用於後續對應 Candidate solution 索引
            for nid in cand_nurses:
                self._task_nurse_pairs.append((tid, nid))
        except (ValueError, IndexError):
            pass

    def _parse_candidate_score(self, parts: List[str]):
        """
        解析 Candidate solution 區段的單行分數資料。
        格式：<sequential_index>,<score>
        例：  0,0.440

        sequential_index 對應 self._task_nurse_pairs 的位置，
        即所有 task 依序展開的 (task_id, nurse_id) 全域序號。
        """
        try:
            if len(parts) < 2:
                return
            idx   = int(parts[0].strip())
            score = float(parts[1].strip())
            if idx < len(self._task_nurse_pairs):
                tid, nid = self._task_nurse_pairs[idx]
                self.task_nurse_score[(tid, nid)] = score
        except (ValueError, IndexError):
            pass

    def _cross_link(self):
        """反向連結：患者 → 任務、任務 → 患者、任務候選 → 護理師"""
        for pid, pat in self.patients.items():
            for tid in pat.task_list:
                if tid in self.tasks:
                    self.tasks[tid].patient_id = pid
        for tid, task in self.tasks.items():
            for nid in task.candidate_nurses:
                if nid in self.nurses:
                    self.nurses[nid].candidate_tasks.add(tid)


# ─────────────────────────── 路線物理引擎 ────────────────────────────────────

class RoutePhysicsEngine:
    """
    嚴格按照 Constraint 9 (Workload = TravelTime + ServiceTime)
    依照任務序列動態計算護理師工時。
    """

    def __init__(self, parser: HHCPFullParser):
        self._p = parser

    def nurse_base(self, nid: int) -> Tuple[float, float]:
        n = self._p.nurses.get(nid)
        return (n.x, n.y) if n else (0.0, 0.0)

    def task_location(self, tid: int) -> Tuple[float, float]:
        t   = self._p.tasks.get(tid)
        pid = t.patient_id if t else None
        p   = self._p.patients.get(pid) if pid is not None else None
        return (p.x, p.y) if p else (0.0, 0.0)

    @staticmethod
    def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    @staticmethod
    def travel_time(distance_m: float) -> float:
        return distance_m / VELOCITY_MPM

    def compute_workload(self, nid: int, task_sequence: List[int]) -> Dict:
        """
        計算護理師 nid 執行 task_sequence 的完整工時。
        回傳：workload, travel_time, service_time, overtime, slack, is_feasible, centroid
        """
        base = self.nurse_base(nid)
        if not task_sequence:
            return dict(workload=0.0, travel=0.0, service=0.0,
                        overtime=0.0, slack=H_MAX, is_feasible=True,
                        centroid=base)

        cx, cy = base
        total_travel_dist = 0.0
        total_service     = 0.0
        sum_x, sum_y      = 0.0, 0.0
        unique_locations = set()
        
        for tid in task_sequence:
            loc  = self.task_location(tid)
            d    = self.dist((cx, cy), loc)
            total_travel_dist += d
            dur  = self._p.tasks.get(tid, TaskSpec(0, 0, 0.0)).duration
            total_service += dur
            cx, cy = loc
            sum_x  += loc[0]
            sum_y  += loc[1]

        # 返回基地
        total_travel_dist += self.dist((cx, cy), base)

        travel   = self.travel_time(total_travel_dist)
        workload = travel + total_service
        overtime = max(0.0, workload - H_STD)
        slack    = max(0.0, H_MAX - workload)
        n        = len(task_sequence)

        return dict(
            workload=workload, travel=travel, service=total_service,
            overtime=overtime, slack=slack,
            is_feasible=(workload <= H_MAX),
            centroid=(sum_x / n, sum_y / n),
        )


# ─────────────────────────── 護理師產能追蹤器 ────────────────────────────────

class NurseCapacityTracker:
    """
    基於 solution_schedule 初始化，並可對事件進行動態增刪更新。
    追蹤每個 (nurse_id, slot) 的任務序列 → 實時工時。
    """

    def __init__(self, solution_path: str, physics: RoutePhysicsEngine):
        self._physics = physics
        # roster: {(nid, slot): [task_id, ...]}
        self.roster: Dict[Tuple[int, int], List[int]] = {}
        self._load(solution_path)

    def _load(self, path: str):
        df = pd.read_csv(path)
        df['GlobalSlot'] = (df['Day'] - 1) * SLOTS_PER_DAY + df['Shift']

        def parse_ids(x):
            return [int(t) for t in str(x).replace(';', ' ').replace(',', ' ').split()
                    if t.isdigit()]

        # Route physics depends on visit order; if CSV rows are unordered the
        # travel-time model would be computed on a scrambled sequence → wrong workload.
        df_sorted = df.sort_values(['NurseID', 'GlobalSlot', 'Sequence'])
        for _, row in df_sorted.iterrows():
            key = (int(row['NurseID']), int(row['GlobalSlot']))
            tasks = parse_ids(row['TaskIDs'])
            if key not in self.roster:
                self.roster[key] = []
            self.roster[key].extend(tasks)

        # 初始化 _slot_wl —— 精確計算每位護理師各 Slot 的基礎工時。
        # 此字典作為 projected_workload 的「既有工時基準」備份，
        # 在 remove_task / remove_nurse_slot 後同步更新，
        # 確保任意 tracker 狀態下均能正確反映護理師當前工時。
        self._slot_wl: Dict[Tuple[int, int], float] = {}
        for (nid, slot), tasks in self.roster.items():
            self._slot_wl[(nid, slot)] = self._physics.compute_workload(nid, tasks)['workload']

    def clone(self) -> 'NurseCapacityTracker':
        """
        建立一個狀態完全獨立的追蹤器副本。
        繞過 __init__ 避免重複讀取 CSV，直接在記憶體中執行深拷貝，確保效能與絕對隔離。
        """
        new_tracker = NurseCapacityTracker.__new__(NurseCapacityTracker)
        new_tracker._physics = self._physics             # 物理引擎無狀態，可安全共用指標
        new_tracker.roster = copy.deepcopy(self.roster)  # 班表陣列必須深拷貝
        new_tracker._slot_wl = copy.deepcopy(self._slot_wl) # 工時快取必須深拷貝
        return new_tracker

    def get_tasks(self, nid: int, slot: int) -> List[int]:
        return list(self.roster.get((nid, slot), []))

    def get_workload(self, nid: int, slot: int) -> Dict:
        tasks = self.get_tasks(nid, slot)
        return self._physics.compute_workload(nid, tasks)

    def add_task(self, nid: int, slot: int, tid: int):
        key = (nid, slot)
        if key not in self.roster:
            self.roster[key] = []
        if tid not in self.roster[key]:
            self.roster[key].append(tid)
            self._slot_wl[key] = self._physics.compute_workload(nid, self.roster[key])['workload']
            
    def remove_task(self, nid: int, slot: int, tid: int):
        key = (nid, slot)
        if key in self.roster and tid in self.roster[key]:
            self.roster[key].remove(tid)
            # 移除後若清單為空，清除鍵值保持與 remove_nurse_slot 一致的狀態語義。
            # 防止空條目在多事件序列中累積，並避免後續 get_workload 對空序列的無效計算。
            if not self.roster[key]:
                self.roster.pop(key)
                self._slot_wl.pop(key, None)          
            else:
                # 任務移除後重算剩餘序列的真實工時，維持 _slot_wl 精確
                self._slot_wl[key] = self._physics.compute_workload(nid, self.roster[key])['workload']

    def remove_nurse_slot(self, nid: int, slot: int) -> List[int]:
        """移除該護理師在該時段的所有任務，回傳孤兒任務清單"""
        key = (nid, slot)
        orphans = list(self.roster.get(key, []))
        self.roster.pop(key, None)
        self._slot_wl.pop(key, None)      # 同步清除工時快取
        return orphans

    def projected_workload(self, nid: int, slot: int, new_tid: int) -> Dict:
        """
        預測加入 new_tid 後的工時（不真正寫入）。

        三層防禦確保現有工時不被遺漏：

        第一層（最精確）：roster 有完整任務序列 → 重算含新任務的完整路線。
          compute_workload([existing_tasks..., new_tid]) 一次性算出正確工時。

        第二層（防禦性回退）：roster 已空但 _slot_wl 有正值
          → 狀態不一致（清空 roster 後 _slot_wl 同步清零，
            理論上不應進入此層；但若仍發生則保守估算）。
          predicted = existing_accumulated + Δtravel(base→new_task→base) + new_service
          此估算保守（起點用護理師基地而非實際最後位置），但能正確
          反映現有工時基礎，避免 Nurse 46 Slot 7 被誤算為 83m 而非 219m。

        第三層：roster 與 _slot_wl 均為空 → 護理師在該 Slot 確實無現有工作
        """
        existing_tasks = self.get_tasks(nid, slot)

        # ── 第一層：有完整序列，直接重算（最精確）──────────────────────────
        if existing_tasks:
            return self._physics.compute_workload(nid, existing_tasks + [new_tid])

        # ── 第二層：roster 空但仍有既有工時紀錄（狀態不一致保護）───────────
        existing_acc = self._slot_wl.get((nid, slot), 0.0)
        if existing_acc > 0.0:
            print(f"  [WARN] Nurse {nid} Slot {slot}: roster 為空但 "
                  f"_slot_wl={existing_acc:.1f}m。以 AccumulatedTime 為基準估算。")
            # 增量：從護理師基地出發訪問新任務再返回（保守估計 Δtravel）
            new_only = self._physics.compute_workload(nid, [new_tid])
            projected = existing_acc + new_only['travel'] + new_only['service']
            slack     = max(0.0, H_MAX - projected)
            return dict(
                workload=projected,
                travel=new_only['travel'],
                service=new_only['service'],
                overtime=max(0.0, projected - H_STD),
                slack=slack,
                is_feasible=(projected <= H_MAX),
                centroid=(0.0, 0.0),
            )

        # ── 第三層：護理師在此 Slot 確實無現有工作，單獨計算新任務 ──────────
        return self._physics.compute_workload(nid, [new_tid])


# ─────────────────────────── 護理師-任務媒合評分器 ───────────────────────────

class NurseTaskScorer:
    """
    可行媒合資訊提供器。
    硬性篩選（任一成立 → 回傳 None，不進候選集）：
      1. 護理師不在 HHCP 任務候選名單中（技能不符已隱含於此）
      2. 護理師在該時段不可出勤
      3. 加入此任務後工時 > H_MAX

    通過篩選者：直接讀取 HHCP Candidate solution 預計算分數，不自行計算。
    """

    def __init__(self, parser: HHCPFullParser,
                 tracker: NurseCapacityTracker):
        self._p  = parser
        self._t  = tracker

    # ── 單次可行性評估與 HHCP 分數讀取 ──────────────────────────────────────

    def score(self, nid: int, tid: int, slot: int) -> Optional[NurseTaskMatch]:
        """
        評估 (nurse, task, slot) 是否可行，並讀取 HHCP 預計算媒合分數。

        硬性排除條件（任一成立 → 回傳 None）：
          1. 護理師不在 HHCP candidate list → 技能不相容，無預計算分數
          2. 護理師在 slot 不可出勤
          3. 加入此任務後 projected_workload > H_MAX

        通過者回傳 NurseTaskMatch（含 HHCP 分數），供上游排班程式使用。
        """
        nurse = self._p.nurses.get(nid)
        task  = self._p.tasks.get(tid)
        if nurse is None or task is None:
            return None

        # ── 硬性排除 1: 不在 HHCP 候選名單（技能不符已隱含）────────────────
        if nid not in task.candidate_nurses:
            return None

        # ── 硬性排除 2: 時段不可出勤 ─────────────────────────────────────
        if slot not in nurse.slots:
            return None

        # ── 物理可行性計算 ────────────────────────────────────────────────
        proj = self._t.projected_workload(nid, slot, tid)

        # ── 硬性排除 3: 工時超過 H_MAX ───────────────────────────────────
        if not proj['is_feasible']:
            return None

        # ── 直接讀取 HHCP 預計算媒合分數 ─────────────────────────────────
        # task_nurse_score 在解析時已對齊 Candidate solution 索引，保證存在
        match_score = self._p.task_nurse_score.get((tid, nid), 0.0)

        return NurseTaskMatch(
            task_id=tid,
            nurse_id=nid,
            slot=slot,
            required_skill=task.required_skill,
            task_duration=task.duration,
            projected_workload=proj['workload'],
            projected_slack=proj['slack'],
            match_score=match_score,
        )

    def feasible_candidates(
        self,
        tid:             int,
        slot:            int,
        exclude_nurses:  Optional[Set[int]] = None,
    ) -> List[NurseTaskMatch]:
        """
        收集指定任務與時段的所有可行護理師。
        僅遍歷 HHCP candidate list 中的護理師（不遍歷全體）。
        回傳列表維持插入順序，不排序 — 由上游排班程式自行處理。
        """
        task = self._p.tasks.get(tid)
        if task is None:
            return []

        results = []
        exc     = exclude_nurses or set()

        # 僅在 HHCP 候選名單中搜尋，大幅減少無效遍歷
        for nid in task.candidate_nurses:
            if nid in exc:
                continue
            m = self.score(nid, tid, slot)
            if m is not None:
                results.append(m)
        return results


# ─────────────────────────── 可行群集建構器 ──────────────────────────────────

class FeasibleClusterBuilder:
    """
    純可行解群集建構器。
    對每個 (task_id, slot) 收集所有通過硬性篩選的護理師，
    不做任何選擇或排序，不修改 tracker 狀態。
    """

    def __init__(self, scorer: NurseTaskScorer):
        self._scorer = scorer

    def build(
        self,
        task_ids:       List[int],
        slot:           int,
        exclude_nurses: Optional[Set[int]] = None,
    ) -> Tuple[Dict[Tuple[int, int], List[NurseTaskMatch]], List[int]]:
        """
        回傳：
          cluster  — {(task_id, slot): [NurseTaskMatch, ...]}，包含所有可行護理師
          no_match — 無任何可行護理師的任務 ID 清單
        """
        cluster:  Dict[Tuple[int, int], List[NurseTaskMatch]] = {}
        no_match: List[int] = []

        for tid in task_ids:
            candidates = self._scorer.feasible_candidates(tid, slot, exclude_nurses)
            key = (tid, slot)
            if candidates:
                cluster[key] = candidates
            else:
                no_match.append(tid)

        return cluster, no_match


# ─────────────────────────── 主優化器 ────────────────────────────────────────

class SubgroupOptimizer:
    """
    D-CHHRP-MD 子群集優化器主類別。

    Usage:
        optimizer = SubgroupOptimizer(hhcp_file, solution_file)
        solution  = optimizer.handle_event(event)
        print(solution.summary)
        # 取得可行群集
        for (task_id, slot), candidates in solution.feasible_cluster.items():
            for m in candidates:
                print(m)
    """

    def __init__(self, hhcp_file: str, solution_file: str):
        print("[SubgroupOptimizer] Initializing full system...")
        self.parser        = HHCPFullParser(hhcp_file)
        self.physics       = RoutePhysicsEngine(self.parser)
        self.base_tracker  = NurseCapacityTracker(solution_file, self.physics)
        self.tracker       = None
        self.scorer        = None
        self.cluster_builder = None
        # 儲存原始 schedule 供 delta 計算
        self._solution_df = pd.read_csv(solution_file)
        self._solution_df['GlobalSlot'] = (
            (self._solution_df['Day'] - 1) * SLOTS_PER_DAY + self._solution_df['Shift']
        )
        print("[SubgroupOptimizer] System ready.")

    # ── 事件分發 ─────────────────────────────────────────────────────────────

    def handle_event(self, event: StrictEvent) -> Optional[CandidateSolution]:
        """
        接收 DynamicEventGenerator 產生的事件，回傳可行解群集。
        """
        print(f"\n{'='*60}")
        print(f"[SubgroupOptimizer] Handling: {event.event_type} @ t={event.t_now}")
        print(f"  {event.description}")

        # 為每個事件建立獨立的沙盒環境 (Sandbox Environment)
        self.tracker = self.base_tracker.clone()
        self.scorer  = NurseTaskScorer(self.parser, self.tracker)
        self.cluster_builder = FeasibleClusterBuilder(self.scorer)

        # ── 時序因果保護 (Temporal Causality Guard) ──────────────────────────────
        # The original code used Python `or` short-circuit:
        #   slot = data.get('slot') or data.get('new_slot') or data.get('old_slot')
        # This has TWO defects:
        #   (a) slot=0 is falsy → the `or` falls through to the next key incorrectly.
        #   (b) Only the FIRST truthy slot is checked; for PATIENT_RESCHEDULE both
        #       old_slot AND new_slot must be in the future.
        # Fix: collect all relevant slot keys and check each one independently.
        # Bug 6 修正：改用嚴格小於（<）以允許對當前進行中時段（slot == t_now）的修改。
        # 原版使用 <=，導致 slot == t_now 被誤判為「歷史時段」而遭封鎖。
        # 語義：只阻擋「嚴格過去」（slot < t_now）的時段；當前時段（slot == t_now）
        # 在實際運營中可能因護士中途請假等情況而需要即時修改，屬合法需求。
        _slot_keys = ['slot', 'new_slot', 'old_slot']
        for _key in _slot_keys:
            _val = event.data.get(_key)
            if _val is not None:                        # explicit None guard (slot=0 is valid)
                _s = int(_val)
                if _s < event.t_now:                   # Bug 6 fix: < (not <=)
                    print(f"  [BLOCKED] {_key}={_s} < t_now={event.t_now}. "
                          f"Historical mutation denied.")
                    return None

        dispatcher = {
            "NURSE_ABSENCE":        self._handle_nurse_absence,
            "PATIENT_RESCHEDULE":   self._handle_patient_reschedule,
            "PATIENT_CANCELLATION": self._handle_patient_cancellation,
        }
        handler = dispatcher.get(event.event_type)
        if handler is None:
            print(f"  [WARN] Unknown event type: {event.event_type}")
            return None

        solution = handler(event)
        solution.summary = self._build_summary(event, solution)
        print(solution.summary)
        return solution

    # ── 事件處理 1：護理師請假 ────────────────────────────────────────────────

    def _handle_nurse_absence(self, event: StrictEvent) -> CandidateSolution:
        """
        護理師 n 請假 → 移除其未來所有班次（事實操作）→
        對每個受影響 (task, slot) 建構可行護理師群集。
        """
        absent_nid     = int(event.data['nurse_id'])
        orphan_tasks   = [int(t) for t in event.data.get('orphan_tasks', [])]
        affected_slots = sorted(event.data.get('affected_slots', []))

        # 移除缺勤護理師未來班次（事實：他確實不在），記錄釋放產能
        freed_capacity: Dict[Tuple, float] = {}
        for s in affected_slots:
            if s < event.t_now:
                continue
            wl = self.tracker.get_workload(absent_nid, s)
            freed_capacity[(absent_nid, s)] = wl['workload']
            self.tracker.remove_nurse_slot(absent_nid, s)

        # 從原始 schedule 取得各時段的孤兒任務（按時段分組）
        task_by_slot: Dict[int, List[int]] = {}
        for s in affected_slots:
            if s < event.t_now:
                continue
            slot_rows = self._solution_df[
                (self._solution_df['NurseID'] == absent_nid) &
                (self._solution_df['GlobalSlot'] == s)
            ]
            slot_tasks = []
            for _, row in slot_rows.iterrows():
                for tid in [int(t) for t in str(row['TaskIDs'])
                            .replace(';', ' ').replace(',', ' ').split()
                            if t.strip().isdigit()]:
                    if tid in orphan_tasks and self.parser.tasks.get(tid):
                        slot_tasks.append(tid)
            if slot_tasks:
                task_by_slot[s] = slot_tasks

        if not task_by_slot:
            first_future = [s for s in affected_slots if s >= event.t_now]
            if first_future:
                task_by_slot[first_future[0]] = [
                    t for t in orphan_tasks if self.parser.tasks.get(t)
                ]

        # 對每個 (task, slot) 建構可行群集
        all_cluster:   Dict[Tuple[int, int], List[NurseTaskMatch]] = {}
        all_unmatched: List[int] = []

        for s, tasks in task_by_slot.items():
            if not tasks:
                continue
            cluster, no_match = self.cluster_builder.build(
                tasks, s, exclude_nurses={absent_nid}
            )
            all_cluster.update(cluster)
            all_unmatched.extend(no_match)

        delta = self._build_delta_remove(
            nurse_id=absent_nid,
            affected_slots=[s for s in affected_slots if s >= event.t_now],
        )
        return CandidateSolution(
            event_type="NURSE_ABSENCE",
            t_now=event.t_now,
            feasible_cluster=all_cluster,
            unassigned_tasks=all_unmatched,
            freed_capacity=freed_capacity,
            schedule_delta=delta,
            summary="",
        )

    # ── 事件處理 2：病患改期 ──────────────────────────────────────────────────

    def _handle_patient_reschedule(self, event: StrictEvent) -> CandidateSolution:
        """
        病患從 old_slot 移至 new_slot：
          Step 1: 從 old_slot 移除病患「所有」任務（事實），記錄每位護理師的釋放產能
          Step 2: 對 new_slot 中「所有」任務建構可行護理師群集
        """
        pid      = int(event.data['patient_id'])
        old_slot = int(event.data['old_slot'])
        new_slot = int(event.data['new_slot'])

        # ── 以 parser 為單一真實來源取所有病患任務 ────────────────
        patient = self.parser.patients.get(pid)
        if patient is None:
            print(f"  [WARN] Patient {pid} not found in parser.")
            return CandidateSolution(
                event_type="PATIENT_RESCHEDULE", t_now=event.t_now,
                feasible_cluster={}, unassigned_tasks=[],
                freed_capacity={}, schedule_delta=pd.DataFrame(), summary="",
            )
        all_task_ids: List[int] = [
            t for t in patient.task_list if self.parser.tasks.get(t) is not None
        ]
        # 若 parser 無資料則退回事件提供的清單（防禦性後備）
        if not all_task_ids:
            all_task_ids = [int(t) for t in event.data.get('task_ids', [])]
        task_id_set = set(all_task_ids)

        freed_capacity: Dict[Tuple, float] = {}

        # ── 找出 old_slot 中所有持有病患任務的護理師 ────────
        # 使用 list() 快照避免在迭代中修改 dict
        # 不使用 break → 確保每位護理師都被處理
        affected: Dict[int, List[int]] = {}   # {nurse_id: [patient_tasks_in_this_slot]}
        for (nid, sl), tasks in list(self.tracker.roster.items()):
            if sl != old_slot:
                continue
            nurse_patient_tasks = [t for t in tasks if t in task_id_set]
            if nurse_patient_tasks:
                affected[nid] = nurse_patient_tasks

        for nid, nurse_tasks in affected.items():
            wl_before = self.tracker.get_workload(nid, old_slot)
            for tid in nurse_tasks:
                self.tracker.remove_task(nid, old_slot, tid)
            wl_after = self.tracker.get_workload(nid, old_slot)
            freed_capacity[(nid, old_slot)] = max(
                0.0, wl_before['workload'] - wl_after['workload']
            )

        # ── Step 2: 對 new_slot 中所有病患任務建構可行群集 ────────────────────
        cluster, no_match = self.cluster_builder.build(all_task_ids, new_slot)

        delta = self._build_delta_reschedule(pid=pid, old_slot=old_slot)
        return CandidateSolution(
            event_type="PATIENT_RESCHEDULE",
            t_now=event.t_now,
            feasible_cluster=cluster,
            unassigned_tasks=no_match,
            freed_capacity=freed_capacity,
            schedule_delta=delta,
            summary="",
        )

    # ── 事件處理 3：病患取消服務 ──────────────────────────────────────────────

    def _handle_patient_cancellation(self, event: StrictEvent) -> CandidateSolution:
        """
        病患取消 slot 的所有服務：
          移除已排班任務（事實），計算釋放的護理師產能。
          不執行任何後續填補或再分配決策。
        """
        # ── 防禦性讀取 slot，避免 KeyError 崩潰 ───────────────
        _raw_slot = event.data.get('slot')
        if _raw_slot is None:
            print(f"  [ERROR] PATIENT_CANCELLATION 事件缺少 'slot' 鍵，無法處理。")
            return CandidateSolution(
                event_type="PATIENT_CANCELLATION", t_now=event.t_now,
                feasible_cluster={}, unassigned_tasks=[],
                freed_capacity={}, schedule_delta=pd.DataFrame(), summary="",
            )
        target_slot = int(_raw_slot)

        # ── 防禦性讀取 patient_id ────────────────────────────
        pid = event.data.get('patient_id')

        # ── 以 parser 為任務真實來源，event 資料僅作後備 ───────────
        if pid is not None:
            patient = self.parser.patients.get(int(pid))
            if patient and patient.task_list:
                # parser 真實來源：取病患全部任務（已在 parser 中驗證存在）
                cancelled_set: Set[int] = {
                    t for t in patient.task_list
                    if self.parser.tasks.get(t) is not None
                }
                print(f" Patient {pid}: parser 任務來源 {sorted(cancelled_set)}"
                      f"（共 {len(cancelled_set)} 個），"
                      f"事件傳入 {event.data.get('cancelled_tasks', [])} 已被覆蓋。")
            else:
                # parser 無資料：退回事件提供的清單（防禦性後備）
                cancelled_set = {int(t) for t in event.data.get('cancelled_tasks', [])}
                print(f"  [WARN] Patient {pid} 在 parser 中無任務資料，"
                      f"退回 event.data['cancelled_tasks']={cancelled_set}。")
        else:
            # 無 patient_id：只能信任事件清單（記錄警告）
            cancelled_set = {int(t) for t in event.data.get('cancelled_tasks', [])}
            print(f"  [WARN] PATIENT_CANCELLATION 事件缺少 'patient_id'，"
                  f"退回 event.data['cancelled_tasks']={cancelled_set}，"
                  f"無法以 parser 核實完整任務清單。")

        freed_capacity: Dict[Tuple, float] = {}

        # 掃描所有護士，找出持有取消任務的護士（不限單一護士）
        # cancelled_set 現已完整，掃描可正確找到全部受影響護士
        affected_nurses: Dict[int, List[int]] = {}
        for (nid, sl), tasks in list(self.tracker.roster.items()):
            if sl != target_slot:
                continue
            nurse_cancelled = [t for t in tasks if t in cancelled_set]
            if nurse_cancelled:
                affected_nurses[nid] = nurse_cancelled

        # ─────────────────────────────────────────────────────────────────────
        # Bug 4 + Bug 1 修正：當 tracker scan 未找到任何受影響護士時（例如先前的
        # NURSE_ABSENCE 事件已清空 roster），採用兩層防禦機制重建任務歸屬：
        #
        # 第二層（主要後備）：從 _solution_df 查詢原始排班，為每位持有
        #   cancelled_set 中任務的護士分別建立正確的任務歸屬映射。
        #   此層保留完整的多護士語義（Multi-Nurse Service），不假設
        #   所有任務都屬於單一護士。
        #
        # 第三層（最終後備）：只有在 _solution_df 也無法找到匹配行時，
        #   才退回 freed_nurse_id 作為最後手段，且僅用於記錄，
        #   不可用於任務歸屬（語義上不正確）。
        # ─────────────────────────────────────────────────────────────────────
        if not affected_nurses:
            # 第二層：從 solution_df 重建任務歸屬
            def _parse_taskids(x: object) -> List[int]:
                return [int(t) for t in str(x).replace(';', ' ').replace(',', ' ').split()
                        if t.strip().isdigit()]

            sol_slot_rows = self._solution_df[
                self._solution_df['GlobalSlot'] == target_slot
            ]
            for _, row in sol_slot_rows.iterrows():
                nid = int(row['NurseID'])
                row_tasks = _parse_taskids(row['TaskIDs'])
                nurse_cancelled = [t for t in row_tasks if t in cancelled_set]
                if nurse_cancelled:
                    if nid not in affected_nurses:
                        affected_nurses[nid] = []
                    # 去重：避免多 Sequence 行重複加入同一 task
                    for t in nurse_cancelled:
                        if t not in affected_nurses[nid]:
                            affected_nurses[nid].append(t)

            if affected_nurses:
                print(f"  [INFO] tracker scan 為空（roster 可能已被先前事件清除），"
                      f"改從 solution_df 重建任務歸屬映射：{ {k: v for k, v in affected_nurses.items()} }")
            else:
                # 第三層（最終後備）：solution_df 也無法找到匹配行
                # 僅記錄警告，不做錯誤的任務歸屬（Bug 1 根本原因）
                hint_nid = event.data.get('freed_nurse_id')
                if hint_nid is not None and int(hint_nid) >= 0:
                    hint_nid = int(hint_nid)
                    # 注意：此處不將 cancelled_set 全部歸屬於單一護士（Bug 1 修正）。
                    # 僅使用 hint_nid 作為日誌提示，不可信賴其任務歸屬語義。
                    # 若 solution_df 也找不到，則 affected_nurses 保持為空，
                    # 下游 freed_capacity 與 schedule_delta 均為空（明確失敗，
                    # 勝於產出錯誤的歸屬資料）。
                    print(f"  [WARN] tracker 與 solution_df 均找不到 Slot {target_slot} "
                          f"中任何取消任務（{sorted(cancelled_set)}）。"
                          f"freed_nurse_id={hint_nid} 僅供參考，無法確認任務歸屬，"
                          f"不產出 schedule_delta 以避免錯誤資料。")

        # cancelled_set 完整後，所有護士的 freed_capacity 均被記錄
        cancelled_by_nurse: Dict[int, List[int]] = {}
        for nid, nurse_tasks in affected_nurses.items():
            wl_before = self.tracker.get_workload(nid, target_slot)
            for tid in nurse_tasks:
                self.tracker.remove_task(nid, target_slot, tid)
            wl_after = self.tracker.get_workload(nid, target_slot)
            freed_capacity[(nid, target_slot)] = max(
                0.0, wl_before['workload'] - wl_after['workload']
            )
            cancelled_by_nurse[nid] = nurse_tasks

        # delta 從真實受影響護士建構，不依賴可能為 -1 的 freed_nurse_id
        delta = self._build_delta_cancel_multi(
            cancelled_by_nurse=cancelled_by_nurse,
            slot=target_slot,
        )
        return CandidateSolution(
            event_type="PATIENT_CANCELLATION",
            t_now=event.t_now,
            feasible_cluster={},     # 取消事件不產生待分配任務
            unassigned_tasks=[],
            freed_capacity=freed_capacity,
            schedule_delta=delta,
            summary="",
        )

    # ── 輔助：Schedule Delta 建構（僅含移除操作）────────────────────────────

    def _build_delta_remove(
        self, nurse_id: int, affected_slots: List[int]
    ) -> pd.DataFrame:
        """護理師請假：記錄該護理師所有受影響班次的移除操作"""
        rows = []
        for s in affected_slots:
            day   = (s // SLOTS_PER_DAY) + 1
            shift = s  %  SLOTS_PER_DAY
            nurse_rows = self._solution_df[
                (self._solution_df['NurseID'] == nurse_id) &
                (self._solution_df['GlobalSlot'] == s)
            ]
            for _, r in nurse_rows.iterrows():
                rows.append({
                    'Action':     'REMOVE',
                    'NurseID':    nurse_id,
                    'Day':        day,
                    'Shift':      shift,
                    'GlobalSlot': s,
                    'TaskIDs':    str(r['TaskIDs']),
                    'Reason':     'Nurse absent',
                })
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _build_delta_reschedule(self, pid: int, old_slot: int) -> pd.DataFrame:
        """病患改期：記錄舊時段任務的移除操作"""
        rows = []
        old_rows = self._solution_df[
            (self._solution_df['PatientID'] == pid) &
            (self._solution_df['GlobalSlot'] == old_slot)
        ]
        for _, r in old_rows.iterrows():
            rows.append({
                'Action':     'REMOVE_OLD_SLOT',
                'NurseID':    int(r['NurseID']),
                'Day':        int(r['Day']),
                'Shift':      int(r['Shift']),
                'GlobalSlot': old_slot,
                'TaskIDs':    str(r['TaskIDs']),
                'Reason':     'Patient rescheduled',
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _build_delta_cancel_multi(
        self,
        cancelled_by_nurse: Dict[int, List[int]],
        slot: int,
    ) -> pd.DataFrame:
        """
        病患取消：為每位受影響護士分別記錄移除操作。
        接受 {nurse_id: [task_ids]} 字典，正確記錄所有護士。
        """
        day   = (slot // SLOTS_PER_DAY) + 1
        shift = slot  %  SLOTS_PER_DAY
        rows  = []
        for nid, tasks in cancelled_by_nurse.items():
            rows.append({
                'Action':     'REMOVE',
                'NurseID':    nid,
                'Day':        day,
                'Shift':      shift,
                'GlobalSlot': slot,
                'TaskIDs':    str(tasks),
                'Reason':     'Patient cancelled',
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _build_delta_cancel(
        self, nurse_id: int, cancelled_tasks: List[int], slot: int
    ) -> pd.DataFrame:
        """舊版單護士取消 delta（保留供外部相容性；內部已改用 _build_delta_cancel_multi）。
        """
        if nurse_id < 0:                          # 防禦：不產生 NurseID=-1 的垃圾列
            return pd.DataFrame()
        day   = (slot // SLOTS_PER_DAY) + 1
        shift = slot  %  SLOTS_PER_DAY
        rows  = [{'Action': 'REMOVE', 'NurseID': nurse_id,
                  'Day': day, 'Shift': shift, 'GlobalSlot': slot,
                  'TaskIDs': str(cancelled_tasks), 'Reason': 'Patient cancelled'}]
        return pd.DataFrame(rows)

    # ── 輔助：資訊摘要生成 ────────────────────────────────────────────────────

    def _build_summary(self, event: StrictEvent, sol: CandidateSolution) -> str:
        # event.math_constraint 由外部事件生成器設定（可能含錯誤）。
        # 對 PATIENT_RESCHEDULE 動態重算正確的約束式，反映實際任務數量。
        #
        # 同樣問題存在於 PATIENT_CANCELLATION：原版直接沿用
        # event.math_constraint，其中任務數可能錯誤（如「2 個任務」而非正確的「5 個任務」）
        # 且護士列表可能只含 Nurse 46 而非所有 4 位受影響護士。
        # 修正：對 PATIENT_CANCELLATION 同樣動態重算 math_line 與護士摘要。
        #
        # event.description 由事件生成器設定，可能含錯誤任務數（如「共 1 個任務」），
        # 與同一 Summary 塊內已修正的 math_line（「5 tasks 已取消」）直接矛盾。
        # 修正：新增 display_description（① 初始化為 event.description 保持其他事件
        # 的現有行為不變）；② 在 PATIENT_CANCELLATION 分支末尾複用已計算的
        # n_tasks / nurses_str / pid / slot_val 覆寫 display_description，
        # 確保描述行與數學行使用完全相同的數值，邏輯上不可能再出現不一致。

        math_line = event.math_constraint
        # 初始化 display_description；對非 CANCELLATION 事件維持原始輸出
        display_description: str = event.description

        if event.event_type == "PATIENT_RESCHEDULE":
            pid      = event.data.get('patient_id')
            old_slot = event.data.get('old_slot')
            new_slot = event.data.get('new_slot')
            n_tasks  = len(sol.feasible_cluster) + len(sol.unassigned_tasks)
            if pid is not None and old_slot is not None and new_slot is not None:
                math_line = (
                    f"sum(x[i, {pid}, {new_slot}]) = {n_tasks}, "
                    f"x[*, {pid}, {old_slot}] = 0"
                )
            # PATIENT_RESCHEDULE 的 description 由事件生成器設定，內容通常正確，
            # 此處不覆寫 display_description（保持 v8 既有行為）。

        elif event.event_type == "PATIENT_CANCELLATION":
            # 動態重算正確的取消約束式
            pid      = event.data.get('patient_id')
            slot_val = event.data.get('slot')

            # 從 freed_capacity 的 key 提取所有受影響護士（已由 handler 完整計算）
            affected_nids: List[int] = [
                nid for (nid, sl) in sorted(sol.freed_capacity.keys())
            ]

            # 以 parser 確認取消任務總數（pid 已在 handler 中驗證過）
            all_cancelled_tasks: List[int] = []
            if pid is not None:
                patient = self.parser.patients.get(int(pid))
                if patient:
                    all_cancelled_tasks = [
                        t for t in patient.task_list
                        if self.parser.tasks.get(t) is not None
                    ]
            n_tasks = len(all_cancelled_tasks) if all_cancelled_tasks else \
                      len(event.data.get('cancelled_tasks', []))

            nurses_str = ", ".join(f"Nurse {n}" for n in affected_nids) \
                         if affected_nids else "（無記錄）"

            math_line = (
                f"x[*, {pid}, {slot_val}] = 0 "
                f"({n_tasks} tasks 已取消; "
                f"受影響護士: {nurses_str})"
            )

            # 覆寫 display_description，確保與 math_line 數值完全一致，
            # 根絕「1 個任務」vs「5 tasks」的描述與數學行矛盾。
            display_description = (
                f"病患 {pid} 取消了 Slot {slot_val} 的所有服務 "
                f"(共 {n_tasks} 個任務; 受影響護士: {nurses_str})"
            )

        lines = [
            f"\n{'─'*60}",
            f"[CandidateSolution] Event: {event.event_type} | t_now={event.t_now}",
            f"  {display_description}",   # 改為輸出 display_description
            f"  Math: {math_line}",
            f"",
        ]

        if sol.feasible_cluster:
            lines.append(f"  可行群集 ({len(sol.feasible_cluster)} 個 task-slot 對):")
            for (tid, slot), matches in sol.feasible_cluster.items():
                task = self.parser.tasks.get(tid)
                skill = task.required_skill if task else '?'
                dur   = task.duration       if task else '?'
                lines.append(
                    f"    Task {tid:>4} [Slot {slot}] "
                    f"需求技能={skill} 服務時間={dur}m | "
                    f"可行護理師數={len(matches)}"
                )
                for m in matches:
                    lines.append(
                        f"      Nurse {m.nurse_id:>3} | "
                        f"預測工時={m.projected_workload:.0f}m "
                        f"剩餘={m.projected_slack:.0f}m | "
                        f"HHCP媒合分數={m.match_score:.3f}"
                    )
        else:
            lines.append("  可行群集：(無待分配任務)")

        if sol.unassigned_tasks:
            lines.append(
                f"\n  無可行護理師的任務: {sol.unassigned_tasks}"
            )

        if sol.freed_capacity:
            lines.append(f"\n  釋放產能:")
            for (nid, sl), mins in sol.freed_capacity.items():
                lines.append(f"    Nurse {nid} Slot {sl}: {mins:.0f} min")

        lines.append(f"{'─'*60}")
        return "\n".join(lines)

    # ── 外部工具：時段護理師工時資訊 ─────────────────────────────────────────

    def slot_status_report(self, slot: int) -> str:
        """回傳指定 Global Slot 所有護理師的原始工時資訊（不含優劣判斷）"""
        lines = [f"\n[Slot {slot} 工時資訊]"]
        active = {nid for (nid, sl) in self.tracker.roster if sl == slot}
        for nid in sorted(active):
            wl    = self.tracker.get_workload(nid, slot)
            tasks = self.tracker.get_tasks(nid, slot)
            lines.append(
                f"  Nurse {nid:>3}: "
                f"工時={wl['workload']:.0f}m  "
                f"剩餘={wl['slack']:.0f}m  "
                f"行程={wl['travel']:.0f}m  "
                f"服務={wl['service']:.0f}m  "
                f"Tasks={tasks}"
            )
        return "\n".join(lines)


# ─────────────────────────── 主程式 / 整合驗證 ───────────────────────────────

if __name__ == "__main__":

    HHCP_FILE = 'HHCP-patient-65-task-256-nurse-59.csv'
    SOL_FILE  = 'solution_schedule.csv'

    print("=" * 60)
    print("  D-CHHRP-MD SubgroupOptimizer — 純可行解群集驗證")
    print("=" * 60)

    optimizer = SubgroupOptimizer(HHCP_FILE, SOL_FILE)

    T_NOW = 4   # Day 2 午班結束後；Slot 5~14 為可修改範圍

    test_events = FeasibilityEngine(HHCP_FILE, SOL_FILE).generate_events(T_NOW)
    

    for event in test_events:
        solution = optimizer.handle_event(event)
        if solution and not solution.schedule_delta.empty:
            print("\n  [Schedule Delta]")
            print(solution.schedule_delta.to_string(index=False))