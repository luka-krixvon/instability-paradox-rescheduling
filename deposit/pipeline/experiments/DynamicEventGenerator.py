import pandas as pd
import random
from dataclasses import dataclass, field
from typing import List, Dict

# --- 資料結構定義 ---
@dataclass
class StrictEvent:
    event_type: str
    t_now: int
    description: str
    math_constraint: str
    data: Dict = field(default_factory=dict)

class FeasibilityEngine:
    """
    可行性引擎：用於驗證動態事件是否符合物理限制
    """
    def __init__(self, hhcp_path, solution_path):
        self.hhcp_path = hhcp_path
        self.solution_path = solution_path
        
        # 核心數據庫
        self.nurse_slots = {}  # {nurse_id: set(slots)}
        self.nurse_skills = {} # {nurse_id: set(skill_levels)}
        self.task_requirements = {} # {task_id: {skill: int, duration: int}}
        self.schedule = pd.DataFrame()
        self.nurse_load = {}   # {nurse_id:{slot_id:minutes_use}}
        self.TOTAL_SLOTS = 15
        
        self._load_and_parse_system()

    def _load_and_parse_system(self):
        """
        深度解析 HHCP 複雜格式，提取技能與時間限制。
        """
        # 1. 解析 HHCP 原始檔
        with open(self.hhcp_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [l.strip() for l in f.readlines()]

        section = None
        for line in lines:
            if line.startswith("nurse,"):
                section = "NURSE"
                continue
            elif line.startswith("task,"):
                section = "TASK"
                continue
            
            if not line or line.startswith("#"): continue
            
            parts = line.split(',')
            
            if section == "NURSE":
                try:
                    # 格式解析: nurse,x,y,class,...,#(slot), slot_list..., #(skill), skill_list
                    clean_parts = [p.strip() for p in parts if p.strip()]
                    if not clean_parts: continue
                    n_id = int(clean_parts[0])
                    
                    # 尋找 #(slot) 的位置
                    num_slots = int(parts[8])
                    slot_start = 9
                    slot_end = 9 + num_slots
                    self.nurse_slots[n_id] = {int(x) for x in clean_parts[slot_start : slot_end]}
                                    
                    # 解析 Skills
                    skill_count_idx = slot_end
                    num_skills = int(clean_parts[skill_count_idx])
                    skill_start = skill_count_idx + 1
                    self.nurse_skills[n_id] = {int(x) for x in clean_parts[skill_start : skill_start + num_skills]}
                
                except (ValueError, IndexError) as e:
                    print(f"DEBUG: Parse Error at Nurse {parts[0] if parts else 'unknown'}: {e}")

            elif section == "TASK":
                try:
                    # task, required skill, time, ...
                    t_id = int(parts[0])
                    req_skill = int(parts[1])
                    duration = int(float(parts[2])) # 有時是 float
                    self.task_requirements[t_id] = {
                        'skill': req_skill,
                        'duration': duration
                    }
                except:
                    pass

        # 2. 解析 Current Solution (x_old)
        try:
            self.schedule = pd.read_csv(self.solution_path)
            self.schedule['GlobalSlot'] = (self.schedule['Day'] - 1) * 3 + self.schedule['Shift']
            
            # 解析 TaskIDs 列表 (處理 "67;" 或 "67 68" 格式)
            def parse_tasks(x):
                clean = str(x).replace(';', ' ').replace(',', ' ')
                return [int(t) for t in clean.split() if t.isdigit()]
            
            self.schedule['TaskID_List'] = self.schedule['TaskIDs'].apply(parse_tasks)
        except Exception as e:
            print(f"Solution load error: {e}")

    def _build_load_cache(self):
        # 依照 NurseID 和 GlobalSlot 分組，加總 ServiceDuration 和 TravelTime
        load_df = self.schedule.groupby(['NurseID', 'GlobalSlot'])[['ServiceDuration', 'TravelTime']].sum().reset_index()
        
        for _, row in load_df.iterrows():
            nid = int(row['NurseID'])
            slot = int(row['GlobalSlot'])
            # 核心物理：負載 = 服務時間 + 路程時間
            total_minutes = row['ServiceDuration'] + row['TravelTime']
            
            if nid not in self.nurse_load: self.nurse_load[nid] = {}
            self.nurse_load[nid][slot] = total_minutes


    def check_reschedule_feasibility(self, task_ids: List[int], target_slot: int) -> bool:
        """
        第一性原理驗證：
        1. [Capability] 技能是否匹配？(質)
        2. [Availability] 該時段是否有排班？(質)
        3. [Capacity] 剩餘時間是否足夠容納任務 + 路程？(量 - 物理守恆)
        """
        
        # --- 1. 定義物理常數 ---
        SLOT_LIMIT = 240  # 假設每個 Slot 上限 240 分鐘
        AVG_TRAVEL_BUFFER = 15 # 啟發式路程懲罰 (分鐘)
        # 為什麼需要 Buffer？因為插入新任務必定產生新的路程。
        # 如果不加這個 Buffer，系統會填滿到 240 分鐘，導致實際路程超時。

        # --- 2. 計算新任務的純物理負載 (Pure Workload) ---
        new_task_duration = 0
        required_skills = set()
        
        for tid in task_ids:
            req = self.task_requirements.get(tid)
            if not req: continue
            new_task_duration += req['duration']
            required_skills.add(req['skill'])
            
        # 基礎物理檢核：如果單個任務組就超過時段上限，直接不可行
        if new_task_duration > SLOT_LIMIT:
            return False

        # --- 3. 尋找可行護理師 (Candidate Search) ---
        feasible_nurses = []
        
        # 取得該時段有排班的所有護理師
        # 優化：先過濾出在 target_slot 有班的人
        candidate_nurses = [n for n, slots in self.nurse_slots.items() if target_slot in slots]
        
        for n_id in candidate_nurses:
            # --- A. 驗證靜態能力 (Capability) ---
            nurse_skill_set = self.nurse_skills.get(n_id, set())
            # 使用集合運算 verify subset，比迴圈更高效且語意清晰
            if not required_skills.issubset(nurse_skill_set):
                continue
                
            # --- B. 驗證動態產能 (Capacity - The Missing Piece) ---
            # 取得該護理師當前的負載
            current_load = 0
            if n_id in self.nurse_load and target_slot in self.nurse_load[n_id]:
                current_load = self.nurse_load[n_id][target_slot]
            
            # 計算預估總負載
            # 這裡引入 Travel Penalty：如果他已經有任務，插入新任務需要額外路程
            travel_penalty = AVG_TRAVEL_BUFFER if current_load > 0 else 0
            
            total_projected_load = current_load + new_task_duration + travel_penalty
            
            # 物理不等式檢查
            if total_projected_load <= SLOT_LIMIT:
                feasible_nurses.append(n_id)
                # 找到一個可行解即可回傳 True (若只需驗證存在性)
                return True 
                
        return False

    def generate_events(self, t_now: int, seed=None):
        # 可重現性修補：固定亂數種子（優先序：seed 參數 > 環境變數 DCHHR_SEED > 不設定）。
        # 向後相容：未提供時行為與原本一致（不設種子）。
        import os
        if seed is None:
            _s = os.environ.get("DCHHR_SEED")
            seed = int(_s) if _s not in (None, "") else None
        if seed is not None:
            random.seed(seed)
            try:
                import numpy as _np
                _np.random.seed(seed)
            except Exception:
                pass
        events = []
        future_df = self.schedule[self.schedule['GlobalSlot'] > t_now]
        
        if future_df.empty: return events

        # --- 1. 生成護理師請假 (Nurse Absence) ---
        # 邏輯：隨機選一位未來有任務的護理師
        if not future_df.empty:
            target_row = future_df.sample(1).iloc[0]
            n_id = int(target_row['NurseID'])
            
            # 找出他未來所有的受影響時段
            future_impact = future_df[future_df['NurseID'] == n_id]
            affected_slots = sorted(future_impact['GlobalSlot'].unique().tolist())
            orphan_tasks = []
            for t_list in future_impact['TaskID_List']:
                orphan_tasks.extend(t_list)
            
            if orphan_tasks:
                events.append(StrictEvent(
                    event_type="NURSE_ABSENCE",
                    t_now=t_now,
                    description=f"護理師 {n_id} 因病請假，釋出 {len(orphan_tasks)} 個任務至孤兒池",
                    math_constraint=f"x[{n_id}, j, q] = 0 FORALL q >= {t_now+1}. Reassign J_orphan={orphan_tasks}",
                    data={"nurse_id": n_id, "orphan_tasks": orphan_tasks, "affected_slots": affected_slots}
                ))

        # --- 2. 生成病患改期 (Patient Reschedule) ---
        # 邏輯：必須通過 check_reschedule_feasibility
        attempts = 0
        while attempts < 20: # 嘗試 20 次尋找可行解
            attempts += 1
            if future_df.empty: break
            
            record = future_df.sample(1).iloc[0]
            p_id = int(record['PatientID'])
            old_slot = int(record['GlobalSlot'])
            task_ids = record['TaskID_List']
            
            # 隨機尋找一個新時段 (t_now 之後)
            candidates = [s for s in range(t_now + 1, self.TOTAL_SLOTS) if s != old_slot]
            random.shuffle(candidates)
            
            found_slot = -1
            for s in candidates:
                if self.check_reschedule_feasibility(task_ids, s):
                    found_slot = s
                    break
            
            if found_slot != -1:
                events.append(StrictEvent(
                    event_type="PATIENT_RESCHEDULE",
                    t_now=t_now,
                    description=f"病患 {p_id} 要求從 Slot {old_slot} 改至 Slot {found_slot} (經技能與資源驗證可行)",
                    math_constraint=f"sum(x[i, {p_id}, {found_slot}]) = 1, x[*, {p_id}, {old_slot}] = 0",
                    data={"patient_id": p_id, "task_ids": task_ids, "old_slot": old_slot, "new_slot": found_slot}
                ))
                break # 成功生成一個就停止
        
        # --- 3. 生成病患取消服務 (Patient Cancellation) ---
        # 邏輯：隨機選擇一個未來時段的任務，模擬病患突然不需要服務（例如住院或康復）
        if not future_df.empty:
            # 從未來的排程中隨機抽樣一個病患的服務記錄
            cancel_record = future_df.sample(1).iloc[0]
            p_id = int(cancel_record['PatientID'])
            target_slot = int(cancel_record['GlobalSlot'])
            tasks_to_cancel = cancel_record['TaskID_List']
            n_id = int(cancel_record['NurseID'])

            events.append(StrictEvent(
                event_type="PATIENT_CANCELLATION",
                t_now=t_now,
                description=f"病患 {p_id} 取消了 Slot {target_slot} 的所有服務 (共 {len(tasks_to_cancel)} 個任務)",
                # 數學約束：強制該時段該病患的所有任務指標變數為 0
                math_constraint=f"x[*, {p_id}, {target_slot}] = 0. Nurse {n_id} idle for this slot.",
                data={
                    "patient_id": p_id, 
                    "cancelled_tasks": tasks_to_cancel, 
                    "slot": target_slot,
                    "freed_nurse_id": n_id
                }
            ))


        return events

# --- 執行單元 ---
if __name__ == "__main__":
    hhcp_file = 'HHCP-patient-65-task-256-nurse-59.csv'
    sol_file = 'solution_schedule.csv'
    
    # 設定當前時間：Day 2 中午 (Slot 4)
    # 意味著 Slot 0,1,2,3,4 已成歷史，只能更動 Slot 5~14
    CURRENT_T_NOW = 4 
    
    print("正在初始化第一性原理引擎...")
    engine = FeasibilityEngine(hhcp_file, sol_file)
    
    print(f"\n--- 生成動態事件 (t_now = Slot {CURRENT_T_NOW}) ---")
    generated_events = engine.generate_events(CURRENT_T_NOW)
    
    if not generated_events:
        print("未生成事件 (可能系統過於緊繃，無可行改期空間)")
    
    for i, e in enumerate(generated_events, 1):
        print(f"\n[Event {i}] {e.event_type}")
        print(f"{e}")
        print(f"  > {e.description}")
        print(f"  > Math Constraints: {e.math_constraint}")