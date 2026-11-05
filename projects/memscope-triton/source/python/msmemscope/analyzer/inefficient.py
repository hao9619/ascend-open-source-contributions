# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import importlib

import csv
import os
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .base import BaseAnalyzer, AnalysisConfig
from .utility_function import safe_convert_int

check_packages = [
    "sqlite3",
]

for package in check_packages:
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"ERROR: Msleaks import {package} failed! Please check it")


MAX_UINT64 = (1 << 64) - 1
MIN_EVENTS_NUM = 2  # state buffer中需要最少有两条内存访问记录
LAST_EVENTS_NUM = 1  # 上一个元素
ATTR_INDEX = 9  # attr在csv中的列索引


class InefficientConfig(AnalysisConfig):
    """低效代码分析配置"""

    def __init__(
        self, input_path: str, mem_size: int = 0, inefficient_type: List[str] = None, idle_threshold: int = 3000
    ):
        super().__init__(input_path=input_path)
        self.mem_size = mem_size
        if inefficient_type is None:
            self.inefficient_type: List[str] = ["early_allocation", "late_deallocation", "temporary_idleness"]
        else:
            self.inefficient_type = inefficient_type
        self.idle_threshold = idle_threshold
        self.__post_init__()

    def __post_init__(self):
        super().__post_init__()
        # 需要额外校验是否可写入
        path = Path(self.input_path)
        if not os.access(str(path), os.W_OK):
            raise PermissionError(f"Write permission denied: {path}")

        def _validate_mem_size(v):
            if not isinstance(v, int) or v < 0:
                raise ValueError("Memory size must be a non-negative integer")
            return v

        def _validate_inefficient_type(v):
            types = [v] if isinstance(v, str) else v
            if not isinstance(types, list) or not types:
                raise ValueError("Inefficient type must be a non-empty list or a single valid string")
            valid_types = {"early_allocation", "late_deallocation", "temporary_idleness"}
            for t in types:
                if t not in valid_types:
                    raise ValueError(f"Invalid inefficient type: {t}, valid values are {list(valid_types)}")
            return types

        def _validate_idle_threshold(v):
            if not isinstance(v, int) or v <= 0:
                raise ValueError("Idle threshold must be a positive integer")
            return v

        validators = {
            "mem_size": _validate_mem_size,
            "inefficient_type": _validate_inefficient_type,
            "idle_threshold": _validate_idle_threshold,
        }
        self.mem_size = validators["mem_size"](self.mem_size)
        self.inefficient_type = validators["inefficient_type"](self.inefficient_type)
        self.idle_threshold = validators["idle_threshold"](self.idle_threshold)


@dataclass
class OriginEvent:
    event_id: int
    event: str
    event_type: str
    name: str
    timestamp: str
    pid: str
    tid: str
    device: str
    ptr: str
    attr: str
    cpp_call_stack: str
    python_call_stack: str
    row_num: str


@dataclass
class IneffEvent:
    # 这里只采集和低效显存识别相关的数据
    event_id: int
    event: str
    event_type: str
    name: str
    pid: int
    allocation_id: int
    size: int
    api_id: int
    inefficient_type: List[str]
    row_num: int
    access_type: str


@dataclass
class PidState:
    # 包含低效显存相关信息
    api_tmp: List[IneffEvent]
    api_id: int
    malloc_api_tmp_id: int
    free_api_tmp_id: int
    is_operation_started: bool


class InefficientAnalyzer(BaseAnalyzer):
    def __init__(self):
        self.memory_state: Dict[str, List[IneffEvent]] = defaultdict(
            list
        )  # 参考MemoryStateKey实现，key是 pid, event_id 的拼接字符串
        self.inefficient_results: Dict[str, List[str]] = defaultdict(list)  # key是row_num, pid, event_id拼接字符串
        self.pta_pid_states: Dict[int, PidState] = defaultdict(PidState)
        self.atb_pid_states: Dict[int, PidState] = defaultdict(PidState)
        self.headers = [
            "ID",
            "Event",
            "Event Type",
            "Name",
            "Timestamp(ns)",
            "Process Id",
            "Thread Id",
            "Device Id",
            "Ptr",
            "Attr",
            "Call Stack(Python)",
            "Call Stack(C)",
        ]
        self.data_format: str
        self.config: InefficientConfig

    def analyze(self, config: InefficientConfig):
        self.config = config
        path = Path(config.input_path)
        # 根据id排序文件
        origin_events = self.file_to_events(path)
        origin_events.sort(key=lambda origin_event: origin_event.event_id)

        # 处理事件流
        for origin_event in origin_events:
            self.event_handle(origin_event)

        # 写回原文件以及清除缓存
        self.write_back_file(path)

    def file_to_events(self, input_path: Path) -> List[OriginEvent]:
        file_suffix = input_path.suffix.lower()
        if file_suffix == ".csv":
            origin_events = self._read_csv_file(input_path)
            self.data_format = "csv"
        elif file_suffix == ".db":
            origin_events = self._read_db_file(input_path)
            self.data_format = "db"
        else:
            raise ValueError(f"Unsupported file format: {file_suffix}.")

        return origin_events

    def _read_csv_file(self, input_path: Path) -> List[OriginEvent]:
        csv_events = []
        with open(input_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing_headers = [h for h in self.headers if h not in reader.fieldnames]
            if missing_headers:
                raise ValueError(f"ERROR: CSV format error: missing headers {missing_headers}")
            print("INFO: Connected to CSV file")
            for row_num, row in enumerate(reader, start=2):
                event_id = safe_convert_int(row["ID"], context=f"line {row_num}")
                if event_id is None:
                    continue
                event_obj = OriginEvent(
                    event_id=event_id,
                    event=row["Event"],
                    event_type=row["Event Type"],
                    name=row["Name"],
                    timestamp=row["Timestamp(ns)"],
                    pid=row["Process Id"],
                    tid=row["Thread Id"],
                    device=row["Device Id"],
                    ptr=row["Ptr"],
                    attr=row["Attr"],
                    python_call_stack=row["Call Stack(Python)"],
                    cpp_call_stack=row["Call Stack(C)"],
                    row_num=row_num,
                )
                csv_events.append(event_obj)

        print("INFO: CSV file read successfully")
        return csv_events

    def _read_db_file(self, input_path: Path) -> List[OriginEvent]:
        db_events = []
        conn = None  # 数据库连接（后续需确保关闭）
        cursor = None  # 游标
        try:
            conn = sqlite3.connect(str(input_path))
            cursor = conn.cursor()
            print(f"INFO: Connected to {input_path.name} (table: memscope_dump)")

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memscope_dump'")
            if not cursor.fetchone():
                raise ValueError(f"Table 'memscope_dump' not found in database {input_path}")

            cursor.execute("PRAGMA table_info(memscope_dump)")
            db_columns = [col[1] for col in cursor.fetchall()]
            missing_headers = [h for h in self.headers if h not in db_columns]
            if missing_headers:
                raise ValueError(f"ERROR: DB format error: missing headers {missing_headers}")

            cursor.execute("SELECT * FROM memscope_dump")
            db_rows = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            row_dict_list = [dict(zip(column_names, row)) for row in db_rows]  # 转换为字典列表（与 CSV 格式对齐）

            for row_num, row in enumerate(row_dict_list, start=1):
                # 跳过ID无法解析为int的脏记录行
                event_id = safe_convert_int(row["ID"], context=f"line {row_num}")
                if event_id is None:
                    continue
                # 构建 OriginEvent 对象（参数与 CSV 完全一致）
                event_obj = OriginEvent(
                    event_id=event_id,
                    event=str(row["Event"]).strip(),
                    event_type=str(row["Event Type"]).strip(),
                    name=str(row["Name"]).strip(),
                    timestamp=str(row["Timestamp(ns)"]),
                    pid=str(row["Process Id"]).strip(),
                    tid=str(row["Thread Id"]).strip(),
                    device=str(row["Device Id"]).strip(),
                    ptr=str(row["Ptr"]).strip(),
                    attr=str(row["Attr"]).strip(),
                    python_call_stack=str(row["Call Stack(Python)"]).strip(),
                    cpp_call_stack=str(row["Call Stack(C)"]).strip(),
                    row_num=row_num,
                )
                db_events.append(event_obj)

        except sqlite3.OperationalError as e:
            print(f"ERROR: Database operation failed → {str(e)}")
        except Exception as e:
            print(f"ERROR: Failed to read database → {str(e)}")
        finally:
            # 关闭数据库连接
            if cursor:
                cursor.close()
            if conn:
                conn.close()

        print("INFO: DB file read successfully")
        return db_events

    def event_handle(self, origin_event: OriginEvent):
        # 处理OP_LAUNCH事件
        ineff_event = self._event_convert(origin_event)
        if ineff_event.event == "OP_LAUNCH":
            if ineff_event.event_type in ("ATEN_START", "ATEN_END", "TRITON_START", "TRITON_END"):
                pid_states = self.pta_pid_states
            else:
                pid_states = self.atb_pid_states
            self._pid_state_init(ineff_event.pid, pid_states)
            self._handle_oplaunch_event(ineff_event, pid_states)
            return
        # 处理MALLOC FREE ACCESS事件
        if ineff_event.event in {"MALLOC", "FREE", "ACCESS"}:
            if ineff_event.event != "ACCESS" and ineff_event.event_type not in {"PTA", "ATB"}:
                return
            # 对于PTA access操作中的name会含有torch字段;atb则没有
            if ineff_event.event_type == "PTA" or (ineff_event.event == "ACCESS" and ineff_event.access_type == "PTA"):
                pid_states = self.pta_pid_states
            else:
                pid_states = self.atb_pid_states

            self._pid_state_init(ineff_event.pid, pid_states)
            self._handle_memory_event(ineff_event, pid_states)  # state通过成员变量memory_state获取，无需传入
            return

    def _pid_state_init(self, pid: int, pid_states: Dict[int, PidState]):
        if pid not in pid_states:
            state = PidState(
                api_tmp=[],
                api_id=0,
                malloc_api_tmp_id=MAX_UINT64,
                free_api_tmp_id=MAX_UINT64,
                is_operation_started=False,
            )
            pid_states[pid] = state

    def _handle_oplaunch_event(self, ineff_event: IneffEvent, pid_states: Dict[int, PidState]):
        event_sub_type = ineff_event.event_type
        pid = ineff_event.pid
        pid_state = pid_states[pid]

        if event_sub_type in ("ATB_START", "ATEN_START", "TRITON_START"):
            pid_state.is_operation_started = True
            pid_state.api_id += 1
            return

        if event_sub_type in ("ATB_END", "ATEN_END", "TRITON_END"):
            pid_state.is_operation_started = False
            self._classify_event_tmp(pid, pid_states)
            return

    def _event_convert(self, origin_event: OriginEvent):
        # 将origin_event类转化为低效显存识别的ineff_event类
        raw_attr = origin_event.attr
        # CSV短行缺失字段为None,统一按空串处理,避免对None调用.strip()抛异常
        if not raw_attr or not raw_attr.strip():
            allocation_id = -1
            size = -1
            api_id = -1
            access_type = ""
        else:
            raw_attr = raw_attr.strip()
            attr_str = raw_attr[1:-1].strip()
            attr_pairs = attr_str.split(',')
            attr_dict = {}

            for pair in attr_pairs:
                key_value = pair.strip().split(':', 1)
                if len(key_value) == 2:
                    key = key_value[0].strip().strip('"')  # 去除双引号
                    value = key_value[1].strip().strip('"')
                    attr_dict[key] = value

            # 容错解析:allocation_id/size非数字时降级为-1(与字段缺失等价),不中断分析
            alloc_id_val = safe_convert_int(
                attr_dict.get('allocation_id', '-1'), context=f"line {origin_event.row_num}, allocation_id"
            )
            size_val = safe_convert_int(attr_dict.get('size', '-1'), context=f"line {origin_event.row_num}, size")
            allocation_id = alloc_id_val if alloc_id_val is not None else -1
            size = size_val if size_val is not None else -1
            access_type = attr_dict.get('type', "")
            api_id = -1

        ineff_obj = IneffEvent(
            event_id=origin_event.event_id,
            event=origin_event.event,
            event_type=origin_event.event_type,
            name=origin_event.name,
            pid=origin_event.pid,
            allocation_id=allocation_id,
            size=size,
            api_id=api_id,
            inefficient_type=[],
            row_num=origin_event.row_num,
            access_type=access_type,
        )
        return ineff_obj

    def _handle_memory_event(self, ineff_event: IneffEvent, pid_states: Dict[int, PidState]):
        tmp_pid = ineff_event.pid
        pid_state = pid_states[tmp_pid]
        if pid_state.is_operation_started is False:
            pid_state.api_id += 1
            ineff_event.api_id = pid_state.api_id
            pid_state.api_tmp.append(ineff_event)
            key = f"{ineff_event.pid}_{ineff_event.allocation_id}"
            self.memory_state[key].append(ineff_event)
            self._classify_event_tmp(tmp_pid, pid_states)
        else:
            ineff_event.api_id = pid_state.api_id
            pid_state.api_tmp.append(ineff_event)
            key = f"{ineff_event.pid}_{ineff_event.allocation_id}"
            self.memory_state[key].append(ineff_event)

        self._inefficient_analysis(ineff_event, pid_states)

    def _classify_event_tmp(self, pid: int, pid_states: Dict[int, PidState]):
        pid_state = pid_states[pid]
        if len(pid_state.api_tmp) == 0:
            return
        local_events = pid_state.api_tmp
        pid_state.api_tmp = []
        has_malloc = False
        has_free = False

        for tmp_event in local_events:
            if tmp_event.event == "MALLOC":
                has_malloc = True
            if tmp_event.event == "FREE":
                has_free = True
        if has_malloc:
            pid_state.malloc_api_tmp_id = pid_state.api_id

        if has_free:
            pid_state.free_api_tmp_id = pid_state.api_id

    def _inefficient_analysis(self, ineff_event: IneffEvent, pid_states: Dict[int, PidState]):
        # 存在开头不是malloc的情况
        key = f"{ineff_event.pid}_{ineff_event.allocation_id}"
        if len(self.memory_state[key]) != 0:
            front_element = self.memory_state[key][0]
            if front_element.event != "MALLOC":
                return

        if ineff_event.event == "ACCESS":
            self._temporary_idleness(ineff_event)
            if pid_states[ineff_event.pid].free_api_tmp_id == MAX_UINT64:
                return
            self._early_allocation(ineff_event, pid_states)

        if ineff_event.event == "FREE" and pid_states[ineff_event.pid].malloc_api_tmp_id != MAX_UINT64:
            self._late_deallocation(ineff_event, pid_states)

    def _early_allocation(self, ineff_event: IneffEvent, pid_states: Dict[int, PidState]):
        key = f"{ineff_event.pid}_{ineff_event.allocation_id}"
        state = self.memory_state[key]
        if "early_allocation" in state[0].inefficient_type:
            return

        eventsLen = len(state)
        first_access_api_id = MAX_UINT64
        malloc_api_id = 0
        pid_state = pid_states[ineff_event.pid]

        # 1.找到MALLOC的API值。2.找到第一个API值不等于MALLOC的API的ACCESS，找到即结束
        for i in range(eventsLen):
            if first_access_api_id < MAX_UINT64:
                break
            if state[i].event == "MALLOC":
                malloc_api_id = state[i].api_id
            elif state[i].event == "ACCESS" and state[i].api_id != malloc_api_id:
                first_access_api_id = state[i].api_id

        # 1.如果没找到第一个ACCESS值，说明此时ACCESS API与MALLOC的相等
        # 2.如果MALLOC和最近的FREE事件在同一个API中，则无法交换。
        # ACCESS所在API此时未结束，还没有将此时API更新至最近MALLOC或者FREE的ID，所以FA 和 FREE事件API值不可能相等
        if first_access_api_id == MAX_UINT64 or malloc_api_id == pid_state.free_api_tmp_id:
            return

        # 如果FREE的API Id在FA与MALLOC之间，则判断MALLOC内存块为过早申请
        # 根据之前识别低效显存的设计,最后的结果都会
        if first_access_api_id > pid_state.free_api_tmp_id and malloc_api_id < pid_state.free_api_tmp_id:
            if state[0].size >= self.config.mem_size and "early_allocation" in self.config.inefficient_type:
                state[0].inefficient_type.append("early_allocation")
                key = f"{state[0].row_num}_{ineff_event.pid}_{state[0].event_id}"
                self.inefficient_results[key].append("early_allocation")

    def _late_deallocation(self, ineff_event: IneffEvent, pid_states: Dict[int, PidState]):
        key = f"{ineff_event.pid}_{ineff_event.allocation_id}"
        state = self.memory_state[key]
        # 1.找到LAST ACCESS。2.判断FREE API与LA API之间有无MALLOC API
        eventsLen = len(state)
        last_access_api_id = MAX_UINT64
        # 此时内存块需要具有MALLOC、ACCESS、FREE的完整事件，且在state->events中具有顺序，因此events中倒数第二个事件即为LA
        if eventsLen >= MIN_EVENTS_NUM:
            if state[eventsLen - MIN_EVENTS_NUM].event == "ACCESS":
                last_access_api_id = state[eventsLen - MIN_EVENTS_NUM].api_id

        pid_state = pid_states[ineff_event.pid]
        # 没有找到LA, 或者LA在当前API中
        if last_access_api_id in (MAX_UINT64, pid_state.api_id):
            return

        # 如果LA与FREE中存在MALLOC的API，则说明为过迟释放
        if last_access_api_id < pid_state.malloc_api_tmp_id and pid_state.api_id > pid_state.malloc_api_tmp_id:
            if state[0].size >= self.config.mem_size and "late_deallocation" in self.config.inefficient_type:
                state[0].inefficient_type.append("late_deallocation")
                key = f"{state[0].row_num}_{ineff_event.pid}_{state[0].event_id}"
                self.inefficient_results[key].append("late_deallocation")

    def _temporary_idleness(self, ineff_event: IneffEvent):
        key = f"{ineff_event.pid}_{ineff_event.allocation_id}"
        front_element = self.memory_state[key][0]
        if "temporary_idleness" in front_element.inefficient_type:
            return

        state = self.memory_state[key]
        eventsLen = len(state)

        # 当events不足2个或者倒数第二个事件不为ACCESS时，不用判断，此时最后一个events必定为ACCESS
        if eventsLen < MIN_EVENTS_NUM or state[eventsLen - MIN_EVENTS_NUM].event != "ACCESS":
            return
        # 当最后一个API值大于倒数第2个API值 并且 其差值大于阈值时，判为临时闲置
        if (
            state[eventsLen - LAST_EVENTS_NUM].api_id > state[eventsLen - MIN_EVENTS_NUM].api_id
            and state[eventsLen - LAST_EVENTS_NUM].api_id - state[eventsLen - MIN_EVENTS_NUM].api_id
            > self.config.idle_threshold
        ):
            if state[0].size >= self.config.mem_size and "temporary_idleness" in self.config.inefficient_type:
                state[0].inefficient_type.append("temporary_idleness")
                key = f"{state[0].row_num}_{ineff_event.pid}_{state[0].event_id}"
                self.inefficient_results[key].append("temporary_idleness")

    def write_back_file(self, input_path: Path):
        file_suffix = input_path.suffix.lower()
        if file_suffix == ".csv":
            self._write_back_csv(input_path)
        elif file_suffix == ".db":
            self._write_back_db(input_path)

    def _write_back_csv(self, input_path: Path):
        with open(input_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            data = list(reader)

        print(f"INFO: Connecting to csv {input_path.name} to write results")
        # 低效显存写回attr
        for inefficient_key, inefficient_value in self.inefficient_results.items():
            parts = inefficient_key.split('_', 2)
            row_num_str, pid_str, id_str = parts
            row_index = int(row_num_str) - 1

            # 读取旧attr并更新
            row_data = data[row_index]
            original_attr = row_data[ATTR_INDEX].strip()
            updated_attr_str = self._update_attr(original_attr, inefficient_value)
            row_data[ATTR_INDEX] = updated_attr_str

        # 提交所有更新
        with open(input_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(data)
        print(f"INFO: All results have been successfully written to csv {input_path.name}")

    def _write_back_db(self, input_path: Path, db_table: str = "leaks_dump"):
        ALLOWED_TABLES = {"leaks_dump"}
        if db_table not in ALLOWED_TABLES:
            raise ValueError("Invalid table name: memscope_dump")
        conn = None
        cursor = None
        try:
            conn = sqlite3.connect(str(input_path))
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")  # 启用WAL模式 加快文件写入
            cursor.execute(
                "PRAGMA synchronous=NORMAL;"
            )  # 降低磁盘同步次数 如果遇到断电、崩溃可能会有风险 但是会大幅提高速度
            cursor.execute("PRAGMA cache_size=-20000;")  # 增大内存缓存大小
            print(f"INFO: Connecting to database {input_path.name} to write results")

            for inefficient_key, inefficient_value in self.inefficient_results.items():
                parts = inefficient_key.split('_', 2)
                row_num_str, pid_str, id_str = parts
                target_id = int(id_str)
                target_pid = int(pid_str)
                row_num = int(row_num_str)

                # 数据库数据不推荐使用行号写入 这里用pid + id定位唯一行
                cursor.execute(
                    "SELECT Attr FROM memscope_dump WHERE ID = ? AND `Process Id` = ?", (target_id, target_pid)
                )

                # 读取旧attr并更新
                result = cursor.fetchone()
                if not result:
                    print(
                        f"WARNING: No record with ID={target_id} found in the database (line number {row_num}), please check"
                    )
                    continue
                original_attr = result[0]
                updated_attr_str = self._update_attr(original_attr, inefficient_value)

                # 执行数据库更新（使用参数化查询避免SQL注入）
                cursor.execute(
                    "UPDATE memscope_dump SET Attr = ? WHERE ID = ? AND `Process Id` = ?",
                    (updated_attr_str, target_id, target_pid),
                )

            # 提交所有更新
            conn.commit()
            print(f"INFO: All results have been successfully written to database {input_path.name}")

        except sqlite3.OperationalError as e:
            print(f"ERROR: Database operation failed → {str(e)} (check if file is a valid SQLite database)")
            if conn:
                conn.rollback()  # 出错时回滚事务,删除错误缓存
        except Exception as e:
            print(f"ERROR: Failed to write to database → {str(e)}")
            if conn:
                conn.rollback()
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def _update_attr(self, original_attr: str, inefficient_value: List[str]):
        # 把低效显存结果写回
        attr_str = original_attr.strip()
        if attr_str.startswith('{') and attr_str.endswith('}'):
            attr_content = attr_str[1:-1].strip()
        else:
            attr_content = attr_str

        # 解析为字典
        attr_dict = {}
        if attr_content:
            for pair in [p.strip() for p in attr_content.split(',')]:
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    attr_dict[key.strip().strip('"')] = value.strip().strip('"')

        # 更新inefficient_type
        attr_dict['inefficient_type'] = ','.join(inefficient_value)

        # 转回字符串格式 csv和db格式略有不同
        if self.data_format == "db":
            updated_attr_pairs = [f'"{k}":"{v}"' for k, v in attr_dict.items()]
        else:
            updated_attr_pairs = [f'{k}:{v}' for k, v in attr_dict.items()]
        return '{' + ','.join(updated_attr_pairs) + '}'


# 对外暴露的便捷低效显存识别函数
def check_inefficient(
    input_path: str, mem_size: int = 0, inefficient_type: List[str] = None, idle_threshold: int = 3000
):
    if inefficient_type is None:
        inefficient_type: List[str] = ["early_allocation", "late_deallocation", "temporary_idleness"]

    config = InefficientConfig(
        input_path=input_path,
        mem_size=mem_size,
        inefficient_type=inefficient_type,
        idle_threshold=idle_threshold,
    )

    return InefficientAnalyzer().analyze(config)
