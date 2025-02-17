import json
import pickle
import copy
from typing import Optional, List, Dict, Tuple
from matplotlib import pyplot as plt
from perf_estimator.utilis.utilis import format_memory
from .blocks import SnapshotTraceBlock, ActivityMemory, SegmentMemory


class SnapshotAnalyser:
    def __init__(self, snapshot_path: str):
        self._data: Optional[Dict] = None
        self._load(snapshot_path)
        self._trace_blocks: Dict[int, List[ActivityMemory]] = {}
        self._segment_blocks: Dict[int, List[SegmentMemory]] = {}
        self._trace_blocks, self._segment_blocks = self._build_trace_blocks()

    @property
    def activity_blocks(self) -> Dict[int, List[ActivityMemory]]:
        return self._trace_blocks

    @property
    def segment_blocks(self) -> Dict[int, List[SegmentMemory]]:
        return self._segment_blocks

    @property
    def segments(self) -> List[Dict]:
        return self._data["segments"]

    @property
    def device_traces(self) -> List[Dict]:
        if (
            len(self._data["device_traces"]) >= 1
            and len(self._data["device_traces"][0]) > 0
        ):
            return self._data["device_traces"][0]
        else:
            return self._data["device_traces"][1]

    @property
    def allocator_setting(self) -> Dict:
        return self._data["allocator_setting"]

    def _load(self, path: str) -> None:
        with open(path, "rb") as f:
            self._data = pickle.load(f)

    def to_json(self) -> str:
        return json.dumps(self._data, indent=4)

    def save_json(self, json_path: str) -> None:
        _data = copy.deepcopy(self._data)
        with open(json_path, "w") as f:
            json.dump(_data, f, indent=4)

    def group_by_file(self, file_path: str):
        code_no_sorted = {}
        activiies = list(self.activity_blocks.values())
        for activity in activiies:
            print(f"Number of activities: {len(code_no_sorted.keys())}")
            for act in activity:
                frames = act._start.frames
                frames.reverse()
                for frame in frames:
                    if file_path in frame["filename"]:
                        code_number = frame["line"]
                        print(code_number)
                        if code_number not in code_no_sorted.keys():
                            code_no_sorted[str(code_number)] = []
                        code_no_sorted[str(code_number)].append(act)
                        break

        return code_no_sorted




    def _build_trace_blocks(self) -> Tuple:
        blocks = {
            "trace": {"collective": {}, "active": {}, "func": ActivityMemory},
            "segment": {"collective": {}, "active": {}, "func": SegmentMemory},
        }
        _current_used_collective_blocks = None
        _current_used_active_blocks = None
        _field_name = None
        for trace in self.device_traces:
            _trace_block: SnapshotTraceBlock = SnapshotTraceBlock(trace)
            if _trace_block.action in ["alloc", "free_requested", "free_completed"]:
                if _trace_block.action == "free_requested":
                    # todo: leave it for now
                    continue
                _field_name = "trace"
            elif _trace_block.action in ["segment_alloc", "segment_free"]:
                _field_name = "segment"
            else:
                # todo: should handle oom and snapshot later
                continue

            _conf_segment = blocks[_field_name]
            collective_blocks = _conf_segment["collective"]
            active_blocks = _conf_segment["active"]
            mem_func = _conf_segment["func"]

            _address = _trace_block.address
            if _address not in active_blocks.keys():
                active_blocks[_address] = mem_func(_trace_block)
            else:
                active_blocks[_address].set_end_block(_trace_block)
                _pop_block = active_blocks.pop(_address)
                if _pop_block.alloc_time not in collective_blocks.keys():
                    collective_blocks[_pop_block.alloc_time] = []
                collective_blocks[_pop_block.alloc_time].append(_pop_block)

        for _type in blocks.keys():
            _active = blocks[_type]["active"]
            for _block in _active.values():
                if _block.alloc_time not in blocks[_type]["collective"].keys():
                    blocks[_type]["collective"][_block.alloc_time] = []
                blocks[_type]["collective"][_block.alloc_time].append(_block)

        ordered_trace_blocks = dict(
            sorted(blocks["trace"]["collective"].items(), key=lambda item: item[0])
        )
        ordered_segment_blocks = dict(
            sorted(blocks["segment"]["collective"].items(), key=lambda item: item[0])
        )
        return ordered_trace_blocks, ordered_segment_blocks

    def breakdown_into_multiple_sections(self):
        sections = {"model": [], "forward": [], "backward": [], "optim": []}
        iterations = {}
        model = False
        forward = False
        backward = False
        optim = False
        field_name = None
        for blocks in self._trace_blocks.values():
            for block in blocks:
                _name = block.ops_name
                if "to_dtype_layout" in _name and not model:
                    model = True
                    forward = False
                    backward = False
                    optim = False
                    field_name = "model"
                elif model and not forward and _name != "to_dtype_layout":
                    model = True
                    forward = True
                    backward = False
                    optim = False
                    field_name = "forward"
                elif model and forward and not backward and "Backward" in _name:
                    model = True
                    forward = True
                    backward = True
                    optim = False
                    field_name = "backward"
                elif (
                    model
                    and forward
                    and backward
                    and not optim
                    and "zeros_like" in _name
                ):
                    model = True
                    forward = True
                    backward = True
                    optim = True
                    field_name = "optim"
                    if "to_dtype_layout" in _name:
                        break

                sections[field_name].append(block)
        return sections

    def forward_backward_memory_sequence(self):
        breakdown_memory = self.breakdown_into_multiple_sections()
        memory_trace = []
        for _forward in breakdown_memory["forward"]:
            memory_trace.append(
                (_forward.alloc_time, _forward.ops_name, _forward.bytes, "F")
            )
            if _forward.free_time:
                memory_trace.append(
                    (_forward.free_time, _forward.ops_name, -_forward.bytes, "F")
                )
        for _backward in breakdown_memory["backward"]:
            memory_trace.append(
                (_backward.alloc_time, _backward.ops_name, _backward.bytes, "B")
            )
            if _backward.free_time:
                memory_trace.append(
                    (_backward.free_time, _backward.ops_name, -_backward.bytes, "B")
                )
        memory_trace = sorted(memory_trace, key=lambda x: x[0])
        return memory_trace

    def gpu_and_segment_memory_changes_data(self):
        seg_mem_by_time = {}
        trace_mem_by_time = {}
        for seg_block in self._segment_blocks.values():
            for seg in seg_block:
                seg_mem_by_time[seg.alloc_time] = seg.bytes
                if seg.free_time:
                    seg_mem_by_time[seg.free_time] = -seg.bytes
        for trace_block in self._trace_blocks.values():
            for trace in trace_block:
                trace_mem_by_time[trace.alloc_time] = trace.bytes
                if trace.free_time:
                    trace_mem_by_time[trace.free_time] = -trace.bytes

        ordered_seg = dict(sorted(seg_mem_by_time.items(), key=lambda item: item[0]))
        ordered_trace = dict(
            sorted(trace_mem_by_time.items(), key=lambda item: item[0])
        )
        return ordered_seg, ordered_trace

    def gpu_and_segment_max_memory_changes_data(self):
        seg_mems, trace_mems = self.gpu_and_segment_memory_changes_data()
        max_seg_mem = 0
        max_trace_mem = 0
        max_seg_mem_list = []
        max_trace_mem_list = []
        for time, size in seg_mems.items():
            max_seg_mem += size
            max_seg_mem_list.append(max_seg_mem)
        for time, size in trace_mems.items():
            max_trace_mem += size
            max_trace_mem_list.append(max_trace_mem)
        return max_seg_mem_list, max_trace_mem_list

    def gpu_and_segment_in_same_time_length(self):
        seg_mems, trace_mems = self.gpu_and_segment_memory_changes_data()
        seg_times = list(seg_mems.keys())
        trace_times = list(trace_mems.keys())
        total_timeline = list(set(seg_times + trace_times))
        total_timeline.sort()
        mem_dict = {"time": [], "seg": [], "trace": []}
        max_seg_mem = 0
        max_trace_mem = 0
        for time in total_timeline:
            max_seg_mem += seg_mems.get(time, 0)
            max_trace_mem += trace_mems.get(time, 0)
            mem_dict["time"].append(time)
            mem_dict["seg"].append(max_seg_mem)
            mem_dict["trace"].append(max_trace_mem)
        return mem_dict

    def fetch_gpu_segment_max_changes_directly(self) -> Tuple[List[int], List[int]]:
        max_trace = 0
        max_seg = 0
        trace_list = []
        seg_list = []
        for trace in self.device_traces:
            if trace["action"] == "alloc":
                max_trace += trace["size"]
            elif trace["action"] == "free_completed":
                max_trace -= trace["size"]
            elif trace["action"] == "segment_alloc":
                max_seg += trace["size"]
            elif trace["action"] == "segment_free":
                max_seg -= trace["size"]
            seg_list.append(max_seg)
            trace_list.append(max_trace)
        return seg_list, trace_list

    def analysis_relationship_between_request_and_max_sgement_memory(
        self, max_allowed_memory_gb: int
    ):
        class Segment:
            def __init__(self, tr):
                self._data = tr
                self.total = 0
                self.used = 0
                self.remaining = 0
                self.blocks = {}
                self.start_addr = None
                self.end_addr = None
                self.index = 0
                self._update()

            def _update(self):
                self.total = self._data["size"]
                self.remaining = self.total
                self.start_addr = self._data["addr"]
                self.end_addr = self.start_addr + self.total

            def insert_block(self, block):
                if self.start_addr <= block["addr"] < self.end_addr:
                    self.blocks[block["addr"]] = block
                    self.remaining -= block["size"]
                    self.used += block["size"]
                    return True
                return False

            def remove_block(self, block):
                if block["addr"] in self.blocks:
                    self.remaining += self.blocks[block["addr"]]["size"]
                    self.used -= self.blocks[block["addr"]]["size"]
                    del self.blocks[block["addr"]]
                    return True
                return False

            def check(self, block):
                if block["addr"] in self.blocks:
                    return True
                return False

            def max_block(self):
                available_blocks = self.get_available_blocks()
                if len(available_blocks) == 0:
                    return 0
                return max([_block["size"] for _block in available_blocks])

            def get_available_blocks(self):
                """
                Returns a list of available (free) blocks within the segment.
                Each available block is represented as a dictionary with 'addr' and 'size'.
                """
                available = []

                # If no blocks are allocated, the entire segment is available
                if not self.blocks:
                    available.append({"addr": self.start_addr, "size": self.total})
                    return available

                # Sort the allocated blocks by their starting address
                sorted_blocks = sorted(self.blocks.values(), key=lambda b: b["addr"])

                # Check for available space before the first allocated block
                first_block = sorted_blocks[0]
                if self.start_addr < first_block["addr"]:
                    gap_size = first_block["addr"] - self.start_addr
                    available.append({"addr": self.start_addr, "size": gap_size})

                # Check for gaps between consecutive allocated blocks
                for i in range(len(sorted_blocks) - 1):
                    current_end = sorted_blocks[i]["addr"] + sorted_blocks[i]["size"]
                    next_start = sorted_blocks[i + 1]["addr"]
                    if current_end < next_start:
                        gap_size = next_start - current_end
                        available.append({"addr": current_end, "size": gap_size})

                # Check for available space after the last allocated block
                last_block = sorted_blocks[-1]
                last_end = last_block["addr"] + last_block["size"]
                if last_end < self.end_addr:
                    gap_size = self.end_addr - last_end
                    available.append({"addr": last_end, "size": gap_size})

                _new_available = []
                for _available in available:
                    _available["is_free"] = True
                    _new_available.append(_available)
                return _new_available

            def plot_memory_state(self):
                """Plot the memory blocks and save the figure."""
                blocks = copy.deepcopy(list(self.blocks.values()))
                blocks.extend(self.get_available_blocks())

                total_size = self.total
                fig, ax = plt.subplots(figsize=(10, 2))
                current_position = 0

                for block in blocks:
                    start = block["addr"] - self.start_addr
                    size = block["size"]
                    is_free = block.get("is_free", False)
                    color = "green" if is_free else "red"
                    ax.barh(
                        0, size, left=start, height=0.5, color=color, edgecolor="black"
                    )
                    # Label the block
                    ax.text(
                        start + size / 2,
                        0,
                        f"{format_memory(size)}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white",
                    )

                ax.set_xlim(0, total_size)
                ax.set_ylim(-0.5, 1)
                ax.axis("off")
                ax.set_title(f"Segment {self.index} Memory Distribution")
                # plt.tight_layout()
                return plt

            def __repr__(self):
                return f"Index: {self.index}: {self.used}/{self.total} (max: {self.max_block()})"

        segments = {}
        max_allowed_memory = max_allowed_memory_gb * 1024 * 1024 * 1024
        max_block = 0
        record = []
        _p_request = 0
        segments_index = 0
        for index, trace in enumerate(self.device_traces):
            action = trace["action"]

            if index == 20:
                pass

            target_seg = None
            request_size = copy.deepcopy(trace["size"])
            trace["size"] = ((trace["size"] + 512 - 1) // 512) * 512
            size = copy.deepcopy(trace["size"])
            if action == "segment_alloc":
                _sge = Segment(trace)
                _sge.index = segments_index
                segments_index += 1
                max_allowed_memory -= _sge.total
                segments[(_sge.start_addr, _sge.end_addr)] = _sge
                target_seg = _sge
                request_size = None
            elif action == "segment_free":
                _sge = Segment(trace)
                max_allowed_memory += _sge.total
                del segments[(_sge.start_addr, _sge.end_addr)]
                target_seg = _sge
            elif action == "alloc":
                for _sge in segments.values():
                    if _sge.insert_block(trace):
                        target_seg = _sge
                        break
            elif action == "free_completed":
                for _sge in segments.values():
                    if _sge.remove_block(trace):
                        target_seg = _sge
                        break
            elif action == "free_requested":
                for _sge in segments.values():
                    if _sge.check(trace):
                        target_seg = _sge
                        break
            else:
                continue

            # kSmallSize = 1048576
            # if target_seg.remaining < kSmallSize or target_seg.total == target_seg.remaining:
            #     size = target_seg.total
            # size = None
            if target_seg is None:
                msg = None
            else:
                msg = f"{None} Block from Segment {target_seg.index} for request {request_size}"
            record.append(
                {
                    "action": action,
                    "size": size,
                    "requested_size": request_size,
                    "block": trace,
                    "msg": msg,
                    "segments": copy.deepcopy(segments),
                }
            )

        return record

    def peak_memory_usage_each_iteration(self):
        iterations = []
        max_trace = 0
        max_seg = 0
        trace_list = []
        seg_list = []
        for trace in self.device_traces:
            if trace["action"] == "alloc":
                max_trace += trace["size"]
            elif trace["action"] == "free_completed":
                max_trace -= trace["size"]
            elif trace["action"] == "segment_alloc":
                max_seg += trace["size"]
            elif trace["action"] == "segment_free":
                max_seg -= trace["size"]
            seg_list.append(max_seg)
            trace_list.append(max_trace)
            frames = trace["frames"]
            for frame in frames:
                filepath = frame["filename"]
                lines = frame["line"]
                if lines == 43:
                    iterations.append({"seg": max(seg_list), "trace": max(trace_list)})
                    break
        return iterations[::3]
