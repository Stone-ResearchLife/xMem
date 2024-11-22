import os
import logging
from typing import Optional, Union
from abc import ABC, abstractmethod
from .interface import InterfaceHostMetric


logger = logging.getLogger()


class _CGroupAbc(ABC):
    def __init__(self, cgroup_dir: str):
        self._cgroup_dir = cgroup_dir

    @property
    def cgroup_dir(self) -> str:
        """
        Get the cgroup directory path
        Returns:
            str: cgroup directory path
        """
        return self._cgroup_dir

    def _get_cgroup_data(self, file_name: str, key: Optional[str] = None, index: Optional[int] = None) -> Optional[str]:
        """Get the data from cgroup file by key and index

        Args:
            file_name (str): cgroup file name
            key (str, optional): key to search, Defaults to None. None will extract all data from the file.
            index (int, optional): index of the value. Defaults to None. None will be converted to 1.

        Returns:
            Optional[Union[str, int]]: value of the key and filter by index

        """
        _index = index or 1
        _file_path = os.path.join(self.cgroup_dir, file_name)
        logger.debug(f"Get data from file {_file_path} with key {key} and index {_index}")
        with open(_file_path, "r") as f:
            _result = None
            if key is not None:
                lines = f.readlines()
                for line in lines:
                    if key in line:
                        _result = line
                        break
                if _result is not None:
                    _result = _result.split()[_index]
                else:
                    raise KeyError(f"Key {key} not found in file {_file_path}")
            else:
                _result = f.read().strip()

        return _result

    @abstractmethod
    def cpu_usage(self) -> int:
        """Get the CPU usage in nanoseconds"""
        pass

    @abstractmethod
    def cpu_system(self) -> int:
        """Get the System CPU usage in nanoseconds"""
        pass

    @abstractmethod
    def cpu_user(self) -> int:
        """Get the user CPU usage in nanoseconds"""
        pass

    @abstractmethod
    def memory_usage(self) -> int:
        """Get the memory usage in bytes"""
        pass

    @abstractmethod
    def memory_max(self) -> Union[int, float]:
        """Get the maximum memory in percentage"""
        pass

    def self_check(self) -> bool:
        try:
            self.cpu_usage()
            self.cpu_system()
            self.cpu_user()
            self.memory_usage()
            self.memory_max()
            return True
        except Exception as e:
            logger.error(f"Self check failed with error: {e}")
            return False


class CGroupV2(_CGroupAbc):
    def cpu_usage(self) -> int:
        return int(self._get_cgroup_data("cpu.stat", "usage_usec"))

    def cpu_system(self) -> int:
        return int(self._get_cgroup_data("cpu.stat", "system_usec"))

    def cpu_user(self) -> int:
        return int(self._get_cgroup_data("cpu.stat", "user_usec"))

    def memory_usage(self) -> int:
        return int(self._get_cgroup_data("memory.current"))

    def memory_max(self) -> float:
        _max = self._get_cgroup_data("memory.max")
        if type(_max) is float or type(_max) is int:
            _max = float(_max * 100)
        else:
            _max = -1
        return _max



class CGroupV1(_CGroupAbc):
    def cpu_usage(self) -> int:
        return int(self._get_cgroup_data("cpuacct/cpuacct.usage"))

    def cpu_system(self) -> int:
        return int(self._get_cgroup_data("cpuacct/cpuacct.stat", "system"))

    def cpu_user(self) -> int:
        return int(self._get_cgroup_data("cpuacct/cpuacct.stat", "user"))

    def memory_usage(self) -> int:
        return int(self._get_cgroup_data("memory/memory.usage_in_bytes"))

    def memory_max(self) -> float:
        _max = self._get_cgroup_data("memory/memory.limit_in_bytes")
        if type(_max) is float or type(_max) is int:
            _max = float(_max * 100)
        else:
            _max = -1
        return _max


class CGroupMonitor(InterfaceHostMetric):
    V2Controller = "/sys/fs/cgroup/cgroup.controllers"

    def __init__(self, pid: Optional[int] = None):
        default_cgroup = "/sys/fs/cgroup"
        if pid is not None:
            logger.info(f"Try to get cgroup for process id: {pid}")
            default_cgroup = f"/proc/{pid}/cgroup"
            logger.info(f"The cgroup path changes to {default_cgroup}")
        self._cgroup = default_cgroup
        self._is_v2 = os.path.exists(self.V2Controller)

    def _get_cgroup(self) -> _CGroupAbc:
        """Get a right verion of cgroup

        Returns:
            _CGroupAbc: cgroup object

        """
        if self._is_v2:
            _cgroup = CGroupV2(self._cgroup)
        else:
            _cgroup = CGroupV1(self._cgroup)
        return _cgroup

    def cpu_usage(self) -> int:
        """Get the CPU usage in nanoseconds"""
        return self._get_cgroup().cpu_usage()

    def cpu_system(self) -> int:
        """Get the System CPU usage in nanoseconds"""
        return self._get_cgroup().cpu_system()

    def cpu_user(self) -> int:
        """Get the user CPU usage in nanoseconds"""
        return self._get_cgroup().cpu_user()

    def memory_usage(self) -> int:
        """Get the memory usage in bytes"""
        return self._get_cgroup().memory_usage()

    def memory_max(self) -> Union[int, float]:
        """Get the maximum memory in percentage"""
        return self._get_cgroup().memory_max()

    def to_json(self) -> dict:
        return {
            "cpu": {
                "usage": self.cpu_usage(),
                "system": self.cpu_system(),
                "user": self.cpu_user()
            },
            "memory": {
                "usage": self.memory_usage(),
                "max": self.memory_max()
            }
        }

    def record(self) -> dict:
        return self.to_json()

    def summary(self, prev_record:dict, current_record:dict, interval_ms:int) -> dict:
        _summary = {}
        for key in current_record["cpu"].keys():
            _cpu_time = (current_record["cpu"][key] - prev_record["cpu"][key]) / 1e6  # convert to milliseconds
            _utilisation = round((_cpu_time/interval_ms) * 100, 2)
            if "cpu" not in _summary.keys():
                _summary["cpu"] = {}
            _summary["cpu"][key] = {
                "util": _utilisation
            }
        for key in current_record["memory"].keys():
            _memory = round(current_record["memory"][key]/1024/1024, 2) # convert to MB
            _summary["memory"] = {
                key: _memory
            }
        return _summary

    def self_check(self) -> bool:
        return self._get_cgroup().self_check()
