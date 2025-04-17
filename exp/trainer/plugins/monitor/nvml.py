import logging
import pynvml as nvml
from typing import Dict
from .interface import InterfaceHostMetric


logger = logging.getLogger()


class GPUMetric:
    def __init__(self, index):
        self.index = index
        self.handle = nvml.nvmlDeviceGetHandleByIndex(self.index)

    def __str__(self):
        return f"Name: {self.name}\n" f"Arch: {self.arch}\n" f"UUID: {self.uuid}\n"

    @property
    def name(self) -> str:
        return self._call_nvml("nvmlDeviceGetName")

    @property
    def uuid(self) -> str:
        return self._call_nvml("nvmlDeviceGetUUID")

    @property
    def arch(self) -> str:
        """
        Returns: string: name of architecture
        """
        arch_list = {
            nvml.NVML_DEVICE_ARCH_KEPLER: "KEPLER",
            nvml.NVML_DEVICE_ARCH_MAXWELL: "MAXWELL",
            nvml.NVML_DEVICE_ARCH_PASCAL: "PASCAL",
            nvml.NVML_DEVICE_ARCH_VOLTA: "VOLTA",
            nvml.NVML_DEVICE_ARCH_TURING: "TURING",
            nvml.NVML_DEVICE_ARCH_AMPERE: "AMPERE",
            nvml.NVML_DEVICE_ARCH_ADA: "ADA",
            nvml.NVML_DEVICE_ARCH_HOPPER: "HOPPER",
        }
        return arch_list.get(self._call_nvml("nvmlDeviceGetArchitecture"), None)

    def _call_nvml(self, func_name, *args, **kwargs):
        try:
            func = getattr(nvml, func_name)
            result = func(self.handle, *args, **kwargs)
        except nvml.NVMLError as e:
            if e.value == nvml.NVML_ERROR_NOT_SUPPORTED:
                logger.error(f"Function {func_name} is not supported on {self.name}")
                result = None
            else:
                logger.error(f"Function {func_name} failed with error: {e}")
                raise RuntimeError(
                    f"Function {func_name} failed with error: {e}"
                ) from e
        except Exception as e:
            logger.error(f"Function {func_name} failed with error: {e}")
            raise RuntimeError(f"Function {func_name} failed with error: {e}") from e
        return result

    def get_running_processes(self, mode=1):
        """
        Args:
            mode (int): fetching mod
                0: fetching running processes for computing
                1: fetching running processes for graph
                2: fetching running processes for MPS

        Returns: List[nvmlFriendlyObject]
            nvmlFriendlyObject:
                pid: int
                userGpuMemory: int
                gpuInstanceId: int
                computeInstanceId: int


        """
        if mode == 0:
            result = self._call_nvml("nvmlDeviceGetComputeRunningProcesses_v3")
        elif mode == 1:
            result = self._call_nvml("nvmlDeviceGetGraphicsRunningProcesses_v3")
        elif mode == 3:
            result = self._call_nvml("nvmlDeviceGetMPSComputeRunningProcesses_v3")
        else:
            logger.error(f"Get running processes failed with mode {mode}")
            raise ValueError(f"Get running processes failed with mode {mode}")
        return result

    def _get_clock_info(self, clock_type):
        return self._call_nvml("nvmlDeviceGetClockInfo", clock_type)

    def get_graphics_clock(self):
        return self._get_clock_info(nvml.NVML_CLOCK_GRAPHICS)

    def get_memory_clock(self):
        return self._get_clock_info(nvml.NVML_CLOCK_MEM)

    def get_sm_clock(self):
        return self._get_clock_info(nvml.NVML_CLOCK_SM)

    def get_memory_info(self, version=None):
        """Get memory information, including total, free, used in bytes

        Returns:
            Dict: total, free, used, utilisation
        """
        memory = self._call_nvml("nvmlDeviceGetMemoryInfo", version=version)
        return {
            "total": memory.total,
            "free": memory.free,
            "used": memory.used,
            "utilisation": round((memory.used / memory.total) * 100, 2),
        }

    def get_temperature(self):
        return self._call_nvml("nvmlDeviceGetTemperature", nvml.NVML_TEMPERATURE_GPU)

    def get_power_usage(self):
        return self._call_nvml("nvmlDeviceGetPowerUsage")

    def get_power_limit(self):
        return self._call_nvml("nvmlDeviceGetPowerManagementLimit")

    def get_power_state(self):
        return self._call_nvml("nvmlDeviceGetPowerState")

    def get_fan_speed(self):
        return self._call_nvml("nvmlDeviceGetFanSpeed")

    def get_gpu_utilisation(self):
        return self._call_nvml("nvmlDeviceGetUtilizationRates").gpu

    def get_memory_utilisation(self):
        return self._call_nvml("nvmlDeviceGetUtilizationRates").memory

    def get_info(self):
        return {
            "temperature": self.get_temperature(),
            # "fan_speed": self.get_fan_speed(),
            "gpu_utilisation": self.get_gpu_utilisation(),
            "memory_utilisation": self.get_memory_utilisation(),
        }

    def get_clock_info(self):
        return {
            "graphics_clock": self.get_graphics_clock(),
            "memory_clock": self.get_memory_clock(),
            "sm_clock": self.get_sm_clock(),
        }

    def get_power_info(self):
        return {
            "usage": self.get_power_usage(),
            "limit": self.get_power_limit(),
            "state": self.get_power_state(),
        }

    def get_pci_info(self):
        pci_info = self._call_nvml("nvmlDeviceGetPciInfo")
        return {
            "bus_id": pci_info.busId,
            "domain": pci_info.domain,
            "bus": pci_info.bus,
            "device": pci_info.device,
            "pci_id": pci_info.pciDeviceId,
            "pci_sub_id": pci_info.pciSubSystemId,
            "bus_id_legacy": pci_info.busIdLegacy,
        }

    def to_json(self):
        return {
            "name": self.name,
            "uuid": self.uuid,
            "arch": self.arch,
            "info": self.get_info(),
            "pci_info": self.get_pci_info(),
            "memory": self.get_memory_info(),
            # "power": self.get_power_info(),
            "clock": self.get_clock_info(),
        }


class HostGPUs(InterfaceHostMetric):
    def __init__(self):
        self._gpus = {}
        self._num_gpus = 0
        self.init()

    def get_gpu(self, index: int) -> GPUMetric:
        return self._gpus[index]

    def init(self):
        nvml.nvmlInit()
        self._num_gpus = nvml.nvmlDeviceGetCount()
        for i in range(self._num_gpus):
            self._gpus[i] = GPUMetric(i)
            logger.info(f"GPU {i}: {self._gpus[i].name} is initialized")

    def shutdown(self):
        logger.info("Shutting down NVML")
        nvml.nvmlShutdown()

    def to_json(self) -> Dict[int, dict]:
        return {i: self._gpus[i].to_json() for i in self._gpus}

    def record(self, *args, **kwargs) -> dict:
        return self.to_json()

    def summary(self, p_record: dict, c_record: dict, interval_ms: int) -> dict:
        # NVML fetch the latest information from GPUs, so there is no calculation needed like CGround and Network
        return c_record

    def self_check(self) -> bool:
        try:
            self.init()
            self.shutdown()
            return True
        except Exception as e:
            logger.error(f"Self check failed with error: {e}")
            return False
