import numpy as np
from typing import List, Union
from abc import ABC, abstractmethod
from pennylane.operation import Operation

from .device_config import *


# To be purged!
class AbstractDeviceDriver(ABC):
    def __init__(self, device_config: DeviceConfig):
        self.device_config = device_config

    def apply_ops(self, ops: Union[Operation, List[Operation]]):
        pass

    def query_support(self) -> DeviceConfig:
        self.device_config


