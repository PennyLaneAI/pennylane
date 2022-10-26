import numpy as np
from typing import List, Union
from abc import ABC, abstractmethod
from pennylane.operation import Operation

from ..device_interface.abstract_device_driver import AbstractDeviceDriver
from ..device_interface.device_config import DeviceConfig


class CustomDeviceDriver(AbstractDeviceDriver):
    def __init__(self, device_config: DeviceConfig):
        super().__init__(device_config)
        # self.device_reference = SomeExternalResource(self.device_config)

    def apply_ops(self, ops: Union[Operation, List[Operation]]):
        # Convert ops to work with self.device_reference
        # Pass operations to self.device_reference
        # Return results
        pass
