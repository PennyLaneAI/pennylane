"""A collection of Custom Devices"""

import pennylane as qml
from pennylane.devices import DefaultExecutionConfig


class BaseCustomDeviceReturnsZero(qml.devices.Device):
    def execute(self, circuits, execution_config=None):
        return 0


class BaseCustomDeviceReturnsTuple(qml.devices.Device):
    def execute(self, circuits, execution_config=None):
        return (0,)


class BaseCustomDeviceReturnsZeroDefaultConfig(qml.devices.Device):
    def execute(self, circuits, execution_config=DefaultExecutionConfig):
        return 0


class BaseCustomDeviceReturnsTupleDefaultConfig(qml.devices.Device):
    def execute(self, circuits, execution_config=DefaultExecutionConfig):
        return (0,)
