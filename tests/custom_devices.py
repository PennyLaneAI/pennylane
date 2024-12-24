"""A collection of Custom Devices"""

import pennylane as qml


class BaseCustomDeviceReturnsZero(qml.devices.Device):
    def execute(self, circuits, execution_config=None):
        return 0


class BaseCustomDeviceReturnsTuple(qml.devices.Device):
    def execute(self, circuits, execution_config=None):
        return (0,)
