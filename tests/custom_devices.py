"""A Factory of Custom Devices"""

from typing import Union

from pennylane.devices import Device
from pennylane.tape import QuantumScriptOrBatch
from pennylane.typing import Result, ResultBatch


def CustomDeviceFactory(return_value=0, return_for_each_circuit=False):
    def _determineReturnValue(circuits):
        if return_for_each_circuit:
            return tuple(return_value for _ in circuits)
        else:
            return return_value

    class BaseCustomDevice(Device):
        def execute(self, circuits, execution_config=None):
            return _determineReturnValue(circuits)

    return BaseCustomDevice
