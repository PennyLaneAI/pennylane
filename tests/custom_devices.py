"""A collection of Custom Devices"""

from typing import Union

from pennylane.devices import DefaultExecutionConfig, Device, ExecutionConfig
from pennylane.tape import QuantumScriptOrBatch
from pennylane.typing import Result, ResultBatch


def CreateBaseCustomDevice(
    return_value=0, return_for_each_circuit=False, circuits_type=None, config=None
):

    class BaseCustomDevice(Device):
        def execute(self, circuits, execution_config=config):
            if return_for_each_circuit:
                return tuple(return_value for _ in circuits)
            else:
                return return_value

    class BaseCustomDeviceQuantumScriptOrBatch(Device):
        def execute(
            self, circuits: QuantumScriptOrBatch, execution_config=config
        ) -> Union[Result, ResultBatch]:
            if return_for_each_circuit:
                return tuple(return_value for _ in circuits)
            else:
                return return_value

    if circuits_type == "QuantumScriptOrBatch":
        return BaseCustomDeviceQuantumScriptOrBatch
    elif circuits_type == None:
        return BaseCustomDevice
    else:
        raise TypeError("Invalid citcuits type")
