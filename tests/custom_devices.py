"""A collection of Custom Devices"""

from typing import Union

from pennylane.devices import DefaultExecutionConfig, Device, ExecutionConfig
from pennylane.tape import QuantumScriptOrBatch
from pennylane.typing import Result, ResultBatch


class BaseCustomDeviceReturnsTupleForDefaultConfig(Device):
    def execute(self, circuits, execution_config=DefaultExecutionConfig):
        return tuple(0.0 for _ in circuits)


def CreateBaseCustomDevice(return_value=0, circuits_type=None, config=None):

    class BaseCustomDevice(Device):
        def execute(self, circuits, execution_config=config):
            return return_value

    class BaseCustomDeviceQuantumScriptOrBatch(Device):
        def execute(
            self, circuits: QuantumScriptOrBatch, execution_config=config
        ) -> Union[Result, ResultBatch]:
            return return_value

    if circuits_type == "QuantumScriptOrBatch":
        return BaseCustomDeviceQuantumScriptOrBatch
    elif circuits_type == None:
        return BaseCustomDevice
    else:
        raise TypeError("Invalid citcuits type")
