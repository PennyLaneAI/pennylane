"""A collection of Custom Devices"""

from typing import Union

from pennylane.devices import DefaultExecutionConfig, Device, ExecutionConfig
from pennylane.tape import QuantumScriptOrBatch
from pennylane.typing import Result, ResultBatch


class BaseCustomDeviceReturnsTupleForDefaultConfig(Device):
    def execute(self, circuits, execution_config=DefaultExecutionConfig):
        return tuple(0.0 for _ in circuits)


class BaseCustomDeviceQuantumScriptOrBatch(Device):
    def execute(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig = None,
    ) -> Union[Result, ResultBatch]:
        return (0,)


def getReturnValue(return_type):
    if return_type == "Tuple":
        return (0,)
    elif return_type == "Int":
        return 0
    elif return_type == "Float":
        return 0.0
    elif return_type == "Literal":
        return "a"
    else:
        raise ValueError("Invalid return type")


def CreateBaseCustomDevice(return_type="Int", config=None):
    execution_config = DefaultExecutionConfig if config == "Default" else None
    return_value = getReturnValue(return_type)

    class BaseCustomDevice(Device):
        def execute(self, circuits, execution_config=execution_config):
            return return_value

    return BaseCustomDevice
