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


def CreateBaseCustomDevice(return_value=0, config=None):

    class BaseCustomDevice(Device):
        def execute(self, circuits, execution_config=config):
            return return_value

    return BaseCustomDevice
