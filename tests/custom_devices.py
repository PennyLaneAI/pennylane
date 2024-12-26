"""A collection of Custom Devices"""

from typing import Union

from pennylane.devices import DefaultExecutionConfig, Device, ExecutionConfig
from pennylane.tape import QuantumScriptOrBatch
from pennylane.typing import Result, ResultBatch


class BaseCustomDeviceReturnsInt(Device):
    def execute(self, circuits, execution_config=None):
        return 0


class BaseCustomDeviceReturnsTuple(Device):
    def execute(self, circuits, execution_config=None):
        return (0,)


class BaseCustomDeviceReturnsFloatDefaultConfig(Device):
    def execute(self, circuits, execution_config=DefaultExecutionConfig):
        return 0.0


class BaseCustomDeviceReturnsTupleDefaultConfig(Device):
    def execute(self, circuits, execution_config=DefaultExecutionConfig):
        return (0,)


class BaseCustomDeviceReturnsTupleForDefaultConfig(Device):
    def execute(self, circuits, execution_config=DefaultExecutionConfig):
        return tuple(0.0 for _ in circuits)


class BaseCustomDeviceReturnsLiteralDefaultConfig(Device):
    def execute(self, circuits, execution_config=DefaultExecutionConfig):
        return "a"


class BaseCustomDeviceQuantumScriptOrBatch(Device):
    def execute(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig = None,
    ) -> Union[Result, ResultBatch]:
        return (0,)
