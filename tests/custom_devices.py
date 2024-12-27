"""A Factory of Custom Devices"""

from typing import Union

from pennylane.devices import Device
from pennylane.tape import QuantumScriptOrBatch
from pennylane.typing import Result, ResultBatch


def CustomDeviceFactory(
    return_value=0, return_for_each_circuit=False, circuits_type=None, config=None
):
    def _determineReturnValue(circuits):
        if return_for_each_circuit:
            return tuple(return_value for _ in circuits)
        else:
            return return_value

    class BaseCustomDevice(Device):
        if circuits_type == "QuantumScriptOrBatch":

            def execute(
                self, circuits: QuantumScriptOrBatch, execution_config=config
            ) -> Union[Result, ResultBatch]:
                return _determineReturnValue()

        elif circuits_type == None:

            def execute(self, circuits, execution_config=config):
                return _determineReturnValue(circuits)

        else:
            raise TypeError("Invalid citcuits type")

    return BaseCustomDevice
