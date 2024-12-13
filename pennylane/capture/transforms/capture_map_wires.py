# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains the MapWiresInterpreter transform."""


import pennylane as qml
from pennylane.capture.base_interpreter import PlxprInterpreter


class MapWiresInterpreter(PlxprInterpreter):
    """Interpreter that maps wires of operations and measurements.

    **Examples:**

    .. code-block:: python

        import jax
        from pennylane.capture.transforms import MapWiresInterpreter

        @MapWiresInterpreter(wire_map={0: 1})
        def circuit():
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

    >>> qml.capture.enable()
    >>> jaxpr = jax.make_jaxpr(circuit)()
    >>> jaxpr
    { lambda ; . let
        _:AbstractOperator() = Hadamard[n_wires=1] 1
        a:AbstractOperator() = PauliZ[n_wires=1] 1
        b:AbstractMeasurement(n_wires=None) = expval_obs a
      in (b,) }

    """

    def __init__(self, wire_map: dict) -> None:
        """Initialize the interpreter."""
        self.wire_map = wire_map
        self._check_wire_map()
        super().__init__()

    def _check_wire_map(self) -> None:
        """Check that the wire map is valid and does not contain dynamic values."""
        if not all(isinstance(k, int) and k >= 0 for k in self.wire_map.keys()):
            raise ValueError("Wire map keys must be constant positive integers.")
        if not all(isinstance(v, int) and v >= 0 for v in self.wire_map.values()):
            raise ValueError("Wire map values must be constant positive integers.")

    def interpret_operation(self, op: "qml.operation.Operation"):
        """Interpret an operation."""
        qml.capture.disable()
        op = op.map_wires(self.wire_map)
        qml.capture.enable()
        return super().interpret_operation(op)

    def interpret_measurement(self, measurement: "qml.measurement.MeasurementProcess"):
        """Interpret a measurement operation."""
        qml.capture.disable()
        measurement = measurement.map_wires(self.wire_map)
        qml.capture.enable()
        return super().interpret_measurement(measurement)
