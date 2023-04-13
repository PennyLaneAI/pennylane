# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The data class which will aggregate all the resource information from a quantum workflow.
"""


class Resources:
    r"""Contains attributes which store key resources such as number of gates, number of wires, shots,
    depth and gate types, tracked over a quantum workflow.

    Args:
        num_wires (int): number of qubits
        num_gates (int): number of gates
        gate_types (dict): dictionary storing operation names (str) as keys
            and the number of times they are used in the circuit (int) as values
        depth (int): the depth of the circuit defined as the maximum number of non-parallel operations
        shots (int): number of samples to generate

    Raises:
        TypeError: If the attributes provided are not of the correct type.
        ValueError: If an attribute provided has value less than 0.
        AttributeError: If an attribute is set after initialization.

    .. details::

        The resources being tracked can be accessed and set as class attributes.
        Additionally, the :code:`Resources` instance can be nicely displayed in the console.

        **Example**

        >>> r = Resources(num_wires=2, num_gates=2, gate_types={'Hadamard': 1, 'CNOT':1}, depth=2)
        >>> print(r)
        wires: 2
        gates: 2
        depth: 2
        shots: 0
        gate_types: {'Hadamard': 1, 'CNOT': 1}
    """

    def __init__(
        self, num_wires=0, num_gates=0, gate_types=None, depth=0, shots=0
    ):  # pylint: disable=too-many-arguments
        """Initialize a Resources instance and perform input type validation."""

        if not all(isinstance(param, int) for param in [num_wires, num_gates, depth, shots]):
            raise TypeError(
                "Incorrect type of input, expected type int for num_wires, num_gates, depth and shots. "
                f"Got {type(num_wires)}, {type(num_gates)}, {type(depth)}, {type(shots)} respectively. "
            )

        if gate_types and isinstance(gate_types, dict):
            raise TypeError(
                f"Incorrect type of input, expected type dict for gate_types. Got {type(gate_types)}."
            )

        if not all(val > -1 for val in [num_wires, num_gates, depth, shots]):
            raise ValueError(
                "Incorrect value of input, expected value of num_wires, num_gates, depth and shots to be >= 0."
                f"Got {num_wires}, {num_gates}, {depth}, {shots} respectively. "
            )

        self._num_wires = num_wires
        self._num_gates = num_gates
        self._gate_types = {} if gate_types is None else gate_types
        self._depth = depth
        self._shots = shots

    @property
    def num_wires(self):
        """Number of wires in the quantum workflow."""
        return self._num_wires

    @num_wires.setter
    def num_wires(self, new_val):
        """Setting is NOT allowed"""
        raise AttributeError(
            f"{type(self)} object does not support assignment for 'num_wires' attribute"
        )

    @property
    def num_gates(self):
        """Number of gates in the quantum workflow."""
        return self._num_gates

    @num_gates.setter
    def num_gates(self, new_val):
        """Setting is NOT allowed"""
        raise AttributeError(
            f"{type(self)} object does not support assignment for 'num_gates' attribute"
        )

    @property
    def gate_types(self):
        """Dictionary of unique quantum gates and counts in the quantum workflow."""
        return self._gate_types

    @gate_types.setter
    def gate_types(self, new_val):
        """Setting is NOT allowed"""
        raise AttributeError(
            f"{type(self)} object does not support assignment for 'gate_types' attribute"
        )

    @property
    def depth(self):
        """Circuit depth of the quantum workflow."""
        return self._depth

    @depth.setter
    def depth(self, new_val):
        """Setting is NOT allowed"""
        raise AttributeError(
            f"{type(self)} object does not support assignment for 'depth' attribute"
        )

    @property
    def shots(self):
        """Number of shots/samples in the quantum workflow."""
        return self._shots

    @shots.setter
    def shots(self, new_val):
        """Setting is NOT allowed"""
        raise AttributeError(
            f"{type(self)} object does not support assignment for 'shots' attribute"
        )

    def __str__(self):
        keys = ["wires", "gates", "depth", "shots", "gate_types"]
        vals = [self.num_wires, self.num_gates, self.depth, self.shots, self.gate_types]
        items = "\n".join([str(i) for i in zip(keys, vals)])
        items = items.replace("('", "")
        items = items.replace("',", ":")
        items = items.replace(")", "")
        items = items.replace("{", "\n{")
        return items

    def __repr__(self):
        return (
            f"<Resource: wires={self.num_wires}, gates={self.num_gates}, "
            f"depth={self.depth}, shots={self.shots}, gate_types={self.gate_types}>"
        )

    def _ipython_display_(self):
        """Displays __str__ in ipython instead of __repr__"""
        print(str(self))

    def __eq__(self, other: "Resources") -> bool:
        return all(
            (
                self.num_wires == other.num_wires,
                self.num_gates == other.num_gates,
                self.depth == other.depth,
                self.shots == other.shots,
                self.gate_types == other.gate_types,
            ),
        )
