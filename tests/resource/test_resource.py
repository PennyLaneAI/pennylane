# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
Test base Resource class and its associated methods
"""
from collections import defaultdict

import pytest
from pennylane.resource import Resources


class TestResources:
    """Test the methods and attributes of the Resource class"""

    resource_quantities = (
        (),
        (5, 0, {}, 0, 0),
        (1, 3, {"Hadamard": 1, "PauliZ": 2}, 3, 10),
        (4, 2, {"Hadamard": 1, "CNOT": 1}, 2, 100),
    )

    @staticmethod
    def _construct_resource(
        num_wires=None, num_gates=None, gate_types=None, depth=None, shots=None
    ):
        """Construct an instances of the Resources class according to the data provided."""
        r = Resources()

        if num_wires:
            r.num_wires = num_wires
        if num_gates:
            r.num_gates = num_gates
        if gate_types:
            r.gate_types = defaultdict(int, gate_types)
        if depth:
            r.depth = depth
        if shots:
            r.shots = shots
        return r

    def test_init(self):
        """Test that the Resource class is instantiated as expected."""
        r = Resources()
        assert r.num_wires == 0
        assert r.num_gates == 0
        assert r.depth == 0
        assert r.shots == 0
        assert r.gate_types == defaultdict(int)

    test_str_data = (
        ("wires: 0\n" + "gates: 0\n" + "depth: 0\n" + "shots: 0\n" + "gate_types: \n" + "{}"),
        ("wires: 5\n" + "gates: 0\n" + "depth: 0\n" + "shots: 0\n" + "gate_types: \n" + "{}"),
        (
            "wires: 1\n"
            + "gates: 3\n"
            + "depth: 3\n"
            + "shots: 10\n"
            + "gate_types: \n"
            + "{'Hadamard': 1, 'PauliZ': 2}"
        ),
        (
            "wires: 4\n"
            + "gates: 2\n"
            + "depth: 2\n"
            + "shots: 100\n"
            + "gate_types: \n"
            + "{'Hadamard': 1, 'CNOT': 1}"
        ),
    )

    @pytest.mark.parametrize("r_params, rep", zip(resource_quantities, test_str_data))
    def test_to_str(self, r_params, rep):
        """Test the string representation of a Resources instance."""
        r = self._construct_resource(*r_params)
        assert str(r) == rep

    test_rep_data = (
        "<Resource: wires=0, gates=0, depth=0, shots=0, gate_types=defaultdict(<class 'int'>, {})>",
        "<Resource: wires=5, gates=0, depth=0, shots=0, gate_types=defaultdict(<class 'int'>, {})>",
        "<Resource: wires=1, gates=3, depth=3, shots=10, "
        "gate_types=defaultdict(<class 'int'>, {'Hadamard': 1, 'PauliZ': 2})>",
        "<Resource: wires=4, gates=2, depth=2, shots=100, "
        "gate_types=defaultdict(<class 'int'>, {'Hadamard': 1, 'CNOT': 1})>",
    )

    @pytest.mark.parametrize("r_params, rep", zip(resource_quantities, test_rep_data))
    def test_repr(self, r_params, rep):
        """Test the repr method of a Resources instance looks as expected."""
        r = self._construct_resource(*r_params)
        assert repr(r) == rep

    @pytest.mark.parametrize("r_params, rep", zip(resource_quantities, test_str_data))
    def test_ipython_display(self, r_params, rep, capsys):
        """Test that the ipython display prints the string representation of a Resources instance."""
        r = self._construct_resource(*r_params)
        r._ipython_display_()  # pylint: disable=protected-access
        captured = capsys.readouterr()
        assert rep in captured.out

    test_add_data = (
        (
            (),
            (1, 3, {"Hadamard": 1, "PauliZ": 2}, 3, 10),
            (1, 3, {"Hadamard": 1, "PauliZ": 2}, 3, 10),
        ),
        (
            (1, 3, {"Hadamard": 1, "PauliZ": 2}, 3, 10),
            (1, 3, {"Hadamard": 1, "PauliZ": 2}, 3, 10),
            (1, 6, {"Hadamard": 2, "PauliZ": 4}, 6, 10),
        ),
        (
            (1, 3, {"Hadamard": 1, "PauliZ": 2}, 3, 10),
            (4, 2, {"Hadamard": 1, "CNOT": 1}, 2, 100),
            (4, 5, {"Hadamard": 2, "PauliZ": 2, "CNOT": 1}, 5, 100),
        ),
        (
            (5, 0, {}, 0, 0),
            (4, 2, {"Hadamard": 1, "CNOT": 1}, 2, 100),
            (5, 2, {"Hadamard": 1, "CNOT": 1}, 2, 100),
        ),
    )

    @pytest.mark.parametrize("r1_params, r2_params, result_params", test_add_data)
    def test_add(self, r1_params, r2_params, result_params):
        """Test adding two Resource instances (sum in series)."""
        r1 = self._construct_resource(*r1_params)
        r2 = self._construct_resource(*r2_params)

        result = r1 + r2
        expected = self._construct_resource(*result_params)
        assert expected == result

    test_matmul_data = (
        (
            (),
            (1, 3, {"Hadamard": 1, "PauliZ": 2}, 3, 10),
            (1, 3, {"Hadamard": 1, "PauliZ": 2}, 3, 10),
        ),
        (
            (1, 3, {"Hadamard": 1, "PauliZ": 2}, 3, 10),
            (1, 3, {"Hadamard": 1, "PauliZ": 2}, 3, 10),
            (2, 6, {"Hadamard": 2, "PauliZ": 4}, 3, 10),
        ),
        (
            (1, 3, {"Hadamard": 1, "PauliZ": 2}, 3, 10),
            (4, 2, {"Hadamard": 1, "CNOT": 1}, 2, 100),
            (5, 5, {"Hadamard": 2, "PauliZ": 2, "CNOT": 1}, 3, 100),
        ),
        (
            (5, 0, {}, 0, 0),
            (4, 2, {"Hadamard": 1, "CNOT": 1}, 2, 100),
            (9, 2, {"Hadamard": 1, "CNOT": 1}, 2, 100),
        ),
    )

    @pytest.mark.parametrize("r1_params, r2_params, result_params", test_matmul_data)
    def test_matmul(self, r1_params, r2_params, result_params):
        """Test the matmul of two Resource instances (sum in parallel)."""
        r1 = self._construct_resource(*r1_params)
        r2 = self._construct_resource(*r2_params)

        result = r1 @ r2
        expected = self._construct_resource(*result_params)
        assert expected == result

    test_combine_data = (
        (True, (5, 5, {"Hadamard": 2, "PauliZ": 2, "CNOT": 1}, 3, 100)),
        (False, (4, 5, {"Hadamard": 2, "PauliZ": 2, "CNOT": 1}, 5, 100)),
    )

    @pytest.mark.parametrize("is_parallel, result_params", test_combine_data)
    def test_combine(self, is_parallel, result_params):
        """Test that we can combine Resource instances between series and parallel combine methods."""
        r0 = self._construct_resource()
        r1 = self._construct_resource(1, 3, {"Hadamard": 1, "PauliZ": 2}, 3, 10)
        r2 = self._construct_resource(4, 2, {"Hadamard": 1, "CNOT": 1}, 2, 100)

        list_of_resources = [r0, r1, r2]
        result = self._construct_resource(*result_params)

        assert Resources.combine(list_of_resources, in_parallel=is_parallel) == result

    def test_invalid_type_error(self):
        """Test that if you try to combine Resources instances with other types an error is raised."""
        r0 = self._construct_resource()
        lst_objs = [0, True, "str", tuple(), []]

        for other in lst_objs:
            with pytest.raises(
                ValueError, match="Can only combine with another instance of `Resources`"
            ):
                _ = r0 + other

            with pytest.raises(
                ValueError, match="Can only combine with another instance of `Resources`"
            ):
                _ = r0 @ other

            with pytest.raises(
                ValueError, match="Can only combine with another instance of `Resources`"
            ):
                _ = Resources.combine([r0, other], in_parallel=True)

            with pytest.raises(
                ValueError, match="Can only combine with another instance of `Resources`"
            ):
                _ = Resources.combine([r0, other], in_parallel=False)
