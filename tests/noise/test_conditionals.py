# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Unit tests for the available conditional utitlities for noise models.
"""

# pylint: disable = too-few-public-methods
import pytest

import pennylane as qml
from pennylane.noise.conditionals import _get_wires, _get_ops


class TestNoiseConditionals:
    """Test for the Conditional classes"""

    def test_noise_conditional_lambda(self):
        """Test for NoiseConditional builds correct objects"""

        func = qml.noise.NoiseConditional(lambda x: x < 10, "less_than_ten")

        assert isinstance(func, qml.BooleanFn)
        assert func(2) and not func(42)
        assert str(func) == "less_than_ten"

    def test_noise_conditional_def(self):
        """Test for NoiseConditional builds correct objects"""

        def greater_than_five(x):
            return x > 5

        func = qml.noise.NoiseConditional(greater_than_five)

        assert isinstance(func, qml.BooleanFn)
        assert not func(3) and func(7)
        assert str(func) == "NoiseConditional(greater_than_five)"

    def test_and_conditionals(self):
        """Test for NoiseConditional supports bitwise AND"""

        def is_int(x):
            return isinstance(x, int)

        def has_bit_length_3(x):
            return x.bit_length() == 3

        func1 = qml.noise.NoiseConditional(is_int, "is_int")
        func2 = qml.noise.NoiseConditional(has_bit_length_3, "has_bit_length_3")
        func3 = func1 & func2

        assert isinstance(func3, qml.noise.AndConditional)
        assert func3(4) and not func3(2.3)
        assert str(func3) == "And(is_int, has_bit_length_3)"

    def test_or_conditionals(self):
        """Test for NoiseConditional supports bitwise OR"""

        def is_int(x):
            return isinstance(x, int)

        def less_than_five(x):
            return x < 5

        func1 = qml.noise.NoiseConditional(is_int, "is_int")
        func2 = qml.noise.NoiseConditional(less_than_five, "less_than_five")
        func3 = func1 | func2

        assert isinstance(func3, qml.noise.OrConditional)
        assert func3(4) and not func3(7.5)
        assert str(func3) == "Or(is_int, less_than_five)"


class TestNoiseFunctions:
    """Test for the Conditional methods"""

    @pytest.mark.parametrize(
        ("obj", "wires", "result"),
        [
            (0, 0, True),
            ([1, 2, 3], 0, False),
            (qml.wires.Wires(["street", "fighter"]), "fighter", True),
            (qml.wires.Wires(1), [0], False),
            (qml.Y(2), 2, True),
            (qml.CNOT(["a", "c"]), "b", False),
        ],
    )
    def test_wire_in(self, obj, wires, result):
        """Test for checking WireIn work as expected for checking if a wire exist in a set of specified wires"""

        func = qml.noise.wire_in(obj)

        assert isinstance(func, qml.noise.NoiseConditional)
        assert str(func) == f"WiresIn({obj})"
        assert func(wires) == result

    @pytest.mark.parametrize(
        ("obj", "wires", "result"),
        [
            (0, 0, True),
            ([1, 2], [1, 2], True),
            (qml.wires.Wires(["street", "fighter"]), "fighter", True),
            (qml.wires.Wires(1), [1], True),
            (qml.Y(2), 2, True),
            (qml.CNOT(["a", "c"]), "b", False),
            (qml.CNOT(["c", "d"]), ["c", "d"], True),
        ],
    )
    def test_wire_eq(self, obj, wires, result):
        """Test for checking WireEq work as expected for checking if a set of wires is equal to specified wires"""

        func = qml.noise.wire_eq(obj)

        assert isinstance(func, qml.noise.NoiseConditional)
        assert str(func) == f"WiresEq({obj})"
        assert func(wires) == result

    def test_get_wires_error(self):
        """Test for checking _get_wires method raise correct error"""

        with pytest.raises(
            ValueError, match="Wires cannot be computed for"
        ):
            _get_wires(qml.RX)

    @pytest.mark.parametrize(
        ("obj", "op", "result"),
        [
            ([qml.RX, qml.RY], qml.RY(1.0, 1), True),
            ([qml.RX, qml.RY], qml.RX, True),
            (qml.RX(0, 1), qml.RY(1.0, 1), False),
            ("RX", qml.RX(0, 1), True),
            (["CZ", "RY", "CNOT"], qml.CNOT([0, 1]), True),
            (qml.Y(1), qml.RY(1.0, 1), False),
            (qml.CNOT(["a", "c"]), qml.CNOT([0, 1]), True),
        ],
    )
    def test_op_in(self, obj, op, result):
        """Test for checking OpIn work as expected for checking if a operation exist in a set of specified operation"""

        func = qml.noise.op_in(obj)

        assert isinstance(func, qml.noise.NoiseConditional)

        op_repr = list(getattr(op, "__name__") for op in _get_ops(obj))
        assert str(func) == f"OpIn({op_repr})"
        assert func(op) == result

    @pytest.mark.parametrize(
        ("obj", "op", "result"),
        [
            (qml.Y(1), qml.RY(1.0, 1), False),
            (qml.CNOT(["a", "c"]), qml.CNOT([0, 1]), True),
            (qml.RX(0, 1), qml.RY(1.0, 1), False),
            ("RX", qml.RX(0, 1), True),
            ([qml.RX, qml.RY], qml.RX, False),
            ([qml.RX, qml.RY], [qml.RX(1.0, 1), qml.RY(2.0, 2)], True),
            (["CZ", "RY"], [qml.CZ([0, 1]), qml.RY(1.0, [1])], True),
        ],
    )
    def test_op_eq(self, obj, op, result):
        """Test for checking WireEq work as expected for checking if an operation is equal to specified operation"""

        func = qml.noise.op_eq(obj)

        assert isinstance(func, qml.noise.NoiseConditional)

        op_repr = str([getattr(op, "__name__") for op in _get_ops(obj)])[1:-1]
        assert str(func) == f"OpEq({op_repr})"
        assert func(op) == result

    def test_partial_wires(self):
        """Test for checking partial_wires work as expected for building callables to make op with correct wires"""

        op = qml.noise.partial_wires(qml.RX(1.2, [12]))(qml.RY(1.0, ["wires"]))
        assert qml.equal(op, qml.RX(1.2, wires=["wires"]))

        op = qml.noise.partial_wires(qml.RX, 3.2)(qml.RY(1.0, [0]))
        assert qml.equal(op, qml.RX(3.2, wires=[0]))

        op = qml.noise.partial_wires(qml.RX, phi=1.2)(qml.RY(1.0, [2]))
        assert qml.equal(op, qml.RX(1.2, wires=[2]))

        op = qml.noise.partial_wires(qml.RX(1.2, [12]), phi=2.3)(qml.RY(1.0, ["light"]))
        assert qml.equal(op, qml.RX(2.3, wires=["light"]))

    def test_partial_wires_error(self):
        """Test for checking partial_wires raise correct error when args are given"""

        with pytest.raises(
            ValueError, match="Args cannot be provided when operation is an instance"
        ):
            qml.noise.partial_wires(qml.RX(1.2, [12]), 1.2)(qml.RY(1.0, ["wires"]))
