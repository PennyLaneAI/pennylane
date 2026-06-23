# Copyright 2026 Xanadu Quantum Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for ``pennylane.core.operator.utils``."""

import numpy as np
import pytest
from operator2_utils import DynOp, MixedHybridOp

from pennylane.core.operator import Operator2
from pennylane.core.operator.utils import abstractify
from pennylane.typing import AbstractArray, AbstractWires, Bool, Complex, Float, Int
from pennylane.wires import Wires


class TestAbstractify:
    """Tests for the ``abstractify`` helper."""

    @pytest.mark.parametrize(
        "x", [0.12, np.ones((2, 3), dtype=np.float32), [0, 1], {"w": Wires([0, 1, 2])}]
    )
    def test_indempotency(self, x):
        """Tests the round-trip behaviour of this helper."""

        assert abstractify(abstractify(x)) == abstractify(x)

    def test_scalar_number(self):
        """Test that Python numbers become scalar ``AbstractArray`` instances."""
        assert abstractify(1.5) == AbstractArray((), float)
        assert abstractify(3) == AbstractArray((), int)

    def test_list(self):
        """Tests that lists correctly retain shape."""
        assert abstractify([0, 1]) == [AbstractArray((), int), AbstractArray((), int)]

    def test_numpy_array(self):
        """Test that numpy arrays are converted to ``AbstractArray``."""
        arr = np.ones((2, 3), dtype=np.float32)
        assert abstractify(arr) == AbstractArray((2, 3), np.float32)

    def test_wires(self):
        """Test that ``Wires`` are converted to ``AbstractWires``."""
        assert abstractify(Wires([0, 1, 2])) == AbstractWires(3)

    def test_abstract_types_as_input(self):
        """Test that abstract types are returned unchanged."""
        aa = AbstractArray((2,), float)
        aw = AbstractWires(2)
        assert abstractify(aa) is aa
        assert abstractify(aw) is aw

    def test_pytree_with_wires_leaves(self):
        """Test that pytrees containing ``Wires`` leaves are abstractified recursively."""
        val = [Wires([0]), (Wires([1, 2]), Wires([3]))]
        result = abstractify(val)
        assert result == [AbstractWires(1), (AbstractWires(2), AbstractWires(1))]

    def test_operator(self):
        """Test that ``Operator2`` instances are abstractified correctly."""
        op = DynOp(0.5, wires=[0, 1])
        result = abstractify(op)

        assert isinstance(result, DynOp)
        assert result.phi == AbstractArray((), float)
        assert result.wires == AbstractWires(2)

    def test_operator_hybrid_args(self):
        """Test that ``Operator2`` instances are abstractified correctly when there
        are hybrid arguments."""
        op = MixedHybridOp(1.5, ops=[DynOp(2.5, 0)], pytree_wires=[1, [2, 3, 4], [5]], wires=6)
        result = abstractify(op)

        assert isinstance(result, MixedHybridOp)
        assert result.phi == AbstractArray((), float)
        assert result.pytree_wires == [AbstractWires(1), AbstractWires(3), AbstractWires(1)]
        assert result.wires == AbstractWires(7)  # len([0,1,2,3,4,5,6])

        inner_op = result.ops[0]
        assert isinstance(result.ops, list) and len(result.ops) == 1
        assert isinstance(inner_op, DynOp)
        assert inner_op.phi == AbstractArray((), float)
        assert inner_op.wires == AbstractWires(1)

    def test_operator_subclass_with_fixed_sig(self):
        """Tests that an operator subclass with fixed sig works correctly."""

        class FixedSigOp(Operator2):  # pylint: disable=too-few-public-methods
            dynamic_argnames = ("phi",)
            wire_argnames = ("wires", "ctrl_wires")
            fixed_sig = (
                AbstractArray((), float),
                AbstractWires(2),
                AbstractWires(1),
            )

            def __init__(self, phi, wires, ctrl_wires):
                super().__init__(phi, wires=wires, ctrl_wires=ctrl_wires)

        result = abstractify(FixedSigOp)
        assert result.phi == AbstractArray((), float)
        assert result.wires == AbstractWires(3)  # 2 + 1
        assert result.ctrl_wires == AbstractWires(1)

    def test_operator_subclass_without_fixed_sig(self):
        """Tests that an error is raised if an operator subclass is used without a defined fixed_sig."""

        class FixedSigOp(Operator2):  # pylint: disable=too-few-public-methods
            dynamic_argnames = ("phi",)
            wire_argnames = ("wires", "ctrl_wires")

            def __init__(self, phi, wires, ctrl_wires):
                super().__init__(phi, wires=wires, ctrl_wires=ctrl_wires)

        with pytest.raises(TypeError, match="must define a 'fixed_sig'"):
            _ = abstractify(FixedSigOp)

    def test_abstract_instance_hash_stable(self):
        """Ensures that we get the same hash value for equal type specifiers."""
        a = abstractify(DynOp(0.5, 0))
        b = abstractify(DynOp(1.5, 0))
        c = abstractify(DynOp(1.5, [0, 1]))
        assert hash(a) == hash(b)  # Structural match, same hash
        assert hash(b) != hash(c)  # Structural mismatch, different hash

    def test_abstract_instance_equal(self):
        """Ensures that equality works on the output of abstractify."""
        phi1 = 0.5
        phi2 = 1.5

        # Different phi, same wire structure
        a = abstractify(DynOp(phi1, [0]))
        b = abstractify(DynOp(phi2, [1]))
        assert a == b

        # Same phi, different wire structure
        a = abstractify(DynOp(phi1, [0]))
        b = abstractify(DynOp(phi1, [0, 1]))
        assert a != b

    @pytest.mark.parametrize(
        "input, abstract_input",
        (
            (float, Float),
            (int, Int),
            (complex, Complex),
            (bool, Bool),
            (np.float32, AbstractArray((), np.float32)),
            ([float, float], [Float, Float]),
            ([float, complex], [Float, Complex]),
        ),
    )
    def test_abstractify_supported_types(self, input, abstract_input):
        """Ensures that the abstract version of types are correct."""

        assert abstractify(input) == abstract_input

    @pytest.mark.parametrize("input", (str, list, tuple))
    def test_abstractify_unsupported_types(self, input):
        """Tests that unsupported types raise an error."""

        with pytest.raises(NotImplementedError, match="Cannot abstractify"):
            _ = abstractify(input)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
