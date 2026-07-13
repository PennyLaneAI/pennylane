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

"""Tests for the Controlled2 class."""

import copy
import pickle
from typing import override

import numpy as np
import pytest

import pennylane as qp
from pennylane.core import Operator2
from pennylane.decomposition.decomposition_rule import register_resources
from pennylane.ops.op_math.controlled import Controlled, ControlledOp
from pennylane.ops.op_math.controlled2 import Controlled2, ControlledOp2
from pennylane.typing import AbstractArray, Bool, Float, Wire
from pennylane.wires import Wires
from tests.core.operator.operator2_utils import DynOp, NonParametricOp, OneWireDynOp

# pylint: disable=unused-argument,too-few-public-methods,useless-parent-delegation


class TestControlled2:
    """Unit tests for the Controlled2 interface."""

    def test_non_parametrized_custom_controlled_op(self):
        """Tests non-parametrized custom controlled op that directly inherits Controlled2"""

        class CH2(Controlled2):
            """A new CH."""

            wire_argnames = ("wires",)

            wire_sizes = (2,)

            def __init__(self, wires):
                super().__init__(qp.H(wires[1]), wires[0])

            @override
            def adjoint(self):
                return CH2(self.wires)

        op = CH2([0, 1])
        assert op.wires == Wires([0, 1])
        assert op.base == qp.H(1)
        assert op.target_wires == Wires([1])
        assert op.control_wires == Wires([0])
        assert op.control_values == [True]
        assert op.work_wires == Wires([])
        assert op.work_wire_type == "borrowed"
        assert op.dynamic_args == {}
        assert op.wire_args == {"wires": Wires([0, 1])}
        assert op.hybrid_args == {}
        assert op.has_adjoint
        assert op.has_matrix
        assert op.has_sparse_matrix
        assert op.has_diagonalizing_gates

    def test_parametric_custom_controlled_op(self):
        """Tests parametric op that inherits from Controlled2."""

        class CRot2(Controlled2):
            """A new CRot."""

            dynamic_argnames = ("phi", "theta", "omega")

            wire_argnames = ("wires",)

            wire_sizes = (2,)

            def __init__(self, phi, theta, omega, wires):
                super().__init__(qp.Rot(phi, theta, omega, wires=wires[1]), control_wires=wires[0])

            @override
            def adjoint(self):
                return CRot2(-self.omega, -self.theta, -self.phi, wires=self.wires)

        op = CRot2(0.1, 0.2, 0.3, wires=[0, 1])
        assert op.wires == Wires([0, 1])
        assert op.base == qp.Rot(0.1, 0.2, 0.3, wires=[1])
        assert op.target_wires == Wires([1])
        assert op.control_wires == Wires([0])
        assert op.control_values == [True]
        assert op.work_wires == Wires([])
        assert op.work_wire_type == "borrowed"
        assert op.dynamic_args == {"phi": 0.1, "theta": 0.2, "omega": 0.3}
        assert op.wire_args == {"wires": Wires([0, 1])}
        assert op.hybrid_args == {}
        assert op.has_adjoint
        assert op.has_matrix
        assert op.has_sparse_matrix

    def test_custom_controlled_op_abstract(self):
        """Test creating an abstract custom controlled op."""

        class Rot2(Operator2):
            """A new Rot."""

            dynamic_argnames = ("phi", "theta", "omega")

            wire_argnames = ("wires",)

            arg_specs = {"phi": Float, "theta": Float, "omega": Float, "wires": Wire[1]}

            def __init__(self, phi, theta, omega, wires):
                super().__init__(phi, theta, omega, wires)

        class CRot2(Controlled2):
            """A new CRot."""

            dynamic_argnames = ("phi", "theta", "omega")

            wire_argnames = ("wires",)

            arg_specs = {"phi": Float, "theta": Float, "omega": Float, "wires": Wire[2]}

            def __init__(self, phi, theta, omega, wires):
                super().__init__(Rot2(phi, theta, omega, wires=wires[1]), control_wires=wires[0])

            def __abstract_init__(self, phi, theta, omega, wires):
                super().__abstract_init__(Rot2(phi, theta, omega, wires[1]), wires[0])

        op = CRot2(Float, 0.5, 0.2, wires=[0, 1])
        assert op.phi == Float
        assert op.theta == Float
        assert op.omega == Float
        assert op.wires == Wire[2]
        assert op.control_wires == Wire[1]
        assert op.control_values == Bool[1]

    def test_custom_controlled_op_default_compute_methods(self):
        """Tests that custom controlled ops can use the default compute_xxx methods."""

        class CRot2(Controlled2):
            """A new CRot."""

            dynamic_argnames = ("phi", "theta", "omega")

            wire_argnames = ("wires",)

            wire_sizes = (2,)

            def __init__(self, phi, theta, omega, wires):
                super().__init__(qp.Rot(phi, theta, omega, wires=wires[1]), control_wires=wires[0])

            @override
            def adjoint(self):
                return CRot2(-self.omega, -self.theta, -self.phi, wires=self.wires)

        op = CRot2(0.1, 0.2, 0.3, wires=[0, 1])

        matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, np.exp(-0.2j) * np.cos(0.1), -np.exp(-0.1j) * np.sin(0.1)],
                [0, 0, np.exp(0.1j) * np.sin(0.1), np.exp(0.2j) * np.cos(0.1)],
            ]
        )
        assert qp.math.allclose(op.matrix(), matrix)
        assert qp.math.allclose(CRot2.compute_matrix(**op.arguments), matrix)
        assert qp.math.allclose(op.sparse_matrix(), matrix)
        assert qp.math.allclose(CRot2.compute_sparse_matrix(**op.arguments), matrix)

        eigvals = np.linalg.eigvals(matrix)
        assert qp.math.allclose(sorted(op.eigvals()), sorted(eigvals))
        assert qp.math.allclose(sorted(CRot2.compute_eigvals(**op.arguments)), sorted(eigvals))

        class CRX2(Controlled2):
            """A new CRX2."""

            dynamic_argnames = ("theta",)

            wire_argnames = ("wires",)

            wire_sizes = (2,)

            def __init__(self, theta, wires):
                super().__init__(qp.RX(theta, wires[1]), control_wires=wires[0])

        op = CRX2(0.5, wires=[0, 1])
        expected = qp.CRX.compute_matrix(0.5)
        assert qp.math.allclose(op.sparse_matrix(), expected)
        assert qp.math.allclose(CRX2.compute_sparse_matrix(**op.arguments), expected)

        class CH2(Controlled2):
            """A new CH."""

            wire_argnames = ("wires",)

            wire_sizes = (2,)

            def __init__(self, wires):
                super().__init__(qp.H(wires[1]), wires[0])

        op = CH2([0, 1])
        gates = [qp.RY(-np.pi / 4, wires=1)]
        assert op.diagonalizing_gates() == gates
        assert CH2.compute_diagonalizing_gates(**op.arguments) == gates

    def test_custom_controlled_op_own_compute_methods(self):
        """Tests when a custom controlled op override its own compute_xxx methods."""

        class CH2(Controlled2):
            """A new CH."""

            wire_argnames = ("wires",)

            wire_sizes = (2,)

            def __init__(self, wires):
                super().__init__(qp.H(wires[1]), wires[0])

            @override
            def adjoint(self):
                return CH2(self.wires)

            @staticmethod
            @override
            def compute_matrix(wires):
                return np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
                        [0, 0, 1 / np.sqrt(2), -1 / np.sqrt(2)],
                    ]
                )

        op = CH2([0, 1])

        matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
                [0, 0, 1 / np.sqrt(2), -1 / np.sqrt(2)],
            ]
        )
        assert qp.math.allclose(op.matrix(), matrix)
        assert qp.math.allclose(CH2.compute_matrix(**op.arguments), matrix)

    def test_subclass_hook(self):
        """Tests that Controlled2 operators are also considered instances of Controlled."""

        base = qp.H(0)
        op = ControlledOp2(base, control_wires=[1, 2])
        assert issubclass(ControlledOp2, Controlled)
        assert issubclass(ControlledOp2, ControlledOp)
        assert isinstance(op, Controlled)
        assert isinstance(op, ControlledOp)

    @pytest.mark.parametrize("op_type", (qp.CNOT, qp.MultiControlledX, qp.Toffoli, qp.CZ))
    def test_subclass_hook_does_not_match_specific_controlled_ops(self, op_type):
        """Test that ControlledOp2 does not pretend to be a concrete controlled gate."""

        op = ControlledOp2(qp.X(0), control_wires=[1])
        assert not issubclass(ControlledOp2, op_type)
        assert not isinstance(op, op_type)

    def test_simplify(self):
        """Tests the simplify method."""

        base = qp.MultiRZ(0.0, wires=[0, 1, 2])
        op = ControlledOp2(base, control_wires=[3, 4])

        simplified_op = op.simplify()
        assert isinstance(simplified_op, qp.Identity)

        base = qp.MultiRZ(0.5 + np.pi * 4, wires=[0, 1, 2])
        op = ControlledOp2(base, control_wires=[3, 4])

        simplified_op = op.simplify()
        qp.assert_equal(simplified_op, qp.ctrl(qp.MultiRZ(0.5, [0, 1, 2]), control=[3, 4]))

    def test_simplify_nested_controlled(self):
        """Tests the simplify method with nested controlled operators."""

        base = ControlledOp2(qp.MultiRZ(0.0, wires=[0, 1, 2]), control_wires=[5])
        op = ControlledOp2(base, control_wires=[3, 4])

        simplified_op = op.simplify()
        assert isinstance(simplified_op, qp.Identity)

        base = ControlledOp2(qp.MultiRZ(0.5 + np.pi * 4, wires=[0, 1, 2]), control_wires=[5])
        op = ControlledOp2(base, control_wires=[3, 4])

        simplified_op = op.simplify()
        qp.assert_equal(simplified_op, qp.ctrl(qp.MultiRZ(0.5, [0, 1, 2]), control=[3, 4, 5]))


class TestControlledOp2:
    """Tests the ControlledOp2 class."""

    def test_initialization(self):
        """Tests initializing a general controlled operator."""

        base = qp.H(0)
        op = ControlledOp2(
            base,
            control_wires=[1, 2],
            control_values=[0, 1],
            work_wires=[3],
            work_wire_type="zeroed",
        )

        assert op.base == base
        assert op.wires == Wires([1, 2, 0])
        assert op.control_wires == Wires([1, 2])
        assert op.control_values == [False, True]
        assert op.target_wires == Wires([0])
        assert op.work_wires == Wires([3])
        assert op.work_wire_type == "zeroed"

    def test_hash(self):
        """Verify that a controlled op is hashable."""

        base = qp.H(0)
        op = ControlledOp2(
            base,
            control_wires=[1, 2],
            control_values=[0, 1],
            work_wires=[3],
            work_wire_type="zeroed",
        )
        base2 = qp.H(0)
        op2 = ControlledOp2(
            base2,
            control_wires=[1, 2],
            control_values=[0, 1],
            work_wires=[3],
            work_wire_type="zeroed",
        )
        assert hash(op) == hash(op2)

    def test_representations(self):
        """Tests the representation methods."""

        base = qp.H(0)
        op = ControlledOp2(
            base,
            control_wires=[1, 2],
            control_values=[0, 1],
            work_wires=[3],
            work_wire_type="zeroed",
        )
        assert op.name == "C(Hadamard)"
        assert (
            repr(op)
            == "Controlled(H(0), control_wires=[1, 2], work_wires=[3], control_values=[False, True])"
        )
        assert op.label() == "H"

    def test_default_arguments(self):
        """Tests default values of the arguments."""

        base = qp.H(0)
        op = ControlledOp2(base, control_wires=[1, 2])
        assert op.control_values == [True, True]
        assert op.work_wires == Wires([])

    def test_single_control_value(self):
        """Tests that a single control value is wrapped."""

        base = qp.H(0)
        op = ControlledOp2(base, control_wires=[1], control_values=0)
        assert op.control_values == [False]

    def test_invalid_arguments(self):
        """Tests that the correct error is raised from invalid init arguments."""

        base = qp.H(0)

        with pytest.raises(ValueError, match="control_wires must not overlap with the base"):
            _ = ControlledOp2(base, control_wires=[0, 1])

        with pytest.raises(ValueError, match="work_wires must not overlap"):
            _ = ControlledOp2(base, control_wires=[1, 2], work_wires=[2, 3])

        with pytest.raises(ValueError, match="work_wire_type must be"):
            _ = ControlledOp2(base, control_wires=[1, 2], work_wires=[3], work_wire_type="hello")

        with pytest.raises(ValueError, match="control_values should be the same length"):
            _ = ControlledOp2(base, control_wires=[2, 1], control_values=[True])

    def test_default_compute_methods(self):
        """Tests the default implementation of compute methods."""

        base = qp.H(0)
        op = ControlledOp2(base, control_wires=[1])

        matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
                [0, 0, 1 / np.sqrt(2), -1 / np.sqrt(2)],
            ]
        )
        assert qp.math.allclose(op.matrix(), matrix)
        assert qp.math.allclose(ControlledOp2.compute_matrix(**op.arguments), matrix)

        assert qp.math.allclose(op.eigvals(), [1, 1, 1, -1])
        assert qp.math.allclose(ControlledOp2.compute_eigvals(**op.arguments), [1, 1, 1, -1])

        gates = [qp.RY(-np.pi / 4, wires=0)]
        assert op.diagonalizing_gates() == gates
        assert ControlledOp2.compute_diagonalizing_gates(**op.arguments) == gates

    def test_batching(self):
        """Tests parameter batching."""

        base = qp.Rot([0.1, 0.1], [0.2, 0.2], [0.3, 0.3], wires=1)
        op = ControlledOp2(base, control_wires=[0])

        single_matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, np.exp(-0.2j) * np.cos(0.1), -np.exp(-0.1j) * np.sin(0.1)],
                [0, 0, np.exp(0.1j) * np.sin(0.1), np.exp(0.2j) * np.cos(0.1)],
            ]
        )
        matrix = np.stack([single_matrix, single_matrix])
        assert qp.math.allclose(op.matrix(), matrix)
        assert qp.math.allclose(ControlledOp2.compute_matrix(**op.arguments), matrix)

    def test_other_methods(self):
        """Tests other operator methods for ControlledOp2."""

        base = qp.H(0)
        op = ControlledOp2(base, control_wires=[1])

        assert op.adjoint() == qp.CH([1, 0])

        base = qp.RX(0.5, wires=0)
        op = ControlledOp2(base, control_wires=[1])

        assert op.adjoint() == qp.CRX(-0.5, wires=[1, 0])
        assert op.has_generator

        generator = qp.Projector([1], wires=1) @ qp.Hamiltonian([-0.5], [qp.PauliX(0)])
        qp.assert_equal(op.generator(), generator)

    def test_create_abstract_op(self):
        """Tests creating an abstract operator."""

        op = ControlledOp2(OneWireDynOp, Wire[2])
        assert op.control_wires == Wire[2]
        assert op.target_wires == Wire[1]
        assert op.control_values == Bool[2]
        assert op.work_wires == Wire[0]
        assert op.wires == Wire[3]

        op = ControlledOp2(OneWireDynOp, Wire[2], control_values=[0, 1])
        assert op.control_values == Bool[2]

        op = ControlledOp2(OneWireDynOp, Wire[2], control_values=AbstractArray((2,), np.float64))
        assert op.control_values == Bool[2]

    def test_create_controlled_op2(self):
        """Tests qp.ctrl on Operator2 creates a ControlledOp2."""

        op = OneWireDynOp(0.5, wires=[0])
        op = qp.ctrl(OneWireDynOp(0.5, wires=[0]), control=[1], control_values=0)
        assert isinstance(op, ControlledOp2)

        op = ControlledOp2(OneWireDynOp(0.5, wires=[0]), control_wires=[], control_values=[])
        assert op.control_values == []

    @pytest.mark.parametrize(
        "copy_fn", (copy.copy, copy.deepcopy, lambda obj: pickle.loads(pickle.dumps(obj)))
    )
    def test_copy_roundtrip(self, copy_fn):
        """Test to make sure that copy roundtrips are sastisfied."""

        op = ControlledOp2(DynOp(0.5, 0), control_wires=1)
        assert op == copy_fn(op)

    def test_old_decomp_integration(self):
        """Tests that ControlledOp2 is compatible with the old decomposition convention."""

        @register_resources({qp.RX: 1})
        def _custom_decomp(wires):
            qp.RX(np.pi / 2, wires=wires)

        with qp.decomposition.local_decomps():

            qp.add_decomps(NonParametricOp, _custom_decomp)
            op = qp.ctrl(NonParametricOp(0), control=1)
            assert op.has_decomposition
            assert op.decomposition() == [qp.CRX(np.pi / 2, wires=[1, 0])]
