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

import numpy as np
import pytest
from typing_extensions import override

import pennylane as qp
from pennylane.ops.op_math.controlled2 import Controlled2, ControlledOp2
from pennylane.wires import Wires

# pylint: disable=unused-argument,too-few-public-methods


class TestControlled2:
    """Unit tests for the Controlled2 interface."""

    def test_non_parametrized_custom_controlled_op(self):
        """Tests non-parametrized custom controlled op that directly inherits Controlled2"""

        class CH2(Controlled2, override_signature=True):
            """A new CH."""

            wire_argnames = ("wires",)

            wire_sizes = (2,)

            def __init__(self, wires):
                super().__init__(qp.H(wires[1]), wires[0], override_init_args={"wires": wires})

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

        # operator methods
        expected = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
                [0, 0, 1 / np.sqrt(2), -1 / np.sqrt(2)],
            ]
        )
        assert qp.math.allclose(op.matrix(), expected)
        assert qp.math.allclose(op.sparse_matrix(), expected)

    def test_parametric_custom_controlled_op(self):
        """Tests parametric op that inherits from Controlled2."""

        class CRot2(Controlled2, override_signature=True):
            """A new CRot."""

            dynamic_argnames = ("phi", "theta", "omega")

            wires_argnames = ("wires",)

            wire_sizes = (2,)

            def __init__(self, phi, theta, omega, wires):
                super().__init__(
                    qp.Rot(phi, theta, omega, wires=wires[1]),
                    control_wires=wires[0],
                    override_init_args={"phi": phi, "theta": theta, "omega": omega, "wires": wires},
                )

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

    def test_custom_controlled_op_default_controlled_methods(self):
        """Tests that custom controlled ops can use the default compute_xxx methods."""

        class CRot2(Controlled2, override_signature=True):
            """A new CRot."""

            dynamic_argnames = ("phi", "theta", "omega")

            wires_argnames = ("wires",)

            wire_sizes = (2,)

            def __init__(self, phi, theta, omega, wires):
                super().__init__(
                    qp.Rot(phi, theta, omega, wires=wires[1]),
                    control_wires=wires[0],
                    override_init_args={"phi": phi, "theta": theta, "omega": omega, "wires": wires},
                )

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

    def test_custom_controlled_op_own_compute_methods(self):
        """Tests when a custom controlled op override its own compute_xxx methods."""

        class CH2(Controlled2, override_signature=True):
            """A new CH."""

            wire_argnames = ("wires",)

            wire_sizes = (2,)

            def __init__(self, wires):
                super().__init__(qp.H(wires[1]), wires[0], override_init_args={"wires": wires})

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

    def test_default_arguments(self):
        """Tests default values of the arguments."""

        base = qp.H(0)
        op = ControlledOp2(base, control_wires=[1, 2])
        assert op.control_values == [True, True]
        assert op.work_wires == Wires([])

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
