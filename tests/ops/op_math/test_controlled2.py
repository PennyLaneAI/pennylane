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
from typing_extensions import override

import pennylane as qp
from pennylane.ops.op_math.controlled2 import Controlled2
from pennylane.wires import Wires

# pylint: disable=unused-argument


def test_non_parametrized_custom_controlled_op():
    """Tests non-parametrized custom controlled op that directly inherits Controlled2"""

    class CH2(Controlled2):
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
            INV_SQRT2 = 1 / np.sqrt(2)
            return np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, INV_SQRT2, INV_SQRT2],
                    [0, 0, INV_SQRT2, -INV_SQRT2],
                ]
            )

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


def test_parametric_custom_controlled_op():
    """Tests parametric op that inherits from Controlled2."""

    class CRot2(Controlled2):  # pylint: disable=too-few-public-methods
        """A new CRot."""

        dynamic_argnames = ("phi", "theta", "omega")

        wires_argnames = ("wires",)

        wire_sizes = (2,)

        def __init__(self, phi, theta, omega, wires):
            super().__init__(
                qp.Rot(phi, theta, omega, wires[1]),
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
