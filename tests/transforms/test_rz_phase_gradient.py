# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the transform ``qml.transform.rz_phase_gradient``"""
from itertools import product

import numpy as np
import pytest

import pennylane as qml
from pennylane.transforms.rz_phase_gradient import _binary_repr_int, _rz_phase_gradient


@pytest.parametrize("string", binary_repr_int_string=list(product([0, 1], repeat=4)))
@pytest.parametrize("precision", [2, 3, 4])
def test_binary_repr_int(string, precision):
    """Test that the binary representation or approximation of the angle is correct

    In particular, this tests that phi = (c1 2^-1 + c2 2^-2 + .. + cp 2^-p + ... + 2^-N) pi
    is correctly represented as (c1, c2, .., cp) for precision p
    """
    phi = np.sum([c * 2 ** (-i - 1) for i, c in enumerate(string)]) * np.pi
    string_str = "".join([str(i) for i in string])
    binary_rep_re = np.binary_repr(_binary_repr_int(phi, precision=precision), width=precision)
    assert (
        binary_rep_re == string_str[:precision]
    ), f"nope: {binary_rep_re}, {string_str[:precision]}, {precision}"


@pytest.parametrize("precision", [2, 3, 4])
def test_units_rz_phase_gradient_private(precision):
    """Test the outputs of _rz_phase_gradient"""
    phi = (1 / 2 + 1 / 4 + 1 / 8 + 1 / 16) * np.pi
    wire = "targ"
    aux_wires = qml.wires.Wires([f"aux_{i}" for i in range(precision)])
    phase_grad_wires = qml.wires.Wires([f"qft_{i}" for i in range(precision)])
    work_wires = qml.wires.Wires([f"work_{i}" for i in range(precision - 1)])

    ops = _rz_phase_gradient(
        qml.RZ(phi, wire),
        aux_wires=aux_wires,
        phase_grad_wires=phase_grad_wires,
        work_wires=work_wires,
    )

    assert isinstance(ops[0], qml.ops.op_math.controlled.ControlledOp)
    assert np.allclose(ops[0].base.parameters, [1] * precision)
    assert ops[0].base.wires == aux_wires

    assert isinstance(ops[1], qml.SemiAdder)
    assert ops[1].wires == aux_wires + phase_grad_wires + work_wires

    assert isinstance(ops[2], qml.ops.op_math.controlled.ControlledOp)
    assert np.allclose(ops[2].base.parameters, [1] * precision)
    assert ops[2].base.wires == aux_wires
