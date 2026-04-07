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
"""Unit tests for the internal reconstruct module."""

import pytest

import pennylane as qml
from pennylane.decomposition.reconstruct import has_reconstructor, reconstruct
from pennylane.decomposition.resources import adjoint_resource_rep, pow_resource_rep, resource_rep


def test_reconstruct_gate():
    """Tests that simple gates can be reconstructed."""

    op = qml.CRX(0.5, wires=[0, 1])
    op_rep = resource_rep(op.__class__, **op.resource_params)

    assert has_reconstructor(op_rep.op_type, op_rep.params)
    reconstructed_op = reconstruct(op.data, op.wires, op_rep.op_type, op_rep.params)
    qml.assert_equal(reconstructed_op, op)


def test_adjoint_op():
    """Tests that the adjoint of a gate can be reconstructed."""

    op = qml.adjoint(qml.CRX(0.5, wires=[0, 1]))
    op_rep = adjoint_resource_rep(op.base.__class__, op.base.resource_params)

    assert has_reconstructor(op_rep.op_type, op_rep.params)
    reconstructed_op = reconstruct(op.data, op.wires, op_rep.op_type, op_rep.params)
    qml.assert_equal(reconstructed_op, op)


def test_pow_op():
    """Tests that the pow of a gate can be reconstructed."""

    op = qml.pow(qml.CRX(0.5, wires=[0, 1]), z=2)
    op_rep = pow_resource_rep(op.base.__class__, op.base.resource_params, z=2)

    assert has_reconstructor(op_rep.op_type, op_rep.params)
    reconstructed_op = reconstruct(op.data, op.wires, op_rep.op_type, op_rep.params)
    qml.assert_equal(reconstructed_op, op)


@pytest.mark.parametrize(
    "op, op_rep",
    [
        (
            qml.adjoint(qml.pow(qml.CRX(0.5, wires=[0, 1]), z=2)),
            adjoint_resource_rep(qml.ops.Pow, {"base_class": qml.CRX, "base_params": {}, "z": 2}),
        ),
        (
            qml.adjoint(qml.adjoint(qml.CRX(0.5, wires=[0, 1]))),
            adjoint_resource_rep(qml.ops.Adjoint, {"base_class": qml.CRX, "base_params": {}}),
        ),
        (
            qml.pow(qml.adjoint(qml.CRX(0.5, wires=[0, 1])), z=2),
            pow_resource_rep(qml.ops.Adjoint, {"base_class": qml.CRX, "base_params": {}}, z=2),
        ),
    ],
)
def test_nested_symbolic_op(op, op_rep):
    """Tests that a nested symbolic op can be recursively reconstructed."""

    assert has_reconstructor(op_rep.op_type, op_rep.params)
    reconstructed_op = reconstruct(op.data, op.wires, op_rep.op_type, op_rep.params)
    qml.assert_equal(reconstructed_op, op)
