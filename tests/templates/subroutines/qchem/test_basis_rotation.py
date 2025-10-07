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
Tests for the BasisRotation template.
"""

import numpy as np
import pytest

import pennylane as qml
from pennylane.ops.functions.assert_valid import _test_decomposition_rule


@pytest.mark.jax
@pytest.mark.parametrize(
    "rotation",
    [
        np.array(
            [
                [0.1478133, 0.58295722, 0.79661853, 0.06091815],
                [-0.34142933, -0.60933128, 0.54474544, -0.46410538],
                [-0.76014658, 0.01047548, 0.0841176, 0.64419847],
                [0.53268604, -0.53737001, 0.24814422, 0.60489958],
            ]
        ),  # orthogonal matrix with determinant -1
        np.array(
            [
                [-0.25619233, 0.81407233, 0.52120219],
                [-0.72789572, -0.5172592, 0.45012302],
                [0.63602933, -0.26406278, 0.72507761],
            ]
        ),  # orthogonal matrix with determinant 1
        np.array(
            [
                [-0.618452, -0.68369054 - 0.38740723j],
                [-0.78582258, 0.53807284 + 0.30489424j],
            ]
        ),  # unitary matrix
    ],
)
def test_standard_validity(rotation):
    """Run standard tests of operation validity."""
    op = qml.BasisRotation(wires=range(len(rotation)), unitary_matrix=rotation)
    qml.ops.functions.assert_valid(op)


class TestDecomposition:
    """Test that the template defines the correct decomposition."""

    @pytest.mark.parametrize(
        ("num_wires", "unitary_matrix", "givens", "diags"),
        [
            (
                2,
                np.array(
                    [
                        [-0.618452, -0.68369054 - 0.38740723j],
                        [-0.78582258, 0.53807284 + 0.30489424j],
                    ]
                ),
                [([0], 2.626062920217307), ([0, 1], 1.808050120433623)],
                [([0], 0.5155297333724864), ([1], 0.5155297333724864)],
            ),
            (
                3,
                np.array(
                    [
                        [0.51378719 + 0.0j, 0.0546265 + 0.79145487j, -0.2051466 + 0.2540723j],
                        [0.62651582 + 0.0j, -0.00828925 - 0.60570321j, -0.36704948 + 0.32528067j],
                        [-0.58608928 + 0.0j, 0.03902657 + 0.04633548j, -0.57220635 + 0.57044649j],
                    ]
                ),
                [
                    ([0], 2.2707802713289267),
                    ([0, 1], 2.9355948424220206),
                    ([1], -1.4869222527726533),
                    ([1, 2], 1.2601662579297865),
                    ([0], 2.3559705032936717),
                    ([0, 1], 1.1748572730890159),
                ],
                [([0], 2.2500537657656356), ([1], -0.7251404204443089), ([2], 2.3577346350335198)],
            ),
        ],
    )
    def test_basis_rotation_operations_complex(self, num_wires, unitary_matrix, givens, diags):
        """Test the correctness of the BasisRotation template including the gate count
        and their order, the wires the operation acts on and the correct use of parameters
        in the circuit."""

        gate_ops, gate_angles, gate_wires = [], [], []

        for indices, angle in diags + givens[::-1]:
            g_op = qml.PhaseShift if len(indices) == 1 else qml.SingleExcitation
            gate_ops.append(g_op)
            gate_angles.append(qml.numpy.array(angle))
            gate_wires.append(list(indices))

        op = qml.BasisRotation(wires=range(num_wires), unitary_matrix=unitary_matrix)
        queue = op.decomposition()

        assert len(queue) == len(gate_ops)  # number of gates

        for idx, _op in enumerate(queue):
            assert isinstance(_op, gate_ops[idx])  # gate operation
            assert np.allclose(_op.parameters[0], gate_angles[idx])  # gate parameter
            assert list(_op.wires) == gate_wires[idx]  # gate wires

        # Tests the decomposition rule defined with the new system
        for rule in qml.list_decomps(qml.BasisRotation):
            _test_decomposition_rule(op, rule)

    @pytest.mark.parametrize(
        ("num_wires", "ortho_matrix", "givens"),
        [
            (
                2,
                np.array(
                    [  # A single Givens matrix obtained from sin(0.61246) and cos(0.61246)
                        [0.8182362852252838, 0.5748820588092205],
                        [-0.5748820588092205, 0.8182362852252838],
                    ]
                ),
                [([0, 1], 2 * 0.61246)],
            ),
            (
                2,
                np.array(
                    [  # Same as above with flipped first row
                        [-0.8182362852252838, -0.5748820588092205],
                        [-0.5748820588092205, 0.8182362852252838],
                    ]
                ),
                [([0, 1], -2 * 0.61246), ([0], np.pi)],
            ),
            (
                3,
                np.array(
                    [  # Random orthogonal matrix with determinant -1
                        [0.41938787, 0.36647513, 0.83054789],
                        [-0.83748936, 0.50924661, 0.19819045],
                        [0.35032183, 0.77869369, -0.52049088],
                    ]
                ),
                [
                    ([0, 1], 2 * 0.42275745323754343),
                    ([1, 2], -2 * 2.118222067141928),
                    ([0, 1], -2 * 1.805041881040138),
                    ([0], np.pi),
                ],
            ),
        ],
    )
    def test_basis_rotation_operations_real(self, num_wires, ortho_matrix, givens):
        """Test the correctness of the BasisRotation template including the gate count
        and their order, the wires the operation acts on and the correct use of parameters
        in the circuit."""

        gate_ops, gate_angles, gate_wires = [], [], []

        for indices, angle in givens[::-1]:
            g_op = qml.PhaseShift if len(indices) == 1 else qml.SingleExcitation
            gate_ops.append(g_op)
            gate_angles.append(qml.numpy.array(angle))
            gate_wires.append(list(indices))

        op = qml.BasisRotation(wires=range(num_wires), unitary_matrix=ortho_matrix)
        queue = op.decomposition()

        assert len(queue) == len(gate_ops)  # number of gates
        assert [type(op) for op in queue].count(qml.PhaseShift) <= 1  # at most one phase shift

        for idx, _op in enumerate(queue):
            assert isinstance(_op, gate_ops[idx])  # gate operation
            assert np.allclose(_op.parameters[0], gate_angles[idx])  # gate parameter
            assert list(_op.wires) == gate_wires[idx]  # gate wires

        # Tests the decomposition rule defined with the new system
        for rule in qml.list_decomps(qml.BasisRotation):
            _test_decomposition_rule(op, rule)

    def test_custom_wire_labels(self, tol):
        """Test that BasisRotation template can deal with non-numeric, nonconsecutive wire labels."""

        weights = qml.math.array(
            [
                [-0.618452, -0.68369054 - 0.38740723j],
                [-0.78582258, 0.53807284 + 0.30489424j],
            ]
        )

        dev = qml.device("default.qubit", wires=2)
        dev2 = qml.device("default.qubit", wires=["z", "a"])

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(wires=[0])
            qml.BasisRotation(
                wires=range(2),
                unitary_matrix=weights,
            )
            return qml.expval(qml.Identity(0)), qml.state()

        @qml.qnode(dev2)
        def circuit2():
            qml.PauliX(wires=["z"])
            qml.BasisRotation(
                wires=["z", "a"],
                unitary_matrix=weights,
            )
            return qml.expval(qml.Identity("z")), qml.state()

        res1, state1 = circuit()
        res2, state2 = circuit2()

        assert np.allclose(res1, res2, atol=tol, rtol=0)
        assert np.allclose(state1, state2, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        ("unitary_matrix", "eigen_values", "exp_state"),
        [
            (
                qml.math.array(
                    [
                        [-0.77228482 + 0.0j, -0.02959195 + 0.63458685j],
                        [0.63527644 + 0.0j, -0.03597397 + 0.77144651j],
                    ]
                ),
                qml.math.array([-1.45183325, 3.47550075]),
                qml.math.array([0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -0.43754907 - 0.89919453j]),
            ),
            (
                qml.math.array(
                    [
                        [0.51378719 + 0.0j, 0.0546265 + 0.79145487j, -0.2051466 + 0.2540723j],
                        [0.62651582 + 0.0j, -0.00828925 - 0.60570321j, -0.36704948 + 0.32528067j],
                        [-0.58608928 + 0.0j, 0.03902657 + 0.04633548j, -0.57220635 + 0.57044649j],
                    ]
                ),
                qml.math.array([-5.91470283, 2.16643686, 3.63229049]),
                qml.math.array(
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        -0.62435779 - 0.15734516j,
                        0.0 + 0.0j,
                        -0.40747865 - 0.20203303j,
                        -0.59904663 - 0.14038088j,
                        0.0 + 0.0j,
                    ]
                ),
            ),
            (
                qml.math.array(
                    [
                        [
                            -0.4544554 + 0.0j,
                            -0.42988728 + 0.08855856j,
                            0.58491507 + 0.30555973j,
                            0.1587327 - 0.37087138j,
                            -0.05022721 - 0.00823365j,
                        ],
                        [
                            0.24312842 + 0.0j,
                            0.10441764 - 0.10283026j,
                            0.14069409 - 0.3754747j,
                            0.6430079 - 0.21405862j,
                            -0.22627141 - 0.49815298j,
                        ],
                        [
                            0.44622072 + 0.0j,
                            0.05853879 - 0.10096448j,
                            0.34305113 + 0.36664395j,
                            -0.46947989 + 0.06727886j,
                            -0.40958506 - 0.37743457j,
                        ],
                        [
                            -0.3381196 + 0.0j,
                            0.06444382 - 0.77075815j,
                            0.13920109 - 0.30283802j,
                            -0.30520162 - 0.02845482j,
                            0.20586607 - 0.20006912j,
                        ],
                        [
                            0.6487843 + 0.0j,
                            -0.3469304 - 0.23167812j,
                            0.19359466 - 0.05525374j,
                            0.03406439 - 0.24067031j,
                            0.43860444 + 0.33623679j,
                        ],
                    ]
                ),
                qml.math.array([-7.74725949, -3.90779537, -1.41680735, 2.54903347, 5.60305197]),
                qml.math.array(
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.03382593 + 0.29268384j,
                        0.0 + 0.0j,
                        -0.1198482 + 0.16323789j,
                        0.13892594 - 0.22857563j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        -0.0291709 + 0.11823507j,
                        -0.13624626 - 0.09903234j,
                        0.0 + 0.0j,
                        -0.10912238 - 0.04514361j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.34983261 - 0.19352721j,
                        -0.47737227 - 0.24440944j,
                        0.0 + 0.0j,
                        0.23242109 - 0.29001262j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        -0.0956275 - 0.38569494j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ]
                ),
            ),
        ],
    )
    def test_basis_rotation_unitary(self, unitary_matrix, eigen_values, exp_state):
        """Test that the BasisRotation template works correctly asserting the prepared state."""

        wires = range(len(unitary_matrix))
        dev = qml.device("default.qubit", wires=wires)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(0)
            qml.PauliX(1)
            qml.adjoint(qml.BasisRotation(wires=wires, unitary_matrix=unitary_matrix))
            for idx, eigenval in enumerate(eigen_values):
                qml.RZ(-eigenval, wires=[idx])
            qml.BasisRotation(wires=wires, unitary_matrix=unitary_matrix)
            return qml.state()

        assert np.allclose([qml.math.fidelity_statevector(circuit(), exp_state)], [1.0], atol=1e-6)


class TestInputs:
    """Test inputs and pre-processing."""

    @pytest.mark.parametrize(
        ("wires", "unitary_matrix", "msg_match"),
        [
            (
                [0, 1, 2],
                np.array(
                    [
                        [0.51378719 + 0.0j, 0.0546265 + 0.79145487j, -0.2051466 + 0.2540723j],
                        [0.62651582 + 0.0j, -0.00828925 - 0.60570321j, -0.36704948 + 0.32528067j],
                    ]
                ),
                "The unitary matrix should be of shape NxN",
            ),
            (
                [0, 1, 2],
                np.array(
                    [
                        [0.21378719 + 0.0j, 0.0546265 - 0.79145487j, -0.2051466 + 0.2540723j],
                        [0.0 + 0.0j, -0.00821925 - 0.60570321j, -0.36704948 + 0.32528067j],
                        [-0.0 + 0.0j, 0.03902657 + 0.04633548j, -0.57220635 + 0.57044649j],
                    ]
                ),
                "The provided transformation matrix should be unitary.",
            ),
            (
                [0],
                np.array([[1.0]]),
                "This template requires at least two wires",
            ),
        ],
    )
    def test_basis_rotation_exceptions(self, wires, unitary_matrix, msg_match):
        """Test that BasisRotation template throws an exception if the parameters have illegal
        shapes, types or values."""

        dev = qml.device("default.qubit", wires=len(wires))

        @qml.qnode(dev)
        def circuit():
            qml.BasisRotation(wires=wires, unitary_matrix=unitary_matrix, check=True)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match=msg_match):
            circuit()

        with pytest.raises(ValueError, match=msg_match):
            qml.BasisRotation.compute_decomposition(
                wires=wires, unitary_matrix=unitary_matrix, check=True
            )

    def test_id(self):
        """Test that the id attribute can be set."""
        template = qml.BasisRotation(
            wires=range(2),
            unitary_matrix=qml.math.array(
                [
                    [-0.77228482 + 0.0j, -0.02959195 + 0.63458685j],
                    [0.63527644 + 0.0j, -0.03597397 + 0.77144651j],
                ]
            ),
            id="a",
        )
        assert template.id == "a"


def circuit_template(unitary_matrix, check=False):
    qml.BasisState(qml.math.array([1, 1, 0]), wires=[0, 1, 2])
    qml.BasisRotation(
        wires=range(3),
        unitary_matrix=unitary_matrix,
        check=check,
    )
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


def circuit_decomposed(weights):
    qml.BasisState(np.array([1, 1, 0]), wires=[0, 1, 2])
    qml.PhaseShift(weights[6], wires=[0])
    qml.PhaseShift(weights[7], wires=[1])
    qml.PhaseShift(weights[8], wires=[2])
    qml.SingleExcitation(weights[5], wires=[0, 1])
    qml.PhaseShift(weights[4], wires=[0])
    qml.SingleExcitation(weights[3], wires=[1, 2])
    qml.PhaseShift(weights[2], wires=[1])
    qml.SingleExcitation(weights[1], wires=[0, 1])
    qml.PhaseShift(weights[0], wires=[0])

    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


class TestInterfaces:
    """Test that the template is compatible with all interfaces, including the computation
    of gradients."""

    @pytest.mark.autograd
    def test_autograd(self, tol):
        """Test the autograd interface."""

        unitary_matrix = qml.numpy.array(
            [
                [0.51378719 + 0.0j, 0.0546265 + 0.79145487j, -0.2051466 + 0.2540723j],
                [0.62651582 + 0.0j, -0.00828925 - 0.60570321j, -0.36704948 + 0.32528067j],
                [-0.58608928 + 0.0j, 0.03902657 + 0.04633548j, -0.57220635 + 0.57044649j],
            ]
        )
        weights = qml.numpy.array(
            [
                2.2707802713289267,  # 0
                2.9355948424220206,  # 1
                -1.4869222527726533,  # 2
                1.2601662579297865,  # 3
                2.3559705032936717,  # 4
                1.1748572730890159,  # 5
                2.2500537657656356,  # 6
                -0.7251404204443089,  # 7
                2.3577346350335198,  # 8
            ]
        )

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(unitary_matrix)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = qml.grad(circuit)
        grads = grad_fn(unitary_matrix)

        assert np.allclose(grads, np.zeros_like(unitary_matrix, dtype=complex), atol=tol, rtol=0)

    @pytest.mark.parametrize("device_name", ("default.qubit", "reference.qubit"))
    @pytest.mark.jax
    def test_jax_jit(self, device_name, tol):
        """Test the jax interface."""

        import jax
        import jax.numpy as jnp

        unitary_matrix = jnp.array(
            [
                [0.51378719 + 0.0j, 0.0546265 + 0.79145487j, -0.2051466 + 0.2540723j],
                [0.62651582 + 0.0j, -0.00828925 - 0.60570321j, -0.36704948 + 0.32528067j],
                [-0.58608928 + 0.0j, 0.03902657 + 0.04633548j, -0.57220635 + 0.57044649j],
            ]
        )

        dev = qml.device(device_name, wires=3)

        circuit = jax.jit(qml.QNode(circuit_template, dev), static_argnames="check")
        circuit2 = qml.QNode(circuit_template, dev)

        res = circuit(unitary_matrix)
        res2 = circuit2(unitary_matrix)
        res3 = circuit2(qml.math.toarray(unitary_matrix))

        assert qml.math.allclose(res, res2, atol=tol, rtol=0)
        assert qml.math.allclose(res, res3, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        grads = grad_fn(unitary_matrix)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(unitary_matrix)

        assert qml.math.allclose(grads, grads2, atol=tol, rtol=0)

    @pytest.mark.slow
    @pytest.mark.tf
    def test_tf(self, tol):
        """Test the tf interface."""

        import tensorflow as tf

        unitary_matrix = tf.Variable(
            [
                [0.51378719 + 0.0j, 0.0546265 + 0.79145487j, -0.2051466 + 0.2540723j],
                [0.62651582 + 0.0j, -0.00828925 - 0.60570321j, -0.36704948 + 0.32528067j],
                [-0.58608928 + 0.0j, 0.03902657 + 0.04633548j, -0.57220635 + 0.57044649j],
            ]
        )
        weights = tf.Variable(
            [
                2.2707802713289267,
                2.9355948424220206,
                -1.4869222527726533,
                1.2601662579297865,
                2.3559705032936717,
                1.1748572730890159,
                2.2500537657656356,
                -0.7251404204443089,
                2.3577346350335198,
            ]
        )

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(unitary_matrix)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        with tf.GradientTape() as tape:
            res = circuit(unitary_matrix)
        grads = tape.gradient(res, [unitary_matrix])

        assert grads[0] is None

    @pytest.mark.torch
    def test_torch(self, tol):
        """Test the torch interface."""

        import torch

        unitary_matrix = torch.tensor(
            [
                [0.51378719 + 0.0j, 0.0546265 + 0.79145487j, -0.2051466 + 0.2540723j],
                [0.62651582 + 0.0j, -0.00828925 - 0.60570321j, -0.36704948 + 0.32528067j],
                [-0.58608928 + 0.0j, 0.03902657 + 0.04633548j, -0.57220635 + 0.57044649j],
            ],
            requires_grad=False,
        )
        weights = torch.tensor(
            [
                2.2707802713289267,
                2.9355948424220206,
                -1.4869222527726533,
                1.2601662579297865,
                2.3559705032936717,
                1.1748572730890159,
                2.2500537657656356,
                -0.7251404204443089,
                2.3577346350335198,
            ],
            requires_grad=False,
        )

        dev = qml.device("default.qubit", wires=3)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(unitary_matrix)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)
