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
"""Tests for the RZLayer template."""

import numpy as np
import pytest

import pennylane as qp
from pennylane.core.operator import abstractify
from pennylane.templates.subroutines.rz_layer import _rzlayer_individual
from pennylane.typing import Float, Wire
from pennylane.wires import Wires

# (angles, wires, work_wires)
SMALL_CASES = [
    ([0.4], [0], []),
    ([0.4], [0, 1, 2], []),
    ([0.3, -1.2, 2.5], [0, 1, 2], []),
    ([0.3, -1.2], [0, 1, 2, 3], []),
    ([0.7], [0, 1], [2]),
    ([0.3, -1.2], [3, 1, 0, 5], [4, 2]),
]


def _random_angles(num_angles, seed):
    return np.random.default_rng(seed).uniform(-2 * np.pi, 2 * np.pi, num_angles)


def _rz_layer_circuit(angles):
    """Apply an RZLayer with the angles ``(a, b)`` on two blocks of two wires to ``|+>`` on all
    four wires, and return the expectation value ``cos(a)**2 * cos(b)**2`` of X on all wires."""
    for wire in range(4):
        qp.H(wire)
    qp.RZLayer(angles, wires=range(4))
    return qp.expval(qp.X(0) @ qp.X(1) @ qp.X(2) @ qp.X(3))


def _rz_layer_circuit_grad(angles):
    """Analytic gradient of :func:`_rz_layer_circuit`."""
    a, b = angles
    return np.array([-np.sin(2 * a) * np.cos(b) ** 2, -np.cos(a) ** 2 * np.sin(2 * b)])


class TestInitialization:
    """Tests for the arguments of RZLayer."""

    @pytest.mark.parametrize(
        "angles, wires, work_wires",
        SMALL_CASES + [([0.3, -1.2], ["a", "b", "c", "d"], ["w0", "w1"])],
    )
    def test_arguments(self, angles, wires, work_wires):
        """Test that the angles are stored as an array and that the work wires are part of
        ``op.wires``."""
        op = qp.RZLayer(angles, wires, work_wires)

        assert isinstance(op.angles, np.ndarray)
        assert np.allclose(op.angles, angles)
        assert op.arguments["wires"] == Wires(wires)
        assert op.work_wires == Wires(work_wires)
        assert op.wires == Wires(wires + work_wires)

    @pytest.mark.parametrize(
        "angles, wires, work_wires, match",
        [
            ([[0.1, 0.2]], [0, 1], [], "one-dimensional"),
            (Float[1, 2], Wire[2], Wire[0], "one-dimensional"),
            ([], [0, 1], [], "at least one angle"),
            (Float[0], Wire[2], Wire[0], "at least one angle"),
            ([0.1, 0.2], [0, 1, 2], [], "multiple of the number of angles"),
            (Float[2], Wire[3], Wire[0], "multiple of the number of angles"),
            ([0.1], [0, 1], [1], "must not overlap"),
        ],
    )
    def test_invalid_arguments(self, angles, wires, work_wires, match):
        """Test that invalid shapes and overlapping wires raise an error."""
        with pytest.raises(ValueError, match=match):
            qp.RZLayer(angles, wires, work_wires)

    @pytest.mark.parametrize(
        "num_angles, block_size, num_work_wires", [(1, 1, 0), (3, 2, 2), (2, 4, 1)]
    )
    def test_abstract_resource_rep(self, num_angles, block_size, num_work_wires, seed):
        """Test that the resource representation is keyed by the numbers of angles, wires and
        work wires only, and can be built from abstract inputs."""
        num_wires = num_angles * block_size
        op = qp.RZLayer(
            _random_angles(num_angles, seed),
            wires=range(num_wires),
            work_wires=range(num_wires, num_wires + num_work_wires),
        )
        abstract_op = qp.RZLayer(Float[num_angles], Wire[num_wires], Wire[num_work_wires])

        assert abstract_op.wires == Wire[num_wires + num_work_wires]
        assert abstractify(op) == abstract_op
        assert hash(abstractify(op)) == hash(abstract_op)

    @pytest.mark.catalyst
    @pytest.mark.usefixtures("enable_graph_decomposition")
    def test_traced_angles_in_list(self):
        """Test that angles given as a list of traced scalars are compiled into one RZLayer."""
        pytest.importorskip("catalyst")

        @qp.qjit(capture=True)
        @qp.decompose(gate_set={"Hadamard", "RZ"})
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit(phi):
            qp.H(0)
            qp.H(1)
            qp.RZLayer([phi / 2], wires=[0, 1])
            return qp.expval(qp.X(0) @ qp.X(1))

        assert qp.analyze(circuit, level=0)(0.3).resources.counts == {"Hadamard": 2, "RZLayer": 1}
        assert np.isclose(circuit(0.3), np.cos(0.15) ** 2)


@pytest.mark.usefixtures("enable_and_disable_capture")
@pytest.mark.parametrize("angles, wires, work_wires", SMALL_CASES)
def test_assert_valid(angles, wires, work_wires):
    """Run the standard Operator2 validity checks, which include those of the decomposition rules
    of RZLayer and its adjoint."""
    qp.testing.assert_valid(qp.RZLayer(angles, wires, work_wires))


class TestDecomposition:
    """Tests for the decomposition rules of RZLayer."""

    @pytest.mark.parametrize(
        "num_angles, block_size, num_work_wires", [(1, 1, 0), (3, 2, 2), (2, 7, 1)]
    )
    def test_individual_rule_resources_from_shapes(self, num_angles, block_size, num_work_wires):
        """Test that the individual rule's resources and work wire spec are computed from the
        abstract shapes ``Float[k]``, ``Wire[k*m]`` and ``Wire[w]``."""
        abstract_args = {
            "angles": Float[num_angles],
            "wires": Wire[num_angles * block_size],
            "work_wires": Wire[num_work_wires],
        }

        resources = _rzlayer_individual.compute_resources(**abstract_args)

        assert resources.gate_counts == {abstractify(qp.RZ): num_angles * block_size}
        assert _rzlayer_individual.get_work_wire_spec(**abstract_args).total == 0

    @pytest.mark.catalyst
    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.parametrize("num_angles, block_size", [(1, 1), (1, 5), (3, 2)])
    def test_qjit_graph_decomposition(self, num_angles, block_size, seed):
        """Test that, compiled with the graph, an RZLayer and its adjoint apply one RZ per wire,
        with ``angles[i]`` and ``-angles[i]`` on block ``i``. On ``|+>``, ``RZ(a)`` gives
        ``<X> = cos(a)`` and ``<Y> = sin(a)``."""
        pytest.importorskip("catalyst")
        angles = _random_angles(num_angles, seed)
        num_wires = num_angles * block_size

        @qp.qjit(capture=True)
        @qp.decompose(gate_set={"Hadamard", "RZ"})
        @qp.qnode(qp.device("lightning.qubit", wires=2 * num_wires + 1))
        def circuit(angles):
            for wire in range(2 * num_wires):
                qp.H(wire)
            qp.RZLayer(angles, wires=range(num_wires), work_wires=[2 * num_wires])
            qp.adjoint(qp.RZLayer(angles, wires=range(num_wires, 2 * num_wires)))
            return [qp.expval(qp.X(w)) for w in range(2 * num_wires)] + [
                qp.expval(qp.Y(w)) for w in range(2 * num_wires)
            ]

        expected = np.concatenate([np.repeat(angles, block_size), -np.repeat(angles, block_size)])
        assert np.allclose(circuit(angles), np.concatenate([np.cos(expected), np.sin(expected)]))
        counts = qp.analyze(circuit, level="device")(angles).resources.counts
        assert counts == {"Hadamard": 2 * num_wires, "RZ": 2 * num_wires}


class TestInterfaces:
    """Tests for the gradient of RZLayer with torch and under ``qjit``."""

    @pytest.mark.torch
    def test_torch(self):
        """Test that the gradient with torch is the analytic gradient."""
        import torch

        angles = [0.3, -1.2]
        circuit = qp.QNode(_rz_layer_circuit, qp.device("default.qubit", wires=4))

        torch_angles = torch.tensor(angles, dtype=torch.float64, requires_grad=True)
        circuit(torch_angles).backward()
        assert np.allclose(torch_angles.grad, _rz_layer_circuit_grad(angles))

    @pytest.mark.catalyst
    def test_qjit(self):
        """Test that the expectation value and its gradient are the analytic ones under ``qjit``."""
        pytest.importorskip("catalyst")
        angles = np.array([0.3, -1.2])
        circuit = qp.QNode(_rz_layer_circuit, qp.device("lightning.qubit", wires=4))

        assert np.isclose(qp.qjit(circuit)(angles), np.prod(np.cos(angles) ** 2))
        assert np.allclose(qp.qjit(qp.grad(circuit))(angles), _rz_layer_circuit_grad(angles))


def test_adjoint_negates_angles():
    """Test that the adjoint negates the angles and keeps the wires and work wires."""
    op = qp.RZLayer([0.3, -1.2], wires=[0, 1, 2, 3], work_wires=[4])

    assert op.has_adjoint
    qp.assert_equal(
        op.adjoint(), qp.RZLayer(np.array([-0.3, 1.2]), wires=[0, 1, 2, 3], work_wires=[4])
    )
    assert np.allclose(qp.matrix(op.adjoint()), qp.matrix(op).conj().T)
