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

import pennylane2 as qp
from pennylane2.core.operator import abstractify
from pennylane2.templates.subroutines.rz_layer import _rzlayer_individual
from pennylane2.typing import Float, Wire
from pennylane2.wires import Wires

# (angle, wires, work_wires)
SMALL_CASES = [
    (0.4, [0], []),
    (-1.2, [0, 1, 2], []),
    (0.7, [0, 1], [2]),
    (2.5, [3, 1, 0, 5], [4, 2]),
]


def _random_angle(seed):
    return np.random.default_rng(seed).uniform(-2 * np.pi, 2 * np.pi)


class TestInitialization:
    """Tests for the arguments of RZLayer."""

    @pytest.mark.parametrize(
        "angle, wires, work_wires", SMALL_CASES + [(0.3, ["a", "b", "c"], ["w0", "w1"])]
    )
    def test_arguments(self, angle, wires, work_wires):
        """Test that the work wires are part of ``op.wires``."""
        op = qp.RZLayer(angle, wires, work_wires)

        assert op.angle == angle
        assert op.arguments["wires"] == Wires(wires)
        assert op.work_wires == Wires(work_wires)
        assert op.wires == Wires(wires + work_wires)

    def test_overlapping_wires(self):
        """Test that overlapping wires and work wires raise an error."""
        with pytest.raises(ValueError, match="must not overlap"):
            qp.RZLayer(0.1, [0, 1], [1])

    @pytest.mark.parametrize("num_wires, num_work_wires", [(1, 0), (6, 2), (8, 1)])
    def test_abstract_resource_rep(self, num_wires, num_work_wires, seed):
        """Test that the resource representation is keyed by the numbers of wires and work wires
        only, and can be built from abstract inputs."""
        op = qp.RZLayer(
            _random_angle(seed),
            wires=range(num_wires),
            work_wires=range(num_wires, num_wires + num_work_wires),
        )
        abstract_op = qp.RZLayer(Float, Wire[num_wires], Wire[num_work_wires])

        assert abstract_op.wires == Wire[num_wires + num_work_wires]
        assert abstractify(op) == abstract_op
        assert hash(abstractify(op)) == hash(abstract_op)

    @pytest.mark.catalyst
    def test_traced_angle(self):
        """Test that a traced angle is compiled into one RZLayer."""

        @qp.qjit(capture=True)
        @qp.decompose(gate_set={"Hadamard", "RZ"})
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit(phi):
            qp.H(0)
            qp.H(1)
            qp.RZLayer(phi / 2, wires=[0, 1])
            return qp.expval(qp.X(0) @ qp.X(1))

        assert qp.analyze(circuit, level=0)(0.3).resources.counts == {"Hadamard": 2, "RZLayer": 1}
        assert np.isclose(circuit(0.3), np.cos(0.15) ** 2)


@pytest.mark.usefixtures("enable_and_disable_capture")
@pytest.mark.parametrize("angle, wires, work_wires", SMALL_CASES)
def test_assert_valid(angle, wires, work_wires):
    """Run the standard Operator2 validity checks, which include those of the decomposition rules
    of RZLayer and its adjoint."""
    qp.testing.assert_valid(qp.RZLayer(angle, wires, work_wires))


class TestDecomposition:
    """Tests for the decomposition rules of RZLayer."""

    @pytest.mark.parametrize("num_wires, num_work_wires", [(1, 0), (6, 2), (7, 1)])
    def test_individual_rule_resources_from_shapes(self, num_wires, num_work_wires):
        """Test that the individual rule's resources and work wire spec are computed from the
        abstract shapes ``Float``, ``Wire[m]`` and ``Wire[w]``."""
        abstract_args = {
            "angle": Float,
            "wires": Wire[num_wires],
            "work_wires": Wire[num_work_wires],
        }

        resources = _rzlayer_individual.compute_resources(**abstract_args)

        assert resources.gate_counts == {abstractify(qp.RZ): num_wires}
        assert _rzlayer_individual.get_work_wire_spec(**abstract_args).total == 0

    @pytest.mark.catalyst
    @pytest.mark.parametrize("num_wires", [1, 5])
    def test_qjit_graph_decomposition(self, num_wires, seed):
        """Test that, compiled with the graph, an RZLayer and its adjoint apply one RZ per wire,
        with ``angle`` and ``-angle``. On ``|+>``, ``RZ(a)`` gives ``<X> = cos(a)`` and
        ``<Y> = sin(a)``."""
        angle = _random_angle(seed)

        @qp.qjit(capture=True)
        @qp.decompose(gate_set={"Hadamard", "RZ"})
        @qp.qnode(qp.device("lightning.qubit", wires=2 * num_wires + 1))
        def circuit(angle):
            for wire in range(2 * num_wires):
                qp.H(wire)
            qp.RZLayer(angle, wires=range(num_wires), work_wires=[2 * num_wires])
            qp.adjoint(qp.RZLayer(angle, wires=range(num_wires, 2 * num_wires)))
            return [qp.expval(qp.X(w)) for w in range(2 * num_wires)] + [
                qp.expval(qp.Y(w)) for w in range(2 * num_wires)
            ]

        expected = np.repeat([angle, -angle], num_wires)
        assert np.allclose(circuit(angle), np.concatenate([np.cos(expected), np.sin(expected)]))
        counts = qp.analyze(circuit, level="device")(angle).resources.counts
        assert counts == {"Hadamard": 2 * num_wires, "RZ": 2 * num_wires}


def test_rule_names():
    """Test the names of the decomposition rules, which select them in ``fixed_decomps``."""
    assert "to_rz" in [rule.name for rule in qp.list_decomps(qp.RZLayer)]
    assert "negate_angle" in [rule.name for rule in qp.list_decomps("Adjoint(RZLayer)")]


def test_adjoint_negates_angle():
    """Test that the adjoint negates the angle and keeps the wires and work wires."""
    op = qp.RZLayer(0.3, wires=[0, 1, 2], work_wires=[3])

    assert op.has_adjoint
    qp.assert_equal(op.adjoint(), qp.RZLayer(-0.3, wires=[0, 1, 2], work_wires=[3]))
    assert np.allclose(qp.matrix(op.adjoint()), qp.matrix(op).conj().T)
