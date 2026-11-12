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
import phase_gradient_helpers
import pytest

import pennylane as qp
from pennylane.core.operator import abstractify
from pennylane.templates.subroutines.rz_layer import _rzlayer_hwp_synthesis, _rzlayer_individual
from pennylane.testing import assert_valid_decomp_rule
from pennylane.transforms import _tape_decompose
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

# (block_size, ANDs, rotations) per block of the Hamming-weight phasing rule. It computes the
# Hamming weight of all wires of a block, except for blocks of four wires, where it computes the
# Hamming weight of three wires and rotates the fourth one separately. Each AND uses one work wire.
HWP_COSTS = [(3, 1, 2), (4, 1, 3), (5, 3, 3), (6, 4, 3), (7, 4, 3)]

# (num_angles, block_size, num_work_wires) with enough work wires for the Hamming-weight phasing
# rule and with one too few, and with few enough wires to compare with a dense matrix
HWP_EQUIVALENCE_CASES = [
    (1, m, w) for m, num_ands, _ in HWP_COSTS for w in (num_ands, num_ands - 1)
]
HWP_EQUIVALENCE_CASES += [(3, 3, 1), (3, 3, 0), (3, 4, 1), (3, 4, 0), (2, 5, 3), (2, 5, 2)]


def _random_angles(num_angles, seed):
    return np.random.default_rng(seed).uniform(-2 * np.pi, 2 * np.pi, num_angles)


def _random_rz_layer(num_angles, block_size, num_work_wires, seed):
    """An RZLayer with random angles on the wires ``0, 1, ...``, followed by its work wires."""
    num_wires = num_angles * block_size
    return qp.RZLayer(
        _random_angles(num_angles, seed),
        wires=range(num_wires),
        work_wires=range(num_wires, num_wires + num_work_wires),
    )


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

    @pytest.mark.usefixtures("enable_and_disable_capture")
    @pytest.mark.parametrize(
        "num_angles, block_size, num_work_wires",
        [(1, 3, 1), (3, 3, 0), (2, 4, 1), (1, 5, 2), (2, 6, 4), (1, 7, 0)],
    )
    def test_hwp_synthesis_rule_valid(self, num_angles, block_size, num_work_wires, seed):
        """Test that the Hamming-weight phasing rule is valid, with enough work wires and with
        too few, in which case it allocates the missing ones."""
        op = _random_rz_layer(num_angles, block_size, num_work_wires, seed)
        assert_valid_decomp_rule(op, _rzlayer_hwp_synthesis, check_matrix=False)

    @pytest.mark.usefixtures("enable_and_disable_capture")
    @pytest.mark.parametrize("num_angles, block_size, num_work_wires", HWP_EQUIVALENCE_CASES)
    def test_hwp_synthesis_realizes_rz_layer(self, num_angles, block_size, num_work_wires, seed):
        """Test that the Hamming-weight phasing rule realizes the rotations including their global
        phase and returns the work wires to zero, also if it allocates the missing ones."""
        op = _random_rz_layer(num_angles, block_size, num_work_wires, seed)

        tape = qp.testing.decomp_rule_to_tape(op, _rzlayer_hwp_synthesis)
        [tape], _ = qp.transforms.resolve_dynamic_wires(tape, min_int=len(op.wires))

        phase_gradient_helpers.assert_decomp_realizes_op(
            op, tape, [], seed, zeroed_wires=op.work_wires
        )

    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.parametrize("num_angles", [1, 3])
    @pytest.mark.parametrize("block_size, num_ands, num_rotations", HWP_COSTS)
    def test_hwp_synthesis_counts(self, num_angles, block_size, num_ands, num_rotations, seed):
        """Test the numbers of ANDs and rotations per block of the Hamming-weight phasing rule,
        which uses one global phase per block and only the work wires of the operator."""
        op = _random_rz_layer(num_angles, block_size, num_ands, seed)

        [decomposed], _ = _tape_decompose(
            qp.tape.QuantumScript([op]),
            gate_set={"CNOT", "TemporaryAND", "Adjoint(TemporaryAND)", "PhaseShift", "GlobalPhase"},
            fixed_decomps={qp.RZLayer: _rzlayer_hwp_synthesis},
        )

        ands = phase_gradient_helpers.count_ands(decomposed.operations)
        names = [o.name for o in phase_gradient_helpers.leaf_ops(decomposed.operations)]
        assert ands == (num_angles * num_ands, 0)
        assert names.count("PhaseShift") == num_angles * num_rotations
        assert names.count("GlobalPhase") == num_angles
        assert set(decomposed.wires) == set(op.wires)

    @pytest.mark.parametrize("num_angles", [1, 3])
    @pytest.mark.parametrize(
        "block_size, num_ands", [(m, num_ands) for m, num_ands, _ in HWP_COSTS]
    )
    def test_hwp_synthesis_work_wire_spec(self, num_angles, block_size, num_ands, seed):
        """Test that the Hamming-weight phasing rule only requests the missing work wires, so that
        Catalyst does not exclude it if the operator has enough work wires."""
        for num_work_wires in range(num_ands + 2):
            op = _random_rz_layer(num_angles, block_size, num_work_wires, seed)
            spec_total = phase_gradient_helpers.work_wire_spec_total(_rzlayer_hwp_synthesis, op)
            assert spec_total == max(num_ands - num_work_wires, 0)

    @pytest.mark.parametrize(
        "num_angles, block_size, num_work_wires", [(1, 3, 0), (3, 4, 1), (2, 7, 4)]
    )
    def test_hwp_synthesis_resources_from_shapes(
        self, num_angles, block_size, num_work_wires, seed
    ):
        """Test that the Hamming-weight phasing rule computes its resources and work wire spec
        from the abstract shapes ``Float[k]``, ``Wire[k*m]`` and ``Wire[w]``."""
        op = _random_rz_layer(num_angles, block_size, num_work_wires, seed)
        abstract_args = {
            "angles": Float[num_angles],
            "wires": Wire[num_angles * block_size],
            "work_wires": Wire[num_work_wires],
        }
        rule = _rzlayer_hwp_synthesis

        assert rule.is_applicable(**abstract_args)
        resources = rule.compute_resources(**abstract_args)
        assert resources.gate_counts == rule.compute_resources(**op.arguments).gate_counts
        assert rule.get_work_wire_spec(**abstract_args) == rule.get_work_wire_spec(**op.arguments)

    @pytest.mark.parametrize("num_angles, block_size", [(1, 1), (3, 1), (1, 2), (2, 2)])
    def test_hwp_synthesis_needs_three_wires_per_block(self, num_angles, block_size):
        """Test that the Hamming-weight phasing rule does not apply to blocks of fewer than three
        wires, which it would not make cheaper."""
        abstract_args = {
            "angles": Float[num_angles],
            "wires": Wire[num_angles * block_size],
            "work_wires": Wire[2],
        }
        assert not _rzlayer_hwp_synthesis.is_applicable(**abstract_args)


def test_adjoint_negates_angles():
    """Test that the adjoint negates the angles and keeps the wires and work wires."""
    op = qp.RZLayer([0.3, -1.2], wires=[0, 1, 2, 3], work_wires=[4])

    assert op.has_adjoint
    qp.assert_equal(
        op.adjoint(), qp.RZLayer(np.array([-0.3, 1.2]), wires=[0, 1, 2, 3], work_wires=[4])
    )
    assert np.allclose(qp.matrix(op.adjoint()), qp.matrix(op).conj().T)
