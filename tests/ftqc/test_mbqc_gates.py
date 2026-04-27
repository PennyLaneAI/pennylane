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
# pylint: disable=no-name-in-module, no-self-use, protected-access
"""Tests for expressing gates in the MBQC representation"""

from functools import partial

import networkx as nx
import numpy as np
import pytest

import pennylane as qp
from pennylane.ftqc import generate_lattice


def generate_random_states(n, n_qubit=1, seed=None):
    """Generate `n` random initial `n_qubit`-states."""
    rng = np.random.default_rng(seed=seed)
    input_state = rng.random((n, 2**n_qubit)) + 1j * rng.random((n, 2**n_qubit))
    input_state /= np.linalg.norm(input_state, axis=1).reshape(-1, 1)

    if n == 1:
        return input_state[0]

    return input_state


def generate_random_rotation_angles(n, lo=0, hi=4 * np.pi, seed=None):
    """Generate `n` random rotation angles on the interval [lo, hi)."""
    rng = np.random.default_rng(seed=seed)
    angles = rng.uniform(lo, hi, n)

    if n == 1:
        return angles[0]

    return angles


@pytest.mark.system
class TestIndividualGates:
    """System-level tests to check that individual gates expressed in the MBQC formalism give
    correct results.
    """

    def test_rz_in_mbqc_representation(self, seed):
        """Test that the RZ gate in the MBQC representation gives correct results."""
        dev = qp.device("default.qubit")

        # Reference RZ circuit
        @qp.qnode(dev)
        def circuit_ref(start_state, angle):
            qp.StatePrep(start_state, wires=0)
            qp.RZ(angle, wires=0)
            return qp.expval(qp.X(0)), qp.expval(qp.Y(0)), qp.expval(qp.Z(0))

        # Define the graph structure for the RZ cluster state (omit node 1 for input):
        # 1 -- 2 -- 3 -- 4 -- 5
        lattice = generate_lattice([4], "chain")

        # Equivalent RZ circuit in the MBQC representation
        @qp.ftqc.diagonalize_mcms
        @qp.qnode(dev, mcm_method="tree-traversal")
        def circuit_mbqc(start_state, angle):
            # prep input node
            qp.StatePrep(start_state, wires=[1])

            # prep graph state
            qp.ftqc.GraphStatePrep(lattice.graph, wires=[2, 3, 4, 5])

            # entangle input and graph state
            qp.CZ([1, 2])

            # RZ measurement pattern from Raussendorf, Browne & Briegel (2003)
            m0 = qp.ftqc.measure_x(1)
            m1 = qp.ftqc.measure_x(2)
            m2 = qp.ftqc.cond_measure(
                m1,
                partial(qp.ftqc.measure_arbitrary_basis, angle=angle),
                partial(qp.ftqc.measure_arbitrary_basis, angle=-angle),
            )(plane="XY", wires=3)
            m3 = qp.ftqc.measure_x(4)

            # corrections based on measurement outcomes
            qp.cond((m0 + m2) % 2, qp.Z)(5)
            qp.cond((m1 + m3) % 2, qp.X)(5)

            return qp.expval(qp.X(5)), qp.expval(qp.Y(5)), qp.expval(qp.Z(5))

        initial_state = generate_random_states(n=1, n_qubit=1, seed=seed)
        rz_angle = generate_random_rotation_angles(n=1, seed=seed)

        result_ref = circuit_ref(initial_state, rz_angle)
        result_mbqc = circuit_mbqc(initial_state, rz_angle)

        assert np.allclose(result_ref, result_mbqc)

    def test_cnot_in_mbqc_representation(self, seed):
        """Test that the CNOT gate in the MBQC representation gives correct results."""
        dev = qp.device("default.qubit")

        # Reference CNOT circuit
        @qp.qnode(dev)
        def circuit_ref(start_state):
            qp.StatePrep(start_state, wires=[0, 1])
            qp.CNOT([0, 1])
            return qp.expval(qp.Z(0)), qp.expval(qp.Z(1))

        # Define the graph structure for the CNOT cluster state (omit nodes 1 and 9 for input)
        # 1 -- 2 -- 3 -- 4 -- 5 -- 6 -- 7
        #                |
        #                8
        #                |
        # 9 -- 10 - 11 - 12 - 13 - 14 - 15
        wires = [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]
        g = nx.Graph()
        g.add_nodes_from(wires)
        g.add_edges_from(
            [
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (4, 8),
                (8, 12),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
            ]
        )

        # Equivalent CNOT circuit in the MBQC representation
        @qp.ftqc.diagonalize_mcms
        @qp.qnode(dev, mcm_method="tree-traversal")
        def circuit_mbqc(start_state):
            qp.StatePrep(start_state, wires=[1, 9])

            # prep graph state
            qp.ftqc.GraphStatePrep(g, wires=wires)

            # entangle
            qp.CZ([1, 2])
            qp.CZ([9, 10])

            # CNOT measurement pattern from Raussendorf, Browne & Briegel (2003)
            m1 = qp.ftqc.measure_x(1)
            m2 = qp.ftqc.measure_y(2)
            m3 = qp.ftqc.measure_y(3)
            m4 = qp.ftqc.measure_y(4)
            m5 = qp.ftqc.measure_y(5)
            m6 = qp.ftqc.measure_y(6)
            m8 = qp.ftqc.measure_y(8)
            m9 = qp.ftqc.measure_x(9)
            m10 = qp.ftqc.measure_x(10)
            m11 = qp.ftqc.measure_x(11)
            m12 = qp.ftqc.measure_y(12)
            m13 = qp.ftqc.measure_x(13)
            m14 = qp.ftqc.measure_x(14)

            # corrections on controls
            x_cor = m2 + m3 + m5 + m6
            z_cor = m1 + m3 + m4 + m5 + m8 + m9 + m11 + 1
            qp.cond(z_cor % 2, qp.Z)(7)
            qp.cond(x_cor % 2, qp.X)(7)

            # corrections on target
            x_cor = m2 + m3 + m8 + m10 + m12 + m14
            z_cor = m9 + m11 + m13
            qp.cond(z_cor % 2, qp.Z)(15)
            qp.cond(x_cor % 2, qp.X)(15)

            return qp.expval(qp.Z(7)), qp.expval(qp.Z(15))

        initial_state = generate_random_states(1, n_qubit=2, seed=seed)

        result_ref = circuit_ref(initial_state)
        result_mbqc = circuit_mbqc(initial_state)

        assert np.allclose(result_ref, result_mbqc)
