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

"""Tests for MultiX"""

# pylint: disable=no-value-for-parameter, import-outside-toplevel
import numpy as np
import pytest

import pennylane as qp
from pennylane.labs.transforms.multix import MultiX
from pennylane.ops.functions.assert_valid import _check_decomposition_new, assert_valid


class TestMultiX:
    """Test the base MultiX decomposition."""

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_assert_valid(self, n):
        """Standard validity test of operator"""
        op = MultiX(wires=range(n))
        assert_valid(op)

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_valid_decomp(self, n):
        """Test that the resource function matches the actual decomposition."""
        op = MultiX(wires=range(n))
        _check_decomposition_new(op)

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_matrix_correctness(self, n):
        """Test that MultiX decomposes to the correct unitary."""
        wires = range(n)

        @qp.transforms.decompose(gate_set={"X"})
        @qp.qnode(qp.device("default.qubit"))
        def circuit():
            MultiX(wires)
            return qp.state()

        state = circuit()
        # MultiX applies X to every wire, so |00...0> -> |11...1>
        expected = np.zeros(2**n)
        expected[2**n - 1] = 1.0
        assert np.allclose(state, expected)


@pytest.mark.usefixtures("enable_graph_decomposition")
class TestControlledMultiX:
    """Test the controlled MultiX decompositions"""

    @pytest.mark.parametrize("n", [2, 3])
    def test_matrix_correctness_no_work_wires(self, n):
        """Test that the controlled MultiX without work wires produces the correct unitary."""
        wires = range(n)
        control = [f"c{i}" for i in range(n)]

        in_state = np.random.rand(2**n)
        in_state /= np.linalg.norm(in_state)

        @qp.transforms.decompose(
            gate_set={
                "StatePrep",
                "Toffoli",
                "MultiControlledX",
            },  # Toffoli for n=2, MultiControlledX for n=3
            num_work_wires=0,
        )
        @qp.qnode(qp.device("default.qubit"))
        def decomposed(in_state):
            qp.StatePrep(in_state, wires=wires)
            qp.ctrl(MultiX(wires), control=control)
            return qp.state()

        @qp.qnode(qp.device("default.qubit"))
        def reference(in_state):
            qp.StatePrep(in_state, wires=wires)
            qp.ctrl(MultiX(wires), control=control)
            return qp.state()

        assert np.allclose(decomposed(in_state), reference(in_state))

    @pytest.mark.catalyst
    @pytest.mark.parametrize("capture", [True, False])
    def test_correctness_without_work_wires_qjit(self, capture):
        """Test that the controlled MultiX without work wires produces the correct unitary."""
        pytest.importorskip("catalyst")
        n = 2
        wires = list(range(n))
        control = list(range(n, 2 * n))

        all_wires = wires + control

        in_state = np.random.rand(2**n)
        in_state /= np.linalg.norm(in_state)

        @qp.qjit(capture=capture)
        @qp.transforms.decompose(
            gate_set={
                "StatePrep",
                "Toffoli",
            },
            num_work_wires=0,
        )
        @qp.qnode(qp.device("lightning.qubit", wires=all_wires))
        def decomposed(in_state):
            qp.StatePrep(in_state, wires=wires)
            qp.ctrl(MultiX(wires), control=control)
            return qp.state()

        @qp.qnode(qp.device("lightning.qubit", wires=all_wires))
        def reference(in_state):
            qp.StatePrep(in_state, wires=wires)
            qp.ctrl(MultiX(wires), control=control)
            return qp.state()

        assert np.allclose(decomposed(in_state), reference(in_state))

    @pytest.mark.catalyst
    @pytest.mark.parametrize("n", [1, 5])
    def test_correctness_with_work_wires_qjit(self, n):
        """Test that the controlled MultiX with work wires produces the correct unitary."""
        pytest.importorskip("catalyst")

        control = list(range(n))
        wires = list(range(n, 2 * n))

        main_wires = control + wires

        work_wires = list(range(2 * n, 3 * n - 1))

        all_wires = main_wires + work_wires

        in_state = np.random.rand(2 ** len(main_wires))
        in_state /= np.linalg.norm(in_state)

        dev = qp.device("lightning.qubit", wires=all_wires)

        @qp.qjit(capture=False)
        @qp.transforms.decompose(
            gate_set={
                "StatePrep",
                "TemporaryAND",
                "Adjoint(TemporaryAND)",
                "CNOT",
            }
        )
        @qp.qnode(dev)
        def decomposed(in_state):
            qp.StatePrep(in_state, wires=main_wires)
            qp.ctrl(MultiX(wires), control=control, work_wires=work_wires)
            return qp.state()

        @qp.qnode(dev)
        def reference(in_state):
            qp.StatePrep(in_state, wires=main_wires)
            qp.ctrl(MultiX(wires), control=control)
            return qp.state()

        assert np.allclose(decomposed(in_state), reference(in_state))
