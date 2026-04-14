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

"""Tests for the decomposition rules for MultiX"""

# pylint: disable=no-value-for-parameter, import-outside-toplevel
import numpy as np
import pytest

import pennylane as qp
from pennylane.labs.transforms.multix import MultiX
from pennylane.ops.functions.assert_valid import _check_decomposition_new
from pennylane.transforms.decompose import DecomposeInterpreter


class TestMultiXDecomp:
    """Test the base MultiX decomposition."""

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


class TestControlledMultiXNoWorkWires:
    """Test the controlled MultiX decomposition without work wires."""

    @pytest.mark.usefixtures("enable_graph_decomposition")
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

    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.parametrize("capture", [True, False])
    def test_matrix_correctness_no_work_wires_capture(self, capture):
        """Test that the controlled MultiX without work wires produces the correct unitary."""
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

    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_base_decomp_capture(self, n):
        """Test that MultiX decomposes to X gates via capture + graph."""
        import jax

        wires = list(range(n))

        def f():
            MultiX(wires)

        decomposed_f = DecomposeInterpreter(gate_set={"X"})(f)
        _ = jax.make_jaxpr(decomposed_f)()
        # If this succeeds without error, the decomposition is capture-compatible
