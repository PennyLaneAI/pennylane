# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
Integration tests for the qnode for all interfaces.
"""
import pytest

import pennylane as qp


@pytest.mark.parametrize(
    "interface",
    (
        pytest.param("autograd", marks=pytest.mark.autograd),
        pytest.param("jax", marks=pytest.mark.jax),
        pytest.param("torch", marks=pytest.mark.torch),
    ),
)
class TestHadamardGradients:

    @pytest.mark.parametrize("diff_method", ["hadamard", "reversed-hadamard"])
    def test_nonstandard_device_wires(self, interface, diff_method):
        """Test that we can automatically determine a work wire with nonstandard wire labels."""

        dev = qp.device("default.qubit", wires=("a", "b"))

        @qp.qnode(dev, diff_method=diff_method)
        def circuit(x):
            qp.RX(x, "a")
            return qp.expval(qp.Z("a"))

        x = qp.math.asarray(0.5, requires_grad=True, like=interface)
        result = qp.math.grad(circuit)(x)
        assert qp.math.allclose(result, -qp.math.sin(x))

    @pytest.mark.parametrize(
        "diff_method, num_executions",
        [
            ("hadamard", 3),
            ("reversed-hadamard", 2),
            ("direct-hadamard", 5),
            ("reversed-direct-hadamard", 3),
        ],
    )
    def test_hamiltonian_generator(self, diff_method, interface, num_executions):
        """Check that we perform the expected number of executions when having a hamiltonian generator."""
        dev = qp.device("default.qubit")

        @qp.qnode(dev, diff_method=diff_method)
        def c(x):
            qp.evolve(qp.X(0) + qp.X(1), x)
            return qp.expval(qp.Z(0))

        x = qp.math.asarray(1.2, requires_grad=True, like=interface)

        with dev.tracker:
            result = qp.math.grad(c)(x)

        expected = -2 * qp.math.sin(2 * x)
        assert qp.math.allclose(result, expected)

        assert dev.tracker.totals["executions"] == num_executions
