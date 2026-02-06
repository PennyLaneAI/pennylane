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
"""
Tests for the while_loop
"""

import pennylane as qp


def test_while_loop_python_fallback():
    """Test that qp.while_loop fallsback to
    Python without qjit"""

    def f(n, m):
        @qp.while_loop(lambda i, _: i < n)
        def outer(i, sm):
            @qp.while_loop(lambda j: j < m)
            def inner(j):
                return j + 1

            return i + 1, sm + inner(0)

        return outer(0, 0)[1]

    assert f(5, 6) == 30  # 5 * 6
    assert f(4, 7) == 28  # 4 * 7


def test_fallback_while_loop_qnode():
    """Test that qp.while_loop inside a qnode falls back to
    Python without qjit"""
    dev = qp.device("default.qubit")

    @qp.qnode(dev)
    def circuit(n):
        @qp.while_loop(lambda v: v[0] < v[1])
        def loop(v):
            qp.PauliX(wires=0)
            return v[0] + 1, v[1]

        loop((0, n))
        return qp.expval(qp.PauliZ(0))

    assert qp.math.allclose(circuit(1), -1.0)

    tape = qp.workflow.construct_tape(circuit)(1)
    expected = [qp.PauliX(0) for i in range(4)]
    _ = [qp.assert_equal(i, j) for i, j in zip(tape.operations, expected)]
