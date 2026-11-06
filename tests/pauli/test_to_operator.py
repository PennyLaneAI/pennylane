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
"""Tests for constructing operators from Pauli representations."""

import pytest

import pennylane as qp
from pennylane.pauli import PauliSentence, PauliWord, to_operator


@pytest.mark.parametrize("value", [1, {}, qp.X(0)])
def test_invalid_input(value):
    """Only Pauli data types are accepted."""
    with pytest.raises(TypeError, match="Expected a PauliWord or PauliSentence"):
        to_operator(value)


def test_batched_coefficients(interface):
    """Conversion retains coefficient tensors and their broadcast dimension."""
    coefficients = qp.math.array([2.0, 3.0], like=interface.replace("-jit", ""))
    word = PauliWord({0: "X"})
    sentence = PauliSentence({word: coefficients})

    op = to_operator(sentence)

    assert op.scalar is coefficients
    assert op.batch_size == 2
    assert qp.math.allclose(qp.matrix(op), [[[0, 2], [2, 0]], [[0, 3], [3, 0]]])


@pytest.mark.jax
def test_capture_after_cache_warmup():
    """Each trace constructs its own Pauli factors after an eager conversion."""
    import jax

    word = PauliWord({0: "X"})
    to_operator(word)

    for _ in range(2):
        with qp.capture.toggle_ctx(True), jax.checking_leaks():
            closed = jax.make_jaxpr(lambda: to_operator(word))()

        assert len(closed.jaxpr.eqns) == 1
        qp.testing.assert_eqn_matches_op(closed.jaxpr.eqns[0], qp.X)
