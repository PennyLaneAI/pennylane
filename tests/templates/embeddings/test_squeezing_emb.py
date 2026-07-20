# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Tests for the SqueezingEmbedding template.
"""

# pylint: disable=protected-access
import pytest

import pennylane as qp


@pytest.mark.jax
def test_standard_validity():
    """Check the operation using the assert_valid function."""
    feature_vector = [1.0, 2.0, 3.0]
    op = qp.SqueezingEmbedding(features=feature_vector, wires=range(3), method="phase", c=0.5)
    # Skip differentiation and capture because it's a CV op.
    qp.ops.functions.assert_valid(op, skip_differentiation=True, skip_capture=True)


def test_flatten_unflatten_methods():
    """Test the _flatten and _unflatten methods."""
    feature_vector = [1, 2, 3]
    op = qp.SqueezingEmbedding(features=feature_vector, wires=range(3), method="phase", c=0.5)
    data, metadata = op._flatten()
    assert op.data == data

    # make sure metadata hashable
    assert hash(metadata)

    new_op = type(op)._unflatten(*op._flatten())
    qp.assert_equal(new_op, op)
    assert new_op is not op
    assert new_op._name == "SqueezingEmbedding"  # make sure initialized


@pytest.mark.parametrize("features", [[1, 2, 3], [-1, 1, -1]])
def test_expansion(features):
    """Checks the queue for the default settings."""

    op = qp.SqueezingEmbedding(features=features, wires=range(3))
    tape = qp.tape.QuantumScript(op.decomposition())

    assert len(tape.operations) == len(features)
    for idx, gate in enumerate(tape.operations):
        assert gate.name == "Squeezing"
        assert gate.parameters[0] == features[idx]
