# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Unit tests for :mod:`pennylane.operation`.
"""

import pytest

import pennylane as qml
import numpy as np

# The BetaTensor class
from pennylane.beta.queuing.operation import BetaTensor

# pylint: disable=no-self-use, no-member, protected-access, pointless-statement


class TestBetaTensor:
    """Unit tests for the BetaTensor class"""

    def test_simple_queue(self):
        """Test that the BetaTensor class works fine using the Queue class."""
        with qml._queuing.Queue() as q:
            A = qml.PauliZ(0)
            B = qml.PauliZ(1)
            tensor_op = BetaTensor(A, B)

        assert q.queue == [A, B, tensor_op]

    @pytest.mark.parametrize(
        "op1,op2",
        [
            (qml.PauliY, qml.PauliX),
            (qml.Hadamard, qml.Hadamard),
            (qml.PauliY, qml.Identity),
            (qml.Identity, qml.Identity),
        ],
    )
    def test_annotating_tensor_obs(self, op1, op2):
        """Test that the ownership related info are updated whenever a BetaTensor
        object is queued."""

        with qml._queuing.AnnotatedQueue() as q:
            A = op1(0)
            B = op2(1)
            tensor_op = BetaTensor(A, B)

        assert q.queue == [A, B, tensor_op]
        assert q.get_info(A) == {"owner": tensor_op}
        assert q.get_info(B) == {"owner": tensor_op}
        assert q.get_info(tensor_op) == {"owns": (A, B)}

    def test_annotating_tensor_obs_hermitian(self):
        """Test that the ownership related info are updated whenever a BetaTensor
        object is queued which contains a Hermitian observable."""

        mx = np.array([[1, 0], [0, 1]])

        with qml._queuing.AnnotatedQueue() as q:
            A = qml.PauliZ(0)
            B = qml.Hermitian(mx, wires=[1])
            tensor_op = BetaTensor(A, B)

        assert q.queue == [A, B, tensor_op]
        assert q.get_info(A) == {"owner": tensor_op}
        assert q.get_info(B) == {"owner": tensor_op}
        assert q.get_info(tensor_op) == {"owns": (A, B)}

    def test_obs_not_in_queue_error_annotated(self):
        """Test that creating a BetaTensor instance with an observable that was not
        queued before raises an error when using AnnotatedQueue."""

        A = qml.PauliZ(0)
        with pytest.raises(ValueError, match="not in the queue"):
            with qml._queuing.AnnotatedQueue() as q:
                B = qml.PauliY(0)
                tensor_op = BetaTensor(A, B)
