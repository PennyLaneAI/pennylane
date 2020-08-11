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
"""Unit tests for the measure module"""
import pytest
import numpy as np

import pennylane as qml
from pennylane.qnodes import QuantumFunctionError


# Beta imports
from pennylane.beta.queuing.operation import BetaTensor
from pennylane.beta.queuing.measure import (
    expval,
    var,
    sample,
    probs,
    Expectation,
    Sample,
    Variance,
    Probability,
)


@pytest.mark.parametrize(
    "stat_func,return_type", [(expval, Expectation), (var, Variance), (sample, Sample)]
)
class TestBetaStatistics:
    """Tests for annotating the return types of the statistics functions"""

    @pytest.mark.parametrize(
        "op", [qml.PauliX, qml.PauliY, qml.PauliZ, qml.Hadamard, qml.Identity],
    )
    def test_annotating_obs_return_type(self, stat_func, return_type, op):
        """Test that the return_type related info is updated for an expval call"""
        with qml._queuing.AnnotatedQueue() as q:
            A = op(0)
            stat_func(A)

        assert q.queue == [A, return_type]
        assert q.get_info(A) == {"return_type": return_type}

    def test_annotating_tensor_hermitian(self, stat_func, return_type):
        """Test that the return_type related info is updated for an expval call
        when called for an Hermitian observable"""

        mx = np.array([[1, 0], [0, 1]])

        with qml._queuing.AnnotatedQueue() as q:
            Herm = qml.Hermitian(mx, wires=[1])
            stat_func(Herm)

        assert q.queue == [Herm, return_type]
        assert q.get_info(Herm) == {"return_type": return_type}

    @pytest.mark.parametrize(
        "op1,op2",
        [
            (qml.PauliY, qml.PauliX),
            (qml.Hadamard, qml.Hadamard),
            (qml.PauliY, qml.Identity),
            (qml.Identity, qml.Identity),
        ],
    )
    def test_annotating_tensor_return_type(self, op1, op2, stat_func, return_type):
        """Test that the return_type related info is updated for an expval call
        when called for an Tensor observable"""
        with qml._queuing.AnnotatedQueue() as q:
            A = op1(0)
            B = op2(1)
            tensor_op = BetaTensor(A, B)
            stat_func(tensor_op)

        assert q.queue == [A, B, tensor_op, return_type]
        assert q.get_info(A) == {"owner": tensor_op}
        assert q.get_info(B) == {"owner": tensor_op}
        assert q.get_info(tensor_op) == {"owns": (A, B), "return_type": return_type}

@pytest.mark.parametrize(
    "stat_func", [expval, var, sample]
)
class TestBetaStatisticsError:
    """Tests for errors arising for the beta statistics functions"""

    def test_not_an_observable(self, stat_func):
        """Test that a QuantumFunctionError is raised if the provided
        argument is not an observable"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.52, wires=0)
            return stat_func(qml.CNOT(wires=[0, 1]))

        with pytest.raises(QuantumFunctionError, match="CNOT is not an observable"):
            res = circuit()


class TestBetaProbs:
    """Tests for annotating the return types of the probs function"""

    @pytest.mark.parametrize("wires", [[0], [0, 1], [1, 0, 2]])
    def test_annotating_probs(self, wires):

        with qml._queuing.AnnotatedQueue() as q:
            probs(wires)

        assert len(q.queue) == 2
        assert isinstance(q.queue[0], qml.Identity)
        assert q.queue[1] == Probability
        assert q.get_info(q.queue[0]) == {"return_type": Probability, "owner":Probability}
