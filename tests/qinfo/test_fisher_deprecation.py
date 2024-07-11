# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Tests for the deprecation of the classical and quantum fisher information matrix in the pennylane.qinfo
"""
import pytest

import pennylane as qml
import pennylane.numpy as pnp
from pennylane.qinfo import classical_fisher, quantum_fisher


@pytest.mark.parametrize("fn", (classical_fisher, quantum_fisher))
def test_qinfo_fisher_fns_raises_warning(fn):
    n_wires = 3
    n_params = 3

    dev = qml.device("default.qubit", wires=n_wires, shots=10000)

    @qml.qnode(dev)
    def circ(params):
        for i in range(n_wires):
            qml.Hadamard(wires=i)

        for x in params:
            for j in range(n_wires):
                qml.RX(x, wires=j)
                qml.RY(x, wires=j)
                qml.RZ(x, wires=j)

        return qml.probs(wires=range(n_wires))

    params = pnp.zeros(n_params, requires_grad=True)

    with pytest.raises(qml.PennyLaneDeprecationWarning, match=f"{fn.__name__} is being migrated"):
        fn(circ)(params)
