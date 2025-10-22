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
"""Unit tests for wrappers for capturing catalyst passes"""

import pytest

import pennylane as qml
from pennylane.capture import make_plxpr
from pennylane.ftqc.catalyst_passes import (
    commute_ppr,
    merge_ppr_ppm,
    ppm_to_mbqc,
    reduce_t_depth,
    to_ppr,
)

pytest.importorskip("jax")


@pytest.mark.usefixtures("enable_disable_plxpr")
@pytest.mark.parametrize(
    "pass_fn", [to_ppr, commute_ppr, merge_ppr_ppm, ppm_to_mbqc, reduce_t_depth]
)
def test_pass_is_captured(pass_fn):

    @pass_fn
    @qml.qnode(qml.device("lightning.qubit", wires=3), shots=1000)
    def circ():
        qml.H(0)
        qml.S(0)
        qml.T(1)
        qml.CNOT([0, 1])
        return qml.sample()

    plxpr = make_plxpr(circ)()
    assert pass_fn.__name__ in str(plxpr)


@pytest.mark.parametrize(
    "pass_fn", [to_ppr, commute_ppr, merge_ppr_ppm, ppm_to_mbqc, reduce_t_depth]
)
def test_pass_without_qjit_raises_error(pass_fn):
    """Test that trying to apply the transform without QJIT raises an error"""

    @pass_fn
    @qml.qnode(qml.device("lightning.qubit", wires=3), shots=1000)
    def circ():
        qml.H(0)
        qml.S(0)
        qml.T(1)
        qml.CNOT([0, 1])
        return qml.sample()

    with pytest.raises(NotImplementedError, match="only implemented when using capture and QJIT"):
        circ()
