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

"""Unit tests for the decomposition transforms in the FTQC module"""

import pytest

import pennylane as qml
from pennylane.capture import make_plxpr
from pennylane.ftqc.decomposition import (
    decompose_clifford_ppr,
    decompose_non_clifford_ppr,
    ppr_to_mbqc,
)

pytest.importorskip("catalyst")
pytestmark = pytest.mark.external


@pytest.mark.catalyst
@pytest.mark.parametrize(
    "pass_fn",
    [
        ppr_to_mbqc,
        decompose_clifford_ppr,
        decompose_non_clifford_ppr,
    ],
)
def test_converstion_to_mlir(pass_fn):
    """Test that we can generate MLIR from the captured circuit and that the generated MLIR
    includes the pass name we are mapping to"""

    @qml.qjit(target="mlir", capture=True)
    @pass_fn
    @qml.qnode(qml.device("lightning.qubit", wires=3), shots=1000)
    def circ():
        qml.H(0)
        qml.S(0)
        qml.T(1)
        qml.CNOT([0, 1])
        return qml.sample()

    assert pass_fn.pass_name in circ.mlir


@pytest.mark.catalyst
@pytest.mark.parametrize(
    "pass_fn",
    [
        ppr_to_mbqc,
        decompose_clifford_ppr,
        decompose_non_clifford_ppr,
    ],
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
    prim = plxpr.eqns[0].primitive
    assert prim.name == "transform"
    assert plxpr.eqns[0].params["transform"] == pass_fn


@pytest.mark.catalyst
@pytest.mark.parametrize(
    "pass_fn",
    [
        ppr_to_mbqc,
        decompose_clifford_ppr,
        decompose_non_clifford_ppr,
    ],
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
