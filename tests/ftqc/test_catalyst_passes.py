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
from pennylane.ftqc.catalyst_pass_aliases import (
    commute_ppr,
    merge_ppr_ppm,
    ppm_to_mbqc,
    reduce_t_depth,
    to_ppr,
)

pytestmark = pytest.mark.external


@pytest.mark.jax
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
    prim = plxpr.eqns[0].primitive
    assert prim.name == pass_fn.__name__ + "_transform"


@pytest.mark.catalyst
@pytest.mark.usefixtures("enable_disable_plxpr")
@pytest.mark.parametrize(
    "pass_fn, pass_name",
    [
        (to_ppr, "to-ppr"),
        (commute_ppr, "commute-ppr"),
        (merge_ppr_ppm, "merge-ppr-ppm"),
        (ppm_to_mbqc, "ppm-to-mbqc"),
        (reduce_t_depth, "reduce-t-depth"),
    ],
)
def test_converstion_to_mlir(pass_fn, pass_name):
    """Test that we can generate MLIR from the captured circuit and that the generated MLIR
    includes the pass name we are mapping to"""

    @qml.qjit(target="mlir")
    @pass_fn
    @qml.qnode(qml.device("lightning.qubit", wires=3), shots=1000)
    def circ():
        qml.H(0)
        qml.S(0)
        qml.T(1)
        qml.CNOT([0, 1])
        return qml.sample()

    assert pass_name in circ.mlir


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
