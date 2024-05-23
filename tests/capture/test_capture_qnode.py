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
Tests for capturing a qnode into jaxpr.
"""
# pylint: disable=protected-access
import pytest

import pennylane as qml
from pennylane.capture.capture_qnode import _get_qnode_prim

qnode_prim = _get_qnode_prim()

pytestmark = pytest.mark.jax

jax = pytest.importorskip("jax")


@pytest.fixture(autouse=True)
def enable_disable_plxpr():
    qml.capture.enable()
    yield
    qml.capture.disable()


@pytest.mark.parametrize("x64_mode", (True, False))
def test_simple_qnode(x64_mode):
    """Test capturing a qnode for a simple use."""

    initial_mode = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", x64_mode)

    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.Z(0))

    jaxpr = jax.make_jaxpr(circuit)(0.5)

    eqn0 = jaxpr.eqns[0]

    assert jaxpr.in_avals == [
        jax.core.ShapedArray(
            (), jax.numpy.float64 if x64_mode else jax.numpy.float32, weak_type=True
        )
    ]

    assert eqn0.primitive == qnode_prim
    assert eqn0.invars[0].aval == jaxpr.in_avals[0]
    assert jaxpr.out_avals[0] == jax.core.ShapedArray(
        (), jax.numpy.float64 if x64_mode else jax.numpy.float32
    )

    assert eqn0.params["device"] == dev
    assert eqn0.params["shots"] == qml.measurements.Shots(None)
    expected_kwargs = {"diff_method": "best"}
    expected_kwargs.update(circuit.execute_kwargs)
    assert eqn0.params["qnode_kwargs"] == expected_kwargs

    qfunc_jaxpr = eqn0.params["qfunc_jaxpr"]
    assert qfunc_jaxpr.eqns[0].primitive == qml.RX._primitive
    assert qfunc_jaxpr.eqns[1].primitive == qml.Z._primitive
    assert qfunc_jaxpr.eqns[2].primitive == qml.measurements.ExpectationMP._obs_primitive

    assert len(eqn0.outvars) == 1
    assert eqn0.outvars[0].aval == jax.core.ShapedArray(
        (), jax.numpy.float64 if x64_mode else jax.numpy.float32
    )

    output = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.5)
    assert qml.math.allclose(output[0], jax.numpy.cos(0.5))

    jax.config.update("jax_enable_x64", initial_mode)
