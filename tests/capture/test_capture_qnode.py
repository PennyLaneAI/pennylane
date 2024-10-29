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
from dataclasses import asdict
from functools import partial

# pylint: disable=protected-access
import pytest

import pennylane as qml

pytestmark = pytest.mark.jax

jax = pytest.importorskip("jax")

# must be below jax importorskip
from pennylane.capture.primitives import qnode_prim  # pylint: disable=wrong-import-position


@pytest.fixture(autouse=True)
def enable_disable_plxpr():
    qml.capture.enable()
    yield
    qml.capture.disable()


def test_error_if_shot_vector():
    """Test that a NotImplementedError is raised if a shot vector is provided."""

    dev = qml.device("default.qubit", wires=1, shots=(50, 50))

    @qml.qnode(dev)
    def circuit():
        return qml.sample()

    with pytest.raises(NotImplementedError, match="shot vectors are not yet supported"):
        jax.make_jaxpr(circuit)()

    with pytest.raises(NotImplementedError, match="shot vectors are not yet supported"):
        circuit()

    jax.make_jaxpr(partial(circuit, shots=50))()  # should run fine
    res = circuit(shots=50)
    assert qml.math.allclose(res, jax.numpy.zeros((50,)))


def test_error_if_overridden_shot_vector():
    """Test that a NotImplementedError is raised if a shot vector is provided on call."""

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit():
        return qml.sample()

    with pytest.raises(NotImplementedError, match="shot vectors are not yet supported"):
        jax.make_jaxpr(partial(circuit, shots=(1, 1, 1)))()


def test_error_if_no_device_wires():
    """Test that a NotImplementedError is raised if the device does not provide wires."""

    dev = qml.device("default.qubit")

    @qml.qnode(dev)
    def circuit():
        return qml.sample()

    with pytest.raises(NotImplementedError, match="devices must specify wires"):
        jax.make_jaxpr(circuit)()

    with pytest.raises(NotImplementedError, match="devices must specify wires"):
        circuit()


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

    res = circuit(0.5)
    assert qml.math.allclose(res, jax.numpy.cos(0.5))

    jaxpr = jax.make_jaxpr(circuit)(0.5)

    assert len(jaxpr.eqns) == 1
    eqn0 = jaxpr.eqns[0]

    fdtype = jax.numpy.float64 if x64_mode else jax.numpy.float32

    assert jaxpr.in_avals == [jax.core.ShapedArray((), fdtype, weak_type=True)]

    assert eqn0.primitive == qnode_prim
    assert eqn0.invars[0].aval == jaxpr.in_avals[0]
    assert jaxpr.out_avals[0] == jax.core.ShapedArray((), fdtype)

    assert eqn0.params["device"] == dev
    assert eqn0.params["qnode"] == circuit
    assert eqn0.params["shots"] == qml.measurements.Shots(None)
    expected_kwargs = {"diff_method": "best"}
    expected_kwargs.update(circuit.execute_kwargs)
    expected_kwargs.update(asdict(expected_kwargs.pop("mcm_config")))
    assert eqn0.params["qnode_kwargs"] == expected_kwargs

    qfunc_jaxpr = eqn0.params["qfunc_jaxpr"]
    assert len(qfunc_jaxpr.eqns) == 3
    assert qfunc_jaxpr.eqns[0].primitive == qml.RX._primitive
    assert qfunc_jaxpr.eqns[1].primitive == qml.Z._primitive
    assert qfunc_jaxpr.eqns[2].primitive == qml.measurements.ExpectationMP._obs_primitive

    assert len(eqn0.outvars) == 1
    assert eqn0.outvars[0].aval == jax.core.ShapedArray((), fdtype)

    output = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.5)
    assert qml.math.allclose(output[0], jax.numpy.cos(0.5))

    jax.config.update("jax_enable_x64", initial_mode)


@pytest.mark.parametrize("x64_mode", (True, False))
def test_overriding_shots(x64_mode):
    """Test that the number of shots can be overridden on call."""
    initial_mode = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", x64_mode)

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit():
        return qml.sample()

    jaxpr = jax.make_jaxpr(partial(circuit, shots=50))()
    assert len(jaxpr.eqns) == 1
    eqn0 = jaxpr.eqns[0]

    assert eqn0.primitive == qnode_prim
    assert eqn0.params["device"] == dev
    assert eqn0.params["shots"] == qml.measurements.Shots(50)
    assert (
        eqn0.params["qfunc_jaxpr"].eqns[0].primitive == qml.measurements.SampleMP._wires_primitive
    )

    assert eqn0.outvars[0].aval == jax.core.ShapedArray(
        (50,), jax.numpy.int64 if x64_mode else jax.numpy.int32
    )

    res = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)
    assert qml.math.allclose(res, jax.numpy.zeros((50,)))

    jax.config.update("jax_enable_x64", initial_mode)


def test_providing_keyword_argument():
    """Test that keyword arguments can be provided to the qnode."""

    @qml.qnode(qml.device("default.qubit", wires=1))
    def circuit(*, n_iterations=0):
        for _ in range(n_iterations):
            qml.X(0)
        return qml.probs()

    jaxpr = jax.make_jaxpr(partial(circuit, n_iterations=3))()

    assert jaxpr.eqns[0].primitive == qnode_prim

    qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
    for i in range(3):
        assert qfunc_jaxpr.eqns[i].primitive == qml.PauliX._primitive
    assert len(qfunc_jaxpr.eqns) == 4

    res = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)
    assert qml.math.allclose(res, jax.numpy.array([0, 1]))

    res2 = circuit(n_iterations=4)
    assert qml.math.allclose(res2, jax.numpy.array([1, 0]))


@pytest.mark.parametrize("x64_mode", (True, False))
def test_multiple_measurements(x64_mode):
    """Test that the qnode can return multiple measurements."""
    initial_mode = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", x64_mode)

    @qml.qnode(qml.device("default.qubit", wires=3, shots=50))
    def circuit():
        return qml.sample(), qml.probs(wires=(0, 1)), qml.expval(qml.Z(0))

    jaxpr = jax.make_jaxpr(circuit)()

    qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]

    assert qfunc_jaxpr.eqns[0].primitive == qml.measurements.SampleMP._wires_primitive
    assert qfunc_jaxpr.eqns[1].primitive == qml.measurements.ProbabilityMP._wires_primitive
    assert qfunc_jaxpr.eqns[2].primitive == qml.Z._primitive
    assert qfunc_jaxpr.eqns[3].primitive == qml.measurements.ExpectationMP._obs_primitive

    assert jaxpr.out_avals[0] == jax.core.ShapedArray(
        (50, 3), jax.numpy.int64 if x64_mode else jax.numpy.int32
    )
    assert jaxpr.out_avals[1] == jax.core.ShapedArray(
        (4,), jax.numpy.float64 if x64_mode else jax.numpy.float32
    )
    assert jaxpr.out_avals[2] == jax.core.ShapedArray(
        (), jax.numpy.float64 if x64_mode else jax.numpy.float32
    )

    res1, res2, res3 = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)
    assert qml.math.allclose(res1, jax.numpy.zeros((50, 3)))
    assert qml.math.allclose(res2, jax.numpy.array([1, 0, 0, 0]))
    assert qml.math.allclose(res3, 1.0)

    res1, res2, res3 = circuit()
    assert qml.math.allclose(res1, jax.numpy.zeros((50, 3)))
    assert qml.math.allclose(res2, jax.numpy.array([1, 0, 0, 0]))
    assert qml.math.allclose(res3, 1.0)

    jax.config.update("jax_enable_x64", initial_mode)


@pytest.mark.parametrize("x64_mode", (True, False))
def test_complex_return_types(x64_mode):
    """Test returning measurements with complex values."""

    initial_mode = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", x64_mode)

    @qml.qnode(qml.device("default.qubit", wires=3))
    def circuit():
        return qml.state(), qml.density_matrix(wires=(0, 1))

    jaxpr = jax.make_jaxpr(circuit)()

    qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]

    assert qfunc_jaxpr.eqns[0].primitive == qml.measurements.StateMP._wires_primitive
    assert qfunc_jaxpr.eqns[1].primitive == qml.measurements.DensityMatrixMP._wires_primitive

    assert jaxpr.out_avals[0] == jax.core.ShapedArray(
        (8,), jax.numpy.complex128 if x64_mode else jax.numpy.complex64
    )
    assert jaxpr.out_avals[1] == jax.core.ShapedArray(
        (4, 4), jax.numpy.complex128 if x64_mode else jax.numpy.complex64
    )

    jax.config.update("jax_enable_x64", initial_mode)


def test_capture_qnode_kwargs():
    """Test that qnode kwargs are captured as parameters."""

    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(
        dev,
        diff_method="parameter-shift",
        grad_on_execution=False,
        cache=True,
        cachesize=10,
        max_diff=2,
    )
    def circuit():
        return qml.expval(qml.Z(0))

    jaxpr = jax.make_jaxpr(circuit)()

    assert jaxpr.eqns[0].primitive == qnode_prim
    expected = {
        "diff_method": "parameter-shift",
        "grad_on_execution": False,
        "cache": True,
        "cachesize": 10,
        "max_diff": 2,
        "device_vjp": False,
        "mcm_method": None,
        "postselect_mode": None,
    }
    assert jaxpr.eqns[0].params["qnode_kwargs"] == expected


def test_qnode_closure_variables():
    """Test that qnode can capture closure variables and consts."""

    a = jax.numpy.array(2.0)

    @qml.qnode(qml.device("default.qubit", wires=2))
    def circuit(w):
        qml.RX(a, w)
        return qml.expval(qml.Z(0))

    jaxpr = jax.make_jaxpr(circuit)(1)
    assert len(jaxpr.eqns[0].invars) == 2  # one closure variable, one arg
    assert jaxpr.eqns[0].params["n_consts"] == 1

    out = jax.core.eval_jaxpr(jaxpr.jaxpr, [jax.numpy.array(0.5)], 0)
    assert qml.math.allclose(out, jax.numpy.cos(0.5))


def test_qnode_pytree_input():
    """Test that we can capture and execute a qnode with a pytree input."""

    @qml.qnode(qml.device("default.qubit", wires=2))
    def circuit(x):
        qml.RX(x["val"], wires=x["wires"])
        return qml.expval(qml.Z(wires=x["wires"]))

    x = {"val": 0.5, "wires": 0}
    res = circuit(x)
    assert qml.math.allclose(res, jax.numpy.cos(0.5))

    jaxpr = jax.make_jaxpr(circuit)(x)
    assert len(jaxpr.eqns[0].invars) == 2


def test_qnode_pytree_output():
    """Test that we can capture and execute a qnode with a pytree output."""

    @qml.qnode(qml.device("default.qubit", wires=2))
    def circuit(x):
        qml.RX(x, 0)
        return {"a": qml.expval(qml.Z(0)), "b": qml.expval(qml.Y(0))}

    out = circuit(1.2)
    assert qml.math.allclose(out["a"], jax.numpy.cos(1.2))
    assert qml.math.allclose(out["b"], -jax.numpy.sin(1.2))
    assert list(out.keys()) == ["a", "b"]


def test_qnode_jvp():
    """Test that JAX can compute the JVP of the QNode primitive via a registered JVP rule."""

    @qml.qnode(qml.device("default.qubit", wires=1))
    def circuit(x):
        qml.RX(x, 0)
        return qml.expval(qml.Z(0))

    x = 0.9
    xt = -0.6
    jvp = jax.jvp(circuit, (x,), (xt,))
    assert qml.math.allclose(jvp, (qml.math.cos(x), -qml.math.sin(x) * xt))
