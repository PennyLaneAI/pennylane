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
from functools import partial
from itertools import product

# pylint: disable=protected-access
import pytest

import pennylane as qml
from pennylane.capture import CaptureError

pytestmark = [pytest.mark.jax, pytest.mark.usefixtures("enable_disable_plxpr")]

jax = pytest.importorskip("jax")
jnp = jax.numpy

# must be below jax importorskip
from pennylane.capture.primitives import qnode_prim  # pylint: disable=wrong-import-position


def get_qnode_output_eqns(jaxpr):
    """Extracts equations related to QNode outputs in the given JAX expression (jaxpr).

    Parameters:
        jaxpr: A JAX expression with equations, containing QNode-related operations.

    Returns:
        List of equations containing QNode outputs.
    """

    qnode_output_eqns = []

    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "qnode":
            qnode_output_eqns.append(eqn)

    return qnode_output_eqns


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
    with pytest.raises(NotImplementedError, match="Overriding shots is not yet supported"):
        res = circuit(shots=50)
        assert qml.math.allclose(res, jnp.zeros((50,)))


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


def test_simple_qnode():
    """Test capturing a qnode for a simple use."""

    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.Z(0))

    res = circuit(0.5)
    assert qml.math.allclose(res, jnp.cos(0.5))

    jaxpr = jax.make_jaxpr(circuit)(0.5)

    assert len(jaxpr.eqns) == 1
    eqn0 = jaxpr.eqns[0]

    fdtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

    assert jaxpr.in_avals == [jax.core.ShapedArray((), fdtype, weak_type=True)]

    assert eqn0.primitive == qnode_prim
    assert eqn0.invars[0].aval == jaxpr.in_avals[0]
    assert jaxpr.out_avals[0] == jax.core.ShapedArray((), fdtype)

    assert eqn0.params["device"] == dev
    assert eqn0.params["qnode"] == circuit
    assert eqn0.params["shots"] == qml.measurements.Shots(None)
    expected_config = qml.devices.ExecutionConfig(
        gradient_method="backprop",
        use_device_gradient=True,
        gradient_keyword_arguments={},
        use_device_jacobian_product=False,
        interface="jax",
        grad_on_execution=False,
        device_options={"max_workers": None, "rng": dev._rng, "prng_key": None},
    )
    assert eqn0.params["execution_config"] == expected_config

    qfunc_jaxpr = eqn0.params["qfunc_jaxpr"]
    assert len(qfunc_jaxpr.eqns) == 3
    assert qfunc_jaxpr.eqns[0].primitive == qml.RX._primitive
    assert qfunc_jaxpr.eqns[1].primitive == qml.Z._primitive
    assert qfunc_jaxpr.eqns[2].primitive == qml.measurements.ExpectationMP._obs_primitive

    assert len(eqn0.outvars) == 1
    assert eqn0.outvars[0].aval == jax.core.ShapedArray((), fdtype)

    output = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.5)
    assert qml.math.allclose(output[0], jnp.cos(0.5))


def test_overriding_shots():
    """Test that the number of shots can be overridden on call."""

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
        (50,), jnp.int64 if jax.config.jax_enable_x64 else jnp.int32
    )

    with pytest.raises(NotImplementedError, match="Overriding shots is not yet supported"):
        res = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)
        assert qml.math.allclose(res, jnp.zeros((50,)))


def test_providing_keyword_argument():
    """Test that keyword arguments can be provided to the qnode."""

    @qml.qnode(qml.device("default.qubit", wires=1), autograph=False)
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
    assert qml.math.allclose(res, jnp.array([0, 1]))

    res2 = circuit(n_iterations=4)
    assert qml.math.allclose(res2, jnp.array([1, 0]))


def test_multiple_measurements():
    """Test that the qnode can return multiple measurements."""

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
        (50, 3), jnp.int64 if jax.config.jax_enable_x64 else jnp.int32
    )
    assert jaxpr.out_avals[1] == jax.core.ShapedArray(
        (4,), jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    )
    assert jaxpr.out_avals[2] == jax.core.ShapedArray(
        (), jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    )

    res1, res2, res3 = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)
    assert qml.math.allclose(res1, jnp.zeros((50, 3)))
    assert qml.math.allclose(res2, jnp.array([1, 0, 0, 0]))
    assert qml.math.allclose(res3, 1.0)

    res1, res2, res3 = circuit()
    assert qml.math.allclose(res1, jnp.zeros((50, 3)))
    assert qml.math.allclose(res2, jnp.array([1, 0, 0, 0]))
    assert qml.math.allclose(res3, 1.0)


def test_complex_return_types():
    """Test returning measurements with complex values."""

    @qml.qnode(qml.device("default.qubit", wires=3))
    def circuit():
        return qml.state(), qml.density_matrix(wires=(0, 1))

    jaxpr = jax.make_jaxpr(circuit)()

    qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]

    assert qfunc_jaxpr.eqns[0].primitive == qml.measurements.StateMP._wires_primitive
    assert qfunc_jaxpr.eqns[1].primitive == qml.measurements.DensityMatrixMP._wires_primitive

    assert jaxpr.out_avals[0] == jax.core.ShapedArray(
        (8,), jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64
    )
    assert jaxpr.out_avals[1] == jax.core.ShapedArray(
        (4, 4), jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64
    )


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
    expected_config = qml.devices.ExecutionConfig(
        gradient_method="parameter-shift",
        use_device_gradient=False,
        grad_on_execution=False,
        derivative_order=2,
        use_device_jacobian_product=False,
        mcm_config=qml.devices.MCMConfig(mcm_method=None, postselect_mode=None),
        interface=qml.math.Interface.JAX,
        device_options={"max_workers": None, "rng": dev._rng, "prng_key": None},
    )
    assert jaxpr.eqns[0].params["execution_config"] == expected_config


def test_qnode_closure_variables():
    """Test that qnode can capture closure variables and consts."""

    a = jnp.array(2.0)

    @qml.qnode(qml.device("default.qubit", wires=2))
    def circuit(w):
        qml.RX(a, w)
        return qml.expval(qml.Z(0))

    jaxpr = jax.make_jaxpr(circuit)(1)
    assert len(jaxpr.eqns[0].invars) == 2  # one closure variable, one arg
    assert jaxpr.eqns[0].params["n_consts"] == 1

    out = jax.core.eval_jaxpr(jaxpr.jaxpr, [jnp.array(0.5)], 0)
    assert qml.math.allclose(out, jnp.cos(0.5))


def test_qnode_pytree_input():
    """Test that we can capture and execute a qnode with a pytree input."""

    @qml.qnode(qml.device("default.qubit", wires=2))
    def circuit(x):
        qml.RX(x["val"], wires=x["wires"])
        return qml.expval(qml.Z(wires=x["wires"]))

    x = {"val": 0.5, "wires": 0}
    res = circuit(x)
    assert qml.math.allclose(res, jnp.cos(0.5))

    jaxpr = jax.make_jaxpr(circuit)(x)
    assert len(jaxpr.eqns[0].invars) == 2


def test_qnode_pytree_output():
    """Test that we can capture and execute a qnode with a pytree output."""

    @qml.qnode(qml.device("default.qubit", wires=2))
    def circuit(x):
        qml.RX(x, 0)
        return {"a": qml.expval(qml.Z(0)), "b": qml.expval(qml.Y(0))}

    out = circuit(1.2)
    assert qml.math.allclose(out["a"], jnp.cos(1.2))
    assert qml.math.allclose(out["b"], -jnp.sin(1.2))
    assert list(out.keys()) == ["a", "b"]


class TestDifferentiation:

    def test_error_backprop_unsupported(self):
        """Test an error is raised with backprop if the device does not support it."""

        # pylint: disable=too-few-public-methods
        class DummyDev(qml.devices.Device):

            def execute(self, *_, **__):
                return 0

        with pytest.raises(qml.QuantumFunctionError, match="does not support backprop"):

            @qml.qnode(DummyDev(wires=2), diff_method="backprop")
            def _(x):
                qml.RX(x, 0)
                return qml.expval(qml.Z(0))

    def test_error_unsupported_diff_method(self):
        """Test an error is raised for a non-backprop diff method."""

        @qml.qnode(qml.device("default.qubit", wires=2), diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        with pytest.raises(
            NotImplementedError, match="diff_method parameter-shift not yet implemented."
        ):
            jax.grad(circuit)(0.5)

    @pytest.mark.parametrize("diff_method", ("best", "backprop"))
    def test_default_qubit_backprop(self, diff_method):
        """Test that JAX can compute the JVP of the QNode primitive via a registered JVP rule."""

        @qml.qnode(qml.device("default.qubit", wires=1), diff_method=diff_method)
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        x = 0.9
        xt = -0.6
        jvp = jax.jvp(circuit, (x,), (xt,))
        assert qml.math.allclose(jvp, (qml.math.cos(x), -qml.math.sin(x) * xt))

    def test_no_gradients_with_lightning(self):
        """Test that we get an error if we try and differentiate a lightning execution."""

        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        with pytest.raises(NotImplementedError, match=r"does not yet support PLXPR jvps."):
            jax.grad(circuit)(0.5)


def test_qnode_jit():
    """Test that executions on default qubit can be jitted."""

    @qml.qnode(qml.device("default.qubit", wires=1))
    def circuit(x):
        qml.RX(x, 0)
        return qml.expval(qml.Z(0))

    x = jnp.array(-0.5)
    res = jax.jit(circuit)(0.5)
    assert qml.math.allclose(res, jnp.cos(x))


# pylint: disable=unused-argument
def test_dynamic_shape_input(enable_disable_dynamic_shapes):
    """Test that the qnode can accept an input with a dynamic shape."""

    @qml.qnode(qml.device("default.qubit", wires=1))
    def circuit(x):
        qml.RX(jnp.sum(x), 0)
        return qml.expval(qml.Z(0))

    jaxpr = jax.make_jaxpr(circuit, abstracted_axes=("a",))(jnp.arange(4))

    [output] = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 3, jnp.arange(3))
    expected = jnp.cos(0 + 1 + 2)
    assert jnp.allclose(expected, output)


# pylint: disable=unused-argument
def test_dynamic_shape_matches_arg(enable_disable_dynamic_shapes):

    @qml.qnode(qml.device("default.qubit", wires=4))
    def circuit(i, x):
        qml.RX(jax.numpy.sum(x), i)
        return qml.expval(qml.Z(i))

    def w(i):
        return circuit(i, jnp.arange(i))

    jaxpr = jax.make_jaxpr(w)(2)
    [res] = qml.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 3)
    expected = jax.numpy.cos(0 + 1 + 2)
    assert qml.math.allclose(res, expected)


# pylint: disable=too-many-public-methods
class TestQNodeVmapIntegration:
    """Tests for integrating JAX vmap with the QNode primitive."""

    @pytest.mark.parametrize(
        "input, expected_shape",
        [
            (jnp.array([0.1], dtype=jnp.float32), (1,)),
            (jnp.array([0.1, 0.2], dtype=jnp.float32), (2,)),
            (jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32), (3,)),
        ],
    )
    def test_qnode_vmap(self, input, expected_shape):
        """Test that JAX can vmap over the QNode primitive via a registered batching rule."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(jax.vmap(circuit))(input)
        eqn0 = jaxpr.eqns[0]

        assert len(eqn0.outvars) == 1
        assert eqn0.outvars[0].aval.shape == expected_shape

        res = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, input)
        assert qml.math.allclose(res, jnp.cos(input))

    def test_qnode_vmap_x64_mode(self):
        """Test that JAX can vmap over the QNode primitive with x64 mode enabled/disabled."""

        dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

        @qml.qnode(qml.device("default.qubit", wires=1))
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        x = jnp.array([0.1, 0.2, 0.3], dtype=dtype)

        jaxpr = jax.make_jaxpr(jax.vmap(circuit))(x)
        eqn0 = jaxpr.eqns[0]

        assert len(eqn0.outvars) == 1
        assert eqn0.outvars[0].aval == jax.core.ShapedArray((3,), dtype)

        res = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)
        assert qml.math.allclose(res, jnp.cos(x))

    def test_vmap_mixed_arguments(self):
        """Test vmap with a mix of batched and non-batched arguments."""

        @qml.qnode(qml.device("default.qubit", wires=2))
        def circuit(arr1, scalar1, arr2, scalar2):
            qml.RX(arr1, 0)
            qml.RY(scalar1, 0)
            qml.RY(arr2, 1)
            qml.RZ(scalar2, 1)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        arr1 = jnp.array([0.1, 0.2, 0.3])
        arr2 = jnp.array([0.2, 0.4, 0.6])
        scalar1 = 1.0
        scalar2 = 2.0

        jaxpr = jax.make_jaxpr(jax.vmap(circuit, in_axes=(0, None, 0, None)))(
            arr1, scalar1, arr2, scalar2
        )

        assert len(jaxpr.out_avals) == 2
        assert jaxpr.out_avals[0].shape == (3,)
        assert jaxpr.out_avals[1].shape == (3,)

        res = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, arr1, scalar1, arr2, scalar2)
        assert qml.math.allclose(res, circuit(arr1, scalar1, arr2, scalar2))
        # compare with jax.vmap to cover all code paths
        assert qml.math.allclose(
            res, jax.vmap(circuit, in_axes=(0, None, 0, None))(arr1, scalar1, arr2, scalar2)
        )

    def test_vmap_multiple_measurements(self):
        """Test that JAX can vmap over the QNode primitive with multiple measurements."""

        @qml.qnode(qml.device("default.qubit", wires=4, shots=5))
        def circuit(x):
            qml.DoubleExcitation(x, wires=[0, 1, 2, 3])
            return qml.sample(), qml.probs(wires=(0, 1, 2)), qml.expval(qml.Z(0))

        x = jnp.array([1.0, 2.0])
        jaxpr = jax.make_jaxpr(jax.vmap(circuit))(x)

        res1_vmap, res2_vmap, res3_vmap = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)

        assert len(jaxpr.eqns[0].outvars) == 3
        assert jaxpr.out_avals[0].shape == (2, 5, 4)
        assert jaxpr.out_avals[1].shape == (2, 8)
        assert jaxpr.out_avals[2].shape == (2,)

        assert qml.math.allclose(res1_vmap, jnp.zeros((2, 5, 4)))
        assert qml.math.allclose(
            res2_vmap, jnp.array([[1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]])
        )
        assert qml.math.allclose(res3_vmap, jnp.array([1.0, 1.0]))

    def test_qnode_vmap_closure(self):
        """Test that JAX can vmap over the QNode primitive with closure variables."""

        const = jnp.array(2.0)

        @qml.qnode(qml.device("default.qubit", wires=2))
        def circuit(x):
            qml.RX(x, 0)
            qml.RY(const, 1)
            return qml.probs(wires=[0, 1])

        x = jnp.array([1.0, 2.0, 3.0])
        jaxpr = jax.make_jaxpr(jax.vmap(circuit))(x)
        eqn0 = jaxpr.eqns[0]

        assert len(eqn0.invars) == 2  # one closure variable, one (batched) arg
        assert eqn0.invars[0].aval.shape == ()
        assert eqn0.invars[1].aval.shape == (3,)

        assert len(eqn0.outvars) == 1
        assert eqn0.outvars[0].aval.shape == (3, 4)

        res = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)
        assert qml.math.allclose(res, circuit(x))

    def test_qnode_vmap_closure_warn(self):
        """Test that a warning is raised when trying to vmap over a batched non-scalar closure variable."""
        dev = qml.device("default.qubit", wires=2)

        const = jnp.array([2.0, 6.6])

        @qml.qnode(dev)
        def circuit(x):
            qml.RY(x, 0)
            qml.RX(const, wires=0)
            return qml.expval(qml.PauliZ(0))

        with pytest.warns(UserWarning, match="Constant argument at index 0 is not scalar. "):
            jax.make_jaxpr(jax.vmap(circuit))(jnp.array([0.1, 0.2]))

    def test_vmap_overriding_shots(self):
        """Test that the number of shots can be overridden on call with vmap."""

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        # pylint: disable=unused-argument
        def circuit(x):
            return qml.sample()

        x = jnp.array([1.0, 2.0, 3.0])

        jaxpr = jax.make_jaxpr(jax.vmap(partial(circuit, shots=50), in_axes=0))(x)
        with pytest.raises(NotImplementedError, match="Overriding shots is not yet supported"):
            res = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)

        assert len(jaxpr.eqns) == 1
        eqn0 = jaxpr.eqns[0]

        assert eqn0.primitive == qnode_prim
        assert eqn0.params["device"] == dev
        assert eqn0.params["shots"] == qml.measurements.Shots(50)
        assert (
            eqn0.params["qfunc_jaxpr"].eqns[0].primitive
            == qml.measurements.SampleMP._wires_primitive
        )

        assert eqn0.outvars[0].aval.shape == (3, 50)

        with pytest.raises(NotImplementedError, match="Overriding shots is not yet supported"):
            res = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)
            assert qml.math.allclose(res, jnp.zeros((3, 50)))

    def test_vmap_error_indexing(self):
        """Test that an IndexError is raised when indexing a batched parameter."""

        @qml.qnode(qml.device("default.qubit", wires=2))
        def circuit(vec, scalar):
            qml.RX(vec[0], 0)
            qml.RY(scalar, 1)
            return qml.expval(qml.Z(0))

        with pytest.raises(IndexError):
            jax.make_jaxpr(jax.vmap(circuit, in_axes=(0, None)))(jnp.array([1.0, 2.0, 3.0]), 5.0)

    def test_vmap_error_empty_array(self):
        """Test that an error is raised when passing an empty array to vmap."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.Z(0))

        with pytest.raises(ValueError, match="Empty tensors are not supported with jax.vmap."):
            jax.make_jaxpr(jax.vmap(circuit))(jnp.array([]))

    def test_warning_bypass_vmap(self):
        """Test that a warning is raised when bypassing vmap."""
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit(param_array, param_array_2):
            qml.RX(param_array, wires=2)
            qml.DoubleExcitation(param_array_2[0], wires=[0, 1, 2, 3])
            return qml.expval(qml.PauliZ(0))

        param_array = jnp.array([1.0, 1.2, 1.3])
        param_array_2 = jnp.array([2.0, 2.1, 2.2])

        with pytest.warns(UserWarning, match="Argument at index 1 has size"):
            jax.make_jaxpr(jax.vmap(circuit, in_axes=(0, None)))(param_array, param_array_2)

    def test_qnode_pytree_input_vmap(self):
        """Test that we can capture and execute a qnode with a pytree input and vmap."""

        @qml.qnode(qml.device("default.qubit", wires=2))
        def circuit(x):
            qml.RX(x["val"], wires=x["wires"])
            return qml.expval(qml.Z(wires=x["wires"]))

        x = {"val": jnp.array([0.1, 0.2]), "wires": 0}
        jaxpr = jax.make_jaxpr(jax.vmap(circuit, in_axes=({"val": 0, "wires": None},)))(x)

        assert len(jaxpr.eqns[0].invars) == 2

        assert len(jaxpr.eqns[0].outvars) == 1
        assert jaxpr.eqns[0].outvars[0].aval.shape == (2,)

        res = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x["val"], x["wires"])
        assert qml.math.allclose(res, jnp.cos(x["val"]))

    def test_qnode_deep_pytree_input_vmap(self):
        """Test vmap over qnodes with deep pytree inputs."""

        @qml.qnode(qml.device("default.qubit", wires=2))
        def circuit(x):
            qml.RX(x["data"]["val"], wires=x["data"]["wires"])
            return qml.expval(qml.Z(wires=x["data"]["wires"]))

        x = {"data": {"val": jnp.array([0.1, 0.2]), "wires": 0}}
        jaxpr = jax.make_jaxpr(jax.vmap(circuit, in_axes=({"data": {"val": 0, "wires": None}},)))(x)

        assert len(jaxpr.eqns[0].invars) == 2

        assert len(jaxpr.eqns[0].outvars) == 1
        assert jaxpr.eqns[0].outvars[0].aval.shape == (2,)

        res = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x["data"]["val"], x["data"]["wires"])
        assert qml.math.allclose(res, jnp.cos(x["data"]["val"]))

    def test_qnode_pytree_output_vmap(self):
        """Test that we can capture and execute a qnode with a pytree output and vmap."""

        @qml.qnode(qml.device("default.qubit", wires=2))
        def circuit(x):
            qml.RX(x, 0)
            return {"a": qml.expval(qml.Z(0)), "b": qml.expval(qml.Y(0))}

        x = jnp.array([1.2, 1.3])
        out = jax.vmap(circuit)(x)

        assert qml.math.allclose(out["a"], jnp.cos(x))
        assert qml.math.allclose(out["b"], -jnp.sin(x))
        assert list(out.keys()) == ["a", "b"]

    def test_simple_multidim_case(self):
        """Test vmap over a simple multidimensional case."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def circuit(x):
            qml.RX(jnp.pi * x[0], wires=0)
            qml.RY(x[1] ** 2, wires=0)
            qml.RX(x[1] * x[2], wires=0)
            return qml.expval(qml.PauliZ(0))

        def cost_fn(x):
            result = circuit(x)
            return jnp.cos(result) ** 2

        x = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        jaxpr = jax.make_jaxpr(jax.vmap(cost_fn))(x)

        assert len(jaxpr.eqns[0].outvars) == 1
        assert jaxpr.out_avals[0].shape == (2,)

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)
        assert len(result[0]) == 2
        assert jnp.allclose(result[0][0], cost_fn(x[0]))
        assert jnp.allclose(result[0][1], cost_fn(x[1]))

    def test_simple_multidim_case_2(self):
        """Test vmap over a simple multidimensional case with a scalar and constant argument."""

        # pylint: disable=import-outside-toplevel
        from scipy.stats import unitary_group

        const = jnp.array(2.0)

        @qml.qnode(qml.device("default.qubit", wires=4))
        def circuit(x, y, U):
            qml.QubitUnitary(U, wires=[0, 1, 2, 3])
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RX(x, wires=2)
            qml.RY(const, wires=3)
            return qml.expval(qml.Z(0) @ qml.X(1) @ qml.Z(2) @ qml.Z(3))

        x = jnp.array([0.4, 2.1, -1.3])
        y = 2.71
        U = jnp.stack([unitary_group.rvs(16) for _ in range(3)])

        jaxpr = jax.make_jaxpr(jax.vmap(circuit, in_axes=(0, None, 0)))(x, y, U)
        assert len(jaxpr.eqns[0].invars) == 4  # 3 args + 1 const
        assert len(jaxpr.eqns[0].outvars) == 1
        assert jaxpr.out_avals[0].shape == (3,)

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x, y, U)
        assert qml.math.allclose(result, circuit(x, y, U))

    def test_vmap_circuit_inside(self):
        """Test vmap of a hybrid workflow."""

        def workflow(x):
            @qml.qnode(qml.device("default.qubit", wires=1))
            def circuit(x):
                qml.RX(jnp.pi * x[0], wires=0)
                qml.RY(x[1] ** 2, wires=0)
                qml.RX(x[1] * x[2], wires=0)
                return qml.expval(qml.PauliZ(0))

            res1 = jax.vmap(circuit)(x)
            res2 = jax.vmap(circuit, in_axes=0)(x)
            res3 = jax.vmap(circuit, in_axes=(0,))(x)
            return res1, res2, res3

        x = jnp.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ]
        )

        jaxpr = jax.make_jaxpr(workflow)(x)

        qnode_output_eqns = get_qnode_output_eqns(jaxpr)
        assert len(qnode_output_eqns) == 3
        for eqn in qnode_output_eqns:
            assert len(eqn.outvars) == 1
            assert eqn.outvars[0].aval.shape == (3,)

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)
        expected = jnp.array([0.93005586, 0.00498127, -0.88789978])
        assert jnp.allclose(result[0], expected)
        assert jnp.allclose(result[1], expected)
        assert jnp.allclose(result[2], expected)

    def test_vmap_nonzero_axes(self):
        """Test vmap of a hybrid workflow with axes > 0."""

        def workflow(x):
            @qml.qnode(qml.device("default.qubit", wires=1))
            def circuit(x):
                qml.RX(jnp.pi * x[0], wires=0)
                qml.RY(x[1] ** 2, wires=0)
                qml.RX(x[1] * x[2], wires=0)
                return qml.expval(qml.PauliZ(0))

            res1 = jax.vmap(circuit, in_axes=1)(x)
            res2 = jax.vmap(circuit, in_axes=(1,))(x)
            return res1, res2

        x = jnp.array(
            [
                [0.1, 0.4],
                [0.2, 0.5],
                [0.3, 0.6],
            ]
        )

        jaxpr = jax.make_jaxpr(workflow)(x)

        qnode_output_eqns = get_qnode_output_eqns(jaxpr)
        assert len(qnode_output_eqns) == 2
        for eqn in qnode_output_eqns:
            assert len(eqn.outvars) == 1
            assert eqn.outvars[0].aval.shape == (2,)

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)
        expected = jnp.array([0.93005586, 0.00498127])
        assert jnp.allclose(result[0], expected)
        assert jnp.allclose(result[1], expected)

    def test_vmap_nonzero_axes_2(self):
        """Test vmap of a hybrid workflow with axes > 0."""

        def workflow(y, x):
            @qml.qnode(qml.device("default.qubit", wires=1))
            def circuit(y, x):
                qml.RX(jnp.pi * x[0] * y, wires=0)
                qml.RY(x[1] ** 2, wires=0)
                qml.RX(x[1] * x[2] * y, wires=0)
                return qml.expval(qml.PauliZ(0))

            res1 = jax.vmap(circuit, in_axes=(None, 1))(y[0], x)
            res2 = jax.vmap(circuit, in_axes=(0, 1))(y, x)
            return res1, res2

        x = jnp.array(
            [
                [0.1, 0.4],
                [0.2, 0.5],
                [0.3, 0.6],
            ]
        )
        y = jnp.array([1, 2])

        jaxpr = jax.make_jaxpr(workflow)(y, x)

        qnode_output_eqns = get_qnode_output_eqns(jaxpr)
        assert len(qnode_output_eqns) == 2
        for eqn in qnode_output_eqns:
            assert len(eqn.outvars) == 1
            assert eqn.outvars[0].aval.shape == (2,)

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, y, x)
        expected = jnp.array([0.93005586, 0.00498127])
        expected2 = jnp.array([0.93005586, -0.97884155])
        assert jnp.allclose(result[0], expected)
        assert jnp.allclose(result[1], expected2)

    def test_vmap_tuple_in_axes(self):
        """Test vmap of a hybrid workflow with tuple in_axes."""

        def workflow(x, y, z):
            @qml.qnode(qml.device("default.qubit", wires=1))
            def circuit(x, y):
                qml.RX(jnp.pi * x[0] + y - y, wires=0)
                qml.RY(x[1] ** 2, wires=0)
                qml.RX(x[1] * x[2], wires=0)
                return qml.expval(qml.PauliZ(0))

            def workflow2(x, y):
                return circuit(x, y) * y

            def workflow3(y, x):
                return circuit(x, y) * y

            def workflow4(y, x, z):
                return circuit(x, y) * y * z

            res1 = jax.vmap(workflow2, in_axes=(0, None))(x, y)
            res2 = jax.vmap(workflow3, in_axes=(None, 0))(y, x)
            res3 = jax.vmap(workflow4, in_axes=(None, 0, None))(y, x, z)
            return res1, res2, res3

        y = jnp.pi
        x = jnp.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ]
        )

        jaxpr = jax.make_jaxpr(workflow)(x, y, 1)

        qnode_output_eqns = get_qnode_output_eqns(jaxpr)
        assert len(qnode_output_eqns) == 3
        for eqn in qnode_output_eqns:
            assert len(eqn.outvars) == 1
            assert eqn.outvars[0].aval.shape == (3,)

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x, y, 1)
        expected = jnp.array([0.93005586, 0.00498127, -0.88789978]) * y
        # note! Any failures here my be a result of a side effect from a different test
        # fails when testing tests/capture, passes with tests/capture/test_capture_qnode
        assert jnp.allclose(result[0], expected)
        assert jnp.allclose(result[1], expected)
        assert jnp.allclose(result[2], expected)

    def test_vmap_pytree_in_axes(self):
        """Test vmap of a hybrid workflow with pytree in_axes."""

        def workflow(x, y, z):
            @qml.qnode(qml.device("default.qubit", wires=1))
            def circuit(x, y):
                qml.RX(jnp.pi * x["arr"][0] + y - y, wires=0)
                qml.RY(x["arr"][1] ** 2, wires=0)
                qml.RX(x["arr"][1] * x["arr"][2], wires=0)
                return qml.expval(qml.PauliZ(0))

            def workflow2(x, y):
                return circuit(x, y) * y

            def workflow3(y, x):
                return circuit(x, y) * y

            def workflow4(y, x, z):
                return circuit(x, y) * y * z

            res1 = jax.vmap(workflow2, in_axes=({"arr": 0, "foo": None}, None))(x, y)
            res2 = jax.vmap(workflow2, in_axes=({"arr": 0, "foo": None}, None))(x, y)
            res3 = jax.vmap(workflow3, in_axes=(None, {"arr": 0, "foo": None}))(y, x)
            res4 = jax.vmap(workflow4, in_axes=(None, {"arr": 0, "foo": None}, None))(y, x, z)
            return res1, res2, res3, res4

        y = jnp.pi
        x = {
            "arr": jnp.array(
                [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                    [0.7, 0.8, 0.9],
                ]
            ),
            "foo": None,
        }

        jaxpr = jax.make_jaxpr(workflow)(x, y, 1)

        qnode_output_eqns = get_qnode_output_eqns(jaxpr)
        assert len(qnode_output_eqns) == 4
        for eqn in qnode_output_eqns:
            assert len(eqn.outvars) == 1
            assert eqn.outvars[0].aval.shape == (3,)

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x["arr"], y, 1)
        expected = jnp.array([0.93005586, 0.00498127, -0.88789978]) * y
        assert jnp.allclose(result[0], expected)
        assert jnp.allclose(result[1], expected)
        assert jnp.allclose(result[2], expected)
        assert jnp.allclose(result[3], expected)

    def test_vmap_circuit_return_tensor(self):
        """Test vmapping over a QNode that returns a tensor."""

        def workflow(x):
            @qml.qnode(qml.device("default.qubit", wires=1))
            def circuit(x):
                qml.RX(jnp.pi * x[0], wires=0)
                qml.RY(x[1] ** 2, wires=0)
                qml.RX(x[1] * x[2], wires=0)
                return qml.state()

            res1 = jax.vmap(circuit)(x)
            res2 = jax.vmap(circuit, out_axes=0)(x)
            return res1, res2

        x = jnp.array([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]])

        jaxpr = jax.make_jaxpr(workflow)(x)

        qnode_output_eqns = get_qnode_output_eqns(jaxpr)
        assert len(qnode_output_eqns) == 2
        for eqn in qnode_output_eqns:
            assert len(eqn.outvars) == 1
            assert eqn.outvars[0].aval.shape == (2, 2)

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)
        expected = jnp.array(
            [
                [0.98235508 + 0.00253459j, 0.0198374 - 0.18595308j],
                [0.10537427 + 0.2120056j, 0.23239136 - 0.94336851j],
            ]
        )
        assert jnp.allclose(result[0], expected)
        assert jnp.allclose(result[1], expected)

    def test_vmap_circuit_return_tensor_pytree(self):
        """Test vmapping over a QNode that returns a pytree tensor."""

        def workflow(x):
            @qml.qnode(qml.device("default.qubit", wires=1))
            def circuit(x):
                qml.RX(jnp.pi * x[0], wires=0)
                qml.RY(x[1] ** 2, wires=0)
                qml.RX(x[1] * x[2], wires=0)
                return qml.state(), qml.probs(0)

            res1 = jax.vmap(circuit)(x)
            return res1

        x = jnp.array([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]])

        jaxpr = jax.make_jaxpr(workflow)(x)

        assert len(jaxpr.eqns[0].outvars) == 2
        assert jaxpr.out_avals[0].shape == (2, 2)
        assert jaxpr.out_avals[1].shape == (2, 2)

        expected_state = jnp.array(
            [
                [0.98235508 + 0.00253459j, 0.0198374 - 0.18595308j],
                [0.10537427 + 0.2120056j, 0.23239136 - 0.94336851j],
            ]
        )
        expected_probs = jnp.array([[0.96502793, 0.03497207], [0.05605011, 0.94394989]])

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)
        assert jnp.allclose(result[0], expected_state)
        assert jnp.allclose(result[1], expected_probs)

    def test_vmap_circuit_return_tensor_out_axes_multiple(self):
        """Test vmapping over a QNode that returns a tensor with multiple out_axes."""

        def workflow(x):
            @qml.qnode(qml.device("default.qubit", wires=1))
            def circuit(x):
                qml.RX(jnp.pi * x[0], wires=0)
                qml.RY(x[1] ** 2, wires=0)
                qml.RX(x[1] * x[2], wires=0)
                return qml.state(), qml.state()

            res1 = jax.vmap(circuit, out_axes=1)(x)
            res2 = jax.vmap(circuit, out_axes=(0, 1))(x)
            return res1, res2

        x = jnp.array([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]])

        jaxpr = jax.make_jaxpr(workflow)(x)

        qnode_output_eqns = get_qnode_output_eqns(jaxpr)
        assert len(qnode_output_eqns) == 2
        for eqn in qnode_output_eqns:
            assert len(eqn.outvars) == 2
            assert eqn.outvars[0].aval.shape == (2, 2)
            assert eqn.outvars[1].aval.shape == (2, 2)

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)
        expected = jnp.array(
            [
                [0.98235508 + 0.00253459j, 0.0198374 - 0.18595308j],
                [0.10537427 + 0.2120056j, 0.23239136 - 0.94336851j],
            ]
        )
        assert jnp.allclose(jnp.transpose(result[0], (1, 0)), expected)
        assert jnp.allclose(jnp.transpose(result[1], (1, 0)), expected)
        assert jnp.allclose(result[2], expected)
        assert jnp.allclose(jnp.transpose(result[3], (1, 0)), expected)


class TestQNodeAutographIntegration:
    """Tests for Autograph integration with QNodes."""

    @pytest.mark.parametrize("autograph", [True, False])
    def test_python_for_loop(self, autograph):
        """Tests that native Python for loops can be used with the QNode."""
        dev = qml.device("default.qubit", wires=[0, 1, 2])

        @qml.qnode(dev, autograph=autograph)
        def circuit(n):
            for i in range(n):
                qml.H(i)
            return qml.state()

        if autograph:
            expected_state = [1 / qml.math.sqrt(8)] * (2**3)
            assert qml.math.allclose(circuit(3), expected_state)
        else:
            with pytest.raises(
                CaptureError,
                match=(
                    "Autograph must be used when Python control flow is dependent on a "
                    r"dynamic variable \(a function input\)"
                ),
            ):
                circuit(3)

    @pytest.mark.parametrize("autograph", [True, False])
    def test_python_while_loop(self, autograph):
        """Tests that native Python while loops can be used with the QNode."""
        dev = qml.device("default.qubit", wires=[0, 1, 2])

        @qml.qnode(dev, autograph=autograph)
        def circuit(n):
            i = 0
            while i < n:
                qml.H(i)
                i += 1
            return qml.state()

        if autograph:
            expected_state = [1 / qml.math.sqrt(8)] * (2**3)
            assert qml.math.allclose(circuit(3), expected_state)
        else:
            with pytest.raises(
                CaptureError,
                match=(
                    "Autograph must be used when Python control flow is dependent on a "
                    r"dynamic variable \(a function input\)"
                ),
            ):
                circuit(3)

    @pytest.mark.parametrize("autograph", [True, False])
    def test_python_conditionals(self, autograph):
        """Test that native Python conditional statements can be used with the QNode."""
        dev = qml.device("default.qubit", wires=[0])

        @qml.qnode(dev, autograph=autograph)
        def circuit(x):
            if x > 1:
                qml.Hadamard(0)
            else:
                qml.I(0)
            return qml.state()

        if autograph:
            assert qml.math.allclose(circuit(0), [1, 0])
            assert qml.math.allclose(circuit(2), [qml.numpy.sqrt(2) / 2, qml.numpy.sqrt(2) / 2])
        else:
            with pytest.raises(
                CaptureError,
                match=(
                    "Autograph must be used when Python control flow is dependent on a "
                    r"dynamic variable \(a function input\)"
                ),
            ):
                circuit(0)

    @pytest.mark.parametrize("autograph", [True, False])
    def test_pennylane_for_loop(self, autograph):
        """Test that a native Pennylane for loop can be used with the QNode."""
        dev = qml.device("default.qubit", wires=[0, 1, 2])

        @qml.qnode(dev, autograph=autograph)
        def circuit(n: int):
            @qml.for_loop(n)
            def loop(i):
                qml.H(wires=i)

            loop()
            return qml.state()

        expected_state = [0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0.0]
        assert qml.math.allclose(circuit(2), expected_state)

    @pytest.mark.parametrize("autograph", [True, False])
    def test_pennylane_while_loop_lambda(self, autograph):
        """Test that a native Pennylane while loop can be used with the QNode."""
        if autograph:
            pytest.xfail(reason="Autograph bug with lambda functions as condition, see sc-82837")

        dev = qml.device("default.qubit", wires=[0, 1, 2])

        @qml.qnode(dev, autograph=autograph)
        def circuit(n: int):
            @qml.while_loop(lambda i: i < n)
            def loop(i):
                qml.H(wires=i)
                return i + 1

            loop(0)
            return qml.state()

        expected_state = [0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0.0]
        assert qml.math.allclose(circuit(2), expected_state)

    @pytest.mark.parametrize("autograph", [True, False])
    def test_pennylane_while_loop(self, autograph):
        """Test that a native Pennylane while loop can be used with the QNode."""
        dev = qml.device("default.qubit", wires=[0, 1, 2])

        @qml.qnode(dev, autograph=autograph)
        def circuit(n: int):
            def condition(i):
                return i < n

            @qml.while_loop(condition)
            def loop(i):
                qml.H(wires=i)
                return i + 1

            loop(0)
            return qml.state()

        expected_state = [0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0.0]
        assert qml.math.allclose(circuit(2), expected_state)

    @pytest.mark.parametrize("autograph", [True, False])
    def test_pennylane_conditional_statements(self, autograph):
        """Test that a native Pennylane conditional statements can be used with the QNode."""
        dev = qml.device("default.qubit", wires=[0, 1])

        @qml.qnode(dev, autograph=autograph)
        def circuit():
            qml.X(0)
            m0 = qml.measure(0)
            qml.cond(m0, qml.X, false_fn=qml.I)(wires=[1])

            return qml.state()

        assert qml.math.allclose(circuit(), [0, 0, 0, 1])


class TestStaticArgnums:
    """Unit tests for `QNode.static_argnums`."""

    @pytest.mark.parametrize("sort_static_argnums", [True, False])
    def test_qnode_static_argnums(self, sort_static_argnums):
        """Test that a QNode's static argnums are used to capture the QNode's quantum function."""
        # Testing using `jax.jit` with `static_argnums` is done in the `TestCaptureCaching` class

        dev = qml.device("default.qubit", wires=2)
        args = (1.5, 2.5, 3.5)
        static_argnums = (0, 1) if sort_static_argnums else (1, 0)

        @qml.qnode(dev, static_argnums=static_argnums)
        def circuit(a, b, c):
            qml.RX(a, 0)
            qml.RY(b + c, 1)

            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        jaxpr = jax.make_jaxpr(circuit, static_argnums=static_argnums)(*args)
        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert len(qfunc_jaxpr.invars) == 1

        assert qml.math.allclose(qfunc_jaxpr.eqns[0].invars[0].val, args[0])
        assert qml.math.allclose(qfunc_jaxpr.eqns[1].invars[0].val, args[1])

        # Empty capture_cache so we don't use cached jaxpr
        circuit.capture_cache.clear()

        res = circuit(*args)
        with qml.capture.pause():
            assert qml.math.allclose(res, circuit(*args))

    def test_qnode_static_argnums_pytree(self):
        """Test that using static argnums with pytree inputs works correctly."""

        dev = qml.device("default.qubit", wires=2)
        args = ({"1": 2.5, "2": 3.5, "3": 4.5}, (1.5, 5.5))
        static_argnums = 1

        @qml.qnode(dev, static_argnums=static_argnums)
        def circuit(a, b):
            qml.RX(a["1"], 0)
            qml.RY(a["2"] + a["3"], 1)
            qml.RX(b[0], 1)
            qml.RY(b[1], 0)

            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        jaxpr = jax.make_jaxpr(circuit, static_argnums=static_argnums)(*args)
        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert len(qfunc_jaxpr.invars) == len(args[0])
        assert qml.math.allclose(qfunc_jaxpr.eqns[3].invars[0].val, args[1][0])
        assert qml.math.allclose(qfunc_jaxpr.eqns[4].invars[0].val, args[1][1])

        # Empty capture_cache so we don't use cached jaxpr
        circuit.capture_cache.clear()

        res = circuit(*args)
        with qml.capture.pause():
            assert qml.math.allclose(res, circuit(*args))

    def test_qnode_static_argnums_autograph(self):
        """Test that static_argnums work as expected with autograph"""

        dev = qml.device("default.qubit", wires=5)

        @qml.qnode(dev, static_argnums=3, autograph=True)
        def circuit(x, y, z, n):

            if z > 5:
                for i in range(n):
                    qml.RX(x, i)
            elif z > 3:
                for i in range(n):
                    qml.RY(y, i)
            else:
                for i in range(n):
                    qml.RZ(z, i)

            for i in range(n - 1):
                qml.CNOT([i, i + 1])

            i = 0
            while i < x + y + z:
                qml.Rot(x, y, z, i % 5)
                i += 1

            return qml.state()

        args = (1.5, 2.5, 3.5, 5)
        res = circuit(*args)
        with qml.capture.pause():
            assert qml.math.allclose(res, circuit(*args))


class TestQNodeCaptureCaching:
    """Unit tests for caching QNode executions with program capture."""

    def check_execution_results(self, circuit, *args, **kwargs):
        """Helper function to compare execution results"""
        res = circuit(*args, **kwargs)
        with qml.capture.pause():
            assert qml.math.allclose(res, circuit(*args, **kwargs))

    def test_caching(self, mocker):
        """Test that caching works correctly."""

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        spy = mocker.spy(jax, "make_jaxpr")
        self.check_execution_results(circuit, jnp.array(1.5))
        spy.assert_called()

        # Cache hit because arguments are of same type/shape
        spy.reset_mock()
        self.check_execution_results(circuit, 100.1)
        spy.assert_not_called()

        # Cache miss because arguments are not of same type/shape
        spy.reset_mock()
        self.check_execution_results(circuit, jnp.array(2))
        spy.assert_called()

        # Cache hit because arguments are of same type/shape
        spy.reset_mock()
        self.check_execution_results(circuit, 10)
        spy.assert_not_called()

    def test_caching_kwargs(self, mocker):
        """Test that caching works correctly when the QNode has kwargs."""

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit(x, y=1):
            qml.RX(x, 0)
            qml.RY(y, 1)
            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        spy = mocker.spy(jax, "make_jaxpr")
        self.check_execution_results(circuit, 1.0, y=1)
        spy.assert_called()

        # Cache hit because same arg shape/type and same kwarg
        spy.reset_mock()
        self.check_execution_results(circuit, 2.0, y=1)
        spy.assert_not_called()

        # Cache miss because same arg shape/type but different kwarg
        spy.reset_mock()
        self.check_execution_results(circuit, 1.0, y=2)
        spy.assert_called()

        # Cache hit because same arg shape/type and same kwarg
        spy.reset_mock()
        self.check_execution_results(circuit, 2.0, y=2)
        spy.assert_not_called()

    def test_caching_pytree(self, mocker):
        """Test that caching works correctly for pytree inputs."""

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x[0], 0)
            qml.RY(x[1], 1)
            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        spy = mocker.spy(jax, "make_jaxpr")
        self.check_execution_results(circuit, (1.5, 2.5))
        spy.assert_called()

        # Cache hit because same arg shape/types
        spy.reset_mock()
        self.check_execution_results(circuit, (jnp.array(5.1), 3.5))
        spy.assert_not_called()

        # Cache miss because different arg shape/types
        spy.reset_mock()
        self.check_execution_results(circuit, (jnp.array(5.1), 3))
        spy.assert_called()

        # Cache hit because same arg shape/types
        spy.reset_mock()
        self.check_execution_results(circuit, (3.5, jnp.array(2)))
        spy.assert_not_called()

    def test_caching_static_argnums(self, mocker):
        """Test that caching works correctly when a QNode has static arguments."""

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev, static_argnums=1)
        def circuit(x, y):
            qml.RX(x, 0)
            qml.RY(y, 1)
            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        spy = mocker.spy(jax, "make_jaxpr")
        self.check_execution_results(circuit, 1.0, 2.0)
        spy.assert_called()

        # Cache hit because same arg shape/type and same static arg
        spy.reset_mock()
        self.check_execution_results(circuit, 1.1, 2.0)
        spy.assert_not_called()

        # Cache miss because same arg shape/type but different static arg
        spy.reset_mock()
        self.check_execution_results(circuit, 1.1, 2.1)
        spy.assert_called()

        # Cache hit because same arg shape/type and same static arg
        spy.reset_mock()
        self.check_execution_results(circuit, 1.0, 2.1)
        spy.assert_not_called()

    def test_caching_static_argnums_pytree(self, mocker):
        """Test that caching works correctly when a QNode has static arguments
        with pytree inputs."""

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev, static_argnums=1)
        def circuit(x, y):
            qml.RX(x[0], 0)
            qml.RX(x[1], 1)
            qml.RY(y[0], 0)
            qml.RY(y[1], 1)
            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        spy = mocker.spy(jax, "make_jaxpr")
        self.check_execution_results(circuit, (1.5, 2.5), (3.5, 4.5))
        spy.assert_called()

        # Cache hit because same arg shape/type and same static args
        spy.reset_mock()
        self.check_execution_results(circuit, (4.5, 5.5), (3.5, 4.5))
        spy.assert_not_called()

        # Cache hit because same arg shape/type and same static args
        spy.reset_mock()
        self.check_execution_results(circuit, (4.5, 5.5), (3.1, 5.5))
        spy.assert_called()

    # pylint: disable=unused-argument
    def test_caching_dynamic_shapes(self, mocker, enable_disable_dynamic_shapes):
        """Test that caching works correctly when a QNode has arguments with
        dynamic shapes."""

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(jnp.sum(x), 0)
            return qml.expval(qml.Z(0))

        spy = mocker.spy(jax, "make_jaxpr")
        _ = jax.make_jaxpr(circuit, abstracted_axes=("a",))(jnp.arange(10))
        assert spy.call_count > 1

        # Only one call to make_jaxpr because of the call we make here. If cache is used,
        # No other calls to make_jaxpr should be made.
        spy.reset_mock()
        _ = jax.make_jaxpr(circuit, abstracted_axes=("a",))(jnp.arange(100))
        assert spy.call_count == 1

        # We changed the dtype, so there will be a cache miss.
        spy.reset_mock()
        _ = jax.make_jaxpr(circuit, abstracted_axes=("a",))(jnp.arange(100, dtype=jnp.complex128))
        assert spy.call_count > 1

    # pylint: disable=unused-argument
    def test_caching_dynamic_shapes_pytree(self, mocker, enable_disable_dynamic_shapes):
        """Test that caching works correctly when a QNode has pytree arguments
        with dynamic shapes."""

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(jnp.sum(x["0"]), 0)
            qml.RY(jnp.sum(x["1"]), 1)
            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        abstracted_axes = ({"0": {0: "a"}, "1": {0: "b"}},)
        spy = mocker.spy(jax, "make_jaxpr")
        _ = jax.make_jaxpr(circuit, abstracted_axes=abstracted_axes)(
            {"0": jnp.arange(10), "1": jnp.arange(100)}
        )
        assert spy.call_count > 1

        # Only one call to make_jaxpr because of the call we make here. If cache is used,
        # No other calls to make_jaxpr should be made.
        spy.reset_mock()
        _ = jax.make_jaxpr(circuit, abstracted_axes=abstracted_axes)(
            {"0": jnp.arange(5), "1": jnp.arange(21)}
        )
        assert spy.call_count == 1

        # We changed the dtype, so there will be a cache miss.
        spy.reset_mock()
        _ = jax.make_jaxpr(circuit, abstracted_axes=abstracted_axes)(
            {"0": jnp.arange(5), "1": jnp.arange(21, dtype=jnp.complex128)}
        )
        assert spy.call_count > 1

    # pylint: disable=unused-argument
    def test_caching_dynamic_shapes_and_static_argnums(self, mocker, enable_disable_dynamic_shapes):
        """Test that caching works correctly when a QNode has arguments with
        dynamic shapes as well as static arguments."""

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev, static_argnums=1)
        def circuit(x, y):
            qml.RX(jnp.sum(x), 0)
            qml.RY(y, 0)
            return qml.expval(qml.Z(0))

        abstracted_axes = ({0: "a"},)
        spy = mocker.spy(jax, "make_jaxpr")
        _ = jax.make_jaxpr(circuit, abstracted_axes=abstracted_axes, static_argnums=1)(
            jnp.arange(10), 3.5
        )
        assert spy.call_count > 1

        # Only one call to make_jaxpr because of the call we make here. If cache is used,
        # No other calls to make_jaxpr should be made.
        spy.reset_mock()
        _ = jax.make_jaxpr(circuit, abstracted_axes=abstracted_axes, static_argnums=1)(
            jnp.arange(100), 3.5
        )
        assert spy.call_count == 1

        # We changed the static arguments, so there will be a cache miss.
        spy.reset_mock()
        _ = jax.make_jaxpr(circuit, abstracted_axes=abstracted_axes, static_argnums=1)(
            jnp.arange(100), 9.2
        )
        assert spy.call_count > 1

        # We changed the dtype, so there will be a cache miss.
        spy.reset_mock()
        _ = jax.make_jaxpr(circuit, abstracted_axes=abstracted_axes, static_argnums=1)(
            jnp.arange(100, dtype=jnp.complex128), 4.5
        )
        assert spy.call_count > 1

    # pylint: disable=unused-argument
    def test_caching_dynamic_shapes_and_static_argnums_pytree(
        self, mocker, enable_disable_dynamic_shapes
    ):
        """Test that caching works correctly when a QNode has pytree arguments
        with dynamic shapes as well as static arguments."""

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev, static_argnums=1)
        def circuit(x, y):
            qml.RX(jnp.sum(x["0"]), 0)
            qml.RY(y[0], 0)
            qml.RX(jnp.sum(x["1"]), 1)
            qml.RY(y[1], 1)
            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        abstracted_axes = ({"0": {0: "a"}, "1": {0: "b"}},)
        spy = mocker.spy(jax, "make_jaxpr")
        _ = jax.make_jaxpr(circuit, abstracted_axes=abstracted_axes, static_argnums=1)(
            {"0": jnp.arange(10), "1": jnp.arange(100)}, (3.5, 4.6)
        )
        assert spy.call_count > 1

        # Only one call to make_jaxpr because of the call we make here. If cache is used,
        # No other calls to make_jaxpr should be made.
        spy.reset_mock()
        _ = jax.make_jaxpr(circuit, abstracted_axes=abstracted_axes, static_argnums=1)(
            {"0": jnp.arange(5), "1": jnp.arange(21)}, (3.5, 4.6)
        )
        assert spy.call_count == 1

        # We changed the static arguments, so there will be a cache miss.
        spy.reset_mock()
        _ = jax.make_jaxpr(circuit, abstracted_axes=abstracted_axes, static_argnums=1)(
            {"0": jnp.arange(5), "1": jnp.arange(21)}, (8.1, 4.6)
        )
        assert spy.call_count > 1

        # We changed the dtype, so there will be a cache miss.
        spy.reset_mock()
        _ = jax.make_jaxpr(circuit, abstracted_axes=abstracted_axes, static_argnums=1)(
            {"0": jnp.arange(5), "1": jnp.arange(21, dtype=jnp.complex128)}, (8.1, 4.6)
        )
        assert spy.call_count > 1

    def test_caching_jit(self):
        """Test that caching does not impact jitting."""

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        jitted_circuit = jax.jit(circuit)
        args = [jnp.array(1.5), jnp.array(2.5)]
        res1 = jitted_circuit(args[0])
        res2 = jitted_circuit(args[1])

        with qml.capture.pause():
            assert qml.math.allclose(res1, circuit(args[0]))
            assert qml.math.allclose(res2, circuit(args[1]))

    def test_caching_jit_static_argnums(self):
        """Test that caching does not impact jitting with static_argnums."""

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev, static_argnums=1)
        def circuit(x, y):
            qml.RX(x, 0)
            qml.RY(y, 1)
            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        jitted_circuit = jax.jit(circuit, static_argnums=1)
        args = [
            (jnp.array(1.5), 2.5),
            (jnp.array(2.5), 2.5),
            (jnp.array(1.5), 3.5),
            (jnp.array(2.5), 3.5),
        ]

        for a in args:
            res = jitted_circuit(*a)

            with qml.capture.pause():
                assert qml.math.allclose(res, circuit(*a))

    def test_caching_with_autograph(self):
        """Test that using autograph works as expected when caching is active."""

        dev = qml.device("default.qubit", wires=5)

        @qml.qnode(dev, autograph=True)
        def circuit(x, y, z, n):

            if z > 5:
                for i in range(n):
                    qml.RX(x, i)
            elif z > 3:
                for i in range(n):
                    qml.RY(y, i)
            else:
                for i in range(n):
                    qml.RZ(z, i)

            for i in range(n - 1):
                qml.CNOT([i, i + 1])

            i = 0
            while i < x + y + z:
                qml.Rot(x, y, z, i % 5)
                i += 1

            return qml.state()

        # Specifying parameters here instead of using @pytest.mark.parametrize
        # to force usage of cache
        xs = [1.5, 2.5, 4, 5]
        ys = [-2, 1.5, 3.5, 2]
        zs = [1.5, 3.5, 5.5, 2, 4, 6]
        ns = list(range(5))

        for x, y, z, n in product(xs, ys, zs, ns):
            self.check_execution_results(circuit, x, y, z, n)
