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

# pylint: disable=protected-access
import pytest

import pennylane as qml
from pennylane.exceptions import CaptureError, QuantumFunctionError

pytestmark = [pytest.mark.jax, pytest.mark.capture]

jax = pytest.importorskip("jax")
jnp = jax.numpy

from pennylane.capture.autograph import run_autograph  # pylint: disable=wrong-import-position

# must be below jax importorskip
from pennylane.capture.primitives import qnode_prim  # pylint: disable=wrong-import-position
from pennylane.tape.plxpr_conversion import (  # pylint: disable=wrong-import-position
    CollectOpsandMeas,
)


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


def test_warning_about_execution_pipeline_unmaintained():
    """Test that a warning is raised saying the native execution is unmaintained."""

    @qml.qnode(qml.device("default.qubit", wires=1))
    def c():
        return qml.probs()

    with pytest.warns(UserWarning, match="Executing PennyLane programs with capture enabled"):
        c()


def test_error_if_no_device_wires():
    """Test that a NotImplementedError is raised if the device does not provide wires."""

    dev = qml.device("default.qubit", wires=None)

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
    assert eqn0.params["shots_len"] == 0
    expected_config = qml.devices.ExecutionConfig(
        gradient_method="best",
        use_device_gradient=None,
        gradient_keyword_arguments={},
        use_device_jacobian_product=False,
        interface="jax",
        grad_on_execution=False,
        device_options={},
        mcm_config=qml.devices.MCMConfig(mcm_method=None, postselect_mode=None),
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
    assert qml.math.allclose(res, jnp.array([0, 1]))

    res2 = circuit(n_iterations=4)
    assert qml.math.allclose(res2, jnp.array([1, 0]))


def test_multiple_measurements():
    """Test that the qnode can return multiple measurements."""

    @qml.qnode(qml.device("default.qubit", wires=3), shots=50)
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
        mcm_method="single-branch-statistics",
    )
    def circuit():
        return qml.expval(qml.Z(0))

    jaxpr = jax.make_jaxpr(circuit)()

    assert jaxpr.eqns[0].primitive == qnode_prim
    expected_config = qml.devices.ExecutionConfig(
        gradient_method="parameter-shift",
        use_device_gradient=None,
        grad_on_execution=False,
        derivative_order=2,
        use_device_jacobian_product=False,
        mcm_config=qml.devices.MCMConfig(
            mcm_method="single-branch-statistics", postselect_mode=None
        ),
        interface=qml.math.Interface.JAX,
        device_options={},
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


class TestShots:
    """Tests for the number of shots."""

    def test_shot_vector(self):
        """Test that a shot vector can be captured."""

        @qml.set_shots(shots=(10, 10, 20))
        @qml.qnode(qml.device("default.qubit", wires=1))
        def c():
            return qml.sample(wires=0)

        jaxpr = jax.make_jaxpr(c)()
        assert len(jaxpr.eqns) == 1
        eqn0 = jaxpr.eqns[0]
        assert eqn0.primitive == qnode_prim
        assert eqn0.params["shots_len"] == 3

        assert eqn0.invars[0].val == 10
        assert eqn0.invars[1].val == 10
        assert eqn0.invars[2].val == 20

        assert len(eqn0.outvars) == 3
        assert eqn0.outvars[0].aval.shape == (10, 1)
        assert eqn0.outvars[1].aval.shape == (10, 1)
        assert eqn0.outvars[2].aval.shape == (20, 1)

    def test_shot_vector_multiple_returns_pytree(self):
        """Test that shot vectors and multiple returns returns the correct shapes."""

        @qml.set_shots(shots=(10, 11))
        @qml.qnode(qml.device("default.qubit", wires=1))
        def c():
            return {"sample": qml.sample(wires=0), "expval": qml.expval(qml.Z(0))}

        def w():
            out = c()
            assert isinstance(out, tuple)
            for i, d in enumerate(out):
                assert isinstance(d, dict)
                assert d["sample"].shape == (10 + i, 1)
                assert d["expval"].shape == ()
            return out

        jaxpr = jax.make_jaxpr(w)()
        eqn0 = jaxpr.eqns[0]
        assert len(eqn0.outvars) == 4
        # for some reason flattening the pytree puts the expval first
        assert eqn0.outvars[0].aval.shape == ()
        assert eqn0.outvars[1].aval.shape == (10, 1)
        assert eqn0.outvars[2].aval.shape == ()
        assert eqn0.outvars[3].aval.shape == (11, 1)

    def test_error_dynamic_shots_dynamic_shapes_not_enabled(self):
        """Test that an error is raised if dynamic shots is not enabled."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def c():
            return qml.sample(wires=0), qml.expval(qml.Z(0))

        def w(num_shots):
            return qml.set_shots(c, num_shots)()

        with pytest.raises(ValueError, match=r"requires setting jax.config.update"):
            jax.make_jaxpr(w)(3)

    @pytest.mark.usefixtures("enable_disable_dynamic_shapes")
    def test_dynamic_shots(self):
        """Test that the shot number can be dynamic."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def c():
            return qml.sample(wires=0), qml.expval(qml.Z(0))

        def w(num_shots):
            return qml.set_shots(c, num_shots)()

        jaxpr = jax.make_jaxpr(w)(3)
        assert len(jaxpr.eqns) == 1
        eqn0 = jaxpr.eqns[0]

        assert eqn0.params["shots_len"] == 1
        assert not isinstance(eqn0.invars[0], jax.extend.core.Literal)

        assert isinstance(eqn0.outvars[0].aval, jax.core.DShapedArray)
        assert isinstance(eqn0.outvars[1].aval, jax.core.ShapedArray)

        assert eqn0.outvars[0].aval.shape[0] is eqn0.invars[0]
        assert eqn0.outvars[1].aval.shape == ()

    @pytest.mark.usefixtures("enable_disable_dynamic_shapes")
    def test_dynamic_shots_shot_vector(self):
        """Test that a shot vector can be used with a shot vector."""

        @qml.qnode(qml.device("default.qubit", wires=2))
        def c():
            return qml.sample(wires=0)

        def w(shots1, shots2):
            out = qml.set_shots(c, (shots1, 2, shots2))()
            return out

        jaxpr = jax.make_jaxpr(w)(5, 6)
        eqn = jaxpr.eqns[-1]
        assert eqn.params["shots_len"] == 3
        assert len(eqn.outvars) == 3

        assert isinstance(eqn.outvars[0].aval, jax.core.DShapedArray)
        assert eqn.outvars[0].aval.shape[0] is eqn.invars[0]
        assert isinstance(eqn.outvars[2].aval, jax.core.DShapedArray)
        assert eqn.outvars[2].aval.shape[0] is eqn.invars[2]

        assert isinstance(eqn.outvars[1].aval, jax.core.ShapedArray)
        assert eqn.outvars[1].aval.shape == (2, 1)


@pytest.mark.parametrize("disable_around_qnode", (True, False))
class TestUserTransforms:
    """Integration tests for applying user transforms to a qnode with program capture."""

    @pytest.mark.unit
    def test_captured_program_qnode_transform(self, disable_around_qnode):
        """Test that a transformed qnode is captured correctly."""

        if disable_around_qnode:
            qml.capture.disable()

        dev = qml.device("default.qubit", wires=3)

        @qml.transforms.cancel_inverses
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, 0)
            qml.X(0)
            qml.X(0)
            return qml.expval(qml.Z(0))

        assert isinstance(circuit, qml.QNode)
        assert qml.transforms.cancel_inverses in circuit.transform_program

        if disable_around_qnode:
            qml.capture.enable()

        jaxpr = jax.make_jaxpr(circuit)(1.5)
        # pylint: disable=protected-access
        assert jaxpr.eqns[0].primitive == qml.transforms.cancel_inverses._primitive
        inner_jaxpr = jaxpr.eqns[0].params["inner_jaxpr"]
        assert inner_jaxpr.eqns[0].primitive == qnode_prim
        qfunc_jaxpr = inner_jaxpr.eqns[0].params["qfunc_jaxpr"]

        collector = CollectOpsandMeas()
        collector.eval(qfunc_jaxpr, [], 1.5)
        assert collector.state["ops"] == [qml.RX(1.5, 0), qml.X(0), qml.X(0)]
        assert collector.state["measurements"] == [qml.expval(qml.Z(0))]

    @pytest.mark.unit
    def test_captured_program_qfunc_transform(self, disable_around_qnode):
        """Test that a qnode with a transformed qfunc is captured correctly."""

        if disable_around_qnode:
            qml.capture.disable()

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        @qml.transforms.cancel_inverses
        def circuit(x):
            qml.RX(x, 0)
            qml.X(0)
            qml.X(0)
            return qml.expval(qml.Z(0))

        if disable_around_qnode:
            qml.capture.enable()

        jaxpr = jax.make_jaxpr(circuit)(1.5)
        assert jaxpr.eqns[0].primitive == qnode_prim
        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        # pylint: disable=protected-access
        assert qfunc_jaxpr.eqns[0].primitive == qml.transforms.cancel_inverses._primitive

        inner_jaxpr = qfunc_jaxpr.eqns[0].params["inner_jaxpr"]
        collector = CollectOpsandMeas()
        collector.eval(inner_jaxpr, [], 1.5)
        assert collector.state["ops"] == [qml.RX(1.5, 0), qml.X(0), qml.X(0)]
        assert collector.state["measurements"] == [qml.expval(qml.Z(0))]

    @pytest.mark.unit
    def test_captured_program_qnode_qfunc_transform(self, disable_around_qnode):
        """Test that a transformed qnode with a transformed qfunc is captured correctly."""

        if disable_around_qnode:
            qml.capture.disable()

        dev = qml.device("default.qubit", wires=3)

        @qml.transforms.cancel_inverses
        @qml.qnode(dev)
        @qml.transforms.merge_rotations
        def circuit(x):
            qml.RX(x, 0)
            qml.X(0)
            qml.X(0)
            return qml.expval(qml.Z(0))

        assert isinstance(circuit, qml.QNode)

        if disable_around_qnode:
            qml.capture.enable()

        jaxpr = jax.make_jaxpr(circuit)(1.5)
        # pylint: disable=protected-access
        assert jaxpr.eqns[0].primitive == qml.transforms.cancel_inverses._primitive
        inner_jaxpr = jaxpr.eqns[0].params["inner_jaxpr"]
        assert inner_jaxpr.eqns[0].primitive == qnode_prim
        qfunc_jaxpr = inner_jaxpr.eqns[0].params["qfunc_jaxpr"]

        assert qfunc_jaxpr.eqns[0].primitive == qml.transforms.merge_rotations._primitive
        inner_jaxpr2 = qfunc_jaxpr.eqns[0].params["inner_jaxpr"]

        collector = CollectOpsandMeas()
        collector.eval(inner_jaxpr2, [], 1.5)
        assert collector.state["ops"] == [qml.RX(1.5, 0), qml.X(0), qml.X(0)]
        assert collector.state["measurements"] == [qml.expval(qml.Z(0))]

    @pytest.mark.unit
    def test_device_jaxpr(self, monkeypatch, disable_around_qnode):
        """Test that jaxpr recieved by a device when executing a transformed qnode has been
        transformed appropriately."""

        device_jaxpr = None

        def dummy_eval_jaxpr(
            jaxpr, consts, *args, execution_config, shots=None
        ):  # pylint: disable=unused-argument
            nonlocal device_jaxpr
            device_jaxpr = jaxpr
            return [1.0]

        if disable_around_qnode:
            qml.capture.disable()

        dev = qml.device("default.qubit", wires=3)
        monkeypatch.setattr(dev, "eval_jaxpr", dummy_eval_jaxpr)

        @partial(qml.transforms.decompose, gate_set=[qml.RX, qml.RY, qml.RZ])
        @qml.qnode(dev)
        @qml.transforms.cancel_inverses
        def circuit(x, y, z):
            qml.Rot(x, y, z, 0)
            qml.X(0)
            qml.X(0)
            return qml.expval(qml.Z(0))

        assert isinstance(circuit, qml.QNode)

        if disable_around_qnode:
            qml.capture.enable()

        _ = circuit(1.5, 2.5, 3.5)
        assert all(
            getattr(eqn.primitive, "prim_type", "") != "transform" for eqn in device_jaxpr.eqns
        )
        assert device_jaxpr.eqns[0].primitive == qml.RZ._primitive
        assert device_jaxpr.eqns[1].primitive == qml.RY._primitive
        assert device_jaxpr.eqns[2].primitive == qml.RZ._primitive
        assert device_jaxpr.eqns[3].primitive == qml.PauliZ._primitive
        assert device_jaxpr.eqns[4].primitive == qml.measurements.ExpectationMP._obs_primitive

    @pytest.mark.integration
    def test_execution(self, disable_around_qnode):
        """Test that a transformed qnode is executed correctly."""

        if disable_around_qnode:
            qml.capture.disable()

        dev = qml.device("default.qubit", wires=3)

        @qml.transforms.cancel_inverses
        @qml.qnode(dev)
        @qml.transforms.merge_rotations
        def circuit(x):
            qml.RX(x, 0)
            qml.RX(4 * x, 0)
            qml.X(0)
            qml.X(0)
            return qml.expval(qml.Z(0))

        if disable_around_qnode:
            qml.capture.enable()

        res = circuit(1.5)
        expected = jnp.cos(5 * 1.5)
        assert jnp.allclose(res, expected)


@pytest.mark.parametrize("dev_name", ["default.qubit", "lightning.qubit"])
class TestDevicePreprocessing:
    """Integration tests for preprocessing and executing qnodes with program capture."""

    def test_non_native_ops_execution(self, dev_name, seed):
        """Test that operators that aren't natively supported by a device can be executed by a qnode."""
        dev = qml.device(dev_name, wires=2, seed=seed)

        @qml.qnode(dev)
        def circuit():
            # QFT not supported on DQ or LQ
            qml.QFT(wires=[0, 1])
            return qml.state()

        assert qml.math.allclose(circuit(), [0.5] * 4)

    @pytest.mark.parametrize("mcm_method", [None, "deferred"])
    @pytest.mark.parametrize("shots", [None, 1000])
    def test_mcms_execution_deferred(self, dev_name, mcm_method, shots, seed):
        """Test that defer_measurements is reflected in the execution results of a device."""

        dev = qml.device(dev_name, wires=3, seed=seed)
        postselect = 1 if dev_name == "default.qubit" else None

        @qml.qnode(dev, mcm_method=mcm_method, shots=shots)
        def circuit():
            qml.Hadamard(0)
            qml.CNOT([0, 1])  # |Φ⁺⟩ = (1/√2) (|00⟩ + |11⟩)
            qml.measure(0, reset=True, postselect=postselect)
            return {
                "expval": (qml.expval(qml.Z(0)), qml.expval(qml.Z(1))),
                "samples": qml.sample(wires=[0, 1]) if shots else None,
            }

        if not shots:
            outcome = -2 * postselect + 1 if postselect else 0
            assert qml.math.allclose(circuit()["expval"], [1, outcome])
        else:
            shots_res = circuit()["samples"]
            if postselect:
                # After post selection and reset (on the first bit)
                # the valid sample is *only* [0, 1] (~shots/2 for bell state)
                assert all(qml.math.allclose(s, [0, 1]) for s in shots_res)
                assert qml.math.isclose(len(shots_res), shots / 2, atol=50)
            else:
                # No longer postselected so |00> and |01> state are valid
                assert all(
                    qml.math.allclose(s, [0, 0]) or qml.math.allclose(s, [0, 1]) for s in shots_res
                )
                assert len(shots_res) == shots
                # Check it's roughly 50/50 by counting the second column of bits
                counts = qml.numpy.bincount(shots_res[:, 1].astype(int))
                assert qml.math.isclose(counts[0] / counts[1], 1, atol=0.3)

    @pytest.mark.parametrize("mcm_method", [None, "deferred"])
    def test_mcm_execution_deferred_fill_shots(self, dev_name, mcm_method, seed):
        """Test that using a qnode with postselect_mode="fill-shots" gives the expected results."""

        shots = 1000
        dev = qml.device(dev_name, wires=3, seed=seed)
        postselect = 1 if dev_name == "default.qubit" else None

        @qml.qnode(dev, mcm_method=mcm_method, postselect_mode="fill-shots", shots=shots)
        def circuit():
            qml.Hadamard(0)
            qml.CNOT([0, 1])  # |Φ⁺⟩ = (1/√2) (|00⟩ + |11⟩)
            qml.measure(0, postselect=postselect)
            return qml.sample(wires=[0, 1])

        shots_res = circuit()
        # pylint: disable = not-an-iterable, unsubscriptable-object
        if postselect:
            # Only postselecting the |11> state ~shots/2 of the time
            # Will fill the rest with |11> state
            assert all(qml.math.allclose(s, [1, 1]) for s in shots_res)
            assert len(shots_res) == shots
        else:
            # No longer postselected so |00> and |11> state are valid
            assert all(
                qml.math.allclose(s, [0, 0]) or qml.math.allclose(s, [1, 1]) for s in shots_res
            )
            assert len(shots_res) == shots
            # Check it's roughly 50/50 by counting the second column of bits
            counts = qml.numpy.bincount(shots_res[:, 1].astype(int))
            assert qml.math.isclose(counts[0] / counts[1], 1, atol=0.3)

    @pytest.mark.parametrize("mcm_method", [None, "deferred"])
    def test_mcm_execution_deferred_hw_like(self, dev_name, mcm_method, seed):
        """Test that using a qnode with postselect_mode="hw-like" gives the expected results."""

        shots = 1000
        dev = qml.device(dev_name, wires=2, seed=seed)
        postselect = 1 if dev_name == "default.qubit" else None
        n_postselects = 3

        @qml.qnode(dev, mcm_method=mcm_method, postselect_mode="hw-like", shots=shots)
        def circuit():

            @qml.for_loop(n_postselects)
            def loop(i):  # pylint: disable=unused-argument
                qml.Hadamard(0)
                qml.measure(0, postselect=postselect)

            loop()
            return qml.sample(wires=[0])

        res = circuit()
        # pylint: disable = not-an-iterable
        if postselect:
            assert all(qml.math.allclose(r, postselect) for r in res)
            num_of_results = len(res)
            assert qml.math.allclose(
                num_of_results,
                int(1000 / (2**n_postselects)),  # 125
                atol=20,
            )
        else:
            assert len(res) == shots
            counts = qml.numpy.bincount(qml.math.squeeze(res))
            assert qml.math.isclose(counts[0] / counts[1], 1, atol=0.3)

    def test_mcms_execution_single_branch_statistics(self, dev_name, seed):
        """Test that single-branch-statistics works as expected."""

        shots = 1000
        dev = qml.device(dev_name, wires=2, seed=seed)

        @qml.qnode(dev, mcm_method="single-branch-statistics", shots=shots)
        def circuit():
            qml.Hadamard(0)
            qml.measure(0)
            return qml.sample(wires=[0])

        # pylint: disable = not-an-iterable
        results = circuit()
        assert all(sample == 0 for sample in results) or all(sample == 1 for sample in results)


class TestDifferentiation:

    def test_error_backprop_unsupported(self):
        """Test an error is raised with backprop if the device does not support it."""

        # pylint: disable=too-few-public-methods
        class DummyDev(qml.devices.Device):

            def execute(self, *_, **__):
                return 0

        with pytest.raises(QuantumFunctionError, match="does not support backprop"):

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
        """Test that JAX can compute the JVP of the QNode primitive via a registered JVP rule on default.qubit."""

        @qml.qnode(qml.device("default.qubit", wires=1), diff_method=diff_method)
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        x = 0.9
        xt = -0.6
        jvp = jax.jvp(circuit, (x,), (xt,))
        assert qml.math.allclose(jvp, (qml.math.cos(x), -qml.math.sin(x) * xt))

    def test_jvp_lightning(self):
        """Test that JAX can compute the JVP of the QNode primitive via a registered rule on lightning.qubit."""

        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        x = 0.9
        xt = -0.6
        jvp = jax.jvp(circuit, (x,), (xt,))
        assert qml.math.allclose(jvp, (qml.math.cos(x), -qml.math.sin(x) * xt))

    def test_grad_lightning(self):
        """Test that JAX can compute the gradient of the QNode primitive via a registered rule on lightning.qubit."""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        grad = jax.grad(circuit)(0.9)
        assert qml.math.allclose(grad, -qml.math.sin(0.9))

    def test_jacobian_lightning(self):
        """Test that JAX can compute the Jacobian of the QNode primitive via a registered rule on lightning.qubit."""

        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit(x):
            qml.RX(x[0], 0)
            qml.RY(x[1], 1)
            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        x = jnp.array([0.9, -0.6])
        jac = jax.jacobian(circuit)(x)
        assert qml.math.allclose(jac, [[-qml.math.sin(0.9), 0], [0, -qml.math.sin(-0.6)]])


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

        @qml.qnode(qml.device("default.qubit", wires=4), shots=5)
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

        jaxpr = jax.make_jaxpr(jax.vmap(qml.set_shots(circuit, shots=50), in_axes=0))(x)
        jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)

        assert len(jaxpr.eqns) == 1
        eqn0 = jaxpr.eqns[0]

        assert eqn0.primitive == qnode_prim
        assert eqn0.params["device"] == dev
        assert eqn0.params["shots_len"] == 1
        assert (
            eqn0.params["qfunc_jaxpr"].eqns[0].primitive
            == qml.measurements.SampleMP._wires_primitive
        )

        assert eqn0.outvars[0].aval.shape == (3, 50, 1)

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

    def test_autograph_with_qnode_transforms(self):
        """Test that autograph can be used when transforms are applied to the qnode."""

        dev = qml.device("default.qubit", wires=[0, 1, 2])

        @qml.capture.run_autograph
        @qml.transforms.merge_rotations
        @qml.transforms.cancel_inverses
        @qml.qnode(dev)
        def c(n):
            for i in range(n):
                qml.H(i)
            return qml.state()

        jaxpr = jax.make_jaxpr(c)(3)
        assert jaxpr.eqns[0].primitive == qml.transforms.merge_rotations._primitive
        j2 = jaxpr.eqns[0].params["inner_jaxpr"]
        assert j2.eqns[0].primitive == qml.transforms.cancel_inverses._primitive
        j3 = j2.eqns[0].params["inner_jaxpr"]
        assert j3.eqns[0].primitive == qnode_prim
        j4 = j3.eqns[0].params["qfunc_jaxpr"]
        assert j4.eqns[0].primitive == qml.capture.primitives.for_loop_prim
        assert j4.eqns[0].invars[1] is j4.invars[0]

    def test_autograph_on_workflow(self):
        """Test autograph can be called on a workflow."""

        @qml.transforms.merge_rotations
        @qml.transforms.cancel_inverses
        @qml.qnode(qml.device("default.qubit", wires=[0, 1, 2]))
        def c(n):
            for i in range(n):
                qml.H(i)
            return qml.expval(qml.Z(0))

        @qml.capture.run_autograph
        def w(n):
            return c(n + 1) + c(n + 3)

        jaxpr = jax.make_jaxpr(w)(3)

        for i in [1, 3]:

            assert jaxpr.eqns[i].primitive == qml.transforms.merge_rotations._primitive
            j2 = jaxpr.eqns[i].params["inner_jaxpr"]
            assert j2.eqns[0].primitive == qml.transforms.cancel_inverses._primitive
            j3 = j2.eqns[0].params["inner_jaxpr"]
            assert j3.eqns[0].primitive == qnode_prim
            j4 = j3.eqns[0].params["qfunc_jaxpr"]
            assert j4.eqns[0].primitive == qml.capture.primitives.for_loop_prim
            # somehow promoted to closure var?
            assert j4.eqns[0].invars[1] is j4.constvars[0]

    @pytest.mark.parametrize("autograph", [True, False])
    def test_python_for_loop(self, autograph):
        """Tests that native Python for loops can be used with the QNode."""
        dev = qml.device("default.qubit", wires=[0, 1, 2])

        @qml.qnode(dev)
        def circuit(n):
            for i in range(n):
                qml.H(i)
            return qml.state()

        if autograph:
            circuit = run_autograph(circuit)
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

        @qml.qnode(dev)
        def circuit(n):
            i = 0
            while i < n:
                qml.H(i)
                i += 1
            return qml.state()

        if autograph:
            circuit = run_autograph(circuit)
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

        @qml.qnode(dev)
        def circuit(x):
            if x > 1:
                qml.Hadamard(0)
            else:
                qml.I(0)
            return qml.state()

        if autograph:
            circuit = run_autograph(circuit)
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

        @qml.qnode(dev)
        def circuit(n: int):
            @qml.for_loop(n)
            def loop(i):
                qml.H(wires=i)

            loop()
            return qml.state()

        circuit = run_autograph(circuit) if autograph else circuit
        expected_state = [0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0.0]
        assert qml.math.allclose(circuit(2), expected_state)

    @pytest.mark.parametrize("autograph", [True, False])
    def test_pennylane_while_loop_lambda(self, autograph):
        """Test that a native Pennylane while loop can be used with the QNode."""
        if autograph:
            pytest.xfail(reason="Autograph bug with lambda functions as condition, see sc-82837")

        dev = qml.device("default.qubit", wires=[0, 1, 2])

        @qml.qnode(dev)
        def circuit(n: int):
            @qml.while_loop(lambda i: i < n)
            def loop(i):
                qml.H(wires=i)
                return i + 1

            loop(0)
            return qml.state()

        circuit = run_autograph(circuit) if autograph else circuit
        expected_state = [0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0.0]
        assert qml.math.allclose(circuit(2), expected_state)

    @pytest.mark.parametrize("autograph", [True, False])
    def test_pennylane_while_loop(self, autograph):
        """Test that a native Pennylane while loop can be used with the QNode."""
        dev = qml.device("default.qubit", wires=[0, 1, 2])

        @qml.qnode(dev)
        def circuit(n: int):
            def condition(i):
                return i < n

            @qml.while_loop(condition)
            def loop(i):
                qml.H(wires=i)
                return i + 1

            loop(0)
            return qml.state()

        circuit = run_autograph(circuit) if autograph else circuit
        expected_state = [0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0.0]
        assert qml.math.allclose(circuit(2), expected_state)

    @pytest.mark.parametrize("autograph", [True, False])
    def test_pennylane_conditional_statements(self, autograph):
        """Test that a native Pennylane conditional statement can be used with the QNode."""
        dev = qml.device("default.qubit", wires=[0, 1, 2])

        @qml.qnode(dev)
        def circuit():
            qml.X(0)
            m0 = qml.measure(0)
            qml.cond(m0, qml.X, false_fn=qml.I)(wires=[1])

            return qml.state()

        circuit = run_autograph(circuit) if autograph else circuit
        assert qml.math.allclose(circuit(), [0, 0, 0, 0, 0, 0, 0, 1])


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

        @qml.qnode(dev, static_argnums=3)
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
        circuit = run_autograph(circuit)
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
        call_count = spy.call_count

        # Cache hit because arguments are of same type/shape
        spy.reset_mock()
        self.check_execution_results(circuit, 100.1)
        assert spy.call_count == call_count - 1

        # Cache miss because arguments are not of same type/shape
        spy.reset_mock()
        self.check_execution_results(circuit, jnp.array(2))
        assert spy.call_count == call_count

        # Cache hit because arguments are of same type/shape
        spy.reset_mock()
        self.check_execution_results(circuit, 10)
        assert spy.call_count == call_count - 1

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
        call_count = spy.call_count

        # Cache hit because same arg shape/type and same kwarg
        spy.reset_mock()
        self.check_execution_results(circuit, 2.0, y=1)
        assert spy.call_count == call_count - 1

        # Cache miss because same arg shape/type but different kwarg
        spy.reset_mock()
        self.check_execution_results(circuit, 1.0, y=2)
        assert spy.call_count == call_count

        # Cache hit because same arg shape/type and same kwarg
        spy.reset_mock()
        self.check_execution_results(circuit, 2.0, y=2)
        assert spy.call_count == call_count - 1

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
        call_count = spy.call_count

        # Cache hit because same arg shape/types
        spy.reset_mock()
        self.check_execution_results(circuit, (jnp.array(5.1), 3.5))
        assert spy.call_count == call_count - 1

        # Cache miss because different arg shape/types
        spy.reset_mock()
        self.check_execution_results(circuit, (jnp.array(5.1), 3))
        assert spy.call_count == call_count

        # Cache hit because same arg shape/types
        spy.reset_mock()
        self.check_execution_results(circuit, (3.5, jnp.array(2)))
        assert spy.call_count == call_count - 1

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
        call_count = spy.call_count

        # Cache hit because same arg shape/type and same static arg
        spy.reset_mock()
        self.check_execution_results(circuit, 1.1, 2.0)
        assert spy.call_count == call_count - 1

        # Cache miss because same arg shape/type but different static arg
        spy.reset_mock()
        self.check_execution_results(circuit, 1.1, 2.1)
        assert spy.call_count == call_count

        # Cache hit because same arg shape/type and same static arg
        spy.reset_mock()
        self.check_execution_results(circuit, 1.0, 2.1)
        assert spy.call_count == call_count - 1

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
        call_count = spy.call_count

        # Cache hit because same arg shape/type and same static args
        spy.reset_mock()
        self.check_execution_results(circuit, (4.5, 5.5), (3.5, 4.5))
        assert spy.call_count == call_count - 1

        # Cache miss because same arg shape/type but different static args
        spy.reset_mock()
        self.check_execution_results(circuit, (4.5, 5.5), (3.1, 5.5))
        assert spy.call_count == call_count

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
    @pytest.mark.xfail  # think JAX 0.5.3 broke dynamic shapes and static argnums
    def test_caching_dynamic_shapes_and_static_argnums(self, mocker, enable_disable_dynamic_shapes):
        """Test that caching works correctly when a QNode has arguments with
        dynamic shapes as well as static arguments."""

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev, static_argnums=1)
        def circuit(x, y):
            qml.RX(jnp.sum(x), 0)
            qml.RY(y, 0)
            return qml.expval(qml.Z(0))

        abstracted_axes = ({0: "a"}, ())
        spy = mocker.spy(jax, "make_jaxpr")
        _ = jax.make_jaxpr(circuit, abstracted_axes=abstracted_axes, static_argnums=1)(
            jnp.arange(10), jnp.array(0.5)
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
    @pytest.mark.xfail  # think JAX 0.5.3 broke dynamic shapes and static argnums
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

        abstracted_axes = ({"0": {0: "a"}, "1": {0: "b"}}, ())
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
