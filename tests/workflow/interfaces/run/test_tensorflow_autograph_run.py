# Copyright 2018-2024 Xanadu Quantum Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the `run` helper function on the 'tf-autograph' interface"""

from dataclasses import replace

import numpy as np
import pytest

# pylint: disable=no-name-in-module
from conftest import atol_for_shots, get_device, test_matrix

import pennylane as qml
from pennylane.tape import QuantumScript
from pennylane.transforms.core import TransformProgram
from pennylane.workflow import _resolve_execution_config, _setup_transform_program, run

tf = pytest.importorskip("tensorflow")


@pytest.mark.tf
@pytest.mark.parametrize("device, config, shots", test_matrix)
class TestTFAutographRun:
    """Test the 'tensorflow-autograph' interface run function integrates well for both forward and backward execution"""

    def test_run(self, device, config, shots, seed):
        """Test execution of tapes on 'tensorflow' interface."""
        if config.gradient_method == "adjoint":
            pytest.xfail("Interface `tf-autograph` doesn't work with Adjoint differentiation.")

        device = get_device(device, seed=seed)
        config = replace(config, interface="tf-autograph")
        resolved_config = _resolve_execution_config(
            config, device, [QuantumScript()], TransformProgram()
        )

        def cost(a, b):
            ops1 = [qml.RY(a, wires=0), qml.RX(b, wires=0)]
            tape1 = qml.tape.QuantumScript(ops1, [qml.expval(qml.PauliZ(0))], shots=shots)

            ops2 = [qml.RY(a, wires="a"), qml.RX(b, wires="a")]
            tape2 = qml.tape.QuantumScript(ops2, [qml.expval(qml.PauliZ("a"))], shots=shots)

            resolved_config = _resolve_execution_config(
                config, device, [tape1, tape2], TransformProgram()
            )
            inner_tp = _setup_transform_program(TransformProgram(), device, resolved_config)[1]
            return run([tape1, tape2], device, resolved_config, inner_tp)

        a = tf.Variable(0.1, dtype="float64")
        b = tf.constant(0.2, dtype="float64")

        with device.tracker:
            res = cost(a, b)

        assert len(res) == 2

        if config.gradient_method == "adjoint" and not resolved_config.use_device_jacobian_product:
            assert device.tracker.totals["execute_and_derivative_batches"] == 1
        else:
            assert device.tracker.totals["batches"] == 1
        assert device.tracker.totals["executions"] == 2

        if not shots.has_partitioned_shots:
            assert res[0].shape == ()
            assert res[1].shape == ()

        assert qml.math.allclose(res[0], tf.cos(a) * tf.cos(b), atol=atol_for_shots(shots))
        assert qml.math.allclose(res[1], tf.cos(a) * tf.cos(b), atol=atol_for_shots(shots))

    def test_scalar_jacobian(self, device, config, shots, seed):
        """Test scalar jacobian calculation"""
        if shots.has_partitioned_shots:
            pytest.xfail(reason="Partitioned shots are not supported yet.")
        device = get_device(device, seed=seed)
        config = replace(config, interface="tf-autograph")
        resolved_config = _resolve_execution_config(
            config, device, [QuantumScript()], TransformProgram()
        )
        device_vjp = resolved_config.use_device_jacobian_product

        def cost(a):
            tape = qml.tape.QuantumScript([qml.RY(a, 0)], [qml.expval(qml.PauliZ(0))], shots=shots)
            resolved_config = _resolve_execution_config(config, device, [tape], TransformProgram())
            inner_tp = _setup_transform_program(TransformProgram(), device, resolved_config)[1]
            return run([tape], device, resolved_config, inner_tp)[0]

        a = tf.Variable(0.1, dtype=tf.float64)
        with tf.GradientTape(persistent=device_vjp) as tape:
            cost_res = cost(a)
        res = tape.jacobian(cost_res, a, experimental_use_pfor=not device_vjp)
        assert res.shape == ()  # pylint: disable=no-member

        expected = -qml.math.sin(a)
        assert expected.shape == ()
        assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)
        assert np.allclose(res, -tf.sin(a), atol=atol_for_shots(shots))

    def test_jacobian(self, device, config, shots, seed):
        """Test jacobian calculation"""
        if device == "param_shift.qubit":
            pytest.xfail(reason="Jacobian not support yet.")

        if shots.has_partitioned_shots:
            pytest.xfail(reason="Partitioned shots are not supported yet.")

        config = replace(config, interface="tf-autograph")
        device = get_device(device, seed=seed)
        resolved_config = _resolve_execution_config(
            config, device, [QuantumScript()], TransformProgram()
        )
        device_vjp = resolved_config.use_device_jacobian_product

        def cost(a, b):
            ops = [qml.RY(a, wires=0), qml.RX(b, wires=1), qml.CNOT(wires=[0, 1])]
            m = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))]
            tape = qml.tape.QuantumScript(ops, m, shots=shots)
            resolved_config = _resolve_execution_config(config, device, [tape], TransformProgram())
            inner_tp = _setup_transform_program(TransformProgram(), device, resolved_config)[1]
            return qml.math.hstack(
                run([tape], device, resolved_config, inner_tp)[0], like="tensorflow"
            )

        a = tf.Variable(0.1)
        b = tf.Variable(0.2)
        with tf.GradientTape(persistent=device_vjp) as tape:
            res = cost(a, b)
        expected = [tf.cos(a), -tf.cos(a) * tf.sin(b)]
        assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        jac = tape.jacobian(res, [a, b], experimental_use_pfor=not device_vjp)
        assert isinstance(jac, list) and len(jac) == 2
        assert jac[0].shape == (2,)
        assert jac[1].shape == (2,)

        expected = ([-tf.sin(a), tf.sin(a) * tf.sin(b)], [0, -tf.cos(a) * tf.cos(b)])
        for _r, _e in zip(jac, expected):
            assert np.allclose(_r, _e, atol=atol_for_shots(shots))
