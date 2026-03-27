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

"""Unit tests for the `run` helper function on the 'jax' interface"""

from dataclasses import replace

import numpy as np
import pytest

import pennylane as qml

# pylint: disable=no-name-in-module
from conftest import atol_for_shots, get_device, test_matrix
from pennylane.workflow import _resolve_execution_config, _setup_transform_program, run

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)


@pytest.mark.jax
@pytest.mark.parametrize("device, config, shots", test_matrix)
class TestJaxRun:
    """Test the 'jax' interface run function integrates well for both forward and backward execution"""

    def test_run(self, device, config, shots, seed):
        """Test execution of tapes on 'jax' interface."""
        device = get_device(device, seed=seed)
        config = replace(config, interface="jax")

        def cost(a, b):
            ops1 = [qml.RY(a, wires=0), qml.RX(b, wires=0)]
            tape1 = qml.tape.QuantumScript(ops1, [qml.expval(qml.PauliZ(0))], shots=shots)

            ops2 = [qml.RY(a, wires="a"), qml.RX(b, wires="a")]
            tape2 = qml.tape.QuantumScript(ops2, [qml.expval(qml.PauliZ("a"))], shots=shots)

            resolved_config = _resolve_execution_config(config, device, [tape1, tape2])
            inner_tp = _setup_transform_program(device, resolved_config)[1]
            return run([tape1, tape2], device, resolved_config, inner_tp)

        a = jnp.array(0.1)
        b = np.array(0.2)

        with device.tracker:
            res = cost(a, b)

        assert len(res) == 2

        if getattr(config, "gradient_method", None) == "adjoint":
            assert device.tracker.totals.get("execute_and_derivative_batches", 0) == 0
        else:
            assert device.tracker.totals["batches"] == 1
        assert device.tracker.totals["executions"] == 2

        if not shots.has_partitioned_shots:
            assert res[0].shape == ()
            assert res[1].shape == ()

        assert qml.math.allclose(res[0], jnp.cos(a) * jnp.cos(b), atol=atol_for_shots(shots))
        assert qml.math.allclose(res[1], jnp.cos(a) * jnp.cos(b), atol=atol_for_shots(shots))

    def test_scalar_jacobian(self, device, config, shots, seed):
        """Test scalar jacobian calculation"""
        device = get_device(device, seed)
        config = replace(config, interface="jax")

        def cost(a):
            tape = qml.tape.QuantumScript([qml.RY(a, 0)], [qml.expval(qml.PauliZ(0))], shots=shots)
            resolved_config = _resolve_execution_config(config, device, [tape])
            inner_tp = _setup_transform_program(device, resolved_config)[1]
            return run([tape], device, resolved_config, inner_tp)[0]

        a = jnp.array(0.1)
        res = jax.jacobian(cost)(a)
        if not shots.has_partitioned_shots:
            assert res.shape == ()  # pylint: disable=no-member

        expected = -qml.math.sin(a)

        assert expected.shape == ()
        assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)
        assert np.allclose(res, -jnp.sin(a), atol=atol_for_shots(shots))

    def test_jacobian(self, device, config, shots, seed):
        """Test jacobian calculation"""
        device = get_device(device, seed)
        config = replace(config, interface="jax")

        def cost(a, b):
            ops = [qml.RY(a, wires=0), qml.RX(b, wires=1), qml.CNOT(wires=[0, 1])]
            m = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))]
            tape = qml.tape.QuantumScript(ops, m, shots=shots)

            resolved_config = _resolve_execution_config(config, device, [tape])
            inner_tp = _setup_transform_program(device, resolved_config)[1]
            return run([tape], device, resolved_config, inner_tp)[0]

        a = jnp.array(0.1)
        b = jnp.array(0.2)
        res = cost(a, b)

        expected = [jnp.cos(a), -jnp.cos(a) * jnp.sin(b)]
        if shots.has_partitioned_shots:
            assert np.allclose(res[0], expected, atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(res[1], expected, atol=2 * atol_for_shots(shots), rtol=0)
        else:
            assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        g = jax.jacobian(cost, argnums=[0, 1])(a, b)
        assert isinstance(g, tuple) and len(g) == 2

        expected = ([-jnp.sin(a), jnp.sin(a) * jnp.sin(b)], [0, -jnp.cos(a) * jnp.cos(b)])

        if shots.has_partitioned_shots:
            for i in (0, 1):
                assert np.allclose(g[i][0][0], expected[0][0], atol=atol_for_shots(shots), rtol=0)
                assert np.allclose(g[i][1][0], expected[0][1], atol=atol_for_shots(shots), rtol=0)
                assert np.allclose(g[i][0][1], expected[1][0], atol=atol_for_shots(shots), rtol=0)
                assert np.allclose(g[i][1][1], expected[1][1], atol=atol_for_shots(shots), rtol=0)
        else:
            assert np.allclose(g[0][0], expected[0][0], atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(g[1][0], expected[0][1], atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(g[0][1], expected[1][0], atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(g[1][1], expected[1][1], atol=atol_for_shots(shots), rtol=0)
