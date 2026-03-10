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

"""Unit tests for the `run` helper function on the 'autograd' interface"""

from dataclasses import replace

import autograd
import numpy as np
import pytest

import pennylane as qml

# pylint: disable=no-name-in-module
from conftest import atol_for_shots, get_device, test_matrix
from pennylane import numpy as pnp
from pennylane.workflow import _resolve_execution_config, _setup_transform_program, run


@pytest.mark.autograd
@pytest.mark.parametrize("device, config, shots", test_matrix)
class TestAutogradRun:
    """Test the 'autograd' interface run function integrates well for both forward and backward execution"""

    def test_run(self, device, config, shots, seed):
        """Test execution of tapes on 'autograd' interface."""
        device = get_device(device, seed=seed)
        config = replace(config, interface="autograd")

        def cost(a, b):
            ops1 = [qml.RY(a, wires=0), qml.RX(b, wires=0)]
            tape1 = qml.tape.QuantumScript(ops1, [qml.expval(qml.PauliZ(0))], shots=shots)

            ops2 = [qml.RY(a, wires="a"), qml.RX(b, wires="a")]
            tape2 = qml.tape.QuantumScript(ops2, [qml.expval(qml.PauliZ("a"))], shots=shots)

            resolved_config = _resolve_execution_config(config, device, [tape1, tape2])
            inner_tp = _setup_transform_program(device, resolved_config)[1]
            return run([tape1, tape2], device, resolved_config, inner_tp)

        a = pnp.array(0.1, requires_grad=True)
        b = pnp.array(0.2, requires_grad=False)

        with device.tracker:
            res = cost(a, b)

        assert len(res) == 2

        if config.grad_on_execution:
            assert device.tracker.totals["execute_and_derivate_batches"] == 1
        else:
            assert device.tracker.totals["batches"] == 1
        assert device.tracker.totals["executions"] == 2

        if not shots.has_partitioned_shots:
            assert res[0].shape == ()
            assert res[1].shape == ()

        assert qml.math.allclose(res[0], np.cos(a) * np.cos(b), atol=atol_for_shots(shots))
        assert qml.math.allclose(res[1], np.cos(a) * np.cos(b), atol=atol_for_shots(shots))

    def test_scalar_jacobian(self, device, config, shots, seed):
        """Test scalar jacobian calculation"""
        device = get_device(device, seed=seed)
        config = replace(config, interface="autograd")

        def cost(a):
            tape = qml.tape.QuantumScript([qml.RY(a, 0)], [qml.expval(qml.PauliZ(0))], shots=shots)
            resolved_config = _resolve_execution_config(config, device, [tape])
            inner_tp = _setup_transform_program(device, resolved_config)[1]
            return run([tape], device, resolved_config, inner_tp)[0]

        a = pnp.array(0.1, requires_grad=True)
        if shots.has_partitioned_shots:
            res = qml.jacobian(lambda x: qml.math.hstack(cost(x)))(a)
        else:
            res = qml.jacobian(cost)(a)
            assert res.shape == ()  # pylint: disable=no-member

        expected = -qml.math.sin(a)

        assert expected.shape == ()
        assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)
        assert np.allclose(res, -np.sin(a), atol=atol_for_shots(shots))

    def test_jacobian(self, device, config, shots, seed):
        """Test jacobian calculation"""
        device = get_device(device, seed=seed)
        config = replace(config, interface="autograd")

        def cost(a, b):
            ops = [qml.RY(a, wires=0), qml.RX(b, wires=1), qml.CNOT(wires=[0, 1])]
            m = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))]
            tape = qml.tape.QuantumScript(ops, m, shots=shots)
            resolved_config = _resolve_execution_config(config, device, [tape])
            inner_tp = _setup_transform_program(device, resolved_config)[1]
            return autograd.numpy.hstack(run([tape], device, resolved_config, inner_tp)[0])

        a = pnp.array(0.1, requires_grad=True)
        b = pnp.array(0.2, requires_grad=True)
        res = cost(a, b)
        expected = [np.cos(a), -np.cos(a) * np.sin(b)]
        if shots.has_partitioned_shots:
            assert np.allclose(res[:2], expected, atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(res[2:], expected, atol=atol_for_shots(shots), rtol=0)
        else:
            assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        res = qml.jacobian(cost)(a, b)
        assert isinstance(res, tuple) and len(res) == 2
        if shots.has_partitioned_shots:
            assert res[0].shape == (4,)
            assert res[1].shape == (4,)

            expected = ([-np.sin(a), np.sin(a) * np.sin(b)], [0, -np.cos(a) * np.cos(b)])
            for _r, _e in zip(res, expected):
                assert np.allclose(_r[:2], _e, atol=atol_for_shots(shots))
                assert np.allclose(_r[2:], _e, atol=atol_for_shots(shots))
        else:
            assert res[0].shape == (2,)
            assert res[1].shape == (2,)

            expected = ([-np.sin(a), np.sin(a) * np.sin(b)], [0, -np.cos(a) * np.cos(b)])
            for _r, _e in zip(res, expected):
                assert np.allclose(_r, _e, atol=atol_for_shots(shots))
