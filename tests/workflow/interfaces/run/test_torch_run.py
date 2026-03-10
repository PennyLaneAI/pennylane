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

"""Unit tests for the `run` helper function on the 'torch' interface"""

from dataclasses import replace

import pytest

import pennylane as qml

# pylint: disable=no-name-in-module
from conftest import atol_for_shots, get_device, test_matrix
from pennylane.workflow import _resolve_execution_config, _setup_transform_program, run

torch = pytest.importorskip("torch")


@pytest.mark.torch
@pytest.mark.parametrize("device, config, shots", test_matrix)
class TestTorchRun:
    """Test the 'torch' interface run function integrates well for both forward and backward execution"""

    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(self):
        torch.set_default_dtype(torch.float64)
        yield
        torch.set_default_dtype(torch.float32)

    def test_run(self, device, config, shots, seed):
        """Test execution of tapes on 'torch' interface."""
        device = get_device(device, seed=seed)
        config = replace(config, interface="torch")

        def cost(a, b):
            ops1 = [qml.RY(a, wires=0), qml.RX(b, wires=0)]
            tape1 = qml.tape.QuantumScript(ops1, [qml.expval(qml.PauliZ(0))], shots=shots)

            ops2 = [qml.RY(a, wires="a"), qml.RX(b, wires="a")]
            tape2 = qml.tape.QuantumScript(ops2, [qml.expval(qml.PauliZ("a"))], shots=shots)

            resolved_config = _resolve_execution_config(config, device, [tape1, tape2])
            inner_tp = _setup_transform_program(device, resolved_config)[1]
            return run([tape1, tape2], device, resolved_config, inner_tp)

        a = torch.tensor(0.1, requires_grad=True)
        b = torch.tensor(0.2, requires_grad=False)

        with device.tracker:
            res = cost(a, b)

        assert len(res) == 2

        if getattr(config, "grad_on_execution", False):
            assert device.tracker.totals["execute_and_derivate_batches"] == 1
        else:
            assert device.tracker.totals["batches"] == 1
        assert device.tracker.totals["executions"] == 2

        if not shots.has_partitioned_shots:
            assert res[0].shape == ()
            assert res[1].shape == ()

        exp = torch.cos(a) * torch.cos(b)
        if shots.has_partitioned_shots:
            for shot in range(2):
                for wire in range(2):
                    assert qml.math.allclose(res[shot][wire], exp, atol=atol_for_shots(shots))
        else:
            for wire in range(2):
                assert qml.math.allclose(res[wire], exp, atol=atol_for_shots(shots))

    def test_scalar_jacobian(self, device, config, shots, seed):
        """Test scalar jacobian calculation"""
        device = get_device(device, seed)
        config = replace(config, interface="torch")

        def cost(a):
            tape = qml.tape.QuantumScript([qml.RY(a, 0)], [qml.expval(qml.PauliZ(0))], shots=shots)
            resolved_config = _resolve_execution_config(config, device, [tape])
            _, inner_tp = _setup_transform_program(device, resolved_config)
            return run([tape], device, resolved_config, inner_tp)[0]

        a = torch.tensor(0.1, requires_grad=True)

        res = torch.autograd.functional.jacobian(cost, a)
        if not shots.has_partitioned_shots:
            assert res.shape == ()  # pylint: disable=no-member

        expected = -qml.math.sin(a)

        assert expected.shape == ()
        if shots.has_partitioned_shots:
            for i in range(shots.num_copies):
                assert torch.allclose(res[i], expected, atol=atol_for_shots(shots), rtol=0)
                assert torch.allclose(res[i], -torch.sin(a), atol=atol_for_shots(shots))
        else:
            assert torch.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)
            assert torch.allclose(res, -torch.sin(a), atol=atol_for_shots(shots))

    def test_jacobian(self, device, config, shots, seed):
        """Test jacobian calculation"""
        device = get_device(device, seed)
        config = replace(config, interface="torch")

        def cost(a, b):
            ops = [qml.RY(a, wires=0), qml.RX(b, wires=1), qml.CNOT(wires=[0, 1])]
            m = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))]
            tape = qml.tape.QuantumScript(ops, m, shots=shots)

            resolved_config = _resolve_execution_config(config, device, [tape])
            _, inner_tp = _setup_transform_program(device, resolved_config)
            res = run([tape], device, resolved_config, inner_tp)[0]

            if shots.has_partitioned_shots:
                return torch.hstack(res[0] + res[1])

            return torch.hstack(res)

        a = torch.tensor(0.1, requires_grad=True)
        b = torch.tensor(0.2, requires_grad=True)

        res = cost(a, b)
        expected = torch.tensor([torch.cos(a), -torch.cos(a) * torch.sin(b)])
        if shots.has_partitioned_shots:
            assert torch.allclose(res[:2], expected, atol=atol_for_shots(shots), rtol=0)
            assert torch.allclose(res[2:], expected, atol=atol_for_shots(shots), rtol=0)
        else:
            assert torch.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        res = torch.autograd.functional.jacobian(cost, (a, b))
        assert isinstance(res, tuple) and len(res) == 2

        expected = (
            torch.tensor([-torch.sin(a), torch.sin(a) * torch.sin(b)]),
            torch.tensor([0, -torch.cos(a) * torch.cos(b)]),
        )
        if shots.has_partitioned_shots:
            assert res[0].shape == (4,)
            assert res[1].shape == (4,)

            for _r, _e in zip(res, expected):
                assert torch.allclose(_r[:2], _e, atol=atol_for_shots(shots))
                assert torch.allclose(_r[2:], _e, atol=atol_for_shots(shots))

        else:
            assert res[0].shape == (2,)
            assert res[1].shape == (2,)

            for _r, _e in zip(res, expected):
                assert torch.allclose(_r, _e, atol=atol_for_shots(shots))
