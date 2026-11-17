# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Unit tests for the ``AdaptiveOptimizer``.
"""

import numpy as np
import pytest

import pennylane as qp
from pennylane import numpy as pnp

symbols = ["H", "H", "H"]
geometry = pnp.array(
    [[0.01076341, 0.04449877, 0.0], [0.98729513, 1.63059094, 0.0], [1.87262415, -0.00815842, 0.0]],
    requires_grad=False,
)
H, qubits = qp.qchem.molecular_hamiltonian(symbols, geometry, charge=1)

hf_state = np.array([1, 1, 0, 0, 0, 0])
dev = qp.device("default.qubit", wires=qubits)

energy_h3p_hf = -1.2465499384199534


@qp.transforms.convert_to_numpy_parameters
@qp.qnode(dev, interface="numpy")
def initial_circuit():
    qp.BasisState(hf_state, wires=range(qubits))
    return qp.expval(H)


@pytest.mark.parametrize(
    "circuit, params, gates, energy_ref",
    [
        (
            initial_circuit,
            np.array([0.0]),
            [qp.DoubleExcitation(0.0, [0, 1, 2, 3])],
            energy_h3p_hf,
        ),
    ],
)
def test_private_circuit(circuit, params, gates, energy_ref):
    r"""Test that _circuit returns the correct output."""
    private_circuit = qp.AdaptiveOptimizer._circuit  # pylint: disable=protected-access
    qnode = qp.QNode(private_circuit, dev, interface="numpy")
    qnode = qp.transforms.convert_to_numpy_parameters(qnode)
    energy = qnode(params, gates, circuit.func)
    assert np.allclose(energy, energy_ref)


@pytest.mark.parametrize(
    "circuit",
    [initial_circuit],
)
def test_append_gate(circuit):
    """Test that append_gate properly adds a gate to a circuit."""

    param = np.array([0.0])
    gate = qp.DoubleExcitation(np.array(0.0), wires=[0, 1, 2, 3])

    final_circuit = qp.optimize.adaptive.append_gate(circuit.func, param, [gate])
    qnode = qp.QNode(final_circuit, dev)
    tape = qp.workflow.construct_tape(qnode)()
    qp.assert_equal(tape.operations[-1], gate)

    final_circuit, fn = qp.optimize.adaptive.append_gate(tape, param, [gate])

    assert isinstance(final_circuit, list)
    assert isinstance(fn(final_circuit), qp.tape.QuantumScript)


class TestGateSelection:
    """Test circuit growth with deterministic data at the quantum derivative boundary."""

    @staticmethod
    def _qnode():
        @qp.transforms.convert_to_numpy_parameters
        @qp.qnode(qp.device("default.qubit", wires=2), interface="numpy", diff_method=None)
        def circuit():
            qp.RY(0.2, 0)
            return qp.expval(qp.Z(0))

        return circuit

    @pytest.mark.parametrize("params_zero", [True, False])
    @pytest.mark.parametrize(
        "drain_pool, gradients, selected_index",
        [(False, [-4.0, -2.0, 0.5], 0), (True, [-2.0, 0.5], 1)],
    )
    def test_selection_and_parameter_updates(
        self, mocker, params_zero, drain_pool, gradients, selected_index
    ):
        """Drain by gate name and wires, select by magnitude, and optimize the selected seed."""
        circuit = self._qnode()
        original_func = circuit.func
        pool = [qp.RY(0.4, 0), qp.RY(0.6, 1), qp.RX(0.8, 0)]
        selected_gate = pool[selected_index]
        pool_gradient = mocker.Mock(return_value=np.array(gradients))
        gradient_factory = mocker.patch(
            "pennylane.optimize.adaptive.grad", return_value=pool_gradient
        )
        parameter_gradient = mocker.Mock(
            side_effect=[np.array([0.5]), np.array([-0.25])], forward=None
        )
        mocker.patch(
            "pennylane.optimize.gradient_descent.get_gradient", return_value=parameter_gradient
        )
        opt = qp.AdaptiveOptimizer(param_steps=2, stepsize=0.2)

        grown, cost, largest_gradient = opt.step_and_cost(
            circuit, pool, drain_pool=drain_pool, params_zero=params_zero
        )

        gradient_factory.assert_called_once_with(grown)
        pool_gradient.assert_called_once()
        candidates = pool[1:] if drain_pool else pool
        assert pool_gradient.call_args.kwargs["gates"] == candidates
        assert pool_gradient.call_args.kwargs["initial_circuit"] is original_func
        assert np.allclose(
            pool_gradient.call_args.args[0], [gate.parameters[0] for gate in candidates]
        )
        seed = 0.0 if params_zero else selected_gate.parameters[0]
        assert parameter_gradient.call_count == 2
        assert np.allclose(parameter_gradient.call_args_list[0].args[0], [seed])
        assert np.allclose(parameter_gradient.call_args_list[1].args[0], [seed - 0.1])
        for call in parameter_gradient.call_args_list:
            assert call.kwargs["gates"] == [selected_gate]
            assert call.kwargs["initial_circuit"] is original_func

        tape = qp.workflow.construct_tape(grown)()
        qp.assert_equal(tape.operations[0], qp.RY(0.2, 0))
        qp.assert_equal(tape.operations[-1], type(selected_gate)(seed - 0.05, selected_gate.wires))
        assert np.allclose(cost, np.cos(0.2))
        assert largest_gradient == (2.0 if drain_pool else 4.0)
        assert grown is not circuit
        assert grown.device is circuit.device
        assert grown.interface == circuit.interface == "numpy"
        assert grown.diff_method is circuit.diff_method is None
        assert circuit.func is original_func
        assert len(qp.workflow.construct_tape(circuit)().operations) == 1
        assert [gate.parameters[0] for gate in pool] == [0.4, 0.6, 0.8]

    @pytest.mark.parametrize("params_zero", [True, False])
    def test_step_returns_grown_qnode(self, mocker, params_zero):
        """The step wrapper preserves the requested parameter initialization."""
        circuit = self._qnode()
        pool = [qp.RX(0.7, 0)]
        mocker.patch(
            "pennylane.optimize.adaptive.grad",
            return_value=mocker.Mock(return_value=np.array([0.5])),
        )
        opt = qp.AdaptiveOptimizer(param_steps=0)

        grown = opt.step(circuit, pool, params_zero=params_zero)

        tape = qp.workflow.construct_tape(grown)()
        qp.assert_equal(tape.operations[-1], qp.RX(0.0 if params_zero else 0.7, 0))
        assert np.allclose(grown(), np.cos(0.2) * np.cos(0.0 if params_zero else 0.7))
