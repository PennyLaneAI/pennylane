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
This test file performs system-level tests with a PennyLane workload against Lightning, both with and without Catalyst.
The workload is performing a single VQE step using molecules from the datasets, and hits the following parts of the pipeline:

* Device creation: "lightning.qubit"
* Loading molecules from the PennyLane datasets with various basis sets: {H2, HeH+, H3+, He2}
* Execution of a templated circuit with and without JITing for expval(H)
* Support for multiple gradient modes: diff_method:={"best", "adjoint", "parameter-shift"}
* Support for correctness with Lightning observable batching: batch_obs:={False, True}
* Support (where capable) for shots with gradients: shots:={None, 1000}
* Support for energy minimization with gradients
"""

from functools import partial

import pytest

import pennylane as qml
from pennylane import numpy as np

pytestmark = [pytest.mark.catalyst, pytest.mark.external, pytest.mark.system, pytest.mark.slow]

catalyst = pytest.importorskip("catalyst")
optax = pytest.importorskip("optax")
jax = pytest.importorskip("jax")

mols_basis_sets = [
    ["H2", "STO-3G"],  # 4 / 15
    ["HeH+", "STO-3G"],  # 4 / 27
    ["H3+", "STO-3G"],  # 6 / 66
    ["He2", "6-31G"],  # 8 / 181
    ["H2", "6-31G"],  # 8 / 185
]


@pytest.mark.parametrize("mol, basis_set", mols_basis_sets)
@pytest.mark.parametrize(
    "diff_method, batch_obs, shots",
    [
        ("best", False, None),
        ("adjoint", False, None),
        ("adjoint", True, None),
        ("parameter-shift", False, None),
        ("parameter-shift", False, 1000),
    ],
)
def test_workload_VQE(mol, basis_set, diff_method, batch_obs, shots):

    dataset = qml.data.load("qchem", molname=mol, basis=basis_set)[0]
    ham, _ = dataset.hamiltonian, len(dataset.hamiltonian.wires)
    hf_state = dataset.hf_state
    ham = dataset.hamiltonian
    wires = ham.wires
    dev = qml.device("lightning.qubit", wires=wires, batch_obs=batch_obs, shots=shots)

    n_electrons = dataset.molecule.n_electrons

    singles, doubles = qml.qchem.excitations(n_electrons, len(wires))

    @qml.qnode(dev, diff_method=diff_method)
    def cost(weights):
        qml.templates.AllSinglesDoubles(weights, wires, hf_state, singles, doubles)
        return qml.expval(ham)

    np.random.seed(42)
    params = np.random.normal(0, np.pi, len(singles) + len(doubles))

    def exec_non_catalyst():
        opt = qml.GradientDescentOptimizer(stepsize=0.2)
        new_params, energy = opt.step_and_cost(cost, params)

        # Asserting execution without error, and for an energy drop
        assert cost(new_params) < energy

    def exec_catalyst():
        opt = optax.adam(learning_rate=0.2)
        cost_jit = qml.qjit(cost)

        @qml.qjit
        def update_step(params, opt_state):
            grads = catalyst.grad(cost_jit, method="auto")(params)
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            return (params, opt_state)

        local_params = jax.numpy.array(params)
        energy = cost(local_params)
        opt_state = opt.init(local_params)
        new_params, opt_state = update_step(local_params, opt_state)

        # Asserting execution without error, and for an energy drop
        assert cost(new_params) < energy

    exec_non_catalyst()
    exec_catalyst()
