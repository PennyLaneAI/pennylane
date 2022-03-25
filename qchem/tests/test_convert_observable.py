import os

import numpy as np
import pytest

import pennylane as qml
from pennylane import qchem


@pytest.mark.parametrize(
    ("name", "core", "active", "mapping", "expected_cost"),
    [
        ("lih", [0], [1, 2], "jordan_WIGNER", -7.255500051039507),
        ("lih", [0], [1, 2], "BRAVYI_kitaev", -7.246409364088741),
        ("h2_pyscf", list(range(0)), list(range(2)), "jordan_WIGNER", 0.19364907363263958),
        ("h2_pyscf", list(range(0)), list(range(2)), "BRAVYI_kitaev", 0.16518000728327564),
        ("gdb3", list(range(11)), [11, 12], "jordan_WIGNER", -130.59816885313248),
        ("gdb3", list(range(11)), [11, 12], "BRAVYI_kitaev", -130.6156540164148),
    ],
)
def test_integration_mol_file_to_vqe_cost(
    name, core, active, mapping, expected_cost, custom_wires, tol
):
    r"""Test if the output of `decompose()` works with `import_operator()`
    to generate `ExpvalCost()`"""
    ref_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_ref_files")
    hf_file = os.path.join(ref_dir, name)
    qubit_hamiltonian = qchem.decompose(
        hf_file,
        mapping=mapping,
        core=core,
        active=active,
    )

    vqe_hamiltonian = qml.import_operator(
        qubit_hamiltonian, wires=custom_wires, format="openfermion"
    )
    assert len(vqe_hamiltonian.ops) > 1  # just to check if this runs

    num_qubits = len(vqe_hamiltonian.wires)
    assert num_qubits == 2 * len(active)

    if custom_wires is None:
        wires = num_qubits
    elif isinstance(custom_wires, dict):
        wires = qml.convert._process_wires(custom_wires)
    else:
        wires = custom_wires[:num_qubits]
    dev = qml.device("default.qubit", wires=wires)

    # can replace the ansatz with more suitable ones later.
    def dummy_ansatz(phis, wires):
        for phi, w in zip(phis, wires):
            qml.RX(phi, wires=w)

    phis = np.load(os.path.join(ref_dir, "dummy_ansatz_parameters.npy"))

    dummy_cost = qml.ExpvalCost(dummy_ansatz, vqe_hamiltonian, dev)
    res = dummy_cost(phis)

    assert np.abs(res - expected_cost) < tol["atol"]
