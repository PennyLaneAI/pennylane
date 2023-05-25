from typing import ClassVar, List, Tuple, TypedDict

import numpy as np
import scipy.sparse

import pennylane as qml
from pennylane.data.base import Dataset, attribute
from pennylane.data.base.dataset import attribute
from pennylane.operation import Operation, Operator
from pennylane.qchem import Molecule

QWCGroupingData = Tuple[List[np.ndarray], List[List[Operator]], List[List[Operator]]]


class QChemDatasetParameters(TypedDict):
    molname: str
    basis: str
    bondlength: str


class QChemDataset(Dataset[QChemDatasetParameters]):
    """A Dataset representing a chemical system."""

    category_id: ClassVar[str] = "qchem"

    molecule: Molecule = attribute(
        doc="PennyLane Molecule object containing description for the system and basis set"
    )
    hf_state: np.ndarray

    # Hamiltonian data
    hamiltonian: qml.Hamiltonian
    sparse_hamiltonian: scipy.sparse.csr_array
    meas_groupings: QWCGroupingData
    fci_energy: float
    fci_spectrum: np.ndarray

    # Auxillary observables
    dipole_op: List[qml.Hamiltonian]
    number_op: qml.Hamiltonian
    spin2_op: qml.Hamiltonian
    spinz_op: qml.Hamiltonian

    # Feature: Tapering data
    symmetries: List[qml.Hamiltonian]
    paulix_ops: List[qml.PauliX]
    optimal_sector: np.ndarray

    # Feature: Tapered observables data
    tapered_hamiltonian: qml.Hamiltonian
    tapered_hf_state: np.ndarray
    tapered_dipole_op: List[qml.Hamiltonian]
    tapered_num_op: qml.Hamiltonian
    tapered_spin2_op: qml.Hamiltonian
    tapered_spinz_op: qml.Hamiltonian

    # VQE data
    vqe_gates: List[Operation]
    vqe_params: np.ndarray
    vqe_energy: float
