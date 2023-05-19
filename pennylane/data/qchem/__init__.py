from pennylane.data.base import Dataset, attribute
from pennylane.data.attributes import DatasetOperatorList
from pennylane.qchem import Molecule
from pennylane.data.base.typing_util import HDF5Group
import pennylane as qml
from pennylane.operation import Operator
import scipy.sparse
import numpy as np
from typing import List, MutableMapping, Tuple


QWCGroupingData = Tuple[List[np.ndarray], List[List[Operator]], List[List[Operator]]]

class QChemDataset(Dataset):
    """A Dataset representing a chemical system."""

    molecule: Molecule = attribute(doc="PennyLane Molecule object containing description for the system and basis set")
    hf_state: np.ndarray

    hamiltonian: qml.Hamiltonian
    sparse_hamiltonian: scipy.sparse.csr_array
    meas_groupings: QWCGroupingData
    fci_energy: float
    fci_spectrum: np.ndarray

    dipole_op: qml.Hamiltonian
    number_op: qml.Hamiltonian
    spin2_op: qml.Hamiltonian
    spinz_op: qml.Hamiltonian

    symmetries: List[qml.Hamiltonian]
    paulix_ops: DatasetOperatorList[]
    optimal_sector: np.ndarray
    tap
