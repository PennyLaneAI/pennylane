# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The data subpackage provides functionality to access, store and manipulate quantum datasets.

Datasets are generally stored and accessed using the :class:`~pennylane.data.Dataset` class.
Pre-computed datasets are available for download and can be accessed using the :func:`~pennylane.data.load` or
:func:`~pennylane.data.load_interactive` functions.
Additionally, users can easily create, write to disk, and read custom datasets using functions within the
:class:`~pennylane.data.Dataset` class.

.. currentmodule:: pennylane.data
.. autosummary::
   :toctree: api

Description
-----------

Datasets
~~~~~~~~
The :class:`Dataset` class provides a portable storage format for information describing a physical
system and its evolution. For example, a dataset for an arbitrary quantum system could have
a Hamiltonian, its ground state, and an efficient state-preparation circuit for that state. Datasets
can contain a range of object types, including:

    - ``numpy.ndarray``
    - any numeric type
    - :class:`qml.qchem.Molecule`
    - most :class:`qml.operation.Operator` types
    - `list` of any supported type
    - `dict` of any supported type, as long as the keys are strings


Creating a Dataset
~~~~~~~~~~~~~~~~~~

To create a new dataset in-memory, initialize a new ``Dataset`` with the desired attributes:

    >>> hamiltonian = qml.Hamiltonian([1., 1.], [qml.PauliZ(wires=0), qml.PauliZ(wires=1)])
    >>> eigvals, eigvecs = np.linalg.eigh(qml.matrix(hamiltonian))
    >>> dataset = qml.data.Dataset(
            hamiltonian = hamiltonian,
            eigen = {"eigvals": eigvals, "eigvecs": eigvecs})
    >>> dataset.hamiltonian
    <Hamiltonian: terms=2, wires=[0, 1]>
    >>> dataset.eigen
    {'eigvals': array([-2.,  0.,  0.,  2.]), 
    'eigvecs': array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])}

Attributes can also be assigned to the instance after creation:

    >>> dataset.ground_state = np.transpose(eigvecs)[np.argmin(eigvals)]
    >>> dataset.ground_state
    array([0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j])


Attribute Metadata
~~~~~~~~~~~~~~~~~~

Dataset attributes can also contain additional metadata, such as docstrings. The :func:`qml.data.attribute`
function can be used to attach metadata on assignment or initialization.

    >>> hamiltonian = qml.Hamiltonian([1., 1.], [qml.PauliZ(wires=0), qml.PauliZ(wires=1)])
    >>> eigvals, eigvecs = np.linalg.eigh(qml.matrix(hamiltonian))
    >>> dataset = qml.data.Dataset(hamiltonian = qml.data.attribute(
            hamiltonian, 
            doc="The hamiltonian of the system"))
    >>> dataset.eigen = attribute(
            {"eigvals": eigvals, "eigvecs": eigvecs}, 
            doc="Eigenvalues and eigenvectors of the hamiltonain")
    
This metadata can then be accessed using the :meth:`Dataset.attr_info` mapping:

    >>> dataset.attr_info["hamiltonian"]["doc"]
    'The hamiltonian of the system'

"""

from .attributes import (
    DatasetArray,
    DatasetDict,
    DatasetJSON,
    DatasetList,
    DatasetMolecule,
    DatasetNone,
    DatasetOperator,
    DatasetScalar,
    DatasetSparseArray,
    DatasetString,
)
from .base.attribute import AttributeInfo, DatasetAttribute, attribute
from .base.dataset import Dataset, field
from .data_manager import (
    DEFAULT,
    FULL,
    list_attributes,
    list_datasets,
    load,
    load_interactive,
)

__all__ = (
    "DatasetAttribute",
    "AttributeInfo",
    "attribute",
    "field",
    "Dataset",
    "DatasetArray",
    "DatasetScalar",
    "DatasetString",
    "DatasetList",
    "DatasetDict",
    "DatasetOperator",
    "DatasetNone",
    "DatasetMolecule",
    "DatasetSparseArray",
    "DatasetJSON",
    "load",
    "load_interactive",
    "list_attributes",
    "list_datasets",
    "DEFAULT",
    "FULL",
)
