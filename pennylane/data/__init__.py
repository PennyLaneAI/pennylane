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
"""The data subpackage provides functionality to access, store and manipulate `quantum datasets <https://pennylane.ai/datasets>`_.

.. note::

    For more details on using datasets, please see the
    :doc:`quantum datasets quickstart guide </introduction/data>`.

Overview
--------

Datasets are generally stored and accessed using the :class:`~pennylane.data.Dataset` class.
Pre-computed datasets are available for download and can be accessed using the :func:`~pennylane.data.load` or
:func:`~pennylane.data.load_interactive` functions.
Additionally, users can easily create, write to disk, and read custom datasets using functions within the
:class:`~pennylane.data.Dataset` class.

.. autosummary::
    :toctree: api

    attribute
    field
    Dataset
    DatasetNotWriteableError
    load
    load_interactive
    list_attributes
    list_datasets

In addition, various dataset types are provided

.. autosummary::
    :toctree: api

    AttributeInfo
    DatasetAttribute
    DatasetArray
    DatasetScalar
    DatasetString
    DatasetList
    DatasetDict
    DatasetOperator
    DatasetNone
    DatasetMolecule
    DatasetSparseArray
    DatasetJSON
    DatasetTuple

Datasets
--------

The :class:`~.Dataset` class provides a portable storage format for information describing a physical
system and its evolution. For example, a dataset for an arbitrary quantum system could have
a Hamiltonian, its ground state, and an efficient state-preparation circuit for that state. Datasets
can contain a range of object types, including:

- ``numpy.ndarray``
- any numeric type
- :class:`~.qchem.Molecule`
- most :class:`~.Operator` types
- ``list`` of any supported type
- ``dict`` of any supported type, as long as the keys are strings


For more details on using datasets, please see the
:doc:`quantum datasets quickstart guide </introduction/data>`.

Creating a Dataset
------------------

To create a new dataset in-memory, initialize a new :class:`~.Dataset` with the desired attributes:

>>> hamiltonian = qml.Hamiltonian([1., 1.], [qml.Z(0), qml.Z(1)])
>>> eigvals, eigvecs = np.linalg.eigh(qml.matrix(hamiltonian))
>>> dataset = qml.data.Dataset(
...   hamiltonian = hamiltonian,
...   eigen = {"eigvals": eigvals, "eigvecs": eigvecs}
... )
>>> dataset.hamiltonian
1.0 * Z(0) + 1.0 * Z(1)
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


Reading and Writing Datasets
----------------------------

Datasets can be saved to disk for later use. Datasets use the HDF5 format for serialization,
which uses the '.h5' file extension.

To save a dataset, use the :meth:`Dataset.write()` method:

>>> my_dataset = Dataset(...)
>>> my_dataset.write("~/datasets/my_dataset.h5")

To open a dataset from a file, use :meth:`Dataset.open()` class method:

>>> my_dataset = Dataset.open("~/datasets/my_dataset.h5", mode="r")

The ``mode`` argument follow the standard library convention --- ``r`` for
reading, ``w-`` and ``w`` for create and overwrite, and 'a' for editing.
``open()`` can be used to create a new dataset directly on disk:

>>> new_dataset = Dataset.open("~/datasets/new_datasets.h5", mode="w")

By default, any changes made to an opened dataset will be committed directly to the file, which will fail
if the file is opened read-only. The ``"copy"`` mode can be used to load the dataset into memory and detach
it from the file:

>>> my_dataset = Dataset.open("~/dataset/my_dataset/h5", mode="copy")
>>> my_dataset.new_attribute = "abc"

.. important::

    Since opened datasets stream data from the disk, it is not possible to simultaneously access the same
    dataset from separately running scripts or multiple Jupyter notebooks. To get around
    this, either make a copy of the dataset in the disk or access the dataset using :meth:`Dataset.open()`
    with ``mode="copy"``.

Attribute Metadata
------------------

Dataset attributes can also contain additional metadata, such as docstrings. The :func:`~.data.attribute`
function can be used to attach metadata on assignment or initialization.

>>> hamiltonian = qml.Hamiltonian([1., 1.], [qml.Z(0), qml.Z(1)])
>>> eigvals, eigvecs = np.linalg.eigh(qml.matrix(hamiltonian))
>>> dataset = qml.data.Dataset(hamiltonian = qml.data.attribute(
...     hamiltonian,
...     doc="The hamiltonian of the system"))
>>> dataset.eigen = qml.data.attribute(
...     {"eigvals": eigvals, "eigvecs": eigvecs},
...     doc="Eigenvalues and eigenvectors of the hamiltonain")

This metadata can then be accessed using the :meth:`Dataset.attr_info` mapping:

>>> dataset.attr_info["eigen"]["doc"]
'Eigenvalues and eigenvectors of the hamiltonain'


Declarative API
---------------

When creating datasets to model a physical system, it is common to collect the same data for
a system under different conditions or assumptions. For example, a collection of datasets describing
a quantum oscillator, which contains the first 1000 energy levels for different masses and force constants.

The datasets declarative API allows us to create subclasses
of :class:`Dataset` that define the required attributes, or 'fields', and
their associated type and documentation:

.. code-block:: python

    class QuantumOscillator(qml.data.Dataset, data_name="quantum_oscillator", identifiers=["mass", "force_constant"]):
        \"""Dataset describing a quantum oscillator.\"""

        mass: float = qml.data.field(doc = "The mass of the particle")
        force_constant: float = qml.data.field(doc = "The force constant of the oscillator")
        hamiltonian: qml.Hamiltonian = qml.data.field(doc = "The hamiltonian of the particle")
        energy_levels: np.ndarray = qml.data.field(doc = "The first 1000 energy levels of the system")

The ``data_name`` keyword specifies a category or descriptive name for the dataset type, and the ``identifiers``
keyword to the class is used to specify fields that function as parameters, i.e they determine the behaviour
of the system.

When a ``QuantumOscillator`` dataset is created, its attributes will have the documentation from the field
definition:

>>> dataset = QuantumOscillator(
...     mass=1,
...     force_constant=0.5,
...     hamiltonian=qml.X(0),
...     energy_levels=np.array([0.1, 0.2])
... )
>>> dataset.attr_info["mass"]["doc"]
'The mass of the particle'

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
    DatasetTuple,
)
from .base import DatasetNotWriteableError
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
    "AttributeInfo",
    "attribute",
    "field",
    "Dataset",
    "DatasetAttribute",
    "DatasetNotWriteableError",
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
    "DatasetTuple",
    "load",
    "load_interactive",
    "list_attributes",
    "list_datasets",
    "DEFAULT",
    "FULL",
)
