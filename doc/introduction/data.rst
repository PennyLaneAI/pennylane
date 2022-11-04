.. role:: html(raw)
   :format: html

.. _intro_ref_data:

Quantum Datasets
================

PennyLane provides the :mod:`~.data` subpackage to download, create, store and manipulate quantum datasets.

.. note::

    The packages `zstd` and `dill` are required to use the :mod:`~pennyalane.data` module.
    These can be installed with `pip install zstd dill`.


Loading Datasets in PennyLane
-----------------------------

We can access data of a desired type with the :func:`~pennylane.data.load` or :func:`~pennylane.data.load_interactive` functions.
These download the desired datasets or load them from local storage if previously downloaded. 
Here we use the :func:`~pennylane.data.load` function, for the
:func:`~pennylane.data.load_interactive` function, please see the relevant documentation.

To specify the dataset to be loaded, the data category (``data_name``) must be
specified, alongside category-specific keyword arguments. For the full list
of available datasets, please see the `datasets website <https://pennylane.ai/qml/datasets.html>`_.
The :func:`~pennylane.data.load` function returns a list with the desired data.

>>> H2_dataset = qml.data.load(data_name="qchem", molname="H2", basis="STO-3G", bondlength="1.0")
>>> print(H2_dataset)
[<pennylane.data.dataset.Dataset object at 0x7f14e4369640>]

When we only want to download portions of a large dataset, we can specify the desired properties  (referred to as `attributes`).
For example, we can download or load only the molecule and energy of a dataset as follows:

>>> H2_partial = qml.data.load(data_name='qchem',molname='H2', basis='STO-3G', bondlength=1.0, attributes=['molecule','fci_energy'])[0]
>>> H2_partial.molecule
<pennylane.qchem.molecule.Molecule at 0x7f56c9d78e50>
>>> H2_partial.fci_energy
-1.1011498981604342

To determine what attributes are available for a type of dataset, we can use the function :func:`~pennylane.data.list_attributes`:

>>> qml.data.list_attributes(data_name='qchem')
['molecule',
'hamiltonian',
'wire_map',
...
'vqe_params',
'vqe_circuit']

Using Datasets in PennyLane
---------------------------

Once loaded, one can access properties of the datasets:

>>> H2_dataset[0].molecule
<pennylane.qchem.molecule.Molecule object at 0x7f890b409280>
>>> print(H2_dataset[0].hf_state)
[1 1 0 0]

The loaded data items are fully compatible with PennyLane. We can therefore
use them directly in a PennyLane circuits as follows:

>>> dev = qml.device('default.qubit',wires=H2_dataset[0].hamiltonian.wires)
>>> @qml.qnode(dev)
... def circuit():
...     return qml.expval(H2_dataset[0].hamiltonian)
>>> print(circuit())
2.173913043478261

Dataset Structure
-----------------

You can call the 
:func:`~.pennylane.data.list_datasets` function to get a snapshot of the currently available data.
This function returns a nested dictionary as we show below. Note that this example limits the results
of the function calls for clarity and that as more data becomes available, the results of these
function calls will change.

>>> available_data = qml.data.list_datasets()
>>> available_data.keys()
dict_keys(['qspin', 'qchem'])
>>> available_data['qchem'].keys()
dict_keys(['HF', 'LiH', ...])
>>> available_data['qchem']['H2'].keys()
dict_keys(['STO-3G'])
>>> print(available_data['qchem']['H2']['STO-3G'])
['2.35', '1.75', '0.6', '1.85', ...]

Creating Custom Datasets
------------------------

The functionality in :mod:`~pennylane.data` also includes creating and reading custom-made datasets.
To create a dataset, we can do the following:

>>> example_hamiltonian = qml.Hamiltonian(coeffs=[1,0.5], observables=[qml.PauliZ(wires=0),qml.PauliX(wires=1)])
>>> example_energies, _ = np.linalg.eigh(qml.matrix(example_hamiltonian)) #Calculate the energies
>>> example_dataset = qml.data.Dataset(data_name = 'Example',hamiltonian=example_hamiltonian,energies=example_energies)
>>> example_dataset.data_name
'Example'
>>> example_dataset.hamiltonian
    (0.5) [X1]
+ (1) [Z0]
>>> example_dataset.energies
array([-1.5, -0.5,  0.5,  1.5])

We can then write this :class:`~pennylane.data.Dataset` to storage and read it as follows:


>>> example_dataset.write('./path/to/dataset.dat')
>>> read_dataset = qml.data.Dataset()
>>> read_dataset.read('./path/to/dataset.dat')
>>> read_dataset.data_name
'Example'
>>> read_dataset.hamiltonian
    (0.5) [X1]
+ (1) [Z0]
>>> read_dataset.energies
array([-1.5, -0.5,  0.5,  1.5])

:html:`<div class="summary-table">`

Quantum Datasets Functions and Classes
--------------------------------------

Classes
^^^^^^^

.. autosummary::
    :nosignatures:

    ~pennylane.data.Dataset

:html:`</div>`

Functions
^^^^^^^^^

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.data.list_datasets
    ~pennylane.data.list_attributes
    ~pennylane.data.load
    ~pennylane.data.load_interactive

:html:`</div>`
