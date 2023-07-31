.. role:: html(raw)
   :format: html

.. _intro_ref_data:

Quantum Datasets
================

PennyLane provides the :mod:`~.data` subpackage to download, create, store and manipulate quantum datasets,
where the quantum dataset is a collection of `quantum data` obtained from various quantum systems that describe it and its evolution.

.. note::

    The packages ``aiohttp``, ``fsspec``, and ``h5py`` are required to use the :mod:`~pennylane.data` module. 
    These can be installed with:
    
    .. code-block:: console
    
        pip install aiohttp fsspec h5py

Loading Datasets in PennyLane
-----------------------------

We can access data of a desired type with the :func:`~pennylane.data.load` or :func:`~pennylane.data.load_interactive` functions.
These download the desired datasets or load them from local storage if previously downloaded.

To specify the dataset to be loaded, the data category (``data_name``) must be
specified, alongside category-specific keyword arguments. For the full list
of available datasets, please see the `datasets website <https://pennylane.ai/qml/datasets.html>`_.
The :func:`~pennylane.data.load` function returns a ``list`` with the desired data.

>>> H2datasets = qml.data.load("qchem", molname="H2", basis="STO-3G", bondlength=1.1)
>>> print(H2datasets)
[<Dataset = molname: H2, basis: STO-3G, bondlength: 1.1, attributes: ['basis', 'basis_rot_groupings', ...]>]
>>> H2data = H2datasets[0]

We can load datasets for multiple parameter values by providing a list of values instead of a single value.
To load all possible values, use the special value :const:`~pennylane.data.FULL` or the string ``"full"``:

>>> H2datasets = qml.data.load("qchem", molname="H2", basis="full", bondlength=[0.5, 1.1])
>>> print(H2datasets)
[<Dataset = molname: H2, basis: 6-31G, bondlength: 0.5, attributes: ['basis', 'bondlength', ...]>,
 <Dataset = molname: H2, basis: 6-31G, bondlength: 1.1, attributes: ['basis', 'bondlength', ...]>,
 <Dataset = molname: H2, basis: STO-3G, bondlength: 0.5, attributes: ['basis', 'basis_rot_groupings', ...]>,
 <Dataset = molname: H2, basis: STO-3G, bondlength: 1.1, attributes: ['basis', 'basis_rot_groupings', ...]>]

When we only want to download portions of a large dataset, we can specify the desired properties  (referred to as 'attributes').
For example, we can download or load only the molecule and energy of a dataset as follows:

>>> part = qml.data.load("qchem", molname="H2", basis="STO-3G", bondlength=1.1, 
...                      attributes=["molecule", "fci_energy"])[0]
>>> part.molecule
<Molecule = H2, Charge: 0, Basis: STO-3G, Orbitals: 2, Electrons: 2>
>>> part.fci_energy
-1.0791924385860894

To determine what attributes are available for a type of dataset, we can use the function :func:`~pennylane.data.list_attributes`:

>>> qml.data.list_attributes(data_name="qchem")
['molname',
 'basis',
 'bondlength',
 ...
 'vqe_params',
 'vqe_energy']

.. note::

    ``"full"`` is the default value for ``attributes``, and it means that all available attributes for the Dataset will be downloaded.

Using Datasets in PennyLane
---------------------------

Once loaded, one can access properties of the datasets:

>>> H2data.molecule
<Molecule = H2, Charge: 0, Basis: STO-3G, Orbitals: 2, Electrons: 2>
>>> print(H2data.hf_state)
[1 1 0 0]

The loaded data items are fully compatible with PennyLane. We can therefore
use them directly in a PennyLane circuits as follows:

>>> dev = qml.device("default.qubit",wires=4)
>>> @qml.qnode(dev)
... def circuit():
...     qml.BasisState(H2data.hf_state, wires = [0, 1, 2, 3])
...     for op in H2data.vqe_gates:
...         qml.apply(op)
...     return qml.expval(H2data.hamiltonian)
>>> print(circuit())
-1.0791430411076344

Dataset Structure
-----------------

You can call the 
:func:`~pennylane.data.list_datasets` function to get a snapshot of the currently available data.
This function returns a nested dictionary as we show below. 

>>> available_data = qml.data.list_datasets()
>>> available_data.keys()
dict_keys(["qspin", "qchem"])
>>> available_data["qchem"].keys()
dict_keys(["H2", "LiH", ...])
>>> available_data['qchem']['H2'].keys()
dict_keys(["6-31G", "STO-3G"])
>>> print(available_data['qchem']['H2']['STO-3G'])
["0.5", "0.54", "0.62", "0.66", ...]

Note that this example limits the results
of the function calls for clarity and that as more data becomes available, the results of these
function calls will change.

Creating Custom Datasets
------------------------

The functionality in :mod:`~pennylane.data` also includes creating and reading custom-made datasets.
We can use custom datasets to store any data generated in PennyLane and its supporting data.
To create a dataset, we can do the following:

>>> coeffs = [1, 0.5]
>>> observables = [qml.PauliZ(wires=0), qml.PauliX(wires=1)]
>>> H = qml.Hamiltonian(coeffs, observables)
>>> energies, _ = np.linalg.eigh(qml.matrix(H)) #Calculate the energies
>>> dataset = qml.data.Dataset(data_name = "Example", hamiltonian=H, energies=energies)
>>> dataset.data_name
"Example"
>>> dataset.hamiltonian
(0.5) [X1]
+ (1) [Z0]
>>> dataset.energies
array([-1.5, -0.5,  0.5,  1.5])

We can then write this :class:`~pennylane.data.Dataset` to storage and read it as follows:


>>> dataset.write("./path/to/dataset.h5")
>>> read_dataset = qml.data.Dataset()
>>> read_dataset.read("./path/to/dataset.h5")
>>> read_dataset.data_name
"Example"
>>> read_dataset.hamiltonian
(0.5) [X1]
+ (1) [Z0]
>>> read_dataset.energies
array([-1.5, -0.5,  0.5,  1.5])

:html:`<div class="summary-table">`

Quantum Datasets Functions and Classes
--------------------------------------

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.data.list_datasets
    ~pennylane.data.list_attributes
    ~pennylane.data.load
    ~pennylane.data.load_interactive
    ~pennylane.data.Dataset

:html:`</div>`
