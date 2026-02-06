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
of available datasets, please see the `datasets website <https://pennylane.ai/datasets>`_.
The :func:`~pennylane.data.load` function returns a ``list`` with the desired data.

>>> H2datasets = qp.data.load("qchem", molname="H2", basis="STO-3G", bondlength=1.1)
>>> print(H2datasets)
[<Dataset = molname: H2, basis: STO-3G, bondlength: 1.1, attributes: ['basis', 'basis_rot_groupings', ...]>]
>>> H2data = H2datasets[0]

We can load datasets for multiple parameter values by providing a list of values instead of a single value.
To load all possible values, use the special value :const:`~pennylane.data.FULL` or the string ``"full"``:

>>> H2datasets = qp.data.load("qchem", molname="H2", basis="full", bondlength=[0.5, 1.1])
>>> print(H2datasets)
[<Dataset = molname: H2, basis: STO-3G, bondlength: 0.5, attributes: ['basis', 'basis_rot_groupings', ...]>,
<Dataset = molname: H2, basis: STO-3G, bondlength: 1.1, attributes: ['basis', 'basis_rot_groupings', ...]>,
<Dataset = molname: H2, basis: CC-PVDZ, bondlength: 0.5, attributes: ['basis', 'basis_rot_groupings', ...]>,
<Dataset = molname: H2, basis: CC-PVDZ, bondlength: 1.1, attributes: ['basis', 'basis_rot_groupings', ...]>,
<Dataset = molname: H2, basis: 6-31G, bondlength: 0.5, attributes: ['basis', 'basis_rot_groupings', ...]>,
<Dataset = molname: H2, basis: 6-31G, bondlength: 1.1, attributes: ['basis', 'basis_rot_groupings', ...]>]

When we only want to download portions of a large dataset, we can specify the desired properties  (referred to as 'attributes').
For example, we can download or load only the molecule and energy of a dataset as follows:

>>> part = qp.data.load("qchem", molname="H2", basis="STO-3G", bondlength=1.1, 
...                      attributes=["molecule", "fci_energy"])[0]
>>> part.molecule
<Molecule = H2, Charge: 0, Basis: STO-3G, Orbitals: 2, Electrons: 2>
>>> part.fci_energy
-1.0791924385860894

To determine what attributes are available for a type of dataset, we can use the function :func:`~pennylane.data.list_attributes`:

>>> qp.data.list_attributes(data_name="qchem")
['molname',
 'basis',
 'bondlength',
 ...
 'vqe_params',
 'vqe_energy']

.. note::

    The default values for attributes are as follows:
    
    - Molecules: ``basis`` is the smallest available basis, usually ``"STO-3G"``, and ``bondlength`` is the optimal bondlength for the molecule or an alternative if the optimal is not known.
    
    - Spin systems: ``periodicity`` is ``"open"``, ``lattice`` is ``"chain"``, and ``layout`` is ``1x4`` for ``chain`` systems and ``2x2`` for ``rectangular`` systems.

Using Datasets in PennyLane
---------------------------

Once loaded, one can access properties of the datasets:

>>> H2data.molecule
<Molecule = H2, Charge: 0, Basis: STO-3G, Orbitals: 2, Electrons: 2>
>>> print(H2data.hf_state)
[1 1 0 0]

The loaded data items are fully compatible with PennyLane. We can therefore
use them directly in a PennyLane circuits as follows:

>>> dev = qp.device("default.qubit",wires=4)
>>> @qp.qnode(dev)
... def circuit():
...     qp.BasisState(H2data.hf_state, wires = [0, 1, 2, 3])
...     for op in H2data.vqe_gates:
...         qp.apply(op)
...     return qp.expval(H2data.hamiltonian)
>>> print(circuit())
-1.0791430411076344

Viewing Available Dataset Names
-------------------------------

We can call the 
:func:`~pennylane.data.list_data_names` function to get a snapshot of the names of the currently available datasets.
This function returns a list of strings as shown below.

>>> qp.data.list_data_names()
["bars-and-stripes",
 "downscaled-mnist",
 "hamlib-max-3-sat",
 "hamlib-maxcut",
 "hamlib-travelling-salesperson-problem",
 "hidden-manifold",
 "hyperplanes",
 "ketgpt",
 "learning-dynamics-incoherently",
 "linearly-separable",
 "mnisq",
 "mqt-bench",
 "plus-minus",
 "qchem",
 "qspin",
 "rydberggpt",
 "two-curves"]

Note that this example limits the results
of the function calls for clarity and that as more data becomes available, the results of these
function calls will change.

Viewing Available Datasets
--------------------------

We can call the 
:func:`~pennylane.data.list_datasets` function to get a snapshot of the currently available data.
This function returns a nested dictionary as shown below. 

>>> available_data = qp.data.list_datasets()
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
>>> observables = [qp.Z(0), qp.X(1)]
>>> H = qp.Hamiltonian(coeffs, observables)
>>> energies, _ = np.linalg.eigh(qp.matrix(H)) #Calculate the energies
>>> dataset = qp.data.Dataset(data_name = "Example", hamiltonian=H, energies=energies)
>>> dataset.data_name
"Example"
>>> dataset.hamiltonian
1.0 * Z(0) + 0.5 * X(1)
>>> dataset.energies
array([-1.5, -0.5,  0.5,  1.5])

We can then write this :class:`~pennylane.data.Dataset` to storage and read it as follows:


>>> dataset.write("./path/to/dataset.h5")
>>> read_dataset = qp.data.Dataset()
>>> read_dataset.read("./path/to/dataset.h5")
>>> read_dataset.data_name
"Example"
>>> read_dataset.hamiltonian
1.0 * Z(0) + 0.5 * X(1)
>>> read_dataset.energies
array([-1.5, -0.5,  0.5,  1.5])

For more details on reading and writing custom datasets, including metadata, please
see the :mod:`~.data` module documentation.

Quantum Datasets Functions and Classes
--------------------------------------

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.data.list_attributes
    ~pennylane.data.list_data_names
    ~pennylane.data.list_datasets
    ~pennylane.data.load
    ~pennylane.data.load_interactive
    ~pennylane.data.Dataset

:html:`</div>`
