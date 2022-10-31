.. role:: html(raw)
   :format: html

.. _intro_ref_data:

Quantum Datasets
================

PennyLane provides the :mod:`~.data` subpackage to download, create, store and manipulate quantum datasets within the PennyLane framework.

Loading Datasets in PennyLane
-----------------------------

We can access data of a desired type with the :func:`~pennylane.data.load` function. This downloads the desired
dataset or loads it from local storage if previously downloaded. Some examples of data categories available for download include 'qchem',
quantum chemistry-related datasets, and 'qspin', spin-system-related datasets. To specify the desired data, we use several arguments.
For 'qchem' this includes the molname, basis, and bondlength, and for 'qspin' this includes sys_name, periodic, lattice, and layout. 
The :func:`~pennylane.data.load` returns a list with the desired data which can then be used within
the usual PennyLane workflow.

.. code-block:: python

    >>> H2_dataset=qml.data.load(data_type='qchem',molname='H2', basis='6-31G', bondlength='0.46')
    >>> print(H2_dataset)
    [<pennylane.data.dataset.Dataset object at 0x7f14e4369640>]


Using Datasets in PennyLane
---------------------------

Once loaded, one can access the attributes in the dataset.

.. code-block:: python

    >>> H2_dataset[0].molecule
    <pennylane.qchem.molecule.Molecule object at 0x7f890b409280>
    >>> H2_dataset[0].hf_state
    [1 1 0 0 0 0 0 0]

The loaded data items are compatible with PennyLane whenever possible. We can therefore
use them directly in PennyLane workflow as follows:

.. code-block:: python

    >>> dev = qml.device('default.qubit',wires=H2_dataset[0].hamiltonian.wires)
    >>> @qml.qnode(dev)
    ... def circuit():
    ...     return qml.expval(H2_dataset[0].hamiltonian)
    >>> print(circuit())
    2.173913043478261

Dataset Structure
-----------------

Two example subcategories in PennyLane's quantum dataset subpackage are `qchem` and `qspin`.
These subcategories offer data relating to molecules and spin systems, respectively. Users can call the 
:func:`~.pennylane.data.list_datasets` function to get a snapshot of the currently available data.
This function returns a nested dictionary as we show below. Note that this example limits the results
of the function calls for clarity and that as more data becomes available, the results of these
function calls will change.

.. code-block:: python

    >>> available_data = qml.data.list_datasets()
    >>> available_data.keys()
    dict_keys(['qspin', 'qchem'])
    >>> available_data['qchem']
    dict_keys(['H2', 'HeH'])
    >>> available_data['qchem']['H2']
    dict_keys(['6-31G'])
    >>> print(available_data['qchem']['H2']['6-31G'])
    ['0.46', '1.16', '1.22', ...]

Filtering Datasets
------------------

In the case that we only wish to download or load portions of a large dataset, we can specify the desired attributes.
For example, we can download only the molecule and hamiltonian of a dataset as follows:

.. code-block:: python

    >>> H2_hamiltonian = qml.data.load(data_type='qchem',molname='H2', basis='6-31G', bondlength='0.46', attributes=['molecule','hamiltonian'])
    >>> H2_hamiltonian
    <Hamiltonian: terms=185, wires=[0, 1, 2, 3, 4, 5, 6, 7]>

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
    ~pennylane.data.load

:html:`</div>`
