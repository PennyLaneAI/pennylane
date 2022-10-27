.. role:: html(raw)
   :format: html

.. _intro_ref_data:

Data Module
============

The data module provides functionality to access, store and manipulate the quantum datasets within the PennyLane framework.

Loading Datasets in PennyLane
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can then download data of the desired type with the :func: `~pennylande.data.load` function, and if the data that has already been downloaded will be loaded from the disk. We make sure to include all of the necessary attributes for the desired data. 
For 'qchem' this includes the `molname`, `basis` and `bondlength`, and for 'qspin' this includes `sys_name`, `periodic`, `lattice` and `layout`. Ideally, the :func:`~pennylande.data.load` returns a list with the desired data which can then be used within
the usual PennyLane workflow.

.. code-block:: python

    >>> H2_dataset=qml.data.load(data_type='qchem',molname='H2', basis='6-31G', bondlength='0.46')
    >>> print(H2_dataset)
    [<pennylane.data.dataset.Dataset object at 0x7f14e4369640>]


Using Dataset in PennyLane
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once loaded, one can access the attributes in the dataset.

.. code-block:: python

    >>> H2_dataset[0].molecule
    <pennylane.qchem.molecule.Molecule object at 0x7f890b409280>
    >>> H2_dataset[0].hf_state
    [1 1 0 0 0 0 0 0]

Since the loaded data items are compatible with PennyLane whenever possible, so we can use them directly in PennyLane worflow as follows:

.. code-block:: python

    >>> dev = qml.device('default.qubit',wires=H2_dataset[0].hamiltonian.wires)
    >>> @qml.qnode(dev)
    ... def circuit():
    ...     return qml.expval(H2_dataset[0].hamiltonian)
    >>> print(circuit())
    2.173913043478261

Dataset Structure
~~~~~~~~~~~~~~~~~~~

PennyLane's quantum dataset currently contains two subcategories: `qchem` and `qspin`, which
contain data regarding to molecules and spin systems, respectively. Users can use the 
:func:`~.pennylane.qdata.list_datasets` method to get a snapshot of the currently available.
This function returns a nested dictionary as we show below. Note that the 

.. code-block:: python

    >>> print('Level 1:'); pprint(qdata.list_datasets(), depth=1)
    Level 1:
    {'qchem': {...}, 'qspin': {...}}
    >>> print('Level 2:'); pprint(qdata.list_datasets(), depth=2)
    Level 2:
    {'qchem': {'H2': {...}, 'LiH': {...}, 'NH3': {...}, ...},
     'qspin': {'Heisenberg': {...}, 'Ising': {...}, ...}}

Filtering Datasets
~~~~~~~~~~~~~~~~~~~

In the case that we only wish to download or load portions of a large dataset, we can specify the desired attributes. For example, we can download only the molecule and hamiltonian of a dataset as follows:

.. code-block:: python

    >>> H2_hamiltonian = qml.data.load(data_type='qchem',molname='H2', basis='6-31G', bondlength='0.46', attributes=['molecule','hamiltonian'])
    >>> H2_hamiltonian
    <Hamiltonian: terms=185, wires=[0, 1, 2, 3, 4, 5, 6, 7]>

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.qdata.Dataset
    ~pennylane.qdata.ChemDataset
    ~pennylane.qdata.SpinDataset

:html:`</div>`


Utility Functions
^^^^^^^^^^^^^^^^^^

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.qdata.list_datasets
    ~pennylane.qdata.get_params
    ~pennylane.qdata.get_keys
    ~pennylane.qdata.load

:html:`</div>`
