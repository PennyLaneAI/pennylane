.. role:: html(raw)
   :format: html

.. _intro_ref_qdata:

Quantum Datasets
=================

PennyLane provides the :mod:`~.data` module to access quantum datasets comprising of quantum
chemsitry data and quantum many-body spin systems data. It contains the functionalities to
download these datasets from a server, store and manipulate it locally. It also provides tools
for users to build such datasets themselves using the :class:`~.pennylane.qdata.QuantumData` class.
Here we explain the structure of the hosted data and how can it be used inside PennyLane.

Dataset Structure
------------------

PennyLane's quantum dataset currently contains two subcategories: `qchem` and `qspin`, which
contains data regarding molecules and spin systems, respectively. Users can use the 
:func:`~.pennylane.qdata.list_datasets` method to get a snapshot of the current state of the
datasets as we show below:

.. code-block:: python

    >>> from pprint import pprint
    >>> print('Level 1:'); pprint(qdata.list_datasets(), depth=1)
    Level 1:
    {'qchem': {...}, 'qspin': {...}}
    >>> print('Level 2:'); pprint(qdata.list_datasets(), depth=2)
    Level 2:
    {'qchem': {'H2': {...}, 'LiH': {...}, 'NH3': {...}, ...},
     'qspin': {'Heisenberg': {...}, 'Ising': {...}, ...}}

This nested-dictionary structure can also be used to generate arguments for the :func:`~.pennylane.qdata.load`
function that allows us to downlaod the dataset to be stored and accessed locally. The main purpose of these
arguments is to proivde users with the flexibility of filtering data as per their needs and downloading what
matches their specified criteria. For example, 

.. code-block:: python

    >>> qdata.get_params(qdata.list_datasets(), "qchem", basis=["STO3G"])
    [{'molname': ['full'], 'basis': ['STO3G'], 'bondlength': ['full']}]
    >>> qdata.get_keys("qchem", qdata.get_params(qdata.list_datasets(), "qchem",)[0])
    ['full']

These arguments can be supplied as it is with the load function or users can manually built these arguments
as per their liking. Upon doing so, they can simply load the data as follows:

.. code-block:: python

    >>> data_type = "qchem"
    >>> data_params = {"molname":"full", "basis":"full", "bondlength":"full"}
    >>> dataset = qdata.load(data_type, data_params)
    Downloading data to datasets/qchem
    [███████████████████████████████████████████████████████ 100.0 %] 146.48 KB/146.48 KB
    >>> dataset
    [<pennylane.qdata.chemdata.Chemdata at 0x1666b2c50>,
     <pennylane.qdata.chemdata.Chemdata at 0x1666b2500>,
     <pennylane.qdata.chemdata.Chemdata at 0x28917aec0>]

Using Datasets in PennyLane
----------------------------

Once downloaded and loaded to the memory, users can access various attributes from each of these datasets. These
attributes can then be used within the usual PennyLane workflow. For example, using the dataset downloaded above

.. code-block:: python

    >>> qchem_dataset = dataset[0]
    >>> {"mol":qchem_dataset.molecule.symbols, "ham":qchem_dataset.hamiltonian}
    {'mol: ['N', 'H', 'H', 'H'],
     'ham': <Hamiltonian: terms=4409, wires=[0, 1, 2, ... , 15]>}
    >>> dev = qml.device('lightning.qubit', wires=16, batch_obs=True)
    >>> @qml.qnode(dev, diff_method="parameter-shift")
    >>> def cost_fn_2():
    ...     qchem_dataset.adaptive_circuit(qchem_dataset.adaptive_params, range(16))
    ...     return qml.expval(qchem_dataset.num_op)
    >>> cost_fn_2()
    tensor(10., requires_grad=True)

Dataset Index
--------------

For each molecule, we obtain the following data for `40` different `geometries`.

.. table:: Quantum chemistry dataset index
    :widths: auto

    +----------------------------+-----------------------------------------------------------------------------------+
    |                            |                                                                                   |
    +============================+===================================================================================+
    | **Molecular Data**         |                                                                                   |
    +----------------------------+-----------------------------------------------------------------------------------+
    | `molecule`                 | PennyLane Molecule object                                                         |
    +----------------------------+-----------------------------------------------------------------------------------+
    | `fci_energy`               | Classical energy of the molecule from exact diagonalization or FCI calculation.   |
    +----------------------------+-----------------------------------------------------------------------------------+
    | **Hamiltonian Data**       |                                                                                   |
    +----------------------------+-----------------------------------------------------------------------------------+
    | `hamiltonian`              | PennyLane Hamiltonian in string format                                            |
    +----------------------------+-----------------------------------------------------------------------------------+
    | `meas_groupings`           | Measurement groupings for the Hamiltonian                                         |
    +----------------------------+-----------------------------------------------------------------------------------+
    | **Auxillary Observables**  | *(in string format)*                                                              |
    +----------------------------+-----------------------------------------------------------------------------------+
    | `dipole_op`                | Dipole moment operator                                                            |
    +----------------------------+-----------------------------------------------------------------------------------+
    | `number_op`                | Number operator                                                                   |
    +----------------------------+-----------------------------------------------------------------------------------+
    | `spin2_op`                 | Total spin operator                                                               |
    +----------------------------+-----------------------------------------------------------------------------------+
    | `spinz_op`                 | Spin projection operator                                                          |
    +----------------------------+-----------------------------------------------------------------------------------+
    | **Tapering Data**          |                                                                                   |
    +----------------------------+-----------------------------------------------------------------------------------+
    | `symmetries`               | Symmetries required for tapering molecular Hamiltonian                            |
    +----------------------------+-----------------------------------------------------------------------------------+
    | `paulix_ops`               | Supporting PauliX ops required to build Clifford U for tapering                   |
    +----------------------------+-----------------------------------------------------------------------------------+
    | `optimal_sector`           | Eigensector of the tapered qubits that would contain the ground state             |
    +----------------------------+-----------------------------------------------------------------------------------+
    | **Tapered Observables**    | *(in string format)*                                                              |
    +----------------------------+-----------------------------------------------------------------------------------+
    | `tapered_hamiltonian`      | Tapered Hamiltonian                                                               |
    +----------------------------+-----------------------------------------------------------------------------------+
    | `tapered_dipole_op`        | Tapered dipole moment operator                                                    |
    +----------------------------+-----------------------------------------------------------------------------------+
    | `tapered_num_op`           | Tapered number operator                                                           |
    +----------------------------+-----------------------------------------------------------------------------------+
    | `tapered_spin2_op`         | Tapered total spin operator                                                       |
    +----------------------------+-----------------------------------------------------------------------------------+
    | `tapered_spinz_op`         | Tapered spin projection operator                                                  |
    +----------------------------+-----------------------------------------------------------------------------------+
    | **VQE Data**               |                                                                                   |
    +----------------------------+-----------------------------------------------------------------------------------+
    | `vqe_circuit`              | Circuit structure for AdaptiveGivens ansatz                                       |
    +----------------------------+-----------------------------------------------------------------------------------+
    | `vqe_params`               | Parameters for the AdaptiveGiven ansatz                                           |
    +----------------------------+-----------------------------------------------------------------------------------+
    | `vqe_energy`               | Energy obtained from VQE with the AdaptiveGivens ansatz                           |
    +----------------------------+-----------------------------------------------------------------------------------+


For each spin system, we obtain the following data for `100` different `parameters`.

.. table:: Quantum many-body spins systems dataset
    :widths: auto

    +----------------------------+---------------------------------------------------------------+
    |                            |                                                               |
    +============================+===============================================================+
    | **Spin System Data**       |                                                               |
    +----------------------------+---------------------------------------------------------------+
    | `spin_system`              | Basic description of the spin system                          |
    +----------------------------+---------------------------------------------------------------+
    | `parameters`               | Variable parameters that define the spin system               |
    +----------------------------+---------------------------------------------------------------+
    | **Hamiltonian Data**       |                                                               |
    +----------------------------+---------------------------------------------------------------+
    | `hamiltonians`             | PennyLane Hamiltonian in string format                        |
    +----------------------------+---------------------------------------------------------------+
    | **Phase Transition Data**  |                                                               |
    +----------------------------+---------------------------------------------------------------+
    | `phase_labels`             | Phase labels according to the known phase transition data     |
    +----------------------------+---------------------------------------------------------------+
    | `order_parameters`         | Observables and their values used for assigning phase labels  |
    +----------------------------+---------------------------------------------------------------+
    | **Ground State Data**      |                                                               |
    +----------------------------+---------------------------------------------------------------+
    | `ground_energies`          | Ground state energies of each system                          |
    +----------------------------+---------------------------------------------------------------+
    | `ground_states`            | Ground state of each system                                   |
    +----------------------------+---------------------------------------------------------------+
    | **Classical Shadow Data**  |                                                               |
    +----------------------------+---------------------------------------------------------------+
    | `classical_shadows`        | Classical shadow representation of each system                |
    +----------------------------+---------------------------------------------------------------+


Quantum dataset APIs and classes
---------------------------------

Dataset Classes
^^^^^^^^^^^^^^^^

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
