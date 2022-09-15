:orphan:

# Release 0.27.0-dev (development release)

<h3>New features since last release</h3>

* `qml.qchem.taper_operation` tapers any gate operation according to the `Z2`
  symmetries of the Hamiltonian. 
  [(#3002)](https://github.com/PennyLaneAI/pennylane/pull/3002)

  ```pycon
    >>> symbols = ['He', 'H']
    >>> geometry =  np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4589]])
    >>> mol = qchem.Molecule(symbols, geometry, charge=1)
    >>> H, n_qubits = qchem.molecular_hamiltonian(symbols, geometry)
    >>> generators = qchem.symmetry_generators(H)
    >>> paulixops = qchem.paulix_ops(generators, n_qubits)
    >>> paulix_sector = qchem.optimal_sector(H, generators, mol.n_electrons)
    >>> qchem.taper_operation(qml.SingleExcitation(3.14159, wires=[0, 2]),
                                generators, paulixops, paulix_sector, wire_order=H.wires)
    [PauliRot(-3.14159+0.j, 'RY', wires=[0])]
    ```

  When used within a QNode, this method applies the tapered operation directly:

  ```pycon
    >>> dev = qml.device('default.qubit', wires=[0, 1])
    >>> @qml.qnode(dev)
    ... def circuit(params):
    ...     qchem.taper_operation(qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3]),
    ...                             generators, paulixops, paulix_sector, H.wires)
    ...     return qml.expval(qml.PauliZ(0)@qml.PauliZ(1))
    >>> drawer = qml.draw(circuit, show_all_wires=True)
    >>> print(drawer(params=[3.14159]))
        0: ─╭RXY(1.570796+0.00j)─╭RYX(1.570796+0.00j)─┤ ╭<Z@Z>
        1: ─╰RXY(1.570796+0.00j)─╰RYX(1.570796+0.00j)─┤ ╰<Z@Z>
  ```

<h3>Improvements</h3>

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Documentation</h3>

<h3>Bug fixes</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola,
Utkarsh Azad,
Soran Jahangiri,
Jay Soni,