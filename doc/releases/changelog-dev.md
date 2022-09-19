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

* Structural improvements are made to `QueuingContext` and `AnnotatedQueue`. None of these changes should
  influence PennyLane behaviour outside of the `queueing.py` module.
  [(#2794)](https://github.com/PennyLaneAI/pennylane/pull/2794)

  * `QueuingContext` should now be the global communication point for putting queuable objects into the active queue.
  * `QueuingContext` is no longer an abstract base class.
  * `AnnotatedQueue` and its children no longer inherit from `QueuingContext`.
  * `QueuingContext` is no longer a context manager.
  * Recording queues should start and stop recording via the `QueuingContext.add_active_queue` and
     `QueueingContext.remove_active_queue` class methods instead of directly manipulating the `_active_contexts` property.
  * `AnnotatedQueue` and its children no longer provide global information about actively recording queues. This information
      is now only available through `QueuingContext`.
  * `AnnotatedQueue` and its children no longer have the private `_append`, `_remove`, `_update_info`, `_safe_update_info`,
      and `_get_info` methods. The public analogues should be used instead.

* Improve `qml.math.expand_matrix` method for dense matrices.
  [(#3064)](https://github.com/PennyLaneAI/pennylane/pull/3064)

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Documentation</h3>

<h3>Bug fixes</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola,
Utkarsh Azad,
Soran Jahangiri,
Christina Lee,
Jay Soni,
