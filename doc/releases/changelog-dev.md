:orphan:

# Release 0.27.0-dev (development release)

<h3>New features since last release</h3>

* Added the `qml.GellMann` qutrit observable, which is the ternary generalization of the Pauli observables. Users must include an index as a
keyword argument when using `GellMann`, which determines which of the 8 Gell-Mann matrices is used as the observable.
  ([#3035](https://github.com/PennyLaneAI/pennylane/pull/3035))

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

* Added the `Operator` attributes `has_decomposition` and `has_adjoint` that indicate
  whether a corresponding `decomposition` or `adjoint` method is available.
  [(#2986)](https://github.com/PennyLaneAI/pennylane/pull/2986)

* Structural improvements are made to `QueuingManager`, formerly `QueuingContext`, and `AnnotatedQueue`.
  [(#2794)](https://github.com/PennyLaneAI/pennylane/pull/2794)
  [(#3061)](https://github.com/PennyLaneAI/pennylane/pull/3061)

   - `QueuingContext` is renamed to `QueuingManager`.
   - `QueuingManager` should now be the global communication point for putting queuable objects into the active queue.
   - `QueuingManager` is no longer an abstract base class.
   - `AnnotatedQueue` and its children no longer inherit from `QueuingManager`.
   - `QueuingManager` is no longer a context manager.
   -  Recording queues should start and stop recording via the `QueuingManager.add_active_queue` and 
     `QueueingContext.remove_active_queue` class methods instead of directly manipulating the `_active_contexts` property.
   - `AnnotatedQueue` and its children no longer provide global information about actively recording queues. This information
      is now only available through `QueuingManager`.
   - `AnnotatedQueue` and its children no longer have the private `_append`, `_remove`, `_update_info`, `_safe_update_info`,
      and `_get_info` methods. The public analogues should be used instead.
   - `QueuingManager.safe_update_info` and `AnnotatedQueue.safe_update_info` are deprecated.  Their functionality is moved to
      `update_info`.

* Added `unitary_check` keyword argument to the constructor of the `QubitUnitary` class which
  indicates whether the user wants to check for unitarity of the input matrix or not. Its default
  value is `false`.
  [(#3063)](https://github.com/PennyLaneAI/pennylane/pull/3063)
   
* Modified the representation of `WireCut` by using `qml.draw_mpl`.
  [(#3067)](https://github.com/PennyLaneAI/pennylane/pull/3067)

* Improve `qml.math.expand_matrix` method for sparse matrices.
  [(#3060)](https://github.com/PennyLaneAI/pennylane/pull/3060)

<h3>Breaking changes</h3>

 * `QueuingContext` is renamed `QueuingManager`.
  [(#3061)](https://github.com/PennyLaneAI/pennylane/pull/3061)

 * `QueuingManager.safe_update_info` and `AnnotatedQueue.safe_update_info` are deprecated. Instead, `update_info` no longer raises errors
   if the object isn't in the queue.

 * Deprecation patches for the return types enum's location and `qml.utils.expand` are removed.
   [(#3092)](https://github.com/PennyLaneAI/pennylane/pull/3092)

<h3>Deprecations</h3>

* `qml.tape.stop_recording` and `QuantumTape.stop_recording` are moved to `qml.QueuingManager.stop_recording`.
  The old functions will still be available untill v0.29.
  [(#3068)](https://github.com/PennyLaneAI/pennylane/pull/3068)

* `qml.tape.get_active_tape` is deprecated. Please use `qml.QueuingManager.active_context()` instead.
  [(#3068)](https://github.com/PennyLaneAI/pennylane/pull/3068)

<h3>Documentation</h3>

<h3>Bug fixes</h3>

* Fixed the `qml.transforms.transpile` transform to work correctly for all multi-qubit operations.
  [(#3104)](https://github.com/PennyLaneAI/pennylane/pull/3104)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso-Linaje,
Juan Miguel Arrazola,
Albert Mitjans Coma,
Utkarsh Azad,
Soran Jahangiri,
Christina Lee,
Mudit Pandey,
Jay Soni,
Antal Száva,
David Wierichs,
