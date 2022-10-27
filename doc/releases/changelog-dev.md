:orphan:

# Release 0.27.0-dev (development release)

<h3>New features since last release</h3>

* The `qml.qchem.basis_rotation` function is added to the `qchem` module. This function returns
  grouped coefficients, grouped observables and basis rotation transformation matrices needed to
  construct a qubit Hamiltonian in the rotated basis of molecular orbitals. In this basis, the
  one-electron integral matrix and the symmetric matrices obtained from factorizing the two-electron
  integrals tensor are diagonal.
  ([#3011](https://github.com/PennyLaneAI/pennylane/pull/3011))

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
    >>> tap_op = qchem.taper_operation(qml.SingleExcitation, generators, paulixops,
    ...                paulix_sector, wire_order=H.wires, op_wires=[0, 2])
    >>> tap_op(3.14159)
    [Exp(1.570795j, 'PauliY', wires=[0])]
  ```

  Moreover, the obtained tapered operation can be directly used within a QNode:

  ```pycon
    >>> dev = qml.device('default.qubit', wires=[0, 1])
    >>> @qml.qnode(dev)
    ... def circuit(params):
    ...     tap_op(params[0])
    ...     return qml.expval(qml.PauliZ(0)@qml.PauliZ(1))
    >>> drawer = qml.draw(circuit, show_all_wires=True)
    >>> print(drawer(params=[3.14159]))
        0: ─Exp(1.570795j PauliY)─┤ ╭<Z@Z>
        1: ───────────────────────┤ ╰<Z@Z>

  ```

<h3>Improvements</h3>

* Added a new `pennylane.tape.QuantumScript` class that contains all the non-queuing behavior of `QuantumTape`. Now `QuantumTape` inherits from `QuantumScript` as well
  as `AnnotatedQueue`.
  This is a developer-facing change, and users should not manipulate `QuantumScript` directly.  Instead, they
  should continue to rely on `QNode`s.
  [(#3097)](https://github.com/PennyLaneAI/pennylane/pull/3097)

* The UCCSD and kUpCCGSD template are modified to remove a redundant flipping of the initial state.
  [(#3148)](https://github.com/PennyLaneAI/pennylane/pull/3148)

* `Adjoint` now supports batching if the base operation supports batching.
  [(#3168)](https://github.com/PennyLaneAI/pennylane/pull/3168)

* `OrbitalRotation` is now decomposed into two `SingleExcitation` operations for faster execution and more efficient parameter-shift gradient calculations on devices that natively support `SingleExcitation`.
  [(#3171)](https://github.com/PennyLaneAI/pennylane/pull/3171)

* Added the `Operator` attributes `has_decomposition` and `has_adjoint` that indicate
  whether a corresponding `decomposition` or `adjoint` method is available.
  [(#2986)](https://github.com/PennyLaneAI/pennylane/pull/2986)

* Structural improvements are made to `QueuingManager`, formerly `QueuingContext`, and `AnnotatedQueue`.
  [(#2794)](https://github.com/PennyLaneAI/pennylane/pull/2794)
  [(#3061)](https://github.com/PennyLaneAI/pennylane/pull/3061)

  * `QueuingContext` is renamed to `QueuingManager`.
  * `QueuingManager` should now be the global communication point for putting queuable objects into the active queue.
  * `QueuingManager` is no longer an abstract base class.
  * `AnnotatedQueue` and its children no longer inherit from `QueuingManager`.
  * `QueuingManager` is no longer a context manager.
  * Recording queues should start and stop recording via the `QueuingManager.add_active_queue` and
     `QueueingContext.remove_active_queue` class methods instead of directly manipulating the `_active_contexts` property.
  * `AnnotatedQueue` and its children no longer provide global information about actively recording queues. This information
      is now only available through `QueuingManager`.
  * `AnnotatedQueue` and its children no longer have the private `_append`, `_remove`, `_update_info`, `_safe_update_info`,
      and `_get_info` methods. The public analogues should be used instead.
  * `QueuingManager.safe_update_info` and `AnnotatedQueue.safe_update_info` are deprecated.  Their functionality is moved to
      `update_info`.

* `qml.Identity` now accepts multiple wires.
    [(#3049)](https://github.com/PennyLaneAI/pennylane/pull/3049)

    ```pycon
    >>> id_op = qml.Identity([0, 1])
    >>> id_op.matrix()
    array([[1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]])
    >>> id_op.sparse_matrix()
    <4x4 sparse matrix of type '<class 'numpy.float64'>'
        with 4 stored elements in Compressed Sparse Row format>
    >>> id_op.eigvals()
    array([1., 1., 1., 1.])
    ```

* Added `unitary_check` keyword argument to the constructor of the `QubitUnitary` class which
  indicates whether the user wants to check for unitarity of the input matrix or not. Its default
  value is `false`.
  [(#3063)](https://github.com/PennyLaneAI/pennylane/pull/3063)

* Modified the representation of `WireCut` by using `qml.draw_mpl`.
  [(#3067)](https://github.com/PennyLaneAI/pennylane/pull/3067)

* Improved the performance of the `qml.math.expand_matrix` function for dense matrices.
  [(#3064)](https://github.com/PennyLaneAI/pennylane/pull/3064)

* Improve `qml.math.expand_matrix` method for sparse matrices.
  [(#3060)](https://github.com/PennyLaneAI/pennylane/pull/3060)

* Support sums and products of `Operator` classes with scalar tensors of any interface
  (numpy, jax, tensorflow, torch...).
  [(#3149)](https://github.com/PennyLaneAI/pennylane/pull/3149)

  ```pycon
  >>> s_prod = torch.tensor(4) * qml.RX(1.23, 0)
  >>> s_prod
  4*(RX(1.23, wires=[0]))
  >>> s_prod.scalar
  tensor(4)

* Added the `map_wires` method to the `Operator` class, which returns a copy of the operator with
  its wires changed according to the given wire map.
  [(#3143)](https://github.com/PennyLaneAI/pennylane/pull/3143)

  ```pycon
  >>> op = qml.Toffoli([0, 1, 2])
  >>> wire_map = {0: 2, 2: 0}
  >>> op.map_wires(wire_map=wire_map)
  Toffoli(wires=[2, 1, 0])
  ```

* Adds caching to the `compute_matrix` and `compute_sparse_matrix` of simple non-parametric operations.
  [(#3134)](https://github.com/PennyLaneAI/pennylane/pull/3134)

* Add details to the output of `Exp.label()`.
  [(#3126)](https://github.com/PennyLaneAI/pennylane/pull/3126)

* `qml.math.unwrap` no longer creates ragged arrays. Lists remain lists.
  [(#3163)](https://github.com/PennyLaneAI/pennylane/pull/3163)

* New `null.qubit` device. The `null.qubit`performs no operations or memory allocations. 
  [(#2589)](https://github.com/PennyLaneAI/pennylane/pull/2589)

* `ControlledQubitUnitary` now has a `control_values` property.
  [(#3206)](https://github.com/PennyLaneAI/pennylane/pull/3206)
  
<h3>Breaking changes</h3>

* `QuantumTape._par_info` is now a list of dictionaries, instead of a dictionary whose keys are integers starting from zero.
  [(#3185)](https://github.com/PennyLaneAI/pennylane/pull/3185)

* `QueuingContext` is renamed `QueuingManager`.
  [(#3061)](https://github.com/PennyLaneAI/pennylane/pull/3061)

* `QueuingManager.safe_update_info` and `AnnotatedQueue.safe_update_info` are deprecated. Instead, `update_info` no longer raises errors
   if the object isn't in the queue.

* Deprecation patches for the return types enum's location and `qml.utils.expand` are removed.
  [(#3092)](https://github.com/PennyLaneAI/pennylane/pull/3092)

* `_multi_dispatch` functionality has been moved inside the `get_interface` function. This function
  can now be called with one or multiple tensors as arguments.
  [(#3136)](https://github.com/PennyLaneAI/pennylane/pull/3136)

  ```pycon
  >>> torch_scalar = torch.tensor(1)
  >>> torch_tensor = torch.Tensor([2, 3, 4])
  >>> numpy_tensor = np.array([5, 6, 7])
  >>> qml.math.get_interface(torch_scalar)
  'torch'
  >>> qml.math.get_interface(numpy_tensor)
  'numpy'
  ```

  `_multi_dispatch` previously had only one argument which contained a list of the tensors to be
  dispatched:

  ```pycon
  >>> qml.math._multi_dispatch([torch_scalar, torch_tensor, numpy_tensor])
  'torch'
  ```

  To differentiate whether the user wants to get the interface of a single tensor or multiple
  tensors, `get_interface` now accepts a different argument per tensor to be dispatched:

  ```pycon
  >>> qml.math.get_interface(*[torch_scalar, torch_tensor, numpy_tensor])
  'torch'
  >>> qml.math.get_interface(torch_scalar, torch_tensor, numpy_tensor)
  'torch'
  ```

<h3>Deprecations</h3>

* `qml.tape.stop_recording` and `QuantumTape.stop_recording` are moved to `qml.QueuingManager.stop_recording`.
  The old functions will still be available untill v0.29.
  [(#3068)](https://github.com/PennyLaneAI/pennylane/pull/3068)

* `qml.tape.get_active_tape` is deprecated. Please use `qml.QueuingManager.active_context()` instead.
  [(#3068)](https://github.com/PennyLaneAI/pennylane/pull/3068)

<h3>Documentation</h3>

* The code block in the usage details of the UCCSD template is updated.
  [(#3140)](https://github.com/PennyLaneAI/pennylane/pull/3140)

<h3>Bug fixes</h3>

* `ControlledQubitUnitary.pow` now copies over the `control_values`.
  [(#3206)](https://github.com/PennyLaneAI/pennylane/pull/3206)

* The evaluation of QNodes that return either `vn_entropy` or `mutual_info` raises an
  informative error message when using devices that define a vector of shots.
  [(#3180)](https://github.com/PennyLaneAI/pennylane/pull/3180)

* Fixed a bug that made `qml.AmplitudeEmbedding` incompatible with JITting.
  [(#3166)](https://github.com/PennyLaneAI/pennylane/pull/3166)

* Fixed the `qml.transforms.transpile` transform to work correctly for all two-qubit operations.
  [(#3104)](https://github.com/PennyLaneAI/pennylane/pull/3104)

* Fixed a bug with the control values of a controlled version of a `ControlledQubitUnitary`.
  [(#3119)](https://github.com/PennyLaneAI/pennylane/pull/3119)

* Fixed a bug where `qml.math.fidelity(non_trainable_state, trainable_state)` failed unexpectedly.
  [(#3160)](https://github.com/PennyLaneAI/pennylane/pull/3160)

* Fixed a bug where `qml.QueuingManager.stop_recording` did not clean up if yielded code raises an exception.
  [(#3182)](https://github.com/PennyLaneAI/pennylane/pull/3182)

* Fixed a bug where `op.eigvals()` would return an incorrect result if the operator was a non-hermitian 
  composite operator.
  [(#3204)](https://github.com/PennyLaneAI/pennylane/pull/3204)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso-Linaje,
Juan Miguel Arrazola,
Albert Mitjans Coma,
Utkarsh Azad,
Amintor Dusko,
Diego Guala,
Soran Jahangiri,
Christina Lee,
Lee J. O'Riordan,
Mudit Pandey,
Matthew Silverman,
Jay Soni,
Antal Száva,
David Wierichs,
