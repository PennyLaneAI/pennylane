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
  
* Added the `qml.ControlledQutritUnitary` qutrit operation for applying a controlled arbitrary unitary matrix to the specified set of wires.
Users can specify the control wires as well as the values to control the operation on.
  ([#2844](https://github.com/PennyLaneAI/pennylane/pull/2844))

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

<h4>Pauli Module</h4>

* Re-organized and grouped all functions in PennyLane responsible for manipulation of Pauli operators into a `pauli` 
  module. Deprecated the `grouping` module and moved logic from `pennylane/grouping` to `pennylane/pauli/grouping`.
  [(#3179)](https://github.com/PennyLaneAI/pennylane/pull/3179)

* The `IntegerComparator` arithmetic operation is now available.
[(#3113)](https://github.com/PennyLaneAI/pennylane/pull/3113)

  Given a basis state :math:`\vert n \rangle`, where :math:`n` is a positive integer, and a fixed positive
  integer :math:`L`, the `IntegerComparator` operator flips a target qubit if :math:`n \geq L`. 
  Alternatively, the flipping condition can be :math:`n < L`. This is accessed via the `geq` keyword
  argument.

  ```python
  dev = qml.device("default.qubit", wires=2)

  @qml.qnode(dev)
  def circuit():
      qml.BasisState(np.array([0, 1]), wires=range(2))
      qml.broadcast(qml.Hadamard, wires=range(2), pattern='single')
      qml.IntegerComparator(2, geq=False, wires=[0, 1])
      return qml.state()
  ```

  ```pycon
  >>> circuit()
  [-0.5+0.j  0.5+0.j -0.5+0.j  0.5+0.j]
  ```

* The `QNode` class now accepts an ``auto`` interface, which automatically detects the interface
  of the given input.
  [(#3132)](https://github.com/PennyLaneAI/pennylane/pull/3132)

  We can therefore execute the same parametrized QNode with parameters from different interfaces,
  and the class will automatically detect the interface:

  ```python
  dev = qml.device("default.qubit", wires=2)
  @qml.qnode(dev, interface="auto")
  def circuit(weight):
      qml.RX(weight[0], wires=0)
      qml.RY(weight[1], wires=1)
      return qml.expval(qml.PauliZ(0))

  interface_tensors = [[0, 1], np.array([0, 1]), torch.Tensor([0, 1]), tf.Variable([0, 1], dtype=float), jnp.array([0, 1])]
  for tensor in interface_tensors:
      res = circuit(weight=tensor)
      print(f"Result value: {res:.2f}; Result type: {type(res)}")
  ```

  ```pycon
  Result value: 1.00; Result type: <class 'pennylane.numpy.tensor.tensor'>
  Result value: 1.00; Result type: <class 'pennylane.numpy.tensor.tensor'>
  Result value: 1.00; Result type: <class 'torch.Tensor'>
  Result value: 1.00; Result type: <class 'tensorflow.python.framework.ops.EagerTensor'>
  Result value: 1.00; Result type: <class 'jaxlib.xla_extension.DeviceArray'>
  ```

* Added the `qml.map_wires` function, that changes the wires of the given operator, `QNode`, queue
  or quantum function according to the given wire map.
  [(#3145)](https://github.com/PennyLaneAI/pennylane/pull/3145)

  Using `qml.map_wires` with an operator:

  ```pycon
  >>> op = qml.RX(0.54, wires=0) + qml.PauliX(1) + (qml.PauliZ(2) @ qml.RY(1.23, wires=3))
  >>> op
  (RX(0.54, wires=[0]) + PauliX(wires=[1])) + (PauliZ(wires=[2]) @ RY(1.23, wires=[3]))
  >>> wire_map = {0: 10, 1: 11, 2: 12, 3: 13}
  >>> qml.map_wires(op, wire_map)
  (RX(0.54, wires=[10]) + PauliX(wires=[11])) + (PauliZ(wires=[12]) @ RY(1.23, wires=[13]))
  ```

  Using `qml.map_wires` with a `QNode`:

  ```pycon
  >>> dev = qml.device("default.qubit", wires=[10, 11, 12, 13])
  >>> @qml.qnode(dev)
  ... def circuit():
  ...     qml.RX(0.54, wires=0)
  ...     qml.PauliX(1)
  ...     qml.PauliZ(2)
  ...     qml.RY(1.23, wires=3)
  ...     return qml.probs(wires=0)
  >>> mapped_circuit = qml.map_wires(circuit, wire_map)
  >>> mapped_circuit()
  tensor([0.92885434, 0.07114566], requires_grad=True)
  >>> print(qml.draw(mapped_circuit)())
  10: ──RX(0.54)─┤  Probs
  11: ──X────────┤       
  12: ──Z────────┤       
  13: ──RY(1.23)─┤  
  ```

* An optimizer is added for building and optimizing quantum circuits adaptively.
  [(#3192)](https://github.com/PennyLaneAI/pennylane/pull/3192)

  The new optimizer, ``AdaptiveOptimizer``, takes an initial circuit and a collection of operators
  as input and adds a selected gate to the circuits at each optimization step. The process of
  growing the circuit can be repeated until the circuit gradients converge to zero within a given
  threshold. The adaptive optimizer can be used to implement algorithms such as ``ADAPT-VQE`` as
  shown in the following example.

  First, the molecule is defined and the Hamiltonian is computed:

  ```python
  symbols = ["H", "H", "H"]
  geometry = np.array([[0.01076341, 0.04449877, 0.0],
                       [0.98729513, 1.63059094, 0.0],
                       [1.87262415, -0.00815842, 0.0]], requires_grad=False)
  H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, charge = 1)
  ```

  The collection of gates to grow the circuit is built to contain all single and double excitations:

  ```python
  n_electrons = 2
  singles, doubles = qml.qchem.excitations(n_electrons, qubits)
  singles_excitations = [qml.SingleExcitation(0.0, x) for x in singles]
  doubles_excitations = [qml.DoubleExcitation(0.0, x) for x in doubles]
  operator_pool = doubles_excitations + singles_excitations
  ```

  An initial circuit that prepares a Hartree-Fock state and returns the expectation value of the
  Hamiltonian is defined:

  ```python
  hf_state = qml.qchem.hf_state(n_electrons, qubits)
  dev = qml.device("default.qubit", wires=qubits)
  @qml.qnode(dev)
  def circuit():
      qml.BasisState(hf_state, wires=range(qubits))
      return qml.expval(H)
  ```

  Finally, the optimizer is instantiated and then the circuit is created and optimized adaptively:

  ```python
  opt = qml.optimize.AdaptiveOptimizer()
  for i in range(len(operator_pool)):
      circuit, energy, gradient = opt.step_and_cost(circuit, operator_pool, drain_pool=True)
      print('Energy:', energy)
      print(qml.draw(circuit)())
      print('Largest Gradient:', gradient)
      print()
      if gradient < 1e-3:
          break
  ```
  
   ```pycon
  Energy: -1.246549938420637
  0: ─╭BasisState(M0)─╭G²(0.20)─┤ ╭<𝓗>
  1: ─├BasisState(M0)─├G²(0.20)─┤ ├<𝓗>
  2: ─├BasisState(M0)─│─────────┤ ├<𝓗>
  3: ─├BasisState(M0)─│─────────┤ ├<𝓗>
  4: ─├BasisState(M0)─├G²(0.20)─┤ ├<𝓗>
  5: ─╰BasisState(M0)─╰G²(0.20)─┤ ╰<𝓗>
  Largest Gradient: 0.14399872776755085

  Energy: -1.2613740231529604
  0: ─╭BasisState(M0)─╭G²(0.20)─╭G²(0.19)─┤ ╭<𝓗>
  1: ─├BasisState(M0)─├G²(0.20)─├G²(0.19)─┤ ├<𝓗>
  2: ─├BasisState(M0)─│─────────├G²(0.19)─┤ ├<𝓗>
  3: ─├BasisState(M0)─│─────────╰G²(0.19)─┤ ├<𝓗>
  4: ─├BasisState(M0)─├G²(0.20)───────────┤ ├<𝓗>
  5: ─╰BasisState(M0)─╰G²(0.20)───────────┤ ╰<𝓗>
  Largest Gradient: 0.1349349562423238

  Energy: -1.2743971719780331
  0: ─╭BasisState(M0)─╭G²(0.20)─╭G²(0.19)──────────┤ ╭<𝓗>
  1: ─├BasisState(M0)─├G²(0.20)─├G²(0.19)─╭G(0.00)─┤ ├<𝓗>
  2: ─├BasisState(M0)─│─────────├G²(0.19)─│────────┤ ├<𝓗>
  3: ─├BasisState(M0)─│─────────╰G²(0.19)─╰G(0.00)─┤ ├<𝓗>
  4: ─├BasisState(M0)─├G²(0.20)────────────────────┤ ├<𝓗>
  5: ─╰BasisState(M0)─╰G²(0.20)────────────────────┤ ╰<𝓗>
  Largest Gradient: 0.00040841755397108586
  ``` 

<h3>Improvements</h3>

* Added the `samples_computational_basis` attribute to the `MeasurementProcess` and `QuantumScript` classes to track
  if computational basis samples are being generated.
  [(#3207)](https://github.com/PennyLaneAI/pennylane/pull/3207)

* The parameters of a basis set containing different number of Gaussian functions are easier to 
  differentiate.
  [(#3213)](https://github.com/PennyLaneAI/pennylane/pull/3213)

* Printing `MultiControlledX` now shows the `control_values`.
[(#3113)](https://github.com/PennyLaneAI/pennylane/pull/3113)

* The matrix passed to `qml.Hermitian` is validated when creating the observable if the input is not abstract.
  [(#3181)](https://github.com/PennyLaneAI/pennylane/pull/3181)

* Added a new `pennylane.tape.QuantumScript` class that contains all the non-queuing behavior of `QuantumTape`. Now `QuantumTape` inherits from `QuantumScript` as well
  as `AnnotatedQueue`.
  This is a developer-facing change, and users should not manipulate `QuantumScript` directly.  Instead, they
  should continue to rely on `QNode`s.
  [(#3097)](https://github.com/PennyLaneAI/pennylane/pull/3097)

* `qml.simplify`, `op_tranform`'s like `qml.matrix`, `batch_transform`, `hamiltonian_expand` and `split_non_commuting` now work with
  `QuantumScript` as well as `QuantumTape`.
  [(#3209)](https://github.com/PennyLaneAI/pennylane/pull/3209)

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
  ```

* Added `overlapping_ops` property to the `Composite` class to improve the
  performance of the `eigvals`, `diagonalizing_gates` and `Prod.matrix` methods.
  [(#3084)](https://github.com/PennyLaneAI/pennylane/pull/3084)

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

* `default.qubit` favours decomposition and avoids matrix construction for `QFT` and `GroverOperator` at larger qubit numbers.
  [(#3193)](https://github.com/PennyLaneAI/pennylane/pull/3193)

* `ControlledQubitUnitary` now has a `control_values` property.
  [(#3206)](https://github.com/PennyLaneAI/pennylane/pull/3206)

* Remove `_wires` properties and setters from the `ControlledClass` and the `SymbolicClass`.
  Stop using `op._wires = new_wires`, use `qml.map_wires(op, wire_map=dict(zip(op.wires, new_wires)))`
  instead.
  [(#3186)](https://github.com/PennyLaneAI/pennylane/pull/3186)

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

* `Operator.compute_terms` is removed. On a specific instance of an operator, `op.terms()` can be used
  instead. There is no longer a static method for this.
  [(#3215)](https://github.com/PennyLaneAI/pennylane/pull/3215)

<h3>Deprecations</h3>

* `qml.tape.stop_recording` and `QuantumTape.stop_recording` are moved to `qml.QueuingManager.stop_recording`.
  The old functions will still be available untill v0.29.
  [(#3068)](https://github.com/PennyLaneAI/pennylane/pull/3068)

* `qml.tape.get_active_tape` is deprecated. Please use `qml.QueuingManager.active_context()` instead.
  [(#3068)](https://github.com/PennyLaneAI/pennylane/pull/3068)

* Deprecate `qml.transforms.qcut.remap_tape_wires`. Use `qml.map_wires` instead.
  [(#3186)](https://github.com/PennyLaneAI/pennylane/pull/3186)

<h3>Documentation</h3>

* The code block in the usage details of the UCCSD template is updated.
  [(#3140)](https://github.com/PennyLaneAI/pennylane/pull/3140)

* Added a "Deprecations" page to the developer documentation.
  [(#3093)](https://github.com/PennyLaneAI/pennylane/pull/3093)

* The example of the `FlipSign` template is updated.
  [(#3219)](https://github.com/PennyLaneAI/pennylane/pull/3219)

<h3>Bug fixes</h3>

* Fixed a bug where `qml.sample()` and `qml.counts()` would output incorrect results when mixed with measurements whose
  operators do not qubit-wise commute with computational basis projectors.
  [(#3207)](https://github.com/PennyLaneAI/pennylane/pull/3207)

* Users no longer see unintuitive errors when inputing sequences to `qml.Hermitian`.
  [(#3181)](https://github.com/PennyLaneAI/pennylane/pull/3181)

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

* Returning `qml.sample()` or `qml.counts()` with other measurements of non-commuting observables
  now raises a QuantumFunctionError (e.g., `return qml.expval(PauliX(wires=0)), qml.sample()`
  now raises an error).
  [(#2924)](https://github.com/PennyLaneAI/pennylane/pull/2924)

* Fixed a bug where `op.eigvals()` would return an incorrect result if the operator was a non-hermitian
  composite operator.
  [(#3204)](https://github.com/PennyLaneAI/pennylane/pull/3204)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso-Linaje,
Juan Miguel Arrazola,
Albert Mitjans Coma,
Utkarsh Azad,
Isaac De Vlugt,
Amintor Dusko,
Lillian M. A. Frederiksen,
Diego Guala,
Soran Jahangiri,
Christina Lee,
Lee J. O'Riordan,
Mudit Pandey,
Matthew Silverman,
Jay Soni,
Antal Száva,
David Wierichs,
