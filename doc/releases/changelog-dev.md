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
<<<<<<< HEAD
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
=======
  >>> dev = qml.device("default.qubit", wires=1)
  >>> def circuit(params):
  ...     qml.RX(params[0], wires=0)
  ...     qml.RY(params[1], wires=0)
  >>> coeffs = [1, 1]
  >>> obs = [qml.PauliX(0), qml.PauliZ(0)]
  >>> H = qml.Hamiltonian(coeffs, obs)
  >>> @qml.qnode(dev)
  ... def cost(params):
  ...     circuit(params)
  ...     return qml.expval(H)
  >>> params = np.random.normal(0, np.pi, (2), requires_grad=True)
  >>> print(params)
  [-5.92774911 -4.26420843]
  >>> print(cost(params))
  0.43866366253270167
  >>> max_iterations = 50
  >>> opt = qml.SPSAOptimizer(maxiter=max_iterations)
  >>> for _ in range(max_iterations):
  ...     params, energy = opt.step_and_cost(cost, params)
  >>> print(params)
  [-6.21193761 -2.99360548]
  >>> print(energy)
  -1.1258709813834058
  ```

* Differentiable zero-noise-extrapolation error mitigation via ``qml.transforms.mitigate_with_zne`` with ``qml.transforms.fold_global`` and ``qml.transforms.poly_extrapolate``.
  [(#2757)](https://github.com/PennyLaneAI/pennylane/pull/2757)

  When using a noisy or real device, you can now create a differentiable mitigated qnode that internally executes folded circuits that increase the noise and extrapolating with a polynomial fit back to zero noise. There will be an accompanying demo on this, see [(PennyLaneAI/qml/529)](https://github.com/PennyLaneAI/qml/pull/529).

  ```python
  # Describe noise
  noise_gate = qml.DepolarizingChannel
  noise_strength = 0.1

  # Load devices
  dev_ideal = qml.device("default.mixed", wires=n_wires)
  dev_noisy = qml.transforms.insert(noise_gate, noise_strength)(dev_ideal)

  scale_factors = [1, 2, 3]
  @mitigate_with_zne(
    scale_factors,
    qml.transforms.fold_global,
    qml.transforms.poly_extrapolate,
    extrapolate_kwargs={'order': 2}
  )
  @qml.qnode(dev_noisy)
  def qnode_mitigated(theta):
      qml.RY(theta, wires=0)
      return qml.expval(qml.PauliX(0))

  theta = np.array(0.5, requires_grad=True)
  grad = qml.grad(qnode_mitigated)
  >>> grad(theta)
  0.5712737447327619
  ```

* The quantum information module now supports computation of relative entropy.
  [(#2772)](https://github.com/PennyLaneAI/pennylane/pull/2772)

  It includes a function in `qml.math`:

  ```pycon
  >>> rho = np.array([[0.3, 0], [0, 0.7]])
  >>> sigma = np.array([[0.5, 0], [0, 0.5]])
  >>> qml.math.relative_entropy(rho, sigma)
  tensor(0.08228288, requires_grad=True)
  ```

  as well as a QNode transform:

  ```python
  dev = qml.device('default.qubit', wires=2)

  @qml.qnode(dev)
  def circuit(param):
      qml.RY(param, wires=0)
      qml.CNOT(wires=[0, 1])
      return qml.state()
  ```

  ```pycon
  >>> relative_entropy_circuit = qml.qinfo.relative_entropy(circuit, circuit, wires0=[0], wires1=[0])
  >>> x, y = np.array(0.4), np.array(0.6)
  >>> relative_entropy_circuit((x,), (y,))
  0.017750012490703237
  ```

* New PennyLane-inspired `sketch` and `sketch_dark` styles are now available for drawing circuit diagram graphics.
  [(#2709)](https://github.com/PennyLaneAI/pennylane/pull/2709)

* Added `QutritDevice` as an abstract base class for qutrit devices.
  ([#2781](https://github.com/PennyLaneAI/pennylane/pull/2781), [#2782](https://github.com/PennyLaneAI/pennylane/pull/2782))
* Added operation `qml.QutritUnitary` for applying user-specified unitary operations on qutrit devices.
  [(#2699)](https://github.com/PennyLaneAI/pennylane/pull/2699)
* Added `default.qutrit` plugin for pure state simulation of qutrits. Currently supports operation `qml.QutritUnitary` and measurements `qml.state()`, `qml.probs()`.
  [(#2783)](https://github.com/PennyLaneAI/pennylane/pull/2783)

  ```pycon
  >>> dev = qml.device("default.qutrit", wires=1)
  >>> @qml.qnode(dev)
  ... def circuit(U):
  ...     qml.QutritUnitary(U, wires=0)
  ...     return qml.probs(wires=0)
  >>> U = np.array([[1, 1, 0], [1, -1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)
  >>> print(circuit(U))
  [0.5 0.5 0. ]
  ```

  * Added `qml.THermitian` observable for using user-specified hermitian matrix as an observable for qutrit circuits.
  ([#2784](https://github.com/PennyLaneAI/pennylane/pull/2784))
  * Added `qml.TShift` operation for qutrit devices, which is the generalized analog of the Pauli X operation.
  ([#2840](https://github.com/PennyLaneAI/pennylane/pull/2840))
  * Added `qml.Clock` operation for qutrit devices, which is the generalized analog of the Pauli Z operation.
  ([#2841](https://github.com/PennyLaneAI/pennylane/pull/2841))
  * Added `qml.TAdd` operation for qutrit devices, which is the generalized analog of the CX operation.
  ([#2842](https://github.com/PennyLaneAI/pennylane/pull/2842))
  * Added `qml.TSWAP` operation for qutrit devices, which swaps the state between two wires.
  ([#2843](https://github.com/PennyLaneAI/pennylane/pull/2843))
  * Added `qml.ControlledQutritUnitary` operation for qutrit devices, which allows users to apply a controlled arbitrary unitary operation.
  ([#2844](https://github.com/PennyLaneAI/pennylane/pull/2844))
  * Added `TRX()` operation, which applies an X rotation to a subspace specified by the user. The subspace determines which 2 of 3 one-qutrit basis states the operation applies to. Updated `pennylane/qnode.py` to support parameter shift differentiation on qutrit devices.
  ([#2845](https://github.com/PennyLaneAI/pennylane/pull/2845))

**Operator Arithmetic:**

* Adds a base class `qml.ops.op_math.SymbolicOp` for single-operator symbolic
  operators such as `Adjoint` and `Pow`.
  [(#2721)](https://github.com/PennyLaneAI/pennylane/pull/2721)

* A `Sum` symbolic class is added that allows users to represent the sum of operators.
  [(#2475)](https://github.com/PennyLaneAI/pennylane/pull/2475)

  The `Sum` class provides functionality like any other PennyLane operator. We can
  get the matrix, eigenvalues, terms, diagonalizing gates and more.

  ```pycon
  >>> summed_op = qml.op_sum(qml.PauliX(0), qml.PauliZ(0))
  >>> summed_op
  PauliX(wires=[0]) + PauliZ(wires=[0])
  >>> qml.matrix(summed_op)
  array([[ 1,  1],
         [ 1, -1]])
  >>> summed_op.terms()
  ([1.0, 1.0], (PauliX(wires=[0]), PauliZ(wires=[0])))
  ```

  The `summed_op` can also be used inside a `qnode` as an observable.
  If the circuit is parameterized, then we can also differentiate through the
  sum observable.

  ```python
  sum_op = Sum(qml.PauliX(0), qml.PauliZ(1))
  dev = qml.device("default.qubit", wires=2)

  @qml.qnode(dev, grad_method="best")
  def circuit(weights):
        qml.RX(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(weights[2], wires=1)
        return qml.expval(sum_op)
  ```
>>>>>>> Update changelog

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

  ```pycon
  >>> dev = qml.device("default.qubit", wires=2, shots=1000)
  >>>
  >>> @qml.qnode(dev)
  >>> def circuit():
  ...   qml.Hadamard(wires=0)
  ...   qml.CNOT(wires=[0, 1])
  ...   return qml.sample(qml.PauliZ(0), counts=True), qml.sample(qml.PauliZ(1), counts=True)
  >>> result = circuit()
  >>> print(result)
  [tensor({-1: 526, 1: 474}, dtype=object, requires_grad=True)
   tensor({-1: 526, 1: 474}, dtype=object, requires_grad=True)]
  ```

* The `qml.state` and `qml.density_matrix` measurements now support custom wire
  labels.
  [(#2779)](https://github.com/PennyLaneAI/pennylane/pull/2779)

* Modified the representation of `WireCut` by using `qml.draw_mpl`.
  [(#3067)](https://github.com/PennyLaneAI/pennylane/pull/3067)

<h3>Breaking changes</h3>

 * `QueuingContext` is renamed `QueuingManager`.
  [(#3061)](https://github.com/PennyLaneAI/pennylane/pull/3061)

 * `QueuingManager.safe_update_info` and `AnnotatedQueue.safe_update_info` are deprecated. Instead, `update_info` no longer raises errors
   if the object isn't in the queue.

<h3>Deprecations</h3>

* `qml.tape.stop_recording` and `QuantumTape.stop_recording` are moved to `qml.QueuingManager.stop_recording`.
  The old functions will still be available untill v0.29.
  [(#3068)](https://github.com/PennyLaneAI/pennylane/pull/3068)

* `qml.tape.get_active_tape` is deprecated. Please use `qml.QueuingManager.active_context()` instead.
  [(#3068)](https://github.com/PennyLaneAI/pennylane/pull/3068)

<h3>Documentation</h3>

<h3>Bug fixes</h3>

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
