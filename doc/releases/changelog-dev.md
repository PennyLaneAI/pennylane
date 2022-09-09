:orphan:

# Release 0.26.0-dev (development release)

<h3>New features since last release</h3>

* Embedding templates now support parameter broadcasting.
  [(#2810)](https://github.com/PennyLaneAI/pennylane/pull/2810)

  Embedding templates like `AmplitudeEmbedding` or `IQPEmbedding` now support
  parameter broadcasting with a leading broadcasting dimension in their variational
  parameters. `AmplitudeEmbedding`, for example, would usually use a one-dimensional input
  vector of features. With broadcasting, we now also can compute

  ```pycon
  >>> features = np.array([
  ...     [0.5, 0.5, 0., 0., 0.5, 0., 0.5, 0.],
  ...     [1., 0., 0., 0., 0., 0., 0., 0.],
  ...     [0.5, 0.5, 0., 0., 0., 0., 0.5, 0.5],
  ... ])
  >>> op = qml.AmplitudeEmbedding(features, wires=[1, 5, 2])
  >>> op.batch_size
  3
  ```

  An exception is `BasisEmbedding`, which is not broadcastable.

* Added `QutritDevice` as an abstract base class for qutrit devices.
  [#2781](https://github.com/PennyLaneAI/pennylane/pull/2781)
  [#2782](https://github.com/PennyLaneAI/pennylane/pull/2782)

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

* Added `qml.THermitian` observable for measuring user-specified Hermitian matrix observables for qutrit circuits.
  ([#2784](https://github.com/PennyLaneAI/pennylane/pull/2784))

* Added the `qml.TShift` and `qml.TClock` qutrit operations for qutrit devices, which are the qutrit analogs of the Pauli X and Pauli Z operations.
  ([#2841](https://github.com/PennyLaneAI/pennylane/pull/2841))

* Added the private `_prod_sort` function that sorts a list of operators by their respective wires
  taking into account their commutativity property.
  [(#2995)](https://github.com/PennyLaneAI/pennylane/pull/2995)

**Classical shadows**

* Added the `qml.classical_shadow` measurement process that can now be returned from QNodes.
  [(#2820)](https://github.com/PennyLaneAI/pennylane/pull/2820)

  The measurement protocol is described in detail in the
  [classical shadows paper](https://arxiv.org/abs/2002.08953). Calling the QNode
  will return the randomized Pauli measurements (the `recipes`) that are performed
  for each qubit, identified as a unique integer:

  * 0 for Pauli X
  * 1 for Pauli Y
  * 2 for Pauli Z

  It also returns the measurement results (the `bits`), which is `0` if the 1 eigenvalue
  is sampled, and `1` if the -1 eigenvalue is sampled.

  For example,

  ```python
  dev = qml.device("default.qubit", wires=2, shots=5)

  @qml.qnode(dev)
  def circuit():
      qml.Hadamard(wires=0)
      qml.CNOT(wires=[0, 1])
      return qml.classical_shadow(wires=[0, 1])
  ```

  ```pycon
  >>> bits, recipes = circuit()
  tensor([[0, 0],
          [1, 0],
          [1, 0],
          [0, 0],
          [0, 1]], dtype=uint8, requires_grad=True)
  >>> recipes
  tensor([[2, 2],
          [0, 2],
          [1, 0],
          [0, 2],
          [0, 2]], dtype=uint8, requires_grad=True)
  ```

* Added the ``shadow_expval`` measurement for differentiable expectation value estimation using classical shadows.
  [(#2871)](https://github.com/PennyLaneAI/pennylane/pull/2871)

  ```python
  H = qml.Hamiltonian([1., 1.], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0) @ qml.PauliX(1)])

  dev = qml.device("default.qubit", wires=range(2), shots=10000)
  @qml.qnode(dev)
  def qnode(x, H):
      qml.Hadamard(0)
      qml.CNOT((0,1))
      qml.RX(x, wires=0)
      return qml.shadow_expval(H)

  x = np.array(0.5, requires_grad=True)

  print(qnode(x, H), qml.grad(qnode)(x, H))
  ```

* Added the `qml.shadows.shadow_expval` and `qml.shadows.shadow_state` QNode transforms for
  computing expectation values and states from a classical shadow measurement. These transforms
  are fully differentiable.
  [(#2968)](https://github.com/PennyLaneAI/pennylane/pull/2968)

  ```python
  dev = qml.device("default.qubit", wires=1, shots=1000)

  @qml.qnode(dev)
  def circuit(x):
      qml.RY(x, wires=0)
      return qml.classical_shadow(wires=[0])
  ```

  ```pycon
  >>> x = np.array(1.2)
  >>> expval_circuit = qml.shadows.shadow_expval(qml.PauliZ(0))(circuit)
  >>> expval_circuit(x)
  tensor(0.282, requires_grad=True)
  >>> qml.grad(expval_circuit)(x)
  -1.0439999999999996
  ```
  ```pycon
  >>> state_circuit = qml.shadows.shadow_state(wires=[0], diffable=True)(circuit)
  >>> state_circuit(x)
  tensor([[0.7055+0.j    , 0.447 +0.0075j],
          [0.447 -0.0075j, 0.2945+0.j    ]], requires_grad=True)
  >>> qml.jacobian(lambda x: np.real(state_circuit(x)))(x)
  array([[-0.477,  0.162],
         [ 0.162,  0.477]])
  ```

* `expand_matrix()` method now allows the sparse matrix representation of an operator to be extended to
  a larger hilbert space.
  [(#2998)](https://github.com/PennyLaneAI/pennylane/pull/2998)

  ```pycon
  >>> from scipy import sparse
  >>> mat = sparse.csr_matrix([[0, 1], [1, 0]])
  >>> qml.math.expand_matrix(mat, wires=[1], wire_order=[0,1]).toarray()
  array([[0., 1., 0., 0.],
         [1., 0., 0., 0.],
         [0., 0., 0., 1.],
         [0., 0., 1., 0.]])
  ```

* `qml.exp` exponentiates an Operator.  An optional scalar coefficient can multiply the
  Operator before exponentiation. Internally, this constructor functions creates the new
  class `qml.ops.op_math.Exp`.
  [(#2799)](https://github.com/PennyLaneAI/pennylane/pull/2799)

  The function can be used to create either observables or generic rotation gates:

  ```pycon
  >>> obs = qml.exp(qml.PauliX(0), 3)
  >>> qml.is_hermitian(obs)
  True
  >>> x = 1.234
  >>> t = qml.PauliX(0) @ qml.PauliX(1) + qml.PauliY(0) @ qml.PauliY(1)
  >>> isingxy = qml.exp(t, 0.25j * x)
  >>> qml.math.allclose(isingxy.matrix(), qml.IsingXY(x, wires=(0,1)).matrix())
  True
  >>> qml.is_unitary(isingxy)
  True
  ```

<h3>Improvements</h3>

* Some methods of the `QuantumTape` class have been simplified and reordered to
  improve both readability and performance. The `Wires.all_wires` method has been rewritten
  to improve performance.
  [(#2963)](https://github.com/PennyLaneAI/pennylane/pull/2963)

* The `qml.qchem.molecular_hamiltonian` function is modified to support observable grouping.
  [(#2997)](https://github.com/PennyLaneAI/pennylane/pull/2997)

* `qml.ops.op_math.Controlled` now has basic decomposition functionality.
  [(#2938)](https://github.com/PennyLaneAI/pennylane/pull/2938)

* Automatic circuit cutting is improved by making better partition imbalance derivations.
  Now it is more likely to generate optimal cuts for larger circuits.
  [(#2517)](https://github.com/PennyLaneAI/pennylane/pull/2517)

* Added `PSWAP` operator.
  [(#2667)](https://github.com/PennyLaneAI/pennylane/pull/2667)

* The `qml.simplify` method can now simplify parametrized operations.
  [(#3012)](https://github.com/PennyLaneAI/pennylane/pull/3012)

  ```pycon
  >>> op1 = qml.RX(30.0, wires=0)
  >>> qml.simplify(op1)
  RX(4.867258771281655, wires=[0])
  >>> op2 = qml.Rot(np.pi / 2, 5.0, -np.pi / 2, wires=0)
  >>> qml.simplify(op2)
  RX(5.0, wires=[0])
  >>> op3 = qml.RX(4 * np.pi, wires=0)
  >>> qml.simplify(op3)
  Identity(wires=[0])
  ```

* The `qml.simplify` method now can compute the adjoint and power of specific operators.
  [(#2922)](https://github.com/PennyLaneAI/pennylane/pull/2922)

  ```pycon
  >>> adj_op = qml.adjoint(qml.RX(1, 0))
  >>> qml.simplify(adj_op)
  RX(-1, wires=[0])
  ```

* Added `sparse_matrix()` support for single qubit observables
  [(#2964)](https://github.com/PennyLaneAI/pennylane/pull/2964)

* Added the `qml.is_hermitian` and `qml.is_unitary` function checks.
  [(#2960)](https://github.com/PennyLaneAI/pennylane/pull/2960)

  ```pycon
  >>> op = qml.PauliX(wires=0)
  >>> qml.is_hermitian(op)
  True
  >>> op2 = qml.RX(0.54, wires=0)
  >>> qml.is_hermitian(op2)
  False
  ```

* Per default, counts returns only the outcomes observed in sampling. Optionally, specifying `qml.counts(all_outcomes=True)`
  will return a dictionary containing all possible outcomes. [(#2889)](https://github.com/PennyLaneAI/pennylane/pull/2889)
  
  ```pycon
  >>> dev = qml.device("default.qubit", wires=2, shots=1000)
  >>>
  >>> @qml.qnode(dev)
  >>> def circuit():
  ...     qml.Hadamard(wires=0)
  ...     qml.CNOT(wires=[0, 1])
  ...     return qml.counts(all_outcomes=True)
  >>> result = circuit()
  >>> print(result)
  {'00': 495, '01': 0, '10': 0,  '11': 505}
  ```
  
* Internal use of in-place inversion is eliminated in preparation for its deprecation.
  [(#2965)](https://github.com/PennyLaneAI/pennylane/pull/2965)

* `qml.is_commuting` is moved to `pennylane/ops/functions` from `pennylane/transforms/commutation_dag.py`.
  [(#2991)](https://github.com/PennyLaneAI/pennylane/pull/2991)

* `qml.simplify` can now be used to simplify quantum functions, tapes and QNode objects.
  [(#2978)](https://github.com/PennyLaneAI/pennylane/pull/2978)

  ```python
    dev = qml.device("default.qubit", wires=2)
    @qml.simplify
    @qml.qnode(dev)
    def circuit():
      qml.adjoint(qml.prod(qml.RX(1, 0) ** 1, qml.RY(1, 0), qml.RZ(1, 0)))
      return qml.probs(wires=0)
  ```

  ```pycon
  >>> circuit()
  >>> list(circuit.tape)
  [RZ(-1, wires=[0]) @ RY(-1, wires=[0]) @ RX(-1, wires=[0]), probs(wires=[0])]
  ```

* Added functionality to `qml.simplify` to allow for grouping of like terms in a sum, resolve
  products of pauli operators and combine rotation angles of identical rotation gates.
  [(#2982)](https://github.com/PennyLaneAI/pennylane/pull/2982)

  ```pycon
  >>> qml.simplify(qml.prod(qml.PauliX(0), qml.PauliY(1), qml.PauliX(0), qml.PauliY(1)))
  Identity(wires=[0]) @ Identity(wires=[1])
  >>> qml.simplify(qml.op_sum(qml.PauliX(0), qml.PauliY(1), qml.PauliX(0), qml.PauliY(1)))
  2*(PauliX(wires=[0])) + 2*(PauliY(wires=[1]))
  >>> qml.simplify(qml.prod(qml.RZ(1, 0), qml.RZ(1, 0)))
  RZ(2, wires=[0])
  ```

* `Controlled` operators now work with `qml.is_commuting`.
  [(#2994)](https://github.com/PennyLaneAI/pennylane/pull/2994)

* `Prod` and `Sum` class now support the `sparse_matrix()` method.
  [(#3006)](https://github.com/PennyLaneAI/pennylane/pull/3006)

  ```pycon
  >>> xy = qml.prod(qml.PauliX(1), qml.PauliY(1))
  >>> op = qml.op_sum(xy, qml.Identity(0))
  >>>
  >>> sparse_mat = op.sparse_matrix(wire_order=[0,1])
  >>> type(sparse_mat)
  <class 'scipy.sparse.csr.csr_matrix'>
  >>> print(sparse_mat.toarray())
  [[1.+1.j 0.+0.j 0.+0.j 0.+0.j]
  [0.+0.j 1.-1.j 0.+0.j 0.+0.j]
  [0.+0.j 0.+0.j 1.+1.j 0.+0.j]
  [0.+0.j 0.+0.j 0.+0.j 1.-1.j]]
  ```

* `qml.Barrier` with `only_visual=True` now simplifies, via `op.simplify()` to the identity
  or a product of identities.
  [(#3016)](https://github.com/PennyLaneAI/pennylane/pull/3016)

* `__repr__` and `label` methods are more correct and meaningful for Operators with an arithmetic
  depth greater than 0. The `__repr__` for `Controlled` show `control_wires` instead of `wires`.
  [(#3013)](https://github.com/PennyLaneAI/pennylane/pull/3013)

* Use `Operator.hash` instead of `Operator.matrix` to cache the eigendecomposition results in `Prod` and
  `Sum` classes. When `Prod` and `Sum` operators have no overlapping wires, compute the eigenvalues
  and the diagonalising gates using the factors/summands instead of using the full matrix.
  [(#3022)](https://github.com/PennyLaneAI/pennylane/pull/3022)

* When computing the (sparse) matrix for `Prod` and `Sum` classes, move the matrix expansion using
  the `wire_order` to the end to avoid computing unnecessary sums and products of huge matrices.
  [(#3030)](https://github.com/PennyLaneAI/pennylane/pull/3030)

* `qml.grouping.is_pauli_word` now returns `False` for operators that don't inherit from `qml.Observable`, instead of raising an error.
  [(#3039)](https://github.com/PennyLaneAI/pennylane/pull/3039)

<h3>Breaking changes</h3>

* Measuring an operator that might not be hermitian as an observable now raises a warning instead of an
  error. To definitively determine whether or not an operator is hermitian, use `qml.is_hermitian`.
  [(#2960)](https://github.com/PennyLaneAI/pennylane/pull/2960)

* The default `execute` method for the `QubitDevice` base class now calls `self.statistics`
  with an additional keyword argument `circuit`, which represents the quantum tape
  being executed.

  Any device that overrides `statistics` should edit the signature of the method to include
  the new `circuit` keyword argument.
  [(#2820)](https://github.com/PennyLaneAI/pennylane/pull/2820)

* The `expand_matrix()` has been moved from `~/operation.py` to
  `~/math/matrix_manipulation.py`
  [(#3008)](https://github.com/PennyLaneAI/pennylane/pull/3008)

<h3>Deprecations</h3>

* In-place inversion is now deprecated. This includes `op.inv()` and `op.inverse=value`. Please
  use `qml.adjoint` instead. Support for these methods will remain till v0.28.
  [(#2988)](https://github.com/PennyLaneAI/pennylane/pull/2988)

  Don't use:

  ```pycon
  >>> v1 = qml.PauliX(0).inv()
  >>> v2 = qml.PauliX(0)
  >>> v2.inverse = True
  ```

  Instead use:

  ```pycon
  >>> qml.adjoint(qml.PauliX(0))
  >>> qml.PauliX(0) ** -1
  ```

  `adjoint` takes the conjugate transpose of an operator, while `op ** -1` indicates matrix
  inversion. For unitary operators, `adjoint` will be more efficient than `op ** -1`, even
  though they represent the same thing.

* The `supports_reversible_diff` device capability is unused and has been removed.
  [(#2993)](https://github.com/PennyLaneAI/pennylane/pull/2993)

<h3>Documentation</h3>

* Corrects the docstrings for diagonalizing gates for all relevant operations. The docstrings used to say that the diagonalizing gates implemented $U$, the unitary such that $O = U \Sigma U^{\dagger}$, where $O$ is the original observable and $\Sigma$ a diagonal matrix. However, the diagonalizing gates actually implement $U^{\dagger}$, since $\langle \psi | O | \psi \rangle = \langle \psi | U \Sigma U^{\dagger} | \psi \rangle$, making $U^{\dagger} | \psi \rangle$ the actual state being measured in the $Z$-basis. [(#2981)](https://github.com/PennyLaneAI/pennylane/pull/2981)

<h3>Bug fixes</h3>

* Fixes a bug where the tape transform `single_qubit_fusion` computed wrong rotation angles
  for specific combinations of rotations.
  [(#3024)](https://github.com/PennyLaneAI/pennylane/pull/3024)

* Jax gradients now work with a QNode when the quantum function was transformed by `qml.simplify`.
  [(#3017)](https://github.com/PennyLaneAI/pennylane/pull/3017)

* Operators that have `num_wires = AnyWires` or `num_wires = AnyWires` raise an error, with
  certain exceptions, when instantiated with `wires=[]`.
  [(#2979)](https://github.com/PennyLaneAI/pennylane/pull/2979)

* Fixes a bug where printing `qml.Hamiltonian` with complex coefficients raises `TypeError` in some cases.
  [(#2979)](https://github.com/PennyLaneAI/pennylane/pull/2979)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola,
Utkarsh Azad,
Olivia Di Matteo,
Josh Izaac,
Soran Jahangiri,
Edward Jiang,
Ankit Khandelwal,
Korbinian Kottmann,
Christina Lee,
Meenu Kumari,
Lillian Marie Austin Frederiksen,
Albert Mitjans Coma,
Rashid N H M,
Zeyue Niu,
Mudit Pandey,
Matthew Silverman,
Jay Soni,
Antal Száva
Cody Wang,
David Wierichs
