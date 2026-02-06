
# Release 0.22.0

<h3>New features since last release</h3>

<h4>Quantum circuit cutting ✂️</h4>

* You can now run `N`-wire circuits on devices with fewer than `N` wires, by
  strategically placing `WireCut` operations that allow their circuit to be
  partitioned into smaller fragments, at a cost of needing to perform a greater
  number of device executions. Circuit cutting is enabled by decorating a QNode
  with the `@qp.cut_circuit` transform.
  [(#2107)](https://github.com/PennyLaneAI/pennylane/pull/2107)
  [(#2124)](https://github.com/PennyLaneAI/pennylane/pull/2124)
  [(#2153)](https://github.com/PennyLaneAI/pennylane/pull/2153)
  [(#2165)](https://github.com/PennyLaneAI/pennylane/pull/2165)
  [(#2158)](https://github.com/PennyLaneAI/pennylane/pull/2158)
  [(#2169)](https://github.com/PennyLaneAI/pennylane/pull/2169)
  [(#2192)](https://github.com/PennyLaneAI/pennylane/pull/2192)
  [(#2216)](https://github.com/PennyLaneAI/pennylane/pull/2216)
  [(#2168)](https://github.com/PennyLaneAI/pennylane/pull/2168)
  [(#2223)](https://github.com/PennyLaneAI/pennylane/pull/2223)
  [(#2231)](https://github.com/PennyLaneAI/pennylane/pull/2231)
  [(#2234)](https://github.com/PennyLaneAI/pennylane/pull/2234)
  [(#2244)](https://github.com/PennyLaneAI/pennylane/pull/2244)
  [(#2251)](https://github.com/PennyLaneAI/pennylane/pull/2251)
  [(#2265)](https://github.com/PennyLaneAI/pennylane/pull/2265)
  [(#2254)](https://github.com/PennyLaneAI/pennylane/pull/2254)
  [(#2260)](https://github.com/PennyLaneAI/pennylane/pull/2260)
  [(#2257)](https://github.com/PennyLaneAI/pennylane/pull/2257)
  [(#2279)](https://github.com/PennyLaneAI/pennylane/pull/2279)

  The example below shows how a three-wire circuit can be run on a two-wire device:

  ```python
  dev = qp.device("default.qubit", wires=2)

  @qp.cut_circuit
  @qp.qnode(dev)
  def circuit(x):
      qp.RX(x, wires=0)
      qp.RY(0.9, wires=1)
      qp.RX(0.3, wires=2)

      qp.CZ(wires=[0, 1])
      qp.RY(-0.4, wires=0)

      qp.WireCut(wires=1)

      qp.CZ(wires=[1, 2])

      return qp.expval(qp.grouping.string_to_pauli_word("ZZZ"))
  ```

  Instead of executing the circuit directly, it will be partitioned into smaller fragments
  according to the `WireCut` locations, and each fragment executed multiple times. Combining the
  results of the fragment executions will recover the expected output of the original uncut circuit.

    ```pycon
  >>> x = np.array(0.531, requires_grad=True)
  >>> circuit(0.531)
  0.47165198882111165
  ```

  Circuit cutting support is also differentiable:

  ```pycon
  >>> qp.grad(circuit)(x)
  -0.276982865449393
  ```

  For more details on circuit cutting, check out the
  [qp.cut_circuit](https://pennylane.readthedocs.io/en/latest/code/api/pennylane.cut_circuit.html)
  documentation page or [Peng et. al](https://arxiv.org/abs/1904.00102).

<h4>Conditional operations: quantum teleportation unlocked 🔓🌀</h4>

* Support for mid-circuit measurements and conditional operations
  has been added, to enable use cases like quantum teleportation, quantum error
  correction and quantum error mitigation.
  [(#2211)](https://github.com/PennyLaneAI/pennylane/pull/2211)
  [(#2236)](https://github.com/PennyLaneAI/pennylane/pull/2236)
  [(#2275)](https://github.com/PennyLaneAI/pennylane/pull/2275)

  Two new functions have been added to support this capability:

  - `qp.measure()` places mid-circuit measurements in the middle of a quantum function.

  - `qp.cond()` allows operations and quantum functions to be conditioned on the result of a
    previous measurement.

  For example, the code below shows how to teleport a qubit from wire 0 to wire 2:

  ```python
  dev = qp.device("default.qubit", wires=3)
  input_state = np.array([1, -1], requires_grad=False) / np.sqrt(2)

  @qp.qnode(dev)
  def teleport(state):
      # Prepare input state
      qp.QubitStateVector(state, wires=0)

      # Prepare Bell state
      qp.Hadamard(wires=1)
      qp.CNOT(wires=[1, 2])

      # Apply gates
      qp.CNOT(wires=[0, 1])
      qp.Hadamard(wires=0)

      # Measure first two wires
      m1 = qp.measure(0)
      m2 = qp.measure(1)

      # Condition final wire on results
      qp.cond(m2 == 1, qp.PauliX)(wires=2)
      qp.cond(m1 == 1, qp.PauliZ)(wires=2)

      # Return state on final wire
      return qp.density_matrix(wires=2)
  ```

  We can double-check that the qubit has been teleported by computing the
  overlap between the input state and the resulting state on wire 2:

  ```pycon
  >>> output_state = teleport(input_state)
  >>> output_state
  tensor([[ 0.5+0.j, -0.5+0.j],
          [-0.5+0.j,  0.5+0.j]], requires_grad=True)
  >>> input_state.conj() @ output_state @ input_state
  tensor(1.+0.j, requires_grad=True)
  ```

  For a full description of new capabilities, refer to the [Mid-circuit
  measurements and conditional
  operations](https://pennylane.readthedocs.io/en/latest/introduction/measurements.html#mid-circuit-measurements-and-conditional-operations)
  section in the documentation.

* Train mid-circuit measurements by deferring them, via the new
  `@qp.defer_measurements` transform.
  [(#2211)](https://github.com/PennyLaneAI/pennylane/pull/2211)
  [(#2236)](https://github.com/PennyLaneAI/pennylane/pull/2236)
  [(#2275)](https://github.com/PennyLaneAI/pennylane/pull/2275)

  If a device doesn't natively support mid-circuit measurements, the `@qp.defer_measurements`
  transform can be applied to the QNode to transform the QNode into one with _terminal_ measurements
  and _controlled_ operations:

  ```python
  dev = qp.device("default.qubit", wires=2)

  @qp.qnode(dev)
  @qp.defer_measurements
  def circuit(x):
      qp.Hadamard(wires=0)

      m = qp.measure(0)

      def op_if_true():
          return qp.RX(x**2, wires=1)

      def op_if_false():
          return qp.RY(x, wires=1)

      qp.cond(m==1, op_if_true, op_if_false)()

      return qp.expval(qp.PauliZ(1))
  ```

  ```pycon
  >>> x = np.array(0.7, requires_grad=True)
  >>> print(qp.draw(circuit, expansion_strategy="device")(x))
  0: ──H─╭C─────────X─╭C─────────X─┤
  1: ────╰RX(0.49)────╰RY(0.70)────┤  <Z>
  >>> circuit(x)
  tensor(0.82358752, requires_grad=True)
  ```

  Deferring mid-circuit measurements also enables differentiation:

  ```pycon
  >>> qp.grad(circuit)(x)
  -0.651546965338656
  ```

<h4>Debug with mid-circuit quantum snapshots 📷</h4>

* A new operation `qp.Snapshot` has been added to assist in debugging quantum functions.
  [(#2233)](https://github.com/PennyLaneAI/pennylane/pull/2233)
  [(#2289)](https://github.com/PennyLaneAI/pennylane/pull/2289)
  [(#2291)](https://github.com/PennyLaneAI/pennylane/pull/2291)
  [(#2315)](https://github.com/PennyLaneAI/pennylane/pull/2315)

  `qp.Snapshot` saves the internal state of devices at arbitrary points of execution.

  Currently supported devices include:

  - `default.qubit`: each snapshot saves the quantum state vector
  - `default.mixed`: each snapshot saves the density matrix
  - `default.gaussian`: each snapshot saves the covariance matrix and vector of means

  During normal execution, the snapshots are ignored:

  ```python
  dev = qp.device("default.qubit", wires=2)

  @qp.qnode(dev, interface=None)
  def circuit():
      qp.Snapshot()
      qp.Hadamard(wires=0)
      qp.Snapshot("very_important_state")
      qp.CNOT(wires=[0, 1])
      qp.Snapshot()
      return qp.expval(qp.PauliX(0))
  ```

  However, when using the `qp.snapshots`
  transform, intermediate device states will be stored and returned alongside the
  results.

  ```pycon
  >>> qp.snapshots(circuit)()
  {0: array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]),
   'very_important_state': array([0.70710678+0.j, 0.        +0.j, 0.70710678+0.j, 0.        +0.j]),
   2: array([0.70710678+0.j, 0.        +0.j, 0.        +0.j, 0.70710678+0.j]),
   'execution_results': array(0.)}
  ```

<h4>Batch embedding and state preparation data 📦</h4>

* Added the `@qp.batch_input` transform to enable batching non-trainable gate parameters.
  In addition, the `qp.qnn.KerasLayer` class has been updated to natively support
  batched training data.
  [(#2069)](https://github.com/PennyLaneAI/pennylane/pull/2069)

  As with other transforms, `@qp.batch_input` can be used to decorate QNodes:
  ```python
  dev = qp.device("default.qubit", wires=2, shots=None)

  @qp.batch_input(argnum=0)
  @qp.qnode(dev, diff_method="parameter-shift", interface="tf")
  def circuit(inputs, weights):
      # add a batch dimension to the embedding data
      qp.AngleEmbedding(inputs, wires=range(2), rotation="Y")
      qp.RY(weights[0], wires=0)
      qp.RY(weights[1], wires=1)
      return qp.expval(qp.PauliZ(1))
  ```

  Batched input parameters can then be passed during QNode evaluation:

  ```pycon
  >>> x = tf.random.uniform((10, 2), 0, 1)
  >>> w = tf.random.uniform((2,), 0, 1)
  >>> circuit(x, w)
  <tf.Tensor: shape=(10,), dtype=float64, numpy=
  array([0.46230079, 0.73971315, 0.95666004, 0.5355225 , 0.66180948,
          0.44519553, 0.93874261, 0.9483197 , 0.78737918, 0.90866411])>
  ```

<h4>Even more mighty quantum transforms 🐛➡🦋</h4>

* New functions and transforms of operators have been added:

  - `qp.matrix()` for computing the matrix representation of one or more unitary operators.
    [(#2241)](https://github.com/PennyLaneAI/pennylane/pull/2241)

  - `qp.eigvals()` for computing the eigenvalues of one or more operators.
    [(#2248)](https://github.com/PennyLaneAI/pennylane/pull/2248)

  - `qp.generator()` for computing the generator of a single-parameter unitary operation.
    [(#2256)](https://github.com/PennyLaneAI/pennylane/pull/2256)

  All operator transforms can be used on instantiated operators,

  ```pycon
  >>> op = qp.RX(0.54, wires=0)
  >>> qp.matrix(op)
  [[0.9637709+0.j         0.       -0.26673144j]
  [0.       -0.26673144j 0.9637709+0.j        ]]
  ```

  Operator transforms can also be used in a functional form:

  ```pycon
  >>> x = torch.tensor(0.6, requires_grad=True)
  >>> matrix_fn = qp.matrix(qp.RX)
  >>> matrix_fn(x, wires=[0])
  tensor([[0.9553+0.0000j, 0.0000-0.2955j],
          [0.0000-0.2955j, 0.9553+0.0000j]], grad_fn=<AddBackward0>)
  ```

  In its functional form, it is fully differentiable with respect to gate arguments:

  ```pycon
  >>> loss = torch.real(torch.trace(matrix_fn(x, wires=0)))
  >>> loss.backward()
  >>> x.grad
  tensor(-0.2955)
  ```

  Some operator transform can also act on multiple operations, by passing
  quantum functions or tapes:

  ```pycon
  >>> def circuit(theta):
  ...     qp.RX(theta, wires=1)
  ...     qp.PauliZ(wires=0)
  >>> qp.matrix(circuit)(np.pi / 4)
  array([[ 0.92387953+0.j,  0.+0.j ,  0.-0.38268343j,  0.+0.j],
  [ 0.+0.j,  -0.92387953+0.j,  0.+0.j,  0. +0.38268343j],
  [ 0. -0.38268343j,  0.+0.j,  0.92387953+0.j,  0.+0.j],
  [ 0.+0.j,  0.+0.38268343j,  0.+0.j,  -0.92387953+0.j]])
  ```

* A new transform has been added to construct the pairwise-commutation directed acyclic graph (DAG)
  representation of a quantum circuit.
  [(#1712)](https://github.com/PennyLaneAI/pennylane/pull/1712)

  In the DAG, each node represents a quantum operation, and edges represent non-commutation
  between two operations.

  This transform takes into account that not all operations can be moved next to each other by
  pairwise commutation:

  ```pycon
  >>> def circuit(x, y, z):
  ...     qp.RX(x, wires=0)
  ...     qp.RX(y, wires=0)
  ...     qp.CNOT(wires=[1, 2])
  ...     qp.RY(y, wires=1)
  ...     qp.Hadamard(wires=2)
  ...     qp.CRZ(z, wires=[2, 0])
  ...     qp.RY(-y, wires=1)
  ...     return qp.expval(qp.PauliZ(0))
  >>> dag_fn = qp.commutation_dag(circuit)
  >>> dag = dag_fn(np.pi / 4, np.pi / 3, np.pi / 2)
  ```

  Nodes in the commutation DAG can be accessed via the `get_nodes()` method, returning a list of
  the  form `(ID, CommutationDAGNode)`:

  ```pycon
  >>> nodes = dag.get_nodes()
  >>> nodes
  NodeDataView({0: <pennylane.transforms.commutation_dag.CommutationDAGNode object at 0x7f461c4bb580>, ...}, data='node')
  ```

  Specific nodes in the commutation DAG can be accessed via the `get_node()` method:

  ```
  >>> second_node = dag.get_node(2)
  >>> second_node
  <pennylane.transforms.commutation_dag.CommutationDAGNode object at 0x136f8c4c0>
  >>> second_node.op
  CNOT(wires=[1, 2])
  >>> second_node.successors
  [3, 4, 5, 6]
  >>> second_node.predecessors
  []
  ```

<h3>Improvements</h3>

* The text-based drawer accessed via `qp.draw()` has been optimized and improved.
  [(#2128)](https://github.com/PennyLaneAI/pennylane/pull/2128)
  [(#2198)](https://github.com/PennyLaneAI/pennylane/pull/2198)

  The new drawer has:

  * a `decimals` keyword for controlling parameter rounding
  * a `show_matrices` keyword for controlling display of matrices
  * a different algorithm for determining positions
  * deprecation of the `charset` keyword
  * additional minor cosmetic changes

  ```python
  @qp.qnode(qp.device('lightning.qubit', wires=2))
  def circuit(a, w):
      qp.Hadamard(0)
      qp.CRX(a, wires=[0, 1])
      qp.Rot(*w, wires=[1])
      qp.CRX(-a, wires=[0, 1])
      return qp.expval(qp.PauliZ(0) @ qp.PauliZ(1))
  ```

  ```pycon
  >>> print(qp.draw(circuit, decimals=2)(a=2.3, w=[1.2, 3.2, 0.7]))
  0: ──H─╭C─────────────────────────────╭C─────────┤ ╭<Z@Z>
  1: ────╰RX(2.30)──Rot(1.20,3.20,0.70)─╰RX(-2.30)─┤ ╰<Z@Z>
  ```

* The frequencies of gate parameters are now accessible as an operation
  property and can be used for circuit analysis, optimization via the
  `RotosolveOptimizer` and differentiation with the parameter-shift rule
  (including the general shift rule).
  [(#2180)](https://github.com/PennyLaneAI/pennylane/pull/2180)
  [(#2182)](https://github.com/PennyLaneAI/pennylane/pull/2182)
  [(#2227)](https://github.com/PennyLaneAI/pennylane/pull/2227)

  ```pycon
  >>> op = qp.CRot(0.4, 0.1, 0.3, wires=[0, 1])
  >>> op.parameter_frequencies
  [(0.5, 1.0), (0.5, 1.0), (0.5, 1.0)]
  ```

  When using `qp.gradients.param_shift`, either a custom `grad_recipe` or the
  parameter frequencies are used to obtain the shift rule for the operation, in
  that order of preference.

  See [Vidal and Theis (2018)](https://arxiv.org/abs/1812.06323) and [Wierichs
  et al. (2021)](https://arxiv.org/abs/2107.12390) for theoretical background
  information on the general parameter-shift rule.

* No two-term parameter-shift rule is assumed anymore by default.
  [(#2227)](https://github.com/PennyLaneAI/pennylane/pull/2227)

  Previously, operations marked for analytic differentiation that
  did not provide a `generator`, `parameter_frequencies` or a
  custom `grad_recipe` were assumed to satisfy the two-term shift
  rule. This now has to be made explicit for custom operations
  by adding any of the above attributes.

* Most compilation transforms, and relevant subroutines, have been updated to
  support just-in-time compilation with `jax.jit`.
  [(#1894)](https://github.com/PennyLaneAI/pennylane/pull/1894/)

* The `qp.draw_mpl` transform supports a `expansion_strategy` keyword argument.
  [(#2271)](https://github.com/PennyLaneAI/pennylane/pull/2271/)

* The `qp.gradients` module has been streamlined and special-purpose functions
  moved closer to their use cases, while preserving existing behaviour.
  [(#2200)](https://github.com/PennyLaneAI/pennylane/pull/2200)

* Added a new `partition_pauli_group` function to the `grouping` module for
  efficiently measuring the `N`-qubit Pauli group with `3 ** N`
  qubit-wise commuting terms.
  [(#2185)](https://github.com/PennyLaneAI/pennylane/pull/2185)

* The Operator class has undergone a major refactor with the following changes:

  * **Matrices**: the static method `Operator.compute_matrices()` defines the matrix representation
    of the operator, and the function `qp.matrix(op)` computes this for a given
    instance.
    [(#1996)](https://github.com/PennyLaneAI/pennylane/pull/1996)

  * **Eigvals**: the static method `Operator.compute_eigvals()` defines the matrix representation
    of the operator, and the function `qp.eigvals(op)` computes this for a given
    instance.
    [(#2048)](https://github.com/PennyLaneAI/pennylane/pull/2048)

  * **Decompositions**: the static method `Operator.compute_decomposition()` defines the matrix representation
    of the operator, and the method `op.decomposition()` computes this for a given
    instance.
    [(#2024)](https://github.com/PennyLaneAI/pennylane/pull/2024)
    [(#2053)](https://github.com/PennyLaneAI/pennylane/pull/2053)

  * **Sparse matrices**: the static method `Operator.compute_sparse_matrix()` defines the sparse
    matrix representation of the operator, and the method `op.sparse_matrix()` computes this for a
    given instance.
    [(#2050)](https://github.com/PennyLaneAI/pennylane/pull/2050)

  * **Linear combinations of operators**: The static method `compute_terms()`, used for representing
    the linear combination of coefficients and operators representing the operator, has been added.
    The method `op.terms()` computes this for a given instance.
    Currently, only the `Hamiltonian` class overwrites `compute_terms()` to store
    coefficients and operators. The `Hamiltonian.terms` property hence becomes
    a proper method called by `Hamiltonian.terms()`.
    [(#2036)](https://github.com/PennyLaneAI/pennylane/pull/2036)

  * **Diagonalization**: The `diagonalizing_gates()` representation has been moved to the
    highest-level `Operator` class and is therefore available to all subclasses. A condition
    `qp.operation.defines_diagonalizing_gates` has been added, which can be used in tape contexts
    without queueing. In addition, a static `compute_diagonalizing_gates` method has been added,
    which is called by default in `diagonalizing_gates()`.
    [(#1985)](https://github.com/PennyLaneAI/pennylane/pull/1985)
    [(#1993)](https://github.com/PennyLaneAI/pennylane/pull/1993)

  * Error handling has been improved for Operator representations. Custom errors subclassing
    `OperatorPropertyUndefined` are raised if a representation has not been defined. This replaces
    the `NotImplementedError` and allows finer control for developers.
    [(#2064)](https://github.com/PennyLaneAI/pennylane/pull/2064)
    [(#2287)](https://github.com/PennyLaneAI/pennylane/pull/2287/)

  * A `Operator.hyperparameters` attribute, used for storing operation parameters that are *never*
    trainable, has been added to the operator class.
    [(#2017)](https://github.com/PennyLaneAI/pennylane/pull/2017)

  * The `string_for_inverse` attribute is removed.
    [(#2021)](https://github.com/PennyLaneAI/pennylane/pull/2021)

  * The `expand()` method was moved from the `Operation` class to the main `Operator` class.
    [(#2053)](https://github.com/PennyLaneAI/pennylane/pull/2053)
    [(#2239)](https://github.com/PennyLaneAI/pennylane/pull/2239)

<h3>Deprecations</h3>

* There are several important changes when creating custom operations:
  [(#2214)](https://github.com/PennyLaneAI/pennylane/pull/2214)
  [(#2227)](https://github.com/PennyLaneAI/pennylane/pull/2227)
  [(#2030)](https://github.com/PennyLaneAI/pennylane/pull/2030)
  [(#2061)](https://github.com/PennyLaneAI/pennylane/pull/2061)

  - The `Operator.matrix` method has been deprecated and `Operator.compute_matrix`
    should be defined instead. Operator matrices should be accessed using `qp.matrix(op)`.
    If you were previously defining the class method `Operator._matrix()`, this is a a **breaking
    change** --- please update your operation to instead overwrite `Operator.compute_matrix`.

  - The `Operator.decomposition` method has been deprecated and `Operator.compute_decomposition`
    should be defined instead. Operator decompositions should be accessed using `Operator.decomposition()`.

  - The `Operator.eigvals` method has been deprecated and `Operator.compute_eigvals`
    should be defined instead. Operator eigenvalues should be accessed using `qp.eigvals(op)`.

  - The `Operator.generator` property is now a method, and should return an *operator instance*
    representing the generator. Note that unlike the other representations above, this is a
    **breaking change**. Operator generators should be accessed using `qp.generator(op)`.

  - The `Operation.get_parameter_shift` method has been deprecated
    and will be removed in a future release.

    Instead, the functionalities for general parameter-shift rules in the
    `qp.gradients` module should be used, together with the operation attributes
    `parameter_frequencies` or `grad_recipe`.

* Executing tapes using `tape.execute(dev)` is deprecated.
  Please use the `qp.execute([tape], dev)` function instead.
  [(#2306)](https://github.com/PennyLaneAI/pennylane/pull/2306)

* The subclasses of the quantum tape, including `JacobianTape`, `QubitParamShiftTape`,
  `CVParamShiftTape`, and `ReversibleTape` are deprecated. Instead of calling
  `JacobianTape.jacobian()` and `JacobianTape.hessian()`,
  please use a standard `QuantumTape`, and apply gradient transforms using
  the `qp.gradients` module.
  [(#2306)](https://github.com/PennyLaneAI/pennylane/pull/2306)

* `qp.transforms.get_unitary_matrix()` has been deprecated and will be removed
  in a future release. For extracting matrices of operations and quantum functions,
  please use `qp.matrix()`.
  [(#2248)](https://github.com/PennyLaneAI/pennylane/pull/2248)

* The `qp.finite_diff()` function has been deprecated and will be removed
  in an upcoming release. Instead,
  `qp.gradients.finite_diff()` can be used to compute purely quantum gradients
  (that is, gradients of tapes or QNode).
  [(#2212)](https://github.com/PennyLaneAI/pennylane/pull/2212)

* The `MultiControlledX` operation now accepts a single `wires` keyword argument for both `control_wires` and `wires`.
  The single `wires` keyword should be all the control wires followed by a single target wire.
  [(#2121)](https://github.com/PennyLaneAI/pennylane/pull/2121)
  [(#2278)](https://github.com/PennyLaneAI/pennylane/pull/2278)

<h3>Breaking changes</h3>

* The representation of an operator as a matrix has been overhauled.
  [(#1996)](https://github.com/PennyLaneAI/pennylane/pull/1996)

  The "canonical matrix", which is independent of wires,
  is now defined in the static method `compute_matrix()` instead of `_matrix`.
  By default, this method is assumed to take all parameters and non-trainable
  hyperparameters that define the operation.

  ```pycon
  >>> qp.RX.compute_matrix(0.5)
  [[0.96891242+0.j         0.        -0.24740396j]
   [0.        -0.24740396j 0.96891242+0.j        ]]
  ```

  If no canonical matrix is specified for a gate, `compute_matrix()`
  raises a `MatrixUndefinedError`.

* The generator property has been updated to an instance method,
  `Operator.generator()`. It now returns an instantiated operation,
  representing the generator of the instantiated operator.
  [(#2030)](https://github.com/PennyLaneAI/pennylane/pull/2030)
  [(#2061)](https://github.com/PennyLaneAI/pennylane/pull/2061)

  Various operators have been updated to specify the generator as either
  an `Observable`, `Tensor`, `Hamiltonian`, `SparseHamiltonian`, or `Hermitian`
  operator.

  In addition, `qp.generator(operation)` has been added to aid in retrieving
  generator representations of operators.

* The argument `wires` in `heisenberg_obs`, `heisenberg_expand` and `heisenberg_tr`
  was renamed to `wire_order` to be consistent with other matrix representations.
  [(#2051)](https://github.com/PennyLaneAI/pennylane/pull/2051)

* The property `kraus_matrices` has been changed to a method, and `_kraus_matrices` renamed to
  `compute_kraus_matrices`, which is now a static method.
  [(#2055)](https://github.com/PennyLaneAI/pennylane/pull/2055)

* The `pennylane.measure` module has been renamed to `pennylane.measurements`.
  [(#2236)](https://github.com/PennyLaneAI/pennylane/pull/2236)

<h3>Bug fixes</h3>

* The `basis` property of `qp.SWAP` was set to `"X"`, which is incorrect; it is
  now set to `None`.
  [(#2287)](https://github.com/PennyLaneAI/pennylane/pull/2287/)

* The `qp.RandomLayers` template now decomposes when the weights are a list of lists.
  [(#2266)](https://github.com/PennyLaneAI/pennylane/pull/2266/)

* The `qp.QubitUnitary` operation now supports just-in-time compilation using JAX.
  [(#2249)](https://github.com/PennyLaneAI/pennylane/pull/2249)

* Fixes a bug in the JAX interface where `Array` objects
  were not being converted to NumPy arrays before executing an
  external device.
  [(#2255)](https://github.com/PennyLaneAI/pennylane/pull/2255)

* The `qp.ctrl` transform now works correctly with gradient transforms
  such as the parameter-shift rule.
  [(#2238)](https://github.com/PennyLaneAI/pennylane/pull/2238)

* Fixes a bug in which passing required arguments into operations as
  keyword arguments would throw an error because the documented call
  signature didn't match the function definition.
  [(#1976)](https://github.com/PennyLaneAI/pennylane/pull/1976)

* The operation `OrbitalRotation` previously was wrongfully registered to satisfy
  the four-term parameter shift rule. The correct eight-term rule will now be used when
  using the parameter-shift rule.
  [(#2180)](https://github.com/PennyLaneAI/pennylane/pull/2180)

* Fixes a bug where `qp.gradients.param_shift_hessian` would produce an
  error whenever all elements of the Hessian are known in advance to be 0.
  [(#2299)](https://github.com/PennyLaneAI/pennylane/pull/2299)

<h3>Documentation</h3>

* The developer guide on adding templates and the architecture overview were rewritten
  to reflect the past and planned changes of the operator refactor.
  [(#2066)](https://github.com/PennyLaneAI/pennylane/pull/2066)

* Links to the Strawberry Fields documentation for information on the CV
  model.
  [(#2259)](https://github.com/PennyLaneAI/pennylane/pull/2259)

* Fixes the documentation example for `qp.QFT`.
  [(#2232)](https://github.com/PennyLaneAI/pennylane/pull/2232)

* Fixes the documentation example for using `qp.sample` with `jax.jit`.
  [(#2196)](https://github.com/PennyLaneAI/pennylane/pull/2196)

* The `qp.numpy` subpackage is now included in the PennyLane
  API documentation.
  [(#2179)](https://github.com/PennyLaneAI/pennylane/pull/2179)

* Improves the documentation of `RotosolveOptimizer` regarding the
  usage of the passed `substep_optimizer` and its keyword arguments.
  [(#2160)](https://github.com/PennyLaneAI/pennylane/pull/2160)

* Ensures that signatures of `@qp.qfunc_transform` decorated functions
  display correctly in the docs.
  [(#2286)](https://github.com/PennyLaneAI/pennylane/pull/2286)

* Docstring examples now display using the updated text-based circuit drawer.
  [(#2252)](https://github.com/PennyLaneAI/pennylane/pull/2252)

* Add docstring to `OrbitalRotation.grad_recipe`.
  [(#2193)](https://github.com/PennyLaneAI/pennylane/pull/2193)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Catalina Albornoz, Jack Y. Araz, Juan Miguel Arrazola, Ali Asadi, Utkarsh Azad,
Sam Banning, Thomas Bromley, Olivia Di Matteo, Christian Gogolin, Diego Guala,
Anthony Hayes, David Ittah, Josh Izaac, Soran Jahangiri, Nathan Killoran,
Christina Lee, Angus Lowe, Maria Fernanda Morris, Romain Moyard, Zeyue Niu, Lee
James O'Riordan, Chae-Yeun Park, Maria Schuld, Jay Soni, Antal Száva, David
Wierichs.
