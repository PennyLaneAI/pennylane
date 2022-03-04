:orphan:

# Release 0.22.0-dev (development release)

<h3>New features since last release</h3>

* A new operation `qml.Snapshot` has been added to assist users in debugging quantum progams.
  The instruction is used to save the internal state of simulator devices at arbitrary points of
  execution, such as the quantum state vector and density matrix in the qubit case, or the
  covariance matrix and vector of means in the continuous variable case. The saved states can be
  retrieved in the form of a dictionary via the top-level `qml.snapshots` function.
  [(#2233)](https://github.com/PennyLaneAI/pennylane/pull/2233)

  ```py
  dev = qml.device("default.qubit", wires=2)
  
  @qml.qnode(dev, interface=None)
  def circuit():
      qml.Snapshot()
      qml.Hadamard(wires=0)
      qml.Snapshot("very_important_state")
      qml.CNOT(wires=[0, 1])
      qml.Snapshot()
      return qml.expval(qml.PauliX(0))
  ```

  ```pycon
  >>> qml.snapshots(circuit)()
  {0: array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]),
   'very_important_state': array([0.70710678+0.j, 0.        +0.j, 0.70710678+0.j, 0.        +0.j]),
   2: array([0.70710678+0.j, 0.        +0.j, 0.        +0.j, 0.70710678+0.j]),
   'execution_results': array(0.)}
  ```

* New functions and transforms of operators have been added. These include:

  - `qml.matrix()` for computing the matrix representation of one or more unitary operators.
    [(#2241)](https://github.com/PennyLaneAI/pennylane/pull/2241)

  - `qml.eigvals()` for computing the eigenvalues of one or more operators.
    [(#2248)](https://github.com/PennyLaneAI/pennylane/pull/2248)

  - `qml.generator()` for computing the generator of a single-parameter unitary operation.
    [(#2256)](https://github.com/PennyLaneAI/pennylane/pull/2256)

  All operator transforms can be used on instantiated operators,

  ```pycon
  >>> op = qml.RX(0.54, wires=0)
  >>> qml.matrix(op)
  [[0.9637709+0.j         0.       -0.26673144j]
  [0.       -0.26673144j 0.9637709+0.j        ]]
  ```

  Operator transforms can also be used in a functional form:

  ```pycon
  >>> x = torch.tensor(0.6, requires_grad=True)
  >>> matrix_fn = qml.matrix(qml.RX)
  >>> matrix_fn(x)
  tensor([[0.9553+0.0000j, 0.0000-0.2955j],
          [0.0000-0.2955j, 0.9553+0.0000j]], grad_fn=<AddBackward0>)
  ```

  In its functional form, it is fully differentiable with respect to gate arguments:

  ```pycon
  >>> loss = torch.real(torch.trace(matrix_fn(x, wires=0)))
  >>> loss.backward()
  >>> x.grad
  tensor(-0.5910)
  ```

  Some operator transform can also act on multiple operations, by passing
  quantum functions or tapes:

  ```pycon
  >>> def circuit(theta):
  ...     qml.RX(theta, wires=1)
  ...     qml.PauliZ(wires=0)
  >>> qml.matrix(circuit)(np.pi / 4)
  array([[ 0.92387953+0.j,  0.+0.j ,  0.-0.38268343j,  0.+0.j],
  [ 0.+0.j,  -0.92387953+0.j,  0.+0.j,  0. +0.38268343j],
  [ 0. -0.38268343j,  0.+0.j,  0.92387953+0.j,  0.+0.j],
  [ 0.+0.j,  0.+0.38268343j,  0.+0.j,  -0.92387953+0.j]])
  ```

* The user-interface for mid-circuit measurements and conditional operations
  has been added to support use cases like quantum teleportation.
  [(#2211)](https://github.com/PennyLaneAI/pennylane/pull/2211)
  [(#2236)](https://github.com/PennyLaneAI/pennylane/pull/2236)
  [(#2275)](https://github.com/PennyLaneAI/pennylane/pull/2275)

  The addition includes the `defer_measurements` device-independent transform
  that can be applied on devices that have no native mid-circuit measurements
  capabilities. This transform is applied by default when evaluating a QNode on a
  device that doesn't support mid-circuit measurements.
  
  For example, the code below shows how to teleport a qubit:

  ```python
  from scipy.stats import unitary_group

  random_state = unitary_group.rvs(2, random_state=1967)[0]

  dev = qml.device("default.mixed", wires=3)

  @qml.qnode(dev)
  def teleport(state):
      # Prepare input state
      qml.QubitStateVector(state, wires=0)

      # Prepare Bell state
      qml.Hadamard(wires=1)
      qml.CNOT(wires=[1, 2])

      # Apply gates
      qml.CNOT(wires=[0, 1])
      qml.Hadamard(wires=0)

      # Measure first two wires
      m1 = qml.measure(0)
      m2 = qml.measure(1)

      # Condition final wire on results
      qml.cond(m2 == 1, qml.PauliX)(wires=2)
      qml.cond(m1 == 1, qml.PauliZ)(wires=2)

      # Return state on final wire
      return qml.density_matrix(wires=2)

  output_projector = teleport(random_state)
  overlap = random_state.conj() @ output_projector @ random_state
  ```
  ```pycon
  >>> overlap
  tensor(1.+0.j, requires_grad=True)
  ```

  For further examples, refer to the [Mid-circuit measurements and conditional
  operations](https://pennylane.readthedocs.io/en/latest/introduction/measurements.html#mid-circuit-measurements-and-conditional-operations)
  section in the documentation.

* A new transform has been added to construct the pairwise-commutation directed acyclic graph (DAG)
  representation of a quantum circuit.
  [(#1712)](https://github.com/PennyLaneAI/pennylane/pull/1712)

  In the DAG, each node represents a quantum operation, and edges represent non-commutation
  between two operations.

  This transform takes into account that not all operations can be moved next to each other by
  pairwise commutation:

  ```pycon
  >>> def circuit(x, y, z):
  ...     qml.RX(x, wires=0)
  ...     qml.RX(y, wires=0)
  ...     qml.CNOT(wires=[1, 2])
  ...     qml.RY(y, wires=1)
  ...     qml.Hadamard(wires=2)
  ...     qml.CRZ(z, wires=[2, 0])
  ...     qml.RY(-y, wires=1)
  ...     return qml.expval(qml.PauliZ(0))
  >>> dag_fn = commutation_dag(circuit)
  >>> dag = dag_fn(np.pi / 4, np.pi / 3, np.pi / 2)
  ```

  Nodes in the commutation DAG can be accessed via the `get_nodes()` method, returning a list of
  the  form `(ID, CommutationDAGNode)`:

  ```pycon
  nodes = dag.get_nodes()
  [(0, <pennylane.transforms.commutation_dag.CommutationDAGNode object at 0x132b03b20>), ...]
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

* The text based drawer accessed via `qml.draw` has been overhauled.
  [(#2128)](https://github.com/PennyLaneAI/pennylane/pull/2128)
  [(#2198)](https://github.com/PennyLaneAI/pennylane/pull/2198)

  The new drawer has:

  * a `decimals` keyword for controlling parameter rounding
  * a `show_matrices` keyword for controlling display of matrices
  * a different algorithm for determining positions
  * deprecation of the `charset` keyword
  * additional minor cosmetic changes

  ```python
  @qml.qnode(qml.device('lightning.qubit', wires=2))
  def circuit(a, w):
      qml.Hadamard(0)
      qml.CRX(a, wires=[0, 1])
      qml.Rot(*w, wires=[1])
      qml.CRX(-a, wires=[0, 1])
      return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
  ```

  ```pycon
  >>> print(qml.draw(circuit, decimals=2)(a=2.3, w=[1.2, 3.2, 0.7]))
  0: ──H─╭C─────────────────────────────╭C─────────┤ ╭<Z@Z>
  1: ────╰RX(2.30)──Rot(1.20,3.20,0.70)─╰RX(-2.30)─┤ ╰<Z@Z>
  ```

* Parametric operations now have the `parameter_frequencies`
  method that returns the frequencies with which a parameter
  enters a circuit. In addition, the general parameter-shift
  rule is now automatically used by `qml.gradients.param_shift`.
  [(#2180)](https://github.com/PennyLaneAI/pennylane/pull/2180)
  [(#2182)](https://github.com/PennyLaneAI/pennylane/pull/2182)
  [(#2227)](https://github.com/PennyLaneAI/pennylane/pull/2227)

  The frequencies can be used for circuit analysis, optimization
  via the `RotosolveOptimizer` and differentiation with the
  parameter-shift rule. They assume that the circuit returns
  expectation values or probabilities, for a variance
  measurement the frequencies will differ.

  By default, the frequencies will be obtained from the
  `generator` property (if it is defined).

  When using `qml.gradients.param_shift`, either a custom `grad_recipe`
  or the parameter frequencies are used to obtain the shift rule
  for the operation, in that order of preference.

  See [Vidal and Theis (2018)](https://arxiv.org/abs/1812.06323)
  and [Wierichs et al. (2021)](https://arxiv.org/abs/2107.12390)
  for theoretical background information on the general
  parameter-shift rule.

* Continued development of the circuit-cutting compiler:

  - A method for converting a quantum tape to a directed multigraph that is amenable
    to graph partitioning algorithms for circuit cutting has been added.
    [(#2107)](https://github.com/PennyLaneAI/pennylane/pull/2107)

  - A method to replace `WireCut` nodes in a directed multigraph with `MeasureNode`
    and `PrepareNode` placeholders has been added.
    [(#2124)](https://github.com/PennyLaneAI/pennylane/pull/2124)

  - A method has been added that takes a directed multigraph with `MeasureNode` and
    `PrepareNode` placeholders and fragments into subgraphs and a communication graph.
    [(#2153)](https://github.com/PennyLaneAI/pennylane/pull/2153)

  - A method has been added that takes a directed multigraph with `MeasureNode`
    and `PrepareNode` placeholder nodes and converts it into a tape.
    [(#2165)](https://github.com/PennyLaneAI/pennylane/pull/2165)

  - A differentiable tensor contraction function `contract_tensors` has been
    added.
    [(#2158)](https://github.com/PennyLaneAI/pennylane/pull/2158)

  - A method has been added that expands a quantum tape over `MeasureNode` and `PrepareNode`
    configurations.
    [(#2169)](https://github.com/PennyLaneAI/pennylane/pull/2169)

  - The postprocessing function for the `cut_circuit` transform has been added.
    [(#2192)](https://github.com/PennyLaneAI/pennylane/pull/2192)
    
  - The `cut_circuit` transform has been added.
    [(#2216)](https://github.com/PennyLaneAI/pennylane/pull/2216)

  - A class `CutStrategy` which acts as an interface and coordinates device/user
    constraints with circuit execution requirements to come up with the best sets
    of graph partitioning parameters.
    [(#2168)](https://github.com/PennyLaneAI/pennylane/pull/2168)

  - A suite of integration tests has been added.
    [(#2231)](https://github.com/PennyLaneAI/pennylane/pull/2231)
    [(#2234)](https://github.com/PennyLaneAI/pennylane/pull/2234)
    [(#2244)](https://github.com/PennyLaneAI/pennylane/pull/2244)
    [(#2251)](https://github.com/PennyLaneAI/pennylane/pull/2251)
    [(#2265)](https://github.com/PennyLaneAI/pennylane/pull/2265)

  - Circuit fragments that are disconnected from the terminal measurements are now removed.
    [(#2254)](https://github.com/PennyLaneAI/pennylane/pull/2254)
  
  - `WireCut` operations that do not lead to a disconnection are now being removed.
    [(#2260)](https://github.com/PennyLaneAI/pennylane/pull/2260)
  
  - Circuit cutting now remaps the wires of fragment circuits to match the available wires on the
    device.
    [(#2257)](https://github.com/PennyLaneAI/pennylane/pull/2257)

<h3>Improvements</h3>

* No two-term parameter-shift rule is assumed anymore by default.
  [(#2227)](https://github.com/PennyLaneAI/pennylane/pull/2227)

  Previously, operations marked for analytic differentiation that
  do not provide a `generator`, `parameter_frequencies` or a
  custom `grad_recipe` were assumed to satisfy the two-term shift
  rule. This now has to be made explicit for custom operations
  by adding any of the above attributes.

* The `qml.draw_mpl` transform supports a `expansion_strategy` keyword argument.
  [(#2271)](https://github.com/PennyLaneAI/pennylane/pull/2271/)

* The `qml.gradients` module has been streamlined and special-purpose functions
  moved closer to their use cases, while preserving existing behaviour.
  [(#2200)](https://github.com/PennyLaneAI/pennylane/pull/2200)

* Added a new `partition_pauli_group` function to the `grouping` module for
  efficiently measuring the `N`-qubit Pauli group with `3 ** N`
  qubit-wise commuting terms.
  [(#2185)](https://github.com/PennyLaneAI/pennylane/pull/2185)

  ```pycon
  >>> qml.grouping.partition_pauli_group(2)
  [['II', 'IZ', 'ZI', 'ZZ'],
   ['IX', 'ZX'],
   ['IY', 'ZY'],
   ['XI', 'XZ'],
   ['XX'],
   ['XY'],
   ['YI', 'YZ'],
   ['YX'],
   ['YY']]
  ```

<h3>Breaking changes</h3>

* The `MultiControlledX` operation now accepts a single `wires` keyword argument for both `control_wires` and `wires`.
  The single `wires` keyword should be all the control wires followed by a single target wire.
  [(#2121)](https://github.com/PennyLaneAI/pennylane/pull/2121)
  [(#2278)](https://github.com/PennyLaneAI/pennylane/pull/2278)

<h3>Deprecations</h3>

* The `qml.operation.Operation.get_parameter_shift` method has been deprecated
  and will be removed in a future release.
  [#2227](https://github.com/PennyLaneAI/pennylane/pull/2227)

  Instead, the functionalities for general parameter-shift rules in the
  `qml.gradients` module should be used, together with the operation attributes
  `parameter_frequencies` or `grad_recipe`.

* The `qml.finite_diff()` function has been deprecated and will be removed
  in an upcoming release. Instead,
  `qml.gradients.finite_diff()` can be used to compute purely quantum gradients
  (that is, gradients of tapes or QNode).
  [#2212](https://github.com/PennyLaneAI/pennylane/pull/2212)

* `qml.transforms.get_unitary_matrix()` has been deprecated and will be removed
  in a future release. For extracting matrices of operations and quantum functions,
  please use `qml.matrix()`.
  [(#2248)](https://github.com/PennyLaneAI/pennylane/pull/2248)

<h3>Bug fixes</h3>

* The `qml.RandomLayers` template now decomposes when the weights are a list of lists.
  [(#2266)](https://github.com/PennyLaneAI/pennylane/pull/2266/)

* The `qml.QubitUnitary` operation now supports jitting. 
  [(#2249)](https://github.com/PennyLaneAI/pennylane/pull/2249)

* Fixes a bug in the JAX interface where ``DeviceArray`` objects
  were not being converted to NumPy arrays before executing an
  external device.
  [(#2255)](https://github.com/PennyLaneAI/pennylane/pull/2255)

* The ``qml.ctrl`` transform now works correctly with gradient transforms
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

<h3>Documentation</h3>

* Link to the strawberry fields docs for information on the CV model.
  [(#2259)](https://github.com/PennyLaneAI/pennylane/pull/2259)

* Fixes the documentation example for `qml.QFT`.
  [(#2232)](https://github.com/PennyLaneAI/pennylane/pull/2232)

* Fixes the documentation example for using `qml.sample` with `jax.jit`.
  [(#2196)](https://github.com/PennyLaneAI/pennylane/pull/2196)

* The `qml.numpy` subpackage is now included in the PennyLane
  API documentation.
  [(#2179)](https://github.com/PennyLaneAI/pennylane/pull/2179)

* Improves the documentation of `RotosolveOptimizer` regarding the
  usage of the passed `substep_optimizer` and its keyword arguments.
  [(#2160)](https://github.com/PennyLaneAI/pennylane/pull/2160)

<h3>Operator class refactor</h3>

The Operator class has undergone a major refactor with the following changes:

* The static `compute_decomposition` method defines the decomposition
  of an operator into a product of simpler operators, and the instance method
  `decomposition()` computes this for a given instance. When a custom
  decomposition does not exist, the code now raises a custom `NoDecompositionError`
  instead of `NotImplementedError`.
  [(#2024)](https://github.com/PennyLaneAI/pennylane/pull/2024)

* The `diagonalizing_gates()` representation has been moved to the highest-level
  `Operator` class and is therefore available to all subclasses. A condition
  `qml.operation.defines_diagonalizing_gates` has been added, which can be used
  in tape contexts without queueing.
  [(#1985)](https://github.com/PennyLaneAI/pennylane/pull/1985)

* A static `compute_diagonalizing_gates` method has been added, which is called
  by default in `diagonalizing_gates()`.
  [(#1993)](https://github.com/PennyLaneAI/pennylane/pull/1993)

* A `hyperparameters` attribute was added to the operator class.
  [(#2017)](https://github.com/PennyLaneAI/pennylane/pull/2017)

* The representation of an operator as a matrix has been overhauled.

  The `matrix()` method now accepts a
  `wire_order` argument and calculates the correct numerical representation
  with respect to that ordering.

  ```pycon
  >>> op = qml.RX(0.5, wires="b")
  >>> op.matrix()
  [[0.96891242+0.j         0.        -0.24740396j]
   [0.        -0.24740396j 0.96891242+0.j        ]]
  >>> op.matrix(wire_order=["a", "b"])
  [[0.9689+0.j  0.-0.2474j 0.+0.j         0.+0.j]
   [0.-0.2474j  0.9689+0.j 0.+0.j         0.+0.j]
   [0.+0.j          0.+0.j 0.9689+0.j 0.-0.2474j]
   [0.+0.j          0.+0.j 0.-0.2474j 0.9689+0.j]]
  ```

  The "canonical matrix", which is independent of wires,
  is now defined in the static method `compute_matrix()` instead of `_matrix`.
  By default, this method is assumed to take all parameters and non-trainable
  hyperparameters that define the operation.

  ```pycon
  >>> qml.RX.compute_matrix(0.5)
  [[0.96891242+0.j         0.        -0.24740396j]
   [0.        -0.24740396j 0.96891242+0.j        ]]
  ```

  If no canonical matrix is specified for a gate, `compute_matrix()`
  raises a `NotImplementedError`.

  The new `matrix()` method is now used in the
  `pennylane.transforms.get_qubit_unitary()` transform.
  [(#1996)](https://github.com/PennyLaneAI/pennylane/pull/1996)

* The `string_for_inverse` attribute is removed.
  [(#2021)](https://github.com/PennyLaneAI/pennylane/pull/2021)

* A `terms()` method and a `compute_terms()` static method were added to `Operator`.
  Currently, only the `Hamiltonian` class overwrites `compute_terms` to store
  coefficients and operators. The `Hamiltonian.terms` property hence becomes
  a proper method called by `Hamiltonian.terms()`.

* The generator property has been updated to an instance method,
  `Operator.generator()`. It now returns an instantiated operation,
  representing the generator of the instantiated operator.
  [(#2030)](https://github.com/PennyLaneAI/pennylane/pull/2030)
  [(#2061)](https://github.com/PennyLaneAI/pennylane/pull/2061)

  Various operators have been updated to specify the generator as either
  an `Observable`, `Tensor`, `Hamiltonian`, `SparseHamiltonian`, or `Hermitian`
  operator.

  In addition, a temporary utility function get_generator has been added
  to the utils module, to automate:

  - Extracting the matrix representation
  - Extracting the 'coefficient' if possible (only occurs if the generator is a single Pauli word)
  - Converting a Hamiltonian to a sparse matrix if there are more than 1 Pauli word present.
  - Negating the coefficient/taking the adjoint of the matrix if the operation was inverted

  This utility logic is currently needed because:

  - Extracting the matrix representation is not supported natively on
    Hamiltonians and SparseHamiltonians.
  - By default, calling `op.generator()` does not take into account `op.inverse()`.
  - If the generator is a single Pauli word, it is convenient to have access to
    both the coefficient and the observable separately.

* Decompositions are now defined in `compute_decomposition`, instead of `expand`.
  [(#2053)](https://github.com/PennyLaneAI/pennylane/pull/2053)

* The `expand` method was moved to the main `Operator` class.
  [(#2053)](https://github.com/PennyLaneAI/pennylane/pull/2053)

* A `sparse_matrix` method and a `compute_sparse_matrix` static method were added
  to the `Operator` class. The sparse representation of `SparseHamiltonian`
  is moved to this method, so that its `matrix` method now returns a dense matrix.
  [(#2050)](https://github.com/PennyLaneAI/pennylane/pull/2050)

* The argument `wires` in `heisenberg_obs`, `heisenberg_expand` and `heisenberg_tr`
  was renamed to `wire_order` to be consistent with other matrix representations.
  [(#2051)](https://github.com/PennyLaneAI/pennylane/pull/2051)

* The property `kraus_matrices` has been changed to a method, and `_kraus_matrices` renamed to
  `compute_kraus_matrices`, which is now a static method.
  [(#2055)](https://github.com/PennyLaneAI/pennylane/pull/2055)

* The developer guide on adding templates and the architecture overview were rewritten
  to reflect the past and planned changes of the operator refactor.
  [(#2066)](https://github.com/PennyLaneAI/pennylane/pull/2066)

* Custom errors subclassing ``OperatorPropertyUndefined`` are raised if a representation
  has not been defined. This replaces the ``NotImplementedError`` and allows finer control
  for developers.
  [(#2064)](https://github.com/PennyLaneAI/pennylane/pull/2064)

* Moved ``expand()`` from ``Operation`` to ``Operator``.
  [(#2239)](https://github.com/PennyLaneAI/pennylane/pull/2239)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Sam Banning, Thomas Bromley, Anthony Hayes, David Ittah, Josh Izaac, Christina Lee, Angus Lowe,
Maria Fernanda Morris, Romain Moyard, Zeyue Niu, Maria Schuld, Jay Soni,
Antal Száva, David Wierichs
