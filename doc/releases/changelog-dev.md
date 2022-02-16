:orphan:

# Release 0.22.0-dev (development release)

<h3>New features since last release</h3>

* The text based drawer accessed via `qml.draw` has been overhauled. The new drawer has 
  a `decimals` keyword for controlling parameter rounding, a different algorithm for determining positions, 
  deprecation of the `charset` keyword, and minor cosmetic changes.
  [(#2128)](https://github.com/PennyLaneAI/pennylane/pull/2128)

  ```
  @qml.qnode(qml.device('lightning.qubit', wires=2))
  def circuit(a, w):
      qml.Hadamard(0)
      qml.CRX(a, wires=[0, 1])
      qml.Rot(*w, wires=[1])
      qml.CRX(-a, wires=[0, 1])
      return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
  ```
  >>> print(qml.draw(circuit, decimals=2)(a=2.3, w=[1.2, 3.2, 0.7]))
  0: ──H─╭C─────────────────────────────╭C─────────┤ ╭<Z@Z>
  1: ────╰RX(2.30)──Rot(1.20,3.20,0.70)─╰RX(-2.30)─┤ ╰<Z@Z>

* Parametric operations now have the `parameter_frequencies`
  method that returns the frequencies with which a parameter
  enters a circuit when using the operation.
  [(#2180)](https://github.com/PennyLaneAI/pennylane/pull/2180)

  The frequencies can be used for circuit analysis, optimization
  via the `RotosolveOptimizer` and differentiation with the
  parameter-shift rule. They assume that the circuit returns
  expectation values or probabilities, for a variance
  measurement the frequencies will differ.

* Continued development of the circuit-cutting compiler:

  A method for converting a quantum tape to a directed multigraph that is amenable
  to graph partitioning algorithms for circuit cutting has been added.
  [(#2107)](https://github.com/PennyLaneAI/pennylane/pull/2107)

  A method to replace `WireCut` nodes in a directed multigraph with `MeasureNode`
  and `PrepareNode` placeholders has been added.
  [(#2124)](https://github.com/PennyLaneAI/pennylane/pull/2124)

  A method has been added that takes a directed multigraph with `MeasureNode` and
  `PrepareNode` placeholders and fragments into subgraphs and a communication graph.
  [(#2153)](https://github.com/PennyLaneAI/pennylane/pull/2153)

  A method has been added that takes a directed multigraph with `MeasureNode`
  and `PrepareNode` placeholder nodes and converts it into a tape.
  [(#2165)](https://github.com/PennyLaneAI/pennylane/pull/2165)

  A differentiable tensor contraction function `contract_tensors` has been
  added.
  [(#2158)](https://github.com/PennyLaneAI/pennylane/pull/2158)

<h3>Improvements</h3>

* Added a new `partition_pauli_group` function to the `grouping` module for
  efficiently measuring the `N`-qubit Pauli group with `3 ** N`
  qubit-wise commuting terms.
  [(#2185)](https://github.com/PennyLaneAI/pennylane/pull/2185)
  
  ```pycon
  >>> qml.grouping.partition_pauli_group(3)
  [['III', 'IIZ', 'IZI', 'IZZ', 'ZII', 'ZIZ', 'ZZI', 'ZZZ'],
   ['IIX', 'IZX', 'ZIX', 'ZZX'],
   ['IIY', 'IZY', 'ZIY', 'ZZY'],
   ['IXI', 'IXZ', 'ZXI', 'ZXZ'],
   ['IXX', 'ZXX'],
   ['IXY', 'ZXY'],
   ['IYI', 'IYZ', 'ZYI', 'ZYZ'],
   ['IYX', 'ZYX'],
   ['IYY', 'ZYY'],
   ['XII', 'XIZ', 'XZI', 'XZZ'],
   ['XIX', 'XZX'],
   ['XIY', 'XZY'],
   ['XXI', 'XXZ'],
   ['XXX'],
   ['XXY'],
   ['XYI', 'XYZ'],
   ['XYX'],
   ['XYY'],
   ['YII', 'YIZ', 'YZI', 'YZZ'],
   ['YIX', 'YZX'],
   ['YIY', 'YZY'],
   ['YXI', 'YXZ'],
   ['YXX'],
   ['YXY'],
   ['YYI', 'YYZ'],
   ['YYX'],
   ['YYY']]
  ```

<h3>Breaking changes</h3>

* The `MultiControlledX` operation now accepts a single `wires` keyword argument for both `control_wires` and `wires`.
  The single `wires` keyword should be all the control wires followed by a single target wire. 
  [(#2121)](https://github.com/PennyLaneAI/pennylane/pull/2121)

<h3>Deprecations</h3>

<h3>Bug fixes</h3>

* Fixes a bug in which passing required arguments into operations as
  keyword arguments would throw an error because the documented call
  signature didn't match the function definition.
  [(#1976)](https://github.com/PennyLaneAI/pennylane/pull/1976)

* The operation `OrbitalRotation` previously was wrongfully registered to satisfy
  the four-term parameter shift rule, it now will be decomposed instead when
  using the parameter-shift rule.
  [(#2180)](https://github.com/PennyLaneAI/pennylane/pull/2180)

<h3>Documentation</h3>

* Fixes the example for using `qml.sample` with `jax.jit`.
  [(#2196)](https://github.com/PennyLaneAI/pennylane/pull/2196)

* The ``pennylane.numpy`` subpackage is now included in the PennyLane
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


<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Thomas Bromley, Anthony Hayes, Josh Izaac, Christina Lee, 
Maria Fernanda Morris, Maria Schuld, Jay Soni, Antal Száva,
David Wierichs

