:orphan:

# Release 0.21.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements</h3>

<h3>Breaking changes</h3>

<h3>Bug fixes</h3>

* Fixes a bug in `classical_jacobian` when used with Torch, where the
  Jacobian of the preprocessing was also computed for non-trainable
  parameters.
  [(#2020)](https://github.com/PennyLaneAI/pennylane/pull/2020)

* Fixes a bug in queueing of the `two_qubit_decomposition` method that
  originally led to circuits with >3 two-qubit unitaries failing when passed
  through the `unitary_to_rot` optimization transform.
  [(#2015)](https://github.com/PennyLaneAI/pennylane/pull/2015)

<h3>Documentation</h3>

<h3>Operator class refactor</h3>

The Operator class has undergone a major refactor with the following changes:

* The `diagonalizing_gates()` representation has been moved to the highest-level
  `Operator` class and is therefore available to all subclasses. A condition
  `qml.operation.defines_diagonalizing_gates` has been added, which can be used
  in tape contexts without queueing.
  [(#1985)](https://github.com/PennyLaneAI/pennylane/pull/1985)

* A `hyperparameters` attribute was added to the operator class.
  [(#2017)](https://github.com/PennyLaneAI/pennylane/pull/2017)

* The `string_for_inverse` attribute is removed.
  [(#2021)](https://github.com/PennyLaneAI/pennylane/pull/2021)

* The generator property has been updated to an instance method,
  `Operation.generator()`. It now returns an instantiated operation,
  representing the generator of the instantiated operator.
  [(#2030)](https://github.com/PennyLaneAI/pennylane/pull/2030)

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

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Olivia Di Matteo, Josh Izaac, Christina Lee, Maria Schuld, David Wierichs.

