:orphan:

# Release 0.21.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements</h3>

<h3>Breaking changes</h3>

* Certain features deprecated in `v0.19.0` have been removed:

  - The `qml.template` decorator;
  - The `default.tensor` and `default.tensor.tf` experimental devices;
  - The `qml.fourier.spectrum` function;
  - The `diag_approx` keyword argument of `qml.metric_tensor` and `qml.QNGOptimizer`.
  [(#1981)](https://github.com/PennyLaneAI/pennylane/pull/1981)

* The default behaviour of the `qml.metric_tensor` transform has been modified:
  By default, the full metric tensor is computed, leading to higher cost than the previous
  default of computing the block diagonal only. At the same time, the Hadamard tests for
  the full metric tensor require an additional wire on the device, so that 

  ```pycon
  >>> qml.metric_tensor(some_qnode)(weights)
  ```

  will revert back to the block diagonal restriction and raise a warning if the
  used device does not have an additional wire.
  [(#1725)](https://github.com/PennyLaneAI/pennylane/pull/1725)

* The `circuit_drawer` module has been renamed `drawer`.
  [(#1949)](https://github.com/PennyLaneAI/pennylane/pull/1949)

* The `par_domain` attribute in the operator class has been removed.
  [(#1907)](https://github.com/PennyLaneAI/pennylane/pull/1907)

- The `mutable` keyword argument has been removed from the QNode.
  [(#1807)](https://github.com/PennyLaneAI/pennylane/pull/1807)

- The reversible QNode differentiation method has been removed.
  [(#1807)](https://github.com/PennyLaneAI/pennylane/pull/1807)

* `QuantumTape.trainable_params` now is a list instead of a set. This
  means that `tape.trainable_params` will return a list unlike before,
  but setting the `trainable_params` with a set works exactly as before.
  [(#1904)](https://github.com/PennyLaneAI/pennylane/pull/1904)

* The `num_params` attribute in the operator class is now dynamic. This makes it easier
  to define operator subclasses with a flexible number of parameters.
  [(#1898)](https://github.com/PennyLaneAI/pennylane/pull/1898)

* The static method `decomposition()`, formerly in the `Operation` class, has
  been moved to the base `Operator` class.
  [(#1873)](https://github.com/PennyLaneAI/pennylane/pull/1873)

* `DiagonalOperation` is not a separate subclass any more.
  [(#1889)](https://github.com/PennyLaneAI/pennylane/pull/1889)

  Instead, devices can check for the diagonal
  property using attributes:

  ``` python
  from pennylane.ops.qubit.attributes import diagonal_in_z_basis

  if op in diagonal_in_z_basis:
      # do something
  ```

<h3>Deprecations</h3>

* The init module, which contains functions to generate random parameters for 
  templates, has been removed. Instead, the templates provide a `shape()` method.
  [(#1963)](https://github.com/PennyLaneAI/pennylane/pull/1963)

<h3>Bug fixes</h3>

* Fixes a bug where the metric tensor was computed incorrectly when using
  gates with `gate.inverse=True`.
  [(#1987)](https://github.com/PennyLaneAI/pennylane/pull/1987)

* Corrects the documentation of `qml.transforms.classical_jacobian`
  for the Autograd interface (and improves test coverage).
  [(#1978)](https://github.com/PennyLaneAI/pennylane/pull/1978)

* Fixes a bug where differentiating a QNode with `qml.state` using the JAX
  interface raised an error.
  [(#1906)](https://github.com/PennyLaneAI/pennylane/pull/1906)

* Fixes a bug where the `ApproxTimeEvolution` template was not correctly
  computing the operation wires from the input Hamiltonian. This did not
  affect computation with the `ApproxTimeEvolution` template, but did
  cause circuit drawing to fail.
  [(#1952)](https://github.com/PennyLaneAI/pennylane/pull/1952)

* Fixes a bug where the classical preprocessing Jacobian
  computed by `qml.transforms.classical_jacobian` with JAX
  returned a reduced submatrix of the Jacobian.
  [(#1935)](https://github.com/PennyLaneAI/pennylane/pull/1935)

* Fixes a bug where the operations are not accessed in the correct order
  in `qml.fourier.qnode_spectrum`, leading to wrong outputs.
  [(#1935)](https://github.com/PennyLaneAI/pennylane/pull/1935)

* Fixes several Pylint errors.
  [(#1951)](https://github.com/PennyLaneAI/pennylane/pull/1951)

* Fixes a bug where the device test suite wasn't testing certain operations.
  [(#1943)](https://github.com/PennyLaneAI/pennylane/pull/1943)

* Fixes a bug where batch transforms would mutate a QNodes execution options.
  [(#1934)](https://github.com/PennyLaneAI/pennylane/pull/1934)

* `qml.draw` now supports arbitrary templates with matrix parameters.
  [(#1917)](https://github.com/PennyLaneAI/pennylane/pull/1917)

* `QuantumTape.trainable_params` now is a list instead of a set, making
  it more stable in very rare edge cases.
  [(#1904)](https://github.com/PennyLaneAI/pennylane/pull/1904)

* `ExpvalCost` now returns corrects results shape when `optimize=True` with
  shots batch.
  [(#1897)](https://github.com/PennyLaneAI/pennylane/pull/1897)

* `qml.circuit_drawer.MPLDrawer` was slightly modified to work with
  matplotlib version 3.5.
  [(#1899)](https://github.com/PennyLaneAI/pennylane/pull/1899)

* `qml.CSWAP` and `qml.CRot` now define `control_wires`, and `qml.SWAP`
  returns the default empty wires object.
  [(#1830)](https://github.com/PennyLaneAI/pennylane/pull/1830)

* The `requires_grad` attribute of `qml.numpy.tensor` objects is now
  preserved when pickling/unpickling the object.
  [(#1856)](https://github.com/PennyLaneAI/pennylane/pull/1856)

* Device tests no longer throw warnings about the `requires_grad`
  attribute of variational parameters.
  [(#1913)](https://github.com/PennyLaneAI/pennylane/pull/1913)

* `AdamOptimizer` and `AdagradOptimizer` had small fixes to their
  optimization step updates.
  [(#1929)](https://github.com/PennyLaneAI/pennylane/pull/1929)

* `AmplitudeEmbedding` template no longer produces a `ComplexWarning`
  when the `features` parameter is batched and provided as a 2D array.
  [(#1990)](https://github.com/PennyLaneAI/pennylane/pull/1990)

* `qml.circuit_drawer.CircuitDrawer` no longer produces an error
  when attempting to draw tapes inside of circuits (e.g. from
  decomposition of an operation or manual placement).
  [(#1994)](https://github.com/PennyLaneAI/pennylane/pull/1994)

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):
