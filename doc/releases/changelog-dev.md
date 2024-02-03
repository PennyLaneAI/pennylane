:orphan:

# Release 0.35.0-dev (development release)

<h3>New features since last release</h3>

* A new `default.clifford` device enables efficient simulation of large-scale Clifford circuits
  defined in PennyLane through the use of [stim](https://github.com/quantumlib/Stim) as a backend.
  [(#4936)](https://github.com/PennyLaneAI/pennylane/pull/4936)

  Given a circuit with only Clifford gates, one can use this device to obtain the usual range
  of PennyLane [measurements](https://docs.pennylane.ai/en/stable/introduction/measurements.html)
  as well as the state represented in the Tableau form of
  [Aaronson & Gottesman (2004)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.70.052328):

  ```python
  import pennylane as qml

  dev = qml.device("default.clifford", tableau=True)

  @qml.qnode(dev)
  def circuit():
      qml.CNOT(wires=[0, 1])
      qml.PauliX(wires=[1])
      qml.ISWAP(wires=[0, 1])
      qml.Hadamard(wires=[0])
      return qml.state()
  ```

  ```pycon
  >>> circuit()
  array([[0, 1, 1, 0, 0],
         [1, 0, 1, 1, 1],
         [0, 0, 0, 1, 0],
         [1, 0, 0, 1, 1]])
  ```

* Adjoint device VJP's are now supported with `jax.jacobian`. `device_vjp=True` is
  is now strictly faster for jax.
  [(#4963)](https://github.com/PennyLaneAI/pennylane/pull/4963)

* New `qml.commutator` function that allows to compute commutators between
  `qml.operation.Operator`, `qml.pauli.PauliWord` and `qml.pauli.PauliSentence` instances.
  [(#5051)](https://github.com/PennyLaneAI/pennylane/pull/5051)

  Basic usage with PennyLane operators.

  ```pycon
  >>> qml.commutator(qml.PauliX(0), qml.PauliY(0))
  2j*(PauliZ(wires=[0]))
  ```

  We can return a `PauliSentence` instance by setting `pauli=True`.

  ```pycon
  >>> op1 = qml.PauliX(0) @ qml.PauliX(1)
  >>> op2 = qml.PauliY(0) + qml.PauliY(1)
  >>> qml.commutator(op1, op2, pauli=True)
  2j * X(1) @ Z(0)
  + 2j * Z(1) @ X(0)
  ```

  We can also input `PauliWord` and `PauliSentence` instances.

  ```pycon
  >>> op1 = PauliWord({0:"X", 1:"X"})
  >>> op2 = PauliWord({0:"Y"}) + PauliWord({1:"Y"})
  >>> qml.commutator(op1, op2, pauli=True)
  2j * Z(0) @ X(1)
  + 2j * X(0) @ Z(1)
  ```

<h3>Improvements 🛠</h3>

* Remove queuing (`AnnotatedQueue`) from `qml.cut_circuit` and `qml.cut_circuit_mc` to improve performance 
  for large workflows.
  [(#5108)](https://github.com/PennyLaneAI/pennylane/pull/5108)

* `device_vjp` can now be used with normal Tensorflow. Support has not yet been added
  for `tf.Function` and Tensorflow Autograph.
  [(#4676)](https://github.com/PennyLaneAI/pennylane/pull/4676)

* Improve the performance of circuit-cutting workloads with large numbers of generated tapes.
  [(#5005)](https://github.com/PennyLaneAI/pennylane/pull/5005)

* Update `tests/ops/functions/conftest.py` to ensure all operator types are tested for validity.
  [(#4978)](https://github.com/PennyLaneAI/pennylane/pull/4978)

* Upgrade Pauli arithmetic:
  You can now multiply `PauliWord` and `PauliSentence` instances by scalars, e.g. `0.5 * PauliWord({0:"X"})` or `0.5 * PauliSentence({PauliWord({0:"X"}): 1.})`.
  You can now intuitively add together
  `PauliWord` and `PauliSentence` as well as scalars, which are treated implicitly as identities.
  For example `ps1 + pw1 + 1.` for some Pauli word `pw1 = PauliWord({0: "X", 1: "Y"})` and Pauli
  sentence `ps1 = PauliSentence({pw1: 3.})`.
  You can now subtract `PauliWord` and `PauliSentence` instances, as well as scalars, from each other. For example `ps1 - pw1 - 1`.
  Overall, you can now intuitively construct `PauliSentence` operators like `0.5 * pw1 - 1.5 * ps1 + 2`.
  [(#4989)](https://github.com/PennyLaneAI/pennylane/pull/4989)
  [(#5001)](https://github.com/PennyLaneAI/pennylane/pull/5001)
  [(#5003)](https://github.com/PennyLaneAI/pennylane/pull/5003)
  [(#5017)](https://github.com/PennyLaneAI/pennylane/pull/5017)

* `qml.matrix` now accepts `PauliWord` and `PauliSentence` instances, `qml.matrix(PauliWord({0:"X"}))`.
  [(#5018)](https://github.com/PennyLaneAI/pennylane/pull/5018)

* Improve efficiency of matrix calculation when operator is symmetric over wires
   [(#3601)](https://github.com/PennyLaneAI/pennylane/pull/3601)

* PennyLane can now use lightning provided VJPs by selecting `device_vjp=True` on the QNode.
  [(#4914)](https://github.com/PennyLaneAI/pennylane/pull/4914)

* A new `pennylane.workflow` module is added. This module now contains `qnode.py`, `execution.py`, `set_shots.py`, `jacobian_products.py`, and the submodule `interfaces`.
  [(#5023)](https://github.com/PennyLaneAI/pennylane/pull/5023)

* Composite operations (eg. those made with `qml.prod` and `qml.sum`) and `SProd` operations convert `Hamiltonian` and
  `Tensor` operands to `Sum` and `Prod` types, respectively. This helps avoid the mixing of
  incompatible operator types.
  [(#5031)](https://github.com/PennyLaneAI/pennylane/pull/5031)
  [(#5063)](https://github.com/PennyLaneAI/pennylane/pull/5063)

* Raise a more informative error when calling `adjoint_jacobian` with trainable state-prep operations.
  [(#5026)](https://github.com/PennyLaneAI/pennylane/pull/5026)

* `CRX`, `CRY`, `CRZ`, `CROT`, and `ControlledPhaseShift` (i.e. `CPhaseShift`) now inherit from `ControlledOp`, giving them additional properties such as `control_wire` and `control_values`. Calling `qml.ctrl` on `RX`, `RY`, `RZ`, `Rot`, and `PhaseShift` with a single control wire will return gates of types `CRX`, `CRY`, etc. as opposed to a general `Controlled` operator.
  [(#5069)](https://github.com/PennyLaneAI/pennylane/pull/5069)

* CI will now fail if coverage data fails to upload to codecov. Previously, it would silently pass
  and the codecov check itself would never execute.
  [(#5101)](https://github.com/PennyLaneAI/pennylane/pull/5101)

<h4>Community contributions 🥳</h4>

* The transform `split_non_commuting` now accepts measurements of type `probs`, `sample` and `counts` which accept both wires and observables.
  [(#4972)](https://github.com/PennyLaneAI/pennylane/pull/4972)

* A function called `apply_operation` has been added to the new `qutrit_mixed` module found in `qml.devices` that applies operations to device-compatible states.
  [(#5032)](https://github.com/PennyLaneAI/pennylane/pull/5032)

* The function `batched_partial_trace` has been refactored to be public-facing for computing the partial trace of matrices other than density matrices.


<h3>Breaking changes 💔</h3>

* Pin Black to `v23.12` to prevent unnecessary formatting changes.
  [(#5112)](https://github.com/PennyLaneAI/pennylane/pull/5112)

* `gradient_analysis_and_validation` is now renamed to `find_and_validate_gradient_methods`. Instead of returning a list, it now returns a dictionary of gradient methods for each parameter index, and no longer mutates the tape.
  [(#5035)](https://github.com/PennyLaneAI/pennylane/pull/5035)

* Passing additional arguments to a transform that decorates a QNode must be done through the use
  of `functools.partial`.
  [(#5046)](https://github.com/PennyLaneAI/pennylane/pull/5046)

* Multiplying two `PauliWord` instances no longer returns a tuple `(new_word, coeff)`
  but instead `PauliSentence({new_word: coeff})`. The old behavior is still available
  with the private method `PauliWord._matmul(other)` for faster processing.
  [(#5045)](https://github.com/PennyLaneAI/pennylane/pull/5054)

* `Observable.return_type` has been removed. Instead, you should inspect the type
  of the surrounding measurement process.
  [(#5044)](https://github.com/PennyLaneAI/pennylane/pull/5044)

* `ClassicalShadow.entropy()` no longer needs an `atol` keyword as a better
  method to estimate entropies from approximate density matrix reconstructions
  (with potentially negative eigenvalues) has been implemented.
  [(#5048)](https://github.com/PennyLaneAI/pennylane/pull/5048)

* The decomposition of an operator created with calling `qml.ctrl` on a parametric operator (specifically `RX`, `RY`, `RZ`, `Rot`, `PhaseShift`) with a single control wire will now be the full decomposition instead of a single controlled gate. For example:
  ```
  >>> qml.ctrl(qml.RX(0.123, wires=1), control=0).decomposition()
  [
    RZ(1.5707963267948966, wires=[1]),
    RY(0.0615, wires=[1]),
    CNOT(wires=[0, 1]),
    RY(-0.0615, wires=[1]),
    CNOT(wires=[0, 1]),
    RZ(-1.5707963267948966, wires=[1])
  ]
  ```
  [(#5069)](https://github.com/PennyLaneAI/pennylane/pull/5069)

* `QuantumScript.is_sampled` and `QuantumScript.all_sampled` have been removed. Users should now
  validate these properties manually.
  [(#5072)](https://github.com/PennyLaneAI/pennylane/pull/5072)

* `qml.transforms.one_qubit_decomposition` and `qml.transforms.two_qubit_decomposition` are removed. Instead,
  you should use `qml.ops.one_qubit_decomposition` and `qml.ops.two_qubit_decomposition`.
  [(#5091)](https://github.com/PennyLaneAI/pennylane/pull/5091)

* `qml.ExpvalCost` has been removed. Users should use `qml.expval()` moving forward.
  [(#5097)](https://github.com/PennyLaneAI/pennylane/pull/5097)

<h3>Deprecations 👋</h3>

* `Operator.validate_subspace(subspace)` has been relocated to the `qml.ops.qutrit.parametric_ops`
  module and will be removed from the Operator class in an upcoming release.
  [(#5067)](https://github.com/PennyLaneAI/pennylane/pull/5067)

* Matrix and tensor products between `PauliWord` and `PauliSentence` instances are done using
  the `@` operator, `*` will be used only for scalar multiplication. Note also the breaking
  change that the product of two `PauliWord` instances now returns a `PauliSentence` instead
  of a tuple `(new_word, coeff)`.
  [(#4989)](https://github.com/PennyLaneAI/pennylane/pull/4989)
  [(#5054)](https://github.com/PennyLaneAI/pennylane/pull/5054)

* `MeasurementProcess.name` and `MeasurementProcess.data` are now deprecated, as they contain dummy
  values that are no longer needed.
  [(#5047)](https://github.com/PennyLaneAI/pennylane/pull/5047)
  [(#5071)](https://github.com/PennyLaneAI/pennylane/pull/5071)
  [(#5076)](https://github.com/PennyLaneAI/pennylane/pull/5076)

* Calling `qml.matrix` without providing a `wire_order` on objects where the wire order could be
  ambiguous now raises a warning. In the future, the `wire_order` argument will be required in
  these cases.
  [(#5039)](https://github.com/PennyLaneAI/pennylane/pull/5039)

* `qml.pauli.pauli_mult` and `qml.pauli.pauli_mult_with_phase` are now deprecated. Instead, you
  should use `qml.simplify(qml.prod(pauli_1, pauli_2))` to get the reduced operator.
  [(#5057)](https://github.com/PennyLaneAI/pennylane/pull/5057)

* The private functions `_pauli_mult`, `_binary_matrix` and `_get_pauli_map` from the
  `pauli` module have been deprecated, as they are no longer used anywhere and the same
  functionality can be achieved using newer features in the `pauli` module.
  [(#5057)](https://github.com/PennyLaneAI/pennylane/pull/5057)

<h3>Documentation 📝</h3>

* The module documentation for `pennylane.tape` now explains the difference between `QuantumTape` and `QuantumScript`.
  [(#5065)](https://github.com/PennyLaneAI/pennylane/pull/5065)

* A typo in a code example in the `qml.transforms` API has been fixed.
  [(#5014)](https://github.com/PennyLaneAI/pennylane/pull/5014)

* Clarification for the definition of `argnum` added to gradient methods
  [(#5035)](https://github.com/PennyLaneAI/pennylane/pull/5035)

* A typo in the code example for `qml.qchem.dipole_of` has been fixed.
  [(#5036)](https://github.com/PennyLaneAI/pennylane/pull/5036)

* Added a development guide on deprecations and removals.
  [(#5083)](https://github.com/PennyLaneAI/pennylane/pull/5083)

* A note about the eigenspectrum of second-quantized Hamiltonians added to `qml.eigvals`.
  [(#5095)](https://github.com/PennyLaneAI/pennylane/pull/5095)

<h3>Bug fixes 🐛</h3>

* Fixed a bug where caching together with JIT compilation and broadcasted tapes yielded wrong results
  `Operator.hash` now depends on the memory location, `id`, of a Jax tracer instead of its string representation.
  [(#3917)](https://github.com/PennyLaneAI/pennylane/pull/3917)

* `qml.transforms.undo_swaps` can now work with operators with hyperparameters or nesting.
  [(#5081)](https://github.com/PennyLaneAI/pennylane/pull/5081)

* `qml.transforms.split_non_commuting` will now pass the original shots along.
  [(#5081)](https://github.com/PennyLaneAI/pennylane/pull/5081)

* If `argnum` is provided to a gradient transform, only the parameters specified in `argnum` will have their gradient methods validated.
  [(#5035)](https://github.com/PennyLaneAI/pennylane/pull/5035)

* `StatePrep` operations expanded onto more wires are now compatible with backprop.
  [(#5028)](https://github.com/PennyLaneAI/pennylane/pull/5028)

* `qml.equal` works well with `qml.Sum` operators when wire labels are a mix of integers and strings.
  [(#5037)](https://github.com/PennyLaneAI/pennylane/pull/5037)

* The return value of `Controlled.generator` now contains a projector that projects onto the correct subspace based on the control value specified.
  [(#5068)](https://github.com/PennyLaneAI/pennylane/pull/5068)

* `CosineWindow` no longer raises an unexpected error when used on a subset of wires at the beginning of a circuit.
  [(#5080)](https://github.com/PennyLaneAI/pennylane/pull/5080)

* Ensure `tf.function` works with `TensorSpec(shape=None)` by skipping batch size computation.
  [(#5089)](https://github.com/PennyLaneAI/pennylane/pull/5089)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Abhishek Abhishek,
Utkarsh Azad,
Gabriel Bottrill,
Astral Cai,
Isaac De Vlugt,
Korbinian Kottmann,
Christina Lee,
Xiaoran Li,
Vincent Michaud-Rioux,
Romain Moyard,
Pablo Antonio Moreno Casares,
Lee J. O'Riordan,
Mudit Pandey,
Alex Preciado,
Matthew Silverman.
Jay Soni,
