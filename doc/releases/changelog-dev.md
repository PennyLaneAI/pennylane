:orphan:

# Release 0.35.0-dev (development release)

<h3>New features since last release</h3>

<h4>Native mid-circuit measurements on default qubit 💡</h4>

<h4>Work easily and efficiently with Pauli operators 🔧</h4>

* New `qml.commutator` function that allows to compute commutators between
  `qml.operation.Operator`, `qml.pauli.PauliWord` and `qml.pauli.PauliSentence` instances.
  [(#5051)](https://github.com/PennyLaneAI/pennylane/pull/5051)
  [(#5052)](https://github.com/PennyLaneAI/pennylane/pull/5052)

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

  We can also compute commutators with Pauli operators natively with the `PauliSentence.commutator` method.

  ```pycon
  >>> op1 = PauliWord({0:"X", 1:"X"})
  >>> op2 = PauliWord({0:"Y"}) + PauliWord({1:"Y"})
  >>> op1.commutator(op2)
  2j * Z(0) @ X(1)
  + 2j * X(0) @ Z(1)
  ```

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

* Composite operations (eg. those made with `qml.prod` and `qml.sum`) and `SProd` operations convert `Hamiltonian` and
  `Tensor` operands to `Sum` and `Prod` types, respectively. This helps avoid the mixing of
  incompatible operator types.
  [(#5031)](https://github.com/PennyLaneAI/pennylane/pull/5031)
  [(#5063)](https://github.com/PennyLaneAI/pennylane/pull/5063)

* `qml.Identity()` can be initialized without wires. Measuring it is currently not possible though.
  [(#5106)](https://github.com/PennyLaneAI/pennylane/pull/5106)

<h4>Easy to inspect transforms 🔎</h4>

<h4>New Clifford and noisy qutrit devices 🦾</h4>

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

* A function called `apply_operation` has been added to the new `qutrit_mixed` module found in `qml.devices` that applies operations to device-compatible states.
  [(#5032)](https://github.com/PennyLaneAI/pennylane/pull/5032)

<h3>Improvements 🛠</h3>

<h4>Faster gradients with VJPs and other performance improvements</h4>

* Adjoint device VJP's are now supported with `jax.jacobian`. `device_vjp=True` is
  is now strictly faster for jax.
  [(#4963)](https://github.com/PennyLaneAI/pennylane/pull/4963)

* `device_vjp` can now be used with normal Tensorflow. Support has not yet been added
  for `tf.Function` and Tensorflow Autograph.
  [(#4676)](https://github.com/PennyLaneAI/pennylane/pull/4676)

* PennyLane can now use lightning provided VJPs by selecting `device_vjp=True` on the QNode.
  [(#4914)](https://github.com/PennyLaneAI/pennylane/pull/4914)

* Remove queuing (`AnnotatedQueue`) from `qml.cut_circuit` and `qml.cut_circuit_mc` to improve performance 
  for large workflows.
  [(#5108)](https://github.com/PennyLaneAI/pennylane/pull/5108)

* Improve the performance of circuit-cutting workloads with large numbers of generated tapes.
  [(#5005)](https://github.com/PennyLaneAI/pennylane/pull/5005)

<h4>Community contributions 🥳</h4>

* `parity_transform` is added for parity mapping of a fermionic Hamiltonian.
   [(#4928)](https://github.com/PennyLaneAI/pennylane/pull/4928)
   It is now possible to transform a fermionic Hamiltonian to a qubit Hamiltonian with parity mapping.

   ```python
   import pennylane as qml
   fermi_ham = qml.fermi.FermiWord({(0, 0) : '+', (1, 1) : '-'})

   qubit_ham = qml.fermi.parity_transform(fermi_ham, n=6)
   ```

   ```pycon
   >>> print(qubit_ham)
   (-0.25j*(PauliY(wires=[0]))) + ((-0.25+0j)*(PauliX(wires=[0]) @ PauliZ(wires=[1]))) +
   ((0.25+0j)*(PauliX(wires=[0]))) + (0.25j*(PauliY(wires=[0]) @ PauliZ(wires=[1])))
   ```

* The transform `split_non_commuting` now accepts measurements of type `probs`, `sample` and `counts` which accept both wires and observables.
  [(#4972)](https://github.com/PennyLaneAI/pennylane/pull/4972)

* Improve efficiency of matrix calculation when operator is symmetric over wires
   [(#3601)](https://github.com/PennyLaneAI/pennylane/pull/3601)

* The module `pennylane/math/quantum.py` has now support for the min-entropy.
  [(#3959)](https://github.com/PennyLaneAI/pennylane/pull/3959/)

<h4>Other improvements</h4>

* Faster `qml.probs` measurements due to an optimization in `_samples_to_counts`.
  [(#5145)](https://github.com/PennyLaneAI/pennylane/pull/5145)

* Cuts down on performance bottlenecks in converting a `PauliSentence` to a `Sum`.
  [(#5141)](https://github.com/PennyLaneAI/pennylane/pull/5141)
  [(#5150)](https://github.com/PennyLaneAI/pennylane/pull/5150)

* The `qml.qsvt` function uses `qml.GlobalPhase` instead of `qml.exp` to define global phase.
  [(#5105)](https://github.com/PennyLaneAI/pennylane/pull/5105)

* Update `tests/ops/functions/conftest.py` to ensure all operator types are tested for validity.
  [(#4978)](https://github.com/PennyLaneAI/pennylane/pull/4978)

* A new `pennylane.workflow` module is added. This module now contains `qnode.py`, `execution.py`, `set_shots.py`, `jacobian_products.py`, and the submodule `interfaces`.
  [(#5023)](https://github.com/PennyLaneAI/pennylane/pull/5023)

* Raise a more informative error when calling `adjoint_jacobian` with trainable state-prep operations.
  [(#5026)](https://github.com/PennyLaneAI/pennylane/pull/5026)

* Adds `qml.workflow.get_transform_program` and `qml.workflow.construct_batch` to inspect the transform program and batch of tapes
  at different stages.
  [(#5084)](https://github.com/PennyLaneAI/pennylane/pull/5084)

* `CRX`, `CRY`, `CRZ`, `CROT`, and `ControlledPhaseShift` (i.e. `CPhaseShift`) now inherit from `ControlledOp`, giving them additional properties such as `control_wire` and `control_values`. Calling `qml.ctrl` on `RX`, `RY`, `RZ`, `Rot`, and `PhaseShift` with a single control wire will return gates of types `CRX`, `CRY`, etc. as opposed to a general `Controlled` operator.
  [(#5069)](https://github.com/PennyLaneAI/pennylane/pull/5069)

* CI will now fail if coverage data fails to upload to codecov. Previously, it would silently pass
  and the codecov check itself would never execute.
  [(#5101)](https://github.com/PennyLaneAI/pennylane/pull/5101)

* String representations of Pauli operators have been improved and there are new aliases `X, Y, Z, I` for `PauliX, PauliY, PauliZ, Identity`.
  ```
  >>> qml.PauliX(0)
  X(0)
  >>> qml.PauliX('a')
  X('a')
  >>> 0.5 * X(0)
  0.5 * X(0)
  >>> 0.5 * (X(0) + Y(1))
  0.5 * (X(0) + Y(1))
  ```
  [(#5116)](https://github.com/PennyLaneAI/pennylane/pull/5116)

* `qml.ctrl` called on operators with custom controlled versions will return instances
  of the custom class, and it will also flatten nested controlled operators to a single
  multi-controlled operation. For `PauliX`, `CNOT`, `Toffoli`, and `MultiControlledX`,
  calling `qml.ctrl` will always resolve to the best option in `CNOT`, `Toffoli`, or
  `MultiControlledX` depending on the number of control wires and control values.
  [(#5125)](https://github.com/PennyLaneAI/pennylane/pull/5125/)

* Remove the unwanted warning filter from tests, and ensure that no PennyLaneDeprecationWarnings
  are being raised unexpectedly.
  [(#5122)](https://github.com/PennyLaneAI/pennylane/pull/5122)


<h4>Community contributions 🥳</h4>

* The function `partial_trace` has been refactored to be public-facing for computing the partial trace of matrices other than density matrices.
  [(#5152)](https://github.com/PennyLaneAI/pennylane/pull/5152)

<h3>Breaking changes 💔</h3>

* Make PennyLane code compatible with the latest version of `black`.
  [(#5112)](https://github.com/PennyLaneAI/pennylane/pull/5112)
  [(#5119)](https://github.com/PennyLaneAI/pennylane/pull/5119)

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

* Controlled operators with a custom controlled version decomposes like how their
  controlled counterpart decomposes, as opposed to decomposing into their controlled version.   
  [(#5069)](https://github.com/PennyLaneAI/pennylane/pull/5069)
  [(#5125)](https://github.com/PennyLaneAI/pennylane/pull/5125/)
  
  For example:
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
  [(#5122)](https://github.com/PennyLaneAI/pennylane/pull/5122)

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

* Documentation `qml.data` has been updated and now mentions a way to access the same dataset simultaneously from multiple environments.
  [(#5029)](https://github.com/PennyLaneAI/pennylane/pull/5029)

* Clarification for the definition of `argnum` added to gradient methods
  [(#5035)](https://github.com/PennyLaneAI/pennylane/pull/5035)

* A typo in the code example for `qml.qchem.dipole_of` has been fixed.
  [(#5036)](https://github.com/PennyLaneAI/pennylane/pull/5036)

* Added a development guide on deprecations and removals.
  [(#5083)](https://github.com/PennyLaneAI/pennylane/pull/5083)

* A note about the eigenspectrum of second-quantized Hamiltonians added to `qml.eigvals`.
  [(#5095)](https://github.com/PennyLaneAI/pennylane/pull/5095)

<h3>Bug fixes 🐛</h3>

* The `qml.TrotterProduct` template is updated to accept `SProd` as input Hamiltonian.
  [(#5073)](https://github.com/PennyLaneAI/pennylane/pull/5073)

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

* `PauliSentence.wires` no longer imposes a false order.
  [(#5041)](https://github.com/PennyLaneAI/pennylane/pull/5041)

* `qml.qchem.import_state` now applies the chemist-to-physicist 
  sign convention when initializing a PennyLane state vector from
  classically pre-computed wavefunctions. That is, it interleaves 
  spin-up/spin-down operators for the same spatial orbital index,
  as standard in PennyLane (instead of commuting all spin-up 
  operators to the left, as is standard in quantum chemistry). 
  [(#5114)](https://github.com/PennyLaneAI/pennylane/pull/5114)

* Multi-wire controlled `CNOT` and `PhaseShift` can now be decomposed correctly.
  [(#5125)](https://github.com/PennyLaneAI/pennylane/pull/5125/) 
  [(#5148)](https://github.com/PennyLaneAI/pennylane/pull/5148)

* `draw_mpl` no longer raises an error when drawing a circuit containing an adjoint of a controlled operation.
  [(#5149)](https://github.com/PennyLaneAI/pennylane/pull/5149)

* `default.mixed` no longer throws `ValueError` when applying a state vector that is not of type `complex128` when used with tensorflow.
  [(#5155)](https://github.com/PennyLaneAI/pennylane/pull/5155)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Abhishek Abhishek,
Utkarsh Azad,
Trenten Babcock,
Gabriel Bottrill,
Astral Cai,
Isaac De Vlugt,
Diksha Dhawan,
Eugenio Gigante,
Diego Guala,
Soran Jahangiri,
Korbinian Kottmann,
Christina Lee,
Xiaoran Li,
Vincent Michaud-Rioux,
Romain Moyard,
Pablo Antonio Moreno Casares,
Lee J. O'Riordan,
Mudit Pandey,
Alex Preciado,
Matthew Silverman,
Jay Soni.

