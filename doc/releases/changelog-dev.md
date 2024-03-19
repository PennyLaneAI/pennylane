:orphan:

# Release 0.36.0-dev (development release)

<h3>New features since last release</h3>

* The `QubitDevice` class and children classes support the `dynamic_one_shot` transform provided that they support `MidMeasureMP` operations natively.
  [(#5317)](https://github.com/PennyLaneAI/pennylane/pull/5317)

* `qml.ops.Sum` now supports storing grouping information. Grouping type and method can be
  specified during construction using the `grouping_type` and `method` keyword arguments of
  `qml.dot`, `qml.sum`, or `qml.ops.Sum`. The grouping indices are stored in `Sum.grouping_indices`.
  [(#5179)](https://github.com/PennyLaneAI/pennylane/pull/5179)

  ```python
  import pennylane as qml

  a = qml.X(0)
  b = qml.prod(qml.X(0), qml.X(1))
  c = qml.Z(0)
  obs = [a, b, c]
  coeffs = [1.0, 2.0, 3.0]

  op = qml.dot(coeffs, obs, grouping_type="qwc")
  ```
  ```pycon
  >>> op.grouping_indices
  ((2,), (0, 1))
  ```

  Additionally, grouping type and method can be set or changed after construction using
  `Sum.compute_grouping()`:

  ```python
  import pennylane as qml

  a = qml.X(0)
  b = qml.prod(qml.X(0), qml.X(1))
  c = qml.Z(0)
  obs = [a, b, c]
  coeffs = [1.0, 2.0, 3.0]

  op = qml.dot(coeffs, obs)
  ```
  ```pycon
  >>> op.grouping_indices is None
  True
  >>> op.compute_grouping(grouping_type="qwc")
  >>> op.grouping_indices
  ((2,), (0, 1))
  ```

  Note that the grouping indices refer to the lists returned by `Sum.terms()`, not `Sum.operands`.

* Added new `SpectralNormError` class to the new error tracking functionality.
  [(#5154)](https://github.com/PennyLaneAI/pennylane/pull/5154)

* The `dynamic_one_shot` transform is introduced enabling dynamic circuit execution on circuits with shots and devices that support `MidMeasureMP` operations natively.
  [(#5266)](https://github.com/PennyLaneAI/pennylane/pull/5266)

* Added new function `qml.operation.convert_to_legacy_H` to convert `Sum`, `SProd`, and `Prod` to `Hamiltonian` instances.
  [(#5309)](https://github.com/PennyLaneAI/pennylane/pull/5309)

* Create the `qml.Reflection` operator, useful for amplitude amplification and its variants.
  [(#5159)](https://github.com/PennyLaneAI/pennylane/pull/5159)

  ```python
  @qml.prod
  def generator(wires):
        qml.Hadamard(wires=wires)

  U = generator(wires=0)

  dev = qml.device('default.qubit')
  @qml.qnode(dev)
  def circuit():

        # Initialize to the state |1>
        qml.PauliX(wires=0)

        # Apply the reflection
        qml.Reflection(U)

        return qml.state()

  ```

  ```pycon
  >>> circuit()
  tensor([1.+6.123234e-17j, 0.-6.123234e-17j], requires_grad=True)
  ```
  
* The `qml.AmplitudeAmplification` operator is introduced, which is a high-level interface for amplitude amplification and its variants.
  [(#5160)](https://github.com/PennyLaneAI/pennylane/pull/5160)

  ```python
  @qml.prod
  def generator(wires):
      for wire in wires:
          qml.Hadamard(wires=wire)

  U = generator(wires=range(3))
  O = qml.FlipSign(2, wires=range(3))

  dev = qml.device("default.qubit")

  @qml.qnode(dev)
  def circuit():

      generator(wires=range(3))
      qml.AmplitudeAmplification(U, O, iters=5, fixed_point=True, work_wire=3)

      return qml.probs(wires=range(3))

  ```
  
  ```pycon
  >>> print(np.round(circuit(), 3))
  [0.013, 0.013, 0.91, 0.013, 0.013, 0.013, 0.013, 0.013]

  ```

<h3>Improvements 🛠</h3>

* The `qml.is_commuting` function now accepts `Sum`, `SProd`, and `Prod` instances.
  [(#5351)](https://github.com/PennyLaneAI/pennylane/pull/5351)

* Operators can now be left multiplied `x * op` by numpy arrays.
  [(#5361)](https://github.com/PennyLaneAI/pennylane/pull/5361)

* The `molecular_hamiltonian` function calls `PySCF` directly when `method='pyscf'` is selected.
  [(#5118)](https://github.com/PennyLaneAI/pennylane/pull/5118)

* All generators in the source code (except those in the `qchem` module) no longer return
  `Hamiltonian` or `Tensor` instances. Wherever possible, these return `Sum`, `SProd`, and `Prod` instances.
  [(#5253)](https://github.com/PennyLaneAI/pennylane/pull/5253)

* Upgraded `null.qubit` to the new device API. Also, added support for all measurements and various modes of differentiation.
  [(#5211)](https://github.com/PennyLaneAI/pennylane/pull/5211)
  
* `ApproxTimeEvolution` is now compatible with any operator that defines a `pauli_rep`.
  [(#5362)](https://github.com/PennyLaneAI/pennylane/pull/5362)

* `Hamiltonian.pauli_rep` is now defined if the hamiltonian is a linear combination of paulis.
  [(#5377)](https://github.com/PennyLaneAI/pennylane/pull/5377)

* Obtaining classical shadows using the `default.clifford` device is now compatible with
  [stim](https://github.com/quantumlib/Stim) `v1.13.0`.
  [(#5409)](https://github.com/PennyLaneAI/pennylane/pull/5409)

<h4>Community contributions 🥳</h4>

* Functions `measure_with_samples` and `sample_state` have been added to the new `qutrit_mixed` module found in
 `qml.devices`. These functions are used to sample device-compatible states, returning either the final measured state or value of an observable.
  [(#5082)](https://github.com/PennyLaneAI/pennylane/pull/5082)

* The `QNode` now defers `diff_method` validation to the device under the new device api `qml.devices.Device`.
  [(#5176)](https://github.com/PennyLaneAI/pennylane/pull/5176)

* `taper_operation` method is compatible with new operator arithmetic.
  [(#5326)](https://github.com/PennyLaneAI/pennylane/pull/5326)

* `qml.transforms.split_non_commuting` will now work with single-term operator arithmetic.
  [(#5314)](https://github.com/PennyLaneAI/pennylane/pull/5314)

* Implemented the method `process_counts` in `ExpectationMP`, `VarianceMP`, and `CountsMP`.
  [(#5256)](https://github.com/PennyLaneAI/pennylane/pull/5256)

<h3>Breaking changes 💔</h3>

* The private functions `_pauli_mult`, `_binary_matrix` and `_get_pauli_map` from the `pauli` module have been removed. The same functionality can be achieved using newer features in the ``pauli`` module.
  [(#5323)](https://github.com/PennyLaneAI/pennylane/pull/5323)

* `qml.matrix()` called on the following will raise an error if `wire_order` is not specified:
  * tapes with more than one wire.
  * quantum functions.
  * Operator class where `num_wires` does not equal to 1
  * QNodes if the device does not have wires specified.
  * PauliWords and PauliSentences with more than one wire.
  [(#5328)](https://github.com/PennyLaneAI/pennylane/pull/5328)
  [(#5359)](https://github.com/PennyLaneAI/pennylane/pull/5359)

* `qml.pauli.pauli_mult` and `qml.pauli.pauli_mult_with_phase` are now removed. Instead, you  should use `qml.simplify(qml.prod(pauli_1, pauli_2))` to get the reduced operator.
  [(#5324)](https://github.com/PennyLaneAI/pennylane/pull/5324)
  
  ```pycon
  >>> op = qml.simplify(qml.prod(qml.PauliX(0), qml.PauliZ(0)))
  >>> op
  -1j*(PauliY(wires=[0]))
  >>> [phase], [base] = op.terms()
  >>> phase, base
  (-1j, PauliY(wires=[0]))
  ```

* `MeasurementProcess.name` and `MeasurementProcess.data` have been removed. Use `MeasurementProcess.obs.name` and `MeasurementProcess.obs.data` instead.
  [(#5321)](https://github.com/PennyLaneAI/pennylane/pull/5321)

* `Operator.validate_subspace(subspace)` has been removed. Instead, you should use `qml.ops.qutrit.validate_subspace(subspace)`.
  [(#5311)](https://github.com/PennyLaneAI/pennylane/pull/5311)

* The contents of `qml.interfaces` is moved inside `qml.workflow`. The old import path no longer exists.
  [(#5329)](https://github.com/PennyLaneAI/pennylane/pull/5329)

* `single_tape_transform`, `batch_transform`, `qfunc_transform`, `op_transform`, `gradient_transform`
  and `hessian_transform` are removed. Instead, switch to using the new `qml.transform` function. Please refer to
  `the transform docs <https://docs.pennylane.ai/en/stable/code/qml_transforms.html#custom-transforms>`_
  to see how this can be done.
  [(#5339)](https://github.com/PennyLaneAI/pennylane/pull/5339)

* Attempting to multiply `PauliWord` and `PauliSentence` with `*` will raise an error. Instead, use `@` to conform with the PennyLane convention.
  [(#5341)](https://github.com/PennyLaneAI/pennylane/pull/5341)

<h3>Deprecations 👋</h3>

* `qml.load` is deprecated. Instead, please use the functions outlined in the *Importing workflows* quickstart guide, such as `qml.from_qiskit`.
  [(#5312)](https://github.com/PennyLaneAI/pennylane/pull/5312)

* Specifying `control_values` with a bit string to `qml.MultiControlledX` is deprecated. Instead, use a list of booleans or 1s and 0s.
  [(#5352)](https://github.com/PennyLaneAI/pennylane/pull/5352)

* `qml.from_qasm_file` is deprecated. Instead, please open the file and then load its content using `qml.from_qasm`.
  [(#5331)](https://github.com/PennyLaneAI/pennylane/pull/5331)

  ```pycon
  >>> with open("test.qasm", "r") as f:
  ...     circuit = qml.from_qasm(f.read())
  ```

<h3>Documentation 📝</h3>

* Removed some redundant documentation for the `evolve` function.
  [(#5347)](https://github.com/PennyLaneAI/pennylane/pull/5347)

* Updated the final example in the `compile` docstring to use transforms correctly.
  [(#5348)](https://github.com/PennyLaneAI/pennylane/pull/5348)

<h3>Bug fixes 🐛</h3>

* We no longer perform unwanted dtype promotion in the `pauli_rep` of `SProd` instances when using tensorflow.
  [(#5246)](https://github.com/PennyLaneAI/pennylane/pull/5246)

* Fixed `TestQubitIntegration.test_counts` in `tests/interfaces/test_jax_qnode.py` to always produce counts for all outcomes.
  [(#5336)](https://github.com/PennyLaneAI/pennylane/pull/5336)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Tarun Kumar Allamsetty,
Guillermo Alonso,
Mikhail Andrenkov,
Utkarsh Azad,
Gabriel Bottrill,
Astral Cai,
Amintor Dusko,
Pietropaolo Frisoni,
Soran Jahangiri,
Korbinian Kottmann,
Christina Lee,
Vincent Michaud-Rioux,
Mudit Pandey,
Matthew Silverman.