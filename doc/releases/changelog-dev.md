:orphan:

# Release 0.36.0-dev (development release)

<h3>New features since last release</h3>

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

<h3>Improvements 🛠</h3>

* Create the `qml.Reflection` operator, useful for amplitude amplification and its variants.
  [(##5159)](https://github.com/PennyLaneAI/pennylane/pull/5159)

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
  
* The `molecular_hamiltonian` function calls `PySCF` directly when `method='pyscf'` is selected.
  [(#5118)](https://github.com/PennyLaneAI/pennylane/pull/5118)
  
* All generators in the source code (except those in the `qchem` module) no longer return 
  `Hamiltonian` or `Tensor` instances. Wherever possible, these return `Sum`, `SProd`, and `Prod` instances.
  [(#5253)](https://github.com/PennyLaneAI/pennylane/pull/5253)

* Upgraded `null.qubit` to the new device API. Also, added support for all measurements and various modes of differentiation.
  [(#5211)](https://github.com/PennyLaneAI/pennylane/pull/5211)
  
<h4>Community contributions 🥳</h4>

* Functions `measure_with_samples` and `sample_state` have been added to the new `qutrit_mixed` module found in
 `qml.devices`. These functions are used to sample device-compatible states, returning either the final measured state or value of an observable.
  [(#5082)](https://github.com/PennyLaneAI/pennylane/pull/5082)

* The `QNode` now defers `diff_method` validation to the device under the new device api `qml.devices.Device`.
  [(#5176)](https://github.com/PennyLaneAI/pennylane/pull/5176)

* `qml.transforms.split_non_commuting` will now work with single-term operator arithmetic.
  [(#5314)](https://github.com/PennyLaneAI/pennylane/pull/5314)

<h3>Breaking changes 💔</h3>

* `qml.matrix()` called on the following will raise an error if `wire_order` is not specified:
  * tapes with more than one wire.
  * quantum functions.
  * QNodes if the device does not have wires specified.
  * PauliWords and PauliSentences with more than one wire.
  [(#5328)](https://github.com/PennyLaneAI/pennylane/pull/5328)

* ``qml.pauli.pauli_mult`` and ``qml.pauli.pauli_mult_with_phase`` are now removed. Instead, you  should use ``qml.simplify(qml.prod(pauli_1, pauli_2))`` to get the reduced operator.
  [(#5324)](https://github.com/PennyLaneAI/pennylane/pull/5324)
  
  ```pycon
  >>> op = qml.simplify(qml.prod(qml.PauliX(0), qml.PauliZ(0)))
  >>> op
  -1j*(PauliY(wires=[0]))
  >>> [phase], [base] = op.terms()
  >>> phase, base
  (-1j, PauliY(wires=[0]))
  ```

* ``MeasurementProcess.name`` and ``MeasurementProcess.data`` have been removed. Use ``MeasurementProcess.obs.name`` and ``MeasurementProcess.obs.data`` instead.
  [(#5321)](https://github.com/PennyLaneAI/pennylane/pull/5321)

* `Operator.validate_subspace(subspace)` has been removed. Instead, you should use `qml.ops.qutrit.validate_subspace(subspace)`.
  [(#5311)](https://github.com/PennyLaneAI/pennylane/pull/5311)

* The contents of ``qml.interfaces`` is moved inside ``qml.workflow``. The old import path no longer exists.
  [(#5329)](https://github.com/PennyLaneAI/pennylane/pull/5329)

<h3>Deprecations 👋</h3>

* ``qml.load`` is deprecated. Instead, please use the functions outlined in the *Importing workflows* quickstart guide, such as ``qml.from_qiskit``.
  [(#5312)](https://github.com/PennyLaneAI/pennylane/pull/5312)

* ``qml.from_qasm_file`` is deprecated. Instead, please open the file and then load its content using ``qml.from_qasm``.
  [(#5331)](https://github.com/PennyLaneAI/pennylane/pull/5331)

  ```pycon
  >>> with open("test.qasm", "r") as f:
  ...     circuit = qml.from_qasm(f.read())
  ```

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* We no longer perform unwanted dtype promotion in the `pauli_rep` of `SProd` instances when using tensorflow.
  [(#5246)](https://github.com/PennyLaneAI/pennylane/pull/5246)

* Fixed `TestQubitIntegration.test_counts` in `tests/interfaces/test_jax_qnode.py` to always produce counts for all outcomes.
  [(#5336)](https://github.com/PennyLaneAI/pennylane/pull/5336)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Korbinian Kottmann,
Guillermo Alonso,
Gabriel Bottrill,
Astral Cai,
Amintor Dusko,
Pietropaolo Frisoni,
Soran Jahangiri,
Korbinian Kottmann,
Christina Lee,
Mudit Pandey,
Matthew Silverman.
