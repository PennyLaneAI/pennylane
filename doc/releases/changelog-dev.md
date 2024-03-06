:orphan:

# Release 0.36.0-dev (development release)

<h3>New features since last release</h3>

* `qml.ops.Sum` now supports storing grouping information. Grouping type and method can be
  specified during construction using the `grouping_type` and `method` keyword arguments of
  `qml.dot`, `qml.sum`, or `qml.ops.Sum`. The grouping indices are stored in `Sum.grouping_indices`.
  [(#5179)](https://github.com/PennyLaneAI/pennylane/pull/5179)

  ```python
  import pennylane as qml

  a = qml.PauliX(0)
  b = qml.PauliX(1)
  c = qml.PauliZ(0)
  obs = [a, b, c]
  coeffs = [1.0, 2.0, 3.0]

  op = qml.dot(coeffs, obs, grouping_type="qwc")
  ```
  ```pycon
  >>> op.grouping_indices
  ((0, 1), (2,))
  ```

  Additionally, grouping type and method can be set or changed after construction using
  `Sum.compute_grouping()`:

  ```python
  import pennylane as qml

  a = qml.PauliX(0)
  b = qml.PauliX(1)
  c = qml.PauliZ(0)
  obs = [a, b, c]
  coeffs = [1.0, 2.0, 3.0]

  op = qml.dot(coeffs, obs)
  ```
  ```pycon
  >>> op.grouping_indices is None
  True
  >>> op.compute_grouping(grouping_type="qwc")
  >>> op.grouping_indices
  ((0, 1), (2,))
  ```

  Note that the grouping indices refer to the lists returned by `Sum.terms()`, not `Sum.operands`.

* Added new `SpectralNormError` class to the new error tracking functionality.
  [(#5154)](https://github.com/PennyLaneAI/pennylane/pull/5154)

* The `dynamic_one_shot` transform is introduced enabling dynamic circuit execution on circuits with shots and devices that support `MidMeasureMP` operations natively.
  [(#5266)](https://github.com/PennyLaneAI/pennylane/pull/5266)

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

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* We no longer perform unwanted dtype promotion in the `pauli_rep` of `SProd` instances when using tensorflow.
  [(#5246)](https://github.com/PennyLaneAI/pennylane/pull/5246)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Korbinian Kottmann,
Guillermo Alonso,
Amintor Dusko
Pietropaolo Frisoni,
Soran Jahangiri,
Korbinian Kottmann,
Mudit Pandey,
Matthew Silverman.
