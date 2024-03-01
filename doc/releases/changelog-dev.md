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

<h3>Improvements 🛠</h3>

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* We no longer perform unwanted dtype promotion in the `pauli_rep` of `SProd` instances when using tensorflow.
  [(#5246)](https://github.com/PennyLaneAI/pennylane/pull/5246)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Korbinian Kottmann,
Mudit Pandey,