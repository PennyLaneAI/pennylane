:orphan:

# Release 0.36.0-dev (development release)

<h3>New features since last release</h3>

<h4>Dynamical Lie Algebra functionality</h4>

* A new `qml.dla.lie_closure` function to compute the Lie closure of a list of operators.
  [(#5161)](https://github.com/PennyLaneAI/pennylane/pull/5161)
  [(#5169)](https://github.com/PennyLaneAI/pennylane/pull/5169)

  The Lie closure, pronounced "Lee closure", is a way to compute the so-called dynamical Lie algebra (DLA) of a set of operators.
  For a list of operators `ops = [op1, op2, op3, ..]`, one computes all nested commutators between `ops` until no new operators are generated from commutation.
  All these operators together form the DLA, see e.g. section IIB of [arXiv:2308.01432](https://arxiv.org/abs/2308.01432).

  Take for example the following ops

  ```python
  ops = [X(0) @ X(1), Z(0), Z(1)]
  ```

  A first round of commutators between all elements yields the new operators `Y(0) @ X(1)` and `X(0) @ Y(1)`.

  ```python
  >>> qml.commutator(X(0) @ X(1), Z(0))
  2j * (Y(0) @ X(1))
  >>> qml.commutator(X(0) @ X(1), Z(0))
  2j * (X(0) @ Y(1))
  ```

  A next round of commutators between all elements further yields the new operator `Y(0) @ Y(1)`.

  ```python
  >>> qml.commutator(X(0) @ Y(1), Z(0))
  -2j * (Y(0) @ Y(1))
  ```

  After that, no new operators emerge from taking nested commutators and we have the resulting DLA.
  This can now be done in short via `qml.dla.lie_closure` as follows.

  ```python
  >>> ops = [X(0) @ X(1), Z(0), Z(1)]
  >>> dla = qml.dla.lie_closure(ops)
  >>> print(dla)
  [1.0 * X(1) @ X(0),
   1.0 * Z(0),
   1.0 * Z(1),
   -1.0 * X(1) @ Y(0),
   -1.0 * Y(1) @ X(0),
   -1.0 * Y(1) @ Y(0)]
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

Korbinian Kottmann