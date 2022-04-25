:orphan:

# Release 0.24.0-dev (development release)

<h3>New features since last release</h3>

* Speed up measuring of commuting Pauli operators
  [(#2425)](https://github.com/PennyLaneAI/pennylane/pull/2425)

  The code that checks for qubit wise commuting (QWC) got a performance boost that is noticable
  when many commuting paulis of the same type are measured.

<h3>Improvements</h3>

* The `gradients` module now uses faster subroutines and uniform
  formats of gradient rules.
  [(#2452)](https://github.com/XanaduAI/pennylane/pull/2452)

* Wires can be passed as the final argument to an `Operator`, instead of requiring
  the wires to be explicitly specified with keyword `wires`. This functionality already
  existed for `Observable`'s, but now extends to all `Operator`'s.
  [(#2432)](https://github.com/PennyLaneAI/pennylane/pull/2432)

  ```pycon
  >>> qml.S(0)
  S(wires=[0])
  >>> qml.CNOT((0,1))
  CNOT(wires=[0, 1])
  ```

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Christian Gogolin, Christina Lee
