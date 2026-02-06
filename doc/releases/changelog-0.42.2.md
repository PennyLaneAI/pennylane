# Release 0.42.2

<h3>Bug fixes 🐛</h3>

* Fixed a recursion error when simplifying operators that are raised to integer powers. For example,

  ```pycon
  >>> class DummyOp(qp.operation.Operator):
  ...     pass
  >>> (DummyOp(0) ** 2).simplify()
  DummyOp(0) @ DummyOp(0)
  ```

  Previously, this would fail with a recursion error.
  [(#8061)](https://github.com/PennyLaneAI/pennylane/pull/8061)
  [(#8064)](https://github.com/PennyLaneAI/pennylane/pull/8064)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Christina Lee,
Andrija Paurevic.