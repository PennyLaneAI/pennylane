
# Release 0.43.1

<h3>Bug fixes 🐛</h3>

* Fixed a bug in the output of :func:`~pennylane.to_openqasm`, where the `creg` declaration for mid-circuit measurements was missing 
  a semicolon and leading to invalid QASM.
  [(#8556)](https://github.com/PennyLaneAI/pennylane/pull/8556)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Yushao Chen,
Marcus Edwards,
Nate Stemen.