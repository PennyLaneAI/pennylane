
# Release 0.14.1

<h3>Bug fixes</h3>

* Fixes a testing bug where tests that required JAX would fail if JAX was not installed.
  The tests will now instead be skipped if JAX can not be imported.
  [(#1066)](https://github.com/PennyLaneAI/pennylane/pull/1066)

* Fixes a bug where inverse operations could not be differentiated
  using backpropagation on `default.qubit`.
  [(#1072)](https://github.com/PennyLaneAI/pennylane/pull/1072)

* The QNode has a new keyword argument, `max_expansion`, that determines the maximum number of times
  the internal circuit should be expanded when executed on a device. In addition, the default number
  of max expansions has been increased from 2 to 10, allowing devices that require more than two
  operator decompositions to be supported.
  [(#1074)](https://github.com/PennyLaneAI/pennylane/pull/1074)

* Fixes a bug where `Hamiltonian` objects created with non-list arguments raised an error for
  arithmetic operations. [(#1082)](https://github.com/PennyLaneAI/pennylane/pull/1082)

* Fixes a bug where `Hamiltonian` objects with no coefficients or operations would return a faulty
  result when used with `ExpvalCost`. [(#1082)](https://github.com/PennyLaneAI/pennylane/pull/1082)

<h3>Documentation</h3>

* Updates mentions of `generate_hamiltonian` to `molecular_hamiltonian` in the
  docstrings of the `ExpvalCost` and `Hamiltonian` classes.
  [(#1077)](https://github.com/PennyLaneAI/pennylane/pull/1077)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Thomas Bromley, Josh Izaac, Antal Száva.


