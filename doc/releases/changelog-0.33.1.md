:orphan:

# Release 0.33.1 (current release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* `qml.defer_measurements` now correctly transforms circuits when terminal measurements include wires
  used in mid-circuit measurements.
  [(#4787)](https://github.com/PennyLaneAI/pennylane/pull/4787)

* Any `ScalarSymbolicOp`, like `Evolution`, now states that it has a matrix if the target
  is a `Hamiltonian`.
  [(#4768)](https://github.com/PennyLaneAI/pennylane/pull/4768)

* In `default.qubit`, initial states are now initialized with the simulator's wire order, not the circuit's
  wire order.
  [(#4781)](https://github.com/PennyLaneAI/pennylane/pull/4781)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Christina Lee,
Mudit Pandey
