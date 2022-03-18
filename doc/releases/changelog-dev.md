:orphan:

# Release 0.23.0-dev (development release)

<h3>New features since last release</h3>

* Development of a circuit-cutting compiler extension to circuits with sampling
  measurements has begun:

  - The existing `qcut.tape_to_graph()` method has been extended to convert a
    sample measurement without an observable specified to multiple single-qubit sample
    nodes.
    [(#2313)](https://github.com/PennyLaneAI/pennylane/pull/2313)

  - The existing `qcut.graph_to_tape()` method has been extended to convert
    graphs containing sample measurement nodes to tapes.
    [(#2321)](https://github.com/PennyLaneAI/pennylane/pull/2321)

<h3>Improvements</h3>

* `QuantumTape` objects are now iterable and can be used in `for` loops or
  converted to sequences such as a `list` to get the operations and
  measurements of the underlying quantum circuit.
  [(#2342)](https://github.com/PennyLaneAI/pennylane/pull/2342)

  ```python
  with qml.tape.QuantumTape() as tape:
      qml.RX(0.432, wires=0)
      qml.RY(0.543, wires=0)
      qml.CNOT(wires=[0, 'a'])
      qml.RX(0.133, wires='a')
      qml.expval(qml.PauliZ(wires=[0]))
  ```
  ```pycon
  >>> for op in tape:
  ...     print(op)
  ```
  RX(0.432, wires=[0])
  RY(0.543, wires=[0])
  CNOT(wires=[0, 'a'])
  RX(0.133, wires=['a'])
  expval(PauliZ(wires=[0]))
  ```pycon
  >>> list(tape)
  ```
  [RX(0.432, wires=[0]),
   RY(0.543, wires=[0]),
   CNOT(wires=[0, 'a']),
   RX(0.133, wires=['a']),
   expval(PauliZ(wires=[0]))]

* The function `qml.ctrl` was given the optional argument `control_values=None`.
  If overridden, `control_values` takes an integer or a list of integers corresponding to
  the binary value that each control value should take. The same change is reflected in
  `ControlledOperation`. Control values of `0` are implemented by `qml.PauliX` applied
  before and after the controlled operation
  [(#2288)](https://github.com/PennyLaneAI/pennylane/pull/2288)

* Circuit cutting now performs expansion to search for wire cuts in contained operations or tapes.
  [(#2340)](https://github.com/PennyLaneAI/pennylane/pull/2340)

<h3>Deprecations</h3>

<h3>Breaking changes</h3>

* The `ObservableReturnTypes` `Sample`, `Variance`, `Expectation`, `Probability`, `State`, and `MidMeasure`
  have been moved to `measurements` from `operation`.
  [(#2329)](https://github.com/PennyLaneAI/pennylane/pull/2329)

* The deprecated QNode, available via `qml.qnode_old.QNode`, has been removed. Please
  transition to using the standard `qml.QNode`.
  [(#2336)](https://github.com/PennyLaneAI/pennylane/pull/2336)

* The deprecated, non-batch compatible interfaces, have been removed.
  [(#2336)](https://github.com/PennyLaneAI/pennylane/pull/2336)

* The deprecated tape subclasses `QubitParamShiftTape`, `JacobianTape`, `CVParamShiftTape`, and
  `ReversibleTape` have been removed.
  [(#2336)](https://github.com/PennyLaneAI/pennylane/pull/2336)

<h3>Bug fixes</h3>

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Karim Alaa El-Din, Thomas Bromley, Anthony Hayes, Josh Izaac, Christina Lee,
Romain Moyard, Antal Száva.
