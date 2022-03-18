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

* The old circuit text drawing infrastructure is being deleted.
  [(#2310)](https://github.com/PennyLaneAI/pennylane/pull/2310)

  - `qml.drawer.CircuitDrawer` is replaced by `qml.drawer.tape_text`.
  - `qml.drawer.CHARSETS` is deleted because we now assume everyone has access to unicode.
  - `Grid` and `qml.drawer.drawable_grid` are removed because the custom data class is replaced
      by list of sets of operators or measurements.
  - `RepresentationResolver` is replaced by the `Operator.label` method.
  - `qml.transforms.draw_old` is replaced by `qml.draw`.
  - `qml.CircuitGraph.greedy_layers` is deleted, as it is no longer needed by the circuit drawer and
      does not seem to have uses outside of that situation.
  - `qml.CircuitGraph.draw` has been deleted, as we draw tapes instead.

The tape method `qml.tape.QuantumTape.draw` now simply calls `qml.drawer.tape_text`. 
In the new pathway, the `charset` keyword is deleted, the `max_length` keyword defaults to `100`, and
the `decimals` and `show_matrices` keywords are added. `qml.drawer.tape_text(tape)`

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

<h3>Bug fixes</h3>

* Fixes a bug in which the `expval`/`var` of a `Tensor(Observable)` would depend on the order 
  in which the observable is defined: 
  ```python
  @qml.qnode(dev)
  def circ(op):
    qml.RX(0.12, wires=0)
    qml.RX(1.34, wires=1)
    qml.RX(3.67, wires=2)
    
    return qml.expval(op)
  
  op1 = qml.Identity(wires=0) @ qml.Identity(wires=1) @ qml.PauliZ(wires=2)
  op2 = qml.PauliZ(wires=2) @ qml.Identity(wires=0) @ qml.Identity(wires=1)
  ```

  ```
  >>> print(circ(op1), circ(op2))
  -0.8636111153905662 -0.8636111153905662
  ```
  [(#2276)](https://github.com/PennyLaneAI/pennylane/pull/2276)

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Karim Alaa El-Din, Thomas Bromley, Anthony Hayes, Josh Izaac, Christina Lee, Jay Soni.
