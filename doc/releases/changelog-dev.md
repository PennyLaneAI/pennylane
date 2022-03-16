:orphan:

# Release 0.23.0-dev (development release)

<h3>New features since last release</h3>

* Added a swap based transpiler transform.
  [(#2118)](https://github.com/PennyLaneAI/pennylane/pull/2118)

  The transpile function takes a quantum function and a coupling map as inputs and compiles the tape to ensure that it can be 
  executed on corresponding hardware. The transform can be used as a decorator in the following way:

  ```python
  dev = qml.device('default.qubit', wires=4)
  
  @qml.qnode(dev)
  @qml.transforms.transpile(coupling_map=[(0, 1), (1, 2), (2, 3)])
  def circuit(param):
      qml.CNOT(wires=[0, 1])
      qml.CNOT(wires=[0, 2])
      qml.CNOT(wires=[0, 3])
      qml.PhaseShift(param, wires=0)
      return qml.probs(wires=[0, 1, 2, 3]) 
  ```

* Development of a circuit-cutting compiler extension to circuits with sampling
  measurements has begun:

  - The existing `qcut.tape_to_graph()` method has been extended to convert a
    sample measurement without an observable specified to multiple single-qubit sample
    nodes.
    [(#2313)](https://github.com/PennyLaneAI/pennylane/pull/2313)

<h3>Improvements</h3>

* The function `qml.ctrl` was given the optional argument `control_values=None`.
  If overridden, `control_values` takes an integer or a list of integers corresponding to
  the binary value that each control value should take. The same change is reflected in
  `ControlledOperation`. Control values of `0` are implemented by `qml.PauliX` applied
  before and after the controlled operation
  [(#2288)](https://github.com/PennyLaneAI/pennylane/pull/2288)
  
<h3>Deprecations</h3>

<h3>Breaking changes</h3>

* The deprecated QNode, available via `qml.qnode_old.QNode`, has been removed. Please
  transition to using the standard `qml.QNode`.
  [(#2336)](https://github.com/PennyLaneAI/pennylane/pull/2336)

* The deprecated, non-batch compatible interfaces, have been removed.
  [(#2336)](https://github.com/PennyLaneAI/pennylane/pull/2336)

* The deprecated tape subclasses `QubitParamShiftTape`, `JacobianTape`, `CVParamShiftTape`, and
  `ReversibleTape` have been removed.
  [(#2336)](https://github.com/PennyLaneAI/pennylane/pull/2336)

<h3>Bug fixes</h3>

* Fixes cases with `qml.measure` where unexpected operations were added to the
  circuit.
  [(#2328)](https://github.com/PennyLaneAI/pennylane/pull/2328)

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Karim Alaa El-Din, Guillermo Alonso-Linaje, Anthony Hayes, Josh Izaac, Maurice Weber
