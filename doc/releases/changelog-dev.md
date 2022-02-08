:orphan:

# Release 0.22.0-dev (development release)

<h3>New features since last release</h3>

* The text based drawer accessed via `qml.draw` has been overhauled. The new drawer has 
  a `decimals` keyword for controlling parameter rounding, a different algorithm for determining positions, 
  deprecation of the `charset` keyword, and minor cosmetic changes.
  [(#2128)](https://github.com/PennyLaneAI/pennylane/pull/2128)

  ```
  @qml.qnode(qml.device('lightning.qubit', wires=2))
  def circuit(a, w):
      qml.Hadamard(0)
      qml.CRX(a, wires=[0, 1])
      qml.Rot(*w, wires=[1])
      qml.CRX(-a, wires=[0, 1])
      return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
  ```
  ```
    >>> print(qml.draw(circuit, decimals=2)(a=2.3, w=[1.2, 3.2, 0.7]))
    0: ──H─╭C─────────────────────────────╭C─────────┤ ╭<Z@Z>
    1: ────╰RX(2.30)──Rot(1.20,3.20,0.70)─╰RX(-2.30)─┤ ╰<Z@Z>
  ```

* Continued development of the circuit-cutting compiler:
  
  A method for converting a quantum tape to a directed multigraph that is amenable
  to graph partitioning algorithms for circuit cutting has been added.
  [(#2107)](https://github.com/PennyLaneAI/pennylane/pull/2107)
  
  A method to replace `WireCut` nodes in a directed multigraph with `MeasureNode` 
  and `PrepareNode` placeholders has been added.
  [(#2124)](https://github.com/PennyLaneAI/pennylane/pull/2124)
  
  A method has been added that takes a directed multigraph with `MeasureNode` and
  `PrepareNode` placeholders and fragments into subgraphs and a communication graph.
  [(#2153)](https://github.com/PennyLaneAI/pennylane/pull/2153)

<h3>Improvements</h3>

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Bug fixes</h3>

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Anthony Hayes, Christina Lee
