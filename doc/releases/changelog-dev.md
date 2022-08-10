:orphan:

# Release 0.26.0-dev (development release)

<h3>New features since last release</h3>

* Added `QutritDevice` as an abstract base class for qutrit devices.
  [#2781](https://github.com/PennyLaneAI/pennylane/pull/2781)
  [#2782](https://github.com/PennyLaneAI/pennylane/pull/2782)
  
* Added operation `qml.QutritUnitary` for applying user-specified unitary operations on qutrit devices.
  [(#2699)](https://github.com/PennyLaneAI/pennylane/pull/2699)
  
* Added `default.qutrit` plugin for pure state simulation of qutrits. Currently supports operation `qml.QutritUnitary` and measurements `qml.state()`, `qml.probs()`.
  [(#2783)](https://github.com/PennyLaneAI/pennylane/pull/2783)

  ```pycon
  >>> dev = qml.device("default.qutrit", wires=1)
  >>> @qml.qnode(dev)
  ... def circuit(U):
  ...     qml.QutritUnitary(U, wires=0)
  ...     return qml.probs(wires=0)
  >>> U = np.array([[1, 1, 0], [1, -1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)
  >>> print(circuit(U))
  [0.5 0.5 0. ]
  ```
  
* Added `qml.THermitian` observable for measuring user-specified Hermitian matrix observables for qutrit circuits.
  [#2784](https://github.com/PennyLaneAI/pennylane/pull/2784)

<h3>Improvements</h3>

* Automatic circuit cutting is improved by making better partition imbalance derivations.
  Now it is more likely to generate optimal cuts for larger circuits.
  [(#2517)](https://github.com/PennyLaneAI/pennylane/pull/2517)

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Documentation</h3>

<h3>Bug fixes</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Olivia Di Matteo,
Josh Izaac,
Korbinian Kottmann,
Zeyue Niu,
Mudit Pandey,
Antal Száva
