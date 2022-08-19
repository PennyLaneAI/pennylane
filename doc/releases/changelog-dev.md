:orphan:

# Release 0.26.0-dev (development release)

<h3>New features since last release</h3>

* Embedding templates now support parameter broadcasting.
  [(#2810)](https://github.com/PennyLaneAI/pennylane/pull/2810)
  
  Embedding templates like `AmplitudeEmbedding` or `IQPEmbedding` now support
  parameter broadcasting with a leading broadcasting dimension in their variational
  parameters. `AmplitudeEmbedding`, for example, would usually use a one-dimensional input
  vector of features. With broadcasting, we now also can compute

  ```pycon
  >>> features = np.array([
  ...     [0.5, 0.5, 0., 0., 0.5, 0., 0.5, 0.],
  ...     [1., 0., 0., 0., 0., 0., 0., 0.],
  ...     [0.5, 0.5, 0., 0., 0., 0., 0.5, 0.5],
  ... ])
  >>> op = qml.AmplitudeEmbedding(features, wires=[1, 5, 2])
  >>> op.batch_size
  3
  ```

  An exception is `BasisEmbedding`, which is not broadcastable.
  
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

**Classical shadows**

* Added the `qml.classical_shadow` measurement process that can now be returned from QNodes.

  The measurement protocol is described in detail in the
  [classical shadows paper](https://arxiv.org/abs/2002.08953). Calling the QNode
  will return the randomized Pauli measurements (the `recipes`) that are performed
  for each qubit, identified as a unique integer:

  - 0 for Pauli X
  - 1 for Pauli Y
  - 2 for Pauli Z

  It also returns the measurement results (the `bits`), which is `0` if the 1 eigenvalue
  is sampled, and `1` if the -1 eigenvalue is sampled.

  For example,

  ```python
  dev = qml.device("default.qubit", wires=2, shots=5)

  @qml.qnode(dev)
  def circuit():
      qml.Hadamard(wires=0)
      qml.CNOT(wires=[0, 1])
      return qml.classical_shadow(wires=[0, 1])
  ```
  ```pycon
  >>> bits, recipes = circuit()
  tensor([[0, 0],
          [1, 0],
          [1, 0],
          [0, 0],
          [0, 1]], dtype=uint8, requires_grad=True)
  >>> recipes
  tensor([[2, 2],
          [0, 2],
          [1, 0],
          [0, 2],
          [0, 2]], dtype=uint8, requires_grad=True)
  ```

<h3>Improvements</h3>

* Automatic circuit cutting is improved by making better partition imbalance derivations.
  Now it is more likely to generate optimal cuts for larger circuits.
  [(#2517)](https://github.com/PennyLaneAI/pennylane/pull/2517)

* Added `PSWAP` operator.
  [(#2667)](https://github.com/PennyLaneAI/pennylane/pull/2667)

* The `qml.simplify` method now can compute the adjoint and power of specific operators.
  [(#2922)](https://github.com/PennyLaneAI/pennylane/pull/2922)

  ```pycon
  >>> adj_op = qml.adjoint(qml.RX(1, 0))
  >>> qml.simplify(adj_op)
  RX(-1, wires=[0])
  ```

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Documentation</h3>

<h3>Bug fixes</h3>

* Operators that have `num_wires = AnyWires` or `num_wires = AnyWires` raise an error, with
  certain exceptions, when instantiated with `wires=[]`.
  [(#2979)](https://github.com/PennyLaneAI/pennylane/pull/2979)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Olivia Di Matteo,
Josh Izaac,
Edward Jiang,
Ankit Khandelwal,
Korbinian Kottmann,
Meenu Kumari,
Rashid N H M,
Zeyue Niu,
Mudit Pandey,
Antal Száva
David Wierichs
