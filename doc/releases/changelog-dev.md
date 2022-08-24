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
  ([#2784](https://github.com/PennyLaneAI/pennylane/pull/2784))

* Added the `qml.TShift` and `qml.TClock` qutrit operations for qutrit devices, which are the qutrit analogs of the Pauli X and Pauli Z operations.
  ([#2841](https://github.com/PennyLaneAI/pennylane/pull/2841))

* Added the `qml.TAdd` and `qml.TSWAP` qutrit operations which are the ternary analogs of the CNOT and SWAP operations respectively.
  ([#2843](https://github.com/PennyLaneAI/pennylane/pull/2843))

**Classical shadows**

* Added the `qml.classical_shadow` measurement process that can now be returned from QNodes.

  The measurement protocol is described in detail in the
  [classical shadows paper](https://arxiv.org/abs/2002.08953). Calling the QNode
  will return the randomized Pauli measurements (the `recipes`) that are performed
  for each qubit, identified as a unique integer:

  * 0 for Pauli X
  * 1 for Pauli Y
  * 2 for Pauli Z

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

* `qml.ops.op_math.Controlled` now has basic decomposition functionality.
  [(#2938)](https://github.com/PennyLaneAI/pennylane/pull/2938)

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
  
* Added `sparse_matrix()` support for single qubit observables
  [(#2964)](https://github.com/PennyLaneAI/pennylane/pull/2964)

* Added the `qml.is_hermitian` and `qml.is_unitary` function checks.
  [(#2960)](https://github.com/PennyLaneAI/pennylane/pull/2960)

  ```pycon
  >>> op = qml.PauliX(wires=0)
  >>> qml.is_hermitian(op)
  True
  >>> op2 = qml.RX(0.54, wires=0)
  >>> qml.is_hermitian(op2)
  False
  ```

* Internal use of in-place inversion is eliminated in preparation for its deprecation.
  [(#2965)](https://github.com/PennyLaneAI/pennylane/pull/2965)

* `qml.is_commuting` is moved to `pennylane/ops/functions` from `pennylane/transforms/commutation_dag.py`.
  [(#2991)](https://github.com/PennyLaneAI/pennylane/pull/2991)

* `qml.simplify` can now be used to simplify quantum functions, tapes and QNode objects.
  [(#2978)](https://github.com/PennyLaneAI/pennylane/pull/2978)

  ```python
    dev = qml.device("default.qubit", wires=2)
    @qml.simplify
    @qml.qnode(dev)
    def circuit():
      qml.adjoint(qml.prod(qml.RX(1, 0) ** 1, qml.RY(1, 0), qml.RZ(1, 0)))
      return qml.probs(wires=0)
  ```

  ```pycon
  >>> circuit()
  >>> list(circuit.tape)
  [RZ(-1, wires=[0]) @ RY(-1, wires=[0]) @ RX(-1, wires=[0]), probs(wires=[0])]
  ```

<h3>Breaking changes</h3>

* Measuring an operator that might not be hermitian as an observable now raises a warning instead of an
  error. To definitively determine whether or not an operator is hermitian, use `qml.is_hermitian`.
  [(#2960)](https://github.com/PennyLaneAI/pennylane/pull/2960)

<h3>Deprecations</h3>

<h3>Documentation</h3>

* Corrects the docstrings for diagonalizing gates for all relevant operations. The docstrings used to say that the diagonalizing gates implemented $U$, the unitary such that $O = U \Sigma U^{\dagger}$, where $O$ is the original observable and $\Sigma$ a diagonal matrix. However, the diagonalizing gates actually implement $U^{\dagger}$, since $\langle \psi | O | \psi \rangle = \langle \psi | U \Sigma U^{\dagger} | \psi \rangle$, making $U^{\dagger} | \psi \rangle$ the actual state being measured in the $Z$-basis. [(#2981)](https://github.com/PennyLaneAI/pennylane/pull/2981)

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
Christina Lee,
Meenu Kumari,
Albert Mitjans Coma,
Rashid N H M,
Zeyue Niu,
Mudit Pandey,
Jay Soni,
Antal Száva
Cody Wang,
David Wierichs
