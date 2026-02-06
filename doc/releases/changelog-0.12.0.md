
# Release 0.12.0

<h3>New features since last release</h3>

<h4>New and improved simulators</h4>

* PennyLane now supports a new device, `default.mixed`, designed for
  simulating mixed-state quantum computations. This enables native
  support for implementing noisy channels in a circuit, which generally
  map pure states to mixed states.
  [(#794)](https://github.com/PennyLaneAI/pennylane/pull/794)
  [(#807)](https://github.com/PennyLaneAI/pennylane/pull/807)
  [(#819)](https://github.com/PennyLaneAI/pennylane/pull/819)

  The device can be initialized as
  ```pycon
  >>> dev = qml.device("default.mixed", wires=1)
  ```

  This allows the construction of QNodes that include non-unitary operations,
  such as noisy channels:

  ```pycon
  >>> @qml.qnode(dev)
  ... def circuit(params):
  ...     qml.RX(params[0], wires=0)
  ...     qml.RY(params[1], wires=0)
  ...     qml.AmplitudeDamping(0.5, wires=0)
  ...     return qml.expval(qml.PauliZ(0))
  >>> print(circuit([0.54, 0.12]))
  0.9257702929524184
  >>> print(circuit([0, np.pi]))
  0.0
  ```

<h4>New tools for optimizing measurements</h4>

* The new `grouping` module provides functionality for grouping simultaneously measurable Pauli word
  observables.
  [(#761)](https://github.com/PennyLaneAI/pennylane/pull/761)
  [(#850)](https://github.com/PennyLaneAI/pennylane/pull/850)
  [(#852)](https://github.com/PennyLaneAI/pennylane/pull/852)

  - The `optimize_measurements` function will take as input a list of Pauli word observables and
    their corresponding coefficients (if any), and will return the partitioned Pauli terms
    diagonalized in the measurement basis and the corresponding diagonalizing circuits.

    ```python
    from pennylane.grouping import optimize_measurements
    h, nr_qubits = qml.qchem.molecular_hamiltonian("h2", "h2.xyz")
    rotations, grouped_ops, grouped_coeffs = optimize_measurements(h.ops, h.coeffs, grouping="qwc")
    ```

    The diagonalizing circuits of `rotations` correspond to the diagonalized Pauli word groupings of
    `grouped_ops`.

  - Pauli word partitioning utilities are performed by the `PauliGroupingStrategy`
    class. An input list of Pauli words can be partitioned into mutually commuting,
    qubit-wise-commuting, or anticommuting groupings.

    For example, partitioning Pauli words into anticommutative groupings by the Recursive Largest
    First (RLF) graph colouring heuristic:

    ```python
    from pennylane import PauliX, PauliY, PauliZ, Identity
    from pennylane.grouping import group_observables
    pauli_words = [
        Identity('a') @ Identity('b'),
        Identity('a') @ PauliX('b'),
        Identity('a') @ PauliY('b'),
        PauliZ('a') @ PauliX('b'),
        PauliZ('a') @ PauliY('b'),
        PauliZ('a') @ PauliZ('b')
    ]
    groupings = group_observables(pauli_words, grouping_type='anticommuting', method='rlf')
    ```

  - Various utility functions are included for obtaining and manipulating Pauli
    words in the binary symplectic vector space representation.

    For instance, two Pauli words may be converted to their binary vector representation:

    ```pycon
    >>> from pennylane.grouping import pauli_to_binary
    >>> from pennylane.wires import Wires
    >>> wire_map = {Wires('a'): 0, Wires('b'): 1}
    >>> pauli_vec_1 = pauli_to_binary(qml.PauliX('a') @ qml.PauliY('b'))
    >>> pauli_vec_2 = pauli_to_binary(qml.PauliZ('a') @ qml.PauliZ('b'))
    >>> pauli_vec_1
    [1. 1. 0. 1.]
    >>> pauli_vec_2
    [0. 0. 1. 1.]
    ```

    Their product up to a phase may be computed by taking the sum of their binary vector
    representations, and returned in the operator representation.

    ```pycon
    >>> from pennylane.grouping import binary_to_pauli
    >>> binary_to_pauli((pauli_vec_1 + pauli_vec_2) % 2, wire_map)
    Tensor product ['PauliY', 'PauliX']: 0 params, wires ['a', 'b']
    ```

    For more details on the grouping module, see the
    [grouping module documentation](https://pennylane.readthedocs.io/en/stable/code/qml_grouping.html)


<h4>Returning the quantum state from simulators</h4>

* The quantum state of a QNode can now be returned using the `qml.state()` return function.
  [(#818)](https://github.com/XanaduAI/pennylane/pull/818)

  ```python
  import pennylane as qp

  dev = qml.device("default.qubit", wires=3)
  qml.enable_tape()

  @qml.qnode(dev)
  def qfunc(x, y):
      qml.RZ(x, wires=0)
      qml.CNOT(wires=[0, 1])
      qml.RY(y, wires=1)
      qml.CNOT(wires=[0, 2])
      return qml.state()

  >>> qfunc(0.56, 0.1)
  array([0.95985437-0.27601028j, 0.        +0.j        ,
         0.04803275-0.01381203j, 0.        +0.j        ,
         0.        +0.j        , 0.        +0.j        ,
         0.        +0.j        , 0.        +0.j        ])
  ```

  Differentiating the state is currently available when using the
  classical backpropagation differentiation method (`diff_method="backprop"`) with a compatible device,
  and when using the new tape mode.

<h4>New operations and channels</h4>

* PennyLane now includes standard channels such as the Amplitude-damping,
  Phase-damping, and Depolarizing channels, as well as the ability
  to make custom qubit channels.
  [(#760)](https://github.com/PennyLaneAI/pennylane/pull/760)
  [(#766)](https://github.com/PennyLaneAI/pennylane/pull/766)
  [(#778)](https://github.com/PennyLaneAI/pennylane/pull/778)

* The controlled-Y operation is now available via `qml.CY`. For devices that do
  not natively support the controlled-Y operation, it will be decomposed
  into `qml.RY`, `qml.CNOT`, and `qml.S` operations.
  [(#806)](https://github.com/PennyLaneAI/pennylane/pull/806)

<h4>Preview the next-generation PennyLane QNode</h4>

* The new PennyLane `tape` module provides a re-formulated QNode class, rewritten from the ground-up,
  that uses a new `QuantumTape` object to represent the QNode's quantum circuit. Tape mode
  provides several advantages over the standard PennyLane QNode.
  [(#785)](https://github.com/PennyLaneAI/pennylane/pull/785)
  [(#792)](https://github.com/PennyLaneAI/pennylane/pull/792)
  [(#796)](https://github.com/PennyLaneAI/pennylane/pull/796)
  [(#800)](https://github.com/PennyLaneAI/pennylane/pull/800)
  [(#803)](https://github.com/PennyLaneAI/pennylane/pull/803)
  [(#804)](https://github.com/PennyLaneAI/pennylane/pull/804)
  [(#805)](https://github.com/PennyLaneAI/pennylane/pull/805)
  [(#808)](https://github.com/PennyLaneAI/pennylane/pull/808)
  [(#810)](https://github.com/PennyLaneAI/pennylane/pull/810)
  [(#811)](https://github.com/PennyLaneAI/pennylane/pull/811)
  [(#815)](https://github.com/PennyLaneAI/pennylane/pull/815)
  [(#820)](https://github.com/PennyLaneAI/pennylane/pull/820)
  [(#823)](https://github.com/PennyLaneAI/pennylane/pull/823)
  [(#824)](https://github.com/PennyLaneAI/pennylane/pull/824)
  [(#829)](https://github.com/PennyLaneAI/pennylane/pull/829)

  - Support for in-QNode classical processing: Tape mode allows for differentiable classical
    processing within the QNode.

  - No more Variable wrapping: In tape mode, QNode arguments no longer become `Variable`
    objects within the QNode.

  - Less restrictive QNode signatures: There is no longer any restriction on the QNode signature;
    the QNode can be defined and called following the same rules as standard Python functions.

  - Unifying all QNodes: The tape-mode QNode merges all QNodes (including the
    `JacobianQNode` and the `PassthruQNode`) into a single unified QNode, with
    identical behaviour regardless of the differentiation type.

  - Optimizations: Tape mode provides various performance optimizations, reducing pre- and
    post-processing overhead, and reduces the number of quantum evaluations in certain cases.

  Note that tape mode is **experimental**, and does not currently have feature-parity with the
  existing QNode. [Feedback and bug reports](https://github.com/PennyLaneAI/pennylane/issues) are
  encouraged and will help improve the new tape mode.

  Tape mode can be enabled globally via the `qml.enable_tape` function, without changing your
  PennyLane code:

  ```python
  qml.enable_tape()
  dev = qml.device("default.qubit", wires=1)

  @qml.qnode(dev, interface="tf")
  def circuit(p):
      print("Parameter value:", p)
      qml.RX(tf.sin(p[0])**2 + p[1], wires=0)
      return qml.expval(qml.PauliZ(0))
  ```

  For more details, please see the [tape mode
  documentation](https://pennylane.readthedocs.io/en/stable/code/qml_tape.html).

<h3>Improvements</h3>

* QNode caching has been introduced, allowing the QNode to keep track of the results of previous
  device executions and reuse those results in subsequent calls.
  Note that QNode caching is only supported in the new and experimental tape-mode.
  [(#817)](https://github.com/PennyLaneAI/pennylane/pull/817)

  Caching is available by passing a `caching` argument to the QNode:

  ```python
  dev = qml.device("default.qubit", wires=2)
  qml.enable_tape()

  @qml.qnode(dev, caching=10)  # cache up to 10 evaluations
  def qfunc(x):
      qml.RX(x, wires=0)
      qml.RX(0.3, wires=1)
      qml.CNOT(wires=[0, 1])
      return qml.expval(qml.PauliZ(1))

  qfunc(0.1)  # first evaluation executes on the device
  qfunc(0.1)  # second evaluation accesses the cached result
  ```

* Sped up the application of certain gates in `default.qubit` by using array/tensor
  manipulation tricks. The following gates are affected: `PauliX`, `PauliY`, `PauliZ`,
  `Hadamard`, `SWAP`, `S`, `T`, `CNOT`, `CZ`.
  [(#772)](https://github.com/PennyLaneAI/pennylane/pull/772)

* The computation of marginal probabilities has been made more efficient for devices
  with a large number of wires, achieving in some cases a 5x speedup.
  [(#799)](https://github.com/PennyLaneAI/pennylane/pull/799)

* Adds arithmetic operations (addition, tensor product,
  subtraction, and scalar multiplication) between `Hamiltonian`,
  `Tensor`, and `Observable` objects, and inline arithmetic
  operations between Hamiltonians and other observables.
  [(#765)](https://github.com/PennyLaneAI/pennylane/pull/765)

  Hamiltonians can now easily be defined as sums of observables:

  ```pycon3
  >>> H = 3 * qml.PauliZ(0) - (qml.PauliX(0) @ qml.PauliX(1)) + qml.Hamiltonian([4], [qml.PauliZ(0)])
  >>> print(H)
  (7.0) [Z0] + (-1.0) [X0 X1]
  ```

* Adds `compare()` method to `Observable` and `Hamiltonian` classes, which allows
  for comparison between observable quantities.
  [(#765)](https://github.com/PennyLaneAI/pennylane/pull/765)

  ```pycon3
  >>> H = qml.Hamiltonian([1], [qml.PauliZ(0)])
  >>> obs = qml.PauliZ(0) @ qml.Identity(1)
  >>> print(H.compare(obs))
  True
  ```

  ```pycon3
  >>> H = qml.Hamiltonian([2], [qml.PauliZ(0)])
  >>> obs = qml.PauliZ(1) @ qml.Identity(0)
  >>> print(H.compare(obs))
  False
  ```

* Adds `simplify()` method to the `Hamiltonian` class.
  [(#765)](https://github.com/PennyLaneAI/pennylane/pull/765)

  ```pycon3
  >>> H = qml.Hamiltonian([1, 2], [qml.PauliZ(0), qml.PauliZ(0) @ qml.Identity(1)])
  >>> H.simplify()
  >>> print(H)
  (3.0) [Z0]
  ```

* Added a new bit-flip mixer to the `qml.qaoa` module.
  [(#774)](https://github.com/PennyLaneAI/pennylane/pull/774)

* Summation of two `Wires` objects is now supported and will return
  a `Wires` object containing the set of all wires defined by the
  terms in the summation.
  [(#812)](https://github.com/PennyLaneAI/pennylane/pull/812)

<h3>Breaking changes</h3>

* The PennyLane NumPy module now returns scalar (zero-dimensional) arrays where
  Python scalars were previously returned.
  [(#820)](https://github.com/PennyLaneAI/pennylane/pull/820)
  [(#833)](https://github.com/PennyLaneAI/pennylane/pull/833)

  For example, this affects array element indexing, and summation:

  ```pycon
  >>> x = np.array([1, 2, 3], requires_grad=False)
  >>> x[0]
  tensor(1, requires_grad=False)
  >>> np.sum(x)
  tensor(6, requires_grad=True)
  ```

  This may require small updates to user code. A convenience method, `np.tensor.unwrap()`,
  has been added to help ease the transition. This converts PennyLane NumPy tensors
  to standard NumPy arrays and Python scalars:

  ```pycon
  >>> x = np.array(1.543, requires_grad=False)
  >>> x.unwrap()
  1.543
  ```

  Note, however, that information regarding array differentiability will be
  lost.

* The device capabilities dictionary has been redesigned, for clarity and robustness. In particular,
  the capabilities dictionary is now inherited from the parent class, various keys have more
  expressive names, and all keys are now defined in the base device class. For more details, please
  [refer to the developer
  documentation](https://pennylane.readthedocs.io/en/stable/development/plugins.html#device-capabilities).
  [(#781)](https://github.com/PennyLaneAI/pennylane/pull/781/files)

<h3>Bug fixes</h3>

* Changed to use lists for storing variable values inside `BaseQNode`
  allowing complex matrices to be passed to `QubitUnitary`.
  [(#773)](https://github.com/PennyLaneAI/pennylane/pull/773)

* Fixed a bug within `default.qubit`, resulting in greater efficiency
  when applying a state vector to all wires on the device.
  [(#849)](https://github.com/PennyLaneAI/pennylane/pull/849)

<h3>Documentation</h3>

* Equations have been added to the `qml.sample` and `qml.probs` docstrings
  to clarify the mathematical foundation of the performed measurements.
  [(#843)](https://github.com/PennyLaneAI/pennylane/pull/843)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Aroosa Ijaz, Juan Miguel Arrazola, Thomas Bromley, Jack Ceroni, Alain Delgado Gran, Josh Izaac,
Soran Jahangiri, Nathan Killoran, Robert Lang, Cedric Lin, Olivia Di Matteo, Nicolás Quesada, Maria
Schuld, Antal Száva.
