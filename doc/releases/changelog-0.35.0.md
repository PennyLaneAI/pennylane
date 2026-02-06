
# Release 0.35.0

<h3>New features since last release</h3>

<h4>Qiskit 1.0 integration 🔌</h4>

* This version of PennyLane makes it easier to import circuits from Qiskit.
  [(#5218)](https://github.com/PennyLaneAI/pennylane/pull/5218)
  [(#5168)](https://github.com/PennyLaneAI/pennylane/pull/5168)

  The `qp.from_qiskit` function converts a Qiskit
  [QuantumCircuit](https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit) into
  a PennyLane
  [quantum function](https://docs.pennylane.ai/en/stable/introduction/circuits.html#quantum-functions).
  Although `qp.from_qiskit` already exists in PennyLane, we have made a number of improvements to
  make importing from Qiskit easier. And yes — `qp.from_qiskit` functionality
  is compatible with both Qiskit
  [1.0](https://docs.quantum.ibm.com/api/qiskit/release-notes/1.0) and earlier
  versions! Here's a comprehensive list of the improvements:

  * You can now append PennyLane measurements onto the quantum function returned by
    `qp.from_qiskit`. Consider this simple Qiskit circuit:

    ```python
    import pennylane as qp
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.rx(0.785, 0)
    qc.ry(1.57, 1)
    ```

    We can convert it into a PennyLane QNode in just a few lines, with PennyLane
    `measurements` easily included:

    ```pycon
    >>> dev = qp.device("default.qubit")
    >>> measurements = qp.expval(qp.Z(0) @ qp.Z(1))
    >>> qfunc = qp.from_qiskit(qc, measurements=measurements)
    >>> qnode = qp.QNode(qfunc, dev)
    >>> qnode()
    tensor(0.00056331, requires_grad=True)
    ```

  * Quantum circuits that already contain Qiskit-side measurements can be faithfully converted with
    `qp.from_qiskit`. Consider this example Qiskit circuit:

    ```python
    qc = QuantumCircuit(3, 2)  # Teleportation

    qc.rx(0.9, 0)  # Prepare input state on qubit 0

    qc.h(1)  # Prepare Bell state on qubits 1 and 2
    qc.cx(1, 2)

    qc.cx(0, 1)  # Perform teleportation
    qc.h(0)
    qc.measure(0, 0)
    qc.measure(1, 1)

    with qc.if_test((1, 1)):  # Perform first conditional
        qc.x(2)
    ```

    This circuit can be converted into PennyLane with the Qiskit measurements still accessible. For example, we can
    use those results as inputs to a mid-circuit measurement in PennyLane:

    ```python
    @qp.qnode(dev)
    def teleport():
        m0, m1 = qp.from_qiskit(qc)()
        qp.cond(m0, qp.Z)(2)
        return qp.density_matrix(2)
    ```

    ```pycon
    >>> teleport()
    tensor([[0.81080498+0.j        , 0.        +0.39166345j],
            [0.        -0.39166345j, 0.18919502+0.j        ]], requires_grad=True)
    ```

  * It is now more intuitive to handle and differentiate parametrized Qiskit circuits. Consider the following circuit:

    ```python
    from qiskit.circuit import Parameter
    from pennylane import numpy as np

    angle0 = Parameter("x")
    angle1 = Parameter("y")

    qc = QuantumCircuit(2, 2)
    qc.rx(angle0, 0)
    qc.ry(angle1, 1)
    qc.cx(1, 0)
    ```

    We can convert this circuit into a QNode with two arguments, corresponding to `x` and `y`:

    ```python
    measurements = qp.expval(qp.PauliZ(0))
    qfunc = qp.from_qiskit(qc, measurements)
    qnode = qp.QNode(qfunc, dev)
    ```

    The QNode can be evaluated and differentiated:

    ```pycon
    >>> x, y = np.array([0.4, 0.5], requires_grad=True)
    >>> qnode(x, y)
    tensor(0.80830707, requires_grad=True)
    >>> qp.grad(qnode)(x, y)
    (tensor(-0.34174675, requires_grad=True),
     tensor(-0.44158016, requires_grad=True))
    ```

    This shows how easy it is to make a Qiskit circuit differentiable with PennyLane.

* In addition to circuits, it is also possible to convert operators from Qiskit to PennyLane with a new function called
  `qp.from_qiskit_op`.
  [(#5251)](https://github.com/PennyLaneAI/pennylane/pull/5251)

  A Qiskit
  [SparsePauliOp](https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.SparsePauliOp) can be
  converted to a PennyLane operator using `qp.from_qiskit_op`:

  ```pycon
  >>> from qiskit.quantum_info import SparsePauliOp
  >>> qiskit_op = SparsePauliOp(["II", "XY"])
  >>> qiskit_op
  SparsePauliOp(['II', 'XY'],
                coeffs=[1.+0.j, 1.+0.j])
  >>> pl_op = qp.from_qiskit_op(qiskit_op)
  >>> pl_op
  I(0) + X(1) @ Y(0)
  ```

  Combined with `qp.from_qiskit`, it becomes easy to quickly calculate quantities like expectation
  values by converting the whole workflow to PennyLane:

  ```python
  qc = QuantumCircuit(2)  # Create circuit
  qc.rx(0.785, 0)
  qc.ry(1.57, 1)

  measurements = qp.expval(pl_op)  # Create QNode
  qfunc = qp.from_qiskit(qc, measurements)
  qnode = qp.QNode(qfunc, dev)
  ```

  ```pycon
  >>> qnode()  # Evaluate!
  tensor(0.29317504, requires_grad=True)
  ```

<h4>Native mid-circuit measurements on Default Qubit 💡</h4>

* Mid-circuit measurements can now be more scalable and efficient in finite-shots mode
  with `default.qubit` by simulating them in a similar way to what happens on quantum hardware.
  [(#5088)](https://github.com/PennyLaneAI/pennylane/pull/5088)
  [(#5120)](https://github.com/PennyLaneAI/pennylane/pull/5120)

  Previously, mid-circuit measurements (MCMs) would be automatically replaced with an additional qubit
  using the `@qp.defer_measurements` transform. The circuit below would have required thousands
  of qubits to simulate.

  Now, MCMs are performed in a similar way to quantum hardware
  with finite shots on `default.qubit`. For each shot and each time an MCM is encountered,
  the device evaluates the probability of projecting onto `|0>` or `|1>` and makes a random choice to
  collapse the circuit state. This approach works well when there are a lot of MCMs
  and the number of shots is not too high.

  ```python
  import pennylane as qp

  dev = qp.device("default.qubit", shots=10)

  @qp.qnode(dev)
  def f():
      for i in range(1967):
          qp.Hadamard(0)
          qp.measure(0)
      return qp.sample(qp.PauliX(0))
  ```

  ```pycon
  >>> f()
  tensor([-1, -1, -1,  1,  1, -1,  1, -1,  1, -1], requires_grad=True)
  ```

<h4>Work easily and efficiently with operators 🔧</h4>

* Over the past few releases, PennyLane's approach to operator arithmetic has been in the process
  of being overhauled. We have a few objectives:

  1. To make it as easy to work with PennyLane operators as it would be with pen and paper.
  2. To improve the efficiency of operator arithmetic.

  The updated operator arithmetic functionality is still being finalized, but can be activated
  using `qp.operation.enable_new_opmath()`. In the next release, the new behaviour will become the
  default, so we recommend enabling now to become familiar with the new system!

  The following updates have been made in this version of PennyLane:

  * You can now easily access Pauli operators via `I`, `X`, `Y`, and `Z`:
    [(#5116)](https://github.com/PennyLaneAI/pennylane/pull/5116)

    ```pycon
    >>> from pennylane import I, X, Y, Z
    >>> X(0)
    X(0)
    ```

    The original long-form names `Identity`, `PauliX`, `PauliY`, and `PauliZ` remain available, but
    use of the short-form names is now recommended.

    The original long-form names `Identity`, `PauliX`, `PauliY`, and `PauliZ` remain available, but
    use of the short-form names is now recommended.

  * A new `qp.commutator` function is now available that allows you to compute commutators between
    PennyLane operators.
    [(#5051)](https://github.com/PennyLaneAI/pennylane/pull/5051)
    [(#5052)](https://github.com/PennyLaneAI/pennylane/pull/5052)
    [(#5098)](https://github.com/PennyLaneAI/pennylane/pull/5098)

    ```pycon
    >>> qp.commutator(X(0), Y(0))
    2j * Z(0)
    ```

  * Operators in PennyLane can have a backend Pauli representation, which can be used to perform faster operator arithmetic. Now, the Pauli
    representation will be automatically used for calculations when available.
    [(#4989)](https://github.com/PennyLaneAI/pennylane/pull/4989)
    [(#5001)](https://github.com/PennyLaneAI/pennylane/pull/5001)
    [(#5003)](https://github.com/PennyLaneAI/pennylane/pull/5003)
    [(#5017)](https://github.com/PennyLaneAI/pennylane/pull/5017)
    [(#5027)](https://github.com/PennyLaneAI/pennylane/pull/5027)

    The Pauli representation can be optionally accessed via `op.pauli_rep`:

    ```pycon
    >>> qp.operation.enable_new_opmath()
    >>> op = X(0) + Y(0)
    >>> op.pauli_rep
    1.0 * X(0)
    + 1.0 * Y(0)
    ```

  * Extensive improvements have been made to the string representations of PennyLane operators,
    making them shorter and possible to copy-paste as valid PennyLane code.
    [(#5116)](https://github.com/PennyLaneAI/pennylane/pull/5116)
    [(#5138)](https://github.com/PennyLaneAI/pennylane/pull/5138)

    ```
    >>> 0.5 * X(0)
    0.5 * X(0)
    >>> 0.5 * (X(0) + Y(1))
    0.5 * (X(0) + Y(1))
    ```

    Sums with many terms are broken up into multiple lines, but can still be copied back as valid
    code:

    ```
    >>> 0.5 * (X(0) @ X(1)) + 0.7 * (X(1) @ X(2)) + 0.8 * (X(2) @ X(3))
    (
        0.5 * (X(0) @ X(1))
      + 0.7 * (X(1) @ X(2))
      + 0.8 * (X(2) @ X(3))
    )
    ```

  * Linear combinations of operators and operator multiplication via `Sum` and `Prod`, respectively,
    have been updated to reach feature parity with `Hamiltonian` and `Tensor`, respectively.
    This should minimize the effort to port over any existing code.
    [(#5070)](https://github.com/PennyLaneAI/pennylane/pull/5070)
    [(#5132)](https://github.com/PennyLaneAI/pennylane/pull/5132)
    [(#5133)](https://github.com/PennyLaneAI/pennylane/pull/5133)

    Updates include support for grouping via the `pauli` module:

    ```pycon
    >>> obs = [X(0) @ Y(1), Z(0), Y(0) @ Z(1), Y(1)]
    >>> qp.pauli.group_observables(obs)
    [[Y(0) @ Z(1)], [X(0) @ Y(1), Y(1)], [Z(0)]]
    ```

<h4>New Clifford device 🦾</h4>

* A new `default.clifford` device enables efficient simulation of large-scale Clifford circuits
  defined in PennyLane through the use of [stim](https://github.com/quantumlib/Stim) as a backend.
  [(#4936)](https://github.com/PennyLaneAI/pennylane/pull/4936)
  [(#4954)](https://github.com/PennyLaneAI/pennylane/pull/4954)
  [(#5144)](https://github.com/PennyLaneAI/pennylane/pull/5144)

  Given a circuit with only Clifford gates, one can use this device to obtain the usual range
  of PennyLane [measurements](https://docs.pennylane.ai/en/stable/introduction/measurements.html)
  as well as the state represented in the Tableau form of
  [Aaronson & Gottesman (2004)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.70.052328):

  ```python
  import pennylane as qp

  dev = qp.device("default.clifford", tableau=True)
  @qp.qnode(dev)
  def circuit():
      qp.CNOT(wires=[0, 1])
      qp.PauliX(wires=[1])
      qp.ISWAP(wires=[0, 1])
      qp.Hadamard(wires=[0])
      return qp.state()
  ```

  ```pycon
  >>> circuit()
  array([[0, 1, 1, 0, 0],
        [1, 0, 1, 1, 1],
        [0, 0, 0, 1, 0],
        [1, 0, 0, 1, 1]])
  ```

  The `default.clifford` device also supports the `PauliError`, `DepolarizingChannel`, `BitFlip` and
  `PhaseFlip`
  [noise channels](https://docs.pennylane.ai/en/latest/introduction/operations.html#noisy-channels)
  when operating in finite-shot mode.

<h3>Improvements 🛠</h3>

<h4>Faster gradients with VJPs and other performance improvements</h4>

* Vector-Jacobian products (VJPs) can result in faster computations when the output of your quantum
  Node has a low dimension. They can be enabled by setting `device_vjp=True` when loading a QNode.
  In the next release of PennyLane, VJPs are planned to be used by default, when available.

  In this release, we have unlocked:

  * Adjoint device VJPs can be used with `jax.jacobian`, meaning that `device_vjp=True` is always
    faster when using JAX with `default.qubit`.
    [(#4963)](https://github.com/PennyLaneAI/pennylane/pull/4963)

  * PennyLane can now use lightning-provided VJPs.
    [(#4914)](https://github.com/PennyLaneAI/pennylane/pull/4914)

  * VJPs can be used with TensorFlow, though support has not yet been added
    for `tf.Function` and Tensorflow Autograph.
    [(#4676)](https://github.com/PennyLaneAI/pennylane/pull/4676)

* Measuring `qp.probs` is now faster due to an optimization in converting samples to counts.
  [(#5145)](https://github.com/PennyLaneAI/pennylane/pull/5145)

* The performance of circuit-cutting workloads with large numbers of generated tapes has been improved.
  [(#5005)](https://github.com/PennyLaneAI/pennylane/pull/5005)

* Queueing (`AnnotatedQueue`) has been removed from `qp.cut_circuit` and `qp.cut_circuit_mc` to improve performance
  for large workflows.
  [(#5108)](https://github.com/PennyLaneAI/pennylane/pull/5108)


<h4>Community contributions 🥳</h4>

* A new function called `qp.fermi.parity_transform` has been added for parity mapping of a fermionic Hamiltonian.
  [(#4928)](https://github.com/PennyLaneAI/pennylane/pull/4928)

  It is now possible to transform a fermionic Hamiltonian to a qubit Hamiltonian with parity mapping.

  ```python
  import pennylane as qp
  fermi_ham = qp.fermi.FermiWord({(0, 0) : '+', (1, 1) : '-'})

  qubit_ham = qp.fermi.parity_transform(fermi_ham, n=6)
  ```

  ```pycon
  >>> print(qubit_ham)
  -0.25j * Y(0) + (-0.25+0j) * (X(0) @ Z(1)) + (0.25+0j) * X(0) + 0.25j * (Y(0) @ Z(1))
  ```

* The transform `split_non_commuting` now accepts measurements of type `probs`, `sample`, and `counts`, which accept both wires and observables.
  [(#4972)](https://github.com/PennyLaneAI/pennylane/pull/4972)

* The efficiency of matrix calculations when an operator is symmetric over a given set of wires has been improved.
  [(#3601)](https://github.com/PennyLaneAI/pennylane/pull/3601)

* The `pennylane/math/quantum.py` module now has support for computing the minimum entropy of a density matrix.
  [(#3959)](https://github.com/PennyLaneAI/pennylane/pull/3959/)

  ```pycon
  >>> x = [1, 0, 0, 1] / np.sqrt(2)
  >>> x = qp.math.dm_from_state_vector(x)
  >>> qp.math.min_entropy(x, indices=[0])
  0.6931471805599455
  ```

* A function called `apply_operation` that applies operations to device-compatible states has been added to the new `qutrit_mixed` module found in `qp.devices`.
  [(#5032)](https://github.com/PennyLaneAI/pennylane/pull/5032)

* A function called `measure` has been added to the new `qutrit_mixed` module found in `qp.devices` that measures device-compatible states for a collection of measurement processes.
  [(#5049)](https://github.com/PennyLaneAI/pennylane/pull/5049)

* A `partial_trace` function has been added to `qp.math` for taking the partial trace of matrices.
  [(#5152)](https://github.com/PennyLaneAI/pennylane/pull/5152)

<h4>Other operator arithmetic improvements</h4>

* The following capabilities have been added for Pauli arithmetic:
  [(#4989)](https://github.com/PennyLaneAI/pennylane/pull/4989)
  [(#5001)](https://github.com/PennyLaneAI/pennylane/pull/5001)
  [(#5003)](https://github.com/PennyLaneAI/pennylane/pull/5003)
  [(#5017)](https://github.com/PennyLaneAI/pennylane/pull/5017)
  [(#5027)](https://github.com/PennyLaneAI/pennylane/pull/5027)
  [(#5018)](https://github.com/PennyLaneAI/pennylane/pull/5018)

  * You can now multiply `PauliWord` and `PauliSentence` instances by scalars (e.g.,
    `0.5 * PauliWord({0: "X"})` or `0.5 * PauliSentence({PauliWord({0: "X"}): 1.})`).

  * You can now intuitively add and subtract `PauliWord` and `PauliSentence` instances and scalars together (scalars are treated implicitly as multiples
    of the identity, `I`). For example, `ps1 + pw1 + 1.` for some Pauli word `pw1 = PauliWord({0: "X", 1: "Y"})` and Pauli sentence `ps1 = PauliSentence({pw1: 3.})`.

  * You can now element-wise multiply `PauliWord`, `PauliSentence`, and operators together with `qp.dot` (e.g.,
    `qp.dot([0.5, -1.5, 2], [pw1, ps1, id_word])` with `id_word = PauliWord({})`).

  * `qp.matrix` now accepts `PauliWord` and `PauliSentence` instances (e.g., `qp.matrix(PauliWord({0: "X"}))`).

  * It is now possible to compute commutators with Pauli operators natively with the new `commutator` method.

    ```pycon
    >>> op1 = PauliWord({0: "X", 1: "X"})
    >>> op2 = PauliWord({0: "Y"}) + PauliWord({1: "Y"})
    >>> op1.commutator(op2)
    2j * Z(0) @ X(1)
    + 2j * X(0) @ Z(1)
    ```

* Composite operations (e.g., those made with `qp.prod` and `qp.sum`) and scalar-product operations
  convert `Hamiltonian` and `Tensor` operands to `Sum` and `Prod` types, respectively. This helps
  avoid the mixing of incompatible operator types.
  [(#5031)](https://github.com/PennyLaneAI/pennylane/pull/5031)
  [(#5063)](https://github.com/PennyLaneAI/pennylane/pull/5063)

* `qp.Identity()` can be initialized without wires. Measuring it is currently not possible, though.
  [(#5106)](https://github.com/PennyLaneAI/pennylane/pull/5106)

* `qp.dot` now returns a `Sum` class even when all the coefficients match.
  [(#5143)](https://github.com/PennyLaneAI/pennylane/pull/5143)

* `qp.pauli.group_observables` now supports grouping `Prod` and `SProd` operators.
  [(#5070)](https://github.com/PennyLaneAI/pennylane/pull/5070)

* The performance of converting a `PauliSentence` to a `Sum` has been improved.
  [(#5141)](https://github.com/PennyLaneAI/pennylane/pull/5141)
  [(#5150)](https://github.com/PennyLaneAI/pennylane/pull/5150)

* Akin to `qp.Hamiltonian` features, the coefficients and operators that make up composite operators formed via `Sum` or `Prod` can now be accessed
  with the `terms()` method.
  [(#5132)](https://github.com/PennyLaneAI/pennylane/pull/5132)
  [(#5133)](https://github.com/PennyLaneAI/pennylane/pull/5133)
  [(#5164)](https://github.com/PennyLaneAI/pennylane/pull/5164)

  ```python
  >>> qp.operation.enable_new_opmath()
  >>> op = X(0) @ (0.5 * X(1) + X(2))
  >>> op.terms()
  ([0.5, 1.0],
   [X(1) @ X(0),
    X(2) @ X(0)])
  ```

* String representations of `ParametrizedHamiltonian` have been updated to match the style of other PL operators.
  [(#5215)](https://github.com/PennyLaneAI/pennylane/pull/5215)

<h4>Other improvements</h4>

* The `pl-device-test` suite is now compatible with the `qp.devices.Device` interface.
  [(#5229)](https://github.com/PennyLaneAI/pennylane/pull/5229)

* The `QSVT` operation now determines its `data` from the block encoding and projector operator data.
  [(#5226)](https://github.com/PennyLaneAI/pennylane/pull/5226)
  [(#5248)](https://github.com/PennyLaneAI/pennylane/pull/5248)

* The `BlockEncode` operator is now JIT-compatible with JAX.
  [(#5110)](https://github.com/PennyLaneAI/pennylane/pull/5110)

* The `qp.qsvt` function uses `qp.GlobalPhase` instead of `qp.exp` to define a global phase.
  [(#5105)](https://github.com/PennyLaneAI/pennylane/pull/5105)

* The `tests/ops/functions/conftest.py` test has been updated to ensure that all operator types are tested for validity.
  [(#4978)](https://github.com/PennyLaneAI/pennylane/pull/4978)

* A new `pennylane.workflow` module has been added. This module now contains `qnode.py`, `execution.py`, `set_shots.py`, `jacobian_products.py`, and the submodule `interfaces`.
  [(#5023)](https://github.com/PennyLaneAI/pennylane/pull/5023)

* A more informative error is now raised when calling `adjoint_jacobian` with trainable state-prep operations.
  [(#5026)](https://github.com/PennyLaneAI/pennylane/pull/5026)

* `qp.workflow.get_transform_program` and `qp.workflow.construct_batch` have been added to inspect the transform program and batch of tapes
  at different stages.
  [(#5084)](https://github.com/PennyLaneAI/pennylane/pull/5084)

* All custom controlled operations such as `CRX`, `CZ`, `CNOT`, `ControlledPhaseShift` now inherit from `ControlledOp`, giving them additional properties such as `control_wire` and `control_values`. Calling `qp.ctrl` on `RX`, `RY`, `RZ`, `Rot`, and `PhaseShift` with a single control wire will return gates of types `CRX`, `CRY`, etc. as opposed to a general `Controlled` operator.
  [(#5069)](https://github.com/PennyLaneAI/pennylane/pull/5069)
  [(#5199)](https://github.com/PennyLaneAI/pennylane/pull/5199)

* The CI will now fail if coverage data fails to upload to codecov. Previously, it would silently pass
  and the codecov check itself would never execute.
  [(#5101)](https://github.com/PennyLaneAI/pennylane/pull/5101)

* `qp.ctrl` called on operators with custom controlled versions will now return instances
  of the custom class, and it will flatten nested controlled operators to a single
  multi-controlled operation. For `PauliX`, `CNOT`, `Toffoli`, and `MultiControlledX`,
  calling `qp.ctrl` will always resolve to the best option in `CNOT`, `Toffoli`, or
  `MultiControlledX` depending on the number of control wires and control values.
  [(#5125)](https://github.com/PennyLaneAI/pennylane/pull/5125/)

* Unwanted warning filters have been removed from tests and no `PennyLaneDeprecationWarning`s
  are being raised unexpectedly.
  [(#5122)](https://github.com/PennyLaneAI/pennylane/pull/5122)

* New error tracking and propagation functionality has been added
  [(#5115)](https://github.com/PennyLaneAI/pennylane/pull/5115)
  [(#5121)](https://github.com/PennyLaneAI/pennylane/pull/5121)

* The method `map_batch_transform` has been replaced with the method `_batch_transform`
  implemented in `TransformDispatcher`.
  [(#5212)](https://github.com/PennyLaneAI/pennylane/pull/5212)

* `TransformDispatcher` can now dispatch onto a batch of tapes, making it easier to compose transforms
  when working in the tape paradigm.
  [(#5163)](https://github.com/PennyLaneAI/pennylane/pull/5163)

* `qp.ctrl` is now a simple wrapper that either calls PennyLane's built in `create_controlled_op`
  or uses the Catalyst implementation.
  [(#5247)](https://github.com/PennyLaneAI/pennylane/pull/5247)

* Controlled composite operations can now be decomposed using ZYZ rotations.
  [(#5242)](https://github.com/PennyLaneAI/pennylane/pull/5242)

* New functions called `qp.devices.modifiers.simulator_tracking` and `qp.devices.modifiers.single_tape_support` have been added
  to add basic default behavior onto a device class.
  [(#5200)](https://github.com/PennyLaneAI/pennylane/pull/5200)

<h3>Breaking changes 💔</h3>

* Passing additional arguments to a transform that decorates a QNode must now be done through the use
  of `functools.partial`.
  [(#5046)](https://github.com/PennyLaneAI/pennylane/pull/5046)

* `qp.ExpvalCost` has been removed. Users should use `qp.expval()` moving forward.
  [(#5097)](https://github.com/PennyLaneAI/pennylane/pull/5097)

* Caching of executions is now turned off by default when `max_diff == 1`, as the classical overhead cost
  outweighs the probability that duplicate circuits exists.
  [(#5243)](https://github.com/PennyLaneAI/pennylane/pull/5243)

* The entry point convention registering compilers with PennyLane has changed.
  [(#5140)](https://github.com/PennyLaneAI/pennylane/pull/5140)

  To allow for packages to register multiple compilers with PennyLane,
  the `entry_points` convention under the designated group name
  `pennylane.compilers` has been modified.

  Previously, compilers would register `qjit` (JIT decorator),
  `ops` (compiler-specific operations), and `context` (for tracing and
  program capture).

  Now, compilers must register `compiler_name.qjit`, `compiler_name.ops`,
  and `compiler_name.context`, where `compiler_name` is replaced
  by the name of the provided compiler.

  For more information, please see the
  [documentation on adding compilers](https://docs.pennylane.ai/en/stable/code/qml_compiler.html#adding-a-compiler).

* PennyLane source code is now compatible with the latest version of `black`.
  [(#5112)](https://github.com/PennyLaneAI/pennylane/pull/5112)
  [(#5119)](https://github.com/PennyLaneAI/pennylane/pull/5119)

* `gradient_analysis_and_validation` has been renamed to `find_and_validate_gradient_methods`. Instead of returning a list, it now returns a dictionary of gradient methods for each parameter index, and no longer mutates the tape.
  [(#5035)](https://github.com/PennyLaneAI/pennylane/pull/5035)

* Multiplying two `PauliWord` instances no longer returns a tuple `(new_word, coeff)`
  but instead `PauliSentence({new_word: coeff})`. The old behavior is still available
  with the private method `PauliWord._matmul(other)` for faster processing.
  [(#5045)](https://github.com/PennyLaneAI/pennylane/pull/5054)

* `Observable.return_type` has been removed. Instead, you should inspect the type
  of the surrounding measurement process.
  [(#5044)](https://github.com/PennyLaneAI/pennylane/pull/5044)

* `ClassicalShadow.entropy()` no longer needs an `atol` keyword as a better
  method to estimate entropies from approximate density matrix reconstructions
  (with potentially negative eigenvalues).
  [(#5048)](https://github.com/PennyLaneAI/pennylane/pull/5048)

* Controlled operators with a custom controlled version decompose like how their controlled
  counterpart decomposes as opposed to decomposing into their controlled version.
  [(#5069)](https://github.com/PennyLaneAI/pennylane/pull/5069)
  [(#5125)](https://github.com/PennyLaneAI/pennylane/pull/5125/)

  For example:

  ```pycon
  >>> qp.ctrl(qp.RX(0.123, wires=1), control=0).decomposition()
  [
    RZ(1.5707963267948966, wires=[1]),
    RY(0.0615, wires=[1]),
    CNOT(wires=[0, 1]),
    RY(-0.0615, wires=[1]),
    CNOT(wires=[0, 1]),
    RZ(-1.5707963267948966, wires=[1])
  ]
  ```

* `QuantumScript.is_sampled` and `QuantumScript.all_sampled` have been removed. Users should now
  validate these properties manually.
  [(#5072)](https://github.com/PennyLaneAI/pennylane/pull/5072)

* `qp.transforms.one_qubit_decomposition` and `qp.transforms.two_qubit_decomposition` have been removed. Instead,
  you should use `qp.ops.one_qubit_decomposition` and `qp.ops.two_qubit_decomposition`.
  [(#5091)](https://github.com/PennyLaneAI/pennylane/pull/5091)

<h3>Deprecations 👋</h3>

* Calling `qp.matrix` without providing a `wire_order` on objects where the wire order could be
  ambiguous now raises a warning. In the future, the `wire_order` argument will be required in
  these cases.
  [(#5039)](https://github.com/PennyLaneAI/pennylane/pull/5039)

* `Operator.validate_subspace(subspace)` has been relocated to the `qp.ops.qutrit.parametric_ops`
  module and will be removed from the Operator class in an upcoming release.
  [(#5067)](https://github.com/PennyLaneAI/pennylane/pull/5067)

* Matrix and tensor products between `PauliWord` and `PauliSentence` instances are done using
  the `@` operator, `*` will be used only for scalar multiplication. Note also the breaking
  change that the product of two `PauliWord` instances now returns a `PauliSentence` instead
  of a tuple `(new_word, coeff)`.
  [(#4989)](https://github.com/PennyLaneAI/pennylane/pull/4989)
  [(#5054)](https://github.com/PennyLaneAI/pennylane/pull/5054)

* `MeasurementProcess.name` and `MeasurementProcess.data` are now deprecated, as they contain dummy
  values that are no longer needed.
  [(#5047)](https://github.com/PennyLaneAI/pennylane/pull/5047)
  [(#5071)](https://github.com/PennyLaneAI/pennylane/pull/5071)
  [(#5076)](https://github.com/PennyLaneAI/pennylane/pull/5076)
  [(#5122)](https://github.com/PennyLaneAI/pennylane/pull/5122)

* `qp.pauli.pauli_mult` and `qp.pauli.pauli_mult_with_phase` are now deprecated. Instead, you
  should use `qp.simplify(qp.prod(pauli_1, pauli_2))` to get the reduced operator.
  [(#5057)](https://github.com/PennyLaneAI/pennylane/pull/5057)

* The private functions `_pauli_mult`, `_binary_matrix` and `_get_pauli_map` from the
  `pauli` module have been deprecated, as they are no longer used anywhere and the same
  functionality can be achieved using newer features in the `pauli` module.
  [(#5057)](https://github.com/PennyLaneAI/pennylane/pull/5057)

* `Sum.ops`, `Sum.coeffs`, `Prod.ops` and `Prod.coeffs` will be deprecated in the future.
  [(#5164)](https://github.com/PennyLaneAI/pennylane/pull/5164)

<h3>Documentation 📝</h3>

* The module documentation for `pennylane.tape` now explains the difference between `QuantumTape` and `QuantumScript`.
  [(#5065)](https://github.com/PennyLaneAI/pennylane/pull/5065)

* A typo in a code example in the `qp.transforms` API has been fixed.
  [(#5014)](https://github.com/PennyLaneAI/pennylane/pull/5014)

* Documentation for `qp.data` has been updated and now mentions a way to access the same dataset simultaneously from multiple environments.
  [(#5029)](https://github.com/PennyLaneAI/pennylane/pull/5029)

* A clarification for the definition of `argnum` added to gradient methods has been made.
  [(#5035)](https://github.com/PennyLaneAI/pennylane/pull/5035)

* A typo in the code example for `qp.qchem.dipole_of` has been fixed.
  [(#5036)](https://github.com/PennyLaneAI/pennylane/pull/5036)

* A development guide on deprecations and removals has been added.
  [(#5083)](https://github.com/PennyLaneAI/pennylane/pull/5083)

* A note about the eigenspectrum of second-quantized Hamiltonians has been added to `qp.eigvals`.
  [(#5095)](https://github.com/PennyLaneAI/pennylane/pull/5095)

* A warning about two mathematically equivalent Hamiltonians undergoing different time evolutions has been added to `qp.TrotterProduct` and `qp.ApproxTimeEvolution`.
  [(#5137)](https://github.com/PennyLaneAI/pennylane/pull/5137)

* A reference to the paper that provides the image of the `qp.QAOAEmbedding` template has been added.
  [(#5130)](https://github.com/PennyLaneAI/pennylane/pull/5130)

* The docstring of `qp.sample` has been updated to advise the use of single-shot expectations
  instead when differentiating a circuit.
  [(#5237)](https://github.com/PennyLaneAI/pennylane/pull/5237)

* A quick start page has been added called "Importing Circuits". This explains
  how to import quantum circuits and operations defined outside of PennyLane.
  [(#5281)](https://github.com/PennyLaneAI/pennylane/pull/5281)

<h3>Bug fixes 🐛</h3>

*  `QubitChannel` can now be used with jitting.
  [(#5288)](https://github.com/PennyLaneAI/pennylane/pull/5288)

* Fixed a bug in the matplotlib drawer where the colour of `Barrier` did not match the requested style.
  [(#5276)](https://github.com/PennyLaneAI/pennylane/pull/5276)

* `qp.draw` and `qp.draw_mpl` now apply all applied transforms before drawing.
  [(#5277)](https://github.com/PennyLaneAI/pennylane/pull/5277)

* `ctrl_decomp_zyz` is now differentiable.
  [(#5198)](https://github.com/PennyLaneAI/pennylane/pull/5198)

* `qp.ops.Pow.matrix()` is now differentiable with TensorFlow with integer exponents.
  [(#5178)](https://github.com/PennyLaneAI/pennylane/pull/5178)

* The `qp.MottonenStatePreparation` template has been updated to include a global phase operation.
  [(#5166)](https://github.com/PennyLaneAI/pennylane/pull/5166)

* Fixed a queuing bug when using `qp.prod` with a quantum function that queues a single operator.
  [(#5170)](https://github.com/PennyLaneAI/pennylane/pull/5170)

* The `qp.TrotterProduct` template has been updated to accept scalar products of operators as an input Hamiltonian.
  [(#5073)](https://github.com/PennyLaneAI/pennylane/pull/5073)

* Fixed a bug where caching together with JIT compilation and broadcasted tapes yielded wrong results
  `Operator.hash` now depends on the memory location, `id`, of a JAX tracer instead of its string representation.
  [(#3917)](https://github.com/PennyLaneAI/pennylane/pull/3917)

* `qp.transforms.undo_swaps` can now work with operators with hyperparameters or nesting.
  [(#5081)](https://github.com/PennyLaneAI/pennylane/pull/5081)

* `qp.transforms.split_non_commuting` will now pass the original shots along.
  [(#5081)](https://github.com/PennyLaneAI/pennylane/pull/5081)

* If `argnum` is provided to a gradient transform, only the parameters specified in `argnum` will have their gradient methods validated.
  [(#5035)](https://github.com/PennyLaneAI/pennylane/pull/5035)

* `StatePrep` operations expanded onto more wires are now compatible with backprop.
  [(#5028)](https://github.com/PennyLaneAI/pennylane/pull/5028)

* `qp.equal` works well with `qp.Sum` operators when wire labels are a mix of integers and strings.
  [(#5037)](https://github.com/PennyLaneAI/pennylane/pull/5037)

* The return value of `Controlled.generator` now contains a projector that projects onto the correct subspace based on the control value specified.
  [(#5068)](https://github.com/PennyLaneAI/pennylane/pull/5068)

* `CosineWindow` no longer raises an unexpected error when used on a subset of wires at the beginning of a circuit.
  [(#5080)](https://github.com/PennyLaneAI/pennylane/pull/5080)

* `tf.function` now works with `TensorSpec(shape=None)` by skipping batch size computation.
  [(#5089)](https://github.com/PennyLaneAI/pennylane/pull/5089)

* `PauliSentence.wires` no longer imposes a false order.
  [(#5041)](https://github.com/PennyLaneAI/pennylane/pull/5041)

* `qp.qchem.import_state` now applies the chemist-to-physicist
  sign convention when initializing a PennyLane state vector from
  classically pre-computed wavefunctions. That is, it interleaves
  spin-up/spin-down operators for the same spatial orbital index,
  as standard in PennyLane (instead of commuting all spin-up
  operators to the left, as is standard in quantum chemistry).
  [(#5114)](https://github.com/PennyLaneAI/pennylane/pull/5114)

* Multi-wire controlled `CNOT` and `PhaseShift` are now be decomposed correctly.
  [(#5125)](https://github.com/PennyLaneAI/pennylane/pull/5125/)
  [(#5148)](https://github.com/PennyLaneAI/pennylane/pull/5148)

* `draw_mpl` no longer raises an error when drawing a circuit containing an adjoint of a controlled operation.
  [(#5149)](https://github.com/PennyLaneAI/pennylane/pull/5149)

* `default.mixed` no longer throws `ValueError` when applying a state vector that is not of type `complex128` when used with tensorflow.
  [(#5155)](https://github.com/PennyLaneAI/pennylane/pull/5155)

* `ctrl_decomp_zyz` no longer raises a `TypeError` if the rotation parameters are of type `torch.Tensor`
  [(#5183)](https://github.com/PennyLaneAI/pennylane/pull/5183)

* Comparing `Prod` and `Sum` objects now works regardless of nested structure with `qp.equal` if the
  operators have a valid `pauli_rep` property.
  [(#5177)](https://github.com/PennyLaneAI/pennylane/pull/5177)

* Controlled `GlobalPhase` with non-zero control wires no longer throws an error.
  [(#5194)](https://github.com/PennyLaneAI/pennylane/pull/5194)

* A `QNode` transformed with `mitigate_with_zne` now accepts batch parameters.
  [(#5195)](https://github.com/PennyLaneAI/pennylane/pull/5195)

* The matrix of an empty `PauliSentence` instance is now correct (all-zeros).
  Further, matrices of empty `PauliWord` and `PauliSentence` instances can now be turned into matrices.
  [(#5188)](https://github.com/PennyLaneAI/pennylane/pull/5188)

* `PauliSentence` instances can handle matrix multiplication with `PauliWord` instances.
  [(#5208)](https://github.com/PennyLaneAI/pennylane/pull/5208)

* `CompositeOp.eigendecomposition` is now JIT-compatible.
  [(#5207)](https://github.com/PennyLaneAI/pennylane/pull/5207)

* `QubitDensityMatrix` now works with JAX-JIT on the `default.mixed` device.
  [(#5203)](https://github.com/PennyLaneAI/pennylane/pull/5203)
  [(#5236)](https://github.com/PennyLaneAI/pennylane/pull/5236)

* When a QNode specifies `diff_method="adjoint"`, `default.qubit` no longer tries to decompose non-trainable operations with non-scalar parameters such as `QubitUnitary`.
  [(#5233)](https://github.com/PennyLaneAI/pennylane/pull/5233)

* The overwriting of the class names of `I`, `X`, `Y`, and `Z` no longer happens in the initialization after causing problems with datasets. This now
  happens globally.
  [(#5252)](https://github.com/PennyLaneAI/pennylane/pull/5252)

* The `adjoint_metric_tensor` transform now works with `jax`.
  [(#5271)](https://github.com/PennyLaneAI/pennylane/pull/5271)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Abhishek Abhishek,
Mikhail Andrenkov,
Utkarsh Azad,
Trenten Babcock,
Gabriel Bottrill,
Thomas Bromley,
Astral Cai,
Skylar Chan,
Isaac De Vlugt,
Diksha Dhawan,
Lillian Frederiksen,
Pietropaolo Frisoni,
Eugenio Gigante,
Diego Guala,
David Ittah,
Soran Jahangiri,
Jacky Jiang,
Korbinian Kottmann,
Christina Lee,
Xiaoran Li,
Vincent Michaud-Rioux,
Romain Moyard,
Pablo Antonio Moreno Casares,
Erick Ochoa Lopez,
Lee J. O'Riordan,
Mudit Pandey,
Alex Preciado,
Matthew Silverman,
Jay Soni.
