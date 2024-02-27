:orphan:

# Release 0.35.0 (current release)

<h3>New features since last release</h3>

<h4>Easy to import circuits 💾</h4>

* This version of PennyLane makes it easier to import workflows from Qiskit.
  [(#5218)](https://github.com/PennyLaneAI/pennylane/pull/5218)
  [(#5168)](https://github.com/PennyLaneAI/pennylane/pull/5168)

  The `qml.from_qiskit` function converts a Qiskit
  [QuantumCircuit](https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit) into
  a PennyLane
  [quantum function](https://docs.pennylane.ai/en/stable/introduction/circuits.html#quantum-functions).
  Although `qml.from_qiskit` already exists in PennyLane, we have made a number of improvements to
  make importing from Qiskit easier:

  * You can now append PennyLane measurements onto the quantum function returned by
    `qml.from_qiskit`. Consider this simple Qiskit circuit:

    ```python
    import pennylane as qml
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.rx(0.785, 0)
    qc.ry(1.57, 1)
    ```
    
    We can convert it into a PennyLane QNode in just a few lines and then add some `measurements`:

    ```pycon
    >>> dev = qml.device("default.qubit")
    >>> measurements = [qml.expval(qml.Z(0) @ qml.Z(1))]
    >>> qfunc = qml.from_qiskit(qc, measurements=measurements)
    >>> qnode = qml.QNode(qfunc, dev)
    >>> qnode()
    [tensor(0.00056331, requires_grad=True)]
    ```

  * Quantum circuits that already contain Qiskit-side measurements can be faithfully converted with
    `qml.from_qiskit`. Consider this example Qiskit circuit:

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
    @qml.qnode(dev)
    def teleport():
        m0, m1 = qml.from_qiskit(qc)()
        qml.cond(m0, qml.CZ)(2)
        return qml.density_matrix(2)
    ```
    
  * It is now more intuitive to handle parametrized Qiskit circuits. Consider the following circuit:

    ```python
    from qiskit.circuit import Parameter

    angle0 = Parameter("x")
    angle1 = Parameter("y")

    qc = QuantumCircuit(2, 2)
    qc.rx(angle0, 0)
    qc.ry(angle1, 1)
    qc.cx(1, 0)
    ```
    
    We can convert this circuit into a QNode with two arguments, corresponding to `x` and `y`:

    ```python
    measurements = qml.expval(qml.PauliZ(0))
    qfunc = qml.from_qiskit(qc, measurements)
    qnode = qml.QNode(qfunc, dev)
    ```
    
    The QNode can be evaluated and differentiated:

    ```pycon
    >>> x, y = qml.numpy.array([0.4, 0.5], requires_grad=True)
    >>> qnode(x, y)
    tensor(0.80830707, requires_grad=True)
    >>> qml.grad(qnode)(x, y)
    (tensor(-0.34174675, requires_grad=True),
     tensor(-0.44158016, requires_grad=True))
    ```

    This shows how easy it is to make a Qiskit circuit differentiable with PennyLane.

  * The `qml.from_qiskit` functionality is compatible with both Qiskit
    [1.0](https://docs.quantum.ibm.com/api/qiskit/release-notes/1.0) and earlier versions.

* In addition to circuits, it is also possible to convert operators from Qiskit to PennyLane with a new function called
   `qml.from_qiskit_op`.
  [(#5251)](https://github.com/PennyLaneAI/pennylane/pull/5251)

  A Qiskit
  [SparsePauliOp](https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.SparsePauliOp) can be
  converted to a PennyLane operator using `qml.from_qiskit_op`:

  ```pycon
  >>> from qiskit.quantum_info import SparsePauliOp
  >>> qiskit_op = SparsePauliOp(["II", "XY"])
  >>> qiskit_op
  SparsePauliOp(['II', 'XY'],
                coeffs=[1.+0.j, 1.+0.j])
  >>> pl_op = qml.from_qiskit_op(qiskit_op)
  I(0) + X(1) @ Y(0)
  ```

  Combined with `qml.from_qiskit`, it becomes easy to quickly calculate quantities like expectation
  values by converting the whole workflow to PennyLane:

  ```python
  qc = QuantumCircuit(2)  # Create circuit
  qc.rx(0.785, 0)
  qc.ry(1.57, 1)

  measurements = [qml.expval(pl_op)]  # Create QNode
  qfunc = qml.from_qiskit(qc, measurements)
  qnode = qml.QNode(qfunc, dev)
  ```
  ```pycon
  >>> qnode()  # Evaluate!
  [tensor(0.29317504, requires_grad=True)]
  ```

<h4>Native mid-circuit measurements on Default Qubit 💡</h4>

* When operating in finite-shots mode, the `default.qubit` device now performs mid-circuit
  measurements in a similar way to quantum hardware. 
  [(#5088)](https://github.com/PennyLaneAI/pennylane/pull/5088)
  [(#5120)](https://github.com/PennyLaneAI/pennylane/pull/5120)
  
  For each shot, when a mid-circuit measurement is encountered, the device evaluates the probability of 
  projecting onto `|0>` or `|1>` and makes a random choice to collapse the circuit state. This approach works 
  well when there are a lot of mid-circuit measurements and the number of shots is not too high.

  ```python
  import pennylane as qml

  dev = qml.device("default.qubit", shots=10)

  @qml.qnode(dev)
  def f():
      for i in range(1967):
          qml.Hadamard(0)
          qml.measure(0)
      return qml.sample(qml.PauliX(0))
  ```
  ```pycon
  >>> f()
  tensor([-1, -1, -1,  1,  1, -1,  1, -1,  1, -1], requires_grad=True)
  ```

  Previously, mid-circuit measurements would be automatically replaced with an additional qubit
  using the `@qml.defer_measurements` transform, so the above circuit would have required thousands
  of qubits to simulate.

<h4>Work easily and efficiently with operators 🔧</h4>

* Over the past few releases, PennyLane's approach to operator arithmetic has been overhauled.
  New classes such as `Sum` and `Prod` have been added in the
  [op_math](https://docs.pennylane.ai/en/stable/code/qml_ops_op_math.html) module, providing
  an extensive range of manipulations and ways to combine PennyLane operators. The updated operator
  arithmetic functionality can be activated using `qml.operation.enable_new_opmath()` and will
  become the default approach in the next release.

  The following updates have been made in this version of PennyLane:

  * You can now easily access Pauli operators via `I`, `X`, `Y`, and `Z`:
    [(#5116)](https://github.com/PennyLaneAI/pennylane/pull/5116)

    ```pycon
    >>> from pennylane import I, X, Y, Z
    >>> X(0)
    X(0)
    ```

  * PennyLane will try to automatically work with a Pauli representation of operators when
    available. The Pauli representation can be optionally accessed via `op.pauli_rep`:
    [(#4989)](https://github.com/PennyLaneAI/pennylane/pull/4989)
    [(#5001)](https://github.com/PennyLaneAI/pennylane/pull/5001)
    [(#5003)](https://github.com/PennyLaneAI/pennylane/pull/5003)
    [(#5017)](https://github.com/PennyLaneAI/pennylane/pull/5017)
    [(#5027)](https://github.com/PennyLaneAI/pennylane/pull/5027)

    ```pycon
    >>> op = X(0) + Y(0)
    >>> type(op.pauli_rep)
    pennylane.pauli.pauli_arithmetic.PauliSentence
    ```

    The `PauliWord` and `PauliSentence` objects in the
    [pauli](https://docs.pennylane.ai/en/stable/code/qml_pauli.html#classes) module provide an
    efficient representation and can be combined using basic arithmetic like addition, products, and
    scalar multiplication. These objects do not need to be directly handled in most workflows
    since manipulation will happen automatically in the background.

  * Extensive improvements have been made to the string representations of PennyLane operators,
    making them shorter and possible to copy as valid PennyLane code:
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

  * The `Sum` and `Prod` classes have been updated to reach feature parity with `Hamiltonian`
    and `Tensor`, respectively. This includes support for grouping via the `pauli` module:
    [(#5070)](https://github.com/PennyLaneAI/pennylane/pull/5070)
    [(#5132)](https://github.com/PennyLaneAI/pennylane/pull/5132)
    [(#5133)](https://github.com/PennyLaneAI/pennylane/pull/5133)

    ```pycon
    >>> obs = [X(0) @ Y(1), Z(0), Y(0) @ Z(1), Y(1)]
    >>> qml.pauli.group_observables(obs)
    [[Y(0) @ Z(1)], [X(0) @ Y(1), Y(1)], [Z(0)]]
    ```

  * A new `qml.commutator` function is now available that allows you to compute commutators between
    `qml.operation.Operator`, `qml.pauli.PauliWord` and `qml.pauli.PauliSentence` instances.
    [(#5051)](https://github.com/PennyLaneAI/pennylane/pull/5051)
    [(#5052)](https://github.com/PennyLaneAI/pennylane/pull/5052)
    [(#5098)](https://github.com/PennyLaneAI/pennylane/pull/5098)

    ```pycon
    >>> qml.commutator(X(0), Y(0))
    2j * Z(0)
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
  import pennylane as qml

  dev = qml.device("default.clifford", tableau=True)
  @qml.qnode(dev)
  def circuit():
      qml.CNOT(wires=[0, 1])
      qml.PauliX(wires=[1])
      qml.ISWAP(wires=[0, 1])
      qml.Hadamard(wires=[0])
      return qml.state()
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

* Adjoint device VJP's are now supported with `jax.jacobian`. `device_vjp=True` is
  now strictly faster for jax.
  [(#4963)](https://github.com/PennyLaneAI/pennylane/pull/4963)

* PennyLane can now use lightning-provided VJPs by selecting `device_vjp=True` on the QNode.
  [(#4914)](https://github.com/PennyLaneAI/pennylane/pull/4914)

* `device_vjp` can now be used with normal Tensorflow. Support has not yet been added
  for `tf.Function` and Tensorflow Autograph.
  [(#4676)](https://github.com/PennyLaneAI/pennylane/pull/4676)

* Queueing has been removed (`AnnotatedQueue`) from `qml.cut_circuit` and `qml.cut_circuit_mc` to improve performance
  for large workflows.
  [(#5108)](https://github.com/PennyLaneAI/pennylane/pull/5108)

* The performance of circuit-cutting workloads with large numbers of generated tapes have been improved.
  [(#5005)](https://github.com/PennyLaneAI/pennylane/pull/5005)

* Measuring `qml.probs`is now faster due to an optimization in `_samples_to_counts`.
  [(#5145)](https://github.com/PennyLaneAI/pennylane/pull/5145)

<h4>Community contributions 🥳</h4>

* A new function called `qml.fermi.parity_transform` has been added for parity mapping of a fermionic Hamiltonian.
  [(#4928)](https://github.com/PennyLaneAI/pennylane/pull/4928)
  It is now possible to transform a fermionic Hamiltonian to a qubit Hamiltonian with parity mapping.

  ```python
  import pennylane as qml
  fermi_ham = qml.fermi.FermiWord({(0, 0) : '+', (1, 1) : '-'})

  qubit_ham = qml.fermi.parity_transform(fermi_ham, n=6)
  ```

  ```pycon
  >>> print(qubit_ham)
  (-0.25j*(PauliY(wires=[0]))) + ((-0.25+0j)*(PauliX(wires=[0]) @ PauliZ(wires=[1]))) +
  ((0.25+0j)*(PauliX(wires=[0]))) + (0.25j*(PauliY(wires=[0]) @ PauliZ(wires=[1])))
  ```

* The transform `split_non_commuting` now accepts measurements of type `probs`, `sample` and `counts` which accept both wires and observables.
  [(#4972)](https://github.com/PennyLaneAI/pennylane/pull/4972)

* The efficiency of matrix calculations when an operator is symmetric over a given set of wires has been improved.
  [(#3601)](https://github.com/PennyLaneAI/pennylane/pull/3601)

* The module `pennylane/math/quantum.py` now has support for the min-entropy.
  [(#3959)](https://github.com/PennyLaneAI/pennylane/pull/3959/)

* A function called `apply_operation` that applies operations to device-compatible states has been added to the new `qutrit_mixed` module found in `qml.devices`.
  [(#5032)](https://github.com/PennyLaneAI/pennylane/pull/5032)

* A function called `measure` has been added to the new `qutrit_mixed` module found in `qml.devices` that measures device-compatible states for a collection of measurement processes.
  [(#5049)](https://github.com/PennyLaneAI/pennylane/pull/5049)

* A function called `apply_operation` has been added to the new `qutrit_mixed` module found in `qml.devices` that applies operations to device-compatible states.
  [(#5032)](https://github.com/PennyLaneAI/pennylane/pull/5032)

* A `partial_trace` function has been added to `qml.math` for taking the partial trace of matrices.
  [(#5152)](https://github.com/PennyLaneAI/pennylane/pull/5152)

<h4>Other operator arithmetic improvements</h4>

* The following capabilities have been added for Pauli arithmetic:
  [(#4989)](https://github.com/PennyLaneAI/pennylane/pull/4989)
  [(#5001)](https://github.com/PennyLaneAI/pennylane/pull/5001)
  [(#5003)](https://github.com/PennyLaneAI/pennylane/pull/5003)
  [(#5017)](https://github.com/PennyLaneAI/pennylane/pull/5017)
  [(#5027)](https://github.com/PennyLaneAI/pennylane/pull/5027)
  [(#5018)](https://github.com/PennyLaneAI/pennylane/pull/5018)

  * You can now multiply `PauliWord` and `PauliSentence` instances by scalars, e.g.
    `0.5 * PauliWord({0:"X"})` or `0.5 * PauliSentence({PauliWord({0:"X"}): 1.})`.

  * You can now intuitively add together
    `PauliWord` and `PauliSentence` as well as scalars, which are treated implicitly as identities.
    For example, `ps1 + pw1 + 1.` for some Pauli word `pw1 = PauliWord({0: "X", 1: "Y"})` and Pauli
    sentence `ps1 = PauliSentence({pw1: 3.})`.

  * You can now subtract `PauliWord` and `PauliSentence` instances, as well as scalars, from each
    other. For example `ps1 - pw1 - 1`.

  * You can now also use `qml.dot` with `PauliWord`, `PauliSentence` and operators, e.g.
    `qml.dot([0.5, -1.5, 2], [pw1, ps1, id_word])` with `id_word = PauliWord({})`.

  * `qml.matrix` now accepts `PauliWord` and `PauliSentence` instances,
    `qml.matrix(PauliWord({0:"X"}))`.

  * It is possible to compute commutators with Pauli operators natively with the
    `PauliSentence.commutator` method.

    ```pycon
    >>> op1 = PauliWord({0: "X", 1: "X"})
    >>> op2 = PauliWord({0: "Y"}) + PauliWord({1: "Y"})
    >>> op1.commutator(op2)
    2j * Z(0) @ X(1)
    + 2j * X(0) @ Z(1)
    ```

* Composite operations (e.g., those made with `qml.prod` and `qml.sum`) and `SProd` operations
  convert `Hamiltonian` and `Tensor` operands to `Sum` and `Prod` types, respectively. This helps
  avoid the mixing of incompatible operator types.
  [(#5031)](https://github.com/PennyLaneAI/pennylane/pull/5031)
  [(#5063)](https://github.com/PennyLaneAI/pennylane/pull/5063)

* `qml.Identity()` can be initialized without wires. Measuring it is currently not possible though.
  [(#5106)](https://github.com/PennyLaneAI/pennylane/pull/5106)

* `qml.dot` now returns a `Sum` class even when all the coefficients match.
  [(#5143)](https://github.com/PennyLaneAI/pennylane/pull/5143)

* `qml.pauli.group_observables` now supports grouping `Prod` and `SProd` operators.
  [(#5070)](https://github.com/PennyLaneAI/pennylane/pull/5070)

* Cuts down on performance bottlenecks in converting a `PauliSentence` to a `Sum`.
  [(#5141)](https://github.com/PennyLaneAI/pennylane/pull/5141)
  [(#5150)](https://github.com/PennyLaneAI/pennylane/pull/5150)

* Upgrade the `Prod.terms()` method to return a tuple `(coeffs, ops)` consisting of coefficients and
  pure product operators.
  [(#5132)](https://github.com/PennyLaneAI/pennylane/pull/5132)

  ```python
  >>> qml.operation.enable_new_opmath()
  >>> op = X(0) @ (0.5 * X(1) + X(2))
  >>> op.terms()
  ([0.5, 1.0],
   [X(1) @ X(0),
    X(2) @ X(0)])
  ```

* Upgrade the `Sum.terms()` method to return a tuple `(coeffs, ops)` consisting of coefficients and
  pure product operators.
  [(#5133)](https://github.com/PennyLaneAI/pennylane/pull/5133)

  ```python
  >>> qml.operation.enable_new_opmath()
  >>> op = 0.5 * X(0) + 0.7 * X(1) + 1.5 * Y(0) @ Y(1)
  >>> op.terms()
  ([0.5, 0.7, 1.5],
   [X(0), X(1), Y(1) @ Y(0)])
  ```

* `Sum.ops`, `Sum.coeffs`, `Prod.ops`, `Prod.coeffs` have been added for feature parity with `qml.Hamiltonian` but will be deprecated in the future.
  [(#5164)](https://github.com/PennyLaneAI/pennylane/pull/5164)

* String representations of `ParametrizedHamiltonian` have been updated to match the style of other PL operators.
  [(#5215)](https://github.com/PennyLaneAI/pennylane/pull/5215)

<h4>Other improvements</h4>

* The `pl-device-test` suite is now compatible with the `qml.devices.Device` interface.
  [(#5229)](https://github.com/PennyLaneAI/pennylane/pull/5229)

* The `QSVT` operation now determines its `data` from the block encoding and projector operator data.
  [(#5226)](https://github.com/PennyLaneAI/pennylane/pull/5226)
  [(#5248)](https://github.com/PennyLaneAI/pennylane/pull/5248)

* Ensure the `BlockEncode` operator is JIT-compatible with JAX.
  [(#5110)](https://github.com/PennyLaneAI/pennylane/pull/5110)

* The `qml.qsvt` function uses `qml.GlobalPhase` instead of `qml.exp` to define global phase.
  [(#5105)](https://github.com/PennyLaneAI/pennylane/pull/5105)

* Update `tests/ops/functions/conftest.py` to ensure all operator types are tested for validity.
  [(#4978)](https://github.com/PennyLaneAI/pennylane/pull/4978)

* A new `pennylane.workflow` module is added. This module now contains `qnode.py`, `execution.py`, `set_shots.py`, `jacobian_products.py`, and the submodule `interfaces`.
  [(#5023)](https://github.com/PennyLaneAI/pennylane/pull/5023)

* Raise a more informative error when calling `adjoint_jacobian` with trainable state-prep operations.
  [(#5026)](https://github.com/PennyLaneAI/pennylane/pull/5026)

* Adds `qml.workflow.get_transform_program` and `qml.workflow.construct_batch` to inspect the transform program and batch of tapes
  at different stages.
  [(#5084)](https://github.com/PennyLaneAI/pennylane/pull/5084)

* All custom controlled operations such as `CRX`, `CZ`, `CNOT`, `ControlledPhaseShift` now inherit from `ControlledOp`, giving them additional properties such as `control_wire` and `control_values`. Calling `qml.ctrl` on `RX`, `RY`, `RZ`, `Rot`, and `PhaseShift` with a single control wire will return gates of types `CRX`, `CRY`, etc. as opposed to a general `Controlled` operator.
  [(#5069)](https://github.com/PennyLaneAI/pennylane/pull/5069)
  [(#5199)](https://github.com/PennyLaneAI/pennylane/pull/5199)

* CI will now fail if coverage data fails to upload to codecov. Previously, it would silently pass
  and the codecov check itself would never execute.
  [(#5101)](https://github.com/PennyLaneAI/pennylane/pull/5101)

* `qml.ctrl` called on operators with custom controlled versions will now return instances
  of the custom class, and it will flatten nested controlled operators to a single
  multi-controlled operation. For `PauliX`, `CNOT`, `Toffoli`, and `MultiControlledX`,
  calling `qml.ctrl` will always resolve to the best option in `CNOT`, `Toffoli`, or
  `MultiControlledX` depending on the number of control wires and control values.
  [(#5125)](https://github.com/PennyLaneAI/pennylane/pull/5125/)

* Unwanted warning filters have been removed from tests and no `PennyLaneDeprecationWarning`s 
  are being raised unexpectedly.
  [(#5122)](https://github.com/PennyLaneAI/pennylane/pull/5122)

* Added new error tracking and propagation functionality.
  [(#5115)](https://github.com/PennyLaneAI/pennylane/pull/5115)
  [(#5121)](https://github.com/PennyLaneAI/pennylane/pull/5121)

* Replacing `map_batch_transform` in the source code with the method `_batch_transform`
  implemented in `TransformDispatcher`.
  [(#5212)](https://github.com/PennyLaneAI/pennylane/pull/5212)

* `TransformDispatcher` can now dispatch onto a batch of tapes, so that it is easier to compose transforms
  when working in the tape paradigm.
  [(#5163)](https://github.com/PennyLaneAI/pennylane/pull/5163)

* `qml.ctrl` is now a simple wrapper that either calls PennyLane's built in `create_controlled_op`
  or uses the Catalyst implementation.
  [(#5247)](https://github.com/PennyLaneAI/pennylane/pull/5247)

* Controlled composite operations can now be decomposed using ZYZ rotations.
  [(#5242)](https://github.com/PennyLaneAI/pennylane/pull/5242)

* New functions called `qml.devices.modifiers.simulator_tracking` and `qml.devices.modifiers.single_tape_support` have been added
  to add basic default behavior onto a device class.
  [(#5200)](https://github.com/PennyLaneAI/pennylane/pull/5200)

<h3>Breaking changes 💔</h3>

* Passing additional arguments to a transform that decorates a QNode must now be done through the use
  of `functools.partial`.
  [(#5046)](https://github.com/PennyLaneAI/pennylane/pull/5046)

* `qml.ExpvalCost` has been removed. Users should use `qml.expval()` moving forward.
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

* Make PennyLane code compatible with the latest version of `black`.
  [(#5112)](https://github.com/PennyLaneAI/pennylane/pull/5112)
  [(#5119)](https://github.com/PennyLaneAI/pennylane/pull/5119)

* `gradient_analysis_and_validation` is now renamed to `find_and_validate_gradient_methods`. Instead of returning a list, it now returns a dictionary of gradient methods for each parameter index, and no longer mutates the tape.
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
  (with potentially negative eigenvalues) has been implemented.
  [(#5048)](https://github.com/PennyLaneAI/pennylane/pull/5048)

* Controlled operators with a custom controlled version decomposes like how their controlled
  counterpart decomposes, as opposed to decomposing into their controlled version.
  [(#5069)](https://github.com/PennyLaneAI/pennylane/pull/5069)
  [(#5125)](https://github.com/PennyLaneAI/pennylane/pull/5125/)

  For example:

  ```pycon
  >>> qml.ctrl(qml.RX(0.123, wires=1), control=0).decomposition()
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

* `qml.transforms.one_qubit_decomposition` and `qml.transforms.two_qubit_decomposition` are removed. Instead,
  you should use `qml.ops.one_qubit_decomposition` and `qml.ops.two_qubit_decomposition`.
  [(#5091)](https://github.com/PennyLaneAI/pennylane/pull/5091)

<h3>Deprecations 👋</h3>

* Calling `qml.matrix` without providing a `wire_order` on objects where the wire order could be
  ambiguous now raises a warning. In the future, the `wire_order` argument will be required in
  these cases.
  [(#5039)](https://github.com/PennyLaneAI/pennylane/pull/5039)

* `Operator.validate_subspace(subspace)` has been relocated to the `qml.ops.qutrit.parametric_ops`
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

* `qml.pauli.pauli_mult` and `qml.pauli.pauli_mult_with_phase` are now deprecated. Instead, you
  should use `qml.simplify(qml.prod(pauli_1, pauli_2))` to get the reduced operator.
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

* A typo in a code example in the `qml.transforms` API has been fixed.
  [(#5014)](https://github.com/PennyLaneAI/pennylane/pull/5014)

* Documentation `qml.data` has been updated and now mentions a way to access the same dataset simultaneously from multiple environments.
  [(#5029)](https://github.com/PennyLaneAI/pennylane/pull/5029)

* Clarification for the definition of `argnum` added to gradient methods
  [(#5035)](https://github.com/PennyLaneAI/pennylane/pull/5035)

* A typo in the code example for `qml.qchem.dipole_of` has been fixed.
  [(#5036)](https://github.com/PennyLaneAI/pennylane/pull/5036)

* Added a development guide on deprecations and removals.
  [(#5083)](https://github.com/PennyLaneAI/pennylane/pull/5083)

* A note about the eigenspectrum of second-quantized Hamiltonians added to `qml.eigvals`.
  [(#5095)](https://github.com/PennyLaneAI/pennylane/pull/5095)

* A warning about two mathematically equivalent Hamiltonians undergoing different time evolutions was added to `qml.TrotterProduct` and `qml.ApproxTimeEvolution`.
  [(#5137)](https://github.com/PennyLaneAI/pennylane/pull/5137)

* Added a reference to the paper that provides the image of the `qml.QAOAEmbedding` template.
  [(#5130)](https://github.com/PennyLaneAI/pennylane/pull/5130)

* The docstring of `qml.sample` has been updated to advise the use of single-shot expectations
  instead when differentiating a circuit.
  [(#5237)](https://github.com/PennyLaneAI/pennylane/pull/5237)

<h3>Bug fixes 🐛</h3>

* `ctrl_decomp_zyz` is now differentiable.
  [(#5198)](https://github.com/PennyLaneAI/pennylane/pull/5198)

* `qml.ops.Pow.matrix()` is now differentiable with TensorFlow with integer exponents.
  [(#5178)](https://github.com/PennyLaneAI/pennylane/pull/5178)

* The `qml.MottonenStatePreparation` template is updated to include a global phase operation.
  [(#5166)](https://github.com/PennyLaneAI/pennylane/pull/5166)

* Fixes a queuing bug when using `qml.prod` with a qfunc that queues a single operator.
  [(#5170)](https://github.com/PennyLaneAI/pennylane/pull/5170)

* The `qml.TrotterProduct` template is updated to accept `SProd` as input Hamiltonian.
  [(#5073)](https://github.com/PennyLaneAI/pennylane/pull/5073)

* Fixed a bug where caching together with JIT compilation and broadcasted tapes yielded wrong results
  `Operator.hash` now depends on the memory location, `id`, of a Jax tracer instead of its string representation.
  [(#3917)](https://github.com/PennyLaneAI/pennylane/pull/3917)

* `qml.transforms.undo_swaps` can now work with operators with hyperparameters or nesting.
  [(#5081)](https://github.com/PennyLaneAI/pennylane/pull/5081)

* `qml.transforms.split_non_commuting` will now pass the original shots along.
  [(#5081)](https://github.com/PennyLaneAI/pennylane/pull/5081)

* If `argnum` is provided to a gradient transform, only the parameters specified in `argnum` will have their gradient methods validated.
  [(#5035)](https://github.com/PennyLaneAI/pennylane/pull/5035)

* `StatePrep` operations expanded onto more wires are now compatible with backprop.
  [(#5028)](https://github.com/PennyLaneAI/pennylane/pull/5028)

* `qml.equal` works well with `qml.Sum` operators when wire labels are a mix of integers and strings.
  [(#5037)](https://github.com/PennyLaneAI/pennylane/pull/5037)

* The return value of `Controlled.generator` now contains a projector that projects onto the correct subspace based on the control value specified.
  [(#5068)](https://github.com/PennyLaneAI/pennylane/pull/5068)

* `CosineWindow` no longer raises an unexpected error when used on a subset of wires at the beginning of a circuit.
  [(#5080)](https://github.com/PennyLaneAI/pennylane/pull/5080)

* Ensure `tf.function` works with `TensorSpec(shape=None)` by skipping batch size computation.
  [(#5089)](https://github.com/PennyLaneAI/pennylane/pull/5089)

* `PauliSentence.wires` no longer imposes a false order.
  [(#5041)](https://github.com/PennyLaneAI/pennylane/pull/5041)

* `qml.qchem.import_state` now applies the chemist-to-physicist
  sign convention when initializing a PennyLane state vector from
  classically pre-computed wavefunctions. That is, it interleaves
  spin-up/spin-down operators for the same spatial orbital index,
  as standard in PennyLane (instead of commuting all spin-up
  operators to the left, as is standard in quantum chemistry).
  [(#5114)](https://github.com/PennyLaneAI/pennylane/pull/5114)

* Multi-wire controlled `CNOT` and `PhaseShift` can now be decomposed correctly.
  [(#5125)](https://github.com/PennyLaneAI/pennylane/pull/5125/)
  [(#5148)](https://github.com/PennyLaneAI/pennylane/pull/5148)

* `draw_mpl` no longer raises an error when drawing a circuit containing an adjoint of a controlled operation.
  [(#5149)](https://github.com/PennyLaneAI/pennylane/pull/5149)

* `default.mixed` no longer throws `ValueError` when applying a state vector that is not of type `complex128` when used with tensorflow.
  [(#5155)](https://github.com/PennyLaneAI/pennylane/pull/5155)

* `ctrl_decomp_zyz` no longer raises a `TypeError` if the rotation parameters are of type `torch.Tensor`
  [(#5183)](https://github.com/PennyLaneAI/pennylane/pull/5183)

* Comparing `Prod` and `Sum` objects now works regardless of nested structure with `qml.equal` if the
  operators have a valid `pauli_rep` property.
  [(#5177)](https://github.com/PennyLaneAI/pennylane/pull/5177)

* Controlled `GlobalPhase` with non-zero control wire no longer throws an error.
  [(#5194)](https://github.com/PennyLaneAI/pennylane/pull/5194)

* A `QNode` transformed with `mitigate_with_zne` now accepts batch parameters.
  [(#5195)](https://github.com/PennyLaneAI/pennylane/pull/5195)

* The matrix of an empty `PauliSentence` instance is now correct (all-zeros).
  Further, matrices of empty `PauliWord` and `PauliSentence` instances can be turned to matrices now.
  [(#5188)](https://github.com/PennyLaneAI/pennylane/pull/5188)

* `PauliSentence.__matmul__` can handle `PauliWord` instances now.
  [(#5208)](https://github.com/PennyLaneAI/pennylane/pull/5208)

* Make `CompositeOp.eigendecomposition` jit-compatible.
  [(#5207)](https://github.com/PennyLaneAI/pennylane/pull/5207)

* `QubitDensityMatrix` now works with jax-jit on the `default.mixed` device.
  [(#5203)](https://github.com/PennyLaneAI/pennylane/pull/5203)
  [(#5236)](https://github.com/PennyLaneAI/pennylane/pull/5236)

* When a QNode specifies `diff_method="adjoint"`, `default.qubit` no longer tries to decompose non-trainable operations with non-scalar parameters such as `QubitUnitary`.
  [(#5233)](https://github.com/PennyLaneAI/pennylane/pull/5233)

* The overwriting of the class names of `I`, `X`, `Y`, and `Z` no longer happens in the init after causing problems with datasets. Now happens globally.
  [(#5252)](https://github.com/PennyLaneAI/pennylane/pull/5252)

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
