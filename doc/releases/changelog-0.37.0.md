
# Release 0.37.0

<h3>New features since last release</h3>

<h4>Execute wide circuits with Default Tensor 🔗</h4>

* A new `default.tensor` device is now available for performing
  [tensor network](https://en.wikipedia.org/wiki/Tensor_network) and
  [matrix product state](https://en.wikipedia.org/wiki/Matrix_product_state) simulations
  of quantum circuits using the
  [quimb backend](https://quimb.readthedocs.io/en/latest/).
  [(#5699)](https://github.com/PennyLaneAI/pennylane/pull/5699)
  [(#5744)](https://github.com/PennyLaneAI/pennylane/pull/5744)
  [(#5786)](https://github.com/PennyLaneAI/pennylane/pull/5786)
  [(#5795)](https://github.com/PennyLaneAI/pennylane/pull/5795)

  Either method can be selected when instantiating the `default.tensor` device by setting the
  `method` keyword argument to `"tn"` (tensor network) or `"mps"` (matrix product state).

  There are
  [several templates in PennyLane](https://docs.pennylane.ai/en/stable/introduction/templates.html#tensor-networks)
  that are tensor-network focused, which are excellent candidates for the `"tn"` method for
  `default.tensor`. The following example shows how a circuit comprising gates in a tree tensor
  network architecture can be efficiently simulated using `method="tn"`.

  ```python
  import pennylane as qp

  n_wires = 16
  dev = qml.device("default.tensor", method="tn")

  def block(weights, wires):
      qml.CNOT(wires=[wires[0], wires[1]])
      qml.RY(weights[0], wires=wires[0])
      qml.RY(weights[1], wires=wires[1])

  n_block_wires = 2
  n_params_block = 2
  n_blocks = qml.TTN.get_n_blocks(range(n_wires), n_block_wires)
  template_weights = [[0.1, -0.3]] * n_blocks

  @qml.qnode(dev)
  def circuit(template_weights):
      for i in range(n_wires):
          qml.Hadamard(i)
      qml.TTN(range(n_wires), n_block_wires, block, n_params_block, template_weights)
      return qml.expval(qml.Z(n_wires - 1))
  ```

  ```pycon
  >>> circuit(template_weights)
  0.3839174759751649
  ```

  For matrix product state simulations (`method="mps"`), we can make the execution be approximate by setting `max_bond_dim` (see the
  [device's documentation](https://docs.pennylane.ai/en/stable/code/api/pennylane.devices.default_tensor.DefaultTensor.html)
  for more details).
  The maximum bond dimension has implications for the speed of the simulation and lets us control the degree of the approximation, as shown in the
  following example. First, set up the circuit:

  ```python
  import numpy as np

  n_layers = 10
  n_wires = 10

  initial_shape, weights_shape = qml.SimplifiedTwoDesign.shape(n_layers, n_wires)
  np.random.seed(1967)
  initial_layer_weights = np.random.random(initial_shape)
  weights = np.random.random(weights_shape)

  def f():
      qml.SimplifiedTwoDesign(initial_layer_weights, weights, range(n_wires))
      return qml.expval(qml.Z(0))
  ```

  The `default.tensor` device is instantiated with a `max_bond_dim` value:

  ```python
  dev_dq = qml.device("default.qubit")
  value_dq = qml.QNode(f, dev_dq)()

  dev_mps = qml.device("default.tensor", max_bond_dim=5)
  value_mps = qml.QNode(f, dev_mps)()
  ```

  With this bond dimension, the expectation values calculated for `default.qubit` and
  `default.tensor` are different:

  ```pycon
  >>> np.abs(value_dq - value_mps)
  tensor(0.0253213, requires_grad=True)
  ```

  Learn more about `default.tensor` and how to configure it by 
  [visiting the how-to guide](https://pennylane.ai/qml/demos/tutorial_How_to_simulate_quantum_circuits_with_tensor_networks/).

<h4>Add noise models to your quantum circuits 📺</h4>

* Support for building noise models and applying them to a quantum circuit has been added
via the `NoiseModel` class and an `add_noise` transform.
  [(#5674)](https://github.com/PennyLaneAI/pennylane/pull/5674)
  [(#5684)](https://github.com/PennyLaneAI/pennylane/pull/5684)
  [(#5718)](https://github.com/PennyLaneAI/pennylane/pull/5718)

  Under the hood, PennyLane's approach to noise models is insertion-based, meaning that noise is included
  by *inserting* additional operators (gates or channels) that describe the noise into the quantum circuit.
  Creating a `NoiseModel` boils down to defining Boolean conditions under which specific noisy operations 
  are inserted. There are several ways to specify conditions for adding noisy operations: 

  * `qml.noise.op_eq(op)`: if the operator `op` is encountered in the circuit, add noise.
  * `qml.noise.op_in(ops)`: if any operators in `ops` are encountered in the circuit, add noise.
  * `qml.noise.wires_eq(wires)`: if an operator is applied to `wires`, add noise.
  * `qml.noise.wires_in(wires)`: if an operator is applied to any wire in `wires`, add noise.
  * custom noise conditions: custom conditions can be defined as functions decorated with `qml.BooleanFn` that return a Boolean value. For example, the following function will insert noise if a `qml.RY` operator is encountered with an angle of rotation that is less than `0.5`:

    ```python
    @qml.BooleanFn
    def c0(op):
        return isinstance(op, qml.RY) and op.parameters[0] < 0.5
    ```

  Conditions can also be combined together with `&`, `and`, `|`, etc.
  Once the conditions under which noise is to be inserted have been stated, we can specify exactly what 
  noise is inserted with the following:

  * `qml.noise.partial_wires(op)`: insert `op` on the wires that are specified by the condition that triggers adding this noise
  * custom noise operations: custom noise can be specified by defining a standard quantum function like below.
  
    ```python
    def n0(op, **kwargs):
        qml.RY(op.parameters[0] * 0.05, wires=op.wires)
    ```

  With that, we can create a `qml.NoiseModel` object whose argument must be a dictionary mapping conditions
  to noise:

  ```python
  c1 = qml.noise.op_eq(qml.X) & qml.noise.wires_in([0, 1])
  n1 = qml.noise.partial_wires(qml.AmplitudeDamping, 0.4)

  noise_model = qml.NoiseModel({c0: n0, c1: n1})
  ```

  ```pycon
  >>> noise_model
  NoiseModel({
      BooleanFn(c0): n0
      OpEq(PauliX) | WiresIn([0, 1]): AmplitudeDamping(gamma=0.4)
  })
  ```

  The noise model created can then be added to a QNode with `qml.add_noise`:

  ```python
  dev = qml.device("lightning.qubit", wires=3)

  @qml.qnode(dev)
  def circuit():
      qml.Y(0)
      qml.CNOT([0, 1])
      qml.RY(0.3, wires=2) # triggers c0
      qml.X(1) # triggers c1
      return qml.state()
  ```

  ```pycon
  >>> print(qml.draw(circuit)())
  0: ──Y────────╭●────┤  State
  1: ───────────╰X──X─┤  State
  2: ──RY(0.30)───────┤  State
  >>> circuit = qml.add_noise(circuit, noise_model)
  >>> print(qml.draw(circuit)())
  0: ──Y────────╭●───────────────────────────────────┤  State
  1: ───────────╰X─────────X──AmplitudeDamping(0.40)─┤  State
  2: ──RY(0.30)──RY(0.01)────────────────────────────┤  State
  ```

  If more than one transform is applied to a QNode, control over when/where the `add_noise` transform is applied 
  in relation to the other transforms can be specified with the `level` keyword argument. By default, `add_noise` is applied
  after all the transforms that have been manually applied to the QNode until that point.
  To learn more about this new functionality, check out our [noise module documentation](https://docs.pennylane.ai/en/stable/code/qml_noise.html)
  and keep your eyes peeled for an in-depth demo!

<h4>Catch bugs with the PennyLane debugger 🚫🐞</h4>

* The new PennyLane quantum debugger allows pausing simulation via the `qml.breakpoint()` command and provides tools for 
  analyzing quantum circuits during execution.
  [(#5680)](https://github.com/PennyLaneAI/pennylane/pull/5680)
  [(#5749)](https://github.com/PennyLaneAI/pennylane/pull/5749)
  [(#5789)](https://github.com/PennyLaneAI/pennylane/pull/5789)
  
  This includes monitoring the circuit via measurements using `qml.debug_state()`, `qml.debug_probs()`, 
  `qml.debug_expval()`, and `qml.debug_tape()`, stepping through the operations in a quantum circuit, and interactively
  adding operations during execution.

  Including `qml.breakpoint()` in a circuit will cause the simulation to pause during execution and bring up the interactive console.
  For example, consider the following code in a Python file called `script.py`:

  ```python
  @qml.qnode(qml.device('default.qubit', wires=(0,1,2)))
  def circuit(x):
      qml.Hadamard(wires=0)
      qml.CNOT(wires=(0,2))
      qml.breakpoint()

      qml.RX(x, wires=1)
      qml.RY(x, wires=2)
      qml.breakpoint()

      return qml.sample()

  circuit(1.2345)
  ```

  Upon executing `script.py`, the simulation pauses at the first breakpoint:

  ```pycon
  > /Users/your/path/to/script.py(8)circuit()
  -> qml.RX(x, wires=1)
  (pldb):
  ```

  While debugging, we can access circuit information.
  For example, `qml.debug_tape()` returns the tape of the circuit, giving access to its operations and drawing:

  ```pycon
  [pldb] tape = qml.debug_tape()
  [pldb] print(tape.draw(wire_order=[0,1,2]))
  0: ──H─╭●─┤  
  2: ────╰X─┤  
  [pldb] tape.operations
  [Hadamard(wires=[0]), CNOT(wires=[0, 2])]
  ```

  While `qml.debug_state()` is equivalent to `qml.state()` and gives the current state:

  ```pycon
  [pldb] print(qml.debug_state())
  [0.70710678+0.j 0.        +0.j 0.        +0.j 0.        +0.j
    1.        +0.j 0.70710678+0.j 0.        +0.j 0.        +0.j]
  ```

  Other debugger functions like `qml.debug_probs()` and `qml.debug_expval()` also function like their simulation counterparts (`qml.probs` and `qml.expval`, respectively) and are described in more detail in the [debugger documentation](https://docs.pennylane.ai/en/stable/code/qml_debugging.html)
  
  Additionally, standard debugging commands are available to navigate through code, including `list`, `longlist`, `next`, `continue`, and `quit`, as described in [the debugging documentation](https://docs.pennylane.ai/en/stable/code/qml_debugging.html#controlling-code-execution-in-the-debugging-context).
  
  Finally, to modify a circuit mid-run, simply call the desired PennyLane operations:

  ```pycon
  [pldb] qml.CNOT(wires=(0,2))
  CNOT(wires=[0, 2])
  [pldb] print(qml.debug_tape().draw(wire_order=[0,1,2]))
  0: ──H─╭●─╭●─┤  
  2: ────╰X─╰X─┤  
  ```

Stay tuned for an in-depth demonstration on using this feature with real-world examples!
<h4>Convert between OpenFermion and PennyLane 🤝</h4>

* Two new functions called `qml.from_openfermion` and `qml.to_openfermion` are now available to convert between 
  OpenFermion and PennyLane objects. This includes both fermionic and qubit operators.
  [(#5773)](https://github.com/PennyLaneAI/pennylane/pull/5773)
  [(#5808)](https://github.com/PennyLaneAI/pennylane/pull/5808)
  [(#5881)](https://github.com/PennyLaneAI/pennylane/pull/5881)
  
  For fermionic operators:

  ```pycon
  >>> import openfermion
  >>> of_fermionic = openfermion.FermionOperator('0^ 2')
  >>> type(of_fermionic)
  <class 'openfermion.ops.operators.fermion_operator.FermionOperator'>
  >>> pl_fermionic = qml.from_openfermion(of_fermionic)
  >>> type(pl_fermionic)
  <class 'pennylane.fermi.fermionic.FermiWord'>
  >>> print(pl_fermionic)
  a⁺(0) a(2)
  ```

  And for qubit operators:

  ```pycon
  >>> of_qubit = 0.5 * openfermion.QubitOperator('X0 X5')
  >>> pl_qubit = qml.from_openfermion(of_qubit)
  >>> print(pl_qubit)
  0.5 * (X(0) @ X(5))
  ```

<h4>Better control over when drawing and specs take place 🎚️</h4>

* It is now possible to control the stage at which `qml.draw`, `qml.draw_mpl`, and `qml.specs` occur
  within a QNode's
  [transform](https://docs.pennylane.ai/en/stable/code/qml_transforms.html) program.
  [(#5855)](https://github.com/PennyLaneAI/pennylane/pull/5855)
  [(#5781)](https://github.com/PennyLaneAI/pennylane/pull/5781/)

  Consider the following circuit which has multiple transforms applied:

  ```python
  @qml.transforms.split_non_commuting
  @qml.transforms.cancel_inverses
  @qml.transforms.merge_rotations
  @qml.qnode(qml.device("default.qubit"))
  def f():
      qml.Hadamard(0)
      qml.Y(0)
      qml.RX(0.4, 0)
      qml.RX(-0.4, 0)
      qml.Y(0)
      return qml.expval(qml.X(0) + 2 * qml.Y(0))
  ```

  We can specify a `level` value when using `qml.draw()`:

  ```pycon
  >>> print(qml.draw(f, level=0)())  # input program
  0: ──H──Y──RX(0.40)──RX(-0.40)──Y─┤  <X+(2.00*Y)>
  >>> print(qml.draw(f, level=1)())  # rotations merged
  0: ──H──Y──Y─┤  <X+(2.00*Y)>
  >>>  print(qml.draw(f, level=2)())  # inverses cancelled
  0: ──H─┤  <X+(2.00*Y)>
  >>>  print(qml.draw(f, level=3)())  # Hamiltonian expanded
  0: ──H─┤  <X>

  0: ──H─┤  <Y>
  ```

  The
  [qml.workflow.get_transform_program function](https://docs.pennylane.ai/en/latest/code/api/pennylane.workflow.get_transform_program.html)
  can be used to see the full transform program.

  ```pycon
  >>> qml.workflow.get_transform_program(f)
  TransformProgram(merge_rotations, cancel_inverses, split_non_commuting, validate_device_wires, mid_circuit_measurements, decompose, validate_measurements, validate_observables, no_sampling)
  ```

  Note that additional transforms can be added automatically from device preprocessing or gradient
  calculations. Rather than providing an integer value to `level`, it is possible to target
  the `"user"`, `"gradient"` or `"device"` stages:

  ```python
  n_wires = 3
  x = np.random.random((2, n_wires))

  @qml.qnode(qml.device("default.qubit"))
  def f():
      qml.BasicEntanglerLayers(x, range(n_wires))
      return qml.expval(qml.X(0))
  ```

  ```pycon
  >>> print(qml.draw(f, level="device")())
  0: ──RX(0.28)─╭●────╭X──RX(0.70)─╭●────╭X─┤  <X>
  1: ──RX(0.52)─╰X─╭●─│───RX(0.65)─╰X─╭●─│──┤     
  2: ──RX(0.00)────╰X─╰●──RX(0.03)────╰X─╰●─┤     
  ```

<h3>Improvements 🛠</h3>

<h4>Community contributions, including UnitaryHACK 💛</h4>

* `default.clifford` now supports arbitrary state-based measurements with `qml.Snapshot`.
  [(#5794)](https://github.com/PennyLaneAI/pennylane/pull/5794)

* `qml.equal` now properly handles `Pow`, `Adjoint`, `Exp`, and `SProd` operators as arguments across 
  different interfaces and tolerances with the addition of four new keyword arguments: `check_interface`, 
  `check_trainability`, `atol` and `rtol`.
  [(#5668)](https://github.com/PennyLaneAI/pennylane/pull/5668)
  
  
* The implementation for `qml.assert_equal` has been updated for `Operator`, `Controlled`, `Adjoint`, 
  `Pow`, `Exp`, `SProd`, `ControlledSequence`, `Prod`, `Sum`, `Tensor` and `Hamiltonian` instances.
  [(#5780)](https://github.com/PennyLaneAI/pennylane/pull/5780)
  [(#5877)](https://github.com/PennyLaneAI/pennylane/pull/5877)
 
* `qml.from_qasm` now supports the ability to convert mid-circuit measurements from `OpenQASM 2` code, 
  and it can now also take an optional argument to specify a list of measurements to be performed at 
  the end of the circuit, just like `qml.from_qiskit`.
  [(#5818)](https://github.com/PennyLaneAI/pennylane/pull/5818)

* Four new operators have been added for simulating noise on the `default.qutrit.mixed` device:
  [(#5502)](https://github.com/PennyLaneAI/pennylane/pull/5502)
  [(#5793)](https://github.com/PennyLaneAI/pennylane/issues/5793)
  [(#5503)](https://github.com/PennyLaneAI/pennylane/pull/5503)
  [(#5757)](https://github.com/PennyLaneAI/pennylane/pull/5757)
  [(#5799)](https://github.com/PennyLaneAI/pennylane/pull/5799)
  [(#5784)](https://github.com/PennyLaneAI/pennylane/pull/5784)

  * `qml.QutritDepolarizingChannel`: a channel that adds depolarizing noise.
  * `qml.QutritChannel`: enables the specification of noise using a collection of (3x3) Kraus matrices.
  * `qml.QutritAmplitudeDamping`: a channel that adds noise processes modelled by amplitude damping.
  * `qml.TritFlip`: a channel that adds trit flip errors, such as misclassification.

<h4>Faster and more flexible mid-circuit measurements</h4>

* The `default.qubit` device supports a depth-first tree-traversal algorithm to accelerate native mid-circuit 
  measurement execution. Accessible through the QNode argument `mcm_method="tree-traversal"`,
  this new implementation supports classical control, collecting statistics, and 
  post-selection, along with all measurements enabled with `qml.dynamic_one_shot`. More information 
  about this new mid-circuit measurement method can be found on our 
  [measurement documentation page](https://docs.pennylane.ai/en/stable/introduction/dynamic_quantum_circuits.html#tree-traversal-algorithm).
  [(#5180)](https://github.com/PennyLaneAI/pennylane/pull/5180)

* `qml.QNode` and the `@qml.qnode` decorator now accept two new keyword arguments: `postselect_mode` 
  and `mcm_method`. These keyword arguments can be used to configure how the device should behave when 
  running circuits with mid-circuit measurements.
  [(#5679)](https://github.com/PennyLaneAI/pennylane/pull/5679)
  [(#5833)](https://github.com/PennyLaneAI/pennylane/pull/5833)
  [(#5850)](https://github.com/PennyLaneAI/pennylane/pull/5850)

  * `postselect_mode="hw-like"` indicates to devices to discard invalid shots when postselecting
    mid-circuit measurements. Use `postselect_mode="fill-shots"` to unconditionally sample the postselected
    value, thus making all samples valid. This is equivalent to sampling until the number of valid samples
    matches the total number of shots.
  * `mcm_method` will indicate which strategy to use for running circuits with mid-circuit measurements.
    Use `mcm_method="deferred"` to use the deferred measurements principle, or `mcm_method="one-shot"`
    to execute once for each shot. If `qml.qjit` is being used (the Catalyst compiler), `mcm_method="single-branch-statistics"`
    is also available. Using this method, a single branch of the execution tree will be randomly explored.

* The `dynamic_one_shot` transform received a few improvements:
  
  * `dynamic_one_shot` is now compatible with `qml.qjit` (the Catalyst compiler).
    [(#5766)](https://github.com/PennyLaneAI/pennylane/pull/5766)
    [(#5888)](https://github.com/PennyLaneAI/pennylane/pull/5888)
  * `dynamic_one_shot` now uses a single auxiliary tape with a shot vector and `default.qubit` implements 
    the loop over shots with `jax.vmap`.
    [(#5617)](https://github.com/PennyLaneAI/pennylane/pull/5617)
  * `dynamic_one_shot` is now compatible with `jax.jit`.
    [(#5557)](https://github.com/PennyLaneAI/pennylane/pull/5557)

* When using `defer_measurements` with postselection, operations that will never be active due to the 
  postselected state are skipped in the transformed quantum circuit. In addition, postselected controls 
  are skipped, as they are evaluated when the transform is applied. This optimization feature can be 
  turned off by setting `reduce_postselected=False`.
  [(#5558)](https://github.com/PennyLaneAI/pennylane/pull/5558)

  Consider a simple circuit with three mid-circuit measurements, two of which are postselecting,
  and a single gate conditioned on those measurements:

  ```python
  @qml.qnode(qml.device("default.qubit"))
  def node(x):
      qml.RX(x, 0)
      qml.RX(x, 1)
      qml.RX(x, 2)
      mcm0 = qml.measure(0, postselect=0, reset=False)
      mcm1 = qml.measure(1, postselect=None, reset=True)
      mcm2 = qml.measure(2, postselect=1, reset=False)
      qml.cond(mcm0 + mcm1 + mcm2 == 1, qml.RX)(0.5, 3)
      return qml.expval(qml.Z(0) @ qml.Z(3))
  ```

  Without the new optimization, we obtain three gates, each controlled on the three measured
  qubits. They correspond to the combinations of controls that satisfy the condition
  `mcm0 + mcm1 + mcm2 == 1`:

  ```pycon
  >>> print(qml.draw(qml.defer_measurements(node, reduce_postselected=False))(0.6))
  0: ──RX(0.60)──|0⟩⟨0|─╭●─────────────────────────────────────────────┤ ╭<Z@Z>
  1: ──RX(0.60)─────────│──╭●─╭X───────────────────────────────────────┤ │
  2: ──RX(0.60)─────────│──│──│───|1⟩⟨1|─╭○────────╭○────────╭●────────┤ │
  3: ───────────────────│──│──│──────────├RX(0.50)─├RX(0.50)─├RX(0.50)─┤ ╰<Z@Z>
  4: ───────────────────╰X─│──│──────────├○────────├●────────├○────────┤
  5: ──────────────────────╰X─╰●─────────╰●────────╰○────────╰○────────┤
  ```

  If we do not explicitly deactivate the optimization, we obtain a much simpler circuit:

  ```pycon
  >>> print(qml.draw(qml.defer_measurements(node))(0.6))
  0: ──RX(0.60)──|0⟩⟨0|─╭●─────────────────┤ ╭<Z@Z>
  1: ──RX(0.60)─────────│──╭●─╭X───────────┤ │
  2: ──RX(0.60)─────────│──│──│───|1⟩⟨1|───┤ │
  3: ───────────────────│──│──│──╭RX(0.50)─┤ ╰<Z@Z>
  4: ───────────────────╰X─│──│──│─────────┤
  5: ──────────────────────╰X─╰●─╰○────────┤
  ```

  There is only one controlled gate with only one control wire.

* Mid-circuit measurement tests have been streamlined and refactored, removing most end-to-end tests 
  from the native MCM test file, but keeping one that validates multiple mid-circuit measurements with 
  any allowed return and interface end-to-end tests.
  [(#5787)](https://github.com/PennyLaneAI/pennylane/pull/5787)

<h4>Access to QROM</h4>

* [The QROM algorithm](https://arxiv.org/abs/1812.00954) is now available in PennyLane with `qml.QROM`. 
  This template allows you to enter classical data in the form of bitstrings.
  [(#5688)](https://github.com/PennyLaneAI/pennylane/pull/5688)

  ```python
  bitstrings = ["010", "111", "110", "000"]

  dev = qml.device("default.qubit", shots = 1)

  @qml.qnode(dev)
  def circuit():
      qml.BasisEmbedding(2, wires = [0,1])

      qml.QROM(bitstrings = bitstrings,
              control_wires = [0,1],
              target_wires = [2,3,4],
              work_wires = [5,6,7])

      return qml.sample(wires = [2,3,4])
  ```
  
  ```pycon
  >>> print(circuit())
  [1 1 0]
  ```

<h4>Capturing and representing hybrid programs</h4>

* A number of templates have been updated to be valid PyTrees and PennyLane operations.
  [(#5698)](https://github.com/PennyLaneAI/pennylane/pull/5698)

* PennyLane operators, measurements, and QNodes can now automatically be captured as instructions in
  JAXPR.
  [(#5564)](https://github.com/PennyLaneAI/pennylane/pull/5564)
  [(#5511)](https://github.com/PennyLaneAI/pennylane/pull/5511)
  [(#5708)](https://github.com/PennyLaneAI/pennylane/pull/5708)
  [(#5523)](https://github.com/PennyLaneAI/pennylane/pull/5523)
  [(#5686)](https://github.com/PennyLaneAI/pennylane/pull/5686)
  [(#5889)](https://github.com/PennyLaneAI/pennylane/pull/5889)

* The `qml.PyTrees` module now has `flatten` and `unflatten` methods for serializing PyTrees.
  [(#5701)](https://github.com/PennyLaneAI/pennylane/pull/5701)

* `qml.sample` can now be used on Boolean values representing mid-circuit measurement results in
  traced quantum functions. This feature is used with Catalyst to enable the pattern
  `m = measure(0); qml.sample(m)`.
  [(#5673)](https://github.com/PennyLaneAI/pennylane/pull/5673)

<h4>Quantum chemistry</h4>

* The `qml.qchem.Molecule` object received a few improvements:
  * `qml.qchem.Molecule` is now the central object used by all qchem functions.
    [(#5571)](https://github.com/PennyLaneAI/pennylane/pull/5571)
  * `qml.qchem.Molecule` now supports Angstrom as a unit.
    [(#5694)](https://github.com/PennyLaneAI/pennylane/pull/5694)
  * `qml.qchem.Molecule` now supports open-shell systems.
    [(#5655)](https://github.com/PennyLaneAI/pennylane/pull/5655)

* The `qml.qchem.molecular_hamiltonian` function now supports parity and Bravyi-Kitaev mappings.
  [(#5657)](https://github.com/PennyLaneAI/pennylane/pull/5657/)

* `qml.qchem.molecular_dipole` function has been added for calculating the dipole operator using the 
  `"dhf"` and `"openfermion"` backends.
  [(#5764)](https://github.com/PennyLaneAI/pennylane/pull/5764)

* The qchem module now has dedicated functions for calling the `pyscf` and `openfermion` backends and 
  the `molecular_hamiltonian` and `molecular_dipole` functions have been moved to `hamiltonian` and
  `dipole` modules.
  [(#5553)](https://github.com/PennyLaneAI/pennylane/pull/5553)
  [(#5863)](https://github.com/PennyLaneAI/pennylane/pull/5863)

* More fermionic-to-qubit tests have been added to cover cases when the mapped operator is different 
  for various mapping schemes.
  [(#5873)](https://github.com/PennyLaneAI/pennylane/pull/5873)

<h4>Easier development</h4>

* Logging now allows for an easier opt-in across the stack and support has been extended to Catalyst.
  [(#5528)](https://github.com/PennyLaneAI/pennylane/pull/5528)

* Three new Pytest markers have been added for easier management of our test suite: `unit`, `integration` 
  and `system`.
  [(#5517)](https://github.com/PennyLaneAI/pennylane/pull/5517)

<h4>Other improvements</h4>

* `qml.MultiControlledX` can now be decomposed even when no `work_wires` are provided. The implementation 
  returns :math:`\mathcal{O}(\mbox{len(control wires)}^2)` operations and is applicable for any multi-controlled 
  unitary gate. This decomposition is provided in [arXiv:quant-ph/9503016](https://arxiv.org/abs/quant-ph/9503016).
  [(#5735)](https://github.com/PennyLaneAI/pennylane/pull/5735)

* A new function called `expectation_value` has been added to `qml.math` to calculate the expectation 
  value of a matrix for pure states.
  [(#4484)](https://github.com/PennyLaneAI/pennylane/pull/4484)

  ```pycon
  >>> state_vector = [1/np.sqrt(2), 0, 1/np.sqrt(2), 0]
  >>> operator_matrix = qml.matrix(qml.PauliZ(0), wire_order=[0,1])
  >>> qml.math.expectation_value(operator_matrix, state_vector)
  tensor(-2.23711432e-17+0.j, requires_grad=True)
  ```

* `param_shift` with the `broadcast=True` option now supports shot vectors and multiple measurements.
  [(#5667)](https://github.com/PennyLaneAI/pennylane/pull/5667)

* `qml.TrotterProduct` is now compatible with resource tracking by inheriting from `ResourcesOperation`. 
  [(#5680)](https://github.com/PennyLaneAI/pennylane/pull/5680)

* `packaging` is now a required package in PennyLane.
  [(#5769)](https://github.com/PennyLaneAI/pennylane/pull/5769)

* `qml.ctrl` now works with tuple-valued `control_values` when applied to any already controlled operation.
  [(#5725)](https://github.com/PennyLaneAI/pennylane/pull/5725)

* The sorting order of parameter-shift terms is now guaranteed to resolve ties in the absolute value 
  with the sign of the shifts.
  [(#5582)](https://github.com/PennyLaneAI/pennylane/pull/5582)

* `qml.transforms.split_non_commuting` can now handle circuits containing measurements of multi-term 
  observables.
  [(#5729)](https://github.com/PennyLaneAI/pennylane/pull/5729)
  [(#5838)](https://github.com/PennyLaneAI/pennylane/pull/5838)
  [(#5828)](https://github.com/PennyLaneAI/pennylane/pull/5828)
  [(#5869)](https://github.com/PennyLaneAI/pennylane/pull/5869)
  [(#5939)](https://github.com/PennyLaneAI/pennylane/pull/5939)
  [(#5945)](https://github.com/PennyLaneAI/pennylane/pull/5945)

* `qml.devices.LegacyDevice` is now an alias for `qml.Device`, so it is easier to distinguish it from
  `qml.devices.Device`, which follows the new device API.
  [(#5581)](https://github.com/PennyLaneAI/pennylane/pull/5581)

* The `dtype` for `eigvals` of `X`, `Y`, `Z` and `Hadamard` is changed from `int` to `float`, making 
  them consistent with the other observables. The `dtype` of the returned values when sampling these 
  observables (e.g. `qml.sample(X(0))`) is also changed to `float`.
  [(#5607)](https://github.com/PennyLaneAI/pennylane/pull/5607)

* The framework for the development of an `assert_equal` function for testing operator comparison has been set up.
  [(#5634)](https://github.com/PennyLaneAI/pennylane/pull/5634)
  [(#5858)](https://github.com/PennyLaneAI/pennylane/pull/5858)

* The `decompose` transform has an `error` keyword argument to specify the type of error that should 
  be raised, allowing error types to be more consistent with the context the `decompose` function is 
  used in.
  [(#5669)](https://github.com/PennyLaneAI/pennylane/pull/5669)

* Empty initialization of `PauliVSpace` is permitted.
  [(#5675)](https://github.com/PennyLaneAI/pennylane/pull/5675)

* `qml.tape.QuantumScript` properties are only calculated when needed, instead of on initialization. 
  This decreases the classical overhead by over 20%. Also, `par_info`, `obs_sharing_wires`, and `obs_sharing_wires_id` 
  are now public attributes.
  [(#5696)](https://github.com/PennyLaneAI/pennylane/pull/5696)
  
* The `qml.data` module now supports PyTree data types as dataset attributes
  [(#5732)](https://github.com/PennyLaneAI/pennylane/pull/5732)

* `qml.ops.Conditional` now inherits from `qml.ops.SymbolicOp`, thus it inherits several useful common 
  functionalities. Other properties such as adjoint and diagonalizing gates have been added using the 
  `base` properties.
  [(##5772)](https://github.com/PennyLaneAI/pennylane/pull/5772)

* New dispatches for `qml.ops.Conditional` and `qml.MeasurementValue` have been added to `qml.equal`.
  [(##5772)](https://github.com/PennyLaneAI/pennylane/pull/5772)

* The `qml.snapshots` transform now supports arbitrary devices by running a separate tape for each snapshot 
  for unsupported devices.
  [(#5805)](https://github.com/PennyLaneAI/pennylane/pull/5805)

* The `qml.Snapshot` operator now accepts sample-based measurements for finite-shot devices.
  [(#5805)](https://github.com/PennyLaneAI/pennylane/pull/5805)

* Device preprocess transforms now happen inside the ML boundary.
  [(#5791)](https://github.com/PennyLaneAI/pennylane/pull/5791)

* Transforms applied to callables now use `functools.wraps` to preserve the docstring and call signature 
  of the original function.
  [(#5857)](https://github.com/PennyLaneAI/pennylane/pull/5857)

* `qml.qsvt()` now supports JAX arrays with angle conversions. 
  [(#5853)](https://github.com/PennyLaneAI/pennylane/pull/5853)

* The sorting order of parameter-shift terms is now guaranteed to resolve ties in the absolute value with the sign of the shifts.
  [(#5583)](https://github.com/PennyLaneAI/pennylane/pull/5583)

<h3>Breaking changes 💔</h3>

* Passing `shots` as a keyword argument to a `QNode` initialization now raises an error instead of ignoring 
  the input.
  [(#5748)](https://github.com/PennyLaneAI/pennylane/pull/5748)

* A custom decomposition can no longer be provided to `qml.QDrift`. Instead, apply the operations in 
  your custom operation directly with `qml.apply`.
  [(#5698)](https://github.com/PennyLaneAI/pennylane/pull/5698)

* Sampling observables composed of `X`, `Y`, `Z` and `Hadamard` now returns values of type `float` instead 
  of `int`.
  [(#5607)](https://github.com/PennyLaneAI/pennylane/pull/5607)

* `qml.is_commuting` no longer accepts the `wire_map` argument, which does not bring any functionality.
  [(#5660)](https://github.com/PennyLaneAI/pennylane/pull/5660)

* `qml.from_qasm_file` has been removed. The user can open files and load their content using `qml.from_qasm`.
  [(#5659)](https://github.com/PennyLaneAI/pennylane/pull/5659)

* `qml.load` has been removed in favour of more specific functions, such as `qml.from_qiskit`, etc.
  [(#5654)](https://github.com/PennyLaneAI/pennylane/pull/5654)

* `qml.transforms.convert_to_numpy_parameters` is now a proper transform and its output signature has changed,
  returning a list of `QuantumScript`s and a post-processing function instead of simply the transformed circuit.
  [(#5693)](https://github.com/PennyLaneAI/pennylane/pull/5693)

* `Controlled.wires` does not include `self.work_wires` anymore. That can be accessed separately through `Controlled.work_wires`.
  Consequently, `Controlled.active_wires` has been removed in favour of the more common `Controlled.wires`.
  [(#5728)](https://github.com/PennyLaneAI/pennylane/pull/5728)

<h3>Deprecations 👋</h3>

* The `simplify` argument in `qml.Hamiltonian` and `qml.ops.LinearCombination` has been deprecated.
  Instead, `qml.simplify()` can be called on the constructed operator.
  [(#5677)](https://github.com/PennyLaneAI/pennylane/pull/5677)

* `qml.transforms.map_batch_transform` has been deprecated, since a transform can be applied directly to a batch of tapes.
  [(#5676)](https://github.com/PennyLaneAI/pennylane/pull/5676)

* The default behaviour of `qml.from_qasm()` to remove measurements in the QASM code has been deprecated. 
  Use `measurements=[]` to keep this behaviour or `measurements=None` to keep the measurements from the QASM code.
  [(#5882)](https://github.com/PennyLaneAI/pennylane/pull/5882)
  [(#5904)](https://github.com/PennyLaneAI/pennylane/pull/5904)

<h3>Documentation 📝</h3>

* The `qml.qchem` docs have been updated to showcase the new improvements.
  [(#5758)](https://github.com/PennyLaneAI/pennylane/pull/5758/)
  [(#5638)](https://github.com/PennyLaneAI/pennylane/pull/5638/)

* Several links to other functions in measurement process docstrings have been fixed.
  [(#5913)](https://github.com/PennyLaneAI/pennylane/pull/5913)

* Information about mid-circuit measurements has been moved from the measurements quickstart page to its own
  [mid-circuit measurements quickstart page](https://docs.pennylane.ai/en/stable/introduction/mid_circuit_measurements.html)
  [(#5870)](https://github.com/PennyLaneAI/pennylane/pull/5870)

* The documentation for the `default.tensor` device has been added.
  [(#5719)](https://github.com/PennyLaneAI/pennylane/pull/5719)

* A small typo was fixed in the docstring for `qml.sample`.
  [(#5685)](https://github.com/PennyLaneAI/pennylane/pull/5685)

* Typesetting for some of the documentation was fixed, (use of left/right delimiters, fractions, and fixing incorrectly set up commands)
  [(#5804)](https://github.com/PennyLaneAI/pennylane/pull/5804)

* The `qml.Tracker` examples have been updated.
  [(#5803)](https://github.com/PennyLaneAI/pennylane/pull/5803)

* The input types for `coupling_map` in `qml.transpile` have been updated to reflect all the allowed 
  input types by `nx.to_networkx_graph`.
  [(#5864)](https://github.com/PennyLaneAI/pennylane/pull/5864)

* The text in the `qml.data` module and datasets quickstart has been slightly modified to lead to the 
  quickstart first and highlight `list_datasets`.
  [(5484)](https://github.com/PennyLaneAI/pennylane/pull/5484)

<h3>Bug fixes 🐛</h3>

* `qml.compiler.active` first checks whether Catalyst is imported at all to avoid changing `jax_enable_x64` on module initialization.
  [(#5960)](https://github.com/PennyLaneAI/pennylane/pull/5960)

* The `__invert__` dunder method of the `MeasurementValue` class uses an array-valued function.
  [(#5955)](https://github.com/PennyLaneAI/pennylane/pull/5955)

* Skip `Projector`-measurement tests on devices that do not support it.
  [(#5951)](https://github.com/PennyLaneAI/pennylane/pull/5951)

* The `default.tensor` device now preserves the order of wires if the initial MPS is created from a dense state vector.
  [(#5892)](https://github.com/PennyLaneAI/pennylane/pull/5892)

* Fixed a bug where `hadamard_grad` returned a wrong shape for `qml.probs()` without wires.
  [(#5860)](https://github.com/PennyLaneAI/pennylane/pull/5860)

* An error is now raised on processing an `AnnotatedQueue` into a `QuantumScript` if the queue
  contains something other than an `Operator`, `MeasurementProcess`, or `QuantumScript`.
  [(#5866)](https://github.com/PennyLaneAI/pennylane/pull/5866)

* Fixed a bug in the wire handling on special controlled ops.
  [(#5856)](https://github.com/PennyLaneAI/pennylane/pull/5856)

* Fixed a bug where `Sum`'s with repeated identical operations ended up with the same hash as
  `Sum`'s with different numbers of repeats.
  [(#5851)](https://github.com/PennyLaneAI/pennylane/pull/5851)

* `qml.qaoa.cost_layer` and `qml.qaoa.mixer_layer` can now be used with `Sum` operators.
  [(#5846)](https://github.com/PennyLaneAI/pennylane/pull/5846)

* Fixed a bug where `qml.MottonenStatePreparation` produces wrong derivatives at special parameter values.
  [(#5774)](https://github.com/PennyLaneAI/pennylane/pull/5774)

* Fixed a bug where fractional powers and adjoints of operators were commuted, which is
  not well-defined/correct in general. Adjoints of fractional powers can no longer be evaluated.
  [(#5835)](https://github.com/PennyLaneAI/pennylane/pull/5835)

* `qml.qnn.TorchLayer` now works with tuple returns.
  [(#5816)](https://github.com/PennyLaneAI/pennylane/pull/5816)

* An error is now raised if a transform is applied to a catalyst qjit object.
  [(#5826)](https://github.com/PennyLaneAI/pennylane/pull/5826)

* `qml.qnn.KerasLayer` and `qml.qnn.TorchLayer` no longer mutate the input `qml.QNode`'s interface.
  [(#5800)](https://github.com/PennyLaneAI/pennylane/pull/5800)

* Docker builds on PR merging has been disabled.
  [(#5777)](https://github.com/PennyLaneAI/pennylane/pull/5777)

* The validation of the adjoint method in `DefaultQubit` correctly handles device wires now.
  [(#5761)](https://github.com/PennyLaneAI/pennylane/pull/5761)

* `QuantumPhaseEstimation.map_wires` on longer modifies the original operation instance.
  [(#5698)](https://github.com/PennyLaneAI/pennylane/pull/5698)

* The decomposition of `qml.AmplitudeAmplification` now correctly queues all operations.
  [(#5698)](https://github.com/PennyLaneAI/pennylane/pull/5698)

* Replaced `semantic_version` with `packaging.version.Version`, since the former cannot
  handle the metadata `.post` in the version string.
  [(#5754)](https://github.com/PennyLaneAI/pennylane/pull/5754)

* The `dynamic_one_shot` transform now has expanded support for the `jax` and `torch` interfaces.
  [(#5672)](https://github.com/PennyLaneAI/pennylane/pull/5672)

* The decomposition of `StronglyEntanglingLayers` is now compatible with broadcasting.
  [(#5716)](https://github.com/PennyLaneAI/pennylane/pull/5716)

* `qml.cond` can now be applied to `ControlledOp` operations when deferring measurements.
  [(#5725)](https://github.com/PennyLaneAI/pennylane/pull/5725)

* The legacy `Tensor` class can now handle a `Projector` with abstract tracer input.
  [(#5720)](https://github.com/PennyLaneAI/pennylane/pull/5720)

* Fixed a bug that raised an error regarding expected versus actual `dtype` when using `JAX-JIT` on a circuit that
  returned samples of observables containing the `qml.Identity` operator.
  [(#5607)](https://github.com/PennyLaneAI/pennylane/pull/5607)

* The signature of `CaptureMeta` objects (like `Operator`) now match the signature of the `__init__` call.
  [(#5727)](https://github.com/PennyLaneAI/pennylane/pull/5727)

* Vanilla NumPy arrays are now used in `test_projector_expectation` to avoid differentiating `qml.Projector` 
  with respect to the state attribute.
  [(#5683)](https://github.com/PennyLaneAI/pennylane/pull/5683)

* `qml.Projector` is now compatible with `jax.jit`.
  [(#5595)](https://github.com/PennyLaneAI/pennylane/pull/5595)

* Finite-shot circuits with a `qml.probs` measurement, both with a `wires` or `op` argument, can now be compiled with `jax.jit`.
  [(#5619)](https://github.com/PennyLaneAI/pennylane/pull/5619)

* `param_shift`, `finite_diff`, `compile`, `insert`, `merge_rotations`, and `transpile` now
  all work with circuits with non-commuting measurements.
  [(#5424)](https://github.com/PennyLaneAI/pennylane/pull/5424)
  [(#5681)](https://github.com/PennyLaneAI/pennylane/pull/5681)

* A correction has been added to `qml.bravyi_kitaev` to call the correct function for a `qml.FermiSentence` input.
  [(#5671)](https://github.com/PennyLaneAI/pennylane/pull/5671)

* Fixed a bug where `sum_expand` produces incorrect result dimensions when combined with shot vectors,
  multiple measurements, and parameter broadcasting.
  [(#5702)](https://github.com/PennyLaneAI/pennylane/pull/5702)

* Fixed a bug in `qml.math.dot` that raises an error when only one of the operands is a scalar.
  [(#5702)](https://github.com/PennyLaneAI/pennylane/pull/5702)

* `qml.matrix` is now compatible with QNodes compiled by `qml.qjit`.
  [(#5753)](https://github.com/PennyLaneAI/pennylane/pull/5753)

* `qml.snapshots` raises an error when a measurement other than `qml.state` is requested from `default.qubit.legacy` 
  instead of silently returning the statevector.
  [(#5805)](https://github.com/PennyLaneAI/pennylane/pull/5805)

* Fixed a bug where `default.qutrit` was falsely determined to be natively compatible with `qml.snapshots`.
  [(#5805)](https://github.com/PennyLaneAI/pennylane/pull/5805)

* Fixed a bug where the measurement of a `qml.Snapshot` instance was not passed on during the `qml.adjoint` and `qml.ctrl` operations.
  [(#5805)](https://github.com/PennyLaneAI/pennylane/pull/5805)

* `qml.CNOT` and `qml.Toffoli` now have an `arithmetic_depth` of `1`, as they are controlled operations.
  [(#5797)](https://github.com/PennyLaneAI/pennylane/pull/5797)

* Fixed a bug where the gradient of `ControlledSequence`, `Reflection`, `AmplitudeAmplification`, and `Qubitization` was incorrect on `default.qubit.legacy` with `parameter_shift`.
  [(#5806)](https://github.com/PennyLaneAI/pennylane/pull/5806)

* Fixed a bug where `split_non_commuting` raises an error when the circuit contains measurements of 
  observables that are not Pauli words.
  [(#5827)](https://github.com/PennyLaneAI/pennylane/pull/5827)

* The `simplify` method for `qml.Exp` now returns an operator with the correct number of Trotter steps, 
  i.e. equal to the one from the pre-simplified operator.
  [(#5831)](https://github.com/PennyLaneAI/pennylane/pull/5831)

* Fixed a bug where `CompositeOp.overlapping_ops` would put overlapping operators in different groups, 
  leading to incorrect results returned by `LinearCombination.eigvals()`.
  [(#5847)](https://github.com/PennyLaneAI/pennylane/pull/5847)

* The correct decomposition for a `qml.PauliRot` with an identity as `pauli_word` has been implemented, 
  i.e. returns a `qml.GlobalPhase` with half the angle.
  [(#5875)](https://github.com/PennyLaneAI/pennylane/pull/5875)

* `qml.pauli_decompose` now works in a jit-ted context, such as `jax.jit` and `qml.qjit`.
  [(#5878)](https://github.com/PennyLaneAI/pennylane/pull/5878)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Tarun Kumar Allamsetty,
Guillermo Alonso-Linaje,
Utkarsh Azad,
Lillian M. A. Frederiksen,
Ludmila Botelho,
Gabriel Bottrill,
Thomas Bromley,
Jack Brown,
Astral Cai,
Ahmed Darwish,
Isaac De Vlugt,
Diksha Dhawan,
Pietropaolo Frisoni,
Emiliano Godinez,
Diego Guala,
Daria Van Hende,
Austin Huang,
David Ittah,
Soran Jahangiri,
Rohan Jain,
Mashhood Khan,
Korbinian Kottmann,
Christina Lee,
Vincent Michaud-Rioux,
Lee James O'Riordan,
Mudit Pandey,
Kenya Sakka,
Jay Soni,
Kazuki Tsuoka,
Haochen Paul Wang,
David Wierichs.