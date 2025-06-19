:orphan:

# Release 0.42.0-dev (development release)

<h3>New features since last release</h3>

* A new decomposition based on *unary iteration* has been added to :class:`qml.Select`.
  This decomposition reduces the :class:`T` count significantly, and uses :math:`c-1`
  auxiliary wires for a :class:`qml.Select` operation with :math:`c` control wires.
  Unary iteration leverages these auxiliary wires to store intermediate values for reuse
  among the different multi-controlled operators, avoiding unnecessary recomputation.
  Check out the documentation for a thorough explanation.
  [(#7623)](https://github.com/PennyLaneAI/pennylane/pull/7623)

* A new function called :func:`qml.from_qasm3` has been added, which converts OpenQASM 3.0 circuits into quantum functions
  that can be subsequently loaded into QNodes and executed. 
  [(#7432)](https://github.com/PennyLaneAI/pennylane/pull/7432)
  [(#7486)](https://github.com/PennyLaneAI/pennylane/pull/7486)
  [(#7488)](https://github.com/PennyLaneAI/pennylane/pull/7488)
  [(#7593)](https://github.com/PennyLaneAI/pennylane/pull/7593)

  ```python
  import pennylane as qml

  dev = qml.device("default.qubit", wires=[0, 1])
  
  @qml.qnode(dev)
  def my_circuit():
      qml.from_qasm3("qubit q0; qubit q1; ry(0.2) q0; rx(1.0) q1; pow(2) @ x q0;", {'q0': 0, 'q1': 1})
      return qml.expval(qml.Z(0))
  ```

  ```pycon
  >>> print(qml.draw(my_circuit)())
  0: ──RY(0.20)──X²─┤  <Z>
  1: ──RX(1.00)─────┤  
  ```
  
  Some gates and operations in OpenQASM 3.0 programs are not currently supported. For more details, 
  please consult the documentation for :func:`qml.from_qasm3` and ensure that you have installed `openqasm3` and 
  `'openqasm3[parser]'` in your environment by following the [OpenQASM 3.0 installation instructions](https://pypi.org/project/openqasm3/).

* A new QNode transform called :func:`~.transforms.set_shots` has been added to set or update the number of shots to be performed, overriding shots specified in the device.
  [(#7337)](https://github.com/PennyLaneAI/pennylane/pull/7337)
  [(#7358)](https://github.com/PennyLaneAI/pennylane/pull/7358)
  [(#7500)](https://github.com/PennyLaneAI/pennylane/pull/7500)

  The :func:`~.transforms.set_shots` transform can be used as a decorator:

  ```python
  @partial(qml.set_shots, shots=2)
  @qml.qnode(qml.device("default.qubit", wires=1))
  def circuit():
      qml.RX(1.23, wires=0)
      return qml.sample(qml.Z(0))
  ```

  ```pycon
  >>> circuit()
  array([1., -1.])
  ```
  
  Additionally, it can be used in-line to update a circuit's `shots`:

  ```pycon
  >>> new_circ = qml.set_shots(circuit, shots=(4, 10)) # shot vector
  >>> new_circ()
  (array([-1.,  1., -1.,  1.]), array([ 1.,  1.,  1., -1.,  1.,  1., -1., -1.,  1.,  1.]))
  ```

* A new function called `qml.to_openqasm` has been added, which allows for converting PennyLane circuits to OpenQASM 2.0 programs.
  [(#7393)](https://github.com/PennyLaneAI/pennylane/pull/7393)

  Consider this simple circuit in PennyLane:
  ```python
  dev = qml.device("default.qubit", wires=2, shots=100)

  @qml.qnode(dev)
  def circuit(theta, phi):
      qml.RX(theta, wires=0)
      qml.CNOT(wires=[0,1])
      qml.RZ(phi, wires=1)
      return qml.sample()
  ```

  This can be easily converted to OpenQASM 2.0 with `qml.to_openqasm`:
  ```pycon
  >>> openqasm_circ = qml.to_openqasm(circuit)(1.2, 0.9)
  >>> print(openqasm_circ)
  OPENQASM 2.0;
  include "qelib1.inc";
  qreg q[2];
  creg c[2];
  rx(1.2) q[0];
  cx q[0],q[1];
  rz(0.9) q[1];
  measure q[0] -> c[0];
  measure q[1] -> c[1];
  ```

* A new template called :class:`~.SelectPauliRot` that applies a sequence of uniformly controlled rotations to a target qubit 
  is now available. This operator appears frequently in unitary decomposition and block encoding techniques. 
  [(#7206)](https://github.com/PennyLaneAI/pennylane/pull/7206)
  [(#7617)](https://github.com/PennyLaneAI/pennylane/pull/7617)

  ```python
  angles = np.array([1.0, 2.0, 3.0, 4.0])

  wires = qml.registers({"control": 2, "target": 1})
  dev = qml.device("default.qubit", wires=3)

  @qml.qnode(dev)
  def circuit():
      qml.SelectPauliRot(
        angles,
        control_wires=wires["control"],
        target_wire=wires["target"],
        rot_axis="Y")
      return qml.state()
  ```
  
  ```pycon
  >>> print(circuit())
  [0.87758256+0.j 0.47942554+0.j 0.        +0.j 0.        +0.j
   0.        +0.j 0.        +0.j 0.        +0.j 0.        +0.j]
  ```

* The transform `convert_to_mbqc_gateset` is added to the `ftqc` module to convert arbitrary 
  circuits to a limited gate-set that can be translated to the MBQC formalism.
  [(#7271)](https://github.com/PennyLaneAI/pennylane/pull/7271)

* Classical shadows with mixed quantum states are now computed with a dedicated method that uses an
  iterative algorithm similar to the handling of shadows with state vectors. This makes shadows with density 
  matrices much more performant.
  [(#6748)](https://github.com/PennyLaneAI/pennylane/pull/6748)
  [(#7458)](https://github.com/PennyLaneAI/pennylane/pull/7458)

* The `RotXZX` operation is added to the `ftqc` module to support definition of a universal
  gate-set that can be translated to the MBQC formalism.
  [(#7271)](https://github.com/PennyLaneAI/pennylane/pull/7271)

* A new iterative angle solver for QSVT and QSP is available in the :func:`poly_to_angles <pennylane.poly_to_angles>` function,
  allowing angle computation for polynomials of large degrees (> 1000).
  Set `angle_solver="iterative"` in the :func:`poly_to_angles  <pennylane.poly_to_angles>` function
  (or from the :func:`qsvt <pennylane.qsvt>` function!) to use it.
  [(6694)](https://github.com/PennyLaneAI/pennylane/pull/6694)

* Two new functions called :func:`~.math.convert_to_su2` and :func:`~.math.convert_to_su4` have been added to `qml.math`, which convert unitary matrices to SU(2) or SU(4), respectively, and optionally a global phase.
  [(#7211)](https://github.com/PennyLaneAI/pennylane/pull/7211)


* A new template :class:`~.TemporaryAND` has been added. The  :class:`~.TemporaryAND` (a.k.a.  :class:`~.Elbow`)
  operation is a three-qubit gate equivalent to an ``AND``, or reversible :class:`~pennylane.Toffoli`, gate
  that leverages extra information about the target wire to enable more efficient circuit decompositions.
  The ``TemporaryAND`` assumes the target qubit to be initialized in ``|0〉``, while the ``Adjoint(TemporaryAND)`` assumes the target output to be ``|0〉``.
  For more details, see Fig. 4 in `arXiv:1805.03662 <https://arxiv.org/abs/1805.03662>`_.
  :class:`~.TemporaryAND` is useful for an efficient decomposition of the :class:`~.Select` template, for example. 
  [(#7472)](https://github.com/PennyLaneAI/pennylane/pull/7472)

  ```python
  dev = qml.device("default.qubit", shots=1)
  @qml.qnode(dev)
  def circuit():
      # |0000⟩
      qml.X(0) # |1000⟩
      qml.X(1) # |1100⟩
      # The target wire is in state |0>, so we can apply TemporaryAND
      qml.TemporaryAND([0,1,2]) # |1110⟩
      qml.CNOT([2,3]) # |1111⟩
      # The target wire will be in state |0> after adjoint(TemporaryAND) gate is applied, so we can apply adjoint(TemporaryAND)
      qml.adjoint(qml.TemporaryAND([0,1,2])) # |1101⟩
      return qml.sample(wires=[0,1,2,3])
  ```
  
  ```pycon
  >>> print(circuit())
  [1 1 0 1]
  ```

* The transform `convert_to_mbqc_formalism` is added to the `ftqc` module to convert a circuit already
  expressed in a limited, compatible gate-set into the MBQC formalism. Circuits can be converted to the 
  relevant gate-set with the `convert_to_mbqc_gateset` transform.
  [(#7355)](https://github.com/PennyLaneAI/pennylane/pull/7355)
  [(#7586)](https://github.com/PennyLaneAI/pennylane/pull/7586)

* A new template :class:`~.SemiAdder` has been added, allowing for quantum-quantum in-place addition.
  This operator performs the plain addition of two integers in the computational basis.
  [(#7494)](https://github.com/PennyLaneAI/pennylane/pull/7494)

  ```python
  x = 3
  y = 4

  wires = qml.registers({"x":3, "y":6, "work":5})

  dev = qml.device("default.qubit", shots=1)

  @qml.qnode(dev)
  def circuit():
      qml.BasisEmbedding(x, wires=wires["x"])
      qml.BasisEmbedding(y, wires=wires["y"])
      qml.SemiAdder(wires["x"], wires["y"], wires["work"])
      return qml.sample(wires=wires["y"])
  ```
  
  ```pycon
  >>> print(circuit())
  [0 0 0 1 1 1]
  ```

<h4>Resource-efficient Decompositions 🔎</h4>

* The :func:`~.transforms.decompose` transform now supports weighting gates in the target `gate_set`, allowing for 
  preferential treatment of certain gates in a target `gate_set` over others.
  [(#7389)](https://github.com/PennyLaneAI/pennylane/pull/7389)

  Gates specified in `gate_set` can be given a numerical weight associated with their effective cost to have in a circuit:
  
  * Gate weights that are greater than 1 indicate a *greater cost* (less preferred).
  * Gate weights that are less than 1 indicate a *lower cost* (more preferred).

  Consider the following toy example.

  ```python
  qml.decomposition.enable_graph()
  
  @partial(
    qml.transforms.decompose, gate_set={qml.Toffoli: 1.23, qml.RX: 4.56, qml.CZ: 0.01, qml.H: 420, qml.CRZ: 100}
  )
  @qml.qnode(qml.device("default.qubit"))
  def circuit():
      qml.CRX(0.1, wires=[0, 1])
      qml.Toffoli(wires=[0, 1, 2])
      return qml.expval(qml.Z(0))
  ```

  ```pycon
  >>> print(qml.draw(circuit)())

  0: ───────────╭●────────────╭●─╭●─┤  <Z>
  1: ──RX(0.05)─╰Z──RX(-0.05)─╰Z─├●─┤     
  2: ────────────────────────────╰X─┤     
  ```

  ```python
  qml.decomposition.enable_graph()

  @partial(
      qml.transforms.decompose, gate_set={qml.Toffoli: 1.23, qml.RX: 4.56, qml.CZ: 0.01, qml.H: 0.1, qml.CRZ: 0.1}
  )
  @qml.qnode(qml.device("default.qubit"))
  def circuit():
      qml.CRX(0.1, wires=[0, 1])
      qml.Toffoli(wires=[0, 1, 2])
      return qml.expval(qml.Z(0))
  ```

  ```pycon
  >>> print(qml.draw(circuit)())

  0: ────╭●───────────╭●─┤  <Z>
  1: ──H─╰RZ(0.10)──H─├●─┤     
  2: ─────────────────╰X─┤  
  ```

  Here, when the Hadamard and ``CRZ`` have relatively high weights, a decomposition involving them is considered *less* 
  efficient. When they have relatively low weights, a decomposition involving them is considered *more* efficient.

* Decomposition rules that can be accessed with the new graph-based decomposition system are
  implemented for the following operators:

  * :class:`~.QubitUnitary`
    [(#7211)](https://github.com/PennyLaneAI/pennylane/pull/7211)

  * :class:`~.ControlledQubitUnitary`
    [(#7371)](https://github.com/PennyLaneAI/pennylane/pull/7371)

  * :class:`~.DiagonalQubitUnitary`
    [(#7625)](https://github.com/PennyLaneAI/pennylane/pull/7625)

  * :class:`~.MultiControlledX`
    [(#7405)](https://github.com/PennyLaneAI/pennylane/pull/7405)

  * :class:`~pennylane.ops.Exp`. 
    [(#7489)](https://github.com/PennyLaneAI/pennylane/pull/7489)

    Specifically, the following decompositions have been added:
    * Suzuki-Trotter decomposition when the `num_steps` keyword argument is specified.
    * Decomposition to a :class:`~pennylane.PauliRot` when the base is a single-term Pauli word.

  * :class:`~.PCPhase`
    [(#7591)](https://github.com/PennyLaneAI/pennylane/pull/7591)

  * :class:`~.QuantumPhaseEstimation`
    [(#7637)](https://github.com/PennyLaneAI/pennylane/pull/7637)

  * :class:`~.BasisRotation`
    [(#7074)](https://github.com/PennyLaneAI/pennylane/pull/7074)

  * :class:`~.PhaseAdder`
    [(#7070)](https://github.com/PennyLaneAI/pennylane/pull/7070)

  * :class:`~.IntegerComparator`
    [(#7636)](https://github.com/PennyLaneAI/pennylane/pull/7636)

* A new decomposition rule that uses a single work wire for decomposing multi-controlled operators is added.
  [(#7383)](https://github.com/PennyLaneAI/pennylane/pull/7383)

* A :func:`~.decomposition.register_condition` decorator is added that allows users to bind a condition to a
  decomposition rule for when it is applicable. The condition should be a function that takes the
  resource parameters of an operator as arguments and returns `True` or `False` based on whether
  these parameters satisfy the condition for when this rule can be applied.
  [(#7439)](https://github.com/PennyLaneAI/pennylane/pull/7439)

  ```python
  import pennylane as qml
  from pennylane.math.decomposition import zyz_rotation_angles
  
  # The parameters must be consistent with ``qml.QubitUnitary.resource_keys``
  def _zyz_condition(num_wires):
    return num_wires == 1

  @qml.register_condition(_zyz_condition)
  @qml.register_resources({qml.RZ: 2, qml.RY: 1, qml.GlobalPhase: 1})
  def zyz_decomposition(U, wires, **__):
      # Assumes that U is a 2x2 unitary matrix
      phi, theta, omega, phase = zyz_rotation_angles(U, return_global_phase=True)
      qml.RZ(phi, wires=wires[0])
      qml.RY(theta, wires=wires[0])
      qml.RZ(omega, wires=wires[0])
      qml.GlobalPhase(-phase)
  
  # This decomposition will be ignored for `QubitUnitary` on more than one wire.
  qml.add_decomps(qml.QubitUnitary, zyz_decomposition)
  ```

* Symbolic operator types (e.g., `Adjoint`, `Controlled`, and `Pow`) can now be specified as strings
  in various parts of the new graph-based decomposition system, specifically:

  * The `gate_set` argument of the :func:`~.transforms.decompose` transform now supports adding symbolic
    operators in the target gate set.
    [(#7331)](https://github.com/PennyLaneAI/pennylane/pull/7331)

  ```python
  from functools import partial
  import pennylane as qml

  qml.decomposition.enable_graph()
  
  @partial(qml.transforms.decompose, gate_set={"T", "Adjoint(T)", "H", "CNOT"})
  @qml.qnode(qml.device("default.qubit"))
  def circuit():
      qml.Toffoli(wires=[0, 1, 2])
  ```
  ```pycon
  >>> print(qml.draw(circuit)())
  0: ───────────╭●───────────╭●────╭●──T──╭●─┤
  1: ────╭●─────│─────╭●─────│───T─╰X──T†─╰X─┤
  2: ──H─╰X──T†─╰X──T─╰X──T†─╰X──T──H────────┤
  ```

  * Symbolic operator types can now be given as strings to the `op_type` argument of :func:`~.decomposition.add_decomps`,
    or as keys of the dictionaries passed to the `alt_decomps` and `fixed_decomps` arguments of the
    :func:`~.transforms.decompose` transform, allowing custom decomposition rules to be defined and
    registered for symbolic operators.
    [(#7347)](https://github.com/PennyLaneAI/pennylane/pull/7347)
    [(#7352)](https://github.com/PennyLaneAI/pennylane/pull/7352)
    [(#7362)](https://github.com/PennyLaneAI/pennylane/pull/7362)
    [(#7499)](https://github.com/PennyLaneAI/pennylane/pull/7499)

  ```python
  @qml.register_resources({qml.RY: 1})
  def my_adjoint_ry(phi, wires, **_):
      qml.RY(-phi, wires=wires)

  @qml.register_resources({qml.RX: 1})
  def my_adjoint_rx(phi, wires, **__):
      qml.RX(-phi, wires)

  # Registers a decomposition rule for the adjoint of RY globally
  qml.add_decomps("Adjoint(RY)", my_adjoint_ry)

  @partial(
      qml.transforms.decompose,
      gate_set={"RX", "RY", "CNOT"},
      fixed_decomps={"Adjoint(RX)": my_adjoint_rx}
  )
  @qml.qnode(qml.device("default.qubit"))
  def circuit():
      qml.adjoint(qml.RX(0.5, wires=[0]))
      qml.CNOT(wires=[0, 1])
      qml.adjoint(qml.RY(0.5, wires=[1]))
      return qml.expval(qml.Z(0))
  ```
  ```pycon
  >>> print(qml.draw(circuit)())
  0: ──RX(-0.50)─╭●────────────┤  <Z>
  1: ────────────╰X──RY(-0.50)─┤
  ```

* A `work_wire_type` argument has been added to :func:`~pennylane.ctrl` and :class:`~pennylane.ControlledQubitUnitary`
  for more fine-grained control over the type of work wire used in their decompositions.
  [(#7612)](https://github.com/PennyLaneAI/pennylane/pull/7612)

* The :func:`~.transforms.decompose` transform now accepts a `stopping_condition` argument with 
  graph-based decomposition enabled, which must be a function that returns `True` if an operator 
  does not need to be decomposed (it meets the requirements as described in `stopping_condition`).
  See the documentation for more details.
  [(#7531)](https://github.com/PennyLaneAI/pennylane/pull/7531)

<h3>Improvements 🛠</h3>

* Caching with finite shots now always warns about the lack of expected noise.
  [(#7644)](https://github.com/PennyLaneAI/pennylane/pull/7644)

* `cache` now defaults to `"auto"` with `qml.execute`, matching the behavior of `QNode` and reducing the 
  performance cost of using `qml.execute` for standard executions.
  [(#7644)](https://github.com/PennyLaneAI/pennylane/pull/7644)

* `qml.grad` and `qml.jacobian` can now handle inputs with dynamic shapes being captured into plxpr.
  [(#7544)](https://github.com/PennyLaneAI/pennylane/pull/7544/)

* Improved the drawing of `GlobalPhase`, `ctrl(GlobalPhase)`, `Identity` and `ctrl(Identity)` operations.
  The labels are grouped together like for other multi-qubit operations, and the drawing
  no longer depends on the wires of `GlobalPhase` or `Identity`. Control nodes of controlled global phases
  and identities no longer receive the operator label, which is in line with other controlled operations.
  [(#7457)](https://github.com/PennyLaneAI/pennylane/pull/7457)

* The decomposition of `qml.PCPhase` is now significantly more efficient for more than 2 qubits.
  [(#7166)](https://github.com/PennyLaneAI/pennylane/pull/7166)

* The decomposition of :class:`~.IntegerComparator` is now significantly more efficient.
  [(#7636)](https://github.com/PennyLaneAI/pennylane/pull/7636)

* :class:`~.QubitUnitary` now supports a decomposition that is compatible with an arbitrary number of qubits. 
  This represents a fundamental improvement over the previous implementation, which was limited to two-qubit systems.
  [(#7277)](https://github.com/PennyLaneAI/pennylane/pull/7277)

* Setting up the configuration of a workflow, including the determination of the best diff
  method, is now done *after* user transforms have been applied. This allows transforms to
  update the shots and change measurement processes with fewer issues.
  [(#7358)](https://github.com/PennyLaneAI/pennylane/pull/7358)

* The decomposition of `DiagonalQubitUnitary` has been updated to a recursive decomposition
  into a smaller `DiagonalQubitUnitary` and a `SelectPauliRot` operation. This is a known
  decomposition [Theorem 7 in Shende et al.](https://arxiv.org/abs/quant-ph/0406176)
  that contains fewer gates than the previous decomposition.
  [(#7370)](https://github.com/PennyLaneAI/pennylane/pull/7370)

* An xDSL `qml.compiler.python_compiler.transforms.MergeRotationsPass` pass for applying `merge_rotations` to an
  xDSL module has been added for the experimental xDSL Python compiler integration.
  [(#7364)](https://github.com/PennyLaneAI/pennylane/pull/7364)
  [(#7595)](https://github.com/PennyLaneAI/pennylane/pull/7595)
  [(#7664)](https://github.com/PennyLaneAI/pennylane/pull/7664)

* An xDSL `qml.compiler.python_compiler.transforms.IterativeCancelInversesPass` pass for applying `cancel_inverses`
  iteratively to an xDSL module has been added for the experimental xDSL Python compiler integration. This pass is
  optimized to cancel self-inverse operations iteratively to cancel nested self-inverse operations.
  [(#7364)](https://github.com/PennyLaneAI/pennylane/pull/7364)
  [(#7595)](https://github.com/PennyLaneAI/pennylane/pull/7595)

* An xDSL `qml.compiler.python_compiler.transforms.MeasurementsFromSamplesPass` pass has been
  added for the experimental xDSL Python compiler integration. This pass replaces all terminal
  measurements in a program with a single :func:`pennylane.sample` measurement, and adds
  postprocessing instructions to recover the original measurement.
  [(#7620)](https://github.com/PennyLaneAI/pennylane/pull/7620)

* An experimental integration for a Python compiler using [xDSL](https://xdsl.dev/index) has been introduced.
  This is similar to [Catalyst's MLIR dialects](https://docs.pennylane.ai/projects/catalyst/en/stable/dev/dialects.html#mlir-dialects-in-catalyst), 
  but it is coded in Python instead of C++.
  [(#7509)](https://github.com/PennyLaneAI/pennylane/pull/7509)
  [(#7357)](https://github.com/PennyLaneAI/pennylane/pull/7357)
  [(#7367)](https://github.com/PennyLaneAI/pennylane/pull/7367)
  [(#7462)](https://github.com/PennyLaneAI/pennylane/pull/7462)
  [(#7470)](https://github.com/PennyLaneAI/pennylane/pull/7470)
  [(#7510)](https://github.com/PennyLaneAI/pennylane/pull/7510)
  [(#7590)](https://github.com/PennyLaneAI/pennylane/pull/7590)

* PennyLane supports `JAX` version 0.6.0.
  [(#7299)](https://github.com/PennyLaneAI/pennylane/pull/7299)

* PennyLane supports `JAX` version 0.5.3.
  [(#6919)](https://github.com/PennyLaneAI/pennylane/pull/6919)

* Computing the angles for uniformly controlled rotations, used in :class:`~.MottonenStatePreparation`
  and :class:`~.SelectPauliRot`, now takes much less computational effort and memory.
  [(#7377)](https://github.com/PennyLaneAI/pennylane/pull/7377)

* The :func:`~.transforms.cancel_inverses` transform no longer changes the order of operations that don't have shared wires, providing a deterministic output.
  [(#7328)](https://github.com/PennyLaneAI/pennylane/pull/7328)

* Alias for Identity (`I`) is now accessible from `qml.ops`.
  [(#7200)](https://github.com/PennyLaneAI/pennylane/pull/7200)

* Add xz encoding related `pauli_to_xz`, `xz_to_pauli` and `pauli_prod` functions to the `ftqc` module.
  [(#7433)](https://github.com/PennyLaneAI/pennylane/pull/7433)

* Add commutation rules for a Clifford gate set (`qml.H`, `qml.S`, `qml.CNOT`) to the `ftqc.pauli_tracker` module,
  accessible via the `commute_clifford_op` function.
  [(#7444)](https://github.com/PennyLaneAI/pennylane/pull/7444)

* Add offline byproduct correction support to the `ftqc` module.
  [(#7447)](https://github.com/PennyLaneAI/pennylane/pull/7447)

* The `ftqc` module `measure_arbitrary_basis`, `measure_x` and `measure_y` functions
  can now be captured when program capture is enabled.
  [(#7219)](https://github.com/PennyLaneAI/pennylane/pull/7219)
  [(#7368)](https://github.com/PennyLaneAI/pennylane/pull/7368)

* `Operator.num_wires` now defaults to `None` to indicate that the operator can be on
  any number of wires.
  [(#7312)](https://github.com/PennyLaneAI/pennylane/pull/7312)

* Shots can now be overridden for specific `qml.Snapshot` instances via a `shots` keyword argument.
  [(#7326)](https://github.com/PennyLaneAI/pennylane/pull/7326)

  ```python
  dev = qml.device("default.qubit", wires=2, shots=10)

  @qml.qnode(dev)
  def circuit():
      qml.Snapshot("sample", measurement=qml.sample(qml.X(0)), shots=5)
      return qml.sample(qml.X(0))
  ```

  ```pycon
  >>> qml.snapshots(circuit)()
  {'sample': array([-1., -1., -1., -1., -1.]),
   'execution_results': array([ 1., -1., -1., -1., -1.,  1., -1., -1.,  1., -1.])}
  ```

* Two-qubit `QubitUnitary` gates no longer decompose into fundamental rotation gates; it now 
  decomposes into single-qubit `QubitUnitary` gates. This allows the decomposition system to
  further decompose single-qubit unitary gates more flexibly using different rotations.
  [(#7211)](https://github.com/PennyLaneAI/pennylane/pull/7211)

* The `gate_set` argument of :func:`~.transforms.decompose` now accepts `"X"`, `"Y"`, `"Z"`, `"H"`, 
  `"I"` as aliases for `"PauliX"`, `"PauliY"`, `"PauliZ"`, `"Hadamard"`, and `"Identity"`. These 
  aliases are also recognized as part of symbolic operators. For example, `"Adjoint(H)"` is now 
  accepted as an alias for `"Adjoint(Hadamard)"`.
  [(#7331)](https://github.com/PennyLaneAI/pennylane/pull/7331)

* PennyLane no longer validates that an operation has at least one wire, as having this check required the abstract
  interface to maintain a list of special implementations.
  [(#7327)](https://github.com/PennyLaneAI/pennylane/pull/7327)

* Two new device-developer transforms have been added to `devices.preprocess`: 
  :func:`~.devices.preprocess.measurements_from_counts` and :func:`~.devices.preprocess.measurements_from_samples`.
  These transforms modify the tape to instead contain a `counts` or `sample` measurement process, 
  deriving the original measurements from the raw counts/samples in post-processing. This allows 
  expanded measurement support for devices that only 
  support counts/samples at execution, like real hardware devices.
  [(#7317)](https://github.com/PennyLaneAI/pennylane/pull/7317)

* Sphinx version was updated to 8.1. Sphinx is upgraded to version 8.1 and uses Python 3.10. References to intersphinx (e.g. `<demos/>` or `<catalyst/>` are updated to remove the :doc: prefix that is incompatible with sphinx 8.1. 
  [(7212)](https://github.com/PennyLaneAI/pennylane/pull/7212)

* Migrated `setup.py` package build and install to `pyproject.toml`
  [(#7375)](https://github.com/PennyLaneAI/pennylane/pull/7375)

* Updated GitHub Actions workflows (`rtd.yml`, `readthedocs.yml`, and `docs.yml`) to use `ubuntu-24.04` runners.
 [(#7396)](https://github.com/PennyLaneAI/pennylane/pull/7396)

* Updated requirements and pyproject files to include the other package.  
  [(#7417)](https://github.com/PennyLaneAI/pennylane/pull/7417)

* Updated documentation check to remove duplicate docstring references. [(#7453)](https://github.com/PennyLaneAI/pennylane/pull/7453)

* Improved performance for `qml.clifford_t_decomposition` transform by introducing caching support and changed the
  default basis set of `qml.ops.sk_decomposition` to `(H, S, T)`, resulting in shorter decomposition sequences.
  [(#7454)](https://github.com/PennyLaneAI/pennylane/pull/7454)

<h3>Labs: a place for unified and rapid prototyping of research software 🧪</h3>

* The imports of dependencies introduced by ``labs`` functionalities have been modified such that
  these dependencies only have to be installed for the functions that use them, not to use
  ``labs`` functionalities in general. This decouples the various submodules, and even functions
  within the same submodule, from each other.
  [(#7650)](https://github.com/PennyLaneAI/pennylane/pull/7650)

* A new module :mod:`pennylane.labs.intermediate_reps <pennylane.labs.intermediate_reps>`
  provides functionality to compute intermediate representations for particular circuits.
  :func:`parity_matrix <pennylane.labs.intermediate_reps.parity_matrix>` computes
  the parity matrix intermediate representation for CNOT circuits.
  :func:`phase_polynomial <pennylane.labs.intermediate_reps.phase_polynomial>` computes
  the phase polynomial intermediate representation for {CNOT, RZ} circuits.
  These efficient intermediate representations are important
  for CNOT routing algorithms and other quantum compilation routines.
  [(#7229)](https://github.com/PennyLaneAI/pennylane/pull/7229)
  [(#7333)](https://github.com/PennyLaneAI/pennylane/pull/7333)
  [(#7629)](https://github.com/PennyLaneAI/pennylane/pull/7629)
  
* The `pennylane.labs.vibrational` module is upgraded to use features from the `concurrency` module
  to perform multiprocess and multithreaded execution of workloads. 
  [(#7401)](https://github.com/PennyLaneAI/pennylane/pull/7401)

* A `rowcol` function is now available in `pennylane.labs.intermediate_reps`.
  Given the parity matrix of a CNOT circuit and a qubit connectivity graph, it synthesizes a
  possible implementation of the parity matrix that respects the connectivity.
  [(#7394)](https://github.com/PennyLaneAI/pennylane/pull/7394)

* A new module :mod:`pennylane.labs.zxopt <pennylane.labs.zxopt>` provides access to the basic optimization
  passes from [pyzx](https://pyzx.readthedocs.io/en/latest/) for PennyLane circuits.
  
    * :func:`basic_optimization <pennylane.labs.zxopt.basic_optimization>` performs peephole optimizations on the circuit and is a useful subroutine for other optimization passes.
    * :func:`full_optimize <pennylane.labs.zxopt.full_optimize>` optimizes [(Clifford + T)](https://pennylane.ai/compilation/clifford-t-gate-set) circuits.
    * :func:`full_reduce <pennylane.labs.zxopt.full_reduce>` can optimize arbitrary PennyLane circuits and follows the pipeline described in the [the pyzx docs](https://pyzx.readthedocs.io/en/latest/simplify.html).
    * :func:`todd <pennylane.labs.zxopt.todd>` performs Third Order Duplicate and Destroy (`TODD <https://arxiv.org/abs/1712.01557>`__) via phase polynomials and reduces T gate counts.

  [(#7471)](https://github.com/PennyLaneAI/pennylane/pull/7471)

<h3>Breaking changes 💔</h3>

* Support for gradient keyword arguments as QNode keyword arguments has been removed. Instead please use the
  new `gradient_kwargs` keyword argument accordingly.
  [(#7648)](https://github.com/PennyLaneAI/pennylane/pull/7648)

* The default value of `cache` is now `"auto"` with `qml.execute`. Like `QNode`, `"auto"` only turns on caching
  when `max_diff > 1`.
  [(#7644)](https://github.com/PennyLaneAI/pennylane/pull/7644)

* A new decomposition for two-qubit unitaries was implemented in `two_qubit_decomposition`.
  It ensures the correctness of the decomposition in some edge cases but uses 3 CNOT gates
  even if 2 CNOTs would suffice theoretically.
  [(#7474)](https://github.com/PennyLaneAI/pennylane/pull/7474)

* The `return_type` property of `MeasurementProcess` has been removed. Please use `isinstance` for type checking instead.
  [(#7322)](https://github.com/PennyLaneAI/pennylane/pull/7322)

* The `KerasLayer` class in `qml.qnn.keras` has been removed because Keras 2 is no longer actively maintained.
  Please consider using a different machine learning framework, like `PyTorch <demos/tutorial_qnn_module_torch>`__ or `JAX <demos/tutorial_How_to_optimize_QML_model_using_JAX_and_Optax>`__.
  [(#7320)](https://github.com/PennyLaneAI/pennylane/pull/7320)

* The `qml.gradients.hamiltonian_grad` function has been removed because this gradient recipe is no
  longer required with the :doc:`new operator arithmetic system </news/new_opmath>`.
  [(#7302)](https://github.com/PennyLaneAI/pennylane/pull/7302)

* Accessing terms of a tensor product (e.g., `op = X(0) @ X(1)`) via `op.obs` has been removed.
  [(#7324)](https://github.com/PennyLaneAI/pennylane/pull/7324)

* The `mcm_method` keyword argument in `qml.execute` has been removed.
  [(#7301)](https://github.com/PennyLaneAI/pennylane/pull/7301)

* The `inner_transform` and `config` keyword arguments in `qml.execute` have been removed.
  [(#7300)](https://github.com/PennyLaneAI/pennylane/pull/7300)

* `Sum.ops`, `Sum.coeffs`, `Prod.ops` and `Prod.coeffs` have been removed.
  [(#7304)](https://github.com/PennyLaneAI/pennylane/pull/7304)

* Specifying `pipeline=None` with `qml.compile` has been removed.
  [(#7307)](https://github.com/PennyLaneAI/pennylane/pull/7307)

* The `control_wires` argument in `qml.ControlledQubitUnitary` has been removed.
  Furthermore, the `ControlledQubitUnitary` no longer accepts `QubitUnitary` objects as arguments as its `base`.
  [(#7305)](https://github.com/PennyLaneAI/pennylane/pull/7305)

* `qml.tape.TapeError` has been removed.
  [(#7205)](https://github.com/PennyLaneAI/pennylane/pull/7205)

<h3>Deprecations 👋</h3>

Here's a list of deprecations made this release. For a more detailed breakdown of deprecations and alternative code to use instead, Please consult the :doc:`deprecations and removals page </development/deprecations>`.

* Top-level access to `DeviceError`, `PennyLaneDeprecationWarning`, `QuantumFunctionError` and `ExperimentalWarning` have been deprecated and will be removed in v0.43. Please import them from the new `exceptions` module.
  [(#7292)](https://github.com/PennyLaneAI/pennylane/pull/7292)
  [(#7477)](https://github.com/PennyLaneAI/pennylane/pull/7477)
  [(#7508)](https://github.com/PennyLaneAI/pennylane/pull/7508)
  [(#7603)](https://github.com/PennyLaneAI/pennylane/pull/7603)

* `qml.operation.Observable` and the corresponding `Observable.compare` have been deprecated, as
  pennylane now depends on the more general `Operator` interface instead. The
  `Operator.is_hermitian` property can instead be used to check whether or not it is highly likely
  that the operator instance is Hermitian.
  [(#7316)](https://github.com/PennyLaneAI/pennylane/pull/7316)

* The boolean functions provided in `pennylane.operation` are deprecated. See the :doc:`deprecations page </development/deprecations>` 
  for equivalent code to use instead. These include `not_tape`, `has_gen`, `has_grad_method`, `has_multipar`,
  `has_nopar`, `has_unitary_gen`, `is_measurement`, `defines_diagonalizing_gates`, and `gen_is_multi_term_hamiltonian`.
  [(#7319)](https://github.com/PennyLaneAI/pennylane/pull/7319)

* `qml.operation.WiresEnum`, `qml.operation.AllWires`, and `qml.operation.AnyWires` are deprecated. To indicate that
  an operator can act on any number of wires, `Operator.num_wires = None` should be used instead. This is the default
  and does not need to be overwritten unless the operator developer wants to add wire number validation.
  [(#7313)](https://github.com/PennyLaneAI/pennylane/pull/7313)

* The :func:`qml.QNode.get_gradient_fn` method is now deprecated. Instead, use :func:`~.workflow.get_best_diff_method` to obtain the differentiation method.
  [(#7323)](https://github.com/PennyLaneAI/pennylane/pull/7323)

<h3>Internal changes ⚙️</h3>

* Update `jax` and `tensorflow` dependencies for `doc` builds.
  [(#7667)](https://github.com/PennyLaneAI/pennylane/pull/7667)

* `Pennylane` has been renamed to `pennylane` in the `pyproject.toml` file 
  to match the expected binary distribution format naming conventions.
  [(#7689)](https://github.com/PennyLaneAI/pennylane/pull/7689)

* The `qml.compiler.python_compiler` submodule has been restructured.
  [(#7645)](https://github.com/PennyLaneAI/pennylane/pull/7645)

* Move program capture code closer to where it is used.
  [(#7608)][https://github.com/PennyLaneAI/pennylane/pull/7608]

* Tests using `OpenFermion` in `tests/qchem` do not fail with NumPy>=2.0.0 any more.
  [(#7626)](https://github.com/PennyLaneAI/pennylane/pull/7626)

* Move `givens_decomposition` and private helpers from `qchem` to `math` module.
  [(#7545)](https://github.com/PennyLaneAI/pennylane/pull/7545)

* Enforce module dependencies in `pennylane` using `tach`.
  [(#7185)](https://github.com/PennyLaneAI/pennylane/pull/7185)
  [(#7416)](https://github.com/PennyLaneAI/pennylane/pull/7416)
  [(#7418)](https://github.com/PennyLaneAI/pennylane/pull/7418)
  [(#7429)](https://github.com/PennyLaneAI/pennylane/pull/7429)
  [(#7430)](https://github.com/PennyLaneAI/pennylane/pull/7430)
  [(#7437)](https://github.com/PennyLaneAI/pennylane/pull/7437)
  [(#7504)](https://github.com/PennyLaneAI/pennylane/pull/7504)
  [(#7538)](https://github.com/PennyLaneAI/pennylane/pull/7538)
  [(#7542)](https://github.com/PennyLaneAI/pennylane/pull/7542)
  [(#7667)](https://github.com/PennyLaneAI/pennylane/pull/7667)

* With program capture enabled, mcm method validation now happens on execution rather than setup.
  [(#7475)](https://github.com/PennyLaneAI/pennylane/pull/7475)

* Add `.git-blame-ignore-revs` file to the PennyLane repository. This file will allow specifying commits that should
  be ignored in the output of `git blame`. For example, this can be useful when a single commit includes bulk reformatting.
  [(#7507)](https://github.com/PennyLaneAI/pennylane/pull/7507)

* Add a `.gitattributes` file to standardize LF as the end-of-line character for the PennyLane
  repository.
  [(#7502)](https://github.com/PennyLaneAI/pennylane/pull/7502)

* `DefaultQubit` now implements `preprocess_transforms` and `setup_execution_config` instead of `preprocess`.
  [(#7468)](https://github.com/PennyLaneAI/pennylane/pull/7468)

* Fix subset of `pylint` errors in the `tests` folder.
  [(#7446)](https://github.com/PennyLaneAI/pennylane/pull/7446)

* Remove and reduce excessively expensive test cases in `tests/templates/test_subroutines/` that do not add value.
  [(#7436)](https://github.com/PennyLaneAI/pennylane/pull/7436)

* Stop using `pytest-timeout` in the PennyLane CI/CD pipeline.
  [(#7451)](https://github.com/PennyLaneAI/pennylane/pull/7451)

* A `RuntimeWarning` raised when using versions of JAX > 0.4.28 has been removed.
  [(#7398)](https://github.com/PennyLaneAI/pennylane/pull/7398)

* Wheel releases for PennyLane now follow the `PyPA binary-distribution format <https://packaging.python.org/en/latest/specifications/binary-distribution-format/>_` guidelines more closely.
  [(#7382)](https://github.com/PennyLaneAI/pennylane/pull/7382)

* `null.qubit` can now support an optional `track_resources` argument which allows it to record which gates are executed.
  [(#7226)](https://github.com/PennyLaneAI/pennylane/pull/7226)
  [(#7372)](https://github.com/PennyLaneAI/pennylane/pull/7372)
  [(#7392)](https://github.com/PennyLaneAI/pennylane/pull/7392)

* A new internal module, `qml.concurrency`, is added to support internal use of multiprocess and multithreaded execution of workloads. This also migrates the use of `concurrent.futures` in `default.qubit` to this new design.
  [(#7303)](https://github.com/PennyLaneAI/pennylane/pull/7303)

* Test suites in `tests/transforms/test_defer_measurement.py` use analytic mocker devices to test numeric results.
  [(#7329)](https://github.com/PennyLaneAI/pennylane/pull/7329)

* Add new `pennylane.exceptions` module for custom errors and warnings.
  [(#7205)](https://github.com/PennyLaneAI/pennylane/pull/7205)
  [(#7292)](https://github.com/PennyLaneAI/pennylane/pull/7292)

* Clean up `__init__.py` files in `math`, `ops`, `qaoa`, `tape` and `templates` to be explicit in what they import. 
  [(#7200)](https://github.com/PennyLaneAI/pennylane/pull/7200)
  
* The `Tracker` class has been moved into the `devices` module.
  [(#7281)](https://github.com/PennyLaneAI/pennylane/pull/7281)

* Moved functions that calculate rotation angles for unitary decompositions into an internal
  module `qml.math.decomposition`
  [(#7211)](https://github.com/PennyLaneAI/pennylane/pull/7211)

* Fixed a failing integration test for `qml.QDrift`  which multiplied the operators of the decomposition incorrectly to evolve the state.
  [(#7621)](https://github.com/PennyLaneAI/pennylane/pull/7621)

* The decomposition test in `assert_valid` no longer checks the matrix of the decomposition if the operator
  does not define a matrix representation.
  [(#7655)](https://github.com/PennyLaneAI/pennylane/pull/7655)

<h3>Documentation 📝</h3>

* The usage examples for `qml.decomposition.DecompositionGraph` have been updated.
  [(#7692)](https://github.com/PennyLaneAI/pennylane/pull/7692)

* The entry in the :doc:`/news/program_capture_sharp_bits` has been updated to include
  additional supported lightning devices: `lightning.kokkos` and `lightning.gpu`.
  [(#7674)](https://github.com/PennyLaneAI/pennylane/pull/7674)

* Updated the circuit drawing for `qml.Select` to include two commonly used symbols for 
  Select-applying, or multiplexing, an operator. Added a similar drawing for `qml.SelectPauliRot`.
  [(#7464)](https://github.com/PennyLaneAI/pennylane/pull/7464)
  
* The entry in the :doc:`/news/program_capture_sharp_bits` page for transforms has been updated; non-native transforms being applied
  to QNodes wherein operators have dynamic wires can lead to incorrect results.
  [(#7426)](https://github.com/PennyLaneAI/pennylane/pull/7426)

* Fixed the wrong `theta` to `phi` in :class:`~pennylane.IsingXY`.
  [(#7427)](https://github.com/PennyLaneAI/pennylane/pull/7427)

* In the :doc:`/introduction/compiling_circuits` page, in the "Decomposition in stages" section,
  circuit drawings now render in a way that's easier to read.
  [(#7419)](https://github.com/PennyLaneAI/pennylane/pull/7419)

* The entry in the :doc:`/news/program_capture_sharp_bits` page for using program capture with Catalyst 
  has been updated. Instead of using ``qjit(experimental_capture=True)``, Catalyst is now compatible 
  with the global toggles ``qml.capture.enable()`` and ``qml.capture.disable()`` for enabling and
  disabling program capture.
  [(#7298)](https://github.com/PennyLaneAI/pennylane/pull/7298)

<h3>Bug fixes 🐛</h3>

* A bug in `qml.draw_mpl` for circuits with work wires has been fixed. The previously
  inconsistent mapping for these wires has been resolved, ensuring accurate assignment during
  drawing.
  [(#7668)](https://github.com/PennyLaneAI/pennylane/pull/7668)

* A bug in `ops.op_math.Prod.simplify()` has been fixed that led to global phases being discarded
  in special cases. Concretely, this problem occurs when Pauli factors combine into the identity
  up to a global phase _and_ there is no Pauli representation of the product operator.
  [(#7671)](https://github.com/PennyLaneAI/pennylane/pull/7671)

* The behaviour of the `qml.FlipSign` operation has been fixed: passing an integer `m` as the wires argument is now
  interpreted as a single wire (i.e. `wires=[m]`). This is different from the previous interpretation of `wires=range(m)`.
  Also, the `qml.FlipSign.wires` attribute is now returning the correct `Wires` object as for all other operations in PennyLane.
  [(#7647)](https://github.com/PennyLaneAI/pennylane/pull/7647)

* `qml.equal` now works with `qml.PauliError`s.
  [(#7618)](https://github.com/PennyLaneAI/pennylane/pull/7618)

* The `qml.transforms.cancel_inverses` transform can be used with `jax.jit`.
  [(#7487)](https://github.com/PennyLaneAI/pennylane/pull/7487)

* `qml.StatePrep` does not validate the norm of statevectors any more, default to `False` during initialization.
  [(#7615)](https://github.com/PennyLaneAI/pennylane/pull/7615)

* `qml.PhaseShift` operation is now working correctly with a batch size of 1.
  [(#7622)](https://github.com/PennyLaneAI/pennylane/pull/7622)

* `qml.metric_tensor` can now be calculated with catalyst.
  [(#7528)](https://github.com/PennyLaneAI/pennylane/pull/7528)

* The mapping to standard wires (consecutive integers) of `qml.tape.QuantumScript` has been fixed
  to correctly consider work wires that are not used otherwise in the circuit.
  [(#7581)](https://github.com/PennyLaneAI/pennylane/pull/7581)

* Fixed a bug where certain transforms with a native program capture implementation give incorrect results when
  dynamic wires were present in the circuit. The affected transforms were:
  * :func:`~pennylane.transforms.cancel_inverses`
  * :func:`~pennylane.transforms.merge_rotations`
  * :func:`~pennylane.transforms.single_qubit_fusion`
  * :func:`~pennylane.transforms.merge_amplitude_embedding`
  [(#7426)](https://github.com/PennyLaneAI/pennylane/pull/7426)

* The `Operator.pow` method has been fixed to raise to the power of 2 the qutrit operators `~.TShift`, `~.TClock`, and `~.TAdd`.
  [(#7505)](https://github.com/PennyLaneAI/pennylane/pull/7505)

* The queuing behavior of the controlled of a controlled operation is fixed.
  [(#7532)](https://github.com/PennyLaneAI/pennylane/pull/7532)

* A new decomposition was implemented for two-qubit `QubitUnitary` operators in `two_qubit_decomposition`
  based on a type-AI Cartan decomposition. It fixes previously faulty edge cases for unitaries
  that require 2 or 3 CNOT gates. Now, 3 CNOTs are used for both cases, using one more
  CNOT than theoretically required in the former case.
  [(#7474)](https://github.com/PennyLaneAI/pennylane/pull/7474)

* The documentation of `qml.pulse.drive` has been updated and corrected.
  [(#7459)](https://github.com/PennyLaneAI/pennylane/pull/7459)

* Fixed a bug in `to_openfermion` where identity qubit-to-wires mapping was not obeyed.
  [(#7332)](https://github.com/PennyLaneAI/pennylane/pull/7332)

* Fixed a bug in the validation of :class:`~.SelectPauliRot` that prevents parameter broadcasting.
  [(#7377)](https://github.com/PennyLaneAI/pennylane/pull/7377)

* Usage of NumPy in `default.mixed` source code has been converted to `qml.math` to avoid
  unnecessary dependency on NumPy and to fix a bug that caused an error when using `default.mixed` with PyTorch and GPUs.
  [(#7384)](https://github.com/PennyLaneAI/pennylane/pull/7384)

* With program capture enabled (`qml.capture.enable()`), `QSVT` no treats abstract values as metadata.
  [(#7360)](https://github.com/PennyLaneAI/pennylane/pull/7360)

* A fix was made to `default.qubit` to allow for using `qml.Snapshot` with defer-measurements (`mcm_method="deferred"`).
  [(#7335)](https://github.com/PennyLaneAI/pennylane/pull/7335)

* Fixes the repr for empty `Prod` and `Sum` instances to better communicate the existence of an empty instance.
  [(#7346)](https://github.com/PennyLaneAI/pennylane/pull/7346)

* Fixes a bug where circuit execution fails with ``BlockEncode`` initialized with sparse matrices.
  [(#7285)](https://github.com/PennyLaneAI/pennylane/pull/7285)

* Adds an informative error if `qml.cond` is used with an abstract condition with
  jitting on `default.qubit` if capture is enabled.
  [(#7314)](https://github.com/PennyLaneAI/pennylane/pull/7314)

* Fixes a bug where using a ``StatePrep`` operation with `batch_size=1` did not work with ``default.mixed``.
  [(#7280)](https://github.com/PennyLaneAI/pennylane/pull/7280)

* Gradient transforms can now be used in conjunction with batch transforms with all interfaces.
  [(#7287)](https://github.com/PennyLaneAI/pennylane/pull/7287)

* Fixes a bug where the global phase was not being added in the ``QubitUnitary`` decomposition.  
  [(#7244)](https://github.com/PennyLaneAI/pennylane/pull/7244)
  [(#7270)](https://github.com/PennyLaneAI/pennylane/pull/7270)

* Using finite differences with program capture without x64 mode enabled now raises a warning.
  [(#7282)](https://github.com/PennyLaneAI/pennylane/pull/7282)

* When the `mcm_method` is specified to the `"device"`, the `defer_measurements` transform will 
  no longer be applied. Instead, the device will be responsible for all MCM handling.
  [(#7243)](https://github.com/PennyLaneAI/pennylane/pull/7243)

* Fixed coverage of `qml.liealg.CII` and `qml.liealg.AIII`.
  [(#7291)](https://github.com/PennyLaneAI/pennylane/pull/7291)

* Fixed a bug where the phase is used as the wire label for a `qml.GlobalPhase` when capture is enabled.
  [(#7211)](https://github.com/PennyLaneAI/pennylane/pull/7211)

* Fixed a bug that caused `CountsMP.process_counts` to return results in the computational basis, even if
  an observable was specified.
  [(#7342)](https://github.com/PennyLaneAI/pennylane/pull/7342)

* Fixed a bug that caused `SamplesMP.process_counts` used with an observable to return a list of eigenvalues 
  for each individual operation in the observable, instead of the overall result.
  [(#7342)](https://github.com/PennyLaneAI/pennylane/pull/7342)

* Fixed a bug where `two_qubit_decomposition` provides an incorrect decomposition for some special matrices.
  [(#7340)](https://github.com/PennyLaneAI/pennylane/pull/7340)

* Fixes a bug where the powers of `qml.ISWAP` and `qml.SISWAP` were decomposed incorrectly.
  [(#7361)](https://github.com/PennyLaneAI/pennylane/pull/7361)

* Returning `MeasurementValue`s from the `ftqc` module's parametric mid-circuit measurements
  (`measure_arbitrary_basis`, `measure_x` and `measure_y`) no longer raises an error in circuits 
  using `diagonalize_mcms`.
  [(#7387)](https://github.com/PennyLaneAI/pennylane/pull/7387)

* Fixes a bug where the :func:`~.transforms.single_qubit_fusion` transform produces a tape that is
  off from the original tape by a global phase.
  [(#7619)](https://github.com/PennyLaneAI/pennylane/pull/7619)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso-Linaje,
Utkarsh Azad,
Astral Cai,
Yushao Chen,
Marcus Edwards,
Lillian Frederiksen,
Pietropaolo Frisoni,
Simone Gasperini,
Korbinian Kottmann,
Christina Lee,
Anton Naim Ibrahim,
Oumarou Oumarou,
Lee J. O'Riordan,
Mudit Pandey,
Andrija Paurevic,
Shuli Shu,
Kalman Szenes,
Marc Vandelle,
David Wierichs,
Jake Zaia
