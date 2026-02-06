
# Release 0.43.0

<h3>New features since last release</h3>

<h4>A brand new resource estimation module 📖 </h4>

A new toolkit dedicated to resource estimation is now available in the :mod:`~.estimator` module!
The functionality therein is designed to rapidly and flexibly estimate the quantum resources
required to execute programs written at different levels of abstraction.  
This new module includes the following features:

* A new :func:`~.estimator.estimate.estimate` function allows users to estimate the quantum 
  resources required to execute a circuit or operation with respect to a given gate set and 
  configuration.
  [(#8203)](https://github.com/PennyLaneAI/pennylane/pull/8203)
  [(#8205)](https://github.com/PennyLaneAI/pennylane/pull/8205)
  [(#8275)](https://github.com/PennyLaneAI/pennylane/pull/8275)
  [(#8227)](https://github.com/PennyLaneAI/pennylane/pull/8227)
  [(#8279)](https://github.com/PennyLaneAI/pennylane/pull/8279)
  [(#8288)](https://github.com/PennyLaneAI/pennylane/pull/8288)
  [(#8311)](https://github.com/PennyLaneAI/pennylane/pull/8311)
  [(#8313)](https://github.com/PennyLaneAI/pennylane/pull/8313)
  [(#8360)](https://github.com/PennyLaneAI/pennylane/pull/8360)

  The :func:`~.estimator.estimate.estimate` function can be used on circuits written at different
  levels of detail to get high-level estimates of gate counts and additional wires *fast*. For 
  workflows that are already defined in detail, like executable QNodes, the 
  :func:`~.estimator.estimate.estimate` function works as follows:

  ```python
  import pennylane as qp
  import pennylane.estimator as qre

  dev = qp.device("null.qubit")

  @qp.qnode(dev)
  def circ():
      for w in range(2):
          qp.Hadamard(wires=w)
      qp.CNOT(wires=[0,1])
      qp.RX(1.23*np.pi, wires=0)
      qp.RY(1.23*np.pi, wires=1)
      qp.QFT(wires=[0, 1, 2])
      return qp.state()
  ```

  ```pycon
  >>> res = qre.estimate(circ)()
  >>> print(res)
  --- Resources: ---
    Total wires: 3
    algorithmic wires: 3
    allocated wires: 0
      zero state: 0
      any state: 0
    Total gates : 408
    'T': 396,
    'CNOT': 9,
    'Hadamard': 3
  ```

  If exact argument values and other details to operators are unknown or not available, 
  :func:`~.estimator.estimate.estimate` can also be used on new lightweight representations of 
  PennyLane operations that require minimal information to obtain high-level estimates. As part of 
  this release, many operations in PennyLane now have a corresponding lightweight version
  that inherits from a new class called :class:`~.estimator.resource_operator.ResourceOperator`,
  which can be found in the :mod:`~.estimator` module.

  For example, the lightweight representation of ``QFT`` is 
  :class:`qre.QFT <~.estimator.templates.QFT>`. By simply specifying the number of wires it acts on, 
  we can obtain resource estimates:

  ```pycon
  >>> qft = qre.QFT(num_wires=3)
  >>> res = qre.estimate(qft)
  >>> print(res)
  --- Resources: ---
    Total wires: 3
      algorithmic wires: 3
      allocated wires: 0
        zero state: 0
        any state: 0
    Total gates : 408
    'T': 396,
    'CNOT': 9,
    'Hadamard': 3
  ```
  
  One can create a circuit comprising these operations with similar syntax as defining a QNode, but 
  with far less detail. Here is an example of a circuit with 50 (logical) algorithmic qubits, which 
  includes a :class:`~.estimator.templates.QROMStatePreparation` acting on 48 qubits. Defining this 
  state preparation for execution would require a state vector of length :math:`2^{48}` (see 
  :class:`qp.QROMStatePreparation <pennylane.QROMStatePreparation>`), but we are able to estimate 
  the required resources with only metadata, bypassing this computational barrier. Even at this 
  scale, the resource estimate is computed in a fraction of a second!

  ```python
  def my_circuit():
      qre.QROMStatePreparation(num_state_qubits=48)
      for w in range(2):
          qre.Hadamard(wires=w)
      qre.QROM(num_bitstrings=32, size_bitstring=8, restored=False)
      qre.CNOT(wires=[0,1])
      qre.RX(wires=0)
      qre.RY(wires=1)
      qre.QFT(num_wires=30)
      return
  ```

  ```pycon
  >>> res = qre.estimate(my_circuit)()
  >>> print(res)
  --- Resources: ---
   Total wires: 129
    algorithmic wires: 50
    allocated wires: 79
      zero state: 71
      any state: 8
    Total gates : 2.702E+16
    'Toffoli': 1.126E+15,
    'T': 5.751E+4,
    'CNOT': 2.027E+16,
    'X': 2.252E+15,
    'Z': 32,
    'S': 64,
    'Hadamard': 3.378E+15
  ```

  Here is a summary of the lightweight operations made available in this release. A complete list
  can be found in the :mod:`~.estimator` module.
  * :class:`~.estimator.ops.Identity`, :class:`~.estimator.ops.GlobalPhase`, and various non-parametric 
    operators and single-qubit parametric operators.
    [(#8240)](https://github.com/PennyLaneAI/pennylane/pull/8240)
    [(#8242)](https://github.com/PennyLaneAI/pennylane/pull/8242)
    [(#8302)](https://github.com/PennyLaneAI/pennylane/pull/8302)
  * Various controlled single and multi qubit operators.
    [(#8243)](https://github.com/PennyLaneAI/pennylane/pull/8243)
  * :class:`~.estimator.ops.Controlled`, and :class:`~.estimator.ops.Adjoint` as symbolic operators.
    [(#8252)](https://github.com/PennyLaneAI/pennylane/pull/8252)
    [(#8349)](https://github.com/PennyLaneAI/pennylane/pull/8349)
  * :class:`~.estimator.ops.Pow`, :class:`~.estimator.ops.Prod`, 
    :class:`~.estimator.ops.ChangeOpBasis`, and parametric multi-qubit operators.
    [(#8255)](https://github.com/PennyLaneAI/pennylane/pull/8255)
  * Templates including :class:`~.estimator.templates.SemiAdder`, :class:`~.estimator.templates.QFT`, 
    :class:`~.estimator.templates.AQFT`, :class:`~.estimator.templates.BasisRotation`, 
    :class:`~.estimator.templates.Select`, :class:`~.estimator.templates.QROM`, 
    :class:`~.estimator.templates.SelectPauliRot`, :class:`~.estimator.templates.QubitUnitary`, 
    :class:`~.estimator.templates.ControlledSequence`, :class:`~.estimator.templates.QPE`,
    :class:`~.estimator.templates.IterativeQPE`, :class:`~.estimator.templates.MPSPrep`, 
    :class:`~.estimator.templates.QROMStatePreparation`, 
    :class:`~.estimator.templates.UniformStatePrep`, :class:`~.estimator.templates.AliasSampling`, 
    :class:`~.estimator.templates.IntegerComparator`, 
    :class:`~.estimator.templates.SingleQubitComparator`, 
    :class:`~.estimator.templates.TwoQubitComparator`,
    :class:`~.estimator.templates.RegisterComparator`, :class:`~.estimator.templates.SelectTHC`, 
    :class:`~.estimator.templates.PrepTHC`, and :class:`~.estimator.templates.QubitizeTHC`.
    [(#8300)](https://github.com/PennyLaneAI/pennylane/pull/8300)
    [(#8305)](https://github.com/PennyLaneAI/pennylane/pull/8305)
    [(#8309)](https://github.com/PennyLaneAI/pennylane/pull/8309)

  For defining your own customized lightweight resource operations that integrate with features in 
  the :mod:`~.estimator` module, check out the documentation for 
  :class:`~.estimator.resource_operator.ResourceOperator`.
  
* Users can define customized configurations to be used during resource estimation using the new 
  :class:`~.estimator.resource_config.ResourceConfig` class. This enables the seamless analysis of 
  tradeoffs between resources required and quantities like individual gate precisions or different
  gate decompositions. 
  [(#8259)](https://github.com/PennyLaneAI/pennylane/pull/8259)

  In the following example, a :class:`~.estimator.resource_config.ResourceConfig` is used to modify 
  the default precision of single qubit rotations, and ``T`` counts are compared between different 
  configurations.

  ```python
  def my_circuit():
      qre.RX(wires=0)
      qre.RY(wires=1)
      qre.RZ(wires=2)
      return

  my_rc = qre.ResourceConfig()
  res1 = qre.estimate(my_circuit, config=my_rc)()
  my_rc.set_single_qubit_rot_precision(1e-2)
  res2 = qre.estimate(my_circuit, config=my_rc)()
  ```

  ```pycon
  >>> t1 = res1.gate_counts['T']
  >>> t2 = res2.gate_counts['T']
  >>> print(t1, t2)
  132 51
  ```

* Hamiltonians are often both expensive to compute and to analyze, but the amount of information 
  required to estimate the resources of Hamiltonian simulation can be surprisingly small in 
  comparison. The :class:`~.estimator.compact_hamiltonian.CDFHamiltonian`,
  :class:`~.estimator.compact_hamiltonian.THCHamiltonian`,
  :class:`~.estimator.compact_hamiltonian.VibronicHamiltonian`,
  and :class:`~.estimator.compact_hamiltonian.VibrationalHamiltonian` classes were added
  to store the metadata of the Hamiltonian of a quantum system pertaining to resource estimation.
  In addition, several resource templates were added that are related to the Suzuki-Trotter method
  for Hamiltonian simulation, including :class:`~.estimator.templates.TrotterProduct`, 
  :class:`~.estimator.templates.TrotterCDF`, :class:`~.estimator.templates.TrotterTHC`,
  :class:`~.estimator.templates.TrotterVibronic`, and 
  :class:`~.estimator.templates.TrotterVibrational`.
  [(#8303)](https://github.com/PennyLaneAI/pennylane/pull/8303)

  Here's a simple example of resource estimation for the simulation of a
  :class:`~.estimator.compact_hamiltonian.CDFHamiltonian`, where we only need to specify 
  two integer arguments (``num_orbitals`` and ``num_fragments``) to get resource estimates:

  ```pycon
  >>> cdf_ham = qre.CDFHamiltonian(num_orbitals=4, num_fragments=4)
  >>> res = qre.estimate(qre.TrotterCDF(cdf_ham, num_steps=1, order=2))
  >>> print(res)
  --- Resources: ---
    Total wires: 8
      algorithmic wires: 8
      allocated wires: 0
        zero state: 0
        any state: 0
    Total gates : 2.238E+4
    'T': 2.075E+4,
    'CNOT': 448,
    'Z': 336,
    'S': 504,
    'Hadamard': 336
  ```

* In addition to the :class:`~.estimator.resource_operator.ResourceOperator` class mentioned above,
  the scalability of the resource estimation functionality in this release 
  is owed to the following new internal classes:    

  * :class:`~.estimator.resources_base.Resources`: A container for counts and other metadata of 
    quantum resources.
    [(#8205)](https://github.com/PennyLaneAI/pennylane/pull/8205)
  * :class:`~.estimator.resource_operator.GateCount`: A class to represent a gate and its number 
    of occurrences in a circuit or decomposition.
  * :class:`~.estimator.resource_operator.CompressedResourceOp`: A lightweight class corresponding 
    to an operator type alongside its parameters.
    [(#8227)](https://github.com/PennyLaneAI/pennylane/pull/8227)
  * The :class:`~.estimator.wires_manager.WireResourceManager`,
    :class:`~.estimator.wires_manager.Allocate`, and :class:`~.estimator.wires_manager.Deallocate`
    classes, which were added to manage and track wire usage during resource estimation and within 
    :class:`~.estimator.resource_operator.ResourceOperator` definitions.
    [(#8203)](https://github.com/PennyLaneAI/pennylane/pull/8203)

The resource estimation tools in the :mod:`~.estimator` module were originally prototyped in the 
:mod:`~.labs` module. Check it out too for the latest cutting-edge research functionality!

<h4>Dynamic wire allocation 🎁</h4>

* Wires can now be dynamically allocated and deallocated in quantum functions with
  the :func:`~.allocate` and :func:`~.deallocate` functions. These features unlock many important applications
  that rely on smart and efficient handling of wires, such as decompositions of gates that require
  auxiliary wires and logical patterns in subroutines that benefit from having dynamic wire 
  management.

  [(#7718)](https://github.com/PennyLaneAI/pennylane/pull/7718)
  [(#8151)](https://github.com/PennyLaneAI/pennylane/pull/8151)
  [(#8163)](https://github.com/PennyLaneAI/pennylane/pull/8163)
  [(#8179)](https://github.com/PennyLaneAI/pennylane/pull/8179)
  [(#8198)](https://github.com/PennyLaneAI/pennylane/pull/8198)
  [(#8381)](https://github.com/PennyLaneAI/pennylane/pull/8381)

  The :func:`~.allocate` function can accept three arguments that dictate how dynamically allocated
  wires are handled:

  * ``num_wires``: the number of wires to dynamically allocate.
  * ``state = "zero"/"any"``: the initial state that the dynamically allocated wires are requested 
    to be in. Currently, supported values are ``"zero"`` (initialize in the all-zero state) or 
    ``"any"`` (any arbitrary state).
  * ``restored = True/False``: a user-guarantee that the allocated wires will be restored to their
    original state (``True``) or not (``False``) when those wires are deallocated.

  The recommended way to safely allocate and deallocate wires is to use :func:`~.allocate` as a
  context manager:

  ```python
  import pennylane as qp

  @qp.qnode(qp.device("default.qubit"))
  def circuit():
      qp.H(0)
      qp.H(1)

      with qp.allocate(2, state="zero", restored=False) as new_wires:
          qp.H(new_wires[0])
          qp.H(new_wires[1])

      return qp.expval(qp.Z(0))
  ```

  ```pycon
  >>> print(qp.draw(circuit)())
              0: ──H───────────────────────┤  <Z>
              1: ──H───────────────────────┤
  <DynamicWire>: ─╭Allocate──H─╭Deallocate─┤
  <DynamicWire>: ─╰Allocate──H─╰Deallocate─┤
  ```

  As illustrated, using :func:`~.allocate` as a context manager ensures that allocation and safe
  deallocation are controlled within a localized scope. Equivalenty, :func:`~.allocate` can be used
  in-line along with :func:`~.deallocate` for manual handling:

  ```python
  new_wires = qp.allocate(2, state="zero", restored=False)
  qp.H(new_wires[0])
  qp.H(new_wires[1])
  qp.deallocate(new_wires)
  ```

  For more complex dynamic allocation in circuits, PennyLane will resolve the dynamic allocation
  calls in the most resource-efficient manner before sending the program to the device. Consider the
  following circuit, which contains two dynamic allocations within a ``for`` loop.

  ```python
  @qp.qnode(qp.device("default.qubit"), mcm_method="tree-traversal")
  def circuit():
      qp.H(0)

      for i in range(2):
          with qp.allocate(1, state="zero", restored=True) as new_qubit1:
              with qp.allocate(1, state="any", restored=False) as new_qubit2:
                  m0 = qp.measure(new_qubit1[0], reset=True)
                  qp.cond(m0 == 1, qp.Z)(new_qubit2[0])
                  qp.CNOT((0, new_qubit2[0]))

      return qp.expval(qp.Z(0))
  ```

  ```pycon
  >>> print(qp.draw(circuit)())
              0: ──H─────────────────────╭●───────────────────────╭●─────────────┤  <Z>
  <DynamicWire>: ──Allocate──┤↗│  │0⟩────│──────────Deallocate────│──────────────┤
  <DynamicWire>: ──Allocate───║────────Z─╰X─────────Deallocate────│──────────────┤
  <DynamicWire>: ─────────────║────────║──Allocate──┤↗│  │0⟩──────│───Deallocate─┤
  <DynamicWire>: ─────────────║────────║──Allocate───║──────────Z─╰X──Deallocate─┤
                              ╚════════╝             ╚══════════╝
  ```

  The user-level circuit drawing shows four separate allocations and deallocations (two per loop
  iteration). However, the circuit that the device receives gets automatically compiled to only use
  **two** additional wires (wires labelled ``1`` and ``2`` in the diagram below). This is due to the
  fact that ``new_qubit1`` and ``new_qubit2`` can both be reused after they've been deallocated in
  the first iteration of the ``for`` loop:

  ```
  >>> print(qp.draw(circuit, level="device")())
  0: ──H───────────╭●──────────────╭●─┤  <Z>
  1: ──┤↗│  │0⟩────│───┤↗│  │0⟩────│──┤
  2: ───║────────Z─╰X───║────────Z─╰X─┤
        ╚════════╝      ╚════════╝
  ```

  Additionally, :func:`~.allocate` and :func:`~.deallocate` work with :func:`~.qjit` with 
  [some restrictions](https://docs.pennylane.ai/projects/catalyst/en/stable/dev/sharp_bits.html#functionality-differences-from-pennylane).

<h4>Resource tracking with Catalyst 🧾</h4>

* Users can now use the :func:`~.specs` function to track the exact resources of programs compiled 
  with :func:`~.qjit`!  
  This new feature is currently only supported when using ``level="device"``.
  [(#8202)](https://github.com/PennyLaneAI/pennylane/pull/8202)

  ```python
  from functools import partial

  gateset = {qp.H, qp.S, qp.CNOT, qp.T, qp.RX, qp.RY, qp.RZ}

  @qp.qjit
  @partial(qp.transforms.decompose, gate_set=gateset)
  @qp.qnode(qp.device("null.qubit", wires=100))
  def circuit():
      qp.QFT(wires=range(100))
      qp.Hadamard(wires=0)
      qp.CNOT(wires=[0, 1])
      qp.OutAdder(
                  x_wires=range(10),
                  y_wires=range(10,20),
                  output_wires=range(20,31)
                  )
      return qp.expval(qp.Z(0) @ qp.Z(1))

  circ_specs = qp.specs(circuit, level="device")()
  ```

  ```pycon
  >>> print(circ_specs['resources'])
  num_wires: 100
  num_gates: 138134
  depth: 90142
  shots: Shots(total=None)
  gate_types:
  {'CNOT': 55313, 'RZ': 82698, 'Hadamard': 123}
  gate_sizes:
  {2: 55313, 1: 82821}
  ```

* The :func:`~.specs` function now accepts a ``compute_depth`` keyword argument, which is set to 
  ``True`` by default. Since depth computation is usually the most expensive resource to calculate, 
  making it optional can increase the performance of :func:`~.specs` when depth is not a desired
  resource to calculate.
  [(#7998)](https://github.com/PennyLaneAI/pennylane/pull/7998)
  [(#8042)](https://github.com/PennyLaneAI/pennylane/pull/8042)

<h4>ZX Calculus transforms 🍪</h4>

* A new set of transforms enable ZX calculus-based circuit optimization.
  These transforms make it easy to implement advanced compilation techniques that use the
  [ZX calculus graphical language](https://pennylane.ai/qml/demos/tutorial_zx_calculus) to
  reduce T-gate counts of 
  [Clifford + T circuits](https://pennylane.ai/compilation/clifford-t-gate-set), optimize 
  [phase polynomials](https://pennylane.ai/compilation/phase-polynomial-intermediate-representation),
  and reduce the number of gates in non-Clifford circuits.
  [(#8025)](https://github.com/PennyLaneAI/pennylane/pull/8025)
  [(#8029)](https://github.com/PennyLaneAI/pennylane/pull/8029)
  [(#8088)](https://github.com/PennyLaneAI/pennylane/pull/8088)
  [(#7747)](https://github.com/PennyLaneAI/pennylane/pull/7747)
  [(#8201)](https://github.com/PennyLaneAI/pennylane/pull/8201)

  These transforms include:

  * :func:`~.transforms.zx.optimize_t_count`: reduces the number of ``T`` gates in a Clifford + T 
    circuit by applying a sequence of passes that combine ZX-based commutation and cancellation 
    rules and the
    [Third Order Duplicate and Destroy (TODD)](https://pennylane.ai/compilation/phase-polynomial-intermediate-representation/compilation#t-gate-optimization) 
    algorithm.

  * :func:`~.transforms.zx.todd`: reduces the number of ``T`` gates in a Clifford + T circuit by 
    using the TODD algorithm. 

  * :func:`~.transforms.zx.reduce_non_clifford`: reduces the number of non-Clifford gates in a 
    circuit by applying a combination of phase gadgetization strategies and Clifford gate 
    simplification rules.

  * :func:`~.transforms.zx.push_hadamards`: reduces the number of large
    [phase-polynomial](https://pennylane.ai/compilation/phase-polynomial-intermediate-representation) 
    blocks in a phase-polynomial + Hadamard circuit by pushing Hadamard gates as far as possible to 
    one side.

  As an example, consider the following circuit:

  ```python
  import pennylane as qp

  dev = qp.device("default.qubit")

  @qp.qnode(dev)
  def circuit():
      qp.T(0)
      qp.CNOT([0, 1])
      qp.S(0)
      qp.T(0)
      qp.T(1)
      qp.CNOT([0, 2])
      qp.T(1)
      return qp.state()
  ```

  ```pycon
  >>> print(qp.draw(circuit)())
  0: ──T─╭●──S──T─╭●────┤  State
  1: ────╰X──T────│───T─┤  State
  2: ─────────────╰X────┤  State
  ```

  We can apply the holistic :func:`~.transforms.zx.optimize_t_count` compilation pass to
  reduce the number of ``T`` gates. In this case, all ``T`` gates can be removed!

  ```pycon
  >>> print(qp.draw(qp.transforms.zx.optimize_t_count(circuit))())
  0: ──Z─╭●────╭●─┤  State
  1: ────╰X──S─│──┤  State
  2: ──────────╰X─┤  State
  ```

<h4>Change operator bases 🍴</h4>

* Users can now benefit from an optimization of the controlled compute-uncompute pattern with
  the new :func:`~.change_op_basis` function and :class:`~.ops.op_math.ChangeOpBasis` class.
  Operators arranged in a compute-uncompute pattern (``U V U†``, which is equivalent to changing the 
  basis in which ``V`` is expressed) can be efficiently controlled, as the only the central (target) 
  operator ``V`` needs to be controlled, and not ``U`` or ``U†``.
  [(#8023)](https://github.com/PennyLaneAI/pennylane/pull/8023)
  [(#8070)](https://github.com/PennyLaneAI/pennylane/pull/8070)
  
  These new features leverage the graph-based decomposition system, enabled with 
  :func:`~.decomposition.enable_graph()`. To illustrate their use, consider the following example. 
  The compute-uncompute pattern is composed of a ``QFT``, followed by a ``PhaseAdder``, and finally 
  an inverse ``QFT``.

  ```python
  from functools import partial

  qp.decomposition.enable_graph()

  dev = qp.device("default.qubit")

  @partial(qp.transforms.decompose, max_expansion=1)
  @qp.qnode(dev)
  def circuit():
      qp.H(0)
      qp.CNOT([1,2])
      qp.ctrl(
          qp.change_op_basis(qp.QFT([1,2]), qp.PhaseAdder(1, x_wires=[1,2])),
          control=0
      )
      return qp.state()
  ```

  When this circuit is decomposed, the ``QFT`` and ``Adjoint(QFT)`` are not controlled, resulting in 
  a much more resource-efficient decomposition:

  ```pycon
  >>> print(qp.draw(circuit)())
  0: ──H──────╭●────────────────┤  State
  1: ─╭●─╭QFT─├PhaseAdder─╭QFT†─┤  State
  2: ─╰X─╰QFT─╰PhaseAdder─╰QFT†─┤  State
  ```

* The decompositions for several templates have been updated to use 
  :class:`~.ops.op_math.ChangeOpBasis`, which makes their decompositions more resource efficient by
  eliminating unnecessary controlled operations. The templates include :class:`~.Adder`, 
  :class:`~.Multiplier`, :class:`~.OutAdder`, :class:`~.OutMultiplier`, :class:`~.PrepSelPrep`.
  [(#8207)](https://github.com/PennyLaneAI/pennylane/pull/8207)

  Here, the optimization is demonstrated when :class:`~.Adder` is controlled:

  ```python
  qp.decomposition.enable_graph()

  dev = qp.device("default.qubit")

  @partial(qp.transforms.decompose, max_expansion=2)
  @qp.qnode(dev)
  def circuit():
      qp.ctrl(qp.Adder(10, x_wires=[1,2,3,4]), control=0)
      return qp.state()
  ```

  ```pycon
  >>> print(qp.draw(circuit)())
  0: ──────╭●────────────────┤  State
  1: ─╭QFT─├PhaseAdder─╭QFT†─┤  State
  2: ─├QFT─├PhaseAdder─├QFT†─┤  State
  3: ─├QFT─├PhaseAdder─├QFT†─┤  State
  4: ─╰QFT─╰PhaseAdder─╰QFT†─┤  State
  ```

<h4>Quantum optimizers compatible with QJIT 🫖</h4>

* Leveraging :func:`~.qjit` to optimize hybrid workflows with the momentum quantum natural gradient 
  optimizer is now possible with :class:`~.MomentumQNGOptimizerQJIT`. This provides better scaling
  than its non-JIT-compatible counterpart.
  [(#7606)](https://github.com/PennyLaneAI/pennylane/pull/7606)

  The v0.42 release saw the addition of the :class:`~.QNGOptimizerQJIT` optimizer, which is a 
  ``qp.qjit``-compatible analogue to :class:`~.QNGOptimizer`. In this release, we've added the 
  :class:`~.MomentumQNGOptimizerQJIT` optimizer, which is the ``qp.qjit``-compatible analogue to 
  :class:`~.MomentumQNGOptimizer`. Both optimizers have an 
  [Optax](https://optax.readthedocs.io/en/stable/getting_started.html#basic-usage-of-optax)-like 
  interface:

  ```python
  import jax.numpy as jnp

  dev = qp.device("lightning.qubit", wires=2)

  @qp.qnode(dev)
  def circuit(params):
      qp.RX(params[0], wires=0)
      qp.RY(params[1], wires=1)
      return qp.expval(qp.Z(0) + qp.X(1))

  opt = qp.MomentumQNGOptimizerQJIT(stepsize=0.1, momentum=0.2)

  @qp.qjit
  def update_step_qjit(i, args):
      params, state = args
      return opt.step(circuit, params, state)

  @qp.qjit
  def optimization_qjit(params, iters):
      state = opt.init(params)
      args = (params, state)
      params, state = qp.for_loop(iters)(update_step_qjit)(args)
      return params
  ```

  Quantum just-in-time compilation works exceptionally well with repeatedly executing the same 
  function in a ``for`` loop. As you can see, :math:`10^5` iterations takes seconds:

  ```pycon
  >>> import time
  >>> params = jnp.array([0.1, 0.2])
  >>> iters = 100_000
  >>> start = time.process_time()
  >>> optimization_qjit(params=params, iters=iters)
  Array([ 3.14159265, -1.57079633], dtype=float64)
  >>> time.process_time() - start
  21.319525
  ```

<h3>Improvements 🛠</h3>

<h4>Resource-efficient decompositions</h4>

* With :func:`~.decomposition.enable_graph()`, dynamically allocated wires with :func:`~.allocate` 
  are now supported in decomposition rules. This provides a smoother overall experience when 
  decomposing operators in a way that requires auxiliary/work wires.
  [(#7861)](https://github.com/PennyLaneAI/pennylane/pull/7861)
  [(#8228)](https://github.com/PennyLaneAI/pennylane/pull/8228)

  Support for :func:`~.allocate` unlocks the following features:

  * The :func:`~.transforms.decompose` transform now accepts a ``max_work_wires`` argument that 
    allows the user to specify the number of work wires available for dynamic allocation during 
    decomposition.
    [(#7963)](https://github.com/PennyLaneAI/pennylane/pull/7963)
    [(#7980)](https://github.com/PennyLaneAI/pennylane/pull/7980)
    [(#8103)](https://github.com/PennyLaneAI/pennylane/pull/8103)
    [(#8236)](https://github.com/PennyLaneAI/pennylane/pull/8236)

  * Decomposition rules were added for the :class:`~.MultiControlledX` that dynamically allocate 
    work wires if none were explicitly specified via the ``work_wires`` argument.
    [(#8024)](https://github.com/PennyLaneAI/pennylane/pull/8024)

* Several templates now have decompositions that can be accessed within the graph-based
  decomposition system (:func:`~.decomposition.enable_graph`), allowing workflows
  that include these templates to be decomposed in a resource-efficient and performant
  manner.
  [(#7779)](https://github.com/PennyLaneAI/pennylane/pull/7779)
  [(#7908)](https://github.com/PennyLaneAI/pennylane/pull/7908)
  [(#7941)](https://github.com/PennyLaneAI/pennylane/pull/7941)
  [(#7943)](https://github.com/PennyLaneAI/pennylane/pull/7943)
  [(#8075)](https://github.com/PennyLaneAI/pennylane/pull/8075)
  [(#8002)](https://github.com/PennyLaneAI/pennylane/pull/8002)

  The included templates are: :class:`~.Adder`, :class:`~.ControlledSequence`, :class:`~.ModExp`, 
  :class:`~.MottonenStatePreparation`, :class:`~.MPSPrep`, :class:`~.Multiplier`, 
  :class:`~.OutAdder`, :class:`~.OutMultiplier`, :class:`~.OutPoly`, :class:`~.PrepSelPrep`, 
  :class:`~.ops.Prod`, :class:`~.Reflection`, :class:`~.StatePrep`, :class:`~.TrotterProduct`, 
  :class:`~.QROM`, :class:`~.GroverOperator`, :class:`~.UCCSD`, :class:`~.StronglyEntanglingLayers`, 
  :class:`~.GQSP`, :class:`~.FermionicSingleExcitation`, :class:`~.FermionicDoubleExcitation`, 
  :class:`~.QROM`, :class:`~.ArbitraryStatePreparation`, :class:`~.CosineWindow`, 
  :class:`~.AmplitudeAmplification`, :class:`~.Permute`, :class:`~.AQFT`, :class:`~.FlipSign`, 
  :class:`~.FABLE`, :class:`~.Qubitization`, and :class:`~.Superposition`.

* Two additions were made to :class:`~.Select`, significantly improving its decomposition:

  * A new keyword argument ``partial`` has been added, which allows for simplifications in the 
    decomposition of :class:`~.Select` under the assumption that the state of the control wires has 
    no overlap with computational basis states that are not used by :class:`~.Select`.

  * A new decomposition rule has been added to :class:`~.Select`. It achieves cost reductions by 
    adding one ``work_wire``. This decomposition is useful to perform efficient :class:`~.QROM` 
    decompositions.

  [(#7385)](https://github.com/PennyLaneAI/pennylane/pull/7385)
  [(#7658)](https://github.com/PennyLaneAI/pennylane/pull/7658)
  [(#8011)](https://github.com/PennyLaneAI/pennylane/pull/8011)
  [(#8276)](https://github.com/PennyLaneAI/pennylane/pull/8276)

* The decomposition of :class:`~.BasisRotation` has been optimized to skip redundant phase shift 
  gates with angle :math:`\pm \pi` for real-valued (orthogonal) rotation matrices. This uses the 
  fact that either one or zero :class:`~.PhaseShift` gates are required in case the matrix has a determinant 
  equal to :math:`\pm 1`.
  [(#7765)](https://github.com/PennyLaneAI/pennylane/pull/7765)

* The :func:`~.transforms.decompose` transform is now able to decompose classically controlled 
  operations (i.e., operations nested inside ``cond``).
  [(#8145)](https://github.com/PennyLaneAI/pennylane/pull/8145)

  ```python
  from functools import partial

  dev = qp.device('default.qubit')

  @partial(qp.transforms.decompose, gate_set={qp.RY, qp.RZ, qp.measurements.MidMeasureMP})
  @qp.qnode(dev)
  def circuit():
      m0 = qp.measure(0)
      qp.cond(m0 == 0, qp.Rot)(qp.numpy.pi / 2, qp.numpy.pi / 2, qp.numpy.pi / 2, wires=1)
      return qp.expval(qp.X(0))
  ```

  ```pycon
  >>> print(qp.draw(circuit, level=0)())
  0: ──┤↗├──────────────────────┤  <X>
  1: ───║───Rot(1.57,1.57,1.57)─┤     
        ╚═══╝  
  >>> print(qp.draw(circuit, level=1)())
  0: ──┤↗├───────────────────────────────┤  <X>
  1: ───║───RZ(1.57)──RY(1.57)──RZ(1.57)─┤     
        ╚═══╩═════════╩═════════╝              
  ```

* Various decompositions of :class:`~.MultiControlledX` now utilize :class:`~.TemporaryAND` in
  place of :class:`~.Toffoli` gates, leading to more resource-efficient decompositions.
  [(#8172)](https://github.com/PennyLaneAI/pennylane/pull/8172)

* ``Controlled(Identity)`` is now directly decomposed to a single Identity operator instead of going
  through a numeric decomposition algorithm.
  [(#8388)](https://github.com/PennyLaneAI/pennylane/pull/8388)

* The internal assignment of basis states in :class:`~.Superposition` was improved, resulting in its
  decomposition being more performant and efficient.
  [(#7880)](https://github.com/PennyLaneAI/pennylane/pull/7880)

* :func:`~.decomposition.has_decomp` and :func:`~.decomposition.list_decomps` now take operator 
  instances as arguments instead of types.
  [(#8286)](https://github.com/PennyLaneAI/pennylane/pull/8286)

  ```pycon
  >>> qp.decomposition.has_decomp(qp.MultiControlledX)
  True
  >>> qp.decomposition.list_decomps(qp.Select)
  [<pennylane.decomposition.decomposition_rule.DecompositionRule at 0x126f99ed0>,
  <pennylane.decomposition.decomposition_rule.DecompositionRule at 0x127002fd0>,
  <pennylane.decomposition.decomposition_rule.DecompositionRule at 0x127034bd0>]
  ```

* With the graph-based decomposition system enabled (:func:`~.decomposition.enable_graph()`), if a 
  decomposition cannot be found for an operator in the circuit in terms of the target gates, it no 
  longer raises an error. Instead, a warning is raised, and ``op.decomposition()`` (the current 
  default method for decomposing gates) is used as a fallback, while the rest of the circuit is 
  still decomposed with the new graph-based system. Additionally, a special warning message is
  raised if the circuit contains a ``GlobalPhase``, reminding the user that ``GlobalPhase`` is not 
  assumed to have a decomposition under the new system.
  [(#8156)](https://github.com/PennyLaneAI/pennylane/pull/8156)

* :func:`~.transforms.decompose` and :func:`~.preprocess.decompose` now have a unified internal 
  implementation to promote feature parity in preparation for the graph-based decomposition system 
  to be the default decomposition method in PennyLane.
  [(#8193)](https://github.com/PennyLaneAI/pennylane/pull/8193)

* A new class called :class:`~.decomposition.decomposition_graph.DecompGraphSolution` has been added 
  to store the solution of a decomposition graph. An instance of this class is returned from the 
  ``solve`` method of the :class:`~.decomposition.decomposition_graph.DecompositionGraph` class.
  [(#8031)](https://github.com/PennyLaneAI/pennylane/pull/8031)

<h4>OpenQASM-PennyLane interoperability</h4>

* The :func:`~.from_qasm3` function can now convert OpenQASM 3.0 circuits that contain
  subroutines, constants, all remaining stdlib gates, qubit registers, and built-in mathematical 
  functions.
  [(#7651)](https://github.com/PennyLaneAI/pennylane/pull/7651)
  [(#7653)](https://github.com/PennyLaneAI/pennylane/pull/7653)
  [(#7676)](https://github.com/PennyLaneAI/pennylane/pull/7676)
  [(#7677)](https://github.com/PennyLaneAI/pennylane/pull/7677)
  [(#7679)](https://github.com/PennyLaneAI/pennylane/pull/7679)
  [(#7690)](https://github.com/PennyLaneAI/pennylane/pull/7690)
  [(#7767)](https://github.com/PennyLaneAI/pennylane/pull/7767)

* :func:`~.to_openqasm` now supports mid-circuit measurements and conditionals of unprocessed 
  measurement values.
  [(#8210)](https://github.com/PennyLaneAI/pennylane/pull/8210)

<h4>Setting shots</h4>

* The number of ``shots`` can now be specified directly in QNodes as a standard keyword argument.
  [(#8073)](https://github.com/PennyLaneAI/pennylane/pull/8073)

  ```python
  @qp.qnode(qp.device("default.qubit"), shots=1000)
  def circuit():
      qp.H(0)
      return qp.expval(qp.Z(0))
  ```

  ```pycon
  >>> circuit.shots
  Shots(total_shots=1000, shot_vector=(ShotCopies(1000 shots x 1),))  
  >>> circuit()
  np.float64(-0.004)
  ```

  Setting the ``shots`` value in a QNode is equivalent to decorating with 
  :func:`~.workflow.set_shots`. However, decorating with :func:`~.workflow.set_shots` overrides 
  QNode ``shots``:

  ```pycon
  >>> new_circ = qp.set_shots(circuit, shots=123)
  >>> new_circ.shots
  Shots(total_shots=123, shot_vector=(ShotCopies(123 shots x 1),))
  ```

* The :func:`~pennylane.set_shots` transform can now be directly applied to a QNode without the need 
  for ``functools.partial``, providing a more user-friendly syntax and negating having to import the 
  ``functools`` package.
  [(#7876)](https://github.com/PennyLaneAI/pennylane/pull/7876)
  [(#7919)](https://github.com/PennyLaneAI/pennylane/pull/7919)

  ```python
  @qp.set_shots(shots=1000)  # or @qp.set_shots(1000)
  @qp.qnode(dev)
  def circuit():
      qp.H(0)
      return qp.expval(qp.Z(0))
  ```

  ```pycon
  >>> circuit()
  0.002
  ```

<h4>Clifford + T decomposition</h4>

* The :func:`~.clifford_t_decomposition` transform with ``method="gridsynth"`` is now compatible
  with quantum just-in-time compilation via the :func:`~.qjit` decorator.
  [(#7711)](https://github.com/PennyLaneAI/pennylane/pull/7711)

  ```python
  @qp.qjit
  @partial(qp.transforms.clifford_t_decomposition, method="gridsynth")
  @qp.qnode(qp.device("lightning.qubit", wires=1))
  def circuit():
      qp.RX(np.pi/3, wires=0)
      qp.RY(np.pi/4, wires=0)
      return qp.state()
  ```

  ```pycon
  >>> circuit()
  Array([0.80011651+0.19132132j, 0.33140586-0.4619306j ], dtype=complex128)
  ```

* The :func:`~.clifford_t_decomposition` transform can now decompose circuits with mid-circuit
  measurements, including Catalyst's measurement operations. It also now handles ``RZ`` and 
  ``PhaseShift`` operations where angles are odd multiples of :math:`\pm \tfrac{\pi}{4}` more 
  efficiently when using ``method="gridsynth"``.
  [(#7793)](https://github.com/PennyLaneAI/pennylane/pull/7793)
  [(#7942)](https://github.com/PennyLaneAI/pennylane/pull/7942)

* The :func:`~.ops.rs_decomposition` method now gives decompositions with exact global phase
  information. 
  [(#7793)](https://github.com/PennyLaneAI/pennylane/pull/7793)

* Users can now specify a relative threshold value for the permissible operator norm error 
  (``epsilon``) that triggers rebuilding of the cache in the :func:`~.clifford_t_decomposition`, via 
  new ``cache_eps_rtol`` keyword argument.
  [(#8056)](https://github.com/PennyLaneAI/pennylane/pull/8056)

<h4>Transforms</h4>

* New transforms called :func:`~.transforms.match_relative_phase_toffoli` and
  :func:`~.transforms.match_controlled_iX_gate` have been added, which compile certain patterns to 
  efficient Clifford + T equivalents.
  [(#7748)](https://github.com/PennyLaneAI/pennylane/pull/7748)

  ```python
  @qp.qnode(qp.device("default.qubit", wires=4))
  def circuit():
    qp.CCZ(wires=[0, 1, 3])
    qp.ctrl(qp.S(wires=[1]), control=[0])
    qp.ctrl(qp.S(wires=[2]), control=[0, 1])
    qp.MultiControlledX(wires=[0, 1, 2, 3])

    return qp.expval(qp.Z(0))
  ```

  ```pycon
  >>> new_circuit = qp.transforms.match_relative_phase_toffoli(circuit)
  >>> print(qp.draw(new_circuit, level=1)())
  0: ─────────────────╭●───────────╭●───────────────────────────┤  <Z>
  1: ─────────────────│─────╭●─────│─────╭●─────────────────────┤
  2: ───────╭●────────│─────│──────│─────│────────────╭●────────┤
  3: ──H──T─╰X──T†──H─╰X──T─╰X──T†─╰X──T─╰X──T†──H──T─╰X──T†──H─┤
  ```

* New intermediate representations (IRs) called :func:`~transforms.parity_matrix` and 
  :func:`~transforms.phase_polynomial` are available in PennyLane. These IRs are used in compilation 
  passes to optimize ``CNOT`` and phase polynomial circuits, respectively. Additionally, the 
  :func:`~transforms.rowcol` has been added, which uses the parity matrix as its IR for ``CNOT``
  routing under constraint connectivity.
  [(#8171)](https://github.com/PennyLaneAI/pennylane/pull/8171)
  [(#8443)](https://github.com/PennyLaneAI/pennylane/pull/8443)

  The example below showcases the use of :func:`~transforms.parity_matrix`, which acts on circuits 
  containing only ``CNOT`` gates. 

  ```python
  dev = qp.device('default.qubit', wires=1)

  @qp.qnode(dev)
  def circuit():
      qp.CNOT((3, 2))
      qp.CNOT((0, 2))
      qp.CNOT((2, 1))
      qp.CNOT((3, 2))
      qp.CNOT((3, 0))
      qp.CNOT((0, 2))
      return qp.state()
  ```

  Upon transforming the above circuit with :func:`~transforms.parity_matrix`, the output is the 
  parity matrix.

  ```pycon
  >>> P = qp.transforms.parity_matrix(circuit, wire_order=range(4))()
  >>> print(P)
  array([[1, 0, 0, 1],
         [1, 1, 1, 1],
         [0, 0, 1, 1],
         [0, 0, 0, 1]])
  ```

  The :func:`~transforms.phase_polynomial` transform functions similarly, operating on circuits 
  containining only ``CNOT`` and ``RZ`` gates and returning the parity matrix, the parity table, and 
  corresponding angles for each parity.

  ```python
  @qp.qnode(dev)
  def circuit():
      qp.CNOT((1, 0))
      qp.RZ(1, 0)
      qp.CNOT((2, 0))
      qp.RZ(2, 0)
      qp.CNOT((0, 1))
      qp.CNOT((3, 1))
      qp.RZ(3, 1)
      return qp.state()
  ```

  ```pycon
  >>> pmat, ptab, angles = qp.transforms.phase_polynomial(circuit, wire_order=range(4))()
  >>> pmat
  [[1 1 1 0]
   [1 0 1 1]
   [0 0 1 0]
   [0 0 0 1]]
  >>> ptab
  [[1 1 1]
   [1 1 0]
   [0 1 1]
   [0 0 1]]
  >>> angles
  [1 2 3]
  ```

* A new transform called :func:`~.transforms.rz_phase_gradient` has been added, which lets you 
  realize arbitrary angle :class:`~.RZ` rotations with a phase gradient resource state and 
  semi-in-place addition (:class:`~.SemiAdder`). This can be a crucial subroutine in FTQC when 
  sufficient auxiliary wires are available, as it saves on ``T`` gates compared to other
  discretization schemes.
  [(#8213)](https://github.com/PennyLaneAI/pennylane/pull/8213)

* A new keyword argument called ``shot_dist`` has been added to the 
  :func:`~.transforms.split_non_commuting` transform. This allows for more customization and 
  efficiency when calculating expectation values across the non-commuting groups of observables that 
  make up a ``Hamiltonian``/``LinearCombination``.
  [(#7988)](https://github.com/PennyLaneAI/pennylane/pull/7988)

  Given a QNode that returns a sample-based measurement (e.g., ``expval``) of a 
  ``Hamiltonian``/``LinearCombination`` with finite ``shots``, the current default behaviour of 
  :func:`~.transforms.split_non_commuting` will perform ``shots`` executions for each group of 
  commuting terms. With the ``shot_dist`` argument, this behaviour can be changed:

  * ``"uniform"``: evenly distributes the number of ``shots`` across all groups of commuting terms
  * ``"weighted"``: distributes the number of ``shots`` according to weights proportional to the L1 
    norm of the coefficients in each group
  * ``"weighted_random"``: same as ``"weighted"``, but the numbers of ``shots`` are sampled from a 
    multinomial distribution
  * or a user-defined function implementing a custom shot distribution strategy

  To show an example about how this works, let's start by defining a simple Hamiltonian:

  ```python
  ham = qp.Hamiltonian(
      coeffs=[10, 0.1, 20, 100, 0.2],
      observables=[
          qp.X(0) @ qp.Y(1),
          qp.Z(0) @ qp.Z(2),
          qp.Y(1),
          qp.X(1) @ qp.X(2),
          qp.Z(0) @ qp.Z(1) @ qp.Z(2)
      ]
  )
  ```

  This Hamiltonian can be split into 3 non-commuting groups of mutually commuting terms.
  With ``shot_dist = "weighted"``, for example, the number of shots will be divided
  according to the L1 norm of each group's coefficients:

  ```python
  from functools import partial
  from pennylane.transforms import split_non_commuting

  dev = qp.device("default.qubit")

  @partial(split_non_commuting, shot_dist="weighted")
  @qp.qnode(dev, shots=10000)
  def circuit():
      return qp.expval(ham)

  with qp.Tracker(dev) as tracker:
      circuit()
  ```

  ```pycon
  >>> print(tracker.history["shots"])
  [2303, 23, 7674]
  ```

* The :func:`~.noise.fold_global` transform has been refactored to collect operators into a list 
  directly rather than relying on queuing.
  [(#8296)](https://github.com/PennyLaneAI/pennylane/pull/8296)

<h4>Choi matrix functionality</h4>

* A new function called :func:`~.math.choi_matrix` is available, which computes the 
  [Choi matrix](https://en.wikipedia.org/wiki/Choi%E2%80%93Jamio%C5%82kowski_isomorphism) of a 
  quantum channel. This is a useful tool in quantum information science and to check circuit 
  identities involving non-unitary operations.
  [(#7951)](https://github.com/PennyLaneAI/pennylane/pull/7951)

  ```pycon
  >>> import numpy as np
  >>> Ks = [np.sqrt(0.3) * qp.CNOT((0, 1)), np.sqrt(1-0.3) * qp.X(0)]
  >>> Ks = [qp.matrix(op, wire_order=range(2)) for op in Ks]
  >>> Lambda = qp.math.choi_matrix(Ks)
  >>> np.trace(Lambda), np.trace(Lambda @ Lambda)
  (np.float64(1.0), np.float64(0.58))
  ```

<h4>Other improvements</h4>

* :func:`~.snapshots` can now be used with ``mcm_method="one-shot"`` and 
  ``mcm_method="tree-traversal"``.
  [(#8140)](https://github.com/PennyLaneAI/pennylane/pull/8140)

  This improvement is particularly useful for extracting the state in finite-shot workflows:

  ```python
  @qp.qnode(qp.device("default.qubit"), mcm_method="one-shot", shots=1)
  def circuit():
      qp.RY(1.23, 0)

      m0 = qp.measure(0)
      qp.cond(m0 == 0, qp.H)(0)
      qp.Snapshot("state", measurement=qp.state())

      return qp.expval(qp.X(0))
  ```

  ```pycon
  >>> qp.snapshots(circuit)()
  {'state': array([0.+0.j, 1.+0.j]), 'execution_results': np.float64(-1.0)}
  ```

  Here, the state is projected onto the corresponding state resulting from the MCM.

* The printing and drawing of :class:`~.TemporaryAND`, also known as ``qp.Elbow``, and its adjoint
  have been improved to be more legible and consistent with how it's depicted in circuits in the 
  literature.
  [(#8017)](https://github.com/PennyLaneAI/pennylane/pull/8017)
  [(#8432)](https://github.com/PennyLaneAI/pennylane/pull/8432)  

  ```python
  import pennylane as qp

  @qp.draw
  @qp.qnode(qp.device("lightning.qubit", wires=4))
  def node():
      qp.TemporaryAND([0, 1, 2], control_values=[1, 0])
      qp.CNOT([2, 3])
      qp.adjoint(qp.TemporaryAND([0, 1, 2], control_values=[1, 0]))
      return qp.expval(qp.Z(3))
  ```

  ```pycon
  print(node())
  0: ─╭●─────●╮─┤
  1: ─├○─────○┤─┤
  2: ─╰──╭●───╯─┤
  3: ────╰X─────┤  <Z>
  ```

* With :func:`~.decomposition.enable_graph`, the ``UserWarning`` that is raised when a decomposition
  cannot be found for an operator in the circuit is now more generic, not making any assumptions 
  about how the unresolved operations will be applied or used in the decompose transformation.
  [(#8361)](https://github.com/PennyLaneAI/pennylane/pull/8361)

* The :func:`~.sample` function can now receive an optional ``dtype`` parameter which sets the type 
  and precision of the samples returned by this measurement process.
  [(#8189)](https://github.com/PennyLaneAI/pennylane/pull/8189)
  [(#8271)](https://github.com/PennyLaneAI/pennylane/pull/8271)

* ``DefaultQubit`` will now default to the tree-traversal MCM method when ``mcm_method="device"``.
  [(#7885)](https://github.com/PennyLaneAI/pennylane/pull/7885)

* ``DefaultQubit`` now determines the ``mcm_method`` in ``Device.setup_execution_config``, making it 
  easier to tell which ``mcm_method`` will be used. This also allows ``defer_measurements`` and 
  ``dynamic_one_shot`` to be applied at different locations in the preprocessing program.
  [(#8184)](https://github.com/PennyLaneAI/pennylane/pull/8184)

* The default implementation of ``Device.setup_execution_config`` now choses ``"device"`` as the 
  default ``mcm_method`` if it is available, as specified by the device TOML file.
  [(#7968)](https://github.com/PennyLaneAI/pennylane/pull/7968)

* ``ExecutionConfig`` and ``MCMConfig`` from ``pennylane.devices`` are now frozen dataclasses whose 
  fields should be updated with ``dataclass.replace``.
  [(#7697)](https://github.com/PennyLaneAI/pennylane/pull/7697)
  [(#8046)](https://github.com/PennyLaneAI/pennylane/pull/8046)

* An error is no longer raised when non-integer wire labels are used in QNodes using 
  ``mcm_method="deferred"``.
  [(#7934)](https://github.com/PennyLaneAI/pennylane/pull/7934)

  ```python
  @qp.qnode(qp.device("default.qubit"), mcm_method="deferred")
  def circuit():
      m = qp.measure("a")
      qp.cond(m == 0, qp.X)("aux")
      return qp.expval(qp.Z("a"))
  ```

  ```pycon
  >>> print(qp.draw(circuit)())
    a: ──┤↗├────┤  <Z>
  aux: ───║───X─┤
          ╚═══╝
  ```

* ``qp.transforms.core.TransformContainer`` now holds onto a ``TransformDispatcher``, ``args``, and 
  ``kwargs``, instead of the transform's defining function and unpacked properties. It can still be 
  constructed via the old signature, as well.
  [(#8306)](https://github.com/PennyLaneAI/pennylane/pull/8306)

* The JAX version is now included in :func:`~.about`.
  [(#8277)](https://github.com/PennyLaneAI/pennylane/pull/8277)

* A warning is now raised when circuits are executed without Catalyst and with 
  ``qp.capture.enable()`` present.
  [(#8291)](https://github.com/PennyLaneAI/pennylane/pull/8291)

* The QNode primitive in the experimental program capture module now captures the unprocessed 
  ``ExecutionConfig``, instead of one processed by the device. This allows for better integration 
  with Catalyst.
  [(#8258)](https://github.com/PennyLaneAI/pennylane/pull/8258)

* ``qp.counts`` can now be captured with program capture. Circuits returning ``counts`` still 
  cannot be interpreted or executed with program capture.
  [(#8229)](https://github.com/PennyLaneAI/pennylane/pull/8229)

* Templates are now compatible with program capture. 
  [(#8211)](https://github.com/PennyLaneAI/pennylane/pull/8211)

* PennyLane ``autograph`` supports standard Python for index assignment (``arr[i] = x``) and 
  updating array elements (``arr[i] += x``) instead of ``jax.numpy`` form (i.e., 
  ``arr = arr.at[i].set(x)`` and ``arr.at[i].add(x)``). 
  [(#8027)](https://github.com/PennyLaneAI/pennylane/pull/8027)
  [(#8076)](https://github.com/PennyLaneAI/pennylane/pull/8076)

  ```python
  import jax.numpy as jnp

  qp.capture.enable()

  @qp.qnode(qp.device("default.qubit", wires=3))
  def circuit(val):
    angles = jnp.zeros(3)
    angles[0:3] += val

    for i, angle in enumerate(angles):
        qp.RX(angle, i)

    return qp.expval(qp.Z(0)), qp.expval(qp.Z(1)), qp.expval(qp.Z(2))
  ```

  ```pycon
  >>> circuit(jnp.pi)
  (Array(-1, dtype=float32),
   Array(-1, dtype=float32),
   Array(-1, dtype=float32))
  ```

* Logical operations (``and``, ``or`` and ``not``) are now supported with PennyLane ``autograph``.
  [(#8006)](https://github.com/PennyLaneAI/pennylane/pull/8006)

  ```python
  qp.capture.enable()

  @qp.qnode(qp.device("default.qubit", wires=1))
  def circuit(param):
      if param >= 0 and param <= 1:
          qp.H(0)
      return qp.state()
  ```

  ```pycon
  >>> circuit(0.5)
  Array([0.70710677+0.j, 0.70710677+0.j], dtype=complex64)
  ```

* With program capture, the ``true_fn`` can now be a subclass of ``Operator`` when no ``false_fn`` 
  is provided. For example, ``qp.cond(condition, qp.X)(0)`` is now valid code.
  [(#8060)](https://github.com/PennyLaneAI/pennylane/pull/8060)
  [(#8101)](https://github.com/PennyLaneAI/pennylane/pull/8101)

* With program capture, an error is now raised if the conditional predicate in, say, an ``if`` 
  statement is not a scalar.
  [(#8066)](https://github.com/PennyLaneAI/pennylane/pull/8066)

* Program capture can now handle dynamic shots, shot vectors, and shots set with 
  :func:`~pennylane.set_shots`.
  [(#7652)](https://github.com/PennyLaneAI/pennylane/pull/7652)

* The error message raised when using unified-compiler transforms with :func:`~.qjit` has been 
  updated with suggested fixes.
  [(#7916)](https://github.com/PennyLaneAI/pennylane/pull/7916)

* Two new ``draw`` and ``generate_mlir_graph`` functions have been introduced in the 
  ``qp.compiler.python_compiler.visualization`` module to visualize circuits with the new unified 
  compiler framework when xDSL and/or Catalyst compilation passes are applied.
  [(#8040)](https://github.com/PennyLaneAI/pennylane/pull/8040)
  [(#8180)](https://github.com/PennyLaneAI/pennylane/pull/8180)
  [(#8091)](https://github.com/PennyLaneAI/pennylane/pull/8091)

* The ``catalyst``, ``qec``, and ``stablehlo` xDSL dialects have been added to the unified compiler 
  framework, containing data structures that support core compiler functionality and quantum error 
  correction and extending the existing StableHLO dialect with missing upstream operations.
  [(#7901)](https://github.com/PennyLaneAI/pennylane/pull/7901)
  [(#7985)](https://github.com/PennyLaneAI/pennylane/pull/7985)
  [(#8036)](https://github.com/PennyLaneAI/pennylane/pull/8036)
  [(#8084)](https://github.com/PennyLaneAI/pennylane/pull/8084)
  [(#8113)](https://github.com/PennyLaneAI/pennylane/pull/8113)

* The ``Quantum`` xDSL dialect now has more strict constraints for operands and results.
  [(#8083)](https://github.com/PennyLaneAI/pennylane/pull/8083)

* A callback mechanism has been added to ``qp.compiler.python_compiler`` submodule to inspect the 
  intermediate representation of the program between multiple compilation passes.
  [(#7964)](https://github.com/PennyLaneAI/pennylane/pull/7964)

* A ``QuantumParser`` class has been added to the ``qp.compiler.python_compiler`` submodule that 
  automatically loads relevant dialects.
  [(#7888)](https://github.com/PennyLaneAI/pennylane/pull/7888)

* Two new operations have been added to the ``Quantum`` dialect of the unified compiler:

  * ``NumQubitsOp``: calculates the number of currently allocated qubits.
    [(#8063)](https://github.com/PennyLaneAI/pennylane/pull/8063)

  * ``AllocQubitOp`` and ``DeallocQubitOp``: allocates and deallocates qubits, respectively.
    [(#7915)](https://github.com/PennyLaneAI/pennylane/pull/7915)
  
* A compilation pass written called 
  ``qp.compiler.python_compiler.transforms.MeasurementsFromSamplesPass`` has been added for 
  integration with the unified compiler framework. This pass replaces all terminal measurements in a 
  program with a single :func:`~.sample` measurement, and adds postprocessing instructions to 
  recover the original measurement.
  [(#7620)](https://github.com/PennyLaneAI/pennylane/pull/7620)

* A combine-global-phase pass has been added to the unified compiler framework. Note that the 
  current implementation can only combine all the global phase operations at the last global phase 
  operation in the same region. In other words, global phase operations inside a control flow region 
  can't be combined with those in their parent region.
  [(#7675)](https://github.com/PennyLaneAI/pennylane/pull/7675)

* The matrix factorization using :func:`~.math.decomposition.givens_decomposition` has been 
  optimized to factor out the redundant sign in the diagonal phase matrix for the real-valued 
  (orthogonal) rotation matrices. For example, in case the determinant of a matrix is :math:`-1`, 
  only a single element of the phase matrix is required.
  [(#7765)](https://github.com/PennyLaneAI/pennylane/pull/7765)

* A new device preprocess transform, `~.devices.preprocess.no_analytic`, is available for hardware 
  devices and hardware-like simulators. It validates that all executions are shot-based.
  [(#8037)](https://github.com/PennyLaneAI/pennylane/pull/8037)

* PennyLane is now compatible with ``quimb == 1.11.2`` after a bug affecting ``default.tensor`` was 
  fixed.
  [(#7931)](https://github.com/PennyLaneAI/pennylane/pull/7931)

* A new :func:`~.transforms.resolve_dynamic_wires` transform can allocate concrete wire values for 
  dynamic wire allocation.
  [(#7678)](https://github.com/PennyLaneAI/pennylane/pull/7678)
  [(#8184)](https://github.com/PennyLaneAI/pennylane/pull/8184)
  [(#8406)](https://github.com/PennyLaneAI/pennylane/pull/8406)

<h3>Labs: a place for unified and rapid prototyping of research software 🧪</h3>

<h4>Labs Resource Estimation</h4>

* State-of-the-art resource estimates have been added to existing templates:
  :class:`~pennylane.labs.resource_estimation.ResourceSelectPauliRot`,
  :class:`~pennylane.labs.resource_estimation.ResourceQubitUnitary`, 
  :class:`~pennylane.labs.resource_estimation.ResourceSingleQubitComparator`, 
  :class:`~pennylane.labs.resource_estimation.ResourceTwoQubitComparator`,
  :class:`~pennylane.labs.resource_estimation.ResourceIntegerComparator`, 
  :class:`~pennylane.labs.resource_estimation.ResourceRegisterComparator`, 
  :class:`~pennylane.labs.resource_estimation.ResourceUniformStatePrep`,
  :class:`~pennylane.labs.resource_estimation.ResourceAliasSampling`, 
  :class:`~pennylane.labs.resource_estimation.ResourceQFT`, 
  :class:`~pennylane.labs.resource_estimation.ResourceAQFT`, and 
  :class:`~pennylane.labs.resource_estimation.ResourceTrotterProduct`.
  [(#7786)](https://github.com/PennyLaneAI/pennylane/pull/7786)
  [(#7857)](https://github.com/PennyLaneAI/pennylane/pull/7857)
  [(#7883)](https://github.com/PennyLaneAI/pennylane/pull/7883)
  [(#7920)](https://github.com/PennyLaneAI/pennylane/pull/7920)
  [(#7910)](https://github.com/PennyLaneAI/pennylane/pull/7910)

* Users can now do resource estimation on QPE and iterative QPE with
  :class:`~pennylane.labs.resource_estimation.ResourceQPE` and
  :class:`~pennylane.labs.resource_estimation.ResourceIterativeQPE`, respectively. Additionally,
  a :class:`~pennylane.labs.resource_estimation.ResourceControlledSequence` template has been added
  that allows estimating resources on controlled sequences of resource operators.
  [(#8053)](https://github.com/PennyLaneAI/pennylane/pull/8053)

* ``estimate_resources`` has been renamed to ``estimate`` to make the function name concise and 
  clearer than ``labs.resource_estimation.estimate_resources``.
  [(#8232)](https://github.com/PennyLaneAI/pennylane/pull/8232)

* A new ``ResourceConfig`` class has been added to help track the configuration for errors, 
  precisions and custom decompositions for the resource estimation pipeline.
  [(#8195)](https://github.com/PennyLaneAI/pennylane/pull/8195)

* The symbolic ``ResourceOperators`` have been updated to use hyperparameters from the ``config`` 
  dictionary.
  [(#8181)](https://github.com/PennyLaneAI/pennylane/pull/8181)

* An internal ``dequeue()`` method has been added to the ``ResourceOperator`` class to simplify the
  instantiation of resource operators which require resource operators as input.
  [(#7974)](https://github.com/PennyLaneAI/pennylane/pull/7974)

* ``ResourceOperator`` instances can now be compared with ``==``.
  [(#8155)](https://github.com/PennyLaneAI/pennylane/pull/8155)

* A mapper function called :func:`~pennylane.labs.resource_estimation.map_to_resource_op`
  has been added to map PennyLane operators to ``ResourceOperator`` equivalents.
  [(#8146)](https://github.com/PennyLaneAI/pennylane/pull/8146)
  [(#8162)](https://github.com/PennyLaneAI/pennylane/pull/8162)

* Several Labs test files have been renamed to prevent conflict with names in mainline PennyLane 
  tests.
  [(#8264)](https://github.com/PennyLaneAI/pennylane/pull/8264)

* A queueing issue in the ``ResourceOperator`` tests has been fixed.
  [(#8204)](https://github.com/PennyLaneAI/pennylane/pull/8204)

<h4>Labs Trotter Error Estimation</h4>

* Parallelization support for ``effective_hamiltonian`` has been added to improve performance.
  [(#8081)](https://github.com/PennyLaneAI/pennylane/pull/8081)
  [(#8257)](https://github.com/PennyLaneAI/pennylane/pull/8257)

* New ``SparseFragment`` and ``SparseState`` classes have been created to allow the use
  of sparse matrices for Hamiltonian Fragments when estimating Trotter error.
  [(#7971)](https://github.com/PennyLaneAI/pennylane/pull/7971)

* The :func:`~pennylane.labs.trotter_error.perturbation_error` function has
  been updated to sum over expectation values instead of states.
  [(#8226)](https://github.com/PennyLaneAI/pennylane/pull/8226)

* The docstring in ``perturbation_error`` has been updated to use the correct positional argument 
  name.
  [(#8174)](https://github.com/PennyLaneAI/pennylane/pull/8174)

<h4>Labs Removals</h4>

* The module ``qp.labs.zxopt`` has been removed. Its functionalities are now available in the
  submodule :mod:`~.transforms.zx`. The same functions are available, but their signature
  may have changed.
  - Instead of ``qp.labs.zxopt.full_optimize``, use :func:`~.transforms.zx.optimize_t_count`
  - Instead of ``qp.labs.zxopt.full_reduce``, use :func:`~.transforms.zx.reduce_non_clifford`
  - Instead of ``qp.labs.zxopt.todd``, use :func:`~.transforms.zx.todd`
  - Instead of ``qp.labs.zxopt.basic_optimization``, use :func:`~.transforms.zx.push_hadamards`
  [(#8177)](https://github.com/PennyLaneAI/pennylane/pull/8177)

<h3>Breaking changes 💔</h3>

* ``autoray``  has been pinned to v0.8.0 for PennyLane v0.43.0 to prevent potential bugs due to 
  breaking changes in autoray releases.
  [(#8412)](https://github.com/PennyLaneAI/pennylane/pull/8412)

  Previous to this change, the ``autoray`` package was upper-bounded in ``pyproject.toml`` to 
  unblock CI failures due to breaking changes in `v0.8.0`. Then, it was unpinned by fixing source 
  code that was broken by the release.
  [(#8110)](https://github.com/PennyLaneAI/pennylane/pull/8110)
  [(#8147)](https://github.com/PennyLaneAI/pennylane/pull/8147)
  [(#8159)](https://github.com/PennyLaneAI/pennylane/pull/8159)
  [(#8160)](https://github.com/PennyLaneAI/pennylane/pull/8160)

* Using ``postselect_mode="fill-shots"`` with ``mcm_method="one-shot"`` or ``"tree-traversal"`` has 
  been disallowed with ``default.qubit``, as it produces incorrect results where the correlation 
  between measurements is not preserved.
  [(#8411)](https://github.com/PennyLaneAI/pennylane/pull/8411)

* ``qp.workflow.construct_batch.expand_fn_transform`` has been deleted as it was no longer getting 
  used.
  [(#8344)](https://github.com/PennyLaneAI/pennylane/pull/8344)

* ``get_canonical_interface_name`` has been removed in favour of overriding ``Enum._missing_`` in   
  ``Interface``.
  [(#8223)](https://github.com/PennyLaneAI/pennylane/pull/8223)

  If you would like to get the canonical interface you can simply use the ``Enum``:

  ```pycon
  >>> from pennylane.math.interface_utils import Interface
  >>> Interface("torch")
  <Interface.TORCH: 'torch'>  
  >>> Interface("jax-jit")
  <Interface.JAX_JIT: 'jax-jit'>
  ```

* :class:`~.PrepSelPrep` has been made more reliable by deriving the attributes ``coeffs`` and `
  ``ops`` from the property ``lcu`` instead of storing them independently. In addition, it is now 
  more consistent with other PennyLane operators, dequeuing its input ``lcu``.
  [(#8169)](https://github.com/PennyLaneAI/pennylane/pull/8169)

* ``MidMeasureMP`` now inherits from ``Operator`` instead of ``MeasurementProcess``, which resolves 
  problems caused by it always acting like an operator.
  [(#8166)](https://github.com/PennyLaneAI/pennylane/pull/8166)

* With the deprecation of the ``shots`` kwarg in ``qp.device``, ``DefaultQubit.eval_jaxpr`` does 
  not use ``self.shots`` from the device anymore; instead, it takes ``shots`` as a keyword argument, and the QNode primitive should process the ``shots`` and call ``eval_jaxpr`` accordingly.
  [(#8161)](https://github.com/PennyLaneAI/pennylane/pull/8161)

* The methods :meth:`~.pauli.PauliWord.operation` and :meth:`~.pauli.PauliSentence.operation`
  no longer queue any operators. This improves the consistency of the queuing behaviour for the operators.
  [(#8136)](https://github.com/PennyLaneAI/pennylane/pull/8136)

* ``qp.sample`` no longer has singleton dimensions squeezed out for single shots or single wires. 
  This cuts down on the complexity of post-processing due to having to handle single shot and single 
  wire cases separately. The return shape will now *always* be ``(shots, num_wires)``.
  [(#7944)](https://github.com/PennyLaneAI/pennylane/pull/7944)
  [(#8118)](https://github.com/PennyLaneAI/pennylane/pull/8118)

  For a simple qnode:

  ```python
  @qp.qnode(qp.device('default.qubit'))
  def circuit():
      return qp.sample(wires=0)
  ```

  Before the change, we had:

  ```pycon
  >>> qp.set_shots(circuit, shots=1)()
  0
  ```

  and now we have:

  ```pycon
  >>> qp.set_shots(circuit, shots=1)()
  array([[0]])
  ```

  Previous behavior can be recovered by squeezing the output:

  ```pycon
  >>> qp.math.squeeze(qp.set_shots(circuit, shots=1)())
  array(0)
  ```

* Functions involving an execution configuration will now default to ``None`` instead of 
  ``pennylane.devices.DefaultExecutionConfig`` and have to be handled accordingly. This prevents the 
  potential mutation of a global object.
  [(#7697)](https://github.com/PennyLaneAI/pennylane/pull/7697)

  This means that functions like,

  ```python
  def some_func(..., execution_config = DefaultExecutionConfig):
      ...
  ```

  should be written as follows,

  ```python
  def some_func(..., execution_config: ExecutionConfig | None = None):
      if execution_config is None:
          execution_config = ExecutionConfig()
  ```

* The :class:`~.HilbertSchmidt` and :class:`~.LocalHilbertSchmidt` templates have been updated and
  their UI has been remarkably simplified. They now accept an operation or a list of operations as  
  unitaries.
  [(#7933)](https://github.com/PennyLaneAI/pennylane/pull/7933)

  In past versions of PennyLane, these templates required providing the ``U`` and ``V`` unitaries as 
  a ``qp.tape.QuantumTape`` and a quantum function, respectively, along with separate parameters 
  and wires.

  With this release, each template has been improved to accept one or more operators as unitaries.
  The wires and parameters of the approximate unitary ``V`` are inferred from the inputs, according 
  to the order provided.

  ```pycon
  >>> U = qp.Hadamard(0)
  >>> V = qp.RZ(0.1, wires=1)
  >>> qp.HilbertSchmidt(V, U)
  HilbertSchmidt(0.1, wires=[0, 1])
  ```

* Support for Python 3.10 has been removed and support for Python 3.13 has been added.
  [(#7935)](https://github.com/PennyLaneAI/pennylane/pull/7935)

* To make the codebase more organized and easier to maintain, custom exceptions were moved into 
  ``exceptions.py``, and a documentation page for them was added in the internals.
  [(#7856)](https://github.com/PennyLaneAI/pennylane/pull/7856)

* The boolean functions provided in ``qp.operation`` have been deprecated. See the
  :doc:`deprecations page </development/deprecations>` for equivalent code to use instead. These
  include ``not_tape``, ``has_gen``, ``has_grad_method``, ``has_multipar``, ``has_nopar``, 
  ``has_unitary_gen``, ``is_measurement``, ``defines_diagonalizing_gates``, and 
  ``gen_is_multi_term_hamiltonian``.
  [(#7924)](https://github.com/PennyLaneAI/pennylane/pull/7924)

* To prevent code duplication, access to ``lie_closure``, ``structure_constants`` and ``center`` via 
  ``qp.pauli`` has been removed. The functions now live in the ``liealg`` module and top level 
  import and usage is advised. 
  [(#7928)](https://github.com/PennyLaneAI/pennylane/pull/7928)
  [(#7994)](https://github.com/PennyLaneAI/pennylane/pull/7994)

  ```python
  import pennylane.liealg
  from pennylane.liealg import lie_closure, structure_constants, center
  ```

* ``qp.operation.Observable`` and the corresponding ``Observable.compare`` have been removed, as
  PennyLane now depends on the more general ``Operator`` interface instead. The
  ``Operator.is_hermitian`` property can instead be used to check whether or not it is highly likely
  that the operator instance is Hermitian.
  [(#7927)](https://github.com/PennyLaneAI/pennylane/pull/7927)

* ``qp.operation.WiresEnum``, ``qp.operation.AllWires``, and ``qp.operation.AnyWires`` have been 
  removed. To indicate that an operator can act on any number of wires, 
  ``Operator.num_wires = None`` should be used instead. This is the default and does not need to be 
  overwritten unless the operator developer wants to validate that the correct number of wires is 
  passed.
  [(#7911)](https://github.com/PennyLaneAI/pennylane/pull/7911)

* The :func:`qp.QNode.get_gradient_fn` function has been removed. Instead, use 
  :func:`qp.workflow.get_best_diff_method` to obtain the differentiation method.
  [(#7907)](https://github.com/PennyLaneAI/pennylane/pull/7907)

* Top-level access to ``DeviceError``, ``PennyLaneDeprecationWarning``, ``QuantumFunctionError`` and 
  ``ExperimentalWarning`` has been removed. Please import these objects from the new 
  ``pennylane.exceptions`` module.
  [(#7874)](https://github.com/PennyLaneAI/pennylane/pull/7874)

* To improve code reliability, ``qp.cut_circuit_mc`` no longer accepts a ``shots`` keyword  
  argument. The shots should instead be set on the tape itself.
  [(#7882)](https://github.com/PennyLaneAI/pennylane/pull/7882)

* :func:`~.tape.tape.expand_tape` has been moved to its own file, and made available at 
  ``qp.tape``.
  [(#8296)](https://github.com/PennyLaneAI/pennylane/pull/8296)

<h3>Deprecations 👋</h3>

* PennyLane and Lightning will no longer ship wheels for Intel MacOS platforms for v0.44 and newer.
  Additionally, MacOS ARM wheels will require a minimum OS version of 14.0 for continued use with 
  v0.44 and newer. This change is needed to account for MacOS officially deprecating support for 
  Intel CPUs in the OS (see their [blog post](https://github.blog/changelog/2025-09-19-github-actions-macos-13-runner-image-is-closing-down/#notice-of-macos-x86_64-intel-architecture-deprecation) for more details).

* Setting shots on a device through the ``shots`` keyword argument (e.g., 
  ``qp.device("default.qubit", wires=2, shots=1000)``) and in QNode calls (e.g., 
  ``qp.QNode(circuit, dev)(shots=1000)``) has been deprecated. Please use the :func:`~pennylane.set_shots` transform to set the number of shots for a QNode instead.  This is done to reduce confusion and 
  code complexity by having a centralized way to set shots.
  [(#7979)](https://github.com/PennyLaneAI/pennylane/pull/7979)
  [(#8161)](https://github.com/PennyLaneAI/pennylane/pull/8161)
  [(#7906)](https://github.com/PennyLaneAI/pennylane/pull/7906)
  
  ```python
  dev = qp.device("default.qubit", wires=2)

  @qp.set_shots(1000)
  @qp.qnode(dev)
  def circuit(x):
      qp.RX(x, wires=0)
      return qp.expval(qp.Z(0))
  ```

* Support for using TensorFlow with PennyLane has been deprecated and will be dropped in Pennylane
  v0.44. Future versions of PennyLane are not guaranteed to work with TensorFlow. Instead, we 
  recommend using the :doc:`JAX </introduction/interfaces/jax>` or 
  :doc:`PyTorch </introduction/interfaces/torch>` interfaces for machine learning applications to 
  benefit from enhanced support and features. Please consult the following demos for more usage 
  information:
  [Turning quantum nodes into Torch Layers](https://pennylane.ai/qml/demos/tutorial_qnn_module_torch) 
  and 
  [How to optimize a QML model using JAX and Optax](https://pennylane.ai/qml/demos/tutorial_How_to_optimize_QML_model_using_JAX_and_Optax).
  [(#7989)](https://github.com/PennyLaneAI/pennylane/pull/7989)
  [(#8106)](https://github.com/PennyLaneAI/pennylane/pull/8106)

* ``pennylane.devices.DefaultExecutionConfig`` has been deprecated and will be removed in v0.44.
  Instead, use ``qp.devices.ExecutionConfig()`` to create a default execution configuration. 
  This helps prevent unintended changes in a workflow's behaviour that could be caused by using a 
  global, mutable object.
  [(#7987)](https://github.com/PennyLaneAI/pennylane/pull/7987)

* Specifying the ``work_wire_type`` argument in ``qp.ctrl`` and other controlled operators as 
  ``"clean"`` or ``"dirty"`` has been deprecated. Use ``"zeroed"`` to indicate that the work wires 
  are initially in the :math:`|0\rangle` state, and ``"borrowed"`` to indicate that the work wires 
  can be in any arbitrary state instead. In both cases, the work wires are restored to their 
  original state upon completing the decomposition. This is done to standardize how work wires are 
  called in PennyLane.
  [(#7993)](https://github.com/PennyLaneAI/pennylane/pull/7993)

* Providing the Trotter number kwarg ``num_steps`` to :func:`pennylane.evolve`, 
  :func:`pennylane.exp`, :class:`pennylane.ops.Evolution`, and :class:`pennylane.ops.Exp` has been 
  deprecated and will be removed in a future release. Instead, use :class:`~.TrotterProduct` for 
  approximate methods, providing the ``n`` parameter to perform the Suzuki-Trotter product
  approximation of a Hamiltonian with the specified number of Trotter steps. This change resolves 
  the ambiguity that arises when using ``num_steps`` on devices that support analytic evolution 
  (e.g., ``default.qubit``).
  [(#7954)](https://github.com/PennyLaneAI/pennylane/pull/7954)
  [(#7977)](https://github.com/PennyLaneAI/pennylane/pull/7977)

  As a concrete example, consider the following case:

  ```python
  coeffs = [0.5, -0.6]
  ops = [qp.X(0), qp.X(0) @ qp.Y(1)]
  H_flat = qp.dot(coeffs, ops)
  ```

  Instead of computing the Suzuki-Trotter product approximation as:

  ```pycon
  >>> qp.evolve(H_flat, num_steps=2).decomposition()
  [RX(0.5, wires=[0]),
  PauliRot(-0.6, XY, wires=[0, 1]),
  RX(0.5, wires=[0]),
  PauliRot(-0.6, XY, wires=[0, 1])]
  ```

  The same result can be obtained using :class:`~.TrotterProduct` as follows:

  ```pycon
  >>> decomp_ops = qp.adjoint(qp.TrotterProduct(H_flat, time=1.0, n=2)).decomposition()
  >>> [simp_op for op in decomp_ops for simp_op in map(qp.simplify, op.decomposition())]
  [RX(0.5, wires=[0]),
  PauliRot(-0.6, XY, wires=[0, 1]),
  RX(0.5, wires=[0]),
  PauliRot(-0.6, XY, wires=[0, 1])]
  ```

* ``MeasurementProcess.expand`` has been deprecated. The relevant method can be replaced with
  ``qp.tape.QuantumScript(mp.obs.diagonalizing_gates(), [type(mp)(eigvals=mp.obs.eigvals(), wires=mp.obs.wires)])``.
  This improves the code design by removing an unused method with undesired dependencies (i.e. 
  circular dependency).
  [(#7953)](https://github.com/PennyLaneAI/pennylane/pull/7953)

* ``QuantumScript.shape`` and ``QuantumScript.numeric_type`` have been deprecated and will be 
  removed in version v0.44. Instead, the corresponding ``.shape`` or ``.numeric_type`` of the 
  ``MeasurementProcess`` class should be used.
  [(#7950)](https://github.com/PennyLaneAI/pennylane/pull/7950)

* Some unnecessary methods of the ``qp.CircuitGraph`` class have been deprecated and will be 
  removed in version v0.44:
  [(#7904)](https://github.com/PennyLaneAI/pennylane/pull/7904)

    - ``print_contents`` in favor of ``print(obj)``
    - ``observables_in_order`` in favor of ``observables``
    - ``operations_in_order`` in favor of ``operations``
    - ``ancestors_in_order`` in favor of ``ancestors(obj, sort=True)``
    - ``descendants_in_order`` in favor of ``descendants(obj, sort=True)``

* The ``QuantumScript.to_openqasm`` method has been deprecated and will be removed in version v0.44.
  Instead, the ``qp.to_openqasm`` function should be used. This change makes the code cleaner by 
  separating out methods that interface with external libraries from PennyLane's internal 
  functionality.
  [(#7909)](https://github.com/PennyLaneAI/pennylane/pull/7909)

* The ``level=None`` argument in the :func:`pennylane.workflow.get_transform_program`, 
  :func:`pennylane.workflow.construct_batch`, ``qp.draw``, ``qp.draw_mpl``, and ``qp.specs`` 
  transforms has been deprecated and will be removed in v0.44. Please use ``level='device'`` instead 
  to apply the noise model at the device level. This reduces ambiguity by making it clear that the 
  default is to apply all transforms to the QNode.
  [(#7886)](https://github.com/PennyLaneAI/pennylane/pull/7886)
  [(#8364)](https://github.com/PennyLaneAI/pennylane/pull/8364)

* ``qp.qnn.cost.SquaredErrorLoss`` has been deprecated and will be removed in version v0.44. 
  Instead, this hybrid workflow can be accomplished with a function like 
  ``loss = lambda *args: (circuit(*args) - target)**2``.
  [(#7527)](https://github.com/PennyLaneAI/pennylane/pull/7527)

* Access to ``add_noise``, ``insert`` and noise mitigation transforms from the ``transforms`` module has been deprecated.
  Instead, these functions should be imported from the ``noise`` module, which is a more appropriate location for them.
  [(#7854)](https://github.com/PennyLaneAI/pennylane/pull/7854)

* The ``qp.QNode.add_transform`` method has been deprecated and will be removed in v0.44.
  Instead, please use 
  ``QNode.transform_program.push_back(transform_container=transform_container)``.
  [(#7855)](https://github.com/PennyLaneAI/pennylane/pull/7855)
  [(#8266)](https://github.com/PennyLaneAI/pennylane/pull/8266)

<h3>Internal changes ⚙️</h3>

* GitHub actions and workflows (`interface-unit-tests.yml`, `tests-labs.yml`, `unit-test.yml`, `upload-nightly-release.yml` and `upload.yml`) have been updated to
  use `ubuntu-24.04` runners.
  [(#8371)](https://github.com/PennyLaneAI/pennylane/pull/8371)

* ``measurements`` is now a "core" module in the ``tach`` specification.
  [(#7945)](https://github.com/PennyLaneAI/pennylane/pull/7945)

* Enforce various modules to follow modular architecture via ``tach``.
  [(#7847)](https://github.com/PennyLaneAI/pennylane/pull/7847)

* CI workflows to test documentation using ``sybil`` have been added. 
  [(#8324)](https://github.com/PennyLaneAI/pennylane/pull/8324)
  [(#8328)](https://github.com/PennyLaneAI/pennylane/pull/8328)
  [(#8329)](https://github.com/PennyLaneAI/pennylane/pull/8329)
  [(#8330)](https://github.com/PennyLaneAI/pennylane/pull/8330)
  [(#8331)](https://github.com/PennyLaneAI/pennylane/pull/8331)
  [(#8386)](https://github.com/PennyLaneAI/pennylane/pull/8386)

* The ``templates/subroutines`` now has ``arithmetic``, ``qchem``, and ``time_evolution`` 
  submodules.
  [(#8333)](https://github.com/PennyLaneAI/pennylane/pull/8333)

* ``test_horizontal_cartan_subalgebra.py`` now uses our fixture ``seed`` for reproducibility and CI stability.
  [(#8304)](https://github.com/PennyLaneAI/pennylane/pull/8304)

* The `qp.compiler.python_compiler` submodule has been restructured to be more cohesive.
  [(#8273)](https://github.com/PennyLaneAI/pennylane/pull/8273)

* ``default.tensor`` now supports graph decomposition (``qp.decomposition.enable_graph()``) during preprocessing.
  [(#8253)](https://github.com/PennyLaneAI/pennylane/pull/8253)

* Legacy interface names from tests have been removed (e.g., ``interface="jax-python"`` or ``interface="pytorch"``)
  [(#8249)](https://github.com/PennyLaneAI/pennylane/pull/8249)

* ``qp.devices.preprocess.decompose`` now works in graph decomposition mode
  when a gateset is provided. ``default.qubit`` and ``null.qubit`` can now use
  graph decomposition mode.
  [(#8225)](https://github.com/PennyLaneAI/pennylane/pull/8225)
  [(#8265)](https://github.com/PennyLaneAI/pennylane/pull/8265)
  [(#8260)](https://github.com/PennyLaneAI/pennylane/pull/8260)

* Usage of the ``pytest.mark.capture`` marker from tests in the ``tests/python_compiler`` directory 
  has been removed.
  [(#8234)](https://github.com/PennyLaneAI/pennylane/pull/8234)

* ``pylint`` has been updated to v3.3.8 in our CI and ``requirements-dev.txt``
  [(#8216)](https://github.com/PennyLaneAI/pennylane/pull/8216)

* Links in the ``README.md`` have been updated.
  [(#8165)](https://github.com/PennyLaneAI/pennylane/pull/8165)

* The `autograph` guide to now reflects new capabilities.
  [(#8132)](https://github.com/PennyLaneAI/pennylane/pull/8132)

* ``strict=True`` is now used to ``zip`` usage in source code.
  [(#8164)](https://github.com/PennyLaneAI/pennylane/pull/8164)
  [(#8182)](https://github.com/PennyLaneAI/pennylane/pull/8182)
  [(#8183)](https://github.com/PennyLaneAI/pennylane/pull/8183)

* The ``autograph`` keyword argument has been removed from the ``QNode`` constructor.
  To enable autograph conversion, use the ``qjit`` decorator together with the 
  ``qp.capture.disable_autograph`` context manager.

* The ability to disable ``autograph`` conversion has been added by using the new 
  ``qp.capture.disable_autograph`` decorator or context manager. Additionally, the 
  ``autograph`` keyword argument has been removed from the ``QNode`` constructor. To enable 
  autograph conversion, use the ``qjit`` decorator together with the 
  ``qp.capture.disable_autograph`` context manager.
  [(#8102)](https://github.com/PennyLaneAI/pennylane/pull/8102)
  [(#8104)](https://github.com/PennyLaneAI/pennylane/pull/8104)

* Roundtrip testing and module verification to the Python compiler is now done in ``run_filecheck`` 
  and ``run_filecheck_qjit`` fixtures.
  [(#8049)](https://github.com/PennyLaneAI/pennylane/pull/8049)

* Various type hints have been improved internally.
  [(#8086)](https://github.com/PennyLaneAI/pennylane/pull/8086)
  [(#8284)](https://github.com/PennyLaneAI/pennylane/pull/8284)

* The ``cond`` primitive with program capture no longer stores missing false branches as ``None``, 
  instead storing them as jaxprs with no output.
  [(#8080)](https://github.com/PennyLaneAI/pennylane/pull/8080)

* Unnecessary execution tests along with accuracy validation in 
  ``tests/ops/functions/test_map_wires.py`` were removed due to stochastic failures.
  [(#8032)](https://github.com/PennyLaneAI/pennylane/pull/8032)

* A new ``all-tests-passed`` gatekeeper job has been added to ``interface-unit-tests.yml`` to ensure 
  all test jobs complete successfully before triggering downstream actions. This reduces the need to
  maintain a long list of required checks in GitHub settings. Also added the previously missing
  ``capture-jax-tests`` job to the list of required test jobs, ensuring this test suite is properly
  enforced in CI.
  [(#7996)](https://github.com/PennyLaneAI/pennylane/pull/7996)

* ``DefaultQubitLegacy`` (test suite only) has been equipped with seeded sampling. This allows for 
  reproducible sampling results of legacy classical shadow across CI.
  [(#7903)](https://github.com/PennyLaneAI/pennylane/pull/7903)

* ``DefaultQubitLegacy`` (test suite only) no longer provides a customized classical shadow
  implementation.
  [(#7895)](https://github.com/PennyLaneAI/pennylane/pull/7895)

* Capture does not block ``wires=0`` anymore. This allows Catalyst to work with zero-wire devices.
  Note that ``wires=None`` is still not allowed.
  [(#7978)](https://github.com/PennyLaneAI/pennylane/pull/7978)

* The readability of ``dynamic_one_shot`` postprocessing has been improved to allow for further 
  modification.
  [(#7962)](https://github.com/PennyLaneAI/pennylane/pull/7962)
  [(#8041)](https://github.com/PennyLaneAI/pennylane/pull/8041)

* PennyLane's top-level ``__init__.py`` file has been updated with imports to improve Python 
  language server support for finding PennyLane submodules.
  [(#7959)](https://github.com/PennyLaneAI/pennylane/pull/7959)

* Type hints in the ``measurements`` module have been improved.
  [(#7938)](https://github.com/PennyLaneAI/pennylane/pull/7938)

* The codebase has been refactored to adopt modern type hint syntax for Python 3.11+.
  [(#7860)](https://github.com/PennyLaneAI/pennylane/pull/7860)
  [(#7982)](https://github.com/PennyLaneAI/pennylane/pull/7982)

* Pre-commit hooks have been updated to add gitleaks for security purposes.
  [(#7922)](https://github.com/PennyLaneAI/pennylane/pull/7922)

* A new fixture called ``run_filecheck_qjit`` has been added, which can be used to run ``FileCheck`` 
  on integration tests for the ``qp.compiler.python_compiler`` submodule.
  [(#7888)](https://github.com/PennyLaneAI/pennylane/pull/7888)

* The minimum supported ``pytest`` version has been updated to ``8.4.1``.
  [(#7853)](https://github.com/PennyLaneAI/pennylane/pull/7853)

* The ``pennylane.io`` module is now a tertiary module.
  [(#7877)](https://github.com/PennyLaneAI/pennylane/pull/7877)

* Tests for the ``split_to_single_terms`` transformation are now seeded.
  [(#7851)](https://github.com/PennyLaneAI/pennylane/pull/7851)

* The ``rc_sync.yml`` file has been updated to work with the latest ``pyproject.toml`` changes.
  [(#7808)](https://github.com/PennyLaneAI/pennylane/pull/7808)
  [(#7818)](https://github.com/PennyLaneAI/pennylane/pull/7818)

* ``LinearCombination`` instances can now be created with ``_primitive.impl`` when capture is 
  enabled and tracing is active.
  [(#7893)](https://github.com/PennyLaneAI/pennylane/pull/7893)

* The ``TensorLike`` type is now compatible with static type checkers.
  [(#7905)](https://github.com/PennyLaneAI/pennylane/pull/7905)

* The supported version of xDSL has been updated to ``0.49``.
  [(#7923)](https://github.com/PennyLaneAI/pennylane/pull/7923)
  [(#7932)](https://github.com/PennyLaneAI/pennylane/pull/7932)
  [(#8120)](https://github.com/PennyLaneAI/pennylane/pull/8120)

* The JAX version used in tests to has been updated to ``0.6.2``
  [(#7925)](https://github.com/PennyLaneAI/pennylane/pull/7925)

* An ``xdsl_extras`` module has been added to the unified compiler framework to house additional 
  utilities and functionality not available upstream in xDSL.
  [(#8067)](https://github.com/PennyLaneAI/pennylane/pull/8067)
  [(#8120)](https://github.com/PennyLaneAI/pennylane/pull/8120)

* Two new xDSL passes have been added to the unified compiler framework: ``decompose-graph-state``, 
  which decomposes ``mbqc.graph_state_prep`` operations into their corresponding set of quantum 
  operations for execution on state simulators, and ``null-decompose-graph-state``, which replaces
  ``mbqc.graph_state_prep`` operations with single quantum-register allocation operations for
  execution on null devices.
  [(#8090)](https://github.com/PennyLaneAI/pennylane/pull/8090)

* The ``mbqc`` xDSL dialect has been added to the unified compiler framework, which is used to 
  represent measurement-based quantum-computing instructions in the xDSL framework.
  [(#7815)](https://github.com/PennyLaneAI/pennylane/pull/7815)
  [(#8059)](https://github.com/PennyLaneAI/pennylane/pull/8059)

* A compilation pass written with xDSL called 
  ``qp.compiler.python_compiler.transforms.ConvertToMBQCFormalismPass`` has been added for the 
  experimental unified compiler framework. This pass converts all gates in the MBQC gate set
  (``Hadamard``, ``S``, ``RZ``, ``RotXZX`` and ``CNOT``) to the textbook MBQC formalism.
  [(#7870)](https://github.com/PennyLaneAI/pennylane/pull/7870)
  [(#8254)](https://github.com/PennyLaneAI/pennylane/pull/8254)

* A ``dialects`` submodule has been added to ``qp.compiler.python_compiler`` which now houses all 
  the xDSL dialects we create. Additionally, the ``MBQCDialect`` and ``QuantumDialect`` dialects 
  have been renamed to ``MBQC`` and ``Quantum``.
  [(#7897)](https://github.com/PennyLaneAI/pennylane/pull/7897)

* The measurement-plane attribute of the unified compiler ``mbqc`` dialect now uses the "opaque 
  syntax" format when printing in the generic IR format. This enables usage of this attribute when 
  IR needs to be passed from xDSL to Catalyst.
  [(#7957)](https://github.com/PennyLaneAI/pennylane/pull/7957)

* A ``diagonalize_mcms`` option has been added to the 
  ``ftqc.decomposition.convert_to_mbqc_formalism`` tape transform that, when set, maps 
  arbitrary-basis mid-circuit measurements into corresponding diagonalizing gates and Z-basis 
  mid-circuit measurements.
  [(#8105)](https://github.com/PennyLaneAI/pennylane/pull/8105)

* The ``mbqc.graph_state_prep`` operation is now integrated into the ``convert_to_mbqc_formalism`` 
  pass.
  [(#8153)](https://github.com/PennyLaneAI/pennylane/pull/8153)
  [(#8301)](https://github.com/PennyLaneAI/pennylane/pull/8301)
  [(#8314)](https://github.com/PennyLaneAI/pennylane/pull/8314)
  [(#8362)](https://github.com/PennyLaneAI/pennylane/pull/8362)

* A ``graph_state_utils`` submodule has been added to ``python_compiler.transforms.mbqc`` for common 
  utilities for MBQC workflows.
  [(#8219)](https://github.com/PennyLaneAI/pennylane/pull/8219)
  [(#8273)](https://github.com/PennyLaneAI/pennylane/pull/8273)

* Support for ``pubchempy`` has been updated to ``1.0.5`` in the unit tests for 
  ``qp.qchem.mol_data``.
  [(#8224)](https://github.com/PennyLaneAI/pennylane/pull/8224)

* A nightly RC builds script has been added to ``.github/workflows``.
  [(#8148)](https://github.com/PennyLaneAI/pennylane/pull/8148)

* The test files for :mod:`~.estimator` were renamed to avoid a dual definition error with the 
  :mod:`~.labs` module.
  [(#8261)](https://github.com/PennyLaneAI/pennylane/pull/8261)

<h3>Documentation 📝</h3>

* The :doc:`program capture sharp bits page </news/program_capture_sharp_bits>` has been updated to include a warning about 
  the experimental nature of the feature.
  [(#8448)](https://github.com/PennyLaneAI/pennylane/pull/8448)

* The :doc:`installation page </development/guide/installation>` has been updated to include
  currently supported Python versions and installation instructions.
  [(#8369)](https://github.com/PennyLaneAI/pennylane/pull/8369)

* The documentation of ``qp.probs`` and ``qp.Hermitian`` has been updated with a warning
  to avoid using them together as the output might be different than expected.
  Furthermore, a warning is raised if a user attempts to use ``qp.probs`` with a Hermitian 
  observable.
  [(#8235)](https://github.com/PennyLaneAI/pennylane/pull/8235)

* "`>>>`" and "`...`" have been removed from "`.. code-block::`" directives in docstrings to 
  facilitate docstring testing and fit best practices.
  [(#8319)](https://github.com/PennyLaneAI/pennylane/pull/8319)

* Three more examples of the deprecated usage of ``qp.device(..., shots=...)`` have been updated in 
  the documentation.
  [(#8298)](https://github.com/PennyLaneAI/pennylane/pull/8298)

* The documentation of ``qp.device`` has been updated to reflect the usage of 
  :func:`~pennylane.set_shots`.
  [(#8294)](https://github.com/PennyLaneAI/pennylane/pull/8294)

* The "Simplifying Operators" section in the 
  :doc:`Compiling circuits </introduction/compiling_circuits>` page has been pushed further down 
  the page to show more relevant sections first.
  [(#8233)](https://github.com/PennyLaneAI/pennylane/pull/8233)

* ``ancilla`` has been renamed to ``auxiliary`` in internal documentation.
  [(#8005)](https://github.com/PennyLaneAI/pennylane/pull/8005)

* Small typos in the docstring for `qp.noise.partial_wires` have been corrected.
  [(#8052)](https://github.com/PennyLaneAI/pennylane/pull/8052)

* The theoretical background section of :class:`~.BasisRotation` has been extended to explain
  the underlying Lie group/algebra homomorphism between the (dense) rotation matrix and the
  performed operations on the target qubits.
  [(#7765)](https://github.com/PennyLaneAI/pennylane/pull/7765)

* The code examples in the documentation of :func:`~.specs` have been updated to replace keyword arguments
  with `gradient_kwargs` in the QNode definition.
  [(#8003)](https://github.com/PennyLaneAI/pennylane/pull/8003)

* The documentation for ``Operator.pow`` and ``Operator.adjoint`` have been updated to clarify 
  optional developer-facing use cases.
  [(#7999)](https://github.com/PennyLaneAI/pennylane/pull/7999)

* The docstring of the `is_hermitian` operator property has been updated to better describe its behaviour.
  [(#7946)](https://github.com/PennyLaneAI/pennylane/pull/7946)

* The docstrings of all optimizers have been improved for consistency and legibility.
  [(#7891)](https://github.com/PennyLaneAI/pennylane/pull/7891)

* The code example in the documentation for :func:`~.transforms.split_non_commuting` has been 
  updated to give the correct output.
  [(#7892)](https://github.com/PennyLaneAI/pennylane/pull/7892)

* The :math:`\LaTeX` rendering in the documentation for `qp.TrotterProduct` and `qp.trotterize` 
  has been corrected.
  [(#8014)](https://github.com/PennyLaneAI/pennylane/pull/8014)

* The docstring of ``ClassicalShadow.entropy`` has been updated to trim out the outdated part of an 
  explanation about the different choices of the ``alpha`` parameter.
  [(#8100)](https://github.com/PennyLaneAI/pennylane/pull/8100)

* A warning has been added to the :doc:`interfaces documentation </introduction/interfaces>`
  under the Pytorch section to explain that all Pytorch floating-point inputs are promoted to 
  ``torch.float64``.
  [(#8124)](https://github.com/PennyLaneAI/pennylane/pull/8124)

* The :doc:`Dynamic Quantum Circuits </introduction/dynamic_quantum_circuits>` page has been updated 
  to include the latest device-dependent mid-circuit measurement method defaults.
  [(#8149)](https://github.com/PennyLaneAI/pennylane/pull/8149)
  [(#8444)](https://github.com/PennyLaneAI/pennylane/pull/8444)

* A syntax rendering issue in the
  :doc:`DefaultQubit documentation </code/api/pennylane.devices.default_qubit.DefaultQubit>` has 
  been fixed to correctly display the ``max_workers`` parameter.
  [(#8289)](https://github.com/PennyLaneAI/pennylane/pull/8289)

<h3>Bug fixes 🐛</h3>

* Fixed a bug in ``default.qubit`` where the device wasn't properly validating the ``mcm_method`` keyword argument.
  [(#8343)](https://github.com/PennyLaneAI/pennylane/pull/8343)

* An error is now raised if postselection is requested for a zero-probability mid-circuit measurement outcome with finite
  shots and :class:`~pennylane.devices.DefaultQubit` when ``mcm_method="deferred"`` and ``postselect_mode="fill-shots"``, as
  this previously led to invalid results.
  [(#8389)](https://github.com/PennyLaneAI/pennylane/pull/8389)

* Applying a transform to a ``QNode`` with capture enabled now returns a ``QNode``. This allows autograph
  to transform the user function when transforms are applied to the ``QNode``.
  [(#8307)](https://github.com/PennyLaneAI/pennylane/pull/8307)
  
* ``qp.compiler.python_compiler.transforms.MergeRotationsPass`` now takes the ``adjoint`` property of
  merged operations correctly into account.
  [(#8429)](https://github.com/PennyLaneAI/pennylane/pull/8429)
  
* Promoting NumPy data to autograd no longer occurs in ``qp.qchem.molecular_hamiltonian``.
  [(#8410)](https://github.com/PennyLaneAI/pennylane/pull/8410)

* Fixed compatibility with JAX and PyTorch input parameters in :class:`~.SpecialUnitary` when large
  numbers of wires are used.
  [(#8209)](https://github.com/PennyLaneAI/pennylane/pull/8209)

* With ``qp.capture.enable()``, AutoGraph will now be correctly applied to functions containing 
  control flow that are then wrapped in :func:`~pennylane.adjoint` or :func:`~pennylane.ctrl`.
  [(#8215)](https://github.com/PennyLaneAI/pennylane/pull/8215)

* Fixed a bug that was causing parameter broadcasting on ``default.mixed`` with diagonal gates in
  the computational basis to raise an error.
  [(#8251)](https://github.com/PennyLaneAI/pennylane/pull/8251)

* ``qp.ctrl(qp.Barrier(), control_wires)`` now just returns the original ``Barrier`` operation, 
  but placed in the circuit where the ``ctrl`` happens.
  [(#8238)](https://github.com/PennyLaneAI/pennylane/pull/8238)

* JIT compilation of :class:`~pennylane.MottonenStatePrep` can now accept statically defined state-vector arrays.
  [(#8222)](https://github.com/PennyLaneAI/pennylane/pull/8222)

* Pauli arithmetic operations (e.g., ``op.simplify()``) can now handle abstract/runtime coefficients 
  when participating in a jitted function.
  [(#8190)](https://github.com/PennyLaneAI/pennylane/pull/8190)

* Operators queued with :func:`pennylane.apply` no longer get dequeued by subsequent dequeuing operations
  (e.g., :func:`pennylane.adjoint`).
  [(#8078)](https://github.com/PennyLaneAI/pennylane/pull/8078)

* Fixed a bug in the decomposition rules of :class:`~.Select` with the graph-based decomposition 
  system that broke the decompositions if the target ``ops`` of the ``Select`` operator were 
  parametrized. This enables the graph-based decomposition system with ``Select`` being provided 
  parametrized target ``ops``.
  [(#8186)](https://github.com/PennyLaneAI/pennylane/pull/8186)

* ``Exp`` and ``Evolution`` now have improved decompositions, allowing them to handle more 
  situations more robustly. In particular, the generator is simplified prior to decomposition. Now, 
  more time evolution operators can be supported on devices that do not natively support them.
  [(#8133)](https://github.com/PennyLaneAI/pennylane/pull/8133)

* A scalar product of a norm one scalar and an operator now decomposes into a ``GlobalPhase`` and 
  the operator. For example, ``-1 * qp.X(0)`` now decomposes into 
  ``[qp.GlobalPhase(-np.pi), qp.X(0)]``. This improves the decomposition of ``Select`` when there 
  are complicated target ``ops``.
  [(#8133)](https://github.com/PennyLaneAI/pennylane/pull/8133)

* Fixed a bug that made the queueing behaviour of 
  :meth:`qp.PauliWord.operation <~.PauliWord.operation>` and 
  :meth:`qmle.PauliSentence.operation <~.PauliSentence.operation>` depndent on the global state of a
  program due to a caching issue.
  [(#8135)](https://github.com/PennyLaneAI/pennylane/pull/8135)

* A more informative error is raised when extremely deep circuits are attempted to be drawn.
  [(#8139)](https://github.com/PennyLaneAI/pennylane/pull/8139)

* An error is now raised if sequences of classically processed mid-circuit measurements
  are used as input to :func:`pennylane.counts` or :func:`pennylane.probs` (e.g., 
  ``qp.counts([2*qp.measure(0), qp.measure(1)])``)
  [(#8109)](https://github.com/PennyLaneAI/pennylane/pull/8109)

* Simplifying operators raised to integer powers no longer causes recursion errors.
  [(#8044)](https://github.com/PennyLaneAI/pennylane/pull/8044)

* Fixed a GPU selection issue in ``qp.math`` with PyTorch when multiple GPUs are present.
  [(#8008)](https://github.com/PennyLaneAI/pennylane/pull/8008)

* The :func:`~.for_loop` function with capture enabled can now properly handle cases when 
  ``start == stop``.
  [(#8026)](https://github.com/PennyLaneAI/pennylane/pull/8026)

* Plxpr primitives now only return dynamically shaped arrays if their outputs actually have dynamic 
  shapes.
  [(#8004)](https://github.com/PennyLaneAI/pennylane/pull/8004)

* Fixed an issue with the tree-traversal MCM method and non-sequential wire orders that produced 
  incorrect results.
  [(#7991)](https://github.com/PennyLaneAI/pennylane/pull/7991)

* Fixed a bug in :func:`~.matrix` where an operator's constituent gates in its decomposition were 
  incorrectly queued, causing extraneous gates to appear in the circuit.
  [(#7976)](https://github.com/PennyLaneAI/pennylane/pull/7976)

* An error is now raised if an ``end`` statement is found in a measurement conditioned branch in a 
  QASM string being imported into PennyLane.
  [(#7872)](https://github.com/PennyLaneAI/pennylane/pull/7872)

* Fixed issue related to :func:`~.transforms.to_zx` adding the support for ``Toffoli`` and ``CCZ``
  gates conversion into their ZX-graph representation.
  [(#7899)](https://github.com/PennyLaneAI/pennylane/pull/7899)

* Fixed ``qp.workflow.get_best_diff_method`` to correctly align with ``execute`` and ``construct_batch`` logic in    
  the workflow module for internal consistency.
  [(#7898)](https://github.com/PennyLaneAI/pennylane/pull/7898)

* Issues were resolved with AutoGraph transforming internal PennyLane library code in addition to 
  user-level code, which was causing downstream errors in Catalyst.
  [(#7889)](https://github.com/PennyLaneAI/pennylane/pull/7889)

* Fixed a bug that caused calls to ``QNode.update`` (e.g., ``circuit.update(...)(shots=10)``) to
  update the shots value as if ``set_shots`` had been applied, causing unnecessary warnings to
  appear.
  [(#7881)](https://github.com/PennyLaneAI/pennylane/pull/7881)

* Fixed attributes and types in the quantum dialect in the unified compiler framework that now 
  allows for types to be inferred correctly when parsing.
  [(#7825)](https://github.com/PennyLaneAI/pennylane/pull/7825)

* Fixed a bug in ``SemiAdder`` that was causing failures when inputs were defined with a single 
  wire.
  [(#7940)](https://github.com/PennyLaneAI/pennylane/pull/7940)
  [(#8437)](https://github.com/PennyLaneAI/pennylane/pull/8437)

* Fixed a bug where ``qp.prod``, ``qp.matrix``, and ``qp.cond`` applied on a quantum function 
  was not dequeueing operators passed as arguments to the function.
  [(#8094)](https://github.com/PennyLaneAI/pennylane/pull/8094)
  [(#8119)](https://github.com/PennyLaneAI/pennylane/pull/8119)
  [(#8078)](https://github.com/PennyLaneAI/pennylane/pull/8078)

* Fixed a bug where a copy of ``ShadowExpvalMP`` was incorrect for a multi-term composite 
  observable.
  [(#8078)](https://github.com/PennyLaneAI/pennylane/pull/8078)

* Fixed a bug where :func:`~.transforms.cancel_inverses`, :func:`~.transforms.merge_rotations`, 
  :func:`~.transforms.single_qubit_fusion`, :func:`~.transforms.commute_controlled`, and 
  :func:`~.transforms.clifford_t_decomposition` were giving incorrect results when acting on 
  [(#8297)](https://github.com/PennyLaneAI/pennylane/pull/8297)
  [(8297)](https://github.com/PennyLaneAI/pennylane/pull/8297)

* When using ``mcm_method="tree-traversal"`` with ``qp.samples``, the data type of the returned
  values is now ``int``. This change ensures consistency with the output of other MCM methods.
  [(#8274)](https://github.com/PennyLaneAI/pennylane/pull/8274)

* The labels for operators that have multiple matrix-valued parameters (e.g. those from 
  :class:`~.operation.Operator`) can now also be drawn correctly (e.g. with ``qp.draw``).     
  [(#8432)](https://github.com/PennyLaneAI/pennylane/pull/8432)

* Fixed a bug with `~.estimator.resource_mapping._map_to_resource_op()` where it was incorrectly
  mapping the `~.TrotterProduct` template.
  [(#8425)](https://github.com/PennyLaneAI/pennylane/pull/8425)
  
* Fixed bugs in the :mod:`~.estimator` module pertaining to tracking resource operator names,
  as well as the handling of decompositions and measurement operators by the mapper
  used by the :func:`~.estimator.estimate.estimate` function.
  [(#8384)](https://github.com/PennyLaneAI/pennylane/pull/8384)
  
* Fixed a bug where :func:`~.ops.rs_decomposition` logic to streamline queuing conditions were 
  applied incorrectly.
  [(#8441)](https://github.com/PennyLaneAI/pennylane/pull/8441)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Runor Agbaire,
Guillermo Alonso,
Ali Asadi,
Utkarsh Azad,
Astral Cai,
Joey Carter,
Yushao Chen,
Isaac De Vlugt,
Diksha Dhawan,
Gabriela Sanchez Diaz,
Marcus Edwards,
Lillian Frederiksen,
Pietropaolo Frisoni,
Simone Gasperini,
Diego Guala,
Sengthai Heng
Austin Huang,
David Ittah,
Soran Jahangiri,
Korbinian Kottmann,
Elton Law,
Mehrdad Malekmohammadi,
Pablo Antonio Moreno Casares,
Anton Naim Ibrahim,
Erick Ochoa,
Lee James O'Riordan,
Mudit Pandey,
Andrija Paurevic,
Justin Pickering,
Alex Preciado,
Shuli Shu,
Jay Soni,
Paul Haochen Wang,
David Wierichs,
Jake Zaia.
