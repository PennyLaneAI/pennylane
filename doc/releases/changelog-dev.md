# Release 0.46.0 (development release)

<h3>New features since last release</h3>

* A new arithmetic template called :class:`~.SignedOutMultiplier` has been added that multiplies numbers encoded in the
  input registers using a two's complement.
  [(#9458)](https://github.com/PennyLaneAI/pennylane/pull/9458)

  ```python
  x = -3
  y = 3

  x_wires = [0, 1, 2]
  y_wires = [3, 4, 5]
  output_wires = [6, 7, 8, 9, 10, 11]
  work_wires = [12, 13, 14, 15]

  dev = qp.device("default.qubit")

  @qp.qnode(dev, shots=1)
  def circuit():
      qp.BasisEmbedding(x, wires=x_wires)
      qp.BasisEmbedding(y, wires=y_wires)
      qp.SignedOutMultiplier(
          x_wires,
          y_wires,
          output_wires,
          work_wires,
          output_wires_zeroed=True,
      )
      return qp.sample(wires=output_wires)
  ```

  ```pycon
  >>> print(circuit())
  [[1 1 0 1 1 1]]

  ```

  The result :math:`[[1 1 0 1 1 1]]`, is the binary representation of :math:`-3 \cdot 3 \; = -9` in 2s complement form.

* Added an :class:`~.Incrementer` template that increments a bitstring encoded in a quantum state
  by 1, in twos complement. Based on `Gidney's blog <https://algassert.com/circuits/2015/06/12/Constructing-Large-Increment-Gates.html>`__.
  [(#9458)](https://github.com/PennyLaneAI/pennylane/pull/9458)

  In this example, we add :math:`2 + 1` to get :math:`3`, using the `Incrementer`.

  ```python
  from pennylane import qnode, device, sample, BasisEmbedding, Incrementer
  import numpy as np

  wires = [0, 1, 2]
  work_wires = [3, 4]
  init_state = [0, 1, 0]  # binary representation of 2

  dev = device("default.qubit", wires=wires + work_wires)

  @qnode(dev, shots=1)
  def increment(wires, init_state, work_wires=None):
      BasisEmbedding(init_state, wires)
      Incrementer(wires, work_wires)
      return sample()

  result = increment(wires, init_state, work_wires)[0]

  ```

  ```pycon
  >>> result[:len(wires)]
  array([0, 1, 1])

  ```

  The result incremented the binary value in the non-work wires by 1: :math:`(010)_2 + (001)_2 = (011)_2`.

* Added new templates :class:`~.OutSquare` and :class:`SignedOutSquare` for out-place squaring
  a quantum register in unsigned or signed encoding convention into another quantum register.
  [(#9003)](https://github.com/PennyLaneAI/pennylane/pull/9003)
  [(#9558)](https://github.com/PennyLaneAI/pennylane/pull/9558)

* A new :func:`~.single_qubit_zyz_angles` function that returns the pre-defined rotation angles
  of a ZYZ decomposition of a single-qubit operator has been added.
  [(#9502)](https://github.com/PennyLaneAI/pennylane/pull/9502)

  ```pycon
  >>> qp.single_qubit_zyz_angles(qp.H(0))
  (3.14159..., 1.57079..., 0.0, 1.57079...)

  ```

  The new function returns a tuple of four values, where the first three corresponds to the rotation
  angles of the ZYZ decomposition of this operator, and the last one corresponds to the global phase.

* :func:`~.specs` will now output symbolic resource information when it encounters a loop that uses dynamic control-flow
  that can't be resolved at compile time.
  In such cases the returned :class:`~.resource.CircuitSpecs` will contain :class:`~.resource.SymbolicSpecsResources` instances instead of the usual :class:`~.resource.SpecsResources` instances.

  ```python
  @qp.qjit(autograph=True)
  @qp.qnode(qp.device("lightning.qubit", wires=1))
  def circuit(x):
      qp.Hadamard(0)
      qp.PauliX(0)
      for _ in range(x):
          qp.PauliX(0)
      return qp.expval(qp.PauliX(0))

  specs_result = qp.specs(circuit, level=0)(5)
  ```

  ```pycon
  >>> print(specs_result)
  Device: lightning.qubit
  Device wires: 1
  Shots: Shots(total=None)
  Level: Before MLIR Passes
  <BLANKLINE>
  Symbolic Variables: a
  Wire allocations: 1
  Total gates: a + 2
  Gate counts:
  - Hadamard: 1
  - PauliX: a + 1
  Measurements:
  - expval(PauliX): 1
  Depth: Not computed

  ```

  These symbolic resources include expressions with variables which can substituted for concrete values to compute the associated resources for a circuit, via the ``subs`` method.

  ```pycon
  >>> res = specs_result.resources
  >>> print(res.subs(a=5))
  Wire allocations: 1
  Total gates: 7
  Gate counts:
  - Hadamard: 1
  - PauliX: 6
  Measurements:
  - expval(PauliX): 1
  Depth: Not computed

  ```

* A new template for probabilistic state preparation based on the flip-flop QRAM construction is now available, named :class:`~.FFQRAM`. Given real amplitudes and a list of bitstring addresses, the template embeds the corresponding sparse computational-basis state, with the desired state obtained by post-selecting on the register qubit.
  [(#9498)](https://github.com/PennyLaneAI/pennylane/pull/9498)
  [(#9598)](https://github.com/PennyLaneAI/pennylane/pull/9598)

  For example, the following circuit creates the state :math:`\sqrt{0.3}|000\rangle + \sqrt{0.7}|001\rangle` in the first three wires when the last wire is measured to be :math:`|1\rangle`.

  ```python
  import pennylane as qp

  addrs = ["000", "001"]
  amps = qp.math.array([qp.math.sqrt(0.3), qp.math.sqrt(0.7)])
  wires = qp.registers({"address": 3, "register": 1})
  shots = 1000

  @qp.set_shots(shots)
  @qp.qnode(qp.device("default.qubit", seed=42))
  def circuit():
      qp.FFQRAM(
          amplitudes=amps,
          wires=wires["address"] + wires["register"],
          address=addrs,
      )
      qp.measure(wires["register"], postselect=1)
      return qp.probs(wires=wires["address"])

  ```

  ```pycon
  >>> print(qp.draw(circuit, level=2)())
  0: ──H──X─╭●─────────X──X─╭●─────────X────┤ ╭Probs
  1: ──H──X─├●─────────X──X─├●─────────X────┤ ├Probs
  2: ──H──X─├●─────────X────├●──────────────┤ ╰Probs
  3: ───────╰RY(1.16)───────╰RY(1.98)──┤↗₁├─┤

  ```

* A new template for Fast Fermionic Fourier Transforms called :class:`~.FFFT` has been added.
  This algorithm is based on [Ferris (2013)](https://arxiv.org/abs/1310.7605) and applies to
  efficient simulation of quantum materials and chemistry systems.
  [(#9354)](https://github.com/PennyLaneAI/pennylane/pull/9354)

  The :class:`~.FFFT` template is decomposed recursively into two parallel FFFTs over
  :math:`\tfrac{n}{2}` sites in each iteration of the recursion. These parallel Fourier transforms
  are followed by a series of two-site linear gates.

  ```python
  import pennylane as qp

  dev = qp.device("default.qubit")

  @qp.qnode(dev)
  def circuit():
      qp.FFFT(wires=(0, 1, 2, 3))
      return qp.state()
  ```

  ```pycon
  >>> print(qp.draw(circuit, level="device")())
  0: ─╭TwoWireFFT────────────────────╭TwoWireFFT──────────────┤ ╭State
  1: ─╰TwoWireFFT───────╭FSWAP(3.14)─╰TwoWireFFT─╭FSWAP(3.14)─┤ ├State
  2: ─╭TwoWireFFT──Z⁰⋅⁰─╰FSWAP(3.14)─╭TwoWireFFT─╰FSWAP(3.14)─┤ ├State
  3: ─╰TwoWireFFT──Z⁰⋅⁵──────────────╰TwoWireFFT──────────────┤ ╰State

  ```

  Alongside the addition of :class:`~.FFFT`, a new operation called :class:`~.TwoWireFFT`
  has been added to enable its implementation: the :class:`~.FFFT` operation is
  decomposed recursively into :class:`~.FermionicSWAP` and :class:`~.TwoWireFFT` operations
  (two-site Fermionic Fourier transforms).

* A new operation :class:`~.QutritDensityMatrix` has been added to initialize density matrix states for the device
  `qp.devices.DefaultQutritMixed`.
  [(#9538)](https://github.com/PennyLaneAI/pennylane/pull/9538)

  ```python
  import pennylane as qp
  nr_wires = 1
  rho = np.zeros((3 ** nr_wires, 3 ** nr_wires), dtype=np.complex128)
  rho[2, 2] = 1  # initialize the pure state density matrix for the |2><2| state

  dev = qp.device("default.qutrit.mixed", wires=1)
  @qp.qnode(dev)
  def circuit():
      qp.QutritDensityMatrix(rho, wires=[0])
      return qp.state()
  ```

  ```pycon
  >>> circuit()
  array([[[0.+0.j, 0.+0.j, 0.+0.j],
          [0.+0.j, 0.+0.j, 0.+0.j],
          [0.+0.j, 0.+0.j, 1.+0.j]]])

  ```

<h3>Improvements 🛠</h3>

* Type aliases `Int`, `Float`, `Complex`, `Bool`, and `Wire` have been introduced to allow for intuitive 
  abstract type notation.  
  [(#9701)](https://github.com/PennyLaneAI/pennylane/pull/9701)

  ```python
  from pennylane.typing import Int, Float, Complex, Bool, Wire
  Int[2, 3]       # Abstract int array with shape (2, 3)
  Float           # Float scalar
  Complex[...]    # Abstract complex array with any shape
  Bool[-1, 3, 4]  # Abstract bool array with dynamic size for the first axis
  Wire            # Single abstract wire
  Wire[4]         # Four abstract wires
  Wire[-1]        # Wire sequence with dynamic size
  ```
  For example, these abstract types can be used to do type-checking on concrete values:
  
  ```pycon
  >>> isinstance(np.array(False), qp.typing.Bool)
  True
  >>> qp.typing.Bool[4]
  AbstractArray(shape=(4,), dtype=dtype('bool'))
  >>> isinstance(np.array(0+1.2j), qp.typing.Complex)
  True
  >>> qp.typing.Complex[..., 2]
  AbstractArray(shape=(Ellipsis, 2), dtype=dtype('complex128'))
  >>> isinstance(qp.wires.Wires([0, 1]), qp.typing.Wire[2])
  True
  >>> qp.typing.Wire[2]
  AbstractWires(num_wires=2)
  
  ```

* `qp.draw` now has improved drawing for dynamic wire allocation with `qp.allocate`.
  [(#9545)](https://github.com/PennyLaneAI/pennylane/pull/9545)

* Data from :func:`~.specs` now have markdown formatting for IPython, improving their readability;
  particularly :class:`~.resource.CircuitSpecs` and :class:`~.resource.SpecsResources`.
  [(#9679)](https://github.com/PennyLaneAI/pennylane/pull/9679)
  [(#9585)](https://github.com/PennyLaneAI/pennylane/pull/9585)

* Added a decomposition of `DiagonalQubitUnitary` into a single `RZ` multiplexer, i.e.
  `SelectPauliRot(..., rot_axis="Z")`, onto an auxiliary qubit. This is a particularly favourable
  decomposition when using phase-gradient based decompositions of multiplexers.
  [(#9593)](https://github.com/PennyLaneAI/pennylane/pull/9593)

* :func:`~pennylane.draw` now renders :class:`~.SelectPauliRot` and :class:`~.QROM` with
  multiplexer selector symbols on the control wires and a Pauli rotation and a "QROM" label,
  respectively, on the target wire(s).
  [(#9604)](https://github.com/PennyLaneAI/pennylane/pull/9604)
  [(#9692)](https://github.com/PennyLaneAI/pennylane/pull/9692)

* `AbstractArray` has been added to
  `pennylane.typing`, and `AbstractWires` has been added to `pennylane.wires`.
  These will support a new method of having compressed operators for resource estimation
  and decomposition.
  [(#9385)](https://github.com/PennyLaneAI/pennylane/pull/9385)

* `Tracker` now has a readable `__repr__` that displays all relevant internals
  (`active`, `totals`, `history`, `latest`, `persistent`, `callback`).
  [(#9575)](https://github.com/PennyLaneAI/pennylane/pull/9575)

  ```pycon
  >>> tracker = qp.Tracker()
  >>> tracker.update(a=2, b="b2", c=1)
  >>> print(tracker)
  Tracker(active=False, totals={'a': 2, 'c': 1}, history={'a': [2], 'b': ['b2'], 'c': [1]}, latest={'a': 2, 'b': 'b2', 'c': 1}, persistent=False, callback=None)

  ```

* :func:`~pennylane.draw`, :func:`~pennylane.draw_mpl`, and :func:`~.specs` now support
  ``functools.partial`` wrappers around supported circuit callables.
  [(#9595)](https://github.com/PennyLaneAI/pennylane/pull/9595)

* Updated the preprocessing of target state vectors for `MottonenStatePreparation` and
  `MultiplexerStatePreparation` to produce only `RY` rotation angles for real target state vectors
  that contain negative signs. This allows the preparation circuits to skip phase gates when the
  phases are purely real, i.e. :math:`\pm 1`.
  [(#9561)](https://github.com/PennyLaneAI/pennylane/pull/9561)

* Instances of `C(Prod)` now have a significantly more efficient decomposition in terms of `TemporaryAND` operators when work wires are provided.

  For example, a controlled multi-target-``X`` operation previously decomposed as

  ```
  c1: ─╭●─╭●─╭●─╭●─┤  State
  c2: ─├●─├●─├●─├●─┤  State
  c3: ─├●─├●─├●─├●─┤  State
   3: ─╰X─│──│──│──┤  State
   2: ────╰X─│──│──┤  State
   1: ───────╰X─│──┤  State
   0: ──────────╰X─┤  State
  ```

  With this upgrade, it decomposes into a ``TemporaryAND`` ladder and individual ``CNOT`` gates when work wires are available:

  ```python
  @qp.transforms.decompose(
      gate_set={"TemporaryAND":4, "Adjoint(TemporaryAND)":1, "MultiControlledX":7, "CNOT":1}
  )
  @qp.qnode(qp.device("default.qubit"))
  def qnode():
      qp.ctrl(qp.X(0) @ qp.X(1) @ qp.X(2) @ qp.X(3), control=["c1", "c2", "c3"], work_wires=["w1", "w2"], work_wire_type="zeroed")
      return qp.state()

  print(qp.draw(qnode)())
  ```

  ```
  c1: ─╭●─────────────────────●╮─┤  State
  c2: ─├●─────────────────────●┤─┤  State
  w1: ─╰⊕─╭●──────────────●╮──⊕╯─┤  State
  c3: ────├●──────────────●┤─────┤  State
  w2: ────╰⊕─╭●─╭●─╭●─╭●──⊕╯─────┤  State
   3: ───────╰X─│──│──│──────────┤  State
   2: ──────────╰X─│──│──────────┤  State
   1: ─────────────╰X─│──────────┤  State
   0: ────────────────╰X─────────┤  State
  ```
  [(#9368)](https://github.com/PennyLaneAI/pennylane/pull/9368)

* Updated `qp.registers` to accept empty registers (e.g., `qp.registers({"algo_wires": 5, "work_wires": 0})).
  [(#9543)](https://github.com/PennyLaneAI/pennylane/pull/9543)

* Removed instances of using the deprecated way to set shots on a device `device(..., shots=...)`.
  [(#9495)](https://github.com/PennyLaneAI/pennylane/pull/9495)

* Added three decompositions of :class:`~.OutMultiplier` that use significantly fewer costly gates
  than the existing QFT-based decomposition, at the cost of more auxiliary wires.
  In addition added a new argument ``output_wires_zeroed`` to ``OutMultiplier`` that can be
  used to indicate ``output_wires`` to be in the :math:`|0\rangle` state, leading to cheaper
  decompositions.
  [(#8900)](https://github.com/PennyLaneAI/pennylane/pull/8900)

* Added a dispatcher for `qp.pauli_measure` to call `catalyst.pauli_measure` when qjit is enabled
  while using the non-capture workflow. This also added an alias for `MidCircuitPauliMeasure` for
  decomposition.
  [(#9506)](https://github.com/PennyLaneAI/pennylane/pull/9506)

* A more informative error message is raised when quantum functions without registered resource
  estimates are passed to the `fixed_decomps` and `alt_decomps` arguments of the :func:`~.transforms.decompose` transform.
  [(#9528)](https://github.com/PennyLaneAI/pennylane/pull/9528)

* The output of :func:`~.decomposition.inspect_decomps` and :func:`~.transforms.decomp_inspector` is
  now formatted for clearer visual inspection when used in a Jupyter notebook environment.
  [(#9518)](https://github.com/PennyLaneAI/pennylane/pull/9518)

* The Ross-Selinger decomposition method :func:`~.ops.rs_decomposition` when used with Catalyst
  no longer queues a Catalyst conditional operator, if the conditional predicate for whether a
  leading T gate exists is statically known, and the `compile_without_static_conditionals` flag
  in Catalyst is set. Instead, the static conditional will be evaluated at trace time, and only
  the correct branch will be queued.
  [(#9630)](https://github.com/PennyLaneAI/pennylane/pull/9630)
  [(#9648)](https://github.com/PennyLaneAI/pennylane/pull/9648)

* The `DecompositionGraph` now skips applying `adjoint` or `ctrl` to decomposition rules that
  contain mid-circuit measurements, also skips applying `adjoint` to decomposition rules that
  contain dynamic wire allocations.
  [(#9629)](https://github.com/PennyLaneAI/pennylane/pull/9629)

* The function `qp.math.partial_trace()` has been changed to include a `qudit_dim` keyword argument to allow for partial traces of
  any qudit density matrices with constant qudit dimension.
  [(#9538)](https://github.com/PennyLaneAI/pennylane/pull/9538)

* Device `default.qutrit.mixed` now implements state preparation operations with batched initial states.
  [(#9538)](https://github.com/PennyLaneAI/pennylane/pull/9538)

<h3>Labs: a place for unified and rapid prototyping of research software 🧪</h3>

* Updated the `make_selectpaulirot_to_phase_gradient_decomp` decomposition rule factory to have
  the decomposition rule validate the number of available work wires against the needed work wires
  to use unary iteration in the decomposition of the used `QROM` operation for the specified 
  number of control wires/angles.
  [(#9655)](https://github.com/PennyLaneAI/pennylane/pull/9655)

* Added a variant of `SumOfSlatersPrep` to labs, accessible as `labs.templates.SumOfSlatersPrep2`.
  This variant handles work wires explicitly instead of allocating them dynamically in the
  decomposition. This enables usage of `SumOfSlatersPrep2` with `qp.qjit` with
  capture _disabled_ (`qp.capture.disable()`).
  [(#9539)](https://github.com/PennyLaneAI/pennylane/pull/9539)

* Updated the `make_selectpaulirot_to_phase_gradient_decomp` and `make_rz_to_phase_gradient_decomp` 
  decomposition rule factories to be compatible with program capture.
  [(#9537)](https://github.com/PennyLaneAI/pennylane/pull/9537)
  [(#9481)](https://github.com/PennyLaneAI/pennylane/pull/9481)

* Created a new ``labs.templates.LeftQuantumComparator`` template for performing inequality test of two quantum registers.
  [(#9277)](https://github.com/PennyLaneAI/pennylane/pull/9277)

  ```python
  import pennylane as qp
  from pennylane.labs.templates import LeftQuantumComparator

  dev = qp.device("lightning.qubit")

  @qp.set_shots(shots=1)
  @qp.qnode(dev)
  def circuit(a, comparator, b):
    x_wires = [0, 3, 6, 9]
    y_wires = [1, 4, 7, 10]
    work_wires = [2, 5, 8]
    qp.BasisState(a, wires=x_wires)
    qp.BasisState(b, wires=y_wires)
    LeftQuantumComparator(x_wires, y_wires, 11, work_wires, comparator)
    qp.CNOT(wires=[11, 12])
    qp.adjoint(LeftQuantumComparator(x_wires, y_wires, 11, work_wires, comparator))

    return qp.sample(wires=[12])
  ```

  ```pycon
    >>> output = circuit(3, ">=", 2)
    >>> print(bool(output))
    True

  ```

* Created a new ``labs.templates.LeftClassicalComparator`` template for performing an inequality
  test of a quantum register and an integer.
  [(#9308)](https://github.com/PennyLaneAI/pennylane/pull/9308)

  ```python
  import pennylane as qp
  from pennylane.labs.templates import LeftClassicalComparator

  dev = qp.device("lightning.qubit", wires=6)

  @qp.set_shots(shots=1)
  @qp.qnode(dev)
  def circuit(x_val, L_val):
    qp.BasisState(x_val, wires=[0, 1, 2])

    LeftClassicalComparator(
        x_wires=[0, 1, 2],
        L=L_val,
        target_wire=3,
        work_wires=[4, 5],
        comparator='>='
    )
    return qp.sample(wires=3)
  ```

  ```pycon
    >>> output = circuit(3, 2)
    >>> print(bool(output)) # 3 >= 2
    True

  ```

* Update phase gradient transforms to use ``BasisState`` instead of ``BasisEmbedding``.
  This is an improvement as the latter is not consistently dispatched to ``C(BasisState)`` in ``controlled_resource_rep``, which
  led to compilation errors when using the old Catalyst frontend ``catalyst.device.decomposition.catalyst_decompose``.
  [(#9493)](https://github.com/PennyLaneAI/pennylane/pull/9493)

* Created a new ``labs.estimator_beta.SelectCopyQROM`` resource operator which uses an optimal
  decomposition to estimate the cost for QROM.
  [(#9500)](https://github.com/PennyLaneAI/pennylane/pull/9500)
  [(#9516)](https://github.com/PennyLaneAI/pennylane/pull/9516)

  ```pycon
    >>> import pennylane.labs.estimator_beta as qre
    >>>
    >>> qrom_op = qre.SelectCopyQROM(
    ...     num_bitstrings = 10**8,
    ...     size_bitstring = 8,
    ...     available_dirty_aux = 300,
    ... )
    >>>
    >>> print(qre.estimate(qrom_op))
    --- Resources: ---
    Total wires: 308
      algorithmic wires: 35
      allocated wires: 273
        zero state: 273
        any state: 0
    Total gates : 4.781E+8
      'Toffoli': 3.520E+6,
      'CNOT': 4.570E+8,
      'X': 7.036E+6,
      'Hadamard': 1.055E+7

  ```

* Created a :func:`~.pennylane.labs.templates.trotter_fragmented` function to run specialized
  Trotter circuits for fragmented Hamiltonians. This is used in modern quantum chemistry
  application algorithms.
  [(#9459)](https://github.com/PennyLaneAI/pennylane/pull/9459)

<h3>Breaking changes 💔</h3>

* Plxpr transforms have been removed.
  [(#9637)](https://github.com/PennyLaneAI/pennylane/pull/9637)

* Support for executing PLxPR without qjit has been removed.
  [(#9678)](https://github.com/PennyLaneAI/pennylane/pull/9678)
  [(#9682)](https://github.com/PennyLaneAI/pennylane/pull/9682)

* :class:`~.IQP` no longer accepts `num_wires`. Instead, `wires` should be passed
  explicitly, to match the behaviour of all other `Operator` classes.
  [(#9419)](https://github.com/PennyLaneAI/pennylane/pull/9419)

  Instead of the following call: `qp.IQP(weights=[0.85, 0.21], num_wires=2, pattern=[[[0]], [[1]]], spin_sym=True)`,
  we would now need to provide the wire labels themselves i.e.

  ```python
  qp.IQP(weights=[0.85, 0.21], wires=[0, 1], pattern=[[[0]], [[1]]], spin_sym=True)
  ```

* `qp.queuing.process_queue` has been moved to `qp.tape.qscript.process_queue`.
  [(#9542)](https://github.com/PennyLaneAI/pennylane/pull/9542)

* The ability to specify shots as a keyword argument on call to a `QNode` is removed. Specifying the
  shots on creation of the `QNode` or using :func:`pennylane.set_shots` should be used instead.
  [(#9469)](https://github.com/PennyLaneAI/pennylane/pull/9469)

* `BoundTransform.transform` has been removed in favor of `BoundTransform.tape_transform`.
  [(#9471)](https://github.com/PennyLaneAI/pennylane/pull/9471/)

* :meth:`QuantumScript.expand`, :func:`~pennylane.tape.qscript.expand` and the related functions :func:`~pennylane.tape.expand_tape`,
  :func:`~pennylane.tape.expand_tape_state_prep`, and :func:`~pennylane.tape.create_expand_trainable_multipar`
  are removed. Instead, please use the
  :func:`qp.transforms.decompose <.transforms.decompose>` function for decomposing circuits.
  [(#9473)](https://github.com/PennyLaneAI/pennylane/pull/9473)

* The `id` keyword argument to :class:`~.qcut.MeasureNode` and :class:`~.qcut.PrepareNode` has been renamed to `node_uid`.
  [(#9467)](https://github.com/PennyLaneAI/pennylane/pull/9467)

* The `id` keyword argument to :class:`~.ops.MidMeasure` has been renamed to `meas_uid`.
  [(#9467)](https://github.com/PennyLaneAI/pennylane/pull/9467)

* The `id` keyword argument to :class:`~.measurements.MeasurementProcess` has been removed.
  [(#9467)](https://github.com/PennyLaneAI/pennylane/pull/9467)

* The `id` keyword argument to :class:`~.Operator` has been removed.
  [(#9467)](https://github.com/PennyLaneAI/pennylane/pull/9467)

* The :func:`~pennylane.workflow.get_transform_program` function has been removed.
  Instead, please use the improved :func:`~pennylane.workflow.get_compile_pipeline` to retrieve the execution pipeline
  of a QNode.
  [(#9466)](https://github.com/PennyLaneAI/pennylane/pull/9466)

* The `transform_program` property of `QNode` has been renamed to `compile_pipeline`.
  The deprecated access through `transform_program` has been removed.
  [(#9465)](https://github.com/PennyLaneAI/pennylane/pull/9465)

* ``Operation.basis`` is deprecated. :func:`~pennylane.is_commuting` instead should be used to determine whether or not
  two operators commute. For example, ``qp.is_commuting(my_op, qp.X(my_op.wires[0]))`` can be used to determine
  if ``my_op`` is in the ``X`` basis.
  [(#9476)](https://github.com/PennyLaneAI/pennylane/pull/9476)

* Providing a value of ``None`` to ``aux_wire`` of ``qp.gradients.hadamard_grad`` with ``mode="reversed"`` or ``mode="standard"``
  is no longer supported as of 0.46. An ``aux_wire`` will no longer be automatically assigned.
  [(#9468)](https://github.com/PennyLaneAI/pennylane/pull/9468)

* Setting `Operator._queue_category=None` and `MeasurementProcess._queue_category=None`
  to avoid processing the operator into the circuit is now removed.
  Instead, `Operator.queue` can be overwritten if needed.
  [(#9470)](https://github.com/PennyLaneAI/pennylane/pull/9470)

* Implementing ``Operator.generator`` as a property is no longer supported. Instead, define a ``generator()`` method for your operator that returns an ``Operator`` instance.
  [(#9662)](https://github.com/PennyLaneAI/pennylane/pull/9662)

* The `return_global_phase` keyword argument has been removed from the following helper methods in `qp.math`:

  - :math:`~.math.decomposition.zyz_rotation_angles`
  - :math:`~.math.decomposition.xyx_rotation_angles`
  - :math:`~.math.decomposition.xzx_rotation_angles`
  - :math:`~.math.decomposition.zxz_rotation_angles`
  - :math:`~.math.convert_to_su2`
  - :math:`~.math.convert_to_su4`

  These methods will now always return the additional global phase.
  [(#9496)](https://github.com/PennyLaneAI/pennylane/pull/9496)

  ```pycon
  >>> # You can always discard the last return value if the global phase is not needed.
  >>> U = qp.Hadamard(0).matrix()
  >>> phi, theta, omega, _ = qp.math.decomposition.zyz_rotation_angles(U)

  ```

* The ``qp.decomposition.reconstruct`` function and all infrastructure built around it has been removed.
  [(#9711)](https://github.com/PennyLaneAI/pennylane/pull/9711)

<h3>Deprecations 👋</h3>

* The ``simplify`` method in ``PauliSentence``, ``FermiSentence``, and ``BoseSentence`` are deprecated in favour of ``prune``, and will be removed in v0.47.
  [(#9487)](https://github.com/PennyLaneAI/pennylane/pull/9487)

* The ``Operator.hash`` and ``MeasurementProcess.hash`` properties have been deprecated and will be removed
  in v0.47. Please use the Python builtin ``hash(obj)`` function instead.
  [(#9488)](https://github.com/PennyLaneAI/pennylane/pull/9488)

* Using :func:`qp.templates.layer <.templates.layer>` is deprecated and will be removed in v0.47. Instead, please apply
  your unitary in a for loop.
  [(#9484)](https://github.com/PennyLaneAI/pennylane/pull/9484)

* The ``QuantumScript.adjoint`` (and ``QuantumTape.adjoint``) methods have been deprecated in v0.46. Instead, please use
  ``QuantumScript([adjoint(op) for op in reversed(tape.operations)])``.
  [(#9483)](https://github.com/PennyLaneAI/pennylane/pull/9483)

* The ``Operation.single_qubit_rot_angles()`` method is deprecated in favour of the new ``qp.single_qubit_zyz_angles(op)`` function, and will be removed in v0.47.

<h3>Internal changes ⚙️</h3>

* The CI workflow `Documentation Tests` has been renamed to `Test Documentation Code Examples`.
  [(#9710)](https://github.com/PennyLaneAI/pennylane/pull/9710)

* The `/benchmark` GitHub comment trigger can now accept additional arguments and has been renamed to `!benchmark`.
  [(#9676)](https://github.com/PennyLaneAI/pennylane/pull/9676)

* The core and JAX CI test suites now use the `least_duration` test-splitting algorithm,
  producing more balanced test groups across parallel CI jobs.
  [(#9519)](https://github.com/PennyLaneAI/pennylane/pull/9519)

* Improve language server support for `qp.capture`.
  [(#9657)](https://github.com/PennyLaneAI/pennylane/pull/9657)

* Bump `codecov-action` to `v7`.
  [(#9615)](https://github.com/PennyLaneAI/pennylane/pull/9615)

* :func:`~.parameter_frequencies` added that allows a user to retrieve `parameter_frequencies` from an
  :class:`~.Operation` or calculate them for an :class:`~.Operator2`.
  [(#9569)](https://github.com/PennyLaneAI/pennylane/pull/9569)

* Adds a new test fixture `preserve_jax_x64` to help automatically restore the `jax.config.jax_enable_x64` to prevent
  accidental context contamination.
  [(#9590)](https://github.com/PennyLaneAI/pennylane/pull/9590)

* New, experimental abstractions for creating PennyLane operators have been added, built around a new
  base class, :class:`~.Operator2`. This is an internal, work-in-progress effort that is being incrementally
  integrated into the PennyLane ecosystem. Supported functionality so far:
  - :func:`qp.equal` can check equality between two :class:`~.Operator2` instances.
  - :class:`~.StatePrepBase2`, based on :class:`~.Operator2`, is added.
  - :meth:`~.Operator2.decomposition` falls back to registered graph decomposition rules
    when ``compute_decomposition`` is not overridden.
  - Arithmetic can be performed with :class:`~.Operator2` instances.
  - :func:`qp.ops.functions.assert_valid` can verify that an :class:`~.Operator2` is defined properly.
  - Integration with :mod:`pennylane.capture`.
  [(#9525)](https://github.com/PennyLaneAI/pennylane/pull/9525)
  [(#9529)](https://github.com/PennyLaneAI/pennylane/pull/9529)
  [(#9526)](https://github.com/PennyLaneAI/pennylane/pull/9526)
  [(#9527)](https://github.com/PennyLaneAI/pennylane/pull/9527)
  [(#9562)](https://github.com/PennyLaneAI/pennylane/pull/9562)
  [(#9607)](https://github.com/PennyLaneAI/pennylane/pull/9607)
  [(#9596)](https://github.com/PennyLaneAI/pennylane/pull/9596)
  [(#9627)](https://github.com/PennyLaneAI/pennylane/pull/9627)
  [(#9659)](https://github.com/PennyLaneAI/pennylane/pull/9659)
  [(#9597)](https://github.com/PennyLaneAI/pennylane/pull/9597)
  [(#9647)](https://github.com/PennyLaneAI/pennylane/pull/9647)
  [(#9649)](https://github.com/PennyLaneAI/pennylane/pull/9649)
  [(#9556)](https://github.com/PennyLaneAI/pennylane/pull/9556)
  [(#9646)](https://github.com/PennyLaneAI/pennylane/pull/9646)
  [(#9674)](https://github.com/PennyLaneAI/pennylane/pull/9674)
  [(#9675)](https://github.com/PennyLaneAI/pennylane/pull/9675)
  [(#9683)](https://github.com/PennyLaneAI/pennylane/pull/9683)
  [(#9693)](https://github.com/PennyLaneAI/pennylane/pull/9693)
  [(#9685)](https://github.com/PennyLaneAI/pennylane/pull/9685)
  [(#9702)](https://github.com/PennyLaneAI/pennylane/pull/9702)

* Adds a new `pennylane/core` module.
  Moves the abstractions from `pennylane/operation` into `pennylane/core/operator`.
  Moves `MeasurementProcess`, `StateMeasurement`, `SampleMeasurement`, `MeasurementTransform`,
  `Shots`, `ShotCopies`, and `ShotsLike` to `pennylane.core`
  [(#9508)](https://github.com/PennyLaneAI/pennylane/pull/9508)
  [(#9586)](https://github.com/PennyLaneAI/pennylane/pull/9586)
  [(#9583)](https://github.com/PennyLaneAI/pennylane/pull/9583)

* ``assert_valid`` will now correctly raise an ``ImportError`` if `skip_capture=False` and JAX is not installed.
  [(#9567)](https://github.com/PennyLaneAI/pennylane/pull/9567)

* CI workflows now install CPU-only PyTorch (`--index-url https://download.pytorch.org/whl/cpu`)
  instead of the default GPU-enabled build. This eliminates transitive NVIDIA package downloads
  and reduces CI install times. The GPU test workflow (`tests-gpu.yml`) is excluded from this change.
  [(#9551)](https://github.com/PennyLaneAI/pennylane/pull/9551)
  [(#9559)](https://github.com/PennyLaneAI/pennylane/pull/9559)

* `Operator._queue_category` and `MeasurementProcess._queue_category` have been removed in favor of `isinstance` checks
  when processing an `AnnotatedQueue` into a `QuantumScript`.
  [(#9530)](https://github.com/PennyLaneAI/pennylane/pull/9530)

* Bump `autoray` package pin to `v0.8.10`.
  [(#9535)](https://github.com/PennyLaneAI/pennylane/pull/9535)

* Fixes imports of exceptions from `pennylane.operation` instead of `pennylane.exceptions`.
  [(#9512)](https://github.com/PennyLaneAI/pennylane/pull/9512)

* Documentation testing workflow now raises `PennyLaneDeprecationWarning` as errors.
  [(#9475)](https://github.com/PennyLaneAI/pennylane/pull/9475)

* Added support for JAX arrays as control wires during JAXpr evaluation.
  [(#9480)](https://github.com/PennyLaneAI/pennylane/pull/9480)

* Replaces arbitrary magic numbers across multiple modules with named, documented constants.
  Raw numeric literals in `pennylane/math`, `pennylane/ops`, `pennylane/devices`,
  `pennylane/gradients`, `pennylane/pauli`, `pennylane/qchem`, `pennylane/liealg`,
  `pennylane/fourier`, and `pennylane/templates` are now module-level constants with
  ``#:`` doc-comments explaining their purpose and origin. Unused constants
  ``eps`` in :mod:`pennylane.math` and ``tolerance`` in ``default_qutrit`` are removed.
  [(#9374)](https://github.com/PennyLaneAI/pennylane/pull/9374)

* Added usage of the `strict` keyword argument for `zip` throughout the codebase.
  [(#9393)](https://github.com/PennyLaneAI/pennylane/pull/9393)
  [(#9406)](https://github.com/PennyLaneAI/pennylane/pull/9406)
  [(#9413)](https://github.com/PennyLaneAI/pennylane/pull/9413)

* The `cond` PLxPR primitive no longer returns an `AbstractOperator` when the branch functions
  are gate-like operators.
  [(#9494)](https://github.com/PennyLaneAI/pennylane/pull/9494)

* The `allocate` PLxPR primitive now returns a list of `AbstractQubit` abstract values instead of a
  list of abstract integer values. This is to better define the set of operations allowed on
  allocated qubits.
  [(#9400)](https://github.com/PennyLaneAI/pennylane/pull/9400)
  [(#9541)](https://github.com/PennyLaneAI/pennylane/pull/9541)

<h3>Documentation 📝</h3>

* A rendering issue was fixed in the docstring for :class:`~.TrotterizedQfunc`.
  [(#9697)](https://github.com/PennyLaneAI/pennylane/pull/9697)

* Enabled documentation testing for the :mod:`pennylane.shadows` module by updating its executable examples and
  removing the module from the documentation-test skip list.
  [(#9566)](https://github.com/PennyLaneAI/pennylane/pull/9566)

* Fixed expected outputs in documentation of `"default.clifford"` device due to a new Stim version.
  [(#9533)](https://github.com/PennyLaneAI/pennylane/pull/9533)

* References to TensorFlow integration have been removed from the documentation following the end of maintenance support as of PennyLane v0.44.
  [(#9486)](https://github.com/PennyLaneAI/pennylane/pull/9486)

* Functions with ``singledispatch`` stop having its signature duplicated in the documentation for every registered dispatch function.
  [(#9502)](https://github.com/PennyLaneAI/pennylane/pull/9502)

* Added examples to the documentation for the :class:`~.CNOT`, :class:`~.Toffoli`, and :class:`~.CCZ` operators.
  [(#9555)](https://github.com/PennyLaneAI/pennylane/pull/9555)

* Clarified the documentation for the :class:`~.QNode` to apply to more than just variational circuits.
  [(#9599)](https://github.com/PennyLaneAI/pennylane/pull/9599)

* Added a warning to the :class:`~.DefaultGaussian` documentation noting that the device may not work as
  expected with recent versions of PennyLane.
  [(#9621)](https://github.com/PennyLaneAI/pennylane/pull/9621)

<h3>Bug fixes 🐛</h3>

* Lazily defers checking program capture mode when taking the adjoint and ctrl of a qfunc.
  [(#9626)](https://github.com/PennyLaneAI/pennylane/pull/9626)

* Fixed a bug where :func:`~.evolve` / :class:`~.ops.op_math.Evolution` silently produced
  incorrect results and wrong gradients under the default autograd/backprop differentiation path
  when the generator was a linear combination of overlapping Pauli words (e.g. a :class:`~.Sum`
  or :class:`~.Hamiltonian`).
  [(#9636)](https://github.com/PennyLaneAI/pennylane/pull/9636)

* Fixed a bug in `change_op_basis` where `TypeError` raised within the body of callable inputs were
  accidentally being masked by internal try/except logic.
  [(#9552)](https://github.com/PennyLaneAI/pennylane/pull/9552)

* Fixed a bug in unary iteration in `Select` where work wires were not restored correctly
  if the number of selected operators is notably smaller than the maximal capacity for the given
  number of control wires. This bug only surfaced for `partial=False`.
  [(#9461)](https://github.com/PennyLaneAI/pennylane/pull/9461)

* Fixed a bug where the construction of ``DecompositionGraph`` enters infinite recursion when a decomposition path
  exists from an operator to a controlled/adjoint version of itself.
  [(#9457)](https://github.com/PennyLaneAI/pennylane/pull/9457)

* Fixed a bug in `MPSPrep` where passing `work_wires` as a NumPy array or an integer caused initialization errors.
  [(#9448)](https://github.com/PennyLaneAI/pennylane/pull/9448)

* The `pl-device-test` no longer uses the deprecated syntax that sets the shots on the device.
  [(#9503)](https://github.com/PennyLaneAI/pennylane/pull/9503)

* Fixed a sign error in the abstract decomposition of :class:`~.BasisState` that produced an
  incorrect global phase (off by −1 per qubit). The decomposition used
  ``GlobalPhase(basis * π/2)`` instead of ``GlobalPhase(-basis * π/2)``, introduced in
  [(#9406)](https://github.com/PennyLaneAI/pennylane/pull/9406).
  [(#9492)](https://github.com/PennyLaneAI/pennylane/pull/9492)

* Fixed a bug where `qp.qnn.TorchLayer` produced incorrect output shape `(n_measurements, batch, 1)`
  instead of `(batch, n_measurements)` when the wrapped QNode returns multiple measurements as a tuple
  (e.g., `return qp.expval(qp.Z(0)), qp.expval(qp.Z(1))`) and receives batched inputs. This previously
  caused shape mismatch errors when feeding the output into downstream `torch.nn.Linear` layers.
  [(#9284)](https://github.com/PennyLaneAI/pennylane/pull/9284)

* Fixed a bug where :class:`~.BasisEmbedding` was not normalized to :class:`~.BasisState` in
  :func:`~.controlled_resource_rep`, causing mismatches in the decomposition resource graph.
  [(#9460)](https://github.com/PennyLaneAI/pennylane/pull/9460)

* Fixes a bug where two ``MeasurementProcess`` of taken of different mid-circuit measurement
  values sometimes incorrectly have the same hash.
  [(#9488)](https://github.com/PennyLaneAI/pennylane/pull/9488)

* Fixed a bug where :func:`~.two_qubit_decomposition` would raise a
  ``TracerArrayConversionError`` when decomposing a :class:`~.QubitUnitary`
  that requires 2 CNOTs under ``qjit``. The guard preventing the 2-CNOT
  decomposition path from being traced with abstract arrays only checked
  ``capture.enabled()``, missing the ``qjit`` context where
  ``compiler.active()`` is ``True``. Both ``two_qubit_decomposition`` and
  ``two_qubit_decomp_rule`` are fixed.
  [(#9520)](https://github.com/PennyLaneAI/pennylane/pull/9520)

* Fixed a bug in the :mod:`~.pennylane.qchem.vibrational` submodule to properly account for the number of modes.
  [(#9522)](https://github.com/PennyLaneAI/pennylane/pull/9522)

* Fixed a bug where :func:`~pennylane.draw` dropped the grouping brackets on measurements that
  span all device wires (such as :func:`~.state`, :func:`~.probs`, :func:`~.sample`, or
  :func:`~.counts` without an explicit ``wires`` argument). The brackets now render consistently
  with the multi-wire case, matching the existing behavior of :func:`~pennylane.draw_mpl`.
  [(#9532)](https://github.com/PennyLaneAI/pennylane/pull/9532)

* Fixed a bug where gate types are overwritten in ``qp.specs`` on the MLIR level.
  [(#9574)](https://github.com/PennyLaneAI/pennylane/pull/9574)

* ``qp.ctrl`` no longer produces ``Controlled(Allocate)`` when applied to quantum functions that
  contain dynamic wire allocation instructions.
  [(#9625)](https://github.com/PennyLaneAI/pennylane/pull/9625)

* Fixed a bug where resource decompositions and parameters were not properly resolved for nested
  symbolic operators.
  [(#9619)](https://github.com/PennyLaneAI/pennylane/pull/9619)

* Fixed bugs where string PyTree leaves were incorrectly
  restored as ``bytes``.
  [(#9687)](https://github.com/PennyLaneAI/pennylane/pull/9687)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Usman Ahmed,
Guillermo Alonso,
Abdullah Al Omar Galib,
Gabriel Bottrill,
Astral Cai,
Daniel Casota,
Miguel Cárdenas,
Yushao Chen,
Diksha Dhawan,
Marcus Edwards,
Korbinian Kottmann,
Christina Lee,
Anton Naim Ibrahim,
Mudit Pandey,
Andrija Paurevic,
Francesco Pernice Botta,
David D.W. Ren,
Jay Soni,
Paul Haochen Wang,
Dennis Wayo,
David Wierichs,
Jake Zaia,
Zinan Zhou.
