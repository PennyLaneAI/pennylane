# Release 0.46.0 (development release)

<h3>New features since last release</h3>

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

* A new template for probabilistic state preparation based on the flip-flop QRAM construction is now available, named :class:`~.FFQRAM`. Given real amplitudes and a list of bitstring addresses, the template embeds the corresponding sparse computational-basis state, with the desired state obtained by post-selecting on the register qubit. [(#9498)](https://github.com/PennyLaneAI/pennylane/pull/9498)

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

<h3>Improvements 🛠</h3>

* `Tracker` now has a readable `__repr__` that displays all relevant internals
  (`active`, `totals`, `history`, `latest`, `persistent`, `callback`).
  [(#9575)](https://github.com/PennyLaneAI/pennylane/pull/9575)

  ```pycon
  >>> tracker = qp.Tracker()
  >>> tracker.update(a=2, b="b2", c=1)
  >>> print(tracker)
  Tracker(active=False, totals={'a': 2, 'c': 1}, history={'a': [2], 'b': ['b2'], 'c': [1]}, latest={'a': 2, 'b': 'b2', 'c': 1}, persistent=False, callback=None)
  
  ```

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

<h3>Labs: a place for unified and rapid prototyping of research software 🧪</h3>

* Added a variant of `SumOfSlatersPrep` to labs, accessible as `labs.templates.SumOfSlatersPrep2`.
  This variant handles work wires explicitly instead of allocating them dynamically in the
  decomposition. This enables usage of `SumOfSlatersPrep2` with `qp.qjit` with 
  capture _disabled_ (`qp.capture.disable()`).
  [(#9539)](https://github.com/PennyLaneAI/pennylane/pull/9539)

* Updated the `make_selectpaulirot_to_phase_gradient_decomp` and `make_rz_to_phase_gradient_decomp` decomposition rule factories to be compatible with program capture.
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
  [(#9502)](https://github.com/PennyLaneAI/pennylane/pull/9502)

<h3>Internal changes ⚙️</h3>

* Bump `codecov-action` to `v7`.
  [(#9615)](https://github.com/PennyLaneAI/pennylane/pull/9615)

* :func:`~.parameter_frequencies` added that allows a user to retrieve `parameter_frequencies` from an
  :class:`~.Operation` or calculate them for an :class:`~.Operator2`.
  [(#9569)](https://github.com/PennyLaneAI/pennylane/pull/9569)

* Adds a new test fixture `preserve_jax_x64` to help automatically restore the `jax.config.jax_enable_x64` to prevent
  accidental context contamination.
  [(#9590)](https://github.com/PennyLaneAI/pennylane/pull/9590)

* New, experimental abstractions for creating PennyLane operators have been added, built around a new
  base class, `Operator2`. This is an internal, work-in-progress effort that is being incrementally
  integrated into the PennyLane ecosystem. Supported functionality so far:
  - :func:`qp.equal` can check equality between two `Operator2` instances.
  [(#9525)](https://github.com/PennyLaneAI/pennylane/pull/9525)
  [(#9529)](https://github.com/PennyLaneAI/pennylane/pull/9529)
  [(#9526)](https://github.com/PennyLaneAI/pennylane/pull/9526)
  [(#9527)](https://github.com/PennyLaneAI/pennylane/pull/9527)

* Adds a new `pennylane/core` module.
  Moves the abstractions from `pennylane/operation` into `pennylane/core/operator`.
  [(#9508)](https://github.com/PennyLaneAI/pennylane/pull/9508)
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

<h3>Bug fixes 🐛</h3>

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

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Usman Ahmed,
Guillermo Alonso,
Abdullah Al Omar Galib,
Astral Cai,
Daniel Casota,
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
