# Release 0.46.0 (development release)

<h3>New features since last release</h3>

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
  0: ─╭TwoWireFFT────────────────────╭TwoWireFFT──────────────┤  State
  1: ─╰TwoWireFFT───────╭FSWAP(3.14)─╰TwoWireFFT─╭FSWAP(3.14)─┤  State
  2: ─╭TwoWireFFT──Z⁰⋅⁰─╰FSWAP(3.14)─╭TwoWireFFT─╰FSWAP(3.14)─┤  State
  3: ─╰TwoWireFFT──Z⁰⋅⁵──────────────╰TwoWireFFT──────────────┤  State

  ```

  Alongside the addition of :class:`~.FFFT`, a new operation called :class:`~.TwoWireFFT`
  has been added to enable its implementation: the :class:`~.FFFT` operation is
  decomposed recursively into :class:`~.FermionicSWAP` and :class:`~.TwoWireFFT` operations 
  (two-site Fermionic Fourier transforms).

<h3>Improvements 🛠</h3>

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

<h3>Labs: a place for unified and rapid prototyping of research software 🧪</h3>

* Created a new ``labs.templates.LeftQuantumComparator`` template for performing inequality test of two quantum registers.
  [(#9277)](https://github.com/PennyLaneAI/pennylane/pull/9277)

  ```python
  import pennylane as qp
  from pennylane.labs.templates import LeftQuantumComparator

  dev = qp.device("lightning.qubit")

  @qp.qnode(dev, shots=1)
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

<h3>Breaking changes 💔</h3>

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

<h3>Internal changes ⚙️</h3>

* Adds a new `pennylane/core` module.
  Moves the abstractions from `pennylane/operation` into `pennylane/core/operator`.

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

* Fixed expected outputs in documentation of `"default.clifford"` device due to a new Stim version.
  [(#9533)](https://github.com/PennyLaneAI/pennylane/pull/9533)

* References to TensorFlow integration have been removed from the documentation following the end of maintenance support as of PennyLane v0.44.
  [(#9486)](https://github.com/PennyLaneAI/pennylane/pull/9486)

<h3>Bug fixes 🐛</h3>

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

* Fixed a bug in the :mod:`~.pennylane.qchem.vibrational` submodule to properly account for the number of modes.
  [(#9522)](https://github.com/PennyLaneAI/pennylane/pull/9522)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Usman Ahmed,
Guillermo Alonso,
Astral Cai,
Daniel Casota,
Yushao Chen,
Marcus Edwards,
Korbinian Kottmann,
Christina Lee,
Anton Naim Ibrahim,
Andrija Paurevic,
Jay Soni,
Paul Haochen Wang,
David Wierichs.
