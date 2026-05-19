# Release 0.46.0 (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* Added three decompositions of :class:`~.OutMultiplier` that use significantly fewer costly gates
  than the existing QFT-based decomposition, at the cost of more auxiliary wires.
  In addition added a new argument ``output_wires_zeroed`` to ``OutMultiplier`` that can be
  used to indicate ``output_wires`` to be in the :math:`|0\rangle` state, leading to cheaper
  decompositions.
  [(#8900)](https://github.com/PennyLaneAI/pennylane/pull/8900)

<h3>Labs: a place for unified and rapid prototyping of research software 🧪</h3>

* Created a new ``~.labs.templates.LeftQuantumComparator`` template for performing inequality test of two quantum registers.
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
  
<h3>Breaking changes 💔</h3>

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

* Using :func:`qp.templates.layer <.templates.layer>` is deprecated and will be removed in v0.47. Instead, please apply
  your unitary in a for loop.
  [(#9484)](https://github.com/PennyLaneAI/pennylane/pull/9484)

* The ``QuantumScript.adjoint`` (and ``QuantumTape.adjoint``) methods have been deprecated in v0.46. Instead, please use
  ``QuantumScript([adjoint(op) for op in reversed(tape.operations)])``.
  [(#9483)](https://github.com/PennyLaneAI/pennylane/pull/9483)

<h3>Internal changes ⚙️</h3>

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
  
<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* Fixed a sign error in the abstract decomposition of :class:`~.BasisState` that produced an
  incorrect global phase (off by −1 per qubit). The decomposition used
  ``GlobalPhase(basis * π/2)`` instead of ``GlobalPhase(-basis * π/2)``, introduced in
  [#9406](https://github.com/PennyLaneAI/pennylane/pull/9406).
  [(#9492)](https://github.com/PennyLaneAI/pennylane/pull/9492)

* Fixed a bug where `qp.qnn.TorchLayer` produced incorrect output shape `(n_measurements, batch, 1)`
  instead of `(batch, n_measurements)` when the wrapped QNode returns multiple measurements as a tuple
  (e.g., `return qp.expval(qp.Z(0)), qp.expval(qp.Z(1))`) and receives batched inputs. This previously
  caused shape mismatch errors when feeding the output into downstream `torch.nn.Linear` layers.
  [(#9284)](https://github.com/PennyLaneAI/pennylane/pull/9284)

* Fixed a bug where :class:`~.BasisEmbedding` was not normalized to :class:`~.BasisState` in
  :func:`~.controlled_resource_rep`, causing mismatches in the decomposition resource graph.
  [(#9460)](https://github.com/PennyLaneAI/pennylane/pull/9460)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Usman Ahmed,
Guillermo Alonso,
Daniel Casota,
Yushao Chen,
Marcus Edwards,
Andrija Paurevic,
David Wierichs
