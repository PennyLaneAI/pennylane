:orphan:

# Release 0.41.0-dev (development release)

<h3>New features since last release</h3>

* ``qml.lie_closure`` now accepts and outputs dense inputs using the ``dense`` keyword.
  Also added ``qml.pauli.trace_inner_product`` that can handle batches of dense matrices.
  [(#6811)](https://github.com/PennyLaneAI/pennylane/pull/6811)

<h3>Improvements 🛠</h3>

* `QNode` objects now have an `update` method that allows for re-configuring settings like `diff_method`, `mcm_method`, and more. This allows for easier on-the-fly adjustments to workflows. Any arguments not specified will retain their original value.
  [(#6803)](https://github.com/PennyLaneAI/pennylane/pull/6803)

  After constructing a `QNode`,
  ```python
  import pennylane as qml

  @qml.qnode(device=qml.device("default.qubit"))
  def circuit():
    qml.H(0)
    qml.CNOT([0,1])
    return qml.probs()
  ```
  its settings can be modified with `update`, which returns a new `QNode` object. Here is an example
  of updating a QNode's `diff_method`:
  ```pycon
  >>> print(circuit.diff_method)
  best
  >>> new_circuit = circuit.update(diff_method="parameter-shift")
  >>> print(new_circuit.diff_method)
  'parameter-shift'
  ```
  
* Finite shot and parameter-shift executions on `default.qubit` can now
  be natively jitted end-to-end, leading to performance improvements.
  Devices can now configure whether or not ML framework data is sent to them
  via an `ExecutionConfig.convert_to_numpy` parameter.
  [(#6788)](https://github.com/PennyLaneAI/pennylane/pull/6788)

* The coefficients of observables now have improved differentiability.
  [(#6598)](https://github.com/PennyLaneAI/pennylane/pull/6598)

* An empty basis set in `qml.compile` is now recognized as valid, resulting in decomposition of all operators that can be decomposed. 
   [(#6821)](https://github.com/PennyLaneAI/pennylane/pull/6821)

* An informative error is raised when a `QNode` with `diff_method=None` is differentiated.
  [(#6770)](https://github.com/PennyLaneAI/pennylane/pull/6770)

<h3>Labs: a place for unified and rapid prototyping of research software 🧪</h3>

* ``qml.labs.dla.lie_closure_dense`` is removed and integrated into ``qml.lie_closure`` using the new ``dense`` keyword.
  [(#6811)](https://github.com/PennyLaneAI/pennylane/pull/6811)

* ``pennylane.labs.dla.structure_constants`` is removed and integrated into ``qml.structure_constants`` using the new ``dense`` keyword.
  [()]()

<h3>Breaking changes 💔</h3>

* `MultiControlledX` no longer accepts strings as control values.
  [(#6835)](https://github.com/PennyLaneAI/pennylane/pull/6835)

* The input argument `control_wires` of `MultiControlledX` has been removed.
  [(#6832)](https://github.com/PennyLaneAI/pennylane/pull/6832)

* `qml.execute` now has a collection of keyword-only arguments.
  [(#6598)](https://github.com/PennyLaneAI/pennylane/pull/6598)

* The ``decomp_depth`` argument in :func:`~pennylane.transforms.set_decomposition` has been removed. 
  [(#6824)](https://github.com/PennyLaneAI/pennylane/pull/6824)

* The ``max_expansion`` argument in :func:`~pennylane.devices.preprocess.decompose` has been removed. 
  [(#6824)](https://github.com/PennyLaneAI/pennylane/pull/6824)

* The ``tape`` and ``qtape`` properties of ``QNode`` have been removed. 
  Instead, use the ``qml.workflow.construct_tape`` function.
  [(#6825)](https://github.com/PennyLaneAI/pennylane/pull/6825)

* The ``gradient_fn`` keyword argument to ``qml.execute`` has been removed. Instead, it has been replaced with ``diff_method``.
  [(#6830)](https://github.com/PennyLaneAI/pennylane/pull/6830)
  
* The ``QNode.get_best_method`` and ``QNode.best_method_str`` methods have been removed. 
  Instead, use the ``qml.workflow.get_best_diff_method`` function. 
  [(#6823)](https://github.com/PennyLaneAI/pennylane/pull/6823)

* The `output_dim` property of `qml.tape.QuantumScript` has been removed. Instead, use method `shape` of `QuantumScript` or `MeasurementProcess` to get the same information.
  [(#6829)](https://github.com/PennyLaneAI/pennylane/pull/6829)

* Removed method `qsvt_legacy` along with its private helper `_qsp_to_qsvt`
  [(#6827)](https://github.com/PennyLaneAI/pennylane/pull/6827)

<h3>Deprecations 👋</h3>

* The ``inner_transform_program`` and ``config`` keyword arguments in ``qml.execute`` have been deprecated.
  If more detailed control over the execution is required, use ``qml.workflow.run`` with these arguments instead.
  [(#6822)](https://github.com/PennyLaneAI/pennylane/pull/6822)

<h3>Internal changes ⚙️</h3>

<h3>Documentation 📝</h3>

* Updated documentation for vibrational Hamiltonians
  [(#6717)](https://github.com/PennyLaneAI/pennylane/pull/6717)

<h3>Bug fixes 🐛</h3>

* `BasisState` now casts its input to integers.
  [(#6844)](https://github.com/PennyLaneAI/pennylane/pull/6844)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):
<<<<<<< HEAD
Diksha Dhawan,
Korbinian Kottmann
=======

Yushao Chen,
Diksha Dhawan,
Pietropaolo Frisoni,
Marcus Gisslén,
Christina Lee,
Andrija Paurevic
>>>>>>> 397273bbfff54621cc52c6c3376b30ae46b9468b
