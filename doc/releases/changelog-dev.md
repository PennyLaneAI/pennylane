:orphan:

# Release 0.41.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* Improved decompositions done by `qml.clifford_t_decomposition` for `qml.PhaseShift` gates.
  [(#6842)](https://github.com/PennyLaneAI/pennylane/pull/6842)

* Finite shot and parameter-shift executions on `default.qubit` can now
  be natively jitted end-to-end, leading to performance improvements.
  Devices can now configure whether or not ML framework data is sent to them
  via an `ExecutionConfig.convert_to_numpy` parameter.
  [(#6788)](https://github.com/PennyLaneAI/pennylane/pull/6788)

* The coefficients of observables now have improved differentiability.
  [(#6598)](https://github.com/PennyLaneAI/pennylane/pull/6598)

* An informative error is raised when a `QNode` with `diff_method=None` is differentiated.
  [(#6770)](https://github.com/PennyLaneAI/pennylane/pull/6770)

<h3>Breaking changes 💔</h3>

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

<h3>Documentation 📝</h3>

* Updated documentation for vibrational Hamiltonians
  [(#6717)](https://github.com/PennyLaneAI/pennylane/pull/6717)

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Utkarsh Azad,
Yushao Chen,
Diksha Dhawan,
Christina Lee,
Andrija Paurevic
