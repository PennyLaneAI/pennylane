:orphan:

# Release 0.32.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* `PauliWord` sparse matrices are much faster, which directly improves `PauliSentence`.
  [(#4272)](https://github.com/PennyLaneAI/pennylane/pull/4272)

* Enable linting of all tests in CI and the pre-commit hook.
  [(#4335)](https://github.com/PennyLaneAI/pennylane/pull/4335)

* Added a function `qml.math.fidelity_statevector` that computes the fidelity between two state vectors.
  [(#4322)](https://github.com/PennyLaneAI/pennylane/pull/4322)

* QNode transforms in `qml.qinfo` now support custom wire labels.
  [#4331](https://github.com/PennyLaneAI/pennylane/pull/4331)

* The default label for a `StatePrep` operator is now `|Ψ⟩`.
  [(#4340)](https://github.com/PennyLaneAI/pennylane/pull/4340)

<h3>Breaking changes 💔</h3>

* The `do_queue` keyword argument in `qml.operation.Operator` has been removed. Instead of
  setting `do_queue=False`, use the `qml.QueuingManager.stop_recording()` context.
  [(#4317)](https://github.com/PennyLaneAI/pennylane/pull/4317)

* The `grouping_type` and `grouping_method` keyword arguments are removed from `qchem.molecular_hamiltonian`.

* `zyz_decomposition` and `xyx_decomposition` are removed. Use `one_qubit_decomposition` instead.

* `LieAlgebraOptimizer` has been removed. Use `RiemannianGradientOptimizer` instead.

* `Operation.base_name` has been removed.

* `QuantumScript.name` has been removed.

* `qml.math.reduced_dm` has been removed. Use `qml.math.reduce_dm` or `qml.math.reduce_statevector` instead.

* The ``qml.specs`` dictionary longer supports direct key access to certain keys. Instead
  these quantities can be accessed as fields of the new ``Resources`` object saved under
  ``specs_dict["resources"]``:

  - ``num_operations`` is no longer supported, use ``specs_dict["resources"].num_gates``
  - ``num_used_wires`` is no longer supported, use ``specs_dict["resources"].num_wires``
  - ``gate_types`` is no longer supported, use ``specs_dict["resources"].gate_types``
  - ``gate_sizes`` is no longer supported, use ``specs_dict["resources"].gate_sizes``
  - ``depth`` is no longer supported, use ``specs_dict["resources"].depth``

* `qml.math.purity`, `qml.math.vn_entropy`, `qml.math.mutual_info`, `qml.math.fidelity`,
  `qml.math.relative_entropy`, and `qml.math.max_entropy` no longer support state vectors as
  input.
  [(#4322)](https://github.com/PennyLaneAI/pennylane/pull/4322)

<h3>Deprecations 👋</h3>

* The CV observables ``qml.X`` and ``qml.P`` have been deprecated. Use ``qml.QuadX`` 
  and ``qml.QuadP`` instead.
  [(#4330)](https://github.com/PennyLaneAI/pennylane/pull/4330)

* `qml.enable_return` and `qml.disable_return` are deprecated. Please avoid calling
  `disable_return`, as the old return system is deprecated along with these switch functions.
  [(#4316)](https://github.com/PennyLaneAI/pennylane/pull/4316)

* The `mode` keyword argument in `QNode` is deprecated, as it was only used in the
  old return system (which is also deprecated). Please use `grad_on_execution` instead.
  [(#4316)](https://github.com/PennyLaneAI/pennylane/pull/4316)

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* Raise a warning if control indicators are hidden when calling `qml.draw_mpl`
  [(#4295)](https://github.com/PennyLaneAI/pennylane/pull/4295)

* `qml.qinfo.purity` now produces correct results with custom wire labels.
  [#4331](https://github.com/PennyLaneAI/pennylane/pull/4331)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Edward Jiang,
Christina Lee,
Borja Requena,
Matthew Silverman
