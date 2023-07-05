:orphan:

# Release 0.32.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

<h3>Breaking changes 💔</h3>

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

<h3>Deprecations 👋</h3>

* `qml.enable_return` and `qml.disable_return` are deprecated. Please ensure that you are using
  the new return system, as the old return system is deprecated along with these switch functions.
  [(#4316)](https://github.com/PennyLaneAI/pennylane/pull/4316)

* The `mode` keyword argument in `QNode` is deprecated, as it was only used in the
  old return system (which is also deprecated). Please use `grad_on_execution` instead.
  [(#4316)](https://github.com/PennyLaneAI/pennylane/pull/4316)

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* Raise a warning if control indicators are hidden when calling `qml.draw_mpl`
  [(#4295)](https://github.com/PennyLaneAI/pennylane/pull/4295)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Christina Lee,
Matthew Silverman