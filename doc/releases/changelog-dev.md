:orphan:

# Release 0.42.0-dev (development release)

<h3>New features since last release</h3>

* The transform `convert_to_mbqc_gateset` is added to the `ftqc` module to convert arbitrary 
  circuits to a limited gate-set that can be translated to the MBQC formalism.
  [(7271)](https://github.com/PennyLaneAI/pennylane/pull/7271)

* The `RotXZX` operation is added to the `ftqc` module to support definition of a universal
  gate-set that can be translated to the MBQC formalism.
  [(7271)](https://github.com/PennyLaneAI/pennylane/pull/7271)

<h3>Improvements 🛠</h3>

* Alias for Identity (`I`) is now accessible from `qml.ops`.
  [(#7200)](https://github.com/PennyLaneAI/pennylane/pull/7200)

* `Sum`, `Prod`, `SProd`, and `LinearCombination` are now represented by a reference in the text
  drawer, with a longer label printed underneath.
  [(#7279)](https://github.com/PennyLaneAI/pennylane/pull/7279)

<h3>Breaking changes 💔</h3>

* The `KerasLayer` class in `qml.qnn.keras` has been removed because Keras 2 is no longer actively maintained.
  Please consider using a different machine learning framework, like :doc:`PyTorch <demos/tutorial_qnn_module_torch>` or :doc:`JAX <demos/tutorial_How_to_optimize_QML_model_using_JAX_and_Optax>`.
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

* The :func:`qml.QNode.get_gradient_fn` method is now deprecated. Instead, use :func:`~.workflow.get_best_diff_method` to obtain the differentiation method.
  [(#7323)](https://github.com/PennyLaneAI/pennylane/pull/7323)

<h3>Internal changes ⚙️</h3>

* Test suites in `tests/transforms/test_defer_measurement.py` use analytic mocker devices to test numeric results.
  [(#7329)](https://github.com/PennyLaneAI/pennylane/pull/7329)

* Introduce module dependency management using `tach`.
  [(#7185)](https://github.com/PennyLaneAI/pennylane/pull/7185)

* Add new `pennylane.exceptions` module for custom errors and warnings.
  [(#7205)](https://github.com/PennyLaneAI/pennylane/pull/7205)

* Clean up `__init__.py` files in `math`, `ops`, `qaoa`, `tape` and `templates` to be explicit in what they import. 
  [(#7200)](https://github.com/PennyLaneAI/pennylane/pull/7200)
  
* The `Tracker` class has been moved into the `devices` module.
  [(#7281)](https://github.com/PennyLaneAI/pennylane/pull/7281)

<h3>Documentation 📝</h3>

* The entry in the :doc:`/news/program_capture_sharp_bits` page for using program capture with Catalyst
  has been updated. Instead of using ``qjit(experimental_capture=True)``, Catalyst is now compatible 
  with the global toggles ``qml.capture.enable()`` and ``qml.capture.disable()`` for enabling and 
  disabling program capture.
  [(#7298)](https://github.com/PennyLaneAI/pennylane/pull/7298)

<h3>Bug fixes 🐛</h3>

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

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso-Linaje,
Yushao Chen,
Lillian Frederiksen,
Pietropaolo Frisoni,
Korbinian Kottmann,
Christina Lee,
Andrija Paurevic
