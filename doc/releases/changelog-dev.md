:orphan:

# Release 0.41.0-dev (development release)

<h3>New features since last release</h3>

* `qml.defer_measurements` can now be used with program capture enabled.
  [(#6838)](https://github.com/PennyLaneAI/pennylane/pull/6838)

  Using `qml.defer_measurements` with program capture enables many new features, including:
  * Significantly richer variety of classical processing on mid-circuit measurement values.
  * Using mid-circuit measurement values as gate parameters.

  Functions such as the following can now be captured:

  ```python
  import jax.numpy as jnp

  qml.capture.enable()

  def processing_fn(m1, m2):
      a = jnp.sin(0.5 * jnp.pi * m1)
      b = a - (m2 + 1) ** 4
      return b

  def f(x):
      m0 = qml.measure(0)
      m1 = qml.measure(0)
      inval = processing_fn(m0, m1)

      qml.s_prod(x, qml.RZ(inval, 0))

      return qml.expval(qml.Z(0))
  ```

<h3>Improvements 🛠</h3>

* The higher order primitives in program capture can now accept inputs with abstract shapes.
  [(#6786)](https://github.com/PennyLaneAI/pennylane/pull/6786)

* The `PlxprInterpreter` classes can now handle creating dynamic arrays via `jnp.ones`, `jnp.zeros`,
  `jnp.arange`, and `jnp.full`.
  [#6865)](https://github.com/PennyLaneAI/pennylane/pull/6865)

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
  
* Devices can now configure whether or not ML framework data is sent to them
  via an `ExecutionConfig.convert_to_numpy` parameter. This is not used on 
  `default.qubit` due to compilation overheads when jitting.
  [(#6788)](https://github.com/PennyLaneAI/pennylane/pull/6788)
  [(#6869)](https://github.com/PennyLaneAI/pennylane/pull/6869)

* The coefficients of observables now have improved differentiability.
  [(#6598)](https://github.com/PennyLaneAI/pennylane/pull/6598)

* An empty basis set in `qml.compile` is now recognized as valid, resulting in decomposition of all operators that can be decomposed. 
   [(#6821)](https://github.com/PennyLaneAI/pennylane/pull/6821)

* An informative error is raised when a `QNode` with `diff_method=None` is differentiated.
  [(#6770)](https://github.com/PennyLaneAI/pennylane/pull/6770)

* `qml.gradients.finite_diff_jvp` has been added to compute the jvp of an arbitrary numeric
  function.
  [(#6853)](https://github.com/PennyLaneAI/pennylane/pull/6853)

* With program capture enabled, `QNode`'s can now be differentiated with `diff_method="finite-diff"`.
  [(#6853)](https://github.com/PennyLaneAI/pennylane/pull/6853)

* The requested `diff_method` is now validated when program capture is enabled.
  [(#6852)](https://github.com/PennyLaneAI/pennylane/pull/6852)

<h3>Breaking changes 💔</h3>

* `MultiControlledX` no longer accepts strings as control values.
  [(#6835)](https://github.com/PennyLaneAI/pennylane/pull/6835)

* The input argument `control_wires` of `MultiControlledX` has been removed.
  [(#6832)](https://github.com/PennyLaneAI/pennylane/pull/6832)
  [(#6862)](https://github.com/PennyLaneAI/pennylane/pull/6862)

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

* The `mcm_method` keyword in `qml.execute` has been deprecated. 
  Instead, use the ``mcm_method`` and ``postselect_mode`` arguments.
  [(#6807)](https://github.com/PennyLaneAI/pennylane/pull/6807)

* Specifying gradient keyword arguments as any additional keyword argument to the qnode is deprecated
  and will be removed in v0.42.  The gradient keyword arguments should be passed to the new
  keyword argument `gradient_kwargs` via an explicit dictionary. This change will improve qnode argument
  validation.
  [(#6828)](https://github.com/PennyLaneAI/pennylane/pull/6828)

* The `qml.gradients.hamiltonian_grad` function has been deprecated.
  This gradient recipe is not required with the new operator arithmetic system.
  [(#6849)](https://github.com/PennyLaneAI/pennylane/pull/6849)

* The ``inner_transform_program`` and ``config`` keyword arguments in ``qml.execute`` have been deprecated.
  If more detailed control over the execution is required, use ``qml.workflow.run`` with these arguments instead.
  [(#6822)](https://github.com/PennyLaneAI/pennylane/pull/6822)
  [(#6879)](https://github.com/PennyLaneAI/pennylane/pull/6879)

<h3>Internal changes ⚙️</h3>

* Improved the `InterfaceEnum` object to prevent direct comparisons to `str` objects.
  [(#6877)](https://github.com/PennyLaneAI/pennylane/pull/6877)

* Added a `QmlPrimitive` class that inherits `jax.core.Primitive` to a new `qml.capture.custom_primitives` module.
  This class contains a `prim_type` property so that we can differentiate between different sets of PennyLane primitives.
  Consequently, `QmlPrimitive` is now used to define all PennyLane primitives.
  [(#6847)](https://github.com/PennyLaneAI/pennylane/pull/6847)

* The `RiemannianGradientOptimizer` has been updated to take advantage of newer features.
  [(#6882)](https://github.com/PennyLaneAI/pennylane/pull/6882)

<h3>Documentation 📝</h3>

* The docstrings for `qml.unary_mapping`, `qml.binary_mapping`, `qml.christiansen_mapping`, 
  `qml.qchem.localize_normal_modes`, and `qml.qchem.VibrationalPES` have been updated to include better 
  code examples.
  [(#6717)](https://github.com/PennyLaneAI/pennylane/pull/6717)

* The docstrings for `qml.qchem.localize_normal_modes` and `qml.qchem.VibrationalPES` have been updated to include
  examples that can be copied.
  [(#6834)](https://github.com/PennyLaneAI/pennylane/pull/6834)

* Fixed a typo in the code example for `qml.labs.dla.lie_closure_dense`.
  [(#6858)](https://github.com/PennyLaneAI/pennylane/pull/6858)

<h3>Bug fixes 🐛</h3>

* `BasisState` now casts its input to integers.
  [(#6844)](https://github.com/PennyLaneAI/pennylane/pull/6844)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Yushao Chen,
Isaac De Vlugt,
Diksha Dhawan,
Pietropaolo Frisoni,
Marcus Gisslén,
Christina Lee,
Mudit Pandey,
Andrija Paurevic
