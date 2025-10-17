:orphan:

# Release 0.44.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* `qml.grad` and `qml.jacobian` now lazily dispatch to catalyst and program
  capture, allowing for `qml.qjit(qml.grad(c))` and `qml.qjit(qml.jacobian(c))` to work.
  [(#8382)](https://github.com/PennyLaneAI/pennylane/pull/8382)

* Both the generic and transform-specific application behavior of a `qml.transforms.core.TransformDispatcher`
  can be overwritten with `TransformDispatcher.generic_register` and `my_transform.register`.
  [(#7797)](https://github.com/PennyLaneAI/pennylane/pull/7797)

<h3>Breaking changes 💔</h3>

* ``pennylane.devices.DefaultExecutionConfig`` has been removed. Instead, use
  ``qml.devices.ExecutionConfig()`` to create a default execution configuration.
  [(#8470)](https://github.com/PennyLaneAI/pennylane/pull/8470)

* Specifying the ``work_wire_type`` argument in ``qml.ctrl`` and other controlled operators as ``"clean"`` or 
  ``"dirty"`` is disallowed. Use ``"zeroed"`` to indicate that the work wires are initially in the :math:`|0\rangle`
  state, and ``"borrowed"`` to indicate that the work wires can be in any arbitrary state. In both cases, the
  work wires are assumed to be restored to their original state upon completing the decomposition.
  [(#8470)](https://github.com/PennyLaneAI/pennylane/pull/8470)

* `QuantumScript.shape` and `QuantumScript.numeric_type` are removed. The corresponding `MeasurementProcess`
  methods should be used instead.
  [(#8468)](https://github.com/PennyLaneAI/pennylane/pull/8468)

* `MeasurementProcess.expand` is removed. 
  `qml.tape.QuantumScript(mp.obs.diagonalizing_gates(), [type(mp)(eigvals=mp.obs.eigvals(), wires=mp.obs.wires)])`
  can be used instead.
  [(#8468)](https://github.com/PennyLaneAI/pennylane/pull/8468)

* The `qml.QNode.add_transform` method is removed.
  Instead, please use `QNode.transform_program.push_back(transform_container=transform_container)`.
  [(#8468)](https://github.com/PennyLaneAI/pennylane/pull/8468)

<h3>Deprecations 👋</h3>

<h3>Internal changes ⚙️</h3>

* The experimental xDSL implementation of `diagonalize_measurements` has been updated to fix a bug
  that included the wrong SSA value for final qubit insertion and deallocation at the end of the circuit. A clear error is not also raised when there are observables with overlapping wires.
  [(#8383)](https://github.com/PennyLaneAI/pennylane/pull/8383)

* The :class:`~pennylane.devices.LegacyDeviceFacade` is slightly refactored to implement `setup_execution_config` and `preprocess_transforms`
  separately as opposed to implementing a single `preprocess` method. Additionally, the `mid_circuit_measurements` transform has been removed
  from the preprocess transform program. Instead, the best mcm method is chosen in `setup_execution_config`.
  [(#8469)](https://github.com/PennyLaneAI/pennylane/pull/8469)
  [(#8486)](https://github.com/PennyLaneAI/pennylane/pull/8486)

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Astral Cai,
Lillian Frederiksen,
Christina Lee,
