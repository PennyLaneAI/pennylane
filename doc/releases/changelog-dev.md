:orphan:

# Release 0.44.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* Both the generic and transform-specific application behavior of a `qml.transforms.core.TransformDispatcher`
  can be overwritten with `TransformDispatcher.generic_register` and `my_transform.register`.
  [(#7797)](https://github.com/PennyLaneAI/pennylane/pull/7797)

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

* `qml.measure`, `qml.measurements.MidMeasureMP`, `qml.measurements.MeasurementValue`,
  and `qml.measurements.get_mcm_predicates` are now located in `qml.ops.mid_measure`.
  `MidMeasureMP` is now renamed to `MidMeasure`.
  `qml.measurements.find_post_processed_mcms` is now `qml.devices.qubit.simulate._find_post_processed_mcms`,
  and is being made private, as it is an utility for tree-traversal.
  [(#8466)](https://github.com/PennyLaneAI/pennylane/pull/8466)

<h3>Internal changes ⚙️</h3>

* The experimental xDSL implementation of `diagonalize_measurements` has been updated to fix a bug
  that included the wrong SSA value for final qubit insertion and deallocation at the end of the circuit. A clear error is not also raised when there are observables with overlapping wires.
  [(#8383)](https://github.com/PennyLaneAI/pennylane/pull/8383)

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Lillian Frederiksen