:orphan:

# Release 0.38.0-dev (development release)

<h3>New features since last release</h3>

* Resolved the bug in `qml.ThermalRelaxationError` where there was a typo from `tq` to `tg`.
  [(#5988)](https://github.com/PennyLaneAI/pennylane/issues/5988)

* A new method `process_density_matrix` has been added to the `ProbabilityMP` and `DensityMatrixMP` classes, allowing for more efficient handling of quantum density matrices, particularly with batch processing support. This method simplifies the calculation of probabilities from quantum states represented as density matrices.
  [(#5830)](https://github.com/PennyLaneAI/pennylane/pull/5830)

* The `qml.PrepSelPrep` template is added. The template implements a block-encoding of a linear
  combination of unitaries.
  [(#5756)](https://github.com/PennyLaneAI/pennylane/pull/5756)

* The `split_to_single_terms` transform is added. This transform splits expectation values of sums
  into multiple single-term measurements on a single tape, providing better support for simulators
  that can handle non-commuting observables but don't natively support multi-term observables.
  [(#5884)](https://github.com/PennyLaneAI/pennylane/pull/5884)

* `SProd.terms` now flattens out the terms if the base is a multi-term observable.
  [(#5885)](https://github.com/PennyLaneAI/pennylane/pull/5885)

<h3>Improvements 🛠</h3>

* `StateMP.process_state` defines rules in `cast_to_complex` for complex casting, avoiding a superfluous state vector copy in Lightning simulations
  [(#5995)](https://github.com/PennyLaneAI/pennylane/pull/5995)

* Port the fast `apply_operation` implementation of `PauliZ` to `PhaseShift`, `S` and `T`.
  [(#5876)](https://github.com/PennyLaneAI/pennylane/pull/5876)

* `qml.UCCSD` now accepts an additional optional argument, `n_repeats`, which defines the number of
  times the UCCSD template is repeated. This can improve the accuracy of the template by reducing
  the Trotter error but would result in deeper circuits.
  [(#5801)](https://github.com/PennyLaneAI/pennylane/pull/5801)

* `QuantumScript.hash` is now cached, leading to performance improvements.
  [(#5919)](https://github.com/PennyLaneAI/pennylane/pull/5919)

* The representation for `Wires` has now changed to be more copy-paste friendly.
  [(#5958)](https://github.com/PennyLaneAI/pennylane/pull/5958)

* Observable validation for `default.qubit` is now based on execution mode (analytic vs. finite shots) and measurement type (sample measurement vs. state measurement).
  [(#5890)](https://github.com/PennyLaneAI/pennylane/pull/5890)

<h3>Breaking changes 💔</h3>

* The `CircuitGraph.graph` rustworkx graph now stores indices into the circuit as the node labels,
  instead of the operator/ measurement itself.  This allows the same operator to occur multiple times in
  the circuit.
  [(#5907)](https://github.com/PennyLaneAI/pennylane/pull/5907)

* `queue_idx` attribute has been removed from the `Operator`, `CompositeOp`, and `SymboliOp` classes.


* ``qml.transforms.map_batch_transform`` has been removed, since transforms can be applied directly to a batch of tapes.
  See :func:`~.pennylane.transform` for more information.
  [(#5981)](https://github.com/PennyLaneAI/pennylane/pull/5981)

* `QuantumScript.interface` has been removed.
  [(#5980)](https://github.com/PennyLaneAI/pennylane/pull/5980)

<h3>Deprecations 👋</h3>

* `pennylane.qinfo.classical_fisher` and `pennylane.qinfo.quantum_fisher` have been deprecated.
  Instead, use `pennylane.gradients.classical_fisher` and `pennylane.gradients.quantum_fisher`.
  [(#5985)](https://github.com/PennyLaneAI/pennylane/pull/5985)

<h3>Documentation 📝</h3>

* Improves the docstring for `QuantumScript.expand` and `qml.tape.tape.expand_tape`.
  [(#5974)](https://github.com/PennyLaneAI/pennylane/pull/5974)

<h3>Bug fixes 🐛</h3>

* `CircuitGraph` can now handle circuits with the same operation instance occuring multiple times.
  [(#5907)](https://github.com/PennyLaneAI/pennylane/pull/5907)

* `qml.QSVT` is updated to store wire order correctly.
  [(#5959)](https://github.com/PennyLaneAI/pennylane/pull/5959)

* `qml.devices.qubit.measure_with_samples` now returns the correct result if the provided measurements
  contain sum of operators acting on the same wire.
  [(#5978)](https://github.com/PennyLaneAI/pennylane/pull/5978)

* `qml.AmplitudeEmbedding` has better support for features using low precision integer data types.
[(#5969)](https://github.com/PennyLaneAI/pennylane/pull/5969)


<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):
Guillermo Alonso,
Ahmed Darwish,
Astral Cai,
Yushao Chen,
Lillian M. A. Frederiksen,
Pietropaolo Frisoni,
Emiliano Godinez,
Christina Lee,
Austin Huang,
William Maxwell,
Vincent Michaud-Rioux,
Mudit Pandey,
Erik Schultheis.
