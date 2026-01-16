# Release 0.45.0 (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* :func:`~.matrix` can now also be applied to a sequence of operators.
  [(#8861)](https://github.com/PennyLaneAI/pennylane/pull/8861)

* The ``qml.estimator.Resources`` class now has a nice string representation in Jupyter Notebooks.
  [(#8880)](https://github.com/PennyLaneAI/pennylane/pull/8880)

<h3>Labs: a place for unified and rapid prototyping of research software 🧪</h3>

* Removed all of the resource estimation functionality from the ``/labs/resource_estimation``
  module. Users can now directly access a more stable version of this functionality using the 
  :mod:`estimator <pennylane.estimator>` module. All experimental development of resource estimation
  will be added to ``/labs/estimator_beta``
  [(#8868)](https://github.com/PennyLaneAI/pennylane/pull/8868)

* The integration test for computing perturbation error of a compressed double-factorized (CDF)
  Hamiltonian in ``labs.trotter_error`` is upgraded to use a more realistic molecular geometry and
  a more reliable reference error.
  [(#8790)](https://github.com/PennyLaneAI/pennylane/pull/8790)

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Internal changes ⚙️</h3>

* Updated internal dependencies `autoray` (to 0.8.4), `tach` (to 0.33).
  [(#8911)](https://github.com/PennyLaneAI/pennylane/pull/8911)

* Relaxed the `torch` dependency from `==2.9.0` to `~=2.9.0` to allow for compatible patch updates.
  [(#8911)](https://github.com/PennyLaneAI/pennylane/pull/8911)

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* Fixes a bug that `qml.QubitDensityMatrix` was applied in `default.mixed` device using `qml.math.partial_trace` incorrectly.
  This would cause wrong results as described in [this issue](https://github.com/PennyLaneAI/pennylane/pull/8932).
  [(#8933)](https://github.com/PennyLaneAI/pennylane/pull/8933)

* Fixes an issue when binding a transform when the first positional arg
  is a `Sequence`, but not a `Sequence` of tapes.
  [(#8920)](https://github.com/PennyLaneAI/pennylane/pull/8920)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):
Omkar Sarkar,
Jay Soni,
David Wierichs,
