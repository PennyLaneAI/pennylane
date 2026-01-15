# Release 0.45.0 (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* :func:`~.matrix` can now also be applied to a sequence of operators.
  [(#8861)](https://github.com/PennyLaneAI/pennylane/pull/8861)

* The ``qml.estimator.Resources`` class now has a nice string representation in Jupyter Notebooks.
  [(#8880)](https://github.com/PennyLaneAI/pennylane/pull/8880)

<h3>Labs: a place for unified and rapid prototyping of research software 🧪</h3>

* Removed all of the resource estimation functionality from the `labs.resource_estimation`
  module. Users can now directly access a more stable version of this functionality using the 
  `estimator` module. All experimental development of resource estimation
  will be added to `.labs.estimator_beta`
  [(#8868)](https://github.com/PennyLaneAI/pennylane/pull/8868)

* The integration test for computing perturbation error of a compressed double-factorized (CDF)
  Hamiltonian in `labs.trotter_error` is upgraded to use a more realistic molecular geometry and
  a more reliable reference error.
  [(#8790)](https://github.com/PennyLaneAI/pennylane/pull/8790)

<h3>Breaking changes 💔</h3>

* The `pennylane.operation.Operator.is_hermitian` property has been removed and replaced 
  with `pennylane.operation.Operator.is_verified_hermitian` as it better reflects the functionality of this property.
  Alternatively, consider using the `pennylane.is_hermitian` function instead as it provides a more reliable check for hermiticity.
  Please be aware that it comes with a higher computational cost.
  [(#8919)](https://github.com/PennyLaneAI/pennylane/pull/8919)

* Passing a function to the `gate_set` argument in the `pennylane.transforms.decompose` transform
  is removed. The `gate_set` argument expects a static iterable of operator type and/or operator names,
  and the function should be passed to the `stopping_condition` argument instead.
  [(#8919)](https://github.com/PennyLaneAI/pennylane/pull/8919)

* `argnum` has been renamed `argnums` in `qml.grad`, `qml.jacobian`, `qml.jvp`, and `qml.vjp`
  to better match Catalyst and JAX.
  [(#8919)](https://github.com/PennyLaneAI/pennylane/pull/8919)

* Access to the following functions and classes from the `~pennylane.resources` module has 
  been removed. Instead, these functions must be imported from the `~pennylane.estimator` module.
  [(#8919)](https://github.com/PennyLaneAI/pennylane/pull/8919)

    - `qml.estimator.estimate_shots` in favor of `qml.resources.estimate_shots`
    - `qml.estimator.estimate_error` in favor of `qml.resources.estimate_error`
    - `qml.estimator.FirstQuantization` in favor of `qml.resources.FirstQuantization`
    - `qml.estimator.DoubleFactorization` in favor of `qml.resources.DoubleFactorization`


<h3>Deprecations 👋</h3>

* Providing a value of ``None`` to ``aux_wire`` of ``qml.gradients.hadamard_grad`` in reversed or standard mode has been
  deprecated and will no longer be supported in 0.46. An ``aux_wire`` will no longer be automatically assigned.
  [(#8905)](https://github.com/PennyLaneAI/pennylane/pull/8905)

<h3>Internal changes ⚙️</h3>

* Updated internal dependencies `autoray` (to 0.8.4), `tach` (to 0.33).
  [(#8911)](https://github.com/PennyLaneAI/pennylane/pull/8911)

* Relaxed the `torch` dependency from `==2.9.0` to `~=2.9.0` to allow for compatible patch updates.
  [(#8911)](https://github.com/PennyLaneAI/pennylane/pull/8911)

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* Fixes an issue when binding a transform when the first positional arg
  is a `Sequence`, but not a `Sequence` of tapes.
  [(#8920)](https://github.com/PennyLaneAI/pennylane/pull/8920)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Andrija Paurevic,
Omkar Sarkar,
Jay Soni,
David Wierichs,
