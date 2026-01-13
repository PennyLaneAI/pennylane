# Release 0.45.0 (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

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

* The :attr:`pennylane.operation.Operator.is_hermitian` property has been removed and replaced 
  with :attr:`pennylane.operation.Operator.is_verified_hermitian` as it better reflects the functionality of this property.
  Alternatively, consider using the :func:`pennylane.is_hermitian` function instead as it provides a more reliable check for hermiticity.
  Please be aware that it comes with a higher computational cost.
  [(#8919)](https://github.com/PennyLaneAI/pennylane/pull/8919)

* Passing a function to the `gate_set` argument in the :func:`~pennylane.transforms.decompose` transform
  is removed. The `gate_set` argument expects a static iterable of operator type and/or operator names,
  and the function should be passed to the `stopping_condition` argument instead.
  [(#8919)](https://github.com/PennyLaneAI/pennylane/pull/8919)

<h3>Deprecations 👋</h3>

<h3>Internal changes ⚙️</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Andrija Paurevic,
Omkar Sarkar,
Jay Soni
