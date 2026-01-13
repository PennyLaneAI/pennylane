# Release 0.45.0 (development release)

<h3>New features since last release</h3>

* Added a ``qml.gate_sets`` that contains pre-defined gate sets such as ``qml.gate_sets.CLIFFORD_T_PLUS_RZ``
  that can be plugged into the ``gate_set`` argument of the :func:`~pennylane.transforms.decompose` transform.
  [(#8915)](https://github.com/PennyLaneAI/pennylane/pull/8915)

<h3>Improvements 🛠</h3>

* The ``qml.estimator.Resources`` class now has a nice string representation in Jupyter Notebooks.
  [(#8880)](https://github.com/PennyLaneAI/pennylane/pull/8880)

<h3>Labs: a place for unified and rapid prototyping of research software 🧪</h3>

* Removed all of the resource estimation functionality from the ``/labs/resource_estimation``
  module. Users can now directly access a more stable version of this functionality using the 
  :mod:`estimator <pennylane.estimator>` module. All experimental development of resource estimation
  will be added to ``/labs/estimator_beta``
  [(#8868)](https://github.com/PennyLaneAI/pennylane/pull/8868)

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Internal changes ⚙️</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Astral Cai,
Omkar Sarkar,
Jay Soni
