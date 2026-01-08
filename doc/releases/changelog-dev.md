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

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Internal changes ⚙️</h3>

<h3>Documentation 📝</h3>

* The docstring for :func:`pennylane.qjit` has been updated to match :func:`catalyst.qjit`.
  [(#8485)](https://github.com/PennyLaneAI/pennylane/pull/8485)

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Omkar Sarkar,
Jay Soni
