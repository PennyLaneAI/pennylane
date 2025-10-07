:orphan:

# Release 0.44.0-dev (development release)

<h3>New features since last release</h3>

* A compilation pass written with xDSL called `qml.compiler.python_compiler.transforms.ParitySynthPass`
  has been added for the experimental xDSL Python compiler integration. This pass resynthesizes
  subcircuits that form a phase polynomial (``CNOT`` and ``RZ`` gates), using ``ParitySynth`` by
  [Vandaele et al.](https://arxiv.org/abs/2104.00934)
  [(#8414)](https://github.com/PennyLaneAI/pennylane/pull/8414)

<h3>Improvements 🛠</h3>

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):
David Wierichs,
