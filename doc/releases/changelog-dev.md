:orphan:

# Release 0.38.0-dev (development release)

<h3>New features since last release</h3>
* A new method `process_density_matrix` has been added to the `ProbabilityMP` and `DensityMatrixMP` classes, allowing for more efficient handling of quantum density matrices, particularly with batch processing support. This method simplifies the calculation of probabilities from quantum states represented as density matrices.
  [(#5830)](https://github.com/PennyLaneAI/pennylane/pull/5830)
  
<h3>Improvements 🛠</h3>

* `qml.devices.LegacyDeviceFacade` has been added to map the legacy devices to the new
  device interface.
  [(#5927)](https://github.com/PennyLaneAI/pennylane/pull/5927)

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Yushao Chen.