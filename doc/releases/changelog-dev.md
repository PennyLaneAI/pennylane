:orphan:

# Release 0.34.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* Autograd can now use vjps provided by the device from the new device API. If a device provides
  a vector Jacobian product, this can be selected by providing `use_device_jacobian_product=True` to
  `qml.execute`.
  [(#4557)](https://github.com/PennyLaneAI/pennylane/pull/4557)

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Christina Lee