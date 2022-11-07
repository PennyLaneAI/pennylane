:orphan:

# Release 0.28.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements</h3>

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

* The following deprecated methods are removed:

  - `qml.tape.get_active_tape`: Use `qml.QueuingManager.active_context()`
  - `qml.transforms.qcut.remap_tape_wires`: Use `qml.map_wires`
  - `qml.tape.QuantumTape.inv()`: Use `qml.tape.QuantumTape.adjoint()`
  - `qml.tape.stop_recording()`: Use `qml.QueuingManager.stop_recording()`
  - `qml.tape.QuantumTape.stop_recording()`: Use `qml.QueuingManager.stop_recording()`
  - `qml.QueuingContext` is now `qml.QueuingManager`
  - `QueuingManager.safe_update_info` and `AnnotatedQueue.safe_update_info`: Use plain `update_info`

<h3>Documentation</h3>

<h3>Bug fixes</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Christina Lee