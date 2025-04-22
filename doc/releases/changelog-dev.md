:orphan:

# Release 0.42.0-dev (development release)

<h3>New features since last release</h3>

* The transform `convert_to_mbqc_gateset` is added to the `ftqc` module to convert arbitrary 
  circuits to a limited gate-set that can be translated to the MBQC formalism.
  [(7271)](https://github.com/PennyLaneAI/pennylane/pull/7271)

* The `RotXZX` operation is added to the `ftqc` module to support definition of a universal
  gate-set that can be translated to the MBQC formalism.
  [(7271)](https://github.com/PennyLaneAI/pennylane/pull/7271)

* Added `convert_to_su2` and `convert_to_su4` to `qml.math` that converts a unitary matrix to SU(2)
  and SU(4) respectively, and optionally a global phase.
  [(#7211)](https://github.com/PennyLaneAI/pennylane/pull/7211)

<h4>Resource-efficient Decompositions 🔎</h4>

* Implemented decomposition rules for `QubitUnitary` that is compatible with the new graph-based
  decomposition system and program capture.
  [(#7211)](https://github.com/PennyLaneAI/pennylane/pull/7211)

<h3>Improvements 🛠</h3>

* Alias for Identity (`I`) is now accessible from `qml.ops`.
  [(#7200)](https://github.com/PennyLaneAI/pennylane/pull/7200)

* Two-qubit `QubitUnitary` gates no longer decompose to fundamental rotation gates, it now 
  decomposes to single-qubit `QubitUnitary` gates. This allows the decomposition system to
  further decompose single-qubit unitary gates more flexibly using different rotations.
  [(#7211)](https://github.com/PennyLaneAI/pennylane/pull/7211)

<h3>Breaking changes 💔</h3>

* `qml.tape.TapeError` has been removed.
  [(#7205)](https://github.com/PennyLaneAI/pennylane/pull/7205)

<h3>Deprecations 👋</h3>

<h3>Internal changes ⚙️</h3>

* Introduce module dependency management using `tach`.
  [(#7185)](https://github.com/PennyLaneAI/pennylane/pull/7185)

* Add new `pennylane.exceptions` module for custom errors and warnings.
  [(#7205)](https://github.com/PennyLaneAI/pennylane/pull/7205)

* Clean up `__init__.py` files in `math`, `ops`, `qaoa`, `tape` and `templates` to be explicit in what they import. 
  [(#7200)](https://github.com/PennyLaneAI/pennylane/pull/7200)
  
* The `Tracker` class has been moved into the `devices` module.
  [(#7281)](https://github.com/PennyLaneAI/pennylane/pull/7281)

* Moved functions that calculate rotation angles for unitary decompositions into an internal
  module `qml.math.decomposition`
  [(#7211)](https://github.com/PennyLaneAI/pennylane/pull/7211)

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* Gradient transforms can now be used in conjunction with batch transforms with all interfaces.
  [(#7287)](https://github.com/PennyLaneAI/pennylane/pull/7287)

* Fixes a bug where the global phase was not being added in the ``QubitUnitary`` decomposition.  
  [(#7244)](https://github.com/PennyLaneAI/pennylane/pull/7244)
  [(#7270)](https://github.com/PennyLaneAI/pennylane/pull/7270)

* Using finite differences with program capture without x64 mode enabled now raises a warning.
  [(#7282)](https://github.com/PennyLaneAI/pennylane/pull/7282)

* When the `mcm_method` is specified to the `"device"`, the `defer_measurements` transform will 
  no longer be applied. Instead, the device will be responsible for all MCM handling.
  [(#7243)](https://github.com/PennyLaneAI/pennylane/pull/7243)

* Fixed coverage of `qml.liealg.CII` and `qml.liealg.AIII`.
  [(#7291)](https://github.com/PennyLaneAI/pennylane/pull/7291)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso-Linaje,
Astral Cai,
Lillian Frederiksen,
Andrija Paurevic,
Korbinian Kottmann,
Christina Lee
