:orphan:

# Release 0.34.0-dev (development release)

<h3>New features since last release</h3>

* `qml.transforms.decomposition.sk_decomposition` method implements the Solovay-Kitaev algorithm for
  approximately decomposing any single-qubit operation to Clifford+T basis.
  [(#4761)](https://github.com/PennyLaneAI/pennylane/pull/4761)

* Approximate Quantum Fourier Transform (AQFT) is now available from `qml.AQFT`.
  [(#4656)](https://github.com/PennyLaneAI/pennylane/pull/4656)

<h3>Improvements 🛠</h3>

* Updates to some relevant Pytests to enable its use as a suite of benchmarks.
  [(#4703)](https://github.com/PennyLaneAI/pennylane/pull/4703)

* Added `__iadd__` method to PauliSentence, which enables inplace-addition using `+=`, we no longer need to perform a copy, leading to performance improvements.
[(#4662)](https://github.com/PennyLaneAI/pennylane/pull/4662) 

* `qml.ArbitraryUnitary` now supports batching.
  [(#4745)](https://github.com/PennyLaneAI/pennylane/pull/4745)

* `qml.draw` and `qml.draw_mpl` now render operator ids.
  [(#4749)](https://github.com/PennyLaneAI/pennylane/pull/4749)

<h3>Breaking changes 💔</h3>

* The `prep` keyword argument has been removed from `QuantumScript` and `QuantumTape`.
  `StatePrepBase` operations should be placed at the beginning of the `ops` list instead.
  [(#4756)](https://github.com/PennyLaneAI/pennylane/pull/4756)

* `qml.gradients.pulse_generator` has become `qml.gradients.pulse_odegen` to adhere to paper naming conventions.
  [(#4769)](https://github.com/PennyLaneAI/pennylane/pull/4769)

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Utkarsh Azad,
Amintor Dusko,
Ankit Khandelwal,
Anurav Modak,
Matthew Silverman,
David Wierichs,
Justin Woodring,
