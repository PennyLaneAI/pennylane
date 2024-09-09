:orphan:

# Release 0.39.0-dev (development release)

<h3>New features since last release</h3>
 
<h3>Improvements 🛠</h3>
* The `diagonalize_measurements` transform now uses a more efficient method of diagonalization 
  when possible, based on the `pauli_rep` of the relevant observables.
  [#6113](https://github.com/PennyLaneAI/pennylane/pull/6113/)

<h4>Capturing and representing hybrid programs</h4>

* Differentiation of hybrid programs via `qml.grad` can now be captured into plxpr.
  When evaluating a captured `qml.grad` instruction, it will dispatch to `jax.grad`,
  which differs from the Autograd implementation of `qml.grad` itself.
  [(#6120)](https://github.com/PennyLaneAI/pennylane/pull/6120)

* Improve unit testing for capturing of nested control flows.
  [(#6111)](https://github.com/PennyLaneAI/pennylane/pull/6111)

* Some custom primitives for the capture project can now be imported via
  `from pennylane.capture.primitives import *`.
  [(#6129)](https://github.com/PennyLaneAI/pennylane/pull/6129)

* The `SampleMP.process_samples` method is updated to support using JAX tracers
  for samples, allowing compatiblity with Catalyst workflows.
  [(#6211)](https://github.com/PennyLaneAI/pennylane/pull/6211)

* Improve `qml.Qubitization` decomposition.
  [(#6182)](https://github.com/PennyLaneAI/pennylane/pull/6182)

* The `__repr__` methods for `FermiWord` and `FermiSentence` now returns a
  unique representation of the object.
  [(#6167)](https://github.com/PennyLaneAI/pennylane/pull/6167)


<h3>Breaking changes 💔</h3>

* The `simplify` argument in `qml.Hamiltonian` and `qml.ops.LinearCombination` has been removed.
  Instead, `qml.simplify()` can be called on the constructed operator.
  [(#6242)](https://github.com/PennyLaneAI/pennylane/pull/6242)

* Legacy operator arithmetic has been deprecated. This includes `qml.ops.Hamiltonian`, `qml.operation.Tensor`,
  `qml.operation.enable_new_opmath`, `qml.operation.disable_new_opmath`, and `qml.operation.convert_to_legacy_H`.
  [(#6242)](https://github.com/PennyLaneAI/pennylane/pull/6242)

* Remove support for Python 3.9.
  [(#6223)](https://github.com/PennyLaneAI/pennylane/pull/6223)

* `DefaultQubitTF` and `DefaultQubitTorch` are removed. Please use `default.qubit` for all interfaces.
  [(#6207)](https://github.com/PennyLaneAI/pennylane/pull/6207)
  [(#6208)](https://github.com/PennyLaneAI/pennylane/pull/6208)

* `expand_fn`, `max_expansion`, `override_shots`, and `device_batch_transform` are removed from the
  signature of `qml.execute`.
  [(#6203)](https://github.com/PennyLaneAI/pennylane/pull/6203)

* `max_expansion` and `expansion_strategy` are removed from the `QNode`.
  [(#6203)](https://github.com/PennyLaneAI/pennylane/pull/6203)

* `expansion_strategy` is removed from `qml.draw`, `qml.draw_mpl`, and `qml.specs`. `max_expansion` is removed from `qml.specs`, as it had no impact on the output.
  [(#6203)](https://github.com/PennyLaneAI/pennylane/pull/6203)

* `qml.transforms.hamiltonian_expand` and `qml.transforms.sum_expand` are removed.
  Please use `qml.transforms.split_non_commuting` instead.
  [(#6204)](https://github.com/PennyLaneAI/pennylane/pull/6204)

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* Fix `qml.PrepSelPrep` template to work with `torch`:
  [(#6191)](https://github.com/PennyLaneAI/pennylane/pull/6191)

* Now `qml.equal` compares correctly `qml.PrepSelPrep` operators.
  [(#6182)](https://github.com/PennyLaneAI/pennylane/pull/6182)

* The ``qml.QSVT`` template now orders the ``projector`` wires first and the ``UA`` wires second, which is the expected order of the decomposition.
  [(#6212)](https://github.com/PennyLaneAI/pennylane/pull/6212)
  
* The ``qml.Qubitization`` template now orders the ``control`` wires first and the ``hamiltonian`` wires second, which is the expected according to other templates.
  [(#6229)](https://github.com/PennyLaneAI/pennylane/pull/6229)

* <h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso,
Utkarsh Azad,
Lillian M. A. Frederiksen,
Christina Lee,
William Maxwell,
Lee J. O'Riordan,
Mudit Pandey,
David Wierichs,
