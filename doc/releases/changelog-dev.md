:orphan:

# Release 0.39.0-dev (development release)

<h3>New features since last release</h3>
 
<h3>Improvements 🛠</h3>

* `qml.qchem.excitations` now optionally returns fermionic operators.
   [(#6171)](https://github.com/PennyLaneAI/pennylane/pull/6171)

* The `diagonalize_measurements` transform now uses a more efficient method of diagonalization 
  when possible, based on the `pauli_rep` of the relevant observables.
  [#6113](https://github.com/PennyLaneAI/pennylane/pull/6113/)

<h4>Capturing and representing hybrid programs</h4>

* Differentiation of hybrid programs via `qml.grad` can now be captured into plxpr.
  When evaluating a captured `qml.grad` instruction, it will dispatch to `jax.grad`,
  which differs from the Autograd implementation of `qml.grad` itself.
  [(#6120)](https://github.com/PennyLaneAI/pennylane/pull/6120)

<h4>Capturing and representing hybrid programs</h4>

* Differentiation of hybrid programs via `qml.grad` and `qml.jacobian` can now be captured
  into plxpr. When evaluating a captured `qml.grad` (`qml.jacobian`) instruction, it will
  dispatch to `jax.grad` (`jax.jacobian`), which differs from the Autograd implementation
  without capture.
  [(#6120)](https://github.com/PennyLaneAI/pennylane/pull/6120)
  [(#6127)](https://github.com/PennyLaneAI/pennylane/pull/6127)

* Improve unit testing for capturing of nested control flows.
  [(#6111)](https://github.com/PennyLaneAI/pennylane/pull/6111)

* Some custom primitives for the capture project can now be imported via
  `from pennylane.capture.primitives import *`.
  [(#6129)](https://github.com/PennyLaneAI/pennylane/pull/6129)

* `FermiWord` and `FermiSentence` classes now have methods to compute adjoints.
  [(#6166)](https://github.com/PennyLaneAI/pennylane/pull/6166)

* The `SampleMP.process_samples` method is updated to support using JAX tracers
  for samples, allowing compatiblity with Catalyst workflows.
  [(#6211)](https://github.com/PennyLaneAI/pennylane/pull/6211)

* Improve `qml.Qubitization` decomposition.
  [(#6182)](https://github.com/PennyLaneAI/pennylane/pull/6182)

* The `__repr__` methods for `FermiWord` and `FermiSentence` now returns a
  unique representation of the object.
  [(#6167)](https://github.com/PennyLaneAI/pennylane/pull/6167)

* The `to_mat` methods for `FermiWord` and `FermiSentence` now optionally return
  a sparse matrix.
  [(#6173)](https://github.com/PennyLaneAI/pennylane/pull/6173)

<h3>Breaking changes 💔</h3>

* Remove support for Python 3.9.
  [(#6223)](https://github.com/PennyLaneAI/pennylane/pull/6223)

* `DefaultQubitTF`, `DefaultQubitTorch`, and `DefaultQubitJax` are removed. Please use `default.qubit` for all interfaces.
  [(#6207)](https://github.com/PennyLaneAI/pennylane/pull/6207)
  [(#6208)](https://github.com/PennyLaneAI/pennylane/pull/6208)
  [(#6209)](https://github.com/PennyLaneAI/pennylane/pull/6209)

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

* Fix a bug where zero-valued JVPs were calculated wrongly in the presence of shot vectors.
  [(#6219)](https://github.com/PennyLaneAI/pennylane/pull/6219)

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
David Wierichs,
