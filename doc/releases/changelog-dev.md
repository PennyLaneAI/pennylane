:orphan:

# Release 0.31.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* The `qchem.molecular_hamiltonian` function is upgraded to support custom wires for constructing
  differentiable Hamiltonians.
  [(4050)](https://github.com/PennyLaneAI/pennylane/pull/4050)

* An error is now raised by `qchem.molecular_hamiltonian` when the `dhf` method is used for an 
  open-shell system. This duplicates a similar error in `qchem.Molecule` but makes it easier to
  inform the users that the `pyscf` backend can be used for open-shell calculations.
  [(4058)](https://github.com/PennyLaneAI/pennylane/pull/4058)

* Added a `shots` property to `QuantumScript`. This will allow shots to be tied to executions instead of devices more
  concretely.
  [(#4067)](https://github.com/PennyLaneAI/pennylane/pull/4067)

* `qml.specs` is compatible with custom operations that have `depth` bigger than 1.
  [(#4033)](https://github.com/PennyLaneAI/pennylane/pull/4033)

* `qml.prod` now accepts a single qfunc input for creating new `Prod` operators.
  [(#4011)](https://github.com/PennyLaneAI/pennylane/pull/4011)

* Added `__repr__` and `__str__` methods to the `Shots` class.
  [(#4081)](https://github.com/PennyLaneAI/pennylane/pull/4081)

* Added a function `measure_with_samples` that returns a sample-based measurement result given a state
  [(#4083)](https://github.com/PennyLaneAI/pennylane/pull/4083)

* Wrap all objects being queued in an `AnnotatedQueue` so that `AnnotatedQueue` is not dependent on
  the hash of any operators/measurement processes.
  [(#4087)](https://github.com/PennyLaneAI/pennylane/pull/4087)

* Support for adjoint differentiation has been added to the `DefaultQubit2` device.
  [(#4037)](https://github.com/PennyLaneAI/pennylane/pull/4037)


<h3>Breaking changes 💔</h3>

* Gradient transforms with Jax do not support `argnum` anymore,  `argnums` needs to be used.
  [(#4076)](https://github.com/PennyLaneAI/pennylane/pull/4076)

* `pennylane.collections`, `pennylane.op_sum`, and `pennylane.utils.sparse_hamiltonian` are removed.

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

* The description of `mult` in the `qchem.Molecule` docstring now correctly states the value
  of `mult` that is supported.
  [(4058)](https://github.com/PennyLaneAI/pennylane/pull/4058)

<h3>Bug fixes 🐛</h3>

* Removes a patch in `interfaces/autograd.py` that checks for the `strawberryfields.gbs` device.  That device
  is pinned to PennyLane <= v0.29.0, so that patch is no longer necessary.

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Isaac De Vlugt,
Soran Jahangiri,
Christina Lee,
Romain Moyard,
Mudit Pandey,
Matthew Silverman,
Jay Soni.
