:orphan:

# Release 0.31.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* An error is raised by `qchem.molecular_hamiltonian` when the `dhf` method is used for an 
  open-shell system. This duplicates a similar error in `qchem.Molecule` but makes it easier to
  inform the users that the `pyscf` backend can be used for open-shell calculations.
  [(4058)](https://github.com/PennyLaneAI/pennylane/pull/4058)

* Added a `shots` property to `QuantumScript`. This will allow shots to be tied to executions instead of devices more
  concretely.
  [(#4067)](https://github.com/PennyLaneAI/pennylane/pull/4067)

* `qml.specs` is compatible with custom operations that have `depth` bigger than 1.
  [(#4033)](https://github.com/PennyLaneAI/pennylane/pull/4033)

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

* The description of `mult` in the `qchem.Molecule` docstring now correctly states the value
  of `mult` that is supported.
  [(4058)](https://github.com/PennyLaneAI/pennylane/pull/4058)

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Isaac De Vlugt,
Soran Jahangiri,
Mudit Pandey,
Jay Soni,
