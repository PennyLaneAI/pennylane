:orphan:

# Release 0.23.0-dev (development release)

<h3>New features since last release</h3>

* A differentiable quantum chemistry module is added to `qml.qchem`. The new module inherits a 
  modified version of the differentiable Hartree-Fock solver from `qml.hf`, contains new functions
  for building a differentiable dipole moment observable and also contains modified functions for 
  building spin and particle number observables independent of external libraries.

  - New functions are added for computing multipole moment molecular integrals
    [(#2166)](https://github.com/PennyLaneAI/pennylane/pull/2166)
  - New functions are added for building a differentiable dipole moment observable
    [(#2173)](https://github.com/PennyLaneAI/pennylane/pull/2173)
  - External dependencies are replaced with local functions for spin and particle number observables
    [(#2197)](https://github.com/PennyLaneAI/pennylane/pull/2197)
  - New functions are added for building fermionic and qubit observables
    [(#2230)](https://github.com/PennyLaneAI/pennylane/pull/2230)
  - A new module is created for hosting openfermion to pennylane observable conversion functions
    [(#2199)](https://github.com/PennyLaneAI/pennylane/pull/2199)
  - Expressive names are used for the Hartree-Fock solver functions
    [(#2272)](https://github.com/PennyLaneAI/pennylane/pull/2272)
  - These new additions are added to a feature branch
    [(#2164)](https://github.com/PennyLaneAI/pennylane/pull/2164)

* Development of a circuit-cutting compiler extension to circuits with sampling
  measurements has begun:

  - The existing `qcut.tape_to_graph()` method has been extended to convert a
    sample measurement without an observable specified to multiple single-qubit sample
    nodes.
    [(#2313)](https://github.com/PennyLaneAI/pennylane/pull/2313)

<h3>Improvements</h3>

* The function `qml.ctrl` was given the optional argument `control_values=None`.
  If overridden, `control_values` takes an integer or a list of integers corresponding to
  the binary value that each control value should take. The same change is reflected in
  `ControlledOperation`. Control values of `0` are implemented by `qml.PauliX` applied
  before and after the controlled operation
  [(#2288)](https://github.com/PennyLaneAI/pennylane/pull/2288)

* Circuit cutting now performs expansion to search for wire cuts in contained operations or tapes.
  [(#2340)](https://github.com/PennyLaneAI/pennylane/pull/2340)

<h3>Deprecations</h3>

<h3>Breaking changes</h3>

* The `ObservableReturnTypes` `Sample`, `Variance`, `Expectation`, `Probability`, `State`, and `MidMeasure`
  have been moved to `measurements` from `operation`.
  [(#2329)](https://github.com/PennyLaneAI/pennylane/pull/2329)

* The deprecated QNode, available via `qml.qnode_old.QNode`, has been removed. Please
  transition to using the standard `qml.QNode`.
  [(#2336)](https://github.com/PennyLaneAI/pennylane/pull/2336)

* The deprecated, non-batch compatible interfaces, have been removed.
  [(#2336)](https://github.com/PennyLaneAI/pennylane/pull/2336)

* The deprecated tape subclasses `QubitParamShiftTape`, `JacobianTape`, `CVParamShiftTape`, and
  `ReversibleTape` have been removed.
  [(#2336)](https://github.com/PennyLaneAI/pennylane/pull/2336)

<h3>Bug fixes</h3>

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Karim Alaa El-Din, Juan Miguel Arrazola, Thomas Bromley, Alain Delgado, Anthony Hayes, Josh Izaac,
Soran Jahangiri, Christina Lee, Jay Soni.
