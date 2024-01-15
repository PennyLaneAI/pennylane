:orphan:

# Release 0.35.0-dev (development release)

<h3>New features since last release</h3>

* New `qml.commutator` function that allows to compute commutators between
  `qml.operation.Operator`, `qml.pauli.PauliWord` and `qml.pauli.PauliSentence` instances.
  ```pycon
  >>> qml.commutator(qml.PauliX(0), qml.PauliY(0))
  2j*(PauliZ(wires=[0]))
  ```

  We can return a `PauliSentence` instance by setting `pauli=True`.

  ```pycon
  >>> op1 = qml.PauliX(0) @ qml.PauliX(1)
  >>> op2 = qml.PauliY(0) + qml.PauliY(1)
  >>> qml.commutator(op1, op2, pauli=True)
  2j * X(1) @ Z(0)
  + 2j * Z(1) @ X(0)
  ```

  We can also input `PauliWord` and `PauliSentence` instances.

  ```pycon
  >>> op1 = PauliWord({0:"X", 1:"X"})
  >>> op2 = PauliWord({0:"Y"}) + PauliWord({1:"Y"})
  >>> qml.commutator(op1, op2, pauli=True)
  2j * Z(0) @ X(1)
  + 2j * X(0) @ Z(1)
  ```

  We can also compute commutators with Pauli operators natively with the `|` (or) operator.
  ```pycon
  >>> op1 = PauliWord({0:"X", 1:"X"})
  >>> op2 = PauliWord({0:"Y"}) + PauliWord({1:"Y"})
  >>> op1 | op2
  2j * Z(0) @ X(1)
  + 2j * X(0) @ Z(1)
  ```
  [(#5051)](https://github.com/PennyLaneAI/pennylane/pull/5051)
  [(#5052)](https://github.com/PennyLaneAI/pennylane/pull/5052)

<h3>Improvements 🛠</h3>

* Improve the performance of circuit-cutting workloads with large numbers of generated tapes.
  [(#5005)](https://github.com/PennyLaneAI/pennylane/pull/5005)

* Update `tests/ops/functions/conftest.py` to ensure all operator types are tested for validity.
  [(#4978)](https://github.com/PennyLaneAI/pennylane/pull/4978)

* Upgrade Pauli arithmetic with multiplying by scalars, e.g. `0.5 * PauliWord({0:"X"})` or `0.5 * PauliSentence({PauliWord({0:"X"}): 1.})`.
  [(#4989)](https://github.com/PennyLaneAI/pennylane/pull/4989)

* Upgrade Pauli arithmetic addition. You can now intuitively add together 
  `PauliWord` and `PauliSentence` as well as scalars, which are treated implicitly as identities.
  For example `ps1 + pw1 + 1.` for some Pauli word `pw1 = PauliWord({0: "X", 1: "Y"})` and Pauli
  sentence `ps1 = PauliSentence({pw1: 3.})`.
  [(#5001)](https://github.com/PennyLaneAI/pennylane/pull/5001)

* Upgrade Pauli arithmetic with subtraction. You can now subtract `PauliWord` and `PauliSentence`
  instances, as well as scalars, from each other.
  For example `ps1 - pw1 - 1` for `pw1 = PauliWord({0: "X", 1: "Y"})` and `ps1 = PauliSentence({pw1: 3.})`.
  [(#5003)](https://github.com/PennyLaneAI/pennylane/pull/5003)
  
* A new `pennylane.workflow` module is added. This module now contains `qnode.py`, `execution.py`, `set_shots.py`, `jacobian_products.py`, and the submodule `interfaces`.
  [(#5023)](https://github.com/PennyLaneAI/pennylane/pull/5023)

* Composite operations (eg. those made with `qml.prod` and `qml.sum`) convert `Hamiltonian` and
  `Tensor` operands to `Sum` and `Prod` types, respectively. This helps avoid the mixing of
  incompatible operator types.
  [(#5031)](https://github.com/PennyLaneAI/pennylane/pull/5031)

* Raise a more informative error when calling `adjoint_jacobian` with trainable state-prep operations.
  [(#5026)](https://github.com/PennyLaneAI/pennylane/pull/5026)

<h4>Community contributions 🥳</h4>

* The transform `split_non_commuting` now accepts measurements of type `probs`, `sample` and `counts` which accept both wires and observables. 
  [(#4972)](https://github.com/PennyLaneAI/pennylane/pull/4972)

<h3>Breaking changes 💔</h3>

* Passing additional arguments to a transform that decorates a QNode must be done through the use
  of `functools.partial`.
  [(#5046)](https://github.com/PennyLaneAI/pennylane/pull/5046)

* `Observable.return_type` has been removed. Instead, you should inspect the type
  of the surrounding measurement process.
  [(#5044)](https://github.com/PennyLaneAI/pennylane/pull/5044)

* `ClassicalShadow.entropy()` no longer needs an `atol` keyword as a better
  method to estimate entropies from approximate density matrix reconstructions
  (with potentially negative eigenvalues) has been implemented.
  [(#5048)](https://github.com/PennyLaneAI/pennylane/pull/5048)

<h3>Deprecations 👋</h3>

* Matrix and tensor products between `PauliWord` and `PauliSentence` instances are done using the `@` operator, `*` will be used only for scalar multiplication.
  [(#4989)](https://github.com/PennyLaneAI/pennylane/pull/4989)

* `MeasurementProcess.name` and `MeasurementProcess.data` are now deprecated, as they contain dummy
  values that are no longer needed.
  [(#5047)](https://github.com/PennyLaneAI/pennylane/pull/5047)

<h3>Documentation 📝</h3>

* A typo in a code example in the `qml.transforms` API has been fixed.
  [(#5014)](https://github.com/PennyLaneAI/pennylane/pull/5014)

* A typo in the code example for `qml.qchem.dipole_of` has been fixed.
  [(#5036)](https://github.com/PennyLaneAI/pennylane/pull/5036) 

<h3>Bug fixes 🐛</h3>

* `StatePrep` operations expanded onto more wires are now compatible with backprop.
  [(#5028)](https://github.com/PennyLaneAI/pennylane/pull/5028)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Abhishek Abhishek,
Pablo Antonio Moreno Casares,
Christina Lee,
Isaac De Vlugt,
Korbinian Kottmann,
Lee J. O'Riordan,
Matthew Silverman.
