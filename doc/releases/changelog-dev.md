:orphan:

# Release 0.36.0-dev (development release)

<h3>New features since last release</h3>

<h4>Dynamical Lie Algebra functionality</h4>

* A new `qml.dla.lie_closure` function to compute the Lie closure of a list of operators.
  [(#5161)](https://github.com/PennyLaneAI/pennylane/pull/5161)
  [(#5169)](https://github.com/PennyLaneAI/pennylane/pull/5169)

  The Lie closure, pronounced "Lee closure", is a way to compute the so-called dynamical Lie algebra (DLA) of a set of operators.
  For a list of operators `ops = [op1, op2, op3, ..]`, one computes all nested commutators between `ops` until no new operators are generated from commutation.
  All these operators together form the DLA, see e.g. section IIB of [arXiv:2308.01432](https://arxiv.org/abs/2308.01432).

  Take for example the following ops

  ```python
  ops = [X(0) @ X(1), Z(0), Z(1)]
  ```

  A first round of commutators between all elements yields the new operators `Y(0) @ X(1)` and `X(0) @ Y(1)`.

  ```python
  >>> qml.commutator(X(0) @ X(1), Z(0))
  2j * (Y(0) @ X(1))
  >>> qml.commutator(X(0) @ X(1), Z(0))
  2j * (X(0) @ Y(1))
  ```

  A next round of commutators between all elements further yields the new operator `Y(0) @ Y(1)`.

  ```python
  >>> qml.commutator(X(0) @ Y(1), Z(0))
  -2j * (Y(0) @ Y(1))
  ```

  After that, no new operators emerge from taking nested commutators and we have the resulting DLA.
  This can now be done in short via `qml.dla.lie_closure` as follows.

  ```python
  >>> ops = [X(0) @ X(1), Z(0), Z(1)]
  >>> dla = qml.dla.lie_closure(ops)
  >>> print(dla)
  [1.0 * X(1) @ X(0),
   1.0 * Z(0),
   1.0 * Z(1),
   -1.0 * X(1) @ Y(0),
   -1.0 * Y(1) @ X(0),
   -1.0 * Y(1) @ Y(0)]
  ```
* Added new `SpectralNormError` class to the new error tracking functionality.
  [(#5154)](https://github.com/PennyLaneAI/pennylane/pull/5154)

* The `dynamic_one_shot` transform is introduced enabling dynamic circuit execution on circuits with shots and devices that support `MidMeasureMP` operations natively.
  [(#5266)](https://github.com/PennyLaneAI/pennylane/pull/5266)

* Added new function `qml.operation.convert_to_legacy_H` to convert `Sum`, `SProd`, and `Prod` to `Hamiltonian` instances.
  [(#5309)](https://github.com/PennyLaneAI/pennylane/pull/5309)    

<h3>Improvements 🛠</h3>

* Create the `qml.Reflection` operator, useful for amplitude amplification and its variants.
  [(##5159)](https://github.com/PennyLaneAI/pennylane/pull/5159)

  ```python
  @qml.prod
  def generator(wires):
        qml.Hadamard(wires=wires)

  U = generator(wires=0)

  dev = qml.device('default.qubit')
  @qml.qnode(dev)
  def circuit():

        # Initialize to the state |1>
        qml.PauliX(wires=0)

        # Apply the reflection
        qml.Reflection(U)

        return qml.state()

  ```
  
  ```pycon
  >>> circuit()
  tensor([1.+6.123234e-17j, 0.-6.123234e-17j], requires_grad=True)
  ```
  
* The `molecular_hamiltonian` function calls `PySCF` directly when `method='pyscf'` is selected.
  [(#5118)](https://github.com/PennyLaneAI/pennylane/pull/5118)
  
* All generators in the source code (except those in the `qchem` module) no longer return 
  `Hamiltonian` or `Tensor` instances. Wherever possible, these return `Sum`, `SProd`, and `Prod` instances.
  [(#5253)](https://github.com/PennyLaneAI/pennylane/pull/5253)

* Upgraded `null.qubit` to the new device API. Also, added support for all measurements and various modes of differentiation.
  [(#5211)](https://github.com/PennyLaneAI/pennylane/pull/5211)

* `qml.transforms.split_non_commuting` will now work with single-term operator arithmetic.
  [(#5314)](https://github.com/PennyLaneAI/pennylane/pull/5314)

<h3>Breaking changes 💔</h3>

* ``MeasurementProcess.name`` and ``MeasurementProcess.data`` have been removed. Use ``MeasurementProcess.obs.name`` and ``MeasurementProcess.obs.data`` instead.
  [(#5321)](https://github.com/PennyLaneAI/pennylane/pull/5321)

* `Operator.validate_subspace(subspace)` has been removed. Instead, you should use `qml.ops.qutrit.validate_subspace(subspace)`.
  [(#5311)](https://github.com/PennyLaneAI/pennylane/pull/5311)

* The contents of ``qml.interfaces`` is moved inside ``qml.workflow``. The old import path no longer exists.
  [(#5329)](https://github.com/PennyLaneAI/pennylane/pull/5329)

<h3>Deprecations 👋</h3>

* ``qml.load`` is deprecated. Instead, please use the functions outlined in the *Importing workflows* quickstart guide, such as ``qml.from_qiskit``.
  [(#5312)](https://github.com/PennyLaneAI/pennylane/pull/5312)

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* We no longer perform unwanted dtype promotion in the `pauli_rep` of `SProd` instances when using tensorflow.
  [(#5246)](https://github.com/PennyLaneAI/pennylane/pull/5246)

* Fixed `TestQubitIntegration.test_counts` in `tests/interfaces/test_jax_qnode.py` to always produce counts for all outcomes.
  [(#5336)](https://github.com/PennyLaneAI/pennylane/pull/5336)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso,
Astral Cai,
Amintor Dusko,
Pietropaolo Frisoni,
Soran Jahangiri,
Korbinian Kottmann,
Christina Lee,
Matthew Silverman.
