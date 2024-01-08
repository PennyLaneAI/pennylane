:orphan:

# Release 0.35.0-dev (development release)

<h3>New features since last release</h3>

* `default.clifford` device enables efficient simulation of larger-scale Clifford circuits
  defined in PennyLane using [stim](https://github.com/quantumlib/Stim).
  [(#4936)](https://github.com/PennyLaneAI/pennylane/pull/4936)

  ```python
  import pennylane as qml

  dev = qml.device("default.clifford", tableau=True)

  @qml.qnode(dev)
  def circuit():
      qml.CNOT(wires=[0, 1])
      qml.PauliX(wires=[1])
      qml.ISWAP(wires=[0, 1])
      qml.Hadamard(wires=[0])
      return qml.state()

  tableau = circuit()
  ```

  Given a circuit with the Clifford gates, one can use this device obtaining the Tableau representation
  as given in [Aaronson & Gottesman (2004)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.70.052328)

  ```pycon
  >>> tableau
  array([[0, 1, 1, 0, 0],
         [1, 0, 1, 1, 1],
         [0, 0, 0, 1, 0],
         [1, 0, 0, 1, 1]])
  ```

<h3>Improvements 🛠</h3>

* Update `tests/ops/functions/conftest.py` to ensure all operator types are tested for validity.
  [(#4978)](https://github.com/PennyLaneAI/pennylane/pull/4978)
  
<h4>Community contributions 🥳</h4>

* The transform `split_non_commuting` now accepts measurements of type `probs`, `sample` and `counts` which accept both wires and observables. 
  [(#4972)](https://github.com/PennyLaneAI/pennylane/pull/4972)

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

* A typo in a code example in the `qml.transforms` API has been fixed.
  [(#5014)](https://github.com/PennyLaneAI/pennylane/pull/5014)

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Abhishek Abhishek,
Utkarsh Azad,
Isaac De Vlugt,
Matthew Silverman.
