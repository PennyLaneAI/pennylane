:orphan:

# Release 0.41.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* The program capture execution pipeline supports control flow if program capture is enabled. 
  [(#)]()

  ```python
  qml.capture.enable()

  @qml.qnode(qml.device("default.qubit"))
  def circuit(x):
      if x > 3:
        qml.Hadamard(0)
        qml.CNOT([0,1])
      else:
        qml.X(0)
      return {"Probabilities": qml.probs(), "State": qml.state()}
  ```
  ```pycon
  >>> circuit(3.5)
  {'Probabilities': Array([0.49999997, 0.        , 0.        , 0.49999997], dtype=float32),
   'State': Array([0.70710677+0.j, 0.        +0.j, 0.        +0.j, 0.70710677+0.j],      dtype=complex64)}
  >>> circuit(2.5)
  {'Probabilities': Array([0., 0., 1., 0.], dtype=float32),
   'State': Array([0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j], dtype=complex64)} 
  ```

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

* Updated documentation for vibrational Hamiltonians
  [(#6717)](https://github.com/PennyLaneAI/pennylane/pull/6717)

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):
Diksha Dhawan