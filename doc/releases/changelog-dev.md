:orphan:

# Release 0.25.0-dev (development release)

<h3>New features since last release</h3>

* Added new device `default.qutrit` for simulation of devices with wires of three dimensions. Users can currently make measurements with `qml.state()` and `qml.probs()`.
* Added operation `qml.QutritUnitary` for basic operation of `default.qutrit` device with user-specified unitaries.



  ```pycon
  >>> dev = qml.device("default.qutrit", wires=1)
  >>> def U = np.multiply(1 / np.sqrt(2), [[1, 1, 0], [1, -1, 0], [0, 0, np.sqrt(2)]])
  >>> @qml.qnode(dev)
  >>> def circuit(U):
  ...     qml.QutritUnitary(U, wires=0)
  ...     return qml.probs(wires=0)
  >>> print(circuit(U))
  [0.5 0.5 0. ]
  ```

<h3>Improvements</h3>

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Mudit Pandey
