:orphan:

# Release 0.28.0-dev (development release)

<h3>New features since last release</h3>

* New basis sets, `6-311g` and `CC-PVDZ`, are added to the qchem basis set repo.
  [#3279](https://github.com/PennyLaneAI/pennylane/pull/3279)

* We can now perform QPE on a quantum function:

  ```python3
  dev = qml.device("default.qubit", wires=2)

  @qml.qnode(dev)
  def unitary():
      qml.RX(np.pi / 2, wires=[0])
      qml.CNOT(wires=[0, 1])
      return qml.state()

  eigenvector = np.array([-1/2, -1/2, 1/2, 1/2])

  n_estimation_wires = 5
  estimation_wires = range(2, n_estimation_wires + 2)
  target_wires = [0, 1]

  dev = qml.device("default.qubit", wires=n_estimation_wires + 2)

  @qml.qnode(dev)
  def circuit():
      qml.QubitStateVector(eigenvector, wires=target_wires)
      QuantumPhaseEstimation(
          unitary,
          target_wires=target_wires,
          estimation_wires=estimation_wires,
      )
      return qml.probs(estimation_wires)

  phase_estimated = np.argmax(circuit()) / 2 ** n_estimation_wires
  ```

<h3>Improvements</h3>

* A representation has been added to the `Molecule` class.
  [#3364](https://github.com/PennyLaneAI/pennylane/pull/3364)

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Documentation</h3>

<h3>Bug fixes</h3>

* Small fix of `MeasurementProcess.map_wires`, where both the `self.obs` and `self._wires`
  attributes were modified.
  [#3292](https://github.com/PennyLaneAI/pennylane/pull/3292)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Astral Cai
