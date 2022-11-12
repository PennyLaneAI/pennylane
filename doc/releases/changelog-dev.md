:orphan:

# Release 0.28.0-dev (development release)

<h3>New features since last release</h3>

<h4>Purity</h4>

* Support for purity computation is added. The `qml.math.purity` function computes the purity from a state vector or a density matrix:

    ```pycon
    >>> x = [1, 0, 0, 1] / np.sqrt(2)
    >>> qml.math.purity(x, [0, 1])
    1.0
    >>> qml.math.purity(x, [0])
    0.5
    
    >>> x = [[1 / 2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1 / 2]]
    >>> qml.math.purity(x, [0, 1])
    0.5
    ```
    The `qml.purity` measurement process can be returned from a QNode:
    
    ```python3
    dev = qml.device("default.mixed", wires=2)

    @qml.qnode(dev)
    def noisy_circuit_purity(p):
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.BitFlip(p, wires=0)
        qml.BitFlip(p, wires=1)
        return qml.purity(wires=[0, 1])
    ```
    ```pycon
    >>> noisy_circuit_purity(0.2)
    0.5648000000000398
    ```
    The `qml.qinfo.purity` can be used to transform a QNode returning a state to a function that returns the mutual information:
    ```python3
    dev = qml.device("default.mixed", wires=2)

    @qml.qnode(dev)
    def circuit(x):
        qml.IsingXX(x, wires=[0, 1])
        return qml.state()
    ```
    ```pycon
    >>> qml.qinfo.purity(circuit, wires=[0])(np.pi / 2)
    0.5
    >>> qml.qinfo.purity(circuit, wires=[0, 1])(np.pi / 2)
    1.0
    ```
    Taking the gradient is also supported:
    ```pycon
    >>> param = np.array(np.pi / 4, requires_grad=True)
    >>> qml.grad(qml.qinfo.purity(circuit, wires=[0]))(param)
    -0.5
    ```

<h3>Improvements</h3>

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
