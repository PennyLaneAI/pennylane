:orphan:

# Release 0.28.0-dev (development release)

<h3>New features since last release</h3>

* Support custom measurement processes:
  * `SampleMeasurement` and `StateMeasurement` classes have been added. They contain an abstract
    method to process samples/quantum state.
    [#3286](https://github.com/PennyLaneAI/pennylane/pull/3286)

  * Add `_MutualInfo` class.
    [#3327](https://github.com/PennyLaneAI/pennylane/pull/3327)

* Functionality for fetching symbols and geometry of a compound from the PubChem Database using `qchem.mol_data`.
  [(#3289)](https://github.com/PennyLaneAI/pennylane/pull/3289)
 
  ```pycon
  >>> mol_data("BeH2")
  (['Be', 'H', 'H'],
  array([[ 4.79405604,  0.29290815,  0.        ],
         [ 3.77946   , -0.29290815,  0.        ],
         [ 5.80884105, -0.29290815,  0.        ]]))

  >>> mol_data(223, "CID")
  (['N', 'H', 'H', 'H', 'H'],
  array([[ 4.79404621,  0.        ,  0.        ],
         [ 5.80882913,  0.5858151 ,  0.        ],
         [ 3.77945225, -0.5858151 ,  0.        ],
         [ 4.20823111,  1.01459396,  0.        ],
         [ 5.3798613 , -1.01459396,  0.        ]]))
  ```

* New basis sets, `6-311g` and `CC-PVDZ`, are added to the qchem basis set repo.
  [#3279](https://github.com/PennyLaneAI/pennylane/pull/3279)

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


* Improve performance of `Wires.all_wires`.
  [(#3302)](https://github.com/PennyLaneAI/pennylane/pull/3302)


* A representation has been added to the `Molecule` class.
  [#3364](https://github.com/PennyLaneAI/pennylane/pull/3364)

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Documentation</h3>

<h3>Bug fixes</h3>

* Small fix of `MeasurementProcess.map_wires`, where both the `self.obs` and `self._wires`
  attributes were modified.
  [#3292](https://github.com/PennyLaneAI/pennylane/pull/3292)

* If the device originally has no shots but finite shots are dynamically specified, Hamiltonian
  expansion now occurs.
  [(#3369)](https://github.com/PennyLaneAI/pennylane/pull/3369)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):
Juan Miguel Arrazola
Utkarsh Azad
Astral Cai
Pieter Eendebak
Soran Jahangiri
Christina Lee
Albert Mitjans Coma
