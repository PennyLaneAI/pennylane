:orphan:

# Release 0.28.0-dev (development release)

<h3>New features since last release</h3>

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

* If the device originally has no shots but finite shots are dynamically specified, Hamiltonian
  expansion now occurs.
  [(#3369)](https://github.com/PennyLaneAI/pennylane/pull/3369)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola
Utkarsh Azad
Soran Jahangiri
Christina Lee
