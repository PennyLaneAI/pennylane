
# Release 0.31.1

<h3>Improvements 🛠</h3>

* `data.Dataset` now uses HDF5 instead of dill for serialization.
  [(#4097)](https://github.com/PennyLaneAI/pennylane/pull/4097)

* The `qchem` functions `primitive_norm` and `contracted_norm` are modified to
  be compatible with higher versions of scipy.
  [(#4321)](https://github.com/PennyLaneAI/pennylane/pull/4321)

<h3>Bug Fixes 🐛</h3>

* Dataset URLs are now properly escaped when fetching from S3.
  [(#4412)](https://github.com/PennyLaneAI/pennylane/pull/4412)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Utkarsh Azad,
Jack Brown,
Soran Jahangiri
