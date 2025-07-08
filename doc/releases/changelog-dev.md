:orphan:

# Release 0.43.0-dev (development release)

<h3>New features since last release</h3>

<h4>State of the art templates and decompositions 🐝</h4>

* The decomposition of :class:`~.BasisRotation` has been updated to skip redundant phase shift
  gates with angle :math:`\pm \pi` for real-valued, i.e., orthogonal, rotation matrices. Only a
  single phase shift is required in case the matrix has determinant :math:`-1`.
  [(#7765)](https://github.com/PennyLaneAI/pennylane/pull/7765)

<h3>Improvements 🛠</h3>

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

* The theoretical background section of :class:`~.BasisRotation` has been extended to explain
  the underlying Lie group/algebra homomorphism between the (dense) rotation matrix and the
  performed operations on the target qubits.
  [(#7765)](https://github.com/PennyLaneAI/pennylane/pull/7765)

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):
David Wierichs,
