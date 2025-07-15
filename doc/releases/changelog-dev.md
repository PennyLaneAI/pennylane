:orphan:

# Release 0.43.0-dev (development release)

<h3>New features since last release</h3>

<h4>State of the art templates and decompositions 🐝</h4>

* The decomposition of :class:`~.BasisRotation` has been optimized to skip redundant phase shift gates
  with angle :math:`\pm \pi` for real-valued, i.e., orthogonal, rotation matrices. This uses the fact that
  no or single :class:`~.PhaseShift` gate is required in case the matrix has a determinant :math:`\pm 1`.
  [(#7765)](https://github.com/PennyLaneAI/pennylane/pull/7765)

<h3>Improvements 🛠</h3>

* The matrix factorization using :func:`~.math.decomposition.givens_decomposition` has
  been optimized to factor out the redundant sign in the diagonal phase matrix for the
  real-valued (orthogonal) rotation matrices. For example, in case the determinant of a matrix is
  :math:`-1`, only a single element of the phase matrix is required.
  [(#7765)](https://github.com/PennyLaneAI/pennylane/pull/7765)

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
