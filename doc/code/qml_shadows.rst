qml.shadows
=============

Overview
--------

.. currentmodule:: pennylane.shadows

.. automodule:: pennylane.shadows

    .. rubric:: Classes

    .. autosummary::
        :toctree:

        pennylane.ClassicalShadow

    .. rubric:: Functions

    .. autosummary::
        :toctree:

        pennylane.classical_shadow
        pennylane.shadow_expval

Classical Shadows formalism
---------------------------

A :class:`ClassicalShadow` is a classical description of a quantum state that is capable of reproducing expectation values of local Pauli observables, see `2002.08953 <https://arxiv.org/abs/2002.08953>`_.

The idea is to capture :math:`T` local snapshots (given by the ``shots`` set in the device) of the state by performing measurements in random Pauli bases at each qubit.
The measurement outcomes, denoted ``bits``, as well as the choices of measurement bases, ``recipes``, are recorded in two ``(T, len(wires))`` integer tensors, respectively.

From the :math:`t`-th measurement, we can reconstruct the ``local_snapshots`` (see :class:`ClassicalShadow` methods)

.. math:: \rho^{(t)} = \bigotimes_{i=1}^{n} 3 U^\dagger_i |b_i \rangle \langle b_i | U_i - \mathbb{I},

where :math:`U_i` is the rotation corresponding to the measurement (e.g. :math:`U_i=H` for measurement in :math:`X`) of qubit :math:`i` at snapshot :math:`t` and
:math:`|b_i\rangle = (1 - b_i, b_i)`
the corresponding computational basis state given the output bit :math:`b_i`.

From these local snapshots, one can compute expectation values of local Pauli strings, where locality refers to the number of non-Identity operators.
The accuracy of the procedure is determined by the number of measurements :math:`T` (``shots``).
To target an error :math:`\epsilon`, one needs of order :math:`T = \mathcal{O}\left( \log(M) 4^\ell/\epsilon^2 \right)` measurements to determine :math:`M` different,
:math:`\ell`-local observables.

One can in principle also reconstruct the global state :math:`\sum_t \rho^{(t)}/T`, though it is not advisable nor practical for larger systems due to its exponential scaling.