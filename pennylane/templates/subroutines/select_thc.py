# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains the template for the tensor hypercontraction ``SELECT`` oracle."""

import numpy as np

from pennylane import math
from pennylane.core.operator import Operator2
from pennylane.typing import Wire
from pennylane.wires import Wires, WiresLike, validate_no_wire_overlaps


def _validate_select_data(chi, t_eigenvectors):
    """Validate the static THC leaf and one-body eigenvector matrices."""
    shapes = {}
    arrays = {}
    for name, matrix in (("chi", chi), ("t_eigenvectors", t_eigenvectors)):
        if (
            not isinstance(matrix, tuple)
            or not matrix
            or not all(isinstance(row, tuple) for row in matrix)
        ):
            raise ValueError(
                f"{name} must be a tuple of tuples of floats; got {type(matrix).__name__}."
            )
        num_columns = len(matrix[0])
        if num_columns == 0 or any(len(row) != num_columns for row in matrix):
            raise ValueError(f"{name} must be a non-empty rectangular matrix.")
        arr = np.asarray(matrix)
        if (
            not np.issubdtype(arr.dtype, np.number)
            or not np.isrealobj(arr)
            or not np.all(np.isfinite(arr))
        ):
            raise ValueError(f"{name} entries must be finite real numbers.")
        shapes[name] = (len(matrix), num_columns)
        arrays[name] = arr

    M, n_half = shapes["chi"]
    if shapes["t_eigenvectors"] != (n_half, n_half):
        raise ValueError(
            f"t_eigenvectors must have shape ({n_half}, {n_half}); got {shapes['t_eigenvectors']}."
        )
    if np.any(np.sum(arrays["chi"] ** 2, axis=1) < 1e-30):
        raise ValueError("Cannot build a rotation from a zero vector in chi.")
    if np.any(np.sum(arrays["t_eigenvectors"] ** 2, axis=0) < 1e-30):
        raise ValueError("Cannot build a rotation from a zero vector in t_eigenvectors.")
    return M, n_half


def _angle_batches(n_half, num_batches):
    r"""Split the Givens pairs into contiguous batches, in application order.

    Args:
        n_half (int): the number of spatial orbitals, ``N/2``
        num_batches (int): the number of batches to split the ``N/2 - 1`` pairs into

    Returns:
        tuple[list[list[int]], int]: the pairs of each batch, followed by the width of the
        widest batch. Pair ``p`` acts on the spatial orbitals ``(p, p + 1)`` and the ladder
        runs from ``N/2 - 2`` down to ``0``. Only the last batch may be short, so the width
        sizes the angle register and the number of batches returned can be smaller than
        ``num_batches``.
    """
    pairs = list(reversed(range(max(n_half - 1, 0))))
    width = max(-(-len(pairs) // num_batches), 1)
    return [pairs[i : i + width] for i in range(0, len(pairs), width)], width


def select_thc_wires(M, N, beth, num_batches=1):
    r"""Return the wire counts required by :class:`~.SelectTHC`.

    Args:
        M (int): the THC rank.
        N (int): the number of spin orbitals.
        beth (int): bits of precision per Givens angle. The realised grid is the odd
            multiples of :math:`\pi / 2^{\mathrm{beth}}`, so the spacing is
            :math:`2\pi / 2^{\mathrm{beth}}`.
        num_batches (int): the number of batches the Givens angles are loaded in. The
            default of ``1`` loads all of them at once. See the note below.

    Returns:
        dict: ``{"system_wires": n_system, "index_wires": n_index,
        "flag_wires": n_flag, "gradient_wires": n_gradient, "work_wires": n_work}``.

        * ``system_wires`` (``N``): the spin orbitals, the ``N/2`` spin-down spatial
          orbitals followed by the ``N/2`` spin-up ones.
        * ``index_wires`` (``2 * ceil(log2(M + 1))``): :math:`\mu` followed by
          :math:`\nu`, as produced by ``PREPARE``. The ``+ 1`` inside the logarithm
          leaves room for the one-body sentinel value :math:`\nu = M`.
        * ``flag_wires`` (``5``): the success flag, the one-body sentinel flag (:math:`\nu = M`),
          the qubit that controls the :math:`\mu \leftrightarrow \nu` swap,
          and the two spin flags.
        * ``gradient_wires`` (``beth + 1``): the phase gradient register. See the note below.
        * ``work_wires`` (``ceil((N/2 - 1) / num_batches) * beth +
          max(ceil(log2(M + 1)), beth)``): the minimum clean scratch, returned to
          :math:`\lvert 0 \rangle`. Zero when ``N/2 == 1``, where the sandwich is a lone
          :math:`Z_1` and no angles are loaded.

    Raises:
        ValueError: if ``M``, ``N``, ``beth`` or ``num_batches`` is not a positive
            integer, if ``N`` is odd, or if the one-body block does not fit the index
            register.

    .. note::

        Only ``work_wires`` is a minimum; the other four are exact and must be matched.
        Extra work wires are forwarded to the internal :class:`~.QROM`, which splits them
        between the unary iteration of its ``Select`` and the ``SelectSwap`` space-time
        trade-off. Which split it picks depends on ``M``, ``N`` and ``beth``, so extra
        wires do not necessarily lower the gate count.

    .. note::

        ``gradient_wires`` must be prepared by the caller in the phase gradient state

        .. math::

            \lvert \phi \rangle = \frac{1}{\sqrt{2^{\mathrm{beth}+1}}}
            \sum_{k=0}^{2^{\mathrm{beth}+1} - 1}
            e^{-2 \pi i k / 2^{\mathrm{beth}+1}} \lvert k \rangle ,

        which can be done with :class:`~.PhaseGradientStatePrep`.
        The ``SELECT`` oracle leaves it unchanged, so it is
        deliberately not allocated internally: one register is prepared once and shared by
        ``PREPARE`` and ``SELECT``.

    **Example**

    >>> import pennylane as qp
    >>> qp.select_thc_wires(M=3, N=4, beth=4)
    {'system_wires': 4, 'index_wires': 4, 'flag_wires': 5, 'gradient_wires': 5, 'work_wires': 8}

    """
    for name, value in (("M", M), ("N", N), ("beth", beth), ("num_batches", num_batches)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")

    if N % 2:
        raise ValueError(f"N must be an even number of spin orbitals; got {N}.")

    n = math.ceil_log2(M + 1)
    n_half = N // 2
    if n_half > 2**n:
        raise ValueError(
            f"The one-body rotations do not fit the index register: N // 2 = {n_half} "
            f"exceeds 2 ** ceil(log2(M + 1)) = {2**n} for M = {M}."
        )

    batches, width = _angle_batches(n_half, num_batches)
    n_angle = width * beth if batches else 0
    return {
        "system_wires": N,
        "index_wires": 2 * n,
        "flag_wires": 5,
        "gradient_wires": beth + 1,
        "work_wires": n_angle + max(n, beth) if n_angle else 0,
    }


class SelectTHC(Operator2):
    r"""Self-inverse Hamiltonian selection oracle for tensor hypercontraction (THC).

    Implements the ``SELECT`` of Fig. 5 of `Lee et al. (2021)
    <https://arxiv.org/abs/2011.03494>`_, with the self-inverse fix of its Eqs. (36)-(42)
    and Fig. 7. On the subspace flagged by the success wire this applies

    .. math::

        \lvert \mu \rangle \lvert \nu \rangle \lvert \psi \rangle \;\longmapsto\;
        \lvert \nu \rangle \lvert \mu \rangle \,
        V_\nu^{\beta} V_\mu^{\alpha} \lvert \psi \rangle ,
        \qquad V_\mu = U_\mu^\dagger Z_1 U_\mu = I - 2 c_\mu^\dagger c_\mu ,

    reducing to the one-body :math:`V_{T, \ell}^{\sigma}` when the one-body sentinel flag
    is set.

    The Hermitian operator that the walk operator block-encodes is given by:

    .. math::

        \langle P \rvert \mathrm{SELECT} \lvert P \rangle = \tfrac{1}{2}
        \left( V_\mu^{\alpha} V_\nu^{\beta} + V_\nu^{\alpha} V_\mu^{\beta} \right) ,

    up to the sign :math:`(-1)^{s_{\mu\nu}}` of the sampled coefficient, which ``PREPARE``
    applies as a controlled phase on its sign qubit.

    Use :func:`~.select_thc_wires` for the register sizes.

    Args:
        chi (tuple[tuple[float]]): the THC leaf matrix of shape ``(M, N/2)``, provided as
            a nested tuple. Each row is assumed normalized; only its direction enters the
            circuit, so any row norm must already be absorbed into :math:`\zeta_{\mu\nu}`
            in ``PREPARE``.
        t_eigenvectors (tuple[tuple[float]]): eigenvectors of the modified one-body matrix
            :math:`T'` as columns, shape ``(N/2, N/2)``, provided as a nested tuple
        beth (int): bits of precision per Givens angle
        system_wires (Sequence[int]): the ``N`` spin orbitals
        index_wires (Sequence[int]): ``2 * ceil(log2(M + 1))`` wires, :math:`\mu`
            followed by :math:`\nu`, as left by ``PREPARE``.
        flag_wires (Sequence[int]): this includes five wires, in order the success flag, the
            one-body flag (:math:`\nu = M`), the qubit that controls the :math:`\mu \leftrightarrow
            \nu` swap, and the two spin flags. All five are assumed to be prepared by
            ``PREPARE``.
        gradient_wires (Sequence[int]): the ``beth + 1`` wires holding the phase gradient state.
            This is assumed to be prepared on entry and left unchanged, as it is reused between
            ``PREPARE`` and ``SELECT`` oracles.
        work_wires (Sequence[int]): clean scratch, returned to :math:`\lvert 0 \rangle`
        num_batches (int): the number of batches the Givens angles are loaded in. The
            default of ``1`` loads all of them at once; larger values shrink
            ``work_wires`` at the cost of more ``QROM`` loads.

    Raises:
        ValueError: if two registers share a wire, or if a register has the wrong size
        ValueError: if ``chi`` or ``t_eigenvectors`` is not a nested tuple of finite,
            real numbers
        ValueError: if ``t_eigenvectors`` does not have shape ``(N/2, N/2)``
        ValueError: if ``beth`` or ``num_batches`` is not a positive integer
        ValueError: if the one-body rotations do not fit the index register, that is if
            ``N/2 > 2 ** ceil(log2(M + 1))``
        ValueError: if a row of ``chi`` or a column of ``t_eigenvectors`` is the zero
            vector, which has no rotation onto :math:`\lvert e_0 \rangle`


    **Example**

    .. code-block:: python

        import numpy as np
        import pennylane as qp
        M, N, beth = 3, 4, 3
        rng = np.random.default_rng(0)
        chi = tuple(map(tuple, rng.standard_normal((M, N // 2))))
        t_eigenvectors = tuple(
            map(tuple, np.linalg.qr(rng.standard_normal((N // 2, N // 2)))[0])
        )

        sizes = qp.select_thc_wires(M, N, beth)
        wires = qp.registers(sizes)
        n_total = sum(sizes.values())

        @qp.qnode(qp.device("default.qubit", wires=n_total))
        def circuit():
            qp.X(wires["flag_wires"][0])              # success flag
            for w in wires["flag_wires"][3:]:              # the two spin flags
                qp.Hadamard(w)
            qp.PhaseGradientStatePrep(wires["gradient_wires"])
            qp.SelectTHC(
                chi, t_eigenvectors, beth, wires["system_wires"], wires["index_wires"],
                wires["flag_wires"], wires["gradient_wires"], wires["work_wires"],
            )
            return qp.probs(wires=wires["system_wires"])

    .. details::
        :title: Available decompositions

        .. decomposition-table:: qp.SelectTHC
            :stacked:
            :symbols: :math:`N` spin orbitals, :math:`M` is the THC rank, :math:`h = N/2` spatial orbitals, :math:`P = h - 1` Givens pairs loaded in
                :math:`B` batches (:math:`B \leq` ``num_batches``, and :math:`B = 0` if
                :math:`P = 0`), and :math:`n = \lceil \log_2(M+1) \rceil` wires per index
            :example: :math:`M=1`, :math:`N=4` (:math:`h=2`, :math:`P=1`, :math:`B=1`, :math:`n=1`)

            .. decomposition-rule:: _select_thc_decomp
                :image: /_static/decomposition_thumbnails/select_thc_double_factorized.svg
                :lightbox:

                :class:`~.QROM` | :math:`2B` | 2
                :class:`Adjoint(QROM) <.QROM>` | :math:`2B` | 2
                :class:`~.CSWAP` | :math:`4h + n + 1` | 10
                :class:`~.Hadamard` | :math:`24P` | 24
                :class:`~.CNOT` | :math:`8P` | 8
                :class:`~.S` | :math:`8P` | 8
                :class:`Adjoint(S) <.S>` | :math:`8P` | 8
                :class:`~.Z` | :math:`8P` | 8
                :class:`~.MultiX` with 1 control | :math:`16P` | 16
                :class:`~.SemiAdder` | :math:`4P` | 4
                :class:`Adjoint(SemiAdder) <.SemiAdder>` | :math:`4P` | 4
                :class:`~.CZ` with 1 control | :math:`1` | 1
                :class:`~.CZ` with 2 controls | :math:`1` | 1
                :class:`~.X` | :math:`3` | 3

    """

    wire_argnames = (
        "system_wires",
        "index_wires",
        "flag_wires",
        "gradient_wires",
        "work_wires",
    )
    compilable_argnames = ("chi", "t_eigenvectors", "beth", "num_batches")
    arg_specs = {
        "system_wires": Wire[-1],
        "index_wires": Wire[-1],
        "flag_wires": Wire[5],
        "gradient_wires": Wire[-1],
        "work_wires": Wire[-1],
    }

    def __init__(
        self,
        chi,
        t_eigenvectors,
        beth,
        system_wires: WiresLike,
        index_wires: WiresLike,
        flag_wires: WiresLike,
        gradient_wires: WiresLike,
        work_wires: WiresLike,
        num_batches=1,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        M, n_half = _validate_select_data(chi, t_eigenvectors)

        requirements = select_thc_wires(M, 2 * n_half, beth, num_batches)
        registers = {
            "system_wires": Wires(system_wires),
            "index_wires": Wires(index_wires),
            "flag_wires": Wires(flag_wires),
            "gradient_wires": Wires(gradient_wires),
            "work_wires": Wires(work_wires),
        }
        for name in ("system_wires", "index_wires", "flag_wires", "gradient_wires"):
            if len(registers[name]) != requirements[name]:
                raise ValueError(
                    f"{name} must have {requirements[name]} entries; got {len(registers[name])}."
                )
        if len(registers["work_wires"]) < requirements["work_wires"]:
            raise ValueError(
                f"work_wires must have at least {requirements['work_wires']} entries; "
                f"got {len(registers['work_wires'])}."
            )
        validate_no_wire_overlaps(registers)

        super().__init__(
            chi,
            t_eigenvectors,
            beth,
            *registers.values(),
            num_batches,
        )

    @property
    def wires(self):
        """All wires involved in the operation."""
        return (
            self.system_wires
            + self.index_wires
            + self.flag_wires
            + self.gradient_wires
            + self.work_wires
        )
