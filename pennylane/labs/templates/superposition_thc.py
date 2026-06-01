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
"""Contains the SuperpositionTHC template, used as a subroutine in tensor
hypercontraction (THC) qubitization."""

from pennylane import adjoint, cond, math
from pennylane.decomposition import (
    add_decomps,
    register_resources,
)
from pennylane.operation import Operation
from pennylane.ops import CNOT, BasisState, Controlled, Hadamard, MultiControlledX, RY, X, Z
from pennylane.queuing import AnnotatedQueue, QueuingManager, apply
from pennylane.templates.subroutines import Elbow
from pennylane.wires import Wires, WiresLike
from pennylane.labs.templates import LeftClassicalComparator, LeftQuantumComparator


class SuperpositionTHC(Operation):
    r"""Prepare the uniform superposition over the valid :math:`(\mu, \nu)` index
    pairs of the tensor hypercontraction (THC) representation.

    This template prepares the state used by the ``SELECT``/``PREPARE`` pair in THC
    qubitization. Starting from two zeroed index registers, :math:`\lvert \mu \rangle`
    and :math:`\lvert \nu \rangle` (plus a flag/auxiliary register), it produces a
    uniform superposition restricted to the set of indices that label the symmetry-unique
    THC factors, namely the one-body terms and the upper-triangular two-body pairs
    :math:`\mu \geq \nu`:

    .. math::

        \lvert 0 \rangle^{\otimes n} \lvert 0 \rangle^{\otimes n} \lvert 0 \rangle \;\mapsto\;
        \frac{1}{\sqrt{d}} \sum_{(\mu, \nu) \in \mathcal{S}} \lvert \mu \rangle \lvert \nu \rangle \lvert \text{flag} \rangle ,

    where :math:`\mathcal{S}` is the valid index set and :math:`d = N/2 + M(M+1)` is its size.
    Because :math:`d` is generally not a power of two, the uniform superposition is obtained
    by combining Hadamards with a single round of amplitude amplification: an :class:`~.RY`
    rotation marks the success amplitude, a reflection amplifies it, and the rotation is
    uncomputed.

    The construction follows the tensor hypercontraction state preparation of
    `Lee et al. (2021), Fig. 3 <https://arxiv.org/abs/2011.03494>`_, and the inequality
    tests reuse the comparator subroutines from
    `Su et al. (2021), Appendix E <https://arxiv.org/abs/2105.12767>`_
    (:class:`~.LeftClassicalComparator` and :class:`~.LeftQuantumComparator`).

    .. note::

        The decomposition is self-contained: every work wire is returned to the zero state
        except the wires that carry the prepared flags, so no external uncomputation is
        required by the caller.

    Args:
        M (int): The THC rank, i.e. the number of auxiliary THC vectors. Together with ``N``
            it determines the size :math:`d = N/2 + M(M+1)` of the prepared superposition.
        N (int): The number of spin orbitals. Used to count the one-body contribution
            :math:`N/2` to the valid index set.
        mu_wires (WiresLike): The :math:`n` wires that store the first THC index :math:`\mu`.
        nu_wires (WiresLike): The :math:`n` wires that store the second THC index :math:`\nu`.
            Must contain the same number of wires as ``mu_wires``.
        work_wires (WiresLike): The auxiliary wires. The first seven encode the flag/ancilla
            register of Fig. 3 of `Lee et al. (2021) <https://arxiv.org/abs/2011.03494>`_;
            the remaining wires are scratch space for the comparators and multi-controlled
            gates. At least :math:`3\,n + 5` zeroed work wires must be provided, where
            :math:`n = \text{len(mu\_wires)}`.

    **Example**

    The template prepares the THC index superposition on the ``mu_wires`` / ``nu_wires``
    registers. Here ``n = 3``, so the minimum number of work wires is :math:`3n + 5 = 14`.

    .. code-block:: python

        import pennylane as qml
        from pennylane.labs.templates import SuperpositionTHC

        n = 3
        M, N = 2, 4
        mu_wires = list(range(0, n))
        nu_wires = list(range(n, 2 * n))
        work_wires = list(range(2 * n, 2 * n + 3 * n + 5))

        dev = qml.device("lightning.qubit", wires=2 * n + 3 * n + 5)

        @qml.qnode(dev)
        def circuit():
            SuperpositionTHC(M, N, mu_wires, nu_wires, work_wires)
            return qml.state()

    .. code-block:: pycon

        >>> op = SuperpositionTHC(M, N, mu_wires, nu_wires, work_wires)
        >>> len(op.decomposition())
        60
    """

    grad_method = None

    resource_keys = {"num_mu_wires"}

    def __init__(
        self,
        M: int,
        N: int,
        mu_wires: WiresLike,
        nu_wires: WiresLike,
        work_wires: WiresLike,
    ):  # pylint: disable=too-many-arguments

        mu_wires = Wires(mu_wires)
        nu_wires = Wires(nu_wires)
        work_wires = Wires(work_wires)

        if len(mu_wires) != len(nu_wires):
            raise ValueError(
                f"mu_wires and nu_wires must contain the same number of wires, but got "
                f"{len(mu_wires)} and {len(nu_wires)}."
            )

        min_work_wires = 3 * len(mu_wires) + 5
        if len(work_wires) < min_work_wires:
            raise ValueError(
                f"At least {min_work_wires} work_wires (3 * len(mu_wires) + 5) should be "
                f"provided, but only {len(work_wires)} were given."
            )

        for name, register in (("mu_wires", mu_wires), ("nu_wires", nu_wires)):
            overlap = work_wires.intersection(register)
            if overlap:
                raise ValueError(
                    f"work_wires and {name} must be disjoint, but share: {list(overlap)}."
                )
        overlap = mu_wires.intersection(nu_wires)
        if overlap:
            raise ValueError(
                f"mu_wires and nu_wires must be disjoint, but share: {list(overlap)}."
            )

        self.hyperparameters["M"] = M
        self.hyperparameters["N"] = N
        self.hyperparameters["mu_wires"] = mu_wires
        self.hyperparameters["nu_wires"] = nu_wires
        self.hyperparameters["work_wires"] = work_wires

        all_wires = Wires.all_wires([mu_wires, nu_wires, work_wires])
        super().__init__(wires=all_wires)

    @property
    def resource_params(self) -> dict:
        return {
            "num_mu_wires": len(self.hyperparameters["mu_wires"]),
        }

    def _flatten(self):
        metadata = tuple((key, value) for key, value in self.hyperparameters.items())
        return tuple(), metadata

    @classmethod
    def _unflatten(cls, data, metadata):
        hyperparams_dict = dict(metadata)
        return cls(*data, **hyperparams_dict)

    def map_wires(self, wire_map: dict) -> "SuperpositionTHC":
        new_dict = {
            key: [wire_map.get(w, w) for w in self.hyperparameters[key]]
            for key in ["mu_wires", "nu_wires", "work_wires"]
        }

        return SuperpositionTHC(
            **new_dict, M=self.hyperparameters["M"], N=self.hyperparameters["N"]
        )

    def decomposition(self):
        r"""Representation of the operator as a product of other operators."""
        return self.compute_decomposition(**self.hyperparameters)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @staticmethod
    def compute_decomposition(
        M, N, mu_wires, nu_wires, work_wires
    ):  # pylint: disable=arguments-differ, too-many-arguments
        r"""Representation of the operator as a product of other operators.

        Args:
            M (int): The THC rank (number of auxiliary THC vectors).
            N (int): The number of spin orbitals.
            mu_wires (WiresLike): The wires that store the first THC index :math:`\mu`.
            nu_wires (WiresLike): The wires that store the second THC index :math:`\nu`.
            work_wires (WiresLike): The auxiliary wires. At least
                :math:`3\,\text{len(mu\_wires)} + 5` zeroed work wires should be provided.

        Returns:
            list[.Operator]: Decomposition of the operator
        """

        with AnnotatedQueue() as q:
            _superposition_thc(M, N, mu_wires, nu_wires, work_wires)

        if QueuingManager.recording():
            for o in q.queue:
                apply(o)

        return q.queue


def left_equalities(M, N, mu_wires, nu_wires, work_wires, keep_eq=False):
    r"""Apply the inequality tests that flag a valid THC index pair.

    Computes the for comparisons that define the valid index set onto dedicated flag
    wires of the ancilla register (Fig. 3 of `Lee et al. (2021)
    <https://arxiv.org/abs/2011.03494>`_):

    * ``work_wires[1]``: :math:`\nu \leq M` (classical comparison against the THC rank).
    * ``work_wires[2]``: :math:`\mu \leq \nu` (quantum comparison between the two registers).
    * ``work_wires[3]``: :math:`\nu = M + 1` (classical equality against the THC rank).
    * ``work_wires[4]``: :math:`\mu > N/2` (classical comparison selecting two-body terms).

    The auxiliary wires used on each comparator are drawn from disjoint slices of ``work_wires``
    starting at index ``7``.

    Args:
        M (int): The THC rank.
        N (int): The number of spin orbitals.
        mu_wires (WiresLike): The wires storing the first THC index :math:`\mu`.
        nu_wires (WiresLike): The wires storing the second THC index :math:`\nu`.
        work_wires (WiresLike): The auxiliary wires.
        keep_eq (bool): only if ``True``, ``work_wires[3]`` is calculated.
    """
    n = len(mu_wires)

    LeftClassicalComparator(
        nu_wires,
        M,
        target_wire=work_wires[1],
        work_wires=work_wires[7 : 7 + n - 1],
        comparator="<=",
    )
    LeftQuantumComparator(
        mu_wires,
        nu_wires,
        target_wire=work_wires[2],
        work_wires=work_wires[7 + n - 1 : 7 + 2 * n - 1],
        comparator="<=",
    )
    LeftClassicalComparator(
        mu_wires,
        N // 2,
        target_wire=work_wires[4],
        work_wires=work_wires[7 + 2 * n - 1 : 7 + 3 * n - 1],
        comparator=">",
    )

    BasisState(M, wires=nu_wires)

    # TODO: replace this zero-controlled MultiControlledX with MultiTemporaryAND.
    cond(keep_eq, MultiControlledX)(
        wires=nu_wires + [work_wires[3]],
        control_values=[0] * len(nu_wires),
        work_wires=work_wires[7 + 3 * n - 1 : 7 + 4 * n - 1],
    )


def _superposition_thc_resources(num_mu_wires):

    # TODO: update this resource estimate to match the full decomposition.
    resources = {
        Elbow: num_mu_wires,
        CNOT: 2 + 5 * (num_mu_wires - 1),
    }

    return resources


@register_resources(_superposition_thc_resources, exact=True)
def _superposition_thc(M, N, mu_wires, nu_wires, work_wires, **_):
    # pylint: disable=too-many-arguments
    #
    # The first seven work_wires correspond to the flag/ancilla register in Fig. 3 of
    # https://arxiv.org/pdf/2011.03494. After the routine, all work wires are returned to
    # the zero state except work_wires[0], work_wires[3] and work_wires[6], which carry
    # the prepared flags.

    n = len(mu_wires)
    extra_work = work_wires[7 + 4 * n - 1 :]

    # 1. Equal superposition over both index registers.
    for wire in mu_wires:
        Hadamard(wire)
    for wire in nu_wires:
        Hadamard(wire)

    # 2. Rotation angle for the single round of amplitude amplification.
    #    The valid index set has size d = N/2 + M(M+1); frac_valid is its fraction of the
    #    full 2^(2n) basis, and the angle marks the corresponding success amplitude.
    n_total_vals = 2 ** len(mu_wires)
    d = N // 2 + M * (M + 1)
    frac_valid = d / n_total_vals**2
    limit = 0.5 / math.sqrt(frac_valid)
    cos_val = math.where(limit < 1.0, limit, 1.0)
    angle = 2 * math.arccos(cos_val)

    RY(angle, wires=work_wires[0])
    X(wires=work_wires[5])

    # 3. Flag the valid index pairs, then mark the "success" subspace with a phase.
    left_equalities(M, N, mu_wires, nu_wires, work_wires)

    Controlled(X(work_wires[5]), control_wires=work_wires[3:5], work_wires=extra_work)
    Controlled(Z(work_wires[5]), control_wires=work_wires[0:3], work_wires=extra_work)
    Controlled(X(work_wires[5]), control_wires=work_wires[3:5], work_wires=extra_work)

    # 4. Uncompute the flags and the amplitude-marking rotation.
    adjoint(left_equalities)(M, N, mu_wires, nu_wires, work_wires)
    RY(-angle, wires=work_wires[0])

    # 5. Reflection about the equal-superposition state (the amplification step).
    for wire in mu_wires:
        Hadamard(wire)
    for wire in nu_wires:
        Hadamard(wire)

    for wire in mu_wires + nu_wires + [work_wires[0]]:
        X(wires=wire)
    Controlled(Z(work_wires[0]), control_wires=mu_wires + nu_wires, work_wires=extra_work)
    for wire in mu_wires + nu_wires + [work_wires[0]]:
        X(wires=wire)

    for wire in mu_wires:
        Hadamard(wire)
    for wire in nu_wires:
        Hadamard(wire)

    # 6. Recompute the flags onto the output ancilla register (work_wires[5], work_wires[6]).
    left_equalities(M, N, mu_wires, nu_wires, work_wires)

    Controlled(X(work_wires[5]), control_wires=work_wires[3:5], work_wires=extra_work)
    Controlled(
        X(work_wires[6]), control_wires=work_wires[1:3] + work_wires[5], work_wires=extra_work
    )
    Controlled(X(work_wires[5]), control_wires=work_wires[3:5], work_wires=extra_work)

    X(wires=work_wires[5])

    # 7. Final uncomputation, keeping the diagonal (mu = nu) equality flag.
    adjoint(left_equalities)(M, N, mu_wires, nu_wires, work_wires, keep_eq=True)


add_decomps(SuperpositionTHC, _superposition_thc)
