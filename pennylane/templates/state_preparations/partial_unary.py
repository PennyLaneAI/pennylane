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
r"""Contains the PartialUnaryStatePreparation template."""

from collections import defaultdict
from itertools import combinations, product

import numpy as np

import pennylane as qp
from pennylane import allocate, for_loop, math
from pennylane.decomposition import (
    add_decomps,
    adjoint_resource_rep,
    register_resources,
    resource_rep,
)
from pennylane.exceptions import DecompositionUndefinedError
from pennylane.operation import Operation


def largest_power_of_two_leq(x: int) -> int:
    """Return the largest power of 2 less than or equal to x."""
    return 1 << qp.math.floor_log2(x)


class PartialUnaryStatePreparation(Operation):
    r"""Prepare an arbitrary quantum state with the partial unary iteration technique.

    This operation prepares an arbitrary state

    .. math:: |\psi\rangle = \sum_{\ell \in L } c_\ell |\ell\rangle,

    where :math:`L` denotes the set of ``indices`` and :math:`c_\ell` is the ``coefficient``
    corresponding to the index :math:`\ell\in L`.
    The state :math:`|\ell\rangle` is a computational basis state, interpreted via the
    binary representation of :math:`\ell`.

    This state preparation technique was introduced in
    `Rupprecht and Wölk, arXiv:2601.09388 <https://arxiv.org/abs/2601.09388>`__.

    Args:
        coefficients (np.ndarray): Coefficients of the sparse state to prepare. The ordering should
            match that in ``indices``.
        wires (qp.wires.WiresLike): Wires on which to prepare the state. All work wires will be
            allocated dynamically with :func:`~.allocate`.
        indices (tuple[int]): Indices of the sparse state to prepare. The ordering should match
            that in ``coefficients``.

    .. warning::

        Note that we require ``coefficients`` to be treated as numerical data in the form of an
        array, whereas the ``indices`` need to be hashable, and thus will be treated as static
        information. This is because ``indices`` significantly impacts the structure and size of
        the circuit that realizes the state preparation.

    **Example**

    #TODO

    """

    resource_keys = {"num_entries", "num_bits", "num_wires"}  # TODO

    @property
    def resource_params(self):
        indices = self.hyperparameters["indices"]
        work_wires = self.hyperparameters["work_wires"]
        n = len(self.wires)
        v_bits = math.int_to_binary(np.array(indices), n).T
        # Process v_bits# TODO
        return {
            "num_entries": len(indices),
            "num_bits": len(selector_ids),
            "num_wires": n,
            "num_work_wires": len(work_wires),
        }  # TODO

    def __init__(self, coefficients, wires, indices, work_wires=None):
        all_wires = Wires.all_wires([wires, work_wires])
        super().__init__(coefficients, wires=all_wires)
        self.hyperparameters["indices"] = indices
        self.hyperparameters["work_wires"] = work_wires

    @property
    def has_decomposition(self):
        """We are using ``qp.allocate`` in the decomposition, so the validation for
        decomposition in the old system breaks. Hence we manually deactivate the fallback
        of ``compute_decomposition`` to the new decomp system that is implemented in
        ``Operator.compute_decomposition``. Accordingly we set ``has_decomposition=False`` here."""
        return False

    @staticmethod
    def compute_decomposition(*_, **__):  # pylint: disable=arguments-differ
        """We are using ``qp.allocate`` in the decomposition, so the validation for
        decomposition in the old system breaks. Hence we manually deactivate the fallback
        of ``compute_decomposition`` to the new decomp system that is implemented in
        ``Operator.compute_decomposition``."""
        raise DecompositionUndefinedError


def _pui_state_prep_resources(num_entries, num_bits, num_wires, num_work_wires):
    """Compute the resources for _pui_state_prep."""
    if num_entries == 1:
        return {resource_rep(qp.BasisState, num_wires=num_wires): 1}

    resources = defaultdict(int)

    return resources


@register_resources(_pui_state_prep_resources)
def _pui_state_prep(coefficients, wires, indices, work_wires, **__):
    """Compute the decomposition of the partial unary iteration state preparation technique."""


add_decomps(SumOfSlatersPrep, _pui_state_prep)
