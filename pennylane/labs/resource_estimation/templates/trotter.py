# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Contains templates for Suzuki-Trotter approximation based subroutines.
"""
from collections import defaultdict
from functools import wraps
from typing import Dict

import pennylane as qml
from pennylane.labs import resource_estimation as re
from pennylane.labs.resource_estimation import (
    CompressedResourceOp,
    ResourceExp,
    ResourceOperator,
    ResourcesNotDefined,
)
from pennylane.templates import TrotterProduct
from pennylane.templates.subroutines.trotter import TrotterizedQfunc

# pylint: disable=arguments-differ


class ResourceTrotterProduct(
    TrotterProduct, ResourceOperator
):  # pylint: disable=too-many-ancestors
    r"""An operation representing the Suzuki-Trotter product approximation for the complex matrix
    exponential of a given Hamiltonian.

    The Suzuki-Trotter product formula provides a method to approximate the matrix exponential of
    Hamiltonian expressed as a linear combination of terms which in general do not commute. Consider
    the Hamiltonian :math:`H = \Sigma^{N}_{j=0} O_{j}`, the product formula is constructed using
    symmetrized products of the terms in the Hamiltonian. The symmetrized products of order
    :math:`m \in [1, 2, 4, ..., 2k]` with :math:`k \in \mathbb{N}` are given by:

    .. math::

        \begin{align}
            S_{1}(t) &= \Pi_{j=0}^{N} \ e^{i t O_{j}} \\
            S_{2}(t) &= \Pi_{j=0}^{N} \ e^{i \frac{t}{2} O_{j}} \cdot \Pi_{j=N}^{0} \ e^{i \frac{t}{2} O_{j}} \\
            &\vdots \\
            S_{m}(t) &= S_{m-2}(p_{m}t)^{2} \cdot S_{m-2}((1-4p_{m})t) \cdot S_{m-2}(p_{m}t)^{2},
        \end{align}

    where the coefficient is :math:`p_{m} = 1 / (4 - \sqrt[m - 1]{4})`. The :math:`m`th order,
    :math:`n`-step Suzuki-Trotter approximation is then defined as:

    .. math:: e^{iHt} \approx \left [S_{m}(t / n)  \right ]^{n}.

    Resources:
        The resources are defined according to the recurrsive formula presented above.
    """

    @staticmethod
    def _resource_decomp(
        n, order, first_order_expansion, **kwargs
    ) -> Dict[CompressedResourceOp, int]:
        k = order // 2
        gate_types = defaultdict(int, {})

        if order == 1:
            for cp_rep in first_order_expansion:
                gate_types[cp_rep] += n
            return gate_types

        cp_rep_first = first_order_expansion[0]
        cp_rep_last = first_order_expansion[-1]
        cp_rep_rest = first_order_expansion[1:-1]

        for cp_rep in cp_rep_rest:
            gate_types[cp_rep] += 2 * n * (5 ** (k - 1))

        gate_types[cp_rep_first] += n * (5 ** (k - 1)) + 1
        gate_types[cp_rep_last] += n * (5 ** (k - 1))

        return gate_types

    @property
    def resource_params(self) -> dict:
        n = self.hyperparameters["n"]
        base = self.hyperparameters["base"]
        order = self.hyperparameters["order"]

        first_order_expansion = [
            ResourceExp.resource_rep(
                **re.ops.op_math.symbolic._extract_exp_params(  # pylint: disable=protected-access
                    op, scalar=1j, num_steps=1
                )
            )
            for op in base.operands
        ]

        return {
            "n": n,
            "order": order,
            "first_order_expansion": first_order_expansion,
        }

    @classmethod
    def resource_rep(cls, n, order, first_order_expansion) -> CompressedResourceOp:
        params = {
            "n": n,
            "order": order,
            "first_order_expansion": first_order_expansion,
        }
        return CompressedResourceOp(cls, params)

    @classmethod
    def resources(cls, *args, **kwargs) -> Dict[CompressedResourceOp, int]:
        """Returns a dictionary containing the counts of each operator type used to
        compute the resources of the operator."""
        return cls._resource_decomp(*args, **kwargs)


class ResourceTrotterizedQfunc(TrotterizedQfunc, ResourceOperator):
    """An internal class which facilitates :code:`qml.resource_trotterize`."""

    @staticmethod
    def _resource_decomp(
        n, order, qfunc_compressed_reps, **kwargs
    ) -> Dict[CompressedResourceOp, int]:
        k = order // 2
        gate_types = defaultdict(int, {})

        if order == 1:
            for cp_rep in qfunc_compressed_reps:
                gate_types[cp_rep] += n
            return gate_types

        for cp_rep in qfunc_compressed_reps:
            gate_types[cp_rep] += 2 * n * (5 ** (k - 1))
        return gate_types

    @property
    def resource_params(self) -> dict:
        with qml.QueuingManager.stop_recording():
            with qml.queuing.AnnotatedQueue() as q:
                base_hyper_params = ("n", "order", "qfunc", "reverse")

                qfunc_args = self.parameters
                qfunc_kwargs = {
                    k: v for k, v in self.hyperparameters.items() if not k in base_hyper_params
                }

                qfunc = self.hyperparameters["qfunc"]
                qfunc(*qfunc_args, wires=self.wires, **qfunc_kwargs)

        try:
            qfunc_compressed_reps = tuple(op.resource_rep_from_op() for op in q.queue)

        except AttributeError as error:
            raise ResourcesNotDefined(
                "Every operation in the TrotterizedQfunc should be a ResourceOperator"
            ) from error

        return {
            "n": self.hyperparameters["n"],
            "order": self.hyperparameters["order"],
            "qfunc_compressed_reps": qfunc_compressed_reps,
        }

    @classmethod
    def resource_rep(cls, n, order, qfunc_compressed_reps, name=None) -> CompressedResourceOp:
        params = {
            "n": n,
            "order": order,
            "qfunc_compressed_reps": qfunc_compressed_reps,
        }
        return CompressedResourceOp(cls, params, name=name)

    def resource_rep_from_op(self) -> CompressedResourceOp:
        """Returns a compressed representation directly from the operator"""
        return self.__class__.resource_rep(**self.resource_params, name=self._name)


def resource_trotterize(qfunc, n=1, order=2, reverse=False):
    r"""Generates higher order Suzuki-Trotter product formulas from a set of
    operations defined in a function.

    The Suzuki-Trotter product formula provides a method to approximate the matrix exponential of
    Hamiltonian expressed as a linear combination of terms which in general do not commute. Consider
    the Hamiltonian :math:`H = \Sigma^{N}_{j=0} O_{j}`, the product formula is constructed using
    symmetrized products of the terms in the Hamiltonian. The symmetrized products of order
    :math:`m \in [1, 2, 4, ..., 2k]` with :math:`k \in \mathbb{N}` are given by:

    .. math::

        \begin{align}
            S_{1}(t) &= \Pi_{j=0}^{N} \ e^{i t O_{j}} \\
            S_{2}(t) &= \Pi_{j=0}^{N} \ e^{i \frac{t}{2} O_{j}} \cdot \Pi_{j=N}^{0} \ e^{i \frac{t}{2} O_{j}} \\
            &\vdots \\
            S_{m}(t) &= S_{m-2}(p_{m}t)^{2} \cdot S_{m-2}((1-4p_{m})t) \cdot S_{m-2}(p_{m}t)^{2},
        \end{align}

    where the coefficient is :math:`p_{m} = 1 / (4 - \sqrt[m - 1]{4})`. The :math:`m`th order,
    :math:`n`-step Suzuki-Trotter approximation is then defined as:

    .. math:: e^{iHt} \approx \left [S_{m}(t / n)  \right ]^{n}.

    Resources:
        The resources are defined according to the recurrsive formula presented above.
    """

    @wraps(qfunc)
    def wrapper(*args, **kwargs):
        time = args[0]
        other_args = args[1:]
        return ResourceTrotterizedQfunc(
            time, *other_args, qfunc=qfunc, n=n, order=order, reverse=reverse, **kwargs
        )

    return wrapper
