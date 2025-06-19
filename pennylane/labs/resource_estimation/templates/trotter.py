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
from pennylane.labs import resource_estimation as plre
from pennylane.labs.resource_estimation import ResourcesNotDefined
from pennylane.labs.resource_estimation.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)
from pennylane.queuing import QueuingManager
from pennylane.wires import Wires

# pylint: disable=arguments-differ


class ResourceTrotterProduct(ResourceOperator):  # pylint: disable=too-many-ancestors
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

    For more details see `J. Math. Phys. 32, 400 (1991) <https://pubs.aip.org/aip/jmp/article-abstract/32/2/400/229229>`_.

    Args:
        hamiltonian (Union[.Hamiltonian, .Sum, .SProd]): The Hamiltonian written as a linear combination
            of operators with known matrix exponentials.
        time (float): The time of evolution, namely the parameter :math:`t` in :math:`e^{iHt}`
        n (int): An integer representing the number of Trotter steps to perform
        order (int): An integer (:math:`m`) representing the order of the approximation (must be 1 or even)
        check_hermitian (bool): A flag to enable the validation check to ensure this is a valid unitary operator

    Resource Parameters:
        * n (int): an integer representing the number of Trotter steps to perform
        * order (int): an integer (:math:`m`) representing the order of the approximation (must be 1 or even)
        * first_order_expansion (list[CompressedResourceOp]): A list of compressed operations corresponding to the exponentiated terms of the hamiltonian (:math:`e^{i t O_{j}}`).

    Resources:
        The resources are defined according to the recursive formula presented above. Specifically, each
        operator in the :code:`first_order_expansion` is called a number of times given by the formula:

        .. math:: C_{O_{j}} = 2n \cdot 5^{\frac{m}{2} - 1}

        Furthermore, the first and last terms of the Hamiltonian appear in pairs due to the symmetric form
        of the recursive formula. Those counts are further simplified by grouping like terms as:

        .. math::

            \begin{align}
                C_{O_{0}} &= n \cdot 5^{\frac{m}{2} - 1} + 1,  \\
                C_{O_{N}} &= n \cdot 5^{\frac{m}{2} - 1}.
            \end{align}

    .. seealso:: :class:`~.TrotterProduct`

    **Example**

    The arguments can be provided directly to the :code:`resources()` function to extract the cost:

    >>> n, order = (1, 2)
    >>> first_order_expansion = [re.ResourceRX.resource_rep(), re.ResourceRZ.resource_rep()]
    >>> re.ResourceTrotterProduct.resources(n, order, first_order_expansion)
    defaultdict(<class 'int'>, {RX: 2, RZ: 1})

    """

    resource_keys = {"cmpr_fragments", "num_steps", "order"}

    def __init__(self, fragments, num_steps, order, wires=None):
        self.queue(fragments)

        self.num_steps = num_steps
        self.order = order
        self.cmpr_fragments = tuple(op.resource_rep_from_op() for op in fragments)

        if wires is not None:
            self.wires = Wires(wires)
            self.num_wires = len(self.wires)
        else:
            ops_wires = [op.wires for op in fragments if op.wires is not None]
            if len(ops_wires) == 0:
                self.wires = None
                self.num_wires = max((op.num_wires for op in fragments))
            else:
                self.wires = Wires.all_wires(ops_wires)
                self.num_wires = len(self.wires)

    def queue(self, remove_fragments=None, context: QueuingManager = QueuingManager):
        """Append the operator to the Operator queue."""
        if remove_fragments:
            for op in remove_fragments:
                context.remove(op)
        context.append(self)
        return self

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * n (int): an integer representing the number of Trotter steps to perform
                * order (int): an integer (:math:`m`) representing the order of the approximation (must be 1 or even)
                * first_order_expansion (list[CompressedResourceOp]): A list of compressed operations corresponding to the exponentiated terms of the hamiltonian (:math:`e^{i t O_{j}}`).
        """
        return {
            "num_steps": self.num_steps,
            "order": self.order,
            "cmpr_fragments": self.cmpr_fragments,
        }

    @classmethod
    def resource_rep(cls, num_steps, order, cmpr_fragments) -> CompressedResourceOp:
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            n (int): an integer representing the number of Trotter steps to perform
            order (int): an integer (:math:`m`) representing the order of the approximation (must be 1 or even)
            first_order_expansion (list[CompressedResourceOp]): A list of compressed operations corresponding to the exponentiated terms of the hamiltonian (:math:`e^{i t O_{j}}`).

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {
            "num_steps": num_steps,
            "order": order,
            "cmpr_fragments": cmpr_fragments,
        }
        return CompressedResourceOp(cls, params)

    @classmethod
    def default_resource_decomp(
        cls, num_steps, order, cmpr_fragments, **kwargs
    ) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

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

        For more details see `J. Math. Phys. 32, 400 (1991) <https://pubs.aip.org/aip/jmp/article-abstract/32/2/400/229229>`_.

        Args:
            n (int): an integer representing the number of Trotter steps to perform
            order (int): an integer (:math:`m`) representing the order of the approximation (must be 1 or even)
            first_order_expansion (list[CompressedResourceOp]): A list of compressed operations corresponding to the exponentiated terms of the hamiltonian (:math:`e^{i t O_{j}}`).

        Resources:
            The resources are defined according to the recurrsive formula presented above. Specifically, each
            operator in the :code:`first_order_expansion` is called a number of times given by the formula:

            .. math:: C_{O_{j}} = 2n \cdot 5^{\frac{m}{2} - 1}

            Furthermore, the first and last terms of the hamiltonian appear in pairs due to the symmetric form
            of the recurrsive formula. Those counts are further simplified by grouping like terms as:

            .. math::

                \begin{align}
                    C_{O_{0}} &= n \cdot 5^{\frac{m}{2} - 1} + 1,  \\
                    C_{O_{N}} &= n \cdot 5^{\frac{m}{2} - 1}.
                \end{align}

        **Example**

        The arguments can be provided directly to the :code:`resources()` function to extract the cost:

        >>> n, order = (1, 2)
        >>> first_order_expansion = [re.ResourceRX.resource_rep(), re.ResourceRZ.resource_rep()]
        >>> re.ResourceTrotterProduct.resources(n, order, first_order_expansion)
        defaultdict(<class 'int'>, {RX: 2, RZ: 1})

        """
        k = order // 2
        gate_list = []

        if order == 1:
            for cp_rep in cmpr_fragments:
                gate_list.append(GateCount(cp_rep, num_steps))
            return gate_list

        cp_rep_first = cmpr_fragments[0]
        cp_rep_last = cmpr_fragments[-1]
        cp_rep_rest = cmpr_fragments[1:-1]

        for cp_rep in cp_rep_rest:
            gate_list.append(GateCount(cp_rep, 2 * num_steps * (5 ** (k - 1))))

        gate_list.append(GateCount(cp_rep_first, num_steps * (5 ** (k - 1)) + 1))
        gate_list.append(GateCount(cp_rep_last, num_steps * (5 ** (k - 1))))

        return gate_list


# class ResourceTrotterizedQfunc(TrotterizedQfunc, ResourceOperator):
#     r"""Generates higher order Suzuki-Trotter product formulas from a set of
#     operations defined in a function.

#     The Suzuki-Trotter product formula provides a method to approximate the matrix exponential of
#     Hamiltonian expressed as a linear combination of terms which in general do not commute. Consider
#     the Hamiltonian :math:`H = \Sigma^{N}_{j=0} O_{j}`, the product formula is constructed using
#     symmetrized products of the terms in the Hamiltonian. The symmetrized products of order
#     :math:`m \in [1, 2, 4, ..., 2k]` with :math:`k \in \mathbb{N}` are given by:

#     .. math::

#         \begin{align}
#             S_{1}(t) &= \Pi_{j=0}^{N} \ e^{i t O_{j}} \\
#             S_{2}(t) &= \Pi_{j=0}^{N} \ e^{i \frac{t}{2} O_{j}} \cdot \Pi_{j=N}^{0} \ e^{i \frac{t}{2} O_{j}} \\
#             &\vdots \\
#             S_{m}(t) &= S_{m-2}(p_{m}t)^{2} \cdot S_{m-2}((1-4p_{m})t) \cdot S_{m-2}(p_{m}t)^{2},
#         \end{align}

#     where the coefficient is :math:`p_{m} = 1 / (4 - \sqrt[m - 1]{4})`. The :math:`m`th order,
#     :math:`n`-step Suzuki-Trotter approximation is then defined as:

#     .. math:: e^{iHt} \approx \left [S_{m}(t / n)  \right ]^{n}.

#     For more details see `J. Math. Phys. 32, 400 (1991) <https://pubs.aip.org/aip/jmp/article-abstract/32/2/400/229229>`_.

#     Args:
#         time (float): the time of evolution, namely the parameter :math:`t` in :math:`e^{iHt}`
#         *trainable_args (tuple): the trainable arguments of the first-order expansion function
#         qfunc (Callable): the first-order expansion given as a callable function which queues operations
#         wires (Iterable): the set of wires the operation will act upon (should be identical to qfunc wires)
#         n (int): an integer representing the number of Trotter steps to perform
#         order (int): an integer (:math:`m`) representing the order of the approximation (must be 1 or even)
#         reverse (bool): if true, reverse the order of the operations queued by :code:`qfunc`
#         **non_trainable_kwargs (dict): non-trainable keyword arguments of the first-order expansion function

#     Resource Parameters:
#         * n (int): an integer representing the number of Trotter steps to perform
#         * order (int): an integer (:math:`m`) representing the order of the approximation (must be 1 or even)
#         * qfunc_compressed_reps (list[CompressedResourceOp]): A list of compressed operations corresponding to the exponentiated terms of the hamiltonian (:math:`e^{i t O_{j}}`).

#     Resources:
#         The resources are defined according to the recurrsive formula presented above. Specifically, each
#         operator in the :code:`first_order_expansion` is called a number of times given by the formula:

#         .. math:: C_{O_{j}} = 2n \cdot 5^{\frac{m}{2} - 1}

#     .. seealso:: :class:`~.TrotterizedQfunc`

#     **Example**

#     The arguments can be provided directly to the :code:`resources()` function to extract the cost:

#         >>> n, order = (1, 2)
#         >>> first_order_expansion = [re.ResourceRX.resource_rep(), re.ResourceRZ.resource_rep()]
#         >>> re.ResourceTrotterizedQfunc.resources(n, order, first_order_expansion)
#         defaultdict(<class 'int'>, {RX: 2, RZ: 2})

#     """

#     @staticmethod
#     def _resource_decomp(
#         n, order, qfunc_compressed_reps, **kwargs
#     ) -> Dict[CompressedResourceOp, int]:
#         r"""Returns a dictionary representing the resources of the operator. The
#         keys are the operators and the associated values are the counts.

#         The Suzuki-Trotter product formula provides a method to approximate the matrix exponential of
#         Hamiltonian expressed as a linear combination of terms which in general do not commute. Consider
#         the Hamiltonian :math:`H = \Sigma^{N}_{j=0} O_{j}`, the product formula is constructed using
#         symmetrized products of the terms in the Hamiltonian. The symmetrized products of order
#         :math:`m \in [1, 2, 4, ..., 2k]` with :math:`k \in \mathbb{N}` are given by:

#         .. math::

#             \begin{align}
#                 S_{1}(t) &= \Pi_{j=0}^{N} \ e^{i t O_{j}} \\
#                 S_{2}(t) &= \Pi_{j=0}^{N} \ e^{i \frac{t}{2} O_{j}} \cdot \Pi_{j=N}^{0} \ e^{i \frac{t}{2} O_{j}} \\
#                 &\vdots \\
#                 S_{m}(t) &= S_{m-2}(p_{m}t)^{2} \cdot S_{m-2}((1-4p_{m})t) \cdot S_{m-2}(p_{m}t)^{2},
#             \end{align}

#         where the coefficient is :math:`p_{m} = 1 / (4 - \sqrt[m - 1]{4})`. The :math:`m`th order,
#         :math:`n`-step Suzuki-Trotter approximation is then defined as:

#         .. math:: e^{iHt} \approx \left [S_{m}(t / n)  \right ]^{n}.

#         For more details see `J. Math. Phys. 32, 400 (1991) <https://pubs.aip.org/aip/jmp/article-abstract/32/2/400/229229>`_.

#         Args:
#             n (int): an integer representing the number of Trotter steps to perform
#             order (int): an integer (:math:`m`) representing the order of the approximation (must be 1 or even)
#             qfunc_compressed_reps (list[CompressedResourceOp]): A list of compressed operations corresponding to the exponentiated terms of the hamiltonian (:math:`e^{i t O_{j}}`).

#         Resources:
#             The resources are defined according to the recurrsive formula presented above. Specifically, each
#             operator in the :code:`first_order_expansion` is called a number of times given by the formula:

#             .. math:: C_{O_{j}} = 2n \cdot 5^{\frac{m}{2} - 1}

#         **Example**

#         The arguments can be provided directly to the :code:`resources()` function to extract the cost:

#             >>> n, order = (1, 2)
#             >>> first_order_expansion = [re.ResourceRX.resource_rep(), re.ResourceRZ.resource_rep()]
#             >>> re.ResourceTrotterizedQfunc.resources(n, order, first_order_expansion)
#             defaultdict(<class 'int'>, {RX: 2, RZ: 2})

#         """
#         k = order // 2
#         gate_types = defaultdict(int, {})

#         if order == 1:
#             for cp_rep in qfunc_compressed_reps:
#                 gate_types[cp_rep] += n
#             return gate_types

#         for cp_rep in qfunc_compressed_reps:
#             gate_types[cp_rep] += 2 * n * (5 ** (k - 1))
#         return gate_types

#     @property
#     def resource_params(self) -> dict:
#         r"""Returns a dictionary containing the minimal information needed to compute the resources.

#         Returns:
#             dict: dictionary containing the resource parameters
#                 * n (int): an integer representing the number of Trotter steps to perform
#                 * order (int): an integer (:math:`m`) representing the order of the approximation (must be 1 or even)
#                 * qfunc_compressed_reps (list[CompressedResourceOp]): A list of compressed operations corresponding to the exponentiated terms of the hamiltonian (:math:`e^{i t O_{j}}`).
#         """
#         with qml.QueuingManager.stop_recording():
#             with qml.queuing.AnnotatedQueue() as q:
#                 base_hyper_params = ("n", "order", "qfunc", "reverse")

#                 qfunc_args = self.parameters
#                 qfunc_kwargs = {
#                     k: v for k, v in self.hyperparameters.items() if not k in base_hyper_params
#                 }

#                 qfunc = self.hyperparameters["qfunc"]
#                 qfunc(*qfunc_args, wires=self.wires, **qfunc_kwargs)

#         try:
#             qfunc_compressed_reps = tuple(op.resource_rep_from_op() for op in q.queue)

#         except AttributeError as error:
#             raise ResourcesNotDefined(
#                 "Every operation in the TrotterizedQfunc should be a ResourceOperator"
#             ) from error

#         return {
#             "n": self.hyperparameters["n"],
#             "order": self.hyperparameters["order"],
#             "qfunc_compressed_reps": qfunc_compressed_reps,
#         }

#     @classmethod
#     def resource_rep(cls, n, order, qfunc_compressed_reps, name=None) -> CompressedResourceOp:
#         """Returns a compressed representation containing only the parameters of
#         the Operator that are needed to compute a resource estimation.

#         Args:
#             n (int): an integer representing the number of Trotter steps to perform
#             order (int): an integer (:math:`m`) representing the order of the approximation (must be 1 or even)
#             qfunc_compressed_reps (list[CompressedResourceOp]): A list of compressed operations corresponding to the exponentiated terms of the hamiltonian (:math:`e^{i t O_{j}}`).

#         Returns:
#             CompressedResourceOp: the operator in a compressed representation
#         """
#         params = {
#             "n": n,
#             "order": order,
#             "qfunc_compressed_reps": qfunc_compressed_reps,
#         }
#         return CompressedResourceOp(cls, params, name=name)

#     def resource_rep_from_op(self) -> CompressedResourceOp:
#         r"""Returns a compressed representation directly from the operator"""
#         return self.__class__.resource_rep(**self.resource_params, name=self._name)


# def resource_trotterize(qfunc, n=1, order=2, reverse=False):
#     r"""Generates higher order Suzuki-Trotter product formulas from a set of
#     operations defined in a function.

#     The Suzuki-Trotter product formula provides a method to approximate the matrix exponential of
#     Hamiltonian expressed as a linear combination of terms which in general do not commute. Consider
#     the Hamiltonian :math:`H = \Sigma^{N}_{j=0} O_{j}`, the product formula is constructed using
#     symmetrized products of the terms in the Hamiltonian. The symmetrized products of order
#     :math:`m \in [1, 2, 4, ..., 2k]` with :math:`k \in \mathbb{N}` are given by:

#     .. math::

#         \begin{align}
#             S_{1}(t) &= \Pi_{j=0}^{N} \ e^{i t O_{j}} \\
#             S_{2}(t) &= \Pi_{j=0}^{N} \ e^{i \frac{t}{2} O_{j}} \cdot \Pi_{j=N}^{0} \ e^{i \frac{t}{2} O_{j}} \\
#             &\vdots \\
#             S_{m}(t) &= S_{m-2}(p_{m}t)^{2} \cdot S_{m-2}((1-4p_{m})t) \cdot S_{m-2}(p_{m}t)^{2},
#         \end{align}

#     where the coefficient is :math:`p_{m} = 1 / (4 - \sqrt[m - 1]{4})`. The :math:`m`th order,
#     :math:`n`-step Suzuki-Trotter approximation is then defined as:

#     .. math:: e^{iHt} \approx \left [S_{m}(t / n)  \right ]^{n}.

#     For more details see `J. Math. Phys. 32, 400 (1991) <https://pubs.aip.org/aip/jmp/article-abstract/32/2/400/229229>`_.

#     Args:
#         qfunc (Callable): A function which queues the operations corresponding to the exponentiated
#             terms of the hamiltonian (:math:`e^{i t O_{j}}`). The operations should be queued according
#             to the first order expression.
#         n (int): an integer representing the number of Trotter steps to perform
#         order (int): an integer (:math:`m`) representing the order of the approximation (must be 1 or even)

#     Resources:
#         The resources are defined according to the recurrsive formula presented above. Specifically, each
#         operator in the :code:`first_order_expansion` is called a number of times given by the formula:

#         .. math:: C_{O_{j}} = 2n \cdot 5^{\frac{m}{2} - 1}

#     .. seealso:: :class:`~.trotterize`

#     **Example**

#     First we define a function which queues the first-order expansion:

#     .. code-block:: python3

#         def first_order_expansion(time, theta, phi, wires):
#             "This is the first order expansion (U_1)."
#             re.ResourceRX(time*theta, wires[0])
#             re.ResourceRY(time*phi, wires[1])

#     The arguments can be provided directly to the :code:`resources()` function to extract the cost:

#         >>> n, order = (1, 2)
#         >>> time, theta, phi = (0.1, 0.2, 0.3)
#         >>> resource_op = re.resource_trotterize(first_order_expansion, n, order)(time, theta, phi, wires=['a', 'b'])
#         >>> resource_op.resources(**resource_op.resource_params)
#         defaultdict(<class 'int'>, {RX: 2, RY: 2})

#     """

#     @wraps(qfunc)
#     def wrapper(*args, **kwargs):
#         time = args[0]
#         other_args = args[1:]
#         return ResourceTrotterizedQfunc(
#             time, *other_args, qfunc=qfunc, n=n, order=order, reverse=reverse, **kwargs
#         )

#     return wrapper


class ResourceTrotterCDF(ResourceOperator):  # pylint: disable=too-many-ancestors
    r"""An operation representing the Suzuki-Trotter product approximation for the complex matrix
    exponential of compressed double factorized Hamiltonian.

    Args:
        compact_ham(CompactHamiltonian): :class:`~pennylane.resource_estimation.CompactHamiltonian` object
            that stores information about the compressed double factorized Hamiltonian
        n (int): An integer representing the number of Trotter steps to perform
        order (int): An integer (:math:`m`) representing the order of the approximation (must be 1 or even)


    Resources:
        The resources are defined according to the recursive formula presented above. Specifically, each
        operator in the :code:`first_order_expansion` is called a number of times given by the formula:

        .. math:: C_{O_{j}} = 2n \cdot 5^{\frac{m}{2} - 1}

        Furthermore, the first and last terms of the Hamiltonian appear in pairs due to the symmetric form
        of the recursive formula. Those counts are further simplified by grouping like terms as:

        .. math::

            \begin{align}
                C_{O_{0}} &= n \cdot 5^{\frac{m}{2} - 1} + 1,  \\
                C_{O_{N}} &= n \cdot 5^{\frac{m}{2} - 1}.
            \end{align}

    .. seealso:: :class:`~.TrotterProduct`

    """

    resource_keys = {"compact_ham", "num_steps", "order"}

    def __init__(self, compact_ham, num_steps, order, wires=None):

        self.num_steps = num_steps
        self.order = order
        self.compact_ham = compact_ham

        if wires is not None:
            self.wires = Wires(wires)
            self.num_wires = len(self.wires)
        else:
            self.wires = None
            self.num_wires = 2 * compact_ham.params["num_orbitals"]
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * compact_ham(CompactHamiltonian): :class:`~pennylane.resource_estimation.CompactHamiltonian` object
                  that stores information about the compressed double factorized Hamiltonian
                * n (int): an integer representing the number of Trotter steps to perform
                * order (int): an integer (:math:`m`) representing the order of the approximation (must be 1 or even)
        """
        return {
            "compact_ham": self.compact_ham,
            "num_steps": self.num_steps,
            "order": self.order,
        }

    @classmethod
    def resource_rep(cls, compact_ham, num_steps, order) -> CompressedResourceOp:
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            compact_ham (~pennylane.resource_estimation.CompactHamiltonian): The compressed double factorized
                Hamiltonian we will be approximately exponentiating.
            n (int): the number of Trotter steps to perform
            order (int): the order of the approximation (must be 1 or even)

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {
            "compact_ham": compact_ham,
            "num_steps": num_steps,
            "order": order,
        }
        return CompressedResourceOp(cls, params)

    @classmethod
    def default_resource_decomp(cls, compact_ham, num_steps, order, **kwargs) -> list[GateCount]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

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

        For more details see `J. Math. Phys. 32, 400 (1991) <https://pubs.aip.org/aip/jmp/article-abstract/32/2/400/229229>`_.

        Args:
            compact_ham(CompactHamiltonian): :class:`~pennylane.resource_estimation.CompactHamiltonian` object
                that stores information about the compressed double factorized Hamiltonian
            n (int): an integer representing the number of Trotter steps to perform
            order (int): an integer (:math:`m`) representing the order of the approximation (must be 1 or even)

        Resources:
            The resources are defined according to the recurrsive formula presented above. Specifically, each
            operator in the single step trotter circuit is called a number of times given by the formula:

            .. math:: C_{O_{j}} = 2n \cdot 5^{\frac{m}{2} - 1}

            Furthermore, the first and last terms of the hamiltonian appear in pairs due to the symmetric form
            of the recurrsive formula. Those counts are further simplified by grouping like terms as:

            .. math::

                \begin{align}
                    C_{O_{0}} &= n \cdot 5^{\frac{m}{2} - 1} + 1,  \\
                    C_{O_{N}} &= n \cdot 5^{\frac{m}{2} - 1}.
                \end{align}

        **Example**

        The resources can be computed as:

        >>> n, order = (1, 2)
        >>> compact_ham = plre.CompactHamiltonian.from_cdf(num_orbitals = 4, num_fragments = 4)
        >>> res = plre.estimate_resources(plre.ResourceTrotterCDF(compact_ham, n, order))
        >>> print(res)
        --- Resources: ---
         Total qubits: 4
         Total gates : 1.120E+4
         Qubit breakdown:
          clean qubits: 0, dirty qubits: 0, algorithmic qubits: 4
         Gate breakdown:
          {'T': 9.912E+3, 'Adjoint(T)': 168.0, 'Hadamard': 336.0, 'S': 168.0, 'Adjoint(S)': 168.0, 'CNOT': 448.0}


        """
        k = order // 2
        gate_list = []
        num_orb = compact_ham.params["num_orbitals"]
        num_frags = compact_ham.params["num_fragments"]

        op_onebody = resource_rep(
            plre.ResourceProd,
            {"cmpr_factors": tuple(plre.ResourceRZ.resource_rep() for i in range(2 * num_orb))},
        )

        op_twobody = resource_rep(
            plre.ResourceProd,
            {
                "cmpr_factors": tuple(
                    plre.ResourceMultiRZ.resource_rep(num_wires=2)
                    for i in range((2 * num_orb - 1) * num_orb)
                )
            },
        )

        basis_rot = resource_rep(plre.ResourceBasisRotation, {"dim_N": num_orb})

        if order == 1:
            print("Basis rot: ", num_frags * num_steps)
            gate_list.append(plre.GateCount(basis_rot, 2 * num_frags * num_steps))

            gate_list.append(plre.GateCount(op_onebody, num_steps))
            gate_list.append(plre.GateCount(op_twobody, (num_frags - 1) * num_steps))
            return gate_list

        # For first and last fragment
        gate_list.append(plre.GateCount(basis_rot, 4 * num_steps * (5 ** (k - 1)) + 2))
        gate_list.append(plre.GateCount(op_onebody, num_steps * (5 ** (k - 1)) + 1))
        gate_list.append(plre.GateCount(op_twobody, num_steps * (5 ** (k - 1))))

        # For rest of the fragments
        gate_list.append(
            plre.GateCount(basis_rot, 4 * num_steps * (num_frags - 2) * (5 ** (k - 1)))
        )
        gate_list.append(
            plre.GateCount(op_twobody, 2 * num_steps * (num_frags - 2) * (5 ** (k - 1)))
        )

        return gate_list

    @classmethod
    def default_controlled_resource_decomp(
        cls, compact_ham, num_steps, order, ctrl_num_ctrl_wires, ctrl_num_ctrl_values, **kwargs
    ):
        """Returns the controlled resource decomposition.

        Resources:
            The original resources are controlled only on the Z rotation gates.

        Args:
            compact_ham(CompactHamiltonian): :class:`~pennylane.resource_estimation.CompactHamiltonian` object
                that stores information about the compressed double factorized Hamiltonian
            num_steps (int): an integer representing the number of Trotter steps to perform
            order (int): an integer (:math:`m`) representing the order of the approximation (must be 1 or even)
            ctrl_num_ctrl_wires (int): the number of control wires for the controlled operations
            ctrl_num_ctrl_values (int): the number of control values for the controlled operations

        Returns:
            list[GateCount]: a list of GateCount objects representing the controlled resource decomposition

        """
        k = order // 2
        gate_list = []
        num_orb = compact_ham.params["num_orbitals"]
        num_frags = compact_ham.params["num_fragments"]

        op_onebody = resource_rep(
            plre.ResourceProd,
            {
                "cmpr_factors": tuple(
                    resource_rep(
                        plre.ResourceControlled,
                        {
                            "base_cmpr_op": plre.ResourceRZ.resource_rep(),
                            "num_ctrl_wires": ctrl_num_ctrl_wires,
                            "num_ctrl_values": ctrl_num_ctrl_values,
                        },
                    )
                    for i in range(2 * num_orb)
                )
            },
        )

        op_twobody = resource_rep(
            plre.ResourceProd,
            {
                "cmpr_factors": tuple(
                    resource_rep(
                        plre.ResourceControlled,
                        {
                            "base_cmpr_op": plre.ResourceMultiRZ.resource_rep(num_wires=2),
                            "num_ctrl_wires": ctrl_num_ctrl_wires,
                            "num_ctrl_values": ctrl_num_ctrl_values,
                        },
                    )
                    for i in range((2 * num_orb - 1) * num_orb)
                )
            },
        )

        basis_rot = resource_rep(plre.ResourceBasisRotation, {"dim_N": num_orb})

        if order == 1:
            print("Basis rot: ", num_frags * num_steps)
            gate_list.append(plre.GateCount(basis_rot, 2 * num_frags * num_steps))

            gate_list.append(plre.GateCount(op_onebody, num_steps))
            gate_list.append(plre.GateCount(op_twobody, (num_frags - 1) * num_steps))
            return gate_list

        # For first and last fragment
        gate_list.append(plre.GateCount(basis_rot, 4 * num_steps * (5 ** (k - 1)) + 2))
        gate_list.append(plre.GateCount(op_onebody, num_steps * (5 ** (k - 1)) + 1))
        gate_list.append(plre.GateCount(op_twobody, num_steps * (5 ** (k - 1))))

        # For rest of the fragments
        gate_list.append(
            plre.GateCount(basis_rot, 4 * num_steps * (num_frags - 2) * (5 ** (k - 1)))
        )
        gate_list.append(
            plre.GateCount(op_twobody, 2 * num_steps * (num_frags - 2) * (5 ** (k - 1)))
        )

        return gate_list


class ResourceTrotterTHC(ResourceOperator):  # pylint: disable=too-many-ancestors
    r"""An operation representing the Suzuki-Trotter product approximation for the complex matrix
    exponential of tensor hypercontracted Hamiltonian.

    Args:
        compact_ham(CompactHamiltonian): :class:`~pennylane.resource_estimation.CompactHamiltonian` object
            that stores information about the tensor hypercontracted Hamiltonian
        n (int): An integer representing the number of Trotter steps to perform
        order (int): An integer (:math:`m`) representing the order of the approximation (must be 1 or even)

    Resource Parameters:
        * compact_ham(CompactHamiltonian): :class:`~pennylane.resource_estimation.CompactHamiltonian` object
          that stores information about the tensor hypercontracted Hamiltonian
        * n (int): an integer representing the number of Trotter steps to perform
        * order (int): an integer (:math:`m`) representing the order of the approximation (must be 1 or even)

    Resources:
        The resources are defined according to the recursive formula presented above. Specifically, each
        operator in the :code:`first_order_expansion` is called a number of times given by the formula:

        .. math:: C_{O_{j}} = 2n \cdot 5^{\frac{m}{2} - 1}

        Furthermore, the first and last terms of the Hamiltonian appear in pairs due to the symmetric form
        of the recursive formula. Those counts are further simplified by grouping like terms as:

        .. math::

            \begin{align}
                C_{O_{0}} &= n \cdot 5^{\frac{m}{2} - 1} + 1,  \\
                C_{O_{N}} &= n \cdot 5^{\frac{m}{2} - 1}.
            \end{align}

    .. seealso:: :class:`~.TrotterProduct`

    """

    resource_keys = {"compact_ham", "num_steps", "order"}

    def __init__(self, compact_ham, num_steps, order, wires=None):

        self.num_steps = num_steps
        self.order = order
        self.compact_ham = compact_ham

        if wires is not None:
            self.wires = Wires(wires)
            self.num_wires = len(self.wires)
        else:
            self.wires = Wires(range(compact_ham.params["tensor_rank"]))
            self.num_wires = len(self.wires)
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * compact_ham(CompactHamiltonian): :class:`~pennylane.resource_estimation.CompactHamiltonian` object
                  that stores information about the compressed double factorized Hamiltonian
                * n (int): an integer representing the number of Trotter steps to perform
                * order (int): an integer (:math:`m`) representing the order of the approximation (must be 1 or even)
        """
        return {
            "compact_ham": self.compact_ham,
            "num_steps": self.num_steps,
            "order": self.order,
        }

    @classmethod
    def resource_rep(cls, compact_ham, num_steps, order) -> CompressedResourceOp:
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            compact_ham(CompactHamiltonian): :class:`~pennylane.resource_estimation.CompactHamiltonian` object
                that stores information about the tensor hypercontracted Hamiltonian
            n (int): an integer representing the number of Trotter steps to perform
            order (int): an integer (:math:`m`) representing the order of the approximation (must be 1 or even)

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {
            "compact_ham": compact_ham,
            "num_steps": num_steps,
            "order": order,
        }
        return CompressedResourceOp(cls, params)

    @classmethod
    def default_resource_decomp(cls, compact_ham, num_steps, order, **kwargs) -> list[GateCount]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

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

        For more details see `J. Math. Phys. 32, 400 (1991) <https://pubs.aip.org/aip/jmp/article-abstract/32/2/400/229229>`_.

        Args:
            compact_ham(CompactHamiltonian): :class:`~pennylane.resource_estimation.CompactHamiltonian` object
                that stores information about the tensor hypercontracted Hamiltonian
            n (int): an integer representing the number of Trotter steps to perform
            order (int): an integer (:math:`m`) representing the order of the approximation (must be 1 or even)

        Resources:
            The resources are defined according to the recurrsive formula presented above. Specifically, each
            operator in the single step trotter circuit is called a number of times given by the formula:

            .. math:: C_{O_{j}} = 2n \cdot 5^{\frac{m}{2} - 1}

            Furthermore, the first and last terms of the hamiltonian appear in pairs due to the symmetric form
            of the recurrsive formula. Those counts are further simplified by grouping like terms as:

            .. math::

                \begin{align}
                    C_{O_{0}} &= n \cdot 5^{\frac{m}{2} - 1} + 1,  \\
                    C_{O_{N}} &= n \cdot 5^{\frac{m}{2} - 1}.
                \end{align}

            The resources for single step trotter circuit are implemented based on Figure `4a` in
            `Luo 2024 <https://arxiv.org/abs/2407.04432>`_.

        **Example**

        The resources can be computed as:

        >>> n, order = (1, 2)
        >>> compact_ham = plre.CompactHamiltonian.from_thc(num_orbitals = 4, tensor_rank = 4)
        >>> plre.estimate_resources(plre.ResourceTrotterTHC(compact_ham, n, order)

        """
        k = order // 2
        gate_list = []
        num_orb = compact_ham.params["num_orbitals"]
        tensor_rank = compact_ham.params["tensor_rank"]

        op_onebody = resource_rep(
            plre.ResourceProd,
            {"cmpr_factors": tuple(plre.ResourceRZ.resource_rep() for i in range(2 * num_orb))},
        )

        op_twobody = resource_rep(
            plre.ResourceProd,
            {
                "cmpr_factors": tuple(
                    plre.ResourceMultiRZ.resource_rep(num_wires=2)
                    for i in range((2 * tensor_rank - 1) * tensor_rank)
                )
            },
        )

        basis_rot_onebody = resource_rep(plre.ResourceBasisRotation, {"dim_N": num_orb})
        basis_rot_twobody = resource_rep(plre.ResourceBasisRotation, {"dim_N": tensor_rank})

        if order == 1:
            gate_list.append(plre.GateCount(basis_rot_onebody, 2 * num_steps))
            gate_list.append(plre.GateCount(basis_rot_twobody, 2 * num_steps))
            gate_list.append(plre.GateCount(op_onebody, num_steps))
            gate_list.append(plre.GateCount(op_twobody, num_steps))
            return gate_list

        # For one-body tensor
        gate_list.append(plre.GateCount(basis_rot_onebody, 2 * num_steps * (5 ** (k - 1)) + 2))
        gate_list.append(plre.GateCount(op_onebody, num_steps * (5 ** (k - 1)) + 1))

        # For two-body tensor
        gate_list.append(plre.GateCount(basis_rot_twobody, 2 * num_steps * (5 ** (k - 1))))
        gate_list.append(plre.GateCount(op_twobody, num_steps * (5 ** (k - 1))))

        return gate_list

    @classmethod
    def default_controlled_resource_decomp(
        cls, compact_ham, num_steps, order, ctrl_num_ctrl_wires, ctrl_num_ctrl_values, **kwargs
    ):
        """Returns the controlled resource decomposition.

        Resources:
            The original resources are controlled only on the Z rotation gates.

        Args:
            compact_ham(CompactHamiltonian): :class:`~pennylane.resource_estimation.CompactHamiltonian` object
                that stores information about the compressed double factorized Hamiltonian
            num_steps (int): an integer representing the number of Trotter steps to perform
            order (int): an integer (:math:`m`) representing the order of the approximation (must be 1 or even)
            ctrl_num_ctrl_wires (int): the number of control wires for the controlled operations
            ctrl_num_ctrl_values (int): the number of control values for the controlled operations

        Returns:
            list[GateCount]: a list of GateCount objects representing the controlled resource decomposition

        """
        k = order // 2
        gate_list = []
        num_orb = compact_ham.params["num_orbitals"]
        tensor_rank = compact_ham.params["tensor_rank"]

        op_onebody = resource_rep(
            plre.ResourceProd,
            {
                "cmpr_factors": tuple(
                    resource_rep(
                        plre.ResourceControlled,
                        {
                            "base_cmpr_op": plre.ResourceRZ.resource_rep(),
                            "num_ctrl_wires": ctrl_num_ctrl_wires,
                            "num_ctrl_values": ctrl_num_ctrl_values,
                        },
                    )
                    for i in range(2 * num_orb)
                )
            },
        )

        op_twobody = resource_rep(
            plre.ResourceProd,
            {
                "cmpr_factors": tuple(
                    resource_rep(
                        plre.ResourceControlled,
                        {
                            "base_cmpr_op": plre.ResourceMultiRZ.resource_rep(num_wires=2),
                            "num_ctrl_wires": ctrl_num_ctrl_wires,
                            "num_ctrl_values": ctrl_num_ctrl_values,
                        },
                    )
                    for i in range((2 * tensor_rank - 1) * tensor_rank)
                )
            },
        )

        basis_rot_onebody = resource_rep(plre.ResourceBasisRotation, {"dim_N": num_orb})
        basis_rot_twobody = resource_rep(plre.ResourceBasisRotation, {"dim_N": tensor_rank})

        if order == 1:
            gate_list.append(plre.GateCount(basis_rot_onebody, 2 * num_steps))
            gate_list.append(plre.GateCount(basis_rot_twobody, 2 * num_steps))
            gate_list.append(plre.GateCount(op_onebody, num_steps))
            gate_list.append(plre.GateCount(op_twobody, num_steps))
            return gate_list

        # For one-body tensor
        gate_list.append(plre.GateCount(basis_rot_onebody, 2 * num_steps * (5 ** (k - 1)) + 2))
        gate_list.append(plre.GateCount(op_onebody, num_steps * (5 ** (k - 1)) + 1))

        # For two-body tensor
        gate_list.append(plre.GateCount(basis_rot_twobody, 2 * num_steps * (5 ** (k - 1))))
        gate_list.append(plre.GateCount(op_twobody, num_steps * (5 ** (k - 1))))

        return gate_list
