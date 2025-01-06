# Copyright 2024 Xanadu Quantum Technologies Inc.

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
from pennylane.labs.resource_estimation import (
    CompressedResourceOp,
    ResourceExp,
    ResourceOperator,
    ResourcesNotDefined,
)
from pennylane.templates import TrotterProduct
from pennylane.templates.subroutines.trotter import TrotterizedQfunc, _scalar


class ResourceTrotterProduct(TrotterProduct, ResourceOperator):
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

    Raises:
        TypeError: The ``hamiltonian`` is not of type :class:`~.Sum`.
        ValueError: The ``hamiltonian`` must have atleast two terms.
        ValueError: One or more of the terms in ``hamiltonian`` are not Hermitian.
        ValueError: The ``order`` is not one or a positive even integer.

    **Example**

    .. code-block:: python3

        coeffs = [0.25, 0.75]
        ops = [qml.X(0), qml.Z(0)]
        H = qml.dot(coeffs, ops)

        dev = qml.device("default.qubit", wires=2)
        @qml.qnode(dev)
        def my_circ():
            # Prepare some state
            qml.Hadamard(0)

            # Evolve according to H
            qml.TrotterProduct(H, time=2.4, order=2)

            # Measure some quantity
            return qml.state()

    >>> my_circ()
    array([-0.13259524+0.59790098j,  0.        +0.j        , -0.13259524-0.77932754j,  0.        +0.j        ])

    .. warning::

        The Trotter-Suzuki decomposition depends on the order of the summed observables. Two
        mathematically identical :class:`~.LinearCombination` objects may undergo different time
        evolutions due to the order in which those observables are stored. The order of observables
        can be queried using the :meth:`~.Sum.terms` method.

    .. warning::

        ``TrotterProduct`` does not automatically simplify the input Hamiltonian, allowing
        for a more fine-grained control over the decomposition but also risking an increased
        runtime and number of gates required. Simplification can be performed manually by
        applying :func:`~.simplify` to your Hamiltonian before using it in ``TrotterProduct``.

    .. details::
        :title: Usage Details

        An *upper-bound* for the error in approximating time-evolution using this operator can be
        computed by calling :func:`~.TrotterProduct.error()`. It is computed using two different methods; the
        "one-norm-bound" scaling method and the "commutator-bound" scaling method. (see `Childs et al. (2021) <https://arxiv.org/abs/1912.08854>`_)

        >>> hamiltonian = qml.dot([1.0, 0.5, -0.25], [qml.X(0), qml.Y(0), qml.Z(0)])
        >>> op = qml.TrotterProduct(hamiltonian, time=0.01, order=2)
        >>> op.error(method="one-norm-bound")
        SpectralNormError(8.039062500000003e-06)
        >>> op.error(method="commutator-bound")
        SpectralNormError(6.166666666666668e-06)

        This operation is similar to the :class:`~.ApproxTimeEvolution`. One can recover the behaviour
        of :class:`~.ApproxTimeEvolution` by taking the adjoint:

        >>> qml.adjoint(qml.TrotterProduct(hamiltonian, time, order=1, n=n))

        We can also compute the gradient with respect to the coefficients of the Hamiltonian and the
        evolution time:

        .. code-block:: python3

            @qml.qnode(dev)
            def my_circ(c1, c2, time):
                # Prepare H:
                H = qml.dot([c1, c2], [qml.X(0), qml.Z(0)])

                # Prepare some state
                qml.Hadamard(0)

                # Evolve according to H
                qml.TrotterProduct(H, time, order=2)

                # Measure some quantity
                return qml.expval(qml.Z(0) @ qml.Z(1))

        >>> args = np.array([1.23, 4.5, 0.1])
        >>> qml.grad(my_circ)(*tuple(args))
        (tensor(0.00961064, requires_grad=True), tensor(-0.12338274, requires_grad=True), tensor(-5.43401259, requires_grad=True))
    """

    @staticmethod
    def _resource_decomp(base, time, n, order, **kwargs) -> Dict[CompressedResourceOp, int]:
        k = order // 2
        first_order_expansion = [
            ResourceExp(op, (time / n) * 1j, num_steps=1).resource_rep_from_op()
            for op in base.operands
        ]

        if order == 1:
            return defaultdict(int, {cp_rep: n for cp_rep in first_order_expansion})

        cp_rep_first = first_order_expansion[0]
        cp_rep_last = first_order_expansion[-1]
        cp_rep_rest = first_order_expansion[1:-1]

        gate_types = defaultdict(int, {cp_rep: 2 * n * (5 ** (k - 1)) for cp_rep in cp_rep_rest})
        gate_types[cp_rep_first] = n * (5 ** (k - 1)) + 1
        gate_types[cp_rep_last] = n * (5 ** (k - 1))

        return gate_types

    def resource_params(self) -> dict:
        return {
            "n": self.hyperparameters["n"],
            "time": self.parameters[-1],
            "base": self.hyperparameters["base"],
            "order": self.hyperparameters["order"],
        }

    @classmethod
    def resource_rep(cls, base, time, n, order) -> CompressedResourceOp:
        params = {
            "n": n,
            "time": time,
            "base": base,
            "order": order,
        }
        return CompressedResourceOp(cls, params)

    @classmethod
    def resources(cls, *args, **kwargs) -> Dict[CompressedResourceOp, int]:
        """Returns a dictionary containing the counts of each operator type used to
        compute the resources of the operator."""
        return cls._resource_decomp(*args, **kwargs)


class ResourceTrotterizedQfunc(TrotterizedQfunc, ResourceOperator):

    @staticmethod
    def _resource_decomp(
        n, order, reverse, qfunc_compressed_reps, **kwargs
    ) -> Dict[CompressedResourceOp, int]:
        k = order // 2
        if order == 1:
            return defaultdict(int, {cp_rep: n for cp_rep in qfunc_compressed_reps})
        return defaultdict(
            int, {cp_rep: 2 * n * (5 ** (k - 1)) for cp_rep in qfunc_compressed_reps}
        )

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

        except AttributeError:
            raise ResourcesNotDefined(
                "Every operation in the TrotterizedQfunc should be a ResourceOperator"
            )

        return {
            "n": self.hyperparameters["n"],
            "order": self.hyperparameters["order"],
            "reverse": self.hyperparameters["reverse"],
            "qfunc_compressed_reps": qfunc_compressed_reps,
        }

    @classmethod
    def resource_rep(
        cls, qfunc_compressed_reps, n, order, reverse, name=None
    ) -> CompressedResourceOp:
        params = {
            "n": n,
            "order": order,
            "reverse": reverse,
            "qfunc_compressed_reps": qfunc_compressed_reps,
        }
        return CompressedResourceOp(cls, params, name=name)

    def resource_rep_from_op(self) -> CompressedResourceOp:
        """Returns a compressed representation directly from the operator"""
        return self.__class__.resource_rep(**self.resource_params(), name=self._name)


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

    For more details see `J. Math. Phys. 32, 400 (1991) <https://pubs.aip.org/aip/jmp/article-abstract/32/2/400/229229>`_.

    Suppose we have direct access to the operators which represent the exponentiated terms of 
    a hamiltonian:

    .. math:: \{ \hat{U}_{j} = e^{i t O_{j}} | for j \in [1, N] \}.
    
    Given a quantum circuit which uses these :math:`\hat{U}_{j}` operators to represents the
    first order expansion :math:`S_{1}(t)`; this function expands it to any higher order Suzuki-Trotter product.

    .. warning::

        :code:`trotterize()` requires the :code:`qfunc` argument is a function with a very specific call 
        signature. The first argument should be a time parameter which will be modified according to the 
        Suzuki-Trotter product formula. The wires required by the circuit should be either the last 
        explicit argument or the first keyword argument. 
        :code:`qfunc((time, arg1, ..., arg_n, wires=[...], kwarg_1, ..., kwarg_n))`
    
    Args:
        qfunc (Callable): the first-order expansion given as a callable function which queues operations
        n (int): an integer representing the number of Trotter steps to perform
        order (int): an integer (:math:`m`) representing the order of the approximation (must be 1 or even)
        reverse (bool): if true, reverse the order of the operations queued by :code:`qfunc`
        name (str): an optional name for the instance
        **non_trainable_kwargs (dict): non-trainable keyword arguments of the first-order expansion function
    
    Returns:
        Callable: a function with the same signature as :code:`qfunc`, when called it queues an instance of 
            :class:`~.TrotterizedQfunc`
    
    **Example**

    .. code-block:: python3

        def first_order_expansion(time, theta, phi, wires, flip=False):
            "This is the first order expansion (U_1)."
            ResourceRX(time*theta, wires[0])
            ResourceRY(time*phi, wires[1])
            if flip:
                ResourceCNOT(wires=wires[:2])

        @qml.qnode(qml.device("default.qubit"))
        def my_circuit(time, theta, phi, num_trotter_steps):
            resource_trotterize(
                first_order_expansion,
                n=num_trotter_steps,
                order=2,
            )(time, theta, phi, wires=['a', 'b', 'c'], flip=True)
            return qml.state()
        
    We can visualize the circuit to see the Suzuki-Trotter product formula being applied:
            
        >>> time = 0.1
        >>> theta, phi = (0.12, -3.45)
        >>> print(qml.draw(my_circuit, level=3)(time, theta, phi, num_trotter_steps=1))
        a: ──RX(0.01)──╭●─╭●──RX(0.01)──┤  State
        b: ──RY(-0.17)─╰X─╰X──RY(-0.17)─┤  State
        >>>
        >>> print(qml.draw(my_circuit, level=3)(time, angles, num_trotter_steps=3))
        a: ──RX(0.00)──╭●─╭●──RX(0.00)───RX(0.00)──╭●─╭●──RX(0.00)───RX(0.00)──╭●─╭●──RX(0.00)──┤  State
        b: ──RY(-0.06)─╰X─╰X──RY(-0.06)──RY(-0.06)─╰X─╰X──RY(-0.06)──RY(-0.06)─╰X─╰X──RY(-0.06)─┤  State

    """

    @wraps(qfunc)
    def wrapper(*args, **kwargs):
        time = args[0]
        other_args = args[1:]
        return ResourceTrotterizedQfunc(
            time, *other_args, qfunc=qfunc, n=n, order=order, reverse=reverse, **kwargs
        )

    return wrapper


@qml.QueuingManager.stop_recording()
def _recursive_qfunc(time, order, qfunc, wires, reverse, *qfunc_args, **qfunc_kwargs):
    """Generate a list of operations using the
    recursive expression which defines the Trotter product.
    Args:
        time (float): the evolution 'time'
        order (int): the order of the Trotter expansion
        ops (Iterable(~.Operators)): a list of terms in the Hamiltonian
    Returns:
        list: the approximation as product of exponentials of the Hamiltonian terms
    """
    if order == 1:
        tape = qml.tape.make_qscript(qfunc)(time, *qfunc_args, wires=wires, **qfunc_kwargs)
        return tape.operations[::-1] if reverse else tape.operations

    if order == 2:
        tape = qml.tape.make_qscript(qfunc)(time / 2, *qfunc_args, wires=wires, **qfunc_kwargs)
        return (
            tape.operations[::-1] + tape.operations
            if reverse
            else tape.operations + tape.operations[::-1]
        )

    scalar_1 = _scalar(order)
    scalar_2 = 1 - 4 * scalar_1

    ops_lst_1 = _recursive_qfunc(
        scalar_1 * time, order - 2, qfunc, wires, reverse, *qfunc_args, **qfunc_kwargs
    )
    ops_lst_2 = _recursive_qfunc(
        scalar_2 * time, order - 2, qfunc, wires, reverse, *qfunc_args, **qfunc_kwargs
    )

    return (2 * ops_lst_1) + ops_lst_2 + (2 * ops_lst_1)
