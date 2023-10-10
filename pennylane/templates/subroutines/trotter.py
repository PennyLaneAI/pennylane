# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
import copy

import pennylane as qml

# from pennylane.math ...
from pennylane import numpy as np
from pennylane.operation import Operation
from pennylane.ops import Sum


def _scalar(order):
    """Assumes that order is an even integer > 2"""
    root = 1 / (order - 1)
    return (4 - 4**root) ** -1


@qml.QueuingManager.stop_recording()
def _recursive_decomposition(x, order, ops):
    """Generate a list of operators."""
    if order == 1:
        return [qml.exp(op, x * 1j) for op in ops]

    if order == 2:
        return [qml.exp(op, x * 0.5j) for op in (ops + ops[::-1])]

    scalar_1 = _scalar(order)
    scalar_2 = 1 - 4 * scalar_1

    ops_lst_1 = _recursive_decomposition(scalar_1 * x, order - 2, ops)
    ops_lst_2 = _recursive_decomposition(scalar_2 * x, order - 2, ops)

    return (2 * ops_lst_1) + ops_lst_2 + (2 * ops_lst_1)


class TrotterProduct(Operation):
    """An operation representing the Suzuki-Trotter product approximation for the complex matrix exponential
    of a given hamiltonian.

    The Suzuki-Trotter product formula provides a method to approximate the matrix exponential of hamiltonian
    expressed as a linear combination of terms which in general do not commute. Consider the hamiltonian
    :math:`H = \Sigma^{N}_{j=0} O_{j}`, the product formula is constructed using symmetrized products of the terms
    in the hamiltonian. The symmetrized products of order :math: `m \in [1, 2, 4, ..., 2k] | k \in \mathbb{N}`
    are given by:

    .. math::

        \begin{align}
            S_{m=1}(t) &= \Pi_{j=0}^{N} \ exp(i t O_{j}) \\
            S_{m=2}(t) &= \Pi_{j=0}^{N} \ exp(i \frac{t}{2} O_{j}) \cdot \Pi_{j=N}^{0} \ exp(i \frac{t}{2} O_{j}) \\
            &\vdots
            S_{m=2k}(t) &= S_{2k-2}(p_{2k}t)^{2} \cdot S_{2k-2}((1-4p_{2k})t) \cdot S_{2k-2}(p_{2k}t)^{2}
        \end{align}

    Where the coefficient is :math:`p_{2k} = \frac{1}{4 - \sqrt[2k - 1]{4}}`.

    The :math:`2k`th order, :math:`n`-step Suzuki-Trotter approximation is then defined as:

    .. math:: exp(iHt) \approx (S_{2k}(\frac{t}{n}))^{n}

    Args:
        hamiltonian (Union[~.Hamiltonian, ~.Sum]):

    Keyword Args:
        n (int): An integer representing the number of Trotter steps to perform.
        order (int): An integer representing the order of the approximation (must be 1 or even).
        check_hermitian (bool): A flag to enable the validation check to ensure this is a valid unitary operator.

    Raises:
        TypeError: The 'hamiltonian' is not of type ~.Hamiltonian, or ~.Sum.
        ValueError: One or more of the terms in 'hamiltonian' are not Hermitian. 
        ValueError: The 'order' is not one or a positive even integer.

    **Example**

    .. code-block:: python3

        coeffs = [0.25, 0.75]
        ops = [qml.PauliX(0), qml.PauliZ(0)]
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
    [-0.13259524+0.59790098j  0.        +0.j         -0.13259524-0.77932754j  0.        +0.j        ]

    .. details::
        :title: Usage Details

        We can also compute the gradient with respect to the coefficients of the hamiltonian and the
        evolution time: 

        .. code-block:: python3

            @qml.qnode(dev)
            def my_circ(c1, c2, time): 
                # Prepare H: 
                H = qml.dot([c1, c2], [qml.PauliX(0), qml.PauliZ(1)])
                    
                # Prepare some state
                qml.Hadamard(0)
                
                # Evolve according to H
                qml.TrotterProduct(H, time, order=2)
                
                # Measure some quantity
                return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        >>> args = qnp.array([1.23, 4.5, 0.1])
        >>> qml.grad(my_circ)(*tuple(args))
        (tensor(0.00961064, requires_grad=True), tensor(-0.12338274, requires_grad=True), tensor(-5.43401259, requires_grad=True))

    """

    def __init__(self, hamiltonian, time, n=1, order=1, check_hermitian=True, id=None):
        """Initialize the TrotterProduct class"""

        if not (order > 0 and (order == 1 or order % 2 == 0)):
            raise ValueError(
                f"The order of a TrotterProduct must be 1 or a positive even integer, got {order}."
            )

        if isinstance(hamiltonian, qml.Hamiltonian):
            coeffs, ops = hamiltonian.terms()
            hamiltonian = qml.dot(coeffs, ops)

        if not isinstance(hamiltonian, Sum):
            raise TypeError(
                f"The given operator must be a PennyLane ~.Hamiltonian or ~.Sum got {hamiltonian}"
            )

        if check_hermitian:
            for op in hamiltonian.operands:
                if not op.is_hermitian:
                    raise ValueError(
                        "One or more of the terms in the Hamiltonian may not be hermitian"
                    )

        self._hyperparameters = {"num_steps": n, "order": order, "base": hamiltonian}
        wires = hamiltonian.wires
        super().__init__(time, wires=wires, id=id)

    @staticmethod
    def compute_decomposition(*args, **kwargs):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. note::

            Operations making up the decomposition should be queued within the
            ``compute_decomposition`` method.

        .. seealso:: :meth:`~.Operator.decomposition`.

        Args:
            *params (list): trainable parameters of the operator, as stored in the ``parameters`` attribute
            wires (Iterable[Any], Wires): wires that the operator acts on
            **hyperparams (dict): non-trainable hyperparameters of the operator, as stored in the ``hyperparameters`` attribute

        Returns:
            list[Operator]: decomposition of the operator
        """
        time = args[0]
        n = kwargs["num_steps"]
        order = kwargs["order"]
        ops = kwargs["base"].operands

        decomp = _recursive_decomposition(time / n, order, ops) * n

        if qml.QueuingManager.recording():
            for op in decomp:
                qml.apply(op)

        return decomp
