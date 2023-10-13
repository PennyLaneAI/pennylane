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
Contains template for QDrift subroutine.
"""
import pennylane as qml
from pennylane.operation import Operation
from pennylane.ops import Sum
import numpy as np


class QDrift(Operation):
    r"""An operation representing the QDrift subroutine for the complex matrix exponential
    of a given Hamiltonian.

    The QDrift subroutine provides a method to approximate the matrix exponential of Hamiltonian
    expressed as a linear combination of terms which in general do not commute. Consider the Hamiltonian
    :math:`H = \Sigma_j h_j H_{j}`, the product formula is constructed by random sampling over the terms
    of the Hamiltonian. With probability :math:`p_j` we will add to the product the operator
    :math:`\exp{(\frac{i \lambda H_j}{n})}`, where :math:`\lambda = \sum_j |h_j|` and :math:`n` is
    the number of terms to be added to the product.
    We calculate the probabilities as :math:`p_j = \frac{|h_j|}{\lambda}`.

    Args:
        hamiltonian (Union[~.Hamiltonian, ~.Sum]): The Hamiltonian of the system.
        time (complex): The time for which the system evolves.

    Keyword Args:
        n (int): The number of terms to be added to the product formula. Default is 1.
        seed (int): The seed for the random number generator.
    Raises:
        TypeError: The 'hamiltonian' is not of type :class:`~.Hamiltonian`, or :class:`~.Sum`.
        ValueError: One or more of the terms in 'hamiltonian' are not Hermitian.

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
            qml.QDrift(H, time=1.2, n = 10)

            # Measure some quantity
            return qml.probs()

    >>> my_circ()
    [0.71061676 0.         0.28938324 0.        ]


    .. details::
        :title: Usage Details

        We can also compute the gradient with respect to the
        evolution time:

        .. code-block:: python3

            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(dev)
            def my_circ(time):
                # Prepare H:
                H = qml.dot([0.2, -0.1], [qml.PauliY(0), qml.PauliZ(1)])

                # Prepare some state
                qml.Hadamard(0)

                # Evolve according to H
                qml.QDrift(H, time, n=10, seed=10)

                # Measure some quantity
                return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


        >>> args = qnp.array([1.23])
        >>> print(qml.grad(my_circ)(*tuple(args)))
        0.27980654844422853
    """

    def __init__(  # pylin: disable=too-many-arguments
        self, hamiltonian, time, n=1, seed=None, id=None
    ):
        r"""Initialize the QDrift class"""

        if isinstance(hamiltonian, qml.Hamiltonian):
            coeffs, ops = hamiltonian.terms()
            hamiltonian = qml.dot(coeffs, ops)

        if not isinstance(hamiltonian, Sum):
            raise TypeError(
                f"The given operator must be a PennyLane ~.Hamiltonian or ~.Sum got {hamiltonian}"
            )

        self._hyperparameters = {
            "n": n,
            "seed": seed,
            "base": hamiltonian,
        }
        super().__init__(time, wires=hamiltonian.wires, id=id)

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
        n = kwargs["n"]
        seed = kwargs["seed"]
        ops = kwargs["base"].operands

        with qml.QueuingManager.stop_recording():
            coeffs, bases = [], []
            for op in ops:
                try:
                    coeffs.append(op.scalar)
                    bases.append(op.base)
                except:
                    coeffs.append(1.0)
                    bases.append(op)

            lmbda = qml.math.sum(qml.math.abs(coeffs))
            probs = qml.math.abs(coeffs) / lmbda
            exps = [
                qml.exp(bases[i], qml.math.sign(coeffs[i]) * lmbda * time * 1j / n)
                for i in range(len(coeffs))
            ]

            choice_rng = np.random.default_rng(seed=seed)
            decomp = choice_rng.choice(exps, p=probs, size=n, replace=True)

        if qml.QueuingManager.recording():
            for op in decomp:  # apply operators in reverse order of expression
                qml.apply(op)

        return decomp
