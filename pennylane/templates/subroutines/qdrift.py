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
"""Contains template for QDrift subroutine."""

import pennylane as qml
from pennylane.operation import Operation
from pennylane.math import requires_grad, unwrap
from pennylane.ops import Sum, SProd, Hamiltonian, LinearCombination


@qml.QueuingManager.stop_recording()
def _sample_decomposition(coeffs, ops, time, n=1, seed=None):
    """Generate the randomly sampled decomposition

    Args:
        coeffs (array): the coefficients of the operations from each term in the Hamiltonian
        ops (list[~.Operator]): the normalized operations from each term in the Hamiltonian
        time (float): time to evolve under the target Hamiltonian
        n (int): number of samples in the product, defaults to 1
        seed (int): random seed. defaults to None

    Returns:
        list[~.Operator]: the decomposition of operations as per the approximation
    """
    normalization_factor = qml.math.sum(qml.math.abs(coeffs))
    probs = qml.math.abs(coeffs) / normalization_factor
    exps = [
        qml.exp(base, (coeff / qml.math.abs(coeff)) * normalization_factor * time * 1j / n)
        for base, coeff in zip(ops, coeffs)
    ]

    choice_rng = qml.math.random.default_rng(seed)
    return tuple(choice_rng.choice(exps, p=probs, size=n, replace=True))


class QDrift(Operation):
    r"""An operation representing the QDrift approximation for the complex matrix exponential
    of a given Hamiltonian.

    The QDrift subroutine provides a method to approximate the matrix exponential of a Hamiltonian
    expressed as a linear combination of terms which in general do not commute. For the Hamiltonian
    :math:`H = \Sigma_j h_j H_{j}`, the product formula is constructed by random sampling from the
    terms of the Hamiltonian with the probability :math:`p_j = h_j / \sum_{j} hj` as:

    .. math::

        \prod_{j}^{n} e^{i \lambda H_j \tau / n},

    where :math:`\tau` is time, :math:`\lambda = \sum_j |h_j|` and :math:`n` is the total number of
    terms to be sampled and added to the product. Note, the terms :math:`H_{j}` are assumed to be
    normalized such that the "impact" of each term is fully encoded in the magnitude of :math:`h_{j}`.

    The number of samples :math:`n` required for a given error threshold can be approximated by:

    .. math::

        n \ \approx \ \frac{2\lambda^{2}t^{2}}{\epsilon}

    For more details see `Phys. Rev. Lett. 123, 070503 (2019) <https://arxiv.org/abs/1811.08017>`_.

    Args:
        hamiltonian (Union[.Hamiltonian, .Sum]): The Hamiltonian written as a sum of operations
        time (float): The time of evolution, namely the parameter :math:`t` in :math:`e^{iHt}`
        n (int): An integer representing the number of exponentiated terms
        seed (int): The seed for the random number generator

    Raises:
        TypeError: The ``hamiltonian`` is not of type :class:`~.Hamiltonian`, or :class:`~.Sum`
        QuantumFunctionError: If the coefficients of ``hamiltonian`` are trainable and are used
            in a differentiable workflow.

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
            qml.QDrift(H, time=1.2, n=10, seed=10)

            # Measure some quantity
            return qml.probs()

    >>> my_circ()
    array([0.65379493, 0.        , 0.34620507, 0.        ])


    .. details::
        :title: Usage Details

        We currently **Do NOT** support computing gradients with respect to the
        coefficients of the input Hamiltonian. We can however compute the gradient
        with respect to the evolution time:

        .. code-block:: python3

            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(dev)
            def my_circ(time):
                # Prepare H:
                H = qml.dot([0.2, -0.1], [qml.Y(0), qml.Z(1)])

                # Prepare some state
                qml.Hadamard(0)

                # Evolve according to H
                qml.QDrift(H, time, n=10, seed=10)

                # Measure some quantity
                return qml.expval(qml.Z(0) @ qml.Z(1))


        >>> time = np.array(1.23)
        >>> print(qml.grad(my_circ)(time))
        0.27980654844422853

        The error in the approximation of time evolution with the QDrift protocol is
        directly related to the number of samples used in the product. We provide a
        method to upper-bound the error:

        >>> H = qml.dot([0.25, 0.75], [qml.X(0), qml.Z(0)])
        >>> print(qml.QDrift.error(H, time=1.2, n=10))
        0.3661197552925645

    """

    def __init__(  # pylint: disable=too-many-arguments
        self, hamiltonian, time, n=1, seed=None, decomposition=None, id=None
    ):
        r"""Initialize the QDrift class"""

        if isinstance(hamiltonian, (Hamiltonian, LinearCombination)):
            coeffs, ops = hamiltonian.terms()

        elif isinstance(hamiltonian, Sum):
            coeffs, ops = [], []
            for op in hamiltonian:
                try:
                    coeffs.append(op.scalar)
                    ops.append(op.base)
                except AttributeError:  # coefficient is 1.0
                    coeffs.append(1.0)
                    ops.append(op)

        else:
            raise TypeError(
                f"The given operator must be a PennyLane ~.Hamiltonian or ~.Sum got {hamiltonian}"
            )

        if len(ops) < 2:
            raise ValueError(
                "There should be atleast 2 terms in the Hamiltonian. Otherwise use `qml.exp`"
            )

        if any(requires_grad(coeff) for coeff in coeffs):
            raise qml.QuantumFunctionError(
                "The QDrift template currently doesn't support differentiation through the "
                "coefficients of the input Hamiltonian."
            )

        if decomposition is None:  # need to do this to allow flatten and _unflatten
            unwrapped_coeffs = unwrap(coeffs)
            decomposition = _sample_decomposition(unwrapped_coeffs, ops, time, n=n, seed=seed)

        self._hyperparameters = {
            "n": n,
            "seed": seed,
            "base": hamiltonian,
            "decomposition": decomposition,
        }
        super().__init__(time, wires=hamiltonian.wires, id=id)

    def queue(self, context=qml.QueuingManager):
        context.remove(self.hyperparameters["base"])
        context.append(self)
        return self

    @classmethod
    def _unflatten(cls, data, metadata):
        """Recreate an operation from its serialized format.

        Args:
            data: the trainable component of the operation
            metadata: the non-trainable component of the operation

        The output of ``Operator._flatten`` and the class type must be sufficient to reconstruct the original
        operation with ``Operator._unflatten``.

        **Example:**

        >>> op = qml.Rot(1.2, 2.3, 3.4, wires=0)
        >>> op._flatten()
        ((1.2, 2.3, 3.4), (<Wires = [0]>, ()))
        >>> qml.Rot._unflatten(*op._flatten())
        >>> op = qml.PauliRot(1.2, "XY", wires=(0,1))
        >>> op._flatten()
        ((1.2,), (<Wires = [0, 1]>, (('pauli_word', 'XY'),)))
        >>> op = qml.ctrl(qml.U2(3.4, 4.5, wires="a"), ("b", "c") )
        >>> type(op)._unflatten(*op._flatten())
        Controlled(U2(3.4, 4.5, wires=['a']), control_wires=['b', 'c'])

        """
        hyperparameters_dict = dict(metadata[1])
        hamiltonian = hyperparameters_dict.pop("base")
        return cls(hamiltonian, *data, **hyperparameters_dict)

    @staticmethod
    def compute_decomposition(*args, **kwargs):  # pylint: disable=unused-argument
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
        decomp = kwargs["decomposition"]

        if qml.QueuingManager.recording():
            for op in decomp:
                qml.apply(op)

        return list(decomp)

    @staticmethod
    def error(hamiltonian, time, n=1):
        r"""A method for determining the upper-bound for the error in the approximation of
        the true matrix exponential.

        The error is bounded according to the following expression:

        .. math::

            \epsilon \ \leq \ \frac{2\lambda^{2}t^{2}}{n}  e^{\frac{2 \lambda t}{n}},

        where :math:`t` is time, :math:`\lambda = \sum_j |h_j|` and :math:`n` is the total number of
        terms to be added to the product. For more details see `Phys. Rev. Lett. 123, 070503 (2019) <https://arxiv.org/abs/1811.08017>`_.

        Args:
            hamiltonian (Union[.Hamiltonian, .Sum]): The Hamiltonian written as a sum of operations
            time (float): The time of evolution, namely the parameter :math:`t` in :math:`e^{-iHt}`
            n (int): An integer representing the number of exponentiated terms. default is 1

        Raises:
            TypeError: The given operator must be a PennyLane .Hamiltonian or .Sum

        Returns:
            float: upper bound on the precision achievable using the QDrift protocol
        """
        if isinstance(hamiltonian, (Hamiltonian, LinearCombination)):
            lmbda = qml.math.sum(qml.math.abs(hamiltonian.coeffs))

        elif isinstance(hamiltonian, Sum):
            lmbda = qml.math.sum(
                qml.math.abs(op.scalar) if isinstance(op, SProd) else 1.0 for op in hamiltonian
            )

        else:
            raise TypeError(
                f"The given operator must be a PennyLane ~.Hamiltonian or ~.Sum got {hamiltonian}"
            )

        return (2 * lmbda**2 * time**2 / n) * qml.math.exp(2 * lmbda * time / n)
