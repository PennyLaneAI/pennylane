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
from pennylane.ops import Sum, SProd, Hamiltonian


@qml.QueuingManager.stop_recording()
def _sample_decomposition(coeffs, ops, time, n=1, seed=None):
    """Generate the randomly sampled decomposition

    Args:
        coeffs (Array): the coefficients of the operations from each term in the hamiltonian.
        ops (List[~.Operator]): the normalized operations from each term in the hamiltonian.
        time (complex): time to evolve under the target hamiltonian.
        n (int): number of samples in the product. Defaults to 1.
        seed (int): random seed. Defaults to None.

    Returns:
        List[~.Operator]: the decomposition of operations as per the approximation.
    """
    normalization_factor = qml.math.sum(qml.math.abs(coeffs))
    probs = qml.math.abs(coeffs) / normalization_factor
    exps = [
        qml.exp(base, qml.math.sign(coeff) * normalization_factor * time * 1j / n)
        for base, coeff in zip(ops, coeffs)
    ]

    choice_rng = qml.math.random.default_rng(seed)
    return choice_rng.choice(exps, p=probs, size=n, replace=True)


class QDrift(Operation):
    r"""An operation representing the QDrift approximation for the complex matrix exponential
    of a given Hamiltonian.

    # TODO: link paper reference, mention that we assume each operator is normalized.

    The QDrift subroutine provides a method to approximate the matrix exponential of a Hamiltonian
    expressed as a linear combination of terms which in general do not commute. For the Hamiltonian
    :math:`H = \Sigma_j h_j H_{j}`, the product formula is constructed by random sampling from the
    terms of the Hamiltonian with the probability :math:`p_j = h_j / \sum_{j} hj` as:

    .. math::

        \prod_{j}^{n} e^{i \lambda H_j \tau / n},

    where :math:`\tau` is time, :math:`\lambda = \sum_j |h_j|` and :math:`n` is the total number of
    terms to be added to the product.

    Args:
        hamiltonian (Union[~.Hamiltonian, ~.Sum]): The Hamiltonian written in terms of products of
            Pauli gates
        time (int or float): The time of evolution, namely the parameter :math:`t` in :math:`e^{-iHt}`
        n (int): An integer representing the number of exponentiated terms
        seed (int): The seed for the random number generator

    Raises:
        TypeError: The ``hamiltonian`` is not of type :class:`~.Hamiltonian`, or :class:`~.Sum`
        QuantumFunctionError: If the coefficients of ``hamiltonian`` are trainable and are used
            in a differentiable workflow.

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

        We can also compute the gradient with respect to the evolution time:

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

    def __init__(self, hamiltonian, time, n=1, seed=None, decomposition=None, id=None):
        r"""Initialize the QDrift class"""

        if isinstance(hamiltonian, Hamiltonian):
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

        if any(requires_grad(coeff) for coeff in coeffs):
            raise qml.QuantumFunctionError(
                "The QDrift template currently doesn't support differentiation through the "
                "coefficients of the input Hamiltonian. Please instantiate the operation "
                "using coefficents with `requires_grad` set to False."
            )

        if decomposition is None:  # need to do this to allow flatten and _unflatten
            unwrapped_coeffs = unwrap(coeffs)
            decomposition = _sample_decomposition(unwrapped_coeffs, ops, time, n=1, seed=None)

        self._hyperparameters = {
            "n": n,
            "seed": seed,
            "base": hamiltonian,
            "decomposition": decomposition,
        }
        super().__init__(time, wires=hamiltonian.wires, id=id)

    @classmethod
    def _unflatten(cls, data, metadata):
        """Recreate an operation from its serialized format.

        Args:
            data: the trainable component of the operation
            metadata: the non-trainable component of the operation.

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
        decomp = kwargs["decomposition"]

        if qml.QueuingManager.recording():
            for op in decomp:
                qml.apply(op)

        return decomp

    @staticmethod
    def error(hamiltonian, time, n=1):
        """Computes the expected precision of the QDrift approximation given the initial parameters.
        # TODO: Add more detail and link paper reference.
        """
        if isinstance(hamiltonian, Hamiltonian):
            num_terms = len(hamiltonian.coeffs)
            max_coeff = max(hamiltonian.coeffs)

        elif isinstance(hamiltonian, Sum):
            num_terms = len(hamiltonian)
            max_coeff = max(op.scalar if isinstance(op, SProd) else 1.0 for op in hamiltonian)

        else:
            raise TypeError(
                f"The given operator must be a PennyLane ~.Hamiltonian or ~.Sum got {hamiltonian}"
            )

        return ((num_terms**2) * (max_coeff**2) * (time**2) / (2 * n)) * qml.math.exp(
            max_coeff * time * num_terms / n
        )
