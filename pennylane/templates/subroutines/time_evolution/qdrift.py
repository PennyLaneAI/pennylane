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
import copy

from pennylane import math
from pennylane.exceptions import QuantumFunctionError
from pennylane.operation import Operation
from pennylane.ops import Evolution, LinearCombination, Sum
from pennylane.queuing import QueuingManager, apply
from pennylane.wires import Wires


def _check_hamiltonian_type(hamiltonian):
    if not isinstance(hamiltonian, Sum):
        raise TypeError(f"The given operator must be a PennyLane ~.Sum, got {hamiltonian}")


def _extract_hamiltonian_coeffs_and_ops(hamiltonian):
    """Extract the coefficients and operators from a Hamiltonian that is
    a ``LinearCombination`` or a ``Sum``."""
    # Note that potentially_trainable_coeffs does *not* contain all coeffs
    if isinstance(hamiltonian, LinearCombination):
        coeffs, ops = hamiltonian.terms()

    elif isinstance(hamiltonian, Sum):
        coeffs, ops = [], []
        for op in hamiltonian:
            coeff = getattr(op, "scalar", None)
            if coeff is None:  # coefficient is 1.0
                coeffs.append(1.0)
                ops.append(op)
            else:
                coeffs.append(coeff)
                ops.append(op.base)

    return coeffs, ops


@QueuingManager.stop_recording()
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
    normalization_factor = math.sum(math.abs(coeffs))
    probs = math.abs(coeffs) / normalization_factor
    exps = [
        Evolution(base, -(coeff / math.abs(coeff)) * normalization_factor * time / n)
        for base, coeff in zip(ops, coeffs)
    ]

    choice_rng = math.random.default_rng(seed)
    return list(choice_rng.choice(exps, p=probs, size=n, replace=True))


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
        n (int): An integer representing the number of exponentiated terms.
        seed (int): The seed for the random number generator.

    Raises:
        TypeError: The ``hamiltonian`` is not of type :class:`~.Sum`
        QuantumFunctionError: If the coefficients of ``hamiltonian`` are trainable and are used
            in a differentiable workflow.
        ValueError: If there is only one term in the Hamiltonian.

    **Example**

    .. code-block:: python

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
    array([0.653..., 0.        , 0.346..., 0.        ])

    .. note::

        The option to pass a custom ``decomposition`` to ``QDrift`` has been removed.
        Instead, the custom decomposition can be applied using :func:`~.pennylane.apply`
        on all operations in the decomposition.

    .. details::
        :title: Usage Details

        We currently **Do NOT** support computing gradients with respect to the
        coefficients of the input Hamiltonian. We can however compute the gradient
        with respect to the evolution time:

        .. code-block:: python

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


        >>> time = qml.numpy.array(1.23)
        >>> print(qml.grad(my_circ)(time))
        0.279...

        The error in the approximation of time evolution with the QDrift protocol is
        directly related to the number of samples used in the product. We provide a
        method to upper-bound the error:

        >>> H = qml.dot([0.25, 0.75], [qml.X(0), qml.Z(0)])
        >>> print(qml.QDrift.error(H, time=1.2, n=10))
        0.3661197552925645

    """

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    def _flatten(self):
        h = self.hyperparameters["base"]
        hashable_hyperparameters = tuple(
            item for item in self.hyperparameters.items() if item[0] != "base"
        )
        return (h, self.data[-1]), hashable_hyperparameters

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(*data, **dict(metadata))

    def __init__(  # pylint: disable=too-many-arguments
        self, hamiltonian, time, n=1, seed=None, id=None
    ):
        r"""Initialize the QDrift class"""

        _check_hamiltonian_type(hamiltonian)
        coeffs, ops = _extract_hamiltonian_coeffs_and_ops(hamiltonian)

        if len(ops) < 2:
            raise ValueError(
                "There should be at least 2 terms in the Hamiltonian. Otherwise use `qml.exp`"
            )

        if any(math.requires_grad(coeff) for coeff in coeffs):
            raise QuantumFunctionError(
                "The QDrift template currently doesn't support differentiation through the "
                "coefficients of the input Hamiltonian."
            )

        self._hyperparameters = {"n": n, "seed": seed, "base": hamiltonian}
        super().__init__(*hamiltonian.data, time, wires=hamiltonian.wires, id=id)

    def map_wires(self, wire_map: dict):
        # pylint: disable=protected-access
        new_op = copy.deepcopy(self)
        new_op._wires = Wires([wire_map.get(wire, wire) for wire in self.wires])
        new_op._hyperparameters["base"] = new_op._hyperparameters["base"].map_wires(wire_map)

        return new_op

    def queue(self, context=QueuingManager):
        context.remove(self.hyperparameters["base"])
        context.append(self)
        return self

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
        time = args[-1]
        hamiltonian = kwargs["base"]
        seed = kwargs["seed"]
        n = kwargs["n"]
        coeffs, ops = _extract_hamiltonian_coeffs_and_ops(hamiltonian)
        decomposition = _sample_decomposition(math.unwrap(coeffs), ops, time, n=n, seed=seed)

        if QueuingManager.recording():
            for op in decomposition:
                apply(op)

        return decomposition

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
            hamiltonian (Sum): The Hamiltonian written as a sum of operations
            time (float): The time of evolution, namely the parameter :math:`t` in :math:`e^{-iHt}`
            n (int): An integer representing the number of exponentiated terms. default is 1

        Raises:
            TypeError: The given operator must be a PennyLane .Hamiltonian or .Sum

        Returns:
            float: upper bound on the precision achievable using the QDrift protocol
        """
        _check_hamiltonian_type(hamiltonian)
        coeffs, _ = _extract_hamiltonian_coeffs_and_ops(hamiltonian)
        lmbda = math.sum(math.abs(coeffs))

        return (2 * lmbda**2 * time**2 / n) * math.exp(2 * lmbda * time / n)
