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
import pennylane as qml
from pennylane.operation import Operation
from pennylane.ops import Sum


def _scalar(order):
    """Compute the scalar used in the recursive expression.

    Args:
        order (int): order of Trotter product (assume order is an even integer > 2).

    Returns:
        float: scalar to be used in the recursive expression.
    """
    root = 1 / (order - 1)
    return (4 - 4**root) ** -1


@qml.QueuingManager.stop_recording()
def _recursive_expression(x, order, ops):
    """Generate a list of operations using the
    recursive expression which defines the Trotter product.

    Args:
        x (complex): the evolution 'time'
        order (int): the order of the Trotter Expansion
        ops (Iterable(~.Operators)): a list of terms in the Hamiltonian

    Returns:
        List: the approximation as product of exponentials of the Hamiltonian terms
    """
    if order == 1:
        return [qml.exp(op, x * 1j) for op in ops]

    if order == 2:
        return [qml.exp(op, x * 0.5j) for op in ops + ops[::-1]]

    scalar_1 = _scalar(order)
    scalar_2 = 1 - 4 * scalar_1

    ops_lst_1 = _recursive_expression(scalar_1 * x, order - 2, ops)
    ops_lst_2 = _recursive_expression(scalar_2 * x, order - 2, ops)

    return (2 * ops_lst_1) + ops_lst_2 + (2 * ops_lst_1)


class TrotterProduct(Operation):
    r"""An operation representing the Suzuki-Trotter product approximation for the complex matrix exponential
    of a given Hamiltonian.

    The Suzuki-Trotter product formula provides a method to approximate the matrix exponential of Hamiltonian
    expressed as a linear combination of terms which in general do not commute. Consider the Hamiltonian
    :math:`H = \Sigma^{N}_{j=0} O_{j}`, the product formula is constructed using symmetrized products of the terms
    in the Hamiltonian. The symmetrized products of order :math: `m \in [1, 2, 4, ..., 2k] | k \in \mathbb{N}`
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

        We can also compute the gradient with respect to the coefficients of the Hamiltonian and the
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

    def __init__(  # pylin: disable=too-many-arguments
        self, hamiltonian, time, n=1, order=1, check_hermitian=True, id=None
    ):
        r"""Initialize the TrotterProduct class"""

        if order <= 0 or order != 1 and order % 2 != 0:
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
                        "One or more of the terms in the Hamiltonian may not be Hermitian"
                    )

        self._hyperparameters = {
            "n": n,
            "order": order,
            "base": hamiltonian,
            "check_hermitian": check_hermitian,
        }
        super().__init__(time, wires=hamiltonian.wires, id=id)

    def _flatten(self):
        """Serialize the operation into trainable and non-trainable components.

        Returns:
            data, metadata: The trainable and non-trainable components.

        See ``Operator._unflatten``.

        The data component can be recursive and include other operations. For example, the trainable component of ``Adjoint(RX(1, wires=0))``
        will be the operator ``RX(1, wires=0)``.

        The metadata **must** be hashable.  If the hyperparameters contain a non-hashable component, then this
        method and ``Operator._unflatten`` should be overridden to provide a hashable version of the hyperparameters.

        **Example:**

        >>> op = qml.Rot(1.2, 2.3, 3.4, wires=0)
        >>> qml.Rot._unflatten(*op._flatten())
        Rot(1.2, 2.3, 3.4, wires=[0])
        >>> op = qml.PauliRot(1.2, "XY", wires=(0,1))
        >>> qml.PauliRot._unflatten(*op._flatten())
        PauliRot(1.2, XY, wires=[0, 1])

        Operators that have trainable components that differ from their ``Operator.data`` must implement their own
        ``_flatten`` methods.

        >>> op = qml.ctrl(qml.U2(3.4, 4.5, wires="a"), ("b", "c") )
        >>> op._flatten()
        ((U2(3.4, 4.5, wires=['a']),),
        (<Wires = ['b', 'c']>, (True, True), <Wires = []>))
        """
        hamiltonian = self.hyperparameters["base"]
        time = self.parameters[0]

        hashable_hyperparameters = tuple(
            (key, value) for key, value in self.hyperparameters.items() if key != "base"
        )
        return (hamiltonian, time), hashable_hyperparameters

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
        hyperparameters_dict = dict(metadata)
        return cls(*data, **hyperparameters_dict)

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
        order = kwargs["order"]
        ops = kwargs["base"].operands

        decomp = _recursive_expression(time / n, order, ops)[-1::-1] * n

        if qml.QueuingManager.recording():
            for op in decomp:  # apply operators in reverse order of expression
                qml.apply(op)

        return decomp
