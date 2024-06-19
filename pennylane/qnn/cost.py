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
This submodule contains frequently used loss and cost functions.
"""
# pylint: disable=too-many-arguments
import pennylane as qml


class SquaredErrorLoss:
    r"""Squared error loss function for circuits with trainable parameters.

    Combines an ansatz circuit with some target observables and calculates
    the squared error between their expectation values and a target.

    Args:
        ansatz (callable): The ansatz for the circuit before the final measurement step.
            Note that the ansatz **must** have the following signature:

            .. code-block:: python

                ansatz(params, **kwargs)

            where ``params`` are the trainable weights of the variational circuit, and
            ``kwargs`` are any additional keyword arguments that need to be passed
            to the template.
        observables (Iterable[.Observable]): observables to measure during the
            final step of each circuit
        device (Device, Sequence[Device]): Corresponding device(s) where the resulting
            function should be executed. This can either be a single device, or a list
            of devices of length matching the number of observables.
        interface (str, None): Which interface to use.
            This affects the types of objects that can be passed to/returned to the function.
            Supports all interfaces supported by the :func:`~.qnode` decorator.
        diff_method (str, None): The method of differentiation to use with the created function.
            Supports all differentiation methods supported by the :func:`~.qnode` decorator.

    Returns:
        callable: a loss function with signature ``loss(*args, target=None, **kwargs)`` that calculates
        the squared error loss between the observables' expectation values and a target.

    .. seealso:: :func:`~.map`

    **Example:**

    First, we create a device and design an ansatz:

    .. code-block:: python

        num_qubits = 3

        dev = qml.device('default.qubit', wires=num_qubits)

        def ansatz(phis, **kwargs):
            for w, phi in enumerate(phis):
                qml.RX(phi, wires=w)

    Now we can create the observables:

    .. code-block:: python3

        obs = [
            qml.Z(0),
            qml.X(0),
            qml.Z(1) @ qml.Z(2)
        ]

    Next, we can define the loss function:

    >>> loss = qml.qnn.cost.SquaredErrorLoss(ansatz, obs, dev, interface="torch")
    >>> phis = np.ones(num_qubits)
    >>> loss(phis, target=np.array([1.0, 0.5, 0.1]))
    array([0.21132197, 0.25      , 0.03683581])

    The loss function can be minimized using any gradient descent-based
    :doc:`optimizer </introduction/interfaces>`.
    """

    def __init__(
        self,
        ansatz,
        observables,
        device,
        measure="expval",
        interface="autograd",
        diff_method="best",
        **kwargs,
    ):
        @qml.qnode(device, diff_method=diff_method, interface=interface, *kwargs)
        def qnode(params, **circuit_kwargs):
            ansatz(params, wires=device.wires, **circuit_kwargs)
            return [getattr(qml, measure)(o) for o in observables]

        self.qnode = qnode

    def loss(self, *args, target=None, **kwargs):
        r"""Calculates the squared error loss between the observables'
        expectation values and a target.

        Keyword Args:
            target (tensor): target values

        Returns:
            array[float]: squared error values
        """

        if target is None:
            raise ValueError("The target cannot be None")

        input_ = self.qnode(*args, **kwargs)

        if len(target) != len(input_):
            raise ValueError(
                f"Input target of incorrect length {len(target)} instead of {len(input_)}"
            )

        return (input_ - target) ** 2

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)
