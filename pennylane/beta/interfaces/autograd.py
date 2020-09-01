# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
This module contains the mixin interface class for creating differentiable quantum tapes with
Autograd.
"""
# pylint: disable=protected-access
import autograd.extend
import autograd.builtins

from pennylane import numpy as np

from pennylane.beta.queuing import AnnotatedQueue


class AutogradInterface(AnnotatedQueue):
    """Mixin class for applying an autograd interface to a :class:`~.QuantumTape`.

    Autograd-compatible quantum tape classes can be created via subclassing:

    .. code-block:: python

        class MyAutogradQuantumTape(AutogradInterface, QuantumTape):

    Alternatively, the autograd interface can be dynamically applied to existing
    quantum tapes via the :meth:`~.apply` class method. This modifies the
    tape **in place**.

    Once created, the autograd interface can be used to perform quantum-classical
    differentiable programming.

    .. note::

        If using a device that supports native autograd computation and backpropagation, such as
        :class:`~.DefaultQubitAutograd`, the Autograd interface **does not need to be applied**. It
        is only applied to tapes executed on non-Autograd compatible devices.

    **Example**

    One an autograd quantum tape has been created, it can be differentiated using autograd:

    .. code-block:: python

        def cost_fn(x, y, z, device):
            tape = AutogradInterface.apply(QuantumTape())

            with tape:
                qml.Rot(x, y ** 2, y * np.sin(z), wires=0)
                expval(qml.PauliX(0))

            return tape.execute(device=device)

    >>> x = np.array(0.1, requires_grad=False)
    >>> y = np.array(0.2, requires_grad=True)
    >>> z = np.array(0.3, requires_grad=True)
    >>> dev = qml.device("default.qubit", wires=2)
    >>> cost_fn(x, y, z, device=dev)
    [0.03991951]
    >>> jac_fn = qml.jacobian(cost_fn)
    >>> jac_fn(x, y, z, device=dev))
    [[ 0.39828408 -0.00045133]]
    """

    # pylint: disable=attribute-defined-outside-init

    cast = staticmethod(np.stack)

    @property
    def interface(self):
        """str, None: automatic differentiation interface used by the quantum tap (if any)"""
        return "autograd"

    def _update_trainable_params(self):
        params = [o.data for o in self.operations + self.observables]
        params = [item for sublist in params for item in sublist]

        trainable_params = set()

        for idx, p in enumerate(params):
            if getattr(p, "requires_grad", True):
                trainable_params.add(idx)

        self.trainable_params = trainable_params
        return params

    def get_parameters(self, free_only=True):  # pylint: disable=missing-function-docstring
        params = self._update_trainable_params()

        if free_only:
            params = [p for idx, p in enumerate(params) if idx in self.trainable_params]

        return np.array(params)

    @autograd.extend.primitive
    def _execute(self, params, device):
        params = autograd.builtins.tuple(params)

        res = self.execute_device(params, device=device)

        if res.dtype == np.dtype("object"):
            return np.hstack(res)

        return res

    @staticmethod
    def vjp(ans, self, params, device):  # pylint: disable=unused-argument
        """Returns the vector-Jacobian product operator for the quantum tape.

        Takes the same arguments as :meth:`~.execute`, plus `ans`.

        Returns:
            function[array[float], array[float]]: vector-Jacobian product operator
        """

        def gradient_product(g):
            jac = self.jacobian(device, params=params)
            vjp = g.flatten() @ jac
            return vjp

        return gradient_product

    @classmethod
    def apply(cls, tape):
        """Apply the autograd interface to an existing tape in-place.

        Args:
            tape (.QuantumTape): a quantum tape to apply the Autograd interface to

        **Example**

        >>> with QuantumTape() as tape:
        ...     qml.RX(0.5, wires=0)
        ...     expval(qml.PauliZ(0))
        >>> AutogradInterface.apply(tape)
        >>> tape
        <AutogradQuantumTape: wires=<Wires = [0]>, params=1>
        """
        tape_class = getattr(tape, "__bare__", tape.__class__)
        tape.__bare__ = tape_class
        tape.__class__ = type("AutogradQuantumTape", (cls, tape_class), {})
        tape._update_trainable_params()
        return tape


autograd.extend.defvjp(AutogradInterface._execute, AutogradInterface.vjp, argnums=[1])
