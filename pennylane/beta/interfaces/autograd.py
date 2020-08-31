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
Differentiable quantum tapes with Autograd interface.
"""
import autograd.extend
import autograd.builtins

from pennylane.utils import unflatten
from pennylane import numpy as np

from pennylane.beta.queuing import AnnotatedQueue
from pennylane.beta.tapes import QuantumTape


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

        If using a device that supports native autograd computation and backpropagation,
        the Autograd interface **does not need to be applied**. It is only applied
        to tapes executed on non-Autograd compatible devices.

        On autograd-compatible devices, it is simply sufficient to set the ``tape.cast``
        attribute:

        >>> tape.cast = AutogradInterface.cast

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

    cast = staticmethod(np.stack)

    @property
    def interface(self):
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

    def get_parameters(self, free_only=True):
        params = self._update_trainable_params()

        if free_only:
            params = [p for idx, p in enumerate(params) if idx in self.trainable_params]

        return np.array(params)

    @autograd.extend.primitive
    def _execute(self, params, device):
        params = autograd.builtins.tuple(params)
        return np.array(self.execute_device(params, device=device))

    @staticmethod
    def vjp(ans, self, params, device):
        def gradient_product(g):
            jac = self.jacobian(device, params=params)
            return g @ jac

        return gradient_product

    @classmethod
    def apply(cls, tape):
        """Apply the autograd interface to an existing tape in-place.

        **Example**

        >>> with QuantumTape() as tape:
        ...     qml.RX(0.5, wires=0)
        ...     expval(qml.PauliZ(0))
        >>> AutogradInterface.apply(tape)
        >>> tape
        <AutogradQuantumTape: wires=<Wires = [0]>, params=1>
        """
        tape.__class__ = type("AutogradQuantumTape", (cls, tape.__class__), {})
        tape._update_trainable_params()
        return tape


autograd.extend.defvjp(AutogradInterface._execute, AutogradInterface.vjp, argnums=[1])
