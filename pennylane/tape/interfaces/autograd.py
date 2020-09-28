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
from autograd.numpy.numpy_boxes import ArrayBox

from pennylane import numpy as np

from pennylane.tape.queuing import AnnotatedQueue


class AutogradInterface(AnnotatedQueue):
    """Mixin class for applying an autograd interface to a :class:`~.JacobianTape`.

    Autograd-compatible quantum tape classes can be created via subclassing:

    .. code-block:: python

        class MyAutogradQuantumTape(AutogradInterface, JacobianTape):

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

    Once an autograd quantum tape has been created, it can be differentiated using autograd:

    .. code-block:: python

        tape = AutogradInterface.apply(JacobianTape())

        with tape:
            qml.Rot(0, 0, 0, wires=0)
            expval(qml.PauliX(0))

        def cost_fn(x, y, z, device):
            tape.set_parameters([x, y ** 2, y * np.sin(z)], trainable_only=False)
            return tape.execute(device=device)

    >>> x = np.array(0.1, requires_grad=False)
    >>> y = np.array(0.2, requires_grad=True)
    >>> z = np.array(0.3, requires_grad=True)
    >>> dev = qml.device("default.qubit", wires=2)
    >>> cost_fn(x, y, z, device=dev)
    [0.03991951]
    >>> jac_fn = qml.jacobian(cost_fn)
    >>> jac_fn(x, y, z, device=dev)
    [[ 0.39828408, -0.00045133]]
    """

    # pylint: disable=attribute-defined-outside-init
    dtype = np.float64

    @property
    def interface(self):  # pylint: disable=missing-function-docstring
        return "autograd"

    def _update_trainable_params(self):
        """Set the trainable parameters.

        Unlike in :class:`~.JacobianTape`, we also set the private attribute
        ``self._all_parameter_values``.

        Since :meth:`~.get_parameters` **always** calls ``_update_trainable_params``, we access this
        private attribute there. This allows the :meth:`~.get_parameters` method to avoid performing
        a redundant parameter extraction.
        """
        params = []

        for p_idx in self._par_info:
            op = self._par_info[p_idx]["op"]
            op_idx = self._par_info[p_idx]["p_idx"]
            params.append(op.data[op_idx])

        trainable_params = set()

        for idx, p in enumerate(params):
            if getattr(p, "requires_grad", False) or isinstance(p, ArrayBox):
                trainable_params.add(idx)

        self.trainable_params = trainable_params
        self._all_parameter_values = params

    def get_parameters(self, trainable_only=True):  # pylint: disable=missing-function-docstring
        self._update_trainable_params()
        params = self._all_parameter_values

        if trainable_only:
            params = [
                p
                for idx, p in enumerate(self._all_parameter_values)
                if idx in self.trainable_params
            ]

        return autograd.builtins.list(params)

    @autograd.extend.primitive
    def _execute(self, params, device):
        # unwrap all NumPy scalar arrays to Python literals
        params = [p.item() if p.shape == tuple() else p for p in params]
        params = autograd.builtins.tuple(params)

        res = self.execute_device(params, device=device)

        if res.dtype == np.dtype("object"):
            return np.hstack(res)

        requires_grad = False

        if self.trainable_params:
            requires_grad = True

        return np.array(res, requires_grad=requires_grad)

    @staticmethod
    def vjp(ans, self, params, device):  # pylint: disable=unused-argument
        """Returns the vector-Jacobian product operator for the quantum tape.
        The returned function takes the arguments as :meth:`~.JacobianTape.execute`.

        Args:
            ans (array): the result of the tape execution
            self (.AutogradQuantumTape): the tape instance
            params (list[Any]): the quantum tape operation parameters
            device (.Device): a PennyLane device that can execute quantum
                operations and return measurement statistics

        Returns:
            function: this function accepts the backpropagation
            gradient output vector, and computes the vector-Jacobian product
        """

        def gradient_product(g):
            jac = self.jacobian(device, params=params, **self.jacobian_options)
            vjp = g.flatten() @ jac
            return vjp

        return gradient_product

    @classmethod
    def apply(cls, tape):
        """Apply the autograd interface to an existing tape in-place.

        Args:
            tape (.JacobianTape): a quantum tape to apply the Autograd interface to

        **Example**

        >>> with JacobianTape() as tape:
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
