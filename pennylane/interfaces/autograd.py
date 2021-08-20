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
This module contains the mixin interface class for creating differentiable quantum tapes with
Autograd.
"""
# pylint: disable=protected-access
import autograd.extend
import autograd.builtins
from autograd.numpy.numpy_boxes import ArrayBox

from pennylane import numpy as np
from pennylane.queuing import AnnotatedQueue


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
        """
        params = self.get_parameters(trainable_only=False, return_arraybox=True)
        trainable_params = set()

        for idx, p in enumerate(params):
            if getattr(p, "requires_grad", False) or isinstance(p, ArrayBox):
                trainable_params.add(idx)

        self.trainable_params = trainable_params
        self._all_parameter_values = params

    def get_parameters(self, trainable_only=True, return_arraybox=False):
        """Return the parameters incident on the tape operations.

        The returned parameters are provided in order of appearance
        on the tape. By default, the returned parameters are wrapped in
        an ``autograd.builtins.list`` container.

        Args:
            trainable_only (bool): if True, returns only trainable parameters
            return_arraybox (bool): if True, the returned parameters are not
                wrapped in an ``autograd.builtins.list`` container
        Returns:
            autograd.builtins.list or list: the corresponding parameter values

        **Example**

        .. code-block:: python

            with JacobianTape() as tape:
                qml.RX(0.432, wires=0)
                qml.RY(0.543, wires=0)
                qml.CNOT(wires=[0, 'a'])
                qml.RX(0.133, wires='a')
                qml.expval(qml.PauliZ(wires=[0]))

        By default, all parameters are trainable and will be returned:

        >>> tape.get_parameters()
        [0.432, 0.543, 0.133]

        Setting the trainable parameter indices will result in only the specified
        parameters being returned:

        >>> tape.trainable_params = {1} # set the second parameter as free
        >>> tape.get_parameters()
        [0.543]

        The ``trainable_only`` argument can be set to ``False`` to instead return
        all parameters:

        >>> tape.get_parameters(trainable_only=False)
        [0.432, 0.543, 0.133]
        """
        params = []
        iterator = self.trainable_params if trainable_only else self._par_info

        for p_idx in iterator:
            op = self._par_info[p_idx]["op"]
            op_idx = self._par_info[p_idx]["p_idx"]
            params.append(op.data[op_idx])

        return params if return_arraybox else autograd.builtins.list(params)

    @autograd.extend.primitive
    def _execute(self, params, device):
        # unwrap all NumPy scalar arrays to Python literals
        params = [p.item() if p.shape == tuple() else p for p in params]
        params = autograd.builtins.tuple(params)

        # unwrap constant parameters
        self._all_params_unwrapped = [
            p.numpy() if isinstance(p, np.tensor) else p for p in self._all_parameter_values
        ]

        # evaluate the tape
        self.set_parameters(self._all_params_unwrapped, trainable_only=False)
        res = self.execute_device(params, device=device)
        self.set_parameters(self._all_parameter_values, trainable_only=False)

        if self.is_sampled:
            return res

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

        # The following dictionary caches the Jacobian and Hessian matrices,
        # so that they can be re-used for different vjp/vhp computations
        # within the same backpropagation call.
        # This dictionary will exist in memory when autograd.grad is called,
        # via closure. Once autograd.grad has returned, this dictionary
        # will no longer be in scope and the memory will be freed.
        saved_grad_matrices = {}

        def _evaluate_grad_matrix(p, grad_matrix_fn):
            """Convenience function for generating gradient matrices
            for the given parameter values.

            This function serves two purposes:

            * Avoids duplicating logic surrounding parameter unwrapping/wrapping.

            * Takes advantage of closure, to cache computed gradient matrices via
              the ``saved_grad_matrices`` attribute, to avoid gradient matrices being
              computed multiple redundant times.

              This is particularly useful when differentiating vector-valued QNodes.
              Because Autograd requests the vector-grad matrix product,
              and *not* the full grad matrix, differentiating vector-valued
              functions will result in multiple backward passes.

            Args:
                p (Sequence): quantum tape parameter values use to evaluate the gradient matrix
                grad_matrix_fn (str): Name of the gradient matrix function. Should correspond to an existing
                    tape method. Currently allowed values include ``"jacobian"`` and ``"hessian"``.

                Returns:
                    array[float]: the gradient matrix
            """
            if grad_matrix_fn in saved_grad_matrices:
                return saved_grad_matrices[grad_matrix_fn]

            self.set_parameters(self._all_params_unwrapped, trainable_only=False)
            grad_matrix = getattr(self, grad_matrix_fn)(device, params=p, **self.jacobian_options)
            self.set_parameters(self._all_parameter_values, trainable_only=False)

            saved_grad_matrices[grad_matrix_fn] = grad_matrix
            return grad_matrix

        def gradient_product(dy):
            """Returns the vector-Jacobian product with given
            parameter values p and output gradient dy"""

            if all(np.ndim(p) == 0 for p in params):
                # only flatten dy if all parameters are single values
                dy = dy.flatten()

            @autograd.extend.primitive
            def jacobian(p):
                """Returns the Jacobian for parameters p"""
                return _evaluate_grad_matrix(p, "jacobian")

            def vhp(ans, p):
                def hessian_product(ddy):
                    """Returns the vector-Hessian product with given
                    parameter values p, output gradient dy, and output
                    second-order gradient ddy"""
                    hessian = _evaluate_grad_matrix(p, "hessian")

                    if dy.size > 1:
                        vhp = dy @ ddy @ hessian @ dy.T / np.linalg.norm(dy) ** 2
                    else:
                        vhp = ddy @ hessian
                        vhp = vhp.flatten()

                    return vhp

                return hessian_product

            # register vhp as the backward method of the jacobian function
            autograd.extend.defvjp(jacobian, vhp, argnums=[0])

            vjp = dy @ jacobian(params)
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
