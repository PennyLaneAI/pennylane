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
"""This module contains utilities for defining custom gradient transforms,
including a decorator for specifying gradient expansions."""
# pylint: disable=too-few-public-methods
import warnings

import pennylane as qml
from pennylane.transforms.tape_expand import expand_invalid_trainable


class gradient_transform(qml.batch_transform):
    """Decorator for defining quantum gradient transforms.

    Quantum gradient transforms are a specific case of :class:`~.batch_transform`.
    All quantum gradient transforms accept a tape, and output
    a batch of tapes to be independently executed on a quantum device, alongside
    a post-processing function that returns the result.

    Args:
        expand_fn (function): An expansion function (if required) to be applied to the
            input tape before the gradient computation takes place. If not provided,
            the default expansion function simply expands all operations that
            have ``Operation.grad_method=None`` until all resulting operations
            have a defined gradient method.
        differentiable (bool): Specifies whether the gradient transform is differentiable or
            not. A transform may be non-differentiable if it does not use an
            autodiff framework for its tensor manipulations. In such a case, setting
            ``differentiable=False`` instructs the decorator
            to mark the output as 'constant', reducing potential overhead.
        hybrid (bool): Specifies whether classical processing inside a QNode
            should be taken into account when transforming a QNode.

            - If ``True``, and classical processing is detected and this
              option is set to ``True``, the Jacobian of the classical
              processing will be computed and included. When evaluated, the
              returned Jacobian will be with respect to the QNode arguments.

            - If ``False``, any internal QNode classical processing will be
              **ignored**. When evaluated, the returned Jacobian will be with
              respect to the **gate** arguments, and not the QNode arguments.

    Supported gradient transforms must be of the following form:

    .. code-block:: python

        @gradient_transform
        def my_custom_gradient(tape, argnum=None, **kwargs):
            ...
            return gradient_tapes, processing_fn

    where:

    - ``tape`` (*QuantumTape*): the input quantum tape to compute the gradient of

    - ``argnum`` (*int* or *list[int]* or *None*): Which trainable parameters of the tape
      to differentiate with respect to. If not provided, the derivatives with respect to all
      trainable inputs of the tape should be returned (``tape.trainable_params``).

    - ``gradient_tapes`` (*List[QuantumTape]*): is a list of output tapes to be evaluated.
      If this list is empty, no quantum evaluations will be made.

    - ``processing_fn`` is a processing function to be applied to the output of the evaluated
      ``gradient_tapes``. It should accept a list of numeric results with length ``len(gradient_tapes)``,
      and return the Jacobian matrix.

    Once defined, the quantum gradient transform can be used as follows:

    >>> gradient_tapes, processing_fn = my_custom_gradient(tape, *gradient_kwargs)
    >>> res = execute(tapes, dev, interface="autograd", gradient_fn=qml.gradients.param_shift)
    >>> jacobian = processing_fn(res)

    Alternatively, gradient transforms can be applied directly to QNodes,
    in which case the execution is implicit:

    >>> fn = my_custom_gradient(qnode, *gradient_kwargs)
    >>> fn(weights) # transformed function takes the same arguments as the QNode
    1.2629730888100839

    .. note::

        The input tape might have parameters of various types, including
        NumPy arrays, JAX DeviceArrays, and TensorFlow and PyTorch tensors.

        If the gradient transform is written in a autodiff-compatible manner, either by
        using a framework such as Autograd or TensorFlow, or by using ``qml.math`` for
        tensor manipulation, then higher-order derivatives will also be supported.

        Alternatively, you may use the ``tape.unwrap()`` context manager to temporarily
        convert all tape parameters to NumPy arrays and floats:

        >>> with tape.unwrap():
        ...     params = tape.get_parameters()  # list of floats
    """

    def __init__(
        self, transform_fn, expand_fn=expand_invalid_trainable, differentiable=True, hybrid=True
    ):
        self.hybrid = hybrid
        super().__init__(transform_fn, expand_fn=expand_fn, differentiable=differentiable)

    @staticmethod
    def _jacobian_trainable_args(args, interface):
        """Return the indices of QNode arguments for which a Jacobian was
        computed by `qml.transforms.classical_jacobian` with argnum=None.
        """
        trainable_args = []

        if interface == "autograd":
            for idx, arg in enumerate(args):
                # TODO: make default False once this change is done in qml.jacobian
                if getattr(arg, "requires_grad", True):
                    trainable_args.append(idx)

        elif interface == "jax":
            trainable_args = [0]

        # Torch and Tensorflow interfaces are not considered since `classical_jacobian`
        # always returns a tuple for them, thus not invoking this function.

        return trainable_args

    def default_qnode_wrapper(self, qnode, targs, tkwargs):
        # Here, we overwrite the QNode execution wrapper in order
        # to take into account that classical processing may be present
        # inside the QNode.
        hybrid = tkwargs.pop("hybrid", self.hybrid)
        _wrapper = super().default_qnode_wrapper(qnode, targs, tkwargs)
        cjac_fn = qml.transforms.classical_jacobian(qnode, expand_fn=expand_invalid_trainable)

        def jacobian_wrapper(*args, **kwargs):
            qjac = _wrapper(*args, **kwargs)

            if any(m.return_type is qml.operation.Probability for m in qnode.qtape.measurements):
                qjac = qml.math.squeeze(qjac)

            if not hybrid:
                return qjac

            kwargs.pop("shots", False)
            cjac = cjac_fn(*args, **kwargs)

            if isinstance(cjac, tuple):
                # Classical processing of multiple arguments is present. Return qjac @ cjac.
                jacs = [
                    qml.math.squeeze(qml.math.tensordot(c, qjac, [[0], [-1]]))
                    for c in cjac
                    if c is not None
                ]
                return jacs

            is_square = cjac.shape == (1,) or (cjac.ndim == 2 and cjac.shape[0] == cjac.shape[1])

            if is_square and qml.math.allclose(cjac, qml.numpy.eye(cjac.shape[0])):
                # Classical Jacobian is the identity. No classical processing
                # is present inside the QNode.
                return qjac

            # Classical processing present of either:
            #   a) a single argument or
            #   b) multiple arguments of the same shape
            # The shape of the classical jacobian returned by qml.jacobian (invoked by
            # classical_jacobian with autograd) depends on the scenario:
            #   a) (# gate args, qnode arg shape)
            #   b) (reverse qnode arg shape, # gate args, # qnode args)
            # It then needs to be contracted with the quantum jacobian of shape:
            #      (qnode output shape, # gate args)
            # The result should have shape:
            #   a) (qnode ouput shape, qnode arg shape)
            #   b) (reverse qnode arg shape, qnode output shape, # qnode args)
            num_gate_args = qml.math.shape(qjac)[-1]
            # Consider only QNode arguments regarded as trainable by the interfaces.
            trainable_args_idx = self._jacobian_trainable_args(args, qnode.interface)
            # Since all arguments have the same shape, obtain shape from the first trainable arg
            qnode_arg_shape = qml.math.shape(args[trainable_args_idx[0]])
            num_qnode_args = len(trainable_args_idx)

            if qml.math.shape(cjac) == (num_gate_args, *qnode_arg_shape):
                # single QNode argument
                jac = qml.math.tensordot(qjac, cjac, [[-1], [0]])
            elif qml.math.shape(cjac) == (*qnode_arg_shape[::-1], num_gate_args, num_qnode_args):
                # multiple QNode arguments with stacking
                cjac = qml.math.swapaxes(cjac, -1, -2)
                jac = qml.math.tensordot(cjac, qjac, [[-1], [-1]])
                jac = qml.math.moveaxis(jac, -1, len(qnode_arg_shape))
            else:  # pragma: no cover
                jac = ()
                warnings.warn(
                    "Unexpected classical Jacobian encoutered, could not compute the hybrid "
                    "Jacobian of the QNode. You can still attempt to obtain the quantum Jacobian "
                    "with the `hybrid=False` parameter.",
                    UserWarning,
                )

            return qml.math.squeeze(jac)

        return jacobian_wrapper
