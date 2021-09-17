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
import pennylane as qml


unsupported_op = lambda op: op.grad_method is None
supported_op = lambda op: op.grad_method is not None
trainable_op = lambda op: any(qml.math.requires_grad(p) for p in op.parameters)


def gradient_expand(tape, depth=10):
    """Expand out a tape so that it supports differentiation
    of requested operations.

    This is achieved by decomposing all trainable operations that have
    ``Operation.grad_method=None`` until all resulting operations
    have a defined gradient method, up to maximum depth ``depth``. Note that this
    might not be possible, in which case the gradient rule will fail to apply.

    Args:
        tape (.QuantumTape): the input tape to expand
        depth (int) : the maximum expansion depth

    Returns:
        .QuantumTape: the expanded tape
    """

    # check if the tape contains unsupported trainable operations
    if any(unsupported_op(op) and trainable_op(op) for op in tape.operations):

        # Define the stopping condition for the expansion
        stop_cond = lambda obj: (
            not isinstance(obj, qml.measure.MeasurementProcess)
            and ((supported_op(obj) and trainable_op(obj)) or not trainable_op(obj))
        )

        return tape.expand(depth=depth, stop_at=stop_cond)

    return tape


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

    def __init__(self, transform_fn, expand_fn=gradient_expand, differentiable=True, hybrid=True):
        self.hybrid = hybrid
        super().__init__(transform_fn, expand_fn=expand_fn, differentiable=differentiable)

    def default_qnode_wrapper(self, qnode, targs, tkwargs):
        # Here, we overwrite the QNode execution wrapper in order
        # to take into account that classical processing may be present
        # inside the QNode.
        hybrid = tkwargs.pop("hybrid", self.hybrid)
        _wrapper = super().default_qnode_wrapper(qnode, targs, tkwargs)
        cjac_fn = qml.transforms.classical_jacobian(qnode)

        def jacobian_wrapper(*args, **kwargs):
            qjac = _wrapper(*args, **kwargs)

            if any(m.return_type is qml.operation.Probability for m in qnode.qtape.measurements):
                qjac = qml.math.squeeze(qjac)

            if not hybrid:
                return qjac

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

            # Classical processing of a single argument is present. Return qjac @ cjac.
            jac = qml.math.squeeze(qml.math.tensordot(qml.math.T(cjac), qjac, [[-1], [-1]]))
            return qml.math.T(jac)

        return jacobian_wrapper
