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
import pennylane as qml


unsupported_op = lambda op: op.grad_recipe is None
supported_op = lambda op: op.grad_recipe is not None
trainable_op = lambda op: any(qml.math.requires_grad(p) for p in op.parameters)


def gradient_expand(tape, depth=10):
    """Expand out a tape so that it supports differentiation
    of requested operations.

    This is achieved by decomposing all trainable operations that have
    ``Operation.grad_method=None`` until all resulting operations
    have a defined gradient method.

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


class BatchReduceTransform:
    """Batch-reduce transforms accept a single tape, and return a list of
    tapes and a processing function for computing some quantity.
    The output tapes, when executed and post-processed by the processing function,
    return the quantity of interest.
    """

    def __init__(self, transform, expand=None, execute=None, differentiable=True, permute_qnode_args=False):
        if not callable(transform):
            raise ValueError(
                f"The batch-reduce transform function to register, {transform}, "
                "does not appear to be a valid Python function or callable."
            )

        self.transform = transform
        self.expand = expand
        self.execute = execute
        self.differentiable
        functools.update_wrapper(self, transform)

    def execute(qnode, *targs, **kwargs):
        def wrapper(*args, **kwargs):
            if permute_qnode_args:
                jac = qml.math.stack(qml.transforms.classical_jacobian(qnode)(*args, **kwargs))
                jac = qml.math.reshape(jac, [qnode.qtape.num_params, -1])

                wrt, perm = np.nonzero(qml.math.toarray(jac))
                perm = np.argsort(np.argsort(perm))

            qnode.construct(args, kwargs)
            res = self.__call__(qnode.qtape, *targs, **kwargs)

            if permute_qnode_args:
                res = qml.math.gather(res, perm)
                res = qml.math.gather(qml.math.T(res), perm)

        return wrapper

    def __call__(self, tape, *args, **kwargs):
        if self.expand is not None:
            tape = self.expand(tape)

        tapes, processing_fn = self.transform(tape, *args, **kwargs)

        if execute is not None:
            return processing_fn(execute(tapes))

        return tapes, processing_fn



def gradient_transform(expand_fn=gradient_expand):
    """Decorator for defining quantum gradient transforms.

    Args:
        expand_fn (callable): An expansion function for decomposing input
            tapes prior to applying the gradient transformation. This function
            must have the signature ``expand_fn(tape, depth=None)``, and must return
            a single expanded tape. If not specified, the default expansion
            function is :func:`.gradient_expand`.

    Supported gradient transforms must be of the following form:

    .. code-block:: python

        @quantum_gradient()
        def my_custom_gradient(tape, argnum=None, **kwargs):
            ...
            return gradient_tapes, processing_fn

    where:

    - ``tape`` (*QuantumTape*): the input quantum tape to compute the gradient of

    - ``argnum`` (*int* or *list[int]* or *None*): Which trainable parameters of the tape
      to differentiate with respect to. If not provided, the derivatives with respect to all
      trainable indices of the tape should be returned (``tape.trainable_params``).

    - ``gradient_tapes`` (*List[QuantumTape]*): is a list of output tapes to be evaluated.
      If this list is empty, no quantum evaluations will be made.

    - ``processing_fn`` is a processing function to be applied to the output of the evaluated
      ``gradient_tapes``. It should accept a list of numeric results with length ``len(gradient_tapes)``,
      and return the Jacobian matrix.

    Once defined, the quantum gradient transform can be used as follows:

    >>> tape = my_custom_gradient.expand(tape)  # optional: expand the tape to support the gradient
    >>> gradient_tapes, processing_fn = my_custom_gradient(tape)
    >>> results = dev.batch_execute(gradient_tapes)
    >>> jacobian = processing_fn(results)

    .. note::

        The input tape might have parameters of various types, including
        NumPy arrays, JAX DeviceArrays, and TensorFlow and PyTorch tensors.

        If the gradient transform is written in a autodiff-compatible manner, either by
        using a framework such as Autograd or TensorFlow, or by using ``qml.math`` for
        tensor manipulation, than higher-order derivatives will also be supported.

        Alternatively, you may use the ``tape.unwrap()`` context manager to temporarily
        convert all tape parameters to NumPy arrays and floats:

        >>> with tape.unwrap():
        ...     params = tape.get_parameters()  # list of floats
    """

    def wrapper(fn):
        fn.expand = expand_fn
        return fn

    return wrapper
