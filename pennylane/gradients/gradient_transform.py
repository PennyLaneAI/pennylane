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


def gradient_analysis(tape, use_graph=True, grad_fn=None):
    """Update the parameter information dictionary of the tape with
    gradient information of each parameter.

    Parameter gradient methods include:

    * ``None``: the parameter does not support differentiation.

    * ``"0"``: the variational circuit output does not depend on this
      parameter (the partial derivative is zero).

    In addition, the operator might define its own grad method
    via :attr:`.Operator.grad_method`.

    Note that this function modifies the input tape in-place.

    Args:
        tape (.QuantumTape): the quantum tape to analyze
        use_graph (bool): whether to use a directed-acyclic graph to determine
            if the parameter has a gradient of 0
        grad_fn (None or callable): The gradient transform performing the analysis.
            This is an optional argument; if provided, and the tape has already
            been analyzed for the gradient information by the same gradient transform,
            the cached gradient analysis will be used.
    """
    # pylint:disable=protected-access
    if grad_fn is not None and getattr(tape, "_gradient_fn", None) is grad_fn:
        # gradient analysis has already been performed on this tape
        return

    if grad_fn is not None:
        tape._gradient_fn = grad_fn

    for idx, info in enumerate(tape._par_info):

        if idx not in tape.trainable_params:
            # non-trainable parameters do not require a grad_method
            info["grad_method"] = None
        else:
            op = tape._par_info[idx]["op"]

            if not qml.operation.has_grad_method(op):
                # no differentiation method is registered for this operation
                info["grad_method"] = None

            elif (tape._graph is not None) or use_graph:
                if not any(tape.graph.has_path(op, ob) for ob in tape.observables):
                    # there is no influence of this operation on any of the observables
                    info["grad_method"] = "0"
                    continue

            info["grad_method"] = op.grad_method


def grad_method_validation(method, tape):
    """Validates if the gradient method requested is supported by the trainable
    parameters of a tape, and returns the allowed parameter gradient methods.

    This method will generate parameter gradient information for the given tape if it
    has not already been generated, and then proceed to validate the gradient method.
    In particular:

    * An exception will be raised if there exist non-differentiable trainable
      parameters on the tape.

    * An exception will be raised if the Jacobian method is ``"analytic"`` but there
      exist some trainable parameters on the tape that only support numeric differentiation.

    If all validations pass, this method will return a tuple containing the allowed parameter
    gradient methods for each trainable parameter.

    Args:
        method (str): the overall Jacobian differentiation method
        tape (.QuantumTape): the tape with associated parameter information

    Returns:
        tuple[str, None]: the allowed parameter gradient methods for each trainable parameter
    """

    diff_methods = {
        idx: info["grad_method"]
        for idx, info in enumerate(tape._par_info)  # pylint: disable=protected-access
        if idx in tape.trainable_params
    }

    # check and raise an error if any parameters are non-differentiable
    nondiff_params = {idx for idx, g in diff_methods.items() if g is None}

    if nondiff_params:
        raise ValueError(f"Cannot differentiate with respect to parameter(s) {nondiff_params}")

    numeric_params = {idx for idx, g in diff_methods.items() if g == "F"}

    # If explicitly using analytic mode, ensure that all parameters
    # support analytic differentiation.
    if method == "analytic" and numeric_params:
        raise ValueError(
            f"The analytic gradient method cannot be used with the parameter(s) {numeric_params}."
        )

    return tuple(diff_methods.values())


def choose_grad_methods(diff_methods, argnum):
    """Chooses the trainable parameters to use for computing the Jacobian
    by returning a map of their indices and differentiation methods.

    When there are fewer parameters specified than the total number of
    trainable parameters, the Jacobian is estimated by using the parameters
    specified using the ``argnum`` keyword argument.

    Args:
        diff_methods (list): the ordered list of differentiation methods
            for each parameter
        argnum (int, list(int), None): Indices for argument(s) with respect
            to which to compute the Jacobian.

    Returns:
        dict: map of the trainable parameter indices and
        differentiation methods
    """
    if argnum is None:
        return dict(enumerate(diff_methods))

    if isinstance(argnum, int):
        argnum = [argnum]

    num_params = len(argnum)

    if num_params == 0:
        warnings.warn(
            "No trainable parameters were specified for computing the Jacobian.",
            UserWarning,
        )
        return {}

    return {idx: diff_methods[idx] for idx in argnum}


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

    def default_qnode_wrapper(self, qnode, targs, tkwargs):
        # Here, we overwrite the QNode execution wrapper in order
        # to take into account that classical processing may be present
        # inside the QNode.
        hybrid = tkwargs.pop("hybrid", self.hybrid)
        _wrapper = super().default_qnode_wrapper(qnode, targs, tkwargs)
        cjac_fn = qml.transforms.classical_jacobian(qnode, expand_fn=expand_invalid_trainable)

        def jacobian_wrapper(*args, **kwargs):
            if not qml.math.get_trainable_indices(args):
                warnings.warn(
                    "Attempted to compute the gradient of a QNode with no trainable parameters. "
                    "If this is unintended, please add trainable parameters in accordance with "
                    "the chosen auto differentiation framework."
                )
                return ()

            qjac = _wrapper(*args, **kwargs)

            if not hybrid:
                return qjac

            kwargs.pop("shots", False)
            cjac = cjac_fn(*args, **kwargs)

            if isinstance(cjac, tuple):
                # Classical processing of multiple arguments is present. Return qjac @ cjac.
                jacs = tuple(
                    qml.math.tensordot(qjac, c, [[-1], [0]]) for c in cjac if c is not None
                )
                return jacs

            is_square = cjac.ndim == 2 and cjac.shape[0] == cjac.shape[1]

            if is_square and qml.math.allclose(cjac, qml.numpy.eye(cjac.shape[0])):
                # Classical Jacobian is the identity. No classical processing
                # is present inside the QNode.
                return qjac

            return qml.math.tensordot(qjac, cjac, [[-1], [0]])

        return jacobian_wrapper
