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
"""This module contains utilities for defining custom Hessian transforms,
including a decorator for specifying Hessian expansions."""
import warnings
from string import ascii_letters as ABC

import pennylane as qml
from pennylane.transforms.tape_expand import expand_invalid_trainable


def _process_jacs(jac, qhess):
    """
    Combine the classical and quantum jacobians
    """
    # Check for a Jacobian equal to the identity matrix.
    if not qml.math.is_abstract(jac):
        shape = qml.math.shape(jac)
        is_square = len(shape) == 2 and shape[0] == shape[1]
        if is_square and qml.math.allclose(jac, qml.numpy.eye(shape[0])):
            return qhess if len(qhess) > 1 else qhess[0]

    hess = []
    for qh in qhess:
        if not isinstance(qh, tuple) or not isinstance(qh[0], tuple):
            # single parameter case
            qh = qml.math.expand_dims(qh, [0, 1])
        else:
            # multi parameter case
            qh = qml.math.stack([qml.math.stack(row) for row in qh])

        jac_ndim = len(qml.math.shape(jac))

        # The classical jacobian has shape (num_params, num_qnode_args)
        # The quantum Hessian has shape (num_params, num_params, output_shape)
        # contracting the quantum Hessian with the classical jacobian twice gives
        # a result with shape (num_qnode_args, num_qnode_args, output_shape)

        qh_indices = "ab..."

        # contract the first axis of the jacobian with the first and second axes of the Hessian
        first_jac_indices = f"a{ABC[2:2 + jac_ndim - 1]}"
        second_jac_indices = f"b{ABC[2 + jac_ndim - 1:2 + 2 * jac_ndim - 2]}"

        result_indices = f"{ABC[2:2 + 2 * jac_ndim - 2]}..."
        qh = qml.math.einsum(
            f"{qh_indices},{first_jac_indices},{second_jac_indices}->{result_indices}",
            qh,
            jac,
            jac,
        )

        hess.append(qh)

    return tuple(hess) if len(hess) > 1 else hess[0]


class hessian_transform(qml.batch_transform):  # pragma: no cover
    """Decorator for defining quantum Hessian transforms.

    Quantum Hessian transforms are a specific case of :class:`~.batch_transform`s,
    similar to the :class:`~.gradient_transform`. Hessian transforms compute the
    second derivative of a quantum function.
    All quantum Hessian transforms accept a tape, and output a batch of tapes to
    be independently executed on a quantum device, alongside a post-processing
    function to return the result.

    Args:
        expand_fn (function): An expansion function (if required) to be applied to the
            input tape before the Hessian computation takes place. If not provided,
            the default expansion function simply expands all operations that
            have ``Operation.grad_method=None`` until all resulting operations
            have a defined gradient method.
        differentiable (bool): Specifies whether the Hessian transform is differentiable
            or not. A transform may be non-differentiable if it does not use an autodiff
            framework for its tensor manipulations. In such a case, setting
            ``differentiable=False`` instructs the decorator to mark the output as
            'constant', reducing potential overhead.
        hybrid (bool): Specifies whether classical processing inside a QNode
            should be taken into account when transforming a QNode.

            - If ``True``, and classical processing is detected, the Jacobian of the
              classical processing will be computed and included. When evaluated, the
              returned Hessian will be with respect to the QNode arguments.

            - If ``False``, any internal QNode classical processing will be **ignored**.
              When evaluated, the returned Hessian will be with respect to the **gate**
              arguments, and not the QNode arguments.

    Supported Hessian transforms must be of the following form:

    .. code-block:: python

        @hessian_transform
        def my_custom_hessian(tape, **kwargs):
            ...
            return hessian_tapes, processing_fn

    where:

    - ``tape`` (*QuantumTape*): the input quantum tape to compute the Hessian of

    - ``hessian_tapes`` (*list[QuantumTape]*): is a list of output tapes to be
      evaluated. If this list is empty, no quantum evaluations will be made.

    - ``processing_fn`` is a processing function to be applied to the output of the
      evaluated ``hessian_tapes``. It should accept a list of numeric results with
      length ``len(hessian_tapes)``, and return the Hessian matrix.

    Once defined, the quantum Hessian transform can be used as follows:

    >>> hessian_tapes, processing_fn = my_custom_hessian(tape, *hessian_kwargs)
    >>> res = execute(tapes, dev, interface="autograd", gradient_fn=qml.gradients.param_shift)
    >>> jacobian = processing_fn(res)

    Alternatively, Hessian transforms can be applied directly to QNodes, in which case
    the execution is implicit:

    >>> fn = my_custom_hessian(qnode, *hessian_kwargs)
    >>> fn(weights)  # transformed function takes the same arguments as the QNode
    1.2629730888100839

    .. note::

        The input tape might have parameters of various types, including NumPy arrays,
        JAX Arrays, and TensorFlow and PyTorch tensors.

        If the Hessian transform is written in a autodiff-compatible manner, either by
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
        # Here, we overwrite the QNode execution wrapper in order to take into account
        # that classical processing may be present inside the QNode.
        hybrid = tkwargs.pop("hybrid", self.hybrid)
        argnums = tkwargs.get("argnums", None)

        old_interface = qnode.interface

        _wrapper = super().default_qnode_wrapper(qnode, targs, tkwargs)
        cjac_fn = qml.transforms.classical_jacobian(qnode, argnum=argnums, expand_fn=self.expand_fn)

        def hessian_wrapper(*args, **kwargs):  # pylint: disable=too-many-branches
            if argnums is not None:
                argnums_ = [argnums] if isinstance(argnums, int) else argnums

                params = qml.math.jax_argnums_to_tape_trainable(
                    qnode, argnums_, self.expand_fn, args, kwargs
                )
                argnums_ = qml.math.get_trainable_indices(params)
                kwargs["argnums"] = argnums_

            if not qml.math.get_trainable_indices(args) and not argnums:
                warnings.warn(
                    "Attempted to compute the Hessian of a QNode with no trainable parameters. "
                    "If this is unintended, please add trainable parameters in accordance with "
                    "the chosen auto differentiation framework."
                )
                return ()

            qhess = _wrapper(*args, **kwargs)

            if old_interface == "auto":
                qnode.interface = "auto"

            if not hybrid:
                return qhess

            if len(qnode.tape.measurements) == 1:
                qhess = (qhess,)

            kwargs.pop("shots", False)

            if argnums is None and qml.math.get_interface(*args) == "jax":
                cjac = qml.transforms.classical_jacobian(
                    qnode, argnum=qml.math.get_trainable_indices(args), expand_fn=self.expand_fn
                )(*args, **kwargs)
            else:
                cjac = cjac_fn(*args, **kwargs)

            has_single_arg = False
            if not isinstance(cjac, tuple):
                has_single_arg = True
                cjac = (cjac,)

            # The classical Jacobian for each argument has shape:
            #   (# gate_args, *qnode_arg_shape)
            # The Jacobian needs to be contracted twice with the quantum Hessian of shape:
            #   (*qnode_output_shape, # gate_args, # gate_args)
            # The result should then have the shape:
            #   (*qnode_output_shape, *qnode_arg_shape, *qnode_arg_shape)
            hessians = []
            for jac in cjac:
                if jac is not None:
                    hess = _process_jacs(jac, qhess)
                    hessians.append(hess)

            return hessians[0] if has_single_arg else tuple(hessians)

        return hessian_wrapper
