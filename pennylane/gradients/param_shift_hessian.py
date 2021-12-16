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
This module contains functions for computing the parameter-shift hessian
of a qubit-based quantum tape.
"""
import warnings
import numpy as np

import pennylane as qml

from pennylane.transforms.tape_expand import expand_invalid_trainable
from .parameter_shift import _gradient_analysis


def _process_gradient_recipe(gradient_recipe, tol=1e-10):
    """Utility function to process gradient recipes."""

    gradient_recipe = np.array(gradient_recipe).T
    # remove all small coefficients, shifts, and multipliers
    gradient_recipe[np.abs(gradient_recipe) < tol] = 0
    # remove columns where the coefficients are 0
    gradient_recipe = gradient_recipe[:, :, ~(gradient_recipe[0, 0] == 0)]
    # sort columns according to abs(shift2) then abs(shift1)
    gradient_recipe = gradient_recipe[:, :, np.lexsort(np.abs(gradient_recipe)[:, -1])]

    return gradient_recipe[0, 0], gradient_recipe[0, 1], gradient_recipe[:, 2].T


def generate_multishifted_tapes(tape, idx, shifts):
    r"""Generate a list of tapes where the corresponding trainable parameter
    indices have been shifted by the values given.

    Args:
        tape (.QuantumTape): input quantum tape
        idx (list[int]): trainable parameter indices to shift the parameters of
        shifts (list[list[float or int]]): nested list of shift values, each
            list containing a value for each index

    Returns:
        list[QuantumTape]: List of quantum tapes. Each tape has multiple parameters
            (indicated by ``idx``) shifted by the values of ``shifts``. The length
            of the returned list of tapes will match the length of ``shifts``.
    """
    params = list(tape.get_parameters())
    tapes = []

    for shift in shifts:
        new_params = params.copy()
        shifted_tape = tape.copy(copy_operations=True)

        for i, s in enumerate(shift):
            new_params[idx[i]] = new_params[idx[i]] + qml.math.convert_like(s, new_params[idx[i]])

        shifted_tape.set_parameters(new_params)
        tapes.append(shifted_tape)

    return tapes


class hessian_transform(qml.batch_transform):
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

    - ``hessian_tapes`` (*List[QuantumTape]*): is a list of output tapes to be
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

    >>> fn = my_custom_gradient(qnode, *gradient_kwargs)
    >>> fn(weights)  # transformed function takes the same arguments as the QNode
    1.2629730888100839

    .. note::

        The input tape might have parameters of various types, including NumPy arrays,
        JAX DeviceArrays, and TensorFlow and PyTorch tensors.

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

        def hessian_wrapper(*args, **kwargs):
            qhess = _wrapper(*args, **kwargs)

            if any(m.return_type is qml.operation.Probability for m in qnode.qtape.measurements):
                qhess = qml.math.squeeze(qhess)

            if not hybrid:
                return qhess

            kwargs.pop("shots", False)
            cjac = cjac_fn(*args, **kwargs)

            if isinstance(cjac, tuple):
                # Classical processing of multiple arguments is present. Return cjac.T @ hess @ cjac
                jacs = [
                    qml.math.squeeze(qml.math.tensordot(c, qhess, [[0], [-1]]))
                    for c in cjac
                    if c is not None
                ]
                return jacs

            is_square = cjac.shape == (1,) or (cjac.ndim == 2 and cjac.shape[0] == cjac.shape[1])

            if is_square and qml.math.allclose(cjac, qml.numpy.eye(cjac.shape[0])):
                # Classical Jacobian is the identity. No classical processing
                # is present inside the QNode.
                return qhess

            # Classical processing present of either:
            #   a) a single argument or
            #   b) multiple arguments of the same shape
            # The shape of the classical jacobian returned by qml.jacobian (gets invoked by
            # qml.transforms.classical_jacobian with autograd) depends on the scenario:
            #   a) (# gate args, qnode arg shape)
            #   b) (reverse qnode arg shape, # gate args, # qnode args)
            # It then needs to be contracted twice (cjac.T @ hess @ cjac for scalar-valued QNodes)
            # with the quantum Hessian of shape:
            #      (qnode output shape, # gate args, # gate args)
            # The result should have shape:
            #   a) (qnode output shape, qnode arg shape, qnode arg shape)
            #   b) (reverse qnode arg shape A, # qnode args B, qnode output shape, qnode arg shape B, # qnode args A)
            num_gate_args = qml.math.shape(qhess)[-1]
            # Consider only QNode arguments regarded as trainable by the interfaces.
            trainable_args_idx = self._jacobian_trainable_args(args, qnode.interface)
            # Since all arguments have the same shape, obtain shape from the first trainable arg
            qnode_arg_shape = qml.math.shape(args[trainable_args_idx[0]])
            num_qnode_args = len(trainable_args_idx)

            if qml.math.shape(cjac) == (num_gate_args, *qnode_arg_shape):
                # single QNode argument
                num_arg_dims = len(cjac.shape) - 1  # number of dimensions in the QNode output
                hess = qml.math.tensordot(qhess, cjac, [[-1], [0]])  # -> (qnode output shape, # gate args, qnode arg shape A)
                hess = qml.math.tensordot(hess, cjac, [[-1 - num_arg_dims], [0]])  # -> (qnode output shape, qnode arg shape A, qnode arg shape B)
                for i in range(num_arg_dims):
                    hess = qml.math.swapaxes(hess, -1 - i, -1 - num_arg_dims - i)  # -> (qnode output shape, qnode arg shape B, qnode arg shape A)
            elif qml.math.shape(cjac) == (*qnode_arg_shape[::-1], num_gate_args, num_qnode_args):
                # multiple QNode arguments with stacking
                num_out_dims = len(cjac.shape) - 2  # number of dimensions in the QNode output
                hess = qml.math.tensordot(cjac, qhess, [[-2], [-2]])  # -> (reverse qnode output shape A, # qnode args A, qnode output shape, # gate args)
                cjac = qml.math.moveaxis(qml.math.transpose(cjac), 0, -1)  # (# gate args, qnode arg shape B, # qnode args B)
                hess = qml.math.tensordot(hess, cjac, [[-1], [0]])  # -> (reverse qnode output shape A, # qnode args A, qnode output shape, qnode arg shape B, # qnode args B)
                hess = qml.math.swapaxes(hess, num_out_dims, -1)  # -> (reverse qnode output shape A, # qnode args B, qnode output shape, qnode arg shape B, # qnode args A)
            else:  # pragma: no cover
                hess = ()
                warnings.warn(
                    "Unexpected classical Jacobian encoutered, could not compute the hybrid "
                    "Hessian of the QNode. You can still attempt to obtain the quantum Hessian "
                    "with the `hybrid=False` parameter.",
                    UserWarning,
                )

            return hess

        return hessian_wrapper


def compute_hessian_tapes(tape, diff_methods, f0=None):
    r"""Generate the Hessian tapes that are used in the computation of the second derivative of a
    quantum tape, using analytical parameter-shift rules to do so exactly. Also define a
    post-processing function to combine the results of evaluating the gradient tapes.

    Args:
        tape (.QuantumTape): input quantum tape
        diff_methods (list[string]): The gradient method to use for each trainable parameter.
            Can be "A" or "0", where "A" is the analytical parameter shift rule and "0" indicates
            a 0 gradient (that is the parameter does not affect the tapes output).
        f0 (tensor_like[float] or None): Output of the evaluated input tape. If provided,
            and the gradient recipe contains an unshifted term, this value is used,
            saving a quantum evaluation.

    Returns:
        tuple[list[QuantumTape], function]: A tuple containing a list of generated tapes, in
            addition to a post-processing function to be applied to the results of the evaluated
            tapes.
    """
    h_dim = tape.num_params

    gradient_tapes = []
    gradient_coeffs = []
    unshifted_coeffs = {}
    shapes = []

    # The Hessian for a 2-term parameter-shift rule can be expressed via the following recipes.
    # Off-diagonal elements of the Hessian require shifts to two different parameter indices.
    # A recipe can thus be expressed via the tape patterns:
    #       [[coeff, dummy], [mult, dummy], [shift1, shift2]]    (dummy values for ndarray creation)
    # Each corresponding to one term in the parameter-shift formula:
    #       didj f(x) = coeff * f(mult*x + shift1*ei + shift2*ej) + ...
    diag_recipe = [[[0.5], [1], [np.pi]], [[-0.5], [1], [0]]]
    off_diag_recipe = [
        [[0.25, 1], [1, 1], [np.pi / 2, np.pi / 2]],
        [[-0.25, 1], [1, 1], [-np.pi / 2, np.pi / 2]],
        [[-0.25, 1], [1, 1], [np.pi / 2, -np.pi / 2]],
        [[0.25, 1], [1, 1], [-np.pi / 2, -np.pi / 2]],
    ]

    # for now assume all operations support the 2-term parameter shift rule
    for i in range(h_dim):
        for j in range(h_dim):
            # optimization: only generate tapes for upper triangular matrix (j >= i)
            # optimization: skip partial derivates that are zero
            if j < i or diff_methods[i] == "0" or diff_methods[j] == "0":
                gradient_coeffs.append([])
                shapes.append(((i, j), 0))
                continue

            recipe = diag_recipe if i == j else off_diag_recipe
            coeffs, _, shifts = _process_gradient_recipe(recipe)

            # optimization: only compute the unshifted tape once
            if all(np.array(shifts[0]) == 0):
                if not unshifted_coeffs and f0 is None:
                    gradient_tapes.insert(0, tape)

                unshifted_coeffs[(i, j)] = coeffs[0]
                coeffs, shifts = coeffs[1:], shifts[1:]

            # generate the gradient tapes
            gradient_coeffs.append(coeffs)
            g_tapes = generate_multishifted_tapes(tape, (i, j), shifts)

            gradient_tapes.extend(g_tapes)
            shapes.append(((i, j), len(g_tapes)))

    def processing_fn(results):
        # The first results dimension is the number of terms/tapes in the parameter-shift
        # rule, the remaining ones are the QNode output dimensions.
        out_dim = qml.math.shape(results)[1:]
        # The desired shape of the Hessian is:
        #       (QNode output dimensions, # trainable gate args, # trainable gate args),
        # but first we accumulate all elements into a list, since no array assingment is possible.
        hessian = []
        # Keep track of tape results already consumed.
        start = 1 if unshifted_coeffs and f0 is None else 0
        # Results of the unshifted tape.
        r0 = results[0] if start == 1 else f0

        for k, ((i, j), s) in enumerate(shapes):
            res = results[start : start + s]
            start = start + s

            # Compute the elements of the Hessian as the linear combination of
            # results and coefficients, barring optimization cases.
            if j < i:
                g = hessian[j * h_dim + i]
            elif s == 0:
                g = qml.math.zeros(out_dim)
            else:
                res = qml.math.stack(res)
                g = qml.math.tensordot(
                    res, qml.math.convert_like(gradient_coeffs[k], res), [[0], [0]]
                )
                if (i, j) in unshifted_coeffs:
                    g += unshifted_coeffs[(i, j)] * r0

            hessian.append(g)

        # Reshape the Hessian to have the dimensions of the QNode output on the outside, that is:
        #         (h_dim, h_dim, out_dim) -> (out_dim, h_dim, h_dim)
        hessian = qml.math.reshape(qml.math.stack(hessian), (h_dim, h_dim) + out_dim)
        dim_indices = list(range(len(out_dim) + 2))
        hessian = qml.math.transpose(hessian, axes=dim_indices[2:] + [0, 1])

        return qml.math.squeeze(hessian)

    return gradient_tapes, processing_fn


@hessian_transform
def param_shift_hessian(tape, f0=None):
    r"""Transform a QNode to compute the parameter-shift Hessian with respect to its trainable
    parameters.

    Args:
        tape (pennylane.QNode or .QuantumTape): quantum tape or QNode to differentiate
        f0 (tensor_like[float] or None): Output of the evaluated input tape. If provided,
            and the gradient recipe contains an unshifted term, this value is used,
            saving a quantum evaluation.

    Returns:
        tensor_like or tuple[list[QuantumTape], function]:

        - If the input is a QNode, a tensor representing the output of the hybrid Hessian matrix
          of size ``(QNode output dimensions, QNode input dimensions, QNode input dimensions)``
          is returned. When the keyword ``hybrid=False`` is specified, the purely quantum Hessian
          matrix is returned instead, with the dimesions
          ``(QNode output dimensions, number of gate arguments, number of gate arguments)``.
          The difference between the two accounts for the mapping of QNode arguments
          to the actual gate arguments, which can include classical computations.

        - If the input is a tape, a tuple containing a list of generated tapes, in addition
          to a post-processing function to be applied to the evaluated tapes.
    """

    # perform gradient method validation
    if any(m.return_type is qml.operation.State for m in tape.measurements):
        raise ValueError(
            "Computing the gradient of circuits that return the state is not supported."
        )

    # The parameter-shift Hessian implementation currently doesn't support variance measurements.
    if any(m.return_type is qml.operation.Variance for m in tape.measurements):
        raise ValueError(
            "Computing the gradient of circuits that return variances is currently not supported."
        )

    if not tape.trainable_params:
        return [], lambda _: []

    # The parameter-shift Hessian implementation currently only supports
    # the two-term parameter-shift rule. Raise an error for unsupported operations.
    supported_ops = (
        "RX",
        "RY",
        "RZ",
        "Rot",
        "PhaseShift",
        "ControlledPhaseShift",
        "MultiRZ",
        "PauliRot",
        "U1",
        "U2",
        "U3",
        "SingleExcitationMinus",
        "SingleExcitationPlus",
        "DoubleExcitationMinus",
        "DoubleExcitationPlus",
        "OrbitalRotation",
    )

    for idx in range(tape.num_params):
        op, _ = tape.get_operation(idx)
        if op.name not in supported_ops:
            raise ValueError(
                f"The operation {op.name} is currently not supported for the parameter-shift "
                f"Hessian. Only two-term parameter shift rules are currently supported."
            )

    _gradient_analysis(tape)
    diff_methods = tape._grad_method_validation("analytic")  # pylint: disable=protected-access

    return compute_hessian_tapes(tape, diff_methods, f0)
