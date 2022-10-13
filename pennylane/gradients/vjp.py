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
This module contains functions for computing the vector-Jacobian product
of tapes.
"""
# pylint: disable=no-member, too-many-branches
import numpy as np
import autograd

import pennylane as qml
from pennylane import math


def _convert(jac, dy_row):
    """Utility to convert and cast the jacobian as dy_row."""
    if isinstance(jac, tuple):
        jac_new = []
        for j in jac:
            j_ = math.convert_like(j, dy_row)
            j_ = math.cast(j_, dy_row.dtype)
            jac_new.append(j_)
        jac = tuple(jac_new)
    else:
        jac = math.convert_like(jac, dy_row)
        jac = math.cast(jac, dy_row.dtype)
    return jac


def compute_vjp_single(dy, jac):
    """Convenience function to compute the vector-Jacobian product for a given
    vector of gradient outputs and a Jacobian for a single measurement tape.

    Args:
        dy (tensor_like): vector of gradient outputs
        jac (tensor_like, tuple): Jacobian matrix

    Returns:
        tensor_like: the vector-Jacobian product

    **Examples**

    1. For a single parameter and a single measurement without shape:

    .. code-block:: pycon

        >>> jac = np.array(0.1)
        >>> dy = np.array(2)
        >>> compute_vjp_single(dy, jac)
        np.array(0.2)

    2. For a single parameter and a single measurment with shape:

    .. code-block:: pycon

        >>> jac = np.array([0.1, 0.2])
        >>> dy = np.array([1.0, 1.0])
        >>> compute_vjp_single(dy, jac)
        np.array([0.3])


    3. For multiple parameters (in this case 2 parameters) and a single measurement without shape:

    .. code-block:: pycon

        >>> jac = tuple([np.array(0.1), np.array(0.2)])
        >>> dy = np.array(2)
        >>> compute_vjp_single(dy, jac)
        np.array([0.2, 0.4])

    3. For multiple parameters (in this case 2 parameters) and a single measurement with shape:

    .. code-block:: pycon

        >>> jac = tuple([np.array([0.1, 0.2]), np.array([0.3, 0.4])])
        >>> dy = np.array([1.0, 2.0])
        >>> compute_vjp_single(dy, jac)
        np.array([0.5, 1.1])


    """
    if jac is None:
        return None

    dy_row = math.reshape(dy, [-1])

    if not isinstance(dy_row, np.ndarray):
        jac = _convert(jac, dy_row)

    try:
        if math.allclose(dy, 0):
            # If the dy vector is zero, then the
            # corresponding element of the VJP will be zero.
            if isinstance(jac, tuple):
                num_params = len(jac)
            else:
                num_params = 1
            res = math.convert_like(np.zeros([num_params]), dy)
            return math.cast(res, dy.dtype)
    except (AttributeError, TypeError):
        pass

    # Single measurement with a single param
    if not isinstance(jac, (tuple, autograd.builtins.SequenceBox)):
        # No trainable parameters
        if jac.shape == (0,):
            res = qml.math.zeros((1, 0))
            return res
        # Single measurement with no dimension e.g. expval
        if dy.shape == ():
            jac = math.reshape(qml.math.squeeze(jac), (1, 1))
            res = math.tensordot(jac, dy_row, [[0], [0]])
        # Single measurement with dimension e.g. probs
        else:
            jac = math.reshape(jac, (len(jac), 1))
            res = math.tensordot(jac, dy_row, [[0], [0]])
    # Single measurement with multiple params
    else:
        # No trainable parameters (adjoint)
        if len(jac) == 0:
            res = qml.math.zeros((1, 0))
            return res
        # Single measurement with no dimension e.g. expval
        if dy.shape == ():
            jac = qml.math.reshape(qml.math.stack(jac), (1, len(jac)))
            res = qml.math.tensordot(jac, dy_row, [[0], [0]])

        # Single measurement with dimension e.g. probs
        else:
            jac = qml.math.stack(jac)
            res = qml.math.tensordot(jac, dy_row, [[1], [0]])
    return res


def compute_vjp_multi(dy, jac):
    """Convenience function to compute the vector-Jacobian product for a given
    vector of gradient outputs and a Jacobian for a multiple measurements tape.

    Args:
        dy (tensor_like): vector of gradient outputs
        jac (tensor_like, tuple): Jacobian matrix

    Returns:
        tensor_like: the vector-Jacobian product

    **Examples**

    1. For a single parameter and multiple measurement (one without shape and one with shape):

    .. code-block:: pycon

        >>> jac = tuple([np.array(0.1), np.array([0.3, 0.4])])
        >>> dy = tuple([np.array(1.0), np.array([1.0, 2.0])])
        >>> compute_vjp_multi(dy, jac)
        np.array([1.2])

    2. For multiple parameters (in this case 2 parameters) and multiple measurement (one without shape and one with shape):

    .. code-block:: pycon

        >>> jac = tuple([tuple([np.array(0.1), np.array(0.2)]), tuple([np.array([0.3, 0.4]), np.array([0.5, 0.6])])])
        >>> dy = tuple([np.array(1.0), np.array([1.0, 2.0])])
        >>> compute_vjp_multi(dy, jac)
        np.array([1.2, 1.9])


    """
    if jac is None:
        return None
    # Single parameter
    if not isinstance(jac[0], (tuple, autograd.builtins.SequenceBox)):
        res = []
        for i, elem in enumerate(dy):
            res.append(compute_vjp_single(elem, jac[i]))
        res = qml.math.stack(res)
        res = qml.math.stack([math.sum(res)])
    # Multiple parameters
    else:
        res = []
        for i, elem in enumerate(dy):
            sub_res = []
            for j in jac[i]:
                sub_res.append(qml.math.squeeze(compute_vjp_single(elem, j)))
            res.append(sub_res)
        res = qml.math.stack([qml.math.sum(x) for x in zip(*res)])
    return res


def compute_vjp(dy, jac, num=None):
    """Convenience function to compute the vector-Jacobian product for a given
    vector of gradient outputs and a Jacobian.

    Args:
        dy (tensor_like): vector of gradient outputs
        jac (tensor_like): Jacobian matrix. For an n-dimensional ``dy``
            vector, the first n-dimensions of ``jac`` should match
            the shape of ``dy``.
        num (int): The length of the flattened ``dy`` argument. This is an
            optional argument, but can be useful to provide if ``dy`` potentially
            has no shape (for example, due to tracing or just-in-time compilation).

    Returns:
        tensor_like: the vector-Jacobian product
    """
    if jac is None:
        return None

    dy_row = math.reshape(dy, [-1])

    if num is None:
        num = math.shape(dy_row)[0]

    if not isinstance(dy_row, np.ndarray):
        jac = math.convert_like(jac, dy_row)
        jac = math.cast(jac, dy_row.dtype)

    jac = math.reshape(jac, [num, -1])

    try:
        if math.allclose(dy, 0):
            # If the dy vector is zero, then the
            # corresponding element of the VJP will be zero.
            num_params = jac.shape[1]
            res = math.convert_like(np.zeros([num_params]), dy)
            return math.cast(res, dy.dtype)
    except (AttributeError, TypeError):
        pass
    return math.tensordot(jac, dy_row, [[0], [0]])


def vjp(tape, dy, gradient_fn, gradient_kwargs=None):
    r"""Generate the gradient tapes and processing function required to compute
    the vector-Jacobian products of a tape.

    Consider a function :math:`\mathbf{f}(\mathbf{x})`. The Jacobian is given by

    .. math::

        \mathbf{J}_{\mathbf{f}}(\mathbf{x}) = \begin{pmatrix}
            \frac{\partial f_1}{\partial x_1} &\cdots &\frac{\partial f_1}{\partial x_n}\\
            \vdots &\ddots &\vdots\\
            \frac{\partial f_m}{\partial x_1} &\cdots &\frac{\partial f_m}{\partial x_n}\\
        \end{pmatrix}.

    During backpropagation, the chain rule is applied. For example, consider the
    cost function :math:`h = y\circ f: \mathbb{R}^n \rightarrow \mathbb{R}`,
    where :math:`y: \mathbb{R}^m \rightarrow \mathbb{R}`.
    The gradient is:

    .. math::

        \nabla h(\mathbf{x}) = \frac{\partial y}{\partial \mathbf{f}} \frac{\partial \mathbf{f}}{\partial \mathbf{x}}
        = \frac{\partial y}{\partial \mathbf{f}} \mathbf{J}_{\mathbf{f}}(\mathbf{x}).

    Denote :math:`d\mathbf{y} = \frac{\partial y}{\partial \mathbf{f}}`; we can write this in the form
    of a matrix multiplication:

    .. math:: \left[\nabla h(\mathbf{x})\right]_{j} = \sum_{i=0}^m d\mathbf{y}_i ~ \mathbf{J}_{ij}.

    Thus, we can see that the gradient of the cost function is given by the so-called
    **vector-Jacobian product**; the product of the row-vector :math:`d\mathbf{y}`, representing
    the gradient of subsequent components of the cost function, and :math:`\mathbf{J}`,
    the Jacobian of the current node of interest.

    Args:
        tape (.QuantumTape): quantum tape to differentiate
        dy (tensor_like): Gradient-output vector. Must have shape
            matching the output shape of the corresponding tape.
        gradient_fn (callable): the gradient transform to use to differentiate
            the tape
        gradient_kwargs (dict): dictionary of keyword arguments to pass when
            determining the gradients of tapes

    Returns:
        tensor_like or None: Vector-Jacobian product. Returns None if the tape
        has no trainable parameters.

    **Example**

    Consider the following quantum tape with PyTorch parameters:

    .. code-block:: python

        import torch

        x = torch.tensor([[0.1, 0.2, 0.3],
                          [0.4, 0.5, 0.6]], requires_grad=True, dtype=torch.float64)

        with qml.tape.QuantumTape() as tape:
            qml.RX(x[0, 0], wires=0)
            qml.RY(x[0, 1], wires=1)
            qml.RZ(x[0, 2], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(x[1, 0], wires=1)
            qml.RY(x[1, 1], wires=0)
            qml.RZ(x[1, 2], wires=1)
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=1)

    We can use the ``vjp`` function to compute the vector-Jacobian product,
    given a gradient-output vector ``dy``:

    >>> dy = torch.tensor([1., 1., 1.], dtype=torch.float64)
    >>> vjp_tapes, fn = qml.gradients.vjp(tape, dy, qml.gradients.param_shift)

    Note that ``dy`` has shape ``(3,)``, matching the output dimension of the tape
    (1 expectation and 2 probability values).

    Executing the VJP tapes, and applying the processing function:

    >>> dev = qml.device("default.qubit", wires=2)
    >>> vjp = fn(qml.execute(vjp_tapes, dev, gradient_fn=qml.gradients.param_shift, interface="torch"))
    >>> vjp
    tensor([-1.1562e-01, -1.3862e-02, -9.0841e-03, -1.3878e-16, -4.8217e-01,
             2.1329e-17], dtype=torch.float64, grad_fn=<ViewBackward>)

    The output VJP is also differentiable with respect to the tape parameters:

    >>> cost = torch.sum(vjp)
    >>> cost.backward()
    >>> x.grad
    tensor([[-1.1025e+00, -2.0554e-01, -1.4917e-01],
            [-1.2490e-16, -9.1580e-01,  0.0000e+00]], dtype=torch.float64)
    """
    gradient_kwargs = gradient_kwargs or {}
    num_params = len(tape.trainable_params)

    if num_params == 0:
        # The tape has no trainable parameters; the VJP
        # is simply none.
        return [], lambda _, num=None: None

    try:
        if math.allclose(dy, 0):
            # If the dy vector is zero, then the
            # corresponding element of the VJP will be zero,
            # and we can avoid a quantum computation.

            def func(_, num=None):  # pylint: disable=unused-argument
                res = math.convert_like(np.zeros([num_params]), dy)
                if qml.active_return():
                    multi = len(tape.measurements) > 1
                    if multi:
                        multi_dy = dy[0]
                        return math.cast(res, multi_dy.dtype)
                return math.cast(res, dy.dtype)

            return [], func
    except (AttributeError, TypeError):
        pass

    gradient_tapes, fn = gradient_fn(tape, **gradient_kwargs)

    def processing_fn(results, num=None):
        # postprocess results to compute the Jacobian
        jac = fn(results)
        if qml.active_return():
            multi = len(tape.measurements) > 1
            if multi:
                return compute_vjp_multi(dy, jac)
            return compute_vjp_single(dy, jac)
        return compute_vjp(dy, jac, num=num)

    return gradient_tapes, processing_fn


def batch_vjp(tapes, dys, gradient_fn, reduction="append", gradient_kwargs=None):
    r"""Generate the gradient tapes and processing function required to compute
    the vector-Jacobian products of a batch of tapes.

    Consider a function :math:`\mathbf{f}(\mathbf{x})`. The Jacobian is given by

    .. math::

        \mathbf{J}_{\mathbf{f}}(\mathbf{x}) = \begin{pmatrix}
            \frac{\partial f_1}{\partial x_1} &\cdots &\frac{\partial f_1}{\partial x_n}\\
            \vdots &\ddots &\vdots\\
            \frac{\partial f_m}{\partial x_1} &\cdots &\frac{\partial f_m}{\partial x_n}\\
        \end{pmatrix}.

    During backpropagation, the chain rule is applied. For example, consider the
    cost function :math:`h = y\circ f: \mathbb{R}^n \rightarrow \mathbb{R}`,
    where :math:`y: \mathbb{R}^m \rightarrow \mathbb{R}`.
    The gradient is:

    .. math::

        \nabla h(\mathbf{x}) = \frac{\partial y}{\partial \mathbf{f}} \frac{\partial \mathbf{f}}{\partial \mathbf{x}}
        = \frac{\partial y}{\partial \mathbf{f}} \mathbf{J}_{\mathbf{f}}(\mathbf{x}).

    Denote :math:`d\mathbf{y} = \frac{\partial y}{\partial \mathbf{f}}`; we can write this in the form
    of a matrix multiplication:

    .. math:: \left[\nabla h(\mathbf{x})\right]_{j} = \sum_{i=0}^m d\mathbf{y}_i ~ \mathbf{J}_{ij}.

    Thus, we can see that the gradient of the cost function is given by the so-called
    **vector-Jacobian product**; the product of the row-vector :math:`d\mathbf{y}`, representing
    the gradient of subsequent components of the cost function, and :math:`\mathbf{J}`,
    the Jacobian of the current node of interest.

    Args:
        tapes (Sequence[.QuantumTape]): sequence of quantum tapes to differentiate
        dys (Sequence[tensor_like]): Sequence of gradient-output vectors ``dy``. Must be the
            same length as ``tapes``. Each ``dy`` tensor should have shape
            matching the output shape of the corresponding tape.
        gradient_fn (callable): the gradient transform to use to differentiate
            the tapes
        reduction (str): Determines how the vector-Jacobian products are returned.
            If ``append``, then the output of the function will be of the form
            ``List[tensor_like]``, with each element corresponding to the VJP of each
            input tape. If ``extend``, then the output VJPs will be concatenated.
        gradient_kwargs (dict): dictionary of keyword arguments to pass when
            determining the gradients of tapes

    Returns:
        List[tensor_like or None]: list of vector-Jacobian products. ``None`` elements corresponds
        to tapes with no trainable parameters.

    **Example**

    Consider the following Torch-compatible quantum tapes:

    .. code-block:: python

        x = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], requires_grad=True, dtype=torch.float64)

        def ansatz(x):
            qml.RX(x[0, 0], wires=0)
            qml.RY(x[0, 1], wires=1)
            qml.RZ(x[0, 2], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(x[1, 0], wires=1)
            qml.RY(x[1, 1], wires=0)
            qml.RZ(x[1, 2], wires=1)

        with qml.tape.QuantumTape() as tape1:
            ansatz(x)
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=1)

        with qml.tape.QuantumTape() as tape2:
            ansatz(x)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        tapes = [tape1, tape2]

    Both tapes share the same circuit ansatz, but have different measurement outputs.

    We can use the ``batch_vjp`` function to compute the vector-Jacobian product,
    given a list of gradient-output vectors ``dys`` per tape:

    >>> dys = [torch.tensor([1., 1., 1.], dtype=torch.float64),
    ...  torch.tensor([1.], dtype=torch.float64)]
    >>> vjp_tapes, fn = qml.gradients.batch_vjp(tapes, dys, qml.gradients.param_shift)

    Note that each ``dy`` has shape matching the output dimension of the tape
    (``tape1`` has 1 expectation and 2 probability values --- 3 outputs --- and ``tape2``
    has 1 expectation value).

    Executing the VJP tapes, and applying the processing function:

    >>> dev = qml.device("default.qubit", wires=2)
    >>> vjps = fn(qml.execute(vjp_tapes, dev, gradient_fn=qml.gradients.param_shift, interface="torch"))
    >>> vjps
    [tensor([-1.1562e-01, -1.3862e-02, -9.0841e-03, -1.3878e-16, -4.8217e-01,
              2.1329e-17], dtype=torch.float64, grad_fn=<ViewBackward>),
     tensor([ 1.7393e-01, -1.6412e-01, -5.3983e-03, -2.9366e-01, -4.0083e-01,
              2.1134e-17], dtype=torch.float64, grad_fn=<ViewBackward>)]

    We have two VJPs; one per tape. Each one corresponds to the number of parameters
    on the tapes (6).

    The output VJPs are also differentiable with respect to the tape parameters:

    >>> cost = torch.sum(vjps[0] + vjps[1])
    >>> cost.backward()
    >>> x.grad
    tensor([[-0.4792, -0.9086, -0.2420],
            [-0.0930, -1.0772,  0.0000]], dtype=torch.float64)
    """
    gradient_kwargs = gradient_kwargs or {}
    reshape_info = []
    gradient_tapes = []
    processing_fns = []

    # Loop through the tapes and dys vector
    for tape, dy in zip(tapes, dys):
        g_tapes, fn = vjp(tape, dy, gradient_fn, gradient_kwargs)
        reshape_info.append(len(g_tapes))
        processing_fns.append(fn)
        gradient_tapes.extend(g_tapes)

    def processing_fn(results, nums=None):
        vjps = []
        start = 0

        if nums is None:
            nums = [None] * len(tapes)

        for t_idx in range(len(tapes)):
            # extract the correct results from the flat list
            res_len = reshape_info[t_idx]
            res_t = results[start : start + res_len]
            start += res_len
            # postprocess results to compute the VJP
            vjp_ = processing_fns[t_idx](res_t, num=nums[t_idx])
            if vjp_ is None:
                if reduction == "append":
                    vjps.append(None)
                continue

            if isinstance(reduction, str):
                getattr(vjps, reduction)(vjp_)
            elif callable(reduction):
                reduction(vjps, vjp_)

        return vjps

    return gradient_tapes, processing_fn
