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
Contains the metric_tensor batch_transform which wraps multiple
methods of computing the metric tensor.
"""
import functools
import warnings

import pennylane as qml

from pennylane.fourier.qnode_spectrum import expand_multi_par_and_no_gen
from .batch_transform import batch_transform
from .metric_tensor_cov_matrix import metric_tensor_cov_matrix


@functools.partial(batch_transform, expand_fn=expand_multi_par_and_no_gen)
def metric_tensor(tape, approx="block diag", diag_approx=None):
    """Returns a function that computes the block-diagonal approximation of the metric tensor
    of a given QNode or quantum tape.

    .. note::

        Only gates that have a single parameter and define a ``generator``
        are supported.
        All other parametrized gates will be decomposed if possible.

    Args:
        qnode (pennylane.QNode or .QuantumTape): quantum tape or QNode to find the metric tensor of
        approx (str): Which approximation of the metric tensor to compute.

            - If ``None``, the full metric tensor is computed

            - If ``"block diag"``, the block diagonal approximation is computed, reducing
              the number of evaluated circuits significantly.

            - If ``"diag"``, only the diagonal approximation is computed, slightly
              reducing the classical overhead but not the quantum resources.

        diag_approx (bool): if True, use the diagonal approximation. If ``False``, a
            block diagonal approximation of the metric tensor is computed.
            This keyword argument is deprecated in favor of ``approx`` and will be removed soon
        hybrid (bool): Specifies whether classical processing inside a QNode
            should be taken into account when transforming a QNode.

            - If ``True``, and classical processing is detected, the Jacobian of the classical
              processing will be computed and included. When evaluated, the
              returned metric tensor will be with respect to the QNode arguments.

            - If ``False``, any internal QNode classical processing will be
              **ignored**. When evaluated, the returned metric tensor will be with
              respect to the **gate** arguments, and not the QNode arguments.

    Returns:
        func: Function which accepts the same arguments as the QNode. When called, this
        function will return the metric tensor.

    **Example**

    Consider the following QNode:

    .. code-block:: python

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev, interface="autograd")
        def circuit(weights):
            # layer 1
            qml.RX(weights[0], wires=0)
            qml.RX(weights[1], wires=1)

            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])

            # layer 2
            qml.RZ(weights[2], wires=0)
            qml.RZ(weights[3], wires=2)

            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)), qml.expval(qml.PauliY(2))

    We can use the ``metric_tensor`` transform to generate a new function that returns the
    metric tensor of this QNode:

    >>> met_fn = qml.metric_tensor(circuit)
    >>> weights = np.array([0.1, 0.2, 0.4, 0.5], requires_grad=True)
    >>> met_fn(weights)
    tensor([[0.25  , 0.    , 0.    , 0.    ],
            [0.    , 0.25  , 0.    , 0.    ],
            [0.    , 0.    , 0.0025, 0.0024],
            [0.    , 0.    , 0.0024, 0.0123]], requires_grad=True)

    The returned metric tensor is also fully differentiable in all interfaces.
    For example, we can compute the gradient of the ``(3, 2)`` element
    with respect to the QNode ``weights``:

    >>> grad_fn = qml.grad(lambda x: met_fn(x)[3, 2])
    >>> grad_fn(weights)
    array([[ 0.04867729, -0.00049502,  0.        ],
           [ 0.        ,  0.        ,  0.        ]])

    .. UsageDetails::

        This transform can also be applied to low-level
        :class:`~.QuantumTape` objects. This will result in no implicit quantum
        device evaluation. Instead, the processed tapes, and post-processing
        function, which together define the metric tensor are directly returned:

        >>> params = np.array([1.7, 1.0, 0.5], requires_grad=True)
        >>> with qml.tape.QuantumTape() as tape:
        ...     qml.RX(params[0], wires=0)
        ...     qml.RY(params[1], wires=0)
        ...     qml.CNOT(wires=[0, 1])
        ...     qml.PhaseShift(params[2], wires=1)
        ...     qml.expval(qml.PauliX(0))
        >>> tapes, fn = qml.metric_tensor(tape)
        >>> tapes
        [<QuantumTape: wires=[0, 1], params=0>,
         <QuantumTape: wires=[0, 1], params=1>,
         <QuantumTape: wires=[0, 1], params=3>]

        This can be useful if the underlying circuits representing the metric tensor
        computation need to be analyzed.

        The output tapes can then be evaluated and post-processed to retrieve
        the metric tensor:

        >>> dev = qml.device("default.qubit", wires=2)
        >>> fn(qml.execute(tapes, dev, None))
        array([[0.25      , 0.        , 0.        ],
               [0.        , 0.00415023, 0.        ],
               [0.        , 0.        , 0.24878844]])
    """
    tape = metric_tensor.expand_fn(tape)

    if diag_approx is not None:
        warnings.warn(
            "The keyword argument diag_approx is deprecated. Please use approx='diag' instead."
        )
        if diag_approx:
            approx = "diag"

    if approx in {"diag", "block diag"}:
        # Only require covariance matrix based transform
        diag_approx = approx == "diag"
        # Cut off excess output of cov_matrix transform
        return metric_tensor_cov_matrix(tape, diag_approx)

    raise NotImplementedError("No method for the full metric tensor has been implemented yet.")


@metric_tensor.custom_qnode_wrapper
def qnode_execution_wrapper(self, qnode, targs, tkwargs):
    """Here, we overwrite the QNode execution wrapper in order
    to take into account that classical processing may be present
    inside the QNode."""
    hybrid = tkwargs.pop("hybrid", True)

    if isinstance(qnode, qml.ExpvalCost):
        if qnode._multiple_devices:  # pylint: disable=protected-access
            warnings.warn(
                "ExpvalCost was instantiated with multiple devices. Only the first device "
                "will be used to evaluate the metric tensor."
            )

        qnode = qnode.qnodes.qnodes[0]

    mt_fn = self.default_qnode_wrapper(qnode, targs, tkwargs)

    # if isinstance(qnode, qml.beta.QNode):
    cjac_fn = qml.transforms.classical_jacobian(qnode, expand_fn=self.expand_fn)
    # else:
    # cjac_fn = qml.transforms.classical_jacobian(qnode)

    def wrapper(*args, **kwargs):
        mt = mt_fn(*args, **kwargs)

        if not hybrid:
            return mt

        kwargs.pop("shots", False)
        cjac = cjac_fn(*args, **kwargs)

        if isinstance(cjac, tuple):
            if len(cjac) == 1:
                cjac = cjac[0]
            else:
                # Classical processing of multiple arguments is present. Return mt @ cjac.
                metric_tensors = []

                for c in cjac:
                    if c is not None:
                        _mt = qml.math.tensordot(mt, c, [[-1], [0]])
                        _mt = qml.math.tensordot(c, _mt, [[0], [0]])
                        metric_tensors.append(_mt)

                return tuple(metric_tensors)

        is_square = cjac.shape == (1,) or (cjac.ndim == 2 and cjac.shape[0] == cjac.shape[1])

        if is_square and qml.math.allclose(cjac, qml.numpy.eye(cjac.shape[0])):
            # Classical Jacobian is the identity. No classical processing
            # is present inside the QNode.
            return mt

        # Classical processing of a single argument is present. Return mt @ cjac.
        cjac = qml.math.convert_like(cjac, mt)
        mt = qml.math.tensordot(mt, cjac, [[-1], [0]])
        mt = qml.math.tensordot(cjac, mt, [[0], [0]])
        return mt

    return wrapper
