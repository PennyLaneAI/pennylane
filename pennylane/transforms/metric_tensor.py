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

import numpy as np
import pennylane as qml

from .batch_transform import batch_transform


def expand_fn(tape, approx="block-diag", diag_approx=None, allow_nonunitary=True):
    """Set the metric tensor based on whether non-unitary gates are allowed."""
    # pylint: disable=unused-argument
    if not allow_nonunitary and approx is None:  # pragma: no cover
        return qml.transforms.expand_nonunitary_gen(tape)
    return qml.transforms.expand_multipar(tape)


@functools.partial(batch_transform, expand_fn=expand_fn)
def metric_tensor(tape, approx="block-diag", diag_approx=None, allow_nonunitary=True):
    """Returns a function that computes the block-diagonal approximation of the metric tensor
    of a given QNode or quantum tape.

    .. note::

        Only gates that have a single parameter and define a ``generator`` are supported.
        All other parametrized gates will be decomposed if possible.

    .. warning::

        While ``approx=None`` is a valid input, the full metric tensor is not implemented yet
        but will be added in an upcoming enhancement. Effectively, this means that only
        ``approx="block-diag"`` and ``approx="diag"`` are currently supported.

    Args:
        tape (pennylane.QNode or .QuantumTape): quantum tape or QNode to find the metric tensor of
        approx (str): Which approximation of the metric tensor to compute.

            - If ``None``, the full metric tensor is computed

            - If ``"block-diag"``, the block diagonal approximation is computed, reducing
              the number of evaluated circuits significantly.

            - If ``"diag"``, only the diagonal approximation is computed, slightly
              reducing the classical overhead but not the quantum resources.

        diag_approx (bool): if True, use the diagonal approximation. If ``False``, a
            block diagonal approximation of the metric tensor is computed.
            This keyword argument is deprecated in favor of ``approx`` and will be removed soon
        allow_nonunitary (bool): Whether non-unitary operations are allowed in circuits
            created by the transform. Only relevant if ``approx`` is ``None``
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
    # pylint: disable=unused-argument
    if diag_approx is not None:
        warnings.warn(
            "The keyword argument diag_approx is deprecated. Please use approx='diag' instead.",
            UserWarning,
        )
        if diag_approx:
            approx = "diag"

    if approx in {"diag", "block-diag"}:
        # Only require covariance matrix based transform
        diag_approx = approx == "diag"
        return _metric_tensor_cov_matrix(tape, diag_approx)

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
                "will be used to evaluate the metric tensor.",
                UserWarning,
            )

        qnode = qnode.qnodes.qnodes[0]

    mt_fn = self.default_qnode_wrapper(qnode, targs, tkwargs)

    _expand_fn = lambda tape: self.expand_fn(tape, *targs, **tkwargs)
    cjac_fn = qml.transforms.classical_jacobian(qnode, expand_fn=_expand_fn)

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


def _metric_tensor_cov_matrix(tape, diag_approx):
    """This is the metric tensor method for the block diagonal, using
    the covariance matrix of the generators of each layer."""
    # get the circuit graph
    graph = tape.graph

    metric_tensor_tapes = []
    obs_list = []
    coeffs_list = []
    params_list = []

    for queue, curr_ops, param_idx, _ in graph.iterate_parametrized_layers():
        params_list.append(param_idx)
        coeffs_list.append([])
        obs_list.append([])

        # for each operation in the layer, get the generator
        for op in curr_ops:
            gen, s = op.generator
            w = op.wires
            coeffs_list[-1].append(s)

            # get the observable corresponding to the generator of the current operation
            if isinstance(gen, np.ndarray):
                # generator is a Hermitian matrix
                obs_list[-1].append(qml.Hermitian(gen, w))

            elif issubclass(gen, qml.operation.Observable):
                # generator is an existing PennyLane operation
                obs_list[-1].append(gen(w))

            else:
                raise qml.QuantumFunctionError(
                    "Can't generate metric tensor, generator {}"
                    "has no corresponding observable".format(gen)
                )

        # Create a quantum tape with all operations
        # prior to the parametrized layer, and the rotations
        # to measure in the basis of the parametrized layer generators.
        with tape.__class__() as layer_tape:
            for op in queue:
                qml.apply(op)

            for o in obs_list[-1]:
                o.diagonalizing_gates()

            qml.probs(wires=tape.wires)

        metric_tensor_tapes.append(layer_tape)

    def processing_fn(probs):
        gs = []

        for prob, obs, coeffs in zip(probs, obs_list, coeffs_list):
            # calculate the covariance matrix of this layer
            scale = qml.math.convert_like(np.outer(coeffs, coeffs), prob)
            scale = qml.math.cast_like(scale, prob)
            g = scale * qml.math.cov_matrix(prob, obs, wires=tape.wires, diag_approx=diag_approx)
            gs.append(g)

        # create the block diagonal metric tensor
        return qml.math.block_diag(gs)

    return metric_tensor_tapes, processing_fn
