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
Contains the metric tensor transform
"""
import warnings

import numpy as np
import pennylane as qml


def _stopping_critera(obj):
    if getattr(obj, "num_params", 0) == 0:
        return True

    if obj.name in ["RX", "RY", "RZ", "PhaseShift"]:
        return True

    return False


def metric_tensor_tape(tape, diag_approx=False, wrt=None):
    """Returns a list of tapes, and a classical processing function, for computing the block
    diagonal metric tensor approximation of an input tape on hardware.

    Args:
        tape (.QuantumTape): the tape to compute the metric tensor of
        diag_approx (bool): If ``True`` the diagonal approximation to the metric
            tensor is computed. If ``False``, a block diagonal approximation
            to the metric tensor is computed.
        wrt (Sequence[int]): Indices of the tape parameters with which to
            compute the metric tensor. Parameter indices not included are
            treated as *fixed* parameters. Defaults to the tape's trainable
            parameters.

    Returns:
        tuple[list[.QuantumTape], func]: Returns a tuple containing a list of
        quantum tapes to be evaluated, and a function to be applied to these
        tape results to compute the metric tensor.

    **Example**

    Given the following quantum tape,

    .. code-block:: python

        with qml.tape.QuantumTape() as tape:
            # layer 1
            qml.RX(0.1, wires=0)
            qml.RX(0.2, wires=1)

            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])

            # layer 2
            qml.RZ(0.4, wires=0)
            qml.RZ(0.5, wires=2)

            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])

            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
            qml.expval(qml.PauliY(2))

    We can use the ``metric_tensor`` transform to generate a new tapes and a classical
    processing function for computing the metric tensor.

    >>> mt_tapes, fn = qml.transforms.metric_tensor_tape(tape)
    >>> print(mt_tapes)
    [<QuantumTape: wires=[0, 1, 2], params=0>, <QuantumTape: wires=[0, 1, 2], params=2>]
    >>> print(mt_tapes[0].draw())
     0: ──H──╭┤ Probs
     1: ──H──├┤ Probs
     2: ─────╰┤ Probs
    >>> print(mt_tapes[1].draw())
     0: ──RX(0.1)──╭C──────╭┤ Probs
     1: ──RX(0.2)──╰X──╭C──├┤ Probs
     2: ───────────────╰X──╰┤ Probs

    We can evaluate these tapes on a device:

    >>> dev = qml.device("default.qubit", wires=3)
    >>> res = dev.batch_execute(mt_tapes)
    >>> print(res)
    [array([[0.25, 0.  , 0.25, 0.  , 0.25, 0.  , 0.25, 0.  ]]),
     array([[9.87560268e-01, 0.00000000e+00, 0.00000000e+00, 9.94181506e-03,
             2.48960206e-05, 0.00000000e+00, 0.00000000e+00, 2.47302134e-03]])]

    Applying the processing function results in the metric tensor:

    >>> fn(res)
    array([[0.25      , 0.        , 0.        , 0.        ],
           [0.        , 0.25      , 0.        , 0.        ],
           [0.        , 0.        , 0.00249168, 0.00244201],
           [0.        , 0.        , 0.00244201, 0.01226071]])
    """

    # For parametrized operations, only the RX, RY, RZ, and PhaseShift gates are supported.
    # Expand out all other gates.
    tape = tape.expand(depth=2, stop_at=_stopping_critera)

    if wrt is not None:
        tape.trainable_params = set(wrt)

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
                op.queue()

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


def metric_tensor(qnode, diag_approx=False, only_construct=False):
    """Returns a function that returns the value of the metric tensor
    of a given QNode.

    .. note::

        Currently, only the :class:`~.RX`, :class:`~.RY`, :class:`~.RZ`, and
        :class:`~.PhaseShift` parametrized gates are supported.
        All other parametrized gates will be decomposed if possible.

    Args:
        qnode (.QNode or .ExpvalCost): QNode(s) to compute the metric tensor of
        diag_approx (bool): iff True, use the diagonal approximation
        only_construct (bool): Iff True, construct the circuits used for computing
            the metric tensor but do not execute them, and return the tapes.

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
            qml.RX(weights[0, 0], wires=0)
            qml.RX(weights[0, 1], wires=1)

            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])

            # layer 2
            qml.RZ(weights[1, 0], wires=0)
            qml.RZ(weights[1, 1], wires=2)

            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)), qml.expval(qml.PauliY(2))

    We can use the ``metric_tensor`` function to generate a new function, that returns the
    metric tensor of this QNode:

    >>> met_fn = qml.metric_tensor(circuit)
    >>> weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], requires_grad=True)
    >>> met_fn(weights)
    tensor([[0.25  , 0.    , 0.    , 0.    ],
            [0.    , 0.25  , 0.    , 0.    ],
            [0.    , 0.    , 0.0025, 0.0024],
            [0.    , 0.    , 0.0024, 0.0123]], requires_grad=True)

    The returned metric tensor is also fully differentiable, in all interfaces.
    For example, differentiating the ``(3, 2)`` element:

    >>> grad_fn = qml.grad(lambda x: met_fn(x)[3, 2])
    >>> grad_fn(weights)
    array([[ 0.04867729, -0.00049502,  0.        ],
           [ 0.        ,  0.        ,  0.        ]])
    """
    if qnode.__class__.__name__ == "ExpvalCost":
        if qnode._multiple_devices:  # pylint: disable=protected-access
            warnings.warn(
                "ExpvalCost was instantiated with multiple devices. Only the first device "
                "will be used to evaluate the metric tensor."
            )

        qnode = qnode.qnodes.qnodes[0]

    def _metric_tensor_fn(*args, **kwargs):
        jac = qml.math.stack(qml.transforms.classical_jacobian(qnode)(*args, **kwargs))
        jac = qml.math.reshape(jac, [qnode.qtape.num_params, -1])

        wrt, perm = np.nonzero(qml.math.toarray(jac))
        perm = np.argsort(np.argsort(perm))

        qnode.construct(args, kwargs)

        metric_tensor_tapes, processing_fn = metric_tensor_tape(
            qnode.qtape,
            diag_approx=diag_approx,
            wrt=wrt.tolist() if qnode.diff_options["method"] == "backprop" else None,
        )

        if only_construct:
            return metric_tensor_tapes

        res = [t.execute(device=qnode.device) for t in metric_tensor_tapes]
        mt = processing_fn(res)

        # permute rows ad columns
        mt = qml.math.gather(mt, perm)
        mt = qml.math.gather(qml.math.T(mt), perm)
        return mt

    return _metric_tensor_fn
