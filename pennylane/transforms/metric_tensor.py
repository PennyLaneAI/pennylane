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

_GEN_TO_CGEN = {
    qml.PauliX: qml.CNOT,
    qml.PauliY: qml.CY,
    qml.PauliZ: qml.CZ,
}

_OP_TO_CGEN = {
    # PhaseShift is the same as RZ up to a global phase
    qml.PhaseShift: qml.CZ,
}

def expand_fn(tape, approx="block-diag", diag_approx=None, allow_nonunitary=True):
    """Set the metric tensor based on whether non-unitary gates are allowed."""
    # pylint: disable=unused-argument
    if not allow_nonunitary and approx is None:  # pragma: no cover
        return qml.transforms.expand_nonunitary_gen(tape)
    return qml.transforms.expand_multipar(tape)


@functools.partial(batch_transform, expand_fn=expand_fn)
def metric_tensor(
    tape, approx="block-diag", diag_approx=None, allow_nonunitary=True, aux_wire=None
):
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
        aux_wire (int or pennylane.wires.Wires): Auxiliary wire to be used for
            Hadamard tests. Defaults to ``tape.num_wires`` if not provided
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

    The block diagonal part of the metric tensor always is computed using the
    covariance-based approach implemented in :func:`.metric_tensor_cov_matrix`.
    If no approximation is selected, the block off-diagonal is computed using
    Hadamard tests.

    .. warning::

        Executing the tapes with the Hadamard tests requires a device
        that has an additional wire as compared to the wires on which the
        original tape was defined. This wire may be specified via ``aux_wire``.
        By default, contiguous wire numbering is assumed and the additional
        wire is set to ``tape.num_wires``.

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
        return _metric_tensor_cov_matrix(tape, diag_approx)[:2]

    return _metric_tensor_hadamard(tape, allow_nonunitary, aux_wire)


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

    return metric_tensor_tapes, processing_fn, obs_list, coeffs_list


@functools.lru_cache
def _get_gen_op(op, allow_nonunitary, aux_wire):
    """Get the controlled-generator operation for a given operation.

    Args:
        op (pennylane.operation.Operation): Operation from which to extract the generator
        allow_nonunitary (bool): Whether non-unitary gates are allowed in the circuit
        aux_wire (int or pennylane.wires.Wires): Auxiliary wire on which to control the operation

    Returns
        qml.Operation: Controlled-generator operation of the generator of ``op``, controlled
        on wire ``aux_wire``.

    Raises
        ValueError: If the generator of ``op`` is not known or it is non-unitary while
        ``allow_nonunitary=False``.

    If ``allow_nonunitary=True``, a general :class:`~.pennylane.ControlledQubitUnitary` is returned,
    otherwise only controlled Pauli operations are used. If the operation has a non-unitary
    generator but ``allow_nonunitary=False``, the operation ``op`` should have been decomposed
    before, leading to a ``ValueError``.
    """
    gen, _ = op.generator
    try:
        if isinstance(gen, np.ndarray) or gen not in _GEN_TO_CGEN:
            cgen = _OP_TO_CGEN[op.__class__]
        else:
            cgen = _GEN_TO_CGEN.get(gen, None)
        return cgen(wires=[aux_wire, *op.wires])

    except KeyError as e:
        if allow_nonunitary:
            if issubclass(gen, qml.operation.Observable):
                gen = gen.matrix
            return qml.ControlledQubitUnitary(gen, control_wires=aux_wire, wires=op.wires)
        raise ValueError(
            f"Generator for operation {op.__name__} not known and non-unitary operations "
            "deactivated via allow_nonunitary=False."
        ) from e


def _get_first_term_tapes(tape, layer_i, layer_j, allow_nonunitary, aux_wire):
    """Obtain the tapes for the first term of all tensor entries
    belonging to an off-diagonal block.

    Args:
        tape (pennylane.tape.QuantumTape): Tape that is being transformed
        layer_i (list): The first layer of parametrized ops, of the format of
            the layers generated by ``iterate_parametrized_layers``
        layer_j (list): The second layer of parametrized ops
        allow_nonunitary (bool): Whether non-unitary operations are allowed
            in the circuit; passed to ``_get_gen_op``
        aux_wire (int or pennylane.wires.Wires): Auxiliary wire on which to
            control the controlled-generator operations

    Returns:
        list[pennylane.tape.QuantumTape]: Transformed tapes that compute the
            first term of the metric tensor for the off-diagonal block belonging
            to the input layers
        list[tuple[int]]: 2-tuple indices assigning the tapes to metric tensor
            entries
    """

    tapes = []
    ids = []
    # Exclude the backwards cone of layer_i from the backwards cone of layer_j
    ops_between_cgens = [op for op in layer_j[0] if op not in layer_i[0]]
    # Iterate over differentiated operation in first layer
    for diffed_op_i, par_idx_i in zip(*layer_i[1:3]):
        gen_op_i = _get_gen_op(diffed_op_i, aux_wire, allow_nonunitary)
        # Iterate over differentiated operation in second layer
        # There will be one tape per pair of differentiated operations
        for diffed_op_j, par_idx_j in zip(*layer_j[1:3]):
            gen_op_j = _get_gen_op(diffed_op_j, aux_wire, allow_nonunitary)
            with tape.__class__() as new_tape:
                # Initialize auxiliary wire
                qml.Hadamard(wires=aux_wire)
                # Apply backward cone of first layer
                for op in layer_i[0]:
                    qml.apply(op)
                # Controlled-generator operation of first diff'ed op
                qml.apply(gen_op_i)
                # Apply first layer and operations between layers
                for op in ops_between_cgens:
                    qml.apply(op)
                # Controlled-generator operation of first diff'ed op
                qml.apply(gen_op_j)
                # Measure auxiliary wire
                qml.expval(qml.PauliX(aux_wire))
            tapes.append(new_tape)
            # Memorize to which metric entry this tape belongs
            ids.append((par_idx_i, par_idx_j))

    return tapes, ids


def _metric_tensor_hadamard(tape, allow_nonunitary, aux_wire):
    r"""Generate the quantum tapes that execute the Hadamard tests
    to compute the first term of block off-diagonal metric entries
    and combine them with the covariance matrix-based block diagonal tapes.

    Args:
        tape (pennylane.tape.QuantumTape): Tape that is being transformed
        allow_nonunitary (bool): Whether non-unitary operations are allowed
            in the circuit; passed to ``_get_gen_op``
        aux_wire (int or pennylane.wires.Wires): Auxiliary wire on which to
            control the controlled-generator operations. Defaults to
            ``tape.num_wires`` if not provided

    Returns:
        list[pennylane.tape.QuantumTape]: Tapes to evaluate the metric tensor
        callable: processing function to obtain the metric tensor from the tape results

    .. warning::

        This method requires the device used to execute the returned tapes to
        have an additional wire. This wire may be specified via ``aux_wire``.
        By default contiguous numbering and an additional wire at the end
        of the input tape wires are assumed.

    This method is based on computing the first term of the metric tensor with Hadamard
    tests.
    This term reads

    .. math ::

        \mathfrak{Re}\left\{\langle \partial_i\psi|\partial_j\psi\rangle\right\}

    and can be computed using an augmented circuit with an additional qubit.
    See for example `the appendix of this paper <https://arxiv.org/pdf/1804.03023.pdf>`__
    for details.
    The block diagonal of the tensor is computed using the covariance matrix approach
    implemented in :func:`.metric_tensor_cov_matrix`. In addition, we may extract the
    factors for the second terms :math:`\langle \psi|\partial_j\psi\rangle`
    of the block off-diagonal tensor from the tape results for the covariance matrix.
    This means that in total only the tapes for the first terms of the block off-diagonal
    are required in addition to ``metric_tensor_cov_matrix``.
    """
    # Prepare aux_wire
    aux_wire = aux_wire or tape.num_wires
    # Get tapes and processing function for the block diagonal metric tensor,
    # as well as the generator observables and generator coefficients for each diff'ed operation
    diag_tapes, diag_proc_fn, obs_list, coeffs = metric_tensor_cov_matrix(tape, diag_approx=False)
    graph = tape.graph
    layers = list(graph.iterate_parametrized_layers())

    first_term_tapes = []
    ids = []
    block_sizes = []
    # Get all tapes for the first term of the metric tensor
    for idx_i, layer_i in enumerate(layers):
        block_sizes.append(len(layer_i[2]))
        for layer_j in layers[idx_i + 1 :]:
            _tapes, _ids = _get_first_term_tapes(tape, layer_i, layer_j, allow_nonunitary)
            first_term_tapes.extend(_tapes)
            ids.extend(_ids)

    # Combine block diagonal and off-diagonal tapes
    tapes = diag_tapes + first_term_tapes
    # prepare block off-diagonal mask
    mask = 1 - qml.math.block_diag([np.ones((bsize, bsize)) for bsize in block_sizes])
    # Required for slicing in processing_fn
    num_diag_tapes = len(diag_tapes)

    def processing_fn(results):
        diag_res, off_diag_res = results[:num_diag_tapes], results[num_diag_tapes:]
        # Get full block diagonal tensor
        diag_mt = diag_proc_fn(diag_res)
        # Initialize block off-diagonal tensor using the stored ids
        first_term = qml.math.zeros_like(diag_mt)
        for result, idx in zip(off_diag_res, ids):
            # The metric tensor is symmetric
            first_term[idx] = first_term[idx[::-1]] = result

        # Second terms of block off-diagonal metric tensor
        expvals = []
        for prob, obs in zip(diag_res, obs_list):
            for o in obs:
                l = qml.math.cast(o.eigvals, dtype=np.float64)
                w = tape.wires.indices(o.wires)
                p = qml.math.marginal_prob(prob, w)
                expvals.append(qml.math.dot(l, p))

        # Construct <\partial_i\psi|\psi><\psi|\partial_j\psi> and mask it
        second_term = np.outer(expvals, expvals)
        second_term = qml.math.convert_like(second_term, results[0])
        second_term = qml.math.cast_like(second_term, results[0]) * mask
        # Subtract second term from first term
        off_diag_mt = first_term - second_term

        # Rescale first and second term
        _coeffs = np.hstack(coeffs)
        scale = qml.math.convert_like(np.outer(_coeffs, _coeffs), results[0])
        scale = qml.math.cast_like(scale, results[0])
        off_diag_mt = scale * off_diag_mt

        # Combine block diagonal and off-diagonal
        mt = off_diag_mt + diag_mt

        return mt

    return tapes, processing_fn
