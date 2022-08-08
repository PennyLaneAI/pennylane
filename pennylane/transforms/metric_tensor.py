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
from pennylane import numpy as np
from pennylane.circuit_graph import LayerData

from .batch_transform import batch_transform


def expand_fn(tape, approx=None, allow_nonunitary=True, aux_wire=None, device_wires=None):
    """Set the metric tensor based on whether non-unitary gates are allowed."""
    # pylint: disable=unused-argument,too-many-arguments
    if not allow_nonunitary and approx is None:  # pragma: no cover
        return qml.transforms.expand_nonunitary_gen(tape)
    return qml.transforms.expand_multipar(tape)


@functools.partial(batch_transform, expand_fn=expand_fn)
def metric_tensor(tape, approx=None, allow_nonunitary=True, aux_wire=None, device_wires=None):
    r"""Returns a function that computes the metric tensor of a given QNode or quantum tape.

    The metric tensor convention we employ here has the following form:

    .. math::

        \text{metric_tensor}_{i, j} = \text{Re}\left[ \langle \partial_i \psi(\bm{\theta}) | \partial_j \psi(\bm{\theta}) \rangle
        - \langle \partial_i \psi(\bm{\theta}) | \psi(\bm{\theta}) \rangle \langle \psi(\bm{\theta}) | \partial_j \psi(\bm{\theta}) \rangle \right]

    with short notation :math:`| \partial_j \psi(\bm{\theta}) \rangle := \frac{\partial}{\partial \theta_j}| \psi(\bm{\theta}) \rangle`.
    It is closely related to the quantum fisher information matrix, see :func:`~.pennylane.qinfo.transforms.quantum_fisher` and eq. (27) in `arxiv:2103.15191 <https://arxiv.org/abs/2103.15191>`_.

    .. note::

        Only gates that have a single parameter and define a ``generator`` are supported.
        All other parametrized gates will be decomposed if possible.

        The ``generator`` of all parametrized operations, with respect to which the
        tensor is computed, are assumed to be Hermitian.
        This is the case for unitary single-parameter operations.

    Args:
        tape (pennylane.QNode or .QuantumTape): quantum tape or QNode to find the metric tensor of
        approx (str): Which approximation of the metric tensor to compute.

            - If ``None``, the full metric tensor is computed

            - If ``"block-diag"``, the block-diagonal approximation is computed, reducing
              the number of evaluated circuits significantly.

            - If ``"diag"``, only the diagonal approximation is computed, slightly
              reducing the classical overhead but not the quantum resources
              (compared to ``"block-diag"``).

        allow_nonunitary (bool): Whether non-unitary operations are allowed in circuits
            created by the transform. Only relevant if ``approx`` is ``None``.
            Should be set to ``True`` if possible to reduce cost.
        aux_wire (int or str or pennylane.wires.Wires): Auxiliary wire to be used for
            Hadamard tests. If ``None`` (the default), a suitable wire is inferred
            from the (number of) used wires in the original circuit and ``device_wires``.
        device_wires (.wires.Wires): Wires of the device that is going to be used for the
            metric tensor. Facilitates finding a default for ``aux_wire`` if ``aux_wire``
            is ``None``.
        hybrid (bool): Specifies whether classical processing inside a QNode
            should be taken into account when transforming a QNode.

            - If ``True``, and classical processing is detected, the Jacobian of the classical
              processing will be computed and included. When evaluated, the
              returned metric tensor will be with respect to the QNode arguments.
              The output shape can vary widely.

            - If ``False``, any internal QNode classical processing will be
              **ignored**. When evaluated, the returned metric tensor will be with
              respect to the **gate** arguments, and not the QNode arguments.
              The output shape is a single two-dimensional tensor.

    Returns:
        func: Function which accepts the same arguments as the QNode. When called, this
        function will return the metric tensor.

    The block-diagonal part of the metric tensor always is computed using the
    covariance-based approach. If no approximation is selected,
    the off block-diagonal is computed using Hadamard tests.

    .. warning::

        Performing the Hadamard tests requires a device
        that has an additional wire as compared to the wires on which the
        original circuit was defined. This wire may be specified via ``aux_wire``.
        The available wires on the device may be specified via ``device_wires``.

        By default (that is, if ``device_wires=None`` ), contiguous wire
        numbering and usage is assumed and the additional
        wire is set to the next wire of the device after the circuit wires.

        If the given or inferred ``aux_wire`` does not exist on the device,
        a warning is raised and the block-diagonal approximation is computed instead.
        It is significantly cheaper in this case to explicitly set ``approx="block-diag"`` .

    The flag ``allow_nonunitary`` should be set to ``True`` whenever the device with
    which the metric tensor is computed supports non-unitary operations.
    This will avoid additional decompositions of gates, in turn avoiding a potentially
    large number of additional Hadamard test circuits to be run.
    State vector simulators, for example, often allow applying operations that are
    not unitary.
    On a real QPU, setting this flag to ``True`` may cause exceptions because the
    computation of the metric tensor will request invalid operations on a quantum
    device.

    **Example**

    Consider the following QNode:

    .. code-block:: python

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev, interface="autograd")
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(weights[2], wires=1)
            qml.RZ(weights[3], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)), qml.expval(qml.PauliY(1))

    We can use the ``metric_tensor`` transform to generate a new function that returns the
    metric tensor of this QNode:

    >>> mt_fn = qml.metric_tensor(circuit)
    >>> weights = np.array([0.1, 0.2, 0.4, 0.5], requires_grad=True)
    >>> mt_fn(weights)
    tensor([[ 0.25  ,  0.    , -0.0497, -0.0497],
            [ 0.    ,  0.2475,  0.0243,  0.0243],
            [-0.0497,  0.0243,  0.0123,  0.0123],
            [-0.0497,  0.0243,  0.0123,  0.0123]], requires_grad=True)

    In order to save cost, one might want to compute only the block-diagonal part of
    the metric tensor, which requires significantly fewer executions of quantum functions
    and does not need an auxiliary wire on the device. This can be done using the
    ``approx`` keyword:

    >>> mt_fn = qml.metric_tensor(circuit, approx="block-diag")
    >>> weights = np.array([0.1, 0.2, 0.4, 0.5], requires_grad=True)
    >>> mt_fn(weights)
    tensor([[0.25  , 0.    , 0.    , 0.    ],
            [0.    , 0.2475, 0.    , 0.    ],
            [0.    , 0.    , 0.0123, 0.0123],
            [0.    , 0.    , 0.0123, 0.0123]], requires_grad=True)

    These blocks are given by parameter groups that belong to groups of commuting gates.

    The tensor can be further restricted to the diagonal via ``approx="diag"``. However,
    this will not save further quantum function evolutions but only classical postprocessing.

    The returned metric tensor is also fully differentiable in all interfaces.
    For example, we can compute the gradient of the Frobenius norm of the metric tensor
    with respect to the QNode ``weights`` :

    >>> norm_fn = lambda x: qml.math.linalg.norm(mt_fn(x), ord="fro")
    >>> grad_fn = qml.grad(norm_fn)
    >>> grad_fn(weights)
    array([-0.0282246 ,  0.01340413,  0.        ,  0.        ])

    .. details::
        :title: Usage Details

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
         <QuantumTape: wires=[0, 1], params=3>,
         <QuantumTape: wires=[2, 0], params=1>,
         <QuantumTape: wires=[2, 0, 1], params=2>,
         <QuantumTape: wires=[2, 0, 1], params=2>]

        This can be useful if the underlying circuits representing the metric tensor
        computation need to be analyzed. We clearly can distinguish the first three
        tapes used for the block-diagonal from the last three tapes that use the
        auxiliary wire ``2`` , which was not used by the original tape.

        The output tapes can then be evaluated and post-processed to retrieve
        the metric tensor:

        >>> dev = qml.device("default.qubit", wires=2)
        >>> fn(qml.execute(tapes, dev, None))
        array([[ 0.25      ,  0.        ,  0.42073549],
               [ 0.        ,  0.00415023, -0.26517488],
               [ 0.42073549, -0.26517488,  0.24878844]])

        The first term of the off block-diagonal entries of the full metric tensor are
        computed with Hadamard tests. This first term reads

        .. math ::

            \mathfrak{Re}\left\{\langle \partial_i\psi|\partial_j\psi\rangle\right\}

        and can be computed using an augmented circuit with an additional qubit.
        See for example the appendix of `McArdle et al. (2019) <https://doi.org/10.1038/s41534-019-0187-2>`__
        for details.
        The block-diagonal of the tensor is computed using the covariance matrix approach.

        In addition, we may extract the factors for the second terms
        :math:`\langle \psi|\partial_j\psi\rangle`
        of the *off block-diagonal* tensor from the quantum function output for the covariance matrix!

        This means that in total only the tapes for the first terms of the off block-diagonal
        are required in addition to the circuits for the block diagonal.
    """
    if not tape.trainable_params:
        warnings.warn(
            "Attempted to compute the metric tensor of a tape with no trainable parameters. "
            "If this is unintended, please mark trainable parameters in accordance with the "
            "chosen auto differentiation framework, or via the 'tape.trainable_params' property."
        )
        return [], lambda _: ()

    # pylint: disable=too-many-arguments
    if approx in {"diag", "block-diag"}:
        # Only require covariance matrix based transform
        diag_approx = approx == "diag"
        return _metric_tensor_cov_matrix(tape, diag_approx)[:2]

    if approx is None:
        return _metric_tensor_hadamard(tape, allow_nonunitary, aux_wire, device_wires)

    raise ValueError(
        f"Unknown value {approx} for keyword argument approx. "
        "Valid values are 'diag', 'block-diag' and None."
    )


def _contract_metric_tensor_with_cjac(mt, cjac):
    """Execute the contraction of pre-computed classical Jacobian(s)
    and the metric tensor of a tape in order to obtain the hybrid
    metric tensor of a QNode.

    Args:
        mt (array): Metric tensor of a tape (2-dimensional)
        cjac (array or tuple[array]): The classical Jacobian of a QNode

    Returns:
        array or tuple[array]: Hybrid metric tensor(s) of the QNode.
        The number of metric tensors depends on the number of QNode arguments
        for which the classical Jacobian was computed, the tensor shape(s)
        depend on the shape of these QNode arguments.
    """
    if isinstance(cjac, tuple):
        # Classical processing of multiple arguments is present. Return cjac.T @ mt @ cjac
        # as a tuple of contractions.
        metric_tensors = tuple(
            qml.math.tensordot(c, qml.math.tensordot(mt, c, axes=[[-1], [0]]), axes=[[0], [0]])
            for c in cjac
            if c is not None
        )
        if len(metric_tensors) == 1:
            return metric_tensors[0]

        return metric_tensors

    is_square = cjac.shape == (1,) or (cjac.ndim == 2 and cjac.shape[0] == cjac.shape[1])

    if is_square and qml.math.allclose(cjac, qml.numpy.eye(cjac.shape[0])):
        # Classical Jacobian is the identity. No classical processing
        # is present inside the QNode.
        return mt

    mt = qml.math.tensordot(cjac, qml.math.tensordot(mt, cjac, axes=[[-1], [0]]), axes=[[0], [0]])

    return mt


@metric_tensor.custom_qnode_wrapper
def qnode_execution_wrapper(self, qnode, targs, tkwargs):
    """Here, we overwrite the QNode execution wrapper in order
    to take into account that classical processing may be present
    inside the QNode."""
    hybrid = tkwargs.pop("hybrid", True)

    tkwargs.setdefault("device_wires", qnode.device.wires)
    mt_fn = self.default_qnode_wrapper(qnode, targs, tkwargs)

    _expand_fn = lambda tape: self.expand_fn(tape, *targs, **tkwargs)
    cjac_fn = qml.transforms.classical_jacobian(qnode, expand_fn=_expand_fn)

    def wrapper(*args, **kwargs):
        if not qml.math.get_trainable_indices(args):
            warnings.warn(
                "Attempted to compute the metric tensor of a QNode with no trainable parameters. "
                "If this is unintended, please add trainable parameters in accordance with the "
                "chosen auto differentiation framework."
            )
            return ()

        try:
            mt = mt_fn(*args, **kwargs)
        except qml.wires.WireError as e:
            if str(e) == "No device wires are unused by the tape.":
                warnings.warn(
                    "The device does not have a wire that is not used by the tape."
                    "\n\nReverting to the block-diagonal approximation. It will often be "
                    "much more efficient to request the block-diagonal approximation directly!"
                )
            else:
                warnings.warn(
                    "An auxiliary wire is not available."
                    "\n\nThis can occur when computing the full metric tensor via the "
                    "Hadamard test, and the device does not provide an "
                    "additional wire or the requested auxiliary wire does not exist "
                    "on the device."
                    "\n\nReverting to the block-diagonal approximation. It will often be "
                    "much more efficient to request the block-diagonal approximation directly!"
                )
            tkwargs["approx"] = "block-diag"
            return self.__call__(qnode, *targs, **tkwargs)(*args, **kwargs)

        if not hybrid:
            return mt

        kwargs.pop("shots", False)
        cjac = cjac_fn(*args, **kwargs)

        return _contract_metric_tensor_with_cjac(mt, cjac)

    return wrapper


def _metric_tensor_cov_matrix(tape, diag_approx):
    r"""This is the metric tensor method for the block diagonal, using
    the covariance matrix of the generators of each layer.

    Args:
        tape (pennylane.QNode or .QuantumTape): quantum tape or QNode to find the metric tensor of
        diag_approx (bool): if True, use the diagonal approximation. If ``False`` , a
            block-diagonal approximation of the metric tensor is computed.
    Returns:
        list[pennylane.tape.QuantumTape]: Transformed tapes that compute the probabilities
            required for the covariance matrix
        callable: Post-processing function that computes the covariance matrix from the
            results of the tapes in the first return value
        list[list[.Observable]]: Observables measured in each tape, one inner list
            corresponding to one tape in the first return value
        list[list[float]]: Coefficients to scale the results for each observable, one inner list
            corresponding to one tape in the first return value

    This method assumes the ``generator`` of all parametrized operations with respect to
    which the tensor is computed to be Hermitian. This is the case for unitary single-parameter
    operations.
    """
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
            obs, s = qml.generator(op)
            obs_list[-1].append(obs)
            coeffs_list[-1].append(s)

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
            scale = qml.math.convert_like(qml.math.outer(coeffs, coeffs), prob)
            scale = qml.math.cast_like(scale, prob)
            g = scale * qml.math.cov_matrix(prob, obs, wires=tape.wires, diag_approx=diag_approx)
            gs.append(g)

        # create the block diagonal metric tensor
        return qml.math.block_diag(gs)

    return metric_tensor_tapes, processing_fn, obs_list, coeffs_list


@functools.lru_cache()
def _get_gen_op(op, allow_nonunitary, aux_wire):
    r"""Get the controlled-generator operation for a given operation.

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
    op_to_cgen = {
        qml.RX: qml.CNOT,
        qml.RY: qml.CY,
        qml.RZ: qml.CZ,
        qml.PhaseShift: qml.CZ,  # PhaseShift is the same as RZ up to a global phase
    }

    try:
        cgen = op_to_cgen[op.__class__]
        return cgen(wires=[aux_wire, *op.wires])

    except KeyError as e:
        if allow_nonunitary:
            mat = qml.matrix(qml.generator(op, format="observable"))
            return qml.ControlledQubitUnitary(mat, control_wires=aux_wire, wires=op.wires)

        raise ValueError(
            f"Generator for operation {op} not known and non-unitary operations "
            "deactivated via allow_nonunitary=False."
        ) from e


def _get_first_term_tapes(tape, layer_i, layer_j, allow_nonunitary, aux_wire):
    r"""Obtain the tapes for the first term of all tensor entries
    belonging to an off-diagonal block.

    Args:
        tape (pennylane.tape.QuantumTape): Tape that is being transformed
        layer_i (list): The first layer of parametrized ops, of the format of
            the layers generated by ``iterate_parametrized_layers``
        layer_j (list): The second layer of parametrized ops
        allow_nonunitary (bool): Whether non-unitary operations are allowed
            in the circuit; passed to ``_get_gen_op``
        aux_wire (object or pennylane.wires.Wires): Auxiliary wire on which to
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
    ops_between_cgens = [op for op in layer_j.pre_ops if op not in layer_i.pre_ops]

    # Iterate over differentiated operation in first layer
    for diffed_op_i, par_idx_i in zip(layer_i.ops, layer_i.param_inds):
        gen_op_i = _get_gen_op(diffed_op_i, allow_nonunitary, aux_wire)

        # Iterate over differentiated operation in second layer
        # There will be one tape per pair of differentiated operations
        for diffed_op_j, par_idx_j in zip(layer_j.ops, layer_j.param_inds):
            gen_op_j = _get_gen_op(diffed_op_j, allow_nonunitary, aux_wire)

            with tape.__class__() as new_tape:
                # Initialize auxiliary wire
                qml.Hadamard(wires=aux_wire)
                # Apply backward cone of first layer
                for op in layer_i.pre_ops:
                    qml.apply(op)
                # Controlled-generator operation of first diff'ed op
                qml.apply(gen_op_i)
                # Apply first layer and operations between layers
                for op in ops_between_cgens:
                    qml.apply(op)
                # Controlled-generator operation of second diff'ed op
                qml.apply(gen_op_j)
                # Measure X on auxiliary wire
                qml.expval(qml.PauliX(aux_wire))

            tapes.append(new_tape)
            # Memorize to which metric entry this tape belongs
            ids.append((par_idx_i, par_idx_j))

    return tapes, ids


def _metric_tensor_hadamard(tape, allow_nonunitary, aux_wire, device_wires):
    r"""Generate the quantum tapes that execute the Hadamard tests
    to compute the first term of off block-diagonal metric entries
    and combine them with the covariance matrix-based block-diagonal tapes.

    Args:
        tape (pennylane.QNode or .QuantumTape): quantum tape or QNode to find the metric tensor of
        allow_nonunitary (bool): Whether non-unitary operations are allowed in circuits
            created by the transform. Only relevant if ``approx`` is ``None``
            Should be set to ``True`` if possible to reduce cost.
        aux_wire (int or .wires.Wires): Auxiliary wire to be used for
            Hadamard tests. By default, a suitable wire is inferred from the number
            of used wires in the original circuit.
        device_wires (.wires.Wires): Wires of the device that is going to be used for the
            metric tensor. Facilitates finding a default for ``aux_wire`` if ``aux_wire``
            is ``None`` .

    Returns:
        list[pennylane.tape.QuantumTape]: Tapes to evaluate the metric tensor
        callable: processing function to obtain the metric tensor from the tape results
    """
    # Get tapes and processing function for the block-diagonal metric tensor,
    # as well as the generator observables and generator coefficients for each diff'ed operation
    diag_tapes, diag_proc_fn, obs_list, coeffs = _metric_tensor_cov_matrix(tape, diag_approx=False)

    # Obtain layers of parametrized operations and account for the discrepancy between trainable
    # and non-trainable parameter indices
    graph = tape.graph
    par_idx_to_trainable_idx = {idx: i for i, idx in enumerate(sorted(tape.trainable_params))}
    layers = [
        LayerData(
            layer.pre_ops,
            layer.ops,
            tuple(par_idx_to_trainable_idx[idx] for idx in layer[2]),
            layer.post_ops,
        )
        for layer in graph.iterate_parametrized_layers()
    ]
    if len(layers) <= 1:
        return diag_tapes, diag_proc_fn

    # Get default for aux_wire
    aux_wire = _get_aux_wire(aux_wire, tape, device_wires)

    # Get all tapes for the first term of the metric tensor and memorize which
    # entry they belong to
    first_term_tapes = []
    ids = []
    block_sizes = []
    for idx_i, layer_i in enumerate(layers):
        block_sizes.append(len(layer_i.param_inds))

        for layer_j in layers[idx_i + 1 :]:
            _tapes, _ids = _get_first_term_tapes(tape, layer_i, layer_j, allow_nonunitary, aux_wire)
            first_term_tapes.extend(_tapes)
            ids.extend(_ids)

    # Combine block-diagonal and off block-diagonal tapes
    tapes = diag_tapes + first_term_tapes
    # prepare off block-diagonal mask
    mask = 1 - qml.math.block_diag([qml.math.ones((bsize, bsize)) for bsize in block_sizes])
    # Required for slicing in processing_fn
    num_diag_tapes = len(diag_tapes)

    def processing_fn(results):
        """Postprocessing function for the full metric tensor."""
        nonlocal mask
        # Split results
        diag_res, off_diag_res = results[:num_diag_tapes], results[num_diag_tapes:]
        # Get full block-diagonal tensor
        diag_mt = diag_proc_fn(diag_res)

        # Prepare the mask to match the used interface
        mask = qml.math.convert_like(mask, diag_mt)

        # Initialize off block-diagonal tensor using the stored ids
        first_term = qml.math.zeros_like(diag_mt)
        if ids:
            off_diag_res = qml.math.stack(off_diag_res, 1)[0]
            inv_ids = [_id[::-1] for _id in ids]
            first_term = qml.math.scatter_element_add(first_term, list(zip(*ids)), off_diag_res)
            first_term = qml.math.scatter_element_add(first_term, list(zip(*inv_ids)), off_diag_res)

        # Second terms of off block-diagonal metric tensor
        expvals = qml.math.zeros_like(first_term[0])
        idx = 0
        for prob, obs in zip(diag_res, obs_list):
            for o in obs:
                l = qml.math.cast(o.eigvals(), dtype=np.float64)
                w = tape.wires.indices(o.wires)
                p = qml.math.marginal_prob(prob, w)
                expvals = qml.math.scatter_element_add(expvals, (idx,), qml.math.dot(l, p))
                idx += 1

        # Construct <\partial_i\psi|\psi><\psi|\partial_j\psi> and mask it
        second_term = qml.math.tensordot(expvals, expvals, axes=0) * mask

        # Subtract second term from first term
        off_diag_mt = first_term - second_term

        # Rescale first and second term
        _coeffs = qml.math.hstack(coeffs)
        scale = qml.math.convert_like(qml.math.tensordot(_coeffs, _coeffs, axes=0), results[0])
        off_diag_mt = scale * off_diag_mt

        # Combine block-diagonal and off block-diagonal
        mt = off_diag_mt + diag_mt

        return mt

    return tapes, processing_fn


def _get_aux_wire(aux_wire, tape, device_wires):
    r"""Determine an unused wire to be used as auxiliary wire for Hadamard tests.

    Args:
        aux_wire (object): Input auxiliary wire. Returned unmodified if not ``None``
        tape (pennylane.tape.QuantumTape): Tape to infer the wire for
        device_wires (.wires.Wires): Wires of the device that is going to be used for the
            metric tensor. Facilitates finding a default for ``aux_wire`` if ``aux_wire``
            is ``None`` .

    Returns:
        object: The auxiliary wire to be used. Equals ``aux_wire`` if it was not ``None`` ,
        and an often reasonable choice else.
    """
    if aux_wire is not None:
        if device_wires is None or aux_wire in device_wires:
            return aux_wire
        raise qml.wires.WireError("The requested aux_wire does not exist on the used device.")

    if device_wires is not None:
        unused_wires = qml.wires.Wires(device_wires.toset().difference(tape.wires.toset()))
        if not unused_wires:
            raise qml.wires.WireError("No device wires are unused by the tape.")
        return unused_wires[0]

    _wires = tape.wires
    for _aux in range(tape.num_wires):
        if _aux not in _wires:
            return _aux

    return tape.num_wires
