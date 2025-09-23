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
from functools import partial

import numpy as np

import pennylane.ops as qops
from pennylane import math
from pennylane.circuit_graph import LayerData
from pennylane.exceptions import WireError
from pennylane.measurements import expval, probs
from pennylane.ops.functions import generator, matrix
from pennylane.queuing import WrappedObj
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import expand_multipar, expand_nonunitary_gen
from pennylane.transforms.core import transform
from pennylane.typing import PostprocessingFn
from pennylane.wires import Wires


def _mt_cjac_tdot(mt, c):
    return math.tensordot(c, math.tensordot(mt, c, axes=[[-1], [0]]), axes=[[0], [0]])


def _contract_metric_tensor_with_cjac(mt, cjac, tape):  # pylint: disable=unused-argument
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
    if isinstance(mt, tuple) and len(mt) == 1:
        mt = mt[0]
    if isinstance(cjac, tuple):
        # Classical processing of multiple arguments is present. Return cjac[i].T @ mt @ cjac[i]
        # as a tuple of contractions.
        metric_tensors = tuple(_mt_cjac_tdot(mt, c) for c in cjac if c is not None)
        return metric_tensors[0] if len(metric_tensors) == 1 else metric_tensors

    if not math.is_abstract(cjac):
        is_square = cjac.shape == (1,) or (cjac.ndim == 2 and cjac.shape[0] == cjac.shape[1])

        if is_square and math.allclose(cjac, math.eye(cjac.shape[0])):
            # Classical Jacobian is the identity. No classical processing in the QNode
            return mt

    return _mt_cjac_tdot(mt, cjac)


# pylint: disable=too-many-positional-arguments
def _expand_metric_tensor(
    tape: QuantumScript,
    argnum=None,
    approx=None,
    allow_nonunitary=True,
    aux_wire=None,
    device_wires=None,
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Set the metric tensor based on whether non-unitary gates are allowed."""
    # pylint: disable=unused-argument,too-many-arguments

    if not allow_nonunitary and approx is None:
        return [expand_nonunitary_gen(tape)], lambda x: x[0]
    return [expand_multipar(tape)], lambda x: x[0]


@partial(
    transform,
    expand_transform=_expand_metric_tensor,
    classical_cotransform=_contract_metric_tensor_with_cjac,
    final_transform=True,
)
def metric_tensor(  # pylint:disable=too-many-arguments, too-many-positional-arguments
    tape: QuantumScript,
    argnum=None,
    approx=None,
    allow_nonunitary=True,
    aux_wire=None,
    device_wires=None,
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    r"""Returns a function that computes the metric tensor of a given QNode or quantum tape.

    The metric tensor convention we employ here has the following form:

    .. math::

        \text{metric_tensor}_{i, j} = \text{Re}\left[ \langle \partial_i \psi(\bm{\theta}) | \partial_j \psi(\bm{\theta}) \rangle
        - \langle \partial_i \psi(\bm{\theta}) | \psi(\bm{\theta}) \rangle \langle \psi(\bm{\theta}) | \partial_j \psi(\bm{\theta}) \rangle \right]

    with short notation :math:`| \partial_j \psi(\bm{\theta}) \rangle := \frac{\partial}{\partial \theta_j}| \psi(\bm{\theta}) \rangle`.
    It is closely related to the quantum fisher information matrix, see :func:`~.pennylane.gradients.quantum_fisher` and eq. (27) in `arxiv:2103.15191 <https://arxiv.org/abs/2103.15191>`_.


    .. note::

        Only gates that have a single parameter and define a ``generator`` are supported.
        All other parametrized gates will be decomposed if possible.

        The ``generator`` of all parametrized operations, with respect to which the
        tensor is computed, are assumed to be Hermitian.
        This is the case for unitary single-parameter operations.

    Args:
        tape (QNode or QuantumTape): quantum circuit to find the metric tensor of
        argnum (int or Sequence[int] or None): Trainable tape-parameter indices with respect to which
            the metric tensor is computed. If ``argnum=None``, the metric tensor with respect to all
            trainable parameters is returned. Excluding tape-parameter indices from this list reduces
            the computational cost and the corresponding metric-tensor elements will be set to 0.

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
        aux_wire (None or int or str or Sequence or pennylane.wires.Wires): Auxiliary wire to
            be used for Hadamard tests. If ``None`` (the default), a suitable wire is inferred
            from the (number of) used wires in the original circuit and ``device_wires``,
            if the latter are given.
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
        qnode (QNode) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the metric tensor in the form of a tensor.

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

    .. note::

        When used with Catalyst, the classical component of the circuit is not included.
        This matches the results of setting ``hybrid=False``.

        For example,

        >>> from jax import numpy as jnp
        >>> @qml.qnode(qml.device('lightning.qubit', wires=4))
        ... def c(x, y):
        ...     qml.RX(2*x, 0)
        ...     qml.RY(y, 0)
        ...     return qml.expval(qml.Z(0))
        ...
        >>> qml.qjit(qml.metric_tensor(c))(jnp.array(0.5), jnp.array(0.6))
        Array([[0.25      , 0.        ],
                [0.        , 0.07298165]], dtype=float64)
        >>> qml.metric_tensor(c, argnums=(0,1))(jnp.array(0.5), jnp.array(0.6))
        (Array(1., dtype=float64), Array(0.07298165, dtype=float64))
        >>> qml.metric_tensor(c, hybrid=False)(qml.numpy.array(0.5), qml.numpy.array(0.6))
        array([[0.25      , 0.        ],
                [0.        , 0.07298165]])

        Here you can see that the ``qjit`` and ``hybrid=False`` options did not postprocess
        the metric tensor to match the shape of the arguments, and they do not include the factor
        of ``4`` from the derivative of ``2*x``.


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
            return qml.expval(qml.Z(0) @ qml.Z(1)), qml.expval(qml.Y(1))

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
        >>> ops = [
        ...     qml.RX(params[0], wires=0),
        ...     qml.RY(params[1], wires=0),
        ...     qml.CNOT(wires=(0,1)),
        ...     qml.PhaseShift(params[2], wires=1),
        ...     ]
        >>> measurements = [qml.expval(qml.X(0))]
        >>> tape = qml.tape.QuantumTape(ops, measurements)
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

        >>> dev = qml.device("default.qubit", wires=3)
        >>> fn(qml.execute(tapes, dev, None))
        tensor([[ 0.25      ,  0.        ,  0.42073549],
                [ 0.        ,  0.00415023, -0.26517488],
                [ 0.42073549, -0.26517488,  0.24878844]], requires_grad=True)

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

        .. warning::

            The ``argnum`` argument can be used to restrict the parameters which are taken into account
            for computing the metric tensor.
            When the metric tensor of a QNode is computed, the ordering of the parameters has to be
            specified as they appear in the corresponding QuantumTape.

        **Example**

        Consider the following QNode in which parameters are used out of order:

        .. code-block:: python

            dev = qml.device("default.qubit")
            @qml.qnode(dev, interface="autograd")
            def circuit(weights):
                qml.RX(weights[1], wires=0)
                qml.RY(weights[0], wires=0)
                qml.CNOT(wires=[0, 1])
                qml.RZ(weights[2], wires=1)
                qml.RZ(weights[3], wires=0)
                return qml.expval(qml.Z(0))

            weights = np.array([0.1, 0.2, 0.4, 0.5], requires_grad=True)
            mt = qml.metric_tensor(circuit, argnum=(0, 2, 3))(weights)

        >>> print(mt)
        [[ 0.          0.          0.          0.        ]
         [ 0.          0.25       -0.02495835 -0.02495835]
         [ 0.         -0.02495835  0.01226071  0.01226071]
         [ 0.         -0.02495835  0.01226071  0.01226071]]

        Because the 0-th element of ``weights`` appears second in the QNode and therefore in the
        underlying tape, it is the 1st tape parameter.
        By setting ``argnum = (0, 2, 3)`` we exclude the 0-th element of ``weights`` from the computation
        of the metric tensor and not the 1st element, as one might expect.
    """
    if not tape.trainable_params:
        warnings.warn(
            "Attempted to compute the metric tensor of a tape with no trainable parameters. "
            "If this is unintended, please mark trainable parameters in accordance with the "
            "chosen auto differentiation framework, or via the 'tape.trainable_params' property."
        )
        return [], lambda _: ()

    if argnum is None:
        argnum = tape.trainable_params
    elif isinstance(argnum, int):
        argnum = [argnum]
    if any(i not in tape.trainable_params for i in argnum):
        raise ValueError(
            "Some parameters specified in argnum are not in the "
            f"trainable parameters {tape.trainable_params} of the tape "
            "and will be ignored. This may be caused by attempting to "
            "differentiate with respect to parameters that are not marked "
            "as trainable."
        )

    if approx in {"diag", "block-diag"}:
        # Only require covariance matrix based transform
        diag_approx = approx == "diag"
        tapes, processing_fn = _metric_tensor_cov_matrix(tape, argnum, diag_approx)[:2]
        return tapes, processing_fn

    if approx is None:
        tapes, processing_fn = _metric_tensor_hadamard(
            tape, argnum, allow_nonunitary, aux_wire, device_wires
        )
        return tapes, processing_fn

    raise ValueError(
        f"Unknown value {approx} for keyword argument approx. "
        "Valid values are 'diag', 'block-diag' and None."
    )


@metric_tensor.custom_qnode_transform
def qnode_execution_wrapper(self, qnode, targs, tkwargs):
    """Here, we overwrite the QNode execution wrapper in order
    to take into account that classical processing may be present
    inside the QNode."""

    tkwargs.setdefault("device_wires", qnode.device.wires)

    mt_fn = self.default_qnode_transform(qnode, targs, tkwargs)

    return mt_fn


def _metric_tensor_cov_matrix(tape, argnum, diag_approx):  # pylint: disable=too-many-statements
    r"""This is the metric tensor method for the block diagonal, using
    the covariance matrix of the generators of each layer.

    Args:
        tape (pennylane.QNode or .QuantumTape): quantum tape or QNode to find the metric tensor of
        argnum (list[int]): Trainable tape-parameter indices with respect to which the metric tensor
            is computed.
        diag_approx (bool): if True, use the diagonal approximation. If ``False`` , a
            block-diagonal approximation of the metric tensor is computed.
    Returns:
        list[pennylane.tape.QuantumTape]: Transformed tapes that compute the probabilities
            required for the covariance matrix
        callable: Post-processing function that computes the covariance matrix from the
            results of the tapes in the first return value
        list[list[.Operator]]: Observables measured in each tape, one inner list
            corresponding to one tape in the first return value
        list[list[float]]: Coefficients to scale the results for each observable, one inner list
            corresponding to one tape in the first return value
        list[list[bool]]: Each inner list corresponds to one tape and therefore also one parametrized
            layer and its elements determine whether a trainable parameter in that layer is
            contained in ``argnum``.
        list[None, int]: Id list representing the layer for each parameter.
        list[None, int]: Id list representing the observables for each parameter.


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
    in_argnum_list = []
    layers_ids = []
    obs_ids = []

    i = 0
    for queue, curr_ops, param_idx, _ in graph.iterate_parametrized_layers():
        params_list.append(param_idx)
        in_argnum_list.append([p in argnum for p in param_idx])

        if not any(in_argnum_list[-1]):
            layers_ids.extend([None] * len(in_argnum_list[-1]))
            obs_ids.extend([None] * len(in_argnum_list[-1]))
            # no tape needs to be created for this block
            continue

        layer_coeffs, layer_obs = [], []

        # for each operation in the layer, get the generator
        j = 0
        for p, op in zip(param_idx, curr_ops):
            layers_ids.append(i)
            if p in argnum:
                obs, s = generator(op)
                layer_obs.append(obs)
                layer_coeffs.append(s)
                obs_ids.append(j)
                j = j + 1
            else:
                obs_ids.append(None)
        i = i + 1

        coeffs_list.append(layer_coeffs)
        obs_list.append(layer_obs)

        # Create a quantum tape with all operations
        # prior to the parametrized layer, and the rotations
        # to measure in the basis of the parametrized layer generators.
        # TODO: Maybe there are gates that do not affect the
        # generators of interest and thus need not be applied.

        for o, param_in_argnum in zip(layer_obs, in_argnum_list[-1]):
            if param_in_argnum:
                queue.extend(o.diagonalizing_gates())

        layer_tape = QuantumScript(queue, [probs(wires=tape.wires)], shots=tape.shots)
        metric_tensor_tapes.append(layer_tape)

    def processing_fn(probabilities):
        gs = []
        probs_idx = 0

        for params_in_argnum in in_argnum_list:
            if not any(params_in_argnum):
                # there is no tape and no probs associated to this layer
                dim = len(params_in_argnum)
                gs.append(math.zeros((dim, dim)))
                continue

            coeffs = coeffs_list[probs_idx]
            obs = obs_list[probs_idx]
            p = probabilities[probs_idx]

            scale = math.convert_like(math.outer(coeffs, coeffs), p)
            scale = math.cast_like(scale, p)
            g = scale * math.cov_matrix(p, obs, wires=tape.wires, diag_approx=diag_approx)
            for i, in_argnum in enumerate(params_in_argnum):
                # fill in rows and columns of zeros where a parameter was not in argnum
                if not in_argnum:
                    dim = g.shape[0]
                    g = math.concatenate((g[:i], math.zeros((1, dim)), g[i:]))
                    g = math.concatenate((g[:, :i], math.zeros((dim + 1, 1)), g[:, i:]), axis=1)
            gs.append(g)
            probs_idx += 1

        # create the block diagonal metric tensor
        return math.block_diag(gs)

    return (
        metric_tensor_tapes,
        processing_fn,
        obs_list,
        coeffs_list,
        in_argnum_list,
        layers_ids,
        obs_ids,
    )


@functools.lru_cache
def _get_gen_op(op, allow_nonunitary, aux_wire):
    r"""Get the controlled-generator operation for a given operation.

    Args:
        op (WrappedObj[Operation]): Wrapped Operation from which to extract the generator. The
            Operation needs to be wrapped for hashability in order to use the lru-cache.
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
        qops.RX: qops.CNOT,
        qops.RY: qops.CY,
        qops.RZ: qops.CZ,
        qops.PhaseShift: qops.CZ,  # PhaseShift is the same as RZ up to a global phase
    }

    op = op.obj
    try:
        cgen = op_to_cgen[op.__class__]
        return cgen(wires=[aux_wire, *op.wires])

    except KeyError as e:
        if allow_nonunitary:
            mat = matrix(generator(op)[0])
            wires = [aux_wire, *op.wires]
            return qops.ControlledQubitUnitary(mat, wires=wires)

        raise ValueError(
            f"Generator for operation {op} not known and non-unitary operations "
            "deactivated via allow_nonunitary=False."
        ) from e


def _get_first_term_tapes(layer_i, layer_j, allow_nonunitary, aux_wire, shots):
    r"""Obtain the tapes for the first term of all tensor entries
    belonging to an off-diagonal block.

    Args:
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
    ops_between_cgens = [
        op1 for op1 in layer_j.pre_ops if not any(op1 is op2 for op2 in layer_i.pre_ops)
    ]

    # Iterate over differentiated operation in first layer
    for diffed_op_i, par_idx_i in zip(layer_i.ops, layer_i.param_inds):
        gen_op_i = _get_gen_op(WrappedObj(diffed_op_i), allow_nonunitary, aux_wire)

        # Iterate over differentiated operation in second layer
        # There will be one tape per pair of differentiated operations
        for diffed_op_j, par_idx_j in zip(layer_j.ops, layer_j.param_inds):
            gen_op_j = _get_gen_op(WrappedObj(diffed_op_j), allow_nonunitary, aux_wire)

            ops = [
                qops.Hadamard(wires=aux_wire),
                *layer_i.pre_ops,
                gen_op_i,
                *ops_between_cgens,
                gen_op_j,
            ]
            new_tape = QuantumScript(ops, [expval(qops.X(aux_wire))], shots=shots)

            tapes.append(new_tape)
            # Memorize to which metric entry this tape belongs
            ids.append((par_idx_i, par_idx_j))

    return tapes, ids


def _metric_tensor_hadamard(
    tape, argnum, allow_nonunitary, aux_wire, device_wires
):  # pylint: disable=too-many-statements
    r"""Generate the quantum tapes that execute the Hadamard tests
    to compute the first term of off block-diagonal metric entries
    and combine them with the covariance matrix-based block-diagonal tapes.

    Args:
        tape (pennylane.QNode or .QuantumTape): quantum tape or QNode to find the metric tensor of
        argnum (list[int]): Trainable tape-parameter indices with respect to which the metric tensor
            is computed.
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
    (
        diag_tapes,
        diag_proc_fn,
        obs_list,
        coeffs_list,
        in_argnum_list,
        layer_ids,
        obs_ids,
    ) = _metric_tensor_cov_matrix(tape, argnum, diag_approx=False)

    # Obtain layers of parametrized operations and account for the discrepancy between trainable
    # and non-trainable parameter indices
    graph = tape.graph
    par_idx_to_trainable_idx = {idx: i for i, idx in enumerate(sorted(tape.trainable_params))}
    layers = []

    for layer, in_argnum in zip(graph.iterate_parametrized_layers(), in_argnum_list):
        if not any(in_argnum):
            # no tapes need to be constructed for this layer
            continue

        pre_ops, ops, param_idx, post_ops = layer
        new_ops = []
        new_param_idx = []

        for o, idx, param_in_argnum in zip(ops, param_idx, in_argnum):
            if param_in_argnum:
                new_ops.append(o)
                new_param_idx.append(par_idx_to_trainable_idx[idx])

        layers.append(LayerData(pre_ops, new_ops, new_param_idx, post_ops))

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
            _tapes, _ids = _get_first_term_tapes(
                layer_i, layer_j, allow_nonunitary, aux_wire, shots=tape.shots
            )
            first_term_tapes.extend(_tapes)
            ids.extend(_ids)

    # Combine block-diagonal and off block-diagonal tapes
    tapes = diag_tapes + first_term_tapes
    # prepare off block-diagonal mask
    blocks = []
    for in_argnum in in_argnum_list:
        d = len(in_argnum)
        blocks.append(math.ones((d, d)))
    mask = 1 - math.block_diag(blocks)

    # Required for slicing in processing_fn
    num_diag_tapes = len(diag_tapes)

    def processing_fn(results):
        """Postprocessing function for the full metric tensor."""
        nonlocal mask
        # Split results
        diag_res, off_diag_res = results[:num_diag_tapes], results[num_diag_tapes:]
        # Get full block-diagonal tensor
        diag_mt = diag_proc_fn(diag_res)

        # the off diag tapes only have a single expval measurement
        off_diag_res = [math.expand_dims(res, 0) for res in off_diag_res]

        # Prepare the mask to match the used interface
        mask = math.convert_like(mask, diag_mt)

        # Initialize off block-diagonal tensor using the stored ids
        first_term = math.zeros_like(diag_mt)
        if ids:
            off_diag_res = math.stack(off_diag_res, 1)[0]

            for loc, r in zip(ids, off_diag_res):
                # not sure if we can promise ordering of locations
                # so need to loop over indices for compatibility with catalyst
                first_term = math.scatter_element_add(first_term, loc, r)
                first_term = math.scatter_element_add(first_term, (loc[1], loc[0]), r)

        # Second terms of off block-diagonal metric tensor
        expvals = math.zeros_like(first_term[0])

        for i, (layer_i, obs_i) in enumerate(zip(layer_ids, obs_ids)):
            if layer_i is not None and obs_i is not None:
                prob = diag_res[layer_i]
                o = obs_list[layer_i][obs_i]
                l = math.cast(o.eigvals(), dtype=np.float64)
                w = tape.wires.indices(o.wires)
                p = math.marginal_prob(prob, w)
                expvals = math.scatter_element_add(expvals, (i,), math.dot(l, p))

        # Construct <\partial_i\psi|\psi><\psi|\partial_j\psi> and mask it
        second_term = math.tensordot(expvals, expvals, axes=0) * mask

        # Subtract second term from first term
        off_diag_mt = first_term - second_term

        # Rescale first and second term
        coeffs_gen = (c for c in math.hstack(coeffs_list))
        # flattened coeffs_list but also with 0s where parameters are not in argnum
        interface = math.get_interface(*results)
        extended_coeffs_list = math.asarray(
            [
                next(coeffs_gen) if param_in_argnum else 0.0
                for param_in_argnum in math.hstack(in_argnum_list)
            ],
            like=interface,
        )
        scale = math.tensordot(extended_coeffs_list, extended_coeffs_list, axes=0)
        off_diag_mt = scale * off_diag_mt

        # Combine block-diagonal and off block-diagonal
        mt = off_diag_mt + diag_mt

        return mt

    return tapes, processing_fn


def _get_aux_wire(aux_wire, tape, device_wires):
    r"""Determine an unused wire to be used as auxiliary wire for Hadamard tests.

    Args:
        aux_wire (object): Input auxiliary wire. May be one of a variety of input formats:
            If ``None``, try to infer a reasonable choice based on the number of wires used
            in the ``tape``, and based on ``device_wires``, if they are not ``None``.
            If an ``int``, a ``str`` or a ``Sequence``, convert the input to a ``Wires``
            object and take the first entry of the result. This leads to consistent behaviour
            between ``_get_aux_wire`` and the ``Wires`` class.
            If a ``Wires`` instance already, the conversion to such an instance is performed
            trivially as well (also see the source code of ``~.Wires``).
        tape (pennylane.tape.QuantumTape): Tape to infer the wire for
        device_wires (.wires.Wires): Wires of the device that is going to be used for the
            metric tensor. Facilitates finding a default for ``aux_wire`` if ``aux_wire``
            is ``None`` .

    Returns:
        object: The auxiliary wire to be used. Equals ``aux_wire`` if it was not ``None``\ ,
        and an often reasonable choice else.
    """
    if aux_wire is not None:
        aux_wire = Wires(aux_wire)[0]
        if aux_wire in tape.wires:
            msg = "The requested auxiliary wire is already in use by the circuit."
            raise WireError(msg)
        if device_wires is None or aux_wire in device_wires:
            return aux_wire
        raise WireError("The requested auxiliary wire does not exist on the used device.")

    if device_wires is not None:
        if len(device_wires) == len(tape.wires):
            raise WireError("The device has no free wire for the auxiliary wire.")
        unused_wires = Wires(device_wires.toset().difference(tape.wires.toset()))
        return unused_wires[0]

    _wires = tape.wires
    for _aux in range(tape.num_wires):
        if _aux not in _wires:
            return _aux

    return tape.num_wires
