# Copyright 2023 Xanadu Quantum Technologies Inc.

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
This module contains functions for computing the Hadamard-test gradient
of a qubit-based quantum tape.
"""
from functools import partial
from itertools import islice

import numpy as np

import pennylane as qml
from pennylane import transform
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.typing import PostprocessingFn

from .gradient_transform import (
    _contract_qjac_with_cjac,
    _no_trainable_grad,
    _try_zero_grad_from_graph_or_get_grad_method,
    assert_no_state_returns,
    assert_no_trainable_tape_batching,
    assert_no_variance,
    choose_trainable_param_indices,
)
from .metric_tensor import _get_aux_wire


def _hadamard_stopping_condition(op) -> bool:
    if not op.has_decomposition:
        # let things without decompositions through without error
        # error will happen when calculating hadamard grad
        return True
    if isinstance(op, qml.operation.Operator) and any(qml.math.requires_grad(p) for p in op.data):
        return op.has_generator
    return True


def _inplace_set_trainable_params(tape):
    """Update all the trainable params in place."""
    params = tape.get_parameters(trainable_only=False)
    tape.trainable_params = qml.math.get_trainable_indices(params)


# pylint: disable=unused-argument
def _expand_transform_hadamard(
    tape: QuantumScript,
    argnum=None,
    aux_wire=None,
    device_wires=None,
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Expand function to be applied before hadamard gradient."""
    batch, postprocessing = qml.devices.preprocess.decompose(
        tape,
        stopping_condition=_hadamard_stopping_condition,
        skip_initial_state_prep=False,
        name="hadamard",
        error=qml.operation.DecompositionUndefinedError,
    )
    if any(
        qml.math.requires_grad(d) for mp in tape.measurements for d in getattr(mp.obs, "data", [])
    ):
        try:
            batch, postprocessing = qml.transforms.split_to_single_terms(batch[0])
        except RuntimeError as e:
            raise ValueError(
                "Can only differentiate Hamiltonian "
                f"coefficients for expectations, not {tape.measurements}."
            ) from e
    if len(batch) > 1 or batch[0] is not tape:  # split to single terms modified the tape
        _ = [_inplace_set_trainable_params(t) for t in batch]
    return batch, postprocessing


@partial(
    transform,
    expand_transform=_expand_transform_hadamard,
    classical_cotransform=_contract_qjac_with_cjac,
    final_transform=True,
)
def hadamard_grad(
    tape: QuantumScript,
    argnum=None,
    aux_wire=None,
    device_wires=None,
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    r"""Transform a circuit to compute the Hadamard test gradient of all gates
    with respect to their inputs.

    Args:
        tape (QNode or QuantumTape): quantum circuit to differentiate
        argnum (int or list[int] or None): Trainable tape parameter indices to differentiate
            with respect to. If not provided, the derivatives with respect to all
            trainable parameters are returned. Note that the indices are with respect to
            the list of trainable parameters.
        aux_wire (pennylane.wires.Wires): Auxiliary wire to be used for the Hadamard tests.
            If ``None`` (the default), a suitable wire is inferred from the wires used in
            the original circuit and ``device_wires``.
        device_wires (pennylane.wires.Wires): Wires of the device that are going to be used for the
            gradient. Facilitates finding a default for ``aux_wire`` if ``aux_wire`` is ``None``.

    Returns:
        qnode (QNode) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.
        Executing this circuit will provide the Jacobian in the form of a tensor, a tuple, or a
        nested tuple depending upon the nesting structure of measurements in the original circuit.

    For a variational evolution :math:`U(\mathbf{p}) \vert 0\rangle` with :math:`N` parameters
    :math:`\mathbf{p}`, consider the expectation value of an observable :math:`O`:

    .. math::

        f(\mathbf{p})  = \langle \hat{O} \rangle(\mathbf{p}) = \langle 0 \vert
        U(\mathbf{p})^\dagger \hat{O} U(\mathbf{p}) \vert 0\rangle.


    The gradient of this expectation value can be calculated via the Hadamard test gradient:

    .. math::

        \frac{\partial f}{\partial \mathbf{p}} = -2 \Im[\bra{0} \hat{O} G \ket{0}] = i \left(\bra{0} \hat{O} G \ket{
        0} - \bra{0} G\hat{O} \ket{0}\right) = -2 \bra{+}\bra{0} \texttt{ctrl}\left(G^{\dagger}\right) (\hat{Y} \otimes \hat{O}) \texttt{ctrl}\left(G\right)
        \ket{+}\ket{0}

    Here, :math:`G` is the generator of the unitary :math:`U`. ``hadamard_grad`` will work on any :math:`U` so long
    as it has a generator :math:`G` defined (i.e., ``op.has_generator == True``). Otherwise, it will try to decompose
    into gates where this is satisfied.

    **Example**

    This transform can be registered directly as the quantum gradient transform
    to use during autodifferentiation:

    >>> import jax
    >>> dev = qml.device("default.qubit")
    >>> @qml.qnode(dev, interface="jax", diff_method="hadamard")
    ... def circuit(params):
    ...     qml.RX(params[0], wires=0)
    ...     qml.RY(params[1], wires=0)
    ...     qml.RX(params[2], wires=0)
    ...     return qml.expval(qml.Z(0)), qml.probs(wires=0)
    >>> params = jax.numpy.array([0.1, 0.2, 0.3])
    >>> jax.jacobian(circuit)(params)
    (Array([-0.3875172 , -0.18884787, -0.38355704], dtype=float64),
     Array([[-0.1937586 , -0.09442394, -0.19177852],
            [ 0.1937586 ,  0.09442394,  0.19177852]], dtype=float64))

    .. details::
        :title: Usage Details

        This gradient method can work with any operator that has a generator:

        .. code-block:: pycon

            >>> dev = qml.device('default.qubit')
            >>> @qml.qnode(dev)
            ... def circuit(x):
            ...     qml.evolve(qml.X(0) @ qml.X(1) + qml.Z(0) @ qml.Z(1) + qml.H(0), x )
            ...     return qml.expval(qml.Z(0))
            ...
            >>> print( qml.draw(qml.gradients.hadamard_grad(circuit))(qml.numpy.array(0.5)) )
            0: ─╭Exp(-0.50j 𝓗)─╭X────┤ ╭<Z@Y>
            1: ─╰Exp(-0.50j 𝓗)─│─────┤ │
            2: ──H─────────────╰●──H─┤ ╰<Z@Y>

            0: ─╭Exp(-0.50j 𝓗)─╭X@X────┤ ╭<Z@Y>
            1: ─╰Exp(-0.50j 𝓗)─├X@X────┤ │
            2: ──H─────────────╰●────H─┤ ╰<Z@Y>

            0: ─╭Exp(-0.50j 𝓗)─╭Z────┤ ╭<Z@Y>
            1: ─╰Exp(-0.50j 𝓗)─│─────┤ │
            2: ──H─────────────╰●──H─┤ ╰<Z@Y>

            0: ─╭Exp(-0.50j 𝓗)─╭Z@Z────┤ ╭<Z@Y>
            1: ─╰Exp(-0.50j 𝓗)─├Z@Z────┤ │
            2: ──H─────────────╰●────H─┤ ╰<Z@Y>

        This gradient transform can be applied directly to :class:`QNode <pennylane.QNode>`
        objects. However, for performance reasons, we recommend providing the gradient transform
        as the ``diff_method`` argument of the QNode decorator, and differentiating with your
        preferred machine learning framework.

        >>> dev = qml.device("default.qubit")
        >>> @qml.qnode(dev)
        ... def circuit(params):
        ...     qml.RX(params[0], wires=0)
        ...     qml.RY(params[1], wires=0)
        ...     qml.RX(params[2], wires=0)
        ...     return qml.expval(qml.Z(0))
        >>> params = np.array([0.1, 0.2, 0.3], requires_grad=True)
        >>> qml.gradients.hadamard_grad(circuit)(params)
        tensor([-0.3875172 , -0.18884787, -0.38355704], requires_grad=True)

        This quantum gradient transform can also be applied to low-level
        :class:`~.QuantumTape` objects. This will result in no implicit quantum
        device evaluation. Instead, the processed tapes, and post-processing
        function, which together define the gradient are directly returned:

        >>> ops = [qml.RX(params[0], 0), qml.RY(params[1], 0), qml.RX(params[2], 0)]
        >>> measurements = [qml.expval(qml.Z(0))]
        >>> tape = qml.tape.QuantumTape(ops, measurements)
        >>> gradient_tapes, fn = qml.gradients.hadamard_grad(tape)
        >>> gradient_tapes
        [<QuantumScript: wires=[0, 1], params=3>,
         <QuantumScript: wires=[0, 1], params=3>,
         <QuantumScript: wires=[0, 1], params=3>]

        This can be useful if the underlying circuits representing the gradient
        computation need to be analyzed.

        Note that ``argnum`` refers to the index of a parameter within the list of trainable
        parameters. For example, if we have:

        >>> tape = qml.tape.QuantumScript(
        ...     [qml.RX(1.2, wires=0), qml.RY(2.3, wires=0), qml.RZ(3.4, wires=0)],
        ...     [qml.expval(qml.Z(0))],
        ...     trainable_params = [1, 2]
        ... )
        >>> qml.gradients.hadamard_grad(tape, argnum=1)

        The code above will differentiate the third parameter rather than the second.

        The output tapes can then be evaluated and post-processed to retrieve the gradient:

        >>> dev = qml.device("default.qubit")
        >>> fn(qml.execute(gradient_tapes, dev, None))
        (tensor(-0.3875172, requires_grad=True),
         tensor(-0.18884787, requires_grad=True),
         tensor(-0.38355704, requires_grad=True))

        This transform can be registered directly as the quantum gradient transform
        to use during autodifferentiation:

        >>> dev = qml.device("default.qubit")
        >>> @qml.qnode(dev, interface="jax", diff_method="hadamard")
        ... def circuit(params):
        ...     qml.RX(params[0], wires=0)
        ...     qml.RY(params[1], wires=0)
        ...     qml.RX(params[2], wires=0)
        ...     return qml.expval(qml.Z(0))
        >>> params = jax.numpy.array([0.1, 0.2, 0.3])
        >>> jax.jacobian(circuit)(params)
        Array([-0.3875172 , -0.18884787, -0.38355704], dtype=float64)

        If you use custom wires on your device, you need to pass an auxiliary wire.

        >>> dev_wires = ("a", "c")
        >>> dev = qml.device("default.qubit", wires=dev_wires)
        >>> gradient_kwargs = {"aux_wire": "c"}
        >>> @qml.qnode(dev, interface="jax", diff_method="hadamard", gradient_kwargs=gradient_kwargs)
        >>> def circuit(params):
        ...    qml.RX(params[0], wires="a")
        ...    qml.RY(params[1], wires="a")
        ...    qml.RX(params[2], wires="a")
        ...    return qml.expval(qml.Z("a"))
        >>> params = jax.numpy.array([0.1, 0.2, 0.3])
        >>> jax.jacobian(circuit)(params)
        Array([-0.3875172 , -0.18884787, -0.38355704], dtype=float64)
    """

    transform_name = "Hadamard test"
    assert_no_state_returns(tape.measurements, transform_name)
    assert_no_variance(tape.measurements, transform_name)
    assert_no_trainable_tape_batching(tape, transform_name)
    if len(tape.measurements) > 1 and tape.shots.has_partitioned_shots:
        raise NotImplementedError(
            "hadamard gradient does not support multiple measurements with partitioned shots."
        )

    if argnum is None and not tape.trainable_params:
        return _no_trainable_grad(tape)

    trainable_param_indices = choose_trainable_param_indices(tape, argnum)

    # Validate or get default for aux_wire
    aux_wire = _get_aux_wire(aux_wire, tape, device_wires)

    g_tapes = []
    coeffs = []
    generators_per_parameter = []

    for trainable_param_idx, _ in enumerate(tape.trainable_params):
        if (
            trainable_param_idx not in trainable_param_indices
            or _try_zero_grad_from_graph_or_get_grad_method(
                tape, tape.trainable_params[trainable_param_idx], True
            )
            == "0"
        ):
            generators_per_parameter.append(0)
        else:
            # can dispatch between different algorithms here in the future
            # hadamard test, direct hadamard test, reversed, reversed direct, and flexible
            batch, new_coeffs = _hadamard_test(tape, trainable_param_idx, aux_wire)
            g_tapes += batch
            coeffs += new_coeffs
            generators_per_parameter.append(len(batch))

    return g_tapes, partial(
        processing_fn, coeffs=coeffs, tape=tape, generators_per_parameter=generators_per_parameter
    )


def _hadamard_test(tape, trainable_param_idx, aux_wire) -> tuple[list, list]:

    trainable_op, idx, _ = tape.get_operation(trainable_param_idx)

    ops_to_trainable_op = tape.operations[: idx + 1]
    ops_after_trainable_op = tape.operations[idx + 1 :]

    # Get a generator and coefficients
    sub_coeffs, generators = _get_pauli_generators(trainable_op)

    measurements = [_new_measurement(mp, aux_wire, tape.wires) for mp in tape.measurements]

    new_batch = []
    for gen in generators:
        ctrl_gen = [qml.ctrl(gen, control=aux_wire)]
        hadamard = [qml.Hadamard(wires=aux_wire)]
        ops = ops_to_trainable_op + hadamard + ctrl_gen + hadamard + ops_after_trainable_op

        new_tape = qml.tape.QuantumScript(ops, measurements, shots=tape.shots)
        new_batch.append(new_tape)
    return new_batch, sub_coeffs


def _new_measurement(mp, aux_wire, all_wires: qml.wires.Wires):
    obs = mp.obs or qml.prod(*(qml.Z(w) for w in mp.wires or all_wires))
    new_obs = qml.simplify(obs @ qml.Y(aux_wire))
    return type(mp)(obs=new_obs)


def _get_pauli_generators(trainable_op):
    """From a trainable operation, extract the unitary generators and their coefficients.
    Any operator with a generator is supported.
    """
    generator = trainable_op.generator()
    if generator.pauli_rep is None:
        mat = qml.matrix(generator, wire_order=generator.wires)
        pauli_rep = qml.pauli_decompose(mat, wire_order=generator.wires, pauli=True)
    else:
        pauli_rep = generator.pauli_rep
    id_pw = qml.pauli.PauliWord({})
    if id_pw in pauli_rep:
        del pauli_rep[qml.pauli.PauliWord({})]  # remove identity term
    sum_op = pauli_rep.operation()
    return sum_op.terms()


def _postprocess_probs(res, measurement, tape):
    projector = np.array([1, -1])
    projector = qml.math.convert_like(projector, res)

    num_wires_probs = len(measurement.wires)
    if num_wires_probs == 0:
        num_wires_probs = tape.num_wires
    res = qml.math.reshape(res, (2**num_wires_probs, 2))
    return qml.math.tensordot(res, projector, axes=[[1], [0]])


def processing_fn(results: qml.typing.ResultBatch, tape, coeffs, generators_per_parameter):
    """Post processing function for computing a hadamard gradient."""

    final_res = []
    for coeff, res in zip(coeffs, results):
        if not isinstance(res, (tuple, list)):
            res = [res]  # add singleton dimension back in for one measurement
        final_res.append([qml.math.convert_like(2 * coeff * r, r) for r in res])

    # Post process for probs
    measurements_probs = [
        idx
        for idx, m in enumerate(tape.measurements)
        if isinstance(m, qml.measurements.ProbabilityMP)
    ]
    if measurements_probs:
        for idx, res in enumerate(final_res):
            for prob_idx in measurements_probs:
                final_res[idx][prob_idx] = _postprocess_probs(
                    res[prob_idx], tape.measurements[prob_idx], tape
                )

    mps = (
        tape.measurements * tape.shots.num_copies
        if tape.shots.has_partitioned_shots
        else tape.measurements
    )
    # first index = into measurements, second index = into parameters
    grads = tuple([] for _ in mps)
    results = iter(final_res)
    for num_generators in generators_per_parameter:
        if num_generators == 0:
            for g_for_parameter, mp in zip(grads, mps):
                zeros_like_mp = np.zeros(
                    mp.shape(num_device_wires=len(tape.wires)), dtype=mp.numeric_type
                )
                g_for_parameter.append(zeros_like_mp)
        else:
            sub_results = islice(results, num_generators)  # take the next number of results
            # sum over batch, iterate over measurements
            summed_sub_results = (sum(r) for r in zip(*sub_results))

            for g_for_parameter, r in zip(grads, summed_sub_results, strict=True):
                g_for_parameter.append(r)

    if len(mps) == 1:
        return grads[0][0] if len(tape.trainable_params) == 1 else grads[0]
    return tuple(g[0] for g in grads) if len(tape.trainable_params) == 1 else grads
