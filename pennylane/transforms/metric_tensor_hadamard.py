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
Contains the transform to compute the full metric tensor using Hadamard tests
and an auxiliary qubit.
"""
import functools
import numpy as np
import pennylane as qml

from .batch_transform import batch_transform
from .metric_tensor import expand_fn, _metric_tensor_core
from pennylane.fourier.qnode_spectrum import expand_multi_par_and_no_gen

_GEN_TO_CGEN = {
    qml.PauliX: qml.CNOT,
    qml.PauliY: qml.CY,
    qml.PauliZ: qml.CZ,
}

_OP_TO_CGEN = {
    # PhaseShift is the same as RZ up to a global phase
    qml.PhaseShift: qml.CZ,
}


# TODO:
# - implement distinction of expand_fn using the allow_nonunitary input


def _get_non_block_diag_indices(size, block_sizes):
    """Get all indices for the upper triangle of a ``size``x``size`` matrix
    which are not within the block diagonal with block sizes ``block_sizes``."""

    ids = {}
    offset = 0
    for block_idx, bsize in enumerate(block_sizes):
        for inner_idx in range(bsize):
            ids[offset + inner_idx] = list(range(offset + bsize, size))
        offset += bsize
    return ids


def _get_gen_op(op, aux_wire, allow_nonunitary):
    gen, _ = op.generator
    if allow_nonunitary:
        if issubclass(gen, qml.operation.Observable):
            gen = gen.matrix
        return qml.ControlledQubitUnitary(gen, control_wires=aux_wire, wires=op.wires)

    if isinstance(gen, np.ndarray) or gen not in _GEN_TO_CGEN:
        cgen = _OP_TO_CGEN[op.__class__]
    else:
        cgen = _GEN_TO_CGEN.get(gen, None)

    return cgen(wires=[aux_wire, *op.wires])


def _get_first_term_tapes(tape, layer_i, layer_j, allow_nonunitary):
    """Obtain the tapes for the first term of all tensor entries
    belonging to an off-diagonal block."""

    tapes = []
    ids = []
    aux_wire = tape.num_wires
    ops_between_layers = [op for op in layer_j[0] if op not in layer_i[0]]
    for diffed_op_i, par_idx_i in zip(*layer_i[1:3]):
        gen_op_i = _get_gen_op(diffed_op_i, aux_wire, allow_nonunitary)
        for diffed_op_j, par_idx_j in zip(*layer_j[1:3]):
            gen_op_j = _get_gen_op(diffed_op_j, aux_wire, allow_nonunitary)
            with tape.__class__() as new_tape:
                qml.Hadamard(wires=aux_wire)
                for op in layer_i[0]:
                    qml.apply(op)
                qml.apply(gen_op_i)
                for op in ops_between_layers:
                    qml.apply(op)
                qml.apply(gen_op_j)
                qml.expval(qml.PauliX(aux_wire))
            tapes.append(new_tape)
            ids.append((par_idx_i, par_idx_j))

    return tapes, ids


def _metric_tensor_hadamard(tape, allow_nonunitary=False, approx=None, cache_states=False):

    diag_approx = approx == "diag"
    diag_tapes, diag_proc_fn, obs_list, coeffs = _metric_tensor_core(tape, diag_approx)
    if approx in {"diag", "block diag"}:
        return diag_tapes, diag_proc_fn

    graph = tape.graph
    layers = list(graph.iterate_parametrized_layers())
    block_sizes = [len(layer[2]) for layer in layers]

    if cache_states:
        raise NotImplementedError("The state-caching version is still WIP")
    else:
        first_term_tapes = []
        ids = []
        for idx_i, layer_i in enumerate(layers):
            for layer_j in layers[idx_i + 1 :]:
                _tapes, _ids = _get_first_term_tapes(tape, layer_i, layer_j, allow_nonunitary)
                first_term_tapes.extend(_tapes)
                ids.extend(_ids)

    num_diag_tapes = len(diag_tapes)
    num_first_term_tapes = len(first_term_tapes)

    tapes = diag_tapes + first_term_tapes

    def processing_fn(results):
        diag_mt = diag_proc_fn(results[:num_diag_tapes])

        off_diag_mt = qml.math.zeros_like(diag_mt)
        for result, idx in zip(results[num_diag_tapes:], ids):
            # calculate the covariance matrix of this layer
            off_diag_mt[idx] = off_diag_mt[idx[::-1]] = result

        # Second term of metric tensor - expectation values
        expvals = []
        for prob, obs in zip(results[:num_diag_tapes], obs_list):
            for o in obs:
                l = qml.math.cast(o.eigvals, dtype=np.float64)
                w = tape.wires.indices(o.wires)
                p = qml.math.marginal_prob(prob, w)
                expvals.append(qml.math.dot(l, p))

        second_term = np.outer(expvals, expvals)
        second_term = qml.math.convert_like(second_term, results[0])
        second_term = qml.math.cast_like(second_term, results[0])
        off_diag_mt = off_diag_mt - second_term

        _coeffs = np.hstack(coeffs)
        scale = qml.math.convert_like(np.outer(_coeffs, _coeffs), results[0])
        scale = qml.math.cast_like(scale, results[0])
        off_diag_mt = scale * off_diag_mt

        return np.where(diag_mt, diag_mt, off_diag_mt)

    return tapes, processing_fn


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

    if isinstance(qnode, qml.beta.QNode):
        cjac_fn = qml.transforms.classical_jacobian(qnode, expand_fn=self.expand_fn)
    else:
        cjac_fn = qml.transforms.classical_jacobian(qnode)

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


def metric_tensor_hadamard(
    tape, allow_nonunitary=False, approx=None, cache_states=False, **tkwargs
):
    if allow_nonunitary:
        _expand_fn = expand_fn
    else:
        _expand_fn = expand_multi_par_and_no_gen

    transform = batch_transform(_metric_tensor_hadamard, expand_fn=_expand_fn)
    transform.custom_qnode_wrapper(qnode_execution_wrapper)

    return transform(tape, approx, cache_states, **tkwargs)
