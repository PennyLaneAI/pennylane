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

from .metric_tensor_cov_matrix import metric_tensor_cov_matrix

_GEN_TO_CGEN = {
    qml.PauliX: qml.CNOT,
    qml.PauliY: qml.CY,
    qml.PauliZ: qml.CZ,
}

_OP_TO_CGEN = {
    # PhaseShift is the same as RZ up to a global phase
    qml.PhaseShift: qml.CZ,
}

@functools.lru_cache
def _get_gen_op(op, aux_wire, allow_nonunitary):
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


def metric_tensor_hadamard(tape, allow_nonunitary, cache_states):
    diag_tapes, diag_proc_fn, obs_list, coeffs = metric_tensor_cov_matrix(tape, diag_approx=False)

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
