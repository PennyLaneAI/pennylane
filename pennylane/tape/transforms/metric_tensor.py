# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Contains the metric tensor tape transform
"""
import numpy as np
import pennylane as qml


def _stopping_critera(obj):
    if getattr(obj, "num_params", 0) == 0:
        return True

    if obj.name in ["RX", "RY", "RZ", "PhaseShift"]:
        return True

    return False


def metric_tensor(tape):
    """Returns a list of tapes, and a classical processing function,
    for computing the metric tensor of an input tape on hardware."""

    # Only the RX, RY, RZ, and PhaseShift gates are supported.
    # Expand out all other gates.
    tape = tape.expand(depth=2, stop_at=_stopping_critera)

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

            if gen is None:
                raise qml.qnodes.QuantumFunctionError(
                    "Can't generate metric tensor, operation {}"
                    "has no defined generator".format(op)
                )

            coeffs_list[-1].append(s)

            # get the observable corresponding to the generator of the current operation
            if isinstance(gen, np.ndarray):
                # generator is a Hermitian matrix
                obs_list[-1].append(qml.Hermitian(gen, w))

            elif issubclass(gen, qml.operation.Observable):
                # generator is an existing PennyLane operation
                obs_list[-1].append(gen(w))

            else:
                raise qml.qnodes.QuantumFunctionError(
                    "Can't generate metric tensor, generator {}"
                    "has no corresponding observable".format(gen)
                )

        with tape.__class__() as layer_tape:
            for op in queue:
                op.queue()

            for o in obs_list[-1]:
                o.diagonalizing_gates()

            qml.probs(wires=tape.wires)

        metric_tensor_tapes.append(layer_tape)

    def processing_fn(probs):
        qml.math.reshape(probs, [len(metric_tensor_tapes), -1])

        gs = []

        for prob, obs, coeffs in zip(probs, obs_list, coeffs_list):
            # calculate the covariance matrix of this layer
            scale = np.outer(coeffs, coeffs)
            g = scale * qml.math.cov_matrix(prob, obs)
            gs.append(g)

        perm = np.array([item for sublist in params_list for item in sublist], dtype=np.int64)

        # create the block diagonal metric tensor
        metric_tensor = qml.math.block_diag(gs)

        # permute rows
        metric_tensor = qml.math.gather(metric_tensor, perm)

        # permute columns
        metric_tensor = qml.math.gather(qml.math.T(metric_tensor), perm)
        return metric_tensor

    return metric_tensor_tapes, processing_fn
