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
Reversible QNode.
"""
import numpy as np

from copy import copy
from functools import reduce
from string import ascii_letters as ABC

from .jacobian import JacobianQNode

ABC_ARRAY = np.array(list(ABC))

class ReversibleQNode(JacobianQNode):

    def __init__(self, func, device, mutable=True, **kwargs):
        super().__init__(func, device, mutable=mutable, **kwargs)

    def _best_method(self, idx):
        meth = super()._best_method(idx)
        if meth in [None, "0"]:
            return meth
        return "A"

    def _pd_analytic(self, idx, args, kwargs, **options):
        """Partial derivative of the node using an analytic method.

        Args:
            idx (int): flattened index of the parameter wrt. which the p.d. is computed
            args (array[float]): flattened positional arguments at which to evaluate the p.d.
            kwargs (dict[str, Any]): auxiliary arguments

        Returns:
            array[float]: partial derivative of the node
        """
        # TODO: cache these so they aren't created on each call from the same `jacobian`
        self.evaluate(args, kwargs)
        state = self.device._pre_rotated_state  # only works if forward pass has occured
        ops = self.circuit.operations_in_order
        obs = self.circuit.observables_in_order

        pd = 0.0
        # find the Operators in which the free parameter appears, use the product rule
        for ctr, (op, p_idx) in enumerate(self.variable_deps[idx]):
            # create a new circuit which rewinds the pre-measurement state to just after `op`,
            # applies the generator of `op`, and then plays forward back to
            # pre-measurement step
            # TODO: likely better to use circuitgraph to determine minimally necessary ops
            generator, multiplier = op.generator  # TODO: this won't work for gates without generators (like `Rot`), but it could be made to)
            op_idx = ops.index(op)
            between_ops = ops[op_idx+1:]
            diff_circuit = [copy(op).inv() for op in between_ops[::-1]] + [generator(op.wires)] + between_ops
            # TODO: consider using shift rather than generator?

            # set the simulator state to be the pre-measurement state
            self.device._state = state

            # evolve the pre-measurement state under this new circuit
            self.device.apply(diff_circuit)
            dstate = self.device._pre_rotated_state

            # compute matrix element <d(state)|O|state> for each observable O
            matrix_elems = np.asarray([self._matrix_elem(dstate, ob, state) for ob in obs])
            # Note: this only works for expvals
            # TODO: handle var and sample

            # post-process to get partial derivative contribution from this op
            pd += 2 * multiplier * np.imag(matrix_elems)

        # reset state back to pre-measurement value
        self.device._pre_rotated_state = state
        return pd

    def _matrix_elem(self, vec1, obs, vec2):
        """Computes the matrix element of observable ``obs`` between the two vectors
        ``vec1`` and ``vec2``, i.e., <vec1|obs|vec2>.
        Unmeasured wires are contracted, and a scalar is returned."""
        mat = np.reshape(obs.matrix, [2] * len(obs.wires) * 2)
        wires = obs.wires

        vec1_indices = ABC[: self.num_wires]
        obs_in_indices = "".join(ABC_ARRAY[wires].tolist())
        obs_out_indices = ABC[self.num_wires : self.num_wires + len(wires)]
        obs_indices = "".join([obs_in_indices, obs_out_indices])
        vec2_indices = reduce(
            lambda old_string, idx_pair: old_string.replace(idx_pair[0], idx_pair[1]),
            zip(obs_in_indices, obs_out_indices),
            vec1_indices,
        )

        einsum_str = "{vec1_indices},{obs_indices},{vec2_indices}->".format(
            vec1_indices=vec1_indices,
            obs_indices=obs_indices,
            vec2_indices=vec2_indices,
        )

        return np.einsum(einsum_str, vec1.conj(), mat, vec2)
