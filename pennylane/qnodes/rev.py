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
ReversibleQNode class.
"""
from copy import copy
from functools import reduce
from string import ascii_letters as ABC

import numpy as np

from .qubit import QubitQNode

ABC_ARRAY = np.array(list(ABC))


class ReversibleQNode(QubitQNode):
    r"""Quantum node for reversible analytic differentiation method.

    This QNode enables a specific kind of differentiation method unique to simulators.

    The ReversibleQNode computes the analytic derivative of the circuit by using
    the following strategy:

    Assume a circuit has a gate :math:`G(\theta)` that we want to differentiate.
    Without loss of generality, we can write the circuit in the form three unitaries: :math:`UGV`.
    Starting from the initial state :math:`\vert 0\rangle`, the quantum state is evolved up to the
    "pre-measurement" state :math:`\vert\psi\rangle=UGV\vert 0\rangle`, which is saved
    (this can be reused for each variable being differentiated).

    We then apply the unitary :math:`V^{-1}` to evolve this state backwards in time
    until just after the gate :math:`G` (hence the name "reversible").
    The generator of :math:`G` is then applied as a gate, and we evolve forward using :math:`V` again.
    At this stage, the state of the simulator is proportional to
    :math:`\frac{\partial}{\partial\theta}\vert\psi\rangle`.
    Some further post-processing of this gives the derivative
    :math:`\frac{\partial}{\partial\theta} \langle \hat{O} \rangle` for any observable O.

    The reversible approach is similar to backpropagation, but trades off extra computation for
    enhanced memory efficiency. Where backpropagation caches the state tensors at each step during
    a forward pass, the reversible method only caches the final pre-measurement state.

    Compared to the parameter-shift rule, the reversible method can
    be faster or slower, depending on the density and location of parametrized gates in a circuit
    (circuits with higher density of parametrized gates near the end of the circuit will see a
    benefit).

    Args:
        func (callable): The *quantum function* of the QNode.
            A Python function containing :class:`~.operation.Operation` constructor calls,
            and returning a tuple of measured :class:`~.operation.Observable` instances.
        device (~pennylane._device.Device): computational device to execute the function on

    Keyword Args:
        mutable (bool): whether the QNode is mutable or not
        use_native_type (bool): If True, return the result in whatever type the device uses
            internally, otherwise convert it into array[float]. Default: True.
    """

    def __init__(self, func, device, mutable=True, **kwargs):
        if not device.capabilities().get("reversible_diff", False):
            raise ValueError(
                "Reversible differentiation method not supported on {}".format(device.short_name)
            )
        super().__init__(func, device, mutable=mutable, **kwargs)

    def _pd_analytic(self, idx, args, kwargs, **options):
        """Partial derivative of the node using the reversible method.

        Args:
            idx (int): flattened index of the parameter wrt. which the p.d. is computed
            args (array[float]): flattened positional arguments at which to evaluate the p.d.
            kwargs (dict[str, Any]): auxiliary arguments

        Returns:
            array[float]: partial derivative of the node
        """
        # pylint: disable=protected-access

        # TODO: cache these so they aren't created on each call from the same `jacobian`
        self.evaluate(args, kwargs)
        state = self.device._pre_rotated_state  # only works if forward pass has occured
        ops = self.circuit.operations_in_order
        obs = self.circuit.observables_in_order

        pd = 0.0
        # find the Operators in which the free parameter appears, use the product rule
        for op, p_idx in self.variable_deps[idx]:

            # create a new circuit which rewinds the pre-measurement state to just after `op`,
            # applies the generator of `op`, and then plays forward back to
            # pre-measurement step
            wires = op.wires
            op_idx = ops.index(op)

            # TODO: likely better to use circuitgraph to determine minimally necessary ops
            between_ops = ops[op_idx + 1 :]
            if op.name == "Rot":
                decomp = op.decomposition(*op.parameters, wires=wires)
                generator, multiplier = decomp[p_idx].generator
                between_ops = decomp[p_idx + 1 :] + between_ops
            else:
                generator, multiplier = op.generator

            # CRX, CRY, CRZ ops have a non-unitary matrix as generator
            # TODO: these can be supported by multiplying ``state`` directly by these generators within this function
            # (or by allowing non-unitary matrix multiplies in the simulator backends)
            if op.name in ["PhaseShift", "CRX", "CRY", "CRZ"]:
                raise ValueError(
                    "The {} gate is not currently supported with the "
                    "reversible gradient method.".format(op.name)
                )
            generator = generator(wires)
            diff_circuit = [copy(op).inv() for op in between_ops[::-1]] + [generator] + between_ops

            # set the simulator state to be the pre-measurement state
            self.device._state = state

            # evolve the pre-measurement state under this new circuit
            self.device.apply(diff_circuit)
            dstate = self.device._pre_rotated_state  # TODO: this will only work for QubitDevices

            # compute matrix element <d(state)|O|state> for each observable O
            matrix_elems = self.device._asarray(
                [self._matrix_elem(dstate, ob, state) for ob in obs]
                # TODO: if all observables act on same number of wires, could
                # do all at once with einsum
            )

            # post-process to get partial derivative contribution from this op
            multiplier *= op.params[p_idx].mult  # possible scalar multiplier
            pd += 2 * multiplier * self.device._imag(matrix_elems)

        # reset state back to pre-measurement value
        self.device._pre_rotated_state = state
        return pd

    def _matrix_elem(self, vec1, obs, vec2):
        """Computes the matrix element of observable ``obs`` between the two vectors
        ``vec1`` and ``vec2``, i.e., <vec1|obs|vec2>.
        Unmeasured wires are contracted, and a scalar is returned."""
        # pylint: disable=protected-access

        mat = self.device._reshape(obs.matrix, [2] * len(obs.wires) * 2)
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
            vec1_indices=vec1_indices, obs_indices=obs_indices, vec2_indices=vec2_indices,
        )
        return self.device._einsum(einsum_str, self.device._conj(vec1), mat, vec2)
