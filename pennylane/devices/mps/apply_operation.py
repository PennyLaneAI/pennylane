# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions to apply an operation to a state vector."""
# pylint: disable=unused-argument

from functools import singledispatch
import numpy as np
import quimb.tensor as qtn

import pennylane as qml


@singledispatch
def apply_operation(op: qml.operation.Operator, state, is_state_batched: bool = False):
    """Apply and operator to a given state.

    Args:
        op (Operator): The operation to apply to ``state``
        state (tensor_like): The starting state.
        is_state_batched (bool): Boolean representing whether the state is batched or not

    Returns:
        ndarray: output state

    .. warning::

        ``apply_operation`` is an internal function, and thus subject to change without a deprecation cycle.

    .. warning::
        ``apply_operation`` applies no validation to its inputs.

        This function assumes that the wires of the operator correspond to indices
        of the state. See :func:`~.map_wires` to convert operations to integer wire labels.

        The shape of state should be ``[2]*num_wires``.

    This is a ``functools.singledispatch`` function, so additional specialized kernels
    for specific operations can be registered like:

    .. code-block:: python

        @apply_operation.register
        def _(op: type_op, state):
            # custom op application method here

    **Example:**

    >>> state = np.zeros((2,2))
    >>> state[0][0] = 1
    >>> state
    tensor([[1., 0.],
        [0., 0.]], requires_grad=True)
    >>> apply_operation(qml.PauliX(0), state)
    tensor([[0., 0.],
        [1., 0.]], requires_grad=True)

    """
    wires = tuple(op.wires)
    gate_opts = {"contract": "swap+split", "cutoff": 0.0, "max_bond": None}
    if len(wires) <= 2:
        m = op.matrix()
        state.gate_(m, wires, **gate_opts)
        return
    # if op.has_decomposition:
    #     # try:
    #     newstate = copy.deepcopy(state)
    #     for d in op.decomposition():
    #         m = d.matrix()
    #         newstate.gate_(m, tuple(d.wires), **gate_opts)
    #     state.__dict__.update(newstate.__dict__)
    #     return
    #     # except:  # pylint: disable=bare-except
    #     #     pass
    mpo = op_2_mpo(op, state)
    newstate = mpo.apply(state, compress=True)
    state.__dict__.update(newstate.__dict__)


def op_2_tensor(op):
    """Returns the Quimb tensor corresponding to a PennyLane operator."""
    wires = tuple(op.wires)
    bra_inds = []
    for _, i in enumerate(wires):
        bra_inds.append(f"b{i}")
    bra_inds = tuple(bra_inds)
    ket_inds = []
    for _, i in enumerate(wires):
        ket_inds.append(f"k{i}")
    ket_inds = tuple(ket_inds)
    array = op.matrix()
    return qtn.Tensor(array.reshape([2] * int(np.log2(array.size))), inds=bra_inds + ket_inds)


def split_tensor(tensor, wires):
    """Returns the MPO factorization of a given tensor."""
    tensors = []
    v0 = tensor
    for c, i in enumerate(wires[0:-1]):
        inds = []
        for side in ["k", "b"]:
            inds.append(f"{side}{i}")
        if c > 0:
            inds.append(v0.inds[0])
        inds = tuple(inds)
        u0, v0 = v0.split(inds, cutoff=0.0)
        tensors.append(u0)
    tensors.append(v0)
    shift_tensor_indices(tensors)
    return tensors


def shift_tensor_indices(tensors):
    """Shifts the ``bra`` and ``ket`` indices to the right."""
    for t in tensors:
        for side in ["b", "k"]:
            for ind in t.inds:
                if ind[0] == side:
                    t.moveindex_(ind, -1)


def tensors_2_arrays(tensors, wires, n):
    """Converts a list of tensors into arrays that can be fed into ``MatrixProductOperator``."""
    arrays = []
    for _ in range(wires[0]):
        arrays.append(np.einsum("ij,kl->ijkl", np.eye(1, 1), np.eye(2, 2)))
    c = 0
    newaxes = 0
    for i in range(wires[0], wires[-1] + 1):
        if i in wires:
            arrays.append(np.expand_dims(tensors[c].data, axis=newaxes))
            c += 1
            newaxes = (1) if c == len(wires) - 1 else ()
        else:
            max_dim = np.max(arrays[-1].shape[0:2])
            arrays.append(np.einsum("ij,kl->ijkl", np.eye(max_dim, max_dim), np.eye(2, 2)))
    for _ in range(wires[-1] + 1, n):
        arrays.append(np.einsum("ij,kl->ijkl", np.eye(1, 1), np.eye(2, 2)))
    return arrays


def op_2_mpo(op, state):
    """Returns the MPO corresponding to the given operator."""
    wires = tuple(op.wires)
    tensor = op_2_tensor(op)
    tensors = split_tensor(tensor, wires)
    arrays = tensors_2_arrays(tensors, wires, state.L)
    return qtn.MatrixProductOperator(arrays, bond_name="x{}")
    # mpo.draw(show_inds="bond-size")
