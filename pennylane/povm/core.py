from typing import Sequence, Callable

import numpy as np

import pennylane as qml
from pennylane.transforms.core import transform


class POVM:

    def __init__(self, obs, validate=False):
        self.obs = obs
        self.wires = qml.wires.Wires.all_wires([ob.wires for ob in self.obs])

        if validate:
            self._validate()

    def _validate(self):

        # TODO: replace matrix comparison with qml.equal once op arithmetic supports it

        if len(self.obs) == 1:
            if not qml.math.allclose(qml.matrix(self.obs[0]), qml.matrix(qml.Identity(self.wires))):
                raise ValueError("Operators should sum to the identity")
        else:
            if not qml.math.allclose(qml.matrix(qml.sum(*self.obs)), qml.matrix(qml.Identity(self.wires))):
                raise ValueError("Operators should sum to the identity")

        for ob in self.obs:
            # check positive semi-definiteness
            eigvals = qml.eigvals(ob)
            if any([eigval < 0 for eigval in eigvals]):
                raise ValueError("All operators should be positive semi-definite; got operator {ob}")

    def __repr__(self):
        ops_str = ",\n  ".join([str(ob) for ob in self.obs])
        return (f"<POVM: n_ops={len(self.obs)}, wires={self.wires.tolist()}, "
                f"ops=[{ops_str}]>")

    def __iter__(self):
        return iter(self.obs)

    def __len__(self):
        return len(self.obs)

    @property
    def data(self):
        d = tuple(ob.data for ob in self.obs)
        return tuple(__d for _d in d for __d in _d)

    @property
    def has_diagonalizing_gates(self):
        return True

    @property
    def is_hermitian(self):
        return True


@transform
def povm_dilate(tape: qml.tape.QuantumTape) -> (Sequence[qml.tape.QuantumTape], Callable):
    ops = tape.operations.copy()

    # more POVMs are technically possible, but keep this prototype simple
    is_prob = isinstance(tape.measurements[0], qml.measurements.ProbabilityMP)
    povm = tape.measurements[0].obs

    # this prototype uses the methods from:
    # https://physics.stackexchange.com/questions/448756/neumarks-theorem-equivalence-of-povm-and-projective-measurements
    # https://quantumcomputing.stackexchange.com/questions/10239/how-can-i-fill-a-unitary-knowing-only-its-first-column

    # TODO: figure out how to dilate using operator arithmetic

    sqrt_ops = qml.math.stack([qml.matrix(qml.pow(op, 0.5)) for op in povm])
    dim = 2 ** len(tape.wires)

    extra_dims = 2 ** int(np.ceil(np.log2(len(povm))))
    if len(povm) <= extra_dims:
        # extend the list of operators with zero matrices (which have probability 0 of occurring)
        sqrt_ops = qml.math.concatenate([sqrt_ops, qml.math.zeros((extra_dims - len(povm), dim, dim))], 0)

    v = qml.math.zeros((extra_dims * dim, dim)).astype(np.complex128)
    l, w = qml.math.linalg.eigh((sqrt_ops[0] + qml.math.eye(dim)) / 2)
    v[:dim] = w * qml.math.sqrt(l[None, :])

    v[dim : extra_dims * dim] = qml.math.reshape(
        sqrt_ops[1:extra_dims] @ qml.math.linalg.inv(qml.math.conj(v[:dim].T)) / 2,
        ((extra_dims - 1) * dim, dim)
    )

    U = 2 * v @ qml.math.conj(v.T) - qml.math.eye(extra_dims * dim)

    new_wires = list(range(-int(np.log2(extra_dims)), 0))
    ops.append(qml.QubitUnitary(U, new_wires + tape.wires))

    if is_prob:
        mp = qml.probs(wires=new_wires + tape.wires)
        new_tape = qml.tape.QuantumScript(ops, [mp], shots=tape.shots)

        def processing_fn(res):
            res = qml.math.reshape(res[0], (extra_dims, dim))
            return qml.math.sum(res, 1)[:len(povm)]

    else:
        mp = qml.sample(wires=new_wires + tape.wires)
        new_tape = qml.tape.QuantumScript(ops, [mp], shots=tape.shots)

        def processing_fn(res):
            res = res[0][:, :len(new_wires)]
            return qml.math.dot(res, 1 << qml.math.arange(len(new_wires) - 1, -1, -1))

    return [new_tape], processing_fn
