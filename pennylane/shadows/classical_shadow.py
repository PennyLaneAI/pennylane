# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Classical Shadows baseclass with processing functions"""
import warnings
from collections.abc import Iterable
import pennylane.numpy as np
import pennylane as qml


@autograd.extend.primitive
def pauli_expval(params, bits, recipes, word, grad_data, shift_results):
    # determine snapshots and qubits that match the word
    # -1 in the word indicates a wild card
    T = recipes.shape[0]
    grad_data, one_unshifted = grad_data

    id_mask = word == -1

    indices = recipes == word
    indices = np.logical_or(indices, np.tile(id_mask, (T, 1)))
    indices = np.all(indices, axis=1)

    bits = bits[:, np.logical_not(id_mask)]
    bits = np.sum(bits, axis=1) % 2

    return np.where(indices, 1 - 2 * bits, 0) * 3 ** np.count_nonzero(np.logical_not(id_mask))


def _param_shift_grad(grad_data, shift_results, id_mask):
    # Apply the same squeezing as in qml.QNode to make the transform output consistent.
    # pylint: disable=protected-access
    grad_data, one_unshifted = grad_data
    shift_results = [qml.math.squeeze(res, axis=0) for res in shift_results]

    grads = []
    start = 1 if one_unshifted else 0
    r0 = shift_results[0]

    for num_tapes, coeffs, fn, unshifted_coeff in grad_data:

        if num_tapes == 0:
            # parameter has zero gradient
            grads.append(qml.math.zeros_like(shift_results[0]))
            continue

        res = shift_results[start : start + num_tapes]
        start = start + num_tapes

        # individual post-processing of e.g. Hamiltonian grad tapes
        if fn is not None:
            res = fn(res)

        # compute the linear combination of results and coefficients
        res = qml.math.stack(res)

        bits = res[:, 0]
        # (tapes, T, n)

        bits = bits[:, :, np.logical_not(id_mask)]
        bits = np.sum(bits, axis=2) % 2
        # bits = np.sum(bits, axis=2, keepdims=True) % 2
        # bits = np.tile(bits, reps=(1, 1, res.shape[3])) / np.count_nonzero(np.logical_not(id_mask))

        g = qml.math.tensordot(bits, qml.math.convert_like(coeffs, bits), [[0], [0]])

        if unshifted_coeff is not None:
            # add the unshifted term
            g = g + unshifted_coeff * r0

        grads.append(g)

    # The following is for backwards compatibility; currently, the device stacks multiple
    # measurement arrays, even if not the same size, resulting in a ragged array.
    # In the future, we might want to change this so that only tuples of arrays are returned.
    for i, g in enumerate(grads):
        if getattr(g, "dtype", None) is np.dtype("object") and qml.math.ndim(g) > 0:
            grads[i] = qml.math.hstack(g)

    return qml.math.stack(grads)


def pauli_expval_vjp(ans, params, bits, recipes, word, grad_data, shift_results):
    # gradient of the parity of bits with respect to the tape parameters
    T = recipes.shape[0]
    grads = _param_shift_grad(grad_data, shift_results, word == -1)

    id_mask = word == -1

    indices = recipes == word
    indices = np.logical_or(indices, np.tile(id_mask, (T, 1)))
    indices = np.all(indices, axis=1)

    def vjp(dy):
        dy = np.where(indices, dy, 0)
        return [-2 * 3 ** np.count_nonzero(np.logical_not(id_mask)) * dy @ grads.T]

    return vjp


autograd.extend.defvjp(pauli_expval, pauli_expval_vjp)


class ClassicalShadowV2:
    def __init__(self, qnode):
        self.qnode = qnode
        self.parameters = None

        # reset these arguments after every QNode evaluation
        self.bitstrings = None
        self.recipes = None
        self.grad_data = None
        self.shifted_tape_results = []

    def _grad_data(self, tape):
        argnum = tape.trainable_params
        gradient_recipes = [None] * len(argnum)
        grad_tapes = []
        grad_data = []
        at_least_one_unshifted = False

        for idx, _ in enumerate(tape.trainable_params):
            op, _ = tape.get_operation(idx)

            recipe = qml.gradients.parameter_shift._choose_recipe(argnum, idx, gradient_recipes, None, tape)
            recipe, at_least_one_unshifted, unshifted_coeff = qml.gradients.parameter_shift._extract_unshifted(
                recipe, at_least_one_unshifted, None, grad_tapes, tape
            )
            coeffs, multipliers, op_shifts = recipe.T

            # generate the gradient tapes
            g_tapes = qml.gradients.generate_shifted_tapes(tape, idx, op_shifts, multipliers)
            grad_data.append((len(g_tapes), coeffs, None, unshifted_coeff))

        return grad_data, at_least_one_unshifted

    def construct(self, *args, **kwargs):
        self.qnode.construct(args, kwargs)

        # find the trainable parameters of the tape
        params = self.qnode.tape.get_parameters(trainable_only=False)
        self.qnode.tape.trainable_parameters = qml.math.get_trainable_indices(params)
        self.parameters = autograd.builtins.tuple(
            [autograd.builtins.list(self.qnode.tape.get_parameters())]
        )

        self.bitstrings, self.recipes = qml.math.squeeze(qml.execute(
            [self.qnode.tape],
            device=self.qnode.device,
            gradient_fn=None,
            interface="autograd"
        )[0], axis=0)
        self.grad_data = self._grad_data(self.qnode.tape)

        grad_tapes = qml.gradients.param_shift(self.qnode.tape)[0]
        self.shifted_tape_results = qml.execute(grad_tapes, device=self.qnode.device, gradient_fn=None, interface="autograd")

    def local_snapshots(self, wires=None, snapshots=None):
        r"""Compute the T x n x 2 x 2 local snapshots
        i.e. compute :math:`3 U_i^\dagger |b_i \rangle \langle b_i| U_i - 1` for each qubit and each snapshot
        """
        bitstrings = self.bitstrings
        recipes = self.recipes

        if isinstance(snapshots, int):
            pick_snapshots = np.random.rand(self.snapshots, size=(snapshots))
            bitstrings = bitstrings[pick_snapshots]
            recipes = recipes[pick_snapshots]

        if isinstance(wires, Iterable):
            bitstrings = bitstrings[:, wires]
            recipes = recipes[:, wires]

        unitaries = [
            np.array([[0, 1], [1, 0]]),
            np.array([[0, -1j], [1j, 0]]),
            np.array([[1, 0], [0, -1]])
        ]

        T, n = bitstrings.shape

        U = np.empty((T, n, 2, 2), dtype="complex")
        for i, u in enumerate(unitaries):
            U[np.where(recipes == i)] = u

        state = ((1 - 2 * bitstrings[:, :, None, None]) * U + np.eye(2)) / 2

        return 3 * state - np.eye(2)[None, None, :, :]

    def _pauli_expval(self, word):
        return pauli_expval(self.parameters, self.bitstrings, self.recipes, np.array(word), self.grad_data, self.shifted_tape_results)

    def _tensor_product(self, *arrs):
        prod = arrs[0]
        for arr in arrs[1:]:
            prod = np.kron(prod, arr)
        return prod

    def _contains(self, arr, ele):
        for other in arr:
            if np.all(ele == other):
                return True
        return False

    def global_snapshots(self, wires=None, snapshots=None):
        """Compute the T x 2**n x 2**n global snapshots"""

        pauli_ops = np.stack([
            np.array([[0, 1], [1, 0]]),
            np.array([[0, -1j], [1j, 0]]),
            np.array([[1, 0], [0, -1]]),
            np.eye(2)
        ], requires_grad=False)

        T, n = self.recipes.shape

        recipes = np.array(self.recipes, requires_grad=False)

        processed_words = []

        snapshots = self._tensor_product(*[np.eye(2) for _ in range(n)])
        snapshots = np.tile(snapshots, (T, 1, 1))

        for word in recipes:
            for id_mask in product(*[[False, True] for _ in range(n)]):
                id_mask = np.array(id_mask)
                if np.all(id_mask):
                    continue

                masked_word = np.where(id_mask, -1, word).astype(np.int32)

                if isinstance(masked_word, autograd.numpy.numpy_boxes.ArrayBox):
                    masked_word = masked_word._value

                if self._contains(processed_words, masked_word):
                    continue

                processed_words.append(masked_word)

                # tensor_op = self._tensor_product(*pauli_ops[masked_word])
                tensor_op = self._tensor_product(*np.take(pauli_ops, masked_word, axis=0))
                # (2^n, 2^n)

                expvals = self._pauli_expval(masked_word)
                # (T, )

                snapshots = snapshots + expvals[:, None, None] * tensor_op[None, :, :]

        return snapshots / 2 ** n

    def _expval_observable(self, observable, k):
        """Compute expectation values of Pauli-string type observables"""
        if isinstance(observable, qml.operation.Tensor):
            os = np.asarray([qml.matrix(o) for o in observable.obs])
        else:
            os = np.asarray([qml.matrix(observable)])  # wont work with other interfaces

        # Picking only the wires with non-trivial Pauli strings avoids computing unnecessary tr(rho_i 1)=1
        local_snapshots = self.local_snapshots(wires=observable.wires)

        means = []
        step = self.snapshots // k

        # Compute expectation value of snapshot = sum_t prod_i tr(rho_i^t P_i)
        # res_i^t = tr(rho_i P_i)
        # res = np.trace(local_snapshots @ os, axis1=-1, axis2=-2)
        res = np.einsum('abcd,bdc->ab', local_snapshots, os)
        res = np.real(res)
        # res^t = prod_i res_i^t

        # res = np.prod(res, axis=-1)
        prods = []
        for i in range(res.shape[0]):
            prod = 1
            for j in range(res.shape[1]):
                prod = prod * res[i, j]
            prods.append(prod)

        res = np.stack(prods)
        return np.real(np.mean(res))

    def expval_pauli(self, word):
        expvals = self._pauli_expval(word)
        return np.mean(expvals)
