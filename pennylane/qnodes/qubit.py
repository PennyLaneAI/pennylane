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
Qubit parameter shift quantum node.

Provides analytic differentiation for all one-parameter gates where the generator
only has two unique eigenvalues; this includes one-parameter single-qubit gates.
"""
import itertools
import copy

import numpy as np
from scipy import linalg

import pennylane as qml
from pennylane.measure import var
from pennylane.utils import expand

from pennylane.operation import Observable, ObservableReturnTypes

from .base import QuantumFunctionError
from .jacobian import JacobianQNode


class QubitQNode(JacobianQNode):
    """Quantum node for qubit parameter shift analytic differentiation"""

    def _best_method(self, idx):
        """Determine the correct partial derivative computation method for a free parameter.

        Use the parameter-shift analytic method iff every gate that depends on the parameter supports it.
        If not, use the finite difference method only.

        Note that if even one dependent Operation does not support differentiation,
        we cannot differentiate with respect to this parameter at all.

        Args:
            idx (int): free parameter index

        Returns:
            str: partial derivative method to be used
        """
        # operations that depend on this free parameter
        ops = [d.op for d in self.variable_deps[idx]]

        # Observables in the circuit
        # (the topological order is the queue order)
        observables = self.circuit.observables_in_order

        # an empty list to store the 'best' partial derivative method
        # for each operator/observable pair
        best = np.empty((len(ops), len(observables)), dtype=object)

        # find the best supported partial derivative method for each operator
        for k_op, op in enumerate(ops):
            if op.grad_method is None:
                # one nondifferentiable item makes the whole nondifferentiable
                op.use_method = None
                continue

            # loop over all observables
            for k_ob, ob in enumerate(observables):
                # get the set of operations betweens the
                # operation and the observable
                S = self.circuit.nodes_between(op, ob)

                # If there is no path between them, p.d. is zero
                # Otherwise, use finite differences
                best[k_op, k_ob] = "0" if not S else op.grad_method

            if all(k == "0" for k in best[k_op, :]):
                # one nondifferentiable item makes the whole nondifferentiable
                op.use_method = "0"
            elif "F" in best[k_op, :]:
                # one non-analytic item makes the whole numeric
                op.use_method = "F"
            else:
                op.use_method = "A"

        # if all ops that depend on the free parameter have a best method
        # of "0", then we can skip the partial derivative altogether
        if all(o.use_method == "0" for o in ops):
            return "0"

        # one nondifferentiable item makes the whole nondifferentiable
        if any(o.use_method is None for o in ops):
            return None

        # one non-analytic item makes the whole numeric
        if any(o.use_method == "F" for o in ops):
            return "F"

        return "A"

    def _pd_analytic(self, idx, args, kwargs, **options):
        """Partial derivative of the node using the analytic parameter shift method.
        Args:
            idx (int): flattened index of the parameter wrt. which the p.d. is computed
            args (array[float]): flattened positional arguments at which to evaluate the p.d.
            kwargs (dict[str, Any]): auxiliary arguments

        Returns:
            array[float]: partial derivative of the node
        """
        n = self.num_variables
        pd = 0.0
        # find the Operators in which the free parameter appears, use the product rule
        for op, p_idx in self.variable_deps[idx]:

            # We temporarily edit the Operator such that parameter p_idx is replaced by a new one,
            # which we can modify without affecting other Operators depending on the original.
            orig = op.params[p_idx]
            assert orig.idx == idx

            # reference to a new, temporary parameter with index n, otherwise identical with orig
            temp_var = copy.copy(orig)
            temp_var.idx = n
            op.params[p_idx] = temp_var

            multiplier, shift = op.get_parameter_shift(p_idx)

            # shifted parameter values
            shift_p1 = np.r_[args, args[idx] + shift]
            shift_p2 = np.r_[args, args[idx] - shift]

            # evaluate the circuit at two points with shifted parameter values
            y2 = np.asarray(self.evaluate(shift_p1, kwargs))
            y1 = np.asarray(self.evaluate(shift_p2, kwargs))
            pd += (y2 - y1) * multiplier

            # restore the original parameter
            op.params[p_idx] = orig

        return pd

    def _pd_analytic_var(self, idx, args, kwargs, **options):
        """Partial derivative of the variance of an observable using the parameter-shift method.

        Args:
            idx (int): flattened index of the parameter wrt. which the p.d. is computed
            args (array[float]): flattened positional arguments at which to evaluate the p.d.
            kwargs (dict[str, Any]): auxiliary arguments

        Returns:
            array[float]: partial derivative of the node
        """
        # boolean mask: elements are True where the return type is a variance, False for expectations
        where_var = [
            e.return_type is ObservableReturnTypes.Variance for e in self.circuit.observables
        ]
        var_observables = [
            e for e in self.circuit.observables if e.return_type == ObservableReturnTypes.Variance
        ]

        # first, replace each var(A) with <A^2>
        new_observables = []
        for e in var_observables:
            # need to calculate d<A^2>/dp
            w = e.wires

            if e.name == "Hermitian":
                # since arbitrary Hermitian observables
                # are not guaranteed to be involutory, need to take them into
                # account separately to calculate d<A^2>/dp

                A = e.params[0]  # Hermitian matrix
                # if not np.allclose(A @ A, np.identity(A.shape[0])):
                new = qml.expval(qml.Hermitian(A @ A, w, do_queue=False))
            else:
                # involutory, A^2 = I
                # For involutory observables (A^2 = I) we have d<A^2>/dp = 0
                new = qml.expval(qml.Hermitian(np.identity(2 ** len(w)), w, do_queue=False))

            # replace the var(A) observable with <A^2>
            self.circuit.update_node(e, new)
            new_observables.append(new)

        # calculate the analytic derivatives of the <A^2> observables
        pdA2 = self._pd_analytic(idx, args, kwargs)

        # restore the original observables, but convert their return types to expectation
        for e, new in zip(var_observables, new_observables):
            self.circuit.update_node(new, e)
            e.return_type = ObservableReturnTypes.Expectation

        # evaluate <A>
        evA = np.asarray(self.evaluate(args, kwargs))

        # evaluate the analytic derivative of <A>
        pdA = self._pd_analytic(idx, args, kwargs)

        # restore return types
        for e in var_observables:
            e.return_type = ObservableReturnTypes.Variance

        # return d(var(A))/dp = d<A^2>/dp -2 * <A> * d<A>/dp for the variances,
        # d<A>/dp for plain expectations
        return np.where(where_var, pdA2 - 2 * evA * pdA, pdA)

    def _construct_metric_tensor(self, *, diag_approx=False):
        """Construct metric tensor subcircuits for qubit circuits.

        Constructs a set of quantum circuits for computing a block-diagonal approximation of the
        Fubini-Study metric tensor on the parameter space of the variational circuit represented
        by the QNode, using the Quantum Geometric Tensor.

        If the parameter appears in a gate :math:`G`, the subcircuit contains
        all gates which precede :math:`G`, and :math:`G` is replaced by the variance
        value of its generator.

        Args:
            diag_approx (bool): iff True, use the diagonal approximation

        Raises:
            QuantumFunctionError: if a metric tensor cannot be generated because no generator
                was defined

        """
        # pylint: disable=too-many-statements, too-many-branches

        self._metric_tensor_subcircuits = {}
        for queue, curr_ops, param_idx, _ in self.circuit.iterate_parametrized_layers():
            obs = []
            scale = []

            Ki_matrices = []
            KiKj_matrices = []
            Ki_ev = []
            KiKj_ev = []
            V = None

            # for each operation in the layer, get the generator and convert it to a variance
            for n, op in enumerate(curr_ops):
                gen, s = op.generator
                w = op.wires

                if gen is None:
                    raise QuantumFunctionError(
                        "Can't generate metric tensor, operation {}"
                        "has no defined generator".format(op)
                    )

                # get the observable corresponding to the generator of the current operation
                if isinstance(gen, np.ndarray):
                    # generator is a Hermitian matrix
                    variance = var(qml.Hermitian(gen, w, do_queue=False))

                    if not diag_approx:
                        Ki_matrices.append((n, expand(gen, w, self.num_wires)))

                elif issubclass(gen, Observable):
                    # generator is an existing PennyLane operation
                    variance = var(gen(w, do_queue=False))

                    if not diag_approx:
                        if issubclass(gen, qml.PauliX):
                            mat = np.array([[0, 1], [1, 0]])
                        elif issubclass(gen, qml.PauliY):
                            mat = np.array([[0, -1j], [1j, 0]])
                        elif issubclass(gen, qml.PauliZ):
                            mat = np.array([[1, 0], [0, -1]])

                        Ki_matrices.append((n, expand(mat, w, self.num_wires)))

                else:
                    raise QuantumFunctionError(
                        "Can't generate metric tensor, generator {}"
                        "has no corresponding observable".format(gen)
                    )

                obs.append(variance)
                scale.append(s)

            if not diag_approx:
                # In order to compute the block diagonal portion of the metric tensor,
                # we need to compute 'second order' <psi|K_i K_j|psi> terms.

                for i, j in itertools.product(range(len(Ki_matrices)), repeat=2):
                    # compute the matrices representing all K_i K_j terms
                    obs1 = Ki_matrices[i]
                    obs2 = Ki_matrices[j]
                    KiKj_matrices.append(((obs1[0], obs2[0]), obs1[1] @ obs2[1]))

                V = np.identity(2 ** self.num_wires, dtype=np.complex128)

                # generate the unitary operation to rotate to
                # the shared eigenbasis of all observables
                for _, term in Ki_matrices:
                    _, S = linalg.eigh(V.conj().T @ term @ V)
                    V = np.round(V @ S, 15)

                V = V.conj().T

                # calculate the eigenvalues for
                # each observable in the shared eigenbasis
                for idx, term in Ki_matrices:
                    eigs = np.diag(V @ term @ V.conj().T).real
                    Ki_ev.append((idx, eigs))

                for idx, term in KiKj_matrices:
                    eigs = np.diag(V @ term @ V.conj().T).real
                    KiKj_ev.append((idx, eigs))

            self._metric_tensor_subcircuits[param_idx] = {
                "queue": queue,
                "observable": obs,
                "Ki_expectations": Ki_ev,
                "KiKj_expectations": KiKj_ev,
                "eigenbasis_matrix": V,
                "result": None,
                "scale": scale,
            }

    def metric_tensor(self, args, kwargs=None, *, diag_approx=False, only_construct=False):
        """Evaluate the value of the metric tensor.

        Args:
            args (tuple[Any]): positional (differentiable) arguments
            kwargs (dict[str, Any]): auxiliary arguments
            diag_approx (bool): iff True, use the diagonal approximation
            only_construct (bool): Iff True, construct the circuits used for computing
                the metric tensor but do not execute them, and return None.

        Returns:
            array[float]: metric tensor
        """
        # pylint:disable=too-many-branches
        kwargs = kwargs or {}
        kwargs = self._default_args(kwargs)

        if self.circuit is None or self.mutable:
            # construct the circuit
            self._construct(args, kwargs)

        if self._metric_tensor_subcircuits is None:
            self._construct_metric_tensor(diag_approx=diag_approx)

        if only_construct:
            return None

        # temporarily store the parameter values in the Variable class
        self._set_variables(args, kwargs)

        tensor = np.zeros([self.num_variables, self.num_variables])

        # execute constructed metric tensor subcircuits
        for params, circuit in self._metric_tensor_subcircuits.items():
            self.device.reset()

            s = np.array(circuit["scale"])
            V = circuit["eigenbasis_matrix"]

            if not diag_approx:
                # block diagonal approximation

                unitary_op = qml.QubitUnitary(V, wires=list(range(self.num_wires)), do_queue=False)

                if isinstance(self.device, qml.QubitDevice):
                    ops = circuit["queue"] + [unitary_op] + [qml.expval(qml.PauliZ(0))]
                    circuit_graph = qml.CircuitGraph(ops, self.variable_deps)
                    self.device.execute(circuit_graph)
                else:
                    self.device.execute(
                        circuit["queue"] + [unitary_op],
                        [
                            qml.expval(qml.PauliZ(wire))
                            for wire in list(range(self.device.num_wires))
                        ],
                    )

                probs = list(self.device.probability())

                first_order_ev = np.zeros([len(params)])
                second_order_ev = np.zeros([len(params), len(params)])

                for idx, ev in circuit["Ki_expectations"]:
                    first_order_ev[idx] = ev @ probs

                for idx, ev in circuit["KiKj_expectations"]:
                    # idx is a 2-tuple (i, j), representing
                    # generators K_i, K_j
                    second_order_ev[idx] = ev @ probs

                    # since K_i and K_j are assumed to commute,
                    # <psi|K_j K_i|psi> = <psi|K_i K_j|psi>,
                    # and thus the matrix of second-order expectations
                    # is symmetric
                    second_order_ev[idx[1], idx[0]] = second_order_ev[idx]

                g = np.zeros([len(params), len(params)])

                for i, j in itertools.product(range(len(params)), repeat=2):
                    g[i, j] = (
                        s[i]
                        * s[j]
                        * (second_order_ev[i, j] - first_order_ev[i] * first_order_ev[j])
                    )

                row = np.array(params).reshape(-1, 1)
                col = np.array(params).reshape(1, -1)
                circuit["result"] = np.diag(g)
                tensor[row, col] = g

            else:
                # diagonal approximation
                if isinstance(self.device, qml.QubitDevice):
                    circuit_graph = qml.CircuitGraph(
                        circuit["queue"] + circuit["observable"], self.variable_deps
                    )
                    variances = self.device.execute(circuit_graph)
                else:
                    variances = self.device.execute(circuit["queue"], circuit["observable"])

                circuit["result"] = s ** 2 * variances
                tensor[np.array(params), np.array(params)] = circuit["result"]

        return tensor
