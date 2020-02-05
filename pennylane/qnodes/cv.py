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
CV parameter shift quantum node.

Provides analytic differentiation for Gaussian operations succeeded by
first and second order observables.
"""
import copy

import numpy as np

import pennylane as qml
from pennylane.operation import ObservableReturnTypes

from .base import QuantumFunctionError
from .jacobian import JacobianQNode


class CVQNode(JacobianQNode):
    """Quantum node for CV parameter shift analytic differentiation"""

    def _best_method(self, idx):
        """Determine the correct partial derivative computation method for a free parameter.

        Use the parameter-shift analytic method iff every gate that depends on the parameter supports it.
        If not, use the finite difference method only (one would have to use it anyway).

        Note that if even one dependent Operation does not support differentiation,
        we cannot differentiate with respect to this parameter at all.

        Args:
            idx (int): free parameter index

        Returns:
            str: partial derivative method to be used
        """
        # pylint: disable=too-many-branches
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
                x = op.grad_method

                if op.grad_method == "A":
                    # for parameter-shift compatible CV gates we need to check both the
                    # intervening gates, and the type of the observable
                    if any(not k.supports_heisenberg for k in S):
                        # non-Gaussian operators present
                        x = "F"
                    elif ob.return_type == ObservableReturnTypes.Variance:
                        if ob.ev_order is None or ob.ev_order >= 2:
                            x = "F"
                    elif ob.ev_order is None or ob.ev_order >= 2:
                        x = "B"
                    else:
                        x = "A"

                # If there is no path between them, p.d. is zero
                # Otherwise, use finite differences
                best[k_op, k_ob] = "0" if not S else x

            if all(k == "0" for k in best[k_op, :]):
                op.use_method = "0"
            elif "F" in best[k_op, :]:
                # one non-analytic item makes the whole numeric
                op.use_method = "F"
            elif "B" in best[k_op, :]:
                op.use_method = "B"
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

    @staticmethod
    def _transform_observable(obs, w, Z):
        """Apply a Gaussian linear transformation to each index of an observable.

        Args:
            obs (Observable): observable to transform
            w (int): number of wires in the circuit
            Z (array[float]): Heisenberg picture representation of the linear transformation

        Returns:
            Observable: transformed observable
        """
        q = obs.heisenberg_obs(w)

        if q.ndim != obs.ev_order:
            raise QuantumFunctionError(
                "Mismatch between the polynomial order of observable and its Heisenberg representation"
            )

        qp = q @ Z
        if q.ndim == 2:
            # 2nd order observable
            qp = qp + qp.T
        elif q.ndim > 2:
            raise NotImplementedError("Transforming observables of order > 2 not implemented.")
        return qml.expval(qml.PolyXP(qp, wires=range(w), do_queue=False))

    def _pd_analytic(self, idx, args, kwargs, **options):
        """Partial derivative of the node using the analytic parameter shift method.

        The 2nd order method can handle also first order observables, but
        1st order method may be more efficient unless it's really easy to
        experimentally measure arbitrary 2nd order observables.

        Args:
            idx (int): flattened index of the parameter wrt. which the p.d. is computed
            args (array[float]): flattened positional arguments at which to evaluate the p.d.
            kwargs (dict[str, Any]): auxiliary arguments

        Keyword Args:
            force_order2 (bool): iff True, use the order-2 method even if not necessary

        Returns:
            array[float]: partial derivative of the node
        """
        force_order2 = options.get("force_order2", False)

        n = self.num_variables
        w = self.num_wires
        pd = np.zeros(self.output_dim)
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

            if not force_order2 and op.use_method != "B":
                # basic parameter-shift method, for Gaussian CV gates
                # succeeded by order-1 observables
                # evaluate the circuit at two points with shifted parameter values
                y2 = np.asarray(self.evaluate(shift_p1, kwargs))
                y1 = np.asarray(self.evaluate(shift_p2, kwargs))
                pd += (y2 - y1) * multiplier
            else:
                # order-2 parameter-shift method, for gaussian CV gates
                # succeeded by order-2 observables
                # evaluate transformed observables at the original parameter point
                # first build the Heisenberg picture transformation matrix Z
                self._set_variables(shift_p1, kwargs)
                Z2 = op.heisenberg_tr(w)
                self._set_variables(shift_p2, kwargs)
                Z1 = op.heisenberg_tr(w)
                Z = (Z2 - Z1) * multiplier  # derivative of the operation

                unshifted_args = np.r_[args, args[idx]]
                self._set_variables(unshifted_args, kwargs)
                Z0 = op.heisenberg_tr(w, inverse=True)
                Z = Z @ Z0

                # conjugate Z with all the descendant operations
                B = np.eye(1 + 2 * w)
                B_inv = B.copy()
                for BB in self._op_descendants(op, "G"):
                    if not BB.supports_heisenberg:
                        # if the descendant gate is non-Gaussian in parameter-shift differentiation
                        # mode, then there must be no observable following it.
                        continue
                    B = BB.heisenberg_tr(w) @ B
                    B_inv = B_inv @ BB.heisenberg_tr(w, inverse=True)
                Z = B @ Z @ B_inv  # conjugation

                # transform the descendant observables into their derivatives using Z
                desc = self._op_descendants(op, "O")
                obs = [self._transform_observable(x, w, Z) for x in desc]
                # Measure the transformed observables.
                # The other observables do not depend on this parameter instance,
                # hence their partial derivatives are zero.
                res = self.evaluate_obs(obs, unshifted_args, kwargs)

                # add the measured pd's to the correct locations
                inds = [self.circuit.observables.index(x) for x in desc]
                pd[inds] += res

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

            # CV first order observable
            # get the heisenberg representation
            # This will be a real 1D vector representing the
            # first order observable in the basis [I, x, p]
            A = e._heisenberg_rep(e.parameters)  # pylint: disable=protected-access

            # take the outer product of the heisenberg representation
            # with itself, to get a square symmetric matrix representing
            # the square of the observable
            A = np.outer(A, A)
            new = qml.expval(qml.PolyXP(A, w, do_queue=False))

            # replace the var(A) observable with <A^2>
            self.circuit.update_node(e, new)
            new_observables.append(new)

        # calculate the analytic derivatives of the <A^2> observables
        pdA2 = self._pd_analytic(idx, args, kwargs, force_order2=True)

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
