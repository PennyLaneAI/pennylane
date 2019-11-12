# Copyright 2019 Xanadu Quantum Technologies Inc.

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
from pennylane.qnode_new.jacobian import JacobianQNode
from pennylane.qnode_new.qnode import QuantumFunctionError


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
        # TODO make sure Identity is recognized as gaussian

        # pylint: disable=too-many-return-statements
        def best_for_op(op):
            """Returns the best gradient method for the Operation op."""
            if op.grad_method == "A":
                # op is Gaussian and has the heisenberg_* methods
                successor_ops = self._op_descendants(op, "G")
                obs_successors = self._op_descendants(op, "E")

                # check that all successor ops are also Gaussian
                if not all(x.supports_heisenberg for x in successor_ops):
                    non_gaussian_ops = [x for x in successor_ops if not x.supports_heisenberg]
                    # a non-Gaussian successor is OK if it isn't succeeded by any observables
                    for x in non_gaussian_ops:
                        if self._op_descendants(x, "E"):
                            return "F"

                # check successor EVs, if any order-2 observables are found return 'A2', else return 'A'
                for observable in obs_successors:
                    if observable.ev_order is None:
                        # ev_order of None corresponds to a non-Gaussian observable
                        return "F"
                    if observable.ev_order == 2:
                        if observable.return_type is ObservableReturnTypes.Variance:
                            # second order observables don't support
                            # analytic diff of variances
                            return "F"
                        op.grad_method = "A2"  # bit of a hack
                return "A"
            return op.grad_method

        # operations that depend on this free parameter
        ops = [d.op for d in self.variable_deps[idx]]
        methods = list(map(best_for_op, ops))

        if all(k in ("A", "A2") for k in methods):
            return "A"

        if None in methods:
            return None

        return "F"

    @staticmethod
    def _transform_observable(obs, w, Z):
        """Apply a linear transformation on an observable.

        Args:
            obs (Observable): observable to transform
            w (int): number of wires
            Z (array[float]): Heisenberg picture representation of the linear transformation

        Returns:
            Observable: transformed observable
        """
        q = obs.heisenberg_obs(w)

        if q.ndim != obs.ev_order:
            raise QuantumFunctionError(
                "Mismatch between polynomial order of observable and heisenberg representation"
            )

        qp = q @ Z
        if q.ndim == 2:
            # 2nd order observable
            qp = qp + qp.T
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
        pd = 0.0
        # find the Operators in which the free parameter appears, use the product rule
        for op, p_idx in self.variable_deps[idx]:

            # We temporarily edit the Operator such that parameter p_idx is replaced by a new one,
            # which we can modify without affecting other Operators depending on the original.
            orig = op.params[p_idx]
            assert orig.idx == idx
            assert orig.name is None

            # reference to a new, temporary parameter with index n, otherwise identical with orig
            temp_var = copy.copy(orig)
            temp_var.idx = n
            op.params[p_idx] = temp_var

            multiplier, shift = op.get_parameter_shift(p_idx)

            # shifted parameter values
            shift_p1 = np.r_[args, args[idx] + shift]
            shift_p2 = np.r_[args, args[idx] - shift]

            if not force_order2 and op.grad_method != "A2":
                # basic parameter-shift method, for Gaussian CV gates succeeded by order-1 observables
                # evaluate the circuit at two points with shifted parameter values
                y2 = np.asarray(self.evaluate(shift_p1, kwargs))
                y1 = np.asarray(self.evaluate(shift_p2, kwargs))
                pd += (y2 - y1) * multiplier
            else:
                # order-2 parameter-shift method, for gaussian CV gates succeeded by order-2 observables
                # evaluate transformed observables at the original parameter point
                # first build the Z transformation matrix
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

                # transform the descendant observables
                desc = self._op_descendants(op, "E")
                obs = [
                    self._transform_observable(x, w, Z) if x in desc else x
                    for x in self.circuit.observables
                ]

                # measure transformed observables
                pd += self.evaluate_obs(obs, unshifted_args, kwargs)

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
