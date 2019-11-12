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
Differentiable quantum nodes.
"""

from collections.abc import Sequence
import copy

import numpy as np

import pennylane as qml
from pennylane.operation import ObservableReturnTypes, CV
from pennylane.utils import _flatten, _inv_dict
from pennylane.qnode_new.qnode import QNode, QuantumFunctionError


class JacobianQNode(QNode):
    """Quantum node that can be differentiated with respect to its positional parameters.
    """

    @property
    def interface(self):
        """str, None: automatic differentiation interface used by the node, if any"""
        return None

    def __init__(self, func, device, mutable=True, properties=None):
        super().__init__(func, device, mutable=mutable, properties=properties)

        #: dict[int, str]: map from free parameter index to the gradient method to be used with that parameter
        self.par_to_grad_method = None

    def __repr__(self):
        """String representation."""
        detail = "<QNode (differentiable): device='{}', func={}, wires={}, interface={}>"
        return detail.format(
            self.device.short_name, self.func.__name__, self.num_wires, self.interface
        )

    def _construct(self, args, kwargs):
        """Constructs the quantum circuit graph by calling the quantum function.

        Like :meth:`.QNode._construct`, additionally determines the best gradient computation method
        for each free parameter.
        """
        super()._construct(args, kwargs)
        self.par_to_grad_method = {k: self._best_method(k) for k in self.variable_deps}

        temp = [
            str(ob)
            for ob in self.circuit.observables
            if ob.return_type is ObservableReturnTypes.Sample
        ]
        if temp:
            raise QuantumFunctionError(
                "Circuits that include sampling can not be differentiated. "
                "The following observables include sampling: {}".format("; ".join(temp))
            )

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
            "Returns the best gradient method for the Operation op."
            # for qubit operations, other ops do not affect the choice

            if not isinstance(op, CV):
                return op.grad_method

            # for CV ops it is more complicated
            if op.grad_method == "A":
                # op is Gaussian and has the heisenberg_* methods

                obs_successors = self._op_descendants(op, "E")
                # if not obs_successors:
                #     # op is not succeeded by any observables, thus analytic method is OK    FIXME actually we can ignore it....
                #     return "A"

                # check that all successor ops are also Gaussian
                successor_ops = self._op_descendants(op, "G")
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

    def jacobian(self, args, kwargs=None, *, wrt=None, method="B", options=None):
        r"""Compute the Jacobian of the QNode.

        Returns the Jacobian of the parametrized quantum circuit encapsulated in the QNode.
        The Jacobian is returned as a two-dimensional array. The (possibly nested) input arguments
        of the QNode are :func:`flattened <_flatten>` so the QNode can be interpreted as a simple
        :math:`\mathbb{R}^m \to \mathbb{R}^n` function.

        The Jacobian can be computed using several methods:

        * Finite differences (``'F'``). The first-order method evaluates the circuit at
          :math:`n+1` points of the parameter space, the second-order method at :math:`2n` points,
          where ``n = len(wrt)``.

        * Parameter-shift method (``'A'``). Analytic, works for all one-parameter gates where the generator
          only has two unique eigenvalues; this includes one-parameter single-qubit gates.
          Additionally, can be used in CV systems for Gaussian circuits containing only first-
          and second-order observables.
          The circuit is evaluated twice for each incidence of each parameter in the circuit.

        * Best known method for each parameter (``'B'``): uses the parameter-shift method if
          possible, otherwise finite difference.

        * Device method (``'D'``): Delegates the computation of the Jacobian to the
          device executing the circuit.

        .. note::
           The finite difference method is sensitive to statistical noise in the circuit output,
           since it compares the output at two points infinitesimally close to each other. Hence the
           'F' method requires exact expectation values, i.e., ``analytic=True`` in simulation plugins.

        Args:
            args (nested Iterable[float] or float): positional arguments to the quantum function (differentiable)
            kwargs (dict[str, Any]): auxiliary arguments to the quantum function (not differentiable)
            wrt (Sequence[int] or None): Indices of the flattened positional parameters with respect
                to which to compute the Jacobian. None means all the parameters.
                Note that you cannot compute the Jacobian with respect to the kwargs.
            method (str): Jacobian computation method, in ``{'F', 'A', 'B', 'D'}``, see above
            options (dict[str, Any]): additional options for the computation methods

                * h (float): finite difference method step size
                * order (int): finite difference method order, 1 or 2
                * force_order2 (bool): force the parameter-shift method to use the second-order method

        Returns:
            array[float]: Jacobian, shape ``(n, len(wrt))``, where ``n`` is the number of outputs returned by the QNode
        """
        # pylint: disable=too-many-branches,too-many-statements
        # arrays are not Sequences... but they ARE Iterables? FIXME decide which types we accept! cf. _flatten
        # TODO: rename "A"->"parameter_shift", "B"->"best", "F"->"finite_diff", "D"->"device"
        if not isinstance(args, (Sequence, np.ndarray)):
            args = (args,)
        kwargs = kwargs or {}

        # apply defaults
        kwargs = self._default_args(kwargs)

        options = options or {}

        # (re-)construct the circuit if necessary
        if self.circuit is None or self.mutable:
            self._construct(args, kwargs)

        if wrt is None:
            wrt = range(self.num_variables)
        else:
            if min(wrt) < 0 or max(wrt) >= self.num_variables:
                raise ValueError(
                    "Tried to compute the gradient with respect to free parameters {} "
                    "(this node has {} free parameters).".format(wrt, self.num_variables)
                )
            if len(wrt) != len(set(wrt)):  # set removes duplicates
                raise ValueError("Parameter indices must be unique.")

        if method == "D":
            return self.device.jacobian(args, kwargs, wrt, self.circuit)  # FIXME placeholder

        # In the following, to evaluate the Jacobian we call self.evaluate several times using
        # modified args (and possibly modified circuit Operators).
        # We do not want evaluate to call _construct again. This would only be necessary if the
        # auxiliary args changed, since only they can change the structure of the circuit,
        # and we do not modify them. To achieve this, we temporarily make the circuit immutable.
        mutable = self.mutable
        self.mutable = False

        # check if the method can be used on the requested parameters
        method_map = _inv_dict(self.par_to_grad_method)

        def inds_using(m):
            """Intersection of ``wrt`` with free params indices whose best grad method is m."""
            return method_map.get(m, set()).intersection(wrt)

        # are we trying to differentiate wrt. params that don't support any method?
        bad = inds_using(None)
        if bad:
            raise ValueError("Cannot differentiate wrt. parameters {}.".format(bad))

        if method in ("A", "F"):
            if method == "A":
                bad = inds_using("F")
                if bad:
                    raise ValueError(
                        "The parameter-shift gradient method cannot be "
                        "used with the parameters {}.".format(bad)
                    )
            method = {k: method for k in wrt}
        elif method == "B":
            # use best known method for each parameter
            method = self.par_to_grad_method
        else:
            raise ValueError("Unknown gradient method.")

        if "F" in method.values():
            if options.get("order", 1) == 1:
                # the value of the circuit at args, computed only once here
                options["y0"] = np.asarray(self.evaluate(args, kwargs))

        # flatten the nested Sequence of input arguments
        flat_args = np.array(list(_flatten(args)), dtype=float)
        variances_required = any(
            ob.return_type is ObservableReturnTypes.Variance for ob in self.circuit.observables
        )

        # compute the partial derivative wrt. each parameter using the appropriate method
        grad = np.zeros((self.output_dim, len(wrt)), dtype=float)
        for i, k in enumerate(wrt):
            if k not in self.variable_deps:
                # unused parameter, partial derivatives wrt. it are zero
                continue

            par_method = method[k]
            if par_method == "A":
                if variances_required:
                    grad[:, i] = self._pd_parameter_shift_var(k, flat_args, kwargs)
                else:
                    grad[:, i] = self._pd_parameter_shift(k, flat_args, kwargs, **options)
            elif par_method == "F":
                grad[:, i] = self._pd_finite_diff(k, flat_args, kwargs, **options)
            else:
                raise ValueError("Unknown gradient method.")

        self.mutable = mutable  # restore original mutability
        return grad

    def _pd_finite_diff(self, idx, args, kwargs, **options):
        """Partial derivative of the node using the finite difference method.

        Args:
            idx (int): flattened index of the parameter wrt. which the p.d. is computed
            args (array[float]): flattened positional arguments at which to evaluate the p.d.
            kwargs (dict[str, Any]): auxiliary arguments

        Keyword Args:
            y0 (array[float], None): value of the circuit at the given arguments
            h (float): step size
            order (int): finite difference method order, 1 or 2

        Returns:
            array[float]: partial derivative of the node
        """
        y0 = options.get("y0", None)
        h = options.get("h", 1e-7)
        order = options.get("order", 1)

        shift_args = args.copy()
        if order == 1:
            # shift the parameter by h
            shift_args[idx] += h
            y = np.asarray(self.evaluate(shift_args, kwargs))
            return (y - y0) / h

        if order == 2:
            # symmetric difference
            # shift the parameter by +-h/2
            shift_args[idx] += 0.5 * h
            y2 = np.asarray(self.evaluate(shift_args, kwargs))
            shift_args[idx] = args[idx] - 0.5 * h
            y1 = np.asarray(self.evaluate(shift_args, kwargs))
            return (y2 - y1) / h

        raise ValueError("Order must be 1 or 2.")

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

    def _pd_parameter_shift(self, idx, args, kwargs, **options):
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
                # basic parameter-shift method, for discrete gates and gaussian CV gates succeeded by order-1 observables
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

    def _pd_parameter_shift_var(self, idx, args, kwargs):
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

            if self.model == "qubit":
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

            else:  # CV circuit, first order observable
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
        # FIXME the force_order2 use here is convoluted and could be better
        pdA2 = self._pd_parameter_shift(idx, args, kwargs, force_order2=(self.model == "cv"))

        # restore the original observables, but convert their return types to expectation
        for e, new in zip(var_observables, new_observables):
            self.circuit.update_node(new, e)
            e.return_type = ObservableReturnTypes.Expectation

        # evaluate <A>
        evA = np.asarray(self.evaluate(args, kwargs))

        # evaluate the analytic derivative of <A>
        pdA = self._pd_parameter_shift(idx, args, kwargs)

        # restore return types
        for e in var_observables:
            e.return_type = ObservableReturnTypes.Variance

        # return d(var(A))/dp = d<A^2>/dp -2 * <A> * d<A>/dp for the variances,
        # d<A>/dp for plain expectations
        return np.where(where_var, pdA2 - 2 * evA * pdA, pdA)
