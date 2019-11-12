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

from collections.abc import Iterable
import copy

import numpy as np

import pennylane as qml
from pennylane.operation import ObservableReturnTypes
from pennylane.utils import _flatten, _inv_dict
from pennylane.qnode_new.qnode import QNode, QuantumFunctionError



class JacobianQNode(QNode):
    """Quantum node that can be differentiated with respect to its positional parameters.
    """
    def __init__(self, func, device, mutable=True, properties=None):
        super().__init__(func, device, mutable=mutable, properties=properties)

        self.par_to_grad_method = None
        """dict[int, str]: map from flattened quantum function positional parameter index
        to the gradient method to be used with that parameter"""

    @property
    def interface(self):
        """str, None: automatic differentiation interface used by the node, if any"""
        return None


    def __repr__(self):
        """String representation."""
        detail = "<QNode (differentiable): device='{}', func={}, wires={}, interface={}>"
        return detail.format(
            self.device.short_name, self.func.__name__, self.num_wires, self.interface
        )


    def _construct(self, args, kwargs):
        """Constructs the quantum circuit graph by calling the quantum function.

        Like :meth:`.QNode._construct`, additionally determines the best gradient computation method
        for each positional parameter.
        """
        super()._construct(args, kwargs)

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

        self.par_to_grad_method = {k: self._best_method(k) for k in self.variable_deps}

    @staticmethod
    def _best_method_combined(methods, B_is_A=False):
        """Combine partial derivative computation methods.

        Given an Iterable of partial derivative computation methods, returns the best method
        compatible with all of them.

        Args:
            methods (Iterable[str, None]): p.d. computation methods {None, 'F', 'V', 'B', 'A', '0'}
            B_is_A (bool): iff True, subsume 'B' into 'A'

        Returns:
            str: p.d. computation method compatible with all the given ones
        """
        if None in methods:
            return None  # one nondifferentiable item makes the whole nondifferentiable
        if 'F' in methods:
            return 'F'   # ditto with items not supporting the par-shift method
        if all(k == '0' for k in methods):
            return '0'   # all the partial derivatives are zero
        if 'V' in methods:
            return 'V'   # variance par-shift method can also handle the B and A cases
        if 'B' in methods:
            return 'A' if B_is_A else 'B'  # observable-transforming par-shift method
        return 'A'  # basic par-shift method


    def _best_method(self, idx):
        """Determine the best partial derivative computation method for a positional parameter.

        Uses the parameter-shift analytic method iff every gate that depends on the parameter supports it.
        If not, use the finite difference method only (one would have to use it anyway).

        Note that if even one dependent Operation does not support differentiation,
        we cannot differentiate with respect to this parameter at all.

        Args:
            idx (int): flattened positional parameter index

        Returns:
            str: partial derivative method to be used
        """
        # pylint: disable=too-many-nested-blocks,too-many-branches

        # all ops which depend on the given parameter
        ops = [d.op for d in self.variable_deps[idx]]
        observables = self.circuit.observables_in_order  # the topological order is the queue order

        # find the best supported partial derivative method for each operator/observable pair
        best = np.empty((len(ops), len(observables)), dtype=object)
        for ka, a in enumerate(ops):
            gm = a.grad_method
            for kb, b in enumerate(observables):
                S = self.circuit.nodes_between(a, b)
                if not S:  # no path between them, p.d. is zero
                    x = '0'
                elif self.model == "qubit":
                    # for qubit circuits the other ops do not matter
                    if gm == 'A' and b.return_type == ObservableReturnTypes.Variance:
                        x = 'V'
                    else:
                        # includes the "op not differentiable" case
                        x = gm
                else:
                    if gm == 'A':
                        # for parameter-shift compatible CV gates we need to check both the
                        # intervening gates, and the type of the observable
                        if any(not k.supports_heisenberg for k in S):  # nongaussian operators
                            x = 'F'
                        elif b.return_type == ObservableReturnTypes.Variance:
                            if b.ev_order >= 2:
                                x = 'F'
                            else:
                                x = 'V'
                        elif b.ev_order >= 2:
                            x = 'B'
                        else:
                            x = 'A'
                    else:
                        # includes the "op not differentiable" case
                        x = gm

                best[ka, kb] = x

            # All the observables are evaluated simultaneously, hence for each dependent op
            # we find out the best method compatible with all the observables.
            a.best_method = self._best_method_combined(best[ka, :])

        return self._best_method_combined([x.best_method for x in ops], B_is_A=True)


    def jacobian(self, args, kwargs=None, *, wrt=None, method='best', options=None):
        r"""Compute the Jacobian of the QNode.

        Returns the Jacobian of the parametrized quantum circuit encapsulated in the QNode.
        The Jacobian is returned as a two-dimensional array. The (possibly nested) input arguments
        of the QNode are :func:`flattened <_flatten>` so the QNode can be interpreted as a simple
        :math:`\mathbb{R}^m \to \mathbb{R}^n` function.

        The Jacobian can be computed using several methods:

        * Finite differences (``'F'``). The first-order method evaluates the circuit at
          :math:`n+1` points of the parameter space, the second-order method at :math:`2n` points,
          where ``n = len(wrt)``.

        * Parameter-shift method (``'A'``). Analytic, works for all one-parameter gates where the
          generator only has two unique eigenvalues; this includes one-parameter single-qubit gates.
          Additionally, can be used in CV circuits iff there are no non-Gaussian operators
          on the paths between the parametrized operation and the observables.
          The circuit is evaluated twice for each incidence of each parameter in the circuit.

        * Best known method for each parameter (``'best'``): uses the parameter-shift method if
          possible, otherwise finite difference.

        * Device method (``'device'``): Delegates the computation of the Jacobian to the
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
            method (str): Jacobian computation method, in ``{'F', 'A', 'best', 'device'}``, see above
            options (dict[str, Any]): additional options for the computation methods

                * h (float): finite difference method step size
                * order (int): finite difference method order, 1 or 2
                * force_order2 (bool): force the parameter-shift method to use the observable-transforming variant

        Returns:
            array[float]: Jacobian, shape ``(n, len(wrt))``, where ``n`` is the number of outputs returned by the QNode
        """
        # TODO: rename "A"->"parameter_shift", "F"->"finite_diff"

        # pylint: disable=too-many-branches,too-many-statements
        # Note: arrays are not Sequences, but they are Iterables
        if not isinstance(args, Iterable):
            args = (args,)
        kwargs = kwargs or {}

        # apply defaults
        kwargs = self._default_args(kwargs)

        options = options or {}

        # (re-)construct the circuit if necessary
        if self.circuit is None or self.mutable:
            self._construct(args, kwargs)

        # check that the wrt parameters are ok
        if wrt is None:
            wrt = range(self.num_variables)
        else:
            if min(wrt) < 0 or max(wrt) >= self.num_variables:
                raise ValueError(
                    "Tried to compute the gradient with respect to parameters {} "
                    "(this node has {} parameters).".format(wrt, self.num_variables)
                )
            if len(wrt) != len(set(wrt)):  # set removes duplicates
                raise ValueError("Parameter indices must be unique.")

        # check if the requested method can be used on the requested parameters
        method_to_pars = _inv_dict(self.par_to_grad_method)
        def inds_using(m):
            """Intersection of ``wrt`` with flattened positional params indices whose best grad method is m."""
            return method_to_pars.get(m, set()).intersection(wrt)

        # are we trying to differentiate wrt. params that don't support any method?
        bad = inds_using(None)
        if bad:
            raise ValueError('Cannot differentiate with respect to the parameters {}.'.format(bad))

        if method == 'device':
            return self.device.jacobian(args, kwargs, wrt, self.circuit)  # FIXME placeholder
        if method == 'A':
            bad = inds_using('F')
            if bad:
                raise ValueError("The parameter-shift gradient method cannot be "
                                 "used with the parameters {}.".format(bad))
            # only variants of the par-shift method remain
            method = self.par_to_grad_method
        elif method == 'F':
            # use the requested method for every parameter
            method = {k: 'F' for k in wrt}
        elif method == 'best':
            # use best known method for each parameter
            method = self.par_to_grad_method
        else:
            raise ValueError('Unknown gradient method.')

        # split the kwargs meant for different submethods
        def split_dict(d, items):
            return {k: v for k, v in d.items() if k in items}

        if 'F' in method.values():
            fd_options = split_dict(options, ['h', 'order'])
            y0 = None
            if fd_options.get('order', 1) == 1:
                # the value of the circuit at args, computed only once here
                y0 = np.asarray(self.evaluate(args, kwargs))
        if 'A' in method.values():
            parshift_options = split_dict(options, ['force_order2'])

        # In the following, to evaluate the Jacobian we call self.evaluate several times using
        # modified args (and possibly modified circuit Operators).
        # We do not want evaluate to call _construct again. This would only be necessary if the
        # auxiliary args changed, since only they can change the structure of the circuit,
        # and we do not modify them. To achieve this, we temporarily make the circuit immutable.
        mutable = self.mutable
        self.mutable = False

        # flatten the nested Sequence of input arguments
        flat_args = np.array(list(_flatten(args)), dtype=float)

        # compute the partial derivative wrt. each parameter using the appropriate method
        grad = np.zeros((self.output_dim, len(wrt)), dtype=float)
        for i, k in enumerate(wrt):
            par_method = method[k]
            if par_method == '0':
                # unused/invisible, partial derivatives wrt. this param are zero
                continue
            if par_method == 'V':
                grad[:, i] = self._pd_parameter_shift_var(k, flat_args, kwargs)
            elif par_method == 'A':
                grad[:, i] = self._pd_parameter_shift(k, flat_args, kwargs, **parshift_options)
            elif par_method == 'F':
                grad[:, i] = self._pd_finite_diff(k, flat_args, kwargs, y0=y0, **fd_options)
            else:
                raise ValueError('Unknown gradient method.')

        self.mutable = mutable  # restore original mutability
        return grad


    def _pd_finite_diff(self, idx, args, kwargs, *, y0=None, h=1e-7, order=1):
        """Partial derivative of the node using the finite difference method.

        Args:
            idx (int): flattened index of the parameter wrt. which the p.d. is computed
            args (array[float]): flattened positional arguments at which to evaluate the p.d.
            kwargs (dict[str, Any]): auxiliary arguments
            y0 (array[float], None): value of the circuit at the given arguments
            h (float): step size
            order (int): finite difference method order, 1 or 2

        Returns:
            array[float]: partial derivative of the node
        """
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
                "Mismatch between the polynomial order of observable and its Heisenberg representation."
            )

        qp = q @ Z
        if q.ndim == 2:
            # 2nd order observable
            qp = qp +qp.T
        elif q.ndim > 2:
            raise NotImplementedError("Transforming observables of order > 2 not implemented.")
        return qml.expval(qml.PolyXP(qp, wires=range(w), do_queue=False))


    def _pd_parameter_shift(self, idx, args, kwargs, *, force_order2=False):
        """Partial derivative of the node using the analytic parameter-shift method.

        The 2nd order method can handle also first order observables, but
        1st order method may be more efficient unless it's really easy to
        experimentally measure arbitrary 2nd order observables.

        Args:
            idx (int): flattened index of the parameter wrt. which the p.d. is computed
            args (array[float]): flattened positional arguments at which to evaluate the p.d.
            kwargs (dict[str, Any]): auxiliary arguments
            force_order2 (bool): iff True, use the order-2 method even if not necessary

        Returns:
            array[float]: partial derivative of the node
        """
        n = self.num_variables
        w = self.num_wires
        pd = 0.0
        # find the Operators in which the parameter appears, use the product rule
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

            if not force_order2 and op.best_method != 'B':
                # basic parameter-shift method, for discrete gates and gaussian CV gates
                # succeeded by order-1 observables
                # evaluates the circuit at two points with shifted parameter values
                y2 = np.asarray(self.evaluate(shift_p1, kwargs))
                y1 = np.asarray(self.evaluate(shift_p2, kwargs))
                pd += (y2 - y1) * multiplier
            else:
                # observable-transforming parameter-shift method, for gaussian CV gates
                # succeeded by observables of order 2 and above
                # evaluates transformed observables at the original parameter point
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
                desc = self._op_descendants(op, "O")
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
        """Partial derivative of a node involving variances of observables using the parameter-shift method.

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
        # the force_order2 use here is a bit hacky
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
