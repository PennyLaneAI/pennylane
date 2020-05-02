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
Differentiable quantum nodes.
"""
from collections.abc import Iterable

import numpy as np

from pennylane.operation import ObservableReturnTypes
from pennylane.utils import _flatten, _inv_dict

from .base import BaseQNode, QuantumFunctionError

DEFAULT_STEP_SIZE = 0.3
DEFAULT_STEP_SIZE_ANALYTIC = 1e-7


class JacobianQNode(BaseQNode):
    """Quantum node that can be differentiated with respect to its positional parameters.
    """

    def __init__(self, func, device, mutable=True, **kwargs):
        super().__init__(func, device, mutable=mutable, **kwargs)

        self.par_to_grad_method = None
        """dict[int, str]: map from flattened quantum function positional parameter index
        to the gradient method to be used with that parameter"""

        analytic = getattr(self.device, "analytic", False)
        """bool: whether the device runs in analytic mode; this attribute is
        not defined for hardware devices so set to False in such cases"""

        default_step_size = DEFAULT_STEP_SIZE_ANALYTIC if analytic else DEFAULT_STEP_SIZE
        self._h = kwargs.get("h", default_step_size)
        """float: step size for the finite difference method"""

        self._order = kwargs.get("order", 1)
        """float: order for the finite difference method"""

    metric_tensor = None

    @property
    def interface(self):
        """str, None: automatic differentiation interface used by the node, if any"""
        return None

    @property
    def h(self):
        """float: step size for the finite difference method"""
        return self._h

    @h.setter
    def h(self, value):
        self._h = value

    @property
    def order(self):
        """float: order for the finite difference method"""
        return self._order

    @order.setter
    def order(self, value):
        self._order = value

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
        self.par_to_grad_method = {k: self._best_method(k) for k in self.variable_deps}

    def _best_method(self, idx):
        """Determine the correct partial derivative computation method for a free parameter.

        Note that if even one dependent Operation does not support differentiation,
        we cannot differentiate with respect to this parameter at all.

        .. note::

            The ``JacobianQNode`` only supports numerical differentiation, so
            this method will always return either ``"F"`` or ``None``. If an inheriting
            QNode supports analytic differentiation for certain operations, make sure
            that this method is overwritten appropriately to return ``"A"`` where
            required.

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
                best[k_op, k_ob] = "0" if not S else "F"

            if all(k == "0" for k in best[k_op, :]):
                op.use_method = "0"
            else:
                op.use_method = "F"

        # if all ops that depend on the free parameter have a best method
        # of "0", then we can skip the partial derivative altogether
        if all(o.use_method == "0" for o in ops):
            return "0"

        # one nondifferentiable item makes the whole nondifferentiable
        if any(o.use_method is None for o in ops):
            return None

        return "F"

    def jacobian(self, args, kwargs=None, *, wrt=None, method="best", options=None):
        r"""Compute the Jacobian of the QNode.

        Returns the Jacobian of the parametrized quantum circuit encapsulated in the QNode.
        The Jacobian is returned as a two-dimensional array. The (possibly nested) input arguments
        of the QNode are :func:`flattened <_flatten>` so the QNode can be interpreted as a simple
        :math:`\mathbb{R}^m \to \mathbb{R}^n` function.

        The Jacobian can be computed using several methods:

        * Finite differences (``'F'``). The first-order method evaluates the circuit at
          :math:`n+1` points of the parameter space, the second-order method at :math:`2n` points,
          where ``n = len(wrt)``.

        * Analytic method (``'A'``). Analytic, if implemented by the inheriting QNode.

        * Best known method for each parameter (``'best'``): uses the analytic method if
          possible, otherwise finite difference.

        * Device method (``'device'``): Delegates the computation of the Jacobian to the
          device executing the circuit. Only supported by devices that provide their
          own method for computing derivatives; support can be checked by
          querying the device capabilities: ``dev.capabilities()['provides_jacobian']`` must
          return ``True``. Examples of supported devices include the experimental
          :class:`"default.tensor.tf" <~.DefaultTensorTF>` device.

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

        Returns:
            array[float]: Jacobian, shape ``(n, len(wrt))``, where ``n`` is the number of outputs returned by the QNode
        """
        # pylint: disable=too-many-branches,too-many-statements
        if not isinstance(args, Iterable):
            args = (args,)
        kwargs = kwargs or {}

        # apply defaults
        kwargs = self._default_args(kwargs)

        options = options or {}

        # Add the step size into the options, if it was not there already
        if "h" not in options.keys():
            options = {"h": self.h, **options}
        if "order" not in options.keys():
            options = {"order": self._order, **options}

        # (re-)construct the circuit if necessary
        if self.circuit is None or self.mutable:
            self._construct(args, kwargs)

        returns_samples = [
            str(ob)
            for ob in self.circuit.observables
            if ob.return_type is ObservableReturnTypes.Sample
        ]
        if returns_samples:
            raise QuantumFunctionError(
                "Circuits that include sampling can not be differentiated. "
                "The following observables include sampling: {}".format("; ".join(returns_samples))
            )

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

        # check if the method can be used on the requested parameters
        method_map = _inv_dict(self.par_to_grad_method)

        def inds_using(m):
            """Intersection of ``wrt`` with free params indices whose best grad method is m."""
            return method_map.get(m, set()).intersection(wrt)

        # are we trying to differentiate wrt. params that don't support any method?
        bad = inds_using(None)
        if bad:
            raise ValueError("Cannot differentiate with respect to the parameters {}.".format(bad))

        if method == "device":
            self._set_variables(args, kwargs)
            return self.device.jacobian(
                self.circuit.operations, self.circuit.observables, self.variable_deps
            )

        if method == "A":
            bad = inds_using("F")
            if bad:
                raise ValueError(
                    "The analytic gradient method cannot be "
                    "used with the parameters {}.".format(bad)
                )
            # only variants of the analytic method remain
            method = self.par_to_grad_method
        elif method == "F":
            # use the requested method for every parameter
            method = {k: "F" for k in wrt}
        elif method == "best":
            # use best known method for each parameter
            method = self.par_to_grad_method
        else:
            raise ValueError("Unknown gradient method.")

        if "F" in method.values():
            if options.get("order", 1) == 1:
                # the value of the circuit at args, computed only once here
                options["y0"] = np.asarray(self.evaluate(args, kwargs))

        # In the following, to evaluate the Jacobian we call self.evaluate several times using
        # modified args (and possibly modified circuit Operators).
        # We do not want evaluate to call _construct again. This would only be necessary if the
        # auxiliary args changed, since only they can change the structure of the circuit,
        # and we do not modify them. To achieve this, we temporarily make the circuit immutable.
        mutable = self.mutable
        self.mutable = False

        # flatten the nested Sequence of input arguments
        flat_args = np.array(list(_flatten(args)), dtype=float)
        variances_required = any(
            ob.return_type is ObservableReturnTypes.Variance for ob in self.circuit.observables
        )

        # compute the partial derivative wrt. each parameter using the appropriate method
        grad = np.zeros((self.output_dim, len(wrt)), dtype=float)
        for i, k in enumerate(wrt):
            par_method = method[k]

            if par_method == "0":
                # unused/invisible, partial derivatives wrt. this param are zero
                continue

            if par_method == "A":
                if variances_required:
                    grad[:, i] = self._pd_analytic_var(k, flat_args, kwargs, **options)
                else:
                    grad[:, i] = self._pd_analytic(k, flat_args, kwargs, **options)
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
        h = options.get("h", self.h)
        order = options.get("order", self.order)

        shift_args = args.copy()
        if order == 1:
            y0 = options.get("y0", None)
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

    def _pd_analytic(self, idx, args, kwargs, **options):
        """Partial derivative of the node using an analytic method.

        Args:
            idx (int): flattened index of the parameter wrt. which the p.d. is computed
            args (array[float]): flattened positional arguments at which to evaluate the p.d.
            kwargs (dict[str, Any]): auxiliary arguments

        Returns:
            array[float]: partial derivative of the node
        """
        raise NotImplementedError

    def _pd_analytic_var(self, idx, args, kwargs, **options):
        """Partial derivative of the variance of an observable using an analytic method.

        Args:
            idx (int): flattened index of the parameter wrt. which the p.d. is computed
            args (array[float]): flattened positional arguments at which to evaluate the p.d.
            kwargs (dict[str, Any]): auxiliary arguments

        Returns:
            array[float]: partial derivative of the node
        """
        raise NotImplementedError

    def to_torch(self):
        """Attach the Torch interface to the Jacobian QNode.

        Raises:
            QuantumFunctionError: if PyTorch is not installed
        """
        # Placing slow imports here, in case the user does not use the Torch interface
        # pylint: disable=import-outside-toplevel
        try:  # pragma: no cover
            from pennylane.interfaces.torch import to_torch as _to_torch
        except ImportError:  # pragma: no cover
            raise QuantumFunctionError(
                "PyTorch not found. Please install " "PyTorch to enable the 'torch' interface."
            ) from None

        return _to_torch(self)

    def to_tf(self):
        """Attach the TensorFlow interface to the Jacobian QNode.

        Raises:
            QuantumFunctionError: if TensorFlow >= 1.12 is not installed
        """
        # Placing slow imports here, in case the user does not use the TF interface
        # pylint: disable=import-outside-toplevel
        try:  # pragma: no cover
            from pennylane.interfaces.tf import to_tf as _to_tf
        except ImportError:  # pragma: no cover
            raise QuantumFunctionError(
                "TensorFlow not found. Please install "
                "the latest version of TensorFlow to enable the 'tf' interface."
            ) from None

        return _to_tf(self)

    def to_autograd(self):
        """Attach the TensorFlow interface to the Jacobian QNode.

        Raises:
            QuantumFunctionError: if Autograd is not installed
        """
        # Placing slow imports here, in case the user does not use the TF interface
        # pylint: disable=import-outside-toplevel
        try:  # pragma: no cover
            from pennylane.interfaces.autograd import to_autograd as _to_autograd
        except ImportError:  # pragma: no cover
            raise QuantumFunctionError(
                "Autograd not found. Please install "
                "the latest version of Autograd to enable the 'autograd' interface."
            ) from None

        return _to_autograd(self)
