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
CV parameter shift quantum tape.

Provides analytic differentiation for variational circuits with parametrized Gaussian CV gates
and first- and second-order observables.
"""
# pylint: disable=attribute-defined-outside-init
import itertools

import numpy as np

import pennylane as qml
from pennylane.beta.queuing import MeasurementProcess, expval

from .tape import QuantumTape


class CVParamShiftTape(QuantumTape):
    r"""Quantum tape for CV parameter-shift analytic differentiation method.

    This class extends the :class:`~.jacobian` method of the quantum tape
    to support analytic gradients of Gaussian CV operations using the parameter-shift rule.
    This gradient method returns *exact* gradients, and can be computed directly
    on quantum hardware. Simply pass ``method=analytic`` when computing the Jacobian:

    >>> tape.jacobian(dev, method="analytic")

    For more details on the quantum tape, please see :class:`~.QuantumTape`.

    **Gradients of expectation values**

    For a variational circuit :math:`U(p_i)|0\rangle` with :math:`N` parameters,
    consider the expectation value of an observable :math:`O`:

    .. math:: f(p_i)  = \langle O \rangle(p_i) = \langle 0 | U(p_i)^\dagger O U(p_i) | 0\rangle.

    The gradient of this expectation value can be calculated using :math:`2N` evaluations
    using the parameter-shift rule:

    .. math:: \frac{\partial f}{\partial p_i} = \frac{1}{2\sin s} \left[ f(p_i + s) - f(p_i -s) \right].

    **Gradients of variances**

    We can extend this to the variance, :math:`g(p_i)=\langle O^2 \rangle (p_i) - \langle O \rangle(p_i)^2`,
    by noting that:

    .. math::

        \frac{\partial g}{\partial p_i}= \frac{\partial}{\partial p_i} \langle O^2 \rangle (p_i)
        - 2 f(p_i) \frac{\partial f}{\partial p_i}.

    This results in :math:`4N + 1` evaluations.

    In the case where :math:`O` is involutory (:math:`O^2 = I`), the first term in the above
    expression vanishes, and we are simply left with

    .. math:: \frac{\partial g}{\partial p_i} = - 2 f(p_i) \frac{\partial f}{\partial p_i},

    allowing us to compute the gradient using :math:`2N + 1` evaluations.
    """

    def _update_circuit_info(self):
        super()._update_circuit_info()

        # set parameter_shift as the analytic_pd method
        self.analytic_pd = self.parameter_shift

        # Make a copy of the original measurements; we will be mutating them
        # during the parameter shift method.
        self._original_measurements = self._measurements.copy()

        # check if the quantum tape contains any measurements
        self.var_mask = [m.return_type is qml.operation.Variance for m in self.measurements]

        if any(self.var_mask):
            # The tape contains variances.
            # Set parameter_shift_var as the analytic_pd method
            self.analytic_pd = self.parameter_shift_var

            # Finally, store the locations of any variance measurements in the
            # measurement queue.
            self.var_idx = np.where(self.var_mask)[0]

    def _grad_method(self, idx, use_graph=True, default_method="F"):
        op = self._par_info[idx]["op"]

        if op.grad_method is None:
            return None

        # an empty list to store the 'best' partial derivative method
        # for each observable
        best = []

        # loop over all observables
        for ob in self.observables:
            # get the set of operations betweens the
            # operation and the observable
            ops_between = self.graph.nodes_between(op, ob)

            if not ops_between:
                # if there is no path between the operation and the observable,
                # the operator has a zero gradient.
                best.append("0")
                continue

            if op.grad_method == "A":
                # Operation supports the CV parameter-shift rule.
                # For parameter-shift compatible CV gates, we need to check both the
                # intervening gates, and the type of the observable.

                if any(not k.supports_heisenberg for k in ops_between):
                    # non-Gaussian operators present in-between the operation
                    # and the observable. Must fallback to numeric differentiation.
                    best.append("F")

                elif ob.return_type is qml.operation.Probability:
                    # probability is a higher order expectation, and thus does not permit
                    # the CV parameter-shift
                    best.append("F")

                elif ob.return_type is qml.operation.Variance:
                    # we only support analytic variance gradients for
                    # first orderobservables
                    if ob.ev_order == 1:
                        best.append("A")
                    else:
                        best.append("F")

                elif ob.ev_order != 1:
                    # If the observable is not first order, we must use the second order
                    # CV parameter shift rule
                    best.append("A2")

                else:
                    # If all other conditions do not hold, we can support
                    # the first order parameter-shift rule.
                    best.append("A")

        if all(k == "0" for k in best):
            return "0"

        if "F" in best:
            # one non-analytic item makes the whole operation gradient numeric
            return "F"

        if "A2" in best:
            # one second order observable makes the whole operation gradient
            # require the second order parameter-shift rule.
            return "A2"

        return "A"

    def _op_descendants(self, op, only):
        """Descendants of the given operator in the quantum circuit.

        Args:
            op (Operator): operator in the quantum circuit
            only (str, None): the type of descendants to return.

                - ``'G'``: only return non-observables (default)
                - ``'O'``: only return observables
                - ``None``: return all descendants

        Returns:
            list[Operator]: descendants in a topological order
        """
        succ = self.graph.descendants_in_order((op,))

        if only == "O":
            return list(filter(qml.circuit_graph._is_observable, succ))

        if only == "G":
            return list(itertools.filterfalse(qml.circuit_graph._is_observable, succ))

        return succ

    @staticmethod
    def _transform_observable(obs, w, Z, device_wires):
        """Apply a Gaussian linear transformation to each index of an observable.

        Args:
            obs (Observable): observable to transform
            w (int): number of wires in the circuit
            Z (array[float]): Heisenberg picture representation of the linear transformation
            device_wires (Wires): wires on the device that the observable gets applied to

        Returns:
            .MeasurementProcess: measurement process with transformed observable
        """
        q = obs.heisenberg_obs(device_wires)

        if q.ndim != obs.ev_order:
            raise qml.QuantumFunctionError(
                "Mismatch between the polynomial order of observable and its Heisenberg representation"
            )

        qp = q @ Z

        if q.ndim == 2:
            # 2nd order observable
            qp = qp + qp.T

        elif q.ndim > 2:
            raise NotImplementedError("Transforming observables of order > 2 not implemented.")

        return expval(qml.PolyXP(qp, wires=range(w)))

    def jacobian(self, device, params=None, **options):
        # The parameter_shift_var method needs to evaluate the circuit
        # at the unshifted parameter values; these are stored in the
        # self._evA attribute. Here, we set the value of the attribute to None
        # before each Jacobian call, so that the expectation value is calculated only once.
        self._evA = None
        return super().jacobian(device, params, **options)

    def parameter_shift(self, idx, device, params, **options):
        r"""Partial derivative using the first- or second-order CV parameter-shift rule of a
        tape consisting of *only* expectation values of observables.

        .. note::

            The 2nd order method can handle also first order observables, but
            1st order method may be more efficient unless it's really easy to
            experimentally measure arbitrary 2nd order observables.

        Args:
            idx (int): trainable parameter index to differentiate with respect to
            device (.Device, .QubitDevice): a PennyLane device
                that can execute quantum operations and return measurement statistics
            params (list[Any]): the quantum tape operation parameters

        Keyword Args:
            force_order2 (bool): iff True, use the order-2 method even if not necessary

        Returns:
            array[float]: 1-dimensional array of length determined by the tape output
            measurement statistics
        """
        grad_method = self._par_info[idx]["grad_method"]

        if options.get("force_order2", grad_method == "A2"):
            return self.parameter_shift_second_order(idx, device, params, **options)

        return self.parameter_shift_first_order(idx, device, params, **options)

    def parameter_shift_first_order(self, idx, device, params, **options):
        r"""Partial derivative using the first-order CV parameter-shift rule of a
        tape consisting of *only* expectation values of observables.

        Args:
            idx (int): trainable parameter index to differentiate with respect to
            device (.Device, .QubitDevice): a PennyLane device
                that can execute quantum operations and return measurement statistics
            params (list[Any]): the quantum tape operation parameters

        Returns:
            array[float]: 1-dimensional array of length determined by the tape output
            measurement statistics
        """
        op = self._par_info[idx]["op"]
        p_idx = self._par_info[idx]["p_idx"]

        recipe = op.grad_recipe[p_idx]
        c, s = (0.5, np.pi / 2) if recipe is None else recipe

        shift = np.zeros_like(params)
        shift[idx] = s

        shift_forward = np.array(self.execute_device(params + shift, device))
        shift_backward = np.array(self.execute_device(params - shift, device))

        return c * (shift_forward - shift_backward)

    def parameter_shift_second_order(self, idx, device, params, **options):
        r"""Partial derivative using the second-order CV parameter-shift rule of a
        tape consisting of *only* expectation values of observables.

        Args:
            idx (int): trainable parameter index to differentiate with respect to
            device (.Device, .QubitDevice): a PennyLane device
                that can execute quantum operations and return measurement statistics
            params (list[Any]): the quantum tape operation parameters

        Returns:
            array[float]: 1-dimensional array of length determined by the tape output
            measurement statistics
        """
        op = self._par_info[idx]["op"]
        p_idx = self._par_info[idx]["p_idx"]

        recipe = op.grad_recipe[p_idx]
        c, s = (0.5, np.pi / 2) if recipe is None else recipe

        shift = np.zeros_like(params)
        shift[idx] = s

        # evaluate transformed observables at the original parameter point
        # first build the Heisenberg picture transformation matrix Z
        self.set_parameters(params + shift)
        Z2 = op.heisenberg_tr(device.wires)

        self.set_parameters(params - shift)
        Z1 = op.heisenberg_tr(device.wires)

        # derivative of the operation
        Z = (Z2 - Z1) * c

        self.set_parameters(params)
        Z0 = op.heisenberg_tr(device.wires, inverse=True)
        Z = Z @ Z0

        # conjugate Z with all the descendant operations
        B = np.eye(1 + 2 * self.num_wires)
        B_inv = B.copy()

        for BB in self._op_descendants(op, "G"):
            if not BB.supports_heisenberg:
                # if the descendant gate is non-Gaussian in parameter-shift differentiation
                # mode, then there must be no observable following it.
                continue

            B = BB.heisenberg_tr(device.wires) @ B
            B_inv = B_inv @ BB.heisenberg_tr(device.wires, inverse=True)

        Z = B @ Z @ B_inv  # conjugation

        # transform the descendant observables into their derivatives using Z
        desc = self._op_descendants(op, "O")
        self._measurements = [self._transform_observable(x, self.num_wires, Z, device.wires) for x in desc]

        # Measure the transformed observables.
        # The other observables do not depend on this parameter instance,
        # hence their partial derivatives are zero.
        res = np.array(self.execute_device(params, device))

        # add the measured pd's to the correct locations
        idx = [self.graph.observables.index(x) for x in desc]
        grad = np.zeros_like(res)
        grad[idx] = res

        # restore the original measurements
        self._measurements = self._original_measurements

        return grad

    def parameter_shift_var(self, idx, device, params, **options):
        r"""Partial derivative using the first-order or second-order parameter-shift rule of a tape
        consisting of a mixture of expectation values and variances of observables.

        .. note::

            The 2nd order method can handle also first order observables, but
            1st order method may be more efficient unless it's really easy to
            experimentally measure arbitrary 2nd order observables.

        Args:
            idx (int): trainable parameter index to differentiate with respect to
            device (.Device, .QubitDevice): a PennyLane device
                that can execute quantum operations and return measurement statistics
            params (list[Any]): the quantum tape operation parameters

        Keyword Args:
            force_order2 (bool): iff True, use the order-2 method even if not necessary

        Returns:
            array[float]: 1-dimensional array of length determined by the tape output
            measurement statistics
        """
        # Temporarily convert all variance measurements on the tape into expectation values
        for i in self.var_idx:
            self._measurements[i].return_type = qml.operation.Expectation

        # Get <A>, the expectation value of the tape with unshifted parameters. This is only
        # calculated once, if `self._evA` is not None.
        if self._evA is None:
            self._evA = np.asarray(self.execute_device(params, device))

        # evaluate the analytic derivative of <A>
        pdA = self.parameter_shift(idx, device, params, **options)

        original = []

        for i in self.var_idx:
            # We need to calculate d<A^2>/dp; to do so, we replace the
            # involutory observables A in the queue with A^2.
            original[:0] = [self._measurements[i]]
            obs = self._measurements[i].obs

            w = obs.wires
            A = obs.matrix

            # CV first order observable
            # get the heisenberg representation
            # This will be a real 1D vector representing the
            # first order observable in the basis [I, x, p]
            A = e._heisenberg_rep(e.parameters)  # pylint: disable=protected-access

            # take the outer product of the heisenberg representation
            # with itself, to get a square symmetric matrix representing
            # the square of the observable
            new_obs = qml.PolyXP(np.outer(A, A), wires=w)
            new_obs.return_type = qml.operation.Expectation

            new_measurement = MeasurementProcess(qml.operation.Expectation, obs=new_obs)
            self._measurements[i] = new_measurement

        # Here, we calculate the analytic derivatives of the <A^2> observables.
        pdA2 = self.parameter_shift(idx, device, params, **options)

        # restore the original observables
        self._measurements = self._original_measurements

        for i in self.var_idx:
            self._measurements[i] = original.pop()

        for i in self.var_idx:
            self._measurements[i].return_type = qml.operation.Variance

        # return d(var(A))/dp = d<A^2>/dp -2 * <A> * d<A>/dp for the variances,
        # d<A>/dp for plain expectations
        return np.where(self.var_mask, pdA2 - 2 * self._evA * pdA, pdA)
