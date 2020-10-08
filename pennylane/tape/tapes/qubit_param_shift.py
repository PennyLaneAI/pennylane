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
Qubit parameter shift quantum tape.

Provides analytic differentiation for all one-parameter gates where the generator
only has two unique eigenvalues; this includes one-parameter single-qubit gates,
and any gate with an involutory generator.
"""
# pylint: disable=attribute-defined-outside-init
import numpy as np

import pennylane as qml
from pennylane.tape.measure import MeasurementProcess

from .jacobian_tape import JacobianTape


class QubitParamShiftTape(JacobianTape):
    r"""Quantum tape for qubit parameter-shift analytic differentiation method.

    This class extends the :class:`~.jacobian` method of the quantum tape
    to support analytic gradients of qubit operations using the parameter-shift rule.
    This gradient method returns *exact* gradients, and can be computed directly
    on quantum hardware. Simply pass ``method=analytic`` when computing the Jacobian:

    >>> tape.jacobian(dev, method="analytic")

    For more details on the quantum tape, please see :class:`~.JacobianTape`.

    **Gradients of expectation values**

    For a variational evolution :math:`U(mathbf{p})\vert 0\rangle` with :math:`N` parameters :math:`mathbf{p}`,

    consider the expectation value of an observable :math:`O`:

    .. math::

        f(mathbf{p})  = \langle hat{O} \rangle(mathbf{p}) = \langle 0 \vert
        U(mathbf{p})^\dagger hat{O} U(mathbf{p}) \vert 0\rangle.


    The gradient of this expectation value can be calculated using :math:`2N` expectation
    values using the parameter-shift rule:

    .. math::

        \frac{\partial f}{\partial mathbf{p}} = \frac{1}{2\sin s} \left[ f(mathbf{p} + s) -
        f(mathbf{p} -s) \right].

    **Gradients of variances**

    We can extend this to the variance,
    :math:`g(mathbf{p})=\langle hat{O}^2 \rangle (mathbf{p}) - [\langle hat{O} \rangle(mathbf{p})]^2`,
    by noting that:

    .. math::

        \frac{\partial g}{\partial mathbf{p}}= \frac{\partial}{\partial mathbf{p}} \langle hat{O}^2 \rangle (mathbf{p})
        - 2 f(mathbf{p}) \frac{\partial f}{\partial mathbf{p}}.

    This results in :math:`4N + 1` evaluations.

    In the case where :math:`O` is involutory (:math:`hat{O}^2 = I`), the first term in the above
    expression vanishes, and we are simply left with

    .. math:: \frac{\partial g}{\partial mathbf{p}} = - 2 f(mathbf{p}) \frac{\partial f}{\partial mathbf{p}},

    allowing us to compute the gradient using :math:`2N + 1` evaluations.
    """

    def _update_circuit_info(self):
        super()._update_circuit_info()

        # set parameter_shift as the analytic_pd method
        self.analytic_pd = self.parameter_shift

        # check if the quantum tape contains any variance measurements
        self.var_mask = [m.return_type is qml.operation.Variance for m in self.measurements]

        # Make a copy of the original measurements; we will be mutating them
        # during the parameter shift method.
        self._original_measurements = self._measurements.copy()

        if any(self.var_mask):
            # The tape contains variances.
            # Set parameter_shift_var as the analytic_pd method
            self.analytic_pd = self.parameter_shift_var

            # Finally, store the locations of any variance measurements in the
            # measurement queue.
            self.var_idx = np.where(self.var_mask)[0]

    def _grad_method(self, idx, use_graph=True, default_method="A"):
        op = self._par_info[idx]["op"]

        if op.grad_method == "F":
            return "F"

        return super()._grad_method(idx, use_graph=use_graph, default_method=default_method)

    def jacobian(self, device, params=None, **options):
        # The parameter_shift_var method needs to evaluate the circuit
        # at the unshifted parameter values; these are stored in the
        # self._evA attribute. Here, we set the value of the attribute to None
        # before each Jacobian call, so that the expectation value is calculated only once.
        self._evA = None
        return super().jacobian(device, params, **options)

    def parameter_shift(self, idx, device, params, **options):
        r"""Partial derivative using the parameter-shift rule of a tape consisting of measurement
        statistics that can be represented as expectation values of observables.

        This includes tapes that output probabilities, since the probability of measuring a
        basis state :math:`|i\rangle` can be written in the form of an expectation value:

        .. math::

            \mathbb{P}_{|i\rangle} = |\langle i | U(\mathbf(p)) | 0 \rangle|^2
                = \langle 0 | U(\mathbf(p))^\dagger | i \rangle\langle i | U(\mathbf(p)) | 0 \rangle
                = \mathbb{E}\left( | i \rangle\langle i | \right)

        Args:
            idx (int): trainable parameter index to differentiate with respect to
            device (.Device, .QubitDevice): a PennyLane device
                that can execute quantum operations and return measurement statistics
            params (list[Any]): the quantum tape operation parameters

        Keyword Args:
            shift (float): the parameter shift value

        Returns:
            array[float]: 1-dimensional array of length determined by the tape output
            measurement statistics
        """
        t_idx = list(self.trainable_params)[idx]
        op = self._par_info[t_idx]["op"]
        p_idx = self._par_info[t_idx]["p_idx"]

        s = (
            np.pi / 2
            if op.grad_recipe is None or op.grad_recipe[p_idx] is None
            else op.grad_recipe[p_idx]
        )
        s = options.get("shift", s)

        shift = np.zeros_like(params)
        shift[idx] = s

        shift_forward = np.array(self.execute_device(params + shift, device))
        shift_backward = np.array(self.execute_device(params - shift, device))

        return (shift_forward - shift_backward) / (2 * np.sin(s))

    def parameter_shift_var(self, idx, device, params, **options):
        r"""Partial derivative using the parameter-shift rule of a tape consisting of a mixture
        of expectation values and variances of observables.

        Args:
            idx (int): trainable parameter index to differentiate with respect to
            device (.Device, .QubitDevice): a PennyLane device
                that can execute quantum operations and return measurement statistics
            params (list[Any]): the quantum tape operation parameters

        Returns:
            array[float]: 1-dimensional array of length determined by the tape output
            measurement statistics
        """
        # Temporarily convert all variance measurements on the tape into expectation values
        for i in self.var_idx:
            obs = self._measurements[i].obs
            self._measurements[i] = MeasurementProcess(qml.operation.Expectation, obs=obs)

        # Get <A>, the expectation value of the tape with unshifted parameters. This is only
        # calculated once, if `self._evA` is not None.
        if self._evA is None:
            self._evA = np.asarray(self.execute_device(params, device))

        # evaluate the analytic derivative of <A>
        pdA = self.parameter_shift(idx, device, params, **options)

        # For involutory observables (A^2 = I) we have d<A^2>/dp = 0.
        # Currently, the only observable we have in PL that may be non-involutory is qml.Hermitian
        involutory = [i for i in self.var_idx if self.observables[i].name != "Hermitian"]

        # If there are non-involutory observables A present, we must compute d<A^2>/dp.
        non_involutory = set(self.var_idx) - set(involutory)

        for i in non_involutory:
            # We need to calculate d<A^2>/dp; to do so, we replace the
            # involutory observables A in the queue with A^2.
            obs = self._measurements[i].obs
            A = obs.matrix

            obs = qml.Hermitian(A @ A, wires=obs.wires, do_queue=False)
            self._measurements[i] = MeasurementProcess(qml.operation.Expectation, obs=obs)

        pdA2 = 0

        if non_involutory:
            # Non-involutory observables are present; the partial derivative of <A^2>
            # may be non-zero. Here, we calculate the analytic derivatives of the <A^2>
            # observables.
            pdA2 = self.parameter_shift(idx, device, params, **options)

            if involutory:
                # We need to explicitly specify that the gradient of
                # the involutory observables is 0, since we saved on processing
                # by not replacing these observables with their square.
                pdA2[np.array(involutory)] = 0

        # restore the original observables
        self._measurements = self._original_measurements.copy()

        # return d(var(A))/dp = d<A^2>/dp -2 * <A> * d<A>/dp for the variances,
        # d<A>/dp for plain expectations
        return np.where(self.var_mask, pdA2 - 2 * self._evA * pdA, pdA)
