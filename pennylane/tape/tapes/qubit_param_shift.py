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
# pylint: disable=attribute-defined-outside-init,protected-access
import numpy as np

import pennylane as qml
from pennylane.tape.measure import MeasurementProcess
from pennylane.tape.tapes import QuantumTape

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
        # at the unshifted parameter values; the result is stored in the
        # self._evA_result attribute. As a result, we want the tape that computes
        # the evA tape to only be generated *once*. We keep track of its generation
        # via the self._append_evA_tape attribute.
        self._append_evA_tape = True
        self._evA_result = None
        return super().jacobian(device, params, **options)

    def parameter_shift(self, idx, params, **options):
        """Generate the tapes and postprocessing methods required to compute the gradient of a
        parameter using the parameter-shift method.

        Args:
            idx (int): trainable parameter index to differentiate with respect to
            params (list[Any]): the quantum tape operation parameters

        Returns:
            tuple[list[QuantumTape], function]: A tuple containing the list of generated tapes,
            in addition to a post-processing function to be applied to the evaluated
            tapes.
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

        shifted_forward = self.copy(copy_operations=True, tape_cls=QuantumTape)
        shifted_forward.set_parameters(params + shift)

        shifted_backward = self.copy(copy_operations=True, tape_cls=QuantumTape)
        shifted_backward.set_parameters(params - shift)

        tapes = [shifted_forward, shifted_backward]

        def processing_fn(results):
            """Computes the gradient of the parameter at index idx via the
            parameter-shift method.

            Args:
                results (list[real]): evaluated quantum tapes

            Returns:
                array[float]: 1-dimensional array of length determined by the tape output
                measurement statistics
            """
            res_forward = np.array(results[0])
            res_backward = np.array(results[1])
            return (res_forward - res_backward) / (2 * np.sin(s))

        return tapes, processing_fn

    def parameter_shift_var(self, idx, params, **options):
        """Generate the tapes and postprocessing methods required to compute the gradient of a
        parameter and its variance using the parameter-shift method.

        Args:
            idx (int): trainable parameter index to differentiate with respect to
            params (list[Any]): the quantum tape operation parameters

        Returns:
            tuple[list[QuantumTape], function]: A tuple containing the list of generated tapes,
            in addition to a post-processing function to be applied to the evaluated
            tapes.
        """
        tapes = []

        # Get <A>, the expectation value of the tape with unshifted parameters.
        evA_tape = self.copy()
        evA_tape.set_parameters(params)

        # Convert all variance measurements on the tape into expectation values
        for i in self.var_idx:
            obs = evA_tape._measurements[i].obs
            evA_tape._measurements[i] = MeasurementProcess(qml.operation.Expectation, obs=obs)

        # evaluate the analytic derivative of <A>
        pdA_tapes, pdA_fn = evA_tape.parameter_shift(idx, params, **options)
        tapes.extend(pdA_tapes)

        # For involutory observables (A^2 = I) we have d<A^2>/dp = 0.
        # Currently, the only observable we have in PL that may be non-involutory is qml.Hermitian
        involutory = [i for i in self.var_idx if self.observables[i].name != "Hermitian"]

        # If there are non-involutory observables A present, we must compute d<A^2>/dp.
        non_involutory = set(self.var_idx) - set(involutory)

        if non_involutory:
            pdA2_tape = self.copy()

            for i in non_involutory:
                # We need to calculate d<A^2>/dp; to do so, we replace the
                # involutory observables A in the queue with A^2.
                obs = pdA2_tape._measurements[i].obs
                A = obs.matrix

                obs = qml.Hermitian(A @ A, wires=obs.wires, do_queue=False)
                pdA2_tape._measurements[i] = MeasurementProcess(qml.operation.Expectation, obs=obs)

            # Non-involutory observables are present; the partial derivative of <A^2>
            # may be non-zero. Here, we calculate the analytic derivatives of the <A^2>
            # observables.
            pdA2_tapes, pdA2_fn = pdA2_tape.parameter_shift(idx, params, **options)
            tapes.extend(pdA2_tapes)

        # Make sure that the expectation value of the tape with unshifted parameters
        # is only calculated once, if `self._append_evA_tape` is True.
        if self._append_evA_tape:
            tapes.append(evA_tape)

            # Now that the <A> tape has been appended, we want to avoid
            # appending it for subsequent parameters, as the result can simply
            # be re-used.
            self._append_evA_tape = False

        def processing_fn(results):
            """Computes the gradient of the parameter at index ``idx`` via the
            parameter-shift method for a circuit containing a mixture
            of expectation values and variances.

            Args:
                results (list[real]): evaluated quantum tapes

            Returns:
                array[float]: 1-dimensional array of length determined by the tape output
                measurement statistics
            """
            pdA = pdA_fn(results[0:2])
            pdA2 = 0

            if non_involutory:
                pdA2 = pdA2_fn(results[2:4])

                if involutory:
                    pdA2[np.array(involutory)] = 0

            # Check if the expectation value of the tape with unshifted parameters
            # has already been calculated.
            if self._evA_result is None:
                # The expectation value hasn't been previously calculated;
                # it will be the last element of the `results` argument.
                self._evA_result = np.array(results[-1])

            # return d(var(A))/dp = d<A^2>/dp -2 * <A> * d<A>/dp for the variances,
            # d<A>/dp for plain expectations
            return np.where(self.var_mask, pdA2 - 2 * self._evA_result * pdA, pdA)

        return tapes, processing_fn
