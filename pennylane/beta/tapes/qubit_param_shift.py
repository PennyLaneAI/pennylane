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
only has two unique eigenvalues; this includes one-parameter single-qubit gates.
"""
# pylint: disable=attribute-defined-outside-init
import numpy as np

import pennylane as qml
from pennylane.beta.queuing import MeasurementProcess

from .tape import QuantumTape


class QubitParamShiftTape(QuantumTape):
    """Quantum tape for qubit parameter-shift analytic differentiation method.

    For more details on the quantum tape, please see :class:`~.QuantumTape`.
    """

    def _update_circuit_info(self):
        super()._update_circuit_info()

        self.analytic_pd = self._parameter_shift

        self.var_idx = None
        self.var_mask = [m.return_type is qml.operation.Variance for m in self.measurements]
        self._original_measurements = self._measurements.copy()

        if any(self.var_mask):
            self.analytic_pd = self._parameter_shift_var
            self.var_idx = np.where(self.var_mask)[0]

        # # expand out the tape, if any operations are not supported by the parameter shift rule
        # if {"CRX", "CRZ", "CRY", "CRot"} & {op.name for op in self.operations}:
        #     stop = set(device.operations) - {"CRX", "CRZ", "CRY", "CRot"}
        #     self = self.expand(depth=2, stop_at=stop)

    def _grad_method(self, idx, use_graph=True, default_method="F"):
        op = self._par_info[idx]["op"]

        if op.grad_method == "F":
            return "F"

        return super()._grad_method(idx, use_graph=use_graph, default_method="A")

    def jacobian(self, device, params=None, **options):
        self._evA = None
        ret = super().jacobian(device, params, **options)
        self._evA = None

        return ret

    def _parameter_shift(self, idx, device, params, **options):
        """Partial derivative of expectation values of an observable using the parameter-shift method.

        Args:
            idx (int): trainable parameter index to differentiate with respect to
            device (~.Device, ~.QubitDevice): a PennyLane device
                that can execute quantum operations and return measurement statistics
            params (list[Any]): the quantum tape operation parameters

        Keyword Args:
            shift (float): the parameter shift value

        Returns:
            array[float]: 1-dimensional array of length determined by the tape output
                measurement statistics
        """
        op = self._par_info[idx[0]]["op"]
        p_idx = self._par_info[idx[0]]["p_idx"]

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

    def _parameter_shift_var(self, idx, device, params, **options):
        """Partial derivative of the variance of an observable using the parameter-shift method.

        Args:
            idx (int): trainable parameter index to differentiate with respect to
            device (~.Device, ~.QubitDevice): a PennyLane device
                that can execute quantum operations and return measurement statistics
            params (list[Any]): the quantum tape operation parameters

        Returns:
            array[float]: 1-dimensional array of length determined by the tape output
                measurement statistics
        """
        for i in self.var_idx:
            self._measurements[i].return_type = qml.operation.Expectation

        # get <A>
        if self._evA is None:
            self._evA = np.asarray(self.execute_device(params, device))

        # evaluate the analytic derivative of <A>
        pdA = self._parameter_shift(idx, device, params, **options)

        # For involutory observables (A^2 = I) and thus we have d<A^2>/dp = 0
        # Currently, the only non-involutory observable we have in PL is Hermitian
        involutory = [i for i in self.var_idx if self.observables[i].name != "Hermitian"]

        # non involutory observables we must compute d<A^2>/dp
        non_involutory = set(self.var_idx) - set(involutory)
        original = []

        for i in non_involutory:
            # need to calculate d<A^2>/dp; replace the involutory observables
            # in the queue with <A^2>.
            original[:0] = [self._measurements[i]]
            obs = self._measurements[i].obs

            w = obs.wires
            A = obs.matrix

            new_obs = qml.Hermitian(A @ A, wires=w)
            new_obs.return_type = qml.operation.Expectation

            new_measurement = MeasurementProcess(qml.operation.Expectation, obs=new_obs)
            self._measurements[i] = new_measurement

        pdA2 = 0

        if non_involutory:
            # calculate the analytic derivatives of the <A^2> observables
            pdA2 = self._parameter_shift(idx, device, params, **options)

            if involutory:
                pdA2[np.array(involutory)] = 0

        # restore the original observables
        self._measurements = self._original_measurements

        for i in non_involutory:
            self._measurements[i] = original.pop()

        for i in self.var_idx:
            self._measurements[i].return_type = qml.operation.Variance

        # return d(var(A))/dp = d<A^2>/dp -2 * <A> * d<A>/dp for the variances,
        # d<A>/dp for plain expectations
        return np.where(self.var_mask, pdA2 - 2 * self._evA * pdA, pdA)
