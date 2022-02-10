# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
from pennylane.measure import MeasurementProcess
from pennylane.tape import QuantumTape
from pennylane.math import toarray

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

    For a variational evolution :math:`U(\mathbf{p}) \vert 0\rangle` with
    :math:`N` parameters :math:`\mathbf{p}`,

    consider the expectation value of an observable :math:`O`:

    .. math::

        f(\mathbf{p})  = \langle \hat{O} \rangle(\mathbf{p}) = \langle 0 \vert
        U(\mathbf{p})^\dagger \hat{O} U(\mathbf{p}) \vert 0\rangle.


    The gradient of this expectation value can be calculated using :math:`2N` expectation
    values using the parameter-shift rule:

    .. math::

        \frac{\partial f}{\partial \mathbf{p}} = \frac{1}{2\sin s} \left[ f(\mathbf{p} + s) -
        f(\mathbf{p} -s) \right].

    **Gradients of variances**

    We can extend this to the variance,
    :math:`g(\mathbf{p})=\langle \hat{O}^2 \rangle (\mathbf{p}) - [\langle \hat{O}
    \rangle(\mathbf{p})]^2`,
    by noting that:

    .. math::

        \frac{\partial g}{\partial \mathbf{p}}= \frac{\partial}{\partial
        \mathbf{p}} \langle \hat{O}^2 \rangle (\mathbf{p})
        - 2 f(\mathbf{p}) \frac{\partial f}{\partial \mathbf{p}}.

    This results in :math:`4N + 1` evaluations.

    In the case where :math:`O` is involutory (:math:`\hat{O}^2 = I`), the first term in the above
    expression vanishes, and we are simply left with

    .. math::

      \frac{\partial g}{\partial \mathbf{p}} = - 2 f(\mathbf{p})
      \frac{\partial f}{\partial \mathbf{p}},

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

        self.hessian_pd = self.parameter_shift_hessian

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

        Keyword Args:
            shift=pi/2 (float): the size of the shift for two-term parameter-shift gradient computations

        Returns:
            tuple[list[QuantumTape], function]: A tuple containing the list of generated tapes,
            in addition to a post-processing function to be applied to the evaluated
            tapes.
        """
        t_idx = list(self.trainable_params)[idx]
        op = self._par_info[t_idx]["op"]
        p_idx = self._par_info[t_idx]["p_idx"]

        s = options.get("shift", np.pi / 2)
        param_shift = op.get_parameter_shift(p_idx, shift=s)

        shift = np.zeros_like(params)
        coeffs = []
        tapes = []

        for c, a, s in param_shift:
            shift[idx] = s
            shifted_tape = self.copy(copy_operations=True, tape_cls=QuantumTape)
            shifted_tape.set_parameters(a * params + shift)

            coeffs.append(c)
            tapes.append(shifted_tape)

        def processing_fn(results):
            """Computes the gradient of the parameter at index idx via the
            parameter-shift method.

            Args:
                results (list[real]): evaluated quantum tapes

            Returns:
                array[float]: 1-dimensional array of length determined by the tape output
                measurement statistics
            """
            results = np.squeeze(list(map(toarray, results)))

            if results.dtype is np.dtype("O"):
                # The evaluated quantum results are a ragged array.
                # Need to use a list comprehension to compute the linear
                # combination.
                return sum([c * r for c, r in zip(coeffs, results)])

            # The evaluated quantum results are a valid NumPy array,
            # can instead apply the dot product along the first axis.
            dot = lambda x: np.dot(coeffs, x)
            return np.apply_along_axis(dot, 0, results)

        return tapes, processing_fn

    def parameter_shift_var(self, idx, params, **options):
        """Generate the tapes and postprocessing methods required to compute the gradient of a
        parameter and its variance using the parameter-shift method.

        Args:
            idx (int): trainable parameter index to differentiate with respect to
            params (list[Any]): the quantum tape operation parameters

        Keyword Args:
            shift=pi/2 (float): the size of the shift for two-term parameter-shift gradient computations

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
                A = obs.get_matrix()

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

    def parameter_shift_hessian(self, i, j, params, **options):
        """Generate the tapes and postprocessing methods required to compute the
        second derivative with respect to tape parameter :math:`i` and :math:`j`
        using the second-order parameter-shift method.

        Args:
            i (int): trainable parameter index to differentiate with respect to
            j (int): trainable parameter index to differentiate with respect to
            params (list[Any]): the quantum tape operation parameters

        Keyword Args:
            s1=pi/2 (float): the size of the shift for index i in the parameter-shift Hessian computations
            s2=pi/2 (float): the size of the shift for index j in the parameter-shift Hessian computations

        Returns:
            tuple[list[QuantumTape], function]: A tuple containing the list of generated tapes,
            in addition to a post-processing function to be applied to the evaluated
            tapes.
        """
        idxs = (i, j)
        idx_shifts = {i: options.get("s1", np.pi / 2), j: options.get("s2", np.pi / 2)}

        param_shifts = []

        if i == j and idx_shifts[i] == idx_shifts[j]:
            if idx_shifts[i] == np.pi / 2:
                # When i = j and s1 = s2 = pi/2, the Hessian parameter shift rule
                # can be simplified to two device executions
                param_shifts = [(0.5, 1, np.pi), (-0.5, 1, 0)]
            elif idx_shifts[i] == np.pi / 4:
                # When i = j and s1 = s2 = pi/4, the Hessian parameter shift rule
                # can be simplified to three device executions
                # TODO: The first and last parameter shift values below are identical
                # to those used when computing the Jacobian with s=pi/2. We should find
                # a way to re-use those values rather than re-calculating them.
                param_shifts = [(0.5, 1, np.pi / 2), (-1, 1, 0), (0.5, 1, -np.pi / 2)]

        coeffs = []
        tapes = []
        shift = np.eye(len(params))

        if param_shifts:
            # Optimizations can be made to reduce amount of tape executions
            for c, a, s in param_shifts:
                shifted_tape = self.copy(copy_operations=True, tape_cls=QuantumTape)
                shifted_tape.set_parameters(a * params + s * shift[i])

                coeffs.append(c)
                tapes.append(shifted_tape)
        else:
            # No optimizations can be made, generate all 4 tapes
            for idx in idxs:
                t_idx = list(self.trainable_params)[idx]
                op = self._par_info[t_idx]["op"]
                p_idx = self._par_info[t_idx]["p_idx"]
                s = idx_shifts[idx]

                param_shift = op.get_parameter_shift(p_idx, shift=s)
                param_shifts.append(param_shift)

            for c1, a, s1 in param_shifts[0]:
                for c2, _, s2 in param_shifts[1]:
                    c = c1 * c2
                    s = s1 * shift[i] + s2 * shift[j]
                    shifted_tape = self.copy(copy_operations=True, tape_cls=QuantumTape)
                    shifted_tape.set_parameters(a * params + s)

                    coeffs.append(c)
                    tapes.append(shifted_tape)

        def processing_fn(results):
            """Computes the second derivative with respect to tape parameters i
            and j using the second-order parameter-shift method.

            Args:
                results (list[real]): evaluated quantum tapes

            Returns:
                array[float]: 1-dimensional array of length determined by the tape output
                measurement statistics
            """
            results = np.squeeze(results)

            if results.dtype is np.dtype("O"):
                # The evaluated quantum results are a ragged array.
                # Need to use a list comprehension to compute the linear
                # combination.
                return sum([c * r for c, r in zip(coeffs, results)])

            # The evaluated quantum results are a valid NumPy array,
            # can instead apply the dot product along the first axis.
            dot = lambda x: np.dot(coeffs, x)
            return np.apply_along_axis(dot, 0, results)

        return tapes, processing_fn

    @property
    def specs(self):
        """Resource information about a quantum circuit.

        Returns:
            dict[str, Union[defaultdict,int]]: dictionaries that contain tape specifications

        **Example**

        .. code-block:: python3

            with qml.tape.QubitParamShiftTape() as tape:
                qml.Hadamard(wires=0)
                qml.RZ(0.26, wires=1)
                qml.CNOT(wires=[1, 0])
                qml.Rot(1.8, -2.7, 0.2, wires=0)
                qml.Hadamard(wires=1)
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        Asking for the specs produces a dictionary as shown below:

        >>> tape.specs['gate_sizes']
        defaultdict(int, {1: 4, 2: 2})
        >>> tape.specs['gate_types']
        defaultdict(int, {'Hadamard': 2, 'RZ': 1, 'CNOT': 2, 'Rot': 1})

        As ``defaultdict`` objects, any key not present in the dictionary returns 0.

        >>> tape.specs['gate_types']['RX']
        0

        In parameter-shift tapes, the number of device executions necessary for a gradient
        is calulated as well:

        >>> tape.specs['num_parameter_shift_executions]
        9

        """

        info = super().specs

        if any(m.return_type is qml.operation.State for m in self.measurements):
            return info

        if len(self._par_info) > 0:
            if "grad_method" not in self._par_info[0]:
                self._update_gradient_info()

        # Initialize with the forward pass execution
        num_executions = 1
        # Loop over all variables
        for _, grad_info in self._par_info.items():

            # if this variable uses parameter-shift
            if grad_info["grad_method"] == "A":

                # get_parameter_shift returns operation specific derivative formula
                # for ops with 4-term gradients, the array will contain 4 pieces of data.
                num_executions += len(grad_info["op"].get_parameter_shift(grad_info["p_idx"]))

        info["num_parameter_shift_executions"] = num_executions

        return info
