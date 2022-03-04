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
CV parameter shift quantum tape.

Provides analytic differentiation for variational circuits with parametrized Gaussian CV gates
and first- and second-order observables.
"""
# pylint: disable=attribute-defined-outside-init,too-many-branches,protected-access
import itertools
import warnings

import numpy as np

import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.tape import QuantumTape

from .qubit_param_shift import QubitParamShiftTape, _get_operation_recipe


class CVParamShiftTape(QubitParamShiftTape):
    r"""Quantum tape for CV parameter-shift analytic differentiation method.

    This class extends the :class:`~.jacobian` method of the quantum tape
    to support analytic gradients of Gaussian CV operations using the parameter-shift rule.
    This gradient method returns *exact* gradients, and can be computed directly
    on quantum hardware. Simply pass ``method=analytic`` when computing the Jacobian:

    >>> tape.jacobian(dev, method="analytic")

    For more details on the quantum tape, please see :class:`~.JacobianTape`.

    This tape supports analytic gradients of photonic circuits that satisfy
    the following constraints with regards to measurements:

    * Expectation values are restricted to observables that are first- and
      second-order in :math:`\hat{x}` :math:`\hat{p}` only.
      This includes :class:`~.X`, :class:`~.P`, :class:`~.QuadOperator`,
      :class:`~.PolyXP`, and :class:`~.NumberOperator`.

      For second-order observables, the device **must support** :class:`~.PolyXP`.

    * Variances are restricted to observables that are first-order
      in :math:`\hat{x}` :math:`\hat{p}` only. This includes :class:`~.X`, :class:`~.P`,
      :class:`~.QuadOperator`, and *some* parameter values of :class:`~.PolyXP`.

      The device **must support** :class:`~.PolyXP`.

    Fock state probabilities (tapes that return :func:`~pennylane.probs` or
    expectation values of :class:`~.FockStateProjector`) are not supported.

    In addition, the tape operations must fulfill the following requirements:

    * Only Gaussian operations are differentiable.

    * Non-differentiable Fock states and Fock operations may *precede* all differentiable Gaussian,
      operations. For example, the following is permissible:

      .. code-block:: python

          with CVParamShiftTape() as tape:
              # Non-differentiable Fock operations
              qml.FockState(2, wires=0)
              qml.Kerr(0.654, wires=1)

              # differentiable Gaussian operations
              qml.Displacement(0.6, 0.5, wires=0)
              qml.Beamsplitter(0.5, 0.1, wires=[0, 1])
              qml.expval(qml.NumberOperator(0))

          tape.trainable_params = {2, 3, 4}

    * If a Fock operation succeeds a Gaussian operation, the Fock operation must
      not contribute to any measurements. For example, the following is allowed:

      .. code-block:: python

          with CVParamShiftTape() as tape:
              qml.Displacement(0.6, 0.5, wires=0)
              qml.Beamsplitter(0.5, 0.1, wires=[0, 1])
              qml.Kerr(0.654, wires=1)  # there is no measurement on wire 1
              qml.expval(qml.NumberOperator(0))

          tape.trainable_params = {0, 1, 2}

    If any of the above constraints are not followed, the tape cannot be differentiated
    via the CV parameter-shift rule. Please use numerical differentiation instead:

    >>> tape.jacobian(dev, method="numeric")
    """

    def _grad_method(self, idx, use_graph=True, default_method="A"):
        op = self._par_info[idx]["op"]

        if op.grad_method in (None, "F"):
            return op.grad_method

        if op.grad_method != "A":
            raise ValueError(f"Operation {op} has unknown gradient method {op.grad_method}")

        if not use_graph:
            raise ValueError(
                "The CV parameter-shift rule must always use the "
                "graph to determine operation gradient methods"
            )

        # Operation supports the CV parameter-shift rule.
        # Create an empty list to store the 'best' partial derivative method
        # for each observable
        best = []

        for m in self.measurements:

            if (m.return_type is qml.operation.Probability) or (m.obs.ev_order not in (1, 2)):
                # Higher-order observables (including probability) only support finite differences.
                best.append("F")
                continue

            # get the set of operations betweens the operation and the observable
            ops_between = self.graph.nodes_between(op, m.obs)

            if not ops_between:
                # if there is no path between the operation and the observable,
                # the operator has a zero gradient.
                best.append("0")
                continue

            # For parameter-shift compatible CV gates, we need to check both the
            # intervening gates, and the type of the observable.
            best_method = "A"

            if any(not k.supports_heisenberg for k in ops_between):
                # non-Gaussian operators present in-between the operation
                # and the observable. Must fallback to numeric differentiation.
                best_method = "F"

            elif m.obs.ev_order == 2:

                if m.return_type is qml.operation.Expectation:
                    # If the observable is second order, we must use the second order
                    # CV parameter shift rule
                    best_method = "A2"

                elif m.return_type is qml.operation.Variance:
                    # we only support analytic variance gradients for
                    # first order observables
                    best_method = "F"

            best.append(best_method)

        if all(k == "0" for k in best):
            # if the operation is independent of *all* observables
            # in the circuit, the gradient will be 0
            return "0"

        if "F" in best:
            # one non-analytic observable path makes the whole operation
            # gradient method fallback to finite-difference
            return "F"

        if "A2" in best:
            # one second order observable makes the whole operation gradient
            # require the second order parameter-shift rule
            return "A2"

        return "A"

    @staticmethod
    def _transform_observable(obs, Z, device_wires):
        """Apply a Gaussian linear transformation to each index of an observable.

        Args:
            obs (.Observable): observable to transform
            Z (array[float]): Heisenberg picture representation of the linear transformation
            device_wires (.Wires): wires on the device the transformed observable is to be
                measured on

        Returns:
            .Observable: the transformed observable
        """
        # Get the Heisenberg representation of the observable
        # in the position/momentum basis. The returned matrix/vector
        # will have been expanded to act on the entire device.
        if obs.ev_order > 2:
            raise NotImplementedError("Transforming observables of order > 2 not implemented.")

        A = obs.heisenberg_obs(device_wires)

        if A.ndim != obs.ev_order:
            raise ValueError(
                "Mismatch between the polynomial order of observable and its Heisenberg representation"
            )

        # transform the observable by the linear transformation Z
        A = A @ Z

        if A.ndim == 2:
            A = A + A.T

        # TODO: if the A matrix corresponds to a known observable in PennyLane,
        # for example qml.X, qml.P, qml.NumberOperator, we should return that
        # instead. This will allow for greater device compatibility.
        return qml.PolyXP(A, wires=device_wires, do_queue=False)

    def parameter_shift_first_order(
        self, idx, params, **options
    ):  # pylint: disable=unused-argument
        """Generate the tapes and postprocessing methods required to compute the gradient of a parameter using the
        first order CV parameter-shift method.

        Args:
            idx (int): trainable parameter index to differentiate with respect to
            params (list[Any]): the quantum tape operation parameters

        Returns:
            tuple[list[QuantumTape], function]: A tuple containing the list of generated tapes,
            in addition to a post-processing function to be applied to the evaluated
            tapes.
        """

        op, p_idx = self.get_operation(idx)
        param_shift = _get_operation_recipe(op, p_idx, None)
        shift = np.zeros_like(params)

        coeffs = []
        tapes = []
        for c, _a, s in zip(*param_shift):

            shift[idx] = s

            # shifted parameter values
            shifted_tape = self.copy(copy_operations=True, tape_cls=QuantumTape)
            shifted_tape.set_parameters(params + shift)
            coeffs.append(c)
            tapes.append(shifted_tape)

        def processing_fn(results):
            """Computes the gradient of the parameter at index idx via the
            first order CV parameter-shift method.

            Args:
                results (list[real]): evaluated quantum tapes

            Returns:
                array[float]: 1-dimensional array of length determined by the tape output
                measurement statistics
            """
            return np.dot(coeffs, results)

        return tapes, processing_fn

    def parameter_shift_second_order(self, idx, params, **options):
        """Generate the tapes and postprocessing methods required to compute the gradient of a
        parameter using the second order CV parameter-shift method.

        Args:
            idx (int): trainable parameter index to differentiate with respect to
            params (list[Any]): the quantum tape operation parameters

        Keyword Args:
            dev_wires (.Wires): wires on the device the parameter-shift method is computed on

        Returns:
            tuple[list[QuantumTape], function]: A tuple containing the list of generated tapes,
            in addition to a post-processing function to be applied to the evaluated
            tapes.
        """

        op, p_idx = self.get_operation(idx)
        param_shift = _get_operation_recipe(op, p_idx, None)

        dev_wires = options["dev_wires"]

        if len(param_shift[0]) != 2:
            # The 2nd order CV parameter-shift rule only accepts two-term shifts
            raise NotImplementedError(
                "Taking the analytic gradient for order-2 operators is "
                "unsupported for {op} which contains a parameter with a "
                "gradient recipe of more than two terms."
            )

        c1, c2 = param_shift[0]
        a1, a2 = param_shift[1]
        s1, s2 = param_shift[2]

        shift = np.zeros_like(params)
        shift[idx] = s1

        # evaluate transformed observables at the original parameter point
        # first build the Heisenberg picture transformation matrix Z
        self.set_parameters(a1 * params + shift)
        Z2 = op.heisenberg_tr(dev_wires)

        shift[idx] = s2
        self.set_parameters(a2 * params + shift)
        Z1 = op.heisenberg_tr(dev_wires)

        # derivative of the operation
        Z = Z2 * c1 + Z1 * c2

        self.set_parameters(params)
        Z0 = op.heisenberg_tr(dev_wires, inverse=True)
        Z = Z @ Z0

        # conjugate Z with all the descendant operations
        B = np.eye(1 + 2 * len(dev_wires))
        B_inv = B.copy()

        succ = self.graph.descendants_in_order((op,))
        operation_descendents = itertools.filterfalse(qml.circuit_graph._is_observable, succ)
        observable_descendents = filter(qml.circuit_graph._is_observable, succ)

        for BB in operation_descendents:
            if not BB.supports_heisenberg:
                # if the descendant gate is non-Gaussian in parameter-shift differentiation
                # mode, then there must be no observable following it.
                continue

            B = BB.heisenberg_tr(dev_wires) @ B
            B_inv = B_inv @ BB.heisenberg_tr(dev_wires, inverse=True)

        Z = B @ Z @ B_inv  # conjugation

        tape = self.copy(copy_operations=True, tape_cls=QuantumTape)

        # change the observable
        # TODO: if the transformation produces only a constant term,
        # `_transform_observable` has only a single non-zero element in the
        # 0th position, then there is no need to execute the device---the constant term
        # represents the gradient.

        # transform the descendant observables into their derivatives using Z
        transformed_obs_idx = []
        for obs in observable_descendents:
            # get the index of the descendent observable
            idx = self.observables.index(obs)
            transformed_obs_idx.append(idx)
            tape._measurements[idx] = MeasurementProcess(
                qml.operation.Expectation, self._transform_observable(obs, Z, dev_wires)
            )

        tapes = [tape]

        def processing_fn(results):
            """Computes the gradient of the parameter at index idx via the
            second order CV parameter-shift method.

            Args:
                results (list[real]): evaluated quantum tapes

            Returns:
                array[float]: 1-dimensional array of length determined by the tape output
                measurement statistics
            """
            res = np.array(results)[0]
            grad = np.zeros_like(res)
            grad[transformed_obs_idx] = res[transformed_obs_idx]
            return grad

        return tapes, processing_fn

    def parameter_shift(self, idx, params, **options):
        r"""Partial derivative using the first- or second-order CV parameter-shift rule of a
        tape consisting of *only* expectation values of observables.

        .. note::

            The 2nd order method can handle also first order observables, but
            1st order method may be more efficient unless it's really easy to
            experimentally measure arbitrary 2nd order observables.

        .. warning::

            The 2nd order method can only be executed on devices that support the
            :class:`~.PolyXP` observable.

        Args:
            idx (int): trainable parameter index to differentiate with respect to
            params (list[Any]): the quantum tape operation parameters

        Keyword Args:
            force_order2 (bool): iff True, use the order-2 method even if not necessary
            device (.Device): A PennyLane device that can execute quantum operations and return
                measurement statistics. This keyword argument is required, as the device labels
                may be needed to generate the quantum tapes for computing the gradient.

        Returns:
            tuple[list[QuantumTape], function]: A tuple containing the list of generated tapes,
            in addition to a post-processing function to be applied to the evaluated
            tapes.
        """
        device = options["device"]
        options["dev_wires"] = device.wires

        t_idx = list(self.trainable_params)[idx]
        grad_method = self._par_info[t_idx]["grad_method"]

        if options.get("force_order2", False) or grad_method == "A2":

            if "PolyXP" not in device.observables:
                # If the device does not support PolyXP, must fallback
                # to numeric differentiation.
                warnings.warn(
                    f"The device {device.short_name} does not support "
                    "the PolyXP observable. The analytic parameter-shift cannot be used for "
                    "second-order observables; falling back to finite-differences.",
                    UserWarning,
                )
                return self.numeric_pd(idx, params, **options)

            return self.parameter_shift_second_order(idx, params=params, **options)

        return self.parameter_shift_first_order(idx, params=params)

    def parameter_shift_var(self, idx, params, **options):
        r"""Partial derivative using the first-order or second-order parameter-shift rule of a tape
        consisting of a mixture of expectation values and variances of observables.

        Expectation values may be of first- or second-order observables,
        but variances can only be taken of first-order variables.

        .. warning::

            This method can only be executed on devices that support the
            :class:`~.PolyXP` observable.

        Args:
            idx (int): trainable parameter index to differentiate with respect to
            params (list[Any]): the quantum tape operation parameters

        Keyword Args:
            force_order2 (bool): iff True, use the order-2 method even if not necessary
            device (.Device): A PennyLane device that can execute quantum operations and return
                measurement statistics. This keyword argument is required, as the device labels
                may be needed to generate the quantum tapes for computing the gradient.

        Returns:
            array[float]: 1-dimensional array of length determined by the tape output
            measurement statistics
        """
        # pylint: disable=protected-access
        device = options["device"]

        if "PolyXP" not in device.observables:
            # If the device does not support PolyXP, must fallback
            # to numeric differentiation.
            warnings.warn(
                f"The device {device.short_name} does not support "
                "the PolyXP observable. The analytic parameter-shift cannot be used for "
                "second-order observables; falling back to finite-differences.",
                UserWarning,
            )

            return self.numeric_pd(idx, params, **options)

        tapes = []

        # Get <A>, the expectation value of the tape with unshifted parameters.
        evA_tape = self.copy()
        evA_tape.set_parameters(params)

        # Temporarily convert all variance measurements on the tape into expectation values
        for i in self.var_idx:
            obs = evA_tape._measurements[i].obs
            evA_tape._measurements[i] = MeasurementProcess(qml.operation.Expectation, obs=obs)

        # evaluate the analytic derivative of <A>
        pdA_tapes, pdA_fn = evA_tape.parameter_shift_first_order(idx, params, **options)
        tapes.extend(pdA_tapes)

        pdA2_tape = self.copy()

        for i in self.var_idx:
            # We need to calculate d<A^2>/dp; to do so, we replace the
            # observables A in the queue with A^2.
            obs = pdA2_tape._measurements[i].obs

            # CV first order observable
            # get the heisenberg representation
            # This will be a real 1D vector representing the
            # first order observable in the basis [I, x, p]
            A = obs._heisenberg_rep(obs.parameters)  # pylint: disable=protected-access

            # take the outer product of the heisenberg representation
            # with itself, to get a square symmetric matrix representing
            # the square of the observable
            obs = qml.PolyXP(np.outer(A, A), wires=obs.wires, do_queue=False)
            pdA2_tape._measurements[i] = MeasurementProcess(qml.operation.Expectation, obs=obs)

        # Here, we calculate the analytic derivatives of the <A^2> observables.
        pdA2_tapes, pdA2_fn = pdA2_tape.parameter_shift_second_order(idx, params, **options)
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
            second order CV parameter-shift method for a circuit containing a mixture
            of expectation values and variances.

            Args:
                results (list[real]): evaluated quantum tapes

            Returns:
                array[float]: 1-dimensional array of length determined by the tape output
                measurement statistics
            """
            pdA = pdA_fn(results[0:2])
            pdA2 = pdA2_fn(results[2:4])

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
