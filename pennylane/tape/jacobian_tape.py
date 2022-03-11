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
This module contains the ``JacobianTape``, which adds differentiation methods
to the ``QuantumTape`` class.
"""
# pylint: disable=too-many-branches

import itertools
import warnings

import numpy as np

import pennylane as qml
from pennylane.operation import State
from pennylane.tape import QuantumTape

# CV ops still need to support state preparation operations prior to any
# other operation for PennyLane-SF tests to pass.
STATE_PREP_OPS = (
    qml.BasisState,
    qml.QubitStateVector,
    # qml.CatState,
    # qml.CoherentState,
    # qml.FockDensityMatrix,
    # qml.DisplacedSqueezedState,
    # qml.FockState,
    # qml.FockStateVector,
    # qml.ThermalState,
    # qml.GaussianState,
)


# pylint: disable=too-many-public-methods
class JacobianTape(QuantumTape):
    """A quantum tape recorder, that records, validates, executes,
    and differentiates variational quantum programs.

    .. note::

        See :mod:`pennylane.tape` for more details.

    .. warning::

        The ``JacobianTape`` as well as the ``JacobianTape.jacobian()`` method is deprecated.
        Please use a standard :class:`~.QuantumTape`, and apply gradient transforms using
        the :mod:`.gradients` module to compute Jacobians.

    Args:
        name (str): a name given to the quantum tape
        do_queue (bool): Whether to queue this tape in a parent tape context.

    **Example**

    In addition to the functionality of the ``QuantumTape`` class,
    the ``JacobianTape`` has the ability to compute Jacobians of a quantum
    circuit. Consider the following

    .. code-block:: python

        import pennylane.tape

        with qml.tape.JacobianTape() as tape:
            qml.RX(0.432, wires=0)
            qml.RY(0.543, wires=0)
            qml.CNOT(wires=[0, 'a'])
            qml.RX(0.133, wires='a')
            qml.expval(qml.PauliZ(wires=[0]))

    The Jacobian is computed using finite difference:

    >>> dev = qml.device('default.qubit', wires=[0, 'a'])
    >>> tape.jacobian(dev)
    [[-0.35846484 -0.46923704  0.        ]]
    >>> tape.jacobian(dev, params=[0.1, 0.1, 0.1])
    [[-0.09933471 -0.09933471  0.        ]]

    Trainable parameters are taken into account when calculating the Jacobian,
    avoiding unnecessary calculations:

    >>> tape.trainable_params = {0} # set only the first parameter as free
    >>> tape.set_parameters([0.56])
    >>> tape.jacobian(dev)
    [[-0.45478169]]
    """

    def __init__(self, name=None, do_queue=True):
        super().__init__(name=name, do_queue=do_queue)
        self.jacobian_options = {}
        self.hessian_options = {}

    def copy(self, copy_operations=False, tape_cls=None):
        copied_tape = super().copy(copy_operations=copy_operations, tape_cls=tape_cls)
        copied_tape.jacobian_options = self.jacobian_options
        return copied_tape

    def _grad_method(self, idx, use_graph=True, default_method="F"):
        """Determine the correct partial derivative computation method for each gate parameter.

        Parameter gradient methods include:

        * ``None``: the parameter does not support differentiation.

        * ``"0"``: the variational circuit output does not depend on this
          parameter (the partial derivative is zero).

        * ``"F"``: the parameter has a non-zero derivative that should be computed
          using finite-differences.

        * ``"A"``: the parameter has a non-zero derivative that should be computed
          using an analytic method.

        .. note::

            The base ``JacobianTape`` class only supports numerical differentiation, so
            this method will always return either ``"F"`` or ``None``. If an inheriting
            tape supports analytic differentiation for certain operations, make sure
            that this method is overwritten appropriately to return ``"A"`` where
            required.

        Args:
            idx (int): parameter index
            use_graph: whether to use a directed-acyclic graph to determine
                if the parameter has a gradient of 0
            default_method (str): the default differentiation value to return
                for parameters that where the grad method is not ``"0"`` or ``None``

        Returns:
            str: partial derivative method to be used
        """
        op = self._par_info[idx]["op"]

        if op.grad_method is None:
            return None

        if (self._graph is not None) or use_graph:
            # an empty list to store the 'best' partial derivative method
            # for each observable
            best = []

            # loop over all observables
            for ob in self.observables:
                # check if op is an ancestor of ob
                has_path = self.graph.has_path(op, ob)

                # Use finite differences if there is a path, else the gradient is zero
                best.append(default_method if has_path else "0")

            if all(k == "0" for k in best):
                return "0"

        return default_method

    def _update_gradient_info(self):
        """Update the parameter information dictionary with gradient information
        of each parameter"""

        for i, info in self._par_info.items():

            if i not in self.trainable_params:
                info["grad_method"] = None
            else:
                info["grad_method"] = self._grad_method(i, use_graph=True)

    def _grad_method_validation(self, method):
        """Validates if the gradient method requested is supported by the trainable
        parameters, and returns the allowed parameter gradient methods.

        This method will generate parameter gradient information if it has not already
        been generated, and then proceed to validate the gradient method. In particular:

        * An exception will be raised if there exist non-differentiable trainable
          parameters on the tape.

        * An exception will be raised if the Jacobian method is ``"analytic"`` but there
          exist some trainable parameters on the tape that only support numeric differentiation.

        If all validations pass, this method will return a tuple containing the allowed parameter
        gradient methods for each trainable parameter.

        Args:
            method (str): the overall Jacobian differentiation method

        Returns:
            tuple[str, None]: the allowed parameter gradient methods for each trainable parameter
        """

        if "grad_method" not in self._par_info[0]:
            self._update_gradient_info()

        diff_methods = {
            idx: info["grad_method"]
            for idx, info in self._par_info.items()
            if idx in self.trainable_params
        }

        # check and raise an error if any parameters are non-differentiable
        nondiff_params = {idx for idx, g in diff_methods.items() if g is None}

        if nondiff_params:
            raise ValueError(f"Cannot differentiate with respect to parameter(s) {nondiff_params}")

        numeric_params = {idx for idx, g in diff_methods.items() if g == "F"}

        # If explicitly using analytic mode, ensure that all parameters
        # support analytic differentiation.
        if method == "analytic" and numeric_params:
            raise ValueError(
                f"The analytic gradient method cannot be used with the argument(s) {numeric_params}."
            )

        return tuple(diff_methods.values())

    @staticmethod
    def _has_trainable_params(params, diff_methods):
        """Determines if there are any trainable parameters.

        Args:
            params (array[float]): one-dimensional array of parameters
            diff_methods (Sequence[str]): The corresponding differentiation method for each parameter.
                A differentiation method of ``"0"`` corresponds to a constant parameter.
        """
        return params.size and not all(g == "0" for g in diff_methods)

    @staticmethod
    def _flatten_processing_result(g):
        """Flattens the output from processing_fn in parameter shift methods."""
        if hasattr(g, "dtype") and g.dtype is np.dtype("object"):
            # - Object arrays cannot be flattened; must hstack them.
            # - We also check that g has attribute dtype first to allow for
            #   Observables that return arbitrary objects.
            g = np.hstack(g)

        if hasattr(g, "flatten"):
            # flatten only if g supports flattening to allow for
            # objects other than numpy ndarrays
            return g.flatten()
        return g

    def numeric_pd(self, idx, params=None, **options):
        """Generate the tapes and postprocessing methods required to compute the gradient of a parameter using the
        finite-difference method.

        Args:
            idx (int): trainable parameter index to differentiate with respect to
            params (list[Any]): The quantum tape operation parameters. If not provided,
               the current tape parameter values are used (via :meth:`~.get_parameters`).

        Keyword Args:
            h=1e-7 (float): finite difference method step size
            order=1 (int): The order of the finite difference method to use. ``1`` corresponds
                to forward finite differences, ``2`` to centered finite differences.

        Returns:
            tuple[list[QuantumTape], function]: A tuple containing the list of generated tapes,
            in addition to a post-processing function to be applied to the evaluated
            tapes.
        """

        if params is None:
            params = np.array(self.get_parameters())

        order = options.get("order", 1)
        h = options.get("h", 1e-7)

        shift = np.zeros_like(params, dtype=np.float64)
        shift[idx] = h

        if order == 1:
            # forward finite-difference.

            tapes = []

            # get the stored result of the original circuit
            y0 = options.get("y0", None)

            shifted = self.copy(copy_operations=True, tape_cls=QuantumTape)
            shifted.set_parameters(params + shift)

            tapes.append(shifted)

            if y0 is None:
                tapes.append(self)

            def processing_fn(results):
                """Computes the gradient of the parameter at index idx via first-order
                forward finite differences.

                Args:
                    results (list[real]): evaluated quantum tapes

                Returns:
                    array[float]: 1-dimensional array of length determined by the tape output
                    measurement statistics
                """
                shifted = qml.math.to_numpy(results[0])
                unshifted = y0

                if unshifted is None:
                    unshifted = np.array(results[1])

                return (shifted - unshifted) / h

            return tapes, processing_fn

        if order == 2:
            # central finite difference

            shifted_forward = self.copy(copy_operations=True, tape_cls=QuantumTape)
            shifted_forward.set_parameters(params + shift / 2)

            shifted_backward = self.copy(copy_operations=True, tape_cls=QuantumTape)
            shifted_backward.set_parameters(params - shift / 2)

            tapes = [shifted_forward, shifted_backward]

            def second_order_processing_fn(results):
                """Computes the gradient of the parameter at index idx via second-order
                centered finite differences.

                Args:
                    results (list[real]): evaluated quantum tapes

                Returns:
                    array[float]: 1-dimensional array of length determined by the tape output
                    measurement statistics
                """
                res0 = np.array(results[0])
                res1 = np.array(results[1])
                return (res0 - res1) / h

            return tapes, second_order_processing_fn

        raise ValueError("Order must be 1 or 2.")

    def device_pd(self, device, params=None, **options):
        """Evaluate the gradient of the tape with respect to
        all trainable tape parameters by querying the provided device.

        Args:
            device (.Device, .QubitDevice): a PennyLane device
                that can execute quantum operations and return measurement statistics
            params (list[Any]): The quantum tape operation parameters. If not provided,
                the current tape parameter values are used (via :meth:`~.get_parameters`).
        """
        jacobian_method = getattr(device, options.get("jacobian_method", "jacobian"))

        if params is None:
            params = np.array(self.get_parameters())

        saved_parameters = self.get_parameters()

        # temporarily mutate the in-place parameters
        self.set_parameters(params)

        # TODO: modify devices that have device Jacobian methods to
        # accept the quantum tape as an argument
        jac = jacobian_method(self, **options.get("device_pd_options", {}))

        # restore original parameters
        self.set_parameters(saved_parameters)
        return jac

    def analytic_pd(self, idx, params, **options):
        """Generate the quantum tapes and classical post-processing function required to compute the
        gradient of the tape with respect to a single trainable tape parameter using an analytic
        method.

        Args:
            idx (int): trainable parameter index to differentiate with respect to
            params (list[Any]): the quantum tape operation parameters

        Returns:
            tuple[list[QuantumTape], function]: A tuple containing the list of generated tapes,
            in addition to a post-processing function to be applied to the evaluated
            tapes.
        """
        raise NotImplementedError

    def hessian_pd(self, i, j, params, **options):
        """Generate the quantum tapes and classical post-processing function required to compute the
        Hessian of the tape with respect to two trainable tape parameter using an analytic
        method.

        Args:
            i (int): trainable parameter index to differentiate with respect to
            j (int): trainable parameter index to differentiate with respect to
            params (list[Any]): the quantum tape operation parameters

        Returns:
            tuple[list[QuantumTape], function]: A tuple containing the list of generated tapes,
            in addition to a post-processing function to be applied to the evaluated
            tapes.
        """
        raise NotImplementedError

    @staticmethod
    def _choose_params_with_methods(diff_methods, argnum):
        """Chooses the trainable parameters to use for computing the Jacobian
        by returning a map of their indices and differentiation methods.

        When there are fewer parameters specified than the total number of
        trainable parameters, the Jacobian is estimated by using the parameters
        specified using the ``argnum`` keyword argument.

        Args:
            diff_methods (list): the ordered list of differentiation methods
                for each parameter
            argnum (int, list(int), None): Indices for which argument(s) to
                compute the Jacobian with respect to.

        Returns:
            enumerate or list: map of the trainable parameter indices and
            differentiation methods
        """
        if argnum is None:
            return enumerate(diff_methods)

        if isinstance(argnum, int):
            argnum = [argnum]

        num_params = len(argnum)

        if num_params == 0:
            warnings.warn(
                "No trainable parameters were specified for computing the Jacobian.",
                UserWarning,
            )
            return []

        diff_methods_to_use = map(diff_methods.__getitem__, argnum)
        return zip(argnum, diff_methods_to_use)

    def jacobian(self, device, params=None, **options):
        r"""Compute the Jacobian of the parametrized quantum circuit recorded by the quantum tape.

        .. warning::

            The ``JacobianTape`` as well as the ``JacobianTape.jacobian()`` method is deprecated.
            Please use a standard :class:`~.QuantumTape`, and apply gradient transforms using
            the :mod:`.gradients` module to compute Jacobians.

        The quantum tape can be interpreted as a simple :math:`\mathbb{R}^m \to \mathbb{R}^n` function,
        mapping :math:`m` (trainable) gate parameters to :math:`n` measurement statistics,
        such as expectation values or probabilities.

        By default, the Jacobian will be computed with respect to all parameters on the quantum tape.
        This can be modified by setting the :attr:`~.trainable_params` attribute of the tape.

        The Jacobian can be computed using several methods:

        * Finite differences (``'numeric'``). The first-order method evaluates the circuit at
          :math:`n+1` points of the parameter space, the second-order method at :math:`2n` points,
          where ``n = tape.num_params``.

        * Analytic method (``'analytic'``). Analytic, if implemented by the inheriting quantum tape.

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
            since it compares the output at two points infinitesimally close to each other. Hence
            the ``'F'`` method works best with exact expectation values when using simulator
            devices.

        Args:
            device (.Device, .QubitDevice): a PennyLane device
                that can execute quantum operations and return measurement statistics
            params (list[Any]): The quantum tape operation parameters. If not provided,
                the current tape parameter values are used (via :meth:`~.get_parameters`).

        Keyword Args:
            method="best" (str): The differentiation method. Must be one of ``"numeric"``,
                ``"analytic"``, ``"best"``, or ``"device"``.
            h=1e-7 (float): finite difference method step size
            order=1 (int): The order of the finite difference method to use. ``1`` corresponds
                to forward finite differences, ``2`` to centered finite differences.
            shift=pi/2 (float): the size of the shift for two-term parameter-shift gradient computations
            argnum=None (int, list(int), None): Which argument(s) to compute the Jacobian
                with respect to. When there are fewer parameters specified than the
                total number of trainable parameters, the jacobian is being estimated.

        Returns:
            array[float]: 2-dimensional array of shape ``(tape.output_dim, tape.num_params)``

        **Example**

        .. code-block:: python

            with JacobianTape() as tape:
                qml.RX(0.432, wires=0)
                qml.RY(0.543, wires=0)
                qml.CNOT(wires=[0, 'a'])
                qml.RX(0.133, wires='a')
                qml.probs(wires=[0, 'a'])

        If parameters are not provided, the existing tape parameters are used:

        >>> dev = qml.device("default.qubit", wires=[0, 'a'])
        >>> tape.jacobian(dev)
        array([[-0.178441  , -0.23358253, -0.05892804],
               [-0.00079144, -0.00103601,  0.05892804],
               [ 0.00079144,  0.00103601,  0.00737611],
               [ 0.178441  ,  0.23358253, -0.00737611]])

        Parameters can be optionally passed during execution:

        >>> tape.jacobian(dev, params=[1.0, 0.0, 1.0])
        array([[-3.24029934e-01, -9.99200722e-09, -3.24029934e-01],
               [-9.67055711e-02, -2.77555756e-09,  3.24029935e-01],
               [ 9.67055709e-02,  3.05311332e-09,  9.67055709e-02],
               [ 3.24029935e-01,  1.08246745e-08, -9.67055711e-02]])

        Parameters provided for execution are temporary, and do not affect
        the tapes' parameters in-place:

        >>> tape.get_parameters()
        [0.432, 0.543, 0.133]

        Explicitly setting the trainable parameters can significantly reduce
        computational resources, as non-trainable parameters are ignored
        during the computation:

        >>> tape.trainable_params = {0} # set only the first parameter as trainable
        >>> tape.jacobian(dev)
        array([[-0.178441  ],
               [-0.00079144],
               [ 0.00079144],
               [ 0.178441  ]])

        If a tape has no trainable parameters, the Jacobian will be empty:

        >>> tape.trainable_params = {}
        >>> tape.jacobian(dev)
        array([], shape=(4, 0), dtype=float64)
        """
        # pylint: disable=too-many-statements

        warnings.warn(
            "Differentiating tapes using JacobianTape.jacobian() is deprecated. "
            "Please use ta standard QuantumTape with gradient transforms from "
            "the qml.gradients module instead."
        )

        if any(m.return_type is State for m in self.measurements):
            raise ValueError("The jacobian method does not support circuits that return the state")

        if self.is_sampled:
            raise qml.QuantumFunctionError(
                "Circuits that include sampling can not be differentiated."
            )

        method = options.get("method", "best")

        if method not in ("best", "numeric", "analytic", "device"):
            raise ValueError(f"Unknown gradient method '{method}'")

        if params is None:
            params = self.get_parameters()

        params = np.array(params)

        if method == "device":
            # Using device mode; simply query the device for the Jacobian
            return self.device_pd(device, params=params, **options)

        # perform gradient method validation
        diff_methods = self._grad_method_validation(method)

        if not self._has_trainable_params(params, diff_methods):
            # Either all parameters have grad method 0, or there are no trainable
            # parameters. Simply return an empty Jacobian.
            return np.zeros((self.output_dim, len(params)), dtype=float)

        if method == "numeric" or "F" in diff_methods:
            # there exist parameters that will be differentiated numerically

            if options.get("order", 1) == 1:
                # First order (forward) finite-difference will be performed.
                # Compute the value of the tape at the current parameters here. This ensures
                # this computation is only performed once, for all parameters.
                # convert to float64 to eliminate floating point errors when params float32
                params_f64 = np.array(params, dtype=np.float64)
                options["y0"] = qml.math.to_numpy(self.execute_device(params_f64, device))

        # some gradient methods need the device or the device wires
        options["device"] = device
        options["dev_wires"] = device.wires

        # collect all circuits (tapes) and postprocessing functions required
        # to compute the jacobian

        all_tapes = []
        reshape_info = []
        processing_fns = []
        nonzero_grad_idx = []

        argnum = options.get("argnum", None)

        params_with_methods = self._choose_params_with_methods(diff_methods, argnum)

        for trainable_idx, param_method in params_with_methods:
            if param_method == "0":
                continue

            nonzero_grad_idx.append(trainable_idx)

            t_idx = list(self.trainable_params)[trainable_idx]
            op = self._par_info[t_idx]["op"]

            if op.name == "Hamiltonian":
                # divert Hamiltonian differentiation to special recipe
                tapes, processing_fn = qml.gradients.hamiltonian_grad(
                    self, trainable_idx, params=params
                )

            elif (method == "best" and param_method[0] == "F") or (method == "numeric"):
                # numeric method
                tapes, processing_fn = self.numeric_pd(trainable_idx, params=params, **options)

            elif (method == "best" and param_method[0] == "A") or (method == "analytic"):
                # analytic method
                tapes, processing_fn = self.analytic_pd(trainable_idx, params=params, **options)

            processing_fns.append(processing_fn)

            # we create a flat list here to feed at once to the device
            all_tapes.extend(tapes)

            # to extract the correct result for this parameter later, remember the number of tapes
            reshape_info.append(len(tapes))

        # execute all tapes at once
        results = device.batch_execute(all_tapes)

        # post-process the results with the appropriate function to fill jacobian columns with gradients
        jac = None
        start = 0

        for i, processing_fn, res_len in zip(nonzero_grad_idx, processing_fns, reshape_info):
            # extract the correct results from the flat list
            res = results[start : start + res_len]
            start += res_len

            # postprocess results to compute the gradient
            g = self._flatten_processing_result(processing_fn(res))

            if jac is None:
                # update the tape's output dimension
                try:
                    self._output_dim = len(g)
                except TypeError:
                    # if g has no len (e.g., because it is not a numpy.ndarray)
                    # assume the dimension is 1
                    self._output_dim = 1
                # create the Jacobian matrix with appropriate dtype
                dtype = g.dtype if isinstance(g, (np.ndarray, float)) else np.object_
                jac = np.zeros((self._output_dim, len(params)), dtype=dtype)

            jac[:, i] = g

        return jac

    def hessian(self, device, params=None, **options):
        r"""Compute the Hessian of the parametrized quantum circuit recorded by the quantum tape.

        .. warning::

            The ``JacobianTape`` as well as the ``JacobianTape.hessian()`` method is deprecated.
            Please use a standard :class:`~.QuantumTape`, and apply gradient transforms using
            the :mod:`.gradients` module to compute Hessians.

        The quantum tape can be interpreted as a simple :math:`\mathbb{R}^m \to \mathbb{R}^n` function,
        mapping :math:`m` (trainable) gate parameters to :math:`n` measurement statistics,
        such as expectation values or probabilities.

        By default, the Hessian will be computed with respect to all parameters on the quantum tape.
        This can be modified by setting the :attr:`~.trainable_params` attribute of the tape.

        The Hessian can be currently computed using only the ``'analytic'`` method.

        Args:
            device (.Device, .QubitDevice): a PennyLane device
                that can execute quantum operations and return measurement statistics
            params (list[Any]): The quantum tape operation parameters. If not provided,
                the current tape parameter values are used (via :meth:`~.get_parameters`).

        Keyword Args:
            method="analytic" (str): The differentiation method. Currently only
                supports ``"analytic"``.
            s1=pi/2 (float): the size of the shift for index i in the parameter-shift Hessian computations
            s2=pi/2 (float): the size of the shift for index j in the parameter-shift Hessian computations

        Returns:
            array[float]: 2-dimensional array of shape ``(tape.num_params, tape.num_params)``

        **Example**

        .. code-block:: python

            n_wires = 5
            weights = [2.73943676, 0.16289932, 3.4536312, 2.73521126, 2.6412488]

            with QubitParamShiftTape() as tape:
                for i in range(n_wires):
                    qml.RX(weights[i], wires=i)

                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[2, 1])
                qml.CNOT(wires=[3, 1])
                qml.CNOT(wires=[4, 3])

                qml.expval(qml.PauliZ(1))

        If parameters are not provided, the existing tape parameters are used:

        >>> dev = qml.device("default.qubit", wires=n_wires)
        >>> tape.hessian(dev)
        array([[ 0.79380556,  0.05549219,  0.10891309, -0.1452963,   0.],
               [ 0.05549219,  0.79380556, -0.04208544,  0.05614438,  0.],
               [ 0.10891309, -0.04208544,  0.79380556,  0.11019314,  0.],
               [-0.1452963,   0.05614438,  0.11019314,  0.79380556,  0.],
               [ 0.,          0.,          0.,          0.,          0.]])

        Parameters can be optionally passed during execution:

        >>> tape.hessian(dev, params=[1.0, 1.0, 2.0, 0, 0])
        array([[ 0.12148432, -0.29466251,  0.41341091,  0.,          0.],
               [-0.29466251,  0.12148432,  0.41341091,  0.,          0.],
               [ 0.41341091,  0.41341091,  0.12148432,  0.,          0.],
               [ 0.,          0.,          0.,          0.12148432,  0.],
               [ 0.,          0.,          0.,          0.,          0.]])

        Parameters provided for execution are temporary, and do not affect
        the tapes' parameters in-place:

        >>> tape.get_parameters()
        [2.73943676, 0.16289932, 3.4536312, 2.73521126, 2.6412488]

        If a tape has no trainable parameters, the Hessian will be empty:

        >>> tape.trainable_params = {}
        >>> tape.hessian(dev)
        array([], shape=(0, 0), dtype=float64)
        """
        warnings.warn(
            "Differentiating tapes using JacobianTape.hessian() is deprecated. "
            "Please use ta standard QuantumTape with gradient transforms from "
            "the qml.gradients module instead."
        )

        if any(m.return_type is State for m in self.measurements):
            raise ValueError("The Hessian method does not support circuits that return the state")

        method = options.get("method", "analytic")

        if method != "analytic":
            raise ValueError(f"Unknown Hessian method '{method}'")

        if params is None:
            params = self.get_parameters()

        params = np.array(params)

        # perform gradient method validation
        diff_methods = self._grad_method_validation(method)

        if not self._has_trainable_params(params, diff_methods):
            # Either all parameters have grad method 0, or there are no trainable
            # parameters. Simply return an empty Hessian.
            return np.zeros((len(params), len(params)), dtype=float)

        # The parameter-shift Hessian implementation currently only supports
        # the two-term parameter-shift rule. Raise an error for unsupported operations.
        supported_ops = (
            "RX",
            "RY",
            "RZ",
            "Rot",
            "PhaseShift",
            "ControlledPhaseShift",
            "MultiRZ",
            "PauliRot",
            "U1",
            "U2",
            "U3",
            "SingleExcitationMinus",
            "SingleExcitationPlus",
            "DoubleExcitationMinus",
            "DoubleExcitationPlus",
            "OrbitalRotation",
        )

        for idx, info in self._par_info.items():
            op = info["op"]

            if idx in self.trainable_params and op.name not in supported_ops:
                raise ValueError(
                    f"The operation {op.name} is currently not supported for the "
                    f"parameter-shift Hessian.\nPlease decompose the operation in your "
                    f"QNode by replacing it with '{op.__str__().replace('(', '.decomposition(')}'"
                )

        # some gradient methods need the device or the device wires
        options["device"] = device
        options["dev_wires"] = device.wires

        # collect all circuits (tapes) and postprocessing functions required
        # to compute the Hessian
        all_tapes = []
        reshape_info = []
        processing_fns = []
        nonzero_grad_idx = []

        # From Schwarz's theorem, the Hessian will be symmetric, so we
        # can compute the upper triangular part only and symmetrize
        # the final Hessian.
        for i, j in itertools.combinations_with_replacement(range(len(diff_methods)), 2):
            if diff_methods[i] == "0" or diff_methods[j] == "0":
                continue

            nonzero_grad_idx.append((i, j))

            tapes, processing_fn = self.hessian_pd(i, j, params=params, **options)

            processing_fns.append(processing_fn)

            # we create a flat list here to feed at once to the device
            all_tapes.extend(tapes)

            # to extract the correct result for this parameter later, remember the number of tapes
            reshape_info.append(len(tapes))

        # execute all tapes at once
        results = device.batch_execute(all_tapes)

        hessian = None
        start = 0

        for (i, j), processing_fn, res_len in zip(nonzero_grad_idx, processing_fns, reshape_info):
            # extract the correct results from the flat list
            res = results[start : start + res_len]
            start += res_len

            # postprocess results to compute the gradient
            g = self._flatten_processing_result(processing_fn(res))

            if hessian is None:
                # create the Hessian matrix
                if self.output_dim is not None:
                    hessian = np.zeros(
                        (len(params), len(params), np.prod(self.output_dim)), dtype=float
                    )
                else:
                    hessian = np.zeros((len(params), len(params)), dtype=float)

            if i == j:
                hessian[i, i] = g
            else:
                hessian[i, j] = hessian[j, i] = g

        if self.output_dim == 1:
            hessian = np.squeeze(hessian, axis=-1)

        return hessian
