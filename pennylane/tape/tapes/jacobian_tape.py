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
This module contains the ``JacobianTape``, which adds differentiation methods
to the ``QuantumTape`` class.
"""
# pylint: disable=too-many-branches

import numpy as np

import pennylane as qml

from pennylane.operation import State
from pennylane.tape.tapes.tape import QuantumTape

STATE_PREP_OPS = (
    qml.BasisState,
    qml.QubitStateVector,
    qml.CatState,
    qml.CoherentState,
    qml.FockDensityMatrix,
    qml.DisplacedSqueezedState,
    qml.FockState,
    qml.FockStateVector,
    qml.ThermalState,
    qml.GaussianState,
)


# pylint: disable=too-many-public-methods
class JacobianTape(QuantumTape):
    """A quantum tape recorder, that records, validates, executes,
    and differentiates variational quantum programs.

    .. note::

        As the quantum tape is a *beta* feature. See :mod:`pennylane.tape`
        for more details.

    Args:
        name (str): a name given to the quantum tape
        caching (int): Number of device executions to store in a cache to speed up subsequent
            executions. A value of ``0`` indicates that no caching will take place. Once filled,
            older elements of the cache are removed and replaced with the most recent device
            executions to keep the cache up to date.

    **Example**

    In addition to the functionality of the ``QuantumTape`` class,
    the ``JacobianTape`` has the ability to compute Jacobians of a quantum
    circuit. Consider the following

    .. code-block:: python

        import pennylane.tape

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.432, wires=0)
            qml.RY(0.543, wires=0)
            qml.CNOT(wires=[0, 'a'])
            qml.RX(0.133, wires='a')
            expval(qml.PauliZ(wires=[0]))

    The Jacobian is computed using finite difference:

    >>> tape.jacobian(dev)
    [[-0.35846484 -0.46923704  0.        ]]
    >>> tape.jacobian(dev, params=[0.1, 0.1, 0.1])
    [[-0.09933471 -0.09933471  0.        ]]

    Trainable parameters are taken into account when calculating the Jacobian,
    avoiding unnecessary calculations:

    >>> tape.trainable_params = {0} # set only the first parameter as free
    >>> tape.set_parameters(0.56)
    >>> tape.jacobian(dev)
    [[-0.45478169]]
    """

    def __init__(self, name=None, caching=0):
        super().__init__(name=name, caching=caching)
        self.jacobian_options = {}

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

        allowed_param_methods = {
            idx: info["grad_method"]
            for idx, info in self._par_info.items()
            if idx in self.trainable_params
        }

        # check and raise an error if any parameters are non-differentiable
        nondiff_params = {idx for idx, g in allowed_param_methods.items() if g is None}

        if nondiff_params:
            raise ValueError(f"Cannot differentiate with respect to parameter(s) {nondiff_params}")

        numeric_params = {idx for idx, g in allowed_param_methods.items() if g == "F"}

        # If explicitly using analytic mode, ensure that all parameters
        # support analytic differentiation.
        if method == "analytic" and numeric_params:
            raise ValueError(
                f"The analytic gradient method cannot be used with the argument(s) {numeric_params}."
            )

        return tuple(allowed_param_methods.values())

    def numeric_pd(self, idx, device, params=None, **options):
        """Evaluate the gradient of the tape with respect to
        a single trainable tape parameter using numerical finite-differences.

        Args:
            idx (int): trainable parameter index to differentiate with respect to
            device (.Device, .QubitDevice): a PennyLane device
                that can execute quantum operations and return measurement statistics
            params (list[Any]): The quantum tape operation parameters. If not provided,
                the current tape parameter values are used (via :meth:`~.get_parameters`).

        Keyword Args:
            h=1e-7 (float): finite difference method step size
            order=1 (int): The order of the finite difference method to use. ``1`` corresponds
                to forward finite differences, ``2`` to centered finite differences.

        Returns:
            array[float]: 1-dimensional array of length determined by the tape output
            measurement statistics
        """
        if params is None:
            params = np.array(self.get_parameters())

        order = options.get("order", 1)
        h = options.get("h", 1e-7)

        shift = np.zeros_like(params, dtype=np.float64)
        shift[idx] = h

        if order == 1:
            # Forward finite-difference.
            # Check if the device has already be pre-computed with
            # unshifted parameter values, to avoid redundant evaluations.
            y0 = options.get("y0", None)

            if y0 is None:
                y0 = np.asarray(self.execute_device(params, device))

            y = np.asarray(self.execute_device(params + shift, device))
            return (y - y0) / h

        if order == 2:
            # central finite difference
            shift_forward = np.array(self.execute_device(params + shift / 2, device))
            shift_backward = np.array(self.execute_device(params - shift / 2, device))
            return (shift_forward - shift_backward) / h

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
        # pylint:disable=unused-argument
        if params is None:
            params = np.array(self.get_parameters())

        saved_parameters = self.get_parameters()

        # temporarily mutate the in-place parameters
        self.set_parameters(params)

        # TODO: modify devices that have device Jacobian methods to
        # accept the quantum tape as an argument
        jac = device.jacobian(self)

        # restore original parameters
        self.set_parameters(saved_parameters)
        return jac

    def analytic_pd(self, idx, device, params=None, **options):
        """Evaluate the gradient of the tape with respect to
        a single trainable tape parameter using an analytic method.

        Args:
            idx (int): trainable parameter index to differentiate with respect to
            device (.Device, .QubitDevice): a PennyLane device
                that can execute quantum operations and return measurement statistics
            params (list[Any]): The quantum tape operation parameters. If not provided,
                the current tape parameter values are used (via :meth:`~.get_parameters`).

        Returns:
            array[float]: 1-dimensional array of length determined by the tape output
                measurement statistics
        """
        raise NotImplementedError

    def jacobian(self, device, params=None, **options):
        r"""Compute the Jacobian of the parametrized quantum circuit recorded by the quantum tape.

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

        Returns:
            array[float]: 2-dimensional array of shape ``(tape.num_params, tape.output_dim)``

        **Example**

        .. code-block:: python

            with JacobianTape() as tape:
                qml.RX(0.432, wires=0)
                qml.RY(0.543, wires=0)
                qml.CNOT(wires=[0, 'a'])
                qml.RX(0.133, wires='a')
                probs(wires=[0, 'a'])

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
        if any([m.return_type is State for m in self.measurements]):
            raise ValueError("The jacobian method does not support circuits that return the state")

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
        allowed_param_methods = self._grad_method_validation(method)

        if not params.size or all(g == "0" for g in allowed_param_methods):
            # Either all parameters have grad method 0, or there are no trainable
            # parameters. Simply return an empty Jacobian.
            return np.zeros((self.output_dim, len(params)), dtype=float)

        if method == "numeric" or "F" in allowed_param_methods:
            # there exist parameters that will be differentiated numerically

            if options.get("order", 1) == 1:
                # First order (forward) finite-difference will be performed.
                # Compute the value of the tape at the current parameters here. This ensures
                # this computation is only performed once, for all parameters.
                options["y0"] = np.asarray(self.execute_device(params, device))

        jac = None
        p_ind = range(len(params))

        # loop through each parameter and compute the gradient
        for idx, (l, param_method) in enumerate(zip(p_ind, allowed_param_methods)):

            if param_method == "0":
                # Independent parameter. Skip, as this parameter has a gradient of 0.
                continue

            if (method == "best" and param_method[0] == "F") or (method == "numeric"):
                # finite difference method
                g = self.numeric_pd(l, device, params=params, **options)

            elif (method == "best" and param_method[0] == "A") or (method == "analytic"):
                # analytic method
                g = self.analytic_pd(l, device, params=params, **options)

            if g.dtype is np.dtype("object"):
                # object arrays cannot be flattened; must hstack them
                g = np.hstack(g)

            if jac is None:
                # The Jacobian matrix has not yet been created, as we needed at least
                # one device execution to occur so that we could ensure that the output
                # dimension is known.
                jac = np.zeros((self.output_dim, len(params)), dtype=float)

            jac[:, idx] = g.flatten()

        return jac
