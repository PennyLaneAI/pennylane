# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Defines classes that take the vjps, jvps, and jacobians of circuits.
"""
import abc
from functools import partial
import inspect
import logging
from typing import Tuple, Callable, Optional, Union

from cachetools import LRUCache

import pennylane as qml
from pennylane.tape import QuantumScript
from pennylane.typing import ResultBatch, TensorLike

Batch = Tuple[QuantumScript]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _compute_vjps(jacs, dy, multi_measurements, has_partitioned_shots):
    """Compute the vjps of multiple tapes, directly for a Jacobian and co-tangents dys."""
    vjps = []
    for i, multi in enumerate(multi_measurements):
        dy_ = dy[i] if has_partitioned_shots else (dy[i],)
        jac_ = jacs[i] if has_partitioned_shots else (jacs[i],)

        shot_vjps = []
        for d, j in zip(dy_, jac_):
            if multi:
                shot_vjps.append(qml.gradients.compute_vjp_multi(d, j))
            else:
                shot_vjps.append(qml.gradients.compute_vjp_single(d, j))

        vjps.append(qml.math.sum(qml.math.stack(shot_vjps), axis=0))

    return tuple(vjps)


def _compute_jvps(jacs, tangents, multi_measurements):
    """Compute the jvps of multiple tapes, directly for a Jacobian and tangents."""
    jvps = []
    for i, multi in enumerate(multi_measurements):
        compute_func = (
            qml.gradients.compute_jvp_multi if multi else qml.gradients.compute_jvp_single
        )
        jvps.append(compute_func(tangents[i], jacs[i]))
    return tuple(jvps)


class JacobianProductCalculator(abc.ABC):
    """Provides methods for calculating the JVP/VJP between the Jacobians of tapes and tangents/cotangents."""

    @abc.abstractmethod
    def execute_and_compute_jvp(
        self, tapes: Batch, tangents: Tuple[Tuple[TensorLike]]
    ) -> Tuple[ResultBatch, Tuple]:
        """Calculate both the results for a batch of tapes and the jvp.

        This method is required to compute JVPs in the JAX interface.

        Args:
            tapes (tuple[`~.QuantumScript`]): The batch of tapes to take the derivatives of
            tangents (Sequence[Sequence[TensorLike]]): the tangents for the parameters of the tape.
                The ``i``th tangent corresponds to the ``i``th tape, and the ``j``th entry into a
                tangent entry corresponds to the ``j``th trainable parameter of the tape.

        Returns:
            ResultBatch, TensorLike: the results of the execution and the jacobian vector product

        **Examples:**

        For an instance of :class:`~.JacobianProductCalculator` ``jpc``, we have:

        >>> tape0 = qml.tape.QuantumScript([qml.RX(0.1, wires=0)], [qml.expval(qml.PauliZ(0))])
        >>> tape1 = qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.expval(qml.PauliZ(0))])
        >>> batch = (tape0, tape1)
        >>> tangents0 = (1.5, )
        >>> tangents1 = (2.0, )
        >>> tangents = (tangents0, tangents1)
        >>> results, jvps = jpc.execute_and_compute_jvp(batch, tangents)
        >>> expected_results = (np.cos(0.1), np.cos(0.2))
        >>> qml.math.allclose(results, expected_results)
        True
        >>> jvps
        (array(-0.14975012), array(-0.39733866))
        >>> expected_jvps = 1.5 * -np.sin(0.1), 2.0 * -np.sin(0.2)
        >>> qml.math.allclose(jvps, expected_jvps)
        True

        While this method could support non-scalar parameters in theory, no implementation currently supports
        jacobians with non-scalar parameters.

        """

    @abc.abstractmethod
    def compute_vjp(self, tapes: Batch, dy: Tuple[Tuple[TensorLike]]) -> Tuple:
        """Compute the vjp for a given batch of tapes.

        This method is used by autograd, torch, and tensorflow to compute VJPs.

        Args:
            tapes (tuple[`~.QuantumScript`]): the batch of tapes to take the derivatives of
            dy (tuple[tuple[TensorLike]]): the derivatives of the results of an execution.
                The ``i``th entry (cotangent) corresponds to the ``i``th tape, and the ``j``th entry of the ``i``th
                cotangent corresponds to the ``j``th return value of the ``i``th tape.

        Returns:
            TensorLike: the vector jacobian product.

        **Examples:**

        For an instance of :class:`~.JacobianProductCalculator` ``jpc``, we have:

        >>> tape0 = qml.tape.QuantumScript([qml.RX(0.1, wires=0)], [qml.expval(qml.PauliZ(0))])
        >>> tape1 = qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(0))])
        >>> batch = (tape0, tape1)
        >>> dy0 = (0.5, )
        >>> dy1 = (2.0, 3.0)
        >>> dys = (dy0, dy1)
        >>> vjps = jpc.compute_vjp(batch, dys)
        >>> vjps
        (array([-0.04991671]), array([2.54286107]))
        >>> expected_vjp0 = 0.5 * -np.sin(0.1)
        >>> qml.math.allclose(vjps[0], expected_vjp0)
        True
        >>> expected_jvp1 = 2.0 * -np.sin(0.2) + 3.0 * np.cos(0.2)
        >>> qml.math.allclose(vjps[1], expected_vjp1)
        True

        While this method could support non-scalar parameters in theory, no implementation currently supports
        jacobians with non-scalar parameters.

        """

    @abc.abstractmethod
    def compute_jacobian(self, tapes: Batch) -> Tuple:
        """Compute the full Jacobian for a batch of tapes.

        This method is required to compute Jacobians in the ``jax-jit`` interface

        Args:
            tapes: the batch of tapes to take the Jacobian of

        **Examples:**

        For an instance of :class:`~.JacobianProductCalculator` ``jpc``, we have:

        >>> tape0 = qml.tape.QuantumScript([qml.RX(0.1, wires=0)], [qml.expval(qml.PauliZ(0))])
        >>> tape1 = qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(0))])
        >>> batch = (tape0, tape1)
        >>> jpc.compute_jacobian(batch)
        (array(-0.09983342), (array(-0.19866933), array(0.98006658)))

        While this method could support non-scalar parameters in theory, no implementation currently supports
        jacobians with non-scalar parameters.

        """


class TransformJacobianProducts(JacobianProductCalculator):
    """Compute VJPs, JVPs and Jacobians via a :class:`~.gradient_transform`.

    Args:
        inner_execute (Callable[[Tuple[QuantumTape]], ResultBatch]): a function that
            executes the batch of circuits and returns their results.
        gradient_transform (pennylane.gradients.gradient_transform): the gradient transform to use.
        gradient_kwargs (dict): Any keyword arguments for the gradient transform.

    >>> inner_execute = qml.device('default.qubit').execute
    >>> gradient_transform = qml.gradients.param_shift
    >>> kwargs = {"broadcast": True}
    >>> jpc = TransformJacobianProducts(inner_execute, gradient_transform, kwargs)

    """

    def __repr__(self):
        return f"TransformJacobianProducts({self._inner_execute}, gradient_transform={self._gradient_transform}, gradient_kwargs={self._gradient_kwargs})"

    def __init__(
        self,
        inner_execute: Callable,
        gradient_transform: "qml.gradients.gradient_transform",
        gradient_kwargs: Optional[dict] = None,
    ):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "TransformJacobianProduct being created with (%s, %s, %s)",
                inner_execute
                if not (logger.isEnabledFor(qml.logging.TRACE) and callable(inner_execute))
                else "\n" + inspect.getsource(inner_execute),
                gradient_transform,
                gradient_kwargs,
            )
        self._inner_execute = inner_execute
        self._gradient_transform = gradient_transform
        self._gradient_kwargs = gradient_kwargs or {}

    def execute_and_compute_jvp(self, tapes: Batch, tangents: Tuple[Tuple[TensorLike]]):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("execute_and_compute_jvp called with (%s, %s)", tapes, tangents)
        num_result_tapes = len(tapes)

        jvp_tapes, jvp_processing_fn = qml.gradients.batch_jvp(
            tapes, tangents, self._gradient_transform, gradient_kwargs=self._gradient_kwargs
        )

        full_batch = tapes + tuple(jvp_tapes)

        full_results = self._inner_execute(full_batch)

        results = full_results[:num_result_tapes]
        jvp_results = full_results[num_result_tapes:]
        jvps = jvp_processing_fn(jvp_results)
        return tuple(results), tuple(jvps)

    def compute_vjp(self, tapes: Batch, dy: Tuple[Tuple[TensorLike]]):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("compute_vjp called with (%s, %s)", tapes, dy)
        vjp_tapes, processing_fn = qml.gradients.batch_vjp(
            tapes, dy, self._gradient_transform, gradient_kwargs=self._gradient_kwargs
        )

        vjp_results = self._inner_execute(vjp_tapes)
        return tuple(processing_fn(vjp_results))

    def compute_jacobian(self, tapes: Batch):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("compute_jacobian called with %s", tapes)
        partial_gradient_fn = partial(self._gradient_transform, **self._gradient_kwargs)
        jac_tapes, batch_post_processing = qml.transforms.map_batch_transform(
            partial_gradient_fn, tapes
        )
        results = self._inner_execute(jac_tapes)
        return tuple(batch_post_processing(results))


class DeviceJacobians(JacobianProductCalculator):
    """Calculate jacobian products via an experimental device.

    Args:
        device (Union[pennylane.Device, pennylane.devices.experimental.Device]): the device for execution and derivatives.
            Must support supports first order gradients with the requested configuration.
        gradient_kwargs (dict): a dictionary of keyword options for the gradients. Only used with a :class:`~.pennylane.Device`
            old device interface.
        execution_config (pennylane.devices.experimental.ExecutionConfig): a datastructure containing the parameters needed to fully
           describe the execution. Only used with :class:`pennylane.devices.experimental.ExecutionConfig` new device interface.

    **Examples:**

    >>> device = qml.devices.experimental.DefaultQubit2()
    >>> config = qml.devices.experimental.ExecutionConfig(gradient_method="adjoint")
    >>> jpc = DeviceJacobians(device, {}, config)

    This same class can also be used with the old device interface.

    >>> device = qml.device('lightning.qubit', wires=5)
    >>> gradient_kwargs = {"method": "adjoint_jacobian"}
    >>> jpc_lightning = DeviceJacobians(device, gradient_kwargs)

    **Technical comments on caching and ``grad_on_execution=True``:**

    In order to store results and jacobian for the backward pass during the forward pass,
    the ``_jacs_cache`` and ``_results_cache`` properties are ``LRUCache`` objects with a maximum size of 10.

    Note that the the results and jacobains are cached based on the ``id`` of the tapes. This is done to separate the key
    from potentially expensive ``QuantumScript.hash``. This means that the batch of tapes must be the
    same instance, not just something that looks the same but has a different location in memory.

    When a forward pass with :meth:`~.execute` is called, both the results and the jacobian for the object are stored.

    >>> tape = qml.tape.QuantumScript([qml.RX(1.0, wires=0)], [qml.expval(qml.PauliZ(0))])
    >>> batch = (tape, )
    >>> with device.tracker:
    ...     results = jpc.execute(batch )
    >>> results
    (0.5403023058681398,)
    >>> device.tracker.totals
    {'execute_and_derivative_batches': 1, 'executions': 1, 'derivatives': 1}
    >>> jpc._jacs_cache
    LRUCache({5660934048: (array(-0.84147098),)}, maxsize=10, currsize=1)

    Then when the vjp, jvp, or jacobian is requested, that cached value is used instead of requesting from
    the device again.

    >>> with device.tracker:
    ...     vjp = jpc.compute_vjp(batch , (0.5, ) )
    >>> vjp
    (array([-0.42073549]),)
    >>> device.tracker.totals
    {}

    """

    def __repr__(self):
        return f"<DeviceJacobians: {self._device.name}, {self._gradient_kwargs}, {self._execution_config}>"

    def __init__(
        self,
        device: Union[qml.devices.experimental.Device, qml.Device],
        gradient_kwargs: dict,
        execution_config: Optional["qml.devices.experimental.ExecutionConfig"] = None,
    ):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "DeviceJacobians created with (%s, %s, %s)",
                device,
                execution_config,
                gradient_kwargs,
            )

        self._device = device
        self._execution_config = execution_config
        self._gradient_kwargs = gradient_kwargs

        self._is_new_device = not isinstance(device, qml.Device)

        # only really need to keep most recent entry, but keeping 10 around just in case
        self._results_cache = LRUCache(maxsize=10)
        self._jacs_cache = LRUCache(maxsize=10)

    def _dev_execute_and_compute_derivatives(self, tapes: Batch):
        """
        Converts tapes to numpy before computing the the results and derivatives on the device.

        Dispatches between the two different device interfaces.
        """
        numpy_tapes = tuple(qml.transforms.convert_to_numpy_parameters(t) for t in tapes)
        if self._is_new_device:
            return self._device.execute_and_compute_derivatives(numpy_tapes, self._execution_config)
        return self._device.execute_and_gradients(numpy_tapes, **self._gradient_kwargs)

    def _dev_execute(self, tapes: Batch):
        """
        Converts tapes to numpy before computing just the results on the device.

        Dispatches between the two different device interfaces.
        """
        numpy_tapes = tuple(qml.transforms.convert_to_numpy_parameters(t) for t in tapes)
        if self._is_new_device:
            return self._device.execute(numpy_tapes, self._execution_config)
        return self._device.batch_execute(numpy_tapes)

    def _dev_compute_derivatives(self, tapes: Batch):
        """
        Converts tapes to numpy before computing the derivatives on the device.

        Dispatches between the two different device interfaces.
        """
        numpy_tapes = tuple(qml.transforms.convert_to_numpy_parameters(t) for t in tapes)
        if self._is_new_device:
            return self._device.compute_derivatives(numpy_tapes, self._execution_config)
        return self._device.gradients(numpy_tapes, **self._gradient_kwargs)

    def execute(self, tapes: Batch):
        """Forward pass used to cache the results and jacobians.

        Args:
            tapes (tuple[`~.QuantumScript`]): the batch of tapes to execute and take derivatives of

        Returns:
            ResultBatch: the results of the execution.

        Side Effects:
            Caches both the results and jacobian into ``_results_cache`` and ``jacs_cache``.

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Forward pass called with %s", tapes)
        results, jac = self._dev_execute_and_compute_derivatives(tapes)
        self._results_cache[id(tapes)] = results
        self._jacs_cache[id(tapes)] = jac
        return results

    def execute_and_compute_jvp(self, tapes: Batch, tangents):
        if id(tapes) not in self._results_cache and id(tapes) not in self._jacs_cache:
            results, jacs = self._dev_execute_and_compute_derivatives(tapes)
            self._results_cache[id(tapes)] = results
            self._jacs_cache[id(tapes)] = jacs
        else:
            if id(tapes) in self._results_cache:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Retrieving results from cache.")
                results = self._results_cache[id(tapes)]
            else:
                results = self._dev_execute(tapes)
                self._results_cache[id(tapes)] = results

            if id(tapes) in self._jacs_cache:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Retrieving jacobian from cache.")
                jacs = self._jacs_cache[id(tapes)]
            else:
                jacs = self._dev_compute_derivatives(tapes)
                self._jacs_cache[id(tapes)] = jacs

        multi_measurements = (len(t.measurements) > 1 for t in tapes)
        jvps = _compute_jvps(jacs, tangents, multi_measurements)
        return results, jvps

    def compute_vjp(self, tapes, dy):
        if id(tapes) in self._jacs_cache:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Retrieving jacobian from cache.")
            jacs = self._jacs_cache[id(tapes)]
        else:
            jacs = self._dev_compute_derivatives(tapes)
            self._jacs_cache[id(tapes)] = jacs

        multi_measurements = (len(t.measurements) > 1 for t in tapes)
        has_partitioned_shots = tapes[0].shots.has_partitioned_shots
        return _compute_vjps(
            jacs, dy, multi_measurements, has_partitioned_shots=has_partitioned_shots
        )

    def compute_jacobian(self, tapes):
        if id(tapes) in self._jacs_cache:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Retrieving jacobian from cache.")
            return self._jacs_cache[id(tapes)]

        jacs = self._dev_compute_derivatives(tapes)
        self._jacs_cache[id(tapes)] = jacs
        return jacs
