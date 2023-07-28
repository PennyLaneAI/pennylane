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
Defines classes taking the vjps and jvps of circuits.
"""
import abc
from typing import Tuple, Callable, Optional

import pennylane as qml
from pennylane.tape import QuantumScript
from pennylane.typing import ResultBatch, TensorLike

from .set_shots import set_shots

Batch = Tuple[QuantumScript]


def _compute_jvps(jacs, tangents, multi_measurements):
    """Compute the jvps of multiple tapes, directly for a Jacobian and tangents."""
    jvps = []
    for i, multi in enumerate(multi_measurements):
        compute_func = (
            qml.gradients.compute_jvp_multi if multi else qml.gradients.compute_jvp_single
        )
        jvps.append(compute_func(tangents[i], jacs[i]))
    return tuple(jvps)


def _compute_vjps(dys, jacs, multi_measurements):
    """Compute the vjps of multiple tapes, directly for a Jacobian and tangents."""
    vjps = []

    for i, multi in enumerate(multi_measurements):
        compute_func = (
            qml.gradients.compute_vjp_multi if multi else qml.gradients.compute_vjp_single
        )
        vjps.append(compute_func(dys[i], jacs[i]))
    return tuple(vjps)


class DerivativeExecutor(abc.ABC):
    """Provides methods for calculating the jvp and vjps for tapes and tangents/ cotangents."""

    @abc.abstractmethod
    def execute_and_compute_jvp(
        self, tapes: Batch, tangents: TensorLike
    ) -> Tuple[ResultBatch, Tuple]:
        """Calculate both the results for a batch of tapes and the jvp.

        Args:
            tapes: The batch of tapes to take the derivatives of
            tangents (Iterable[TensorLike]): the tangents for the parameters of the tape

        Returns:
            ResultBatch, TensorLike

        >>> tape0 = qml.tape.QuantumScript([qml.RX(0.1, wires=0)], [qml.expval(qml.PauliZ(0))])
        >>> batch = (tape0, )
        >>> tangent_variables = (1.5, )
        >>> derivatives_executor.execute_and_compute_vjp(batch, tangent_variables)
        ((array(0.99500417),), (-0.14975012497024237,))

        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_jvp(self, tapes: Batch, tangents: TensorLike) -> Tuple:
        """Calculate both the results for a batch of tapes and the jvp.

        Args:
            tapes: The batch of tapes to take the derivatives of
            tangents (Iterable[TensorLike]): the tangents for the parameters of the tape

        Returns:
            TensorLike

        >>> tape0 = qml.tape.QuantumScript([qml.RX(0.1, wires=0)], [qml.expval(qml.PauliZ(0))])
        >>> batch = (tape0, )
        >>> tangent_variables = (1.5, )
        >>> derivatives_executor.compute_vjp(batch, tangent_variables)
        (-0.14975012497024237, )

        """
        raise NotImplementedError

    @abc.abstractmethod
    def execute_and_compute_vjp(self, tapes: Batch, dy: TensorLike) -> Tuple[ResultBatch, Tuple]:
        """Compute the vjp for a given batch of tapes.

        Args:
            tapes: the batch of tapes to the the derivatives of
            dy: the derivatives of the results of an execution

        Returns:
            ResultBatch, TensorLike

        >>> tape0 = qml.tape.QuantumScript([qml.RX(0.1, wires=0)], [qml.expval(qml.PauliZ(0))])
        >>> batch = (tape0, )
        >>> derivatives_executor.execute_and_compute_vjp(batch, (0.5, ))
        ((array(0.99500417),), (-0.04991670832341412,))
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_vjp(self, tapes, dy) -> Tuple:
        """Compute the vjp for a given batch of tapes.

        Args:
            tapes: the batch of tapes to the the derivatives of
            dy: the derivatives of the results of an execution

        Returns:
            TensorLike

        >>> tape0 = qml.tape.QuantumScript([qml.RX(0.1, wires=0)], [qml.expval(qml.PauliZ(0))])
        >>> batch = (tape0, )
        >>> derivatives_executor.compute_vjp(batch, (0.5, ))
        (-0.04991670832341412,)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_jacobian(self, tapes, use_pure_callback=False) -> Tuple:
        raise NotImplementedError


class TransformDerivatives(DerivativeExecutor):
    """Compute vjp and jvps via a gradient transform.

    Args:
        next_executor (Executor): where to execute the gradient tapes.
        gradient_transform (TransformContainer): the gradient transform to use.

    Note that this class accepts a :class:`~.TransformContainer` with bound keyword arguments
    instead of the ``gradient_transform`` itself.

    >>> dev_ex = DeviceExecutor(ExecutionConfig(), DefaultQubit2())
    >>> gradient_kwargs = {}
    >>> par_shift = TransformContainer(qml.gradients.param_shift, kwargs = gradient_kwargs)
    >>> derivatives_executor = TransformDerivatives(dev_ex, par_shift)

    To accomplish higher order derivatives, the provided executor can already be registered with
    a machine learning library.

    >>> jax_layer = get_interface_layer("jax")(dev_ex, derivatives_executor)
    >>> second_order_derivatives = TransformDerivatives(jax_layer, par_shift)
    >>> jax_layer2 = get_interface_layer("jax")(jax_layer, second_order_derivatives)

    """

    def __repr__(self):
        return f"TransformDerivatives({self._inner_execute}, gradient_transform={self._gradient_transform}, gradient_kwargs={self._gradient_kwargs})"

    def __init__(
        self,
        inner_execute: Callable,
        gradient_transform: "qml.gradients.gradient_transform",
        gradient_kwargs: Optional[dict] = None,
    ):
        self._inner_execute = inner_execute
        self._gradient_transform = gradient_transform
        self._gradient_kwargs = gradient_kwargs or {}

    def execute_and_compute_jvp(self, tapes, tangents):
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

    def compute_jvp(self, tapes, tangents):
        jvp_tapes, jvp_processing_fn = qml.gradients.batch_jvp(
            tapes, tangents, self._gradient_transform, gradient_kwargs=self._gradient_kwargs
        )
        jvp_results = self._inner_execute(jvp_tapes)

        return tuple(jvp_processing_fn(jvp_results))

    def execute_and_compute_vjp(self, tapes, dy) -> Tuple[ResultBatch, Tuple]:
        num_result_tapes = len(tapes)

        jvp_tapes, vjp_processing_fn = qml.gradients.batch_vjp(
            tapes, dy, self._gradient_transform, gradient_kwargs=self._gradient_kwargs
        )

        full_batch = tuple(tapes) + tuple(jvp_tapes)

        full_results = self._inner_execute(full_batch)

        results = full_results[:num_result_tapes]
        vjp_results = full_results[num_result_tapes:]

        vjps = vjp_processing_fn(vjp_results)
        return tuple(results), tuple(vjps)

    def compute_vjp(self, tapes, dy):
        vjp_tapes, processing_fn = qml.gradients.batch_vjp(
            tapes, dy, self._gradient_transform, gradient_kwargs=self._gradient_kwargs
        )

        vjp_results = self._inner_execute(vjp_tapes)

        return tuple(processing_fn(vjp_results))

    def compute_jacobian(self, tapes, use_pure_callback=False):
        jacobians = []
        for new_t in tapes:
            jac_tapes, res_processing_fn = self._gradient_transform(new_t, **self._gradient_kwargs)
            jacs_results = self._inner_execute(jac_tapes)
            jacs = res_processing_fn(jacs_results)
            jacobians.append(jacs)
        return tuple(jacobians)


class OldDeviceDerivatives(DerivativeExecutor):
    def __init__(self, device, gradient_kwargs, override_shots):
        self._device = device
        self._gradient_kwargs = gradient_kwargs
        self._override_shots = override_shots
        self._jacobian_cache = {}
        self._results_cache = {}

    def __call__(self, tapes):
        tapes = tuple(tapes)
        if tapes not in self._jacobian_cache:
            unwrapped_tapes = tuple(qml.transforms.convert_to_numpy_parameters(t) for t in tapes)
            results, jacs = set_shots(self._device, self._override_shots)(
                self._device.execute_and_gradients
            )(unwrapped_tapes, **self._gradient_kwargs)
            results = tuple(results)
            jacs = tuple(jacs)
            self._results_cache[tapes] = results
            self._jacobian_cache[tapes] = jacs
        return self._results_cache[tapes]

    def compute_jvp(self, tapes, tangents):
        return None

    def execute_and_compute_jvp(self, tapes, tangents):
        tapes = tuple(qml.transforms.convert_to_numpy_parameters(t) for t in tapes)
        multi_measurements = [len(tape.measurements) > 1 for tape in tapes]

        if tapes not in self._jacobian_cache:
            results, jacs = set_shots(self._device, self._override_shots)(
                self._device.execute_and_gradients
            )(tapes, **self._gradient_kwargs)
            results = tuple(results)
            jacs = tuple(jacs)
            self._jacobian_cache[tapes] = jacs
            self._results_cache[tapes] = results

        jvps = _compute_jvps(self._jacobian_cache[tapes], tangents, multi_measurements)
        return self._results_cache[tapes], jvps

    def execute_and_compute_vjp(self, tapes, dy) -> Tuple[ResultBatch, Tuple]:
        return None

    def compute_vjp(self, tapes, dy):
        tapes = tuple(qml.transforms.convert_to_numpy_parameters(t) for t in tapes)
        multi_measurements = (len(t.measurements) > 1 for t in tapes)

        if tapes not in self._jacobian_cache:
            jacs = set_shots(self._device, self._override_shots)(self._device.gradients)(
                tapes, **self._gradient_kwargs
            )
            self._jacobian_cache[tapes] = jacs

        return _compute_vjps(dy, self._jacobian_cache[tapes], multi_measurements)

    def compute_jacobian(self, tapes, use_pure_callback=False):
        tapes = tuple(qml.transforms.convert_to_numpy_parameters(t) for t in tapes)
        if not use_pure_callback:
            return set_shots(self._device, self._override_shots)(self._device.gradients)(
                tapes, **self._gradient_kwargs
            )
        from .pure_callback import _old_device_jac_via_callback

        return set_shots(self._device, self._override_shots)(_old_device_jac_via_callback)(
            tapes, self._device, self._gradient_kwargs
        )


class DeviceDerivatives(DerivativeExecutor):
    """Compute vjp and jvps using device provided derivatives.

    Args:
        device (Device): the device to use to compute derivatives
        execution_config (ExecutionConfig): the configuration for the device

    >>> dev = DefaultQubit2()
    >>> config = ExecutionConfig(use_device_gradient=True, gradient_method="adjoint")
    >>> derivatives_executor = DeviceDerivatives(dev, config_device_derivative)

    To calculate and cache the jacobian on the forward pass of an excution, this class can
    also be used as an ``Executor``.

    >>> tape0 = qml.tape.QuantumScript([qml.RX(0.1, wires=0)], [qml.expval(qml.PauliZ(0))])
    >>> batch = (tape0, )
    >>> with dev.tracker:
    ...     derivatives_executor(batch)
    >>> dev.tracker.totals
    {'execute_and_compute_derivative_batches': 1,
    'batches': 1,
    'executions': 1,
    'derivative_batches': 1,
    'derivatives': 1}
    >>> with dev.tracker:
    ...     derivatives_executor.compute_vjp(batch, tangent_variables)
    >>> dev.tracker.totals
    {}
    >>> derivatives_executor._jacobian_cache
    {(<QuantumScript: wires=[0], params=1>,): (array(-0.09983342),)}

    Note that caching is based on the *identity* of the input, not its contents. Identical
    batches at different locations in memory will be cached separately. This is to spend less
    time computing the hash.
    """

    def __repr__(self):
        return f"Device derivatives({self._device})"

    def __init__(self, device, execution_config, use_pure_callback=False):
        self._device = device
        self._execution_config = execution_config
        self._jacobian_cache = {}
        self._results_cache = {}
        self._use_pure_callback = use_pure_callback

    def __call__(self, tapes):
        if not self._use_pure_callback:
            tapes = tuple(tapes)
            if tapes not in self._jacobian_cache:
                unwrapped_tapes = tuple(
                    qml.transforms.convert_to_numpy_parameters(t) for t in tapes
                )
                results, jacs = self._device.execute_and_compute_derivatives(
                    unwrapped_tapes, self._execution_config
                )
                self._results_cache[tapes] = results
                self._jacobian_cache[tapes] = jacs
            return self._results_cache[tapes]

        from .pure_callback import _new_device_execute_and_jac

        res, jacs = _new_device_execute_and_jac(tapes, self._device, self._execution_config)
        self._jacobian_cache[tapes] = jacs
        return res

    def execute_and_compute_jvp(self, tapes, tangents):
        tapes = tuple(qml.transforms.convert_to_numpy_parameters(t) for t in tapes)
        multi_measurements = [len(tape.measurements) > 1 for tape in tapes]

        if tapes not in self._jacobian_cache:
            results, jacs = self._device.execute_and_compute_derivatives(
                tapes, self._execution_config
            )
            self._jacobian_cache[tapes] = jacs
            self._results_cache[tapes] = results

        jvps = _compute_jvps(self._jacobian_cache[tapes], tangents, multi_measurements)
        return self._results_cache[tapes], jvps

    def compute_jvp(self, tapes, tangents):
        tapes = tuple(qml.transforms.convert_to_numpy_parameters(t) for t in tapes)
        if tapes not in self._jacobian_cache:
            jacs = self._device.compute_derivatives(tapes, self._execution_config)
            self._jacobian_cache[tapes] = jacs

        multi_measurements = [len(tape.measurements) > 1 for tape in tapes]
        return _compute_jvps(self._jacobian_cache[tapes], tangents, multi_measurements)

    def execute_and_compute_vjp(self, tapes, dy) -> Tuple[ResultBatch, Tuple]:
        tapes = tuple(qml.transforms.convert_to_numpy_parameters(t) for t in tapes)
        if tapes not in self._jacobian_cache:
            results, jacs = self._device.execute_and_compute_derivatives(
                tapes, self._execution_config
            )
            self._results_cache[tapes] = results
            self._jacobian_cache[jacs] = jacs

        multi_measurements = [len(tape.measurements) > 1 for tape in tapes]
        vjps = _compute_vjps(dy, self._jacobian_cache[tapes], multi_measurements)
        return tuple(results), tuple(vjps)

    def compute_vjp(self, tapes, dy):
        tapes = tuple(qml.transforms.convert_to_numpy_parameters(t) for t in tapes)
        multi_measurements = (len(t.measurements) > 1 for t in tapes)

        if tapes not in self._jacobian_cache:
            jacs = self._device.compute_derivatives(tapes, self._execution_config)
            self._jacobian_cache[tapes] = jacs

        return _compute_vjps(dy, self._jacobian_cache[tapes], multi_measurements)

    def compute_jacobian(self, tapes):
        if not self._use_pure_callback:
            tapes = tuple(qml.transforms.convert_to_numpy_parameters(t) for t in tapes)

            if tuple(tapes) not in self._jacobian_cache:
                jacs = self._device.compute_derivatives(tapes, self._execution_config)
                self._jacobian_cache[tapes] = jacs

            return self._jacobian_cache[tapes]

        from .pure_callback import _new_device_jac_via_callback

        return _new_device_jac_via_callback(tapes, self._device, self._execution_config)
