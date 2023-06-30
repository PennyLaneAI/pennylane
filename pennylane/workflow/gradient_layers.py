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
from typing import Tuple

from pennylane.gradients import (
    batch_jvp,
    batch_vjp,
    compute_jvp_multi,
    compute_jvp_single,
    compute_vjp_multi,
    compute_vjp_single,
)
from pennylane.tape import QuantumScript
from pennylane.transforms.core import TransformContainer
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.typing import ResultBatch, TensorLike

from .executor import Executor

Batch = Tuple[QuantumScript]


def _compute_jvps(jacs, tangents, multi_measurements):
    """Compute the jvps of multiple tapes, directly for a Jacobian and tangents."""
    jvps = []
    for i, multi in enumerate(multi_measurements):
        compute_func = compute_jvp_multi if multi else compute_jvp_single
        jvps.append(compute_func(tangents[i], jacs[i]))
    return jvps


def _compute_vjps(dys, jacs, multi_measurements, reduction_method="extend"):
    """Compute the vjps of multiple tapes, directly for a Jacobian and tangents."""
    vjps = []

    for i, multi in enumerate(multi_measurements):
        compute_func = compute_vjp_multi if multi else compute_vjp_single
        if reduction_method == "extend":
            vjps.extend(compute_func(dys[i], jacs[i]))
        else:
            vjps.append(compute_func(dys[i], jacs[i]))
    return vjps


class DerivativeExecutor(abc.ABC):
    """Provides methods for calculating the jvp and vjps for tapes and tangents/ cotangents."""

    @abc.abstractmethod
    def execute_and_compute_jvp(
        self, tapes: Batch, tangents: TensorLike, reduction_method: str = "extend"
    ) -> Tuple[ResultBatch, Tuple]:
        """Calculate both the results for a batch of tapes and the jvp.

        Args:
            tapes: The batch of tapes to take the derivatives of
            tangents (Iterable[TensorLike]): the tangents for the parameters of the tape
            reduction_method (str): Either ``"append"`` or ``"extend"``

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
    def compute_jvp(self, tapes: Batch, tangents: TensorLike, reduction_method="extend") -> Tuple:
        """Calculate both the results for a batch of tapes and the jvp.

        Args:
            tapes: The batch of tapes to take the derivatives of
            tangents (Iterable[TensorLike]): the tangents for the parameters of the tape
            reduction_method (str): Either ``"append"`` or ``"extend"``

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
    def execute_and_compute_vjp(
        self, tapes: Batch, dy: TensorLike, reduction_method: str = "extend"
    ) -> Tuple[ResultBatch, Tuple]:
        """Compute the vjp for a given batch of tapes.

        Args:
            tapes: the batch of tapes to the the derivatives of
            dy: the derivatives of the results of an execution
            reduction_method (str): Either ``"append"`` or ``"extend"``

        Returns:
            ResultBatch, TensorLike

        >>> tape0 = qml.tape.QuantumScript([qml.RX(0.1, wires=0)], [qml.expval(qml.PauliZ(0))])
        >>> batch = (tape0, )
        >>> derivatives_executor.execute_and_compute_vjp(batch, (0.5, ))
        ((array(0.99500417),), (-0.04991670832341412,))
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_vjp(self, tapes, dy, reduction_method: str = "extend") -> Tuple:
        """Compute the vjp for a given batch of tapes.

        Args:
            tapes: the batch of tapes to the the derivatives of
            dy: the derivatives of the results of an execution
            reduction_method (str): Either ``"append"`` or ``"extend"``

        Returns:
            TensorLike

        >>> tape0 = qml.tape.QuantumScript([qml.RX(0.1, wires=0)], [qml.expval(qml.PauliZ(0))])
        >>> batch = (tape0, )
        >>> derivatives_executor.compute_vjp(batch, (0.5, ))
        (-0.04991670832341412,)
        """
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
        next_executor_str = f"{self._next_executor}".replace("\n", "\n\t")
        return f"Transform Derivatives({self._gradient_transform},\n\t{next_executor_str}\n)"

    def __init__(self, next_executor: Executor, gradient_transform: TransformContainer):
        self._next_executor = next_executor
        self._gradient_transform = gradient_transform

    def execute_and_compute_jvp(self, tapes, tangents, reduction_method="extend"):
        """tangents[0] ane tangent vectors for those tapes."""

        num_result_tapes = len(tapes)

        jvp_tapes, jvp_processing_fn = batch_jvp(
            tapes, tangents, self._gradient_transform, reduction=reduction_method
        )

        full_batch = tapes + jvp_tapes

        full_results = self._next_executor(full_batch)

        results = full_results[:num_result_tapes]
        jvp_results = full_results[num_result_tapes:]
        print(jvp_results)
        jvps = jvp_processing_fn(jvp_results)

        return tuple(results), tuple(jvps)

    def compute_jvp(self, tapes, tangents, reduction_method="extend"):

        jvp_tapes, jvp_processing_fn = batch_jvp(
            tapes, tangents, self._gradient_transform, reduction=reduction_method
        )
        print(jvp_tapes)
        jvp_results = self._next_executor(jvp_tapes)
        print(jvp_results)

        return tuple(jvp_processing_fn(jvp_results))

    def execute_and_compute_vjp(
        self, tapes, dy, reduction_method="extend"
    ) -> Tuple[ResultBatch, Tuple]:
        num_result_tapes = len(tapes)

        jvp_tapes, vjp_processing_fn = batch_vjp(
            tapes, dy, self._gradient_transform, reduction=reduction_method
        )

        full_batch = tuple(tapes) + tuple(jvp_tapes)

        full_results = self._next_executor(full_batch)

        results = full_results[:num_result_tapes]
        vjp_results = full_results[num_result_tapes:]

        vjps = vjp_processing_fn(vjp_results)
        return tuple(results), tuple(vjps)

    def compute_vjp(self, tapes, dy, reduction_method="extend"):
        vjp_tapes, processing_fn = batch_vjp(
            tapes,
            dy,
            self._gradient_transform,
            reduction=reduction_method,
        )

        vjp_results = self._next_executor(vjp_tapes)

        return tuple(processing_fn(vjp_results))


class DeviceDerivatives(Executor, DerivativeExecutor):
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

    def __init__(self, device, execution_config):
        self._device = device
        self._execution_config = execution_config
        self._jacobian_cache = {}
        self._results_cache = {}

    def __call__(self, tapes):
        if tapes not in self._jacobian_cache:
            unwrapped_tapes = tuple(convert_to_numpy_parameters(t)[0][0] for t in tapes)
            results, jacs = self._device.execute_and_compute_derivatives(
                unwrapped_tapes, self._execution_config
            )
            self._results_cache[tapes] = results
            self._jacobian_cache[tapes] = jacs
        print(self._jacobian_cache)
        return self._results_cache[tapes]

    def execute_and_compute_jvp(self, tapes, tangents, reduction_method="extend"):
        tapes = tuple(convert_to_numpy_parameters(t)[0][0] for t in tapes)
        multi_measurements = [len(tape.measurements) > 1 for tape in tapes]

        if tapes not in self._jacobian_cache:
            results, jacs = self._device.execute_and_compute_derivatives(
                tapes, self._execution_config
            )
            self._jacobian_cache[tapes] = jacs
            self._results_cache[tapes] = results

        jvps = _compute_jvps(self._jacobian_cache[tapes], tangents, multi_measurements)

        return self._results_cache[tapes], jvps

    def compute_jvp(self, tapes, tangents, reduction_method="extend"):
        tapes = tuple(convert_to_numpy_parameters(t)[0][0] for t in tapes)
        if tapes not in self._jacobian_cache:
            jacs = self._device.compute_derivatives(tapes, self._execution_config)
            self._jacobian_cache[tapes] = jacs

        multi_measurements = [len(tape.measurements) > 1 for tape in tapes]
        return _compute_jvps(self._jacobian_cache[tapes], tangents, multi_measurements)

    def execute_and_compute_vjp(
        self, tapes, dy, reduction_method="extend"
    ) -> Tuple[ResultBatch, Tuple]:
        tapes = tuple(convert_to_numpy_parameters(t)[0][0] for t in tapes)
        if tapes not in self._jacobian_cache:
            results, jacs = self._device.execute_and_compute_derivatives(
                tapes, self._execution_config
            )
            self._results_cache[tapes] = results
            self._jacobian_cache[jacs] = jacs

        multi_measurements = [len(tape.measurements) > 1 for tape in tapes]
        vjps = _compute_vjps(
            dy, self._jacobian_cache[tapes], multi_measurements, reduction_method=reduction_method
        )
        return tuple(results), tuple(vjps)

    def compute_vjp(self, tapes, dy, reduction_method="extend"):

        multi_measurements = (len(t.measurements) > 1 for t in tapes)

        if tapes not in self._jacobian_cache:
            unwrapped_tapes = tuple(convert_to_numpy_parameters(t)[0][0] for t in tapes)
            jacs = self._device.compute_derivatives(unwrapped_tapes, self._execution_config)
            self._jacobian_cache[tapes] = jacs

        return _compute_vjps(
            dy, self._jacobian_cache[tapes], multi_measurements, reduction_method=reduction_method
        )
