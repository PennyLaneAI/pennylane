import abc
from typing import Tuple

from .executor import Executor
from pennylane.transforms.core import TransformContainer
from pennylane.transforms import convert_to_numpy_parameters

from pennylane.typing import ResultBatch

from pennylane.gradients import (
    batch_jvp,
    batch_vjp,
    compute_jvp_multi,
    compute_jvp_single,
    compute_vjp_multi,
    compute_vjp_single,
)


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
    @abc.abstractmethod
    def execute_and_compute_jvp(
        self, tapes, tangents, reduction_method="extend"
    ) -> Tuple[ResultBatch, Tuple]:
        pass

    @abc.abstractmethod
    def compute_jvp(self, tapes, tangent_variables, reduction_method="extend") -> Tuple:
        pass

    @abc.abstractmethod
    def execute_and_compute_vjp(
        self, tapes, dy, reduction_method="extend"
    ) -> Tuple[ResultBatch, Tuple]:
        pass

    @abc.abstractmethod
    def compute_vjp(self, tapes, dy, reduction_method="extend") -> Tuple:
        pass


class TransformDerivatives(DerivativeExecutor):
    def __repr__(self):
        return f"Transform Derivatives({self._gradient_transform},\n{self._next_executor})"

    def __init__(self, next_executor: Executor, gradient_transform: TransformContainer):
        self._next_executor = next_executor
        self._gradient_transform = gradient_transform

    def execute_and_compute_jvp(self, tapes, tangent_variables, reduction_method="extend"):
        """tangents[0] ane tangent vectors for those tapes."""

        num_result_tapes = len(tapes)

        jvp_tapes, jvp_processing_fn = batch_jvp(
            tapes, tangent_variables, self._gradient_transform, reduction=reduction_method
        )

        full_batch = tapes + jvp_tapes

        full_results = self._next_executor(full_batch)

        results = full_results[:num_result_tapes]
        jvp_results = full_results[num_result_tapes:]

        jvps = jvp_processing_fn(jvp_results)

        return tuple(results), tuple(jvps)

    def compute_jvp(self, tapes, tangent_variables, reduction_method="extend"):

        jvp_tapes, jvp_processing_fn = batch_jvp(
            tapes, tangent_variables, self._gradient_transform, reduction=reduction_method
        )
        jvp_results = self._next_executor(jvp_tapes)

        return jvp_processing_fn(jvp_results)

    def execute_and_compute_vjp(
        self, tapes, dy, reduction_method="extend"
    ) -> Tuple[ResultBatch, Tuple]:
        num_result_tapes = len(tapes)

        jvp_tapes, vjp_processing_fn = batch_vjp(
            tapes, dy, self._gradient_transform, reduction=reduction_method
        )

        full_batch = tapes + jvp_tapes

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

        return processing_fn(vjp_results)


class DeviceDerivatives(DerivativeExecutor):
    def __repr__(self):
        return f"Device derivatives({self._device}, \n{self._next_executor})"

    def __init__(self, next_executor: Executor, device, execution_config):
        self._next_executor = next_executor
        self._device = device
        self._execution_config = execution_config

    def execute_and_compute_jvp(self, tapes, tangent_variables, reduction_method="extend"):

        results, jacs = self._device.execute_and_compute_derivatives(tapes, self._execution_config)
        multi_measurements = [len(tape.measurements) > 1 for tape in tapes]
        jvps = _compute_jvps(jacs, tangent_variables, multi_measurements)

        return results, jvps

    def compute_jvp(self, tapes, tangent_variables, reduction_method="extend"):
        jacs = self._device.compute_derivatives(tapes, self._execution_config)
        multi_measurements = [len(tape.measurements) > 1 for tape in tapes]
        return _compute_jvps(jacs, tangent_variables, multi_measurements)

    def execute_and_compute_vjp(
        self, tapes, dy, reduction_method="extend"
    ) -> Tuple[ResultBatch, Tuple]:
        results, jacs = self._device.execute_and_compute_derivatives(tapes, self._execution_config)
        multi_measurements = [len(tape.measurements) > 1 for tape in tapes]
        vjps = _compute_vjps(dy, jacs, multi_measurements, reduction_method=reduction_method)
        return tuple(results), tuple(vjps)

    def compute_vjp(self, tapes, dy, reduction_method="extend"):

        multi_measurements = (len(t.measurements) > 1 for t in tapes)

        unwrapped_tapes = tuple(convert_to_numpy_parameters(t)[0][0] for t in tapes)
        jacs = self._device.compute_derivatives(unwrapped_tapes, self._execution_config)
        return _compute_vjps(dy, jacs, multi_measurements, reduction_method=reduction_method)
