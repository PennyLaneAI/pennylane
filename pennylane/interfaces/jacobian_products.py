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
from typing import Tuple, Callable, Optional

import pennylane as qml
from pennylane.tape import QuantumScript
from pennylane.typing import ResultBatch, TensorLike

Batch = Tuple[QuantumScript]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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
