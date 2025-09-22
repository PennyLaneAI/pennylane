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
Contains the general execute function, for executing tapes on devices with auto-
differentiation support.
"""
from __future__ import annotations

import inspect
import logging
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from cachetools import Cache

import pennylane as qml
from pennylane.exceptions import _TF_DEPRECATION_MSG, PennyLaneDeprecationWarning
from pennylane.math.interface_utils import Interface
from pennylane.transforms.core import TransformProgram

from ._setup_transform_program import _setup_transform_program
from .resolution import _resolve_execution_config, _resolve_interface
from .run import run

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


if TYPE_CHECKING:
    from pennylane.concurrency.executors import ExecBackends
    from pennylane.tape import QuantumScriptBatch
    from pennylane.transforms.core import TransformDispatcher
    from pennylane.typing import ResultBatch
    from pennylane.workflow.qnode import SupportedDeviceAPIs
    from pennylane.workflow.resolution import SupportedDiffMethods


# pylint: disable=too-many-arguments
def execute(
    tapes: QuantumScriptBatch,
    device: SupportedDeviceAPIs,
    diff_method: Callable | SupportedDiffMethods | TransformDispatcher | None = None,
    interface: Interface | str | None = Interface.AUTO,
    *,
    grad_on_execution: bool | Literal["best"] = "best",
    cache: bool | dict | Cache | Literal["auto"] | None = "auto",
    cachesize: int = 10000,
    max_diff: int = 1,
    device_vjp: bool | None = False,
    postselect_mode: Literal["hw-like", "fill-shots"] | None = None,
    mcm_method: Literal["deferred", "one-shot", "tree-traversal"] | None = None,
    gradient_kwargs: dict | None = None,
    transform_program: TransformProgram | None = None,
    executor_backend: ExecBackends | str | None = None,
) -> ResultBatch:
    """A function for executing a batch of tapes on a device with compatibility for auto-differentiation.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        device (pennylane.devices.LegacyDevice): Device to use to execute the batch of tapes.
            If the device does not provide a ``batch_execute`` method,
            by default the tapes will be executed in serial.
        diff_method (Optional[str | TransformDispatcher]): The gradient transform function to use
            for backward passes. If "device", the device will be queried directly
            for the gradient (if supported).
        interface (str, Interface): The interface that will be used for classical auto-differentiation.
            This affects the types of parameters that can exist on the input tapes.
            Available options include ``autograd``, ``torch``, ``tf``, ``jax``, and ``auto``.
        transform_program(.TransformProgram): A transform program to be applied to the initial tape.
        grad_on_execution (bool, str): Whether the gradients should be computed
            on the execution or not. It only applies
            if the device is queried for the gradient; gradient transform
            functions available in ``qml.gradients`` are only supported on the backward
            pass. The 'best' option chooses automatically between the two options and is default.
        cache="auto" (str or bool or dict or Cache): Whether to cache evalulations.
            ``"auto"`` indicates to cache only when ``max_diff > 1``. This can result in
            a reduction in quantum evaluations during higher order gradient computations.
            If ``True``, a cache with corresponding ``cachesize`` is created for each batch
            execution. If ``False``, no caching is used. You may also pass your own cache
            to be used; this can be any object that implements the special methods
            ``__getitem__()``, ``__setitem__()``, and ``__delitem__()``, such as a dictionary.
        cachesize (int): the size of the cache.
        max_diff (int): If ``diff_method`` is a gradient transform, this option specifies
            the maximum number of derivatives to support. Increasing this value allows
            for higher-order derivatives to be extracted, at the cost of additional
            (classical) computational overhead during the backward pass.
        device_vjp=False (Optional[bool]): whether or not to use the device-provided Jacobian
            product if it is available.
        postselect_mode (Optional[str]): Configuration for handling shots with mid-circuit measurement
            postselection. Use ``"hw-like"`` to discard invalid shots and ``"fill-shots"`` to
            keep the same number of shots. Default is ``None``.
        mcm_method (Optional[str]): Strategy to use when executing circuits with mid-circuit measurements.
            ``"deferred"`` is ignored. If mid-circuit measurements are found in the circuit,
            the device will use ``"tree-traversal"`` if specified and the ``"one-shot"`` method
            otherwise. For usage details, please refer to the
            :doc:`dynamic quantum circuits page </introduction/dynamic_quantum_circuits>`.
        gradient_kwargs (Optional[dict]): dictionary of keyword arguments to pass when
            determining the gradients of tapes.
        executor_backend (Optional[str | ExecBackends]): concurrent task-based executor for function dispatch.
            If supported by a device, the configured executor provides an abstraction for task-based function execution, which can provide speed-ups for computationally demanding execution. Defaults to ``None``.


    Returns:
        list[tensor_like[float]]: A nested list of tape results. Each element in
        the returned list corresponds in order to the provided tapes.

    **Example**

    Consider the following cost function:

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=2)

        def cost_fn(params, x):
            ops1 = [qml.RX(params[0], wires=0), qml.RY(params[1], wires=0)]
            measurements1 = [qml.expval(qml.Z(0))]
            tape1 = qml.tape.QuantumTape(ops1, measurements1)

            ops2 = [
                qml.RX(params[2], wires=0),
                qml.RY(x[0], wires=1),
                qml.CNOT(wires=(0,1))
            ]
            measurements2 = [qml.probs(wires=0)]
            tape2 = qml.tape.QuantumTape(ops2, measurements2)

            tapes = [tape1, tape2]

            # execute both tapes in a batch on the given device
            res = qml.execute(tapes, dev, diff_method=qml.gradients.param_shift, max_diff=2)

            return res[0] + res[1][0] - res[1][1]

    In this cost function, two **independent** quantum tapes are being
    constructed; one returning an expectation value, the other probabilities.
    We then batch execute the two tapes, and reduce the results to obtain
    a scalar.

    Let's execute this cost function while tracking the gradient:

    >>> params = np.array([0.1, 0.2, 0.3], requires_grad=True)
    >>> x = np.array([0.5], requires_grad=True)
    >>> cost_fn(params, x)
    1.93050682

    Since the ``execute`` function is differentiable, we can
    also compute the gradient:

    >>> qml.grad(cost_fn)(params, x)
    (array([-0.0978434 , -0.19767681, -0.29552021]), array([5.37764278e-17]))

    Finally, we can also compute any nth-order derivative. Let's compute the Jacobian
    of the gradient (that is, the Hessian):

    >>> x.requires_grad = False
    >>> qml.jacobian(qml.grad(cost_fn))(params, x)
    array([[-0.97517033,  0.01983384,  0.        ],
           [ 0.01983384, -0.97517033,  0.        ],
           [ 0.        ,  0.        , -0.95533649]])
    """
    if not isinstance(device, qml.devices.Device):
        device = qml.devices.LegacyDeviceFacade(device)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            (
                """Entry with args=(tapes=%s, device=%s, diff_method=%s, interface=%s, """
                """grad_on_execution=%s, gradient_kwargs=%s, cache=%s, cachesize=%s,"""
                """ max_diff=%s) called by=%s"""
            ),
            tapes,
            repr(device),
            (
                diff_method
                if not (logger.isEnabledFor(qml.logging.TRACE) and inspect.isfunction(diff_method))
                else "\n" + inspect.getsource(diff_method) + "\n"
            ),
            interface,
            grad_on_execution,
            gradient_kwargs,
            cache,
            cachesize,
            max_diff,
            "::L".join(str(i) for i in inspect.getouterframes(inspect.currentframe(), 2)[1][1:3]),
        )

    if not tapes:
        return ()

    ### Apply the user transforms ####
    transform_program = transform_program or TransformProgram()
    tapes, user_post_processing = transform_program(tapes)
    if transform_program.is_informative:
        return user_post_processing(tapes)

    if not tapes:
        return user_post_processing(())

    ### Specifying and preprocessing variables ###

    interface = _resolve_interface(interface, tapes)

    if interface in {Interface.TF, Interface.TF_AUTOGRAPH}:  # pragma: no cover
        warnings.warn(_TF_DEPRECATION_MSG, PennyLaneDeprecationWarning, stacklevel=4)

    config = qml.devices.ExecutionConfig(
        interface=interface,
        gradient_method=diff_method,
        grad_on_execution=None if grad_on_execution == "best" else grad_on_execution,
        use_device_jacobian_product=device_vjp,
        mcm_config=qml.devices.MCMConfig(postselect_mode=postselect_mode, mcm_method=mcm_method),
        gradient_keyword_arguments=gradient_kwargs or {},
        derivative_order=max_diff,
        executor_backend=executor_backend,
    )
    config = _resolve_execution_config(config, device, tapes)

    outer_transform, inner_transform = _setup_transform_program(device, config, cache, cachesize)

    #### Executing the configured setup #####
    tapes, outer_post_processing = outer_transform(tapes)

    assert not outer_transform.is_informative, "should only contain device preprocessing"

    results = run(tapes, device, config, inner_transform)
    return user_post_processing(outer_post_processing(results))
