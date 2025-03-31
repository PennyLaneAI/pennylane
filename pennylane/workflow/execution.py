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

import inspect
import logging
from typing import Callable, Literal, Optional, Union
from warnings import warn

from cachetools import Cache

import pennylane as qml
from pennylane.math import Interface, InterfaceLike
from pennylane.tape import QuantumScriptBatch
from pennylane.transforms.core import TransformDispatcher, TransformProgram
from pennylane.typing import ResultBatch
from pennylane.workflow.resolution import SupportedDiffMethods

from ._setup_transform_program import _setup_transform_program
from .resolution import _resolve_execution_config, _resolve_interface
from .run import run

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# pylint: disable=too-many-arguments
def execute(
    tapes: QuantumScriptBatch,
    device: Union["qml.devices.LegacyDevice", "qml.devices.Device"],
    diff_method: Optional[Union[Callable, SupportedDiffMethods, TransformDispatcher]] = None,
    interface: Optional[InterfaceLike] = Interface.AUTO,
    *,
    transform_program: TransformProgram = None,
    grad_on_execution: Literal[True, False, "best"] = "best",
    cache: Union[None, bool, dict, Cache] = True,
    cachesize: int = 10000,
    max_diff: int = 1,
    device_vjp: Union[bool, None] = False,
    postselect_mode: Literal[None, "hw-like", "fill-shots"] = None,
    mcm_method: Literal[None, "deferred", "one-shot", "tree-traversal"] = None,
    gradient_kwargs: dict = None,
    mcm_config="unset",
    config="unset",
    inner_transform="unset",
) -> ResultBatch:
    """A function for executing a batch of tapes on a device with compatibility for auto-differentiation.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        device (pennylane.devices.LegacyDevice): Device to use to execute the batch of tapes.
            If the device does not provide a ``batch_execute`` method,
            by default the tapes will be executed in serial.
        diff_method (None, str, TransformDispatcher): The gradient transform function to use
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
        cache (None, bool, dict, Cache): Whether to cache evaluations. This can result in
            a significant reduction in quantum evaluations during gradient computations.
        cachesize (int): the size of the cache.
        max_diff (int): If ``diff_method`` is a gradient transform, this option specifies
            the maximum number of derivatives to support. Increasing this value allows
            for higher-order derivatives to be extracted, at the cost of additional
            (classical) computational overhead during the backward pass.
        device_vjp=False (Optional[bool]): whether or not to use the device-provided Jacobian
            product if it is available.
        postselect_mode (str): Configuration for handling shots with mid-circuit measurement
            postselection. Use ``"hw-like"`` to discard invalid shots and ``"fill-shots"`` to
            keep the same number of shots. Default is ``None``.
        mcm_method (str): Strategy to use when executing circuits with mid-circuit measurements.
            ``"deferred"`` is ignored. If mid-circuit measurements are found in the circuit,
            the device will use ``"tree-traversal"`` if specified and the ``"one-shot"`` method
            otherwise. For usage details, please refer to the
            :doc:`dynamic quantum circuits page </introduction/dynamic_quantum_circuits>`.
        gradient_kwargs (dict): dictionary of keyword arguments to pass when
            determining the gradients of tapes.
        mcm_config="unset": **DEPRECATED**. This keyword argument has been replaced by ``postselect_mode``
            and ``mcm_method`` and will be removed in v0.42.
        config="unset": **DEPRECATED**. This keyword argument has been deprecated and
            will be removed in v0.42.
        inner_transform="unset": **DEPRECATED**. This keyword argument has been deprecated
            and will be removed in v0.42.

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

    if config != "unset":
        warn(
            "The config argument has been deprecated and will be removed in v0.42. "
            "The provided config argument will be ignored. "
            "If more detailed control over the execution is required, use ``qml.workflow.run`` with these arguments instead.",
            qml.PennyLaneDeprecationWarning,
        )

    if inner_transform != "unset":
        warn(
            "The inner_transform argument has been deprecated and will be removed in v0.42. "
            "The provided inner_transform argument will be ignored. "
            "If more detailed control over the execution is required, use ``qml.workflow.run`` with these arguments instead.",
            qml.PennyLaneDeprecationWarning,
        )

    if mcm_config != "unset":
        warn(
            "The mcm_config argument is deprecated and will be removed in v0.42, use mcm_method and postselect_mode instead.",
            qml.PennyLaneDeprecationWarning,
        )
        mcm_method = mcm_config.mcm_method
        postselect_mode = mcm_config.postselect_mode

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

    ### Specifying and preprocessing variables ####

    interface = _resolve_interface(interface, tapes)

    config = qml.devices.ExecutionConfig(
        interface=interface,
        gradient_method=diff_method,
        grad_on_execution=None if grad_on_execution == "best" else grad_on_execution,
        use_device_jacobian_product=device_vjp,
        mcm_config=qml.devices.MCMConfig(postselect_mode=postselect_mode, mcm_method=mcm_method),
        gradient_keyword_arguments=gradient_kwargs or {},
        derivative_order=max_diff,
    )
    config = _resolve_execution_config(config, device, tapes, transform_program=transform_program)

    transform_program = transform_program or qml.transforms.core.TransformProgram()
    transform_program, inner_transform = _setup_transform_program(
        transform_program, device, config, cache, cachesize
    )

    #### Executing the configured setup #####
    tapes, post_processing = transform_program(tapes)

    if transform_program.is_informative:
        return post_processing(tapes)

    results = run(tapes, device, config, inner_transform)
    return post_processing(results)
