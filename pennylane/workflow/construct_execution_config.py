# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License rif is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains a function to construct an execution configuration from a QNode instance."""
import functools

import pennylane as qml
from pennylane.math import Interface
from pennylane.workflow import construct_tape
from pennylane.workflow.resolution import _resolve_execution_config


def construct_execution_config(qnode: "qml.QNode", resolve: bool = True):
    """Constructs the execution configuration of a QNode instance.

    Args:
        qnode (QNode): the qnode we want to get execution configuration for
        resolve (bool): Whether or not to validate and fill in undetermined values like `"best"`. Defaults to ``True``.

    Returns:
        config (qml.devices.ExecutionConfig): the execution configuration

    **Example**

    .. code-block:: python

        @qml.qnode(qml.device("default.qubit", wires=1))
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

    First, let's import ``pprint`` to make it easier to read the execution configuration objects.

    >>> from pprint import pprint

    If we wish to construct an unresolved execution configuration, we can specify
    ``resolve=False``. This will leave properties like ``gradient_method`` and ``interface``
    in their unrefined state (e.g. ``"best"`` or ``"auto"`` respectively).

    >>> config = qml.workflow.construct_execution_config(circuit, resolve=False)(1)
    >>> pprint(config)
    ExecutionConfig(grad_on_execution=None,
                    use_device_gradient=None,
                    use_device_jacobian_product=False,
                    gradient_method='best',
                    gradient_keyword_arguments={},
                    device_options={},
                    interface=<Interface.AUTO: 'auto'>,
                    derivative_order=1,
                    mcm_config=MCMConfig(mcm_method=None, postselect_mode=None),
                    convert_to_numpy=True)

    Specifying ``resolve=True`` will then resolve these properties appropriately for the
    given ``QNode`` configuration that was provided,

    >>> resolved_config = qml.workflow.construct_execution_config(circuit, resolve=True)(1)
    >>> pprint(resolved_config)
    ExecutionConfig(grad_on_execution=False,
                    use_device_gradient=True,
                    use_device_jacobian_product=False,
                    gradient_method='backprop',
                    gradient_keyword_arguments={},
                    device_options={'max_workers': None,
                                    'prng_key': None,
                                    'rng': Generator(PCG64) at 0x15F6BB680},
                    interface=<Interface.NUMPY: 'numpy'>,
                    derivative_order=1,
                    mcm_config=MCMConfig(mcm_method=None, postselect_mode=None),
                        convert_to_numpy=True)
    """

    @functools.wraps(qnode)
    def wrapper(*args, **kwargs):
        mcm_config = qml.devices.MCMConfig(
            postselect_mode=qnode.execute_kwargs["postselect_mode"],
            mcm_method=qnode.execute_kwargs["mcm_method"],
        )

        grad_on_execution = qnode.execute_kwargs["grad_on_execution"]
        if qnode.interface in {Interface.JAX.value, Interface.JAX_JIT.value}:
            grad_on_execution = False
        elif grad_on_execution == "best":
            grad_on_execution = None

        config = qml.devices.ExecutionConfig(
            interface=qnode.interface,
            gradient_method=qnode.diff_method,
            grad_on_execution=grad_on_execution,
            use_device_jacobian_product=qnode.execute_kwargs["device_vjp"],
            derivative_order=qnode.execute_kwargs["max_diff"],
            gradient_keyword_arguments=qnode.gradient_kwargs,
            mcm_config=mcm_config,
        )

        if resolve:
            tape = construct_tape(qnode, level=0)(*args, **kwargs)
            # pylint:disable=protected-access
            config = _resolve_execution_config(
                config, qnode.device, (tape,), qnode._transform_program
            )

        return config

    return wrapper
