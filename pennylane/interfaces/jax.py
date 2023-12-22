# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
This module contains functions for binding JVP's or VJP's to the JAX interface.

See JAX documentation on this process `here <https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html>`_ .

**Basic examples:**

.. code-block:: python

    def f(x):
        return x**2

    def f_and_jvp(primals, tangents):
        x = primals[0]
        dx = tangents[0]
        print("in custom jvp function: ", x, dx)
        return x**2, 2*x*dx

    registered_f_jvp = jax.custom_jvp(f)

    registered_f_jvp.defjvp(f_and_jvp)

>>> jax.grad(registered_f)(jax.numpy.array(2.0))
in custom jvp function:  2.0 Traced<ShapedArray(float64[], weak_type=True):JaxprTrace(level=1/0)>
Array(4., dtype=float64, weak_type=True)


We can do something similar for the VJP as well:

.. code-block:: python

    def f_fwd(x):
        print("in forward pass: ", x)
        return f(x), x

    def f_bwd(residual, dy):
        print("in backward pass: ", residual, dy)
        return (dy*2*residual,)

    registered_f_vjp = jax.custom_vjp(f)
    registered_f_vjp.defvjp(f_fwd, f_bwd)

>>> jax.grad(registered_f_vjp)(jax.numpy.array(2.0))
in forward pass:  2.0
in backward pass:  2.0 1.0
Array(4., dtype=float64, weak_type=True)

**JVP versus VJP:**

When JAX can trace the product between the Jacobian and the cotangents, it can turn the JVP calculation into a VJP calculation. Through this
process, JAX can support both JVP and VJP calculations by registering only the JVP.

Unfortunately, :meth:`~pennylane.devices.Device.compute_jvp` uses pure numpy to perform the Jacobian product and cannot
be traced by JAX.

For example, if we replace the definition of ``f_and_jvp`` from above with one that breaks tracing,

.. code-block:: python

    def f_and_jvp(primals, tangents):
        x = primals[0]
        dx = qml.math.unwrap(tangents[0]) # This line breaks tracing
        return x**2, 2*x*dx

>>> jax.grad(registered_f_jvp)(jax.numpy.array(2.0))
ValueError: Converting a JAX array to a NumPy array not supported when using the JAX JIT.

Note that the comment about ``JIT`` is generally a comment about not being able to trace code.

But if we used the VJP instead:

.. code-block:: python

    def f_bwd(residual, dy):
        dy = qml.math.unwrap(dy)
        return (dy*2*residual,)

We would be able to calculate the gradient without error.

Since the VJP calculation offers access to ``jax.grad`` and ``jax.jacobian``, we register the VJP when we have to choose
between either the VJP or the JVP.

**Pytrees:**

The trainable arguments for the registered functions can be any valid pytree.

.. code-block:: python

    def f(x):
        return x['a']**2

    def f_and_jvp(primals, tangents):
        x = primals[0]
        dx = tangents[0]
        print("in custom jvp function: ", x, dx)
        return x['a']**2, 2*x['a']*dx['a']

    registered_f_jvp = jax.custom_jvp(f)

    registered_f_jvp.defjvp(f_and_jvp)

>>> jax.grad(registered_f_jvp)({'a': jax.numpy.array(2.0)})
in custom jvp function:  {'a': Array(2., dtype=float64, weak_type=True)} {'a': Traced<ShapedArray(float64[], weak_type=True):JaxprTrace(level=1/0)>}
{'a': Array(4., dtype=float64, weak_type=True)}

As we can see here, the tangents are packed into the same pytree structure as the trainable arguments.

We use the fact that ``QuantumTape``, ``Opereator``, and `MeasurementProcess`` are all valid pytrees 

"""
# pylint: disable=unused-argument
import logging
from typing import Tuple, Callable

import jax
import jax.numpy as jnp

import pennylane as qml
from pennylane.typing import ResultBatch

dtype = jnp.float64

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


Batch = Tuple[qml.tape.QuantumTape]
ExecuteFn = Callable[[Batch], qml.typing.ResultBatch]


def get_jax_interface_name(tapes):
    """Check all parameters in each tape and output the name of the suitable
    JAX interface.

    This function checks each tape and determines if any of the gate parameters
    was transformed by a JAX transform such as ``jax.jit``. If so, it outputs
    the name of the JAX interface with jit support.

    Note that determining if jit support should be turned on is done by
    checking if parameters are abstract. Parameters can be abstract not just
    for ``jax.jit``, but for other JAX transforms (vmap, pmap, etc.) too. The
    reason is that JAX doesn't have a public API for checking whether or not
    the execution is within the jit transform.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute

    Returns:
        str: name of JAX interface that fits the tape parameters, "jax" or
        "jax-jit"
    """
    for t in tapes:
        for op in t:
            # Unwrap the observable from a MeasurementProcess
            op = op.obs if hasattr(op, "obs") else op
            if op is not None:
                # Some MeasurementProcess objects have op.obs=None
                for param in op.data:
                    if qml.math.is_abstract(param):
                        return "jax-jit"

    return "jax"


def _to_jax(result: qml.typing.ResultBatch) -> qml.typing.ResultBatch:
    """Converts an arbitrary result batch to one with jax arrays.
    Args:
        result (ResultBatch): a nested structure of lists, tuples, dicts, and numpy arrays
    Returns:
        ResultBatch: a nested structure of tuples, dicts, and jax arrays
    """
    if isinstance(result, dict):
        return result
    if isinstance(result, (list, tuple)):
        return tuple(_to_jax(r) for r in result)
    return jnp.array(result)


def _execute_wrapper(tapes, execute_fn, jpc, device) -> ResultBatch:
    """Executes ``tapes`` with ``params`` via ``execute_fn``"""
    return _to_jax(execute_fn(tapes))


def _execute_and_compute_jvp(execute_fn, jpc, device, primals, tangents):
    """Compute the results and jvps for ``tapes`` with ``primals[0]`` parameters via
    ``jpc``.
    """
    tangents = tuple(t.get_parameters() for t in tangents[0])
    res, jvps = jpc.execute_and_compute_jvp(primals[0], tangents)
    return _to_jax(res), _to_jax(jvps)


def _vjp_fwd(tapes, execute_fn, jpc, device):
    """Perform the forward pass execution, return results and empty residuals."""
    return _to_jax(execute_fn(tapes)), tapes


def _vjp_bwd(execute_fn, jpc, device, tapes, dy):
    """Perform the backward pass of a vjp calculation, returning the vjp."""
    vjps = _to_jax(jpc.compute_vjp(tapes, dy))
    bound_vjps = tuple(
        t.bind_new_parameters(vjp, t.trainable_params) for vjp, t in zip(vjps, tapes)
    )
    return (tuple(bound_vjps),)


jax_jvp_execute = jax.custom_jvp(_execute_wrapper, nondiff_argnums=[1, 2, 3])
jax_jvp_execute.defjvp(_execute_and_compute_jvp)

jax_vjp_execute = jax.custom_vjp(_execute_wrapper, nondiff_argnums=[1, 2, 3])
jax_vjp_execute.defvjp(_vjp_fwd, _vjp_bwd)
