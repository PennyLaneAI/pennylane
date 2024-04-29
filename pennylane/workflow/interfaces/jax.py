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

>>> jax.grad(registered_f_jvp)(jax.numpy.array(2.0))
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

**Pytrees and Non-diff argnums:**

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

Currently, :class:`~.QuantumScript` is a valid pytree *most* of the time. Once it is a valid pytree *all* of the
time and can store tangents in place of the variables, we can use a batch of tapes as our trainable argument. Until then, the tapes
must be a non-pytree non-differenatible argument that accompanies the tree leaves.

"""
# pylint: disable=unused-argument
import logging
from typing import Tuple, Callable

import dataclasses

import jax
import jax.numpy as jnp

import pennylane as qml
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.typing import ResultBatch

dtype = jnp.float64

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


Batch = Tuple[qml.tape.QuantumTape]
ExecuteFn = Callable[[Batch], qml.typing.ResultBatch]


@dataclasses.dataclass
class _NonPytreeWrapper:
    """We aren't quite ready to switch to having tapes as pytrees as our
    differentiable argument due to:

    * Operators that aren't valid pytrees: ex. ParametrizedEvolution, ParametrizedHamiltonian, HardwareHamiltonian
    * Validation checks on initialization: see BasisStateProjector, StatePrep that does not allow the operator to store the cotangents
    * Jitting non-jax parametrized circuits.  NumPy parameters turn into abstract parameters during the pytree process.

    ``jax.custom_vjp`` forbids any non-differentiable argument to be a pytree, so we need to wrap it in a non-pytree type.

    When the above issues are fixed, we can treat ``tapes`` as the differentiable argument.

    """

    vals: Batch = None


def _set_copy_and_unwrap_tape(t, a, unwrap=True):
    """Copy a given tape with operations and set parameters"""
    tc = t.bind_new_parameters(a, t.trainable_params)
    return convert_to_numpy_parameters(tc) if unwrap else tc


def set_parameters_on_copy_and_unwrap(tapes, params, unwrap=True):
    """Copy a set of tapes with operations and set parameters"""
    return tuple(_set_copy_and_unwrap_tape(t, a, unwrap=unwrap) for t, a in zip(tapes, params))


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
            if not isinstance(op, qml.ops.Prod):
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


def _execute_wrapper(params, tapes, execute_fn, jpc) -> ResultBatch:
    """Executes ``tapes`` with ``params`` via ``execute_fn``"""
    new_tapes = set_parameters_on_copy_and_unwrap(tapes.vals, params, unwrap=False)
    return _to_jax(execute_fn(new_tapes))


def _execute_and_compute_jvp(tapes, execute_fn, jpc, primals, tangents):
    """Compute the results and jvps for ``tapes`` with ``primals[0]`` parameters via
    ``jpc``.
    """
    new_tapes = set_parameters_on_copy_and_unwrap(tapes.vals, primals[0], unwrap=False)
    res, jvps = jpc.execute_and_compute_jvp(new_tapes, tangents[0])
    return _to_jax(res), _to_jax(jvps)


_execute_jvp = jax.custom_jvp(_execute_wrapper, nondiff_argnums=[1, 2, 3])
_execute_jvp.defjvp(_execute_and_compute_jvp)


def jax_jvp_execute(tapes: Batch, execute_fn: ExecuteFn, jpc, device=None):
    """Execute a batch of tapes with JAX parameters using JVP derivatives.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        execute_fn (Callable[[Sequence[.QuantumTape]], ResultBatch]): a function that turns a batch of circuits into results
        jpc (JacobianProductCalculator): a class that can compute the Jacobian vector product (JVP)
            for the input tapes.

    Returns:
        TensorLike: A nested tuple of tape results. Each element in
        the returned tuple corresponds in order to the provided tapes.

    """
    if logger.isEnabledFor(logging.DEBUG):  # pragma: no cover
        logger.debug("Entry with (tapes=%s, execute_fn=%s, jpc=%s)", tapes, execute_fn, jpc)

    parameters = tuple(tuple(t.get_parameters()) for t in tapes)

    return _execute_jvp(parameters, _NonPytreeWrapper(tuple(tapes)), execute_fn, jpc)
