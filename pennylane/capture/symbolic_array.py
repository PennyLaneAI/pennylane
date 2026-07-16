# Copyright 2026 Xanadu Quantum Technologies Inc.
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
A tool for capturing dummy arrays that can be used for resource estimation.
"""

from functools import lru_cache
from importlib.util import find_spec

from .switches import enabled

has_jax = find_spec("jax") is not None


@lru_cache
def _symbolic_array_primitive():
    if not has_jax:
        raise ImportError("jax is required for creating a jax primitive.")  # pragma: no cover

    import jax  # pylint: disable=import-outside-toplevel

    import pennylane  # pylint: disable=import-outside-toplevel

    estimation_p = pennylane.capture.custom_primitives.QpPrimitive("symbolic_array")

    @estimation_p.def_abstract_eval
    def _estimation_p_abstract_eval(shape, dtype):
        return jax.core.ShapedArray(shape, dtype)

    @estimation_p.def_impl
    def _estimation_p_impl(shape, dtype):
        raise NotImplementedError(
            "symbolic_arrays can only be produced for abstract evaluation and cannot be executed."
        )

    return estimation_p


def symbolic_array(shape: tuple[int, ...], dtype: type):
    """**EXPERIMENTAL** Creates a dummy array that can be used in resource dry-runs with catalyst.

    Args:
        shape (tuple[int,...]): the shape of the array
        dtype (type): the data type of the array.

    Returns:
        A jax tracer with the specified shape and dtype.

    .. warning::

        This function is **EXPERIMENTAL**.

        This function can only be used with ``qjit`` and with capture enabled.

        The result of this function is not concretely defined. It can be used for inspecting intermediate values (e.g. via :func:`~.specs`), but not for execution.

    .. code-block:: python

        @qp.qjit(capture=True, target="mlir")
        @qp.transforms.merge_rotations
        @qp.qnode(qp.device('null.qubit', wires=2))
        def c():
            x = qp.capture.symbolic_array((), float)
            y = 2*x + 1
            qp.RX(x, 0)
            qp.RX(y, 0)
            return qp.state()

    Even though we do not have actual values for ``x`` and ``y``, we can still see
    the effect of the ``merge_rotations`` pass on the resources.

    >>> qp.specs(c, level=0)().resources.quantum_operations # doctest: +SKIP
    {'RX': 2}
    >>> qp.specs(c, level=1)().resources.quantum_operations # doctest: +SKIP
    {'RX': 1}

    Trying to execute or calculate specs at ``level="device"`` will result in errors.

    >>> c() # doctest: +SKIP
    CompileError: catalyst failed with error code 1: Failed to run pipeline: BufferizationStage
    ...

    """
    if not enabled():
        raise NotImplementedError("symbolic_array requires program capture to be enabled.")
    from jax.numpy import dtype as jnp_dtype  # pylint: disable=import-outside-toplevel

    if not all(isinstance(s, int) and s > 0 for s in shape):
        raise ValueError(
            f"All shape dimensions must be integers greater than zero. Got shape {shape}."
        )
    return _symbolic_array_primitive().bind(shape=shape, dtype=jnp_dtype(dtype))
