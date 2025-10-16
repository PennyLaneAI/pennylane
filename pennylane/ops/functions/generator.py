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
This module contains the qml.generator function.
"""

import inspect
import warnings

import numpy as np

import pennylane as qml
from pennylane.exceptions import PennyLaneDeprecationWarning, QuantumFunctionError
from pennylane.ops import LinearCombination, Prod, SProd, Sum


def _generator_hamiltonian(gen, op):
    """Return the generator as type :class:`~ops.LinearCombination`."""

    if isinstance(gen, LinearCombination):
        return gen

    if isinstance(gen, (qml.Hermitian, qml.SparseHamiltonian)):
        if isinstance(gen, qml.Hermitian):
            mat = gen.parameters[0]

        elif isinstance(gen, qml.SparseHamiltonian):
            mat = gen.parameters[0].toarray()

        return qml.pauli_decompose(mat, wire_order=op.wires, hide_identity=True)

    if isinstance(gen, (SProd, Prod, Sum)):
        coeffs, ops = gen.terms()
        return qml.Hamiltonian(coeffs, ops)

    return qml.Hamiltonian([1.0], [gen])


# pylint: disable=no-member
def _generator_prefactor(gen):
    r"""Return the generator as ```(obs, prefactor)`` representing
    :math:`G=p \hat{O}`, where

    - prefactor :math:`p` is a float
    - observable `\hat{O}` is one of :class:`~.Hermitian`,
      :class:`~.SparseHamiltonian`, or a tensor product
      of Pauli words.
    """

    prefactor = 1.0

    gen = qml.simplify(gen) if isinstance(gen, Prod) else gen

    if isinstance(gen, LinearCombination):
        gen = qml.dot(gen.coeffs, gen.ops)  # convert to Sum

    if isinstance(gen, Prod):
        coeffs, ops = gen.terms()
        return ops[0], coeffs[0]

    if isinstance(gen, Sum):
        ops = [o.base if isinstance(o, SProd) else o for o in gen]
        coeffs = [o.scalar if isinstance(o, SProd) else 1 for o in gen]
        abs_coeffs = list(qml.math.abs(coeffs))
        if qml.math.allequal(coeffs[0], coeffs):
            # case where the Hamiltonian coefficients are all the same
            return qml.sum(*ops), coeffs[0]
        if qml.math.allequal(abs_coeffs[0], abs_coeffs):
            # absolute value of coefficients is the same
            prefactor = abs_coeffs[0]
            coeffs = [c / prefactor for c in coeffs]
            return qml.dot(coeffs, ops), prefactor

    elif isinstance(gen, SProd):
        return gen.base, gen.scalar

    return gen, prefactor


def _generator_backcompatibility(op):
    r"""Preserve backwards compatibility behaviour for PennyLane
    versions <=0.22, where generators returned List[type or ndarray, float].
    This function raises a deprecation warning, and converts to the new
    format where an instantiated Operator is returned."""
    warnings.warn(
        "The Operator.generator property is deprecated. Please update the operator so that "
        "\n\t1. Operator.generator() is a method, and"
        "\n\t2. Operator.generator() returns an Operator instance representing the operator.",
        PennyLaneDeprecationWarning,
    )
    gen = op.generator

    if inspect.isclass(gen[0]):
        return gen[1] * gen[0](wires=op.wires)

    if isinstance(gen[0], np.ndarray) and len(gen[0].shape) == 2:
        return gen[1] * qml.Hermitian(gen[0], wires=op.wires)

    raise qml.operation.GeneratorUndefinedError


def generator(op: qml.operation.Operator, format="prefactor"):
    r"""Returns the generator of an operation.

    Args:
        op (.Operator or Callable): A single operator, or a function that
            applies a single quantum operation.
        format (str): The format to return the generator in. Must be one of ``'prefactor'``,
            ``'observable'``, or ``'hamiltonian'``. See below for more details.

    Returns:
        .Operator or tuple[.Operator, float]: The returned generator, with format/type
        dependent on the ``format`` argument.

        * ``"prefactor"``: Return the generator as ``(obs, prefactor)`` (representing
          :math:`G=p \hat{O}`), where:

          - observable :math:`\hat{O}` is one of :class:`~.Hermitian`,
            :class:`~.SparseHamiltonian`, or a tensor product
            of Pauli words.
          - prefactor :math:`p` is a float.

        * ``"observable"``: Return the generator as a single observable as directly defined
          by ``op``. Returned generators may be any type of observable, including
          :class:`~.Hermitian`, :class:`~.SparseHamiltonian`, or :class:`~.ops.LinearCombination`.

        * ``"hamiltonian"``: Similar to ``"observable"``, however the returned observable
          will always be converted into :class:`~.ops.LinearCombination` regardless of how ``op``
          encodes the generator.

        * ``"arithmetic"``: Similar to ``"hamiltonian"``, however the returned observable
          will always be converted into an arithmetic operator. The returned generator may be
          any type, including:
          :class:`~.ops.op_math.SProd`, :class:`~.ops.op_math.Prod`, :class:`~.ops.op_math.Sum`, or the operator itself.

    **Example**

    Given an operation, ``qml.generator`` returns the generator representation:

    >>> op = qml.CRX(0.6, wires=[0, 1])
    >>> qml.generator(op)
    (X(1) @ Projector(array([1]), wires=[0]), np.float64(-0.5))

    It can also be used in a functional form:

    >>> qml.generator(qml.CRX)(0.6, wires=[0, 1])
    (X(1) @ Projector(array([1]), wires=[0]), np.float64(-0.5))

    By default, ``generator`` will return the generator in the format of ``(obs, prefactor)``,
    corresponding to :math:`G=p \hat{O}`, where the observable :math:`\hat{O}` will
    always be given in tensor product form, or as a dense/sparse matrix.

    By using the ``format`` argument, the returned generator representation can
    be altered:

    >>> op = qml.RX(0.2, wires=0)
    >>> qml.generator(op, format="prefactor")  # output will always be (obs, prefactor)
    (X(0), -0.5)
    >>> qml.generator(op, format="hamiltonian")  # output will be a LinearCombination
    -0.5 * X(0)
    >>> qml.generator(qml.PhaseShift(0.1, wires=0), format="observable")  # output will be a simplified obs where possible
    Projector(array([1]), wires=[0])
    >>> qml.generator(op, format="arithmetic")  # output is an instance of `SProd`
    -0.5 * X(0)
    """

    def processing_fn(*args, **kwargs):
        if callable(op):
            with qml.queuing.QueuingManager.stop_recording():
                gen_op = op(*args, **kwargs)
        else:
            gen_op = op

        if gen_op.num_params != 1:
            raise ValueError(
                f"Operation {gen_op.name} is not written in terms of a single parameter"
            )

        try:
            gen = gen_op.generator()
        except TypeError:
            # For backwards compatibility with PennyLane
            # versions <=0.22, assume gen_op.generator is a property
            gen = _generator_backcompatibility(gen_op)

        if not gen.is_hermitian:
            raise QuantumFunctionError(
                f"Generator {gen.name} of operation {gen_op.name} is not hermitian"
            )

        if format == "prefactor":
            return _generator_prefactor(gen)

        if format == "hamiltonian":
            return _generator_hamiltonian(gen, gen_op)

        if format == "arithmetic":
            h = _generator_hamiltonian(gen, gen_op)
            return qml.dot(h.coeffs, h.ops)

        if format == "observable":
            return gen

        raise ValueError(
            "format must be one of ('prefactor', 'hamiltonian', 'observable', 'arithmetic')"
        )

    if callable(op):
        return processing_fn
    return processing_fn()
