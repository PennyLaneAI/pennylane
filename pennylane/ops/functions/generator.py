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
# pylint: disable=protected-access
import inspect
import warnings

import numpy as np

import pennylane as qml


def _generator_observable(gen, op):
    """Return the generator as type :class:`~.Hermitian`,
    :class:`~.SparseHamiltonian`, or :class:`~.Hamiltonian`,
    as provided by the original gate.
    """
    if isinstance(gen, (qml.Hermitian, qml.SparseHamiltonian)):
        if not op.inverse:
            return gen

        param = gen.parameters[0]
        wires = gen.wires

        return gen.__class__(-param, wires=wires)

    if op.inverse:
        gen = -1.0 * gen

    return gen


def _generator_hamiltonian(gen, op):
    """Return the generator as type :class:`~.Hamiltonian`."""
    wires = op.wires

    if isinstance(gen, qml.Hamiltonian):
        H = gen

    elif isinstance(gen, (qml.Hermitian, qml.SparseHamiltonian)):

        if isinstance(gen, qml.Hermitian):
            mat = gen.parameters[0]

        elif isinstance(gen, qml.SparseHamiltonian):
            mat = gen.parameters[0].toarray()

        H = qml.pauli_decompose(mat, wire_order=wires, hide_identity=True)

    elif isinstance(gen, qml.operation.Observable):
        H = 1.0 * gen

    if op.inverse:
        H = -1.0 * H

    return H


def _generator_prefactor(gen, op):
    r"""Return the generator as ```(obs, prefactor)`` representing
    :math:`G=p \hat{O}`, where

    - prefactor :math:`p` is a float
    - observable `\hat{O}` is one of :class:`~.Hermitian`,
      :class:`~.SparseHamiltonian`, or a tensor product
      of Pauli words.
    """
    if isinstance(gen, (qml.Hermitian, qml.SparseHamiltonian)):
        obs = gen
        prefactor = 1.0

    elif isinstance(gen, qml.operation.Observable):
        # convert to a qml.Hamiltonian
        gen = 1.0 * gen

        if len(gen.ops) == 1:
            # case where the Hamiltonian is a single Pauli word
            obs = gen.ops[0]
            prefactor = gen.coeffs[0]
        else:
            obs = gen
            prefactor = 1.0

    if op.inverse:
        prefactor *= -1.0

    return obs, prefactor


def _generator_backcompatibility(op):
    r"""Preserve backwards compatibility behaviour for PennyLane
    versions <=0.22, where generators returned List[type or ndarray, float].
    This function raises a deprecation warning, and converts to the new
    format where an instantiated Operator is returned."""
    warnings.warn(
        "The Operator.generator property is deprecated. Please update the operator so that "
        "\n\t1. Operator.generator() is a method, and"
        "\n\t2. Operator.generator() returns an Operator instance representing the operator.",
        UserWarning,
    )
    gen = op.generator

    if inspect.isclass(gen[0]):
        return gen[1] * gen[0](wires=op.wires)

    if isinstance(gen[0], np.ndarray) and len(gen[0].shape) == 2:
        return gen[1] * qml.Hermitian(gen[0], wires=op.wires)

    raise qml.operation.GeneratorUndefinedError


@qml.op_transform
def generator(op, format="prefactor"):
    r"""Returns the generator of an operation.

    Args:
        op (.Operator or Callable): A single operator, or a function that
            applies a single quantum operation.
        format (str): The format to return the generator in. Must be one of ``'prefactor'``,
            ``'observable'``, or ``'hamiltonian'``. See below for more details.

    Returns:
        .Observable or tuple[.Observable, float]: The returned generator, with format/type
        dependent on the ``format`` argument.

        * ``"prefactor"``: Return the generator as ``(obs, prefactor)`` (representing
          :math:`G=p \hat{O}`), where:

          - observable :math:`\hat{O}` is one of :class:`~.Hermitian`,
            :class:`~.SparseHamiltonian`, or a tensor product
            of Pauli words.
          - prefactor :math:`p` is a float.

        * ``"observable"``: Return the generator as a single observable as directly defined
          by ``op``. Returned generators may be any type of observable, including
          :class:`~.Hermitian`, :class:`~.Tensor`,
          :class:`~.SparseHamiltonian`, or :class:`~.Hamiltonian`.

        * ``"hamiltonian"``: Similar to ``"observable"``, however the returned observable
          will always be converted into :class:`~.Hamiltonian` regardless of how ``op``
          encodes the generator.

    **Example**

    Given an operation, ``qml.generator`` returns the generator representation:

    >>> op = qml.CRX(0.6, wires=[0, 1])
    >>> qml.generator(op)
    (Projector([1], wires=[0]) @ PauliX(wires=[1]), -0.5)

    It can also be used in a functional form:

    >>> qml.generator(qml.CRX)(0.6, wires=[0, 1])
    (Projector([1], wires=[0]) @ PauliX(wires=[1]), -0.5)

    By default, ``generator`` will return the generator in the format of ``(obs, prefactor)``,
    corresponding to :math:`G=p \hat{O}`, where the observable :math:`\hat{O}` will
    always be given in tensor product form, or as a dense/sparse matrix.

    By using the ``format`` argument, the returned generator representation can
    be altered:

    >>> op = qml.RX(0.2, wires=0)
    >>> qml.generator(op, format="prefactor")  # output will always be (prefactor, obs)
    (PauliX(wires=[0]), -0.5)
    >>> qml.generator(op, format="hamiltonian")  # output will always be a Hamiltonian
    <Hamiltonian: terms=1, wires=[0]>
    >>> qml.generator(qml.PhaseShift(0.1, wires=0), format="observable")  # ouput will be a simplified obs where possible
    Projector([1], wires=[0])

    """
    if op.num_params != 1:
        raise ValueError(f"Operation {op.name} is not written in terms of a single parameter")

    try:
        gen = op.generator()
    except TypeError:
        # For backwards compatibility with PennyLane
        # versions <=0.22, assume op.generator is a property
        gen = _generator_backcompatibility(op)

    if not isinstance(gen, qml.operation.Observable):
        raise qml.QuantumFunctionError(
            f"Generator {gen.name} of operation {op.name} is not an observable"
        )

    if format == "prefactor":
        return _generator_prefactor(gen, op)

    if format == "hamiltonian":
        return _generator_hamiltonian(gen, op)

    if format == "observable":
        return _generator_observable(gen, op)

    raise ValueError("format must be one of ('prefactor', 'hamiltonian', 'observable')")
