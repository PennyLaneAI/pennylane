# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Aliases for pauli-based computation passes from Catalyst's passes module."""

from functools import partial

from pennylane.transforms.core import transform


@partial(transform, pass_name="to-ppr")
def to_ppr(tape):
    r"""A quantum compilation pass that converts Clifford+T gates into Pauli Product Rotation (PPR)
    gates.

    .. note::

        This transform requires decorating the QNode with :func:`@qml.qjit <pennylane.qjit>` and for
        program capture to be enabled via :func:`qml.capture.enable() <pennylane.capture.enable>`.

    Clifford gates are defined as :math:`\exp(-{iP\tfrac{\pi}{4}})`, where :math:`P` is a Pauli word.
    Non-Clifford gates are defined as :math:`\exp(-{iP\tfrac{\pi}{8}})`.

    For more information on the PPM compilation pass, check out the
    `compilation hub <https://pennylane.ai/compilation/pauli-product-measurement>`__.

    .. note::

        The circuits that generated from this pass are currently not executable on any backend.
        This pass is only for analysis with the ``null.qubit`` device and potential future execution
        when a suitable backend is available.

    The full list of supported gates and operations are
    ``qml.H``,
    ``qml.S``,
    ``qml.T``,
    ``qml.X``,
    ``qml.Y``,
    ``qml.Z``,
    ``qml.adjoint(qml.S)``,
    ``qml.adjoint(qml.T)``,
    ``qml.CNOT``, and
    ``catalyst.measure``.

    Args:
        fn (QNode): QNode to apply the pass to

    Returns:
        :class:`QNode <pennylane.QNode>`

    **Example**

    The ``to_ppr`` compilation pass can be applied as a dectorator on a QNode:

    .. code-block:: python

        import pennylane as qml

        qml.capture.enable()

        @qml.qjit(target="mlir")
        @qml.transforms.to_ppr
        @qml.qnode(qml.device("null.qubit", wires=2))
        def circuit():
            qml.H(0)
            qml.CNOT([0, 1])
            qml.T(0)
            return qml.expval(qml.Z(0))

    For clear and inspectable results, use ``target="mlir"`` in the ``qjit`` decorator, ensure that
    PennyLane's program capture is enabled, :func:`pennylane.capture.enable`, and call ``to_ppr``
    from the PennyLane frontend (``qml.transforms.to_ppr``) instead of with
    ``catalyst.passes.to_ppr``.

    >>> print(qml.specs(circuit, level="all")()['resources'])
    {
        'No transforms': ...,
        'Before MLIR Passes (MLIR-0)': ...,
        'to-ppr (MLIR-1)': Resources(
            num_wires=2,
            num_gates=7,
            gate_types=defaultdict(<class 'int'>, {'PPR-pi/4-w1': 5, 'PPR-pi/4-w2': 1, 'PPR-pi/8-w1': 1}),
            gate_sizes=defaultdict(<class 'int'>, {1: 6, 2: 1}),
            depth=None,
            shots=Shots(total_shots=None, shot_vector=())
        )
    }

    In the above output, ``PPR-theta-weight`` denotes the type of PPR present in the circuit, where
    ``theta`` is the PPR angle (:math:`\theta`) and ``weight`` is the PPR weight.

    """

    raise NotImplementedError(
        "The to_ppr compilation pass has no tape implementation, and can only be applied when decorating the entire worfklow with @qml.qjit and when it is placed after all transforms that only have a tape implementation."
    )


@partial(transform, pass_name="commute_ppr")
def commute_ppr(tape, *, max_pauli_size=0):
    r"""A quantum compilation pass that commutes Clifford Pauli product rotation (PPR) gates,
    :math:`\exp(-{iP\tfrac{\pi}{4}})`, past non-Clifford PPRs gates,
    :math:`\exp(-{iP\tfrac{\pi}{8}})`, where :math:`P` is a Pauli word.

    .. note::

        This transform requires decorating the QNode with :func:`@qml.qjit <pennylane.qjit>` and for
        program capture to be enabled via :func:`qml.capture.enable() <pennylane.capture.enable>`.

    For more information on PPRs, check out the
    `Compilation Hub <https://pennylane.ai/compilation/pauli-product-measurement>`_.

    .. note::

        The circuits that generated from this pass are currently not executable on any backend.
        This pass is only for analysis with the ``null.qubit`` device and potential future execution
        when a suitable backend is available.

    Args:
        fn (QNode): QNode to apply the pass to.
        max_pauli_size (int):
            The maximum size of Pauli strings resulting from commutation. If a commutation results
            in a PPR that acts on more than ``max_pauli_size`` qubits, that commutation will not be
            performed.

    Returns:
        :class:`QNode <pennylane.QNode>`

    **Example**

    The ``commute_ppr`` compilation pass can be applied as a dectorator on a QNode:

    .. code-block:: python

        import pennylane as qml
        from functools import partial
        import jax.numpy as jnp

        qml.capture.enable()

        @qml.qjit(target="mlir")
        @partial(qml.transforms.commute_ppr, max_pauli_size=2)
        @qml.qnode(qml.device("null.qubit", wires=2))
        def circuit():

            # equivalent to a Hadamard gate
            qml.PauliRot(jnp.pi / 2, pauli_word="Z", wires=0)
            qml.PauliRot(jnp.pi / 2, pauli_word="X", wires=0)
            qml.PauliRot(jnp.pi / 2, pauli_word="Z", wires=0)

            # equivalent to a CNOT gate
            qml.PauliRot(jnp.pi / 2, pauli_word="ZX", wires=[0, 1])
            qml.PauliRot(-jnp.pi / 2, pauli_word="Z", wires=0)
            qml.PauliRot(-jnp.pi / 2, pauli_word="X", wires=1)

            # equivalent to a T gate
            qml.PauliRot(jnp.pi / 4, pauli_word="Z", wires=0)

            return qml.expval(qml.Z(0))

    For clear and inspectable results, use ``target="mlir"`` in the ``qjit`` decorator, ensure that
    PennyLane's program capture is enabled, :func:`pennylane.capture.enable`, and call
    ``commute_ppr`` from the PennyLane frontend (``qml.transforms.commute_ppr``) instead of with
    ``catalyst.passes.commute_ppr``.

    >>> print(qml.specs(circuit, level="all")()['resources'])
    {
        'No transforms': ...,
        'Before MLIR Passes (MLIR-0)': ...,
        'commute-ppr (MLIR-1)': Resources(
            num_wires=2,
            num_gates=7,
            gate_types=defaultdict(<class 'int'>, {'PPR-pi/8-w1': 1, 'PPR-pi/4-w1': 5, 'PPR-pi/4-w2': 1}),
            gate_sizes=defaultdict(<class 'int'>, {1: 6, 2: 1}),
            depth=None,
            shots=Shots(total_shots=None, shot_vector=()))
    }

    In the example above, the Clifford PPRs (``H`` and ``CNOT``) will be commuted past the
    non-Clifford PPR (``T``). In the output above, ``PPR-theta-weight`` denotes the type of PPR
    present in the circuit, where ``theta`` is the PPR angle (:math:`\theta`) and ``weight`` is the
    PPR weight.

    Note that if a commutation resulted in a PPR acting on more than ``max_pauli_size`` qubits
    (here, ``max_pauli_size = 2``), that commutation would be skipped.

    """

    raise NotImplementedError(
        "The commute_ppr compilation pass has no tape implementation, and can only be applied when decorating the entire worfklow with @qml.qjit and when it is placed after all transforms that only have a tape implementation."
    )


@partial(transform, pass_name="merge_ppr_ppm")
def merge_ppr_ppm(tape=None, *, max_pauli_size=0):
    R"""
    A quantum compilation pass that absorbs Clifford Pauli product rotation (PPR) operations,
    :math:`\exp{-iP\tfrac{\pi}{4}}`, into the final Pauli product measurements (PPMs).
    """
    raise NotImplementedError(
        "The merge_ppr_ppm compilation pass has no tape implementation, and can only be applied when decorating the entire worfklow with @qml.qjit and when it is placed after all transforms that only have a tape implementation."
    )


@partial(transform, pass_name="ppr_to_ppm")
def ppr_to_ppm(tape=None, *, decompose_method="pauli-corrected", avoid_y_measure=False):
    R"""
    A quantum compilation pass that decomposes Pauli product rotations (PPRs),
    :math:`P(\theta) = \exp(-iP\theta)`, into Pauli product measurements (PPMs).
    """
    raise NotImplementedError(
        "The ppr_to_ppm compilation pass has no tape implementation, and can only be applied when decorating the entire worfklow with @qml.qjit and when it is placed after all transforms that only have a tape implementation."
    )


@partial(transform, pass_name="ppm-compilation")
def ppm_compilation(
    tape=None, *, decompose_method="pauli-corrected", avoid_y_measure=False, max_pauli_size=0
):
    R"""
    A quantum compilation pass that transforms Clifford+T gates into Pauli product measurements
    (PPMs).
    """
    raise NotImplementedError(
        "The ppm_compilation compilation pass has no tape implementation, and can only be applied when decorating the entire worfklow with @qml.qjit and when it is placed after all transforms that only have a tape implementation."
    )


@partial(transform, pass_name="reduce-t-depth")
def reduce_t_depth(qnode):
    R"""
    A quantum compilation pass that reduces the depth and count of non-Clifford Pauli product
    rotation (PPR, :math:`P(\theta) = \exp(-iP\theta)`) operators (e.g., ``T`` gates) by commuting
    PPRs in adjacent layers and merging compatible ones (a layer comprises PPRs that mutually
    commute). For more details, see Figure 6 of
    `A Game of Surface Codes <https://arXiv:1808.02892v3>`_.
    """

    raise NotImplementedError(
        "The reduce_t_depth compilation pass has no tape implementation, and can only be applied when decorating the entire worfklow with @qml.qjit and when it is placed after all transforms that only have a tape implementation."
    )
