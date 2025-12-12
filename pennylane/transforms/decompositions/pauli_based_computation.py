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

    .. warning::

        This transform requires QJIT and capture to be enabled (via :func:`qml.capture.enable() <pennylane.capture.enable>`),
        as it is a wrapper for Catalyst's ``to_ppr`` compilation pass designed to only work with
        program capture is enabled.

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

    raise NotImplementedError(  # pragma: no cover
        "This transform pass (to_ppr) is only implemented when using program capture and QJIT. They can be activated by `qml.capture.enable()` and applying the `@qml.qjit` decorator."
    )


@partial(transform, pass_name="commute_ppr")
def commute_ppr(tape, *, max_pauli_size=0):
    r"""A quantum compilation pass that commutes Clifford Pauli product rotation (PPR) gates,
    :math:`\exp(-{iP\tfrac{\pi}{4}})`, past non-Clifford PPRs gates,
    :math:`\exp(-{iP\tfrac{\pi}{8}})`, where :math:`P` is a Pauli word.

    .. warning::

        This transform requires QJIT and capture to be enabled (via :func:`qml.capture.enable() <pennylane.capture.enable>`),
        as it is a wrapper for Catalyst's ``commute_ppr`` compilation pass designed to only work with
        program capture is enabled.

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

    raise NotImplementedError(  # pragma: no cover
        "This transform pass (commute_ppr) is only implemented when using program capture and QJIT. They can be activated by `qml.capture.enable()` and applying the `@qml.qjit` decorator."
    )


@partial(transform, pass_name="merge_ppr_ppm")
def merge_ppr_ppm(tape=None, *, max_pauli_size=0):
    R"""
    A quantum compilation pass that absorbs Clifford Pauli product rotation (PPR) operations,
    :math:`\exp{-iP\tfrac{\pi}{4}}`, into the final Pauli product measurements (PPMs).

    .. warning::

        This transform requires QJIT and capture to be enabled (via :func:`qml.capture.enable() <pennylane.capture.enable>`),
        as it is a wrapper for Catalyst's ``commute_ppr`` compilation pass designed to only work with
        program capture is enabled.

    For more information on PPRs and PPMs, check out
    the `Compilation Hub <https://pennylane.ai/compilation/pauli-product-measurement>`_.

    .. note::

        The circuits that generated from this pass are currently not executable on any backend.
        This pass is only for analysis with the ``null.qubit`` device and potential future execution
        when a suitable backend is available.

    Args:
        fn (QNode): QNode to apply the pass to
        max_pauli_size (int):
            The maximum size of Pauli strings resulting from merging. If a merge results in a PPM
            that acts on more than ``max_pauli_size`` qubits, that merge will not be performed. The
            default value is ``0`` (no limit).

    Returns:
        :class:`QNode <pennylane.QNode>`

    **Example**

    The ``merge_ppr_ppm`` compilation pass can be applied as a dectorator on a QNode:

    .. code-block:: python

        import pennylane as qml
        from functools import partial
        import jax.numpy as jnp

        qml.capture.enable()

        @qml.qjit(target="mlir")
        @partial(qml.transforms.merge_ppr_ppm, max_pauli_size=2)
        @qml.qnode(qml.device("null.qubit", wires=2))
        def circuit():
            qml.PauliRot(jnp.pi / 2, pauli_word="Z", wires=0)
            qml.PauliRot(jnp.pi / 2, pauli_word="X", wires=0)
            qml.PauliRot(jnp.pi / 2, pauli_word="Z", wires=0)

            ppm = qml.pauli_measure(pauli_word="ZX", wires=[0, 1])

            return

    In the above example, every PPR (``PauliRot``) and the PPM (``pauli_measure``) can be merged
    into one PPM that acts on two qubits. For clear and inspectable results, use ``target="mlir"``
    in the ``qjit`` decorator, ensure that PennyLane's program capture is enabled,
    :func:`pennylane.capture.enable`, and call ``ppr_to_ppm`` from the PennyLane frontend
    (``qml.transforms.merge_ppr_ppm``) instead of with ``catalyst.passes.merge_ppr_ppm``.

    >>> print(qml.specs(circuit, level="all")()['resources'])
    {
        'No transforms': ...,
        'Before MLIR Passes (MLIR-0)': ...,
        'merge-ppr-ppm (MLIR-1)': Resources(
            num_wires=2,
            num_gates=1,
            gate_types=defaultdict(<class 'int'>, {'PPM-w2': 1}),
            gate_sizes=defaultdict(<class 'int'>, {2: 1}),
            depth=None,
            shots=Shots(total_shots=None, shot_vector=())
        )
    }

    In the above output, ``PPM-weight`` denotes the type of PPM present in the circuit, where
    ``weight`` is the PPM weight.

    If a merging resulted in a PPM acting on more than ``max_pauli_size`` qubits, that merging
    operation would be skipped.
    """
    raise NotImplementedError(  # pragma: no cover
        "This transform pass (merge_ppr_ppm) is only implemented when using program capture and QJIT. They can be activated by `qml.capture.enable()` and applying the `@qml.qjit` decorator."
    )


@partial(transform, pass_name="ppr_to_ppm")
def ppr_to_ppm(tape=None, *, decompose_method="pauli-corrected", avoid_y_measure=False):
    R"""
    A quantum compilation pass that decomposes Pauli product rotations (PPRs),
    :math:`P(\theta) = \exp(-iP\theta)`, into Pauli product measurements (PPMs).

    .. warning::

        This transform requires QJIT and capture to be enabled (via :func:`qml.capture.enable() <pennylane.capture.enable>`),
        as it is a wrapper for Catalyst's ``ppr_to_ppm`` compilation pass designed to only work with
        program capture is enabled.

    This pass is used to decompose both non-Clifford and Clifford PPRs into PPMs. The non-Clifford
    PPRs (:math:`\theta = \tfrac{\pi}{8}`) are decomposed first, then Clifford PPRs
    (:math:`\theta = \tfrac{\pi}{4}`) are decomposed.

    For more information on PPRs and PPMs, check out
    the `Compilation Hub <https://pennylane.ai/compilation/pauli-product-measurement>`_.

    .. note::

        The circuits that generated from this pass are currently not executable on any backend.
        This pass is only for analysis with the ``null.qubit`` device and potential future execution
        when a suitable backend is available.

    Args:
        qnode (QNode): QNode to apply the pass to.
        decompose_method (str, optional): The method to use for decomposing non-Clifford PPRs.
            Options are ``"pauli-corrected"``, ``"auto-corrected"``, and ``"clifford-corrected"``.
            Defaults to ``"pauli-corrected"``.
            ``"pauli-corrected"`` uses a reactive measurement for correction that is based on Figure
            13 in `arXiv:2211.15465 <https://arxiv.org/pdf/2211.15465>`_.
            ``"auto-corrected"`` uses an additional measurement for correction that is based on
            Figure 7 in `A Game of Surface Codes <https://arxiv.org/abs/1808.02892>`__, and
            ``"clifford-corrected"`` uses a Clifford rotation for correction that is based on
            Figure 17(b) in `A Game of Surface Codes <https://arxiv.org/abs/1808.02892>`__.

        avoid_y_measure (bool): Rather than performing a Pauli-Y measurement for Clifford rotations
            (sometimes more costly), a :math:`Y` state (:math:`Y\vert 0 \rangle`) is used instead
            (requires :math:`Y`-state preparation). This is currently only supported when using the
            ``"clifford-corrected"`` and ``"pauli-corrected"`` decomposition method. Defaults to
            ``False``.

    Returns:
        :class:`QNode <pennylane.QNode>`

    **Example**

    The ``ppr_to_ppm`` compilation pass can be applied as a dectorator on a QNode:

    .. code-block:: python

        import pennylane as qml
        from functools import partial
        import jax.numpy as jnp

        qml.capture.enable()

        @qml.qjit(target="mlir")
        @partial(ppr_to_ppm, decompose_method="auto-corrected")
        @qml.qnode(qml.device("null.qubit", wires=2))
        def circuit():
            # equivalent to a Hadamard gate
            qml.PauliRot(jnp.pi / 2, pauli_word="Z", wires=0)
            qml.PauliRot(jnp.pi / 2, pauli_word="X", wires=0)
            qml.PauliRot(jnp.pi / 2, pauli_word="Z", wires=0)

            # equivalent to a CNOT gate
            qml.PauliRot(jnp.pi / 2, pauli_word="ZX", wires=[0, 1])
            qml.PauliRot(-jnp.pi / 2, pauli_word="Z", wires=[0])
            qml.PauliRot(-jnp.pi / 2, pauli_word="X", wires=[1])

            # equivalent to a T gate
            qml.PauliRot(jnp.pi / 4, pauli_word="Z", wires=0)

            return

    For clear and inspectable results, use ``target="mlir"`` in the ``qjit`` decorator, ensure that
    PennyLane's program capture is enabled, :func:`pennylane.capture.enable`, and call
    ``ppr_to_ppm`` from the PennyLane frontend (``qml.transforms.ppr_to_ppm``) instead of with
    ``catalyst.passes.ppr_to_ppm``.

    >>> print(qml.specs(circuit, level="all")()['resources'])
    {
        'No transforms': ...,
        'Before MLIR Passes (MLIR-0)': ...,
        'ppr-to-ppm (MLIR-1)': Resources(
            num_wires=8,
            num_gates=21,
            gate_types=defaultdict(<class 'int'>, {'PPM-w2': 6, 'PPM-w1': 7, 'PPR-pi/2-w1': 6, 'PPM-w3': 1, 'PPR-pi/2-w2': 1}),
            gate_sizes=defaultdict(<class 'int'>, {2: 7, 1: 13, 3: 1}),
            depth=None,
            shots=Shots(total_shots=None, shot_vector=())
        )
    }

    In the above output, ``PPM-weight`` denotes the type of PPM present in the circuit, where
    ``weight`` is the PPM weight. ``PPR-theta-weight`` denotes the type of PPR present in the
    circuit, where ``theta`` is the PPR angle (:math:`\theta`) and ``weight`` is the PPR weight.
    Note that :math:`\theta = \tfrac{\pi}{2}` PPRs correspond to Pauli operators:
    :math:`P(\tfrac{\pi}{2}) = \exp(-iP\tfrac{\pi}{2}) = P`. Pauli operators can be commuted to the
    end of the circuit and absorbed into terminal measurements.
    """
    raise NotImplementedError(  # pragma: no cover
        "This transform pass (ppr_to_ppm) is only implemented when using program capture and QJIT. They can be activated by `qml.capture.enable()` and applying the `@qml.qjit` decorator."
    )


@partial(transform, pass_name="ppm-compilation")
def ppm_compilation(
    tape=None, *, decompose_method="pauli-corrected", avoid_y_measure=False, max_pauli_size=0
):
    R"""
    A quantum compilation pass that transforms Clifford+T gates into Pauli product measurements
    (PPMs).

    .. warning::

        This transform requires QJIT and capture to be enabled (via :func:`qml.capture.enable() <pennylane.capture.enable>`),
        as it is a wrapper for Catalyst's ``ppm_compilation`` compilation pass designed to only work with
        program capture is enabled.

    This pass combines multiple sub-passes:

    - :func:`~.transforms.to_ppr` : Converts gates into Pauli Product Rotations (PPRs)
    - :func:`~.transforms.commute_ppr` : Commutes PPRs past non-Clifford PPRs
    - :func:`~.transforms.merge_ppr_ppm` : Merges PPRs into Pauli Product Measurements (PPMs)
    - :func:`~.transforms.ppr_to_ppm` : Decomposes PPRs into PPMs

    The ``avoid_y_measure`` and ``decompose_method`` arguments are passed to the
    :func:`~.transforms.ppr_to_ppm` pass. The ``max_pauli_size`` argument is passed to the
    :func:`~.transforms.commute_ppr` and :func:`~.transforms.merge_ppr_ppm` passes.

    For more information on PPRs and PPMs, check out
    the `Compilation Hub <https://pennylane.ai/compilation/pauli-product-measurement>`_.

    .. note::

        The circuits that generated from this pass are currently not executable on any backend.
        This pass is only for analysis with the ``null.qubit`` device and potential future execution
        when a suitable backend is available.

    Args:
        qnode (QNode, optional): QNode to apply the pass to. If ``None``, returns a decorator.
        decompose_method (str, optional): The method to use for decomposing non-Clifford PPRs.
            Options are ``"pauli-corrected"``, ``"auto-corrected"``, and ``"clifford-corrected"``.
            Defaults to ``"pauli-corrected"``.
            ``"pauli-corrected"`` uses a reactive measurement for correction that is based on Figure
            13 in `arXiv:2211.15465 <https://arxiv.org/pdf/2211.15465>`_.
            ``"auto-corrected"`` uses an additional measurement for correction that is based on
            Figure 7 in `A Game of Surface Codes <https://arxiv.org/abs/1808.02892>`__, and
            ``"clifford-corrected"`` uses a Clifford rotation for correction that is based on
            Figure 17(b) in `A Game of Surface Codes <https://arxiv.org/abs/1808.02892>`__.

        avoid_y_measure (bool): Rather than performing a Pauli-Y measurement for Clifford rotations
            (sometimes more costly), a :math:`Y` state (:math:`Y\vert 0 \rangle`) is used instead
            (requires :math:`Y`-state preparation). This is currently only supported when using the
            ``"clifford-corrected"`` and ``"pauli-corrected"`` decomposition method. Defaults to
            ``False``.

        max_pauli_size (int): The maximum size of the Pauli strings after commuting or merging.
            Defaults to 0 (no limit).

    Returns:
        :class:`QNode <pennylane.QNode>`

    **Example**

    The ``commute_ppr`` compilation pass can be applied as a dectorator on a QNode:

    .. code-block:: python

        import pennylane as qml
        from functools import partial

        qml.capture.enable()

        @qml.qjit(target="mlir")
        @partial(qml.transforms.ppm_compilation, decompose_method="clifford-corrected", max_pauli_size=2)
        @qml.qnode(qml.device("null.qubit", wires=2))
        def circuit():
            qml.H(0)
            qml.CNOT([0, 1])
            qml.T(0)
            return

    For clear and inspectable results, use ``target="mlir"`` in the ``qjit`` decorator, ensure that
    PennyLane's program capture is enabled, :func:`pennylane.capture.enable`, and call
    ``ppm_compilation`` from the PennyLane frontend (``qml.transforms.ppm_compilation``) instead of
    with ``catalyst.passes.ppm_compilation``.

    >>> print(qml.specs(circuit, level="all")()['resources'])
    {
        'No transforms': ...,
        'Before MLIR Passes (MLIR-0)': ...,
        'ppm-compilation (MLIR-1)': Resources(
            num_wires=7,
            num_gates=18,
            gate_types=defaultdict(<class 'int'>, {'PPM-w2': 5, 'PPM-w1': 6, 'PPR-pi/2-w1': 5, 'PPM-w3': 1, 'PPR-pi/2-w2': 1}),
            gate_sizes=defaultdict(<class 'int'>, {2: 6, 1: 11, 3: 1}),
            depth=None,
            shots=Shots(total_shots=None, shot_vector=())
        )
    }

    In the above output, ``PPM-weight`` denotes the type of PPM present in the circuit, where
    ``weight`` is the PPM weight. ``PPR-theta-weight`` denotes the type of PPR present in the
    circuit, where ``theta`` is the PPR angle (:math:`\theta`) and ``weight`` is the PPR weight.
    Note that :math:`\theta = \tfrac{\pi}{2}` PPRs correspond to Pauli operators:
    :math:`P(\tfrac{\pi}{2}) = \exp(-iP\tfrac{\pi}{2}) = P`. Pauli operators can be commuted to the
    end of the circuit and absorbed into terminal measurements.

    Note that if a commutation or merge resulted in a PPR or PPN acting on more than
    ``max_pauli_size`` qubits (here, ``max_pauli_size = 2``), that commutation or merge would be
    skipped.
    """
    raise NotImplementedError(  # pragma: no cover
        "This transform pass (ppm_compilation) is only implemented when using program capture and QJIT. They can be activated by `qml.capture.enable()` and applying the `@qml.qjit` decorator."
    )


@partial(transform, pass_name="reduce-t-depth")
def reduce_t_depth(qnode):
    R"""
    A quantum compilation pass that reduces the depth and count of non-Clifford Pauli product
    rotation (PPR, :math:`P(\theta) = \exp(-iP\theta)`) operators (e.g., ``T`` gates) by commuting
    PPRs in adjacent layers and merging compatible ones (a layer comprises PPRs that mutually
    commute). For more details, see Figure 6 of
    `A Game of Surface Codes <https://arXiv:1808.02892v3>`_.

    .. warning::

        This transform requires QJIT and capture to be enabled (via :func:`qml.capture.enable() <pennylane.capture.enable>`),
        as it is a wrapper for Catalyst's ``commute_ppr`` compilation pass designed to only work with
        program capture is enabled.

    .. note::

        The circuits that generated from this pass are currently not executable on any backend.
        This pass is only for analysis with the ``null.qubit`` device and potential future execution
        when a suitable backend is available.

    Args:
        qnode (QNode): QNode to apply the pass to.

    Returns:
        ~.QNode: Returns decorated QNode.

    **Example**

    In the example below, after performing the :func:`catalyst.passes.to_ppr` and
    :func:`catalyst.passes.merge_ppr_ppm` passes, the circuit contains a depth of four of
    non-Clifford PPRs. Subsequently applying the ``reduce_t_depth`` pass will move PPRs around via
    commutation, resulting in a circuit with a smaller PPR depth.

    Specifically, in the circuit below, the Pauli-:math:`X` PPR (:math:`\exp(iX\tfrac{\pi}{8})`) on
    qubit Q1 will be moved to the first layer, which results in a depth of three non-Clifford PPRs.

    .. code-block:: python

        import pennylane as qml
        from catalyst import qjit, measure
        from catalyst.passes import to_ppr, commute_ppr, reduce_t_depth, merge_ppr_ppm

        pips = [("pipe", ["quantum-compilation-stage"])]


        @qjit(pipelines=pips, target="mlir")
        @reduce_t_depth
        @merge_ppr_ppm
        @commute_ppr
        @to_ppr
        @qml.qnode(qml.device("null.qubit", wires=3))
        def circuit():
            n = 3
            for i in range(n):
                qml.H(wires=i)
                qml.S(wires=i)
                qml.CNOT(wires=[i, (i + 1) % n])
                qml.T(wires=i)
                qml.H(wires=i)
                qml.T(wires=i)

            return

        >>> print(circuit.mlir_opt)

        . . .
        %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
        %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
        // layer 1
        %3 = qec.ppr ["X"](8) %1 : !quantum.bit
        %4 = qec.ppr ["X"](8) %2 : !quantum.bit

        // layer 2
        %5 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit
        %6:2 = qec.ppr ["Y", "X"](8) %3, %4 : !quantum.bit, !quantum.bit
        %7 = qec.ppr ["X"](8) %5 : !quantum.bit
        %8:3 = qec.ppr ["X", "Y", "X"](8) %6#0, %6#1, %7:!quantum.bit, !quantum.bit, !quantum.bit

        // layer 3
        %9:3 = qec.ppr ["X", "X", "Y"](8) %8#0, %8#1, %8#2:!quantum.bit, !quantum.bit, !quantum.bit
        . . .
    """

    raise NotImplementedError(  # pragma: no cover
        "This transform pass (reduce_t_depth) is only implemented when using program capture and QJIT. They can be activated by `qml.capture.enable()` and applying the `@qml.qjit` decorator."
    )
