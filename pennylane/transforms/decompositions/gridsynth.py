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
"""Alias transform function for the Ross-Selinger decomposition (GridSynth) for qjit."""

from pennylane.transforms.core import transform


def gridsynth_setup_inputs(
    epsilon: float = 1e-4, ppr_basis: bool = False, method: str = "deterministic"
):
    r"""Decomposes RZ and PhaseShift gates into the Clifford+T basis or the PPR basis.

    .. warning::

        This transform must be applied within a workflow compiled with :func:`pennylane.qjit`,
        as it is a frontend for Catalyst's ``gridsynth`` compilation pass.
        Consult the Catalyst documentation for more information.

    Args:
        tape (QNode): A quantum circuit.
        epsilon (float): The accuracy per rotation gate. Defaults to ``1e-4``. With
            ``"deterministic"``, each rotation has operator norm error at most ``epsilon``, and
            therefore diamond norm error at most :math:`2\epsilon`. With ``"mixed"`` and
            ``"mixed-fallback"``, the channel averaged over samples has diamond norm error at most
            :math:`2\epsilon`.
        ppr_basis (bool): If True, decompose into the PPR basis. If False, decompose into the Clifford+T basis. Defaults to ``False``.
        method (str): The synthesis method. ``"deterministic"`` (default) uses the standard
            Ross-Selinger discretization. ``"mixed"`` uses the mixed diagonal approximation
            (`arXiv:2203.10064 <https://arxiv.org/abs/2203.10064>`__ Section 3.4). It randomly
            applies either an under- or an over-rotated sequence, additionally twirled over the
            group :math:`\{I, S, Z, S^\dagger\}`. This roughly halves the T-count at the same
            accuracy. ``"mixed-fallback"`` uses the mixed fallback approximation (Section 3.5).
            It randomly applies one of two projective rotations on an ancilla qubit, followed by
            a mixed diagonal fallback if measuring the ancilla signals failure. This roughly
            halves the T-count of ``"mixed"`` again.

    .. note::

        With ``method="mixed"`` or ``method="mixed-fallback"``, the gate sequence of each
        rotation is sampled at execution time from the runtime random number generator, which is
        seeded by the ``seed`` argument of :func:`~.qjit`. The accuracy guarantee holds for the
        channel averaged over these samples, while each sampled sequence has an error of order
        :math:`\sqrt{\epsilon}`. By default, and with ``mcm_method="single-branch-statistics"``,
        a call samples one sequence per rotation, shared by all its shots. With
        ``mcm_method="one-shot"``, every shot samples its own.

        With ``method="mixed-fallback"``, each rotation uses a temporary ancilla qubit and a
        mid-circuit measurement.

        Resource estimates from :func:`~.specs` report the expected gate counts, since the actual
        sequences are only known at execution time. With ``method="mixed-fallback"``, they also
        report one additional wire per rotation for its temporary ancilla.

    **Example**

    .. code-block:: python

        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit(x):
            qp.Hadamard(0)
            qp.RZ(x, 0)
            qp.PhaseShift(x * 0.2, 0)
            return qp.state()

        gridsynth_circuit = qp.transforms.gridsynth(circuit, epsilon=1e-4)
        qjitted_circuit = qp.qjit(gridsynth_circuit)

    >>> circuit(1.1)
    array([0.60282587-0.36959568j, 0.5076395 +0.49224195j])
    >>> qjitted_circuit(1.1)
    Array([0.60282699-0.36962758j, 0.50758458+0.49227326j], dtype=complex128)

    Mixed synthesis reaches the same accuracy with roughly half the T gates, and mixed fallback
    synthesis with roughly half of that:

    .. code-block:: python

        def t_count(method):
            @qp.qjit(capture=True, target="mlir")
            @qp.transforms.gridsynth(epsilon=1e-6, method=method)
            @qp.qnode(qp.device("null.qubit", wires=1))
            def circuit(x: float):
                qp.RZ(x, 0)
                return qp.expval(qp.Z(0))

            return qp.specs(circuit, level="user")(1.1).resources.quantum_operations["T"]

    >>> t_count("deterministic"), t_count("mixed"), t_count("mixed-fallback")
    (63, 31, 16)
    """
    if not isinstance(ppr_basis, bool):
        raise ValueError(f"ppr_basis must be of type bool. Got {ppr_basis}")
    if not isinstance(epsilon, float):
        raise ValueError(f"epsilon must be of type float. Got {epsilon}.")
    if method not in ("deterministic", "mixed", "mixed-fallback"):
        raise ValueError(
            f"method must be 'deterministic', 'mixed' or 'mixed-fallback'. Got {method!r}."
        )
    return (), {"epsilon": epsilon, "ppr_basis": ppr_basis, "method": method}


gridsynth = transform(pass_name="gridsynth", setup_inputs=gridsynth_setup_inputs)
