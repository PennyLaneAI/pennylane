# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Provides transforms for mitigating quantum circuits."""
from typing import Any, Dict, Optional, Sequence, Union

from pennylane import QNode, apply
from pennylane.math import mean
from pennylane.tape import QuantumTape
from pennylane.transforms import batch_transform


# pylint: disable=too-many-arguments, protected-access
@batch_transform
def mitigate_with_zne(
    circuit: Union[QNode, QuantumTape],
    scale_factors: Sequence[float],
    folding: callable,
    extrapolate: callable,
    folding_kwargs: Optional[Dict[str, Any]] = None,
    extrapolate_kwargs: Optional[Dict[str, Any]] = None,
    reps_per_factor=1,
) -> float:
    r"""Mitigate an input circuit using zero-noise extrapolation.

    Error mitigation is a precursor to error correction and is compatible with near-term quantum
    devices. It aims to lower the impact of noise when evaluating a circuit on a quantum device by
    evaluating multiple variations of the circuit and post-processing the results into a
    noise-reduced estimate. This transform implements the zero-noise extrapolation (ZNE) method
    originally introduced by
    `Temme et al. <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.119.180509>`__ and
    `Li et al. <https://journals.aps.org/prx/abstract/10.1103/PhysRevX.7.021050>`__.

    Details on the functions passed to the ``folding`` and ``extrapolate`` arguments of this
    transform can be found in the usage details. This transform is compatible with functionality
    from the `Mitiq <https://mitiq.readthedocs.io/en/stable/>`__ package (version 0.11.0 and above),
    see the example and usage details for further information.

    .. warning::

        Calculating the gradient of mitigated circuits is not supported when using the Mitiq
        package as a backend for folding or extrapolation.

    Args:
        circuit (callable or QuantumTape): the circuit to be error-mitigated
        scale_factors (Sequence[float]): the range of noise scale factors used
        folding (callable): a function that returns a folded circuit for a specified scale factor
        extrapolate (callable): a function that returns an extrapolated result when provided a
            range of scale factors and corresponding results
        folding_kwargs (dict): optional keyword arguments passed to the ``folding`` function
        extrapolate_kwargs (dict): optional keyword arguments passed to the ``extrapolate`` function
        reps_per_factor (int): Number of circuits generated for each scale factor. Useful when the
            folding function is stochastic.

    Returns:
        float: the result of evaluating the circuit when mitigated using ZNE

    **Example:**

    We first create a noisy device using ``default.mixed`` by adding :class:`~.AmplitudeDamping` to
    each gate of circuits executed on the device using the :func:`~.transforms.insert` transform:

    .. code-block:: python3

        import pennylane as qml

        noise_strength = 0.05

        dev = qml.device("default.mixed", wires=2)
        dev = qml.transforms.insert(qml.AmplitudeDamping, noise_strength)(dev)

    We can now set up a mitigated QNode by harnessing functionality from the
    `Mitiq <https://mitiq.readthedocs.io/en/stable/>`__ package:

    .. code-block:: python3

        from pennylane import numpy as np
        from pennylane import qnode

        from mitiq.zne.scaling import fold_global
        from mitiq.zne.inference import RichardsonFactory

        n_wires = 2
        n_layers = 2

        shapes = qml.SimplifiedTwoDesign.shape(n_wires, n_layers)
        np.random.seed(0)
        w1, w2 = [np.random.random(s) for s in shapes]

        @qml.transforms.mitigate_with_zne([1, 2, 3], fold_global, RichardsonFactory.extrapolate)
        @qnode(dev)
        def circuit(w1, w2):
            qml.SimplifiedTwoDesign(w1, w2, wires=range(2))
            return qml.expval(qml.PauliZ(0))

    Executions of ``circuit`` will now be mitigated:

    >>> circuit(w1, w2)
    0.19113067083636542

    The unmitigated circuit result is ``0.33652776`` while the ideal circuit result is
    ``0.23688169`` and we can hence see that mitigation has helped reduce our estimation error.

    .. UsageDetails::

        **Theoretical details**

        A summary of ZNE can be found in `LaRose et al. <https://arxiv.org/abs/2009.04417>`__. The
        method works by assuming that the amount of noise present when a circuit is run on a
        noisy device is enumerated by a parameter :math:`\gamma`. Suppose we have an input circuit
        that experiences an amount of noise equal to :math:`\gamma = \gamma_{0}` when executed.
        Ideally, we would like to evaluate the result of the circuit in the :math:`\gamma = 0`
        noise-free setting.

        To do this, we create a family of equivalent circuits whose ideal noise-free value is the
        same as our input circuit. However, when run on a noisy device, each circuit experiences
        a noise equal to :math:`\gamma = s \gamma_{0}` for some scale factor :math:`s`. By
        evaluating the noisy outputs of each circuit, we can extrapolate to :math:`s=0` to estimate
        the result of running a noise-free circuit.

        A key element of ZNE is the ability to run equivalent circuits for a range of scale factors
        :math:`s`. When the noise present in a circuit scales with the number of gates, :math:`s`
        can be varied using `unitary folding <https://ieeexplore.ieee.org/document/9259940>`__.
        Unitary folding works by noticing that a unitary :math:`U` is equivalent to
        :math:`U U^{\dagger} U`. This type of transform can be applied to individual gates in the
        circuit or to the whole circuit. When no folding occurs, the scale factor is
        :math:`s=1` and we are running our input circuit. On the other hand, when each gate has been
        folded once, we have tripled the amount of noise in the circuit so that :math:`s=3`. For
        :math:`s \geq 3`, each gate in the circuit will be folded more than once. A typical choice
        of scale parameters is :math:`(1, 2, 3)`.

        **Unitary folding**

        This transform applies ZNE to an input circuit using the unitary folding approach. It
        requires a callable to be passed as the ``folding`` argument with signature

        .. code-block:: python

            fn(circuit, scale_factor, **folding_kwargs)

        where

        - ``circuit`` is a quantum tape,

        - ``scale_factor`` is a float, and

        - ``folding_kwargs`` are optional keyword arguments.

        The output of the function should be the folded circuit as a quantum tape.
        Folding functionality is available from the
        `Mitiq <https://mitiq.readthedocs.io/en/stable/>`__ package (version 0.11.0 and above)
        in the
        `zne.scaling.folding <https://mitiq.readthedocs.io/en/stable/apidoc.html#module-mitiq.zne.scaling.folding>`__
        module.

        **Extrapolation**

        This transform also requires a callable to be passed to the ``extrapolate`` argument that
        returns the extrapolated value(s). Its function should be

        .. code-block:: python

            fn(scale_factors, results, **extrapolate_kwargs)

        where

        - ``scale_factors`` are the ZNE scale factors,

        - ``results`` are the execution results of the circuit at the specified scale
          factors of shape ``(len(scale_factors), len(qnode_returns))``, and

        - ``extrapolate_kwargs`` are optional keyword arguments.

        The output of the extrapolate ``fn`` should be a flat array of
        length ``len(qnode_returns)``.

        Extrapolation functionality is available using ``extrapolate``
        methods of the factories in the
        `mitiq.zne.inference <https://mitiq.readthedocs.io/en/stable/apidoc.html#module-mitiq.zne.inference>`__
        module.
    """
    folding_kwargs = folding_kwargs or {}
    extrapolate_kwargs = extrapolate_kwargs or {}

    tape = circuit.expand(stop_at=lambda op: not isinstance(op, QuantumTape))

    with QuantumTape() as tape_removed:
        for op in tape._ops:
            apply(op)

    tapes = [
        [folding(tape_removed, s, **folding_kwargs) for _ in range(reps_per_factor)]
        for s in scale_factors
    ]
    tapes = [tape_ for tapes_ in tapes for tape_ in tapes_]  # flattens nested list

    out_tapes = []

    for tape_ in tapes:
        # pylint: disable=expression-not-assigned
        with QuantumTape() as t:
            [apply(p) for p in tape._prep]
            [apply(op) for op in tape_.operations]
            [apply(m) for m in tape.measurements]
        out_tapes.append(t)

    def processing_fn(results):
        """Maps from input tape executions to an error-mitigated estimate"""
        results = [
            results[i : i + reps_per_factor] for i in range(0, len(results), reps_per_factor)
        ]  # creates nested list according to reps_per_factor
        results = mean(results, axis=1)
        extrapolated = extrapolate(scale_factors, results, **extrapolate_kwargs)
        return extrapolated[0] if len(extrapolated) == 1 else extrapolated

    return out_tapes, processing_fn
