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
"""TODO"""
from typing import Any, Dict, Sequence, Optional
from pennylane.transforms import batch_transform, support_preparations_and_measurements
from pennylane.tape import QuantumTape
from pennylane.math import mean


@batch_transform
def mitigate_with_zne(
    tape: QuantumTape,
    scale_factors: Sequence[float],
    folding: callable,
    extrapolate: callable,
    folding_kwargs: Optional[Dict[str, Any]] = None,
    extrapolate_kwargs: Optional[Dict[str, Any]] = None,
    reps_per_factor=1,
) -> float:
    """Mitigate an input circuit using zero-noise extrapolation.

    Error mitigation is a precursor to error correction and is compatible with near-term quantum
    devices. It aims to lower the impact of noise when evaluating a circuit on a quantum device by
    evaluating multiple variations of the circuit and post-processing the results into a
    noise-reduced estimate. This transform implements the zero-noise extrapolation (ZNE) method
    originally introduced by
    `Temme et al. <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.119.180509>`__ and
    `Li et al. <https://journals.aps.org/prx/abstract/10.1103/PhysRevX.7.021050>`__.

    A summary of ZNE can be found in `LaRose et al. <https://arxiv.org/abs/2009.04417>`__. The
    method works by assuming that the circuit experiences a fixed amount of noise when executed on
    a noisy device that is enumerated by the parameter :math:`\gamma`. If an equivalent circuit can
    be run for a range of noise parameters :math:`\gamma`, then the results can be extrapolated to
    the :math:`\gamma = 0` noise case.

    A key element of ZNE is the ability to run equivalent circuits for a range of noise parameters
    :math:`\gamma`. When :math:`\gamma` scales with the number of gates in the circuit, it can be
    varied using `unitary folding <https://ieeexplore.ieee.org/document/9259940>`__. Unitary folding
    works by noticing that a unitary :math:`U` is equivalent to :math:`U U^{\dagger} U`. This type
    of transform can be applied to individual gates in the circuit or to the whole circuit and is
    controlled by a scale parameter :math:`s` which is calibrated so that :math:`s = 1` corresponds
    to the (unfolded) input circuit and :math:`s = 3` is a folding of all gates in the circuit once.

    This transform applies ZNE to an input circuit using the unitary folding approach. It requires
    a callable to be passed as the ``folding`` argument with signature
    ``fn(circuit, scale_factor, **folding_kwargs)`` where ``circuit`` is a quantum tape,
    ``scale_factor`` is a float, and ``folding_kwargs`` are optional arguments passed to the folding
    function. The output of the function should be the folded circuit as a quantum tape. Folding
    functionality is available from the `Mitiq <https://mitiq.readthedocs.io/en/stable/>`__ package
    in the `zne.scaling.folding <https://mitiq.readthedocs.io/en/stable/apidoc.html#module-mitiq.zne.scaling.folding>`__
    module.

    This transform also requires a callable to be passed to the ``extrapolate`` argument that
    returns the extrapolated value. Its function should be
    ``fn(scale_factors, results, **extrapolate_kwargs)`` where ``scale_factors`` are the ZNE scale
    factors, ``results`` are the execution results of the circuit at the specified scale factors,
    and ``extrapolate_kwargs`` are optional keyword arguments. Extrapolation functionality is
    available using ``extrapolate`` methods of the factories in the
    `mitiq.zne.inference <https://mitiq.readthedocs.io/en/stable/apidoc.html#module-mitiq.zne.inference>`__
    module.

    Args:
        tape (QuantumTape): the circuit to be error mitigated
        scale_factors (Sequence[float]): the range of scale factors used
        folding (callable): a function that returns a folded circuit for a specified scale factor
        extrapolate (callable): a function that returns an extrapolated result when provided a
            range of scale factors and corresponding results
        folding_kwargs (dict): optional keyword arguments passed to the ``folding`` function
        extrapolate_kwargs (dict): optional keyword arguments passed to the ``extrapolate`` function
        reps_per_factor (int): Number of circuits generated for each scale factor. Useful when the
            folding function is stochastic.

    Returns:
        float: the result of evaluating the circuit when mitigated using ZNE


    """
    folding_kwargs = folding_kwargs or {}
    extrapolate_kwargs = extrapolate_kwargs or {}
    folding = support_preparations_and_measurements(folding)

    tapes = [
        [folding(tape, s, **folding_kwargs) for _ in range(reps_per_factor)] for s in scale_factors
    ]
    tapes = [tape_ for tapes_ in tapes for tape_ in tapes_]  # flattens nested list

    def processing_fn(results):
        results = [
            results[i: i + reps_per_factor] for i in range(0, len(results), reps_per_factor)
        ]  # creates nested list according to reps_per_factor
        results = mean(results, axis=1)

        return extrapolate(scale_factors, results, **extrapolate_kwargs)

    return tapes, processing_fn
