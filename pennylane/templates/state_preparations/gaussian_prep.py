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
r"""Contains the GaussianStatePreparation template."""

from pennylane.decomposition import (
    add_decomps,
    adjoint_resource_rep,
    register_condition,
    register_resources,
    resource_rep,
)
from pennylane.operation import Operation
from pennylane.ops import adjoint
from pennylane.templates import AQFT, SignedOutSquare
from pennylane.wires import Wires, WiresLike


class GaussianStatePreparation(Operation):
    r"""Prepare a Gaussian state.

    This operation prepares an :math:`n`-qubit Gaussian state

    .. math::

        |\chi\rangle =
        \frac{1}{Z}\sum_{x=0}^{N} \exp(-\frac{\pi}{N}\left(x-\tfrac{N}{2}\right)^2 |x\rangle,

    where :math:`N=2^n` is the Hilbert space dimension and :math:`Z` is the normalization constant
    such that :math:`\| |\chi\rangle\|^2=1`.

    Args:
        coefficients (np.ndarray): Coefficients of the sparse state to prepare. The ordering should
            match that in ``indices``.
        wires (qp.wires.WiresLike): Wires on which to prepare the state. All work wires will be
            allocated dynamically with :func:`~.allocate`.
        indices (tuple[int]): Indices of the sparse state to prepare. The ordering should match
            that in ``coefficients``.

    **Example**

    blah

    .. details::
        :title: Usage details

        blah
    """

    # pylint:disable=too-few-public-methods

    num_params = 0
    resource_keys = {"num_target_wires", "num_work_wires", "num_phase_gradient_wires"}

    @property
    def resource_params(self):
        """The resoure parameters for an instance of GaussianStatePreparation."""
        return {
            "num_target_wires": len(self.hyperparameters["target_wires"]),
            "num_work_wires": len(self.hyperparameters["work_wires"]),
            "num_phase_gradient_wires": len(self.hyperparameters["phase_gradient_wires"]),
        }

    def __init__(
        self,
        target_wires: WiresLike,
        work_wires: WiresLike = None,
        phase_gradient_wires: WiresLike = None,
    ):
        if work_wires is None:
            work_wires = []
        if phase_gradient_wires is None:
            phase_gradient_wires = []
        all_wires = Wires.all_wires([target_wires, work_wires, phase_gradient_wires])
        super().__init__(wires=all_wires)
        self.hyperparameters["target_wires"] = target_wires
        self.hyperparameters["work_wires"] = work_wires
        self.hyperparameters["phase_gradient_wires"] = phase_gradient_wires


def _gaussian_with_square_aqft_condition(
    num_target_wires: int, num_work_wires: int, num_phase_gradient_wires: int
) -> bool:
    return (
        num_phase_gradient_wires > 0 and num_work_wires >= num_target_wires - 1
    )  # TODO: refine work wire requirement


def _gaussian_with_square_aqft_resources(
    num_target_wires: int, num_work_wires: int, num_phase_gradient_wires: int
) -> dict:
    aqft_order = num_phase_gradient_wires  # Refine
    return {
        resource_rep(
            SignedOutSquare,
            num_x_wires=num_target_wires,
            num_output_wires=num_phase_gradient_wires,
            num_work_wires=num_work_wires,
            output_wires_zeroed=False,
        ): 1,
        resource_rep(AQFT, order=aqft_order): 1,
        adjoint_resource_rep(AQFT, base_params={"order": aqft_order}): 1,
    }


@register_condition(_gaussian_with_square_aqft_condition)
@register_resources(_gaussian_with_square_aqft_resources)
def _gaussian_with_square_aqft(target_wires, work_wires, phase_gradient_wires):
    aqft_order = len(phase_gradient_wires)  # Refine
    adjoint(AQFT)(aqft_order, target_wires)
    SignedOutSquare(target_wires, phase_gradient_wires, work_wires, output_wires_zeroed=False)
    AQFT(aqft_order, target_wires)


add_decomps(GaussianStatePreparation, _gaussian_with_square_aqft)
