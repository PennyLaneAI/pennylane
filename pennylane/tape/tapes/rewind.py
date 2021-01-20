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
TODO
"""
# pylint: disable=protected-access
import pennylane as qml
from pennylane.devices import DefaultMixed
from .jacobian_tape import JacobianTape


class RewindTape(JacobianTape):
    r"""TODO
    """

    def jacobian(self, device, params=None, **options):
        """TODO

        Args:
            device:
            params:
            **options:

        Returns:
        """

        # The rewind tape only support differentiating expectation values of observables for now.
        for m in self.measurements:
            if (
                m.return_type is not qml.operation.Expectation
            ):
                raise ValueError(
                    f"The {m.return_type.value} return type is not supported with the rewind gradient method"
                )

        method = options.get("method", "analytic")

        if method == "device":
            # Using device mode; simply query the device for the Jacobian
            return self.device_pd(device, params=params, **options)
        elif method == "numeric":
            raise ValueError("RewindTape does not support numeric differentiation")

        if not device.capabilities().get("returns_state") or isinstance(device, DefaultMixed):
            # TODO: consider renaming returns_state to, e.g., uses_statevector
            # TODO: consider adding a capability for mixed/pure state
            raise qml.QuantumFunctionError("The rewind gradient method is only supported on statevector-based devices")

        # Perform the forward pass
        # TODO: Could we use lower-level like device.apply, since we just need the state?
        self.execute(device, params=params)
        phi = device.state







