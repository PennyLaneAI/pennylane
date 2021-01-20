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
                m.return_type is qml.operation.Variance
                or m.return_type is qml.operation.Probability
            ):
                raise ValueError(
                    f"{m.return_type} is not supported with the rewind gradient method"
                )


