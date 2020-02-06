# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Device Jacobian QNode.

A QNode that delegates all gradient computations directly to the device.
"""
from .jacobian import JacobianQNode


class DeviceJacobianQNode(JacobianQNode):
    """Quantum node that delegates gradient computation to the device"""

    # pylint: disable=abstract-method

    def _best_method(self, idx):
        """Determine the correct partial derivative computation method for a positional parameter.

        For this QNode, the partial derivative of every free parameter will be
        computed using the device; only parameters used in operations with
        ``grad_method=None`` will be marked as non-differentiable.

        Args:
            idx (int): free parameter index

        Returns:
            str: partial derivative method to be used
        """
        # operations that depend on this free parameter
        ops = [d.op for d in self.variable_deps[idx]]
        methods = [op.grad_method for op in ops]

        # one nondifferentiable item makes the whole nondifferentiable
        if None in methods:
            return None

        return "A"

    def jacobian(
        self, args, kwargs=None, *, wrt=None, options=None
    ):  # pylint: disable=arguments-differ
        return super().jacobian(args, kwargs=kwargs, wrt=wrt, method="device", options=options)
