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
This module contains the QNode class and qnode decorator.
"""
from functools import lru_cache

from pennylane.beta.tapes import QuantumTape

from pennylane.beta.interfaces.autograd import AutogradInterface


class QuantumFunctionError(Exception):
    """Exception raised when an illegal operation is defined in a quantum function."""


class QNode:
    def __init__(self, func, dev, interface="autograd", diff_method="best", diff_options=None):
        self.func = func
        self.dev = dev
        self.qtape = None

        if self.interface not in self.INTERFACE_MAP:
            raise QuantumFunctionError(
                f"Unkown interface {interface}. Interface must be "
                f"one of {self.interface_map.values()}."
            )

        self.interface = interface
        self.diff_method = diff_method
        self.diff_options = diff_options or []

    def construct(self, args, kwargs):
        """Call the quantum function with a tape context,
        ensuring the operations get queued."""

        with QuantumTape() as self.qtape:
            # Note that the quantum function doesn't
            # return anything. We assume that all classical
            # pre-processing happens *prior* to the quantum
            # tape, and the tape is the final step in the function.
            self.func(*args, **kwargs)

        # apply the interface (if any)
        self.INTERFACE_MAP[self.interface](self)

        # provide the jacobian options
        self.qtape.jacobian_options = self.diff_options

        # bind some pre-existing methods for backward compatibility
        self.jacobian = self.qtape.jacobian

    def __call__(self, *args, **kwargs):
        # construct the tape
        self.construct(args, kwargs)

        # execute the tape
        return self.qtape.execute(device=self.dev)

    def to_tf(self):
        """Apply the TensorFlow interface to the internal quantum tape.

        Raises:
            QuantumFunctionError: if TensorFlow >= 2.1 is not installed
        """
        try:
            from pennylane.beta.interfaces.tf import TFInterface

            TFInterface.apply(self.qtape)
        except ImportError:
            raise QuantumFunctionError(
                "TensorFlow not found. Please install the latest "
                "version of TensorFlow to enable the 'tf' interface."
            )

    def to_torch(self):
        """Apply the Torch interface to the internal quantum tape.

        Raises:
            QuantumFunctionError: if PyTorch >= 1.3 is not installed
        """
        try:
            from pennylane.beta.interfaces.tf import TFInterface

            TFInterface.apply(self.qtape)
        except ImportError:
            raise QuantumFunctionError(
                "PyTorch not found. Please install "
                "version of PyTorch to enable the 'torch' interface."
            )

    def to_autograd(self):
        """Apply the Autograd interface to the internal quantum tape."""
        AutogradInterface.apply(self.qtape)

    INTERFACE_MAP = {"autograd": to_autograd, "torch": to_torch, "tf": to_tf}


def qnode(device, *, interface="autograd", diff_method="best"):
    """Decorator for creating QNodes."""

    @lru_cache()
    def qfunc_decorator(func):
        """The actual decorator"""
        return QNode(func, device, interface=interface, diff_method=diff_method)

    return qfunc_decorator
