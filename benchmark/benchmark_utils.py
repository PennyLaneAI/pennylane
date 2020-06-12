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
Benchmarking utilities.
"""
import abc

from types import ModuleType

import pennylane as qml


class BaseBenchmark(abc.ABC):
    """ABC for benchmarks.

    Args:
        device (~pennylane.Device): device for executing the benchmark (if needed)
        verbose (bool): If True, print debugging info during the benchmark. Note that this
            may invalidate the benchmark due to printing latency that should be irrelevant.
    """

    name = None  #: str: benchmark name
    min_wires = 1  #: int: minimum number of quantum wires required by the benchmark
    n_vals = None  #: Sequence[Any]: range of benchmark parameter values for perfplot

    def __init__(self, device=None, qnode_type=None, verbose=False):
        self.device = device
        self.qnode_type = qnode_type
        if device is not None:
            if device.num_wires < self.min_wires:
                raise ValueError(
                    "'{}' requires at least {} wires.".format(self.name, self.min_wires)
                )
            self.n_wires = device.num_wires
        else:
            self.n_wires = None
        self.verbose = verbose

    def setup(self):
        """Set up the benchmark.

        This method contains the initial part of the benchmark that should not be timed.
        """

    def teardown(self):
        """Tear down the benchmark.

        This method contains the final part of the benchmark that should not be timed.
        """

    @abc.abstractmethod
    def benchmark(self, n):
        """The benchmark test itself.

        Args:
            n (int): benchmark size parameter

        Returns:
            Any: Result of the benchmark. Must not return None.
        """


def create_qnode(qfunc, device, mutable=True, interface="autograd", qnode_type="QNode"):
    """Utility function for creating a quantum node.

    Takes care of the backwards compatibility of the benchmarks.

    By default, uses the parameter-shift method for computing Jacobians.

    Args:
        qfunc (Callable): quantum function defining a circuit
        device (~pennylane.Device): device for executing the circuit
        mutable (bool): whether the QNode should mutable
        interface (str, None): interface used for classical backpropagation,
            in ('autograd', 'torch', 'tf', None)
        qnode_type (str): name of the specific QNode subclass to use

    Returns:
        BaseQNode: constructed QNode
    """
    try:
        qnode_type = getattr(qml.qnodes, qnode_type)
        qnode = qnode_type(
            qfunc, device, mutable=mutable, interface=interface,
        )
    except AttributeError:
        # versions before the "new-style" QNodes
        try:
            qnode = qml.QNode(qfunc, device, cache=not mutable)
            if interface == "torch":
                return qnode.to_torch()
            if interface == "tf":
                return qnode.to_tf()
        except TypeError:
            # versions before mutable arg
            qnode = qml.QNode(qfunc, device)

    return qnode


def expval(obs):
    """Returns the expectation value of an observable ``obs``, in a way that is
    compatible with all versions of PennyLane."""
    if type(qml.expval) == ModuleType:  # pylint: disable=unidiomatic-typecheck
        return getattr(obs, qml.expval)
    return qml.expval(obs)
