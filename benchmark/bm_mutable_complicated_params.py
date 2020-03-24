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
Mutable QNode, complicated primary parameters benchmark.
"""
# pylint: disable=invalid-name
import numpy as np

import pennylane as qml
import benchmark_utils as bu


def circuit(p, *, aux=0):
    """A very simple, lightweight mutable quantum circuit."""
    qml.RX(p[aux][2], wires=[0])
    return qml.expval(qml.PauliZ(0))


class Benchmark(bu.BaseBenchmark):
    """
    This benchmark attempts to measure the efficiency of :meth:`JacobianQNode._construct` for
    mutable QNodes, using an extreme case where the QNode has lots of primary parameters with
    a complicated nested structure, but relatively few auxiliary parameters, and only a few
    of the primary parameters are actually used in the circuit.

    When the QNode is constructed, a VariableRef is built for each primary parameter,
    and the qfunc re-evaluated. In this test this is meant to be time-consuming, but it is only
    strictly necessary if the auxiliary parameters change.
    The main reasons why there are significant differences in the execution speed of this test
    between different PL commits:

      * :meth:`BaseQNode._construct` should only reconstruct the QNode if the auxiliary params
        have changed.
      * Most of the primary params are not used in the circuit, hence
        :meth:`JacobianQNode._construct` should efficiently figure out that partial derivatives
        wrt. them are always zero.
    """

    name = "mutable qnode, complicated primary params"
    min_wires = 1
    n_vals = range(6, 13, 1)

    def __init__(self, device=None, verbose=False):
        super().__init__(device, verbose)
        self.qnode = None

    def setup(self):
        self.qnode = bu.create_qnode(circuit, self.device, mutable=True, interface=None)

    def benchmark(self, n=8):
        # n is the number of levels in the primary parameter tree.
        # Hence the number of primary parameters depends exponentially on n.

        def create_params(n):
            """Recursively builds a tree structure with n levels."""
            if n <= 0:
                # the leaves are arrays
                return np.random.randn(2)
            # the other nodes have two branches and a scalar
            return [create_params(n - 1), create_params(n - 1), np.random.randn()]

        p = create_params(n)

        def evaluate(aux):
            """Evaluates the qnode using the given auxiliary params."""
            res = self.qnode(p, aux=aux)
            # check the result
            assert np.allclose(res, np.cos(p[aux][2]))

        # first evaluation and construction
        evaluate(0)
        # evaluate the node several times more with a different auxiliary argument
        # (it does not matter if p changes or not, the VariableRefs handle it)
        for _ in range(1, 10):
            # If we had evaluate(i % 2) here instead the auxiliary arguments would change
            # every time, which would negate most possible speedups.
            evaluate(1)

        return True
