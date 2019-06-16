# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""QGT optimizer"""
from collections import Sequence

import autograd
import autograd.numpy as np

from pennylane.utils import _flatten, unflatten


class QGTOptimizer:
    r"""Optimizer with adaptive learning rate.

    Because this optimizer performs quantum evaluations to determine
    the learning rate, either:

    * The objective function must be a QNode, or

    * If the objective function is not a QNode, all QNode dependencies of
      the objective function **must be provided**.

    .. note::

        Currently, we only support the use-case where all provided
        QNode dependencies contain the same circuit ansatz (but may
        differ in expectation value).

    Args:
        stepsize (float): the user-defined stepsize parameter :math:`\eta`
        tol (float): tolerance used for inverse
    """

    def __init__(self, stepsize=0.01, tol=1e-6):
        self._stepsize = stepsize
        self.metric_tensor = None
        self.tol = tol
        self.qnodes = []

    def step(self, objective_fn, x, qnodes=None):
        """Update x with one step of the optimizer.

        Args:
            objective_fn (Union[function, QNode]): the objective function for optimization
            qnodes (List[QNode]): list of QNodes that the objective function depends on.
                Must be provided if ``objective_fn`` is not a QNode.
            x (array): NumPy array containing the current values of the variables to be updated

        Returns:
            array: the new variable values :math:`x^{(t+1)}`
        """

        g = self.compute_grad(objective_fn, x, qnodes=qnodes)

        x_out = self.apply_grad(g, x)

        return x_out

    def compute_grad(self, objective_fn, x, qnodes=None):
        r"""Compute gradient of the objective function at the point x.

        Args:
            objective_fn (Union[function, QNode]): the objective function for optimization
            qnodes (List[QNode]): list of QNodes that the objective function depends on.
                Must be provided if ``objective_fn`` is not a QNode.
            x (array): NumPy array containing the current values of the variables to be updated

        Returns:
            array: NumPy array containing the gradient :math:`\nabla f(x^{(t)})`
        """
        if self.metric_tensor is None:
            # if the metric tensor has not been calculated,
            # first we must construct the subcircuits before
            # we call the gradient function

            # check if the objective function is a QNode
            if hasattr(objective_fn, "construct_subcircuits"):
                # objective function is the qnode!

                # Note: we pass the parameters 'x' to this method,
                # but the values themselves are not used.
                # Rather, they are simply needed for the JIT
                # circuit construction, to determine expected parameter shapes.
                objective_fn.construct_subcircuits([x])
                self.qnodes = [objective_fn]

            else:
                # objective function is a classical node

                if qnodes is None:
                    raise ValueError(
                        "As the provided objective function is not a QNode, "
                        "the qnode argument must be provided, containing a "
                        "list of all QNodes the objective function depends on."
                    )

                # use the user provided qnode dependencies
                for q in qnodes:
                    try:
                        q.construct_subcircuits([x])
                    except AttributeError:
                        raise ValueError(
                            "Item {} in list of provided QNodes is not a QNode.".format(q)
                        )

                self.qnodes = qnodes

        # calling the gradient function will implicitly
        # evaluate the subcircuit expectations
        g = autograd.grad(objective_fn)(x)  # pylint: disable=no-value-for-parameter

        if not isinstance(x, Sequence):
            x = np.array([x])

        if self.metric_tensor is None:
            # metric tensor has not already been previously calculated.
            # calculate metric tensor elements for each qnode, and verify
            # they are identical.
            metric_tensor = np.zeros([len(self.qnodes), len(x.flatten())])

            for idx, q in enumerate(self.qnodes):
                for i in range(len(x.flatten())):
                    # evaluate metric tensor diagonals
                    metric_tensor[idx, i] = q.subcircuits[i]['result']

                if idx > 0:
                    # verify metric tensor is the same as previous metric tensor
                    same_tensor = np.allclose(metric_tensor[idx], metric_tensor[idx - 1])

                    if not same_tensor:
                        # stop the loop and raise an exception
                        raise ValueError(
                            "QNodes containing different circuits currently not supported"
                        )

            # since all metric tensors are identical, just keep the first one
            self.metric_tensor = metric_tensor[0]
        else:
            # metric tensor has already been previously calculated.
            # we now know they are all identical for each qnode, so we can
            # just use the first qnodes subcircuits to save time

            for i in range(len(x.flatten())):
                # evaluate metric tensor diagonals
                self.metric_tensor[i] = self.qnodes[0].subcircuits[i]['result']

        return g

    def apply_grad(self, grad, x):
        r"""Update the variables x to take a single optimization step. Flattens and unflattens
        the inputs to maintain nested iterables as the parameters of the optimization.

        Args:
            grad (array): The gradient of the objective
                function at point :math:`x^{(t)}`: :math:`\nabla f(x^{(t)})`
            x (array): the current value of the variables :math:`x^{(t)}`

        Returns:
            array: the new values :math:`x^{(t+1)}`
        """
        grad_flat = list(_flatten(grad))
        x_flat = _flatten(x)

        # inverse metric tensor
        # note: in the cases where np.abs(self.metric_tensor) > self.tol, we
        # should raise a warning to let the user know that tol should be reduced
        G_inv = np.where(np.abs(self.metric_tensor) > self.tol, 1 / self.metric_tensor, 0)

        x_new_flat = [e - self._stepsize * g * d for e, g, d in zip(x_flat, G_inv, grad_flat)]

        return unflatten(x_new_flat, x)
