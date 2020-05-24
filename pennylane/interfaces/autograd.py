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
Differentiable quantum nodes with Autograd interface.
"""
import autograd.extend
import autograd.builtins

from pennylane.numpy import array, ndarray, zeros, fromiter, int64
from pennylane.utils import _flatten, unflatten


def to_autograd(qnode):
    """Function that accepts a :class:`~.QNode`, and returns an Autograd-compatible QNode.

    Args:
        qnode (~pennylane.qnode.QNode): a PennyLane QNode

    Returns:
        AutogradQNode: an Autograd-compatible QNode
    """

    class AutogradQNode(qnode.__class__):
        """QNode that works with Autograd."""

        @property
        def interface(self):
            """str, None: automatic differentiation interface used by the node, if any"""
            return "autograd"

        # mark the evaluate method as an Autograd primitive
        evaluate = autograd.extend.primitive(qnode.__class__.evaluate)

        def __call__(self, *args, **kwargs):
            # prevents autograd boxed arguments from going through to evaluate
            args = autograd.builtins.tuple(args)  # pylint: disable=no-member
            return self.evaluate(args, kwargs)

        @staticmethod
        def QNode_vjp(ans, self, args, kwargs):
            """Returns the vector-Jacobian product operator for the QNode.

            Takes the same arguments as :meth:`evaluate`, plus ``ans``.

            Returns:
                function[array[float], array[float]]: vector-Jacobian product operator
            """
            # pylint: disable=unused-argument
            def gradient_product(g):
                """Vector-Jacobian product operator.

                Args:
                    g (array[float]): scalar or vector multiplying the Jacobian
                        from the left (output side)

                Returns:
                    nested Sequence[float]: vector-Jacobian product, arranged
                    into the nested structure of the input arguments in ``args``
                """
                diff_args = []
                diff_indices = None
                non_diff_indices = set()

                for arg, arg_variable in zip(args, self.arg_vars):
                    if not getattr(arg, "requires_grad", True):
                        indices = [i.idx for i in _flatten(arg_variable)]
                        non_diff_indices.update(indices)
                    else:
                        diff_args.append(arg)

                if non_diff_indices:
                    diff_indices = set(range(self.num_variables)) - non_diff_indices

                # Jacobian matrix of the circuit
                jac = self.jacobian(args, kwargs, wrt=diff_indices)

                if not g.shape:
                    vjp = g * jac  # numpy treats 0d arrays as scalars, hence @ cannot be used
                else:
                    vjp = g @ jac

                if non_diff_indices:
                    # Autograd requires we return a gradient of size (num_variables,)
                    res = zeros([self.num_variables])
                    indices = fromiter(diff_indices, dtype=int64)
                    res[indices] = vjp
                    vjp = res

                # Restore the nested structure of the input args.
                vjp = unflatten(vjp.flat, args)
                return vjp

            return gradient_product

    # define the vector-Jacobian product function for AutogradQNode.evaluate
    autograd.extend.defvjp(AutogradQNode.evaluate, AutogradQNode.QNode_vjp, argnums=[1])
    qnode.__class__ = AutogradQNode
    return qnode
