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
"""Contains the SuperpositionTHC template used as a subroutine in THC Qubitization."""

from pennylane import capture, compiler, for_loop, math, adjoint, BasisState, cond
from pennylane.decomposition import (
    add_decomps,
    register_resources,
)
from pennylane.operation import Operation
from pennylane.ops import CNOT, X, Hadamard, RY, Z, Controlled
from pennylane.queuing import AnnotatedQueue, QueuingManager, apply
from pennylane.templates.subroutines import Elbow
from pennylane.wires import Wires, WiresLike
from pennylane.labs.templates import LeftQuantumComparator, LeftClassicalComparator

from pennylane.pennylane.ops import MultiControlledX


class SuperpositionTHC(Operation):
    r"""Perform an inequality test :math:`\lvert x\rangle\lvert y\rangle\lvert 0\rangle \mapsto \lvert x\rangle \lvert y\rangle\lvert x \leq y\rangle` between two states in separate quantum registers.

    This operator performs an inequality test between two quantum registers :math:`x` and
    :math:`y`, storing the result in a zeroed target qubit. The
    ``comparator`` argument can be one of four possible string values ``"<", "<=", ">", ">="`` to determine the type of inequality test. For example, choosing ``comparator="<"`` we have the following operation:

    .. math::

        \text{LeftQuantumComparator}_{<} \lvert x\rangle \lvert y\rangle \lvert 0\rangle = \lvert x\rangle \lvert y\rangle \lvert x < y\rangle

    The decomposition is defined as the left block in Figure 6 in Appendix E
    of `Su et al. (2021) <https://arxiv.org/abs/2105.12767>`_. Note that the decomposition uses auxiliary wires
    and in order to clean them, we must apply the adjoint of this operator via ``Adjoint(LeftQuantumComparator)``
    after using the target qubit, as shown in the example below.

    Args:
        x_wires (WiresLike): The wires that store the integer :math:`x`.
        y_wires (WiresLike): The wires that store the integer :math:`y`. The number of ``y_wires`` should be equal to
            the number of ``x_wires``.
        target_wire (WiresLike): The zeroed target wire that outputs the value of the inequality test.
        work_wires (WiresLike): The auxiliary wires to use for the addition.
            At least ``len(y_wires) - 1`` zeroed work wires should be provided. They are not returned in the zero state.
        comparator (str): The operator used in the inequality. The value could be '<', '<=', '>=' and '>'.

    **Example**

    In this example, we will use the ``LeftQuantumComparator``, generating the output on wire :math:`11`. After this,
    we will copy the result to wire :math:`12` using a ``CNOT`` gate, and then apply the ``adjoint(LeftQuantumComparator)``
    to clean up the auxiliary qubits used.

    .. code-block:: python

        import pennylane as qp
        from pennylane.labs.templates import LeftQuantumComparator

        dev = qp.device("lightning.qubit")

        @qp.qnode(dev, shots=1)
        def circuit(a, comparator, b):
            x_wires = [0, 3, 6, 9]
            y_wires = [1, 4, 7, 10]
            work_wires = [2, 5, 8]
            qp.BasisState(a, wires=x_wires)
            qp.BasisState(b, wires=y_wires)
            LeftQuantumComparator(x_wires, y_wires, 11, work_wires, comparator)

            # We copy the output in wire=12
            qp.CNOT(wires=[11, 12])

            # We clean the work wires used in LeftQuantumComparator
            qp.adjoint(LeftQuantumComparator(x_wires, y_wires, 11, work_wires, comparator))

            return qp.sample(wires=[12])

    .. code-block:: pycon

        >>> output = circuit(3, ">=", 2)
        >>> print(bool(output))
        True
    """

    grad_method = None

    resource_keys = {"num_y_wires", "comparator"}

    def __init__(
        self,
        M: int,
        N: int,
        mu_wires: WiresLike,
        nu_wires: WiresLike,
        work_wires: str,
    ):  # pylint: disable=too-many-arguments

        mu_wires = Wires(mu_wires)
        nu_wires = Wires(nu_wires)
        work_wires = Wires(work_wires)

        '''
        if comparator not in ["<", "<=", ">=", ">"]:
            raise ValueError("Allowed values for 'comparator' are: '<', '<=', '>=' and '>'.")

        if len(work_wires) < len(y_wires) - 1:
            raise ValueError(f"At least {len(y_wires)-1} work_wires should be provided.")
        if work_wires.intersection(target_wire):
            raise ValueError("None of the wires in work_wires should be the target wire.")
        if work_wires.intersection(x_wires):
            raise ValueError("None of the wires in work_wires should be included in x_wires.")
        if work_wires.intersection(y_wires):
            raise ValueError("None of the wires in work_wires should be included in y_wires.")
        if len(x_wires) != len(y_wires):
            raise ValueError("The number of y_wires should be equal to the number of x_wires")
        if x_wires.intersection(target_wire):
            raise ValueError("None of the wires in x_wires should be the target wire.")
        if x_wires.intersection(y_wires):
            raise ValueError("None of the wires in y_wires should be included in x_wires.")
        if y_wires.intersection(target_wire):
            raise ValueError("None of the wires in y_wires should be the target wire.")
        '''

        self.hyperparameters["M"] = M
        self.hyperparameters["N"] = N
        self.hyperparameters["mu_wires"] = mu_wires
        self.hyperparameters["nu_wires"] = nu_wires
        self.hyperparameters["work_wires"] = work_wires

        all_wires = [mu_wires, nu_wires, work_wires]
        all_wires = Wires.all_wires(all_wires)
        super().__init__(wires=all_wires)

    @property
    def resource_params(self) -> dict:
        return {
            "num_y_wires": len(self.hyperparameters["y_wires"]),
            "comparator": self.hyperparameters["comparator"],
        }

    @property
    def num_params(self):
        return 2

    def _flatten(self):
        metadata = tuple((key, value) for key, value in self.hyperparameters.items())
        return tuple(), metadata

    @classmethod
    def _unflatten(cls, data, metadata):
        hyperparams_dict = dict(metadata)
        return cls(*data, **hyperparams_dict)

    def map_wires(self, wire_map: dict) -> "SuperpositionTHC":
        new_dict = {
            key: [wire_map.get(w, w) for w in self.hyperparameters[key]]
            for key in ["mu_wires", "nu_wires", "work_wires"]
        }

        return SuperpositionTHC(**new_dict, M=self.hyperparameters["M"] , N=self.hyperparameters["N"])

    def decomposition(self):
        r"""Representation of the operator as a product of other operators."""
        return self.compute_decomposition(**self.hyperparameters)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @staticmethod
    def compute_decomposition(
        M, N, mu_wires, nu_wires, work_wires, comparator
    ):  # pylint: disable=arguments-differ, too-many-arguments
        r"""Representation of the operator as a product of other operators.

        Args:
            x_wires (WiresLike): The wires that store the integer :math:`x`.
            y_wires (WiresLike): The wires that store the integer :math:`y`.
            target_wire (WiresLike): The wire that stores the value of the inequality test.
            work_wires (WiresLike): The auxiliary wires to use for the addition.
                At least ``len(y_wires) - 1`` work wires should be provided.
            comparator (str): The operator used in the inequality. The value could be '<', '<=', '>=' and '>'.

        Returns:
            list[.Operator]: Decomposition of the operator
        """

        with AnnotatedQueue() as q:
            _superposition_thc(
                M, N, mu_wires, nu_wires
            )

        if QueuingManager.recording():
            for o in q.queue:
                apply(o)

        return q.queue



def left_equalities(M, N, mu_wires, nu_wires, work_wires, keep_eq = False):

    LeftClassicalComparator(nu_wires, M, target_wire=work_wires[1], work_wires=work_wires[7: 7 + len(mu_wires) - 1],
                            comparator="<=")
    LeftQuantumComparator(mu_wires, nu_wires, target_wire=work_wires[2],
                          work_wires=work_wires[7 + len(mu_wires) - 1: 7 + 2 * len(mu_wires) - 1], comparator="<=")
    LeftClassicalComparator(mu_wires, N // 2, target_wire=work_wires[4],
                            work_wires=work_wires[7 + 2 * len(mu_wires) - 1: 7 + 3 * len(mu_wires) - 1], comparator=">")

    BasisState(M, wires=nu_wires)

    # TODO: change this for MultiTemporaryAND
    cond(keep_eq, MultiControlledX)(wires=nu_wires + [work_wires[3]], control_values=[0] * (len(nu_wires)),
                          work_wires=work_wires[7 + 3 * len(mu_wires) - 1: 7 + 4 * len(mu_wires) - 1])

def _left_quantum_comparator_resources(num_y_wires, comparator):

    resources = {
        Elbow: num_y_wires,
        CNOT: 2 + 5 * (num_y_wires - 1),
    }

    if comparator in [">=", "<="]:
        resources[X] = 1

    return resources


@register_resources(_left_quantum_comparator_resources, exact=True)
def _superposition_thc(
    M, N, mu_wires, nu_wires, work_wires, **_
):  # pylint: disable=too-many-arguments

    # the first seven work_wires are the one Fig 3 https://arxiv.org/pdf/2011.03494,
    # All are returned to zero other than work_wire 0, 3 and 6
    for wire in mu_wires:
        Hadamard(wire)

    for wire in nu_wires:
        Hadamard(wire)

    # angle calculation
    n_total_vals = 2 ** len(mu_wires)
    d = N // 2 + M * (M + 1)
    frac_valid = d / n_total_vals ** 2
    limit = 0.5 / math.sqrt(frac_valid)
    cos_val = math.where(limit < 1.0, limit, 1.0)
    angle = 2 * math.arccos(cos_val)

    RY(angle, wires=work_wires[0])
    X(wires=work_wires[5])

    left_equalities(M, N, mu_wires, nu_wires, work_wires)

    Controlled(X(work_wires[5]), control_wires=work_wires[3:5], work_wires=work_wires[7 + 4 * len(mu_wires) - 1:])
    Controlled(Z(work_wires[5]), control_wires=work_wires[0:3], work_wires=work_wires[7 + 4 * len(mu_wires) - 1:])
    Controlled(X(work_wires[5]), control_wires=work_wires[3:5], work_wires=work_wires[7 + 4 * len(mu_wires) - 1:])

    adjoint(left_equalities)(M, N, mu_wires, nu_wires, work_wires)

    RY(-angle, wires=work_wires[0])

    for wire in mu_wires:
        Hadamard(wire)

    for wire in nu_wires:
        Hadamard(wire)

    for wires in mu_wires + nu_wires + [work_wires[0]]:
        X(wires=wires)
    Controlled(Z(work_wires[0]), control_wires=mu_wires + nu_wires,
               work_wires=work_wires[7 + 4 * len(mu_wires) - 1:])
    for wires in mu_wires + nu_wires + [work_wires[0]]:
        X(wires=wires)

    for wire in mu_wires:
        Hadamard(wire)

    for wire in nu_wires:
        Hadamard(wire)

    left_equalities(M, N, mu_wires, nu_wires, work_wires)

    Controlled(X(work_wires[5]), control_wires=work_wires[3:5], work_wires=work_wires[7 + 4 * len(mu_wires) - 1:])
    Controlled(X(work_wires[6]), control_wires=work_wires[1:3] + work_wires[5],
               work_wires=work_wires[7 + 4 * len(mu_wires) - 1:])
    Controlled(X(work_wires[5]), control_wires=work_wires[3:5], work_wires=work_wires[7 + 4 * len(mu_wires) - 1:])

    X(wires=work_wires[5])

    adjoint(left_equalities)(M, N, mu_wires, nu_wires, work_wires, keep_eq=True)


add_decomps(SuperpositionTHC, _superposition_thc)
