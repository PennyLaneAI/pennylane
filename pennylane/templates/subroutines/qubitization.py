# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
This submodule contains the template for Qubitization.
"""

import copy

from pennylane.decomposition import add_decomps, register_resources, resource_rep
from pennylane.operation import Operation
from pennylane.ops import I, Prod, prod
from pennylane.wires import Wires

from .prepselprep import PrepSelPrep
from .reflection import Reflection


class Qubitization(Operation):
    r"""Applies the `Qubitization <https://arxiv.org/abs/2204.11890>`__ operator.

    This operator encodes a Hamiltonian, written as a linear combination of unitaries, into a unitary operator.
    It is implemented with a quantum walk operator that takes a Hamiltonian as input and generates:

    .. math::
        Q =  \text{Prep}_{\mathcal{H}}^{\dagger} \text{Sel}_{\mathcal{H}} \text{Prep}_{\mathcal{H}}(2|0\rangle\langle 0| - I).



    .. seealso:: :class:`~.AmplitudeEmbedding` and :class:`~.Select`.

    Args:
        hamiltonian (Union[.Hamiltonian, .Sum, .Prod, .SProd, .LinearCombination]): The Hamiltonian written as a linear combination of unitaries.
        control (Iterable[Any], Wires): The control qubits for the Qubitization operator.

    **Example**

    This operator, when applied in conjunction with QPE, allows computing the eigenvalue of an eigenvector of the Hamiltonian.

    .. code-block:: python

        H = qml.dot([0.1, 0.3, -0.3], [qml.Z(0), qml.Z(1), qml.Z(0) @ qml.Z(2)])

        @qml.qnode(qml.device("default.qubit"))
        def circuit():

            # initiate the eigenvector
            qml.PauliX(2)

            # apply QPE
            measurements = qml.iterative_qpe(
                qml.Qubitization(H, control = [3,4]), aux_wire = 5, iters = 3
            )
            return qml.probs(op = measurements)

        output = circuit()

        # post-processing
        lamb = sum([abs(c) for c in H.terms()[0]])

    >>> print("eigenvalue: ", lamb * np.cos(2 * np.pi * (np.argmax(output)) / 8))
    eigenvalue: 0.7
    """

    grad_method = None

    resource_keys = {"num_control_wires", "hamiltonian"}

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    def __init__(self, hamiltonian, control, id=None):
        wires = Wires(control) + hamiltonian.wires

        self._hyperparameters = {
            "hamiltonian": hamiltonian,
            "control": Wires(control),
        }

        super().__init__(*hamiltonian.data, wires=wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {
            "num_control_wires": len(self.hyperparameters["control"]),
            "hamiltonian": self.hyperparameters["hamiltonian"],
        }

    def _flatten(self):
        data = (self.hyperparameters["hamiltonian"],)
        metadata = tuple(item for item in self.hyperparameters.items() if item[0] != "hamiltonian")
        return data, metadata

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(*data, **dict(metadata))

    def __copy__(self):

        clone = Qubitization.__new__(Qubitization)

        # Ensure the operators in the hyper-parameters are copied instead of aliased.
        clone._hyperparameters = {
            "hamiltonian": copy.copy(self._hyperparameters["hamiltonian"]),
            "control": copy.copy(self._hyperparameters["control"]),
        }

        for attr, value in vars(self).items():
            if attr != "_hyperparameters":
                setattr(clone, attr, value)

        return clone

    def map_wires(self, wire_map: dict):
        # pylint: disable=protected-access
        new_op = copy.deepcopy(self)
        new_op._wires = Wires([wire_map.get(w, w) for w in self.wires])
        new_op._hyperparameters["hamiltonian"] = new_op._hyperparameters["hamiltonian"].map_wires(
            wire_map
        )
        new_op._hyperparameters["control"] = Wires(
            [wire_map.get(w, w) for w in self._hyperparameters["control"]]
        )
        return new_op

    @staticmethod
    def compute_decomposition(*_, **kwargs):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.Qubitization.decomposition`.

        Args:
            *params (list): trainable parameters of the operator, as stored in the ``parameters`` attribute
            wires (Iterable[Any], Wires): wires that the operator acts on
            **hyperparams (dict): non-trainable hyperparameters of the operator, as stored in the ``hyperparameters`` attribute

        Returns:
            list[Operator]: decomposition of the operator

        **Example:**

        .. code-block:: python

            import pennylane as qml
            from pennylane.wires import Wires

        >>> print(qml.Qubitization.compute_decomposition(hamiltonian=0.1 * qml.Z(0), control=Wires(1)))
        [Reflection(3.141592653589793, wires=[1]), PrepSelPrep(lcu=0.1 * Z(0), control=Wires([1]))]
        """

        hamiltonian = kwargs["hamiltonian"]
        control = kwargs["control"]

        decomp_ops = []

        identity = prod(*[I(wire) for wire in control])

        decomp_ops.append(Reflection(identity))
        decomp_ops.append(PrepSelPrep(hamiltonian, control=control))

        return decomp_ops


def _qubitization_resources(num_control_wires, hamiltonian):
    return {
        resource_rep(
            Reflection,
            base_class=Prod,
            base_params={"resources": {resource_rep(I): num_control_wires}},
            num_wires=1,
            num_reflection_wires=1,
        ): 1,
        resource_rep(
            PrepSelPrep,
            op_reps=(resource_rep(type(hamiltonian), **hamiltonian.resource_params),),
            num_control=num_control_wires,
        ): 1,
    }


@register_resources(_qubitization_resources)
def _qubitization_decomposition(*_, **kwargs):
    hamiltonian = kwargs["hamiltonian"]
    control = kwargs["control"]

    Reflection(Prod(*[I(wire) for wire in control]))
    PrepSelPrep(hamiltonian, control=control)


add_decomps(Qubitization, _qubitization_decomposition)
