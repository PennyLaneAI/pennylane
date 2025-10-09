# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Contains the SelectPauliRot template.
"""

from pennylane import math
from pennylane.decomposition import add_decomps, adjoint_resource_rep, register_resources
from pennylane.operation import Operation
from pennylane.ops import CNOT, RZ, Hadamard, S, adjoint
from pennylane.queuing import AnnotatedQueue, QueuingManager, apply
from pennylane.templates.state_preparations.mottonen import _apply_uniform_rotation_dagger
from pennylane.wires import Wires


class SelectPauliRot(Operation):
    r"""Applies individual single-qubit Pauli rotations depending on the state of
    designated control qubits.

    This operator, also called a **multiplexed rotation** or **uniformly controlled rotation**,
    applies a sequence of multi-controlled rotations about the same axis to a single target qubit.
    The rotation angles are selected based on the state of the control qubits.
    Its definition is given by:

    .. math::

       \sum_i | i \rangle \langle i | \otimes R_P(\alpha_i),

    where :math:`| i \rangle` refers to the computational basis state of the control register,
    the :math:`\{\alpha_i\}` are the rotation angles, and :math:`R_P` denotes a Pauli rotation
    about the Pauli operator :math:`P` applied to the target qubit.

    .. figure:: ../../../doc/_static/templates/subroutines/select_pauli_rot.png
                    :align: center
                    :width: 70%
                    :target: javascript:void(0);

    For more details, see `Möttönen and Vartiainen (2005), Fig 7a <https://arxiv.org/abs/quant-ph/0504100>`_.

    Args:
        angles (tensor_like): The rotation angles to be applied. The length of the angles array must
            be :math:`2^n`, where :math:`n` is the number of ``control_wires``.
        control_wires (Sequence[int]): The control qubits used to select the rotation.
        target_wire (Sequence[int]): The wire where the rotations are applied.
        rot_axis (str): The axis around which the rotation is performed.
            It can take the value ``X``, ``Y`` or ``Z``. Default is ``Z``.

    Raises:
        ValueError: If the length of the angles array is not :math:`2^n`, where :math:`n` is the number
            of ``control_wires``.
        ValueError: If ``rot_axis`` has a value different from ``X``, ``Y`` or ``Z``.
        ValueError: If the number of the target wires is not one.

    .. seealso:: :class:`~.Select`.

    **Example**

    .. code-block:: python

        angles = np.array([1.0, 2.0, 3.0, 4.0])

        wires = qml.registers({"control": 2, "target": 1})
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit():
            qml.SelectPauliRot(
                angles,
                control_wires=wires["control"],
                target_wire=wires["target"],
                rot_axis="Y",
            )
            return qml.state()

    >>> print(circuit()) # doctest: +SKIP
    [0.8776+0.j 0.4794+0.j 0.    +0.j 0.    +0.j 0.    +0.j 0.    +0.j
     0.    +0.j 0.    +0.j]
    """

    grad_method = None
    ndim_params = (1,)

    resource_keys = {"num_wires", "rot_axis"}

    def __init__(
        self, angles, control_wires, target_wire, rot_axis="Z", id=None
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments

        self.hyperparameters["control_wires"] = Wires(control_wires)
        self.hyperparameters["target_wire"] = Wires(target_wire)
        self.hyperparameters["rot_axis"] = rot_axis

        if math.shape(angles)[-1] != 2 ** len(control_wires):
            raise ValueError("Number of angles must be 2^(len(control_wires))")

        if rot_axis not in ["X", "Y", "Z"]:
            raise ValueError("'rot_axis' can only take the values 'X', 'Y' and 'Z'.")

        if len(self.hyperparameters["target_wire"]) != 1:
            raise ValueError("Only one target wire can be specified")

        all_wires = self.hyperparameters["control_wires"] + self.hyperparameters["target_wire"]
        super().__init__(angles, wires=all_wires, id=id)

    def _flatten(self):
        metadata = tuple((key, value) for key, value in self.hyperparameters.items())
        return self.data, metadata

    @classmethod
    def _unflatten(cls, data, metadata):
        hyperparams_dict = dict(metadata)
        return cls(*data, **hyperparams_dict)

    @property
    def resource_params(self) -> dict:
        return {"rot_axis": self.hyperparameters["rot_axis"], "num_wires": len(self.wires)}

    def map_wires(self, wire_map: dict):
        """Map the control and target wires using the provided wire map."""
        new_dict = {
            key: [wire_map.get(w, w) for w in self.hyperparameters[key]]
            for key in ["control_wires", "target_wire"]
        }
        return SelectPauliRot(
            self.parameters[0],
            new_dict["control_wires"],
            new_dict["target_wire"],
            rot_axis=self.hyperparameters["rot_axis"],
        )

    # pylint: disable=arguments-differ
    @classmethod
    def _primitive_bind_call(cls, angles, control_wires, target_wire, **kwargs):
        wires = [*control_wires, target_wire]
        return super()._primitive_bind_call(angles, wires=wires, **kwargs)

    def decomposition(self):
        """Return the operator's decomposition using its parameters and hyperparameters."""
        return self.compute_decomposition(self.parameters[0], **self.hyperparameters)

    # pylint: disable=arguments-differ
    @staticmethod
    def compute_decomposition(angles, control_wires, target_wire, rot_axis):
        r"""
        Computes the decomposition operations for the given state vector.

        Args:
            angles (tensor_like): The rotation angles to be applied.
            control_wires (Sequence[int]): The control qubits used to select the rotation.
            target_wire (Sequence[int]): The wire where the rotations are applied.
            rot_axis (str): The axis around the rotation is performed.
                It can take the value ``X``, ``Y`` or ``Z``. Default is ``Z``.

        Returns:
            list: List of decomposition operations.
        """

        control_wires = Wires(control_wires)
        target_wire = Wires(target_wire)

        with AnnotatedQueue() as q:
            decompose_select_pauli_rot(angles, control_wires + target_wire, rot_axis)

        if QueuingManager.recording():
            for op in q.queue:
                apply(op)

        return q.queue


def _select_pauli_rot_resource(num_wires, rot_axis):
    return {
        RZ: 2 ** (num_wires - 1),
        CNOT: 2 ** (num_wires - 1) if num_wires > 1 else 0,
        Hadamard: 0 if rot_axis == "Z" else 2,
        S: 1 if rot_axis == "Y" else 0,
        adjoint_resource_rep(S, {}): 1 if rot_axis == "Y" else 0,
    }


# Not exact resources because rotations might be skipped based on angles
@register_resources(_select_pauli_rot_resource, exact=False)
def decompose_select_pauli_rot(angles, wires, rot_axis, **__):
    r"""Decomposes the SelectPauliRot"""

    if rot_axis == "X":
        Hadamard(wires[-1])
    elif rot_axis == "Y":
        adjoint(S(wires[-1]))
        Hadamard(wires[-1])

    _apply_uniform_rotation_dagger(RZ, angles, wires[-2::-1], wires[-1])

    if rot_axis == "X":
        Hadamard(wires[-1])
    elif rot_axis == "Y":
        Hadamard(wires[-1])
        S(wires[-1])


add_decomps(SelectPauliRot, decompose_select_pauli_rot)

# pylint: disable=protected-access
if SelectPauliRot._primitive is not None:

    @SelectPauliRot._primitive.def_impl
    def _(*args, n_wires, **kwargs):
        (angles,), (*control_wires, target_wire) = args[:-n_wires], args[-n_wires:]
        return type.__call__(SelectPauliRot, angles, control_wires, target_wire, **kwargs)
