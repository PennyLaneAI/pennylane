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
r"""
Contains the AllSinglesDoubles template.
"""

# pylint: disable=too-many-arguments,protected-access
import copy
from collections.abc import Sequence

import numpy as np

from pennylane import capture, math
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources, resource_rep
from pennylane.operation import Operation
from pennylane.ops import BasisState, DoubleExcitation, SingleExcitation
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike

has_jax = True
try:
    from qpjax import numpy as jnp
except (ModuleNotFoundError, ImportError) as import_error:  # pragma: no cover
    has_jax = False  # pragma: no cover


class AllSinglesDoubles(Operation):
    r"""Builds a quantum circuit to prepare correlated states of molecules
    by applying all :class:`~.pennylane.SingleExcitation` and
    :class:`~.pennylane.DoubleExcitation` operations to
    the initial Hartree-Fock state.

    The template initializes the :math:`n`-qubit system to encode
    the input Hartree-Fock state and applies the particle-conserving
    :class:`~.pennylane.SingleExcitation` and
    :class:`~.pennylane.DoubleExcitation` operations which are implemented as
    `Givens rotations <https://en.wikipedia.org/wiki/Givens_rotation>`_ that act
    on the subspace of two and four qubits, respectively. The total number of
    excitation gates and the indices of the qubits they act on are obtained
    using the :func:`~.excitations` function.

    For example, the quantum circuit for the case of two electrons and six qubits
    is sketched in the figure below:

    |

    .. figure:: ../../_static/templates/subroutines/all_singles_doubles.png
        :align: center
        :width: 70%
        :target: javascript:void(0);

    |

    In this case, we have four single and double excitations that preserve the total-spin
    projection of the Hartree-Fock state. The :class:`~.pennylane.SingleExcitation` gate
    :math:`G` act on the qubits ``[0, 2], [0, 4], [1, 3], [1, 5]`` as indicated by the
    squares, while the :class:`~.pennylane.DoubleExcitation` operation :math:`G^{(2)}` is
    applied to the qubits ``[0, 1, 2, 3], [0, 1, 2, 5], [0, 1, 2, 4], [0, 1, 4, 5]``.

    The resulting unitary conserves the number of particles and prepares the
    :math:`n`-qubit system in a superposition of the initial Hartree-Fock state and
    other states encoding multiply-excited configurations.

    Args:
        weights (TensorLike): size ``(len(singles) + len(doubles),)`` tensor containing the
            angles entering the :class:`~.pennylane.SingleExcitation` and
            :class:`~.pennylane.DoubleExcitation` operations, in that order
        wires (WiresLike): wires that the template acts on
        hf_state (Sequence[int]): Length ``len(wires)`` occupation-number vector representing the
            Hartree-Fock state. ``hf_state`` is used to initialize the wires.
        singles (Sequence[tuple[int, int]] | None): An optional sequence of lists with the indices of the two qubits
            the :class:`~.pennylane.SingleExcitation` operations act on
        doubles (Sequence[tuple[int, int, int, int]] | None): An optional sequence of lists with the indices of the four qubits
            the :class:`~.pennylane.DoubleExcitation` operations act on

    .. details::
        :title: Usage Details

        Notice that:

        #. The number of wires has to be equal to the number of spin orbitals included in
           the active space.

        #. The single and double excitations can be generated with the function
           :func:`~.excitations`. See example below.

        An example of how to use this template is shown below:

        .. code-block:: python

            import pennylane as qml
            import numpy as np

            electrons = 2
            qubits = 4

            # Define the HF state
            hf_state = qml.qchem.hf_state(electrons, qubits)

            # Generate all single and double excitations
            singles, doubles = qml.qchem.excitations(electrons, qubits)

            # Define the device
            dev = qml.device('default.qubit', wires=qubits)

            wires = range(qubits)

            @qml.qnode(dev)
            def circuit(weights, hf_state, singles, doubles):
                qml.templates.AllSinglesDoubles(weights, wires, hf_state, singles, doubles)
                return qml.expval(qml.Z(0))

            # Evaluate the QNode for a given set of parameters
            params = np.random.normal(0, np.pi, len(singles) + len(doubles))
            circuit(params, hf_state, singles=singles, doubles=doubles)
    """

    grad_method = None

    resource_keys = {"num_singles", "num_doubles", "num_wires"}

    def __init__(
        self,
        weights: TensorLike,
        wires: WiresLike,
        hf_state: Sequence[int],
        singles: Sequence[tuple[int, int]] | None = None,
        doubles: Sequence[tuple[int, int, int, int]] | None = None,
        id=None,
    ):
        wires = Wires(wires)
        if len(wires) < 2:
            raise ValueError(
                f"The number of qubits (wires) can not be less than 2; got len(wires) = {len(wires)}"
            )

        if doubles is not None:
            if any(len(d_wires) != 4 for d_wires in doubles):
                raise ValueError("Expected all entries of 'doubles' to be of size 4.")
        doubles = () if doubles is None else tuple(tuple(d) for d in doubles)

        if singles is not None:
            if any(len(s_wires) != 2 for s_wires in singles):
                raise ValueError("Expected all entries of 'singles' to be of size 2.")
        singles = () if singles is None else tuple(tuple(s) for s in singles)

        weights_shape = math.shape(weights)
        exp_shape = self.shape(singles, doubles)
        if weights_shape != exp_shape:
            raise ValueError(f"'weights' tensor must be of shape {exp_shape}; got {weights_shape}.")

        if len(hf_state) != len(wires):
            raise ValueError(
                f"Expected length of 'hf_state' to match number of wires ({len(wires)})."
            )

        if (hf_dtype := hf_state[0].dtype) != np.dtype("int"):
            raise ValueError(f"Elements of 'hf_state' must be integers, got {hf_dtype}.")
        hf_state = tuple(hf_state)

        self._hyperparameters = {
            "hf_state": hf_state,
            "singles": singles,
            "doubles": doubles,
        }

        super().__init__(weights, wires=wires, id=id)

    @classmethod
    def _primitive_bind_call(
        cls, weights, wires, hf_state, singles=None, doubles=None, id=None
    ):  # pylint: disable=arguments-differ
        singles = math.array(singles) if singles is not None else math.array(((),))
        doubles = math.array(doubles) if doubles is not None else math.array(((),))
        wires = math.array(wires)
        hf_state = math.array(hf_state)
        weights = math.array(weights)
        return cls._primitive.bind(weights, wires, hf_state, singles, doubles, id=id)

    @property
    def resource_params(self) -> dict:
        return {
            "num_singles": len(self.hyperparameters["singles"]),
            "num_doubles": len(self.hyperparameters["doubles"]),
            "num_wires": len(self.wires),
        }

    def map_wires(self, wire_map: dict):
        new_op = copy.deepcopy(self)
        new_op._wires = Wires([wire_map.get(wire, wire) for wire in self.wires])
        for key in ["singles", "doubles"]:
            new_op._hyperparameters[key] = tuple(
                tuple(wire_map[w] for w in wires) for wires in new_op._hyperparameters[key]
            )
        return new_op

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(
        weights, wires, hf_state, singles, doubles
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.AllSinglesDoubles.decomposition`.

        Args:
            weights (tensor_like): size ``(len(singles) + len(doubles),)`` tensor containing the
                angles entering the :class:`~.pennylane.SingleExcitation` and
                :class:`~.pennylane.DoubleExcitation` operations, in that order
            wires (Any or Iterable[Any]): wires that the BasisState operator acts on
            hf_state (array[int]): Length ``len(wires)`` occupation-number vector representing the
                Hartree-Fock state. ``hf_state`` is used to initialize the wires.
            singles (Sequence[Sequence]): sequence of lists with the indices of the two qubits
                the :class:`~.pennylane.SingleExcitation` operations act on
            doubles (Sequence[Sequence]): sequence of lists with the indices of the four qubits
                the :class:`~.pennylane.DoubleExcitation` operations act on
        Returns:
            list[.Operator]: decomposition of the operator
        """
        op_list = []

        op_list.append(BasisState(hf_state, wires=wires))

        for i, d_wires in enumerate(doubles):
            op_list.append(DoubleExcitation(weights[len(singles) + i], wires=d_wires))

        for j, s_wires in enumerate(singles):
            op_list.append(SingleExcitation(weights[j], wires=s_wires))

        return op_list

    @staticmethod
    def shape(singles, doubles):
        r"""Returns the expected shape of the tensor that contains the circuit parameters.

        Args:
            singles (Sequence[Sequence]): sequence of lists with the indices of the two qubits
                the :class:`~.pennylane.SingleExcitation` operations act on
            doubles (Sequence[Sequence]): sequence of lists with the indices of the four qubits
                the :class:`~.pennylane.DoubleExcitation` operations act on

        Returns:
            tuple(int): shape of the tensor containing the circuit parameters
        """
        if (singles is None or len(singles) == 0) and (doubles is None or len(doubles) == 0):
            raise ValueError(
                f"'singles' and 'doubles' lists can not be both empty;"
                f" got singles = {singles}, doubles = {doubles}"
            )

        if singles is None or len(singles) == 0:
            shape_ = (len(doubles),)
        elif doubles is None or len(doubles) == 0:
            shape_ = (len(singles),)
        else:
            shape_ = (len(singles) + len(doubles),)

        return shape_


if AllSinglesDoubles._primitive is not None:

    @AllSinglesDoubles._primitive.def_impl
    def _(*args, **kwargs):  # pylint: disable=unused-argument
        # need to convert array values into integers
        # for plxpr, all wires must be integers
        # could be abstract when using tracing evaluation in interpreter
        wires = tuple(w if math.is_abstract(w) else int(w) for w in args[1])
        return type.__call__(AllSinglesDoubles, args[0], wires, args[2], args[3], args[4])


def _all_singles_doubles_resouces(num_singles, num_doubles, num_wires):
    return {
        resource_rep(BasisState, num_wires=num_wires): 1,
        resource_rep(DoubleExcitation): num_doubles,
        resource_rep(SingleExcitation): num_singles,
    }


@register_resources(_all_singles_doubles_resouces)
def _all_singles_doubles_decomposition(weights, wires, hf_state, singles, doubles):
    BasisState(hf_state, wires=wires)

    if has_jax and capture.enabled():
        weights, doubles, singles = jnp.array(weights), jnp.array(doubles), jnp.array(singles)

    @for_loop(len(doubles))
    def doubles_loop(i):
        DoubleExcitation(weights[len(singles) + i], wires=doubles[i])

    doubles_loop()  # pylint: disable=no-value-for-parameter

    @for_loop(len(singles))
    def singles_loop(j):
        SingleExcitation(weights[j], wires=singles[j])

    singles_loop()  # pylint: disable=no-value-for-parameter


add_decomps(AllSinglesDoubles, _all_singles_doubles_decomposition)
