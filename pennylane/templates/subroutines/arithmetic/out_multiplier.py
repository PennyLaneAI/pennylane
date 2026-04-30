# Copyright 2018-2026 Xanadu Quantum Technologies Inc.

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
Contains the OutMultiplier template.
"""

from collections import defaultdict

from pennylane.decomposition import (
    add_decomps,
    adjoint_resource_rep,
    change_op_basis_resource_rep,
    controlled_resource_rep,
    register_condition,
    register_resources,
)
from pennylane.decomposition.resources import resource_rep
from pennylane.operation import Operation
from pennylane.ops import (
    CNOT,
    BasisState,
    H,
    Prod,
    X,
    adjoint,
    change_op_basis,
    ctrl,
    prod,
)
from pennylane.queuing import AnnotatedQueue, QueuingManager, apply
from pennylane.templates.subroutines.arithmetic import SemiAdder, TemporaryAND
from pennylane.templates.subroutines.arithmetic.semi_adder import _semiadder, _semiadder_resources
from pennylane.templates.subroutines.controlled_sequence import ControlledSequence
from pennylane.templates.subroutines.qft import QFT
from pennylane.wires import Wires, WiresLike

from .phase_adder import PhaseAdder


class OutMultiplier(Operation):
    r"""Performs the out-place modular multiplication operation.

    This operator performs the modular multiplication of integers :math:`x` and :math:`y` modulo
    :math:`mod` in the computational basis:

    .. math::
        \text{OutMultiplier}(mod) |x \rangle |y \rangle |b \rangle = |x \rangle |y \rangle |b + x \cdot y \; \text{mod} \; mod \rangle,

    There are three implementations available, which differ in the auxiliary wires
    and in the gate counts they require, and in whether or not they support arbitrary values for
    the modulus ``mod``. See the usage details for more information.

    .. note::

        To obtain the correct result, :math:`x`, :math:`y` and :math:`b` must be smaller than :math:`mod`.

    .. seealso:: :class:`~.PhaseAdder` and :class:`~.Multiplier`.

    Args:
        x_wires (Sequence[int]): the wires that store the integer :math:`x`
        y_wires (Sequence[int]): the wires that store the integer :math:`y`
        output_wires (Sequence[int]): the wires that store the multiplication result. If the register is in a non-zero state :math:`b`, the solution will be added to this value
        mod (int): the modulo for performing the multiplication. If not provided, it will be set to its maximum value, :math:`2^{\text{len(output_wires)}}`
        work_wires (Sequence[int]): the auxiliary wires to use for the multiplication. The
            work wires are not needed if :math:`mod=2^{\text{len(output_wires)}}`, otherwise at least two work wires
            should be provided. Defaults to empty tuple.
        output_wires_zeroed (bool): Whether the ``output_wires`` are guaranteed to be in state
            :math:`|0\rangle` initially.

    **Example**

    This example performs the multiplication of two integers :math:`x=2` and :math:`y=7` modulo :math:`mod=12`.
    We'll let :math:`b=0`. See Usage Details for :math:`b \neq 0`.

    .. code-block:: python

        x = 2
        y = 7
        mod = 12

        x_wires = [0, 1]
        y_wires = [2, 3, 4]
        output_wires = [6, 7, 8, 9]
        work_wires = [5, 10]

        dev = qp.device("default.qubit")

        @qp.qnode(dev, shots=1)
        def circuit():
            qp.BasisEmbedding(x, wires=x_wires)
            qp.BasisEmbedding(y, wires=y_wires)
            qp.OutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)
            return qp.sample(wires=output_wires)

    >>> print(circuit())
    [[0 0 1 0]]

    The result :math:`[[0 0 1 0]]`, is the binary representation of
    :math:`2 \cdot 7 \; \text{modulo} \; 12 = 2`.

    .. details::
        :title: Usage Details

        This template takes as input four different registers of wires.

        The first register is ``x_wires`` which is used
        to encode the integer :math:`x < mod` in the computational basis.

        The second register is ``y_wires`` which is used
        to encode the integer :math:`y < mod` in the computational basis.

        The third register is ``output_wires`` which is used
        to encode the integer :math:`(b+ x \cdot y) \; \text{mod} \; mod` in the computational
        basis. Therefore, it will require at least :math:`\lceil \log_2(mod)\rceil` wires
        Note that these wires can be initialized with any integer :math:`b < mod`.

        The fourth register is ``work_wires`` containing the auxiliary qubits used to
        perform the modular multiplication operation. The number of auxiliary wires determines
        which decomposition is available (also see below).

        **Initial state of output wires**

        As indicated above, the initial state of ``output_wires`` can encode any value
        :math:`b<mod`. The following is an example for :math:`b = 1`.

        .. code-block:: python

            b = 1
            x = 2
            y = 7
            mod = 12

            x_wires = [0, 1]
            y_wires = [2, 3, 4]
            output_wires = [6, 7, 8, 9]
            work_wires = [5, 10]

            dev = qp.device("default.qubit")

            @qp.qnode(dev, shots=1)
            def circuit():
                qp.BasisEmbedding(x, wires=x_wires)
                qp.BasisEmbedding(y, wires=y_wires)
                qp.BasisEmbedding(b, wires=output_wires)
                qp.OutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)
                return qp.sample(wires=output_wires)

        >>> print(circuit())
        [[0 0 1 1]]

        The result :math:`(0011)_2`, is the binary representation of
        :math:`(1 + 2 \cdot 7)\; \text{modulo} \; 12 = 3`:

        If the initial state on the output wires is guaranteed to be :math:`|0\rangle`, this
        can be indicated to ``OutMultiplier`` by setting ``output_wires_zeroed=True``. This
        simplifies some of the available decompositions (also see below), saving quantum resources.

        **Different decompositions**

        There are three decompositions, which differ in the required number of work wires and
        gates, and in whether they support ``mod!=2**len(output_wires)``.

        - The first implementation is based on the quantum Fourier transform (QFT) method presented in
          `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_. It requires zero (two) auxiliary
          wires for ``mod=2**len(output_wires)`` (for other values of ``mod``), and it uses a doubly
          controlled sequence (a nested :class:`~.ControlledSequence`) and two QFTs. Any value
          for ``mod`` is supported, subject to the description above.

        - The second implementation uses controlled :class:`~.SemiAdder`\ s to realize the
          multiplication. For :math:`n` ``x_wires``, :math:`m` ``y_wires`` and :math:`k`
          ``output_wires``, we need :math:`L = \min(k, n)` adders with usually varying sizes
          :math:`\min(k - i, m + 1)` for :math:`0\leq i<L`. The concrete :class:`~.Toffoli` count
          resulting from this is a bit verbose in general. The implementation is shown in
          Fig. 2a) (for :math:`k=2m=2n`) and Fig. 2c) (for :math:`k=m=n`) in
          `arXiv:2410.00899 <https://arxiv.org/abs/2410.00899>`__.

        - The third implementation uses controlled addition/subtraction to replace the controlled
          ``SemiAdder``\ s from the previous implementation, based on Litinski's
          `arXiv:2410.00899 <https://arxiv.org/abs/2410.00899>`__.
          For :math:`n` ``x_wires``, :math:`m` ``y_wires`` and :math:`k`
          ``output_wires``, we need :math:`L=\min(k, n)` controlled add/subtract operations
          of usually varying size :math:`\min(k + 1 - i, m + 1)` for :math:`0\leq i<L`,
          three ``SemiAdder``\ s of sizes :math:`\min(k + 1 - m, n + 1)`,
          :math:`\min(k + 1 - n, m + 1)` and :math:`k+1`, as well as an incrementer on
          :math:`\min(k + 1, n + m)` qubits and Pauli gates. For :math:`n=m` and
          :math:`k=2n`, this implementation is shown in Fig. 2b), for :math:`k=n=m` in Fig. 2d).

    """

    grad_method = None

    resource_keys = {
        "num_output_wires",
        "num_x_wires",
        "num_y_wires",
        "num_work_wires",
        "mod",
        "output_wires_zeroed",
    }

    def __init__(
        self,
        x_wires: WiresLike,
        y_wires: WiresLike,
        output_wires: WiresLike,
        mod=None,
        work_wires: WiresLike = (),
        output_wires_zeroed: bool = False,
        id=None,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments

        x_wires = Wires(x_wires)
        y_wires = Wires(y_wires)
        output_wires = Wires(output_wires)
        work_wires = Wires(() if work_wires is None else work_wires)

        num_work_wires = len(work_wires)

        if mod is None:
            mod = 2 ** len(output_wires)
        if mod != 2 ** len(output_wires):
            if num_work_wires < 2:
                raise ValueError(
                    f"If mod is not 2^{len(output_wires)}, at least two work wires should be provided."
                )
            work_wires = work_wires[:2]
        if mod > 2 ** (len(output_wires)):
            raise ValueError(
                "OutMultiplier must have enough wires to represent mod. The maximum mod "
                f"with len(output_wires)={len(output_wires)} is {2 ** len(output_wires)}, but received {mod}."
            )

        if len(work_wires) != 0:
            if any(wire in work_wires for wire in x_wires):
                raise ValueError("None of the wires in work_wires should be included in x_wires.")
            if any(wire in work_wires for wire in y_wires):
                raise ValueError("None of the wires in work_wires should be included in y_wires.")

        if any(wire in y_wires for wire in x_wires):
            raise ValueError("None of the wires in y_wires should be included in x_wires.")
        if any(wire in x_wires for wire in output_wires):
            raise ValueError("None of the wires in x_wires should be included in output_wires.")
        if any(wire in y_wires for wire in output_wires):
            raise ValueError("None of the wires in y_wires should be included in output_wires.")

        wires_list = [x_wires, y_wires, output_wires, work_wires]
        wires_name = ["x_wires", "y_wires", "output_wires", "work_wires"]

        for name, wires in zip(wires_name, wires_list):
            self.hyperparameters[name] = Wires(wires)
        self.hyperparameters["mod"] = mod
        self.hyperparameters["output_wires_zeroed"] = output_wires_zeroed

        # pylint: disable=consider-using-generator
        all_wires = sum([self.hyperparameters[name] for name in wires_name], start=[])
        super().__init__(wires=all_wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {
            "num_output_wires": len(self.hyperparameters["output_wires"]),
            "num_x_wires": len(self.hyperparameters["x_wires"]),
            "num_y_wires": len(self.hyperparameters["y_wires"]),
            "num_work_wires": len(self.hyperparameters["work_wires"]),
            "mod": self.hyperparameters["mod"],
            "output_wires_zeroed": self.hyperparameters["output_wires_zeroed"],
        }

    @property
    def num_params(self):
        return 0

    def _flatten(self):
        metadata = tuple((key, value) for key, value in self.hyperparameters.items())
        return tuple(), metadata

    @classmethod
    def _unflatten(cls, data, metadata):
        hyperparams_dict = dict(metadata)
        return cls(**hyperparams_dict)

    def map_wires(self, wire_map: dict):
        new_dict = {
            key: [wire_map.get(w, w) for w in self.hyperparameters[key]]
            for key in ["x_wires", "y_wires", "output_wires", "work_wires"]
        }

        return OutMultiplier(
            new_dict["x_wires"],
            new_dict["y_wires"],
            new_dict["output_wires"],
            self.hyperparameters["mod"],
            new_dict["work_wires"],
        )

    def decomposition(self):
        return self.compute_decomposition(**self.hyperparameters)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @staticmethod
    def compute_decomposition(
        x_wires: WiresLike,
        y_wires: WiresLike,
        output_wires: WiresLike,
        mod,
        work_wires: WiresLike,
        output_wires_zeroed: bool = False,
    ):  # pylint: disable=arguments-differ, too-many-arguments, unused-argument
        r"""Representation of the operator as a product of other operators.

        Args:
            x_wires (Sequence[int]): the wires that store the integer :math:`x`
            y_wires (Sequence[int]): the wires that store the integer :math:`y`
            output_wires (Sequence[int]): the wires that store the multiplication result. If the register is in a non-zero state :math:`b`, the solution will be added to this value
            mod (int): the modulo for performing the multiplication. If not provided, it will be set to its maximum value, :math:`2^{\text{len(output_wires)}}`
            work_wires (Sequence[int]): the auxiliary wires to use for the multiplication. The
                work wires are not needed if :math:`mod=2^{\text{len(output_wires)}}`, otherwise two work wires
                should be provided.

        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        >>> qp.OutMultiplier.compute_decomposition(x_wires=[0,1], y_wires=[2,3], output_wires=[5,6], mod=4, work_wires=[4,7])
        [(Adjoint(QFT(wires=[5, 6]))) @ (ControlledSequence(ControlledSequence(PhaseAdder(wires=[5, 6]), control=[0, 1]), control=[2, 3])) @ QFT(wires=[5, 6])]
        """
        if mod != 2 ** len(output_wires):
            qft_output_wires = work_wires[:1] + output_wires
            work_wire = work_wires[1:2]
        else:
            qft_output_wires = output_wires
            work_wire = ()

        if output_wires_zeroed:
            compute_op = prod(*(H(w) for w in qft_output_wires))
        else:
            compute_op = QFT(qft_output_wires)
        uncompute_op = adjoint(QFT)(qft_output_wires)

        target_op = ControlledSequence(
            ControlledSequence(PhaseAdder(1, qft_output_wires, mod, work_wire), control=x_wires),
            control=y_wires,
        )
        op_list = [change_op_basis(compute_op, target_op, uncompute_op)]
        return op_list


def _out_multiplier_with_qft_resources(
    num_output_wires, num_x_wires, num_y_wires, mod, output_wires_zeroed, **_
) -> dict:
    qft_wires = num_output_wires + 1 if mod != 2**num_output_wires else num_output_wires

    if output_wires_zeroed:
        compute_rep = resource_rep(Prod, resources={resource_rep(H): qft_wires})
    else:
        compute_rep = resource_rep(QFT, num_wires=qft_wires)

    uncompute_rep = adjoint_resource_rep(QFT, base_params={"num_wires": qft_wires})
    target_rep = resource_rep(
        ControlledSequence,
        base_class=ControlledSequence,
        base_params={
            "base_class": PhaseAdder,
            "base_params": {"num_x_wires": qft_wires, "mod": mod},
            "num_control_wires": num_x_wires,
        },
        num_control_wires=num_y_wires,
    )
    return {change_op_basis_resource_rep(compute_rep, target_rep, uncompute_rep): 1}


def _out_multiplier_with_qft_condition(num_output_wires, mod, num_work_wires, **_):
    return mod in (None, 2**num_output_wires) or num_work_wires >= 2


@register_condition(_out_multiplier_with_qft_condition)
@register_resources(_out_multiplier_with_qft_resources)
def _out_multiplier_with_qft(
    x_wires: WiresLike,
    y_wires: WiresLike,
    output_wires: WiresLike,
    mod,
    work_wires: WiresLike,
    output_wires_zeroed: bool,
    **_,
):  # pylint: disable=too-many-arguments, unused-argument
    OutMultiplier.compute_decomposition(
        x_wires, y_wires, output_wires, mod, work_wires, output_wires_zeroed
    )


def _out_multiplier_with_adder_resources(
    num_output_wires, num_x_wires, num_y_wires, output_wires_zeroed, num_work_wires, **_
) -> dict:
    """Resources for OutMultiplier decomposition with controlled adders."""
    n = num_x_wires
    m = num_y_wires
    k = num_output_wires

    resources = defaultdict(int)
    if output_wires_zeroed:
        resources[resource_rep(TemporaryAND)] += min(m, k)

    for i in range(int(output_wires_zeroed), min(k, n)):
        if output_wires_zeroed:
            size = min(k - i, m + 1)
        else:
            size = k - i
        resources[
            controlled_resource_rep(
                base_class=SemiAdder,
                base_params={
                    "num_x_wires": m,
                    "num_y_wires": size,
                    "num_work_wires": num_work_wires,
                },
                num_control_wires=1,
                num_zero_control_values=0,
            )
        ] += 1
    return dict(resources)


def _out_multiplier_with_adder_condition(
    num_output_wires, num_y_wires, mod, num_work_wires, output_wires_zeroed, **_
):
    k = num_output_wires
    m = num_y_wires
    # Controlled adder takes as many work wires as the output register size. The largest controlled
    # adder is the first one in the loop, with size `min(k - 1, m+1)` if output_wires_zeroed=True
    # (because in that case the very first adder is replaced by ctrl(copy)) and size `k` else.
    if output_wires_zeroed:
        min_num_work_wires = min(k - 1, m + 1)
    else:
        min_num_work_wires = k
    return mod in (None, 2**num_output_wires) and num_work_wires >= min_num_work_wires


@register_condition(_out_multiplier_with_adder_condition)
@register_resources(_out_multiplier_with_adder_resources)
def _out_multiplier_with_adder(
    x_wires: WiresLike,
    y_wires: WiresLike,
    output_wires: WiresLike,
    mod,
    work_wires: WiresLike,
    output_wires_zeroed: bool,
    **__,
):  # pylint: disable=unused-argument, too-many-arguments
    """Implementation of Schoolbook multiplication via controlled adders as sole building block,
    except for a potential simplification for the very first adder.
    The j-th building block adds y⋅x_{n-1-j}⋅2^j to the output register, by controlling the
    addition of y on x_{n-1-j} and shifting the output wires of the addition by j bits.
    Overall, we thus add

    sum_{j=0}^{n-1} 2^j⋅x_{n-1-j}⋅y = x⋅y

    to the output register. Note that the size of the addition output registers as well as the
    upper limit of the sum are adjusted depending on the sizes n, m, and k of the three registers
    x_wires, y_wires, and output_wires.
    """
    m = len(y_wires)
    k = len(output_wires)

    # If the output wires are zeroed, the first controlled adder is just a controlled copy.
    if output_wires_zeroed:
        for y_wire, out_wire in zip(
            y_wires[::-1], output_wires[max(0, k - (m + 1)) : k][::-1], strict=False
        ):
            TemporaryAND([x_wires[-1], y_wire, out_wire])

    # If the output wires are zeroed, we already did the first controlled adder above
    start = int(output_wires_zeroed)
    for i, x_wire in enumerate(x_wires[::-1][start:k], start=start):
        # Slice the output wires according to the shift in control, and bounded by its own size,
        # and the size of the y_wires
        if output_wires_zeroed:
            out_wires = output_wires[max(0, k - (m + 1 + i)) : k - i]
        else:
            out_wires = output_wires[: k - i]
        # Add y wires to shifted output, controlled by current x_wire
        ctrl(SemiAdder(y_wires, out_wires, work_wires=work_wires), control=x_wire)


def _out_multiplier_with_caddsub_resources(
    num_output_wires, num_x_wires, num_y_wires, num_work_wires, output_wires_zeroed, **_
) -> dict:
    n = num_x_wires
    m = num_y_wires
    k = num_output_wires + 1  # augmented output register
    num_passed_ww = num_work_wires - 1  # One work wire is used by the arithmetic logic itself.

    resources = defaultdict(int)

    # Some resource reps we will need:
    cnot_on_0_kwargs = {"base_params": {}, "num_control_wires": 1, "num_zero_control_values": 1}
    cnot_on_0_rep = controlled_resource_rep(X, **cnot_on_0_kwargs)
    x_rep = resource_rep(X)

    # Controlled add-subtract loop
    loop_size = min(k, n)
    # Bit flips on the y_wires, controlled on |0>: two per ctrl-add-subtract
    if num_y_wires > 1:
        c_flips = controlled_resource_rep(
            BasisState,
            base_params={"num_wires": num_y_wires - 1},
            num_control_wires=1,
            num_zero_control_values=1,
        )
        resources[c_flips] += 2 * loop_size

    # Bit flip of LSB output wire, controlled on |0>: two per ctrl-add-subtract
    resources[cnot_on_0_rep] += 2 * loop_size
    # Bit flip on LSB work wire, controlled on |0>: two per ctrl-add-subtract that has work wires
    c_add_subs_with_work_wires = min(n, k - 1)
    resources[cnot_on_0_rep] += 2 * c_add_subs_with_work_wires

    # Decomposed SemiAdder of y_wires onto output_wires: One per ctrl-add-subtract, varying size
    for i in range(loop_size):
        size = min(k - i, m + 1) if output_wires_zeroed else k - i
        adder_resources = _semiadder_resources(num_x_wires=m, num_y_wires=size)
        for key, value in adder_resources.items():
            resources[key] += value

    # Add 2^m(x+1)
    adder_resources = _semiadder_resources(num_x_wires=n, num_y_wires=k - m)
    for key, value in adder_resources.items():
        resources[key] += value
    # bit flips corresponding to input carry activated. Accounts for the fact that
    # we don't need to flip a work wire if k=m+1, in which case there are no work wires.
    has_work_wires = int(k > m + 1)
    resources[x_rep] += 4 + 2 * has_work_wires

    # Subtract y+2^(n+m)
    # First negation
    resources[x_rep] += k
    # Add y
    resources[
        resource_rep(SemiAdder, num_x_wires=m, num_y_wires=k, num_work_wires=num_passed_ww)
    ] += 1

    # increment 2^(n+m) bit
    size = k - n - m
    if size > 0:
        if size > 1:
            resources[resource_rep(TemporaryAND)] += size - 2
            resources[adjoint_resource_rep(TemporaryAND)] += size - 2
        resources[resource_rep(CNOT)] += size - 1
        resources[x_rep] += 1

    # Second negation
    resources[x_rep] += k

    # Add 2^n y
    if k > n:
        resources[
            resource_rep(SemiAdder, num_x_wires=m, num_y_wires=k - n, num_work_wires=num_passed_ww)
        ] += 1

    return dict(resources)


def _out_multiplier_with_caddsub_condition(num_output_wires, mod, num_work_wires, **_) -> bool:
    # Adder sizes are (using n=num_x_wires, m=num_y_wires, k=num_output_wires+1):
    # - min(k, m+1) # Largest size occurring in controlled add/sub loop
    # - k-m, # Add 2^m(x+1)
    # - k, # Add y during subtracting 2^(n+m)+y     <-- Largest one
    # - k-n, # Add 2^n y
    largest_adder_size = num_output_wires + 1
    # One work wire for temporarily enlarged output register. Adder takes size-1 work wires.
    min_num_work_wires = 1 + (largest_adder_size - 1)
    return mod in (None, 2**num_output_wires) and num_work_wires >= min_num_work_wires


def _add_plus_one(x_wires, y_wires, work_wires):
    """This qfunc implements ``(x, y, 0) -> (x, (x+y+1) % 2**m, 0)`` for ``m`` the number of
    bits in ``y``. Note that it will produce the right behaviour in a circuit both when decomposing
    the right elbows into measurement + CZ and when using a decomposition into unitary operators.
    This is because we use a measurement with reset rather than a simple bit flip to return
    the last work wire to the state |0>.
    This circuit is similar to the one shown in Fig. 1c) in
    `arXiv:2410.00899 <https://arxiv.org/abs/2410.00899>`__, just without the bit flips on the
    ``x_wires`` before and after the adder. We replace the explicit input carry in that figure
    by bit flips on the least significant bits of all three registers, handling the clean-up
    bit flip on the least significant work qubit with measurement+reset.
    """
    work_wires = work_wires[: len(y_wires) - 1]
    X(x_wires[-1])
    X(y_wires[-1])
    with AnnotatedQueue() as q:
        _semiadder(x_wires, y_wires, work_wires)
    adder_ops = q.queue
    if work_wires:
        # We insert work wire bit flips where a carry-in qubit would cause them,
        # i.e., after the very first left elbow and before the last right elbow
        with QueuingManager.stop_recording():
            work_wire_flip = X(work_wires[-1])
        adder_ops.insert(1, work_wire_flip)
        adder_ops.insert(-2, work_wire_flip)
    if QueuingManager.recording():
        for op in adder_ops:
            apply(op)
    X(y_wires[-1])
    X(x_wires[-1])


def _increment(wires, work_wires):
    """Increment the input `wires` by one, using zeroed `work_wires`.
    We use a left elbow ladder together with a CNOT+right elbow uncompute ladder.
    This is a manually reduced decomposition of the standard incrementer via MCX gates if
    work wires are available:

    Generic decomposition:
    0: ─╭X────────────────┤
    1: ─├●─╭X─────────────┤
    2: ─├●─├●─╭X──────────┤
    3: ─├●─├●─├●─╭X───────┤
    4: ─├●─├●─├●─├●─╭X────┤
    5: ─╰●─╰●─╰●─╰●─╰●──X─┤

    Decompose all MCX gates into elbows and CNOTs:
       0: ─────────────╭X──────────────────────────────────────────────────────────────────────────┤
       1: ──────────╭●─│───●╮──────────────────────╭X──────────────────────────────────────────────┤
       2: ───────╭●─│──│────│──●╮───────────────╭●─│───●╮───────────────╭X─────────────────────────┤
       3: ────╭●─│──│──│────│───│──●╮────────╭●─│──│────│──●╮────────╭●─│───●╮────────╭X───────────┤
       4: ─╭●─│──│──│──│────│───│───│──●╮─╭●─│──│──│────│───│──●╮─╭●─│──│────│──●╮─╭●─│───●╮─╭X────┤
       5: ─├●─│──│──│──│────│───│───│──●┤─├●─│──│──│────│───│──●┤─├●─│──│────│──●┤─├●─│───●┤─╰●──X─┤
    aux0: ─│──│──├⊕─├●─│───●┤──⊕┤───│───│─│──│──│──│────│───│───│─│──│──│────│───│─│──│────│───────┤
    aux1: ─│──├⊕─╰●─│──│────│──●╯──⊕┤───│─│──├⊕─├●─│───●┤──⊕┤───│─│──│──│────│───│─│──│────│───────┤
    aux2: ─╰⊕─╰●────│──│────│──────●╯──⊕╯─╰⊕─╰●─│──│────│──●╯──⊕╯─╰⊕─├●─│───●┤──⊕╯─│──│────│───────┤
    aux3: ──────────╰⊕─╰●──⊕╯───────────────────╰⊕─╰●──⊕╯────────────╰⊕─╰●──⊕╯─────╰⊕─╰●──⊕╯───────┤

    Cancel neighbouring right and left elbows (moving some work wire usage around in the process)
       0: ─────────────╭X───────────────────────────────┤
       1: ──────────╭●─│───●╮─╭X────────────────────────┤
       2: ───────╭●─│──│────│─│──●╮──╭X─────────────────┤
       3: ────╭●─│──│──│────│─│───│──│──●╮─╭X───────────┤
       4: ─╭●─│──│──│──│────│─│───│──│───│─│───●╮─╭X────┤
       5: ─├●─│──│──│──│────│─│───│──│───│─│───●┤─╰●──X─┤
    aux0: ─│──│──├⊕─├●─│───●┤─╰●─⊕┤──│───│─│────│───────┤
    aux1: ─│──├⊕─╰●─│──│────│────●╯──╰●─⊕┤─│────│───────┤
    aux2: ─╰⊕─╰●────│──│────│───────────●╯─╰●──⊕╯───────┤
    aux3: ──────────╰⊕─╰●──⊕╯───────────────────────────┤

    We see a leading ladder of left elbows and a backwards ladder of CNOT+right elbow pairs.
    This circuit is derived, e.g., in
    `Gidney's blog <https://algassert.com/circuits/2015/06/12/Constructing-Large-Increment-Gates.html>`__,
    see "Incrementer from n-2 Zeroed bits".
    """
    wires = wires[::-1]
    if len(wires) > 1:
        # Construct the wires on which the ladder will act.
        all_wires = wires[:1] + list(sum(zip(wires[1:], work_wires), start=tuple()))
        # Forward ladder
        for k in range(len(wires) - 2):
            TemporaryAND(all_wires[2 * k : 2 * k + 3])
        # Backward ladder
        for k in range(len(wires) - 3, -1, -1):
            CNOT([all_wires[2 * k + 2], all_wires[2 * k + 3]])
            adjoint(TemporaryAND)(all_wires[2 * k : 2 * k + 3])
        # Trailing CNOT
        CNOT(wires[:2])
    X(wires[0])


def _c_add_sub(c_wire, x_wires, y_wires, work_wires):
    r"""Controlled add/subtract operation. If the control wire ``c_wire`` is in the
    state :math:`|1\rangle`, simply adds :math:`x`, the integer stored in ``x_wires``,
    to :math:`y`, the value in ``y_wires``. If the control wire is in
    the state :math:`|0\rangle`, adds :math:`2^n-x` to :math:`y` instead where :math:`n`
    is the length of ``x_wires``. In short:

    |0>|x>|y>  ->  |0>|x>|y+2^n-x>
    |1>|x>|y>  ->  |1>|x>|y+x>

    This is shown in Fig. 1f) in `arXiv:2410.00899 <https://arxiv.org/abs/2410.00899>`__.
    Note that the figure explicitly shows an input carry for the adder, which
    we do not represent here. Instead, we introduce (controlled) bit flips on the least significant
    bits of each register that correspond to an input carry being set to one.
    """
    if len(x_wires) > 1:
        ctrl(BasisState([1] * (len(x_wires) - 1), x_wires[:-1]), control=c_wire, control_values=[0])

    work_wires = work_wires[: len(y_wires) - 1]
    ctrl(X(y_wires[-1]), control=c_wire, control_values=[0])

    with AnnotatedQueue() as q:
        _semiadder(x_wires, y_wires, work_wires)
    adder_ops = q.queue
    if work_wires:
        # We insert controlled work wire bit flips where a carry-in qubit would cause them,
        # i.e., after the very first left elbow and before the last right elbow
        with QueuingManager.stop_recording():
            work_wire_flip = ctrl(X(work_wires[-1]), control=c_wire, control_values=[0])
        adder_ops.insert(1, work_wire_flip)
        adder_ops.insert(-2, work_wire_flip)
    if QueuingManager.recording():
        for op in adder_ops:
            apply(op)

    ctrl(X(y_wires[-1]), control=c_wire, control_values=[0])

    if len(x_wires) > 1:
        ctrl(BasisState([1] * (len(x_wires) - 1), x_wires[:-1]), control=c_wire, control_values=[0])


@register_condition(_out_multiplier_with_caddsub_condition)
@register_resources(_out_multiplier_with_caddsub_resources)
def _out_multiplier_with_caddsub(
    x_wires: WiresLike,
    y_wires: WiresLike,
    output_wires: WiresLike,
    mod: None,
    work_wires: WiresLike,
    output_wires_zeroed: bool,
    **__,
):  # pylint: disable=unused-argument, too-many-arguments
    """Implementation of improved Schoolbook multiplication via controlled add/subtract blocks,
    combined with some correction steps. After appending a work wire to the output register,
    effectively multiplying it with two, we first have a bulk computation with n steps (where
    n is the size of x_wires):

    The j-th building block adds y⋅(2x_{n-1-j}-1)⋅2^j+2^(j+m)⋅(1-x_{n-1-j}) to the output register,
    by controlling between addition and subtraction of y on x_{n-1-j}, and shifting the output
    wires of the addition by j bits.
    Overall, we thus computed (including the initial multiplication with two)

    2⋅z + sum_{j=0}^{n-1} (2^j⋅(2x_{n-1-j}-1)⋅y + 2^{j+m}⋅(1-x_{n-1-j})
    = 2⋅z + 2⋅x⋅y - (2^n-1)⋅y + 2^{n+m}-2^m⋅(1+x)

    to the output register. Note that the size of the addition output registers as well as the
    upper limit of the sum are adjusted depending on the sizes n, m, and k of the three registers
    x_wires, y_wires, and output_wires.

    Afterwards, we correct for the additional terms in three steps:
    - Add 2^m⋅(x+1)
    - Subtract 2^{n+m}+y
    - Add 2^n⋅y
    We are left with 2⋅(z+x⋅y), an even number, which we can divide by two by splitting off the
    least significant bit (which is exactly the one we appended initially) from the output register.
    """
    # We extend our output by one wire because we need to store 2x*y intermediately, instead
    # of x*y. This also multiplies the value stored in output_wires with two.
    output_wires = output_wires + [work_wires[0]]
    # The other work wires can be used for arithmetic building blocks
    work_wires = work_wires[1:]
    n = len(x_wires)
    m = len(y_wires)
    k = len(output_wires)

    # Controlled add-subtract loop
    for i, x_wire in enumerate(x_wires[::-1][:k]):
        # Slice the output wires according to the shift in control, and bounded by its own size,
        # and the size of the y_wires.
        output_msb = max(0, k - (m + 1 + i)) if output_wires_zeroed else 0
        output = output_wires[output_msb : k - i]
        _c_add_sub(x_wire, y_wires, output, work_wires)

    # Add 2^m(x+1)
    _add_plus_one(x_wires, output_wires[: k - m], work_wires)

    # Implement |y> |z> -> |y> |z-2^(n+m)-y>, i.e. subtract 2^(n+m)+y in four steps:
    # - Negate z: |y> |z> -> |y> |2^k-1-z>
    # - Add y: |y> |2^k-1-z> -> |y> |2^k-1-z+y>
    # - Add 2^(n+m) by incrementing the (k-(n+m)) most significant bits
    #   |y> |2^k-1-z+y> -> |y> |2^k-1-z+y+2^(n+m)>
    # - Negate z again: |y> |2^k-1-z+y+2^(n+m)> -> |y> |z-y-2^(n+m)>
    # The third step only is needed if k>n+m, otherwise those bits to increment do not exist.
    _ = [X(w) for w in output_wires]
    SemiAdder(y_wires, output_wires, work_wires)
    if k > n + m:
        increment_wires = output_wires[: k - n - m]
        _increment(increment_wires, work_wires)
    _ = [X(w) for w in output_wires]

    # Add (2^n·y) if 2^k > 2^n (otherwise it just vanishes in the modulus)
    if k > n:
        SemiAdder(y_wires, output_wires[: k - n], work_wires)

    # Note that dividing by two does not have to happen explicitly, because the registers are
    # not explicit return values.


def _out_multiplier_with_cache_condition(
    num_output_wires, num_work_wires, output_wires_zeroed, **_
):
    return num_work_wires >= 2 * num_output_wires - 1 and not output_wires_zeroed


def _out_multiplier_with_cache_resources(
    num_output_wires, num_x_wires, num_y_wires, num_work_wires, output_wires_zeroed, mod, **_
):  # pylint: disable=unused-argument,too-many-arguments
    new_num_work_wires = num_work_wires - num_output_wires
    mult_params = {
        "num_x_wires": num_x_wires,
        "num_y_wires": num_y_wires,
        "num_output_wires": num_output_wires,
        "num_work_wires": new_num_work_wires,
        "mod": mod,
        "output_wires_zeroed": True,
    }
    adder_params = {
        "num_x_wires": num_output_wires,
        "num_y_wires": num_output_wires,
        "num_work_wires": new_num_work_wires,
    }
    return {
        resource_rep(OutMultiplier, **mult_params): 1,
        resource_rep(SemiAdder, **adder_params): 1,
        adjoint_resource_rep(OutMultiplier, base_params=mult_params): 1,
    }


@register_condition(_out_multiplier_with_cache_condition)
@register_resources(_out_multiplier_with_cache_resources)
def _out_multiplier_with_cache(
    x_wires: WiresLike,
    y_wires: WiresLike,
    output_wires: WiresLike,
    mod: None,
    work_wires: WiresLike,
    output_wires_zeroed,
    **__,
):  # pylint: disable=unused-argument,too-many-arguments
    cache_wires = work_wires[: len(output_wires)]
    work_wires = work_wires[len(output_wires) :]
    OutMultiplier(
        x_wires, y_wires, cache_wires, mod=mod, work_wires=work_wires, output_wires_zeroed=True
    )
    SemiAdder(cache_wires, output_wires, work_wires)
    adjoint(OutMultiplier)(
        x_wires, y_wires, cache_wires, mod=mod, work_wires=work_wires, output_wires_zeroed=True
    )


add_decomps(
    OutMultiplier,
    _out_multiplier_with_qft,
    _out_multiplier_with_adder,
    _out_multiplier_with_caddsub,
    _out_multiplier_with_cache,
)
