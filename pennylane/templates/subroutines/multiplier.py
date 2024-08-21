import numpy as np

import pennylane as qml
from pennylane.operation import Operation


def _mul_out_k_mod(k, x_wires, mod, work_wire_aux, wires_aux):
    """Performs x*k in the registers wires_aux"""
    op_list = []

    op_list.append(qml.QFT(wires=wires_aux))
    op_list.append(
        qml.ControlledSequence(qml.PhaseAdder(k, wires_aux, mod, work_wire_aux), control=x_wires)
    )
    op_list.append(qml.adjoint(qml.QFT(wires=wires_aux)))
    return op_list


class Multiplier(Operation):
    r"""Performs the Inplace Multiplication operation.

    This operator multiplies the integer :math:`k` modulo :math:`mod` in the computational basis:

    .. math::

        \text{Multiplier}(k,mod) |x \rangle = | x*k mod mod \rangle,

    The quantum circuit that represents the Multiplier operator is:


    Args:
        k (int): number that wants to be added
        wires (Sequence[int]): the wires the operation acts on. There are needed at least enough wires to represent :math:`k` and :math:`mod`.
        mod (int): modulo with respect to which the multiplication is performed, default value will be ``2^len(wires)``
        work_wires (Sequence[int]): the auxiliary wires to use for the multiplication modulo :math:`mod`

    **Example**

    Multiplication of two integers :math:`m=3` and :math:`k=4` modulo :math:`mod=7`. Note that to perform this multiplication using qml.Multiplier we need that :math:`m,k < mod`
    and that :math:`k` has inverse, :math:`k^-1`, modulo :math:`mod`. That means :math:`k*k^-1 modulo mod = 1`, which will only be possible if :math:`k` and :math:`mod` are coprime.

    .. code-block::
        x = 3
        k = 4
        mod = 7
        wires_m =[0,1,2]
        work_wires=[3,4,5,6,7]
        dev = qml.device("default.qubit", shots=1)
        @qml.qnode(dev)
        def multiplier_modulo(x, k, mod, wires_m, work_wires):
            qml.BasisEmbedding(x, wires=wires_m)
            qml.Multiplier(k, wires_m, mod, work_wires)
            return qml.sample(wires=wires_m)

    .. code-block:: pycon

        >>> print(f"The ket representation of {m} * {k} mod {mod} is {multiplier_modulo(m, k, mod, wires_m, work_wires)}")
        The ket representation of 3 * 4 mod 7 is [1 0 1]

    We can see that the result [1 0 1] corresponds to 5, which comes from :math:`3+4=12 \longrightarrow 12 mod 7 = 5`.
    """

    grad_method = None

    def __init__(
        self, k, x_wires, mod=None, work_wires=None, id=None
    ):  # pylint: disable=too-many-arguments
        if any(wire in work_wires for wire in x_wires):
            raise ValueError("None wire in work_wires should be included in x_wires.")
        if mod is None:
            mod = 2 ** len(x_wires)
        if mod != 2 ** len(x_wires):
            if len(work_wires) < (len(x_wires) + 2):
                raise ValueError("Multiplier needs as many work_wires as x_wires plus two.")
        elif len(work_wires) < len(x_wires):
            raise ValueError("Multiplier needs as many work_wires as x_wires.")
        k = k % mod
        if (not hasattr(x_wires, "__len__")) or (mod > 2 ** len(x_wires)):
            raise ValueError("Multiplier must have at least enough wires to represent mod.")

        if np.gcd(k, mod) != 1:
            raise ValueError("Since k has no inverse modulo mod, the work_wires cannot be cleaned.")

        self.hyperparameters["k"] = k
        self.hyperparameters["mod"] = mod
        self.hyperparameters["work_wires"] = qml.wires.Wires(work_wires)
        self.hyperparameters["x_wires"] = qml.wires.Wires(x_wires)
        all_wires = qml.wires.Wires(x_wires) + qml.wires.Wires(work_wires)
        super().__init__(wires=all_wires, id=id)

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
            for key in ["x_wires", "work_wires"]
        }

        return Multiplier(
            self.hyperparameters["k"],
            new_dict["x_wires"],
            self.hyperparameters["mod"],
            new_dict["work_wires"],
        )

    @property
    def wires(self):
        """All wires involved in the operation."""
        return self.hyperparameters["x_wires"] + self.hyperparameters["work_wires"]

    def decomposition(self):  # pylint: disable=arguments-differ
        return self.compute_decomposition(**self.hyperparameters)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @staticmethod
    def compute_decomposition(k, x_wires, mod, work_wires):
        r"""Representation of the operator as a product of other operators.
        Args:
            k (int): number that wants to be added
            mod (int): modulo of the sum
            work_wires (Sequence[int]): the auxiliary wires to use for the sum modulo :math:`mod` when :math:`mod \neq 2^{\textrm{len(wires)}}`
            wires (Sequence[int]): the wires the operation acts on
        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        >>> qml.Multiplier.compute_decomposition(k=3,mod=8,wires=[0,1,2],work_wires=[3,4,5,6,7])
        [QFT(wires=[5, 6, 7]),
        ControlledSequence(PhaseAdder(wires=[5, 6, 7]), control=[0, 1, 2]),
        Adjoint(QFT(wires=[5, 6, 7])),
        SWAP(wires=[0, 5]),
        SWAP(wires=[1, 6]),
        SWAP(wires=[2, 7]),
        Adjoint(Adjoint(QFT(wires=[5, 6, 7]))),
        Adjoint(ControlledSequence(PhaseAdder(wires=[5, 6, 7]), control=[0, 1, 2])),
        Adjoint(QFT(wires=[5, 6, 7]))]
        """

        op_list = []
        if mod != 2 ** len(x_wires):
            work_wire_aux = work_wires[:1]
            wires_aux = work_wires[1:]
            wires_aux_swap = wires_aux[1:]
        else:
            work_wire_aux = None
            wires_aux = work_wires[:3]
            wires_aux_swap = wires_aux
        op_list.extend(_mul_out_k_mod(k, x_wires, mod, work_wire_aux, wires_aux))
        for i in range(len(x_wires)):
            op_list.append(qml.SWAP(wires=[x_wires[i], wires_aux_swap[i]]))
        inv_k = pow(k, -1, mod)
        op_list.extend(qml.adjoint(_mul_out_k_mod)(inv_k, x_wires, mod, work_wire_aux, wires_aux))
        return op_list
