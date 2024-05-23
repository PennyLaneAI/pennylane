# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains DatasetAttribute definition for pennylane operators, and lists
of operators."""

import json
import typing
from functools import lru_cache
from typing import Dict, FrozenSet, Generic, List, Type, TypeVar

import numpy as np

import pennylane as qml
from pennylane.data.base.attribute import DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Group, h5py
from pennylane.operation import Operator, Tensor

from ._wires import wires_to_json

Op = TypeVar("Op", bound=Operator)


class DatasetOperator(Generic[Op], DatasetAttribute[HDF5Group, Op, Op]):
    """``DatasetAttribute`` for ``pennylane.operation.Operator`` classes.

    Supports all operator types that meet the following conditions:
        - The ``__init__()`` method matches the signature of ``Operator.__init__``,
            or any additional arguments are optional and do not affect the value of
            the operator
        - The ``data`` and ``wires`` attributes will produce an identical copy of
            operator if passed into the classes' ``__init__()`` method. Generally,
            this means ``__init__()`` do not mutate the ``identifiers`` and ``wires``
            arguments.
        - Hyperparameters are not used or are automatically derived by ``__init__()``.

    Almost all operators meet these conditions. This type also supports serializing the
    ``Hamiltonian`` and ``Tensor`` operators.
    """

    type_id = "operator"

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: Op) -> HDF5Group:
        return self._ops_to_hdf5(bind_parent, key, [value])

    def hdf5_to_value(self, bind: HDF5Group) -> Op:
        return self._hdf5_to_ops(bind)[0]

    def _ops_to_hdf5(
        self, bind_parent: HDF5Group, key: str, value: typing.Sequence[Operator]
    ) -> HDF5Group:
        """Serialize op sequence ``value``, and create nested sequences for any
        composite ops in ``value``.

        Since operators are commonly used in larger composite operations, we handle
        sequences of operators as the default case. This allows for performant (in
        time and space) serialization of large and nested operator sums, products, etc.
        """
        bind = bind_parent.create_group(key)

        op_wire_labels = []
        op_class_names = []
        for i, op in enumerate(value):
            op_key = f"op_{i}"
            if isinstance(op, (qml.ops.Prod, qml.ops.SProd, qml.ops.Sum)):
                op = op.simplify()
            if type(op) not in self.consumes_types():
                raise TypeError(
                    f"Serialization of operator type '{type(op).__name__}' is not supported."
                )

            if isinstance(op, Tensor):
                self._ops_to_hdf5(bind, op_key, op.obs)
                op_wire_labels.append("null")
            elif isinstance(op, (qml.ops.Hamiltonian, qml.ops.LinearCombination)):
                coeffs, ops = op.terms()
                ham_grp = self._ops_to_hdf5(bind, op_key, ops)
                ham_grp["hamiltonian_coeffs"] = coeffs
                op_wire_labels.append("null")
            elif isinstance(op, (qml.ops.Prod, qml.ops.Sum)):
                self._ops_to_hdf5(bind, op_key, op.operands)
                op_wire_labels.append("null")
            elif isinstance(op, qml.ops.SProd):
                coeffs, ops = op.terms()
                sprod_grp = self._ops_to_hdf5(bind, op_key, ops)
                sprod_grp["sprod_scalar"] = coeffs
                op_wire_labels.append("null")
            else:
                bind[op_key] = op.data if len(op.data) else h5py.Empty("f")
                op_wire_labels.append(wires_to_json(op.wires))

            op_class_names.append(type(op).__name__)

        bind["op_wire_labels"] = op_wire_labels
        bind["op_class_names"] = op_class_names

        return bind

    def _hdf5_to_ops(self, bind: HDF5Group) -> List[Operator]:
        """Load list of serialized ops from ``bind``."""
        ops = []

        names_bind = bind["op_class_names"]
        wires_bind = bind["op_wire_labels"]
        op_class_names = [] if names_bind.shape == (0,) else names_bind.asstr()
        op_wire_labels = [] if wires_bind.shape == (0,) else wires_bind.asstr()
        with qml.QueuingManager.stop_recording():
            for i, op_class_name in enumerate(op_class_names):
                op_key = f"op_{i}"

                op_cls = self._supported_ops_dict()[op_class_name]
                if op_cls is Tensor:
                    ops.append(Tensor(*self._hdf5_to_ops(bind[op_key])))
                elif op_cls in (qml.ops.Hamiltonian, qml.ops.LinearCombination):
                    ops.append(
                        qml.Hamiltonian(
                            coeffs=list(bind[op_key]["hamiltonian_coeffs"]),
                            observables=self._hdf5_to_ops(bind[op_key]),
                        )
                    )
                elif op_cls in (qml.ops.Prod, qml.ops.Sum):
                    ops.append(op_cls(*self._hdf5_to_ops(bind[op_key])))
                elif op_cls is qml.ops.SProd:
                    ops.append(
                        qml.ops.s_prod(
                            scalar=bind[op_key]["sprod_scalar"][0],
                            operator=self._hdf5_to_ops(bind[op_key])[0],
                        )
                    )
                else:
                    wire_labels = json.loads(op_wire_labels[i])
                    op_data = bind[op_key]
                    if op_data.shape is not None:
                        params = np.zeros(shape=op_data.shape, dtype=op_data.dtype)
                        op_data.read_direct(params)
                        ops.append(op_cls(*params, wires=wire_labels))
                    else:
                        ops.append(op_cls(wires=wire_labels))

        return ops

    @classmethod
    @lru_cache(1)
    def _supported_ops_dict(cls) -> Dict[str, Type[Operator]]:
        """Returns a dict mapping ``Operator`` subclass names to the class."""
        return {op.__name__: op for op in cls.consumes_types()}
