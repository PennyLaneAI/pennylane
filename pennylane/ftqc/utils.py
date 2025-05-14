# Copyright 2025 Xanadu Quantum Technologies Inc.

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
This module contains utility data-structures and algorithms supporting functionality in the
ftqc module.
"""
from threading import RLock
from typing import List, Tuple

import pennylane as qml
from pennylane import numpy as np


class QubitMgr:
    r"""
    The ``QubitMgr`` object maintains a list of active and inactive qubit wire indices used and for use
    during execution of a workload. Its purpose is to allow tracking of free qubit indices that
    are in the :math:`\vert 0 \rangle` state to participate in MCM-based workloads, under the assumption of reset
    upon measurement. Qubit wires indices will be tracked with a monotonically increasing set
    of values, starting from the initial input ``start_idx``.

    Args:
        num_qubits (int): Total number of wire indices to track.
        start_idx (int): Starting index of wires to track. Defaults to 0.

    **Example:**
        The following MBQC example workload uses the ``QubitMgr`` to assist with recycling of indices
        between iterations

        .. code-block:: python3

            import pennylane as qml
            from pennylane.ftqc import QubitGraph, diagonalize_mcms, generate_lattice, measure_x, measure_y
            dev = qml.device('null.qubit')

            @qml.qnode(dev, mcm_method="one-shot")
            def circuit_mbqc(start_state, angles):
                q_mgr = QubitMgr(num_qubits=5, start_idx=0)
                input_idx = q_mgr.acquire_qubit()

                # prep input node
                qml.StatePrep(start_state, wires=[input_idx])

                # prep and consume graph state iteratively
                for i in range(num_iter):
                    # Acquire 4 free qubit indices
                    graph_wires = q_mgr.acquire_qubits(4)

                    # Denote the index for the final output state
                    output_idx = graph_wires[-1]

                    # Prepare the state
                    qml.ftqc.GraphStatePrep(lattice.graph, wires=graph_wires)

                    # entangle input and graph using first qubit
                    qml.CZ([input_idx, graph_wires[0]])

                    # MBQC Z rotation: X, X, +/- angle, X
                    # Reset operations allow qubits to be returned to the pool
                    m0 = measure_x(input_idx, reset=True)
                    m1 = measure_x(graph_wires[0], reset=True)
                    m2 = cond_measure(m1, partial(measure, angle=angle, reset=True), partial(measure, angle=-angle, reset=True))(plane="XY", wires=graph_wires[1])
                    m3 = measure_x(graph_wires[2], reset=True)

                    # corrections based on measurement outcomes
                    qml.cond((m0+m2)%2, qml.Z)(graph_wires[3])
                    qml.cond((m1+m3)%2, qml.X)(graph_wires[3])

                    # The input qubit can be freed and the output qubit becomes the next iteration's input
                    q_mgr.release_qubit(input_idx)
                    input_idx = output_idx

                    # We can now free all but the last qubit, which has become the new input_idx
                    q_mgr.release_qubits(graph_wires[0:-1])

                # Perform the measurements on the output qubit from the last iteration
                return qml.expval(X(output_idx)), qml.expval(Y(output_idx)), qml.expval(Z(output_idx))

        For each loop iteration, the measured and reset wire labels are returned to the ``QubitMgr`` instance, which are then reallocated
        on the next step, which when combined with the MCM resets allows for qubit index recycling.

    """

    def __init__(self, num_qubits: int = 0, start_idx: int = 0):
        # All resources are protected via a re-entrant lock, to ensure exclusive access when
        # acquiring/accessing/releasing qubits.
        self._lock = RLock()
        self._num_qubits = num_qubits
        self._active = set()
        is_valid = lambda x: (isinstance(x, int) and x >= 0)
        if is_valid(num_qubits) and is_valid(start_idx):
            self._inactive = set(range(start_idx, start_idx + num_qubits, 1))
        else:
            raise TypeError(
                f"Index counts and starting values must be positive integers. Received {num_qubits} and {start_idx}"
            )

    def __repr__(self):
        return f"QubitMgr(num_qubits={self.num_qubits}, active={self.active}, inactive={self.inactive})"

    @property
    def num_qubits(self) -> int:
        """Defines the total number of wire indices tracked by the manager.

        Returns:
            int: total number of qubit wire indices
        """

        return self._num_qubits

    @property
    def active(self) -> set:
        """
        Defines the active wire indices. Any wire index in this set is unavailable for use, as it may
        be participating in existing algorithms and/or not be in a reset state.

        Returns:
            set[int]: active wire indices
        """
        with self._lock:
            return self._active

    @property
    def inactive(self) -> set:
        r"""
        Defines the inactive wire indices. Any wire index in this set is available for use, and is
        assumed to be in a reset (:math:`\vert 0 \rangle`) state.

        Returns:
            set[int]: inactive wire indices
        """

        with self._lock:
            return self._inactive

    @property
    def all_qubits(self) -> set:
        """
        Defines all active and inactive wire indices.

        Returns:
            set[int]: union of active and inactive wire indices
        """
        with self._lock:
            return self.inactive | self.active

    def acquire_qubit(self) -> int:
        """
        Acquires an available qubit wire index from the inactive pool, and makes it active.
        If there are no inactive qubits available a RuntimeError will be raised.

        Returns:
            int: newly activated qubit wire index
        """
        with self._lock:
            try:
                idx = self._inactive.pop()
                self._active.add(idx)
                return idx
            except Exception as exc:
                raise RuntimeError(
                    "Cannot allocate any additional wire indices. Execution aborted."
                ) from exc

    def acquire_qubits(self, num_qubits: int) -> list[int]:
        """
        Acquires num_qubits qubit wire indices from the inactive pool, and makes them active.
        If there are no inactive qubits available a RuntimeError will be raised.

        .. seealso:: :meth:`~.QubitMgr.acquire_qubit`.

        Returns:
            list[int]: newly activated qubit wire indices
        """
        indices = []
        if num_qubits > 0:
            with self._lock:
                while True:
                    indices.append(self.acquire_qubit())
                    if len(indices) == num_qubits:
                        break
        return indices

    def release_qubit(self, idx: int) -> None:
        """
        Release an active qubit wire index, idx, from the active pool, and makes it inactive.
        If idx is not in the active pool a RuntimeError will be raised.
        """
        with self._lock:
            try:
                self._active.remove(idx)
            except Exception as exc:
                raise RuntimeError(
                    f"Wire index {idx} not found in active set. Execution aborted."
                ) from exc
            self._inactive.add(idx)

    def release_qubits(self, indices: list[int]) -> None:
        """
        Release the list of active qubit wire indices, indices, from the active pool, and makes them inactive.
        If any of the given indices are not in the active pool a RuntimeError will be raised.

        .. seealso:: :meth:`~.QubitMgr.release_qubit`.
        """
        with self._lock:
            for idx in indices:
                self.release_qubit(idx)

    def reserve_qubit(self, idx: int) -> None:
        """
        Explicitly reserve the qubit wire index, idx, to be active.
        If given index is not in the active pool a RuntimeError will be raised.
        """
        with self._lock:
            if idx in self._inactive:
                self._inactive.remove(idx)
                self._active.add(idx)
            else:
                raise RuntimeError(
                    f"Qubit index {idx} not found in inactive set. Execution aborted."
                )


_ENCODE_XZ_OPS = {
    qml.I: (0, 0),
    qml.X: (1, 0),
    qml.Y: (1, 1),
    qml.Z: (0, 1),
}

_paulis = frozenset({qml.X, qml.Y, qml.Z, qml.I})


def pauli_encode_xz(op: qml.operation.Operator) -> Tuple[np.uint8, np.uint8]:
    """
    Encode a `Pauli` operator to its `xz` representation up to a global phase, i.e., :math:`encode_{xz}(Pauli)=(x,z)=X^xZ^z)`, where
    :math:`x` is the exponent of the :class:`~pennylane.X` and :math:`z` is the exponent of
    the :class:`~pennylane.Z`, meaning :math:`encode_{xz}(I) = (0, 0)`, :math:`encode_{xz}(X) = (1, 0)`,
    :math:`encode_{xz}(Y) = (1, 1)` and :math:`encode_{xz}(Z) = (0, 1)`.

    Args:
        op (qml.operation.Operator): A Pauli operator.

    Return:
        A tuple of xz encoding data, :math:`x` is the exponent of the :class:`~pennylane.X`, :math:`z` is the exponent of
    the :class:`~pennylane.Z`.
    """

    if op in _paulis:
        return _ENCODE_XZ_OPS[op]
    raise NotImplementedError(f"{op.name} gate does not support xz encoding.")


def pauli_prod_to_xz(ops: List[qml.operation.Operator]) -> Tuple[np.uint8, np.uint8]:
    """
    Get the result of a product of list of Pauli operators. The result is encoded with `xz` representation.

    Args:
        ops (List[qml.operation.Operator]): A list of Pauli operators with the same target wire.

    Return:
        A tuple of `xz` encoding data, :math:`x` is the exponent of the :class:`~pennylane.X`, :math:`z` is the exponent of
    the :class:`~pennylane.Z`.
    """

    if len(ops) == 0:
        raise ValueError("Please ensure that a valid list of operators are passed to the method.")

    res_x, res_z = pauli_encode_xz(ops.pop())

    while len(ops) > 0:
        op_x, op_z = pauli_encode_xz(ops.pop())
        res_x ^= op_x
        res_z ^= op_z

    return (res_x, res_z)
