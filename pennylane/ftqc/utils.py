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

"""This module contains utility data-structures and algorithms supporting functionality in the
ftqc module.
"""


class QubitMgr:
    """
    The QubitMgr object maintains a list of active and inactive qubit indices used and for use
    during execution of a workload. Its purpose is to allow tracking of free qubit indices that
    are in the |0> state to participate in MCM-based workloads, under the assumption of reset
    upon measurement.

    This class assumes single-producer, single-consumer serialized CRUD operations only, and may 
    not behave correctly in a concurrent execution environment.
    """

    def __init__(self, num_qubits: int = 0, start_idx: int = 0):
        self._num_qubits = num_qubits
        self._inactive = set(range(start_idx, start_idx + num_qubits, 1))
        self._active = set()

    @property
    def num_qubits(self):
        """
        Return the total number of qubits tracked by the manager.
        """
        return self._num_qubits

    @property
    def active(self):
        """
        Return the active qubit indices. Any qubit in this set is unavailable for use, as it may
        be participating in existing algorithms and/or not be in a reset state.
        """
        return self._active

    @property
    def inactive(self):
        """
        Return the inactive qubit indices. Any qubit in this set is available for use, and is
        assumed to be in a reset (|0>) state.
        """
        return self._inactive

    @property
    def all_qubits(self):
        """
        Return all active and inactive qubit indices.
        """
        return self.inactive | self.active

    def acquire_qubit(self):
        """
        Return an available inactive qubit index and make it active.
        """
        try:
            idx = self._inactive.pop()
            self._active.add(idx)
            return idx
        except Exception as exc:
            raise RuntimeError("Cannot allocate any additional qubits. Execution aborted.") from exc

    def acquire_qubits(self, num_qubits):
        """
        Return num_qubits number of inactive qubits and make them active.
        """
        indices = []
        for i in range(num_qubits):
            indices.append(self.get_qubit())
        return indices

    def release_qubit(self, idx: int):
        """
        Return the active qubit idx to the inactive pool.
        """
        try:
            self._active.remove(idx)
        except Exception as exc:
            raise RuntimeError(
                f"Qubit index {idx} not found in active set. Execution aborted."
            ) from exc
        self._inactive.add(idx)

    def release_qubits(self, indices: list[int]):
        """
        Return the active qubit indices to the inactive pool.
        """
        for idx in indices:
            self.free_qubit(idx)

    def reserve_qubit(self, idx: int):
        """
        Explicitly request the qubit index idx to be active.
        """
        if idx in self._inactive:
            self._inactive.remove(idx)
            self._active.add(idx)
        else:
            raise RuntimeError(f"Qubit index {idx} not found in inactive set. Execution aborted.")
