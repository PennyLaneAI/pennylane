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
from threading import RLock


class QubitMgr:
    """
    The QubitMgr object maintains a list of active and inactive qubit wire indices used and for use
    during execution of a workload. Its purpose is to allow tracking of free qubit indices that
    are in the |0> state to participate in MCM-based workloads, under the assumption of reset
    upon measurement. Qubit wires indices will be tracked with a monotonically increasing set
    of values, starting from the initial input `start_idx`.

    This class assumes single-producer, single-consumer serialized CRUD operations only, and may
    not behave correctly in a concurrent execution environment.

    Args:
        num_qubits (int): Total number of wire indices to track.
        start_idx (int): Starting index of wires to track. Defaults to `0`.

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
    def num_qubits(self):
        """
        Return the total number of wire indices tracked by the manager.
        """
        return self._num_qubits

    @property
    def active(self):
        """
        Return the active wire indices. Any wire index in this set is unavailable for use, as it may
        be participating in existing algorithms and/or not be in a reset state.
        """
        with self._lock:
            return self._active

    @property
    def inactive(self):
        """
        Return the inactive wire indices. Any wire index in this set is available for use, and is
        assumed to be in a reset (|0>) state.
        """
        with self._lock:
            return self._inactive

    @property
    def all_qubits(self):
        """
        Return all active and inactive wire indices.
        """
        with self._lock:
            return self.inactive | self.active

    def acquire_qubit(self):
        """
        Return an available inactive wire index and make it active.
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

    def acquire_qubits(self, num_qubits: int):
        """
        Return num_qubits number of inactive wires and make them active.
        """
        indices = []
        if num_qubits > 0:
            with self._lock:
                while True:
                    indices.append(self.acquire_qubit())
                    if len(indices) == num_qubits:
                        break
        return indices

    def release_qubit(self, idx: int):
        """
        Return the active wire idx to the inactive pool.
        """
        with self._lock:
            try:
                self._active.remove(idx)
            except Exception as exc:
                raise RuntimeError(
                    f"Wire index {idx} not found in active set. Execution aborted."
                ) from exc
            self._inactive.add(idx)

    def release_qubits(self, indices: list[int]):
        """
        Return the active wire indices to the inactive pool.
        """
        with self._lock:
            for idx in indices:
                self.release_qubit(idx)

    def reserve_qubit(self, idx: int):
        """
        Explicitly request the wire index idx to be active.
        """
        with self._lock:
            if idx in self._inactive:
                self._inactive.remove(idx)
                self._active.add(idx)
            else:
                raise RuntimeError(
                    f"Qubit index {idx} not found in inactive set. Execution aborted."
                )
