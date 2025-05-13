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
r"""Core class for qubit management in RE workflows."""


def _chech_active_manger():
    """Check that there is an active QubitManager instance to track wires"""
    if not QubitManager.in_active_context():
        raise ValueError("No active QubitManger found.")


def clean_qubits() -> int:
    _chech_active_manger()
    qm = QubitManager.active_context()
    return qm.clean_qubits


def tight_qubit_budget():
    _chech_active_manger()
    qm = QubitManager.active_context()
    return qm.tight_budget


def dirty_qubits() -> int:
    _chech_active_manger()
    qm = QubitManager.active_context()
    return qm.dirty_qubits


def borrowable_qubits() -> int:
    _chech_active_manger()
    qm = QubitManager.active_context()
    return qm.borrowable_qubits


def grab_qubits(num_qubits: int) -> None:
    _chech_active_manger()
    qm = QubitManager.active_context()
    qm.grab_clean_qubits(num_qubits)
    return


def free_qubits(num_qubits: int) -> None:
    _chech_active_manger()
    qm = QubitManager.active_context()
    qm.free_qubits(num_qubits)
    return


class QubitManager:

    _active_managers = []

    @classmethod
    def in_active_context(cls):
        return len(cls._active_managers) > 0

    @classmethod
    def active_context(cls):
        return cls._active_managers[-1] if cls.in_active_context else None

    def __init__(self, work_wires, tight_budget=False) -> None:

        if isinstance(work_wires, dict):
            clean_wires = work_wires["clean"]
            dirty_wires = work_wires["dirty"]
        else:
            clean_wires = work_wires
            dirty_wires = 0

        self.tight_budget = tight_budget
        self._logic_qubit_counts = 0
        self._clean_qubit_counts = clean_wires
        self._dirty_qubit_counts = dirty_wires

    def __str__(self):
        return f"QubitManager(clean={self.clean_qubits}, dirty={self.dirty_qubits}, logic={self.algo_qubits}, tight_budget={self.tight_budget})"

    def __repr__(self) -> str:
        return str(self)

    def __enter__(self):
        """Add the current instance of the QubitManager to the global stack"""
        self.__class__._active_managers.append(self)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Remove the current instance of the QubitManager from the global stack"""
        self.__class__._active_managers.pop()
        return

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return (
                (self.clean_qubits == other.clean_qubits) and 
                (self.dirty_qubits == other.dirty_qubits) and 
                (self.algo_qubits == other.algo_qubits)
            )

    @property
    def clean_qubits(self):
        return self._clean_qubit_counts

    @property
    def dirty_qubits(self):
        return self._dirty_qubit_counts

    @property
    def algo_qubits(self):
        return self._logic_qubit_counts

    @property
    def borrowable_qubits(self):
        return self.clean_qubits + self.dirty_qubits + self.algo_qubits

    @algo_qubits.setter
    def algo_qubits(self, count: int):
        self._logic_qubit_counts = count

    def allocate_qubits(self, num_qubits: int):
        self._clean_qubit_counts += num_qubits

    def grab_clean_qubits(self, num_qubits: int):
        available_clean = self.clean_qubits

        if num_qubits > available_clean:
            missing_qubits = num_qubits - available_clean
            if self.tight_budget:
                raise ValueError(
                    f"Not enough work qubits, trying to allocate {num_qubits} qubits with "
                    + f"only {available_clean} available qubits, please allocate more qubits."
                )

            self.allocate_qubits(missing_qubits)

        self._clean_qubit_counts -= num_qubits
        self._dirty_qubit_counts += num_qubits
        return

    def free_qubits(self, num_qubits: int):
        available_dirty = self.dirty_qubits

        if (num_qubits > available_dirty) and self.tight_budget:
            raise ValueError(
                f"Freeing more qubits than allocated. Allocated {available_dirty} dirty qubits, but releasing {num_qubits} qubits"
            )

        self._dirty_qubit_counts -= min(available_dirty, num_qubits)
        self._clean_qubit_counts += min(available_dirty, num_qubits)
