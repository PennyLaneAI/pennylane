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
"""Fallback implementation of a progress bar, used when ``rich`` is not available."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class _Task:
    """Data class for progress bar state."""

    description: str
    completed: float = 0
    total: Optional[float] = None

    def update(
        self,
        *,
        advance: Optional[float] = None,
        completed: Optional[float] = None,
        total: Optional[float] = None,
    ) -> None:
        """Update the state of the progress bar and set the display
        string."""
        if completed:
            self.completed = completed
        if advance:
            self.completed += advance
        if total:
            self.total = total

        if self.total is None:
            self.display = f"{self.description} {self.completed / 1e6:.2f} MB"
        else:
            self.display = (
                f"{self.description} {self.completed / 1e6:.2f}/{self.total / 1e6:.2f} MB"
            )

    @property
    def finished(self) -> bool:
        return self.total is not None and (self.total - self.completed < 0.1)

    def __post_init__(self):
        """Initialize the display string."""
        self.update()


class Progress:
    """Progress bar implementation compatible with Rich."""

    def __init__(self, **_):
        self._tasks: list[_Task] = []
        self._finished_task_ids = set()
        self._total_downloaded_mb: float = 0

    def __enter__(self, *_, **__):
        pass

    def __exit__(self, *_, **__):
        pass

    def add_task(self, description: str, total: Optional[float] = None) -> int:
        """Add a new progress bar task and return its task id."""
        self._tasks.append(
            _Task(
                description=description,
                total=total,
            )
        )

        return len(self._tasks) - 1

    def update(
        self,
        task_id: int,
        *,
        advance: Optional[float] = None,
        completed: Optional[float] = None,
        total: Optional[float] = None,
    ):  # pylint:disable=unused-argument
        """Implement an update method."""
        task = self._tasks[task_id]
        task.update(advance=advance, completed=completed, total=total)

        if task.finished and task_id not in self._finished_task_ids:
            self._finished_task_ids.add(task_id)
            self._total_downloaded_mb += task.total / 1e6

        self._print()

    def _print(self):
        print(
            f"Downloading datasets: {len(self._finished_task_ids)}/{len(self._tasks)} complete ({self._total_downloaded_mb:.2f} MB)",
            end="\r",
        )


def make_progress() -> Progress:
    return Progress()


__all__ = [
    "make_progress",
]
