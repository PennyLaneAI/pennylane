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

import shutil
from dataclasses import dataclass
from typing import Any, Optional

from pennylane.data.data_manager.progress._default import term


@dataclass
class Task:
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


class DefaultProgress:
    """Implements a progress bar."""

    tasks: list[Task]

    _active: bool
    _term_columns: int
    _term_lines: int
    _task_display_lines: list[str]
    _task_display_lines_max: int
    _description_width_max: int

    def __init__(self):
        self.tasks = []
        self._active = False

    def __enter__(self) -> "DefaultProgress":
        if self._active:
            raise RuntimeError("Progress context already active.")

        self._term_columns, self._term_lines = shutil.get_terminal_size()
        self._task_display_lines = []
        self._task_display_lines_max = self._term_lines - 2
        self._description_width_max = int(self._term_columns * 0.6)

        self._active = True

        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Any,
    ) -> None:
        self._print_final()
        self._active = False
        del self._task_display_lines

    def add_task(self, description: str, total: Optional[float] = None) -> int:
        """Add a task."""
        task = Task(description=description, total=total)

        self.tasks.append(Task(description=description, total=total))

        task_id = len(self.tasks) - 1
        if task_id < self._task_display_lines_max:
            self._task_display_lines.append(self._get_task_display_line(task))
        elif task_id == self._task_display_lines_max:
            self._task_display_lines.append("...")

        return len(self.tasks) - 1

    def update(
        self,
        task_id: int,
        completed: Optional[float] = None,
        total: Optional[float] = None,
        advance: Optional[float] = None,
    ):
        """Update task with given ``task_id`` and refresh its progress.

        Args:
            task_id: ID of task
            completed: Set the completed state of the task
            total: Set the total for the task
            advance: Advance the completion state of the task
        """
        task = self.tasks[task_id]
        task.update(advance=advance, completed=completed, total=total)

        if task_id < self._task_display_lines_max:
            self._task_display_lines[task_id] = self._get_task_display_line(task)

        self._print()

    def _get_task_display_line(self, task: Task) -> str:
        """Get display line for the task."""
        if task.total is None:
            progress_column = f"{task.completed / 1e6:.2f} MB"
        else:
            progress_column = f"{task.completed / 1e6:.2f}/{task.total / 1e6:.2f} MB"

        display = _truncate(
            f"{_truncate(task.description, self._description_width_max - 1).ljust(self._description_width_max)}"
            f"{progress_column}",
            self._term_columns,
        )

        return f"{term.erase_line()}{display}"

    def _print_final(self):
        """Print all tasks without any terminal control. Should be called when
        progress is complete."""
        print(*(self._get_task_display_line(task) for task in self.tasks), sep="\n", flush=True)

    def _print(self):
        """Prints up to ``_task_display_lines_max`` lines and returns the terminal cursor to
        the starting point."""
        print(
            *self._task_display_lines,
            sep="\n",
            end=f"\r{term.cursor_up(len(self._task_display_lines) - 1)}",
            flush=True,
        )


def make_progress() -> DefaultProgress:
    """Factory function for a progress instance."""
    return DefaultProgress()


def _truncate(s: str, maxlen: int) -> str:
    """If ``s`` is longer than ``maxlen``, truncate
    it and replace the last 3 characters with '...'.

    >>> _truncate("abcdef", 6)
    "abcdef"
    >>> _truncate("abcdef", 5)
    "ab..."
    """
    if len(s) > maxlen:
        return f"{s[:maxlen - 3]}..."

    return s


__all__ = ["make_progress", "DefaultProgress"]
