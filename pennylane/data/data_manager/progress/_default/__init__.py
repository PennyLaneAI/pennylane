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
    total: float | None = None

    def update(
        self,
        *,
        advance: float | None = None,
        completed: float | None = None,
        total: float | None = None,
    ) -> None:
        """Update the state of the progress bar and set the display
        string."""
        if completed:
            self.completed = completed
        if advance:
            self.completed += advance
        if total:
            self.total = total


class TerminalInfo:
    """Contains information on  the dimensions of
    the terminal."""

    # pylint: disable=too-few-public-methods

    def __init__(self):
        self.columns, self.lines = shutil.get_terminal_size()
        self.max_display_lines = self.lines - 2
        self.description_len_max = int(self.columns * 0.6) - 1


class DefaultProgress:
    """Implements a progress bar."""

    tasks: list[Task]

    def __init__(self):
        self.tasks = []

        self._active = False
        self._term_info = TerminalInfo()
        self._task_display_lines = []
        self._curr_longest_description = 0

    def __enter__(self) -> "DefaultProgress":
        self._active = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        self._print_final()
        self._active = False

    def add_task(self, description: str, total: float | None = None) -> int:
        """Add a task."""
        description = _truncate(description, self._term_info.description_len_max)
        self.tasks.append(Task(description=description, total=total))

        self._curr_longest_description = max(self._curr_longest_description, len(description))
        self.refresh()

        return len(self.tasks) - 1

    def refresh(self, task_id: int | None = None):
        """Refresh display liens for one or all tasks."""
        if task_id is None:
            self._task_display_lines.clear()
            self._task_display_lines.extend(
                self._get_task_display_line(task)
                for task in self.tasks[: self._term_info.max_display_lines]
            )

            if len(self.tasks) > self._term_info.max_display_lines:
                self._task_display_lines.append(f"{term.erase_line()}...")

        elif task_id < self._term_info.max_display_lines:
            self._task_display_lines[task_id] = self._get_task_display_line(self.tasks[task_id])

    # pylint: disable = too-many-arguments
    def update(
        self,
        task_id: int,
        *,
        completed: float | None = None,
        total: float | None = None,
        advance: float | None = None,
        refresh: bool = False,
    ):
        """Update task with given ``task_id`` and refresh its progress.

        Args:
            task_id: ID of task
            completed: Set the completed state of the task
            total: Set the total for the task
            advance: Advance the completion state of the task
            refresh: Included for compatability with ``rich.Progress``, has no effect
        """
        del refresh
        task = self.tasks[task_id]
        task.update(advance=advance, completed=completed, total=total)

        self.refresh(task_id)

        if self._active:
            self._print()

    def _get_task_display_line(self, task: Task) -> str:
        """Get display line for the task."""
        if task.total is None:
            progress_column = f"{task.completed / 1e6:.2f} MB"
        else:
            progress_column = f"{task.completed / 1e6:.2f}/{task.total / 1e6:.2f} MB"

        display = _truncate(
            f"{task.description.ljust(self._curr_longest_description + 1)}" f"{progress_column}",
            self._term_info.columns,
        )

        return f"{term.erase_line()}{display}"

    def _print_final(self):
        """Print all tasks without any terminal control. Should be called when
        progress is complete."""
        print(*(self._get_task_display_line(task) for task in self.tasks), sep="\n", flush=True)

    def _print(self):
        """Prints up to ``_task_display_lines_max`` lines and returns the terminal cursor to
        the starting point."""
        if len(self._task_display_lines) > 1:
            end = f"\r{term.cursor_up(len(self._task_display_lines) - 1)}"
        else:
            end = "\r"

        print(
            *self._task_display_lines,
            sep="\n",
            end=end,
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
