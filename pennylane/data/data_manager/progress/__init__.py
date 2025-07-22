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
"""A library for showing loading progress, using ``rich`` or basic stdout."""

from typing import Any, Optional

from pennylane.data.data_manager.progress._default import make_progress as make_progress_default

try:
    from pennylane.data.data_manager.progress._rich import make_progress as make_progress_rich
except ImportError:  # pragma: no cover
    make_progress_rich = None


class Task:  # pylint: disable=too-few-public-methods
    """Represents progress display for a single dataset download."""

    def __init__(self, _task_id: Any, _progress: Any):
        """Private constructor."""
        self._progress = _progress
        self._task_id = _task_id

    def update(
        self,
        *,
        completed: float | None = None,
        advance: float | None = None,
        total: float | None = None,
    ):
        """Update download state.

        Args:
            advance: Adds to number of bytes downloaded so far
            completed: Sets the number of bytes downloaded so far
            total: Sets the total number of bytes for the download
        """
        self._progress.update(
            self._task_id, completed=completed, total=total, advance=advance, refresh=True
        )


class Progress:
    """Displays dataset download progress on the terminal. Will use
    ``rich.progress.Progress`` if available, otherwise it will fall back to the
    default implementation.

    Must be used as a context manager to ensure correct output.
    """

    def __init__(self) -> None:
        """Initialize progress."""
        if make_progress_rich is not None:
            self.progress = make_progress_rich()
        else:
            self.progress = make_progress_default()

    def __enter__(self) -> "Progress":
        """Enter progress context."""
        self.progress.__enter__()

        return self

    def __exit__(self, *args):
        """Exit progress context."""
        return self.progress.__exit__(*args)

    def add_task(self, description: str, total: float | None = None) -> Task:
        """Add a task.

        Args:
            description: Description for the task
            total: Total size of the dataset download in bytes, if available.
        """
        task_id = self.progress.add_task(description=description, total=total)

        return Task(task_id, self.progress)


__all__ = ["Progress", "Task"]
