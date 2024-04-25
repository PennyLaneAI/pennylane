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
try:
    from rich.progress import (
        Progress,
        FileSizeColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
    )

    _PROGRESS_COLUMNS = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        FileSizeColumn(),
    )
except ImportError:
    _PROGRESS_COLUMNS = ()

    class Progress:  # pragma: no cover
        """A simple implementation of Progress that just writes to stdout with carriage-return."""

        __task_name = None
        __current = 0
        __total = None
        __fstring = None

        def __init__(self, **_):
            super().__init__()

        def __enter__(self, *_, **__):
            return self

        def __exit__(self, *_, **__):
            pass

        def add_task(self, task_name: str, total: float):
            """Implement an add_task method."""
            if self.__task_name is not None:
                raise ValueError("non-rich progress bar can only handle one task")
            self.__task_name = task_name
            self.__total = f"{total:0.2f}"
            self.__fstring = "{:>" + str(len(self.__total)) + ".2f}"
            print(f"{self.__task_name} {self.__fstring.format(0)}/{self.__total} KB", end="\r")

        def update(self, task_id, advance=None, completed=None):  # pylint:disable=unused-argument
            """Implement an update method."""
            if self.__task_name is None:
                raise ValueError("no task found to update")
            if advance:
                self.__current += advance
            elif completed:
                self.__current = completed
            print(
                f"{self.__task_name} {self.__fstring.format(self.__current)}/{self.__total} KB",
                end="\r",
            )


def progress():
    """Create a Progress object with some pre-selected defaults."""
    return Progress(*_PROGRESS_COLUMNS, refresh_per_second=1000, transient=False)
