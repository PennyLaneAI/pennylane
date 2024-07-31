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
    from pennylane.data.data_manager.progress._rich import make_progress
except ImportError:
    from pennylane.data.data_manager.progress._default import make_progress


from typing import Any, Optional


class Task:

    def __init__(self, _progress: Any, _task_id: Any):
        self._progress = _progress
        self._task_id = _task_id

    def update(
        self,
        *,
        completed: Optional[float] = None,
        advance: Optional[float] = None,
        total: Optional[float] = None,
    ):
        self._progress.update(self._task_id, completed=completed, total=total, advance=advance)


class Progress:
    """"""

    def __init__(self) -> None:
        self._progress = make_progress()

    def __enter__(self) -> "Progress":
        self._progress.__enter__()

        return self

    def __exit__(self, *args):
        return self._progress.__exit__(*args)

    def add_task(self, description: str, total: Optional[float] = None) -> Task:
        task_id = self._progress.add_task(description=description, total=total)

        return Task(self._progress, task_id)


__all__ = ["Progress", "Task"]
