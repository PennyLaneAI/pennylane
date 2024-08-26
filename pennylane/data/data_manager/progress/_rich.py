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
"""Progress bar using ``rich``."""

import rich.progress
from rich.progress import Progress as RichProgress


def make_progress() -> RichProgress:
    """Factory function for a progress instance."""
    return rich.progress.Progress(
        rich.progress.TextColumn("[progress.description]{task.description}"),
        rich.progress.BarColumn(),
        rich.progress.TaskProgressColumn(),
        rich.progress.TransferSpeedColumn(),
        rich.progress.DownloadColumn(),
        refresh_per_second=10,
        transient=False,
    )


__all__ = ["make_progress", "RichProgress"]
