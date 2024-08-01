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
"""
Tests for :class:`pennylane.data.progress`
"""
import pytest
import rich.progress

from pennylane.data.data_manager.progress import Progress, Task
from pennylane.data.data_manager.progress._default import DefaultProgress


class TestProgress:
    """Tests for :class:`pennylane.data.progress.Progress`."""

    @pytest.fixture(params=[True, False])
    def progress(self, request):
        progress = Progress(use_rich=request.param)

        yield progress

    @pytest.mark.parametrize(
        "use_rich, expect_cls", [(False, DefaultProgress), (True, rich.progress.Progress)]
    )
    def test_init(self, use_rich, expect_cls):
        """Test that ``__init__()`` uses the correct implementation based
        on the value of ``use_rich``."""
        # pylint: disable=protected-access

        prog = Progress(use_rich=use_rich)
        assert isinstance(prog._progress, expect_cls)

    @pytest.mark.parametrize("total", [100, None])
    def test_add_task(self, progress, total):
        """Test that ``add_task()`` returns a new Task instance."""
        task = progress.add_task(description="abc", total=total)

        assert isinstance(task, Task)
