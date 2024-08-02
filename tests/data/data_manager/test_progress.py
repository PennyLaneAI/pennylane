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
import shutil

import pennylane.data.data_manager.progress
import pennylane.data.data_manager.progress._default
import pytest
import rich.progress
from pennylane.data.data_manager.progress import Progress, Task
from pennylane.data.data_manager.progress._default import DefaultProgress, term


class TestProgress:
    """Tests for :class:`pennylane.data.progress.Progress`."""

    @pytest.fixture(params=[True, False])
    def progress(self, request):
        """Progress bar fixture."""
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

    def test_init_no_rich(self, monkeypatch):
        """Test that an ImportError is raised if rich is requested but is
        not installed."""
        monkeypatch.setattr(pennylane.data.data_manager.progress, "make_progress_rich", None)

        with pytest.raises(
            ImportError, match=r"Module 'rich' is not installed. Install it with 'pip install rich'"
        ):
            Progress(use_rich=True)

    @pytest.mark.parametrize("total", [100, None])
    def test_add_task(self, progress, total):
        """Test that ``add_task()`` returns a new Task instance."""
        task = progress.add_task(description="abc", total=total)

        assert isinstance(task, Task)

    def test_context(self, progress):
        """Test that ``__enter__()`` returns the instance."""
        with progress as progress_ctx:
            assert progress_ctx is progress


class TestDefaultProgress:
    """Tests for :class:`pennylane.data.data_manger.progress._default.DefaultProgress`."""

    @pytest.fixture(autouse=True)
    def terminal_size(self, monkeypatch):
        """Patches terminal size for testing."""

        def get_terminal_size(fallback=None):
            return (40, 4)

        monkeypatch.setattr(shutil, "get_terminal_size", get_terminal_size)

        yield (40, 4)

    @pytest.fixture()
    def progress(self):
        """Progress bar fixture."""
        yield Progress(use_rich=False)

    def test_task_update_with_total(self, progress: Progress, capsys: pytest.CaptureFixture):
        """Tests for updating a task with a total."""
        task_1 = progress.add_task(description="Task-1", total=100 * 1e6)
        progress.add_task(description="Task-2", total=None)

        with progress:
            task_1.update(advance=50 * 1e6)
            out, _ = capsys.readouterr()

            assert out == (
                f"{term.erase_line()}Task-1 50.00/100.00 MB\n"
                f"{term.erase_line()}Task-2 0.00 MB\r{term.cursor_up(1)}"
            )

    @pytest.mark.parametrize(
        "kwds, numbers",
        [
            ({"advance": 50 * 1e6}, "50.00"),
            ({"completed": 100 * 1e6, "total": 100 * 1e6}, "100.00/100.00"),
        ],
    )
    def test_task_update_without_total(
        self, progress: Progress, capsys: pytest.CaptureFixture, kwds, numbers
    ):
        """Tests for updating a task without a total."""
        progress.add_task(description="Task-1", total=100 * 1e6)
        task_2 = progress.add_task(description="Task-2", total=None)

        with progress:
            task_2.update(**kwds)
            out, _ = capsys.readouterr()

            assert out == (
                f"{term.erase_line()}Task-1 0.00/100.00 MB\n"
                f"{term.erase_line()}Task-2 {numbers} MB\r{term.cursor_up(1)}"
            )

    def test_task_lines_truncated(self, progress: Progress, capsys: pytest.CaptureFixture):
        """Tests that task lines will not be printed if there is not enough
        terminal lines available."""
        task_1 = progress.add_task(description="Task-1", total=100e6)
        progress.add_task(description="Task-2", total=300e6)

        invisible = progress.add_task(description="Task-3", total=100e6)
        progress.add_task(description="Task-4", total=100e6)

        with progress:
            task_1.update(advance=50e6)
            out, _ = capsys.readouterr()

            invisible.update(advance=50e6)

            assert out.encode("utf-8") == (
                f"{term.erase_line()}Task-1 50.00/100.00 MB\n"
                f"{term.erase_line()}Task-2 0.00/300.00 MB\n"
                f"{term.erase_line()}...\r{term.cursor_up(2)}"
            ).encode("utf-8")

    def test_description_truncated(self, progress: Progress, capsys: pytest.CaptureFixture):
        """Test that long task descriptions will be truncated."""
        task_1 = progress.add_task(description="Task-1-with-a-too-long-name", total=100e6)
        progress.add_task(description="Task-2", total=300e6)

        with progress:
            task_1.update(advance=50e6)
            out, _ = capsys.readouterr()

            assert out == (
                f"{term.erase_line()}Task-1-with-a-too-lo... 50.00/100.00 MB\n"
                f"{term.erase_line()}Task-2                  0.00/300.00 MB\r{term.cursor_up(1)}"
            )

    def test_final_print(self, progress: Progress, capsys: pytest.CaptureFixture):
        """Test that all task lines are printed, without cursor control codes,
        when the progress bar exits."""
        progress.add_task(description="Task-1", total=100e6)
        progress.add_task(description="Task-2", total=300e6)

        task_3 = progress.add_task(description="Task-3", total=100e6)

        with progress:
            task_3.update(advance=50e6)
            capsys.readouterr()

        out, _ = capsys.readouterr()

        assert out == (
            f"{term.erase_line()}Task-1 0.00/100.00 MB\n"
            f"{term.erase_line()}Task-2 0.00/300.00 MB\n"
            f"{term.erase_line()}Task-3 50.00/100.00 MB\n"
        )
