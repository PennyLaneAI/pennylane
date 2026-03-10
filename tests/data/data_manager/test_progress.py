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

import pytest

import pennylane.data.data_manager.progress
import pennylane.data.data_manager.progress._default
from pennylane.data.data_manager.progress import Progress, Task
from pennylane.data.data_manager.progress._default import term

# pylint: disable=redefined-outer-name


@pytest.fixture
def disable_rich(request, monkeypatch):
    """Sets ``progress.make_progress_rich`` to ``None``, mocking the result of
    a failed import.

    Also skips the calling test if rich is not disabled, but is not
    installed.
    """
    if getattr(request, "param", True):
        monkeypatch.setattr(pennylane.data.data_manager.progress, "make_progress_rich", None)
        return True

    try:
        from pennylane.data.data_manager.progress._rich import RichProgress

        del RichProgress
    except ImportError:
        pytest.skip("'rich' is not installed")

    return False


class TestProgress:
    """Tests for :class:`pennylane.data.progress.Progress`."""

    @pytest.mark.parametrize(
        "disable_rich, expect_cls",
        [
            (True, "pennylane.data.data_manager.progress._default.DefaultProgress"),
            (False, "rich.progress.Progress"),
        ],
        indirect=["disable_rich"],
    )
    def test_init(self, disable_rich, expect_cls):
        """Test that ``__init__()`` uses the correct implementation based
        on the value of ``disable_rich``."""
        del disable_rich

        prog_cls = type(Progress().progress)
        assert f"{prog_cls.__module__}.{prog_cls.__name__}" == expect_cls

    @pytest.mark.usefixtures("disable_rich")
    @pytest.mark.parametrize("disable_rich", [True, False], indirect=True)
    @pytest.mark.parametrize("total", [100, None])
    def test_add_task(self, total):
        """Test that ``add_task()`` returns a new Task instance."""
        progress = Progress()
        task = progress.add_task(description="abc", total=total)

        assert isinstance(task, Task)

    @pytest.mark.usefixtures("disable_rich")
    @pytest.mark.parametrize("disable_rich", [True, False], indirect=True)
    def test_context(self):
        """Test that ``__enter__()`` returns the instance."""
        progress = Progress()
        with progress as progress_ctx:
            assert progress_ctx is progress


@pytest.mark.usefixtures("disable_rich")
class TestDefaultProgress:
    """Tests for :class:`pennylane.data.data_manger.progress._default.DefaultProgress`."""

    @pytest.fixture(autouse=True)
    def terminal_size(self, monkeypatch):
        """Patches terminal size for testing."""

        def get_terminal_size(fallback=None):
            # pylint: disable=unused-argument
            return (40, 4)

        monkeypatch.setattr(shutil, "get_terminal_size", get_terminal_size)

        yield (40, 4)

    @pytest.fixture()
    def progress(self):
        """Progress bar fixture."""
        yield Progress()

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

    def test_task_update_one_task(self, progress: Progress, capsys: pytest.CaptureFixture):
        """Test for updating with only one task."""
        task = progress.add_task(description="Task-1", total=100 * 1e6)

        with progress:
            task.update(advance=50 * 1e6)
            out, _ = capsys.readouterr()

            assert out == f"{term.erase_line()}Task-1 50.00/100.00 MB\r"

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

            assert (
                out.encode("utf-8")
                == (
                    f"{term.erase_line()}Task-1 50.00/100.00 MB\n"
                    f"{term.erase_line()}Task-2 0.00/300.00 MB\n"
                    f"{term.erase_line()}...\r{term.cursor_up(2)}"
                ).encode()
            )

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
