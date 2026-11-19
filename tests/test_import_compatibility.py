# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Import isolation checks for the PennyLane2 namespace."""

import os
import subprocess
import sys
from pathlib import Path


def _run_python(code, extra_path=None):
    """Run without inheriting the parent's imported PennyLane modules."""
    package_root = str(Path(__file__).resolve().parents[1])
    paths = [package_root]
    if extra_path is not None:
        paths.insert(0, str(extra_path))
    if os.environ.get("PYTHONPATH"):
        paths.append(os.environ["PYTHONPATH"])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(paths)}
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_classic_import_rejected_before_initialization(tmp_path):
    """A loaded Classic module is rejected before the numerical backend is initialized."""
    classic = tmp_path / "pennylane"
    classic.mkdir()
    (classic / "__init__.py").write_text('generation = "classic"\n', encoding="utf-8")
    result = _run_python(
        """
import sys
import pennylane
try:
    import pennylane2
except ImportError as error:
    assert "Classic PennyLane" in str(error)
    assert "restart the notebook kernel" in str(error)
    assert "pennylane2.math" not in sys.modules
    assert pennylane.generation == "classic"
else:
    raise AssertionError("Mixed imports were accepted")
""",
        extra_path=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_installed_classic_is_not_imported(tmp_path):
    """Classic's presence on disk does not prevent a beta-only process."""
    classic = tmp_path / "pennylane"
    classic.mkdir()
    (classic / "__init__.py").write_text(
        'raise RuntimeError("Classic must not be imported")\n', encoding="utf-8"
    )
    result = _run_python(
        """
import importlib.util
import sys
assert importlib.util.find_spec("pennylane") is not None
import pennylane2 as qp
assert qp.RX(0.2, wires=0).name == "RX"
assert "pennylane" not in sys.modules
""",
        extra_path=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_classic_import_rejected_after_failed_initialization(tmp_path):
    """Retrying a failed beta import still rejects a subsequently imported Classic."""
    classic = tmp_path / "pennylane"
    classic.mkdir()
    (classic / "__init__.py").write_text('generation = "classic"\n', encoding="utf-8")
    result = _run_python(
        """
import importlib.abc
import sys

class FailInitialization(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pennylane2.exceptions":
            raise ImportError("Simulated initialization failure")
        return None

finder = FailInitialization()
sys.meta_path.insert(0, finder)
try:
    import pennylane2
except ImportError as error:
    assert "Simulated initialization failure" in str(error)
else:
    raise AssertionError("The initial import should have failed")
finally:
    sys.meta_path.remove(finder)

assert "pennylane2" not in sys.modules
assert "pennylane2._compatibility" in sys.modules

import pennylane
try:
    import pennylane2
except ImportError as error:
    assert "Classic PennyLane" in str(error)
    assert "pennylane2.math" not in sys.modules
else:
    raise AssertionError("Mixed imports were accepted after a failed import")
""",
        extra_path=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
