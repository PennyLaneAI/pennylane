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
"""
This file generates the circuit diagrams shown by the ``decomposition-table`` directive
(``doc/_ext/decomposition_table.py``), one SVG per decomposition rule, in the style of the
PennyLane illustration guidelines.

It is not intended to be used in any Continuous Integration, but saves time and hassle for
developers adding or changing a decomposition rule. Requires ``matplotlib`` and ``fonttools``.

Usage, from this folder::

    python generate.py              # regenerate every diagram
    python generate.py select_unary # regenerate only the named diagrams

Diagrams are registered with ``@drawing("<name>")`` in the ``drawings_*.py`` modules and written
to ``../<name>.svg``.
"""

import importlib
import pathlib
import sys
import tempfile

import matplotlib
import requests
from matplotlib.font_manager import fontManager

folder = pathlib.Path(__file__).parent
FONT_URL = "https://github.com/google/fonts/raw/main/ofl/quicksand/Quicksand%5Bwght%5D.ttf"


def quicksand_semibold():
    """Download the Quicksand variable font and instance it at SemiBold weight, once."""
    path = pathlib.Path(tempfile.gettempdir()) / "pennylane-decomp-thumbnails" / "Quicksand-600.ttf"
    if path.exists():
        return path

    # pylint: disable=import-outside-toplevel
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont

    path.parent.mkdir(parents=True, exist_ok=True)
    variable = path.parent / "Quicksand.ttf"
    response = requests.get(FONT_URL, timeout=60)
    response.raise_for_status()
    variable.write_bytes(response.content)
    font = instantiateVariableFont(TTFont(variable), {"wght": 600})
    # drop the dagger glyph so that maths labels use the (clearer) STIX dagger instead
    for table in font["cmap"].tables:
        table.cmap.pop(0x2020, None)
    font.save(path)
    return path


def use_quicksand():
    """Use Quicksand for text and maths labels. This must run before the drawings are imported,
    as matplotlib resolves the maths fonts when it is first used."""
    fontManager.addfont(str(quicksand_semibold()))
    matplotlib.rcParams.update(
        {
            "mathtext.fontset": "custom",
            "mathtext.rm": "Quicksand",
            "mathtext.it": "Quicksand",
            "mathtext.bf": "Quicksand",
            "mathtext.sf": "Quicksand",
            "mathtext.fallback": "stixsans",
        }
    )


if __name__ == "__main__":
    use_quicksand()
    sys.path.insert(0, str(folder))
    common = importlib.import_module("common")
    for module in sorted(folder.glob("drawings_*.py")):
        importlib.import_module(module.stem)

    names = sys.argv[1:] or list(common.D)
    unknown = set(names) - set(common.D)
    if unknown:
        sys.exit(f"Unknown diagrams: {', '.join(sorted(unknown))}")
    common.render(names, folder.parent)
    print(f"Wrote {len(names)} diagrams to {folder.parent}")
