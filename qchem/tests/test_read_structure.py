import pytest
from pennylane import qchem
import os
import subprocess

ref_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_ref_files")


@pytest.mark.parametrize(
    "names",
    [
        ["h2.SDF", "h2_ref.xyz"],
        ["gdb3.mol5.PDB", "gdb3.mol5_ref.xyz"],
        ["gdb3.mol5.XYZ", "gdb3.mol5_ref.xyz"],
    ],
)
def test_conversion_from_folder(names, tmpdir, requires_babel):
    r"""Test the conversion of molecular structure file with different formats"""

    filename = os.path.join(ref_dir, names[0])
    qchem.read_structure(filename, outpath=tmpdir)

    with open(tmpdir.join("structure.xyz")) as g:
        gen_file = g.readlines()[2:]

    with open(os.path.join(ref_dir, names[1])) as f:
        ref_file = f.readlines()[2:]

    assert gen_file == ref_file


def test_reading_xyz_file(tmpdir):
    r"""Test reading of the generated file 'structure.xyz'"""

    ref_geometry = [
        ["C", (0.361, -0.452, -0.551)],
        ["C", (-0.714, 0.125, 0.327)],
        ["N", (0.683, 0.133, 0.745)],
        ["H", (0.442, -1.529, -0.619)],
        ["H", (0.672, 0.102, -1.428)],
        ["H", (-1.364, -0.56, 0.857)],
        ["H", (-1.149, 1.08, 0.060)],
        ["H", (1.093, 1.063, 0.636)],
    ]

    name = os.path.join(ref_dir, "gdb3.mol5.XYZ")
    geometry = qchem.read_structure(name, outpath=tmpdir)

    assert geometry == ref_geometry


def test_subprocess_run(monkeypatch, requires_babel):
    r"""Test 'subprocess.run' function running babel to convert the molecular structure
     file to xyz format"""

    with monkeypatch.context() as m:

        def fake_run(*args, **kwargs):
            raise subprocess.CalledProcessError(1, "obabel")

        m.setattr(subprocess, "run", fake_run)

        with pytest.raises(
            RuntimeError, match="Open Babel error. See the following Open Babel output for details"
        ):
            qchem.read_structure("fake_mol_geo.SDF")
