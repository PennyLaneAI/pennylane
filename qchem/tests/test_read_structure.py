import pytest
from pennylane import qchem
import numpy as np
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

    ref_symbols = ["C", "C", "N", "H", "H", "H", "H", "H"]
    ref_coords = np.array(
        [
            0.68219113,
            -0.85415621,
            -1.04123909,
            -1.34926445,
            0.23621577,
            0.61794044,
            1.29068294,
            0.25133357,
            1.40784596,
            0.83525895,
            -2.88939124,
            -1.16974047,
            1.26989596,
            0.19275206,
            -2.69852891,
            -2.57758643,
            -1.05824663,
            1.61949529,
            -2.17129532,
            2.04090421,
            0.11338357,
            2.06547065,
            2.00877887,
            1.20186582,
        ]
    )
    name = os.path.join(ref_dir, "gdb3.mol5.XYZ")
    symbols, coordinates = qchem.read_structure(name, outpath=tmpdir)

    assert symbols == ref_symbols
    assert np.allclose(coordinates, ref_coords)


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
