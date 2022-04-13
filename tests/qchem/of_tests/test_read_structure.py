import os

import numpy as np

from pennylane import qchem

ref_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_ref_files")


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
