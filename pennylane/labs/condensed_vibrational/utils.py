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
This module contains the main workflow for condensed-phase vibrational
calculations using QMM integration.
"""

import numpy as np
from pennylane.qchem.vibrational.vibrational_class import _get_dipole, _single_point

from qmm_workflow import run_sampling_and_scf_wrapper, setup_environment_wrapper as _qmm_setup_environment_wrapper

# Keys accepted by the QMM sampling wrapper (steps 2-4)
_QMM_SAMPLING_KEYS = {
    "box_size_angstroms",
    "simulation_time_ps",
    "n_snapshots",
    "temperature",  # also used in environment setup
    "snapshot_interval_ps",
    "displacement_amplitude",
    "qm_basis",
    "qm_method",
    "charge",
    "spin",
    "forcefield_xmls",
}

# Environment-only keys (step 1)
_QMM_ENV_KEYS = {
    "work_dir",
    "temperature",
    "pressure",
    "debug",
    "force_environment",
    "crystal_system",
    "packing_density",
    "supercell_size",
    "optimize_structure",
    "auto_determine_params",
    "heuristic_density",
}

def setup_environment_wrapper(molecule, **kwargs):
    """Thin adapter around qmm_workflow.setup_environment_wrapper returning only the workflow.

    The original QMM helper returns (workflow, result). PennyLane drivers expect just the
    workflow object, so we discard the second element while still allowing environment kwargs.
    Environment kwargs are filtered to avoid passing sampling-only keys to step 1.
    """
    env_kwargs = {k: v for k, v in kwargs.items() if k in _QMM_ENV_KEYS}
    workflow, env_result = _qmm_setup_environment_wrapper(molecule, **env_kwargs)
    # Stash the raw environment result for optional downstream inspection
    setattr(workflow, "_environment_result", env_result)
    return workflow

def snapshot_and_average(molecule, qmm_workflow, do_dipole=False, method="rhf", **kwargs):
    """Generate (or reuse) QM/MM SCF objects and compute average energy/dipole.

    Parameters
    ----------
    molecule : pennylane.qchem.Molecule
        Reference PennyLane molecule (geometry is not used for SCF if snapshots already contain SCF objects).
    qmm_workflow : QMMWorkflow
        Prepared workflow instance (after environment setup).
    do_dipole : bool
        Whether to compute averaged dipole.
    method : str
        Electronic structure method label passed through to fallback single-point (if ever used).
    **kwargs : dict
        Extra sampling kwargs (filtered to internal accepted set).
    """
    if qmm_workflow is None:
        raise ValueError("Trying to generate snapshots but no qmm_workflow provided for QMM calculations!")

    sampling_kwargs = {k: v for k, v in kwargs.items() if k in _QMM_SAMPLING_KEYS}
    scf_snapshots = run_sampling_and_scf_wrapper(qmm_workflow, molecule, **sampling_kwargs)

    num_snapshots = len(scf_snapshots)
    if num_snapshots == 0:
        raise RuntimeError("No SCF snapshots returned from workflow")

    avg_energy = 0.0
    avg_dipole = np.zeros(3) if do_dipole else None

    for scf_i in scf_snapshots:
        # Stored energy from workflow (preferred)
        energy = getattr(scf_i, '_qmmm_meta', {}).get('energy') if hasattr(scf_i, '_qmmm_meta') else None
        if energy is None:
            # Fallback (should not normally occur) – run single point on reference molecule
            energy = _single_point(molecule, method)
        avg_energy += energy

        if do_dipole:
            dip = None
            if hasattr(scf_i, '_qmmm_meta'):
                dip = scf_i._qmmm_meta.get('dipole')
            if dip is None:
                # Try PySCF integral-based dipole (origin at COM)
                try:
                    dm = scf_i.make_rdm1()
                    with scf_i.mol.with_common_origin([0.0, 0.0, 0.0]):
                        dip_ints = scf_i.mol.intor('int1e_r', comp=3)
                    # dipole expectation value: -Tr(D * r)
                    # dip_ints shape (3, nao, nao)
                    dip = -np.einsum('ab,xab->x', dm, dip_ints)
                    if hasattr(scf_i, '_qmmm_meta'):
                        scf_i._qmmm_meta['dipole'] = dip
                except Exception:
                    # Fallback to PennyLane dipole on reference molecule
                    dip = _get_dipole(molecule, method)
            avg_dipole += dip

    avg_energy /= num_snapshots
    if do_dipole:
        avg_dipole /= num_snapshots

    return avg_energy, avg_dipole
