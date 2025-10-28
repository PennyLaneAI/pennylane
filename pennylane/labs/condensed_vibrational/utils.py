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
calculations using QMM integration with enhanced snapshot caching.
"""

import numpy as np
from pennylane.qchem.vibrational.vibrational_class import _get_dipole, _single_point

try:
    from qmm_workflow import run_sampling_and_scf_wrapper, setup_environment_wrapper
    QMM_AVAILABLE = True
except ImportError:
    QMM_AVAILABLE = False
    raise ValueError("qmm_workflow not available from QMM package, cannot run condensed phase workflow!")

# Export list for public API
__all__ = [
    'setup_environment_wrapper',
    'snapshot_and_average', 
    'precompute_snapshots',
    'setup_cached_workflow',
    'generate_scf_from_cached_snapshots'
]

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

def precompute_snapshots(molecule, qmm_workflow, **kwargs):
    """Precompute MD snapshots and store them in the workflow object for reuse.

    Parameters
    ----------
    molecule : pennylane.qchem.Molecule
        Reference PennyLane molecule for the initial trajectory generation.
    qmm_workflow : QMMWorkflow
        Prepared workflow instance (after environment setup).
    **kwargs : dict
        Extra sampling kwargs (filtered to internal accepted set).
    
    Returns
    -------
    qmm_workflow : QMMWorkflow
        The workflow object with cached snapshots stored in _cached_snapshots attribute.
    """
    if not QMM_AVAILABLE:
        raise RuntimeError("QMM workflow not available. Cannot precompute snapshots.")
    
    if qmm_workflow is None:
        raise ValueError("Trying to generate snapshots but no qmm_workflow provided for QMM calculations!")
    
    # Check if snapshots are already cached
    if hasattr(qmm_workflow, '_cached_snapshots') and qmm_workflow._cached_snapshots is not None:
        return qmm_workflow
    
    sampling_kwargs = {k: v for k, v in kwargs.items() if k in _QMM_SAMPLING_KEYS}
    
    # Generate snapshots and cache them in the workflow
    snapshots = run_sampling_and_scf_wrapper(qmm_workflow, molecule, **sampling_kwargs)
    
    # Store snapshots in workflow object for reuse
    setattr(qmm_workflow, '_cached_snapshots', snapshots)
    
    return qmm_workflow

def generate_scf_from_cached_snapshots(cached_snapshots, molecule, **kwargs):
    """Generate SCF objects from cached MD snapshots using a new molecule geometry.
    
    This function reuses the MD environment snapshots but applies them to a new 
    molecule geometry for SCF calculations.
    
    Parameters
    ----------
    cached_snapshots : list
        List of cached SCF snapshot objects from previous calculations
    molecule : pennylane.qchem.Molecule  
        New molecule geometry to use for SCF calculations
    **kwargs : dict
        QM method parameters (basis, method, charge, spin, etc.)
    
    Returns
    -------
    scf_snapshots : list
        List of SCF objects with updated molecule geometry
    """
    import numpy as np
    import os
    from pyscf import gto, scf as pyscf_scf
    
    scf_snapshots = []
    
    # Extract QM parameters
    basis = kwargs.get('qm_basis', 'sto-3g')
    method = kwargs.get('qm_method', 'hf')
    charge = kwargs.get('charge', 0)
    spin = kwargs.get('spin', 0)
    
    # Get molecule geometry
    mol_coords = molecule.coordinates
    mol_symbols = molecule.symbols
    
    for i, cached_scf in enumerate(cached_snapshots):
        # Create new PySCF molecule with updated geometry but same environment
        # Preserve the MM environment from the cached snapshot
        if hasattr(cached_scf, '_qmmm_meta'):
            meta = cached_scf._qmmm_meta
            
            # Try to extract MM environment from the QM/MM object itself
            mm_coords = None
            mm_charges = None
            
            # Method 1: Check if it's a PySCF QM/MM object with built-in environment
            if hasattr(cached_scf, 'mm_mol') and cached_scf.mm_mol is not None:
                try:
                    mm_coords = cached_scf.mm_mol.atom_coords()  # Get MM coordinates
                    mm_charges = cached_scf.mm_mol.atom_charges()  # Get MM charges
                except AttributeError:
                    pass
            
            # Method 2: Try to extract from metadata file if available
            if mm_coords is None and 'snapshot_file' in meta and 'n_env_atoms' in meta:
                try:
                    # Try to load MM environment from snapshot file
                    snapshot_file = meta['snapshot_file']
                    if os.path.exists(snapshot_file):
                        # This would need to be adapted based on the actual file format
                        # TODO: Load MM coords and charges from file
                        pass
                except Exception as e:
                    pass
            
            # Method 3: Check for direct MM data in the QM/MM object
            if mm_coords is None:
                # Try other common attributes where MM data might be stored
                for attr_name in ['_mm_coords', 'mm_coords', 'env_coords', '_env_coords']:
                    if hasattr(cached_scf, attr_name):
                        mm_coords = getattr(cached_scf, attr_name)
                        break
                
                for attr_name in ['_mm_charges', 'mm_charges', 'env_charges', '_env_charges']:
                    if hasattr(cached_scf, attr_name):
                        mm_charges = getattr(cached_scf, attr_name)
                        break
            
            if mm_coords is not None and mm_charges is not None:
                # Use extracted MM environment with updated QM geometry
                
                # Build atom string for new molecule geometry
                atom_str = []
                for i, (symbol, coord) in enumerate(zip(mol_symbols, mol_coords)):
                    atom_str.append(f"{symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}")
                
                # Create new PySCF molecule
                mol = gto.Mole()
                mol.atom = '; '.join(atom_str)
                mol.basis = basis
                mol.charge = charge
                mol.spin = spin
                mol.build()
                
                # Set up SCF with same method
                if method.lower() == 'hf':
                    mf = pyscf_scf.RHF(mol) if spin == 0 else pyscf_scf.UHF(mol)
                else:
                    mf = pyscf_scf.RKS(mol) if spin == 0 else pyscf_scf.UKS(mol)
                    if 'qm_functional' in kwargs:
                        mf.xc = kwargs['qm_functional']
                
                # Silence verbose SCF output
                mf.verbose = 0
                
                # Add MM environment as external potential
                # Use PySCF's built-in QM/MM functionality for proper integration
                from pyscf import qmmm
                
                # Convert coordinates to proper format (Bohr units)
                mm_coords_bohr = np.array(mm_coords) / 0.52917721067  # Convert Angstrom to Bohr
                mm_charges_array = np.array(mm_charges)
                
                # Set up QM/MM calculation with proper external potential
                mf = qmmm.add_mm_charges(mf, mm_coords_bohr, mm_charges_array)
                
                # Silence verbose output
                mf.verbose = 0
                
                # Store metadata
                mf._qmmm_meta = {
                    'mm_coords': mm_coords,
                    'mm_charges': mm_charges,
                    'energy': None,  # Will be computed when needed
                    'dipole': None   # Will be computed when needed
                }
                
                scf_snapshots.append(mf)
            else:
                raise RuntimeError(
                    f"Could not extract MM environment from cached snapshot {i}. "
                    f"Snapshot type: {type(cached_scf)}. "
                    f"Available attributes: {[attr for attr in dir(cached_scf) if not attr.startswith('__')]}. "
                    f"QM/MM condensed-phase calculations require MM environment data."
                )
    
    return scf_snapshots

def snapshot_and_average(molecule, qmm_workflow, do_dipole=False, method="rhf", use_cached=True, **kwargs):
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
    use_cached : bool
        Whether to use cached snapshots if available. If False, generates new snapshots.
    **kwargs : dict
        Extra sampling kwargs (filtered to internal accepted set).
    """
    if qmm_workflow is None:
        raise ValueError("Trying to generate snapshots but no qmm_workflow provided for QMM calculations!")

    # Use cached snapshots if available and requested
    if use_cached and hasattr(qmm_workflow, '_cached_snapshots') and qmm_workflow._cached_snapshots is not None:
        scf_snapshots = generate_scf_from_cached_snapshots(qmm_workflow._cached_snapshots, molecule, **kwargs)
    else:
        # Generate new snapshots
        if not QMM_AVAILABLE:
            raise RuntimeError("QMM workflow not available. Cannot generate new snapshots.")
        sampling_kwargs = {k: v for k, v in kwargs.items() if k in _QMM_SAMPLING_KEYS}
        scf_snapshots = run_sampling_and_scf_wrapper(qmm_workflow, molecule, **sampling_kwargs)

    num_snapshots = len(scf_snapshots)
    if num_snapshots == 0:
        raise RuntimeError("No SCF snapshots returned from workflow")

    avg_energy = 0.0
    avg_dipole = np.zeros(3) if do_dipole else None

    for scf_i in scf_snapshots:
        energy = scf_i.kernel()
        avg_energy += energy

        if do_dipole:
            dip = _get_dipole(scf_i, method)
            avg_dipole += dip

    avg_energy /= num_snapshots
    if do_dipole:
        avg_dipole /= num_snapshots

    return avg_energy, avg_dipole

def setup_cached_workflow(molecule, **kwargs):
    """Setup a QMM workflow with pre-computed snapshots for efficient reuse.
    
    This function creates a workflow, sets up the environment, and pre-computes
    MD snapshots that can be reused for multiple PES point calculations.
    
    Parameters
    ----------
    molecule : pennylane.qchem.Molecule
        Reference molecule for environment setup and initial trajectory
    **kwargs : dict
        Combined environment and sampling kwargs
        
    Returns
    -------
    qmm_workflow : QMMWorkflow
        Workflow object with cached snapshots stored in _cached_snapshots attribute
    """
    # Setup environment
    workflow = setup_environment_wrapper(molecule, **kwargs)
    
    # Pre-compute snapshots
    workflow = precompute_snapshots(molecule, workflow, **kwargs)
    
    return workflow
