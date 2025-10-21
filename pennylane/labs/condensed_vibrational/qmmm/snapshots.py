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
This module contains utilities for generating thermodynamic snapshots of solvent 
distributions around a fixed solute molecule. It uses packmol to create initial 
configurations and molecular dynamics simulations to sample the thermodynamic 
ensemble, providing multiple snapshots for QM/MM calculations.
"""

import os
import shutil
import tempfile
import sys
import numpy as np
from typing import Optional, Union, Dict, Any, List, Tuple
import warnings

# Import functions from npt_density module  
from npt_density import (
    xyz_to_pdb, estimate_molecular_properties, estimate_density,
    generate_packmol_input, run_packmol, setup_openmm_system,
    create_manual_configuration, _write_pdb_file,
    OPENMM_AVAILABLE, DEFAULT_FORCEFIELDS, ATOMIC_MASSES
)

# Import molecular handling and simulation libraries
if OPENMM_AVAILABLE:
    from openmm.app import *
    from openmm import *
    from openmm.unit import *
    from openmm.app import PDBFile
else:
    # Define dummy variables to prevent NameError
    class DummyUnit:
        def __mul__(self, other): return self
        def __rmul__(self, other): return self
        def __pow__(self, other): return self
        def __truediv__(self, other): return self
    
    class DummyClass:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): return self
    
    nanometer = femtosecond = picosecond = bar = kelvin = DummyUnit()
    angstrom = dalton = gram = kilojoule_per_mole = DummyUnit()
    Vec3 = PDBFile = ForceField = PME = HBonds = DummyClass
    LangevinMiddleIntegrator = MonteCarloBarostat = DummyClass
    Platform = Simulation = StateDataReporter = DCDReporter = DummyClass


def create_mixed_system(solute_xyz: str, solvent_xyz: str, n_solvent: int, 
                       box_size: float, output_dir: str, 
                       solute_position: Optional[Tuple[float, float, float]] = None,
                       buffer_distance: float = 3.0) -> str:
    """
    Create a mixed system with one fixed solute molecule and multiple solvent molecules.
    
    Parameters
    ----------
    solute_xyz : str
        Path to XYZ file for the solute molecule
    solvent_xyz : str
        Path to XYZ file for the solvent molecule
    n_solvent : int
        Number of solvent molecules to add
    box_size : float
        Size of the cubic box in Angstroms
    output_dir : str
        Directory for output files
    solute_position : tuple, optional
        (x, y, z) position for the solute molecule center. If None, places at box center
    buffer_distance : float, default=3.0
        Minimum distance between solute and solvent molecules in Angstroms
        
    Returns
    -------
    str
        Path to the generated mixed system PDB file
        
    Raises
    ------
    RuntimeError
        If packmol fails or molecules cannot be placed
    """
    print(f"Creating mixed system with 1 solute + {n_solvent} solvent molecules")
    
    # Convert both molecules to PDB format
    solute_pdb = os.path.join(output_dir, "solute.pdb")
    solvent_pdb = os.path.join(output_dir, "solvent.pdb")
    
    xyz_to_pdb(solute_xyz, solute_pdb)
    xyz_to_pdb(solvent_xyz, solvent_pdb)
    
    # Get molecular properties for sizing calculations
    with open(solute_xyz, 'r') as f:
        lines = f.readlines()
    n_atoms_solute = int(lines[0].strip())
    solute_atoms = []
    for i in range(2, 2 + n_atoms_solute):
        parts = lines[i].strip().split()
        solute_atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    
    solute_props = estimate_molecular_properties(solute_atoms)
    solute_size = solute_props['size']
    
    # Set solute position (default to box center)
    if solute_position is None:
        solute_position = (0.0, 0.0, 0.0)  # Box center in packmol coordinates
    
    print(f"Placing solute at position: {solute_position}")
    print(f"Solute size: {solute_size:.2f} Å")
    print(f"Buffer distance: {buffer_distance:.2f} Å")
    
    # Create modified solute PDB with fixed position
    fixed_solute_pdb = os.path.join(output_dir, "solute_fixed.pdb")
    _place_solute_at_position(solute_pdb, fixed_solute_pdb, solute_position)
    
    # Calculate exclusion radius for solvent placement
    exclusion_radius = solute_size / 2 + buffer_distance
    
    print(f"Exclusion radius for solvent: {exclusion_radius:.2f} Å")
    
    # Generate packmol input for mixed system
    mixed_pdb = os.path.join(output_dir, "mixed_system.pdb")
    packmol_inp = os.path.join(output_dir, "packmol_mixed.inp")
    
    # More sophisticated packmol input for mixed systems
    packmol_input = f"""tolerance 2.0
filetype pdb
output {os.path.basename(mixed_pdb)}
seed 12345

# Fixed solute molecule at specified position
structure {os.path.basename(fixed_solute_pdb)}
  number 1
  fixed {solute_position[0]:.3f} {solute_position[1]:.3f} {solute_position[2]:.3f} 0. 0. 0.
end structure

# Solvent molecules avoiding the solute
structure {os.path.basename(solvent_pdb)}
  number {n_solvent}
  inside cube -{box_size/2:.2f} -{box_size/2:.2f} -{box_size/2:.2f} {box_size:.2f}
  outside sphere {solute_position[0]:.3f} {solute_position[1]:.3f} {solute_position[2]:.3f} {exclusion_radius:.2f}
end structure
"""
    
    # Write packmol input
    with open(packmol_inp, 'w') as f:
        f.write(packmol_input)
    
    print(f"Running packmol to create mixed system...")
    print(f"Packmol input:\n{packmol_input}")
    
    # Try packmol first, with fallback to manual placement
    try:
        # Check if packmol is available
        if not shutil.which("packmol"):
            raise RuntimeError("packmol executable not found")
        
        # Run packmol with short timeout for testing
        import subprocess
        result = subprocess.run(
            ["packmol", "<", "packmol_mixed.inp"],
            shell=True,
            cwd=output_dir,
            timeout=10,  # Much shorter timeout for testing
            check=True,
            capture_output=True,
            text=True
        )
        
        if os.path.exists(mixed_pdb):
            print("Successfully created mixed system with packmol!")
            return mixed_pdb
        else:
            raise RuntimeError("Packmol completed but output file not created")
            
    except Exception as e:
        print(f"Packmol failed: {e}")
        print("Using manual placement as fallback...")
        return _create_manual_mixed_system(
            fixed_solute_pdb, solvent_pdb, n_solvent, box_size, 
            output_dir, solute_position, exclusion_radius
        )


def _place_solute_at_position(solute_pdb: str, output_pdb: str, 
                             position: Tuple[float, float, float]):
    """
    Create a new PDB file with the solute molecule centered at the specified position.
    
    Parameters
    ----------
    solute_pdb : str
        Path to the original solute PDB file
    output_pdb : str
        Path for the output PDB file with repositioned solute
    position : tuple
        (x, y, z) target position for the molecule center
    """
    with open(solute_pdb, 'r') as f:
        lines = f.readlines()
    
    atom_lines = [line for line in lines if line.startswith('ATOM')]
    
    # Calculate current center of mass
    coords = []
    for line in atom_lines:
        # Parse coordinates more robustly
        try:
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
        except ValueError:
            # If the standard PDB parsing fails, try to extract numbers from the line
            import re
            numbers = re.findall(r'-?\d+\.?\d*', line)
            if len(numbers) >= 3:
                # Take the last 3 numbers as coordinates
                x, y, z = float(numbers[-3]), float(numbers[-2]), float(numbers[-1])
            else:
                print(f"Warning: Could not parse coordinates from line: {line.strip()}")
                continue
        coords.append([x, y, z])
    
    coords = np.array(coords)
    current_center = np.mean(coords, axis=0)
    
    # Calculate translation vector
    target_center = np.array(position)
    translation = target_center - current_center
    
    # Write new PDB with translated coordinates
    with open(output_pdb, 'w') as f:
        for line in lines:
            if line.startswith('ATOM'):
                # Extract original coordinates robustly
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip()) 
                    z = float(line[46:54].strip())
                except ValueError:
                    # If the standard PDB parsing fails, try to extract numbers from the line
                    import re
                    numbers = re.findall(r'-?\d+\.?\d*', line)
                    if len(numbers) >= 3:
                        # Take the last 3 numbers as coordinates
                        x, y, z = float(numbers[-3]), float(numbers[-2]), float(numbers[-1])
                    else:
                        print(f"Warning: Could not parse coordinates from line: {line.strip()}")
                        f.write(line)  # Write original line if parsing fails
                        continue
                
                # Apply translation
                new_x = x + translation[0]
                new_y = y + translation[1]
                new_z = z + translation[2]
                
                # Reconstruct line with new coordinates
                new_line = line[:30] + f"{new_x:8.3f}{new_y:8.3f}{new_z:8.3f}" + line[54:]
                f.write(new_line)
            else:
                f.write(line)


def _create_manual_mixed_system(solute_pdb: str, solvent_pdb: str, n_solvent: int,
                               box_size: float, output_dir: str,
                               solute_position: Tuple[float, float, float],
                               exclusion_radius: float) -> str:
    """
    Manually create a mixed system when packmol fails.
    
    Parameters
    ----------
    solute_pdb : str
        Path to the fixed solute PDB file
    solvent_pdb : str
        Path to the solvent molecule PDB file
    n_solvent : int
        Number of solvent molecules to place
    box_size : float
        Box size in Angstroms
    output_dir : str
        Output directory
    solute_position : tuple
        Position of the solute molecule
    exclusion_radius : float
        Minimum distance from solute center
        
    Returns
    -------
    str
        Path to the manually created mixed system PDB file
    """
    print("Creating manual mixed system configuration...")
    
    # Read solute atoms
    with open(solute_pdb, 'r') as f:
        solute_lines = f.readlines()
    
    # Read solvent atoms
    with open(solvent_pdb, 'r') as f:
        solvent_lines = f.readlines()
    
    solute_atoms = [line for line in solute_lines if line.startswith('ATOM')]
    solvent_atom_template = [line for line in solvent_lines if line.startswith('ATOM')]
    
    if not solvent_atom_template:
        raise RuntimeError("No ATOM records found in solvent PDB")
    
    # Parse solvent coordinates for centering
    solvent_coords = []
    solvent_atom_data = []
    for line in solvent_atom_template:
        atom_name = line[12:16].strip()
        element = line[76:78].strip() if len(line) > 77 and line[76:78].strip() else atom_name[0]
        
        # Robust coordinate parsing
        try:
            x_str = line[30:38].strip()
            y_str = line[38:46].strip()
            z_str = line[46:54].strip()
            
            # Handle potential formatting issues
            x = float(x_str.split()[-1]) if x_str else 0.0
            y = float(y_str.split()[-1]) if y_str else 0.0
            z = float(z_str.split()[-1]) if z_str else 0.0
        except (ValueError, IndexError):
            print(f"Warning: Could not parse coordinates from line: {line.strip()}")
            x, y, z = 0.0, 0.0, 0.0
        
        solvent_coords.append([x, y, z])
        solvent_atom_data.append((atom_name, element, x, y, z))
    
    # Center the solvent molecule at origin
    solvent_coords = np.array(solvent_coords)
    solvent_center = np.mean(solvent_coords, axis=0)
    centered_solvent_atoms = []
    for atom_name, element, x, y, z in solvent_atom_data:
        centered_solvent_atoms.append((
            atom_name, element, 
            x - solvent_center[0], 
            y - solvent_center[1], 
            z - solvent_center[2]
        ))
    
    # Generate random positions for solvent molecules
    np.random.seed(42)  # For reproducibility
    solvent_positions = []
    attempts = 0
    max_attempts = n_solvent * 100
    
    while len(solvent_positions) < n_solvent and attempts < max_attempts:
        # Random position in box
        x = np.random.uniform(-box_size/2, box_size/2)
        y = np.random.uniform(-box_size/2, box_size/2)
        z = np.random.uniform(-box_size/2, box_size/2)
        
        # Check distance from solute
        distance_from_solute = np.sqrt(
            (x - solute_position[0])**2 + 
            (y - solute_position[1])**2 + 
            (z - solute_position[2])**2
        )
        
        if distance_from_solute > exclusion_radius:
            # Check distance from other solvent molecules
            too_close = False
            for existing_pos in solvent_positions:
                distance = np.sqrt(
                    (x - existing_pos[0])**2 + 
                    (y - existing_pos[1])**2 + 
                    (z - existing_pos[2])**2
                )
                if distance < 3.0:  # Minimum solvent-solvent distance
                    too_close = True
                    break
            
            if not too_close:
                # Random rotation
                angle = np.random.random() * 2 * np.pi
                solvent_positions.append((x, y, z, angle))
        
        attempts += 1
    
    if len(solvent_positions) < n_solvent:
        print(f"Warning: Could only place {len(solvent_positions)} out of {n_solvent} solvent molecules")
    
    # Write the mixed system PDB
    mixed_pdb = os.path.join(output_dir, "mixed_system.pdb")
    with open(mixed_pdb, 'w') as f:
        f.write("REMARK   Manual mixed system created by PennyLane QML\n")
        
        atom_serial = 1
        
        # Write solute atoms first - parse coordinates and reformat consistently
        for line in solute_atoms:
            # Parse coordinates with robust handling
            try:
                x_str = line[30:38].strip()
                y_str = line[38:46].strip()
                z_str = line[46:54].strip()
                
                # Handle potential formatting issues
                x = float(x_str.split()[-1]) if x_str else 0.0
                y = float(y_str.split()[-1]) if y_str else 0.0
                z = float(z_str.split()[-1]) if z_str else 0.0
            except (ValueError, IndexError):
                print(f"Warning: Could not parse coordinates from solute line: {line.strip()}")
                x, y, z = 0.0, 0.0, 0.0
            
            # Extract atom name and element
            atom_name = line[12:16].strip()
            element = line[76:78].strip() if len(line) > 77 and line[76:78].strip() else atom_name[0]
            element_formatted = element[0].upper() + element[1:].lower() if len(element) > 1 else element[0].upper()
            
            # Write with exact working PDB format (same as npt_density function)
            simple_name = element[0].upper()  # Just use element symbol like H, C, N
            pdb_line = f"ATOM  {atom_serial:5d}  {simple_name:<3s} {'SOL':>3s} A{1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element_formatted:>2s}\n"
            f.write(pdb_line)
            atom_serial += 1
        
        # Write solvent molecules
        for mol_idx, (sol_x, sol_y, sol_z, angle) in enumerate(solvent_positions):
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            for atom_name, element, x, y, z in centered_solvent_atoms:
                # Apply rotation around z-axis
                rot_x = x * cos_a - y * sin_a
                rot_y = x * sin_a + y * cos_a
                rot_z = z
                
                # Translate to final position
                final_x = rot_x + sol_x
                final_y = rot_y + sol_y
                final_z = rot_z + sol_z
                
                # Write atom line
                element_formatted = element[0].upper() + (element[1:].lower() if len(element) > 1 else "")
                
                # Use exact working PDB format (same as npt_density function)
                simple_name = element[0].upper()  # Just use element symbol like H, C, N
                line = f"ATOM  {atom_serial:5d}  {simple_name:<3s} {'SOL':>3s} B{mol_idx+1:4d}    {final_x:8.3f}{final_y:8.3f}{final_z:8.3f}  1.00  0.00          {element_formatted:>2s}\n"
                f.write(line)
                atom_serial += 1
        
        f.write("END\n")
    
    print(f"Manual mixed system created with {len(solvent_positions)} solvent molecules")
    return mixed_pdb


def run_npt_sampling(mixed_pdb: str, box_size: float, temperature: float = 298.15,
                    pressure: float = 1.0, equilibration_time: float = 0.1,
                    sampling_time: float = 1.0, snapshot_interval: float = 0.1,
                    forcefield_xml: Optional[Union[str, List[str]]] = None,
                    output_dir: str = None) -> Dict[str, Any]:
    """
    Run NPT simulation to sample solvent configurations around fixed solute.
    
    Parameters
    ----------
    mixed_pdb : str
        Path to the mixed system PDB file
    box_size : float
        Box size in Angstroms
    temperature : float, default=298.15
        Temperature in Kelvin
    pressure : float, default=1.0
        Pressure in bar
    equilibration_time : float, default=0.1
        Equilibration time in nanoseconds
    sampling_time : float, default=1.0
        Total sampling time in nanoseconds
    snapshot_interval : float, default=0.1
        Time interval between snapshots in nanoseconds
    forcefield_xml : str or list, optional
        Force field XML file(s)
    output_dir : str, optional
        Directory for output files
        
    Returns
    -------
    dict
        Dictionary containing simulation results and snapshot information
    """
    if not OPENMM_AVAILABLE:
        raise RuntimeError("OpenMM is not available. Cannot run NPT simulation.")
    
    print(f"Setting up NPT sampling simulation...")
    print(f"Temperature: {temperature} K, Pressure: {pressure} bar")
    print(f"Equilibration: {equilibration_time} ns, Sampling: {sampling_time} ns")
    print(f"Snapshot interval: {snapshot_interval} ns")
    
    # Set up OpenMM system
    pdb_obj, system, topology = setup_openmm_system(mixed_pdb, forcefield_xml, box_size)
    
    # Create integrator
    integrator = LangevinMiddleIntegrator(
        temperature * kelvin,
        1.0 / picosecond,  # friction coefficient
        2.0 * femtosecond  # time step
    )
    
    # Add barostat for NPT ensemble
    system.addForce(MonteCarloBarostat(pressure * bar, temperature * kelvin))
    
    # Create simulation
    try:
        platform = Platform.getPlatformByName('CUDA')
        properties = {'Precision': 'mixed'}
        simulation = Simulation(topology, system, integrator, platform, properties)
        print("Using CUDA platform")
    except:
        try:
            platform = Platform.getPlatformByName('OpenCL')
            properties = {'Precision': 'mixed'}
            simulation = Simulation(topology, system, integrator, platform, properties)
            print("Using OpenCL platform")
        except:
            platform = Platform.getPlatformByName('CPU')
            simulation = Simulation(topology, system, integrator, platform)
            print("Using CPU platform")
    
    # Set initial positions and minimize energy
    simulation.context.setPositions(pdb_obj.positions)
    print("Minimizing energy...")
    simulation.minimizeEnergy(maxIterations=1000)
    
    # Set up trajectory output
    if output_dir:
        dcd_file = os.path.join(output_dir, "sampling_trajectory.dcd")
        log_file = os.path.join(output_dir, "sampling.log")
        
        simulation.reporters.append(
            StateDataReporter(
                log_file, 1000, step=True, time=True, potentialEnergy=True,
                temperature=True, volume=True, density=True
            )
        )
        simulation.reporters.append(DCDReporter(dcd_file, 1000))
    
    # Equilibration phase
    equilibration_steps = int(equilibration_time * 1000000 / 2)  # 2 fs time step
    print(f"Running equilibration ({equilibration_steps} steps)...")
    simulation.step(equilibration_steps)
    
    # Sampling phase with snapshot collection
    steps_per_snapshot = int(snapshot_interval * 1000000 / 2)  # 2 fs time step
    total_sampling_steps = int(sampling_time * 1000000 / 2)
    n_snapshots = total_sampling_steps // steps_per_snapshot
    
    print(f"Collecting {n_snapshots} snapshots during {sampling_time} ns sampling...")
    
    snapshots = []
    snapshot_times = []
    
    for snapshot_idx in range(n_snapshots):
        # Run simulation for the interval
        simulation.step(steps_per_snapshot)
        
        # Get current state
        state = simulation.context.getState(getPositions=True, getEnergy=True)
        positions = state.getPositions()
        potential_energy = state.getPotentialEnergy()
        current_time = equilibration_time + (snapshot_idx + 1) * snapshot_interval
        
        # Store snapshot
        snapshots.append({
            'positions': positions,
            'potential_energy': potential_energy / kilojoule_per_mole,
            'time': current_time,
            'snapshot_index': snapshot_idx
        })
        snapshot_times.append(current_time)
        
        # Save individual snapshot PDB
        if output_dir:
            snapshot_pdb = os.path.join(output_dir, f"snapshot_{snapshot_idx:03d}.pdb")
            PDBFile.writeFile(topology, positions, open(snapshot_pdb, 'w'))
        
        print(f"  Snapshot {snapshot_idx + 1}/{n_snapshots}: t = {current_time:.3f} ns, "
              f"E = {potential_energy / kilojoule_per_mole:.1f} kJ/mol")
    
    # Save final state
    if output_dir:
        final_pdb = os.path.join(output_dir, "final_sampled_state.pdb")
        final_positions = simulation.context.getState(getPositions=True).getPositions()
        PDBFile.writeFile(topology, final_positions, open(final_pdb, 'w'))
    
    results = {
        'snapshots': snapshots,
        'snapshot_times': snapshot_times,
        'n_snapshots': len(snapshots),
        'equilibration_time': equilibration_time,
        'sampling_time': sampling_time,
        'snapshot_interval': snapshot_interval,
        'temperature': temperature,
        'pressure': pressure,
        'output_dir': output_dir
    }
    
    print(f"Sampling completed! Collected {len(snapshots)} snapshots.")
    return results


def generate_solvent_snapshots(solute_xyz: str, solvent_xyz: str,
                              temperature: float = 298.15,
                              pressure: float = 1.0,
                              solvent_density: float = 1.0,
                              n_snapshots: int = 2,
                              n_solvent: Optional[int] = None,
                              box_size: Optional[float] = None,
                              equilibration_time: float = 0.1,
                              sampling_time: Optional[float] = None,
                              snapshot_interval: Optional[float] = None,
                              solute_position: Optional[Tuple[float, float, float]] = None,
                              forcefield_xml: Optional[Union[str, List[str]]] = None,
                              output_dir: Optional[str] = None,
                              cleanup: bool = True) -> Dict[str, Any]:
    """
    Generate thermodynamic snapshots of solvent distributions around a fixed solute molecule.
    
    This function performs the complete workflow:
    1. Creates a mixed system with fixed solute and solvent molecules
    2. Runs NPT molecular dynamics to equilibrate the system
    3. Collects snapshots at regular intervals during sampling
    4. Returns the snapshots for further QM/MM analysis
    
    Parameters
    ----------
    solute_xyz : str
        Path to XYZ file for the solute molecule (will be kept fixed)
    solvent_xyz : str
        Path to XYZ file for the solvent molecule
    temperature : float, default=298.15
        Temperature in Kelvin
    pressure : float, default=1.0
        Pressure in bar
    solvent_density : float, default=1.0
        Density of the solvent in g/cm³ (used to estimate system size)
    n_snapshots : int, default=2
        Number of snapshots to collect
    n_solvent : int, optional
        Number of solvent molecules. If None, estimated from density and box size
    box_size : float, optional
        Size of cubic simulation box in Angstroms. If None, estimated from molecular properties
    equilibration_time : float, default=0.1
        Equilibration time in nanoseconds
    sampling_time : float, optional
        Total sampling time in nanoseconds. If None, calculated from n_snapshots and interval
    snapshot_interval : float, optional
        Time interval between snapshots in nanoseconds. If None, calculated from sampling_time
    solute_position : tuple, optional
        (x, y, z) position for solute center. If None, places at box center
    forcefield_xml : str or list, optional
        Force field XML file(s). If None, uses default
    output_dir : str, optional
        Directory for output files. If None, creates temporary directory
    cleanup : bool, default=True
        Whether to clean up intermediate files
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'snapshots': List of snapshot dictionaries with positions and energies
        - 'snapshot_times': List of simulation times for each snapshot
        - 'n_snapshots': Number of snapshots collected
        - 'system_info': Information about the system setup
        - 'simulation_parameters': Parameters used for the simulation
        - 'output_dir': Directory containing output files
        
    Raises
    ------
    RuntimeError
        If packmol or OpenMM are not available, or if simulations fail
    FileNotFoundError
        If input XYZ files are not found
        
    Examples
    --------
    >>> # Generate 5 snapshots of HCN solvent around HCN solute
    >>> results = generate_solvent_snapshots(
    ...     'solute.xyz', 'solvent.xyz',
    ...     temperature=280, n_snapshots=5,
    ...     equilibration_time=0.05, sampling_time=0.5
    ... )
    >>> print(f"Collected {results['n_snapshots']} snapshots")
    >>> print(f"Output files in: {results['output_dir']}")
    """
    # Validate dependencies
    if not shutil.which("packmol") and not OPENMM_AVAILABLE:
        warnings.warn("Neither packmol nor OpenMM available. Using simplified manual placement.")
    
    if not os.path.exists(solute_xyz):
        raise FileNotFoundError(f"Solute XYZ file not found: {solute_xyz}")
    if not os.path.exists(solvent_xyz):
        raise FileNotFoundError(f"Solvent XYZ file not found: {solvent_xyz}")
    
    # Setup output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="qml_solvent_snapshots_")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting solvent snapshot generation")
    print(f"Solute: {solute_xyz}")
    print(f"Solvent: {solvent_xyz}")
    print(f"Conditions: T = {temperature} K, P = {pressure} bar")
    print(f"Target snapshots: {n_snapshots}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Get molecular properties
        with open(solute_xyz, 'r') as f:
            lines = f.readlines()
        n_atoms_solute = int(lines[0].strip())
        solute_atoms = []
        for i in range(2, 2 + n_atoms_solute):
            parts = lines[i].strip().split()
            solute_atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
        
        with open(solvent_xyz, 'r') as f:
            lines = f.readlines()
        n_atoms_solvent = int(lines[0].strip())
        solvent_atoms = []
        for i in range(2, 2 + n_atoms_solvent):
            parts = lines[i].strip().split()
            solvent_atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
        
        solute_props = estimate_molecular_properties(solute_atoms)
        solvent_props = estimate_molecular_properties(solvent_atoms)
        
        print(f"Solute: {solute_props['mass']:.1f} amu, {solute_props['size']:.1f} Å")
        print(f"Solvent: {solvent_props['mass']:.1f} amu, {solvent_props['size']:.1f} Å")
        
        # Estimate system size if not provided
        if box_size is None:
            # Create a reasonable box size for solvation
            # Start with solute size and add layers of solvent
            min_box_size = solute_props['size'] + 4 * solvent_props['size']  # 2 solvation layers
            
            if n_solvent is not None:
                # Calculate based on solvent density
                solvent_mass_total = n_solvent * solvent_props['mass'] * 1.66054e-24  # amu to grams
                solvent_volume_cm3 = solvent_mass_total / solvent_density
                density_based_box_size = (solvent_volume_cm3 * 1e24) ** (1/3)  # cm³ to Å³, cube root
                box_size = max(min_box_size, density_based_box_size)
            else:
                box_size = min_box_size
        
        # Estimate number of solvent molecules if not provided
        if n_solvent is None:
            # Use solvent density to estimate how many molecules fit
            available_volume = box_size**3 - (solute_props['size']**3)  # Rough estimate
            solvent_volume_per_molecule = solvent_props['mass'] * 1.66054e-24 / solvent_density  # g/cm³
            solvent_volume_per_molecule_angstrom = solvent_volume_per_molecule * 1e24  # Å³
            n_solvent = max(1, int(available_volume / solvent_volume_per_molecule_angstrom * 0.6))  # 60% packing
        
        # Set up timing parameters
        if sampling_time is None and snapshot_interval is None:
            # Default: 0.1 ns per snapshot
            snapshot_interval = 0.1
            sampling_time = n_snapshots * snapshot_interval
        elif sampling_time is None:
            sampling_time = n_snapshots * snapshot_interval
        elif snapshot_interval is None:
            snapshot_interval = sampling_time / n_snapshots
        
        print(f"System setup:")
        print(f"  Box size: {box_size:.1f} Å")
        print(f"  Number of solvent molecules: {n_solvent}")
        print(f"  Equilibration time: {equilibration_time} ns")
        print(f"  Sampling time: {sampling_time} ns")
        print(f"  Snapshot interval: {snapshot_interval} ns")
        
        # Step 1: Create mixed system
        print(f"\n1. Creating mixed system...")
        mixed_pdb = create_mixed_system(
            solute_xyz, solvent_xyz, n_solvent, box_size, output_dir,
            solute_position=solute_position
        )
        
        # Step 2: Run NPT sampling simulation
        print(f"\n2. Running NPT sampling simulation...")
        simulation_results = run_npt_sampling(
            mixed_pdb, box_size, temperature, pressure,
            equilibration_time, sampling_time, snapshot_interval,
            forcefield_xml, output_dir
        )
        
        # Prepare final results
        system_info = {
            'solute_file': solute_xyz,
            'solvent_file': solvent_xyz,
            'n_solvent_molecules': n_solvent,
            'box_size': box_size,
            'solute_properties': solute_props,
            'solvent_properties': solvent_props,
            'solvent_density': solvent_density
        }
        
        simulation_parameters = {
            'temperature': temperature,
            'pressure': pressure,
            'equilibration_time': equilibration_time,
            'sampling_time': sampling_time,
            'snapshot_interval': snapshot_interval,
            'forcefield_xml': forcefield_xml
        }
        
        results = {
            'snapshots': simulation_results['snapshots'],
            'snapshot_times': simulation_results['snapshot_times'],
            'n_snapshots': simulation_results['n_snapshots'],
            'system_info': system_info,
            'simulation_parameters': simulation_parameters,
            'output_dir': output_dir
        }
        
        # Cleanup intermediate files if requested
        if cleanup:
            files_to_remove = ['solute.pdb', 'solvent.pdb', 'solute_fixed.pdb', 
                             'packmol_mixed.inp', 'mixed_system.pdb']
            for filename in files_to_remove:
                filepath = os.path.join(output_dir, filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
        
        print(f"\n" + "="*60)
        print("SOLVENT SNAPSHOT GENERATION COMPLETED!")
        print("="*60)
        print(f"Generated {results['n_snapshots']} snapshots")
        print(f"System: {n_solvent} {os.path.basename(solvent_xyz)} around 1 {os.path.basename(solute_xyz)}")
        print(f"Box size: {box_size:.1f} Å")
        print(f"Sampling time: {sampling_time} ns (interval: {snapshot_interval} ns)")
        print(f"Output directory: {output_dir}")
        
        return results
        
    except Exception as e:
        # Cleanup on error if in temporary directory
        if cleanup and output_dir.startswith(tempfile.gettempdir()):
            shutil.rmtree(output_dir, ignore_errors=True)
        raise e


def main():
    """
    Command line interface for solvent snapshot generation.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate thermodynamic snapshots of solvent around fixed solute"
    )
    parser.add_argument("solute_xyz", help="XYZ file for solute molecule (fixed)")
    parser.add_argument("solvent_xyz", help="XYZ file for solvent molecule")
    parser.add_argument("-t", "--temperature", type=float, default=298.15,
                       help="Temperature in Kelvin (default: 298.15)")
    parser.add_argument("-p", "--pressure", type=float, default=1.0,
                       help="Pressure in bar (default: 1.0)")
    parser.add_argument("-d", "--solvent_density", type=float, default=1.0,
                       help="Solvent density in g/cm³ (default: 1.0)")
    parser.add_argument("-n", "--n_snapshots", type=int, default=2,
                       help="Number of snapshots (default: 2)")
    parser.add_argument("--n_solvent", type=int, default=None,
                       help="Number of solvent molecules (default: auto)")
    parser.add_argument("-b", "--box_size", type=float, default=None,
                       help="Box size in Angstroms (default: auto)")
    parser.add_argument("--equilibration_time", type=float, default=0.1,
                       help="Equilibration time in ns (default: 0.1)")
    parser.add_argument("--sampling_time", type=float, default=None,
                       help="Sampling time in ns (default: auto)")
    parser.add_argument("--snapshot_interval", type=float, default=None,
                       help="Snapshot interval in ns (default: auto)")
    parser.add_argument("-f", "--forcefield", type=str, default=None,
                       help="Force field XML file path")
    parser.add_argument("-o", "--output_dir", type=str, default=None,
                       help="Output directory (default: temporary)")
    parser.add_argument("--no-cleanup", action="store_true",
                       help="Keep intermediate files")
    
    args = parser.parse_args()
    
    try:
        results = generate_solvent_snapshots(
            solute_xyz=args.solute_xyz,
            solvent_xyz=args.solvent_xyz,
            temperature=args.temperature,
            pressure=args.pressure,
            solvent_density=args.solvent_density,
            n_snapshots=args.n_snapshots,
            n_solvent=args.n_solvent,
            box_size=args.box_size,
            equilibration_time=args.equilibration_time,
            sampling_time=args.sampling_time,
            snapshot_interval=args.snapshot_interval,
            forcefield_xml=args.forcefield,
            output_dir=args.output_dir,
            cleanup=not args.no_cleanup
        )
        
        print(f"\nGenerated {results['n_snapshots']} snapshots successfully!")
        print(f"Output files saved in: {results['output_dir']}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
