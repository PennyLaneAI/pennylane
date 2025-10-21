#!/usr/bin/env python3
"""
NPT Sampler for Solute-Solvent Systems

This module provides functionality to sample thermodynamic distributions of solvent
molecules around a fixed solute using NPT molecular dynamics simulations with OpenMM.

The workflow:
1. Takes a fixed solute molecule and a solvent molecule
2. Uses packmol to create initial liquid configurations
3. Runs NPT simulation at specified temperature/pressure
4. Samples N snapshots of solvent distributions around the solute
5. Returns coordinate snapshots for further analysis

Author: Generated for PennyLane QML vibrational workflows
Date: September 2025
"""

import os
import sys
import tempfile
import shutil
import subprocess
import numpy as np
import argparse
import json
from typing import List, Dict, Any, Optional, Tuple, Union
import time

# Import the debugged functions from npt_density.py
from npt_density import xyz_to_pdb, create_manual_configuration

# Try to import OpenMM
try:
    from openmm.app import PDBFile, Simulation, ForceField
    from openmm import unit
    import openmm as mm
    OPENMM_AVAILABLE = True
except ImportError:
    import warnings
    warnings.warn("OpenMM not available. NPT sampling will not work.")
    OPENMM_AVAILABLE = False
    PDBFile = None
    ForceField = None
    Simulation = None


def read_xyz_properties(xyz_file: str) -> Dict[str, Any]:
    """
    Read XYZ file and extract molecular properties.
    
    Parameters
    ----------
    xyz_file : str
        Path to XYZ file
        
    Returns
    -------
    dict
        Dictionary with molecular properties
    """
    atomic_masses = {
        'H': 1.008, 'C': 12.01, 'N': 14.007, 'O': 15.999, 'F': 18.998,
        'P': 30.974, 'S': 32.06, 'Cl': 35.45, 'Br': 79.904, 'I': 126.90
    }
    
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    
    n_atoms = int(lines[0].strip())
    elements = []
    coordinates = []
    total_mass = 0.0
    
    for i in range(2, 2 + n_atoms):
        parts = lines[i].strip().split()
        element = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        
        elements.append(element)
        coordinates.append([x, y, z])
        total_mass += atomic_masses.get(element, 15.0)
    
    coords_array = np.array(coordinates)
    size = np.max(np.ptp(coords_array, axis=0))  # Maximum extent
    
    return {
        'elements': elements,
        'coordinates': coordinates,
        'mass': total_mass,
        'size': size,
        'n_atoms': n_atoms
    }

# OpenMM imports
try:
    import openmm
    from openmm import app
    from openmm.app import PDBFile, Simulation, Modeller
    from openmm.unit import *
    from openmm.unit import nanometer, kelvin, bar, picosecond, picoseconds, kilojoule_per_mole
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False
    print("Warning: OpenMM not available. NPT sampling will not work.")


def create_manual_solute_solvent_config(solute_pdb: str, solvent_pdb: str,
                                        n_solvents: int, box_size: float,
                                        output_pdb: str, 
                                        solute_center: Tuple[float, float, float] = (0, 0, 0),
                                        min_distance: float = 3.5) -> str:
    """
    Create manual solute-solvent configuration using the debugged function from npt_density.py.
    This is a wrapper that places one solute molecule and then uses the existing manual
    configuration for the solvent molecules.
    
    Parameters
    ----------
    solute_pdb : str
        Path to solute PDB file
    solvent_pdb : str
        Path to solvent PDB file
    n_solvents : int
        Number of solvent molecules
    box_size : float
        Size of simulation box in Angstroms
    output_pdb : str
        Output PDB file path
    solute_center : tuple
        Position to place solute molecule (x, y, z)
    min_distance : float
        Minimum distance between molecules in Angstroms
        
    Returns
    -------
    str
        Path to output PDB file
    """
    # For now, let's use the existing create_manual_configuration and then
    # manually add the solute. This is a simple approach for debugging.
    
    # First create a configuration with all solvent molecules
    temp_solvent_config = output_pdb.replace('.pdb', '_temp_solvent.pdb')
    create_manual_configuration(solvent_pdb, n_solvents, box_size, 
                               os.path.dirname(output_pdb), temp_solvent_config)
    
    # Read the solute PDB
    solute_lines = []
    with open(solute_pdb, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                solute_lines.append(line.strip())
    
    # Combine solute at center with solvent configuration
    with open(output_pdb, 'w') as out_f:
        out_f.write("REMARK   Manual solute-solvent configuration\n")
        
        atom_serial = 1
        
        # Write solute at center (residue 1)
        for line in solute_lines:
            # Parse and update coordinates
            x_orig = float(line[30:38])
            y_orig = float(line[38:46]) 
            z_orig = float(line[46:54])
            
            # Translate to center
            new_x = x_orig + solute_center[0]
            new_y = y_orig + solute_center[1]
            new_z = z_orig + solute_center[2]
            
            # Update line with new serial and coordinates
            new_line = line[:6] + f"{atom_serial:5d}" + line[11:22] + "   1" + line[26:30]
            new_line += f"{new_x:8.3f}{new_y:8.3f}{new_z:8.3f}" + line[54:]
            out_f.write(new_line + "\n")
            atom_serial += 1
        
        # Read and append solvent molecules (starting from residue 2)
        with open(temp_solvent_config, 'r') as temp_f:
            for line in temp_f:
                if line.startswith('ATOM'):
                    # Update atom serial and residue number
                    res_num = int(line[22:26]) + 1  # Shift residue numbers
                    new_line = line[:6] + f"{atom_serial:5d}" + line[11:22] + f"{res_num:4d}" + line[26:]
                    out_f.write(new_line)
                    atom_serial += 1
                elif line.startswith('END'):
                    out_f.write(line)
        
        out_f.write("END\n")
    
    # Clean up temporary file
    if os.path.exists(temp_solvent_config):
        os.remove(temp_solvent_config)
    
    print(f"Manual configuration created with 1 solute + {n_solvents} solvents")
    return output_pdb


def run_packmol_solute_solvent(solute_pdb: str, solvent_pdb: str, 
                               n_solvents: int, box_size: float,
                               output_pdb: str, solute_center: Tuple[float, float, float] = (0, 0, 0),
                               min_distance: float = 2.5) -> str:
    """
    Use packmol to create a system with one solute molecule surrounded by solvent.
    
    Parameters
    ----------
    solute_pdb : str
        Path to solute PDB file
    solvent_pdb : str
        Path to solvent PDB file
    n_solvents : int
        Number of solvent molecules
    box_size : float
        Size of simulation box in Angstroms
    output_pdb : str
        Output PDB file path
    solute_center : tuple
        Position to place solute molecule (x, y, z)
    min_distance : float
        Minimum distance between molecules in Angstroms
        
    Returns
    -------
    str
        Path to output PDB file
    """
    # Create packmol input file
    packmol_input = f"""tolerance {min_distance:.1f}
filetype pdb
output {os.path.basename(output_pdb)}
seed -1

# Fixed solute molecule at center
structure {os.path.basename(solute_pdb)}
  number 1
  fixed {solute_center[0]:.3f} {solute_center[1]:.3f} {solute_center[2]:.3f} 0. 0. 0.
end structure

# Solvent molecules distributed in box
structure {os.path.basename(solvent_pdb)}
  number {n_solvents}
  inside box {-box_size/2:.3f} {-box_size/2:.3f} {-box_size/2:.3f} {box_size/2:.3f} {box_size/2:.3f} {box_size/2:.3f}
end structure
"""
    
    # Write packmol input file
    input_dir = os.path.dirname(output_pdb)
    packmol_input_file = os.path.join(input_dir, "packmol_input.inp")
    
    with open(packmol_input_file, 'w') as f:
        f.write(packmol_input)
    
    # Run packmol
    try:
        print(f"Running packmol (timeout: 60s)...")
        result = subprocess.run(
            ['packmol', '<', packmol_input_file],
            cwd=input_dir,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60  # 1 minute timeout for debugging
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Packmol failed: {result.stderr}")
            
        if not os.path.exists(output_pdb):
            raise RuntimeError("Packmol did not create output file")
            
        print(f"Packmol completed successfully. Created: {output_pdb}")
        return output_pdb
        
    except (subprocess.TimeoutExpired, RuntimeError, FileNotFoundError) as e:
        print(f"Packmol failed: {e}")
        print("Using manual configuration as fallback...")
        return create_manual_solute_solvent_config(
            solute_pdb, solvent_pdb, n_solvents, box_size, 
            output_pdb, solute_center, min_distance
        )


def create_combined_forcefield(solute_ff: str, solvent_ff: str, output_ff: str,
                              solute_residue: str = "SOL", solvent_residue: str = "SLV") -> str:
    """
    Create a combined force field file from solute and solvent force fields.
    This handles the case where the same molecule is used for both solute and solvent.
    
    Parameters
    ----------
    solute_ff : str
        Path to solute force field file
    solvent_ff : str  
        Path to solvent force field file
    output_ff : str
        Path for output combined force field
    solute_residue : str
        Residue name for solute
    solvent_residue : str
        Residue name for solvent
        
    Returns
    -------
    str
        Path to combined force field file
    """
    import xml.etree.ElementTree as ET
    
    # If files are the same, create a copy with different residue names
    if solute_ff == solvent_ff:
        tree = ET.parse(solute_ff)
        root = tree.getroot()
        
        # Find residues section
        residues = root.find('Residues')
        if residues is not None:
            # Get the first residue as template
            template_residue = residues.find('Residue')
            if template_residue is not None:
                # Create new residue for solvent with different name
                new_residue = ET.deepcopy(template_residue)
                new_residue.set('name', solvent_residue)
                residues.append(new_residue)
                
                # Update original residue name for solute
                template_residue.set('name', solute_residue)
        
        tree.write(output_ff)
        return output_ff
    else:
        # Different force fields - would need more complex merging
        # For now, just copy the solute force field
        import shutil
        shutil.copy(solute_ff, output_ff)
        return output_ff


def setup_openmm_solute_solvent_system(packed_pdb: str, 
                                       solute_forcefield: Optional[str] = None,
                                       solvent_forcefield: Optional[str] = None,
                                       box_size: float = 30.0,
                                       output_dir: str = None) -> Tuple[Any, Any, Any]:
    """
    Set up OpenMM system for solute-solvent simulation.
    
    Parameters
    ----------
    packed_pdb : str
        Path to packmol-generated PDB file
    solute_forcefield : str, optional
        Path to solute force field XML file
    solvent_forcefield : str, optional
        Path to solvent force field XML file
    box_size : float
        Simulation box size in Angstroms
        
    Returns
    -------
    tuple
        (pdb_object, system, topology)
    """
    if not HAS_OPENMM:
        raise RuntimeError("OpenMM is required for NPT simulations")
    
    # Load PDB file
    pdb_obj = PDBFile(packed_pdb)
    
    # Set up force field
    if solute_forcefield or solvent_forcefield:
        # Use custom force fields
        if solute_forcefield and solvent_forcefield:
            # Create combined force field if both specified
            combined_ff = os.path.join(output_dir or os.path.dirname(packed_pdb), "combined.xml")
            create_combined_forcefield(solute_forcefield, solvent_forcefield, combined_ff)
            forcefield = app.ForceField(combined_ff)
        elif solute_forcefield:
            forcefield = app.ForceField(solute_forcefield)
        else:
            forcefield = app.ForceField(solvent_forcefield)
    else:
        # Use default AMBER force field
        forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    
    # Create modeller and add periodic box
    modeller = Modeller(pdb_obj.topology, pdb_obj.positions)
    box_size_nm = box_size * 0.1  # Convert Angstroms to nanometers
    modeller.topology.setPeriodicBoxVectors([
        [box_size_nm, 0, 0] * nanometer,
        [0, box_size_nm, 0] * nanometer,
        [0, 0, box_size_nm] * nanometer
    ])
    
    # Create system
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=min(1.0, box_size_nm * 0.4) * nanometer,
        constraints=app.HBonds
    )
    
    return pdb_obj, system, modeller.topology


def run_npt_sampling(pdb_obj: Any, system: Any, topology: Any,
                     temperature: float = 298.15, pressure: float = 1.0,
                     equilibration_time: float = 1.0, sampling_time: float = 5.0,
                     n_snapshots: int = 10, snapshot_interval: float = None,
                     output_dir: str = None) -> List[Dict[str, Any]]:
    """
    Run NPT simulation and collect snapshots of solvent configurations.
    
    Parameters
    ----------
    pdb_obj : PDBFile
        OpenMM PDB object
    system : System
        OpenMM system object
    topology : Topology
        OpenMM topology object
    temperature : float
        Temperature in Kelvin
    pressure : float
        Pressure in bar
    equilibration_time : float
        Equilibration time in nanoseconds
    sampling_time : float
        Total sampling time in nanoseconds
    n_snapshots : int
        Number of snapshots to collect
    snapshot_interval : float, optional
        Time interval between snapshots in nanoseconds
        If None, calculated as sampling_time / n_snapshots
    output_dir : str, optional
        Directory to save snapshot PDB files
        
    Returns
    -------
    list
        List of dictionaries containing snapshot information
    """
    if not HAS_OPENMM:
        raise RuntimeError("OpenMM is required for NPT simulations")
    
    # Calculate snapshot interval if not provided
    if snapshot_interval is None:
        snapshot_interval = sampling_time / n_snapshots
    
    # Set up integrator and simulation
    integrator = openmm.LangevinMiddleIntegrator(
        temperature * kelvin, 1/picosecond, 0.002 * picoseconds
    )
    
    # Add barostat for NPT
    system.addForce(openmm.MonteCarloBarostat(pressure * bar, temperature * kelvin))
    
    # Create simulation
    simulation = Simulation(topology, system, integrator)
    simulation.context.setPositions(pdb_obj.positions)
    
    # Minimize energy
    print("Minimizing energy...")
    simulation.minimizeEnergy()
    
    # Equilibration
    equilibration_steps = int(equilibration_time * 1000000 / 2)  # 2 fs timestep
    print(f"Running equilibration for {equilibration_time} ns ({equilibration_steps} steps)...")
    simulation.step(equilibration_steps)
    
    # Production run with snapshot collection
    snapshot_steps = int(snapshot_interval * 1000000 / 2)  # 2 fs timestep
    snapshots = []
    
    print(f"Collecting {n_snapshots} snapshots every {snapshot_interval} ns...")
    
    for i in range(n_snapshots):
        # Run simulation for snapshot interval
        simulation.step(snapshot_steps)
        
        # Get current state
        state = simulation.context.getState(getPositions=True, getEnergy=True)
        positions = state.getPositions()
        potential_energy = state.getPotentialEnergy()
        
        # Create snapshot dictionary
        snapshot = {
            'snapshot_id': i + 1,
            'time_ns': equilibration_time + (i + 1) * snapshot_interval,
            'positions': positions,
            'potential_energy': potential_energy.value_in_unit(kilojoule_per_mole),
            'box_vectors': state.getPeriodicBoxVectors()
        }
        
        # Save PDB file if output directory provided
        if output_dir:
            snapshot_pdb = os.path.join(output_dir, f"snapshot_{i+1:03d}.pdb")
            with open(snapshot_pdb, 'w') as f:
                PDBFile.writeFile(topology, positions, f)
            snapshot['pdb_file'] = snapshot_pdb
        
        snapshots.append(snapshot)
        
        print(f"  Snapshot {i+1}/{n_snapshots} collected at {snapshot['time_ns']:.3f} ns, "
              f"Energy: {potential_energy.value_in_unit(kilojoule_per_mole):.1f} kJ/mol")
    
    return snapshots


def sample_solvent_distributions(solute_xyz: str, solvent_xyz: str,
                                 n_solvents: int = 50, 
                                 temperature: float = 298.15, pressure: float = 1.0,
                                 solvent_density: float = 1.0,
                                 equilibration_time: float = 1.0, sampling_time: float = 5.0,
                                 n_snapshots: int = 2, snapshot_interval: float = None,
                                 solute_forcefield: str = None, solvent_forcefield: str = None,
                                 output_dir: str = None, cleanup: bool = True) -> Dict[str, Any]:
    """
    Sample thermodynamic distributions of solvent molecules around a fixed solute.
    
    This function creates NPT simulations with a fixed solute molecule surrounded by
    solvent molecules, then samples snapshots of the solvent configurations for 
    further analysis.
    
    Parameters
    ----------
    solute_xyz : str
        Path to XYZ file containing solute molecule geometry
    solvent_xyz : str
        Path to XYZ file containing solvent molecule geometry  
    n_solvents : int, default=50
        Number of solvent molecules to include
    temperature : float, default=298.15
        Simulation temperature in Kelvin
    pressure : float, default=1.0
        Simulation pressure in bar
    solvent_density : float, default=1.0
        Target density of solvent in g/cm³ (used for box sizing)
    equilibration_time : float, default=1.0
        Equilibration time in nanoseconds
    sampling_time : float, default=5.0
        Total sampling time in nanoseconds
    n_snapshots : int, default=2
        Number of configuration snapshots to collect
    snapshot_interval : float, optional
        Time interval between snapshots in nanoseconds
        If None, calculated as sampling_time / n_snapshots
    solute_forcefield : str, optional
        Path to custom force field XML file for solute
    solvent_forcefield : str, optional
        Path to custom force field XML file for solvent
    output_dir : str, optional
        Directory for output files. If None, creates temporary directory
    cleanup : bool, default=True
        Whether to clean up intermediate files
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'snapshots': List of snapshot dictionaries with positions and energies
        - 'solute_info': Information about the solute molecule
        - 'solvent_info': Information about the solvent molecule
        - 'simulation_params': Simulation parameters used
        - 'output_dir': Directory containing output files
        
    Examples
    --------
    >>> # Sample water around HCN
    >>> results = sample_solvent_distributions(
    ...     solute_xyz="HCN.xyz",
    ...     solvent_xyz="H2O.xyz", 
    ...     n_solvents=30,
    ...     temperature=280.0,
    ...     n_snapshots=5
    ... )
    >>> print(f"Collected {len(results['snapshots'])} snapshots")
    """
    if not HAS_OPENMM:
        raise RuntimeError("OpenMM is required for NPT sampling")
    
    # Create output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="npt_sampling_")
        cleanup_dir = cleanup
    else:
        os.makedirs(output_dir, exist_ok=True)
        cleanup_dir = False
    
    try:
        print(f"Starting NPT solvent sampling")
        print(f"Solute: {solute_xyz}")
        print(f"Solvent: {solvent_xyz}")
        print(f"Conditions: T = {temperature} K, P = {pressure} bar")
        print(f"System: 1 solute + {n_solvents} solvents")
        print(f"Output directory: {output_dir}")
        
        # Read molecule information using debugged function
        solute_props = read_xyz_properties(solute_xyz)
        solvent_props = read_xyz_properties(solvent_xyz)
        
        solute_mass = solute_props['mass']
        solvent_mass = solvent_props['mass']
        solute_size = solute_props['size']
        solvent_size = solvent_props['size']
        
        print(f"\nSolute: {len(solute_props['elements'])} atoms, mass = {solute_mass:.2f} amu")
        print(f"Solvent: {len(solvent_props['elements'])} atoms, mass = {solvent_mass:.2f} amu")
        
        # Estimate box size based on solvent density
        total_solvent_mass_g = n_solvents * solvent_mass * 1.66054e-24  # amu to grams
        solvent_volume_cm3 = total_solvent_mass_g / solvent_density
        solvent_volume_angstrom3 = solvent_volume_cm3 * 1e24  # cm³ to Å³
        
        # Add some space for the solute and buffer
        total_volume = solvent_volume_angstrom3 * 1.5  # 50% extra space
        box_size = total_volume ** (1/3)
        
        # Ensure minimum box size for stability
        min_box_size = max(15.0, max(solute_size, solvent_size) * 3)
        box_size = max(box_size, min_box_size)
        
        print(f"\nEstimated box size: {box_size:.2f} Å")
        
        # Convert molecules to PDB format using debugged function
        print("\n1. Converting molecules to PDB format...")
        solute_pdb = xyz_to_pdb(solute_xyz, os.path.join(output_dir, "solute.pdb"))
        solvent_pdb = xyz_to_pdb(solvent_xyz, os.path.join(output_dir, "solvent.pdb"))
        
        # Create initial configuration with packmol
        print("\n2. Creating initial configuration with packmol...")
        packed_pdb = os.path.join(output_dir, "packed_system.pdb")
        
        # Use packmol with fallback to manual configuration
        run_packmol_solute_solvent(
            solute_pdb, solvent_pdb, n_solvents, box_size, 
            packed_pdb, solute_center=(0, 0, 0), min_distance=2.5
        )
        
        # Set up OpenMM system
        print("\n3. Setting up OpenMM system...")
        pdb_obj, system, topology = setup_openmm_solute_solvent_system(
            packed_pdb, solute_forcefield, solvent_forcefield, box_size, output_dir
        )
        
        # Run NPT sampling
        print("\n4. Running NPT simulation and collecting snapshots...")
        snapshots = run_npt_sampling(
            pdb_obj, system, topology, temperature, pressure,
            equilibration_time, sampling_time, n_snapshots, 
            snapshot_interval, output_dir
        )
        
        # Prepare results
        results = {
            'snapshots': snapshots,
            'solute_info': {
                'xyz_file': solute_xyz,
                'symbols': solute_props['elements'],
                'coordinates': solute_props['coordinates'],
                'mass_amu': solute_mass,
                'n_atoms': len(solute_props['elements'])
            },
            'solvent_info': {
                'xyz_file': solvent_xyz,
                'symbols': solvent_props['elements'],
                'coordinates': solvent_props['coordinates'],
                'mass_amu': solvent_mass,
                'n_atoms': len(solvent_props['elements']),
                'n_molecules': n_solvents
            },
            'simulation_params': {
                'temperature_K': temperature,
                'pressure_bar': pressure,
                'solvent_density_g_cm3': solvent_density,
                'box_size_angstrom': box_size,
                'equilibration_time_ns': equilibration_time,
                'sampling_time_ns': sampling_time,
                'n_snapshots': n_snapshots,
                'snapshot_interval_ns': snapshot_interval or (sampling_time / n_snapshots)
            },
            'output_dir': output_dir
        }
        
        print(f"\n" + "="*60)
        print("NPT SOLVENT SAMPLING COMPLETED!")
        print("="*60)
        print(f"Collected {len(snapshots)} snapshots")
        print(f"Snapshot files saved in: {output_dir}")
        print(f"Energy range: {min(s['potential_energy'] for s in snapshots):.1f} to "
              f"{max(s['potential_energy'] for s in snapshots):.1f} kJ/mol")
        
        return results
        
    except Exception as e:
        # Cleanup on error if in temporary directory
        if cleanup_dir and output_dir.startswith(tempfile.gettempdir()):
            shutil.rmtree(output_dir, ignore_errors=True)
        raise e


if __name__ == "__main__":
    """
    Example usage and testing
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Sample solvent distributions around solute")
    parser.add_argument("solute_xyz", help="Solute molecule XYZ file")
    parser.add_argument("solvent_xyz", help="Solvent molecule XYZ file")
    parser.add_argument("-n", "--n_solvents", type=int, default=20,
                       help="Number of solvent molecules (default: 20)")
    parser.add_argument("-T", "--temperature", type=float, default=298.15,
                       help="Temperature in K (default: 298.15)")
    parser.add_argument("-P", "--pressure", type=float, default=1.0,
                       help="Pressure in bar (default: 1.0)")
    parser.add_argument("-d", "--density", type=float, default=1.0,
                       help="Solvent density in g/cm³ (default: 1.0)")
    parser.add_argument("--equilibration_time", type=float, default=0.5,
                       help="Equilibration time in ns (default: 0.5)")
    parser.add_argument("--sampling_time", type=float, default=2.0,
                       help="Sampling time in ns (default: 2.0)")
    parser.add_argument("--n_snapshots", type=int, default=2,
                       help="Number of snapshots (default: 2)")
    parser.add_argument("--solute_ff", type=str, default=None,
                       help="Solute force field XML file")
    parser.add_argument("--solvent_ff", type=str, default=None,
                       help="Solvent force field XML file")
    parser.add_argument("-o", "--output_dir", type=str, default=None,
                       help="Output directory")
    
    args = parser.parse_args()
    
    try:
        results = sample_solvent_distributions(
            solute_xyz=args.solute_xyz,
            solvent_xyz=args.solvent_xyz,
            n_solvents=args.n_solvents,
            temperature=args.temperature,
            pressure=args.pressure,
            solvent_density=args.density,
            equilibration_time=args.equilibration_time,
            sampling_time=args.sampling_time,
            n_snapshots=args.n_snapshots,
            solute_forcefield=args.solute_ff,
            solvent_forcefield=args.solvent_ff,
            output_dir=args.output_dir
        )
        
        print(f"\nSampling completed successfully!")
        print(f"Results available in: {results['output_dir']}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
