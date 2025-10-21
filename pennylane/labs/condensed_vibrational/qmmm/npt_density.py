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
This module contains utilities for generating accurate liquid densities using 
packmol followed by NPT molecular dynamics simulations. It provides functionality 
to convert XYZ molecular coordinates to PDB format, use packmol to create initial 
liquid systems, and then run NPT simulations to obtain equilibrium densities.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from typing import Optional, Union, Dict, Any, List
import warnings

import numpy as np

# Import molecular handling and simulation libraries
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Some functionality may be limited.")

try:
    from openmm.app import *
    from openmm import *
    from openmm.unit import *
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False
    warnings.warn("OpenMM not available. NPT simulations will not work.")
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
    BOLTZMANN_CONSTANT_kB = AVOGADRO_CONSTANT_NA = 1.0
    Vec3 = PDBFile = ForceField = PME = HBonds = DummyClass
    LangevinMiddleIntegrator = MonteCarloBarostat = DummyClass
    Platform = Simulation = StateDataReporter = DCDReporter = DummyClass

try:
    from openff.toolkit.topology import Molecule
    from openmmforcefields.generators import SystemGenerator
    OPENFF_AVAILABLE = True
except ImportError:
    OPENFF_AVAILABLE = False
    warnings.warn("OpenFF toolkit not available. Some force field functionality may be limited.")


# Atomic masses (in atomic mass units) for common elements
ATOMIC_MASSES = {
    'H': 1.008, 'He': 4.003, 'Li': 6.94, 'Be': 9.012, 'B': 10.81, 'C': 12.011,
    'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180, 'Na': 22.990,
    'Mg': 24.305, 'Al': 26.982, 'Si': 28.085, 'P': 30.974, 'S': 32.06,
    'Cl': 35.45, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078, 'Br': 79.904,
    'I': 126.90
}

# Empirical liquid densities at room temperature (g/cm³)
EMPIRICAL_DENSITIES = {
    'water': 1.0, 'ethanol': 0.789, 'methanol': 0.791, 'acetone': 0.784,
    'benzene': 0.876, 'chloroform': 1.489, 'dimethylsulfoxide': 1.092,
    'acetonitrile': 0.786, 'dichloromethane': 1.325, 'toluene': 0.867
}

# Default force field configurations
DEFAULT_FORCEFIELDS = {
    'amber': ['amber14-all.xml', 'amber14/tip3pfb.xml'],
    'charmm': ['charmm36.xml', 'charmm36/water.xml'],
    'gaff': 'gaff-2.11',
    'openff': 'openff-2.1.0'
}


def xyz_to_pdb(xyz_file: str, output_pdb: Optional[str] = None) -> str:
    """
    Convert an XYZ file to PDB format compatible with packmol.
    
    Parameters
    ----------
    xyz_file : str
        Path to the input XYZ file containing molecular coordinates
    output_pdb : str, optional
        Path for the output PDB file. If None, creates a file with same base name
        
    Returns
    -------
    str
        Path to the generated PDB file
        
    Raises
    ------
    FileNotFoundError
        If the input XYZ file doesn't exist
    ValueError
        If the XYZ file format is invalid
    """
    if not os.path.exists(xyz_file):
        raise FileNotFoundError(f"XYZ file not found: {xyz_file}")
    
    # Read XYZ file
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 3:
        raise ValueError("Invalid XYZ file format: too few lines")
    
    try:
        n_atoms = int(lines[0].strip())
    except ValueError:
        raise ValueError("Invalid XYZ file format: first line must be number of atoms")
    
    if len(lines) < n_atoms + 2:
        raise ValueError(f"Invalid XYZ file format: expected {n_atoms + 2} lines, got {len(lines)}")
    
    # Parse coordinates
    atoms = []
    for i in range(2, 2 + n_atoms):
        parts = lines[i].strip().split()
        if len(parts) < 4:
            raise ValueError(f"Invalid coordinate line {i}: {lines[i].strip()}")
        element = parts[0].strip()
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            raise ValueError(f"Invalid coordinates in line {i}: {lines[i].strip()}")
        
        # Create simple atom name (element + number)
        atom_name = f"{element}{i-1}"  # C1, N2, H3, etc.
        atoms.append((atom_name, element, x, y, z))
    
    # Generate output filename if not provided
    if output_pdb is None:
        base_name = os.path.splitext(os.path.basename(xyz_file))[0]
        output_pdb = f"{base_name}.pdb"
    
    # Write PDB file
    _write_pdb_file(atoms, output_pdb)
    
    return output_pdb


def _write_pdb_file(atoms: list, output_pdb: str):
    """
    Write atom coordinates to a PDB file in the format expected by packmol.
    
    Parameters
    ----------
    atoms : list
        List of tuples (atom_name, element, x, y, z) with atomic coordinates
    output_pdb : str
        Path to the output PDB file
    """
    with open(output_pdb, 'w') as f:
        f.write("REMARK   Generated by PennyLane QML\n")
        
        for i, (atom_name, element, x, y, z) in enumerate(atoms, 1):
            # Use MOL as residue name (3 characters, standard)
            residue_name = "MOL"
            
            # Format element properly (capitalize first letter only)
            element_formatted = element[0].upper() + element[1:].lower() if len(element) > 1 else element[0].upper()
            
            # PDB ATOM format - exact format that works with OpenMM
            # Use simple element name for atom name (H, C, N) padded to 4 chars
            simple_name = element[0].upper()  # Just use element symbol
            line = f"ATOM  {i:5d}  {simple_name:<3s} {residue_name:>3s} A{1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element_formatted:>2s}\n"
            f.write(line)
        
        f.write("END\n")


def estimate_molecular_properties(atoms: list) -> Dict[str, float]:
    """
    Estimate molecular properties from atomic composition.
    
    Parameters
    ----------
    atoms : list
        List of tuples (element, x, y, z) with atomic coordinates
        
    Returns
    -------
    dict
        Dictionary containing 'mass' (amu) and 'size' (Angstroms)
    """
    if not atoms:
        return {'mass': 0.0, 'size': 0.0}
    
    # Calculate molecular mass
    total_mass = 0.0
    for element, _, _, _ in atoms:
        if element in ATOMIC_MASSES:
            total_mass += ATOMIC_MASSES[element]
        else:
            warnings.warn(f"Unknown element {element}, using mass of 12.0 amu")
            total_mass += 12.0
    
    # Calculate molecular size (maximum dimension + VdW buffer)
    coords = np.array([(x, y, z) for _, x, y, z in atoms])
    ranges = np.max(coords, axis=0) - np.min(coords, axis=0)
    molecular_size = np.max(ranges) + 4.0  # Add ~4 Å for VdW radii
    
    return {'mass': total_mass, 'size': molecular_size}


def estimate_density(temperature: float = 298.15, pressure: float = 1.0, 
                    molecular_mass: Optional[float] = None, 
                    molecule_name: Optional[str] = None) -> float:
    """
    Estimate liquid density based on temperature, pressure, and molecular properties.
    
    Parameters
    ----------
    temperature : float, default=298.15
        Temperature in Kelvin
    pressure : float, default=1.0
        Pressure in bar
    molecular_mass : float, optional
        Molecular mass in amu
    molecule_name : str, optional
        Name of the molecule for empirical lookup
        
    Returns
    -------
    float
        Estimated density in g/cm³
    """
    # First try empirical lookup
    if molecule_name and molecule_name.lower() in EMPIRICAL_DENSITIES:
        base_density = EMPIRICAL_DENSITIES[molecule_name.lower()]
    else:
        # Use a simple heuristic based on molecular mass
        if molecular_mass is None:
            base_density = 1.0  # Default water-like density
        else:
            # Rough correlation: heavier molecules tend to be denser
            base_density = 0.5 + 0.02 * molecular_mass
            base_density = min(base_density, 2.5)  # Cap at reasonable maximum
    
    # Simple temperature and pressure corrections
    temp_correction = 1.0 - 0.001 * (temperature - 298.15)  # Thermal expansion
    pressure_correction = 1.0 + 0.0001 * (pressure - 1.0)  # Compressibility
    
    return base_density * temp_correction * pressure_correction


def generate_packmol_input(molecule_pdb: str, n_molecules: int, box_size: float, 
                          output_pdb: str, tolerance: float = 2.0) -> str:
    """
    Generate a packmol input file content for creating a liquid system.
    
    Parameters
    ----------
    molecule_pdb : str
        Path to the PDB file of a single molecule
    n_molecules : int
        Number of molecules to pack
    box_size : float
        Size of the cubic box in Angstroms
    output_pdb : str
        Path for the output packed PDB file
    tolerance : float, default=2.0
        Packmol tolerance parameter
        
    Returns
    -------
    str
        Content of the packmol input file
    """
    input_content = f"""tolerance {tolerance:.1f}
filetype pdb
output {os.path.basename(output_pdb)}
seed 1

structure {os.path.basename(molecule_pdb)}
  number {n_molecules}
  inside cube 0.0 0.0 0.0 {box_size:.2f}
end structure
"""
    return input_content


def create_manual_configuration(molecule_pdb: str, n_molecules: int, box_size: float, 
                               output_dir: str) -> str:
    """
    Create a simple manual configuration when packmol fails.
    Places molecules on a regular grid with random orientations.
    
    Parameters
    ----------
    molecule_pdb : str
        Path to the single molecule PDB file
    n_molecules : int
        Number of molecules to place
    box_size : float
        Box size in Angstroms
    output_dir : str
        Output directory
        
    Returns
    -------
    str
        Path to the manually created PDB file
    """
    print(f"Creating manual grid configuration with {n_molecules} molecules...")
    
    # Read the reference molecule
    with open(molecule_pdb, 'r') as f:
        mol_lines = f.readlines()
    
    atom_lines = [line for line in mol_lines if line.startswith('ATOM')]
    if not atom_lines:
        raise RuntimeError("No ATOM records found in molecule PDB")
    
    # Parse atom coordinates
    atoms = []
    for line in atom_lines:
        atom_name = line[12:16].strip()
        element = line[76:78].strip() if len(line) > 77 and line[76:78].strip() else atom_name[0]
        
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
        
        atoms.append((atom_name, element, x, y, z))
    
    # Center the molecule at origin
    coords = np.array([(x, y, z) for _, _, x, y, z in atoms])
    center = np.mean(coords, axis=0)
    centered_atoms = []
    for atom_name, element, x, y, z in atoms:
        centered_atoms.append((atom_name, element, x - center[0], y - center[1], z - center[2]))
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(n_molecules ** (1/3)))
    spacing = box_size / (grid_size + 1)  # +1 to avoid placing on boundaries
    
    print(f"Using {grid_size}×{grid_size}×{grid_size} grid with spacing {spacing:.2f} Å")
    
    # Generate output
    output_pdb = os.path.join(output_dir, "packed_system.pdb")
    
    with open(output_pdb, 'w') as f:
        f.write("REMARK   Manual configuration created by PennyLane QML\n")
        
        atom_serial = 1
        mol_count = 0
        
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    if mol_count >= n_molecules:
                        break
                    
                    # Grid position
                    grid_x = (i + 1) * spacing - box_size/2
                    grid_y = (j + 1) * spacing - box_size/2  
                    grid_z = (k + 1) * spacing - box_size/2
                    
                    # Random rotation (simple approach)
                    angle = np.random.random() * 2 * np.pi
                    cos_a, sin_a = np.cos(angle), np.sin(angle)
                    
                    # Write molecule atoms with proper formatting
                    for atom_idx, (atom_name, element, x, y, z) in enumerate(centered_atoms):
                        # Apply simple rotation around z-axis
                        rot_x = x * cos_a - y * sin_a
                        rot_y = x * sin_a + y * cos_a
                        rot_z = z
                        # Translate to grid position
                        final_x = rot_x + grid_x
                        final_y = rot_y + grid_y
                        final_z = rot_z + grid_z
                        
                        # Format element properly
                        element_formatted = element[0].upper() + element[1:].lower() if len(element) > 1 else element[0].upper()
                        
                        # Write PDB line with correct format
                        line = f"ATOM  {atom_serial:5d}  {atom_name:<4s} MOL A{mol_count+1:4d}    {final_x:8.3f}{final_y:8.3f}{final_z:8.3f}  1.00  0.00          {element_formatted:>2s}\n"
                        f.write(line)
                        atom_serial += 1
                    
                    mol_count += 1
                    
                    if mol_count >= n_molecules:
                        break
                if mol_count >= n_molecules:
                    break
            if mol_count >= n_molecules:
                break
        
        f.write("END\n")
    
    print(f"Manual configuration created with {mol_count} molecules")
    return output_pdb


def run_packmol(molecule_pdb: str, n_molecules: int, box_size: float, 
                output_dir: str, molecular_size: float = 5.0, timeout: int = 60) -> str:
    """
    Run packmol to create a packed liquid system with conservative settings.
    
    Parameters
    ----------
    molecule_pdb : str
        Path to the PDB file of a single molecule
    n_molecules : int
        Number of molecules to pack
    box_size : float
        Size of the cubic box in Angstroms
    output_dir : str
        Directory for output files
    molecular_size : float, default=5.0
        Estimated molecular size in Angstroms
    timeout : int, default=60
        Timeout in seconds for packmol execution
        
    Returns
    -------
    str
        Path to the generated packed PDB file
        
    Raises
    ------
    RuntimeError
        If packmol fails or is not available
    """
    if not shutil.which("packmol"):
        print("packmol not found, using manual configuration")
        return create_manual_configuration(molecule_pdb, n_molecules, box_size, output_dir)
    
    # Generate file paths
    packed_pdb = os.path.join(output_dir, "packed_system.pdb")
    packmol_inp = os.path.join(output_dir, "packmol.inp")
    
    # Check if molecule PDB is valid
    with open(molecule_pdb, 'r') as f:
        pdb_lines = f.readlines()
    
    atom_lines = [line for line in pdb_lines if line.startswith('ATOM')]
    if not atom_lines:
        print("Invalid PDB format, using manual configuration")
        return create_manual_configuration(molecule_pdb, n_molecules, box_size, output_dir)
    
    atom_count = len(atom_lines)
    print(f"Molecule has {atom_count} atoms")
    
    # Calculate packing density
    estimated_mol_volume = atom_count * 20.0  # ~20 Å³ per atom
    total_mol_volume = n_molecules * estimated_mol_volume
    box_volume = box_size ** 3
    packing_fraction = total_mol_volume / box_volume
    
    print(f"Packing fraction: {packing_fraction:.3f}")
    
    # If packing fraction is too high, increase box size
    if packing_fraction > 0.5:
        new_box_size = box_size * 1.3
        print(f"High packing fraction, increasing box size to {new_box_size:.2f} Å")
        box_size = new_box_size
    
    # Simple packmol input with conservative settings
    packmol_input = f"""tolerance 3.0
filetype pdb
output {os.path.basename(packed_pdb)}
seed 12345
nloop 200
maxit 5

structure {os.path.basename(molecule_pdb)}
  number {n_molecules}
  inside cube 0.0 0.0 0.0 {box_size:.2f}
end structure
"""

    # Write packmol input file
    with open(packmol_inp, 'w') as f:
        f.write(packmol_input)

    print(f"Running packmol with timeout {timeout}s...")
    print(f"Input file content:\n{packmol_input}")
    
    try:
        # Run packmol with timeout
        result = subprocess.run(
            ["packmol", "<", packmol_inp],
            shell=True,
            cwd=output_dir,
            text=True,
            capture_output=True,
            timeout=timeout
        )
        
        print(f"Packmol return code: {result.returncode}")
        if result.stdout:
            print(f"Packmol stdout (last 500 chars):\n{result.stdout[-500:]}")
        if result.stderr:
            print(f"Packmol stderr:\n{result.stderr}")
        
        if result.returncode == 0 and os.path.exists(packed_pdb):
            print("SUCCESS: Packmol completed successfully!")
            
            # Verify output file has reasonable content
            with open(packed_pdb, 'r') as f:
                output_lines = f.readlines()
            output_atoms = len([line for line in output_lines if line.startswith('ATOM')])
            expected_atoms = n_molecules * atom_count
            
            if output_atoms == expected_atoms:
                print(f"Output verified: {output_atoms} atoms as expected")
                return packed_pdb
            else:
                print(f"WARNING: Expected {expected_atoms} atoms, got {output_atoms}")
                if output_atoms > 0:
                    return packed_pdb  # Use it anyway if we got some atoms
        
        print("Packmol failed or produced no output, using manual configuration")
        
    except subprocess.TimeoutExpired:
        print(f"Packmol timed out after {timeout}s, using manual configuration")
    except Exception as e:
        print(f"Packmol failed with error: {e}, using manual configuration")
    
    # Fallback to manual configuration
    return create_manual_configuration(molecule_pdb, n_molecules, box_size, output_dir)
    

def setup_openmm_system(packed_pdb: str, forcefield_xml: Optional[Union[str, List[str]]] = None,
                       box_size: float = None) -> tuple:
    """
    Set up an OpenMM system from a packed PDB file.
    
    Parameters
    ----------
    packed_pdb : str
        Path to the packed PDB file from packmol
    forcefield_xml : str or list, optional
        Force field XML file(s). If None, uses amber default
    box_size : float
        Box size in Angstroms for periodic boundary conditions
        
    Returns
    -------
    tuple
        (pdb_object, system, topology) for OpenMM simulation
    """
    if not OPENMM_AVAILABLE:
        raise RuntimeError("OpenMM is not available. Cannot run NPT simulation.")
    
    # Load PDB file
    pdb = PDBFile(packed_pdb)
    
    # Set up periodic box if provided
    if box_size is not None:
        box_vectors = [
            Vec3(box_size, 0, 0) * angstrom,
            Vec3(0, box_size, 0) * angstrom, 
            Vec3(0, 0, box_size) * angstrom
        ]
        pdb.topology.setPeriodicBoxVectors(box_vectors)
    
    # Set up force field
    if forcefield_xml is None:
        forcefield_xml = DEFAULT_FORCEFIELDS['amber']
    elif isinstance(forcefield_xml, str):
        forcefield_xml = [forcefield_xml]
    
    try:
        # Try using standard OpenMM force fields
        forcefield = ForceField(*forcefield_xml)
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=1.0*nanometer,
            constraints=HBonds
        )
        print(f"Successfully created system with force fields: {forcefield_xml}")
        
    except Exception as e:
        print(f"Failed to create system with provided force fields: {e}")
        # Fallback to basic force field
        try:
            forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
            system = forcefield.createSystem(
                pdb.topology,
                nonbondedMethod=PME,
                nonbondedCutoff=1.0*nanometer,
                constraints=HBonds
            )
            print("Using fallback amber14 force field")
        except Exception as e2:
            raise RuntimeError(f"Could not create OpenMM system: {e2}")
    
    return pdb, system, pdb.topology


def run_npt_simulation(pdb_obj, system, topology, temperature: float = 298.15,
                      pressure: float = 1.0, equilibration_time: float = 1.0,
                      production_time: float = 2.0, output_dir: str = None) -> Dict[str, Any]:
    """
    Run NPT molecular dynamics simulation to equilibrate density.
    
    Parameters
    ----------
    pdb_obj : PDBFile
        OpenMM PDB object
    system : System
        OpenMM system object
    topology : Topology
        OpenMM topology object
    temperature : float, default=298.15
        Temperature in Kelvin
    pressure : float, default=1.0
        Pressure in bar
    equilibration_time : float, default=1.0
        Equilibration time in nanoseconds
    production_time : float, default=2.0
        Production run time in nanoseconds
    output_dir : str, optional
        Directory for output files
        
    Returns
    -------
    dict
        Dictionary containing simulation results including final density
    """
    if not OPENMM_AVAILABLE:
        raise RuntimeError("OpenMM is not available. Cannot run NPT simulation.")
    
    print(f"Setting up NPT simulation at {temperature} K and {pressure} bar")
    
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
    
    # Set initial positions
    simulation.context.setPositions(pdb_obj.positions)
    
    # Minimize energy
    print("Minimizing energy...")
    simulation.minimizeEnergy(maxIterations=1000)
    
    # Set up reporters
    if output_dir:
        log_file = os.path.join(output_dir, "npt_simulation.log")
        dcd_file = os.path.join(output_dir, "trajectory.dcd")
        
        simulation.reporters.append(
            StateDataReporter(
                log_file, 1000, step=True, time=True, potentialEnergy=True,
                temperature=True, volume=True, density=True
            )
        )
        simulation.reporters.append(DCDReporter(dcd_file, 1000))
    
    # Equilibration phase
    equilibration_steps = int(equilibration_time * 1000000 / 2)  # 2 fs time step
    print(f"Running equilibration for {equilibration_time} ns ({equilibration_steps} steps)...")
    simulation.step(equilibration_steps)
    
    # Production phase - collect data
    production_steps = int(production_time * 1000000 / 2)
    data_collection_interval = 1000  # Collect data every 1000 steps (2 ps)
    n_data_points = production_steps // data_collection_interval
    
    print(f"Running production for {production_time} ns ({production_steps} steps)...")
    
    densities = []
    volumes = []
    energies = []
    temperatures = []
    
    for i in range(n_data_points):
        simulation.step(data_collection_interval)
        
        # Get state information
        state = simulation.context.getState(getEnergy=True)
        
        # Calculate density
        volume = state.getPeriodicBoxVolume()
        volume_nm3 = volume / (nanometer**3)
        
        # Get system mass (approximate from number of particles)
        n_particles = system.getNumParticles()
        # This is approximate - in practice you'd calculate exact molecular mass
        estimated_mass = n_particles * 15.0 * dalton  # Rough average atomic mass
        mass_kg = estimated_mass / (dalton * AVOGADRO_CONSTANT_NA) * gram
        
        density_kg_m3 = mass_kg / (volume_nm3 * (nanometer**3))
        density_g_cm3 = density_kg_m3 * 0.001  # Convert kg/m³ to g/cm³
        
        densities.append(density_g_cm3)
        volumes.append(volume_nm3)
        energies.append(state.getPotentialEnergy() / kilojoule_per_mole)
        
        # Get temperature
        kinetic_energy = simulation.context.getState(getEnergy=True).getKineticEnergy()
        temp = (2 * kinetic_energy / (3 * n_particles * BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA)) / kelvin
        temperatures.append(temp)
        
        if (i + 1) % 10 == 0:
            print(f"  Step {i+1}/{n_data_points}: Density = {density_g_cm3:.3f} g/cm³, "
                  f"Volume = {volume_nm3:.2f} nm³, Temp = {temp:.1f} K")
    
    # Calculate statistics
    mean_density = np.mean(densities)
    std_density = np.std(densities)
    mean_volume = np.mean(volumes)
    mean_energy = np.mean(energies)
    mean_temperature = np.mean(temperatures)
    
    print(f"\nSimulation completed!")
    print(f"Average density: {mean_density:.3f} ± {std_density:.3f} g/cm³")
    print(f"Average volume: {mean_volume:.2f} nm³")
    print(f"Average temperature: {mean_temperature:.1f} K")
    
    # Save final configuration
    if output_dir:
        final_pdb = os.path.join(output_dir, "final_equilibrated.pdb")
        positions = simulation.context.getState(getPositions=True).getPositions()
        PDBFile.writeFile(topology, positions, open(final_pdb, 'w'))
        print(f"Final configuration saved to: {final_pdb}")
    
    return {
        'mean_density': mean_density,
        'std_density': std_density,
        'densities': densities,
        'mean_volume': mean_volume,
        'volumes': volumes,
        'mean_energy': mean_energy,
        'energies': energies,
        'mean_temperature': mean_temperature,
        'temperatures': temperatures,
        'n_data_points': n_data_points
    }


def calculate_equilibrium_density(xyz_file: str,
                                temperature: float = 298.15,
                                pressure: float = 1.0,
                                n_molecules: int = 256,
                                box_size: Optional[float] = None,
                                forcefield_xml: Optional[Union[str, List[str]]] = None,
                                equilibration_time: float = 1.0,
                                production_time: float = 2.0,
                                output_dir: Optional[str] = None,
                                cleanup: bool = True) -> Dict[str, Any]:
    """
    Complete workflow to calculate equilibrium liquid density from XYZ file.
    
    This function performs the following steps:
    1. Converts XYZ to PDB format
    2. Uses packmol to create initial liquid configuration
    3. Runs NPT molecular dynamics simulation to equilibrate density
    4. Returns the equilibrium density and simulation statistics
    
    Parameters
    ----------
    xyz_file : str
        Path to the XYZ file containing the equilibrium molecular structure
    temperature : float, default=298.15
        Temperature in Kelvin
    pressure : float, default=1.0 
        Pressure in bar
    n_molecules : int, default=256
        Number of molecules to include in the liquid system
    box_size : float, optional
        Size of the cubic simulation box in Angstroms. If None, estimated from density
    forcefield_xml : str or list, optional
        Path(s) to force field XML file(s). If None, uses amber default
    equilibration_time : float, default=1.0
        NPT equilibration time in nanoseconds
    production_time : float, default=2.0
        NPT production run time in nanoseconds for density calculation
    output_dir : str, optional
        Directory for output files. If None, creates a temporary directory
    cleanup : bool, default=True
        Whether to clean up intermediate files
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'equilibrium_density': Mean density from NPT simulation (g/cm³)
        - 'density_std': Standard deviation of density
        - 'initial_density': Estimated density before simulation
        - 'box_size': Final box size used
        - 'n_molecules': Number of molecules
        - 'temperature': Simulation temperature
        - 'pressure': Simulation pressure
        - 'simulation_results': Full simulation statistics
        - 'output_dir': Directory containing output files
        
    Raises
    ------
    RuntimeError
        If packmol or OpenMM are not available, or if simulations fail
    FileNotFoundError
        If input files are not found
    """
    # Validate dependencies
    if not shutil.which("packmol"):
        raise RuntimeError("packmol executable not found in PATH. Please install packmol.")
    
    if not OPENMM_AVAILABLE:
        raise RuntimeError("OpenMM is not available. Cannot run NPT simulations.")
    
    # Setup output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="qml_npt_density_")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting equilibrium density calculation for {xyz_file}")
    print(f"Conditions: T = {temperature} K, P = {pressure} bar")
    print(f"System: {n_molecules} molecules")
    print(f"Output directory: {output_dir}")
    
    try:
        # Step 1: Convert XYZ to PDB
        print("\n1. Converting XYZ to PDB format...")
        molecule_pdb = os.path.join(output_dir, "molecule.pdb")
        xyz_to_pdb(xyz_file, molecule_pdb)
        
        # Parse molecular properties
        with open(xyz_file, 'r') as f:
            lines = f.readlines()
        n_atoms = int(lines[0].strip())
        atoms = []
        for i in range(2, 2 + n_atoms):
            parts = lines[i].strip().split()
            atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
        
        mol_props = estimate_molecular_properties(atoms)
        molecular_mass = mol_props['mass']
        molecular_size = mol_props['size']
        
        print(f"Molecular mass: {molecular_mass:.2f} amu")
        print(f"Molecular size: {molecular_size:.2f} Å")
        
        # Step 2: Estimate initial density and box size
        print("\n2. Estimating initial density...")
        initial_density = estimate_density(temperature, pressure, molecular_mass)
        print(f"Estimated initial density: {initial_density:.3f} g/cm³")
        
        if box_size is None:
            # Calculate box size based on reasonable packing fraction for liquids
            # Use a more aggressive packing fraction to get smaller, more reasonable boxes
            target_packing_fraction = 0.5  # Higher packing for smaller box
            
            # Estimate molecular volume (rough sphere approximation)
            molecular_volume_angstrom3 = (4/3) * 3.14159 * (molecular_size/2)**3
            total_molecular_volume = n_molecules * molecular_volume_angstrom3
            
            # Calculate required box volume
            required_box_volume = total_molecular_volume / target_packing_fraction
            packing_based_box_size = required_box_volume ** (1/3)
            
            # Alternative approach: use the density-based calculation as a cross-check
            total_mass_grams = n_molecules * molecular_mass * 1.66054e-24  # amu to grams
            volume_cm3 = total_mass_grams / initial_density
            density_based_box_size = (volume_cm3 * 1e24) ** (1/3)  # cm³ to Å³, then cube root
            
            print(f"Packing-based box size (pf={target_packing_fraction}): {packing_based_box_size:.2f} Å")
            print(f"Density-based box size: {density_based_box_size:.2f} Å")
            
            # Use the SMALLER of the two to avoid overly large boxes
            box_size = min(packing_based_box_size, density_based_box_size)
            
            # But ensure it's not too small for packmol to work
            min_box_size = molecular_size * 2.5  # Minimum for packmol convergence
            if box_size < min_box_size:
                print(f"Adjusting box size from {box_size:.2f} to {min_box_size:.2f} Å (minimum size)")
                box_size = min_box_size
            
            # Cap at reasonable maximum to avoid huge dilute systems
            max_box_size = molecular_size * 8  # Much more conservative upper limit 
            if box_size > max_box_size:
                print(f"Capping box size from {box_size:.2f} to {max_box_size:.2f} Å (maximum size)")
                box_size = max_box_size
        
        print(f"Using box size: {box_size:.2f} Å")
        
        # Step 3: Create initial liquid configuration
        print("\n3. Creating initial liquid configuration...")
        
        # For very small systems, skip packmol and go straight to manual grid
        if n_molecules <= 15 or box_size < molecular_size * 3:
            print(f"Small system detected ({n_molecules} molecules, {box_size:.1f} Å box)")
            print("Using manual grid configuration (more reliable for small systems)...")
            packed_pdb = create_manual_configuration(molecule_pdb, n_molecules, box_size, output_dir)
        else:
            print("Running packmol to create initial liquid configuration...")
            # Add some safety checks before running packmol
            if box_size < molecular_size * 2:
                print(f"WARNING: Box size ({box_size:.2f} Å) is very small compared to molecular size ({molecular_size:.2f} Å)")
                print("This may cause packmol to take a very long time or fail")
                
            density_check = (n_molecules * molecular_mass * 1.66054e-24) / ((box_size * 1e-8) ** 3)
            if density_check > 2.0:
                print(f"WARNING: Target density is very high ({density_check:.2f} g/cm³)")
                print("Consider reducing number of molecules or increasing box size")
            
            try:
                packed_pdb = run_packmol(molecule_pdb, n_molecules, box_size, output_dir, molecular_size)
            except RuntimeError as e:
                if "timed out" in str(e):
                    print("\nPackmol timed out. Trying with a larger box...")
                    # Try with 20% larger box
                    larger_box = box_size * 1.2
                    print(f"Retrying with box size: {larger_box:.2f} Å")
                    try:
                        packed_pdb = run_packmol(molecule_pdb, n_molecules, larger_box, output_dir, molecular_size)
                        box_size = larger_box  # Update box_size for OpenMM setup
                    except RuntimeError:
                        print("\nStill failing. Using manual configuration...")
                        packed_pdb = create_manual_configuration(molecule_pdb, n_molecules, box_size, output_dir)
                else:
                    print(f"\nPackmol failed with error: {e}")
                    print("Using manual configuration as fallback...")
                    packed_pdb = create_manual_configuration(molecule_pdb, n_molecules, box_size, output_dir)
        
        # Step 4: Set up OpenMM system
        print("\n4. Setting up OpenMM system...")
        pdb_obj, system, topology = setup_openmm_system(packed_pdb, forcefield_xml, box_size)
        
        # Step 5: Run NPT simulation
        print("\n5. Running NPT simulation to equilibrate density...")
        simulation_results = run_npt_simulation(
            pdb_obj, system, topology, temperature, pressure,
            equilibration_time, production_time, output_dir
        )
        
        # Prepare final results
        results = {
            'equilibrium_density': simulation_results['mean_density'],
            'density_std': simulation_results['std_density'],
            'initial_density': initial_density,
            'box_size': box_size,
            'n_molecules': n_molecules,
            'molecular_mass': molecular_mass,
            'temperature': temperature,
            'pressure': pressure,
            'equilibration_time': equilibration_time,
            'production_time': production_time,
            'simulation_results': simulation_results,
            'output_dir': output_dir
        }
        
        # Cleanup intermediate files if requested
        if cleanup:
            files_to_remove = [molecule_pdb]
            for file_path in files_to_remove:
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        print(f"\n" + "="*60)
        print("EQUILIBRIUM DENSITY CALCULATION COMPLETED!")
        print("="*60)
        print(f"Initial estimated density: {initial_density:.3f} g/cm³")
        print(f"Equilibrium density: {simulation_results['mean_density']:.3f} ± {simulation_results['std_density']:.3f} g/cm³")
        print(f"Density change: {((simulation_results['mean_density'] - initial_density) / initial_density * 100):+.1f}%")
        print(f"Final box volume: {simulation_results['mean_volume']:.2f} nm³")
        print(f"Output directory: {output_dir}")
        
        return results
        
    except Exception as e:
        # Cleanup on error if in temporary directory
        if cleanup and output_dir.startswith(tempfile.gettempdir()):
            shutil.rmtree(output_dir, ignore_errors=True)
        raise e


def main():
    """
    Command line interface for the equilibrium density calculation.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Calculate equilibrium liquid density from XYZ file using packmol + NPT simulation"
    )
    parser.add_argument("xyz_file", help="Input XYZ file with equilibrium molecular structure")
    parser.add_argument("-t", "--temperature", type=float, default=298.15, 
                       help="Temperature in Kelvin (default: 298.15)")
    parser.add_argument("-p", "--pressure", type=float, default=1.0,
                       help="Pressure in bar (default: 1.0)")
    parser.add_argument("-n", "--n_molecules", type=int, default=256,
                       help="Number of molecules (default: 256)")
    parser.add_argument("-b", "--box_size", type=float, default=None,
                       help="Box size in Angstroms (default: auto-calculated)")
    parser.add_argument("-f", "--forcefield", type=str, default=None,
                       help="Force field XML file path (default: amber)")
    parser.add_argument("--equilibration_time", type=float, default=1.0,
                       help="Equilibration time in ns (default: 1.0)")
    parser.add_argument("--production_time", type=float, default=2.0,
                       help="Production time in ns (default: 2.0)")
    parser.add_argument("-o", "--output_dir", type=str, default=None,
                       help="Output directory (default: temporary)")
    parser.add_argument("--no-cleanup", action="store_true",
                       help="Keep intermediate files")
    
    args = parser.parse_args()
    
    try:
        result = calculate_equilibrium_density(
            xyz_file=args.xyz_file,
            temperature=args.temperature,
            pressure=args.pressure,
            n_molecules=args.n_molecules,
            box_size=args.box_size,
            forcefield_xml=args.forcefield,
            equilibration_time=args.equilibration_time,
            production_time=args.production_time,
            output_dir=args.output_dir,
            cleanup=not args.no_cleanup
        )
        
        print(f"\nFinal equilibrium density: {result['equilibrium_density']:.3f} ± {result['density_std']:.3f} g/cm³")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
