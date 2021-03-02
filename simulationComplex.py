import sys, time
from openforcefield.topology import Molecule
from openmmforcefields.generators import SystemGenerator
from simtk import unit, openmm
from simtk.openmm import app, Platform, LangevinIntegrator
from simtk.openmm.app import PDBFile, Simulation, DCDReporter, Modeller, PDBReporter
#from dcdreporter import DCDReporter
from parmed.openmm import StateDataReporter, NetCDFReporter
from rdkit import Chem
import pytraj as pt
import parmed

#temperature = 300 * unit.kelvin
#equilibration_steps = 200
#opt.interval = 1000
from argparse import ArgumentParser

parser = ArgumentParser()
group = parser.add_argument_group('Input File Options')
group.add_argument('-p', '--receptor', dest='receptor', metavar='<PROTEIN_FILE>', required=True,
                   help='''Receptor file for pdb format which is precessed using PDBFixer.''')
group.add_argument('-l', '--ligand', dest='ligand', metavar='<LIGAND_FILE>',
                   help='''Ligand file for sdf format which is docked.''')
group.add_argument('-s', '--other', dest='other', metavar='FILE', nargs='+', default=None,
                   help='''Other ligand for sdf format that is not calcuated binding free energy.''')
group = parser.add_argument_group('Output File Options')
group.add_argument('-r', '--restart', dest='restart', default='restart.nc',
                   metavar='FILE', help='''NetCDF file with information to
                   restart the simulation with another run''')
group.add_argument('-o' , '--output', dest='output', default='md.log',
                   metavar='FILE', help='''Output file for energies''')
group.add_argument('-x', '--trajectory', dest='trajectory', default='md.dcd',
                   metavar='FILE', help='''NetCDF trajectory to generate.
                   Snapshots written every --interval steps.''')
group.add_argument('--interval', dest='interval', default=500, metavar='INT',
                   help='Interval between printing state data. Default 500',
                   type=int)
group = parser.add_argument_group('Positional Restraint Options')
group.add_argument('--restrain', dest='restraints', metavar='MASK',
                   help='restraint mask (default None)', default=None)
group.add_argument('--restrainbond', dest='bond_restraints', metavar='FILE',
                   help='bond force(default None)', default=None)
group.add_argument('-k', '--force-constant', dest='force_constant', type=float,
                   metavar='FLOAT', help='''Force constant for cartesian
                   constraints. Default 10 kcal/mol/A^2''', default=10)
group = parser.add_argument_group('Simulation Options')
group.add_argument('-n', '--num-steps', dest='num_steps', required=True, type=int,
                   help='Number of MD steps to run. Required', metavar='INT')
group.add_argument('--dt', dest='timestep', type=float,
                   metavar='FLOAT', help='''time step for integrator (outer
                   time-step for RESPA integrator) Default 2 fs''', default=2.0)
group.add_argument('--temp', dest='temp', type=float,
                   metavar='FLOAT', help='''target temperature for NVT
                   simulation. Default %(default)s K''', default=300.0)

opt = parser.parse_args()


#if len(sys.argv) != 5:
#    print('Usage: python simulateComplexWithSolvent2.py input.pdb input.mol output num_steps')
#    print('Prepares complex of input.pdb and input.mol and generates complex named output_complex.pdb,')
#    print(' minimised complex named output_minimised.pdb and MD trajectory named output_traj.pdb and/or output_traj.dcd')
#    exit(1)


#pdb_in = sys.argv[1]
#mol_in = sys.argv[2]
output_complex = 'Cmp_complex.pdb'
#output_traj_pdb = sys.argv[3] + '_traj.pdb'
#output_traj_dcd = sys.argv[3] + '_traj.nc'
output_min = 'Cmp_minimised.pdb'
#num_steps = int(sys.argv[4])
#print('Processing', pdb_in, 'and', mol_in, 'with', num_steps, 'steps generating outputs',
#      output_complex, output_min, output_traj_pdb, output_traj_dcd)

# check whether we have a GPU platform and if so set the precision to mixed
speed = 0
for i in range(Platform.getNumPlatforms()):
    p = Platform.getPlatform(i)
    # print(p.getName(), p.getSpeed())
    if p.getSpeed() > speed:
        platform = p
        speed = p.getSpeed()

if platform.getName() == 'CUDA' or platform.getName() == 'OpenCL':
    platform.setPropertyDefaultValue('Precision', 'mixed')
    print('Set precision for platform', platform.getName(), 'to mixed')


# Read the molfile into RDKit, add Hs and create an openforcefield Molecule object
other_mols = []
if opt.other is not None:
    for step, other in enumerate(opt.other):
        o_mol = Chem.MolFromMolFile(other)
        o_molh = Chem.AddHs(o_mol, addCoords=True)
        o_molh.SetProp('_Name', 'MO'+str(step+1))
        Chem.AssignAtomChiralTagsFromStructure(o_molh)
        other_mol = Molecule(o_molh)
        other_mols.append(other_mol)

if opt.ligand:
    print('Reading ligand')
    rdkitmol = Chem.MolFromMolFile(opt.ligand)
    print('Adding hydrogens')
    rdkitmolh = Chem.AddHs(rdkitmol, addCoords=True)
    rdkitmolh.SetProp('_Name', 'LIG')
    # ensure the chiral centers are all defined
    Chem.AssignAtomChiralTagsFromStructure(rdkitmolh)
    ligand_mol = Molecule(rdkitmolh)
    other_mols.append(ligand_mol)

print('Preparing system')
# Initialize a SystemGenerator using the GAFF for the ligand and tip3p for the water.
# forcefield_kwargs = {'constraints': app.HBonds, 'rigidWater': True, 'removeCMMotion': False, 'hydrogenMass': 4*unit.amu }
forcefield_kwargs = {'constraints': None, 'rigidWater': None, 'removeCMMotion': False, 'hydrogenMass': 4*unit.amu }
system_generator = SystemGenerator(
    forcefields=['amber/ff14SB.xml', 'amber/tip3p_standard.xml'],
    small_molecule_forcefield='openff_unconstrained-1.2.1.offxml',
    # small_molecule_forcefield='gaff-2.11',
    molecules=other_mols,
    forcefield_kwargs=forcefield_kwargs)

# Use Modeller to combine the protein and ligand into a complex
print('Reading protein')
protein_pdb = PDBFile(opt.receptor)

print('Preparing complex')
modeller = Modeller(protein_pdb.topology, protein_pdb.positions)
print('System has %d atoms' % modeller.topology.getNumAtoms())

# This next bit is black magic.
# Modeller needs topology and positions. Lots of trial and error found that this is what works to get these from
# an openforcefield Molecule object that was created from a RDKit molecule.
# The topology part is described in the openforcefield API but the positions part grabs the first (and only)
# conformer and passes it to Modeller. It works. Don't ask why!
if len(other_mols) != 0:
    for other_mol in other_mols:
        modeller.add(other_mol.to_topology().to_openmm(), other_mol.conformers[0])
    #modeller.add(ligand_mol.to_topology().to_openmm(), ligand_mol.conformers[0])

# Generate ligand with solvent for FEP
if opt.ligand:
    modeller_org = Modeller(ligand_mol.to_topology().to_openmm(), ligand_mol.conformers[0])
    modeller_org.addSolvent(system_generator.forcefield, model='tip3p', ionicStrength=0.1*unit.molar, padding=10.0*unit.angstroms)
    system_org = system_generator.create_system(modeller_org.topology, molecules=ligand_mol)
    system_org.addForce(openmm.MonteCarloBarostat(1*unit.atmospheres, opt.temp * unit.kelvin, 25))

print('System has %d atoms' % modeller.topology.getNumAtoms())

# Solvate
print('Adding solvent...')
# we use the 'padding' option to define the periodic box. The PDB file does not contain any
# unit cell information so we just create a box that has a 10A padding around the complex.
#modeller.addSolvent(system_generator.forcefield, model='tip3p', ionicStrength=0.1*unit.molar, padding=10.0*unit.angstroms)
modeller.addSolvent(system_generator.forcefield, model='tip3p', ionicStrength=0.1*unit.molar, padding=10.0*unit.angstroms)
print('System has %d atoms' % modeller.topology.getNumAtoms())

with open(output_complex, 'w') as outfile:
    PDBFile.writeFile(modeller.topology, modeller.positions, outfile)

# Create the system using the SystemGenerator
system = system_generator.create_system(modeller.topology, molecules=other_mols)

# Save Protein-Ligand Complex
structure = parmed.openmm.load_topology(modeller.topology, system, xyz=modeller.positions)
structure.save('Cmp.pdb', overwrite=True)
#structure.save('Cmp.gro', format='gro', overwrite=True)
structure.save('Cmp.prmtop', overwrite=True)
structure.save('Cmp.inpcrd', overwrite=True)

# Add cartesian restraints if desired
if opt.restraints:
    print('Adding restraints (k=%s kcal/mol/A^2) from %s' %
            (opt.force_constant, opt.restraints)); sys.stdout.flush()
    sel = parmed.amber.AmberMask(modeller, opt.restraints).Selection()
#    print('restrain atom is [%s]' %sel)
    const = opt.force_constant * unit.kilocalories_per_mole/unit.angstroms**2
    const = const.value_in_unit_system(unit.md_unit_system)
    force = openmm.CustomExternalForce('k*periodicdistance(x, y, z, x0, y0, z0)^2')
    force.addGlobalParameter('k', const)
    force.addPerParticleParameter('x0')
    force.addPerParticleParameter('y0')
    force.addPerParticleParameter('z0')
    for i, atom_crd in enumerate(modeller.positions):
        if sel[i]:
            force.addParticle(i, atom_crd.value_in_unit(unit.nanometers))
    system.addForce(force)

# Add bond restraints if desired
if opt.bond_restraints:
    flat_bottom_force = openmm.CustomBondForce(
	        'step(r-r0) * (k/2) * (r-r0)^2')
    flat_bottom_force.addPerBondParameter('r0')
    flat_bottom_force.addPerBondParameter('k')

    # restraints.txt consists of one restraint per line
    # with the format:
    # atom_index_i 	atom_index_j	r0	k
    with open(opt.bond_restraints) as input_file:
        for line in input_file:
            columns = line.split()
            atom_index_i = int(columns[0])
            atom_index_j = int(columns[1])
            r0 = float(columns[2])
            k = float(columns[3])
            flat_bottom_force.addBond(
                atom_index_i, atom_index_j, [r0, k])
            print('Adding %sA distance restraints (k=%s kcal/mol/A^2) between %s and %s' %
                (r0, k, atom_index_i, atom_index_j)); sys.stdout.flush()
    system.addForce(flat_bottom_force)


integrator = LangevinIntegrator(opt.temp * unit.kelvin, 1 / unit.picosecond, opt.timestep * unit.femtoseconds)
#system.addForce(openmm.MonteCarloBarostat(1*unit.atmospheres, opt.temp * unit.kelvin, 25))
print('Default Periodic box:', system.getDefaultPeriodicBoxVectors())

simulation = Simulation(modeller.topology, system, integrator, platform=platform)
context = simulation.context
context.setPositions(modeller.positions)

# # Save to Gromacs format File
# # Save Protein-Ligand Complex
# structure = parmed.openmm.load_topology(modeller.topology, system, xyz=modeller.positions)
# structure.save('Cmp.pdb', overwrite=True)
# #structure.save('Cmp.gro', format='gro', overwrite=True)
# structure.save('Cmp.prmtop', overwrite=True)
# structure.save('Cmp.inpcrd', overwrite=True)

from simtk.openmm import XmlSerializer
serialized_system_Cmp = XmlSerializer.serialize(system)
outfile = open('Cmp.xml','w')
outfile.write(serialized_system_Cmp)
outfile.close()

# Save Ligand
if opt.ligand:
    structure_org = parmed.openmm.load_topology(modeller_org.topology, system_org, xyz=modeller_org.positions)
    structure_org.save('Lig.pdb', overwrite=True)
    #structure.save('Lig.gro', format='gro', overwrite=True)

    serialized_system_Org = XmlSerializer.serialize(system_org)
    outfile = open('Lig.xml','w')
    outfile.write(serialized_system_Org)
    outfile.close()

print('Minimising ...')
simulation.minimizeEnergy()

# Write out the minimised PDB. The 'enforcePeriodicBox=False' bit is important otherwise the different
# components can end up in different periodic boxes resulting in really strange looking output.
with open(output_min, 'w') as outfile:
    PDBFile.writeFile(modeller.topology, context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(), file=outfile, keepIds=True)

# equilibrate
#simulation.context.setVelocitiesToTemperature(opt.temp)
#print('Equilibrating ...')
#simulation.step(10000)

# Run the simulation.
# The enforcePeriodicBox arg to the reporters is important.
# It's a bit counter-intuitive that the value needs to be False, but this is needed to ensure that
# all parts of the simulation end up in the same periodic box when being output.

simulation.context.setVelocitiesToTemperature(opt.temp)

simulation.reporters.append(
        parmed.openmm.StateDataReporter(opt.output, reportInterval=opt.interval,
                        volume=True,density=True,separator='\t')
)
simulation.reporters.append(
        parmed.openmm.ProgressReporter(str(opt.output) + '.info', opt.interval,  opt.num_steps)
)
simulation.reporters.append(
        #parmed.openmm.NetCDFReporter(opt.trajectory, opt.interval*10)
        DCDReporter(opt.trajectory, opt.interval*10, enforcePeriodicBox=False)
)
#simulation.reporters.append(
#        parmed.openmm.RestartReporter(opt.restart, opt.interval*100, netcdf=True)
#)
print('Starting simulation with', opt.num_steps, 'steps ...')

#simulation.context.setPositions(modeller.positions)
#simulation.context.setVelocitiesToTemperature(opt.temp)

t0 = time.time()
simulation.step(opt.num_steps)
t1 = time.time()
print('Simulation complete in', t1 - t0, 'seconds at', opt.temp * unit.kelvin)

import mdtraj as md
mdtraj_topology = md.Topology.from_openmm(modeller.topology)
trajectory = md.Trajectory(simulation.context.getState(getPositions=True).getPositions(asNumpy=True)/unit.nanometers, mdtraj_topology)
trajectory[-1:].save('restart.nc')
trajectory[-1:].save('restart.pdb')
#atoms_to_keep = [r.index for r in trajectory.topology.residues if r.name == 'LIG']
if opt.ligand:
    atoms_to_keep = trajectory.topology.select('resname LIG')
    trajectory.restrict_atoms(atoms_to_keep)  # this acts inplace on the trajectory
    trajectory.save('Lig_dry.pdb')
