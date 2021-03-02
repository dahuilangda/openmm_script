#!/usr/bin/env python

from __future__ import division, print_function
import sys

# OpenMM Imports
import simtk.openmm as mm
import simtk.openmm.app as app

# ParmEd Imports
from parmed import load_file, unit as u
from parmed.openmm import StateDataReporter, NetCDFReporter

from argparse import ArgumentParser
import sys

import parmed as pmd

parser = ArgumentParser()
group = parser.add_argument_group('Input File Options')
group.add_argument('-p', '--prmtop', dest='prmtop', metavar='<PRMTOP_FILE>', required=True,
                   help='''Prmtop file for amber.''')
group.add_argument('-i', '--inpcrd', dest='inpcrd', metavar='<INPCRD_FILE>', required=True,
                   help='''Inpcrd file for amber.''')
group.add_argument('-s', '--state', dest='state', metavar='FILE', default=None,
                   help='''Restart file (any format)''')
group = parser.add_argument_group('Positional Restraint Options')
group.add_argument('--restrain', dest='restraints', metavar='MASK',
                   help='restraint mask (default None)', default=None)
group.add_argument('--restrainbond', dest='bond_restraints', metavar='FILE',
                   help='bond force(default None)', default=None)
group.add_argument('-k', '--force-constant', dest='force_constant', type=float,
                   metavar='FLOAT', help='''Force constant for cartesian
                   constraints. Default 10 kcal/mol/A^2''', default=10)
group = parser.add_argument_group('Output File Options')
group.add_argument('-r', '--restart', dest='restart', default='restart.nc',
                   metavar='FILE', help='''NetCDF file with information to
                   restart the simulation with another run''')
group.add_argument('-o' , '--output', dest='output', default=sys.stdout,
                   metavar='FILE', help='''Output file for energies''')
group.add_argument('-x', '--trajectory', dest='trajectory', default='md.nc',
                   metavar='FILE', help='''NetCDF trajectory to generate.
                   Snapshots written every --interval steps.''')
group.add_argument('--checkpoint', dest='checkpoint', metavar='FILE',
                   default=None, help='''Name of a checkpoint file to write
                   periodically throughout the simulation. Primarily useful for
                   debugging intermittent and rare errors.''')
group.add_argument('--interval', dest='interval', default=500, metavar='INT',
                   help='Interval between printing state data. Default 500',
                   type=int)
group = parser.add_argument_group('Simulation Options')
group.add_argument('-m', '--num-min', dest='num_min_steps', type=int,
                   help='Number of min steps to run. Required', metavar='INT')
group.add_argument('-n', '--num-steps', dest='num_steps', required=True, type=int,
                   help='Number of MD steps to run. Required', metavar='INT')
group.add_argument('--dt', dest='timestep', type=float,
                   metavar='FLOAT', help='''time step for integrator (outer
                   time-step for RESPA integrator) Default 1 fs''', default=1.0)
group.add_argument('--temp', dest='temp', type=float,
                   metavar='FLOAT', help='''target temperature for NVT
                   simulation. Default %(default)s K''', default=300.0)

opt = parser.parse_args()

# Load the Amber files
print('Loading AMBER files...'); sys.stdout.flush()
Cmp_solv = load_file(opt.prmtop, opt.inpcrd)

# Create the OpenMM system
print('Creating OpenMM System'); sys.stdout.flush()
system = Cmp_solv.createSystem(nonbondedMethod=app.PME,
                                nonbondedCutoff=8.0*u.angstroms,
                                constraints=app.HBonds,
)

# Add cartesian restraints if desired
if opt.restraints:
    print('Adding restraints (k=%s kcal/mol/A^2) from %s' %
            (opt.force_constant, opt.restraints)); sys.stdout.flush()
    sel = pmd.amber.AmberMask(Cmp_solv, opt.restraints).Selection()
#    print('restrain atom is [%s]' %sel)
    const = opt.force_constant * u.kilocalories_per_mole/u.angstroms**2
    const = const.value_in_unit_system(u.md_unit_system)
    force = mm.CustomExternalForce('k*periodicdistance(x, y, z, x0, y0, z0)^2')
    force.addGlobalParameter('k', const)
    force.addPerParticleParameter('x0')
    force.addPerParticleParameter('y0')
    force.addPerParticleParameter('z0')
    for i, atom_crd in enumerate(Cmp_solv.positions):
        if sel[i]:
            force.addParticle(i, atom_crd.value_in_unit(u.nanometers))
    system.addForce(force)

# Add bond restraints if desired
if opt.bond_restraints:
    flat_bottom_force = mm.CustomBondForce(
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

# Create the integrator to do Langevin dynamics
integrator = mm.LangevinIntegrator(
                        opt.temp*u.kelvin,       # Temperature of heat bath
                        1.0/u.picoseconds,  # Friction coefficient
                        opt.timestep*u.femtoseconds, # Time step
)

# Define the platform to use; CUDA, OpenCL, CPU, or Reference. Or do not specify
# the platform to use the default (fastest) platform
platform = mm.Platform.getPlatformByName('CUDA')
prop = dict(CudaPrecision='mixed') # Use mixed single/double precision

# Create the Simulation object
sim = app.Simulation(Cmp_solv.topology, system, integrator, platform, prop)

# Set the particle positions
sim.context.setPositions(Cmp_solv.positions)

# Minimize the energy
#print('Minimizing energy'); sys.stdout.flush()
#sim.minimizeEnergy(maxIterations=500)

sim.reporters.append(
        pmd.openmm.StateDataReporter(opt.output, reportInterval=opt.interval,
                        volume=True,density=True,separator='\t')
)
sim.reporters.append(
        pmd.openmm.ProgressReporter(str(opt.output) + '.info', opt.interval, opt.num_steps)
)
sim.reporters.append(
        pmd.openmm.NetCDFReporter(opt.trajectory, opt.interval*10)

# # Set up the reporters to report energies and coordinates
# sim.reporters.append(
#         StateDataReporter(sys.stdout, 100, step=True, potentialEnergy=True,
#                           kineticEnergy=True, temperature=True, volume=True,
#                           density=True)
)
# sim.reporters.append(NetCDFReporter('aa_solv.nc', opt.interval*10, crds=True))
sim.reporters.append(
        pmd.openmm.RestartReporter(opt.restart, opt.interval*100, netcdf=True)
)

# Run dynamics
# print('Running dynamics')
# sim.step(100000)

if opt.state is not None:
    print('Setting coordinates and velocities from restart file %s' %
        opt.state); sys.stdout.flush()

    if opt.state[-3:] == 'xml':
        with open(opt.state, 'r') as f:
            sim.context.setState(mm.XmlSerializer.deserialize(f.read()))
    elif opt.state[-3:] == 'chk':
        sim.loadCheckpoint(opt.state)
    else:
#       jason's code that is supposed to work for any restart file type:
        rst = pmd.load_file(opt.state)
        sim.context.setPositions(rst.coordinates[-1]*u.angstroms)
        sim.context.setVelocities(rst.velocities[-1]*u.angstroms/u.picoseconds)
        sim.context.setPeriodicBoxVectors(*pmd.geometry.box_lengths_and_angles_to_vectors(*rst.box))
        if hasattr(rst, 'time'):
            try:
                sim.context.setTime(rst.time[-1])
            except TypeError:
                sim.context.setTime(rst.time)

else:
    print('Setting coordinates from PDB file %s' % opt.prmtop); sys.stdout.flush()
    sim.context.setPositions(Cmp_solv.positions)
    sim.context.setVelocitiesToTemperature(opt.temp)

# Minimize the energy
if opt.num_min_steps:
    print('Minimizing energy'); sys.stdout.flush()
    sim.minimizeEnergy(maxIterations=opt.num_min_steps)

print('Running the simulation for %d steps!' % opt.num_steps); sys.stdout.flush()
sim.step(opt.num_steps)

# The last step may not have resulted in a restart file being written. Force it
# here
state = sim.context.getState(getPositions=True, getVelocities=True,
        getEnergy=True, getForces=True,
        enforcePeriodicBox=system.usesPeriodicBoundaryConditions())
for rep in sim.reporters:
    if isinstance(rep, pmd.openmm.RestartReporter):
        rep.report(sim, state)


# ~/Simulation/amber16/bin/pmemd.cuda -O -i Min_1.md -o Min_1.mdout -p Cmp.prmtop -c Cmp.inpcrd -ref Cmp.inpcrd -r Min_1_rst.nc < /dev/null
# python simulateAmber_V2.0.py -p Cmp.prmtop -i Min_1_rst.nc -n 10000000 -r MD_rst_1.nc -o md_1.log -x md_1.nc
# python simulateAmber_V2.0.py -p Cmp_1264.prmtop -i MD_rst_1.nc -n 10000000 -r MD_rst_2.nc -o md_2.log -x md_2.nc
# python simulateAmber_V2.0.py -p Cmp_1264.prmtop -i MD_rst_2.nc -n 10000000 -r MD_rst_3.nc -o md_3.log -x md_3.nc
# python simulateAmber_V2.0.py -p Cmp_1264.prmtop -i MD_rst_3.nc -n 10000000 -r MD_rst_4.nc -o md_4.log -x md_4.nc
# python simulateAmber_V2.0.py -p Cmp_1264.prmtop -i MD_rst_4.nc -n 10000000 -r MD_rst_5.nc -o md_5.log -x md_5.nc
# python simulateAmber_V2.0.py -p Cmp_1264.prmtop -i MD_rst_5.nc -n 10000000 -r MD_rst_6.nc -o md_6.log -x md_6.nc
# python simulateAmber_V2.0.py -p Cmp_1264.prmtop -i MD_rst_6.nc -n 10000000 -r MD_rst_7.nc -o md_7.log -x md_7.nc
# python simulateAmber_V2.0.py -p Cmp_1264.prmtop -i MD_rst_7.nc -n 10000000 -r MD_rst_8.nc -o md_8.log -x md_8.nc
# python simulateAmber_V2.0.py -p Cmp_1264.prmtop -i MD_rst_8.nc -n 10000000 -r MD_rst_9.nc -o md_9.log -x md_9.nc
# python simulateAmber_V2.0.py -p Cmp_1264.prmtop -i MD_rst_9.nc -n 10000000 -r MD_rst_10.nc -o md_10.log -x md_10.nc
