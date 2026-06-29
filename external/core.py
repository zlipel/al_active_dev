"""
Trimmed copy of the cluster-side LAMMPS helper library (`core.py`),
vendored into al_active_dev so the project is self-contained.

Vendored from: ~/scripts/utility_scripts/core.py on Princeton Stellar.
Trimmed to the 9 symbols actually consumed by simulation/ and analysis/.
"""
import os
import random
import numpy as np


def calculate_mass(sequence):
    """
    Calculate the mass of an amino acid sequence.

    This function takes an amino acid sequence as a string and calculates the mass
    of the sequence based on the amino acid masses defined in the `amino_acid_masses`
    dictionary.

    Args:
        sequence (str): A string representing an amino acid sequence.

    Returns:
        float: The mass of the amino acid sequence.
    """

    # Define a dictionary for amino acid masses
    amino_acid_masses = {
        'A': 71.0,
        'C': 103.1,
        'D': 115.0,
        'E': 129.1,
        'F': 147.1,
        'G': 57.0,
        'H': 137.1,
        'I': 113.1,
        'K': 128.1,
        'L': 113.1,
        'M': 131.1,
        'N': 114.1,
        'P': 97.1,
        'Q': 128.1,
        'R': 156.1,
        'S': 87.0,
        'T': 101.1,
        'V': 99.1,
        'W': 186.2,
        'Y': 163.1,
        'B': 156.1,  # Amino acids I created. B has same mass as R
        'O': 115.0,  # O has same mass as D
        'U': 129.1,  # U has same mass as E
        'J': 128.1,  # J has same mass as K
    }

    return sum(amino_acid_masses.get(letter, 0) for letter in sequence)


def convert_amino_acids_to_numbers(lines):
    """
    Convert amino acid sequences into numerical sequences as
    defined by the alphabetical order of single letter monomer identifiers.

    This function takes a list of amino acid sequences and converts each sequence
    into a list of numbers based on the alphabetical order of single letter monomer identifiers.

    Args:
        lines (list): A list of strings representing amino acid sequences.

    Returns:
        list: A list of lists, where each list is the list of amino acids represented as numbers.
    """

    amino_acid_dict = {
        'A': 0,
        'C': 1,
        'D': 2,
        'E': 3,
        'F': 4,
        'G': 5,
        'H': 6,
        'I': 7,
        'K': 8,
        'L': 9,
        'M': 10,
        'N': 11,
        'P': 12,
        'Q': 13,
        'R': 14,
        'S': 15,
        'T': 16,
        'V': 17,
        'W': 18,
        'Y': 19,
        'B': 20,  # Amino acids I created. B has same mass as R
        'O': 21,  # O has same mass as D
        'U': 22,  # U has same mass as E
        'J': 23,  # J has same mass as K
    }

    with open('numbers.txt', 'w') as p:
        for line in lines:
            for letter in line:
                p.write(str(amino_acid_dict.get(letter))+' ')
            p.write('\n')


def calc_box_length(mass, rho, units='Angstrom'):
    """
    Calculate the cubic box lengths of a simulation based on the given mass and density.

    Parameters:
    mass (float): The mass of the object.
    rho (float): Density of simulation in g/mL.

    Returns:
    float: The cubic box length of the simulation in the units specified.
    """

    mass = float(mass) # mass/mole * N_polymers
    avo = (6.022 * 10**23) # molecules/mole
    mass = mass / avo # mass/polymer * N_polymers

    rho = rho / (10**24) # g/Ang^3
    V = mass/rho
    L = V**(1/3)
            
    if units == 'Angstrom':
        return L
    elif units == 'nm':
        L = L/10
        return L


def generate_gendata_inputs(path, Ns, align='no'):
    """
    Generate gendata inputs.

    Args:
        path (str): The path to the directory containing the poly_pdb_files.
        N (int, optional): The value of N. Defaults to 100.
        align (str, optional): The alignment option. Defaults to 'no'.
    """

    # Append '/poly_pdb_files' to the path
    path = path + '/poly_pdb_files'
    
    # Get a list of all files in the directory specified by path
    files = os.listdir(path)

    # Loop through each file in the directory
    for filename, N in zip(files, Ns):
        # Open a new file in write mode with the name 'gendata_list-{filename}.txt'
        with open(f'gendata_list-{filename}.txt','w') as f:
            # Write the path of the file, the value of N, and the alignment option to the new file
            f.write(f'{path}/{filename} {N} {align}')


def write_partition(path, cpus_per_partition, num_pol = None, partitions=None, max_cores=3000, max_cores_per_node=96, classes=None, plog=None, pscreen=None, lmp_name='lmp.in', model=None):
    """
    Write a SLURM partition submit script.

    Args:
        path (str): The path to write the partition submit script.
        nodes (int, optional): The number of nodes. Defaults to 1.
        cpus_per_node (int, optional): The number of CPUs per node. Defaults to 96.
        partitions (int, optional): The number of partitions. Defaults to 96.
        cpus_per_partition (int, optional): The number of CPUs per partition. Defaults to 1.
        plog (str, optional): The plog option. Defaults to None.
        pscreen (str, optional): The pscreen option. Defaults to None.
        type (str, optional): The type of partition variable. Defaults to 'world'.
    """
    
    lmp_binary = '/home/zl4808/software/lammps-2Aug2023/build_cpu/lmp_stellarCpu' #if model == 'mpipi' else '/home/mawebb/lammps_mittal/mittal/lmp_stellarCpuMittal'

    if classes is not None:
        ## write partition string
        class_ids = np.unique(classes)
        num_class = [np.sum(classes == i) for i in class_ids]   
        partition_string = ' '.join([f'{num_class[i]*num_pol}x{cpus_per_partition[i]}' for i in range(len(class_ids))])
        partitions = np.sum(num_class)
        total_cores = np.sum([num_class[i]*cpus_per_partition[i] for i in range(len(class_ids))])
    else:
        total_cores = partitions*cpus_per_partition

        while total_cores > max_cores:
            partitions -= 1
            total_cores -= cpus_per_partition
        
        partition_string = f'{partitions}x{cpus_per_partition}'
        #total_cores = partitions * cpus_per_partition

    ### find number of nodes
    nodes = int(np.ceil(total_cores/max_cores_per_node)) 
    if total_cores <= 5000:
        time_alotted = '23:59:59'
    if total_cores <= 3000:
        time_alotted = '47:59:59'
    if max_cores <= 2500:
        time_alotted = '86:59:59'

    with open(path+'/partition.submit', 'w') as p:
        p.write(f'''#!/bin/bash
#SBATCH --job-name=uni_{model}	 # create a short name for your job
#SBATCH --nodes={nodes}                # node count
#SBATCH --ntasks={total_cores} #           # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=200MB        # memory per cpu-core (4G is default)
#SBATCH --time={time_alotted}          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --constraint=cascade
#SBATCH --mail-user=zl4808@princeton.edu
#SBATCH --output=output
#SBATCH --error=error

module purge
module load intel/2022.2.0 intel-mpi/intel/2021.7.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

srun {lmp_binary} -in {lmp_name} -partition {partition_string} {'-plog none' if plog is None else ''} {'-pscreen none' if pscreen is None else ''}''')


def write_universe(path, subdirectories, lammps_input, name='in.universe'):
    """
    Write the universe file for LAMMPS simulation.

    Args:
        path (str): The path to the directory where the universe file will be created.
        subdirectories (list): A list of subdirectories to be included in the universe file.
        lammps_input (str): Name of the lammps input file to use.
    """

    with open(os.path.join(path,name), 'w') as p:
        p.write(f'''variable d universe {' '.join(reversed(subdirectories))}
shell cd $d
log log.lammps
include {str(lammps_input)}
clear
shell cd {str(path)}
next d
jump {name}''')


def generate_EOS(path, name='in.eos', nsteps=10000000, nequil=1000000, dt=10, density=None, box_length=None, model=None, charge=1):
    """
    Generate a LAMMPS input file for EOS calculations.

    Parameters:
        path (str): Path to save the LAMMPS input file.
        name (str): Name of the LAMMPS input file. Defaults to 'in.eos'.
        nsteps (int): Number of production steps. Defaults to 10 million.
        nequil (int): Number of equilibration steps. Defaults to 1 million.
        dt (int): Timestep in femtoseconds.
        density (float): System density in g/mL.
        box_length (float): Cubic box length in Ångströms.
        model (str): Interaction model (e.g., 'hps_urry').
    """
    if density is None or box_length is None:
        raise ValueError("Both density and box length must be specified for EOS calculations.")
    
    nequilsplit = nequil//2

    # Pair style based on model
    pair_style = 'ljlambda 0.1 0.0 35.0'
    if model == 'mpipi':
        pair_style = 'hybrid/overlay wf/cut 25.0 coul/debye 0.131 0.0' if charge else 'hybrid wf/cut 25.0'
    elif model == 'calvados':
        pair_style = 'ljlambda 0.1041 0.0 0.0'

    # Random seeds
    random_int1 = np.random.randint(1, 10000000)
    random_int2 = np.random.randint(1, 10000000)
    random_int3 = np.random.randint(1, 10000000)
    random_int4 = np.random.randint(1, 10000000)

    # Generate the LAMMPS input script
    with open(os.path.join(path, name), 'w') as f:
        f.write(f"""# LAMMPS input script for EOS
variable        data_name      index 	sys.data
variable        settings_name  index    sys.settings
variable        nsteps         index    {nsteps}
variable        nequil         index    {nequil}
variable        nequilsplit    index    {nequilsplit}
variable        Tinit          index    300
variable        T0             index    300
variable        Tf             index    300
variable        Tdamp          index    1000
variable        vseed1         index    {random_int1}
variable        vseed2         index    {random_int2}
variable        vseed3         index    {random_int3}
variable        vseed4         index    {random_int4}

#===========================================================
# SYSTEM DEFINITION
#===========================================================
units		real	# m = grams/mole, x = Angstroms, E = kcal/mole
dimension	3	# 3 dimensional simulation
newton		on	# use Newton's 3rd law
boundary	p p p	# shrink wrap conditions
atom_style	full    # molecular + charge

#===========================================================
# FORCE FIELD DEFINITION
#===========================================================
pair_style     {pair_style}
bond_style     hybrid harmonic
special_bonds  fene
angle_style    none
dihedral_style none
kspace_style   none
improper_style none                 # no impropers
dielectric     {'77.7' if model == 'calvados' else '80.0'}

#===========================================================
# SETUP SIMULATIONS
#===========================================================
# READ IN COEFFICIENTS/COORDINATES/TOPOLOGY
read_data ${{data_name}} 
include ${{settings_name}}

# SET RUN PARAMETERS
#neighbor 3.5 multi
neighbor 5.0 multi
#neigh_modify every 1 delay 0 check yes
neigh_modify every 10 delay 0
comm_style tiled              #could be removed
timestep {dt}
run_style	verlet 		# Velocity-Verlet integrator

# DECLARE RELEVANT OUTPUT VARIABLES
variable        my_step   equal   step
variable        my_temp   equal   temp
variable        my_rho    equal   density
variable        my_pe     equal   pe
variable        my_ke     equal   ke
variable        my_etot   equal   etotal
variable        my_ent    equal   enthalpy
variable        my_P      equal   press
variable        my_vol    equal   vol

#===========================================================
# PERFORM ENERGY MINIMIZATION
#===========================================================
minimize 1.0e-4 1.0e-6 500000 50000000

#===========================================================
# EQUILIBRATION PHASE
#===========================================================
velocity all create ${{Tinit}} ${{vseed1}} mom yes rot yes
fix momentum_fix all momentum 1000 linear 1 1 1
fix bal      all balance 1000 1.0 shift xyz 10 1.1        # Load balancing


fix lang0     all langevin ${{T0}} ${{Tf}} ${{Tdamp}} ${{vseed2}} 
fix dynamics0 all nve/limit 0.1
thermo 10000
thermo_style custom step temp pe ke etotal press

run ${{nequilsplit}}   # Run equilibration for nequil steps

unfix dynamics0
unfix lang0

fix lang1 all langevin ${{T0}} ${{Tf}} ${{Tdamp}} ${{vseed3}} 
fix dynamics1 all nve

run ${{nequilsplit}}   # Run equilibration for nequil steps

unfix dynamics1
unfix lang1

unfix momentum_fix
reset_timestep 0   # Reset timestep for production run

#===========================================================
# PRODUCTION PHASE
#===========================================================
fix lang2     all langevin ${{T0}} ${{Tf}} ${{Tdamp}} ${{vseed4}}   # Langevin thermostat for production
fix dynamics2 all nve                                      # Integrate equations of motion


# Output setup for production
thermo 1000
thermo_style custom step temp pe ke etotal press density

#===========================================================
# SET OUTPUTS
#===========================================================
fix  averages all ave/time 100 1 100 v_my_temp v_my_etot v_my_pe v_my_ke v_my_ent v_my_P v_my_rho file thermo.avg


restart       50000 restart1.tmp restart2.tmp
run             ${{nsteps}}
write_data      end.data pair ij
""")


def generate_diff(path, name='in.diff', nsteps=15000000, nequil=1000000, dt=10, density=0.5, box_length=None, model=None, charge=1):
    """
    Generate a LAMMPS input file for EOS calculations.

    Parameters:
        path (str): Path to save the LAMMPS input file.
        name (str): Name of the LAMMPS input file. Defaults to 'in.eos'.
        nsteps (int): Number of production steps. Defaults to 10 million.
        nequil (int): Number of equilibration steps. Defaults to 1 million.
        dt (int): Timestep in femtoseconds.
        density (float): System density in g/mL.
        box_length (float): Cubic box length in Ångströms.
        model (str): Interaction model (e.g., 'hps_urry').
    """
    if density is None or box_length is None:
        raise ValueError("Both density and box length must be specified for EOS calculations.")

    # Pair style based on model
    pair_style = 'ljlambda 0.1 0.0 35.0'
    if model == 'mpipi':
        pair_style = 'hybrid/overlay wf/cut 25.0 coul/debye 0.131 0.0' if charge else 'hybrid wf/cut 25.0'
    elif model == 'calvados':
        pair_style = 'ljlambda 0.1041 0.0 0.0'

    # Random seeds
    random_int1 = np.random.randint(1, 10000000)
    random_int2 = np.random.randint(1, 10000000)
    random_int3 = np.random.randint(1, 10000000)
    random_int4 = np.random.randint(1, 10000000)

    nequil1 = int(0.65*nequil)
    nequil2 = int(0.35*nequil)

    # Generate the LAMMPS input script
    with open(os.path.join(path, name), 'w') as f:
        f.write(f"""# LAMMPS input script for calculating diffusivity
variable        data_name      index 	sys.data
variable        settings_name  index    sys.settings
variable        nsteps         index    {nsteps}
variable        nequil         index    {nequil}
variable        nequil1        index    {nequil1}
variable        nequil2        index    {nequil2}
variable        Tinit          index    300
variable        T0             index    300
variable        Tf             index    300
variable        Tdamp          index    1000
variable        P0             index    1
variable        Pf             index    1
variable        Pdamp          index    10000
variable        vseed1         index    {random_int1}
variable        vseed2         index    {random_int2}
variable        vseed3         index    {random_int3}
variable        vseed4         index    {random_int4}
variable        coords_freq    index    100000
variable        print_thermo   index    1000

#===========================================================
# SYSTEM DEFINITION
#===========================================================
units		real	# m = grams/mole, x = Angstroms, E = kcal/mole
dimension	3	# 3 dimensional simulation
newton		on	# use Newton's 3rd law
boundary	p p p	# shrink wrap conditions
atom_style	full    # molecular + charge

#===========================================================
# FORCE FIELD DEFINITION
#===========================================================
pair_style     {pair_style}
bond_style     hybrid harmonic
special_bonds  fene
angle_style    none
dihedral_style none
kspace_style   none
improper_style none                 # no impropers
dielectric     {'77.7' if model == 'calvados' else '80.0'}

#===========================================================
# SETUP SIMULATIONS
#===========================================================
# READ IN COEFFICIENTS/COORDINATES/TOPOLOGY
read_data ${{data_name}} 
include ${{settings_name}}

# SET RUN PARAMETERS
#neighbor 3.5 multi
neighbor 5.0 multi
neigh_modify every 1 delay 0 check yes
comm_style tiled              #could be removed
timestep {dt}
run_style	verlet 		# Velocity-Verlet integrator

# BOX VARIABLES (used for rescaling)
variable        lx equal lx
variable        ly equal ly
variable        lz equal lz

# DECLARE RELEVANT OUTPUT VARIABLES
variable        my_step   equal   step
variable        my_temp   equal   temp
variable        my_rho    equal   density
variable        my_pe     equal   pe
variable        my_ke     equal   ke
variable        my_etot   equal   etotal
variable        my_ent    equal   enthalpy
variable        my_P      equal   press
variable        my_vol    equal   vol

#===========================================================
# PERFORM ENERGY MINIMIZATION
#===========================================================
minimize 1.0e-6 1.0e-6 500000 500000000

#===========================================================
# EQUILIBRATION PHASE
#===========================================================
velocity all create ${{Tinit}} ${{vseed1}} mom yes rot yes
fix momentum_fix all momentum 1000 linear 1 1 1
fix bal      all balance 1000 1.0 shift xyz 10 1.1        # Load balancing

fix equil all nvt temp ${{T0}} ${{Tf}} ${{Tdamp}}     # Equilibration using NVT ensemble
thermo 10000
thermo_style custom step temp pe ke etotal press density vol

#fix bal all balance 1000 1.05 rcb


run 500000   # Run equilibration for nequil steps

# fix equil_berendsen all press/berendsen iso ${{P0}} ${{P0}} ${{Pdamp}}

# run 400000   # Short Berendsen run (~50-100 ps)

# unfix equil_berendsen

unfix equil
unfix momentum_fix
reset_timestep 0   # Reset timestep for production run

#===========================================================
# PRE-NPT SHORT RUN
#===========================================================

# print "--------------- pre-NPT starts ------------"


# fix lang1 all langevin ${{T0}} ${{Tf}} 10000.0 ${{vseed2}} # 10 ps damping
# fix baro1 all nph iso 1.0 1.0 50000.0  # 50 ps damping

# fix             2 all ave/time 1000 100 ${{nequil}} v_my_vol ave one

# variable mean_Vol equal f_2


# run ${{nequil}}  # because we switch barostat styles

# print           "Average volume after pre-NPT = ${{mean_Vol}}"
# unfix           2

# unfix lang1
# unfix baro1

# reset_timestep 0

# print "--------------- pre-NPT ends ------------"

#===========================================================
# NPT EQUILIBRIATION
#===========================================================

print "--------------- NPT starts ------------"


fix lang2 all langevin ${{T0}} ${{Tf}} 10000.0 ${{vseed3}} # 10 ps damping
fix baro2 all nph iso 1.0 1.0 50000.0  # 50 ps damping


fix             3 all ave/time 5000 200 ${{nequil}} v_my_vol ave one

run 400000


run ${{nequil}}  # because we switch barostat styles

variable mean_Vol equal f_3

print           "Average volume after NPT = ${{mean_Vol}}"
print           "Last volume after NPT = ${{my_vol}}"


variable target_L equal v_mean_Vol^(1.0/3.0)

# Calculate scale factors for each dimension
variable scaleX equal v_target_L/v_lx
variable scaleY equal v_target_L/v_ly
variable scaleZ equal v_target_L/v_lz


print           "scaleX = ${{scaleX}}"
print           "scaleY = ${{scaleY}}"
print           "scaleZ = ${{scaleZ}}"

# Apply the scaling to fix the volume
change_box  all x scale ${{scaleX}}      &
                y scale ${{scaleY}}      &
                z scale ${{scaleZ}}      &
                remap

print           "New volume value after change_box = ${{my_vol}}"
print "--------------- NPT ends  ------------"

unfix           3

unfix lang2
unfix baro2

# reset_timestep 0

#===========================================================
# PRODUCTION RUN
#===========================================================
minimize 1.0e-4 1.0e-6 1000 10000 # to get rid of potential overlaps

fix lang all langevin ${{T0}} ${{Tf}} 100000.0 ${{vseed4}} # 100 ps damping light but present thermostat
fix dynamics all nve

run 200000

reset_timestep 0


# Compute Center-of-Mass of each molecule
compute molchunk all chunk/atom molecule
compute commol all com/chunk molchunk
fix com_mol all ave/time 1000 1 1000 c_commol[*] file com_mol.dat mode vector

# Coordinates output setup
thermo 1000
thermo_style custom step temp pe ke etotal press density vol

dump crds_dcd all dcd ${{coords_freq}} coords.dcd
fix           fixcentro all recenter INIT INIT INIT

#===========================================================
# RUN SIMULATION
#===========================================================
restart       50000 restart1.tmp restart2.tmp
run           ${{nsteps}}

unfix lang
unfix dynamics
unfix fixcentro
unfix bal
write_data    end.data pair ij""")


def generate_diff_restart(path, name='in.restart', nsteps=15000000, dt=10, model=None, charge=1, restart_file='restart.lmp'):
    """
    Generate a LAMMPS input file for EOS calculations.

    Parameters:
        path (str): Path to save the LAMMPS input file.
        name (str): Name of the LAMMPS input file. Defaults to 'in.eos'.
        nsteps (int): Number of production steps. Defaults to 10 million.
        nequil (int): Number of equilibration steps. Defaults to 1 million.
        dt (int): Timestep in femtoseconds.
        density (float): System density in g/mL.
        box_length (float): Cubic box length in Ångströms.
        model (str): Interaction model (e.g., 'hps_urry').
    """
    # Pair style based on model
    pair_style = 'ljlambda 0.1 0.0 35.0'
    if model == 'mpipi':
        pair_style = 'hybrid/overlay wf/cut 25.0 coul/debye 0.131 0.0' if charge else 'hybrid wf/cut 25.0'
    elif model == 'calvados':
        pair_style = 'ljlambda 0.1041 0.0 0.0'

    # Random seeds
    random_int1 = np.random.randint(1, 10000000)
    random_int2 = np.random.randint(1, 10000000)

    # Generate the LAMMPS input script
    with open(os.path.join(path, name), 'w') as f:
        f.write(f"""# LAMMPS input script for calculating diffusivity
variable        settings_name  index    sys.settings
variable        nsteps         index    {nsteps}
variable        Tinit          index    300
variable        T0             index    300
variable        Tf             index    300
variable        Tdamp          index    1000
variable        P0             index    1
variable        Pf             index    1
variable        Pdamp          index    10000
variable        vseed1         index    {random_int1}
variable        vseed2         index    {random_int2}
variable        coords_freq    index    10000


#===========================================================
# FORCE FIELD DEFINITION
#===========================================================
pair_style     {pair_style}
bond_style     hybrid harmonic
special_bonds  fene
angle_style    none
dihedral_style none
kspace_style   none
improper_style none                 # no impropers
dielectric     {'77.7' if model == 'calvados' else '80.0'}

#===========================================================
# SETUP SIMULATIONS
#===========================================================
# READ IN COEFFICIENTS/COORDINATES/TOPOLOGY
read_restart {restart_file}
include ${{settings_name}}

# SET RUN PARAMETERS
#neighbor 3.5 multi
neighbor 5.0 bin 

comm_style tiled              #could be removed
timestep {dt}
run_style	verlet 		# Velocity-Verlet integrator

# DECLARE RELEVANT OUTPUT VARIABLES
variable        my_step   equal   step
variable        my_temp   equal   temp
variable        my_rho    equal   density
variable        my_pe     equal   pe
variable        my_ke     equal   ke
variable        my_etot   equal   etotal
variable        my_ent    equal   enthalpy
variable        my_P      equal   press
variable        my_vol    equal   vol


#===========================================================
# PRODUCTION PHASE
#===========================================================
fix npt_fix all npt temp ${{T0}} ${{Tf}} ${{Tdamp}} iso ${{P0}} ${{P0}} ${{Pdamp}}

fix bal      all balance 500 1.0 shift xyz 20 1.2        # Load balancing

fix           fixcentro all recenter INIT INIT INIT


# Compute Center-of-Mass of each molecule
compute molchunk all chunk/atom molecule
compute commol all com/chunk molchunk
fix com_mol all ave/time 100 1 100 c_commol[*] file com_mol_restart.dat mode vector

neigh_modify every 5 delay 20 check yes

# Output setup
thermo 1000
thermo_style custom step temp pe ke etotal press density vol

dump crds_dcd all dcd ${{coords_freq}} coords_restart.dcd

#===========================================================
# RUN SIMULATION
#===========================================================
restart       50000 restart1.tmp restart2.tmp
run           ${{nsteps}} upto
write_data    end.data pair ij
unfix npt_fix""")
