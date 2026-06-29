
import numpy as np
import subprocess
import os
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from joblib import Parallel, delayed
import ray
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from external.core import (
    calculate_mass,
    calc_box_length,
    convert_amino_acids_to_numbers,
    generate_gendata_inputs,
    generate_diff,
    generate_diff_restart,
    write_partition,
    write_universe,
)
import pandas as pd
import argparse

def main(
    model: str = 'hps_urry',
    num_polymers: int = 100,
    nsim: int = 5,
    rho_init : Optional[List[float]] = None,
    sequence_file: Optional[str] = None,
    parent_dir: Optional[str] = None,
    check_finished: bool = False,
    cpus_per_sim: int = 16,
    max_cores: int = 480,
):
    """
    Generate LAMMPS diffusivity input scripts for a batch of IDP sequences.

    Args:
        model: Force-field model (hps_urry, mpipi, calvados, etc.)
        num_polymers: Chains per simulation box.
        nsim: Number of independent production runs.
        rho_init: Initial density (g/mL) per sequence; read from eos_results.csv when not provided.
        sequence_file: Path to sequence list (.txt, one sequence per line).
        parent_dir: Root output directory for all simulation subdirs.
        check_finished: When True, skip already-completed simulations.
        cpus_per_sim: CPUs allocated per simulation partition. Increase for dense sequences.
        max_cores: Total core ceiling for this submission. Adjust to match your SLURM partition
                   (Stellar medium: 3000, long: 2500).
    """
    config = SimulationConfig.create(
        model=model,
        num_polymers=num_polymers,
        nsim=nsim,
        rho_init=rho_init,
        sequence_file=Path(sequence_file),
        parent_dir=Path(parent_dir)
    )

    manager = SimulationManager(config)

    try:
        print(f'Simulating {len(manager.sequences)} sequences for {nsim} production runs.')

        manager.setup_environment()
        commands = manager.create_simulation_commands()

        if check_finished:
            print("Filtering based on completed simulations...\n", flush=True)
            command_temp = Parallel(n_jobs=-1)(delayed(manager.check_complete)(cmd) for cmd in commands)
            filtered_commands = [cmd for cmd in command_temp if cmd != False]
            print(f"Remaining simulations: {len(filtered_commands)}", flush=True)

            if len(filtered_commands) == 0:
                print("All done!!!\n")
                return 0
        else:
            filtered_commands = commands

        paths = manager.run_simulations(filtered_commands)

        max_cores_per_node = 96  # Stellar hardware constant
        max_partitions = len(paths)

        # Fit as many partitions as possible within the core ceiling,
        # rounding down to a whole number of nodes.
        partitions = min(max_partitions, max_cores // cpus_per_sim)
        total_cores = partitions * cpus_per_sim

        if max_partitions * cpus_per_sim > 1000:
            while total_cores % max_cores_per_node != 0 and partitions > 0:
                partitions -= 1
                total_cores = partitions * cpus_per_sim
        else:
            partitions = max_partitions
            total_cores = partitions * cpus_per_sim

        if partitions == 0:
            raise ValueError("Could not find valid partitioning under core/node constraints.")

        write_partition(
            str(config.parent_dir),
            cpus_per_sim,
            partitions=partitions,
            model=config.model,
            max_cores=total_cores,
            lmp_name='in.diff'
        )

        write_universe(
            str(config.parent_dir),
            paths,
            'diff.lmp',
            name='in.diff'
        )
    finally:
        manager.cleanup()


@dataclass
class SimulationConfig:
    model: str
    num_polymers: int
    nsim: int
    rho_init: List[float]
    sequence_file: Path
    parent_dir: Path

    @classmethod
    def create(cls, model: str, num_polymers: int, nsim: int, rho_init: List[float], sequence_file: str, parent_dir: str):
        return cls(
            model=model,
            num_polymers=num_polymers,
            nsim=nsim,
            rho_init=rho_init,
            sequence_file=Path(sequence_file),
            parent_dir=Path(parent_dir)
        )

class ModelType(Enum):
    MOFF = 'moff'
    MPIPI = 'mpipi'
    HPS_KR = 'hps_kr'
    HPS_URRY = 'hps_urry'
    CALVADOS = 'calvados'


class SimulationManager:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.sequences = self.load_sequences()
        self.masses  = [calculate_mass(seq) for seq in self.sequences]
        self.charges = [self.get_mpipi_charge(seq) for seq in self.sequences]

        self.gendata_params = {
            'hps_urry': ['amino_acid_Urry.py', 'Urry'],
            'hps_kr'  : ['amino_acid_KR.py', 'arithmetic'],
            'mpipi'   : ['mpipi.py', None],
            'calvados': ['calvados.py', 'arithmetic']
        }

    def load_sequences(self) -> List[str]:
        with open(self.config.sequence_file, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def setup_environment(self):
        self.config.parent_dir.mkdir(exist_ok=True)
        convert_amino_acids_to_numbers(self.sequences)

        path = os.path.expanduser('~/scripts/POLYMERIZE/polymerize.py')
        monomer_list = os.path.expanduser('~/scripts/AA_monomer_list.txt')
        subprocess.run(['python',
                        path,
                        monomer_list, '-Np', str(len(self.sequences)), '-seq',
                        "numbers.txt", '-max_rot', '0.6',
                        '-model', self.config.model], check=True)
        if self.config.model != "moff":
            generate_gendata_inputs(os.getcwd(), Ns=[self.config.num_polymers]*len(self.sequences))

    def create_simulation_commands(self) -> List[Tuple]:
        """Create commands for LAMMPS simulations."""
        commands = []
        for run in range(self.config.nsim):
            for i, seq in enumerate(self.sequences):
                commands.append((self.masses[i], self.charges[i], i, run, str(self.config.parent_dir), 0))
        return commands

    def check_complete(self, command: Tuple, nsteps: int = 15000000, nfreq: int = 1000, nchains: int = 100) -> Tuple:
        mass, charge, i, run, parent_dir, rstart = command
        output_dir = Path(parent_dir) / f'poly{i}/{run}'
        com_data = output_dir / 'com_mol.dat'

        if com_data.exists():
            try:
                data = np.loadtxt(str(com_data), usecols=0, comments='#')
                nsteps_out = (data.shape[0] / ((nchains + 1) - 1)) * nfreq
                if nsteps_out >= nsteps:
                    return False
            except Exception as e:
                print(f"File corrupted in {output_dir}: {e}, keeping directory.\n")
                return (mass, charge, i, run, parent_dir, 0)

        print(f"Keeping directory: {parent_dir}/poly{i}/{run}\n")
        return (mass, charge, i, run, parent_dir, 0)

    def run_simulations(self, commands: List[Tuple]) -> List[str]:
        results = Parallel(n_jobs=-1)(
            delayed(self.run_single_simulation)(cmd) for cmd in commands
        )
        return [res for res in results if res is not None]

    def run_single_simulation(self, command: Tuple) -> str:
        """Runs LAMMPS input generation using generate_diff()."""
        mass, charge, i, run, parent_dir, restart = command
        output_dir = Path(parent_dir) / f'poly{i}/{run}'

        if restart == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            L = calc_box_length(self.config.num_polymers * mass, self.config.rho_init[i])
            database, mix = self.gendata_params[self.config.model]
            pdb_file = f'gendata_list-poly-{i}.pdb.txt'
            subprocess.run(['cp', pdb_file, output_dir])

            gendata_inp = [
                'python', os.environ.get('GENDATA'), pdb_file, database,
                '-init', 'random',
                '-box', f"{L} {L} {L}",
                '-exclude', 'angles dihedrals',
                '-avoid_overlap', 'False'
            ]
            if mix:
                gendata_inp.extend(['-mix', mix])
            else:
                gendata_inp.extend(['-nonbond', 'yes'])
            if self.config.model == 'calvados' or self.config.model == 'hps_urry':
                gendata_inp.extend(['-model', self.config.model])

            try:
                subprocess.run(gendata_inp, cwd=output_dir, check=True)
                generate_diff(str(output_dir), name='diff.lmp', density=self.config.rho_init[i],
                        box_length=calc_box_length(self.config.num_polymers * mass, self.config.rho_init[i]),
                        model=self.config.model, charge=charge)
                return str(output_dir)
            except subprocess.CalledProcessError as e:
                print(f"Error running gen_data.py for model {self.config.model} for sequence {i}: {e}")
                return None
        else:
            rfile = f"restart{restart}.tmp"
            generate_diff_restart(str(output_dir), name='diff_restart.lmp', model=self.config.model,
                                  charge=charge, restart_file=rfile)
            return str(output_dir)

    def cleanup(self):
        """Removes temporary files created by polymerize and gendata."""
        subprocess.run(['rm', '-r', 'poly_pdb_files', 'poly_xyz_files', 'numbers.txt'])
        for i in range(len(self.sequences)):
            subprocess.run(['rm', f'gendata_list-poly-{i}.pdb.txt'])

    @staticmethod
    def get_mpipi_charge(sequence: str) -> int:
        charge = 0
        for aa in sequence:
            if aa in ['D', 'E', 'H', 'K', 'R']:
                charge = 1
        return charge


if __name__ == "__main__":
    args = sys.argv[1:]
    parser = argparse.ArgumentParser(description='Generate LAMMPS diffusivity input scripts for IDP sequences.')
    parser.add_argument('--model', type=str, default='hps_urry', help='Force-field model (hps_urry, mpipi, calvados, etc.)')
    parser.add_argument('--num_polymers', type=int, default=100, help='Chains per simulation box.')
    parser.add_argument('--nsim', type=int, default=5, help='Number of independent production runs.')
    parser.add_argument('--sequence_file', type=str, required=True, help='Path to sequence list (.txt).')
    parser.add_argument('--parent_dir', type=str, required=True, help='Root output directory for simulation subdirs.')
    parser.add_argument('--check_finished', action=argparse.BooleanOptionalAction, help='Skip already-completed simulations.')
    parser.add_argument('--test', action=argparse.BooleanOptionalAction, help='Use a fixed initial density (testing only).')
    parser.add_argument('--quick', type=int, default=0, help='Quick mode: 1 = fixed init density without EOS lookup.')
    parser.add_argument('--cpus_per_sim', type=int, default=16,
                        help='CPUs per simulation partition. Increase for dense/large sequences (default: 16).')
    parser.add_argument('--max_cores', type=int, default=480,
                        help='Total core ceiling for this submission. Match to your SLURM partition '
                             '(Stellar medium: 3000, long: 2500; default: 480).')
    args = parser.parse_args(args)

    if args.quick == 0:
        if not args.test:
            eos_results = pd.read_csv(os.path.join(args.parent_dir, 'eos_results.csv'))

            psp     = eos_results['psp'].values
            density = eos_results['density'].values
            exp_rhos = eos_results['exp_density'].values

            rho_init = []
            for i, psp in enumerate(psp):
                if psp == 0:
                    rho_init.append(0.25)
                elif psp != 0:
                    rho_init.append(float(density[i]))
                if exp_rhos[i] == 0.0 and psp == 0:
                    rho_init.append(1.2)
                elif exp_rhos[i] - density[i] >= 1.0 and psp != 0:
                    rho_init.append(float(1.2))
        else:
            with open(args.sequence_file, 'r') as f:
                sequences = [line.strip() for line in f.readlines()]
            rho_init = [0.45 for seq in sequences]
    else:
        with open(args.sequence_file, 'r') as f:
            sequences = [line.strip() for line in f.readlines()]
        rho_init = [0.75 for seq in sequences]

    main(
        model=args.model.lower(),
        num_polymers=args.num_polymers,
        nsim=args.nsim,
        rho_init=rho_init,
        sequence_file=args.sequence_file,
        parent_dir=args.parent_dir,
        check_finished=args.check_finished,
        cpus_per_sim=args.cpus_per_sim,
        max_cores=args.max_cores,
    )
