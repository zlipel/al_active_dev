
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

# Add the repo root to sys.path so `analysis` and `external` (siblings of
# `simulation/`) are importable when run as a script. The pathlib chain is
# equivalent to os.path.dirname(os.path.abspath(__file__)) + '/..', just
# resolved to an absolute path once so downstream logic can reason about it.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Ray workers spawn as fresh subprocesses and do NOT inherit the driver's
# sys.path modifications — they only see PYTHONPATH from the environment.
# Without this, `@ray.remote def _calc_exp_density` fails to import
# `analysis.process_eos_sims` on the worker with ModuleNotFoundError, even
# though the driver import above works fine.
_existing_pp = os.environ.get("PYTHONPATH", "")
if _REPO_ROOT not in _existing_pp.split(os.pathsep):
    os.environ["PYTHONPATH"] = (
        f"{_REPO_ROOT}{os.pathsep}{_existing_pp}" if _existing_pp else _REPO_ROOT
    )
from external.core import (
    calculate_mass,
    calc_box_length,
    convert_amino_acids_to_numbers,
    generate_gendata_inputs,
    generate_EOS,
    write_partition,
    write_universe,
)
from analysis.process_eos_sims import bootstrap_exp_dens_from_path
import argparse


def main(
    model: str = 'hps_urry',
    num_polymers: int = 100,
    density_values: Optional[List[float]] = None,
    sequence_file: Optional[str] = None,
    parent_dir: Optional[str] = None,
    check_densities: bool = False,
    check_finished: bool = False,
    cpus_per_sim: int = 12,
    max_cores: int = 1500,
):
    """
    Generate LAMMPS EOS input scripts for a batch of IDP sequences.

    Args:
        model: Force-field model (hps_urry, mpipi, calvados, etc.)
        num_polymers: Chains per simulation box.
        density_values: List of densities (g/mL) to simulate.
        sequence_file: Path to sequence list (.txt, one sequence per line).
        parent_dir: Root output directory for all simulation subdirs.
        check_densities: When True, filter by expenditure density before running.
        check_finished: When True, skip already-completed simulations.
        cpus_per_sim: CPUs allocated per simulation partition. Increase for dense sequences.
        max_cores: Total core ceiling for this submission. Adjust to match your SLURM partition
                   (Stellar medium: 3000, long: 2500).
    """
    if density_values is None:
        density_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]

    config = SimulationConfig.create(
        model=model,
        num_polymers=num_polymers,
        density_values=np.asarray(density_values),
        sequence_file=Path(sequence_file),
        parent_dir=Path(parent_dir)
    )
    print(f"Using model: {config.model}, num_polymers: {config.num_polymers}, "
          f"density_values: {config.density_values} with sequence file: {config.sequence_file}", flush=True)

    manager = SimulationManager(config)
    print(f"Loaded {len(manager.sequences)} sequences from {config.sequence_file}.", flush=True)

    try:
        print(f'Simulating {len(manager.sequences)} sequences at {len(config.density_values)} densities.')

        manager.setup_environment()
        commands = manager.create_simulation_commands()

        if check_finished:
            print('Filtering simulations based on completion...', flush=True)
            filtered_commands = manager.filter_by_completion(commands)
        elif check_densities:
            print('Filtering simulations based on expenditure density...', flush=True)
            filtered_commands = manager.filter_by_density(commands, 0.2, 200)
        else:
            filtered_commands = commands

        print(f'Remaining simulations: {len(filtered_commands)}', flush=True)

        if len(filtered_commands) == 0:
            print("All done!!!\n")
            return 0

        paths = manager.run_simulations(filtered_commands)

        max_cores_per_node = 96  # Stellar hardware constant
        max_partitions = len(paths)

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
            lmp_name='in.eos'
        )

        write_universe(
            str(config.parent_dir),
            paths,
            'eos.lmp',
            name='in.eos'
        )
    finally:
        manager.cleanup()


@dataclass
class SimulationConfig:
    model: str
    num_polymers: int
    density_values: np.ndarray
    sequence_file: Path
    parent_dir: Path

    @classmethod
    def create(cls, model: str, num_polymers: int, density_values: List[float], sequence_file: str, parent_dir: str):
        return cls(
            model=model,
            num_polymers=num_polymers,
            density_values=np.array(density_values),
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
        for density in self.config.density_values:
            for i, seq in enumerate(self.sequences):
                commands.append((self.masses[i], self.charges[i], i, density, str(self.config.parent_dir)))
        return commands

    @staticmethod
    @ray.remote
    def _calc_exp_density(path: Path, frac: float, num_bootstrap: int) -> float:
        return bootstrap_exp_dens_from_path(str(path), frac, num_bootstrap=num_bootstrap)

    def filter_by_completion(self, commands: List[Tuple], nsteps: int = 10000000) -> List[Tuple]:
        """
        Filters out already-completed simulations based on the last timestep in thermo.avg.
        Simulations with a missing or incomplete thermo.avg are kept for re-run.
        """
        kept_commands = []
        for command in commands:
            _, _, i, density, parent_dir = command
            output_dir = Path(parent_dir) / f'poly{i}/rho{density}'
            thermo_file = output_dir / 'thermo.avg'

            if not output_dir.exists():
                kept_commands.append(command)
                continue

            if thermo_file.exists():
                try:
                    with open(thermo_file, 'r') as f:
                        last_line = f.readlines()[-1].strip()
                        last_timestep = int(last_line.split()[0])
                    if last_timestep < nsteps:
                        print(f"Simulation {i} at density {density} is incomplete. "
                              f"Last timestep: {last_timestep}. Restarting from scratch.", flush=True)
                        kept_commands.append(command)
                except Exception as e:
                    print(f"Warning: Could not parse thermo.avg in {output_dir}: {e}")
                    kept_commands.append(command)
            else:
                print(f"Simulation {i} at density {density} has no thermo.avg. Restarting from scratch.", flush=True)
                kept_commands.append(command)

        return kept_commands

    def filter_by_density(self, commands: List[Tuple], threshold: float, num_bootstrap: int,
                          nsteps: int = 10000000) -> List[Tuple]:
        """
        Filters out simulations whose expenditure density has already been reached.
        Also skips any simulation whose thermo.avg reports >= 95% of nsteps completed.
        """
        ray.init(num_cpus=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
        kept_commands = []

        exp_density_futures = [
            self._calc_exp_density.remote(self.config.parent_dir / f'poly{i}', 0.5, num_bootstrap)
            for i in range(len(self.sequences))
        ]
        exp_densities = ray.get(exp_density_futures)

        for command in commands:
            _, _, k, density, parent_dir = command
            output_dir = os.path.join(parent_dir, f'poly{k}/rho{density}')
            thermo_file = os.path.join(output_dir, 'thermo.avg')

            if Path(thermo_file).exists():
                try:
                    with open(thermo_file, 'r') as f:
                        last_line = f.readlines()[-1].strip()
                        last_timestep = int(last_line.split()[0])
                    if last_timestep >= int(0.95 * nsteps):
                        continue
                    else:
                        print(f"Simulation {k} at density {density} is incomplete. "
                              f"Last timestep: {last_timestep}. Restarting from scratch.", flush=True)
                        kept_commands.append(command)
                except Exception as e:
                    print(f"Warning: Could not parse thermo.avg in {output_dir}: {e}")
            else:
                exp_rho = exp_densities[k % len(self.sequences)]
                if density <= exp_rho + threshold or exp_rho == -1:
                    print(f"Seq {k} has exp density: {exp_rho}, keeping for further analysis.", flush=True)
                    kept_commands.append(command)
                else:
                    print(f"Seq {k} is done with exp density: {exp_rho}.\n")

        ray.shutdown()
        return kept_commands

    def run_simulations(self, commands: List[Tuple]) -> List[str]:
        results = Parallel(n_jobs=-1)(
            delayed(self.run_single_simulation)(cmd) for cmd in commands
        )
        return [res for res in results if res is not None]

    def run_single_simulation(self, command: Tuple) -> str:
        """Runs LAMMPS input generation using generate_EOS()."""
        mass, charge, i, density, parent_dir = command
        output_dir = Path(parent_dir) / f'poly{i}/rho{density}'
        output_dir.mkdir(parents=True, exist_ok=True)

        L = calc_box_length(self.config.num_polymers * mass, density)
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
            generate_EOS(str(output_dir), name='eos.lmp', density=density,
                         box_length=calc_box_length(self.config.num_polymers * mass, density),
                         model=self.config.model, charge=charge)
            return str(output_dir)
        except subprocess.CalledProcessError as e:
            print(f"Error running gen_data.py for model {self.config.model} for sequence {i}: {e}")
            return None

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
    parser = argparse.ArgumentParser(description='Generate LAMMPS EOS input scripts for IDP sequences.')
    parser.add_argument('--model', type=str, default='hps_urry', help='Force-field model (hps_urry, mpipi, calvados, etc.)')
    parser.add_argument('--num_polymers', type=int, default=100, help='Chains per simulation box.')
    parser.add_argument('--density_start', type=float, required=True, help='First density to simulate (g/mL).')
    parser.add_argument('--density_end', type=float, required=True, help='Final density to simulate (g/mL).')
    parser.add_argument('--density_step', type=float, required=True, help='Density increment (g/mL).')
    parser.add_argument('--sequence_file', type=str, required=True, help='Path to sequence list (.txt).')
    parser.add_argument('--parent_dir', type=str, required=True, help='Root output directory for simulation subdirs.')
    parser.add_argument('--check_densities', action=argparse.BooleanOptionalAction,
                        help='Filter simulations based on expenditure density.')
    parser.add_argument('--check_finished', action='store_true',
                        help='Skip already-completed simulations.')
    parser.add_argument('--cpus_per_sim', type=int, default=12,
                        help='CPUs per simulation partition. Increase for dense sequences (default: 12).')
    parser.add_argument('--max_cores', type=int, default=1500,
                        help='Total core ceiling for this submission. Match to your SLURM partition '
                             '(Stellar medium: 3000, long: 2500; default: 1500).')
    args = parser.parse_args(args)

    rho_i = np.round(args.density_start, decimals=2)
    rho_f = np.round(args.density_end, decimals=2)
    drho  = np.round(args.density_step, decimals=2)

    steps = round((rho_f - rho_i) / drho)
    density_values = [np.round(rho, decimals=2) for rho in np.linspace(rho_i, rho_f, steps + 1)]

    print(f"Densities to be simulated: {density_values}.", flush=True)

    main(
        model=args.model.lower(),
        num_polymers=args.num_polymers,
        density_values=density_values,
        sequence_file=args.sequence_file,
        parent_dir=args.parent_dir,
        check_densities=args.check_densities,
        check_finished=args.check_finished,
        cpus_per_sim=args.cpus_per_sim,
        max_cores=args.max_cores,
    )
