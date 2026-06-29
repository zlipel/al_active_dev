# Active Learning IDP Pipeline

Active learning pipeline for discovering phase-separating intrinsically disordered
proteins (IDPs) using molecular dynamics simulation. Each iteration proposes new
sequences, generates LAMMPS input files, submits simulations, analyzes results, and
feeds experimental observations back into the surrogate model.

---

## Repository Structure

```
al_active_dev/
├── al_pipeline/        # Full active learning optimization loop (Python package)
├── simulation/         # Generate LAMMPS input files for EOS and diffusivity runs
├── analysis/           # Analyze completed MD simulations
├── beam_search/        # Surrogate-model beam search over sequence space (post-optimization)
├── external/           # Vendored cluster code: core.py + md_calcs C++ extension
├── submit/             # All SLURM job-submission scripts (consolidated, top-level)
├── utils/              # Shared visualization and Pareto-front utilities
├── tests/              # pytest suite (~100 tests covering AL math + helpers)
├── config/             # cluster.env + environment.yml
├── runs/               # (gitignored) HOME-side artifacts: .pt models, plots, logs
└── archive/            # (gitignored) Superseded scripts — preserved locally for reference
```

---

## Installation

The full pipeline (al_pipeline + simulation + analysis + beam search) has cluster
dependencies (intel compilers, openmpi for beam search, a built C++ extension)
that aren't required for local development. The cluster bootstrap is documented
in its own section; for local dev just create the conda env.

### Local dev (any platform)

```bash
conda env create -f config/environment.yml
conda activate al_active_dev
pip install -e .              # installs al_pipeline in editable mode

pytest tests/                 # 102 tests; should pass with no extra setup
```

The vendored C++ MSD extension and `mpi4py` are stubbed by the test suite, so
neither is needed locally. See `external/md_calcs/README.md` if you want to
build the extension on macOS.

### Cluster bootstrap (Princeton Stellar)

One-time setup on a login node. Each step assumes you're inside
`~/PROJECTS/al_active_dev` after cloning.

**1. Conda env**

```bash
module purge
module load anaconda3/2024.6
conda env create -f config/environment.yml
conda activate al_active_dev
pip install -e .
```

**2. Build the md_calcs C++ extension**

The build uses **cluster modules**, not conda toolchain — `cxx-compiler`,
`openmp`, and `cmake` are deliberately absent from `environment.yml` so conda
doesn't shadow the cluster's intel/intel-mpi/libstdc++.

```bash
module purge
module load anaconda3/2024.6
module load intel/2022.2
module load cmake
conda activate al_active_dev

cd external/md_calcs
mkdir -p build && cd build
cmake ..
make -j4
cd ../../..
```

Result: `external/md_calcs/md_calcs_par.cpython-312-x86_64-linux-gnu.so`. Re-run
the build whenever the conda env's Python version changes. See
`external/md_calcs/README.md` for details.

**3. Build mpi4py against the cluster's MPI** (beam search only)

Conda's `mpi4py` ships with its own MPI implementation that can't talk to the
cluster's openmpi. So mpi4py must be built from source with `mpicc` pointing
at the cluster's openmpi wrapper:

```bash
module purge
module load anaconda3/2024.6
module load openmpi/gcc/4.1.2
conda activate al_active_dev

MPICC=$(which mpicc) pip install --no-binary=mpi4py mpi4py
python -c "from mpi4py import MPI; print('mpi4py:', MPI.Get_library_version())"
```

**4. Sanity check**

```bash
conda activate al_active_dev
pytest tests/                       # should be 102 passed
CLUSTER_TESTS=1 pytest tests/       # also runs @pytest.mark.cluster tests
```

**5. Cluster-specific env vars**

`config/cluster.env` defines `HOME_AL`, `SCRATCH_AL`, `SCRATCH_AL_ACQ`,
`DB_PATH`, etc. Submit scripts source this file, so check the paths match your
account. The `GENDATA` env var must also be set if not already in your
`~/.bashrc`:

```bash
export GENDATA="${HOME}/scripts/GENDATA/gendata.py"   # adjust path
```

**6. (One-time) Migrate HOME-side artifacts to `runs/`**

If you ran an earlier version of this pipeline that wrote GPR checkpoints and
logs to `~/PROJECTS/MODEL_COMPARISON/<MODEL>/`, move them under the new
`runs/` artifact root so the current `ALConfig.base_path` default finds them:

```bash
mkdir -p ~/PROJECTS/al_active_dev/runs
for model in MPIPI CALVADOS HPS_URRY HPS_KR MARTINI; do
  if [[ -d ~/PROJECTS/MODEL_COMPARISON/$model ]]; then
    mv ~/PROJECTS/MODEL_COMPARISON/$model ~/PROJECTS/al_active_dev/runs/$model
    echo "moved $model"
  fi
done
```

The loop is a no-op for models you haven't trained. After this, existing `.pt`
checkpoints resume working at the new path — no retraining needed.

After all of the above, `git pull` is enough to roll forward when changes land
on main.

---

## Pipeline Stages

The pipeline runs in a cycle. Each full iteration consists of five stages. Between stages
3 and 4 there is an intentionally manual resource decision (see §Between-Stage Decisions).

---

### Stage 1 — Sequence Optimization (`al_pipeline`)

`al_pipeline/` is the active learning loop. It trains a multitask GP surrogate on the
current experimental observations, runs an EHVI (or Monte Carlo EHVI) acquisition function
to select the next batch of candidate sequences, and writes them to the ALPaths directory
structure.

The CLI has two entry points:

- **`python -m al_pipeline.cli.master`** — orchestration layer; sets up iteration directory,
  trains GPR, generates Pareto front, writes sequence candidates.
- **`python -m al_pipeline.cli.child`** — GA inner loop, one per candidate slot; called in
  parallel by the master script via `srun`.

Key options (see `--help` for full list):

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | — | Force field: `CALVADOS`, `HPS_URRY`, `MPIPI`, `HPS_KR`, `MARTINI` |
| `--front` | `upper` | Pareto front direction: `upper` (maximize both) or `lower` |
| `--ehvi_variant` | `epsilon` | `epsilon` (ε-EHVI) or `standard` |
| `--exploration_strategy` | `kriging_believer` | `kriging_believer`, `constant_liar_min/mean/max`, `similarity_penalty` |
| `--transform` | `yeoj` | Label transform: `yeoj` (Yeo-Johnson) or `log` |
| `--mc_ehvi` | off | Monte Carlo EHVI (use `pygmo.hypervolume` backend) |

The pipeline writes results to the `ALPaths` directory tree rooted at
`$SCRATCH_AL/$MODEL/`, including: `features_gen{N}.csv`, `labels_gen{N}.csv`,
`normalization_stats.json`, and the GPR checkpoint `GPR_iter{N}_{tag}.pt`.

---

### Stage 2 — Generate LAMMPS Input Files (`simulation/`)

After Stage 1 produces candidate sequences, generate the MD input files:

```bash
# Equation of state (phase behavior)
python simulation/make_eos.py \
  --model CALVADOS --iter 10 --front upper \
  --scratch_dir $SCRATCH_AL --home_dir $HOME_AL --db_root $DB_ROOT

# Diffusivity
python simulation/make_diff.py \
  --model CALVADOS --iter 10 --front upper \
  --scratch_dir $SCRATCH_AL --home_dir $HOME_AL --db_root $DB_ROOT
```

Both scripts write LAMMPS universe files into the iteration's EOS/ and DIFF/
subdirectories under `$SCRATCH_AL/$MODEL/GENERATIONS/iteration_{N}/`.

`make_diff.py` can optionally use EOS simulation outputs as starting configurations
(pass `--use_eos_results`). This is recommended for dense / phase-separating sequences
where equilibration from random configurations is slow.

Both scripts require `core.py` from `$CORE_LIB` (see §External Dependencies).

Submit via the consolidated top-level `submit/`:
```bash
sbatch submit/make_eos.sh  --model CALVADOS --iter 10 --rho_i 0.05 --rho_f 1.4 --drho 0.1
sbatch submit/make_diff.sh --model CALVADOS --iter 10
```

---

### Stage 3 — Submit LAMMPS Simulations (manual)

Navigate to the generated EOS or DIFF directory and submit the universe files as
individual SLURM jobs. This step is cluster-specific and intentionally manual — see
§Between-Stage Decisions for guidance on resource allocation.

Typical simulation length:
- **EOS:** 100 ns per simulation
- **DIFF:** 150 ns per simulation

Dense or phase-separating sequences run significantly longer to equilibrate; plan
for batches taking over a week in realistic conditions.

---

### Stage 4 — Analyze Simulation Results (`analysis/`)

After LAMMPS simulations complete:

```bash
# EOS analysis (phase coexistence densities, Beff/exp_density)
submit/eos_calc.sh CALVADOS 500 10        # MODEL NBOOT ITER

# Diffusivity analysis
submit/diff_calc.sh CALVADOS 10           # MODEL ITER [INNER_JOBS] [OMP_THREADS] [NSEQ_JOBS]
```

(The `*_calc.sh` scripts are thin wrappers — they `sbatch` the actual SLURM job
script `process_{eos,diff}_sims.sh` and set per-(model, iter) `--job-name`,
`--output`, `--error` on the CLI.)

The submit wrappers call `process_eos_sims.sh` / `process_diff_sims.sh` via
`sbatch --job-name=... --output=... --error=...` (no `sed -i` mutation). Results
are written to `eos_results.csv` and `diff_results.csv` in the iteration directory.

Python analysis scripts:
- `analysis/process_eos_sims.py` — fits CubicSpline to density profiles, extracts
  coexistence densities, uses `bootstrap_exp_dens_from_path` for the effective second
  virial coefficient.
- `analysis/process_diff_sims.py` — computes MSD-based diffusivities from LAMMPS
  trajectories using the `md_calcs_par` compiled module (see §External Dependencies).

---

### Stage 5 — Repeat

Return to Stage 1 with `--iter N+1`. The `al_pipeline` master script reads the
accumulated observations from all previous iterations and fits a new surrogate model.

```bash
sbatch submit/al_master.sh --model CALVADOS --iter 11 --front upper
```

---

## Submit scripts (`submit/`)

All SLURM submission scripts live in a single top-level `submit/` directory.
Every script:
- Self-locates the repo via `${BASH_SOURCE}` (no hardcoded `${HOME}/...` paths).
- Sources `config/cluster.env` for module versions, env name, and project paths.
- Calls Python scripts via `${REPO_ROOT}/...` absolute paths — invariant to the
  shell's working directory at submission time.

| Script | Purpose |
|---|---|
| `submit/make_eos.sh` | Generate LAMMPS EOS input files for a (model, iter) batch |
| `submit/make_diff.sh` | Generate LAMMPS diffusivity input files |
| `submit/eos_calc.sh` | Thin wrapper that sbatches `process_eos_sims.sh` with parameterized job-name/output |
| `submit/process_eos_sims.sh` | The actual EOS analysis SLURM job |
| `submit/diff_calc.sh` | Wrapper for the diffusivity analysis |
| `submit/process_diff_sims.sh` | The actual diffusivity analysis SLURM job |
| `submit/al_master.sh` | **One sbatch per AL iteration** — production AL run |
| `submit/al_master_acq_test.sh` | Acquisition-function diagnostic (separate scratch path) |
| `submit/run_acq_sweep.sh` | Convenience wrapper to submit acq-test runs per `(model, ehvi, explore)` tuple |

### AL master examples

```bash
# Production run, iteration 0
sbatch submit/al_master.sh --model MPIPI --iter 0 --front upper

# Override exploration strategy
sbatch submit/al_master.sh --model MPIPI --iter 3 --front upper \
       --exploration_strategy similarity_penalty

# Disable the default --pessimism flag
sbatch submit/al_master.sh --model MPIPI --iter 0 --no-pessimism

# Pass-through power-user flags
sbatch submit/al_master.sh --model MPIPI --iter 0 -- --ref_point_mode in_line
```

### Acquisition-function sweep

```bash
# One (model, ehvi, explore) per invocation
./submit/run_acq_sweep.sh MPIPI epsilon kriging_believer

# Bash loop for full sweep
for m in MPIPI CALVADOS HPS_URRY; do
  for e in epsilon standard; do
    for x in kriging_believer similarity_penalty; do
      ./submit/run_acq_sweep.sh "$m" "$e" "$x"
    done
  done
done
```

Acq-test runs write to `${SCRATCH_AL_ACQ}` (a separate scratch root from
`${SCRATCH_AL}`) so production state is never touched.

---

## Between-Stage Decisions

Resource allocation between Stages 2/3 and 3/4 is intentionally manual. Before
submitting LAMMPS jobs (Stage 3), decide:

- **Core count per simulation:** Default 12 cores/sim for initial EOS runs. Dense or
  phase-separating sequences with slow equilibration benefit from 16–24 cores.
  Diffusivity simulations default to 16 cores/sim.
- **Density extensions:** If the EOS run does not reach the dilute-phase plateau
  (common for highly phase-separating sequences), extend the density range manually
  before resubmitting.
- **Selective rerun:** If a small fraction of simulations fail (LAMMPS crashes, node
  preemption), manually rerun those sequences rather than regenerating the full batch.

The `--extend_no_finished` and `--extra_steps` flags in the beam search scripts
(Stage 1 surrogate) follow a similar philosophy: extend only what is needed, preserve
everything else.

---

## Cluster Environment

All SLURM job scripts source `config/cluster.env` at the top of their body, after the
`#SBATCH` headers. This file centralizes all module names, environment names, and cluster
paths. Do not hardcode these values in individual scripts.

```bash
# config/cluster.env provides (among others):
CONDA_MODULE="anaconda3/2024.6"
CONDA_ENV="torch-chemistry"
OPENMPI_MODULE="openmpi/gcc/4.1.2"
HOME_AL="${HOME}/PROJECTS/MODEL_COMPARISON"
SCRATCH_AL="/scratch/gpfs/zl4808/PROJECTS/MODEL_COMPARISON"
DB_ROOT="${HOME}/scripts/GENDATA"
CORE_LIB="${HOME}/scripts/utility_scripts"
MD_CALCS="${HOME}/scripts/md_analysis/src"
```

The cluster is Princeton HPC Stellar (SLURM scheduler). Jobs are submitted from the
`beam_search/submit/` and `analysis/submit/` directories on the cluster.

---

## External Dependencies Not in This Repository

The following are required at runtime but live outside `al_active_dev/`:

| Dependency | Cluster location (`$VAR`) | Used by |
|------------|--------------------------|---------|
| `core.py` | `$CORE_LIB` (`/home/zl4808/scripts/utility_scripts/`) | `simulation/make_diff.py`, `make_eos.py`, `analysis/process_eos_sims.py` |
| `md_calcs_par` (compiled C extension) | `$MD_CALCS` (`/home/zl4808/scripts/md_analysis/src/`) | `analysis/process_diff_sims.py` |
| Sequence feature databases | `$DB_PATH` (`$DB_ROOT/databases`) | `al_pipeline` featurizer, `beam_search/` model loading |
| GPR checkpoint `.pt` files | `$SCRATCH_AL/$MODEL/MODELS/` | `beam_search/cross_paths/model_io.py` |
| Training CSVs | `$SCRATCH_AL/$MODEL/GENERATIONS/iteration_{N}/` | `beam_search/cross_paths/model_io.py` |

---

## Beam Search (`beam_search/`)

The `beam_search/` component implements surrogate-model-guided beam search over protein
sequence space. It is used to systematically explore the GP surrogate's predictions and
identify high-predicted-performance sequences to add to the simulation queue, independent
of the GA-based Stage 1 optimization.

The beam search runs in parallel across MPI ranks using `mpi4py`:

```bash
# Fresh run
bash beam_search/submit/submit_beams.sh CALVADOS 10 50 5 false

# Resume an interrupted run (preserves completed beams)
bash beam_search/submit/submit_resume_beams.sh CALVADOS 10 false

# Append phase-separating endpoints then resume
bash beam_search/submit/submit_phase_separated.sh CALVADOS 10 false
```

Key design decisions:
- **Default is resume-safe:** existing beam search results are never deleted by default.
  Pass `--clear_paths` explicitly to clear before a fresh run.
- **Stagnation termination:** use `--stagnation_patience N --stagnation_delta D` to
  stop beams that are not improving.
- **ALPaths integration:** all path resolution uses `ALPaths` from `al_pipeline.core.paths`;
  no hardcoded directory strings in the Python entry points.

`beam_search/cross_paths/` is a candidate for future absorption into `al_pipeline` as
`al_pipeline.beam_search`. The MPI dispatch in `run_beams_mpi.py` would likely remain as
a standalone cluster entry point even after that integration.

---

## Archive

`archive/` contains scripts that were active in the original `MODEL_COMPARISON_STELLAR/`
workspace but are superseded in this reorganization:

- **Component 1 standalone scripts** (`generate_features.py`, `train_gpr_multitask.py`,
  `ga_iterk_selection_testing.py`, etc.) — replaced by `al_pipeline`
- **Legacy `TRAINING/` and `UTILS/` modules** — replaced by `al_pipeline.training`,
  `al_pipeline.data_prep`, `al_pipeline.acquisition`
- **Old beam search versions** (`beam.py`, `beam_revised.py`, `io.py`, `io_test.py`,
  `run_beams_mpi_v2.py`, etc.) — superseded by `cross_paths/beam_search.py` and
  `cross_paths/model_io.py`
- **Profiling and testing harnesses** — `profile_beams.sh`, `parse_timing_logs.py`,
  `acq_testing.sh`, `featurizer_equivalence.py`
- **One-time initialization scripts** — `generate_init_data.sh`, `generate_init_gen.sh`
- **Monolithic predecessors** — `ActiveLearning_IterK.sh`, `ActiveLearning_IterK_Legacy.sh`

Nothing in `archive/` is imported or called by any active script.
