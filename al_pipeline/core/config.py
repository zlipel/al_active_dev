from __future__ import annotations

from dataclasses import dataclass, fields, MISSING
from pathlib import Path
from typing import get_args, get_type_hints
from .paths import ALPaths, ensure_dirs

import argparse


@dataclass(frozen=True)
class ALConfig:
    #### Baseline configurations ####
    model: str
    iteration: int
    front: str

    #### Training configurations ####
    batch_size: int = 32
    epochs: int = 1000
    learning_rate: float = 0.1
    patience: int = 3
    save_best_fold: bool = False
    train_model_type: str = "gpr_multitask"
    k_folds: int = 5

    #### AL Strategy configurations ####
    ehvi_variant: str = "epsilon"
    epsilon_scale: float = 2.0
    exploration_strategy: str = "kriging_believer"
    pessimism: bool = False
    transform: str = "yeoj"

    #### Objectives for AL ####
    obj1: str = "exp_density"
    obj2: str = "diff"


    #### Auxiliary properties (from sims) ####
    aux1_obj1: str = "density"

    #### How many children + how many candidate GA runs ####
    seed_size: int = 120
    ngen: int = 24
    ncands: int = 96

    #### Paths ####
    # base_path  : HOME-side artifact root (.pt checkpoints, parity plots, logs).
    #              Defaults to runs/ inside the repo so artifacts ship with the
    #              project tree (runs/ is .gitignored). Override via --base_path
    #              or by editing config/cluster.env's HOME_AL.
    # scratch_path : per-iteration data (features/labels, sim outputs). Lives
    #                on the cluster's scratch filesystem.
    # db_path    : gendata databases used by simulation input generation.
    base_path: Path = Path("/home/zl4808/PROJECTS/al_active_dev/runs/")
    scratch_path: Path = Path("/scratch/gpfs/zl4808/PROJECTS/MODEL_COMPARISON/")
    db_path: Path = Path("/home/zl4808/scripts/GENDATA/databases/")

    #### Reproducibility / worker threading ####
    seed_base: int = 12345
    worker_threads: int = 1  # each GA worker must be single-threaded

    #### GA hyperparameters ####
    ga_max_iter: int = 200
    ga_mutation_prob: float = 0.5
    ga_crossover_prob: float = 0.5
    ga_deletion_prob: float = 0.2
    ga_growth_prob: float = 0.05
    ga_elit_ratio: float = 0.01
    ga_parents_portion: float = 0.3
    ga_max_no_improv: int = 50
    ga_Lmin: int = 20
    ga_Lmax: int = 160

    # EHVI choices
    ref_point_mode: str = "frac" # other options 'in_line', 'halfway'
    ref_point_frac: float = 0.5
    ref_point_tau: float = 0.05
    ref_point_cap: float = 0.5
    epsilon_scale: float = 2.0
    mc_ehvi: bool = False


    #### MC-EHVI tuning (used only when mc_ehvi=True) ####
    mc_min_samples: int = 64
    mc_max_samples: int = 512
    mc_chunk_size: int = 128
    mc_stderr_tol: float = 1e-4

    acq_test: bool = False

    def validate(self) -> None:
        """Validate the configuration parameters."""
        if self.iteration < 0:
            raise ValueError("Iteration number must be non-negative.")
        if self.ngen <= 0:
            raise ValueError("Number of generations must be positive.")
        if self.ncands <= 0:
            raise ValueError("Number of candidates must be positive.")    
        if self.obj1 == self.obj2:
            raise ValueError("Objectives must be different.")
        if self.iteration > 0:
            p = self.paths
            if not p.prev_features_csv.exists():
                raise FileNotFoundError(f"Missing {p.prev_features_csv}")
            if not p.prev_labels_csv.exists():
                raise FileNotFoundError(f"Missing {p.prev_labels_csv}")
            
        if self.train_model_type == "gpr_singletask" and (not self.obj1 or not self.obj2):
            raise ValueError("gpr_singletask requires obj1 and obj2")
        _ = self.paths.tag
        
    @property
    def paths(self) -> ALPaths:
        return ALPaths(
            base_path=self.base_path,
            scratch_path=self.scratch_path,
            iteration=self.iteration,
            front=self.front,
            model=self.model,
            ehvi_variant=self.ehvi_variant,
            exploration_strategy=self.exploration_strategy,
            transform=self.transform,
            mc_ehvi=self.mc_ehvi
        )
    
    
    def ensure(self) -> ALPaths:
        """Ensure that all necessary directories exist."""
        self.validate()
        p = self.paths
        ensure_dirs(p)
        return p
    

    @classmethod
    def from_cli(cls) -> "ALConfig":
        parser = argparse.ArgumentParser("Active Learning Configuration")

        # required fields (just the basics, the rest are more flexible)
        parser.add_argument("--model", required=True, type=str)
        parser.add_argument("--iter", dest="iteration", required=True, type=int)
        parser.add_argument("--front", required=True, choices=["upper", "lower"], type=str)
        parser.add_argument("--train_model_type", required=True, type=str, 
                            choices=['gpr_singletask', 
                                     'gpr_multitask',
                                     'dnn'])
        parser.add_argument("--ehvi_variant", required=True, type=str, 
                            choices=['standard', 
                                     'epsilon'])
        parser.add_argument("--exploration_strategy", required=True, type=str, 
                            choices=['standard', 
                                     'kriging_believer', 
                                     'similarity_penalty',
                                     'constant_liar_min',
                                     'constant_liar_mean',
                                     'constant_liar_max'])
        parser.add_argument("--transform", required=True, type=str, 
                            choices=['log', 
                                     'yeoj'])
        parser.add_argument("--obj1", required=True, type=str)
        parser.add_argument("--obj2", required=True, type=str)



        CLI_EXCLUDE = {
         "aux1_obj1",  
         "model",
            "iteration",
            "front",
            "train_model_type",
            "ehvi_variant",
            "exploration_strategy",
            "transform",
            "obj1",
            "obj2",
        }


        type_hints = get_type_hints(cls)

        # auto-add all other dataclass fields (ga params, other tuning params)
        for f in fields(cls):
            if f.name in CLI_EXCLUDE:
                continue

            default = f.default
            if default is MISSING:
                continue  # skip fields without default values for now

            arg = f"--{f.name}"

            t = type_hints.get(f.name, f.type) 

            # bool flags
            if t is bool:
                if default is False: # right now all bools default to False but just in case
                    parser.add_argument(arg, action="store_true", help=f"(default: {default})")
                else:
                    parser.add_argument(arg, action="store_false", help=f"(default: {default})")
                continue

            # Path
            if t is Path:
                parser.add_argument(arg, type=Path, default=default)
                continue

            # int/float/str
            if t in (int, float, str):
                parser.add_argument(arg, type=t, default=default)
                continue

            # fallback, just in case...usually we just enter some string
            parser.add_argument(arg, type=str, default=str(default))

        args = parser.parse_args()
        cfg = cls(**vars(args))
        cfg.validate()
        return cfg