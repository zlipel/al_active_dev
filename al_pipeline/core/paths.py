from __future__ import annotations

from pathlib import Path
from typing import List
from dataclasses import dataclass


def _tag(ehvi_variant: str, 
         exploration_strategy: str, 
         transform: str,
         front: str,
         mc_ehvi: bool) -> str:
    """
    Generate a reusable tag depending on the chosen strategy.
    Parameters:
    ----------
    ehvi_variant : str
        The EHVI variant used (e.g., 'standard' or 'epsilon').
    exploration_strategy : str
        The exploration strategy used (e.g., 'standard', 'similarity_penalty', 'kriging_believer', etc.).
    transform : str
        The transformation applied to the labels (e.g., 'log' or 'yeoj').
    mc_ehvi : bool
        Whether MC EHVI is used (if not, we use the analytic 2d formula).
    Returns:
    -------
    str
        The generated tag.
    """
    tag = f"{ehvi_variant}_{exploration_strategy}_{transform}_{front}"
    if mc_ehvi:
        tag += "_mc"
    return tag

#### Potential TODO: make the paths more modular, right now it's EOS and DIFF centric, but in future may have other objectives.  ####
@dataclass(frozen=True)
class ALPaths:
    #### Path configurations ####
    base_path:            Path # for me, this is /home/zl4808/PROJECTS/MODEL_COMPARISON/
    scratch_path:         Path # e.g., /scratch/gpfs/zl4808/PROJECTS/MODEL_COMPARISON/
    #iter_dir:             Path 
    #log_dir:              Path
    iteration:            int  # current iteration number
    front:                str  # 'upper' or 'lower'
    model:                str  # e.g., HPS_URRY, MPIPI, CALVADOS, etc.

    #### AL Strategy configurations ####
    ehvi_variant:         str  = 'epsilon' # default is epsilon
    exploration_strategy: str  = 'kriging_believer' # default is epsilon
    transform:            str  = 'yeoj' # default is Yeo-Johnson Transform
    mc_ehvi:              bool = False # default is analytic

    @property
    def tag(self) -> str:
        return _tag(self.ehvi_variant, self.exploration_strategy, self.transform, self.front, self.mc_ehvi)
    
    #### model directories ####
    @property
    def model_home_dir(self) -> Path:
        return self.base_path / self.model
    
    @property
    def model_scratch_dir(self) -> Path:
        return self.scratch_path / self.model
    
    #### iteration directories ####
    @property
    def iter_scratch_dir(self) -> Path:
        return self.model_scratch_dir / "GENERATIONS" / f"iteration_{self.iteration}"
    
    @property
    def iter_front_dir(self) -> Path:
        return self.model_scratch_dir / f"{self.front}" / f"iteration_{self.iteration}"
    
    #### Simulaton directories ####
    @property
    def sim_dir(self) -> Path:
        return self.iter_scratch_dir / "SIMULATIONS"
    
    @property
    def eos_dir(self) -> Path:
        return self.sim_dir / "EOS"
    
    @property
    def eos_csv(self) -> Path:
        return self.eos_dir / f"eos_results.csv"

    @property
    def diff_dir(self) -> Path:
        return self.sim_dir / "DIFF"
    
    @property
    def diff_csv(self) -> Path:
        return self.diff_dir / f"diffusivities.csv"
    
    #### home outputs ####
    @property
    def models_dir(self) -> Path:
        return self.model_home_dir / "MODELS"
    
    @property
    def logs_dir(self) -> Path:
        return self.model_home_dir / "logs" / f"iteration_{self.front}_{self.iteration}"
    
    #### data files####
    @property
    def features_csv(self) -> Path:
        return self.iter_scratch_dir / f"features_gen{self.iteration}.csv"
    
    @property
    def labels_csv(self) -> Path:
        return self.iter_scratch_dir / f"labels_gen{self.iteration}.csv"
    
    @property
    def seq_gen_txt(self) -> Path:
        return self.iter_scratch_dir / f"seq_gen{self.iteration}.txt"
    
    #### normalized and tagged instances ####
    @property
    def features_norm_csv(self) -> Path:
        return self.iter_scratch_dir / f"features_gen{self.iteration}_NORM_{self.tag}.csv"
    
    @property
    def labels_norm_csv(self) -> Path:
        return self.iter_scratch_dir / f"labels_gen{self.iteration}_NORM_{self.tag}.csv"
    
    @property
    def seq_gen_temp_txt(self) -> Path:
        return self.iter_scratch_dir / f"seq_gen{self.iteration}_TEMP_{self.tag}.txt"
    
    @property
    def norm_stats(self) -> Path:
        return self.iter_scratch_dir / f"normalization_stats.json"
    
    @property
    def prev_iter_scratch_dir(self) -> Path:
        if self.iteration <= 0:
            raise ValueError("iteration=0 has no previous iter")
        return self.model_scratch_dir / "GENERATIONS" / f"iteration_{self.iteration - 1}"

    @property
    def prev_features_csv(self) -> Path:
        return self.prev_iter_scratch_dir / f"features_gen{self.iteration - 1}.csv"

    @property
    def prev_labels_csv(self) -> Path:
        return self.prev_iter_scratch_dir / f"labels_gen{self.iteration - 1}.csv"
    
    #### special files### 
    @property
    def labels_no_pessimism(self) -> Path:
        return self.iter_scratch_dir / f"labels_gen{self.iteration}_NORM_NO_PESS_{self.tag}.csv"
    
    #### GA files ####
    @property
    def ga_children_dir(self) -> Path:
        return self.iter_front_dir / f"children_{self.tag}"
    
    @property
    def ga_candidates_dir(self) -> Path:
        return self.iter_front_dir / f"candidates_{self.tag}"
    
    @property
    def parent_seqs_txt(self) -> Path:
        return self.iter_front_dir / f"sequences_parent_{self.tag}.txt"
    
    @property
    def parent_features_csv(self) -> Path:
        return self.iter_front_dir / f"features_parent_{self.tag}.csv"
    
    @property
    def parent_labels_csv(self) -> Path:
        return self.iter_front_dir / f"labels_parent_{self.tag}.csv"
    
    @property
    def parent_features_norm_csv(self) -> Path:
        return self.iter_front_dir / f"features_parent_NORM_{self.tag}.csv"
    
    @property
    def parent_labels_norm_csv(self) -> Path:
        return self.iter_front_dir / f"labels_parent_NORM_{self.tag}.csv"
    
    @property
    def parent_seqs_temp_txt(self) -> Path:
        return self.iter_front_dir / f"sequences_parent_TEMP_{self.tag}.txt"
    
    #### ML Model chkpt paths ####
    def gpr_multitask_chkpt(self, temp: bool) -> Path:
        fname =f"GPR_iter{self.iteration}_{self.tag}_TEMP.pt" if temp else f"GPR_iter{self.iteration}_{self.tag}.pt"
        return self.models_dir / fname
    
    
    def gpr_singletask_chkpt(self, label_name: List, temp: bool) -> List[Path]:
        fnames =[f"GPR_iter{self.iteration}_{label}_{self.tag}_TEMP.pt" if temp else f"GPR_iter{self.iteration}_{label}_{self.tag}.pt" for label in label_name]
        return [self.models_dir / fname for fname in fnames]

    #### MoE checkpoint paths (mirror gpr_multitask shape per expert + the RF bundle).
    # No "all" expert — global GPR comparisons use `gpr_multitask_chkpt`.
    def moe_ps_chkpt(self, temp: bool) -> Path:
        suffix = "_TEMP" if temp else ""
        return self.models_dir / f"MOE_PS_iter{self.iteration}_{self.tag}{suffix}.pt"

    def moe_nonps_chkpt(self, temp: bool) -> Path:
        suffix = "_TEMP" if temp else ""
        return self.models_dir / f"MOE_NONPS_iter{self.iteration}_{self.tag}{suffix}.pt"

    def moe_rf_bundle(self, temp: bool) -> Path:
        suffix = "_TEMP" if temp else ""
        return self.models_dir / f"MOE_RF_iter{self.iteration}_{self.tag}{suffix}.pkl"

    #### Diagnostic outputs (retrospective analyses, plots, etc.) ####
    # Lives under model_home_dir so it ships with the AL artifacts, not on scratch.
    @property
    def diagnostic_dir(self) -> Path:
        return self.model_home_dir / "DIAGNOSTIC"

    #### TODO: implement DNN ensemble paths later ####

    #### next simulation directory ####
    @property
    def next_iter_scratch_dir(self) -> Path:
        nxt = self.iteration + 1
        return self.model_scratch_dir / f"GENERATIONS" / f"iteration_{nxt}"
    
    @property
    def next_iter_scratch_sim_dir(self) -> Path:
        nxt = self.iteration + 1
        return self.next_iter_scratch_dir / "SIMULATIONS"
    
    @property
    def next_iter_candidates_file(self) -> Path:
        nxt = self.iteration + 1
        return self.next_iter_scratch_sim_dir / f"simulation_candidates_gen{nxt}_{self.front}.txt"
    
    @property
    def next_iter_eos_dir(self) -> Path:
        return self.next_iter_scratch_sim_dir / "EOS"
    
    @property
    def next_iter_diff_dir(self) -> Path:
        return self.next_iter_scratch_sim_dir / "DIFF"
    
def ensure_dirs(p: ALPaths) -> None:
    """
    Ensure that all necessary directories exist.
    Parameters:
    ----------
    p : ALPaths
        The ALPaths instance containing the paths to make sure exist.
    """

    p.model_home_dir.mkdir(parents=True, exist_ok=True)
    p.model_scratch_dir.mkdir(parents=True, exist_ok=True)
    p.iter_scratch_dir.mkdir(parents=True, exist_ok=True)
    p.iter_front_dir.mkdir(parents=True, exist_ok=True)
    p.sim_dir.mkdir(parents=True, exist_ok=True)
    p.eos_dir.mkdir(parents=True, exist_ok=True)
    p.diff_dir.mkdir(parents=True, exist_ok=True)
    p.models_dir.mkdir(parents=True, exist_ok=True)
    p.logs_dir.mkdir(parents=True, exist_ok=True)
    p.ga_children_dir.mkdir(parents=True, exist_ok=True)
    p.ga_candidates_dir.mkdir(parents=True, exist_ok=True)
    p.next_iter_scratch_dir.mkdir(parents=True, exist_ok=True)
    p.next_iter_scratch_sim_dir.mkdir(parents=True, exist_ok=True)
    p.next_iter_eos_dir.mkdir(parents=True, exist_ok=True)
    p.next_iter_diff_dir.mkdir(parents=True, exist_ok=True)

    if p.iteration > 0:
        p.prev_iter_scratch_dir.mkdir(parents=True, exist_ok=True)


