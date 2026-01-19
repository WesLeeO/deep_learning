## Repository layout

* `src/data`: synthetic problem generation plus datasets for blackboard and CoT
  representations (addition, subtraction, multi-addition, sampling helpers).  
* `src/models`: transformer definitions and positional encodings (1D, 2D, relative,
  rotary, and hybrids).  
* `src/training`: experiment scripts that train on blackboard or CoT data, including
  generalization, transfer, sample/weight efficiency, denoising, and mechanistic
  interpretability runs.  
* `models`: default output directory for saved checkpoints plots corresponding to Phase 0 of the paper

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Note: `requirements.txt` lists `torch`, `numpy`, `tqdm`, and `matplotlib`.

## Key experiments

The `src/training` folder contains scripts for the main experimental families:

* **Blackboard training**: `src/training/blackboard.py` trains blackboard-style
  transformers on stepwise arithmetic.  
* **Chain-of-Thought baseline**: `src/training/cot.py` trains 1D CoT transformers for
  comparison.  
* **Length/generalization studies**:
  `src/training/training_length_generalization.py`, and
  `src/training/training_local_generalization_all_pes.py`.  
* **Transfer/interpolation**: multi-addition grid transfer and subtraction transfer
  scripts under `src/training/` (e.g.,
  `multi_addition_grid_transfer_all_pes.py`, `multi_addition_grid_transfer_dif_heads.py`,
  `subtraction_transfer_all_pes.py`, `subtraction_transfer_dif_heads.py`).  
* **Efficiency studies**: `sample_efficiency_trainer.py` (for 2.4 A,C,D in the paper)
* **Denoising/backtracking variant**: `blackboard_denoising.py` implements the denoising
  objective described in the code comments. cluster_sweep_direction2.py for the full running code (all settings/PEs).
* **Mechanistic interpretability**: `mechanistic_interpretability.py` generates attention
  visualizations over blackboard grids and `error_distributions_abs_rel.py` plots the error distribution of abs and rel PEs

## Running a training script

Experiments are run as Python modules. Example commands:

```bash
python -m src.training.blackboard
python -m src.training.cot
python -m src.training.blackboard_denoising --setting local --denoise-rate 0.15 --p-revert 0.25
python -m src.training.cluster_sweep_direction2 
```

Adjust hyperparameters, dataset sizes, and output directories by editing the script or
passing CLI flags (where provided).
