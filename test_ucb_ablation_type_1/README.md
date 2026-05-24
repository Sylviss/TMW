# UCB Ablation (mask_type=1)

This notebook runs a TMW hyperparameter ablation on PTB-DB normal data with
`mask_type` base value set to 1.

## How To Run

1. Open `ablation_tmw.ipynb`.
2. In the **ABLATION STUDY CONFIGURATION** cell, update:
   - `BASE` (default values for `lambda_ts`, `mask_type`, `eps_threshold`)
   - `SWEEP` (values to sweep, one parameter at a time)
   - `BASE_PRETRAIN_EPOCHS`
   - `TRAIN_CONFIG` (including `data_path`)
   - `EVAL_ONLY` (set True to skip training)
3. Run all cells top to bottom.

## Hyperparameters (Defaults)

- `BASE`: default ablation point (`lambda_ts=0.5`, `mask_type=1`, `eps_threshold=0.2`).
- `BASE_PRETRAIN_EPOCHS`: warm-start epochs (`50`).
- `BASE_PRETRAIN_CONFIG`: pretrain settings (`lambda_ts=0.0`, `mask_type=1`, `eps_threshold=0.2`).
- `SWEEP`: sweep ranges (`lambda_ts=[0.05, 0.1]`, `mask_type=[1]`, `eps_threshold=[0.2]`).
- `EVAL_ONLY`: skip training (`False`).
- `TRAIN_CONFIG`: training settings (defaults below).

```python
TRAIN_CONFIG = {
   "epochs": 1000,
   "batch_size": 1024,
   "g_lr": 3e-4,
   "c_lr": 1e-3,
   "latent_dim": 100,
   "seq_len": 50,
   "channels": 1,
   "patch_size": 10,
   "n_critic": 3,
   "lambda_gp": 10,
   "lambda_sm": 1,
   "device": "cuda" if torch.cuda.is_available() else "cpu",
   "save_interval": 50,
   "num_eval_samples": 1000,
   "data_path": "../consolidated_datasets/pcb/ptbdb_normal.csv",
   "output_dir": "ablation_results",
}
```

- `TMW_FIXED_PARAMS`: fixed TMW settings (defaults below).

```python
TMW_FIXED_PARAMS = {
   "cost_function": "L2",
   "reg": 1,
   "max_iterations": 50,
   "thres": 1e-5,
   "masked": True,
   "rescale": True,
   "device": TRAIN_CONFIG["device"],
}
```

## Outputs

Results and checkpoints are saved under `ablation_results/`.
