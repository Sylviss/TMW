# UCB Normal (PTB-DB) GAN Training

This notebook trains and evaluates a GAN on PTB-DB normal ECG data using
TMW and other time-series losses.

## How To Run

1. Open `train_and_test.ipynb`.
2. In the **CONFIGURATION** cell, update:
   - `TRAIN_CONFIG["data_path"]` (default: `../consolidated_datasets/pcb/ptbdb_normal.csv`)
   - `TRAIN_CONFIG` values (epochs, batch size, etc.)
   - `LOSSES_TO_RUN`
   - `LOSS_CONFIG_OVERRIDES` (per-loss training overrides)
   - `LOSS_HYPERPARAMS` (per-loss settings)
   - `EVAL_ONLY` (set True to skip training)
   - To change all hyperparameters, edit `TRAIN_CONFIG`, `LOSS_CONFIG_OVERRIDES`, and `LOSS_HYPERPARAMS` here.
3. Run all cells top to bottom.

## Hyperparameters (Defaults)

- `LOSSES_TO_RUN`: losses to train/evaluate (`["tmw", "opw", "taot", "sdtw", "awswd", "tcot", "pow", "gow"]`).
- `EVAL_ONLY`: skip training (`False`).
- `TRAIN_CONFIG`: core training settings (defaults below).

```python
TRAIN_CONFIG = {
   "epochs": 3000,
   "batch_size": 1024,
   "g_lr": 3e-4,
   "c_lr": 1e-3,
   "latent_dim": 100,
   "seq_len": 50,
   "channels": 1,
   "patch_size": 10,
   "n_critic": 3,
   "lambda_gp": 10,
   "lambda_ts": 0.05,
   "lambda_sm": 1,
   "device": "cuda" if torch.cuda.is_available() else "cpu",
   "save_interval": 50,
   "num_eval_samples": 1000,
   "data_path": "../consolidated_datasets/pcb/ptbdb_normal.csv",
   "output_dir": ".",
}
```

- `LOSS_CONFIG_OVERRIDES`: per-loss training overrides (defaults below).

```python
LOSS_CONFIG_OVERRIDES = {
   "opw": {"lambda_ts": 0.05},
   "taot": {"lambda_ts": 0.1},
   "tcot": {"lambda_ts": 0.1},
   "awswd": {"lambda_ts": 0.1},
   "tmw": {"lambda_ts": 0.5},
   "sdtw": {"lambda_ts": 0.5},
   "pow": {"lambda_ts": 0.5},
   "gow": {"lambda_ts": 0.5},
}
```

- `LOSS_HYPERPARAMS`: per-loss parameters (defaults below).

```python
LOSS_HYPERPARAMS = {
   "opw": {"lambda1": 20, "lambda2": 1.0, "sigma": 1.0, "num_iter": 20},
   "taot": {"reg_lambda": 1.0, "time_weight": 1.0, "num_iter": 100},
   "tcot": {"reg_lambda": 1.0, "num_iter": 100},
   "awswd": {"reg_lambda": 1.0, "l_window": 5, "k_steep": 0.1, "num_sinkhorn": 50, "num_outer": 5},
   "tmw": {"cost_function": "L2", "mask_type": 2, "reg": 1, "max_iterations": 50, "thres": 1e-5, "eps_threshold": 0.2, "masked": True, "rescale": True, "device": TRAIN_CONFIG["device"]},
   "sdtw": {"use_cuda": True, "gamma": 1.0, "normalize": True, "bandwidth": None},
   "pow": {"order_reg": 1.0, "sinkhorn_reg": 1, "m_mass": 0.8, "num_iter": 20},
   "gow": {"lambda1": 10.0, "lambda2": 1, "max_iter": 5, "sinkhorn_iter": 30, "fw_iter": 20},
   "none": {},
}
```

## Outputs

- Checkpoints are saved under `ckpt_*` folders (per loss).
- Metrics and logs are written to .txt files in this folder.

Note: For GAN tests, TMW checkpoints are saved in `ckpt_tmw_type_1` or
`ckpt_tmw_type_2`. Rename the folder to `ckpt_tmw` (remove the `_type_1` or
`_type_2` suffix) to use the checkpoint.
