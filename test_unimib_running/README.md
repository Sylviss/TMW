# UniMiB-SHAR (Running) GAN Training

This notebook trains and evaluates a GAN on UniMiB-SHAR data filtered to the
`running` activity.

## How To Run

1. Open `train_and_test.ipynb`.
2. In the **CONFIGURATION** cell, update:
   - `TRAIN_CONFIG` (epochs, batch size, etc.)
   - `LOSSES_TO_RUN`, `LOSS_CONFIG_OVERRIDES`, and `LOSS_HYPERPARAMS`
   - `EVAL_ONLY` (set True to skip training)
   - To change all hyperparameters, edit `TRAIN_CONFIG`, `LOSS_CONFIG_OVERRIDES`, and `LOSS_HYPERPARAMS` here.
3. In the **DATASET** section, verify:
   - `data_path` (default: `../consolidated_datasets/UniMiB-SHAR/data/`)
   - `data_type` (default: `adl`)
   - Activity selection uses `clean_activity_names.index('running')`
     (change the activity name if needed)
4. Run all cells top to bottom.

## Hyperparameters (Defaults)

- `LOSSES_TO_RUN`: losses to train/evaluate (`["tmw", "opw", "taot", "sdtw", "awswd", "tcot", "pow", "gow"]`).
- `EVAL_ONLY`: skip training (`False`).
- `TRAIN_CONFIG`: core training settings (defaults below).

```python
TRAIN_CONFIG = {
   "epochs": 3000,
   "batch_size": 128,
   "g_lr": 3e-4,
   "c_lr": 1e-3,
   "latent_dim": 100,
   "seq_len": 150,
   "channels": 3,
   "patch_size": 30,
   "n_critic": 3,
   "lambda_gp": 10,
   "lambda_ts": 0.05,
   "lambda_sm": 1,
   "device": "cuda" if torch.cuda.is_available() else "cpu",
   "save_interval": 50,
   "num_eval_samples": 1000,
   "output_dir": ".",
}
```

- `LOSS_CONFIG_OVERRIDES`: per-loss training overrides (defaults below).

```python
LOSS_CONFIG_OVERRIDES = {
   "opw": {"lambda_ts": 0.1},
   "taot": {"lambda_ts": 0.1},
   "tcot": {"lambda_ts": 0.1},
   "awswd": {"lambda_ts": 0.1},
   "tmw": {"lambda_ts": 0.5},
   "sdtw": {"lambda_ts": 0.1},
   "pow": {"lambda_ts": 0.1},
   "gow": {"lambda_ts": 0.1},
}
```

- `LOSS_HYPERPARAMS`: per-loss parameters (defaults below).

```python
LOSS_HYPERPARAMS = {
   "opw": {"lambda1": 20, "lambda2": 1.0, "sigma": 1.0, "num_iter": 20},
   "taot": {"reg_lambda": 50.0, "time_weight": 1.0, "num_iter": 100},
   "tcot": {"reg_lambda": 10.0, "num_iter": 100},
   "awswd": {"reg_lambda": 50.0, "l_window": 5, "k_steep": 0.1, "num_sinkhorn": 50, "num_outer": 5},
   "tmw": {"cost_function": "L2", "mask_type": 2, "reg": 1, "max_iterations": 50, "thres": 1e-5, "eps_threshold": 0.1, "masked": True, "rescale": True, "device": TRAIN_CONFIG["device"]},
   "sdtw": {"use_cuda": True, "gamma": 0.1, "normalize": False, "bandwidth": None},
   "pow": {"order_reg": 1.0, "sinkhorn_reg": 0.1, "m_mass": 0.9, "num_iter": 20},
   "gow": {"lambda1": 5.0, "lambda2": 0.1, "max_iter": 5, "sinkhorn_iter": 20, "fw_iter": 10},
}
```

## Outputs

- Checkpoints are saved under `ckpt_*` folders (per loss).
- Result logs are written to .txt files in this folder.
