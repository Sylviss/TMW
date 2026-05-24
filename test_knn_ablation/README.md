# KNN Ablation (TMW eps_threshold and reg)

This notebook runs two ablation studies on a UCR dataset using KNN with TMW
precomputed distances:

1. Sweep `eps_threshold` for `mask_type` 1 and 2.
2. Sweep `reg` at a fixed `eps_threshold`.

## How To Run

1. Open `ablation_eps_threshold.ipynb`.
2. In the **CONFIGURATION** cell, update:
   - `DATASET_NAME` (default: `ToeSegmentation2`)
   - `EPS_THRESHOLD_VALUES`, `MASK_TYPES_FOR_EPS_ABLATION`
   - `REG_VALUES`, `MASK_TYPES_FOR_REG_ABLATION`
   - `KNN_K_VALUES`
3. Run all cells top to bottom.

## Hyperparameters (Defaults)

- `DATA_PATH`: dataset root (`../consolidated_datasets/aeon_datasets`).
- `RESULTS_DIR`: output directory (`ablation_results`).
- `DATASET_NAME`: dataset for ablation (`ToeSegmentation2`).
- `KNN_K_VALUES`: k values (`[1, 3, 5, 7, 9, 11, 13, 15]`).
- `EPS_THRESHOLD_DEFAULT`: baseline eps threshold (`0.4`).
- `REG_DEFAULT`: baseline regularization (`0.01`).
- `EPS_THRESHOLD_VALUES`: sweep values (`[0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]`).
- `MASK_TYPES_FOR_EPS_ABLATION`: mask types for eps sweep (`[1, 2]`).
- `REG_VALUES`: sweep values (`[0.0001, 0.001, 0.01, 0.1, 1, 10]`).
- `MASK_TYPES_FOR_REG_ABLATION`: mask types for reg sweep (`[1, 2]`).
- `TMW_FIXED_PARAMS`: fixed TMW settings (defaults below).

```python
TMW_FIXED_PARAMS = {
   "cost_function": "L2",
   "mask_type": 1,
   "reg": 0.01,
   "max_iterations": 2000,
   "thres": 1e-5,
   "masked": True,
   "rescale": True,
}
```

## Outputs

Results are saved under `ablation_results/`:

- `eps_threshold_ablation_results.csv`
- `regularization_ablation_results.csv`
- `*_errors.txt` for any failures
