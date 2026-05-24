# KNN Benchmark (Multivariate UEA + Local)

This notebook benchmarks KNN classification using multiple distances on
multivariate datasets. It supports UEA datasets (via aeon) and local datasets
(Weizmann, SpokenArabicDigits).

## How To Run

1. Open `test_multi_distances.ipynb`.
2. In the **CONFIGURATION** cell, update:
   - `DATA_PATH_UEA` (default: `../consolidated_datasets/aeon_datasets`)
   - `LOCAL_DATASETS` and `UEA_DATASETS`
   - `DISTANCES_TO_RUN` and `DISTANCE_HYPERPARAMS`
   - `KNN_K_VALUES`
3. Run all cells. The main entry point is `run_multi_dataset_benchmark()`.

## Hyperparameters (Defaults)

- `DATA_PATH_UEA`: dataset root (`../consolidated_datasets/aeon_datasets`).
- `RESULTS_FILE`: output CSV (`multivariate_knn_results.csv`).
- `ERROR_LOG_FILE`: failure log (`multivariate_failed_log.txt`).
- `LOCAL_DATASETS`: local datasets (`["Weizmann"]`).
- `UEA_DATASETS`: UEA datasets (`["BasicMotions", "CharacterTrajectories"]`).
- `KNN_K_VALUES`: k values for KNN (`[1, 3, 5, 7, 9, 11, 13, 15]`).
- `DISTANCES_TO_RUN`: distances to evaluate (`["tmw", "tcot", "taot", "pow", "opw", "gow", "dtw", "awswd"]`).
- `DISTANCE_HYPERPARAMS`: per-distance overrides (defaults below).

```python
DISTANCE_HYPERPARAMS = {
   "opw": {"lambda1": 50.0, "lambda2": 0.1, "sigma": 1.0, "num_iter": 100},
   "taot": {"reg_lambda": 10.0, "time_weight": 10.0, "num_iter": 1000},
   "tcot": {"reg_lambda": 10.0, "num_iter": 1000},
   "awswd": {"reg_lambda": 10.0, "l_window": 5, "k_steep": 0.1, "num_sinkhorn": 100, "num_outer": 5},
   "tmw": {"cost_function": "L2", "mask_type": 2, "reg": 0.01, "max_iterations": 500, "thres": 1e-5, "eps_threshold": 0.2, "masked": True, "rescale": True},
   "dtw": {"global_constraint": None, "sakoe_chiba_radius": None, "itakura_max_slope": None},
   "gow": {"lambda1": 5.0, "lambda2": 0.1, "max_iter": 20, "sinkhorn_iter": 10, "fw_iter": 10},
   "pow": {"order_reg": 1.0, "sinkhorn_reg": 0.05, "m_mass": 0.8, "num_iter": 200}
}
```

## Outputs

- `multivariate_knn_results.csv`: best-k accuracy and mAP per dataset/distance
- `multivariate_failed_log.txt`: any failures or skipped runs

## Add More UEA Datasets

Add any UEA dataset name to `UEA_DATASETS`. The notebook uses
`aeon.datasets.load_classification` and will download data into
`consolidated_datasets/aeon_datasets` as needed.

Tip: the full list of UEA names is available as
`aeon.datasets.tsc_datasets.multivariate` (imported as `uea_names`).

## Local Dataset Notes

- **Weizmann**: expects files under `../consolidated_datasets/wei_dataset_feature/binary`.
- **SpokenArabicDigits**: expects files under `../consolidated_datasets/spoken_arabic`.
