# KNN Benchmark (Univariate UCR)

This notebook runs KNN classification with multiple distance functions over a
selected set of univariate UCR datasets.

## How To Run

1. Open `test_multi_distance.ipynb`.
2. In the **CONFIGURATION** cell, update:
   - `DATA_PATH` (default: `../consolidated_datasets/aeon_datasets`)
   - `SELECTED_DATASETS` (names of UCR datasets)
   - `DISTANCES_TO_RUN` and `DISTANCE_HYPERPARAMS`
   - `KNN_K_VALUES`
3. Run all cells. The main entry point is `run_multi_distance_knn()`.

## Hyperparameters (Defaults)

- `DATA_PATH`: dataset root (`../consolidated_datasets/aeon_datasets`).
- `RESULTS_FILE`: output CSV (`multi_distance_knn_results.csv`).
- `ERROR_LOG_FILE`: failure log (`multi_distance_failed_log.txt`).
- `SELECTED_DATASETS`: dataset list (default: BME, CBF, Chinatown, DodgerLoopDay, DodgerLoopWeekend, FreezerRegularTrain, FreezerSmallTrain, GesturePebbleZ1, GesturePebbleZ2, GunPointAgeSpan, GunPointMaleVersusFemale, Lightning7, MelbournePedestrian, PickupGestureWiimoteZ, ShakeGestureWiimoteZ, SmoothSubspace, ToeSegmentation1, ToeSegmentation2, Trace, UMD).
- `KNN_K_VALUES`: k values for KNN (`[1, 3, 5, 7, 9, 11, 13, 15]`).
- `DISTANCES_TO_RUN`: distances to evaluate (`["tmw", "opw", "taot", "tcot", "awswd", "otw", "dtw", "gow", "pow"]`).
- `DISTANCE_HYPERPARAMS`: per-distance overrides (defaults below).

```python
DISTANCE_HYPERPARAMS = {
   "opw": {"lambda1": 50.0, "lambda2": 0.1, "sigma": 1.0, "num_iter": 100},
   "taot": {"reg_lambda": 10.0, "time_weight": 10.0, "num_iter": 1000},
   "tcot": {"reg_lambda": 10.0, "num_iter": 1000},
   "awswd": {"reg_lambda": 10.0, "l_window": 5, "k_steep": 0.1, "num_sinkhorn": 100, "num_outer": 5},
   "tmw": {"cost_function": "L2", "mask_type": 2, "reg": 0.01, "max_iterations": 2000, "thres": 1e-5, "eps_threshold": 0.2, "masked": True, "rescale": True},
   "otw": {"m_cost": 1.0, "s_window": -1, "beta_smooth_l1": 1.0, "strategy_neg": "direct"},
   "dtw": {"global_constraint": None, "sakoe_chiba_radius": None, "itakura_max_slope": None},
   "gow": {"lambda1": 5.0, "lambda2": 10.0, "max_iter": 15, "sinkhorn_iter": 200, "fw_iter": 100},
   "pow": {"order_reg": 1.0, "sinkhorn_reg": 0.1, "m_mass": 0.8, "num_iter": 200},
   "euclidean": {}
}
```

## Outputs

- `multi_distance_knn_results.csv`: best-k accuracy and mAP per dataset/distance
- `multi_distance_failed_log.txt`: any failures or skipped runs

## Add More UCR/UEA Datasets

Add any dataset name to `SELECTED_DATASETS`. The notebook uses
`aeon.datasets.load_classification`, which will download the dataset (if needed)
into `consolidated_datasets/aeon_datasets`.

Tip: the full list of available UCR names is available as
`aeon.datasets.tsc_datasets.univariate` (imported as `ucr_names`).
