# 3D Embedding Notebook

This folder contains a single notebook for visualizing original vs generated
samples with 3D embeddings (PCA or t-SNE).

Notebook: [single_3d_type_dataset_embedding.ipynb](single_3d_type_dataset_embedding.ipynb)

## How To Use

1. Open the notebook and run the import cell.
2. In the **User config** cell, update:
   - `DATASET_KEY` (UCB Normal, UCB Abnormal, UniMiB Running, UniMiB Jumping)
   - `GEN_TYPE` (`type_1` or `type_2`)
   - `EMBED_METHOD` (`pca` or `tsne`)
   - `NUM_SAMPLES`, `RANDOM_STATE`
   - `SAVE_FIG` (set True to save an HTML file)
3. Run all cells. The final cell calls `plot_single_3d_embedding(...)`.

## Outputs

- Interactive HTML is saved under `outputs/` when `SAVE_FIG=True`.
- The HTML file is named like `single_3d_<dataset>_<gen_type>_<method>.html`.
