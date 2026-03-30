# Cross-Species Transfer Learning for Electrophysiology-to-Transcriptomics Mapping in Cortical GABAergic Interneurons

**Theo Schwider, Ramin Ramezani**

This repository contains the analysis code accompanying:

> *Human to Mouse Cross-Species Transfer Learning for Electrophysiology-to-Transcriptomics Mapping in Cortical GABAergic Interneurons*

We replicate and extend the electrophysiology-to-transcriptomics framework of [Gouwens et al. (2020)](https://doi.org/10.1016/j.cell.2020.09.057) using publicly available Allen Institute Patch-seq datasets from mouse and human cortex. After reproducing major inhibitory subclass separations with engineered IPFX features and a sparse-PCA + random forest baseline, we develop an attention-based BiLSTM that operates directly on the 12-family IPFX sequence representation and evaluate cross-species transfer from mouse to human for an aligned 4-class inhibitory subclass task. The article is currently up at https://arxiv.org/abs/2603.11000 and submitted to MDPI Data.

---

## Datasets

Raw NWB files are downloaded directly inside `Feature_Extraction.ipynb` from the [DANDI Archive](https://dandiarchive.org/). No raw data is included in this repository.

| Species | DANDI ID | DOI | $N$ cells (post-QC) |
|---|---|---|---|
| Mouse (visual cortex) | [000020](https://dandiarchive.org/dandiset/000020) | [10.48324/dandi.000020/0.210913.1639](https://doi.org/10.48324/dandi.000020/0.210913.1639) | 3,699 |
| Human (neocortex, neurosurgical resection) | [000636](https://dandiarchive.org/dandiset/000636) | [10.48324/dandi.000636/0.241120.0510](https://doi.org/10.48324/dandi.000636/0.241120.0510) | 506 |

Human inhibitory subclass metadata is taken from the Allen Institute [`human_patchseq_gaba`](https://github.com/AllenInstitute/human_patchseq_gaba/tree/main/data) repository associated with [Lee et al. (2023)](https://doi.org/10.1126/science.adf6484). Place `LeeDalley_manuscript_metadata.csv` in the working directory before running feature extraction.

### Class distributions (post-QC)

| Mouse subclass | $N$ (%) | Human subclass | $N$ (%) |
|---|---|---|---|
| Sst | 1,663 (45.0%) | Pvalb | 293 (57.9%) |
| Pvalb | 745 (20.1%) | Sst | 96 (19.0%) |
| Vip | 691 (18.7%) | Vip | 67 (13.2%) |
| Lamp5 | 402 (10.9%) | Lamp5 | 50 (9.9%) |
| Sncg | 198 (5.4%) | — | — |

For cross-species experiments, labels are harmonized to a shared 4-class space (`Lamp5`, `Pvalb`, `Sst`, `Vip`), with mouse `Sncg` merged into `Vip` following [Hodge et al. (2019)](https://doi.org/10.1038/s41586-019-1506-7).

---

## Notebooks

### `Feature_Extraction.ipynb`
Downloads NWB files from DANDI, extracts the 12 Gouwens electrophysiological feature families per cell using [IPFX](https://github.com/AllenInstitute/ipfx), maps cells to coarse inhibitory t-type labels, and exports the sequence arrays used by all downstream notebooks.

**Steps:**
1. Install dependencies (allensdk, IPFX, DANDI, pynwb)
2. Download NWB files — `000636` for human, `000020` for mouse
3. Extract the 12 IPFX feature families per cell (see table below)
4. Map cells to t-type labels from accompanying metadata CSVs
5. Filter excluded t-types (`Serpinf1`, `Meis2`, `L2/3`)
6. Pad each feature family to a common width; stack into `(N, 12, D)` sequence arrays
7. Export to `.npy` / `.json`

**Outputs:**

| File | Shape | Description |
|---|---|---|
| `human_X_seq.npy` | `(N, 12, D)` | Human feature sequence array |
| `mouse_X_seq.npy` | `(N, 12, D)` | Mouse feature sequence array |
| `human_y_ttypes.json` | `(N,)` | Human t-type label list |
| `mouse_y_ttypes.json` | `(N,)` | Mouse t-type label list |

**12 IPFX feature families:**

| Family | Description |
|---|---|
| `first_ap_v` | Voltage features of the first AP (threshold, peak, AHP) |
| `first_ap_dv` | $dV/dt$ features around the first spike |
| `isi_shape` | Inter-spike interval metrics and adaptation indices |
| `inst_freq` | Instantaneous firing rate as a function of time and stimulus amplitude |
| `spiking_threshold_v` | Spike threshold estimates across sweeps |
| `spiking_peak_v` | Spike peak voltages across sweeps |
| `spiking_width` | Spike half-width and temporal width measures |
| `spiking_fast_trough_v` | Post-spike fast trough amplitudes |
| `spiking_upstroke_downstroke_ratio` | Max upstroke / downstroke derivative ratios |
| `step_subthresh` | Subthreshold voltage responses to step stimuli (sag, etc.) |
| `subthresh_norm` | Normalized subthreshold features (input resistance, time constant) |
| `psth` | Peristimulus time histogram summaries of spiking |

---

### `Classification.ipynb`
Loads the exported arrays and runs all supervised classification experiments. Models are evaluated across 10 independent runs; k-fold experiments use 5-fold stratified CV with an inner 20% validation split.

**Cell structure:**

| Cell(s) | Content |
|---|---|
| 1 | Data loading and sanity checks |
| 2 | Imports |
| 3 | Shared utilities — `set_seed`, `standardize_train_only`, `SeqDataset`, `inverse_freq_weights`, `collect_attn`, `attn_to_df`, `plot_attn_heatmap` |
| 4 | Core models — `BiLSTMAttentionClassifier`, `BiLSTMNoAttnClassifier` |
| 5 | Label coarsening and shared `LabelEncoder` (aligned 4-class across both species) |
| 6 | Single-dataset baselines — no-attn/no-SMOTE, attn/no-SMOTE, attn+SMOTE — 10 runs on mouse |
| 7 | Single-dataset ArcFace — ArcFace + CB-Focal loss + NAT prior correction — 10 runs on mouse |
| 8 | Human k-fold baselines — BiLSTM+attn ± SMOTE, 5-fold × 10 runs |
| 9 | Human ArcFace k-fold — ArcFace + CB-Focal + NAT prior, 5-fold × 10 runs |
| 10 | Normal transfer — mouse pretrain → encoder freeze → head warmup → full fine-tune on human |
| 11 | Joint MTL + fine-tune — shared encoder, two heads (human + mouse), joint then human-only fine-tune |
| 12–13 | Attention aggregation — `collect_attn` over human / mouse test sets |
| 14–15 | Attention heatmaps — paper-ready viridis heatmap, row-normalised, annotated |

---

### `Baseline_Visualizations.ipynb` *(forthcoming)*
Sparse PCA, UMAP embeddings, and random forest baseline. Reproduces Figures 1–2 and Tables 2–3 from the paper.

---

## Results

All metrics are mean ± SD. Mouse results use a stratified 60/20/20 hold-out split over 10 runs. Human results use 5-fold stratified CV repeated over 10 runs.

**Mouse BiLSTM variants:**

| Model | Macro-F1 | Accuracy |
|---|---|---|
| Baseline BiLSTM | 0.8601 ± 0.0162 | 0.9062 ± 0.0091 |
| BiLSTM + Attention | 0.8738 ± 0.0136 | 0.9145 ± 0.0088 |
| BiLSTM + Attention + SMOTE | 0.8856 ± 0.0214 | 0.9193 ± 0.0128 |
| ArcFace BiLSTM + Attention + SMOTE | 0.8869 ± 0.0186 | 0.9212 ± 0.0092 |

**Human BiLSTM variants:**

| Model | Macro-F1 | Accuracy |
|---|---|---|
| BiLSTM + Attention | 0.6685 ± 0.0141 | 0.7798 ± 0.0080 |
| BiLSTM + Attention + SMOTE | 0.6754 ± 0.0180 | 0.7822 ± 0.0145 |
| ArcFace BiLSTM + Attention + SMOTE | 0.6729 ± 0.0195 | 0.7818 ± 0.0145 |

**Transfer learning (aligned 4-class, 5-fold CV):**

| Model | Macro-F1 | Accuracy |
|---|---|---|
| Human-only baseline BiLSTM | 0.6580 ± 0.0286 | 0.7710 ± 0.0172 |
| Joint MTL + human fine-tune | 0.6795 ± 0.0120 | 0.7905 ± 0.0096 |

---

## Setup

```bash
git clone https://github.com/tschwider09/cross-species-ephys-transfer.git
cd cross-species-ephys-transfer
pip install torch scikit-learn pandas matplotlib imbalanced-learn
```

For `Feature_Extraction.ipynb` only:
```bash
pip install allensdk --no-deps
pip install requests h5py pandas matplotlib scipy "numpy>=1.21,<2.0"
pip install SimpleITK methodtools pynwb dandi ipfx
```

Both notebooks are designed to run in Google Colab. The classification notebook runs on CPU but benefits substantially from a GPU runtime.

---

## Repository structure

```
.
├── Feature_Extraction.ipynb        # DANDI download → IPFX extraction → export arrays
├── Classification.ipynb            # All BiLSTM experiments and attention figures
├── Baseline_Visualizations.ipynb   # sPCA, UMAP, random forest (forthcoming)
└── README.md
```

---

## Citation

If you use this code or build on this pipeline, please also cite the datasets and tools it depends on:

- Gouwens et al. (2020). *Integrated morphoelectric and transcriptomic classification of cortical GABAergic cells.* Cell, 183(4), 935–953. https://doi.org/10.1016/j.cell.2020.09.057
- Lee et al. (2023). *Signature morphoelectric properties of diverse GABAergic interneurons in the human neocortex.* Science, 382(6667), eadf6484. https://doi.org/10.1126/science.adf6484
- Allen Institute for Brain Science. IPFX v2.1.2. https://github.com/AllenInstitute/ipfx
- Mouse Patch-seq dataset: https://doi.org/10.48324/dandi.000020/0.210913.1639
- Human Patch-seq dataset: https://doi.org/10.48324/dandi.000636/0.241120.0510

---

## Acknowledgements

The authors thank the Allen Institute for Brain Science for open data releases and documentation resources that made this analysis possible.
