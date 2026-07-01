# LPnets: Longitudinal Patient Networks for Clinical Outcome Prediction

LPnets is a modular, reproducible pipeline for constructing single-sample networks (SSNs) from longitudinal clinical data and using them for predictive modelling. The framework extends network medicine approaches to irregular, sparse, and temporally structured clinical data, enabling graph-based representations of patient trajectories for downstream machine learning tasks.

The pipeline supports large-scale experimental evaluation across multiple datasets, network construction strategies, temporal aggregation methods, and machine learning models.

## Overview

LPnets transforms patient data into patient-specific graphs, where:

- **Nodes** correspond to clinical variables (e.g., laboratory measurements)
- **Edges** represent statistical dependencies (e.g., correlations or co-variation patterns)

Each patient, or each time window within a patient's record, is represented as its own graph.

From these graphs, LPnets derives:

| Feature type | Examples |
|---|---|
| Node-level | Degree centrality |
| Edge-level | High-variance or strong interactions |
| Graph-level | Global statistics and motifs |

These representations can be used for outcome prediction either independently or in combination with original clinical features.

## Installation

**Requirements:** Python 3.9+

```bash
conda env create -f environment.yml
conda activate lpnets
```

## Repository Structure

```
lpnets/
├── config/            Experiment and model configurations
├── datasets/          Dataset loaders and utilities
├── edges/             Graph construction methods
├── features/          Graph feature extraction
├── ml_training/       Machine learning models and training logic
├── pipeline/          Main execution pipeline
├── preprocessing/     Clinical data preprocessing scripts
└── __init__.py
```

## Data Preprocessing

Supported datasets:

- MIMIC-III / MIMIC-IV ICU cohorts
- PhysioNet challenge datasets
- Pan-cancer omics datasets (for benchmarking)

Run the relevant preprocessing script before building graphs:

```bash
python -m lpnets.preprocessing.preprocess_mimic_iii
python -m lpnets.preprocessing.preprocess_physionet_2012
python -m lpnets.preprocessing.preprocess_pancancer
```

> Each script expects raw source files to already be downloaded locally; see the corresponding script's `--help` output for expected input paths.

## Graph Construction

Graphs are constructed using configurable edge estimation methods, aggregation functions, and temporal strategies for longitudinal data.

**Build all configurations in a grid:**

```bash
python -m lpnets.pipeline.run_pipeline --mode build --build_all chemo_grid
```

**Build a single, specific configuration:**

```bash
python -m lpnets.pipeline.run_pipeline \
  --mode build \
  --cohort mimic_cohort_aplasia_45_days \
  --fold 0 \
  --bin day \
  --time_strategy TS1 \
  --edge_method SSN \
  --agg_method napyPCC
```

### Key arguments

| Argument | Description |
|---|---|
| `--cohort` | Name of the patient cohort to build graphs for |
| `--fold` | Cross-validation fold index |
| `--bin` | Temporal binning resolution (e.g., `day`) |
| `--time_strategy` | Temporal aggregation strategy (e.g., `TS1`) |
| `--edge_method` | Edge estimation method (e.g., `SSN`) |
| `--agg_method` | Correlation/aggregation method used to compute edges (e.g., `PCC`, `napyPCC`) |

## Model Training

LPnets supports training on original clinical features, graph-derived features, or a combination of both.

**Full pipeline (build + train, grid search):**

```bash
python -m lpnets.pipeline.run_pipeline --mode train --build_all chemo_grid --train_all train_grid
```

**Default training (uses default hyperparameters):**

```bash
python -m lpnets.pipeline.run_pipeline --mode train --build_all chemo_grid
```

**Training with statistical filtering (z-score/significance thresholding):**

```bash
python -m lpnets.pipeline.run_pipeline \
  --mode train \
  --build_all chemo_grid \
  --significant \
  --zscores \
  --zmode graphwise \
  --threshold p_val
```

**Single configuration training:**

```bash
python -m lpnets.pipeline.run_pipeline \
  --mode train \
  --cohort mimic_cohort_aplasia_45_days \
  --fold 0 \
  --bin day \
  --time_strategy TS1 \
  --edge_method SSN \
  --agg_method PCC \
  --zscores \
  --zmode edgewise
```

### Key arguments

| Argument | Description |
|---|---|
| `--significant` | Restrict training to statistically significant features |
| `--zscores` | Enable z-score normalisation of features |
| `--zmode` | Scope of z-score computation: `graphwise` or `edgewise` |
| `--threshold` | Metric used for significance filtering (e.g., `p_val`) |

## Feature Extraction

Graph-derived features fall into three categories:

- **Node-level metrics** — e.g., degree centrality
- **Edge-level statistics** — e.g., variance-based filtering
- **Graph-level descriptors** — e.g., density, motifs, global measures

## Machine Learning Models

Supported model families:

- Tree-based methods (Random Forest, Gradient Boosting)
- Linear models
- Standard scikit-learn-style pipelines

Training utilities are implemented in `lpnets/ml_training/`.

## Pipeline Execution

Main entry point:

```bash
python -m lpnets.pipeline.run_pipeline
```

**Modes:**

| Mode | Description |
|---|---|
| `build` | Construct graphs only |
| `train` | Train models only (requires previously built graphs) |
| `build` + `train` | Full pipeline: build graphs, then train models |

Run `python -m lpnets.pipeline.run_pipeline --help` for the complete list of available arguments.

## Configuration System

Experiment settings are defined as JSON files under `lpnets/config/`.

| File | Purpose |
|---|---|
| `chemo_grid.json` | Chemotherapy cohort experiment definitions |
| `omic_grid.json` | Omics benchmark experiment definitions |
| `train_grid.json` | ML hyperparameter grids |
| `model_config.json` | Model definitions |

