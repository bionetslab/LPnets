# Longitudinal Patient Networks for Clinical Outcome Prediction

LPnets is a modular and reproducible pipeline for constructing single-sample networks (SSNs) from longitudinal clinical data and using them for predictive modelling. The framework extends network medicine approaches to irregular, sparse, and temporally structured clinical data, enabling graph-based representations of patient trajectories for downstream machine learning tasks.

The pipeline supports large-scale experimental evaluation across multiple datasets, network construction strategies, temporal aggregation methods, and machine learning models.

---

## Overview

LPnets transforms patient data into patient-specific graphs, where nodes correspond to clinical variables (e.g., laboratory measurements) and edges represent statistical dependencies (e.g., correlations or co-variation patterns). Each patient or time window is represented as a graph.

From these graphs, LPnets derives node-level features (e.g., degree centrality), edge-level features (e.g., high-variance or strong interactions), and graph-level features (e.g., global statistics and motifs). These representations are used for outcome prediction, either independently or in combination with original clinical features.

---

## Installation

Create the environment using conda:

conda env create -f environment.yml  
conda activate lpnets  

Alternatively:

pip install -r requirements.txt  

---

## Repository Structure

lpnets/  
config/              Experiment and model configurations  
datasets/            Dataset loaders and utilities  
edges/               Graph construction methods  
features/            Graph feature extraction  
ml_training/         Machine learning models and training logic  
pipeline/            Main execution pipeline  
preprocessing/       Clinical data preprocessing scripts  
__init__.py  

---

## Data Preprocessing

Supported datasets include MIMIC-III / MIMIC-IV ICU cohorts, PhysioNet challenge datasets, and pancancer omics datasets for benchmarking.

Preprocessing scripts:

python -m lpnets.preprocessing.preprocess_mimic_iii  
python -m lpnets.preprocessing.preprocess_physionet_2012  
python -m lpnets.preprocessing.preprocess_pancancer  

---

## Graph Construction

Graphs are constructed using configurable edge estimation methods, aggregation functions, and temporal strategies for longitudinal data.

Build all configurations:

python -m lpnets.pipeline.run_pipeline --mode build --build_all chemo_grid  

Build a specific configuration:

python -m lpnets.pipeline.run_pipeline --mode build --cohort mimic_cohort_aplasia_45_days --fold 0 --bin day --time_strategy TS1 --edge_method SSN --agg_method napyPCC  

---

## Model Training

LPnets supports training on original clinical features, graph-derived features, or their combination.

Full pipeline (build + train, grid search):

python -m lpnets.pipeline.run_pipeline --mode train --build_all chemo_grid --train_all train_grid  

Default training:

python -m lpnets.pipeline.run_pipeline --mode train --build_all chemo_grid  

Training with statistical filtering:

python -m lpnets.pipeline.run_pipeline --mode train --build_all chemo_grid --significant --zscores --zmode graphwise --threshold p_val  

Single configuration training:

python -m lpnets.pipeline.run_pipeline --mode train --cohort mimic_cohort_aplasia_45_days --fold 0 --bin day --time_strategy TS1 --edge_method SSN --agg_method PCC --zscores --zmode edgewise  

---

## Feature Extraction

Graph-derived features include node-level metrics (degree centrality), edge-level statistics (variance-based filtering), and graph-level descriptors (density, motifs, global measures).

Implemented in:
lpnets/features/feature_computer.py  
lpnets/features/graph_metric_computer.py  
lpnets/features/graph_filter.py  

---

## Machine Learning Models

Supported models include tree-based methods (Random Forest, Gradient Boosting), linear models, and standard machine learning pipelines.

Training utilities are implemented in:
lpnets/ml_training/  

---

## Pipeline Execution

Main entry point:

python -m lpnets.pipeline.run_pipeline  

Modes:
- build: construct graphs  
- train: train models  
- build + train: full pipeline execution  

---

## Configuration System

Experiment settings are defined in:
lpnets/config/  

Key files:
- chemo_grid.json: chemotherapy cohort experiments  
- omic_grid.json: omics benchmarks  
- train_grid.json: ML hyperparameters  
- model_config.json: model definitions  

