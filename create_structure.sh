#!/bin/bash

# Change to the diffusion_models_evaluation directory
# (assuming it already exists)
# cd diffusion_models_evaluation || { echo "Error: diffusion_models_evaluation directory not found."; exit 1; }

# Create config structure
mkdir -p config/downstream_tasks
touch config/models.yaml
touch config/metrics.yaml
touch config/evaluation.yaml
touch config/downstream_tasks/classification.yaml
touch config/downstream_tasks/segmentation.yaml
touch config/downstream_tasks/detection.yaml

# Create models structure
mkdir -p models/sd21_controlnet
mkdir -p models/sd35_controlnet
mkdir -p models/ldm

# Create scripts structure
mkdir -p scripts/preparation
mkdir -p scripts/inference
mkdir -p scripts/evaluation
mkdir -p scripts/downstream_tasks/classification
mkdir -p scripts/downstream_tasks/segmentation
mkdir -p scripts/downstream_tasks/detection

# Create workflow structure
mkdir -p workflow/snakemake/rules
touch workflow/snakemake/Snakefile
touch workflow/snakemake/rules/inference.smk
touch workflow/snakemake/rules/evaluation.smk
touch workflow/snakemake/rules/downstream.smk
mkdir -p workflow/airflow/dags

# Create utils directory
mkdir -p utils

echo "Directory structure for diffusion models evaluation created successfully!"