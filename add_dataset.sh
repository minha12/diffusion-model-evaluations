#!/bin/bash

# Function to add a new dataset structure
add_dataset() {
    local dataset_name=$1
    
    echo "Creating structure for dataset: ${dataset_name}"
    
    # Create data directories for this dataset
    mkdir -p "data/${dataset_name}/patches"
    mkdir -p "data/${dataset_name}/processed/pathology/patches"
    mkdir -p "data/${dataset_name}/processed/pathology/masks"
    mkdir -p "data/${dataset_name}/processed/pathology/annotations"
    mkdir -p "data/${dataset_name}/processed/pathology/metadata"
    mkdir -p "data/${dataset_name}/prompts"
    mkdir -p "data/${dataset_name}/segmentation"

    # Create results directories for this dataset
    mkdir -p "results/${dataset_name}/generated_images/sd21_controlnet"
    mkdir -p "results/${dataset_name}/generated_images/sd35_controlnet"
    mkdir -p "results/${dataset_name}/generated_images/ldm"
    mkdir -p "results/${dataset_name}/metrics/sd21_controlnet"
    mkdir -p "results/${dataset_name}/metrics/sd35_controlnet"
    mkdir -p "results/${dataset_name}/metrics/ldm"
    mkdir -p "results/${dataset_name}/downstream_tasks/classification/sd21_controlnet"
    mkdir -p "results/${dataset_name}/downstream_tasks/classification/sd35_controlnet"
    mkdir -p "results/${dataset_name}/downstream_tasks/classification/ldm"
    mkdir -p "results/${dataset_name}/downstream_tasks/segmentation/sd21_controlnet"
    mkdir -p "results/${dataset_name}/downstream_tasks/segmentation/sd35_controlnet"
    mkdir -p "results/${dataset_name}/downstream_tasks/segmentation/ldm"
    mkdir -p "results/${dataset_name}/downstream_tasks/detection/sd21_controlnet"
    mkdir -p "results/${dataset_name}/downstream_tasks/detection/sd35_controlnet"
    mkdir -p "results/${dataset_name}/downstream_tasks/detection/ldm"

    echo "✓ Dataset structure for '${dataset_name}' created successfully!"
}

# Check if dataset name is provided
if [ $# -eq 0 ]; then
    echo "Error: No dataset name provided."
    echo "Usage: $0 <dataset_name> [<dataset_name2> ...]"
    exit 1
fi

# Add structure for each dataset provided
for dataset in "$@"; do
    # Check if dataset already exists
    if [ -d "data/${dataset}" ] || [ -d "results/${dataset}" ]; then
        read -p "Dataset '${dataset}' appears to already exist. Overwrite? (y/n): " confirm
        if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
            add_dataset "$dataset"
        else
            echo "Skipping dataset '${dataset}'."
        fi
    else
        add_dataset "$dataset"
    fi
done

echo "All requested dataset structures have been created."