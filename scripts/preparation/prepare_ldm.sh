#!/bin/bash

# Get the output directory from the first argument
OUTPUT_DIR="$1"

# Create models directory if it doesn't exist
if [ ! -d models ]; then
    mkdir -p models
fi

# Clone or update the repository
if [ ! -d "$OUTPUT_DIR" ]; then
    git clone git@github.com:minha12/latent-diffusion-semantic.git "$OUTPUT_DIR"
else
    cd "$OUTPUT_DIR" && git pull
fi