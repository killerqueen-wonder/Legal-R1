#!/bin/bash

# Enable strict mode: exit immediately if a command exits with a non-zero status
set -e

# Define the directory containing environment files
ENV_DIR="env"

# Define the list of YAML configuration files to install
YML_FILES=(
    "Legal-R1.yml"
    "retriever_filter.yml"
    "SFT.yml"
    "vllm_server.yml"
)

echo "🚀 Starting the one-click installation of all Conda environments..."
echo "================================================="

# Iterate through the file list and execute installation
for yml in "${YML_FILES[@]}"; do
    file_path="${ENV_DIR}/${yml}"
    
    if [ -f "$file_path" ]; then
        echo "⏳ [Processing] Building environment from: $yml ..."
        # Create the environment using conda env create
        conda env create -f "$file_path"
        echo "✅ [Success] The environment for $yml is ready!"
        echo "-------------------------------------------------"
    else
        echo "❌ [Error] Configuration file not found: $file_path"
        echo "Please ensure you run this script from the project root directory, and the $ENV_DIR folder contains the file."
        exit 1
    fi
done

echo "🎉 Awesome! All 4 running environments have been successfully installed!"
echo "👉 Next, you can use 'conda activate <environment_name>' to start each service respectively."