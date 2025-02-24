#!/bin/bash
# init.sh
#
# This script sets up the 'activetigger' conda environment with Python 3.11,
# installs required packages, prepares a configuration file, and launches the server.
#
# Usage: ./setup_and_run.sh [-p PORT]
#   -p PORT : Optional flag to specify the port number (default is 5000)

# Default port
PORT=5000

# Ensure conda is available
if [ -z "$CONDA_EXE" ]; then
    # Try to source conda.sh from common installation locations
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        echo "Conda not found. Please install Anaconda or Miniconda."
        exit 1
    fi
fi

# Create the conda environment if it doesn't exist
if conda info --envs | awk '{print $1}' | grep -q '^activetigger$'; then
    echo "Conda environment 'activetigger' already exists. Activating..."
else
    echo "Creating conda environment 'activetigger' with Python 3.11..."
    conda create -n activetigger python=3.11 -y
fi

# Activate the conda environment
echo "Activating environment 'activetigger'..."
conda activate activetigger

# Install required Python packages
echo "Installing required packages from activetigger/api/requirements.txt..."
pip install -r activetigger/api/requirements.txt

# Check for a config.yaml file in the api directory
if [ ! -f activetigger/api/config.yaml ]; then
    if [ -f activetigger/api/config.yaml.sample ]; then
        echo "No config.yaml found. Copying config.yaml.sample to config.yaml..."
        cp activetigger/api/config.yaml.sample activetigger/api/config.yaml
        echo "You can now edit activetigger/api/config.yaml to modify paths for static files and database."
    else
        echo "No config.yaml.sample found in activetigger/api. Please ensure your configuration is set as needed."
    fi
fi

# Launch the server
echo "Launching the server on port $PORT..."
cd activetigger/api || { echo "Directory activetigger/api not found"; exit 1; }
python -m activetigger -p "$PORT"