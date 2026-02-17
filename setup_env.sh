#!/bin/bash

# Physics Simulation Research - Environment Setup Script
# This script helps you set up the development environment for different research directions

set -e  # Exit on error

echo "=================================="
echo "Physics Simulation Research Setup"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    print_warning "Conda not found. Please install Anaconda or Miniconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

print_success "Conda found"

# Prompt for environment name
read -p "Enter environment name (default: physics-sim): " ENV_NAME
ENV_NAME=${ENV_NAME:-physics-sim}

# Check if environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    print_warning "Environment '$ENV_NAME' already exists."
    read -p "Do you want to remove and recreate it? (y/N): " RECREATE
    if [[ $RECREATE =~ ^[Yy]$ ]]; then
        print_warning "Removing existing environment..."
        conda env remove -n $ENV_NAME -y
    else
        print_error "Setup cancelled."
        exit 0
    fi
fi

# Create conda environment
echo ""
echo "Creating conda environment '$ENV_NAME' with Python 3.10..."
conda create -n $ENV_NAME python=3.10 -y
print_success "Environment created"

# Activate environment
echo ""
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME
print_success "Environment activated"

# Install base dependencies
echo ""
echo "Installing base dependencies..."
pip install -r requirements.txt
print_success "Base dependencies installed"

# Ask user which research directions they want to set up
echo ""
echo "Which research directions do you want to set up?"
echo "You can enter multiple numbers separated by spaces (e.g., '1 2 4')"
echo ""
echo "1) Differentiable Physics (Brax, MuJoCo, Taichi)"
echo "2) Physics Foundation Models (The Well data tools)"
echo "3) GNN Particle Simulation (PyTorch Geometric)"
echo "4) World Models (DreamerV3 dependencies)"
echo "5) PINNs (DeepXDE)"
echo "6) Multimodal Fusion (PyTorch vision tools)"
echo "7) All of the above"
echo ""
read -p "Enter your choice(s): " CHOICES

# Function to install PyTorch
install_pytorch() {
    echo ""
    echo "Detecting CUDA availability..."
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2)
        print_success "CUDA $CUDA_VERSION detected"
        
        if [[ $CUDA_VERSION == "11.8" ]]; then
            echo "Installing PyTorch with CUDA 11.8..."
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        elif [[ $CUDA_VERSION == "12.1" ]]; then
            echo "Installing PyTorch with CUDA 12.1..."
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        else
            print_warning "CUDA version $CUDA_VERSION detected. Installing PyTorch with default CUDA support..."
            pip install torch torchvision
        fi
    else
        print_warning "No CUDA detected. Installing CPU-only PyTorch..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
    print_success "PyTorch installed"
}

# Function to install JAX
install_jax() {
    echo ""
    if command -v nvidia-smi &> /dev/null; then
        echo "Installing JAX with CUDA support..."
        pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    else
        print_warning "No CUDA detected. Installing CPU-only JAX..."
        pip install jax
    fi
    print_success "JAX installed"
}

# Process choices
if [[ $CHOICES == *"7"* ]] || [[ $CHOICES == *"all"* ]]; then
    CHOICES="1 2 3 4 5 6"
fi

# Direction 1: Differentiable Physics
if [[ $CHOICES == *"1"* ]]; then
    echo ""
    echo "=== Setting up Differentiable Physics ==="
    install_jax
    pip install brax mujoco taichi
    print_success "Differentiable physics tools installed"
fi

# Direction 2: Physics Foundation Models
if [[ $CHOICES == *"2"* ]]; then
    echo ""
    echo "=== Setting up Physics Foundation Models ==="
    install_pytorch
    pip install the-well
    print_success "Physics foundation model tools installed"
fi

# Direction 3: GNN Particle Simulation
if [[ $CHOICES == *"3"* ]]; then
    echo ""
    echo "=== Setting up GNN Particle Simulation ==="
    if ! command -v pip show torch &> /dev/null; then
        install_pytorch
    fi
    pip install torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cu$(python -c "import torch; print(torch.version.cuda.replace('.', ''))").html
    print_success "PyTorch Geometric installed"
fi

# Direction 4: World Models
if [[ $CHOICES == *"4"* ]]; then
    echo ""
    echo "=== Setting up World Models ==="
    if ! command -v pip show torch &> /dev/null; then
        install_pytorch
    fi
    pip install gymnasium[atari,accept-rom-license] dm-control
    print_success "World model dependencies installed"
fi

# Direction 5: PINNs
if [[ $CHOICES == *"5"* ]]; then
    echo ""
    echo "=== Setting up PINNs ==="
    if ! command -v pip show torch &> /dev/null; then
        install_pytorch
    fi
    pip install deepxde
    print_success "DeepXDE installed"
fi

# Direction 6: Multimodal Fusion
if [[ $CHOICES == *"6"* ]]; then
    echo ""
    echo "=== Setting up Multimodal Fusion ==="
    if ! command -v pip show torch &> /dev/null; then
        install_pytorch
    fi
    pip install opencv-python pillow albumentations
    print_success "Multimodal fusion tools installed"
fi

# Install experiment tracking tools (optional)
echo ""
read -p "Do you want to install experiment tracking tools (wandb, tensorboard)? (y/N): " INSTALL_TRACKING
if [[ $INSTALL_TRACKING =~ ^[Yy]$ ]]; then
    pip install wandb tensorboard
    print_success "Experiment tracking tools installed"
fi

# Summary
echo ""
echo "=================================="
print_success "Setup complete!"
echo "=================================="
echo ""
echo "Environment name: $ENV_NAME"
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To deactivate, run:"
echo "  conda deactivate"
echo ""
echo "Next steps:"
echo "  1. Read GETTING_STARTED.md for learning paths"
echo "  2. Check RESOURCES.md for code repositories"
echo "  3. Explore the research directions in subdirectories"
echo ""
