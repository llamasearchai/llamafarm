#!/bin/bash

# LlamaFarmAI Installation Script
# This script sets up the environment for LlamaFarmAI, installing all necessary dependencies.

set -e

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Display banner
function display_banner() {
    echo -e "${GREEN}"
    echo 'LlamaFarmAI: Precision Agriculture Platform'
    echo -e "${NC}"
}

display_banner

# Check system requirements
function check_requirements() {
    echo -e "${YELLOW}Checking system requirements...${NC}"
    # Check Python version
    if command -v python3 &> /dev/null; then
        python_version=$(python3 --version | cut -d' ' -f2)
        echo -e "  - ${GREEN}✓${NC} Python: $python_version"
    else
        echo -e "  - ${RED}✗${NC} Python 3.9+ required"
        exit 1
    fi
    # Check Docker
    if command -v docker &> /dev/null; then
        docker_version=$(docker --version | cut -d' ' -f3 | tr -d ',')
        echo -e "  - ${GREEN}✓${NC} Docker: $docker_version"
    else
        echo -e "  - ${YELLOW}⚠${NC} Docker not found (required for containerized deployment)"
        install_docker=true
    fi
    # Check Git
    if command -v git &> /dev/null; then
        git_version=$(git --version | cut -d' ' -f3)
        echo -e "  - ${GREEN}✓${NC} Git: $git_version"
    else
        echo -e "  - ${YELLOW}⚠${NC} Git not found"
        install_git=true
    fi
    echo ""
}

# Install dependencies
function install_dependencies() {
    echo -e "${YELLOW}Installing dependencies...${NC}"
    # Install system packages
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update
        sudo apt-get install -y python3-pip python3-venv libatlas-base-dev libopenjp2-7 libhdf5-dev
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install python3 hdf5 openjpeg
    fi
    # Install Docker if needed
    if [[ "$install_docker" == true ]]; then
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        rm get-docker.sh
    fi
    # Install Git if needed
    if [[ "$install_git" == true ]]; then
        sudo apt-get install -y git
    fi
    echo -e "  - ${GREEN}✓${NC} Dependencies installed"
}

# Create Python virtual environment and install Python packages
function setup_python_env() {
    echo -e "${YELLOW}Setting up Python environment...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo -e "  - ${GREEN}✓${NC} Python environment set up"
}

# Install PyTorch separately
function install_pytorch() {
    echo -e "${YELLOW}Installing PyTorch...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    echo -e "  - ${GREEN}✓${NC} PyTorch installed"
}

# Main installation process
check_requirements
install_dependencies
setup_python_env
install_pytorch

echo -e "${GREEN}LlamaFarmAI installation complete!${NC}" 