#!/bin/bash
#
# LlamaFarms Installation Script
# This script installs the LlamaFarms precision agriculture platform
#

set -e  # Exit on error

# Text colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
echo "==============================================================="
echo "              LlamaFarms Installation Script"
echo "    Advanced Precision Agriculture Platform with MLX Support"
echo "==============================================================="
echo -e "${NC}"

# Check for Python 3.9+
echo -e "${YELLOW}Checking system requirements...${NC}"
if command -v python3 >/dev/null 2>&1; then
    python_version=$(python3 --version | cut -d " " -f 2)
    python_major=$(echo $python_version | cut -d. -f1)
    python_minor=$(echo $python_version | cut -d. -f2)
    
    if [ "$python_major" -ge 3 ] && [ "$python_minor" -ge 9 ]; then
        echo -e "${GREEN}✓ Python ${python_version} detected${NC}"
    else
        echo -e "${RED}✗ Python 3.9+ required, found ${python_version}${NC}"
        echo "Please install Python 3.9 or newer and try again."
        exit 1
    fi
else
    echo -e "${RED}✗ Python 3 not found${NC}"
    echo "Please install Python 3.9 or newer and try again."
    exit 1
fi

# Check for pip
if command -v pip3 >/dev/null 2>&1; then
    echo -e "${GREEN}✓ pip detected${NC}"
else
    echo -e "${RED}✗ pip not found${NC}"
    echo "Please install pip and try again."
    exit 1
fi

# Check for Apple Silicon (for MLX optimization)
if [[ $(uname -m) == "arm64" ]] && [[ $(uname) == "Darwin" ]]; then
    echo -e "${GREEN}✓ Apple Silicon detected - MLX acceleration will be available${NC}"
    apple_silicon=true
else
    echo -e "${YELLOW}⚠ Apple Silicon not detected - MLX acceleration will not be available${NC}"
    apple_silicon=false
fi

# Create virtual environment
echo -e "\n${YELLOW}Setting up virtual environment...${NC}"
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip
pip install wheel setuptools

# Install core requirements
echo -e "${YELLOW}Installing core requirements...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✓ Core requirements installed${NC}"

# Install development dependencies if requested
if [[ "$*" == *--dev* ]]; then
    echo -e "${YELLOW}Installing development dependencies...${NC}"
    pip install pytest black flake8 mypy sphinx mkdocs
    echo -e "${GREEN}✓ Development dependencies installed${NC}"
fi

# Install package in development mode
echo -e "${YELLOW}Installing LlamaFarms in development mode...${NC}"
pip install -e .
echo -e "${GREEN}✓ LlamaFarms installed in development mode${NC}"

# Set up environment variables
echo -e "\n${YELLOW}Setting up environment variables...${NC}"
if [ ! -f ".env" ]; then
    cat > .env << EOF
# LlamaFarms Environment Configuration
LLAMAFARMS_ENV=development
OPENAI_API_KEY=your_api_key_here
EOF
    echo -e "${GREEN}✓ Created .env file${NC}"
    echo -e "${YELLOW}⚠ Please edit .env file to add your API keys${NC}"
else
    echo "Existing .env file found. Skipping creation."
fi

# Create data and model directories
echo -e "\n${YELLOW}Creating data and model directories...${NC}"
mkdir -p data/raw data/processed models/vision models/llm
echo -e "${GREEN}✓ Directories created${NC}"

# Success message
echo -e "\n${GREEN}=============================================${NC}"
echo -e "${GREEN}Installation completed successfully!${NC}"
echo -e "${GREEN}=============================================${NC}"
echo
echo -e "To get started with LlamaFarms:"
echo -e "1. Activate the virtual environment: ${BLUE}source venv/bin/activate${NC}"
echo -e "2. Run the API server: ${BLUE}python -m llamafarms.api.server${NC}"
echo -e "3. Or use the CLI: ${BLUE}llamafarms --help${NC}"
echo
echo -e "For more information, see the documentation in ${BLUE}docs/${NC}"
echo -e "or visit ${BLUE}https://github.com/yourusername/llamafarms${NC}"
echo

exit 0 