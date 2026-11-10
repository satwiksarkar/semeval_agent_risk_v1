#!/usr/bin/env bash
# setup_structure.sh
# Creates PEP-compliant ML research structure for semeval_agent_risk_v1

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Setting up repository structure in: $BASE_DIR"

mkdir -p "$BASE_DIR/data/raw"
mkdir -p "$BASE_DIR/data/processed/phase_1_dag"
mkdir -p "$BASE_DIR/src/agent_risk/simulator"
mkdir -p "$BASE_DIR/src/agent_risk/models"
mkdir -p "$BASE_DIR/experiments"
mkdir -p "$BASE_DIR/tests"

touch "$BASE_DIR/src/agent_risk/__init__.py"
touch "$BASE_DIR/src/agent_risk/simulator/__init__.py"
touch "$BASE_DIR/src/agent_risk/models/__init__.py"
touch "$BASE_DIR/experiments/__init__.py"
touch "$BASE_DIR/tests/__init__.py"

touch "$BASE_DIR/data/raw/.gitkeep"
touch "$BASE_DIR/data/processed/phase_1_dag/.gitkeep"

cat << 'EOF' > "$BASE_DIR/.gitignore"
# Byte-compiled / optimized files
__pycache__/
*.py[cod]
*$py.class

# PyTorch checkpoints and tensors
*.pt
*.pth
*.bin

# Data directories
data/raw/*
!data/raw/.gitkeep
data/processed/phase_1_dag/*
!data/processed/phase_1_dag/.gitkeep

# Environments & IDEs
.venv/
env/
venv/
.idea/
.vscode/

# Build & packaging
dist/
build/
*.egg-info/
EOF

echo "Repository structure initialized successfully."

