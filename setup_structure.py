#!/usr/bin/env python3
"""
setup_structure.py
Automates the creation of a clean, PEP-compliant ML research repository
structure for semeval_agent_risk_v1 (SemEval 2027 Task 10 - AgentRisk Subtask 1).
"""

from pathlib import Path

def create_directory_structure(base_dir: Path):
    directories = [
        base_dir / "data" / "raw",
        base_dir / "data" / "processed" / "phase_1_dag",
        base_dir / "src" / "agent_risk" / "simulator",
        base_dir / "src" / "agent_risk" / "models",
        base_dir / "experiments",
        base_dir / "tests",
    ]

    for d in directories:
        d.mkdir(parents=True, exist_ok=True)
        print(f"[+] Created directory: {d.relative_to(base_dir)}")

    # Create __init__.py files
    init_files = [
        base_dir / "src" / "agent_risk" / "__init__.py",
        base_dir / "src" / "agent_risk" / "simulator" / "__init__.py",
        base_dir / "src" / "agent_risk" / "models" / "__init__.py",
        base_dir / "experiments" / "__init__.py",
        base_dir / "tests" / "__init__.py",
    ]

    for init_file in init_files:
        if not init_file.exists():
            init_file.write_text('"""Package initialization."""\n', encoding="utf-8")
            print(f"[+] Created init file: {init_file.relative_to(base_dir)}")

    # Create .gitkeep in data directories
    gitkeeps = [
        base_dir / "data" / "raw" / ".gitkeep",
        base_dir / "data" / "processed" / "phase_1_dag" / ".gitkeep",
    ]
    for gk in gitkeeps:
        if not gk.exists():
            gk.touch()

    # Create .gitignore
    gitignore = base_dir / ".gitignore"
    if not gitignore.exists() or gitignore.stat().st_size == 0:
        gitignore.write_text(
            """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# PyTorch checkpoints and tensors
*.pt
*.pth
*.bin

# Data directories (raw and processed artifacts)
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
*.swp

# Distribution / Packaging
dist/
build/
*.egg-info/
""",
            encoding="utf-8",
        )
        print(f"[+] Created .gitignore")

    print("\nRepository structure setup completed successfully.")

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent
    create_directory_structure(repo_root)

