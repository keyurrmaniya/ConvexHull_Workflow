# LAMMPS Convex Hull Workflow

A robust Python CLI tool for automating the calculation and plotting of binary system phase diagrams (convex hulls) using LAMMPS. It directly queries the Materials Project API (via `mp-api`) for all known structures of a given system, sets up LAMMPS simulations using ML potentials (like ACE/GRACE), and plots the lowest-energy convex hull.

## Features
- Fully automated fetch of structural polymorphs from Materials Project
- Handles disconnected compute nodes (via the `--setup-only` flag on a login node)
- Skip-aware: Resumes safely without recalculating completed structures
- Automatic parsing of crystallographic Space Groups for plot labels
- Generates a mathematically strictly-convex lower hull plot and a CSV of formation energies
- **Multi-Model Comparison:** Run calculations for multiple potentials (e.g., GRACE and ACE) and compare their convex hulls on a single, beautifully formatted plot with smart, deduplicated structure labels.

## Installation

You can install this directly into any conda environment.

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/ConvexHull_Workflow.git
   cd ConvexHull_Workflow
   ```

2. Install as an editable package:
   ```bash
   pip install -e .
   ```

## Usage

Create an `input.yaml` file defining your system. You can specify a single model or compare multiple models.

### Multi-Model Comparison (Recommended)
```yaml
elements:
  - Ni
  - Al
api_key: "YOUR_MP_API_KEY"
compare_models: true
models:
  - name: "GRACE"
    potential: "/path/to/grace"
    lammps_exec: "lmp"
    pair_style: "pair_style grace"
    pair_coeff: "pair_coeff * * /path/to/grace Ni Al"
    output_dir: "run_grace"
  - name: "ACE"
    potential: "/path/to/ace"
    lammps_exec: "lmp"
    pair_style: "pair_style pace"
    pair_coeff: "pair_coeff * * /path/to/ace Ni Al"
    output_dir: "run_ace"
```

### Single Model (Legacy Format)
```yaml
elements:
  - Ni
  - Al
api_key: "YOUR_MP_API_KEY"
potential: "/path/to/your/potential/file"
pair_style: "pair_style pace"
pair_coeff: ""
output_dir: "test_run"
lammps_exec: "lmp"
```

### HPC Workflow (Recommended)
1. **On the Login Node** (where internet is available):
   ```bash
   convex_hull -i input.yaml --setup-only
   ```
   This will download all structures and prepare the LAMMPS inputs in the designated `output_dir`(s).

2. **On the Compute Node** (submit via Slurm script):
   ```bash
   convex_hull -i input.yaml
   ```
   This will run LAMMPS offline in the prepared directories, extract all energies, and plot the convex hull.
