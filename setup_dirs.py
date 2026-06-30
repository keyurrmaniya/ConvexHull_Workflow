import os, yaml
from workflow import get_structures, setup_lammps_dir

with open("input.yaml", "r") as f:
    config = yaml.safe_load(f)

api_key = config["api_key"]
elements = config["elements"]
potential = config["potential"]
pair_style = config["pair_style"]
pair_coeff = config["pair_coeff"]
output_dir = config["output_dir"]
lammps_exec = config["lammps_exec"]

os.makedirs(output_dir, exist_ok=True)
docs = get_structures(api_key, elements)
for doc in docs:
    setup_lammps_dir(output_dir, doc, elements, potential, pair_style, pair_coeff)
print("Directories setup complete!")
