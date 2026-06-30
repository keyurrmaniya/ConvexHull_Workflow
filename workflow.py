import os
import sys
import argparse
import subprocess
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write

def get_structures(api_key, elements):
    """
    Query Materials Project for pure elements and the binary system.
    """
    structures = []
    print(f"Connecting to MP-API with key: {api_key[:5]}...")
    
    with MPRester(api_key) as mpr:
        # Query pure elements
        for el in elements:
            print(f"Querying pure {el} structures...")
            docs = mpr.materials.summary.search(chemsys=el)
            structures.extend(docs)
            
        # Query binary system
        binary_chemsys = "-".join(sorted(elements))
        print(f"Querying binary {binary_chemsys} structures...")
        docs = mpr.materials.summary.search(chemsys=binary_chemsys)
        structures.extend(docs)
        
    print(f"Found {len(structures)} total structures.")
    return structures

def setup_lammps_dir(base_dir, doc, elements, potential_file, pair_style, pair_coeff):
    """
    Setup directory and LAMMPS input for a given structure.
    """
    mp_id = doc.material_id
    formula = doc.formula_pretty
    dir_name = f"{mp_id}_{formula}"
    dir_path = os.path.join(base_dir, dir_name)
    os.makedirs(dir_path, exist_ok=True)
    
    # Write LAMMPS data file
    atoms = AseAtomsAdaptor.get_atoms(doc.structure)
    data_file = os.path.join(dir_path, "data.lammps")
    write(data_file, atoms, format='lammps-data', specorder=elements)
    
    # Calculate masses for LAMMPS input if needed, but read_data usually handles it
    # However, sometimes LAMMPS requires mass explicitly. 
    # ase.io.write includes masses if atom types are mapped.
    
    # Write LAMMPS input script
    elements_str = " ".join(elements)
    
    # Generate mass commands
    from pymatgen.core import Element
    mass_lines = ""
    for i, el_name in enumerate(elements):
        mass = float(Element(el_name).atomic_mass)
        mass_lines += f"mass {i+1} {mass}\n"
        
    lammps_in_content = f"""# LAMMPS input for {formula} ({mp_id})
units metal
boundary p p p
atom_style atomic

read_data data.lammps

{mass_lines}
{pair_style}
{pair_coeff}

thermo 10
thermo_style custom step pe lx ly lz press pxx pyy pzz

fix 1 all box/relax iso 0.0 vmax 0.001
min_style cg
minimize 1e-25 1e-25 5000 10000

print "FINAL_ENERGY: $(pe)"
print "FINAL_ATOMS: $(atoms)"
"""
    in_file = os.path.join(dir_path, "lammps.in")
    with open(in_file, 'w') as f:
        f.write(lammps_in_content)
        
    return dir_path

def run_lammps(dir_path, lammps_exec):
    """
    Run LAMMPS in the specified directory.
    """
    in_file = os.path.join(dir_path, "lammps.in")
    log_file = os.path.join(dir_path, "log.lammps")
    
    print(f"Running LAMMPS in {dir_path}...")
    
    # Check if already completed
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            if "FINAL_ENERGY" in f.read():
                print(f"Calculation already completed in {dir_path}. Skipping.")
                return True
                
    
    # Try to find python's tensorflow library path to add to LD_LIBRARY_PATH
    import sys
    import glob
    tf_lib_path = glob.glob(f"{sys.exec_prefix}/lib/python*/site-packages/tensorflow")
    ld_path_setup = f"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{tf_lib_path[0]} && " if tf_lib_path else ""
    
    cmd = f"cd {dir_path} && {ld_path_setup}{lammps_exec} -in lammps.in > lammps.out"
    
    try:
        res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Even if it returns non-zero, check if the output file contains the result
        # because ML potentials (like GRACE/Tensorflow) sometimes segfault on exit.
        if res.returncode != 0:
            print(f"LAMMPS returned non-zero exit code for {dir_path}, but checking if calculation finished.")
        return True
    except Exception as e:
        print(f"Error running LAMMPS in {dir_path}:\n{e}")
        return False

def parse_energy(dir_path):
    """
    Parse the final potential energy and atom count from LAMMPS output.
    """
    log_file = os.path.join(dir_path, "log.lammps")
    if not os.path.exists(log_file):
        return None, None
        
    energy = None
    atoms = None
    with open(log_file, 'r') as f:
        for line in f:
            if line.startswith("FINAL_ENERGY:"):
                energy = float(line.split()[1])
            elif line.startswith("FINAL_ATOMS:"):
                atoms = int(float(line.split()[1]))
                
    if energy is not None and atoms is not None:
        return energy, atoms
    return None, None

def main():
    parser = argparse.ArgumentParser(description="LAMMPS Convex Hull Workflow")
    parser.add_argument("-i", "--input", dest="config", required=True, help="Path to input YAML configuration file")
    parser.add_argument("--setup-only", action="store_true", help="Only fetch structures and setup directories, do not run LAMMPS")
    
    args = parser.parse_args()
    
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    elements = config.get("elements")
    api_key = config.get("api_key")
    potential = os.path.abspath(config.get("potential"))
    lammps_exec = config.get("lammps_exec", "lmp")
    pair_style = config.get("pair_style", "pair_style pace")
    pair_coeff = config.get("pair_coeff")
    output_dir = config.get("output_dir", "hull_workflow_output")
    
    if len(elements) != 2:
        print("Warning: This script is currently optimized for binary systems plotting, but calculations will proceed.")
        
    if not pair_coeff:
        elements_str = " ".join(elements)
        pair_coeff = f"pair_coeff * * {potential} {elements_str}"
        
    # Phase 1: Get Structures
    docs = None
    
    # Check for internet access to prevent infinite hanging on compute nodes
    import requests
    try:
        requests.get("https://api.materialsproject.org", timeout=3)
        has_internet = True
    except (requests.ConnectionError, requests.Timeout):
        has_internet = False
        
    if has_internet:
        try:
            print("Fetching structures from MP-API...")
            docs = get_structures(api_key, elements)
        except Exception as e:
            print(f"Failed to fetch structures from MP-API. Error: {e}")
            print("Falling back to existing directories...")
    else:
        print("No internet access detected (likely running on a compute node).")
        print("Bypassing MP-API fetch and falling back to existing directories in output_dir...")
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Phase 2 & 3: Setup and Run LAMMPS
    results = []
    
    if docs:
        for doc in docs:
            mp_id = doc.material_id
            formula = doc.formula_pretty
            
            dir_path = setup_lammps_dir(output_dir, doc, elements, potential, pair_style, pair_coeff)
            
            if args.setup_only:
                continue
                
            success = run_lammps(dir_path, lammps_exec)
            
            if success:
                energy, n_atoms = parse_energy(dir_path)
                if energy is not None:
                    # Extract spacegroup from data.lammps
                    sg_symbol = ""
                    try:
                        from ase.io import read
                        from ase.spacegroup import get_spacegroup
                        atoms = read(os.path.join(dir_path, "data.lammps"), format="lammps-data", style="atomic")
                        element_map = {i+1: el for i, el in enumerate(elements)}
                        symbols = [element_map[t] for t in atoms.get_atomic_numbers()]
                        atoms.set_chemical_symbols(symbols)
                        sg_symbol = get_spacegroup(atoms, symprec=0.1).symbol
                    except Exception as e:
                        pass
                        
                    res = {
                        "mp_id": mp_id,
                        "formula": formula,
                        "spacegroup": sg_symbol,
                        "energy": energy,
                        "n_atoms": n_atoms,
                        "energy_per_atom": energy / n_atoms
                    }
                    for el in elements:
                        res[f"frac_{el}"] = doc.composition.get_atomic_fraction(el)
                    results.append(res)
    else:
        # Fallback loop: if no docs fetched, just loop over existing directories
        if args.setup_only:
            print("Cannot run --setup-only without internet access. Exiting.")
            return
            
        from pymatgen.core import Composition
        existing_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
        if not existing_dirs:
            print("No existing directories found in output_dir and MP-API fetch failed. Exiting.")
            return
            
        print(f"Found {len(existing_dirs)} directories in {output_dir}. Running LAMMPS...")
        for d in existing_dirs:
            dir_path = os.path.join(output_dir, d)
            parts = d.split("_")
            if len(parts) != 2: continue
            mp_id = parts[0]
            formula = parts[1]
            composition = Composition(formula)
            
            # Setup is already done, just run
            success = run_lammps(dir_path, lammps_exec)
            
            if success:
                energy, n_atoms = parse_energy(dir_path)
                if energy is not None:
                    # Extract spacegroup from data.lammps
                    sg_symbol = ""
                    try:
                        from ase.io import read
                        from ase.spacegroup import get_spacegroup
                        atoms = read(os.path.join(dir_path, "data.lammps"), format="lammps-data", style="atomic")
                        element_map = {i+1: el for i, el in enumerate(elements)}
                        symbols = [element_map[t] for t in atoms.get_atomic_numbers()]
                        atoms.set_chemical_symbols(symbols)
                        sg_symbol = get_spacegroup(atoms, symprec=0.1).symbol
                    except Exception as e:
                        pass
                        
                    res = {
                        "mp_id": mp_id,
                        "formula": formula,
                        "spacegroup": sg_symbol,
                        "energy": energy,
                        "n_atoms": n_atoms,
                        "energy_per_atom": energy / n_atoms
                    }
                    for el in elements:
                        res[f"frac_{el}"] = composition.get_atomic_fraction(el)
                    results.append(res)
                else:
                    print(f"Could not parse energy for {mp_id}")
                
    if args.setup_only:
        print(f"Setup complete! {len(docs)} directories created in {output_dir}.")
        return
        
    if not results:
        print("No successful LAMMPS runs. Exiting.")
        return
        
    # Phase 4: Compute Formation Energy
    df = pd.DataFrame(results)
    
    # Find pure element references
    ref_energies = {}
    for el in elements:
        pure_phases = df[df[f"frac_{el}"] == 1.0]
        if not pure_phases.empty:
            min_e = pure_phases["energy_per_atom"].min()
            ref_energies[el] = min_e
            print(f"Reference energy for pure {el}: {min_e:.4f} eV/atom")
        else:
            print(f"WARNING: No successful calculations for pure {el}! Cannot compute formation energies.")
            # Set to 0 just to allow the script to finish, but the hull will be wrong.
            ref_energies[el] = 0.0
            
    # Calculate formation energy
    def calc_formation_energy(row):
        e_ref_sum = sum(row[f"frac_{el}"] * ref_energies[el] for el in elements)
        return row["energy_per_atom"] - e_ref_sum
        
    df["formation_energy"] = df.apply(calc_formation_energy, axis=1)
    
    # Save CSV
    csv_path = os.path.join(output_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
    
    # Phase 5: Convex Hull Plotting (for binary only)
    if len(elements) == 2:
        el_A, el_B = elements
        x_all = df[f"frac_{el_B}"].values
        y_all = df["formation_energy"].values
        
        # Plot all structures in the background
        plt.figure(figsize=(12, 8))
        plt.scatter(x_all, y_all, color='blue', alpha=0.3, label='All Structures')
        
        # Filter for strict minimum at each composition to avoid vertical lines on hull ends
        min_energy_by_x = {}
        for i, row in df.iterrows():
            x_val = row[f"frac_{el_B}"]
            y_val = row["formation_energy"]
            if y_val <= 0.05:  # Only consider relatively stable points
                x_round = round(x_val, 4)
                if x_round not in min_energy_by_x or y_val < min_energy_by_x[x_round]["y"]:
                    min_energy_by_x[x_round] = {"x": x_val, "y": y_val, "row": row}
                    
        valid_points_list = [(v["x"], v["y"]) for v in min_energy_by_x.values()]
        valid_points = np.array(valid_points_list)
        
        if len(valid_points) >= 3:
            try:
                hull = ConvexHull(valid_points)
                
                # Extract hull vertices
                hull_x = valid_points[hull.vertices, 0]
                hull_y = valid_points[hull.vertices, 1]
                
                # Sort hull points by x for proper lower hull line plotting
                sort_idx = np.argsort(hull_x)
                hull_x = hull_x[sort_idx]
                hull_y = hull_y[sort_idx]
                
                # Plot the lower convex hull line
                plt.plot(hull_x, hull_y, 'r-', linewidth=2, marker='o', markersize=8, label='Lower Convex Hull')
                
                # Label points that are on or very close to the convex hull
                for x_val, y_val in zip(hull_x, hull_y):
                    # Find the corresponding row
                    row = None
                    for v in min_energy_by_x.values():
                        if abs(v["x"] - x_val) < 1e-4 and abs(v["y"] - y_val) < 1e-4:
                            row = v["row"]
                            break
                    if row is not None:
                        sg = row.get("spacegroup", "")
                        label_text = f"{row['formula']}\n({sg})" if sg else row['formula']
                        plt.annotate(label_text, (x_val, y_val), 
                                     textcoords="offset points", xytext=(0,-15), ha='center', va='top', fontsize=9,
                                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
                        
                plt.xlabel(f"Atomic Fraction of {el_B}")
                plt.ylabel("Formation Energy (eV/atom)")
                plt.title(f"Convex Hull for {el_A}-{el_B} System")
                plt.axhline(0, color='black', linestyle='--', linewidth=1)
                plt.legend()
                plt.grid(True, linestyle=':', alpha=0.6)
                
                plot_path = os.path.join(output_dir, "convex_hull.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Saved convex hull plot to {plot_path}")
                
            except Exception as e:
                print(f"Error plotting convex hull: {e}")
                plt.scatter(x_all, y_all, color='blue', alpha=0.6)
                plt.xlabel(f"Atomic Fraction of {el_B}")
                plt.ylabel("Formation Energy (eV/atom)")
                plt.savefig(os.path.join(output_dir, "scatter.png"))
        else:
             print("Not enough stable points to draw a convex hull.")
             plt.scatter(x_all, y_all, color='blue', alpha=0.6)
             plt.xlabel(f"Atomic Fraction of {el_B}")
             plt.ylabel("Formation Energy (eV/atom)")
             plt.savefig(os.path.join(output_dir, "scatter.png"))
             
if __name__ == "__main__":
    main()
