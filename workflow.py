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
from pymatgen.core import Composition, Element
from ase.io import write
import re
import yaml
import requests

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

def setup_lammps_dir(base_dir, doc, elements, pair_style, pair_coeff):
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
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            if "FINAL_ENERGY" in f.read():
                print(f"Calculation already completed in {dir_path}. Skipping.")
                return True
                
    tf_lib_path = glob.glob(f"{sys.exec_prefix}/lib/python*/site-packages/tensorflow")
    ld_path_setup = f"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{tf_lib_path[0]} && " if tf_lib_path else ""
    
    cmd = f"cd {dir_path} && {ld_path_setup}{lammps_exec} -in lammps.in > lammps.out"
    
    try:
        res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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

def format_formula(formula):
    # Format formula numbers into subscripts (e.g., Ni3Al -> Ni$_{3}$Al)
    return re.sub(r'(\d+)', r'$_{\1}$', formula)

def main():
    parser = argparse.ArgumentParser(description="LAMMPS Convex Hull Workflow")
    parser.add_argument("-i", "--input", dest="config", required=True, help="Path to input YAML configuration file")
    parser.add_argument("--setup-only", action="store_true", help="Only fetch structures and setup directories, do not run LAMMPS")
    parser.add_argument("--plot-only", action="store_true", help="Skip fetching and calculations, just redraw the plot from existing results.csv")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    elements = config.get("elements")
    api_key = config.get("api_key")
    
    compare_models = config.get("compare_models", False)
    if "models" in config:
        models = config["models"]
        if not compare_models:
            models = [models[0]]
    else:
        # Fallback to old format
        models = [{
            "name": "Model",
            "lammps_exec": config.get("lammps_exec", "lmp"),
            "pair_style": config.get("pair_style", "pair_style pace"),
            "pair_coeff": config.get("pair_coeff"),
            "output_dir": config.get("output_dir", "hull_workflow_output")
        }]
        
    if len(elements) != 2:
        print("Warning: This script is currently optimized for binary systems plotting, but calculations will proceed.")
        
    all_results = {}
    
    if args.plot_only:
        print("Plot-only mode active. Loading existing results.csv files...")
        for model in models:
            model_name = model.get("name", "Model")
            output_dir = model.get("output_dir", f"hull_workflow_output_{model_name}")
            csv_path = os.path.join(output_dir, "results.csv")
            if os.path.exists(csv_path):
                all_results[model_name] = pd.read_csv(csv_path)
                print(f"Loaded {csv_path}")
            else:
                print(f"Warning: {csv_path} not found for {model_name}")
                
        if not all_results:
            print("No results.csv files found to plot. Exiting.")
            return
    else:
        # Phase 1: Get Structures
        docs = None
        
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
            
        for model in models:
            model_name = model.get("name", "Model")
            print(f"\n--- Processing Model: {model_name} ---")
            
            lammps_exec = model.get("lammps_exec", "lmp")
            pair_style = model.get("pair_style", "pair_style pace")
            pair_coeff = model.get("pair_coeff")
            output_dir = model.get("output_dir", f"hull_workflow_output_{model_name}")
            
            os.makedirs(output_dir, exist_ok=True)
            
            results = []
            
            if docs:
                for doc in docs:
                    mp_id = doc.material_id
                    formula = doc.formula_pretty
                    
                    dir_path = setup_lammps_dir(output_dir, doc, elements, pair_style, pair_coeff)
                    
                    if args.setup_only:
                        continue
                        
                    success = run_lammps(dir_path, lammps_exec)
                    
                    if success:
                        energy, n_atoms = parse_energy(dir_path)
                        if energy is not None:
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
                if args.setup_only:
                    print(f"Cannot run --setup-only without internet access for {model_name}. Skipping.")
                    continue
                    
                if not os.path.exists(output_dir):
                    print(f"Directory {output_dir} does not exist. Skipping.")
                    continue
                    
                existing_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
                if not existing_dirs:
                    print(f"No existing directories found in {output_dir} and MP-API fetch failed. Skipping.")
                    continue
                    
                print(f"Found {len(existing_dirs)} directories in {output_dir}. Running LAMMPS...")
                for d in existing_dirs:
                    dir_path = os.path.join(output_dir, d)
                    parts = d.split("_")
                    if len(parts) != 2: continue
                    mp_id = parts[0]
                    formula = parts[1]
                    composition = Composition(formula)
                    
                    success = run_lammps(dir_path, lammps_exec)
                    
                    if success:
                        energy, n_atoms = parse_energy(dir_path)
                        if energy is not None:
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
                print(f"Setup complete for {model_name}!")
                continue
                
            if not results:
                print(f"No successful LAMMPS runs for {model_name}.")
                continue
                
            df = pd.DataFrame(results)
            
            ref_energies = {}
            for el in elements:
                pure_phases = df[df[f"frac_{el}"] == 1.0]
                if not pure_phases.empty:
                    min_e = pure_phases["energy_per_atom"].min()
                    ref_energies[el] = min_e
                    print(f"Reference energy for pure {el} ({model_name}): {min_e:.4f} eV/atom")
                else:
                    print(f"WARNING: No successful calculations for pure {el} in {model_name}! Cannot compute formation energies accurately.")
                    ref_energies[el] = 0.0
                    
            def calc_formation_energy(row):
                e_ref_sum = sum(row[f"frac_{el}"] * ref_energies[el] for el in elements)
                return row["energy_per_atom"] - e_ref_sum
                
            df["formation_energy"] = df.apply(calc_formation_energy, axis=1)
            csv_path = os.path.join(output_dir, "results.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved results to {csv_path}")
            
            all_results[model_name] = df

    if args.setup_only:
        return

    # Phase 5: Convex Hull Plotting (for binary only)
    if len(elements) == 2 and all_results:
        el_A, el_B = elements
        plt.figure(figsize=(12, 8))
        
        line_styles = ['-', '--', ':', '-.']
        markers = ['o', 's', '^', 'D', 'v', 'p', '*']
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        labeled_points = {}  # { (round(x, 4), formula): True }
        show_only_negative = config.get("show_only_negative_energies", False)
        
        for idx, (model_name, df) in enumerate(all_results.items()):
            style = line_styles[idx % len(line_styles)]
            marker = markers[idx % len(markers)]
            color = colors[idx % len(colors)]
            
            if show_only_negative:
                plot_df = df[df["formation_energy"] <= 0.05]
            else:
                plot_df = df
                
            x_all = plot_df[f"frac_{el_B}"].values
            y_all = plot_df["formation_energy"].values
            
            min_energy_by_x = {}
            for i, row in plot_df.iterrows():
                x_val = row[f"frac_{el_B}"]
                y_val = row["formation_energy"]
                if y_val <= 0.05:
                    x_round = round(x_val, 4)
                    if x_round not in min_energy_by_x or y_val < min_energy_by_x[x_round]["y"]:
                        min_energy_by_x[x_round] = {"x": x_val, "y": y_val, "row": row}
                        
            valid_points_list = [(v["x"], v["y"]) for v in min_energy_by_x.values()]
            valid_points = np.array(valid_points_list)
            
            if len(valid_points) >= 3:
                try:
                    hull = ConvexHull(valid_points)
                    
                    hull_x = valid_points[hull.vertices, 0]
                    hull_y = valid_points[hull.vertices, 1]
                    
                    sort_idx = np.argsort(hull_x)
                    hull_x = hull_x[sort_idx]
                    hull_y = hull_y[sort_idx]
                    
                    # Plot hull line WITHOUT label so it doesn't show up as a line in the legend
                    plt.plot(hull_x, hull_y, linestyle=style, color=color, linewidth=2, zorder=1)
                    
                    # Plot hull solid points WITH label
                    plt.scatter(hull_x, hull_y, marker=marker, facecolors=color, edgecolors=color, 
                                s=64, zorder=2, label=f'{model_name}')
                    
                    # Separate points into hull and non-hull
                    hull_points_set = set((round(x, 4), round(y, 4)) for x, y in zip(hull_x, hull_y))
                    non_hull_x, non_hull_y = [], []
                    for x, y in zip(x_all, y_all):
                        if (round(x, 4), round(y, 4)) not in hull_points_set:
                            non_hull_x.append(x)
                            non_hull_y.append(y)
                            
                    # Plot non-hull (hollow)
                    if non_hull_x:
                        plt.scatter(non_hull_x, non_hull_y, marker=marker, facecolors='none', edgecolors=color, 
                                    alpha=0.6, zorder=2, label=f'{model_name} (above convex hull)')
                    
                    for x_val, y_val in zip(hull_x, hull_y):
                        row = None
                        for v in min_energy_by_x.values():
                            if abs(v["x"] - x_val) < 1e-4 and abs(v["y"] - y_val) < 1e-4:
                                row = v["row"]
                                break
                        if row is not None:
                            sg = row.get("spacegroup", "")
                            form = row['formula']
                            x_round = round(x_val, 4)
                            
                            label_key = (x_round, form)
                            if label_key not in labeled_points:
                                labeled_points[label_key] = True
                                formatted_form = format_formula(form)
                                plt.annotate(formatted_form, (x_val, y_val), 
                                         textcoords="offset points", xytext=(0,-10), ha='center', va='top', fontsize=10)
                except Exception as e:
                    print(f"Error plotting convex hull for {model_name}: {e}")
                    # If error, plot all as hollow
                    if len(x_all) > 0:
                        plt.scatter(x_all, y_all, marker=marker, facecolors='none', edgecolors=color, alpha=0.6, label=f'{model_name} (above convex hull)')
            else:
                 print(f"Not enough stable points to draw a convex hull for {model_name}.")
                 # Plot all as hollow if no hull
                 if len(x_all) > 0:
                     plt.scatter(x_all, y_all, marker=marker, facecolors='none', edgecolors=color, alpha=0.6, label=f'{model_name} (above convex hull)')

        plt.xlabel(f"X$_{{{el_B}}}$ (atomic fraction)", fontsize=14)
        plt.ylabel(r"E$_f$ (eV/atom)", fontsize=14)
        plt.title(f"{el_A}-{el_B} System", fontsize=16)
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        
        ymin, ymax = plt.ylim()
        plt.ylim(ymin - 0.1 * (ymax - ymin), ymax)
        
        general_output_dir = "comparison_output"
        if len(models) == 1:
            general_output_dir = models[0].get("output_dir", "comparison_output")
        else:
            os.makedirs(general_output_dir, exist_ok=True)
            
        plot_path = os.path.join(general_output_dir, "convex_hull_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved convex hull plot to {plot_path}")

if __name__ == "__main__":
    main()
