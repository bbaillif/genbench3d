# GenBench3D
Benchmarking deep learning models generating molecules in 3D

## Requirements
- Python > 3.9
- RDKit > 2022.09 (molecule handling + 2022.09 required for rdDetermineBonds)
- openbabel (for molecule protonation + second bond determination option)
- [ADFRsuite](https://ccsb.scripps.edu/adfr/downloads/)
- vina (for Vina docking score)
- access to Schrodinger Glide CLI (optional)
- [csd python api](https://downloads.ccdc.cam.ac.uk/documentation/API/installation_notes.html) (for Gold PLP score, optional)
- meeko (for ligand preparation for Vina)
- mdanalysis (for protein extraction)
- PDBFixer (fix the chain identification that is duplicated in CrossDocked files)

## Installation
I might put it on pip/conda if I have time, but for now it is a standalone. 
Run these commands to download genbench3d, then install in development mode using pip:  
```bash
git clone https://github.com/bbaillif/genbench3D.git
cd genbench3D
conda env create -f environment.yml
conda activate genbench3d
pip install -e . # install genbench3d in current environment
```

Before using GenBench3D, you must define working relative/absolute paths in the config/default.yaml file.

For the reference data, we used the CSD Drug subset that can be found [here](https://ars.els-cdn.com/content/image/1-s2.0-S0022354918308104-mmc2.zip) along with the CSD Python API to query the September 2023 release of the CSD, but you can use any list of molecules you want. We recommend using the publicly available LigBoundConf PDB subset (minimized version) that can be downloaded from [here](https://pubs.acs.org/doi/suppl/10.1021/acs.jcim.0c01197/suppl_file/ci0c01197_si_002.zip)

The original CrossDocked v1.1 dataset can be downloaded from [here](http://bits.csb.pitt.edu/files/crossdock2020/) (make sure you have enough space because there are a lot of files), while the processed CrossDocked dataset (extracting pockets and ligands only for RMSD < 1A) used in e.g. Pocket2Mol can be downloaded [here](https://drive.google.com/drive/folders/1CzwxmTpjbrt83z_wBzcQncq84OVDPurM). You can run the convert_crossdocked_split.py script with another conda environment you own that has pytorch installed to transform the datasplit (in .pt pickle format containing torch objects) to a .p format without torch object to be run with minimal dependancies with the provided genbench3D environment.

## Usage

The first step is to load the configuration file to indicate the working directories and default parameters.
```python
import yaml
config_path = 'config/default.yaml' # change path accordingly
config = yaml.safe_load(open(args.config_path, 'r'))
```

The second step is to setup a reference geometry from a data source. The CSDDrug source was used in our manuscript, but you can define any reference from a list of molecules. We provide an example for LigBoundConf, a public subset of the PDB with high quality structural data.
```python
from genbench3d.data.source import LigBoundConf
from genbench3d.geometry import ReferenceGeometry

ligboundconf_path = config['data']['ligboundconf_path'] # Set accordingly
source = LigBoundConf(ligands_path=ligboundconf_path)
reference_geometry = ReferenceGeometry(source=source, root=config['benchmark_dirpath'], minimum_pattern_values=config['genbench3D']['minimum_pattern_values'],)
```
The root argument is used to save the values and kernel densities for the extracted reference geometry, and the minimum pattern values is the number of values required for a pattern to have its kernel density computed (if lower than 50, the default behaviour during validity3D evaluation is to simplify the pattern, and if simplified, to consider the geometry as valid in the absence of sufficient data)

The main usage is to compute all metrics from a list of RDKit molecules `mol_list`:
```python
from genbench3d import GenBench3D
# mol_list is a list of RDKit molecules
genbench3D = GenBench3D(reference_geometry=reference_geometry,
                        config=config['genbench3D'])
results = genbench3D.get_results_for_mol_list(mol_list)
```

Under the hood, the GenBench3D is transforming the molecule list (generated molecules or training molecules) into a `ConfEnsembleLibrary`, a structure that groups the conformations of the same molecule (i.e. molecule topological graph and stereochemistry) into unique `ConfEnsemble` (wrapper around a single RDKit molecule having multiple Conformer), under a default name that is the SMILES representation
```python
from conf_ensemble import ConfEnsembleLibrary
cel = ConfEnsembleLibrary.from_mol_list(mol_list)
```
`cel.library` is a dictionary in the form {SMILES (or other specified name): conf_ensemble}  
`conf_ensemble.mol` gives the RDKit molecule containing each listed conformation of that molecule in the original `mol_list`

To compute metrics that depends on the training molecules (e.g. training similarity, novelty2D and 3D), you need to set the training mols with the corresponding method before running the get_results:
```python
# training_mols is a list of RDKit molecules corresponding to training ligands
training_cel = ConfEnsembleLibrary.from_mol_list(training_mols)
benchmark.set_training_cel(training_cel)
```

When dealing with pocket-based generative models, you can use the `SBGenBench3D` version to add metrics related to activity. This new object needs the pocket (extracted from the protonated = "clean" protein produced during Vina processing) and native ligands (as `native_ligand` RDKit molecule) as input.
You can setup the Vina calculation by creating a `VinaProtein` object: this legacy class preprocess the raw pdb files from CrossDocked (select protein only, add hydrogens) and generate pdbqt files for Vina input. 
You can also setup the Glide calculation with a `GlideProtein` or a Gold calculation with a `VinaProtein`. The benchmark is setup as followed

```python
from genbench3d import SBGenBench3D
from genbench3d.data.structure import VinaProtein, Pocket, GlideProtein
# native_ligand is a RDKit molecule
# original_structure_path is a string pdb path to the initial protein file
vina_protein = VinaProtein(pdb_filepath=original_structure_path,
                            prepare_receptor_bin_path=config['bin']['prepare_receptor_bin_path'],)
glide_protein = GlideProtein(pdb_filepath=vina_protein.protein_clean_filepath,
                            native_ligand=native_ligand,
                            glide_output_dirpath=config['glide_working_dir'],
                            glide_path=config['bin']['glide_path'],
                            structconvert_path=config['bin']['structconvert_path'],)
pocket = Pocket(pdb_filepath=vina_protein.protein_clean_filepath, 
                native_ligand=native_ligand,
                distance_from_ligand=config['pocket_distance_from_ligand'])
sbgenbench3D = SBGenBench3D(reference_geometry=reference_geometry,
                            config=config['genbench3D'],
                            pocket=pocket,
                            native_ligand=native_ligand)
sbgenbench3D.setup_vina(vina_protein,
                        config['vina'],
                        add_minimized=True)
sbgenbench3D.setup_glide(glide_protein,
                            glide_path=config['bin']['glide_path'],
                            add_minimized=True)
sbgenbench3D.setup_gold_plp(vina_protein)
sbgenbench3D.set_training_cel(training_cel)
results = sbgenbench3D.get_results_for_mol_list(mols=gen_mols,
                                                n_total_mols=n_total_mols)
```

Additionnally, you can relax molecules inside the binding pocket (i.e. local energy minimization using MMFF94s forcefield, keeping protein atom fixed and allowing up to 1 Angstrom ligand heavy atom deviation, maximum 1000 steps) with the ComplexMinimizer:

```python
from genbench3d.data import ComplexMinimizer
# mol is a RDKit molecule with a single conformer
complex_minimizer = ComplexMinimizer(pocket,
                                    config=config['minimization'])
mini_mol = complex_minimizer.minimize_ligand(mol)
```

Aggregated compiled metrics can be computed for the benchmark results:
```python
import numpy as np
import pandas as pd
summary = {}
for metric_name, values in results.items():
    if isinstance(values, dict): # e.g. Ring proportion
        for key, value in values:
            summary[metric_name + key] = value
    elif isinstance(values, list):
        summary[metric_name] = np.nanmedian(values) # values can have nan
    else: # float or int
        summary[metric_name] = values
print(summary)
```

## Implemented metrics

### Based on topological graph (traditionally used to evaluate SMILES or molecular graph generators)

| Metric | Definition |
| --- | --- |
| Validity2D | Fraction of molecules parsed by RDKit (the n_total_mols can be setup manually in all get_metrics functions) |
| Uniqueness2D | Fraction of unique molecules (based on SMILES with stereochemistry embeded) |
| Novelty2D | Fraction of molecules with a stereo-SMILES not in training molecules (if input) |
| Diversity2D | Average Tanimoto dissimilarity (1 - similarity) of Morgan Fingerprints radius 3 with stereochemistry considered between all generated molecules|
| Ring size proportion | Distribution of the proportion of observed ring sizes |
| Molecular weight (MW) | Self-explanatory |
| logP | Using RDKit |
| SAScore | Using RDKit implementation of SAScore |
| Quantitative Estimate of Drug-likeness (QED) | Using RDKit implementation of QED |

### Based on molecular 3D conformation

| Metric | Definition |
| --- | --- |
| Validity3D | Fraction of molecular conformations with valid bond lengths and valence angles, based on kernel density estimations from values observed in the CSD Drug Subset, see `Validity3D` class for details, flat aromatic rings and no intramolecular steric clash |
| Uniqueness3D | Fraction of (3D-valid by default) unique conformations based on Torsion Fingerprint Deviation (TFD) with a default threshold of 0.2. Threshold can be input in `GenBench3D` with the `tfd_threshold` argument |
| Novelty3D | Fraction of (3D-valid by default) conformations different from those observed in the training set (only on the set of common molecules between training and generated)  |
| Diversity3D | Average interconformation deviation computed using the TFD (on 3D-valid by default)|
| MMFF94s strain energy | Using up to 1000 minimization step, computed using the MMFF94s RDKit implementation |

### Pocket-based metrics (based on a given protein pocket and native ligand)

| Metric | Definition |
| --- | --- |
| Steric clash | Fraction of molecules clashing with the protein |
| Out of pocket | Fraction of molecules having a centroid 10 Angstrom away from the native ligand centroid |
| Absolute Vina score | Using Vina Python package |
| Vina score relative to test ligand | Using Vina Python package |
| Absolute Gold PLP score | Using CSD Python API, CCDC tools required |
| Gold PLP score relative to test ligand | Using CSD Python API, CCDC tools required |
| Absolute Glide score | Requires Schrodinger Glide command line interface |
| Glide score relative to test ligand | Requires Schrodinger Glide command line interface |

## Set of generated molecules

You can download the data used in the manuscript on [figshare](https://figshare.com/articles/dataset/Data_for_Benchmarking_structure-based_3D_generative_models_with_GenBench3D/26139496)

## Previous versions details

Earlier versions of GenBench3D used the ESPSIM and Interaction FingerPrints (IFP), that are commented out in the main code, we decided to put the emphasis on scoring functions

You have different ways of inputing molecules in the benchmark.
If the model generated xyz coordinates in an ASE db (i.e. GSchNet) you can use the ASEDBReader
```python
ase_db_path = "your favorite path"
ase_reader = ASEDBReader(ase_db_path)
```

This reader is directly embedded in the benchmark:
```python
benchmark = GenBench3D()
metrics = benchmark.get_metrics_for_ase_db(filepath=ase_db_path,
                                            cel_name='generated_molecules')
