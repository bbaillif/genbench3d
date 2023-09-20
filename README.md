# GenBench3D
Benchmarking deep learning models generating molecules in 3D

## Requirements
- Python 3.11
- RDKit > 2022.09 (molecule handling + 2022.09 required for rdDetermineBonds)
- ProLIF (IFP similarity)
- espsim (electrostatic shape similarity)
- ase (read asedb files, useful to store xyz coordinates)
- openbabel (for molecule protonation + second bond determination option)
- vina (for Vina docking score)

## Installation
I am planning to put it on pip/conda, but for now it is a standalone
Run these commands to download genbench3d (and espsim), then install in development mode in pip:  
```bash
git clone https://github.com/bbaillif/genbench3d.git
cd genbench3d
conda env create -f environment.yml
conda activate genbench3d
git clone https://github.com/hesther/espsim.git
cd espsim
pip install -e . # install espsim in current environment
cd ..
pip install -e . # install genbench3d in current environment
```

Final step is to change relative/absolute paths in the genbench3d/params.py file. I will add config files later on in the development process, but it is a quick fix for now. LigBoundConf might need to be downloaded from [here](https://pubs.acs.org/doi/full/10.1021/acs.jcim.0c01197)

## Usage
The main usage is to compute all metrics from a list of RDKit molecules `mol_list`:
```python
from genbench3d import GenBench3D
benchmark = GenBench3D()
results = benchmark.get_results_for_mol_list(mol_list)
```

To compute metrics that depends on the training molecules (e.g. training similarity, novelty2D and 3D), you need to input the `training_mols` list when creating the benchmark object:
```python
benchmark = GenBench3D(training_mols=training_mols)
results = benchmark.get_results_for_mol_list(mol_list)
```

When dealing with structure-based (= using pocket as input) generative models, you can use the `SBGenBench3D` version to add metrics related to activity. This new object needs the Vina preparated protein (done with the `VinaProtein` class from the original `pdb_filepath`), the pocket (extracted from the protonated = "clean" protein produced during Vina processing) and native ligands (as `native_ligand` RDKit molecule) as input to estimate relative activity.
```python
from genbench3d import SBGenBench3D
from genbench3d.data.structure import VinaProtein, Protein, Pocket
vina_protein = VinaProtein(pdb_filepath=pdb_filepath)
protein_clean = Protein(vina_protein.protein_clean_filepath)
pocket = Pocket(protein=protein_clean,
                native_ligand=native_ligand)
sb_benchmark = SBGenBench3D(vina_protein, pocket, native_ligand)
results = sb_benchmark.get_results_for_mol_list(mol_list)
```

You can then aggregate all the compiled metrics:
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
| SAScore | Using RDKit implementation of SAScore TODO: put reference |
| Quantitative Estimate of Drug-likeness (QED) | Using RDKit implementation of QED TODO: put reference |

### Based on molecular 3D conformation

| Metric | Definition |
| --- | --- |
| Validity3D | Fraction of molecular conformations with valid bond lengths and valence angles (based on range of values observed in LigBoundConf, see `Validity3D` class for details) |
| Number of invalid bonds | A bond length is invalid if it is outside ranges observed in LigBoundConf |
| Number of invalid angles | A valence angle value is invalid if it is outside ranges observed in LigBoundConf |
| Uniqueness3D | Fraction of (3D-valid by default) unique conformations based on Torsion Fingerprint Deviation (TFD) with a default threshold of 0.2. Threshold can be input in `GenBench3D` with the `tfd_threshold` argument |
| Novelty3D | Fraction of (3D-valid by default) conformations different from those observed in the training set (only on the set of common molecules between training and generated)  |
| Diversity3D | Average interconformation deviation computed using the TFD|
| MMFF94s strain energy | Using up to 1000 minimization step, computed using the MMFF94s RDKit implementation |

### Structure-based metrics (based on a given protein pocket and native ligand)

| Metric | Definition |
| --- | --- |
| Absolute Vina score | Using Vina Python package |
| Vina score relative to test ligand | Using Vina Python package |
| Interaction FingerPrint (IFP) similarity to test ligand | Using ProLiF Python package |
| Electrostatic shape similarity (ESPSIM) to test ligand | Using ESPSIM Python package |

## Set of generated molecules

Will be uploaded soon to reproduce poster/presentation results

## Details
Under the hood, the GenBench3D is transforming the molecule list (generated molecules or training molecules) into a `ConfEnsembleLibrary`, a structure that groups the conformations of the same molecule (i.e. molecule topological graph and stereochemistry) into unique `ConfEnsemble` (wrapper around a single RDKit molecule having multiple Conformer), under a default name that is the SMILES representation
```python
from conf_ensemble import ConfEnsembleLibrary
cel = ConfEnsembleLibrary.from_mol_list(mol_list)
```
`cel.library` is a dictionary in the form {SMILES (or other specified name): conf_ensemble}  
`conf_ensemble.mol` gives the RDKit molecule containing each listed conformation of that molecule in the original `mol_list`

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
