# GenBench3D
Benchmarking deep learning models generating molecules in 3D

# Installation
TODO, I am planning to put it on pip/conda, but for now it is a standalone

# Usage
The main usage is to compute all metrics from a list of RDKit molecules `mol_list`:
```python
from genbench3d import GenBench3D
benchmark = GenBench3D()
metrics = benchmark.get_metrics_for_mol_list(mol_list)
```

To compute metrics that depends on the training molecules (e.g. training similarity, novelty2D and 3D), you need to input the `training_mols` list when creating the benchmark object:
```python
benchmark = GenBench3D(training_mols=training_mols)
metrics = benchmark.get_metrics_for_mol_list(mol_list)
```

When dealing with structure-based (= using pocket as input) generative models, you can use the `SBGenBench3D` version to add metrics related to activity. This new object needs the protein (as `original_structure_path` pdb filepath, and currently as an `clean_mda_prot` MDAnalysis Universe) and native ligands (as `native_ligand` RDKit molecule) as input to estimate relative activity.
```python
from genbench3d import SBGenBench3D
sb_benchmark = SBGenBench3D(original_structure_path, clean_mda_prot, native_ligand)
metrics = sb_benchmark.get_metrics_for_mol_list(mol_list)
```

# Implemented metrics

## Based on topological graph (traditionally used to evaluate SMILES or molecular graph generators)

| Metric | Definition |
| --- | --- |
| Validity2D | Fraction of molecules parsed by RDKit (the n_total_mols can be setup manually in all get_metrics functions) |
| Uniqueness2D | Fraction of unique molecules (based on SMILES with stereochemistry embeded) |
| Novelty2D | Fraction of molecules with a stereo-SMILES not in training molecules (if input) |
| Diversity2D | Average Tanimoto dissimilarity (1 - similarity) of Morgan Fingerprints radius 3 with stereochemistry considered between all generated molecules|
| Median molecular weight (MW) | Self-explanatory |
| Median logP | Using RDKit |
| Median SAScore | Using RDKit implementation of SAScore TODO: put reference |
| Median Quantitative Estimate of Drug-likeness (QED) | Using RDKit implementation of QED TODO: put reference |

## Based on molecular 3D conformation

| Metric | Definition |
| --- | --- |
| Validity3D | Fraction of molecular conformations with valid bond lengths and valence angles (based on range of values observed in LigBoundConf, see `Validity3D` class for details) |
| Average number of invalid bonds | Based on LigBoundConf observed ranges |
| Average number of invalid angles | Based on LigBoundConf observed ranges |
| Uniqueness3D | Fraction of (3D-valid by default) unique conformations based on Torsion Fingerprint Deviation (TFD) with a default threshold of 0.2. Threshold can be input in `GenBench3D` with the `tfd_threshold` argument |
| Novelty3D | Fraction of (3D-valid by default) conformations different from those observed in the training set (only on the set of common molecules between training and generated)  |
| Diversity3D | Average interconformation deviation computed using the TFD|
| Median MMFF94s strain energy | Using 1000 minimization step, computed using the MMFF94s RDKit implementation |

## Structure-based metrics (based on a given protein pocket and native ligand)

| Metric | Definition |
| --- | --- |
| Median absolute Vina score | Using Vina Python package |
| Median Vina score relative to test ligand | Using Vina Python package |
| Median Interaction FingerPrint (IFP) similarity to test ligand | Using ProLiF Python package |
| Median Electrostatic shape similarity (ESPSIM) to test ligand | Using ESPSIM Python package |


# Details
Under the hood, the GenBench3D is transforming the molecule list (generated molecules and training molecules) into a `ConfEnsembleLibrary`, a structure that groups the conformations of the same molecule (i.e. molecule topological graph and stereochemistry) into unique `ConfEnsemble` (wrapper around a single RDKit molecule), under a default name that is the SMILES representation
```python
from conf_ensemble import ConfEnsembleLibrary
cel = ConfEnsembleLibrary.from_mol_list(mol_list)
```
`cel.library` is a dictionary in the form {SMILES: conf_ensemble}
`conf_ensemble.mol` gives the RDKit molecule containing each listed conformation of that molecule in the original `mol_list`

You have different ways on inputing molecules in the benchmark.
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
```# 3DGenMolBenchmark
Benchmarking deep learning models generating molecules in 3D
