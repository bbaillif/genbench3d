import prolif as plf
import logging
import numpy as np

from genbench3d.conf_ensemble import GeneratedCEL
from ..metric import Metric
from rdkit.Chem import Mol
from rdkit import Chem
from rdkit import DataStructs
from collections import defaultdict
from MDAnalysis import Universe


class IFPSimilarity(Metric):
    
    def __init__(self, 
                 universe: Universe,
                 native_ligand: Mol,
                 name: str = 'IFP similarity') -> None:
        super().__init__(name)
        self._universe = universe
        self.prolif_mol = plf.Molecule.from_mda(self._universe)
        self.native_ligand = native_ligand
        self.ifp_sims = None
        
        
    def get(self, 
            cel: GeneratedCEL) -> float:
        all_ifp_sims = []
        self.ifp_sims = defaultdict(list)
        all_mols = []
        mol_names = []
        for name, ce in cel.items():
            mols = ce.to_mol_list()
            all_mols.extend(mols)
            mol_names.extend([name for _ in range(len(mols))])
        
        try:
            ligands = [Chem.AddHs(ligand, addCoords=True) for ligand in all_mols]
            
            fp = plf.Fingerprint()
            lig_list = [plf.Molecule.from_rdkit(self.native_ligand)] \
                + [plf.Molecule.from_rdkit(ligand) for ligand in ligands]
            fp.run_from_iterable(lig_iterable=lig_list,
                                 prot_mol=self.prolif_mol, 
                                 progress=False)
            df = fp.to_dataframe()
            
            bvs = plf.to_bitvectors(df)
            assert len(mol_names) == len(bvs[1:])
            for i, bv in enumerate(bvs[1:]):
                name = mol_names[i]
                ifp_sim = DataStructs.TanimotoSimilarity(bvs[0], bv)
                self.ifp_sims[name].append(ifp_sim)
                all_ifp_sims.append(ifp_sim)
        except Exception as e:
            logging.warning(f'IFP computation error: {e}')
            # import pdb;pdb.set_trace()
            
        return all_ifp_sims