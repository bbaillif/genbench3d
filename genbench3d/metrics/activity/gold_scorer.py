import os
import tempfile

from rdkit import Chem
from rdkit.Chem import Mol
from ccdc.docking import Docker
from ccdc.io import Entry, EntryWriter
from genbench3d.utils import rdkit_conf_to_ccdc_mol
from typing import List

class GoldScorer():
    
    def __init__(self,
                 protein_path: str,
                 native_ligand: Mol,
                 ) -> None:
        self.docker = Docker()
        settings = self.docker.settings
        self.temp_dir = tempfile.mkdtemp()
        settings.output_directory = self.temp_dir
        settings.fitness_function = None
        settings.rescore_function = 'plp'
        
        settings.add_protein_file(protein_path)

        protein = settings.proteins[0]
        native_ligand_ccdc = rdkit_conf_to_ccdc_mol(native_ligand)
        settings.binding_site = settings.BindingSiteFromLigand(protein=protein, 
                                                               ligand=native_ligand_ccdc, 
                                                               distance=10.)
        
    def score_mols(self,
                   mols: List[Mol]):
        self.docker.settings.clear_ligand_files()
        for i, gen_mol in enumerate(mols):
            ligand = rdkit_conf_to_ccdc_mol(gen_mol)
            ligand.identifier = f'Ligand_{i}'
            ligand_entry = Entry.from_molecule(ligand)

            ligand_preparation = Docker.LigandPreparation()
            prepared_ligand = ligand_preparation.prepare(ligand_entry)
            prepared_ligand_filename = os.path.join(self.temp_dir, f'test_ligand_{i}.mol2')
            with EntryWriter(prepared_ligand_filename) as writer:
                writer.write(prepared_ligand)
            self.docker.settings.add_ligand_file(file_name=prepared_ligand_filename)

        self.docker.dock()
        
        scores = []
        for docked_ligand in self.docker.results.ligands:
            score = docked_ligand.attributes['Gold.PLP.Fitness']
            scores.append(float(score))
            
        return scores