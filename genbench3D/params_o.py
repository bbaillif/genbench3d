import os

ROOT_DIRPATH = '/home/bb596/hdd'
if not os.path.exists(ROOT_DIRPATH):
    os.mkdir(ROOT_DIRPATH)
    
BENCHMARK_DIRNAME = 'ThreeDGenMolBenchmark'
BENCHMARK_DIRPATH = os.path.join(ROOT_DIRPATH,
                                 BENCHMARK_DIRNAME)
if not os.path.exists(BENCHMARK_DIRPATH):
    os.mkdir(BENCHMARK_DIRPATH)

DATA_DIRNAME = 'data'
DATA_DIRPATH = os.path.join(BENCHMARK_DIRPATH, DATA_DIRNAME)
if not os.path.exists(DATA_DIRPATH):
    os.mkdir(DATA_DIRPATH)

PREPARE_RECEPTOR_BIN_PATH = '/home/bb596/ADFRsuite/bin/prepare_receptor'

LIGBOUNDCONF_MINIMIZED_FILENAME = 'LigBoundConf/minimized/S2_LigBoundConf_minimized.sdf'
LIGBOUNDCONF_MINIMIZED_FILEPATH = os.path.join(ROOT_DIRPATH, LIGBOUNDCONF_MINIMIZED_FILENAME)

SCHRODINGER_PATH = '/usr/local/shared/schrodinger/current/'
GLIDE_OUTPUT_DIRPATH = '/home/bb596/genbench3d/glide_working_dir/'

CSD_DRUG_SUBSET_PATH = os.path.join(BENCHMARK_DIRPATH, 'CSD_Drug_Subset.gcd')

CROSSDOCKED_PATH = os.path.join(ROOT_DIRPATH, 'CrossDocked/')
CROSSDOCKED_DATA_PATH = os.path.join(ROOT_DIRPATH, 'crossdocked_v1.1_rmsd1.0/crossdocked_v1.1_rmsd1.0/')
CROSSDOCKED_POCKET10_PATH = os.path.join(CROSSDOCKED_PATH, 'crossdocked_pocket10/')
CROSSDOCKED_SPLITS_PT_PATH = os.path.join(CROSSDOCKED_PATH, 'split_by_name.pt')
CROSSDOCKED_SPLITS_P_PATH = os.path.join(CROSSDOCKED_PATH, 'split_by_name.p')

MINIMIZED_DIRNAME = 'minimized/'
MINIMIZED_DIRPATH = os.path.join(BENCHMARK_DIRPATH, MINIMIZED_DIRNAME)

MIN_PATTERN_VALUES = 50
DEFAULT_TFD_THRESHOLD = 0.2
Q_VALUE_THRESHOLD = 0.001
CLASH_SAFETY_RATIO = 0.75
MAX_RING_PLANE_DISTANCE = 0.1
CONSIDER_HYDROGENS = False
INCLUDE_TORSIONS = False
ADD_MINIMIZED_DOCKING_SCORES = True
OVERWRITE = False

TARGETDIFF_RESULTS_FILEPATH = os.path.join(BENCHMARK_DIRPATH, 'targetdiff/targetdiff_vina_docked.p')
THREEDSBDD_GEN_DIRPATH = os.path.join(BENCHMARK_DIRPATH, 'AR/test_set/')
POCKET2MOL_GEN_DIRPATH = os.path.join(BENCHMARK_DIRPATH, 'Pocket2Mol/test_set/')
DIFFSBDD_GEN_DIRPATH =  os.path.join(BENCHMARK_DIRPATH, 'DiffSBDD/crossdocked_fullatom_joint/')
LIGAN_GEN_DIRPATH = os.path.join(BENCHMARK_DIRPATH, 'LiGAN/molecules/')
RESGEN_GEN_DIRPATH = os.path.join(BENCHMARK_DIRPATH, 'ResGen/test_set/')