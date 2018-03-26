#__author__ = "Niccolò Pretto"
#__email__ = "niccolo.pretto_at_dei.unipd.it"
#__copyright__ = "Copyright 2018, Università degli Studi di Padova, Universitat Pompeu Fabra"
#__license__ = "GPL"
#__version__ = "0.1"

import os

# change parameters here to customize the directories where data will be stored.
# NB: the source directory is the main directory of the notebook
DATA_DIR = 'data'
RECORDINGS_DIR = os.path.join(DATA_DIR, 'documents')
EXPERIMENT_DIR = os.path.join(DATA_DIR, 'experiment')
EXPERIMENT_RECORDINGS_DIR = os.path.join(EXPERIMENT_DIR, 'documents')

# prefix applied to json file with the information from Dunya or derived by them
PREFIX_JSON = 'andalusian_'

DF_LISTS = ['recording', 'tab', 'nawba', 'mizan', 'form', 'description']
COLUMNS_NAMES = ['name', 'transliterated_name']
COLUMNS_RECORDINGS = ['title', 'transliterated_title', 'archive_url', 'musescore_url']
COLUMNS_DESCRIPTION = ['mbid', 'section', 'tab', 'nawba', 'mizan', 'form', 'start_time', 'end_time', 'duration']
STATISTIC_TYPE = ['# recordings', '# sections', 'overall sections time', 'avg sections time']

# File name in Dunya-Desktop
DUNYA_PREFIX_ANALYSIS = 'audioanalysis--'
FN_PITCH = DUNYA_PREFIX_ANALYSIS + 'pitch.json'
FN_TONIC_NO_FILT = DUNYA_PREFIX_ANALYSIS + 'tonic_no_filt.json'
FN_TONIC_FILT = DUNYA_PREFIX_ANALYSIS + 'tonic_filt.json'
FN_TONIC_SEC = DUNYA_PREFIX_ANALYSIS + 'tonic_sec.json'
FN_PD = DUNYA_PREFIX_ANALYSIS + 'pitch_distribution.json'
FN_PITCH_FILT = DUNYA_PREFIX_ANALYSIS + 'pitch_filtered.json'
FN_METADATA = DUNYA_PREFIX_ANALYSIS + 'metadata.json'
FN_TONIC_TYPE = [FN_TONIC_NO_FILT, FN_TONIC_FILT, FN_TONIC_SEC]

# File name for text file
FNT_PITCH = 'pitch.txt'
FNT_PITCH_FILT = 'pitch_filtered.txt'
XML_SUFFIX = '-symbtrxml.xml'

ACCEPTED_TYPE = ['mp3', 'score', 'wav', FN_PITCH, FNT_PITCH, FN_TONIC_NO_FILT, FN_TONIC_FILT, FN_TONIC_SEC, FN_PITCH_FILT, FNT_PITCH_FILT,
                 FN_PD, FN_METADATA]

# Path in Dunya
API_PATH = 'api/andalusian/'
DUNYA_TOKEN = '1caad80ef12b4d47accfd26b01c9062da834825f' # put your DUNYA_TOKEN here

# GUI
OPTION_LIST = ['mp3', 'score', 'MB metadata', 'analysis json', 'analysis text', 'wav']

# distance measures for experiments
DISTANCE_MEASURES = ["city block (L1)", "euclidian (L2)", "correlation", "canberra"]
