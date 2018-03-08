#__author__ = "Niccolò Pretto"
#__email__ = "niccolo.pretto_at_dei.unipd.it"
#__copyright__ = "Copyright 2018, Università degli Studi di Padova, Universitat Pompeu Fabra"
#__license__ = "GPL"
#__version__ = "0.1"

import os
import json
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from music21 import *

from shutil import copyfile
from utilities.constants import *
from utilities.recordingcomputation import *
from external_utilities.corpusbasestatistics import *

# ---------------------------------------- DATASET -------------------------------------------------

TAB_DATASET = [1, 2, 3, 5, 9, 10, 14, 16, 18]
TAB_FUNDAMENTAL_GRADE_IT = ["DO", "RE", "RE", "RE", "DO", "RE", "SOL", "RE", "DO"]
TAB_FUNDAMENTAL_GRADE = [ "C", "D", "D", "D", "C", "D", "G", "D", "C"]
ATTRIBUTES = ["mbid", "tab", "tonic_not_filt", "tonic_filt", "tonic_sec", "verified", "observation"]

class DataSet:

    def __init__(self, cm, tabs_list, num_recording_per_tab, not_valid_rmbid_list):

        df_tab = cm.get_dataframe(DF_LISTS[1])
        # avoid too long list
        if len(tabs_list) > len(df_tab.index.values):
            raise Exception("tab_list contains to many values: " + str(len(tabs_list)))

        # avoid duplicates
        if len(set(tabs_list)) != len(tabs_list):
            raise Exception("tab_list contains duplicates")

        # avoid wrong entry
        for tab_index in tabs_list:
            if not (tab_index in df_tab.index.values):
                raise Exception("tab index " + str(tab_index) + " do not exist")

        self.df_dataset = pd.DataFrame(columns=ATTRIBUTES)
        tabs_matrix = list()

        # for every dataset
        for tab_index in tabs_list:
            temp_tab_rmbid_list = cm.search_recording(tab_index, 'all', 'all', 'all')
            # the dataset could have only analyzed recording with score
            temp_tab_rmbid_list = check_files_of_rmbid_lists(RECORDINGS_DIR, temp_tab_rmbid_list, ['score', 'analysis json'],
                                                   [True, True])
            tab_rmbid_list = list()
            for rmbid in temp_tab_rmbid_list:
                if len(tab_rmbid_list) < num_recording_per_tab:
                    if rmbid != not_valid_rmbid_list:
                        tab_rmbid_list.append(rmbid)
            if len(tab_rmbid_list) < num_recording_per_tab:
                raise Exception("Tab " + str(tab_index) + " have only " +  str(len(tab_rmbid_list)))
            tabs_matrix.append(tab_rmbid_list)

        list_of_dict = list()
        for index in range(len(tabs_list)):
            for rmbid in tabs_matrix[index]:
                characteristic_list = [rmbid, tabs_list[index], \
                                       get_tonic_value(os.path.join(RECORDINGS_DIR, rmbid), FN_TONIC_NO_FILT), \
                                       get_tonic_value(os.path.join(RECORDINGS_DIR, rmbid), FN_TONIC_FILT), \
                                       get_tonic_value(os.path.join(RECORDINGS_DIR, rmbid), FN_TONIC_SEC),\
                                       False, ""]
                df_row = pd.DataFrame([characteristic_list], columns=ATTRIBUTES)
                self.df_dataset = self.df_dataset.append(df_row,  ignore_index=True)

    def export_dataset(self, experiment_dir, file_name, tonic_type, format):

        if not (tonic_type == 'all' or tonic_type in FN_TONIC_TYPE):
            raise Exception("Tonic type " + str(tonic_type) + " do not exist")

        if not(format == 'csv' or format == 'json'):
            raise Exception("Format " + str(format) + " unknown")

        attributes_list  = list()
        for attr in ATTRIBUTES:
            if not ((tonic_type == FN_TONIC_TYPE[0] and (attr == ATTRIBUTES[3] or attr == ATTRIBUTES[4])) or \
                    (tonic_type == FN_TONIC_TYPE[1] and (attr == ATTRIBUTES[2] or attr == ATTRIBUTES[4])) or \
                    (tonic_type == FN_TONIC_TYPE[2] and (attr == ATTRIBUTES[2] or attr == ATTRIBUTES[3]))):
                attributes_list.append(attr)

        #print(attributes_list)
        # create a new dataframe
        if tonic_type == 'all':
            df_temp = self.df_dataset
        else:
            df_temp = pd.DataFrame(columns = attributes_list)
            for row_index in self.df_dataset.index.values:
                row_values = list()
                for attr in attributes_list:
                    #print("[" + str(row_index) + ", " + str(attr) + "]")
                    row_values.append(self.df_dataset.loc[row_index, attr])
                df_row = pd.DataFrame([row_values], columns = attributes_list)
                df_temp = df_temp.append(df_row,  ignore_index=True)

        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)

        if format == 'csv':
            df_temp.to_csv(os.path.join(experiment_dir, file_name + '.csv'), sep=';', encoding="utf-8")
            print(str(file_name) + '.csv created')

        if format == 'json':
            df_temp.to_json(path_or_buf=os.path.join(experiment_dir, file_name + '.json') ,orient='records')
            print(str(file_name) + '.json created')

    def move_dataset_mp3(self, recordings_dir, experiment_mp3_dir):

        # list of recordings by mbid
        rmbid_list = list()
        for row_index in self.df_dataset.index.values:
            rmbid_list.append(self.df_dataset.loc[row_index, ATTRIBUTES[0]])

        # check if all the files exists
        if check_files_of_rmbid_lists(recordings_dir, rmbid_list, ['mp3', FNT_PITCH, FNT_PITCH_FILT], [True, True, True]):
            for rmbid in rmbid_list:
                FILE_NAMES = [rmbid + '.mp3', FNT_PITCH, FNT_PITCH_FILT]
                for i in range(len(FILE_NAMES)):
                    origin_file = os.path.join(recordings_dir, rmbid, FILE_NAMES[i])
                    destination_dir = os.path.join(experiment_mp3_dir, rmbid)
                    if not os.path.exists(destination_dir):
                        os.makedirs(destination_dir)
                    destination_file = os.path.join(destination_dir, FILE_NAMES[i])
                    print(destination_file)
                    copyfile(origin_file, destination_file)
            print("All files are copied in the new directory")
        else:
            raise Exception("Some requested files do not exist")

