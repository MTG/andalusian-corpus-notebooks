#__author__ = "Niccolò Pretto"
#__email__ = "niccolo.pretto_at_dei.unipd.it"
#__copyright__ = "Copyright 2018, Università degli Studi di Padova, Universitat Pompeu Fabra"
#__license__ = "GPL"
#__version__ = "0.1"

import os
import json
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from music21 import *

from sklearn.model_selection import train_test_split
from shutil import copyfile
from utilities.constants import *
from utilities.recordingcomputation import *
from external_utilities.corpusbasestatistics import *

# ---------------------------------------- DATASET -------------------------------------------------

#TAB_DATASET = [1, 2, 3, 5, 9, 10, 14, 16, 18]
#TAB_FUNDAMENTAL_GRADE_IT = ["DO", "RE", "RE", "RE", "DO", "RE", "SOL", "RE", "DO"]
#TAB_FUNDAMENTAL_GRADE = [ "C", "D", "D", "D", "C", "D", "G", "D", "C"]
ATTRIBUTES = ["mbid", "tab", "tonic_not_filt", "tonic_filt", "tonic_sec", "verified", "observation"]
DATASET_ATTRIBUTES = ["tab", "scale"]
DISTANCE_MEASURES = ["city block (L1)", "euclidian (L2)", "correlation", "intersection", "camberra"]

class DataSet:

    def __init__(self, cm, exp_name, rmbid_list, tab_list, scale_division):
        self.cm = cm
        self.tab_list = tab_list
        self.scale_division = scale_division
        # create a directory for the experiment
        exp_dir = os.path.join(EXPERIMENT_DIR, exp_name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
            print("Directory {} created".format(exp_dir))
        self.exp_dir = exp_dir
        self.exp_name = exp_name

        # create a dataframe from the rmbid_list checking if the tab of every rmbid is in
        # the tab list passed as parameter
        self.df_dataset = pd.DataFrame(columns = DATASET_ATTRIBUTES, index=rmbid_list)
        for rmbid in rmbid_list:
            rmbid_tab = cm.get_characteristic(rmbid, COLUMNS_DESCRIPTION[2])
            if not (rmbid_tab in tab_list):
                raise Exception("Tab {} of the recording {} is not included in the experiment".format(rmbid_tab, rmbid))
            rmbid_scale = self.get_scale_index_from_tab(rmbid_tab)
            self.df_dataset.loc[rmbid, DATASET_ATTRIBUTES[0]] = rmbid_tab
            self.df_dataset.loc[rmbid, DATASET_ATTRIBUTES[1]] = rmbid_scale

    def get_scale_index_from_tab(self, tab):
        ''' Return the scale related to tab passed as parameter

        :param tab: index of the tab
        :return: related scale
        '''
        list_of_scale_index = list()
        for i in range(len(self.scale_division)):
            if tab in self.scale_division[i]:
                list_of_scale_index.append(i)

        if len(list_of_scale_index) == 0:
            raise Exception("Tab {} has not scale information".format(tab))
        if len(list_of_scale_index) > 1:
            raise Exception("Tab {} has more than one scale".format(tab))
        return list_of_scale_index[0]

    def get_tab_list(self):
        ''' Return the list of tab included in the experiment

        :return: list of tab indexes
        '''
        return self.tab_list

    def get_list_of_recording_per_tab(self, tab):
        ''' Return the list of the rmbid in the dataset with a specified tab

        :param tab_index: index of the tab
        :return: list of rmbid
        '''
        rmbid_list = list()
        for rmbid in self.df_dataset.index.values.tolist():
            if self.df_dataset.loc[rmbid, DATASET_ATTRIBUTES[0]] == tab:
                rmbid_list.append(rmbid)
        return rmbid_list

    def get_dataset_dataframe(self):
        return self.df_dataset

class Tab_Scale_Recognition_Experiment:

    def __init__(self, dataset_object, test_size, random_state, standard_deviation_list):

        self.do = dataset_object
        self.df_dataset = dataset_object.get_dataset_dataframe()
        self.std_list = standard_deviation_list
        self.distance_measure = DISTANCE_MEASURES

        # divide the dataset in training set and test set
        X = list()
        y = list()
        for rmbid in self.df_dataset.index.values.tolist():
            X.append(rmbid)
            y.append(self.df_dataset.loc[rmbid, DATASET_ATTRIBUTES[0]])

        self.X_mbid_train, self.X_mbid_test, self.y_train, self.y_test = train_test_split(X, y, stratify=y,\
                                                                                          test_size=test_size,\
                                                                                            random_state=random_state)
        # create the dataframe that will contain the results of the experiments
        self.experiment_attributes = list()
        for attr in DATASET_ATTRIBUTES:
            self.experiment_attributes.append("{}".format(attr))
            for measure in DISTANCE_MEASURES:
                self.experiment_attributes.append("{}-{}".format(attr, measure))

        self.summary_attributes = copy.deepcopy(self.experiment_attributes)
        self.summary_attributes.remove(DATASET_ATTRIBUTES[0])
        self.summary_attributes.remove(DATASET_ATTRIBUTES[1])

        # one dataframe for every standard deviation
        self.df_experiment_list = list()
        for std in standard_deviation_list:
            self.df_experiment_list.append(pd.DataFrame(columns=self.experiment_attributes, index = self.X_mbid_test))

        self.df_summary = pd.DataFrame(0, columns=self.summary_attributes, index = standard_deviation_list)

        # create the histograms of the duration of the notes
        self.notes_avg_tab_list, self.y_avg_tab_list = compute_folded_avg_scores(self)

    def get_train_mbid_by_tab(self, tab):
        rmbid_list = list()
        for i in range(len(self.y_train)):
            if self.y_train[i] == tab:
                rmbid_list.append(self.X_mbid_train[i])
        return rmbid_list

    def get_test_mbid_by_tab(self, tab):
        rmbid_list = list()
        for i in range(len(self.y_test)):
            if self.y_test[i] == tab:
                rmbid_list.append(self.X_mbid_test[i])
        return rmbid_list

    def get_test_dataset(self):
        rmbid_list = list()
        for i in range(len(self.y_test)):
            rmbid_list.append(self.X_mbid_test[i])
        return rmbid_list, self.y_test

    def get_tab_list(self):
        return self.do.get_tab_list()

    def run(self):
        counter_1 = 0
        counter_2 = 0
        for i in range(len(self.std_list)):

            # convert notes histograms in models
            x_model, y_models_list = convert_folded_scores_in_models(self.y_avg_tab_list, self.std_list[i])

            # for every recording in the test set
            for rmbid in self.df_experiment_list[i].index.values.tolist():
                self.df_experiment_list[i].loc[rmbid, DATASET_ATTRIBUTES[0]] = self.do.df_dataset.loc[rmbid, DATASET_ATTRIBUTES[0]]
                self.df_experiment_list[i].loc[rmbid, DATASET_ATTRIBUTES[1]] = self.do.df_dataset.loc[rmbid, DATASET_ATTRIBUTES[1]]

                for distance_type in DISTANCE_MEASURES:
                    resulting_tab = get_tab_using_models_from_scores(self, rmbid, y_models_list, distance_type)
                    column_tab = "{}-{}".format(DATASET_ATTRIBUTES[0], distance_type)
                    self.df_experiment_list[i].loc[rmbid,column_tab] = resulting_tab
                    resulting_scale = self.do.get_scale_index_from_tab(resulting_tab)
                    column_scale = "{}-{}".format(DATASET_ATTRIBUTES[1], distance_type)
                    self.df_experiment_list[i].loc[rmbid, column_scale] = resulting_scale

            print(self.std_list[i])
            print(self.df_experiment_list[i])

    def compute_summary(self):

        for i in range(len(self.std_list)):
            for distance_type in DISTANCE_MEASURES:
                for rmbid in self.df_experiment_list[i].index.values.tolist():
                    tab_attributes = "{}-{}".format(DATASET_ATTRIBUTES[0], distance_type)
                    scale_attributes = "{}-{}".format(DATASET_ATTRIBUTES[1], distance_type)
                    resulting_tab = self.df_experiment_list[i].loc[rmbid, tab_attributes]
                    resulting_scale = self.df_experiment_list[i].loc[rmbid, scale_attributes]
                    correct_tab = self.df_experiment_list[i].loc[rmbid, DATASET_ATTRIBUTES[0]]
                    correct_scale = self.df_experiment_list[i].loc[rmbid, DATASET_ATTRIBUTES[1]]
                    if resulting_tab == correct_tab:
                        self.df_summary.loc[self.std_list[i], tab_attributes] +=1
                    if resulting_scale == correct_scale:
                        self.df_summary.loc[self.std_list[i], scale_attributes] += 1
        self.df_summary.loc[:,:] /= len(self.df_experiment_list[0].index.values.tolist())
        print(self.df_summary)

# -------------------------------------------------- OLD --------------------------------------------------

    def create_equal_distributed_tabs_dataset(self, tabs_list, num_recording_per_tab, not_valid_rmbid_list):
        ''' Create a dataset of random recordings by using the tabs contained in the list passed by input.
        The number of element for every tab is a parameter of a function. The recording indicated in
        not_valid_rmbid_list are not considered in the results

        :param tabs_list: list of tab indexes that will be used in the dataset
        :param num_recording_per_tab: number of recording for each tab
        :param not_valid_rmbid_list: list of rmbid to not include in the dataset
        '''
        df_tab = self.cm.get_dataframe(DF_LISTS[1])
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

        # for every tab in the dataset
        for tab_index in tabs_list:
            temp_tab_rmbid_list = self.cm.search_recording(tab_index, 'all', 'all', 'all')
            # the dataset could have only analyzed recording with score
            temp_tab_rmbid_list = check_files_of_rmbid_lists(RECORDINGS_DIR, temp_tab_rmbid_list, ['score', 'analysis json'],
                                                   [True, True])
            # delete not valid rmbid
            a = set(temp_tab_rmbid_list)
            b = set(not_valid_rmbid_list)
            temp_tab_rmbid_list = a-b
            if len(temp_tab_rmbid_list) < num_recording_per_tab:
                raise Exception("Tab " + str(tab_index) + " have only " + str(len(temp_tab_rmbid_list)))
            # select randomly the elements from the list
            temp_tab_rmbid_list = random.sample(temp_tab_rmbid_list, num_recording_per_tab)
            # append it to the list
            tabs_matrix.append(list(temp_tab_rmbid_list))

        for index in range(len(tabs_list)):
            for rmbid in tabs_matrix[index]:
                characteristic_list = [rmbid, tabs_list[index], \
                                       get_tonic_value(os.path.join(RECORDINGS_DIR, rmbid), FN_TONIC_NO_FILT), \
                                       get_tonic_value(os.path.join(RECORDINGS_DIR, rmbid), FN_TONIC_FILT), \
                                       get_tonic_value(os.path.join(RECORDINGS_DIR, rmbid), FN_TONIC_SEC),\
                                       False, ""]
                df_row = pd.DataFrame([characteristic_list], columns=ATTRIBUTES)
                self.df_dataset = self.df_dataset.append(df_row,  ignore_index=True)
        self.tabs_list = tabs_list

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

    def export_dataset_csv_json(self):
        #NAME_TONIC_TYPE = ['not_filt', 'filt', 'sec']
        #for i in range(len(FN_TONIC_TYPE)):
        #    self.export_dataset(EXPERIMENT_DIR, suffix + NAME_TONIC_TYPE[i], FN_TONIC_TYPE[i], 'csv')
        #    self.export_dataset(EXPERIMENT_DIR, suffix + NAME_TONIC_TYPE[i], FN_TONIC_TYPE[i], 'json')
        self.export_dataset(self.exp_dir, self.exp_name, 'all', 'csv')# + "all"
        self.export_dataset(self.exp_dir, self.exp_name, 'all', 'json')#+ "all"

    def import_dataset_from_csv(self, path, file_name):
        complete_path = os.path.join(path,file_name)
        if not os.path.exists(complete_path):
            raise Exception("Path {} doesn't exist".format(complete_path))
        self.df_dataset = pd.read_csv(complete_path, sep = ';', encoding="utf-8", index_col=0)
        print(self.df_dataset)

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



    def save_scores_models(self, notes_avg_tab_list, y_avg_tab_list, x_model, y_models_list, param_name, param):

        for i in range(len(notes_avg_tab_list)):

            f, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 20))
            x_fake = list((i) for i in range(len(notes_avg_tab_list[i])))
            ax1.bar(x_fake, y_avg_tab_list[i], tick_label=notes_avg_tab_list[i])
            ax1.set_title("Avarage score - tab {}".format(self.tabs_list[i]))
            ax1.set_xlabel("Notes")
            ax1.set_ylabel("Occurances %")
            ax2.plot(x_model, y_models_list[i])
            ax2.set_xlabel("Cents")
            ax2.set_ylabel("Occurances %")
            ax2.set_title("Model with standard deviation {} - tab {}".format(param, self.tabs_list[i]))
            file_name = "avg_score_model-tab{}-{}{}".format(self.tabs_list[i], param_name, param)
            dir_name = os.path.join(self.exp_dir, str(param), "avg_scores_models-{}_{}".format(param_name, param))
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            f.savefig(os.path.join(dir_name, file_name))
            plt.close(f)

    def save_best_shifted_recording_plot(self, rmbid, x_s, y_s, y_s_f, shift, param_name, param, tab):
        fig = plt.figure()
        plt.plot(x_s, y_s_f, label="model{}".format(tab))
        plt.plot(x_s, y_s, label="shifted recording")

        plt.title("{} - shift {} - {} {}".format(rmbid, shift, param_name, param))
        plt.xlabel("Cents")
        plt.ylabel("Occurances")
        plt.legend()
        file_name = "{}_shift_{}".format(rmbid, shift)
        dir_name = os.path.join(self.exp_dir, str(param), "best_plot-{}_{}".format(param_name, param))
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plt.savefig(os.path.join(dir_name, file_name))
        plt.close(fig)

    def export_experiment_results_to_csv(self, df_exper, param_name, param):
        file_name = "{}-{}_{}.csv".format(self.exp_name, param_name, param)
        dir_name = os.path.join(self.exp_dir, str(param), "results")
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        df_exper.sort_values(by=['tab']).to_csv(os.path.join(dir_name, file_name), sep=';', encoding="utf-8")