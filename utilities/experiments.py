#__author__ = "Niccolò Pretto"
#__email__ = "niccolo.pretto_at_dei.unipd.it"
#__copyright__ = "Copyright 2018, Università degli Studi di Padova, Universitat Pompeu Fabra"
#__license__ = "GPL"
#__version__ = "0.1"

import os
import json
import copy
import random
import itertools
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from music21 import *
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from shutil import copyfile
from utilities.constants import *
from utilities.recordingcomputation import *
from external_utilities.corpusbasestatistics import *

font = {'size'   : '40'}

matplotlib.rc('font', **font)
# ---------------------------------------- DATASET -------------------------------------------------

DATASET_ATTRIBUTES = [DF_LISTS[2]]

class DataSet:

    def __init__(self, cm, rmbid_list = []):
        ''' Constructor to create an object Dataset

        :param cm: Metadata Object
        :param rmbid_list: list of recording mbids
        '''

        self.cm = cm
        self.rmbid_list = rmbid_list
        self.nawba_list = []

        # create an empty dataframe
        self.df_dataset = pd.DataFrame(columns=DATASET_ATTRIBUTES)

        # check if the mbid list of the recordings are empty
        if len(rmbid_list) > 0:
            self.add_rmbid_list_to_dataframe(rmbid_list)

    def add_rmbid_list_to_dataframe(self, rmbid_list):
        ''' Add a list of MusibBrainz id to the dataframe of the Dataset object

        :param rmbid_list: list of music brainz id
        '''

        # check if the list of mbid in the corpora
        self.rmbid_list += rmbid_list
        correct_list, incorrect_list = self.cm.check_rmbid_list_before_download(rmbid_list)
        if len(incorrect_list) > 0:
            raise Exception("The mbid\s {} are not in Dunya".format(incorrect_list))
        for rmbid in rmbid_list:
            # get the nawba information
            rmbid_nawba = self.cm.get_characteristic(rmbid, DATASET_ATTRIBUTES[0])
            self.df_dataset.loc[rmbid, DATASET_ATTRIBUTES[0]] = int(rmbid_nawba)
            if not (rmbid_nawba in self.nawba_list):
                self.nawba_list.append(rmbid_nawba)
        self.nawba_list = sorted(self.nawba_list)

    def get_nawba_list(self):
        ''' Return the list of nawba included in the experiment

        :return: list of nawba indexes
        '''
        return self.nawba_list

    def get_list_of_recording_per_nawba(self, nawba_index):
        ''' Return the list of the rmbid in the dataset with a specified nawba

        :param nawba_index: index of the nawba
        :return: list of rmbid
        '''
        rmbid_list = list()
        for rmbid in self.df_dataset.index.values.tolist():
            if self.df_dataset.loc[rmbid, DATASET_ATTRIBUTES[0]] == nawba_index:
                rmbid_list.append(rmbid)
        return rmbid_list

    def get_dataset_dataframe(self):
        ''' the entire dataframe with rmbid and related nawba

        :return: pandas dataframe
        '''
        return self.df_dataset

    def get_rmbid_list(self):
        ''' Get the list of the recordings in the dataset

        :return:
        '''
        return self.rmbid_list

    def import_dataset_from_csv(self, path):
        rmbid_list = self.cm.import_rmbid_list_from_file(path)
        self.add_rmbid_list_to_dataframe(rmbid_list)

class Nawba_Recognition_Experiment:

    def __init__(self, dataset_object, i_element, random_state, standard_deviation_list, distance_measure_list, experiment_name, path_dir):
        ''' Initialize the Experiment Object

        :param dataset_object: Dataset object with the information about the dataset used in the experiment
        :param i_element: index to select the recordings that will be part of the test set
        :param random_state: index to reproduce the pseudo-random order of an experiment
        :param standard_deviation_list: list of standard deviation values used to create the templates in the experiment
        :param distance_measure_list: list of distance measure used to recognize the nawba of a recording
        :param experiment_name: string of the experiment name
        :param path_dir: path of the directory where will be saved all the results of the experiment
        '''

        self.do = dataset_object
        self.df_dataset = dataset_object.get_dataset_dataframe()
        self.std_list = standard_deviation_list

        # check of the distance measure
        for distance in distance_measure_list:
            if not (distance in DISTANCE_MEASURES):
                raise Exception("Distance measure {} is not part of the experiment".format(distance))

        self.distance_measure_list = distance_measure_list
        self.experiment_name = experiment_name
        self.source_dir = path_dir

        # create a directory for the experiment
        experiment_dir = os.path.join(path_dir, experiment_name)
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        self.experiment_dir = experiment_dir

        # # divide the dataset in training set and test set
        # X = list()
        # y = list()
        # for rmbid in self.df_dataset.index.values.tolist():
        #     X.append(rmbid)
        #     y.append(self.df_dataset.loc[rmbid, DATASET_ATTRIBUTES[0]])
        #
        # self.X_mbid_train, self.X_mbid_test, self.y_train, self.y_test = train_test_split(X, y, stratify=y,\
        #                                                                                   test_size=test_size,\
        #                                                                                     random_state=random_state)

        # divide the dataset in training set and test set
        self.X_mbid_train = list()
        self.X_mbid_test = list()
        self.y_train = list()
        self.y_test = list()

        for i in range(len(self.do.get_nawba_list())):
            temp_list_single_nawba = self.df_dataset.index.values.tolist()[i*7:(i+1)*7]
            temp_list_single_nawba = shuffle(temp_list_single_nawba, random_state=random_state)
            for j in range(len(temp_list_single_nawba)):
                if j == i_element:
                    self.X_mbid_test.append(temp_list_single_nawba[j])
                else:
                    self.X_mbid_train.append(temp_list_single_nawba[j])

        for mbid in self.X_mbid_train:
            self.y_train.append(self.df_dataset.loc[mbid, "nawba"])

        for mbid in self.X_mbid_test:
            self.y_test.append(self.df_dataset.loc[mbid, "nawba"])

        # create the dataframe that will contain the results of the experiments
        self.experiment_attributes = list()
        for attr in DATASET_ATTRIBUTES:
            self.experiment_attributes.append("{}".format(attr))
            for measure in self.distance_measure_list:
                self.experiment_attributes.append("{}-{}".format(attr, measure))

        self.summary_attributes = list()
        for attr in DATASET_ATTRIBUTES:
            for measure in self.distance_measure_list:
                self.summary_attributes.append("{}-{}".format(attr, measure))

        # one dataframe for every standard deviation
        self.df_experiment_list = list()
        for std in standard_deviation_list:
            self.df_experiment_list.append(pd.DataFrame(columns=self.experiment_attributes, index = self.X_mbid_test))

        self.df_summary = pd.DataFrame(0, columns=self.summary_attributes, index = standard_deviation_list)

        # create the histograms of the duration of the notes
        self.notes_avg_nawba_list, self.y_avg_nawba_list = compute_folded_avg_scores(self)

    def get_train_mbid_by_nawba(self, nawba):
        ''' Return the list of recordings in the training set belong to a selected nawba

        :param nawba: nawba index
        :return: list of MusicBrainz ids in the training set belong to a selected nawba
        '''

        rmbid_list = list()
        for i in range(len(self.y_train)):
            if self.y_train[i] == nawba:
                rmbid_list.append(self.X_mbid_train[i])
        return rmbid_list

    def get_test_mbid_by_nawba(self, nawba):
        ''' Return the list of recordings in the test set belong to a selected nawba

        :param nawba: nawba index
        :return: list of MusicBrainz ids in the test set belong to a selected nawba
        '''

        rmbid_list = list()
        for i in range(len(self.y_test)):
            if self.y_test[i] == nawba:
                rmbid_list.append(self.X_mbid_test[i])
        return rmbid_list

    def get_test_dataset(self):
        ''' Return the list of recordings in the test set

        :return: list of musicbrainz ids
        '''
        rmbid_list = list()
        for i in range(len(self.y_test)):
            rmbid_list.append(self.X_mbid_test[i])
        return rmbid_list, self.y_test

    def get_nawba_list(self):
        ''' List of nawbat considered in the experiment

        :return: list of musicbrainz ids
        '''
        return self.do.get_nawba_list()

    def run(self, plot_flag = False):
        ''' Function to run the nawba recognition experiment
        '''
        for i in range(len(self.std_list)):

            # convert pitch class distribution in templates
            x_model, y_models_list = convert_folded_scores_in_models(self.y_avg_nawba_list, self.std_list[i])
            # save figures
            template_dir = os.path.join(self.experiment_dir, str(self.std_list[i]) ,"templates")
            if plot_flag:
                self.save_scores_models(self.notes_avg_nawba_list, self.y_avg_nawba_list, x_model, y_models_list, self.std_list[i], template_dir)

            # for every recording in the test set
            for rmbid in self.df_experiment_list[i].index.values.tolist():
                self.df_experiment_list[i].loc[rmbid, DATASET_ATTRIBUTES[0]] = self.do.df_dataset.loc[rmbid, DATASET_ATTRIBUTES[0]]

                for distance_type in self.distance_measure_list:
                    resulting_nawba = get_nawba_using_models_from_scores(self, rmbid, y_models_list, distance_type, self.std_list[i], plot_flag)
                    column_nawba = "{}-{}".format(DATASET_ATTRIBUTES[0], distance_type)
                    self.df_experiment_list[i].loc[rmbid,column_nawba] = resulting_nawba
            print(" - sub_exp_{} completed".format(self.std_list[i]))

    def compute_summary(self):
        ''' Compute the summary of the results of the experiment. A dataframe is created to contains the results.
        '''

        for i in range(len(self.std_list)):
            for distance_type in self.distance_measure_list:
                for rmbid in self.df_experiment_list[i].index.values.tolist():
                    nawba_attributes = "{}-{}".format(DATASET_ATTRIBUTES[0], distance_type)
                    resulting_nawba = self.df_experiment_list[i].loc[rmbid, nawba_attributes]
                    correct_nawba = self.df_experiment_list[i].loc[rmbid, DATASET_ATTRIBUTES[0]]
                    if resulting_nawba == correct_nawba:
                        self.df_summary.loc[self.std_list[i], nawba_attributes] +=1

        self.df_summary.loc[:,:] /= len(self.df_experiment_list[0].index.values.tolist())
        #print(self.df_summary)

    def compute_confusion_matrix(self, distance, std):
        ''' Compute the confusion matrix using the result of the experiment with the selected distance and standard
        deviation standard

        :param distance: distance measure
        :param std: standard deviation value
        :return: confusion matrix
        '''

        index_std = self.std_list.index(std)
        y_pred = list()
        for rmbid in self.df_experiment_list[index_std].index.values.tolist():
            y_pred.append(self.df_experiment_list[index_std].loc[rmbid, distance])

        cnf_matrix = confusion_matrix(self.y_test, y_pred)
        return cnf_matrix

    def save_scores_models(self, notes_avg_nawba_list, y_avg_nawba_list, x_model, y_models_list, std, dir_path):
        ''' Save the plot of templates used in the experiment in the experiment directory

        :param notes_avg_nawba_list: label of the notes in the pitch class distribution (x-axis)
        :param y_avg_nawba_list: values of the pitch class distribution
        :param x_model: x-axis values of the templates
        :param y_models_list: y-axis values of the templates
        :param std: standard deviation
        :param dir_path: directory where the figure will be saved
        '''

        for i in range(len(notes_avg_nawba_list)):
            emph_fontsize = 30
            normal_fontsize = 24
            f, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 20))
            x_fake = list((i) for i in range(len(notes_avg_nawba_list[i])))
            ax1.tick_params(labelsize=normal_fontsize)
            ax1.bar(x_fake, y_avg_nawba_list[i], tick_label=notes_avg_nawba_list[i])
            ax1.set_title("Avarage Pitch Class Distribution - nawba {}".format(self.get_nawba_list()[i]), fontsize=emph_fontsize)
            ax1.set_xlabel("Notes", fontsize=emph_fontsize)
            ax1.set_ylabel("Occurances %", fontsize=emph_fontsize)
            ax2.tick_params(labelsize=normal_fontsize)
            ax2.plot(x_model, y_models_list[i], linewidth=4)
            ax2.set_xlabel("Cents", fontsize=emph_fontsize)
            ax2.set_ylabel("Occurances %", fontsize=emph_fontsize)
            ax2.set_title("Template with standard deviation {} - nawba {}".format(std, self.get_nawba_list()[i]), fontsize=emph_fontsize)
            file_name = "avg_score_template-nawba{}".format(self.get_nawba_list()[i])
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            f.savefig(os.path.join(dir_path, file_name)) #, dpi=300
            plt.close(f)

    def save_best_shifted_recording_plot(self, rmbid, x_s, y_s, y_s_f, shift, predicted_nawba, dir_path):
        ''' Save the plot of the best match with the template that provide the predicted nawba

        :param rmbid: musicbrainz id of the tested recording
        :param x_s: x-axis values for the pitch distribution of the recording and of the template
        :param y_s: y-axis values for the pitch distribution of the recording
        :param y_s_f: y-axis values of the template
        :param shift: shift value to obtain the best match
        :param predicted_nawba: predicted nawba
        :param dir_path: directory where the figure will be saved
        '''

        emph_fontsize = 30
        normal_fontsize = 24
        fig = plt.figure(figsize=(20, 10))
        plt.plot(x_s, y_s_f, label="template_{}".format(predicted_nawba), linewidth=4)
        plt.plot(x_s, y_s, label="shifted recording", linewidth=4)

        plt.title("{} - shift {}".format(rmbid, shift), fontsize=emph_fontsize)
        plt.xlabel("Cents", fontsize=emph_fontsize)
        plt.ylabel("Occurances", fontsize=emph_fontsize)
        plt.xticks(fontsize = normal_fontsize)
        plt.yticks(fontsize = normal_fontsize)

        plt.legend(fontsize = normal_fontsize)
        file_name = "{}_shift_{}".format(rmbid, shift)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        plt.savefig(os.path.join(dir_path, file_name))
        plt.close(fig)

# ---------------------------------------- END EXPERIMENT CLASS ----------------------------------------

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, plot=False, path_directory=""):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    emph_fontsize = 30
    normal_fontsize = 24
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = "{} - normalized".format(title)
    plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=emph_fontsize)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0, fontsize=normal_fontsize)
    plt.yticks(tick_marks, classes, fontsize=normal_fontsize)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=emph_fontsize)

    plt.tight_layout()
    plt.ylabel('True nawba', fontsize=emph_fontsize)
    plt.xlabel('Predicted nawba', fontsize=emph_fontsize)

    if print:
        if not os.path.exists(path_directory) and not path_directory =="":
            os.makedirs(path_directory)
        file_name_path = os.path.join(path_directory, title)
        plt.savefig(file_name_path, dpi=300)

    plt.show()

def calculate_overall_confusion_matrix(experiment_list, distance, std):
    ''' Calculate the overall confusion matrix of a set of experiment

    :param experiment_list: list of experiment object
    :param distance: distance measure
    :param std: standard deviation value
    :return: np matrix with the confusion matrix
    '''
    cnf_list = list()
    df_cnf_list = list()
    for experiment in experiment_list:
        cnf_temp = experiment.compute_confusion_matrix(distance, std)
        cnf_list.append(cnf_temp)
        header = range(len(cnf_temp))
        df_temp = pd.DataFrame(0, columns=header, index=header)
        for i in header:
            for j in header:
                df_temp.loc[i,j] = cnf_temp[i][j]
        df_cnf_list.append(df_temp)

    df_cnf_total = df_cnf_list[0]
    for i in range(1,len(df_cnf_list)):
        df_cnf_total = df_cnf_total.add(df_cnf_list[i])

    return np.asarray(df_cnf_total.values.tolist())

def export_overall_experiment(experiment_list):
    ''' export in csv the results of the single experiment and the overall ones

    :param experiment_list: list of experiment object
    '''

    for i in range(len(experiment_list)):
        exp_dir = experiment_list[i].experiment_dir
        exp_name = experiment_list[i].experiment_name
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        for j in range(len(experiment_list[0].std_list)):
            std_dir = os.path.join(exp_dir, str(experiment_list[0].std_list[j]))
            if not os.path.exists(std_dir):
                os.makedirs(std_dir)
            experiment_result_filename = "{} - {}.csv".format(exp_name, experiment_list[i].std_list[j])
            path_filename_result = os.path.join(std_dir, experiment_result_filename)
            experiment_list[i].df_experiment_list[j].to_csv(path_filename_result, sep=';', encoding="utf-8")
        summary_filename = "exp_{} - summary.csv".format(i+1)
        path_filename_summary = os.path.join(exp_dir, summary_filename)
        experiment_list[i].df_summary.to_csv(path_filename_summary, sep=';', encoding="utf-8")

    df_overall = experiment_list[0].df_summary
    for index in range(len(experiment_list) - 1):
        df_overall = df_overall.add(experiment_list[index + 1].df_summary)
    df_overall = df_overall.divide(len(experiment_list))
    overall_filename = "overall_results.csv"
    source_path = experiment_list[0].source_dir
    path_filename_overall = os.path.join(source_path, overall_filename)
    df_overall.to_csv(path_filename_overall, sep=';', encoding="utf-8")

    # find best result
    max_value_index = df_overall.max().values.tolist().index(max(df_overall.max().values.tolist()))
    max_distance = df_overall.columns.tolist()[max_value_index]
    index_value = df_overall[max_distance].tolist().index(max(df_overall[max_distance].tolist()))
    max_std = df_overall.index.values.tolist()[index_value]
    overall_conf_matrix = calculate_overall_confusion_matrix(experiment_list, max_distance, max_std)
    classes = experiment_list[0].get_nawba_list()
    plot_confusion_matrix(overall_conf_matrix, classes, normalize=False,
                          title="Confusion matrix - {} - {}".format(experiment_list[0].distance_measure_list[max_value_index], max_std), cmap=plt.cm.Blues,
                          plot=True, path_directory=source_path)
    plot_confusion_matrix(overall_conf_matrix, classes, normalize=True,
                          title="Confusion matrix - {} - {}".format(experiment_list[0].distance_measure_list[max_value_index], max_std), cmap=plt.cm.Blues,
                          plot=True, path_directory=source_path)

    # -------------------------------------------------- OLD --------------------------------------------------
    # -------------------------------------------------- NOT CHECKED --------------------------------------------------



    # def export_dataset(self, experiment_dir, file_name, tonic_type, format):
    #
    #     if not (tonic_type == 'all' or tonic_type in FN_TONIC_TYPE):
    #         raise Exception("Tonic type " + str(tonic_type) + " do not exist")
    #
    #     if not(format == 'csv' or format == 'json'):
    #         raise Exception("Format " + str(format) + " unknown")
    #
    #     attributes_list  = list()
    #     for attr in ATTRIBUTES:
    #         if not ((tonic_type == FN_TONIC_TYPE[0] and (attr == ATTRIBUTES[3] or attr == ATTRIBUTES[4])) or \
    #                 (tonic_type == FN_TONIC_TYPE[1] and (attr == ATTRIBUTES[2] or attr == ATTRIBUTES[4])) or \
    #                 (tonic_type == FN_TONIC_TYPE[2] and (attr == ATTRIBUTES[2] or attr == ATTRIBUTES[3]))):
    #             attributes_list.append(attr)
    #
    #     #print(attributes_list)
    #     # create a new dataframe
    #     if tonic_type == 'all':
    #         df_temp = self.df_dataset
    #     else:
    #         df_temp = pd.DataFrame(columns = attributes_list)
    #         for row_index in self.df_dataset.index.values:
    #             row_values = list()
    #             for attr in attributes_list:
    #                 #print("[" + str(row_index) + ", " + str(attr) + "]")
    #                 row_values.append(self.df_dataset.loc[row_index, attr])
    #             df_row = pd.DataFrame([row_values], columns = attributes_list)
    #             df_temp = df_temp.append(df_row,  ignore_index=True)
    #
    #     if not os.path.exists(experiment_dir):
    #         os.makedirs(experiment_dir)
    #
    #     if format == 'csv':
    #         df_temp.to_csv(os.path.join(experiment_dir, file_name + '.csv'), sep=';', encoding="utf-8")
    #         print(str(file_name) + '.csv created')
    #
    #     if format == 'json':
    #         df_temp.to_json(path_or_buf=os.path.join(experiment_dir, file_name + '.json') ,orient='records')
    #         print(str(file_name) + '.json created')
    #
    # def export_dataset_csv_json(self):
    #     #NAME_TONIC_TYPE = ['not_filt', 'filt', 'sec']
    #     #for i in range(len(FN_TONIC_TYPE)):
    #     #    self.export_dataset(EXPERIMENT_DIR, suffix + NAME_TONIC_TYPE[i], FN_TONIC_TYPE[i], 'csv')
    #     #    self.export_dataset(EXPERIMENT_DIR, suffix + NAME_TONIC_TYPE[i], FN_TONIC_TYPE[i], 'json')
    #     self.export_dataset(self.exp_dir, self.exp_name, 'all', 'csv')# + "all"
    #     self.export_dataset(self.exp_dir, self.exp_name, 'all', 'json')#+ "all"
    #
    # def import_dataset_from_csv(self, path, file_name):
    #     complete_path = os.path.join(path,file_name)
    #     if not os.path.exists(complete_path):
    #         raise Exception("Path {} doesn't exist".format(complete_path))
    #     self.df_dataset = pd.read_csv(complete_path, sep = ';', encoding="utf-8", index_col=0)
    #     print(self.df_dataset)
    #
    # def move_dataset_mp3(self, recordings_dir, experiment_mp3_dir):
    #
    #     # list of recordings by mbid
    #     rmbid_list = list()
    #     for row_index in self.df_dataset.index.values:
    #         rmbid_list.append(self.df_dataset.loc[row_index, ATTRIBUTES[0]])
    #
    #     # check if all the files exists
    #     if check_files_of_rmbid_lists(recordings_dir, rmbid_list, ['mp3', FNT_PITCH, FNT_PITCH_FILT], [True, True, True]):
    #         for rmbid in rmbid_list:
    #             FILE_NAMES = [rmbid + '.mp3', FNT_PITCH, FNT_PITCH_FILT]
    #             for i in range(len(FILE_NAMES)):
    #                 origin_file = os.path.join(recordings_dir, rmbid, FILE_NAMES[i])
    #                 destination_dir = os.path.join(experiment_mp3_dir, rmbid)
    #                 if not os.path.exists(destination_dir):
    #                     os.makedirs(destination_dir)
    #                 destination_file = os.path.join(destination_dir, FILE_NAMES[i])
    #                 print(destination_file)
    #                 copyfile(origin_file, destination_file)
    #         print("All files are copied in the new directory")
    #     else:
    #         raise Exception("Some requested files do not exist")


