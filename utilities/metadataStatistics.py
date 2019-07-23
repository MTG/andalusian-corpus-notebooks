#__author__ = "Niccolò Pretto"
#__email__ = "niccolo.pretto_at_dei.unipd.it"
#__copyright__ = "Copyright 2018, Università degli Studi di Padova, Universitat Pompeu Fabra"
#__license__ = "GPL"
#__version__ = "0.1"

import os
import json
import pandas as pd
import numpy as np
from utilities.constants import *
from utilities.generalutilities import *

class CollectionMetadata:

    def __init__(self):
        ''' The constructor fill dataframes for recording, tab, nawba, mizan, form, description,
                analizying the related json files

        '''
        # initialize recording_dir and create a directory if doesn't exist
        if not os.path.exists(RECORDINGS_DIR):
            os.makedirs(RECORDINGS_DIR)

        # Create a dataframe for every json file
        json_recording_list_name = PREFIX_JSON + DF_LISTS[0] + '.json'
        json_description_name = PREFIX_JSON + DF_LISTS[5] + '.json'
        self.df_recording = pd.DataFrame(columns=COLUMNS_RECORDINGS)
        self.df_description = pd.DataFrame(columns = COLUMNS_DESCRIPTION)
        self.mbid_no_sections = list()

        df_list = []
        id_name = COLUMNS_NAMES[3]
        # NB: indexes of the following dataframes are the 'uuid' of every object in the json file
        for i in range(1,5):
            # create a dataframe for each list
            df = pd.DataFrame(columns=COLUMNS_NAMES[0:3])
            with open(os.path.join(DATA_DIR, PREFIX_JSON + DF_LISTS[i] + '.json')) as json_file:
                file = json.load(json_file)

            for row in file['results']:
                if (not COLUMNS_NAMES[3] in row) and id_name == COLUMNS_NAMES[3]:
                    id_name = COLUMNS_NAMES[4]

                new_row = pd.DataFrame({COLUMNS_NAMES[0]: row[COLUMNS_NAMES[0]], COLUMNS_NAMES[1]:row[COLUMNS_NAMES[1]]}, index = [row[id_name]])
                df = pd.concat([df, new_row])
            df_list.append(df)

        self.df_tab = df_list[0]
        self.df_nawba = df_list[1]
        self.df_mizan = df_list[2]
        self.df_form = df_list[3]

        # read the documents
        with open(os.path.join(DATA_DIR, json_recording_list_name)) as json_file:
            recordings_file = json.load(json_file)
        with open(os.path.join(DATA_DIR, json_description_name)) as json_file:
            descriptions_file = json.load(json_file)

        # fill the first part of recordings dataframe with the data from recording dataframe
        for row in recordings_file['results']:
            new_row = pd.DataFrame({COLUMNS_RECORDINGS[0]: row[COLUMNS_RECORDINGS[0]], \
                                    COLUMNS_RECORDINGS[1]: row[COLUMNS_RECORDINGS[1]], \
                                    COLUMNS_RECORDINGS[2]: None, \
                                    COLUMNS_RECORDINGS[3]: None \
                                    }, index=[row[COLUMNS_DESCRIPTION[0]]])
            self.df_recording = pd.concat([self.df_recording, new_row])

        counter = 0
        for row in descriptions_file:
            # fill the second part of the recording dataframe
            self.df_recording.at[row[COLUMNS_DESCRIPTION[0]], COLUMNS_RECORDINGS[2]] = row[COLUMNS_RECORDINGS[2]]
            self.df_recording.at[row[COLUMNS_DESCRIPTION[0]], COLUMNS_RECORDINGS[3]] = row[COLUMNS_RECORDINGS[3]]

            # fill the description dataframe.
            # NB: They doesn't include the recording without the sections
            mbid = row[COLUMNS_DESCRIPTION[0]]
            counter_sec = 0
            if not row['sections']:
                self.mbid_no_sections.append(mbid)

            for section in row['sections']:
                # COLUMNS_DESCRIPTION = ['mbid', 'section', 'tab', 'nawba', 'mizan', 'form', 'start_time', 'end_time', 'duration']
                # COLUMNS_NAMES = ['name', 'transliterated_name', 'display_order', 'uuid']

                t = section[COLUMNS_DESCRIPTION[2]][id_name]
                n = section[COLUMNS_DESCRIPTION[3]][id_name]
                m = section[COLUMNS_DESCRIPTION[4]][id_name]
                f = section[COLUMNS_DESCRIPTION[5]][id_name]
                s = get_seconds(section[COLUMNS_DESCRIPTION[6]])
                e = get_seconds(section[COLUMNS_DESCRIPTION[7]])
                mi = get_interval(section[COLUMNS_DESCRIPTION[7]], section[COLUMNS_DESCRIPTION[6]])

                self.df_description.loc[counter] = [mbid, counter_sec, t, n, m, f, s, e, mi]
                counter_sec += 1
                counter += 1

# -------------------------------------------------- RECORDING LISTS --------------------------------------------------
    def get_list_of_recordings(self):
        ''' get the list of all recording mbid in the the dataframe

        :return: list of all recording mbid
        '''
        return self.get_dataframe('recording').index.tolist()

    def search_recordings_by_type(self, type_c, id_c):
        ''' Search recordings discriminating the track using id in one of the four characteristic
                'tab', 'nawba', 'mizan', 'form'

        :param type_c: characteristic. Possible value: ['tab', 'nawba', 'mizan', 'form']
        :param id_c: id of the selected characteristic on which will be discriminate the recordings
        :return: list of Music Brainz id with the id in the selected characteristic
        '''
        characteristic = ['tab', 'nawba', 'mizan', 'form']
        list_mbid = list()
        if type_c in characteristic:
            flag = True
            for row in self.df_description.index.tolist():
                # if list_mbid exists (manage the first iteration)
                if list_mbid:
                    flag = False
                    # if the mbid is not in the mbid_list, I can add it.
                    # if it already in the list, it won't be add again (avoid duplicate)
                    if self.df_description.loc[row, 'mbid'] != list_mbid[-1]:
                        flag = True

                if self.df_description.loc[row, type_c] == id_c and flag:
                    list_mbid.append(self.df_description.loc[row, 'mbid'])
        return list_mbid

    def search_recording(self, id_tab='all', id_nawba='all', id_mizan='all', id_form='all'):
        ''' Search the list of recordings by id in tab, nawba, mizan, form (or a subset of them)

        :param id_tab: id of the tab
        :param id_nawba: id of the nawba
        :param id_mizan: id of mizan
        :param id_form: id of form
        :return: list of Music Brainz id with all the characteristics requested by input
        '''

        mbid_list = self.df_recording.index.tolist()
        # search for tab
        if id_tab in self.df_tab.index.tolist():
            mbid_list = list_intersection(mbid_list, self.search_recordings_by_type('tab', id_tab))

        # search for nawba
        if id_nawba in self.df_nawba.index.tolist():
            mbid_list = list_intersection(mbid_list, self.search_recordings_by_type('nawba', id_nawba))

        # search for mizan
        if id_mizan in self.df_mizan.index.tolist():
            mbid_list = list_intersection(mbid_list, self.search_recordings_by_type('mizan', id_mizan))

        # search for form
        if id_form in self.df_form.index.tolist():
            mbid_list = list_intersection(mbid_list, self.search_recordings_by_type('form', id_form))

        return mbid_list

    def get_recordings_with_diff_(self, characteristic):
        ''' List of recordings with different types of the same characteristic

        :param characteristic: possible value ['tab', 'nawba', 'mizan', 'form']
        :return: list of recording mbids
        '''

        list_mdib_diff = list()
        reference_mbid = 0
        temp_characteristic = ""
        flag = False

        for i in self.df_description.index.tolist():
            if reference_mbid != self.df_description.loc[i, COLUMNS_DESCRIPTION[0]]:
                if flag == True:
                    list_mdib_diff.append(reference_mbid)
                reference_mbid =  self.df_description.loc[i, COLUMNS_DESCRIPTION[0]]
                temp_characteristic =  self.df_description.loc[i, characteristic]
                flag = False
            else:
                if self.df_description.loc[i, characteristic] != temp_characteristic:
                    flag = True

        return list_mdib_diff

    def get_recordings_without(self, attribute):
        ''' Create a list of mbid adding the recording without an attribute (attribute is empty)

        :param attribute: column of the dataframe that is checked to find empty fields
        :return: the list of mbid
        '''
        recordings = list()
        for rec in self.df_recording.index:
            if not self.df_recording.loc[rec,attribute]:
                recordings.append(rec)
        return recordings

    def get_characteristic(self, rmbid, characteristic):
        if not (characteristic in COLUMNS_DESCRIPTION[2:6]):
            raise Exception("{} is not a valid characteristic".format(characteristic))
        tab_list = list()
        for i in self.df_description.index.tolist():
            if rmbid == self.df_description.loc[i, COLUMNS_DESCRIPTION[0]]:
                tab_list.append(self.df_description.loc[i, characteristic])

        if len(set(tab_list)) == 0:
            raise Exception("rmbid {} has not {} description".format(rmbid, characteristic))
        if len(set(tab_list)) > 1:
            raise Exception("rmbid {} has more than one {} in the same recording".format(rmbid, characteristic))
        return tab_list[0]

    def import_rmbid_list_from_file(self, path):
        if not os.path.exists(path):
            raise Exception("Path {} does not exist".format(path))

        rmbid_list = list()
        valid_rmbid_list = self.get_list_of_recordings()
        with open(path, 'r') as f:
            for line in f.readlines():
                mbid = line.split("\n")[0]
                if not(mbid in valid_rmbid_list):
                    raise Exception("Rmbid {} is not in valid".format(mbid))
                rmbid_list.append(mbid)

        return rmbid_list

# -------------------------------------------------- DATAFRAME --------------------------------------------------

    def get_dataframe(self, type_info):
        ''' Return the requested dataframe

        :param type: possible value ['recording', 'tab', 'nawba', 'mizan', 'form', 'description']
        :return: dataframe
        '''
        if type_info == DF_LISTS[0]:
            return self.df_recording
        if type_info == DF_LISTS[1]:
            return self.df_tab
        if type_info == DF_LISTS[2]:
            return self.df_nawba
        if type_info == DF_LISTS[3]:
            return self.df_mizan
        if type_info == DF_LISTS[4]:
            return self.df_form
        if type_info == DF_LISTS[5]:
            return self.df_description

    def get_overall_sections_time(self):
        ''' Calculate the overall time of all the sections in the recordings.
                NB: tracks without sections are not considered

        :return: the sum of all the duration of the sections
        '''
        tot = 0
        df_temp = self.get_times('nawba')
        for row in df_temp.index.tolist():
            tot += df_temp.loc[row,'time']
        return tot

    def get_times(self, type_dataframe):
        ''' Calculate the amount of time for every row in a dataframe

        :param type_dataframe: possible values: 'nawba', 'tab', 'mizan', 'form'
        :return: dataframe with the id of the nawba as indexes and the related amount of time in seconds as column
        '''

        if type_dataframe == DF_LISTS[2]:
            indexes_dataframe = self.df_nawba.index.values
        if type_dataframe == DF_LISTS[1]:
            indexes_dataframe = self.df_tab.index.values
        if type_dataframe == DF_LISTS[3]:
            indexes_dataframe = self.df_mizan.index.values
        if type_dataframe == DF_LISTS[4]:
            indexes_dataframe = self.df_form.index.values

        df_nt = pd.DataFrame(0, columns=["time"], index=indexes_dataframe)
        for row in range(0, len(self.df_description.index)):
            id_type = self.df_description.at[row, type_dataframe]
            section_min = self.df_description.at[row, "duration"]
            df_nt.at[id_type, "time"] += section_min
        return df_nt

    def get_num_sections(self, type_dataframe):
        ''' Calculate the amount of sections for every row in a dataframe

        :param type_dataframe: possible values: 'nawba', 'tab', 'mizan', 'form'
        :return: dataframe with the id of the nawba as indexes and the related amount of time in seconds as column
        '''

        if type_dataframe == DF_LISTS[2]:
            indexes_dataframe = self.df_nawba.index.values
        if type_dataframe == DF_LISTS[1]:
            indexes_dataframe = self.df_tab.index.values
        if type_dataframe == DF_LISTS[3]:
            indexes_dataframe = self.df_mizan.index.values
        if type_dataframe == DF_LISTS[4]:
            indexes_dataframe = self.df_form.index.values

        df_nt = pd.DataFrame(0, columns=["num_sections"], index=indexes_dataframe)
        for row in range(0, len(self.df_description.index)):
            id_type = self.df_description.at[row, type_dataframe]
            #section_min = self.df_description.at[row, "duration"]
            df_nt.at[id_type, "num_sections"] += 1
        return df_nt

    def get_avarage_sections_time(self, type_dataframe):
        df_times = self.get_times(type_dataframe)
        df_num_sections = self.get_num_sections(type_dataframe)

        df_avg = pd.DataFrame(0, columns=["avg"], index=df_times.index.values)
        for index in df_times.index.values.tolist():
            if df_times.loc[index,'time'] == 0:
                df_avg.loc[index, 'avg'] = 0
            else:
                df_avg.loc[index, 'avg'] = int(df_times.loc[index,'time'] / df_num_sections.loc[index,'num_sections'])

        return df_avg

    def get_recording_translitered_title(self, rmbid):
        return self.df_recording.loc[rmbid, 'transliterated_title']

    def get_recording_orchestra_name(self, rmbid):
        return str('('+self.df_recording.loc[rmbid, 'archive_url'].split('/')[-1].split('_')[0]+')')

    def convert_id(self, id, type_dataframe, type_name):
        ''' Convert the id in its 'name' or 'transliterated_name'

        :param id: id that will be converted
        :param type_dataframe: possible values: 'nawba', 'tab', 'mizan', 'form'
        :param type_name: possible values: 'name' or 'transliterated_name'
        :return: related name or transliterated_name of the id in the type_dataframe
        '''

        if type_name == COLUMNS_NAMES[0] or type_name == COLUMNS_NAMES[1]:
            if type_dataframe == DF_LISTS[2]:
                return self.df_nawba.loc[id, type_name]
            if type_dataframe == DF_LISTS[1]:
                return self.df_tab.loc[id, type_name]
            if type_dataframe == DF_LISTS[3]:
                return self.df_mizan.loc[id, type_name]
            if type_dataframe == DF_LISTS[4]:
                return self.df_form.loc[id, type_name]
        return None

    def get_cross_dataframe(self, column_type, row_type, statistic_type):
        ''' Create a dataframe from selected statistic crossing two characteristics

        :param column_type: characteristic for columns in dataframe
        :param row_type: characteristic for index in dataframe
        :param statistic_type: type of statistic
        :return: dataframe with the statistics
        '''
        if not (column_type in DF_LISTS[1:5] and row_type in DF_LISTS[1:5] and statistic_type in STATISTIC_TYPE):
            raise Exception("Incorrect parameter/s")
        df_column = self.get_dataframe(column_type)
        df_index = self.get_dataframe(row_type)

        df_cross = pd.DataFrame(0, index=df_index.index.tolist(), columns=df_column.index.tolist())
        df_cross_temp = pd.DataFrame(0, index=df_index.index.tolist(), columns=df_column.index.tolist())
        for i in self.df_description.index.tolist():
            ind = self.df_description.loc[i, row_type]
            col = self.df_description.loc[i, column_type]
            # STATISTIC_TYPE = ['# recordings', '# sections', 'overall sections time', 'avg sections time']
            if statistic_type == STATISTIC_TYPE[0]:
                # num recordings
                if i == 0:
                    df_cross.loc[ind, col] += 1
                else:
                    if self.df_description.loc[i, COLUMNS_DESCRIPTION[0]] != self.df_description.loc[
                            i - 1, COLUMNS_DESCRIPTION[0]]:
                        df_cross.loc[ind, col] += 1
            else:
                # overall sections time
                sec = self.df_description.loc[i, 'duration']
                df_cross.loc[ind, col] += sec

                # num sections
                df_cross_temp.loc[ind, col] += 1

        if statistic_type == STATISTIC_TYPE[1]:
            df_cross = df_cross_temp

        if statistic_type == STATISTIC_TYPE[3]:
            df_cross = df_cross.div(df_cross_temp)

        if statistic_type == STATISTIC_TYPE[2] or statistic_type == STATISTIC_TYPE[3]:
            df_cross = self.convert_dataframe_second_to_time(df_cross)

        return df_cross

    def convert_dataframe_second_to_time(self, df_seconds):
        df_time = pd.DataFrame(0, index=df_seconds.index.tolist(), columns=df_seconds.columns.values)
        for col in df_seconds.columns.values:
            for row in df_seconds.index.tolist():
                if np.isnan(df_seconds.loc[row,col]) or df_seconds.loc[row,col] == 0:
                    df_time.loc[row,col] = get_time(0)
                else:
                    df_time.loc[row, col] = get_time(int(df_seconds.loc[row,col]))
        return df_time

    def get_description_of_single_recording(self, rmbid):
        ''' Get the rows from description dataframe of a single recording and return a dataframe cointaing them

        :param rmbid: Music Brainz id of the desidered recording
        :return: dataframe as description dataframe cointaining only the rows of the selected recording
        '''

        # create an empty dataframe as description
        df_recording = pd.DataFrame(columns = self.df_description.columns.values)

        for row_index in self.df_description.index.values.tolist():
            if self.df_description.loc[row_index, 'mbid'] == rmbid:
                df_row = self.df_description.loc[row_index, :]
                df_recording = df_recording.append(df_row, ignore_index=True)

        return df_recording

# -------------------------------------------------- CHECK --------------------------------------------------

    def check_rmbid_list_before_download(self,rmbid_list):
        ''' Check if the list of rmbid is in Dunya.

        :param rmbid_list: list of recording Music Brainz id
        :return: list of rmbid in dunya and a list of incorrect rmbid
        '''

        if not isinstance(rmbid_list, list):
            raise Exception("Parameter provided is not a list")

        if not rmbid_list:
            raise Exception("Empty list")

        rmbid_list_in_Dunya = self.get_list_of_recordings()

        rmbid_in_dunya = list()
        incorrect_rmbid = list()

        for rmbid in rmbid_list:
            if rmbid in rmbid_list_in_Dunya:
                rmbid_in_dunya.append(rmbid)
            else:
                incorrect_rmbid.append(rmbid)

        return rmbid_in_dunya, incorrect_rmbid

# -------------------------------------------------- EXPORT --------------------------------------------------

    def dataframe_to_csv(self,data_dir, name):
        ''' Export in csv the dataframe related to the selected characteristic

        :param data_dir: directory where the data will be stored
        :param name: name of the characteristic to export
        '''
        if name in DF_LISTS:
            dt = self.get_dataframe(name)
            dt.to_csv(os.path.join(data_dir, name + '.csv'), sep=';', encoding="utf-8")
        else:
            raise Exception("Name {} doesn'exist".format(name) )

# -------------------------------------------------- END REVISED --------------------------------------------------

    #
    # def convert_id_indexes(self, df_n, type_dataframe, type_name):
    #     ''' Convert all the indexes of a dataframe from id to 'name' or 'transliterated_name'.
    #             The correspondences are located in the dataframe indicated with the parameter type_dataframe
    #
    #     :param df_n: dataframe that will be modified
    #     :param type_dataframe: possible values: 'nawba', 'tab', 'mizan', 'form'
    #     :param type_name: possible values: 'name' or 'transliterated_name'
    #     :return: dataframe with the translated indexes
    #     '''
    #     index_as_list = df_n.index.tolist()
    #     for element in index_as_list:
    #         idx = index_as_list.index(element)
    #         index_as_list[idx] = self.convert_id(element, type_dataframe, type_name)
    #     df_n.index = index_as_list
    #     return df_n
    #
    # def convert_id_columns(self, df_n, type_dataframe, type_name):
    #     ''' Convert all the column name of a dataframe from id to 'name or 'transliterated_name'.
    #             The correspondences are located in the dataframe indicated with the parameter type_dataframe
    #
    #     :param df_n: dataframe that will be modified
    #     :param type_dataframe: possible values: 'nawba', 'tab', 'mizan', 'form'
    #     :param type_name: possible values: 'name' or 'transliterated_name'
    #     :return: dataframe with the translated column
    #     '''
    #     new_col_list = list()
    #     for col in df_n:
    #         new_col_list.append(self.convert_id(col, type_dataframe, type_name))
    #
    #     df_n.columns = new_col_list
    #     return df_n
    #
