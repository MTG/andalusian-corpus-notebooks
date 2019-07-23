#__author__ = "Niccolò Pretto"
#__email__ = "niccolo.pretto@dei.unipd.it"
#__copyright__ = "Copyright 2018, Università degli Studi di Padova, Universitat Pompeu Fabra"
#__license__ = "GPL"
#__version__ = "0.1"

import os
import json
import pandas as pd
import ipywidgets as widgets
import numpy as np

import matplotlib.pyplot as plt
from IPython.display import display

from utilities.constants import *
from utilities.metadataStatistics import *
from utilities.generalutilities import *


# -------------------------------------------------- DATAFRAME --------------------------------------------------

class VisualizeDataframeGui:

    def __init__(self, MetadataObject):

        self.cm = MetadataObject

        self.children = list()

        # values from metadata
        self.num_recordings_values = list()
        self.num_sections_values = list()
        self.times_values = list()
        self.avg_times_values = list()
        self.name_values = list()

        # self.column_titles = ['# recordings', '# sections', 'overall sections time', 'avg sections time']
        self.column_titles = STATISTIC_TYPE
        self.tab_title = CHARACTERISTICS_NAMES # ['Tubu', 'Nawabat', 'Myazen', 'Forms']

        # create the tabs content
        for characteristic in DF_LISTS[1:5]:

            df = self.cm.get_dataframe(characteristic)
            df_times = self.cm.get_times(characteristic)
            df_sections = self.cm.get_num_sections(characteristic)
            df_avg = self.cm.get_avarage_sections_time(characteristic)
            columns = list()

            # labels
            #label_box_layout = widgets.Layout(height='40px')
            #indexes = [widgets.Box(children=[widgets.Label('id')], layout=label_box_layout)] #
            indexes = [widgets.Label('id')]
            names = [widgets.Label(COLUMNS_NAMES[0])]
            transliterated_names = [widgets.Label(COLUMNS_NAMES[1])]
            num_recordings = [widgets.Label(self.column_titles[0])]
            num_sections = [widgets.Label(self.column_titles[1])]
            times = [widgets.Label(self.column_titles[2])]
            avg_times = [widgets.Label(self.column_titles[3])]

            # list of values
            name_values = list()
            num_recordings_values = list()
            num_sections_values = list()
            times_values = list()
            avg_times_values = list()

            for index in df.index.values.tolist():

                # number of track with this characteristic
                if characteristic == DF_LISTS[1]:
                    list_value = self.cm.search_recording(index, 'all', 'all', 'all')
                if characteristic == DF_LISTS[2]:
                    list_value = self.cm.search_recording('all', index, 'all', 'all')
                if characteristic == DF_LISTS[3]:
                    list_value = self.cm.search_recording('all', 'all', index, 'all')
                if characteristic == DF_LISTS[4]:
                    list_value = self.cm.search_recording('all', 'all', 'all', index)

                if len(list_value) != 0:

                    # column with indexes
                    indexes.append(widgets.Label(str(index)))

                    # second and third columns - COLUMNS_NAMES = ['name', 'transliterated_name']
                    names.append(widgets.Label(str(df.loc[index,COLUMNS_NAMES[0]])))
                    transliterated_names.append(widgets.Label(str(df.loc[index,COLUMNS_NAMES[1]])))
                    name_values.append(str(df.loc[index,COLUMNS_NAMES[1]]))

                    num_recordings_values.append(len(list_value))
                    num_recordings.append(widgets.Label(str(len(list_value))))

                    # number of sections with this characteristic
                    num_sections_values.append(df_sections.loc[index,'num_sections'])
                    num_sections.append(widgets.Label(str(df_sections.loc[index,'num_sections'])))

                    # overall amount of time
                    times_values.append(df_times.loc[index,'time'])
                    times.append(widgets.Label(str(get_time(df_times.loc[index,'time']))))

                    # avarage amount of time
                    avg_times_values.append(df_avg.loc[index,'avg'])
                    avg_times.append(widgets.Label(str(get_time(df_avg.loc[index,'avg']))))

            id_layout = widgets.Layout(display='flex',\
                                flex_flow='column',\
                                align_items='stretch',\
                                #border='solid 1px',\
                                width='6%')

            tran_layout = widgets.Layout(display='flex', \
                                        flex_flow='column', \
                                        align_items='stretch', \
                                        # border='solid 1px',\
                                        width='21%')

            box_layout = widgets.Layout(display='flex',\
                                flex_flow='column',\
                                align_items='stretch',\
                                #border='solid 1px',\
                                width='13%')



            columns += [#widgets.VBox(children=indexes, layout=id_layout), \
                        widgets.VBox(children=names, layout=box_layout),\
                        widgets.VBox(children=transliterated_names, layout=tran_layout),\
                        widgets.VBox(children=num_recordings, layout=box_layout),\
                        widgets.VBox(children=num_sections, layout=box_layout), \
                        widgets.VBox(children=times, layout=box_layout), \
                        widgets.VBox(children=avg_times, layout=box_layout)]


            # add values to lists
            self.num_recordings_values.append(num_recordings_values)
            self.num_sections_values.append(num_sections_values)
            self.times_values.append(times_values)
            self.avg_times_values.append(avg_times_values)
            self.name_values.append(name_values)

            # add tab to children list
            self.children.append(widgets.HBox(columns))

        self.tab = widgets.Tab()
        self.tab.children = self.children

        for i in range(len(self.children)):
            self.tab.set_title(i, self.tab_title[i])

        self.mainbox = widgets.VBox([self.tab])

        # create a selector for type of data
        label_dropdown_data = widgets.Label('Select the type of data:')
        self.type_selector = widgets.Dropdown(
            options=self.column_titles,
            value=self.column_titles[1],
        )
        self.type_selector.description = ""

        # create a selector for the tab
        self.options_tab = DF_LISTS[1:5]
        label_dropdown_tab = widgets.Label('Select the type of data:')
        self.tab_selector = widgets.Dropdown(
            options=self.options_tab,
            value=DF_LISTS[1],
        )
        self.tab_selector.description = ""
        self.tab_selector.layout.display = 'none'

        # create an interactive space
        figure_plot = widgets.interactive(self.plot_bar_histogram, type=self.type_selector, tab=self.tab_selector)

        self.tab.observe(self.on_tab_change, names=self.tab.selected_index)

        display(widgets.VBox([self.mainbox, label_dropdown_data, figure_plot]))


    def on_tab_change(self, change):
        self.tab_selector.value = DF_LISTS[self.tab.selected_index + 1]

    def plot_bar_histogram(self, type, tab):            # NB: tab is the widget, not the musical entity
        tab_index = self.options_tab.index(tab)
        self.type_selector.description = ""
        # Title
        title = CHARACTERISTICS_NAMES[tab_index]  + " - " + type
        plt.figure(figsize=(20, 8))
        ax = plt.gca()
        ax.grid(True)
        plt.title(title, fontsize=24)
        plt.xlabel(CHARACTERISTICS_NAMES[tab_index], fontsize=22)

        if type == self.column_titles[0]:
            y = self.num_recordings_values[tab_index]
            plt.ylabel("# recordings", fontsize=22)
        else:
            if type == self.column_titles[1]:
                y = self.num_sections_values[tab_index]
                plt.ylabel("# sections", fontsize=22)
            else:
                if type == self.column_titles[2]:
                    y = self.times_values[tab_index]
                    plt.ylabel("hours", fontsize=22)
                    plt.yticks(np.arange(0,max(self.times_values[tab_index]), 10800), (np.arange(0,max(self.times_values[tab_index]), 10800)/3600).astype(int))
                else:
                    if type == self.column_titles[3]:
                        y = self.avg_times_values[tab_index]
                        plt.ylabel("minutes", fontsize=22)
                        plt.yticks(np.arange(0, max(self.avg_times_values[tab_index]), 120), (np.arange(0, max(self.avg_times_values[tab_index]), 120)/60).astype(int))
                    else:
                        raise Exception("Value not exists")
        plt.yticks(fontsize=18)
        # x value
        x = range(1,len(y)+1)
        #df = self.cm.get_dataframe(tab)
        #x_label = df[COLUMNS_NAMES[1]].tolist()
        x_label = self.name_values[tab_index]
        plt.tight_layout()
        plt.xticks(x, x_label, rotation=45, ha='right', fontsize=18)
        plt.bar(x,y)

# -------------------------------------------------- CROSS INFORMATION --------------------------------------------------

class CrossMetadataVisualization:

    def __init__(self, MetadataObject):
        self.cm = MetadataObject
        # labels
        labels = ["Column:", "Row: ", "Title: ", "Statistics:"]
        self.dropbox_menus = list()
        top_boxes = list()
        style_boxdropdown = widgets.Layout(width='25%')
        style_dropdown = widgets.Layout(width='180px') # TODO: fix the width using percentage
        for i in range(4):
            label = widgets.Label(labels[i])
            if i < 2:
                self.dropbox_menus.append(widgets.Dropdown(options = DF_LISTS[1:5], value = DF_LISTS[i+1], layout = style_dropdown))
            else:
                if i == 2:
                    # options_title = ['id', COLUMNS_NAMES[0], COLUMNS_NAMES[1]]
                    options_title = [COLUMNS_NAMES[0], COLUMNS_NAMES[1]]
                    self.dropbox_menus.append(widgets.Dropdown(options = options_title, value = options_title[0], layout = style_dropdown))
                else:
                    self.options_title = STATISTIC_TYPE[1:4]
                    self.dropbox_menus.append(widgets.Dropdown(options= self.options_title, value = self.options_title[0], layout = style_dropdown))
            top_boxes.append(widgets.VBox([label, self.dropbox_menus[i]], style = style_boxdropdown))
        first_line = widgets.HBox(top_boxes)
        self.second_line = widgets.Label("Results:")
        self.third_line = widgets.VBox([])
        self.updateResults(0)

        main_box = widgets.VBox([first_line, self.second_line, self.third_line])

        display(main_box)

        for i in range(4):
            self.dropbox_menus[i].observe(self.updateResults, names ="value")

    def updateResults(self, change):
        self.second_line.description = "Computing..."
        # Title type
        column_type = self.dropbox_menus[0].value
        row_type = self.dropbox_menus[1].value
        title_type = self.dropbox_menus[2].value
        option_type = self.dropbox_menus[3].value

        vbox_layout = widgets.Layout(min_width="70px")

        # column 1
        first_column_label = [widgets.Label(column_type + " /\n" + row_type)]
        df_column = self.cm.get_dataframe(row_type)
        values_first_column = df_column.index.values.tolist()
        translated_id_values = list()
        if title_type != 'id':
            for id in values_first_column:
                translated_id_values.append(self.cm.convert_id(id, row_type, title_type))
        else:
            translated_id_values = values_first_column

        for element in translated_id_values:
            first_column_label.append(widgets.Label(str(element)))

        first_column = widgets.VBox(children = first_column_label, layout = vbox_layout)

        # other column
        df_columns = self.cm.get_dataframe(column_type)
        values_other_columns = df_columns.index.values.tolist()
        translated_other_id_values = list()
        if title_type != 'id':
            for id in values_other_columns:
                translated_other_id_values.append(self.cm.convert_id(id, column_type, title_type))
        else:
            translated_other_id_values = values_other_columns

        df_results = self.cm.get_cross_dataframe(column_type, row_type, option_type)
        #print(df_results)
        all_columns = [first_column]
        for col in list(df_results.columns.values):
            index = list(df_results.columns.values).index(col)
            single_column = [widgets.Label(str(translated_other_id_values[index]))]
            for element in df_results.index.values.tolist():
                #if (option_type == STATISTIC_TYPE[2] or option_type == STATISTIC_TYPE[3]) and df_results.loc[element, col]!= get_time(0):
                if df_results.loc[element, col] == get_time(0) or df_results.loc[element, col] == 0:
                    single_column.append(widgets.Label(str(df_results.loc[element, col])))
                else:
                    layout_box = widgets.Layout(border='solid 1px red')
                    single_column.append(\
                        widgets.Box(children=[widgets.Label(str(df_results.loc[element, col]))], layout=layout_box))
            all_columns.append(widgets.VBox(children = single_column, layout = vbox_layout))

        #layout_hbox = widgets.Layout(overflow_x='scroll', flex_direction='row', display='flex')
        self.third_line.children = [widgets.HBox(children = all_columns)]#, layout = layout_hbox)]

        self.second_line.description = "Results:"

# -------------------------------------------------- SINGLE INFORMATION --------------------------------------------------

class SingleRecordingVisualization:

    def __init__(self, MetadataObject, rmbid):
        self.cm = MetadataObject

        title_label = widgets.Label("MBID: " + str(rmbid))
        dunya_info = widgets.Label("DUNYA API info: dunya.compmusic.upf.edu/api/andalusian/recording/"+str(rmbid))
        mainBox = widgets.VBox([title_label, dunya_info, self.get_single_recording_box(rmbid)])

        display(mainBox)

    def get_single_recording_box(self, rmbid):
        ''' Display a table with all the sections description from the description file

        :param rmbid: Music Brainz id of the selected recording
        :return: a HBox with the description table
        '''
        df_recording = self.cm.get_description_of_single_recording(rmbid)
        columns_header = df_recording.columns.values.tolist()
        columns = list()
        for i in range(len(columns_header)):
            single_column = [widgets.Label(str(columns_header[i]))]
            columns.append(single_column)

        for row_index in df_recording.index.values.tolist():
            for column_index in columns_header:
                value = str(df_recording.loc[row_index, column_index])
                if column_index in DF_LISTS[1:5]:
                    value = self.cm.convert_id(value, column_index, COLUMNS_NAMES[1])
                temp_widget = widgets.Label(value)
                index = columns_header.index(column_index)
                columns[index].append(temp_widget)

        column_layout = widgets.Layout(display='flex', \
                                     flex_flow='column', \
                                     align_items='stretch', \
                                     # border='solid 1px',\
                                     width='12%')
        column_VBoxes = list()
        for col in range(len(columns)):
            if col != 0:
                column_VBoxes.append(widgets.VBox(children = columns[col], layout = column_layout))

        mainBox = widgets.HBox(column_VBoxes)

        return mainBox

