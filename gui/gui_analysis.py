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
from utilities.recordingcomputation import *

# -------------------------------------------------- TONIC COMPARISON --------------------------------------------------
class NawbaComparison:

    def __init__(self, MetadataObject):
        self.cm = MetadataObject

        # first column:

        # create checkboxes to select the types of graph to plot
        checkboxes_labels = ["Single recording PD with not filtered tonic", \
                             "Single Recording PD with filtered tonic", \
                             "Single Recording PD with sections tonic", \
                             "Overall nawba PD with not filtered tonic", \
                             "Overall nawba PD with filtered tonic", \
                             "Overall nawba PD with sections tonic", \
                             "Average nawba PD with not filtered tonic", \
                             "Avarage nawba PD with filtered tonic",
                             "Avarage nawba PD with sections tonic"]

        self.graphs_checkboxes = list()
        for cb in checkboxes_labels:
            self.graphs_checkboxes.append(widgets.Checkbox(value=False, description= cb, indent=False))

        label1 = widgets.Label("Graphs:")
        column1_widgets = [label1] + self.graphs_checkboxes
        column1 = widgets.VBox(column1_widgets)

        # second column:

        # Dropdown menu nawba
        key_list = list()
        values_list = list()
        for row in self.cm.get_dataframe(DF_LISTS[2]).index.tolist():
            key_list.append(str(self.cm.convert_id(row, DF_LISTS[2], COLUMNS_NAMES[1])))
            values_list.append(row)
        vals = list(zip(key_list, values_list))
        self.nawba_widget = widgets.Dropdown(options=vals, \
                                       value=values_list[0], layout=widgets.Layout(width='80%'))

        # Dropdown menu alignment
        self.align_dropdown = widgets.Dropdown(options=[("aligned graphs (3 octaves)",0), ("complete graphs", 1)], \
                                          value=1, layout=widgets.Layout(width='80%'))

        # Checkbox folded and unfolded
        self.fold_widget = widgets.Checkbox(value=False, description="Fold graphs", indent=False)

        label2 = widgets.Label("Options:")
        column2 = widgets.VBox([label2, self.nawba_widget, self.align_dropdown, self.fold_widget])

        row1 = widgets.HBox([column1,column2])

        display(row1) #checkboxes_grid, , figure_plot
        widgets.interact_manual(self.plot_tonic_histograms)

    def plot_tonic_histograms(self):

        pitch = self.graphs_checkboxes[0].value
        pitch_filt = self.graphs_checkboxes[1].value
        pitch_sec = self.graphs_checkboxes[2].value
        overall_tonic = self.graphs_checkboxes[3].value
        overall_filtered = self.graphs_checkboxes[4].value
        overall_sec = self.graphs_checkboxes[5].value
        avg_tonic = self.graphs_checkboxes[6].value
        avg_filt = self.graphs_checkboxes[7].value
        avg_sec = self.graphs_checkboxes[8].value

        nawba = self.nawba_widget.value
        align = self.align_dropdown.value
        fold = self.fold_widget.value

        not_checked_mbid_list = self.cm.search_recording('all', nawba, 'all','all')

        mbid_list = check_files_of_rmbid_lists(RECORDINGS_DIR, not_checked_mbid_list, ['score', 'analysis json'], [True,True])

        for element in not_checked_mbid_list:
            if not (element in mbid_list):
                print(str(element) + " without score or pitch analysis")

        if pitch or pitch_filt or pitch_sec:
            iteration_list =  mbid_list
        else:
            iteration_list = ["Num of tracks: " + str(len(mbid_list))] # TODO: print # of file analyzed/# num total

        if overall_tonic or overall_filtered or overall_sec:
            if overall_tonic:
                x3, y3 = get_customized_histogram_from_rmbid_list(RECORDINGS_DIR, mbid_list, FN_TONIC_NO_FILT, fold, align)
                label_all_tonic = 'Overall nawba PD - Not Filtered Tonic'
            if overall_filtered:
                x4, y4 = get_customized_histogram_from_rmbid_list(RECORDINGS_DIR, mbid_list, FN_TONIC_FILT, fold, align)
                label_all_tonic_filt = 'Overall nawba PD - Filtered Tonic'
            if overall_sec:
                x4sec, y4sec = get_customized_histogram_from_rmbid_list(RECORDINGS_DIR, mbid_list, FN_TONIC_SEC, fold, align)
                label_all_tonic_sec = 'Overall nawba PD - Sections Tonic'

            # score
            x2s, y2s = get_customized_score_histogram(RECORDINGS_DIR, mbid_list, fold)
            label_overall = 'Overall notes distribution'
            x2s_fake = list((i) for i in range(len(x2s)))

        if avg_tonic or avg_filt or avg_sec:
            if avg_tonic:
                x5, y5 = get_customized_histogram_from_rmbid_list(RECORDINGS_DIR, mbid_list, FN_TONIC_NO_FILT, fold, align)
                label_avg_tonic = 'Avarage nawba PD - Not Filtered Tonic'
                y5[:] = [y / len(mbid_list) for y in y5] # NB: if the histogram is centered, the result could be not reliable

            if avg_filt:
                x6, y6 = get_customized_histogram_from_rmbid_list(RECORDINGS_DIR, mbid_list, FN_TONIC_FILT, fold, align)
                label_avg_tonic_filt = 'Avarage nawba PD - Filtered Tonic'
                y6[:] = [y / len(mbid_list) for y in y6]

            if avg_sec:
                x6sec, y6sec = get_customized_histogram_from_rmbid_list(RECORDINGS_DIR, mbid_list, FN_TONIC_SEC, fold, align)
                label_avg_tonic_sec = 'Avarage nawba PD - Sections Tonic'
                y6sec[:] = [y / len(mbid_list) for y in y6sec]

            x3s, y3s = get_customized_score_histogram(RECORDINGS_DIR, mbid_list, fold)
            label_avg = 'Avarage notes distribution'
            x3s_fake = list((i) for i in range(len(x3s)))
            y3s[:] = [y / len(mbid_list) for y in y3s]

        for rmbid in iteration_list:

            f, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,6))

            # subplot 1
            if len(rmbid) == 1:
                ax1.set_title('Pitch Distribution - ' + str(rmbid))
            else:
                ax1.set_title('Pitch Distributions - ' + str(rmbid))
            ax1.set_xlabel('Cents')
            ax1.set_ylabel('Occurances')

            # if the 'Single recording PD with not filtered tonic' checkbox is selected
            if overall_tonic:
                ax1.plot(x3, y3, label=label_all_tonic)

            if overall_filtered:
                ax1.plot(x4, y4, label=label_all_tonic_filt)

            if overall_sec:
                ax1.plot(x4sec, y4sec, label=label_all_tonic_sec)

            if avg_tonic:
                ax1.plot(x5, y5, label=label_avg_tonic)

            if avg_filt:
                ax1.plot(x6, y6, label=label_avg_tonic_filt)

            if avg_sec:
                ax1.plot(x6sec, y6sec, label=label_avg_tonic_sec)

            if pitch:
                x1, y1 = get_customized_histogram_from_rmbid_list(RECORDINGS_DIR, [rmbid], FN_TONIC_NO_FILT, fold, align)
                label_tonic = 'Single PD - Tonic No Filt: ' + str(int(get_tonic_value(os.path.join(RECORDINGS_DIR, rmbid), FN_TONIC_NO_FILT))) + " Hz"
                ax1.plot(x1, y1, label=label_tonic)

            # if the 'Single Recording PD with filtered tonic' checkbox is selected
            if pitch_filt:
                x2, y2 = get_customized_histogram_from_rmbid_list(RECORDINGS_DIR, [rmbid], FN_TONIC_FILT, fold, align)
                label_tonic_filt = 'Single PD - Tonic Filt: ' + str(int(get_tonic_value(os.path.join(RECORDINGS_DIR, rmbid), FN_TONIC_FILT))) + " Hz"
                ax1.plot(x2, y2, label=label_tonic_filt)

            if pitch_sec:
                x2sec, y2sec = get_customized_histogram_from_rmbid_list(RECORDINGS_DIR, [rmbid], FN_TONIC_SEC, fold, align)
                label_tonic_sec = 'Single PD - Tonic Sec: ' + str(int(get_tonic_value(os.path.join(RECORDINGS_DIR, rmbid), FN_TONIC_SEC))) + " Hz"
                ax1.plot(x2sec, y2sec, label=label_tonic_sec)

            if not(pitch == False and pitch_filt == False and pitch_sec == False and \
                               overall_tonic == False and overall_filtered == False and overall_sec == False and \
                               avg_tonic == False and avg_filt == False and avg_sec == False):
                ax1.legend()
                ax1.grid()

            # subplot 2
            if len(rmbid) == 1:
                ax2.set_title('Note Distributions - ' + str(rmbid))
            else:
                ax2.set_title('Note Distributions - ' + str(rmbid))
            ax2.set_xlabel('Notes')
            ax2.set_ylabel('Duration')

            if overall_tonic or overall_filtered or overall_sec:
                ax2.bar(x2s_fake, y2s, tick_label=x2s, label=label_overall) #
                ax2.set_xticklabels(x2s, rotation='vertical')

            if avg_filt or avg_tonic or avg_sec:
                ax2.bar(x3s_fake, y3s, tick_label=x3s, label=label_avg) #

            if pitch or pitch_filt or pitch_sec:
                x1, y1 = get_customized_score_histogram(RECORDINGS_DIR, [rmbid], fold)
                label_score_single_rec = 'Single recording distribution'
                x_fake = list((i) for i in range(len(x1)))
                ax2.bar(x_fake, y1, tick_label=x1, label=label_score_single_rec) #

            if not(pitch == False and pitch_filt == False and pitch_sec == False and \
                               overall_tonic == False and overall_filtered == False and overall_sec == False and \
                               avg_tonic == False and avg_filt == False and avg_sec == False):
                ax2.legend()

            plt.setp(plt.xticks()[1], rotation=90)
            print("Plots of " + str(rmbid))
            plt.show()

