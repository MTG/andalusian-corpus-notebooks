#__author__ = "Niccolò Pretto"
#__email__ = "niccolo.pretto@dei.unipd.it"
#__copyright__ = "Copyright 2018, Università degli Studi di Padova, Universitat Pompeu Fabra"
#__license__ = "GPL"
#__version__ = "0.1"

import ipywidgets as widgets
from IPython.display import display
from utilities.constants import *
from utilities.recordingcomputation import *
from utilities.dunyautilities import *

# -------------------------------------------------- DOWNLOAD --------------------------------------------------

class DownloadGui:

    def __init__(self, rmbid_list, MetadataObject):
        ''' The constructor create an interface based on widgets to select the type of file to download
            for each recording in the list

        :param rmbid_list: list of recordings that will be analyzed
        :param MetadataObject: object that collect metadata from Dunya
        '''

        self.cm = MetadataObject

        # Check if the list of rmbid is in Dunya. If rmbid_list is empty or is not a list launch an Exception
        try:
            correct_list, incorrect_list = self.cm.check_rmbid_list_before_download(rmbid_list)
        except Exception as e:
            print(str(e))
            return

        if not correct_list:
            print("All the recordings are not in Dunya")
            return

        self.rmbid_list = list(set(correct_list))

        # checkboxs
        style = {'description_width': 'initial'}
        self.check_mp3 = widgets.Checkbox(
            value=True,
            description='Mp3 from Dunya',
            disabled=False,
            style= style
        )
        self.check_score = widgets.Checkbox(
            value=True,
            description='XML Score (if available)',
            disabled=False,
            style=style

        )

        self.check_metadata = widgets.Checkbox(
            value=True,
            description='Metadata from Music Brainz',
            disabled=False,
            style= style

        )

        self.download_option = widgets.Checkbox(
            value=False,
            description='Overwrite existing files',
            disabled=False,
            style= style
        )

        self.download_button = widgets.Button(
            description='Download',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            layout=widgets.Layout(align_self = 'center'),
        )

        left_label_box = widgets.Label("Select data type:" , layout = widgets.Layout(width='50%'))
        right_label_box = widgets.Label("Select option:", layout = widgets.Layout(width='50%'))
        list_check = [self.check_mp3, self.check_score, self.check_metadata]
        left_check_box = widgets.VBox(list_check, layout = widgets.Layout(width='50%'))
        right_check_box = widgets.VBox([self.download_option], layout = widgets.Layout(width='50%'))
        center_button_box = widgets.HBox([self.download_button], layout = widgets.Layout(width='100%'))
        main_box = widgets.VBox([widgets.HBox([left_label_box, right_label_box]), widgets.HBox([left_check_box, right_check_box]), center_button_box])
        display(main_box)

        self.download_button.on_click(self.download_selected)

        if len(incorrect_list) != 0:
            print("List of incorrect recordings: ")
            print(incorrect_list)
            print()

        if len(correct_list) != len(set(correct_list)):
            print("Duplicated entry/ies removed")
            print()


    def download_selected(self, b):
        ''' Function activated by the button. Download the requested files

            :param b: parameter passed by the button
        '''
        if not self.check_mp3.value and not self.check_score.value and not self.check_metadata.value:
            print("No Data type selected")
        else:
            # disable all the checkboxs and the button
            self.disable_all()
            download_list_of(RECORDINGS_DIR, self.rmbid_list, self.check_mp3.value, self.check_score.value, self.check_metadata.value, self.download_option.value)
            self.enable_all()
            print()
            print("Download complete")

    def disable_all(self):
        ''' Disable all the checkboxes and the button

        '''
        self.check_mp3.disabled = True
        self.check_score.disabled = True
        self.check_metadata.disabled = True
        self.download_option.disabled = True
        self.download_button = True

    def enable_all(self):
        ''' Enable all the checkboxes and the button

        '''
        self.check_mp3.disabled = False
        self.check_score.disabled = False
        self.check_metadata.disabled = False
        self.download_option.disabled = False
        self.download_button = False

# -------------------------------------------------- CHECK --------------------------------------------------

def check_before_download(rmbid_list, cm):
    '''

    :param rmbid_list: list of recording Music Brainz id
    :param cm: MetadataObject: object that collect metadata from Dunya
    '''
    try:
        correct_list, uncorrect_list = cm.check_rmbid_list_before_download(rmbid_list)
    except Exception as e:
        print(str(e))
        return

    print("List of recordings in Dunya: ")
    print(correct_list)
    print()
    print("List of uncorrect recordings: ")
    print(uncorrect_list)

# -------------------------------------------------- COMPUTATION --------------------------------------------------

class ComputationGui:

    def __init__(self, rmbid_list, MetadataObject):
        ''' The constructor create an interface based on widgets to select the computation to run
            on the list of recording
            :param rmbid_list: list of recordings that will be analyzed
            :param MetadataObject: object that collect metadata from Dunya
        '''
        self.cm = MetadataObject

        # Check if the list of rmbid is in Dunya. If rmbid_list is empty or is not a list launch an Exception
        try:
            correct_list, incorrect_list = self.cm.check_rmbid_list_before_download(rmbid_list)
        except Exception as e:
            print(str(e))
            return

        if not correct_list:
            print("All the recordings are not in Dunya")
            return

        # avoid duplicate
        self.rmbid_list = list(set(correct_list))

        # check if the mp3s exist in the correct directory and continue with only the file with mp3
        self.rmbid_list = check_files_of_rmbid_lists(RECORDINGS_DIR, rmbid_list, ['mp3'], [True])

        if not self.rmbid_list:
            print("All the recordings have not mp3 files")
            return

        # checkboxs
        style = {'description_width': 'initial'}
        self.check_json = widgets.Checkbox(
            value=True,
            description='Pitch analysis in json',
            disabled=False,
            style=style
        )
        self.check_txt = widgets.Checkbox(
            value=True,
            description='Pitch analysis in txt',
            disabled=False,
            style=style
        )

        self.check_wav = widgets.Checkbox(
            value=False,
            description='Convert mp3 in wav',
            #TODO: change in False the following parameter when the methods to convert in wav will be developed
            disabled=True,
            style=style
        )

        self.download_option = widgets.Checkbox(
            value=False,
            description='Overwrite existing files',
            disabled=False,
            style=style
        )

        self.compute_button = widgets.Button(
            description='Compute',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            layout=widgets.Layout(align_self='center'),
        )

        left_label_box = widgets.Label("Select data type:", layout=widgets.Layout(width='50%'))
        right_label_box = widgets.Label("Select option:", layout=widgets.Layout(width='50%'))
        list_check = [self.check_json, self.check_txt, self.check_wav]
        left_check_box = widgets.VBox(list_check, layout=widgets.Layout(width='50%'))
        right_check_box = widgets.VBox([self.download_option], layout=widgets.Layout(width='50%'))
        center_button_box = widgets.HBox([self.compute_button], layout=widgets.Layout(width='100%'))
        main_box = widgets.VBox(
            [widgets.HBox([left_label_box, right_label_box]), widgets.HBox([left_check_box, right_check_box]),
             center_button_box])
        display(main_box)

        self.compute_button.on_click(self.compute_selected)

        if len(incorrect_list) != 0:
            print("List of incorrect recordings: ")
            print(incorrect_list)
            print()

        if len(correct_list) != len(set(correct_list)):
            print("Duplicated entry/ies removed")
            print()

    def compute_selected(self, b):
        ''' Function activated by the button. Perform the requested computation

        :param b: parameter passed by the button
        '''
        if not self.check_json.value and not self.check_txt.value and not self.check_wav.value:
            print("No Data type selected")
        else:
            # disable all the checkboxs and the button
            self.disable_all()
            compute_list_of(self.cm, RECORDINGS_DIR, self.rmbid_list, self.check_json.value, self.check_txt.value,
                             self.check_wav.value, self.download_option.value)
            self.enable_all()
            print()
            print("Computation completed")

    def disable_all(self):
        ''' Disable all the checkboxes and the button

        '''
        self.check_json.disabled = True
        self.check_txt.disabled = True
        self.check_wav.disabled = True
        self.download_option.disabled = True
        self.compute_button = True

    def enable_all(self):
        ''' Enable all the checkboxes and the button

        '''
        self.check_json.disabled = False
        self.check_txt.disabled = False
        self.check_wav.disabled = False
        self.download_option.disabled = False
        self.compute_button = False

        #TODO: delete the following line when the methods to convert in wav will be developed
        self.check_wav.disabled = True



# -------------------------------------------------- SELECTOR --------------------------------------------------

class SelectionGui:

    def __init__(self, MetadataObject, tab_len):
        ''' The constructor create an interface based on widgets to select a list of recording using filters
                and checkbox
        :param MetadataObject: object that collect metadata from Dunya
        :param tab_len: number of checkbox in a tab
        '''
        self.cm = MetadataObject
        self.tab_len = tab_len

        # First Line - VBOXs ['tab', 'nawba', 'mizan', 'form']
        dataframe_type_list = ['tab', 'nawba', 'mizan', 'form']
        self.first_line_labels = list()
        self.first_line_drop_down_menus = list()
        self.first_line_vboxes = list()

        self.second_line_labels = list()
        self.second_line_drop_down_menus = list()
        self.second_line_vboxes = list()

        self.first_title = widgets.Label("   SELECT CHARACTERISTICS: ")

        for characteristic in dataframe_type_list:
            # create a list of labels
            self.first_line_labels.append(widgets.Label("   " + characteristic))
            # create a list of dropdown menus
            key_list = ['all']
            values_list = [0]
            for row in self.cm.get_dataframe(characteristic).index.tolist():
                key_list.append(str(row) + ' - ' + str(self.cm.convert_id(row, characteristic, COLUMNS_NAMES[1])))
                values_list.append(row)
            vals = list(zip(key_list, values_list))
            temp_widget = widgets.Dropdown(options= vals,\
                            value = 0, layout = widgets.Layout(width='96%'))
            self.first_line_drop_down_menus.append(temp_widget)
            # create a list of box
            temp_list = [self.first_line_labels[dataframe_type_list.index(characteristic)], \
                         self.first_line_drop_down_menus[dataframe_type_list.index(characteristic)] ]
            if characteristic == 'form':
                temp_widget = widgets.VBox(temp_list, layout = widgets.Layout(width='31%', border='solid 1px', border_width='1px'))
            else:
                temp_widget = widgets.VBox(temp_list, layout=widgets.Layout(width='23%', border='solid 1px', border_width='1px'))

            self.first_line_vboxes.append(temp_widget)

        # add first line in a HBox
        self.first_line_box = widgets.HBox(self.first_line_vboxes, layout=widgets.Layout(border='solid 1px', justify_content='center'))

        # Second line - VBOXs ['mp3', 'score', 'MB metadata', 'analysis json', 'analysis text', 'wav']
        self.second_title = widgets.Label("   SELECT OPTIONS: ")


        for element in OPTION_LIST:
            # create label
            self.second_line_labels.append(widgets.Label(element))
            key_list = ['all', 'with ' + element, 'without ' + element]
            values_list = [0, 1, 2]
            vals = list(zip(key_list, values_list))
            temp_widget = widgets.Dropdown(options=vals, \
                                           value=0, layout=widgets.Layout(width='96%'))
            self.second_line_drop_down_menus.append(temp_widget)
            # create a list of box
            temp_list = [self.second_line_labels[OPTION_LIST.index(element)], \
                         self.second_line_drop_down_menus[OPTION_LIST.index(element)]]
            temp_widget = widgets.VBox(temp_list, \
                                       layout=widgets.Layout(width='16.6667%', border='solid 1px', border_width='1px'))
            self.second_line_vboxes.append(temp_widget)

        self.second_line_box = widgets.HBox(self.second_line_vboxes, layout=widgets.Layout(border='solid 1px', justify_content='center'))

        # third line - TAB
        self.rmbid_list = self.cm.search_recording('all','all','all','all')
        self.third_title = widgets.Label("   RESULTS (" + str(len(self.rmbid_list))  + "): ")

        self.tabs = widgets.Tab()
        self.update_tab(self.rmbid_list)

        lines_list = [self.first_title, self.first_line_box, self.second_title, self.second_line_box, self.third_title, self.tabs]
        self.main_box = widgets.VBox(lines_list)

        # main box
        display(self.main_box)

        # apply functions
        for drop_down_menu in self.first_line_drop_down_menus:
            drop_down_menu.observe(self.update_recordings, names='value')
        for drop_down_menu in self.second_line_drop_down_menus:
            drop_down_menu.observe(self.update_recordings, names='value')


    def update_tab(self, rmbid_list):
        ''' Update the tab widget with the new list of recodings

        :param rmbid_list: list of Music Brainz id
        '''

        self.tabs_children = list()
        self.results_check_boxes = list()
        self.results_descriptions_rmbid = list()
        self.results_descriptions_name = list()
        self.results_hboxes = list()

        for rmbid in rmbid_list:
            i = rmbid_list.index(rmbid)
            self.results_descriptions_rmbid.append(widgets.Label(rmbid))
            description_name_temp = self.cm.get_recording_translitered_title(rmbid)
            self.results_descriptions_name.append(widgets.Label(description_name_temp))
            style = {'description_width': 'initial'}
            self.results_check_boxes.append(widgets.Checkbox(value=True, \
                                                             description=rmbid, \
                                                             style=style))
            self.results_hboxes.append(widgets.HBox([self.results_check_boxes[i],\
                                                     self.results_descriptions_name[i]]))

        self.third_title.value = "   RESULTS (" + str(len(rmbid_list))  + "): "

        num_tab = int((len(rmbid_list))/self.tab_len)+1
        if (num_tab-1)*self.tab_len == len(rmbid_list):
            num_tab -= 1
        for i in range(0,num_tab):
            min = i * self.tab_len
            max = (i+1) * self.tab_len
            if max > len(rmbid_list):
                max = len(rmbid_list)
            check_boxes_in_tab_temp = list()
            for j in range(min,max):
                check_boxes_in_tab_temp.append(self.results_hboxes[j])
            temp_box = widgets.VBox(check_boxes_in_tab_temp)
            self.tabs_children.append(temp_box)

        self.tabs.children = self.tabs_children
        for i in range(len(self.tabs_children)):
            self.tabs.set_title(i, str(i+1))

    def get_rmbid_list(self):
        ''' Get the list of selected checkbox

        :return: list of selected checkbox
        '''

        selected_rmbid_list = list()
        for i in range(len(self.rmbid_list)):
            if self.results_check_boxes[i].value:
                selected_rmbid_list.append(self.rmbid_list[i])
        return selected_rmbid_list

    def update_recordings(self, change):
        ''' This function is called every time a filter change its value. Run a new search with the new parameter
            and update the tab widget

        :param change: changed value
        '''
        self.third_title.value = "   SEARCHING... "
        attributes_list = list()
        # search recordings with the selected characteristic
        for drop_down_menu in self.first_line_drop_down_menus:
            attributes_list.append(drop_down_menu.value)
        rmbid_results = self.cm.search_recording(attributes_list[0],attributes_list[1],attributes_list[2],attributes_list[3])

        # search recordings with files
        attributes_list = list()
        for drop_down_menu in self.second_line_drop_down_menus:
            attributes_list.append(drop_down_menu.value)
        rmbid_results = check_files_of_rmbid_lists(RECORDINGS_DIR, rmbid_results, OPTION_LIST, attributes_list)
        self.rmbid_list = rmbid_results

        self.update_tab(self.rmbid_list)
