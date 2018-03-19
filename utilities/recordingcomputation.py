#__author__ = "Niccolò Pretto"
#__email__ = "niccolo.pretto_at_dei.unipd.it"
#__copyright__ = "Copyright 2018, Università degli Studi di Padova, Universitat Pompeu Fabra"
#__license__ = "GPL"
#__version__ = "0.1"

import os
import json
import copy
import numpy as np
import scipy.integrate as integrate
from scipy.spatial import distance
# import scipy.spatial.distance.cityblock as cityblock
# import scipy.spatial.distance.euclidean as euclidean
# import scipy.spatial.distance.correlation as correlation
# import scipy.spatial.distance.canberra as canberra

import matplotlib.pyplot as plt
from music21 import *
from collections import deque

from shutil import copyfile
from utilities.constants import *
from utilities.generalutilities import *
from external_utilities.predominantmelodymakam import PredominantMelodyMakam
from external_utilities.pitchfilter import PitchFilter
from external_utilities.toniclastnote import TonicLastNote
from external_utilities.pitchdistribution import PitchDistribution
from external_utilities.corpusbasestatistics import *

NUM_CENTS = 1200

# Parameter from tomato
_pd_params = {'kernel_width': 7.5, 'step_size': 7.5}

list_notes = ['C', 'C#','D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
pair_notes = ['C#', 'D-', 'D#', 'E-', 'F#', 'G-', 'G#', 'A-', 'A#', 'B-' ]

# ---------------------------------------- CHECK FILES -------------------------------------------------

def check_file_of_rmbid(recordings_dir, rmbid, file_type_name):
    ''' Check if a file is within the directory of a recording

    :param recordings_dir: directory where are contained all the recording directories
    :param rmbid: Music Brainz Id of the recording
    :param file_type_name: type of file. Possible values: 'mp3', 'pitch', 'pitch_distribution', 'tonic', 'score'
    :return: True if the file exists. Otherwise False
    '''

    if file_type_name in ACCEPTED_TYPE:
        if file_type_name == 'mp3':
            fn = rmbid + '.mp3'
        else:
            if file_type_name == 'score':
                fn = rmbid + XML_SUFFIX
            else:
                if file_type_name == 'wav':
                    fn = rmbid + '.wav'
                else:
                    fn = file_type_name
    else:
        raise Exception("file_type_name %s doesn't exist" % file_type_name)

    if os.path.exists(os.path.join(recordings_dir, rmbid, fn)):
        return True
    else:
        return False

def check_files_of_rmbid_lists(recordings_dir, rmbid_list, file_type_list, file_type_value):
    ''' Check a list of type of files for every recording in rmbid_list. The options are translated in a
        list of file to check

    :param recordings_dir: directory where are contained all the recording directories
    :param rmbid_list: list of Music Brainz Id to check
    :param file_type_list: list with options.
    :param file_type_value: list with the same length of the previous. Each option MUST have a related flag
    :return: a list of the rmbid of the recordings with all the files selected in file_type_list and file_type_value
    '''
    if len(file_type_list) != len(file_type_value):
        raise Exception("file_type_list and file_type_value have different lengths")

    rmbid_with_all_files = list()
    extended_file_type_list = list()
    extended_file_type_value = list()

    # translate options passed as parameter in files to check
    for i in range(len(file_type_list)):
        if file_type_list[i] == OPTION_LIST[2]: # 'MB metadata'
            extended_file_type_list.append(FN_METADATA)
            extended_file_type_value.append(file_type_value[i])
        else:
            if file_type_list[i] == OPTION_LIST[3]:  # 'analysis json'
                temp_type_list = [FN_PITCH, FN_PITCH_FILT, FN_TONIC_NO_FILT, FN_TONIC_FILT, FN_PD, FN_TONIC_SEC]
                temp_value_list = [file_type_value[i]] * len(temp_type_list)
                extended_file_type_list += temp_type_list
                extended_file_type_value += temp_value_list
            else:
                if file_type_list[i] == OPTION_LIST[4]:  #'analysis text'
                    temp_type_list = [FNT_PITCH, FNT_PITCH_FILT]
                    temp_value_list = [file_type_value[i]] * len(temp_type_list)
                    extended_file_type_list += temp_type_list
                    extended_file_type_value += temp_value_list
                else:
                    extended_file_type_list.append(file_type_list[i])
                    extended_file_type_value.append(file_type_value[i])

    for rmbid in rmbid_list:
        flag = True
        for i in range(len(extended_file_type_list)):
            # check only if the value is different from 'all' (0) or the recording is not discarded yet
            if extended_file_type_value[i] != 0 and flag:
                # translate the value in boolean
                if extended_file_type_value[i] == 1:
                    file_type_boolean = True
                else:
                    file_type_boolean = False
                if check_file_of_rmbid(recordings_dir, rmbid, extended_file_type_list[i]) != file_type_boolean:
                    flag = False
        if flag:
            rmbid_with_all_files.append(rmbid)

    return rmbid_with_all_files

def check_pitch_json_files(recordings_dir, rmbid):
    ''' Check if a recording has the json files related to pitch analysis

    :param recordings_dir: directory where are contained all the recording directories
    :param rmbid: Music Brainz Id of the recording
    :return: return false if one or more of this files are missing
    '''
    p_flag = check_file_of_rmbid(recordings_dir, rmbid, FN_PITCH)
    pf_flag = check_file_of_rmbid(recordings_dir, rmbid, FN_PITCH_FILT)
    t_flag = check_file_of_rmbid(recordings_dir, rmbid, FN_TONIC_NO_FILT)
    tf_flag = check_file_of_rmbid(recordings_dir, rmbid, FN_TONIC_FILT)
    ts_flag = check_file_of_rmbid(recordings_dir, rmbid, FN_TONIC_SEC)
    pd_flag = check_file_of_rmbid(recordings_dir, rmbid, FN_PD)
    return p_flag and pf_flag and t_flag and pd_flag and tf_flag and ts_flag

def check_pitch_txt_files(recordings_dir, rmbid):
    ''' Check if a recording has the txt files related to pitch analysis

    :param recordings_dir: directory where are contained all the recording directories
    :param rmbid: Music Brainz Id of the recording
    :return: return false if one or more of this files are missing
    '''
    p_flag = check_file_of_rmbid(recordings_dir, rmbid, FNT_PITCH)
    pf_flag = check_file_of_rmbid(recordings_dir, rmbid, FNT_PITCH_FILT)
    return p_flag and pf_flag

# -------------------------------------------------- COMPUTATION --------------------------------------------------

def compute_pitch_information(MetadataObject, single_recording_dir, mbid):
    ''' # Calculate pitch, pitch filtered, tonic, tonic_filtered and pitch distribution and save them in five different json files

    :param MetadataObject: Object with all the statistic from Dunya
    :param single_recording_dir:  directory where the audio file is located and where the computed files
            will be saved
    :param mbid: MusicBrainz ID of the recording
    '''

    # Compute pitch
    extractor = PredominantMelodyMakam(filter_pitch=False)
    results = extractor.run(os.path.join(single_recording_dir, mbid + '.mp3'))
    pitch = results['settings']  # collapse the keys in settings
    pitch['pitch'] = results['pitch']

    # Create json file for pitch
    json.dump(pitch, open(os.path.join(single_recording_dir, FN_PITCH), 'w'))

    # Compute pitch filtered
    pitch_filter = PitchFilter()
    pitch_filt = copy.deepcopy(pitch)
    pitch_filt['pitch'] = pitch_filter.run(pitch_filt['pitch'])
    pitch_filt['citation'] = 'Bozkurt, B. (2008). An automatic pitch ' \
                             'analysis method for Turkish maqam music. ' \
                             'Journal of New Music Research, 37(1), 1-13.'

    # Create json file for pitch filtered
    pitch_filt['pitch'] = pitch_filt['pitch'].tolist()
    json.dump(pitch_filt, open(os.path.join(single_recording_dir, FN_PITCH_FILT), 'w'))

    # Compute tonic with the not filtered pitch
    tonic_identifier = TonicLastNote()
    tonic = tonic_identifier.identify(pitch['pitch'])

    # Create json file for tonic not filtered
    json.dump(tonic[0], open(os.path.join(single_recording_dir, FN_TONIC_NO_FILT), 'w'))
    # print(tonic[0]['value'])

    # Compute tonic with filtered pitch
    tonic_identifier_filt = TonicLastNote()
    tonic_filt = tonic_identifier_filt.identify(pitch_filt['pitch'])

    # Create json file for tonic
    json.dump(tonic_filt[0], open(os.path.join(single_recording_dir, FN_TONIC_FILT), 'w'))

    # Compute the tonic using last note for sections
    compute_sections_last_note_tonic_json(MetadataObject, single_recording_dir, mbid)

    pitch_distribution = PitchDistribution.from_hz_pitch(np.array(pitch_filt['pitch'])[:, 1],
                                                                           **_pd_params)
    pitch_distribution.cent_to_hz()
    # pitch_distribution = pitch_distribution.to_dict()

    # Create json file
    pitch_distribution.to_json(os.path.join(single_recording_dir, FN_PD))

def pitch_to_txt(single_recording_dir, type_pitch):
    ''' Convert pitch or pitch filtered json file in a text file readable by Sonic Visualizer

    :param single_recording_dir: directory of the recording
    :param type_pitch: possible values FN_PITCH or FN_PITCH_FILT
    '''

    if type_pitch == FN_PITCH:
        fnt = FNT_PITCH
    else:
        if type_pitch == FN_PITCH_FILT:
            fnt = FNT_PITCH_FILT
        else:
            raise Exception("type_pitch %s incorrect" % type_pitch)

    with open(os.path.join(single_recording_dir, type_pitch)) as json_data:
        d = json.load(json_data)

    p = np.array(d['pitch'])

    file = open(os.path.join(single_recording_dir, fnt), 'w')
    for p_triplet in p:
        file.write(str(p_triplet[0]))
        file.write('\t')
        file.write(str(np.round(p_triplet[1])) + '\n')
    file.close()

def compute_tonic_with_pitch_filtered(single_recording_dir):
    ''' Compute tonic using pitch filtered file and save this one in a json file

    :param single_recording_dir:  directory of the recording
    '''
    # load pitch filtered
    with open(os.path.join(single_recording_dir, FN_PITCH_FILT)) as json_data:
        d = json.load(json_data)

    # Compute tonic
    tonic_identifier = TonicLastNote()
    tonic = tonic_identifier.identify(d['pitch'])

    # Create json file for tonic
    json.dump(tonic[0], open(os.path.join(single_recording_dir, FN_TONIC_FILT), 'w'))

def create_filtered_pitch_json(single_recording_dir):
    ''' Create a json file containing all the value of the filtered pitch

    :param single_recording_dir:
    '''
    # read pitch_json ()
    with open(os.path.join(single_recording_dir, FN_PITCH)) as json_file:
        pitch = json.load(json_file)

    # Compute pitch filtered
    pitch_filter = PitchFilter()
    pitch_filt = copy.deepcopy(pitch)
    pitch_filt['pitch'] = pitch_filter.run(pitch_filt['pitch']).tolist()
    pitch_filt['citation'] = 'Bozkurt, B. (2008). An automatic pitch ' \
                              'analysis method for Turkish maqam music. ' \
                              'Journal of New Music Research, 37(1), 1-13.'

    # Create json file for pitch
    json.dump(pitch_filt, open(os.path.join(single_recording_dir, FN_PITCH_FILT), 'w'))

def compute_list_of(cm, recordings_dir, rmbid_list, json_flag, txt_flag, wav_flag, ow_flag):
    ''' Compute a list of recordings with options indicated by flags

    :param cm: metadata object
    :param recordings_dir: directory where are contained all the recording directories
    :param rmbid_list: Music Brainz Id of the recording
    :param json_flag: if True the pitch analysis will be performed for all the recordings in the list
    :param txt_flag: if True the pitch analysis will be traslated in txt for all the recordings in the list
    :param wav_flag: if True the mp3 will be converted in Wav for all the recordings in the list
    :param ow_flag: if True the new files will overwrite the old ones
    '''

    if not json_flag and not txt_flag and not wav_flag:
        print("No type of data selected")
    else:
        for rmbid in rmbid_list:
            single_recording_dir = os.path.join(recordings_dir, rmbid)
            print()
            print("Computing data for recording " + rmbid)

            if json_flag:
                # If ow_flag is False and all the files exist skip the computation. Otherwise compute the json files
                if not ow_flag and check_pitch_json_files(recordings_dir, rmbid):
                    print(" - Json files already exist")
                else:
                    compute_pitch_information(cm, single_recording_dir, rmbid)
                    print(" - Json files computed")

            if txt_flag:
                # If ow_flag is False and all the files exist skip the computation. Otherwise compute the txt files
                if not ow_flag and check_pitch_txt_files(recordings_dir, rmbid):
                    print(" - Txt files already exist")
                else:
                    # If the related json file are missing, compute also the json files
                    if not check_pitch_json_files(recordings_dir, rmbid):
                        compute_pitch_information(cm, single_recording_dir, rmbid)
                        print(" - Json files computed (necessary for the computation of txt files)")
                    pitch_to_txt(single_recording_dir, FN_PITCH)
                    pitch_to_txt(single_recording_dir, FN_PITCH_FILT)
                    print(" - Txt files computed")

            if wav_flag:
                # If ow_flag is False and the wav file exists, skip the computation. Otherwise convert the mp3 in wav
                if not ow_flag and check_file_of_rmbid(recordings_dir, rmbid, 'wav'):
                    print(" - Wav file already exists")
                else:
                    convert_mp3_in_wav(single_recording_dir, rmbid)
                    print(" - Mp3 converted in wav file")

def convert_mp3_in_wav(single_recording_dir, rmbid):
    # TODO: convert mp3 in wav
    return

def get_sections_last_note_tonic(cm, rmbid):
    ''' Compute the tonic of the selected recording analyzing the tonic obtained from last note of each section

    :param cm: metadata object
    :param rmbid: Music Brainz id of the selected recording
    :return: selected tonic and a list of all the tonics of the sections
    '''

    # get information about the end of every sections of a recording
    section_delimiter_list = [0]

    df_row = cm.get_description_of_single_recording(rmbid)
    for row in df_row.index.values.tolist():
        section_delimiter_list.append(df_row.loc[row, 'end_time'] + 1)  # plus one to round the end second
    #print(section_delimiter_list)

    # if in the description is not present any section for the track an Exception will be raised
    if len(section_delimiter_list) == 1:
        raise Exception("The recording {} has not sections".format(rmbid))

    # extract pitch from json file of FN_PITCH_FILT or FN_PITCH
    with open(os.path.join(RECORDINGS_DIR, rmbid, FN_PITCH_FILT)) as json_file:
        pitch_filt = json.load(json_file)
    # print(len(pitch_filt['pitch']))

    # if the last delimiter is bigger that the last element of the pitch histogram change that value with the last element-1
    if section_delimiter_list[-1] > pitch_filt['pitch'][-1][0]:
        section_delimiter_list[-1] = pitch_filt['pitch'][-2][0]

    # Compute the indexes
    borders_list_indexes = list()

    for i in range(len(section_delimiter_list)):
        min_flag = False
        #max_flag = False
        # print("-------------------------------")
        for j in range(len(pitch_filt['pitch'])):
            # find the index of min value
            if pitch_filt['pitch'][j][0] > section_delimiter_list[i] and min_flag == False:
                # print('MAX' + str(ending_second_list[i]))
                # print(pitch_filt['pitch'][j][0])
                # print(j)
                borders_list_indexes.append(j)
                min_flag = True
    #print(borders_list_indexes)

    # find list of tonic
    tonic_list = list()
    tonic_identifier = TonicLastNote()


    for i in range(len(borders_list_indexes) - 1):
        temp_pitch = pitch_filt['pitch'][borders_list_indexes[i]: borders_list_indexes[i + 1]]
        tonic_temp = tonic_identifier.identify(temp_pitch)
        # print(tonic_temp[0]["value"])
        tonic_list.append(tonic_temp[0]["value"])

    #print(tonic_list)
    # find the tonic
    epsilon = 5
    near_value_counter = list()
    near_values_list = list()
    # for every element search how many other tonic in the list are in the interval [tonic-epsilon,tonic+epsilon]
    for i in tonic_list:
        counter = 0
        near_values = list()
        for j in tonic_list:
            if j > i - epsilon and j < i + epsilon:
                counter += 1
                near_values.append(j)
        near_value_counter.append(counter)
        near_values_list.append(near_values)

    # print(near_value_counter)
    # print(near_values_list)

    # calculate the tonic with the maximum number of other tonic in the interval.
    # the cycle start from the end. In case of equal result the last one win
    max_index = -1
    max_value = 0
    n = len(near_value_counter)
    for i in range(n):
        # print(near_value_counter[n-i-1])
        if near_value_counter[n - i - 1] > max_value:
            max_value = near_value_counter[n - i - 1]
            max_index = n - i - 1
    # print("Max: " + str(max_index))

    # print("Tonic values considered: " + str(near_values_list[max_index]))
    final_tonic = sum(near_values_list[max_index]) / len(near_values_list[max_index])
    #print("Avarage tonic value: " + str(final_tonic))
    return final_tonic, tonic_list

def compute_sections_last_note_tonic_json(cm, single_recording_dir, rmbid):
    tonic = 0
    tonic_list = None
    try:
        tonic, tonic_list = get_sections_last_note_tonic(cm, rmbid)
    except Exception as e:
        print (e)


    tonic_dict = dict()
    tonic_dict["value"] = tonic
    tonic_dict["unit"] = "Hz"
    tonic_dict["procedure"] = "Tonic identification by last note detection in all the sections"
    tonic_dict["citation"] = "To be defined"
    tonic_dict["allTonics"] = tonic_list
    tonic_dict["octaveWrapped"] = False

    json.dump(tonic_dict, open(os.path.join(single_recording_dir, FN_TONIC_SEC), 'w'))

# -------------------------------------------------- PITCH HISTOGRAM --------------------------------------------------

def load_pd(pd_path):
    ''' Load a pitch distribution from a json file and return the values (vals and bins)

    :param pd_path: path of the json file
    :return: vals and related bins of the pitch distribution
    '''
    pd = json.load(open(pd_path))
    vals = pd["vals"]
    bins = pd["bins"]
    return vals, bins

def load_tonic(tonic_path):
    ''' Load the value of the tonic from a json file and return its value

    :param tonic_path: path of the json file
    :return: value of the tonic
    '''
    tnc = json.load(open(tonic_path))
    try:
        return [tnc['value']]
    except KeyError:
        return [work['value'] for work in tnc.values()]

def get_histogram_from_rmbid_list(recording_folder, list_of_rmbid, type_tonic):
    ''' Create an histogram for all the recordings in the list

    :param recording_folder: directory where are contained all the recording directories
    :param list_of_rmbid: list of the recordings that will be trasformed in histogram
    :param type_tonic: type of tonic used for the offset of the pitch histogram [FN_TONIC_FILT, FN_TONIC_NO_FILT]
    :return: coordinates of the overall histogram
    '''

    histograms = {}
    for mbid in list_of_rmbid:
        vals, bins = load_pd(os.path.join(recording_folder, mbid, FN_PD))
        if type_tonic in FN_TONIC_TYPE:
            tonic = load_tonic(os.path.join(recording_folder, mbid, type_tonic))
        else:
            raise Exception("Incorrect type_tonic " + str(type_tonic))

        histograms[mbid] = [[vals, bins], tonic]
        x, y = compute_overall_histogram(histograms)
    return x, y

def get_tonic_value(single_recording_dir, tonic_type):
    ''' Get the value of the tonic in a json file containing the tonic information

    :param single_recording_dir: directory of the recording
    :param tonic_type: possible values FN_TONIC_NO_FILT or FN_TONIC_FILT
    :return:
    '''
    if tonic_type in FN_TONIC_TYPE:
        if os.path.exists(os.path.join(single_recording_dir, tonic_type)):
            with open(os.path.join(single_recording_dir, tonic_type)) as json_data:
                t = json.load(json_data)
        else:
            raise Exception("tonic_type " + str(tonic_type) + " incorrect")

    return t["value"]

def fold_histogram(x,y,cents,shift):
    ''' Fold and shift the values of a pitch distribution histogram by number of cents

    :param x: values of the x axis
    :param y: values of the y axis
    :param cents: length of the folding interval in cents
    :param shift: shift value in cents
    :return: new x and y related to the folded (and shifted) pitch distribution histogram
    '''
    y_folded = [0] * cents

    # unfold value
    for element in x:
        old_index = x.index(element)
        new_index = element % cents
        y_folded[new_index] += y[old_index]

    # shift in cents
    x_shifted = list(range(-shift, cents - shift))
    y_shifted = y_folded[cents - shift:] + y_folded[0:cents - shift]
    # normalize
    y_shifted[:] = [y / sum(y_shifted) for y in y_shifted]

    return x_shifted, y_shifted

def get_customized_histogram_from_rmbid_list(recording_folder, list_of_rmbid, type_tonic, folding_flag, center):

    # calculate histogram
    x, y = get_histogram_from_rmbid_list(recording_folder, list_of_rmbid, type_tonic)

    # fold the histogram. NB: if the histogram is folded, it is already centered
    if folding_flag:
        return fold_histogram(x, y, NUM_CENTS, 50)
    # histograms remain unfolded
    else:
        # center the histogram and delimitate the graph in three octaves [-1800,1800]
        if center == 0:
            bound = 1800
            # create new lists [-bound,bound]
            x_lim = list(range(-bound,bound))
            y_lim = [0] * (2*bound)
            # copy elements that are within the bounds
            for element in x:
                if element > -bound-1 and element < bound:
                    old_index = x.index(element)
                    new_index = x_lim.index(element)
                    y_lim[new_index] = y[old_index]
            return x_lim,y_lim
    return x,y

# -------------------------------------------------- SCORE HISTOGRAM --------------------------------------------------

def get_folded_score_histogram(recordings_folder, mbid_list):
    ''' get the folded score distribution of the notes in score/s

    :param recordings_folder: directory where are contained all the recording directories
    :param mbid_list: list of the recordings to compute
    :return: the two axis values
    '''
    hist = dict((note, 0) for note in list_notes)
    for rmbid in mbid_list:
        rmbid_parsing = converter.parse(os.path.join(recordings_folder, rmbid, rmbid) + '-symbtrxml.xml')
        rmbid_stream = rmbid_parsing.recurse().notes
        for myNote in rmbid_stream:
            #   list_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            #   pair_notes = ['C#', 'D-', 'D#', 'E-', 'F#', 'G-', 'G#', 'A-', 'A#', 'B-']

            # if myNote is a Chord print and discard
            if myNote.isChord:
                print(str(myNote) + " discarded")
            else:
                if myNote.name in list_notes:
                    hist[myNote.name] = hist[myNote.name] + myNote.duration.quarterLength
                else:
                    if myNote.name in pair_notes:
                        i = pair_notes.index(myNote.name)
                        name_translated = pair_notes[i - 1]
                        hist[name_translated] = hist[name_translated] + myNote.duration.quarterLength
                    else:
                        print(str(myNote) + " discarded")
                        # print(rmbid)

    hist_y = [0] * len(list_notes)
    for i in range(len(list_notes)):
        hist_y[i] = hist[list_notes[i]]

    return list_notes, hist_y

def get_unfolded_score_histogram(recordings_folder, mbid_list):
    ''' get unfolded distribution of the notes in score/s

    :param recordings_folder: directory where are contained all the recording directories
    :param mbid_list: list of the recordings to compute
    :return: the two axis values
    '''
    min_octave = 500
    max_octave = 0
    # get the numbers of octaves finding max and min
    for rmbid in mbid_list:
        rmbid_parsing = converter.parse(os.path.join(recordings_folder, rmbid, rmbid) + '-symbtrxml.xml')
        rmbid_stream = rmbid_parsing.recurse().notes
        for myNote in rmbid_stream:
            if not myNote.isChord:
                if myNote.octave < min_octave:
                    min_octave = myNote.octave
                if myNote.octave > max_octave:
                    max_octave = myNote.octave

    # create an empty histogram and the list of notes
    extended_list_notes = list()
    hist = dict()
    for octave in range(min_octave,max_octave+1):
        for note in list_notes:
            hist[str(note) + str(octave)] = 0
            extended_list_notes.append(str(note) + str(octave))

    # fill the histogram
    for rmbid in mbid_list:
        rmbid_parsing = converter.parse(os.path.join(recordings_folder, rmbid, rmbid) + '-symbtrxml.xml')
        rmbid_stream = rmbid_parsing.recurse().notes
        for myNote in rmbid_stream:
            #   list_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            #   pair_notes = ['C#', 'D-', 'D#', 'E-', 'F#', 'G-', 'G#', 'A-', 'A#', 'B-']

            if myNote.isChord:
                print(str(myNote) + " discarded")
            else:
                if myNote.name in list_notes:
                    hist[str(myNote.name)+str(myNote.octave)] += myNote.duration.quarterLength
                else:
                    if myNote.name in pair_notes:
                        i = pair_notes.index(myNote.name)
                        name_translated = pair_notes[i - 1]
                        hist[str(name_translated) + str(myNote.octave)] += myNote.duration.quarterLength
                    else:
                        print(str(myNote) + " discarded")
                        # print(rmbid)

    # convert histogram in list
    hist_y = [0] * len(extended_list_notes)
    for i in range(len(extended_list_notes)):
        hist_y[i] = hist[extended_list_notes[i]]

    return extended_list_notes, hist_y

def get_customized_score_histogram(recordings_folder, mbid_list, fold_flag):
    ''' get the score histogram for a list of rmbid. The histogram could be folded or unfolded.

    :param recordings_folder: directory where are contained all the recording directories
    :param mbid_list: list of the recordings to compute
    :param fold_flag: True = folded. False = unfolded
    :return: the two axis values of the histogram
    '''
    if fold_flag:
        return get_folded_score_histogram(recordings_folder, mbid_list)
    else:
        return get_unfolded_score_histogram(recordings_folder, mbid_list)

def compute_folded_avg_scores(exp):
    # get the different tab in the Training set
    tab_list = exp.get_tab_list()
    notes_avg_tab_list = list()
    y_avg_tab_list = list()
    # compute the avarage score bar for every tab
    for tab in tab_list:
        tab_mbid_list = exp.get_train_mbid_by_tab(tab)
        list_notes_temp, y_temp = get_folded_score_histogram(RECORDINGS_DIR, tab_mbid_list)
        # convert duration in percentage
        tot_y_temp = sum(y_temp)
        y_temp[:] = [y / tot_y_temp for y in y_temp]
        notes_avg_tab_list.append(list_notes_temp)
        y_avg_tab_list.append(y_temp)
    return notes_avg_tab_list, y_avg_tab_list

def convert_folded_scores_in_models(y_list, std):
    # convert the score in distribution using gaussian
    min_bound = -50
    max_bound = 1150
    interval = 100
    num_bins = 1200
    x_distribution = np.linspace(min_bound, max_bound, num_bins)
    y_distribution_list = list()

    for i in range(len(y_list)):
        y_temp = [0] * len(x_distribution)
        for j in range(len(y_list[i])):
            c = std   # standard deviation
            a = y_list[i][j]/ np.sqrt(2 * np.pi * np.power(c, 2.))  # curve height
            b = j * 100  # center of the curve
            y_temp += a * gaussian(x_distribution, b, c)
        tot_curve = sum(y_temp)
        y_temp[:] = [y / tot_curve for y in y_temp]
        y_distribution_list.append(y_temp)

    return x_distribution, y_distribution_list

def get_tab_using_models_from_scores(exp, rmbid, y_models_list, distance_type):
    if not (rmbid in exp.do.cm.get_list_of_recordings()):
        raise Exception("rmbid {} does not exist in the corpora".format(rmbid))
    if not (distance_type in exp.distance_measure):
        raise Exception("distance {} is not a valid distance metric".format(distance))

    # load the pitch distribution of the recording
    vals, bins = load_pd(os.path.join(RECORDINGS_DIR, rmbid, 'audioanalysis--pitch_distribution.json'))
    # use the max peak as fake_tonic
    fake_tonic_index = vals.index(max(vals))
    fake_tonic = bins[fake_tonic_index]

    # calculate the unfolded pitch distribution with the fake tonic
    histograms = {}
    histograms[rmbid] = [[vals, bins], fake_tonic]
    x, y = compute_overall_histogram(histograms)
    x_reference, y_reference = fold_histogram(x, y, NUM_CENTS, 50)

    recording_value_queue = deque(y_reference)


    best_model = 0
    best_distance_value = 0
    best_shift = 0

    for model_index in range(len(exp.y_avg_tab_list)):
        for shift_value in range(NUM_CENTS):
            shifted_recording_values = list(recording_value_queue)
            distance_value = get_distance(exp, shifted_recording_values, y_models_list[model_index], distance_type)
            recording_value_queue.rotate(1)
            if distance_value > best_distance_value:
                best_distance_value = distance_value
                best_model = model_index
                best_shift = shift_value
    tablist = exp.get_tab_list()
    return tablist[best_model]


def get_distance(exp, shifted_recording, tab_model, distance_type):
    # DISTANCE_MEASURES = ["city block (L1)", "euclidian (L2)", "correlation", "intersection", "camberra", "K-L"]

    # city-block (L1)
    if distance_type == exp.distance_measure[0]:
        return 1 - distance.cityblock(shifted_recording, tab_model)
    # euclidian (L2)
    if distance_type == exp.distance_measure[1]:
        return 1 - distance.euclidean(shifted_recording, tab_model)
    # correlation
    if distance_type == exp.distance_measure[2]:
        return np.dot(shifted_recording, tab_model)
    # intersection
    if distance_type == exp.distance_measure[3]:
        temp_min_function = np.minimum(shifted_recording, tab_model)
        return sum(temp_min_function)
    # camberra
    if distance_type == exp.distance_measure[4]:
        return -distance.canberra(shifted_recording, tab_model)

# def get_tab_by_folded_score_cross(do, rmbid, y_avg_tab_list, std):
#     if do.df_dataset is None:
#         raise Exception("Dataset not created or imported")
#     if do.X_mbid_train is None:
#         raise Exception("Dataset not stratified")
#     # if not (rmbid in cm.get_list_of_recordings):
#     #    raise Exception("rmbid is does not exist in the corpora")
#     if [rmbid] in do.X_mbid_train:
#         raise Exception("rmbid {} is in the train set".format(rmbid))
#     # comment this to use with other rmbid
#     if not ([rmbid] in do.X_mbid_test):
#         raise Exception("rmbid {} is not in the test set".format(rmbid))
#
#     # load the pitch distribution of the recording
#     vals, bins = load_pd(os.path.join(RECORDINGS_DIR, rmbid, 'audioanalysis--pitch_distribution.json'))
#     # use the max peak as fake_tonic
#     fake_tonic_index = vals.index(max(vals))
#     fake_tonic = bins[fake_tonic_index]
#
#     # calculate the unfolded pitch distribution with the fake tonic
#     histograms = {}
#     histograms[rmbid] = [[vals, bins], fake_tonic]
#     x, y = compute_overall_histogram(histograms)
#
#     # calculate the folded distribution
#     bests_shift = list()
#     bests_correlation_index = list()
#     bests_corralation_value = list()
#     for i in range(len(y_avg_tab_list)):
#         temp_best_shift = 0
#         temp_best_correlation_index = 0
#         temp_best_corralation_value = 0
#         for j in range(12):
#             x_f, y_f = fold_histogram(x, y, NUM_CENTS, 50 + 100 * j)
#             #print("Fold_histogram_sum {}".format(sum(y_f)))
#
#             temp_corr = np.correlate(y_f, y_avg_tab_list[i], "full")# "valid","same","full"
#
#             xmin = len(y_f)-20
#             xmax = len(y_f)+20+1
#             #print("len temp corr: {}".format(len(temp_corr)))
#             #print("max value: {}".format(max(temp_corr)))
#             #print("max value index: {}".format(temp_corr.tolist().index(max(temp_corr))))
#             #print(temp_corr[xmin:xmax])
#             max_corr_value = np.max(temp_corr[xmin:xmax])
#
#             if max_corr_value > temp_best_corralation_value:
#                 temp_best_corralation_value = max_corr_value
#                 temp_best_correlation_index = temp_corr.tolist().index(max_corr_value)
#                 temp_best_shift = j
#
#             plt.title("Shift: {} + Corr: {}".format(50 + 100 * j,max_corr_value + temp_best_correlation_index%1200))
#             plt.plot(x_f, y_f, label = "recording")
#             plt.plot(x_f, y_avg_tab_list[i], label = "model tab {}".format(do.get_list_of_dataset_tab()[i]))
#             plt.show()
#
#         bests_shift.append(temp_best_shift)
#         bests_correlation_index.append(temp_best_correlation_index)
#         bests_corralation_value.append(temp_best_corralation_value)
#
#     #dict_corr_values = dict(zip(do.get_list_of_dataset_tab(), bests_corralation_value))
#     # print(dict_corr_values)
#     # print(max(bests_corralation_value))
#     index = bests_corralation_value.index(max(bests_corralation_value))
#     tab = do.get_list_of_dataset_tab()[index]
#
#     x_f, y_f = fold_histogram(x, y, NUM_CENTS, 50 + 100 * max(bests_shift))
#     do.save_best_shifted_recording_plot(rmbid, x_f, y_f, y_avg_tab_list[index], 50 + 100 * max(bests_shift), 'std', std, tab)
#     # print("Resulting ")
#     # print("Tab {}".format(tab))
#     # print("Shift {}".format(bests_shift[index]))
#     return tab, max(bests_shift)
#
#
# def get_tab_by_folded_score_auc(do, rmbid, y_avg_tab_list, std):
#     if do.df_dataset is None:
#         raise Exception("Dataset not created or imported")
#     if do.X_mbid_train is None:
#         raise Exception("Dataset not stratified")
#     # if not (rmbid in cm.get_list_of_recordings):
#     #    raise Exception("rmbid is does not exist in the corpora")
#     if [rmbid] in do.X_mbid_train:
#         raise Exception("rmbid {} is in the train set".format(rmbid))
#     # comment this to use with other rmbid
#     if not ([rmbid] in do.X_mbid_test):
#         raise Exception("rmbid {} is not in the test set".format(rmbid))
#
#     # load the pitch distribution of the recording
#     vals, bins = load_pd(os.path.join(RECORDINGS_DIR, rmbid, 'audioanalysis--pitch_distribution.json'))
#     # use the max peak as fake_tonic
#     fake_tonic_index = vals.index(max(vals))
#     fake_tonic = bins[fake_tonic_index]
#
#     # calculate the unfolded pitch distribution with the fake tonic
#     histograms = {}
#     histograms[rmbid] = [[vals, bins], fake_tonic]
#     x, y = compute_overall_histogram(histograms)
#
#     # find list of best shift, max area and correction for every tab
#
#     bests_shift_list = list()
#     bests_correction_list = list()
#     bests_max_area_list = list()
#
#     # for every model
#     for i in range(len(y_avg_tab_list)):
#         best_shift_tab = 0
#         best_max_area_tab = 0
#         best_correction_tab = 0
#
#         #for every 100 cent
#         for j in range(12):
#             temp_max_area_tab = 0
#             temp_best_correction_tab = 0
#
#             # find best correction
#             correction_value = 5
#             for k in range(-correction_value,correction_value):
#                 x_f, y_f = fold_histogram(x, y, NUM_CENTS, 50 + 100 * j + k)
#
#                 # compare the minimum value between every couple of values in the two distribution
#                 temp_min_function = np.minimum(y_f,y_avg_tab_list[i])
#                 temp_max_area = sum(temp_min_function)
#
#                 if temp_max_area > temp_max_area_tab:
#                     temp_max_area_tab = temp_max_area
#                     temp_best_correction_tab = k
#
#             if temp_max_area_tab > best_max_area_tab:
#                 best_max_area_tab = temp_max_area_tab
#                 best_shift_tab = j
#                 best_correction_tab = temp_best_correction_tab
#
#         bests_shift_list.append(best_shift_tab)
#         bests_max_area_list.append(best_max_area_tab)
#         bests_correction_list.append(best_correction_tab)
#
#     index = bests_max_area_list.index(max(bests_max_area_list))
#     tab = do.get_list_of_dataset_tab()[index]
#
#     x_f, y_f = fold_histogram(x, y, NUM_CENTS, 50 + 100 * max(bests_shift_list))
#     do.save_best_shifted_recording_plot(rmbid, x_f, y_f, y_avg_tab_list[index], 50 + 100 * bests_shift_list[index] + bests_correction_list[index], 'std', std, tab)
#     # print("Resulting ")
#     # print("Tab {}".format(tab))
#     # print("Shift {}".format(bests_shift[index]))
#     return tab, 50 + 100 * bests_shift_list[index] + bests_correction_list[index]

# --------------------------------------------------               --------------------------------------------------
# -------------------------------------------------- OLD FUNCTIONS --------------------------------------------------
# -------------------------------------------------- not verified  --------------------------------------------------

# -------------------------------------------------- CHECK FILES --------------------------------------------------

# def check_elaborated_recordings_in_a_folder(recording_folder):
#     ''' Check if in the folder all the recordings contain the following file:
#             mp3, pitch.json, tonic_no_filt.json, pitch_distribution.json
#
#     :param recording_folder: folder where are located the recording directories
#     :return: list of successfully elaborated recordings and the list of not successfully elaborated recordings
#     '''
#     tree = os.walk(recording_folder)
#     mbid_list = list()
#     list_dir = [x[0] for x in tree]
#
#     list_elaborated = list()
#     list_not_elaborated = list()
#
#     for recording in list_dir:
#         mbid = recording.split(recording_folder)[1]
#         if mbid:
#             mbid_list.append(mbid)
#
#     assessments = ['mp3', FN_PITCH, FN_PITCH_FILT, FN_TONIC_NO_FILT, FN_TONIC_FILT, FN_PD]
#     for rmbid in mbid_list:
#         flag = True
#         for ass in assessments:
#             if not check_file_of_rmbid(recording_folder, rmbid, ass):
#                 flag = False
#         if flag:
#             list_elaborated.append(rmbid)
#         else:
#             list_not_elaborated.append(rmbid)
#
#     return list_elaborated, list_not_elaborated
#
# def get_all_recordings_with_score_in_list(recording_folder, mbid_list):
#     with_score_list = list()
#     without_score_list = list()
#     for rmbid in mbid_list:
#         if check_file_of_rmbid(recording_folder, rmbid, 'score'):
#             with_score_list.append(rmbid)
#         else:
#             without_score_list.append(rmbid)
#     return with_score_list, without_score_list
#
# def get_all_analyzed_files(recording_folder):
#     ''' Get the list of all the recording correctly analyzed in a directory
#
#     :param recording_folder: directory where the recordings are stored
#     :return: list of analyzed recordings in a directory
#     '''
#     e, n = check_elaborated_recordings_in_a_folder(recording_folder)
#     return e
#
# def check_list(list_mbid, recording_folder):
#     ''' Check if the recordings in the list are analyzed or not
#
#     :param list_mbid: list of recording that will be checked
#     :param recording_folder: folder where are the recordings directory are stored
#     :return: return two list: one with the analyzed recordings and one for the not analyzed ones
#     '''
#     a = get_all_analyzed_files(recording_folder)
#     analyzed_recording_list = list()
#     not_analyzed_recording_list = list()
#
#     for  rmbid in list_mbid:
#         if rmbid in a:
#             analyzed_recording_list.append(rmbid)
#         else:
#             not_analyzed_recording_list.append(rmbid)
#
#     return analyzed_recording_list, not_analyzed_recording_list
#
# def remove_not_analyzed(list_mbid, recording_folder):
#     ''' Remove the not analyzed recordings form the list in input and return the list of all the analyzed recording
#             in the input list
#
#     :param list_mbid: list of recordings indicated by mbid
#     :param recording_folder: folder where the recording directory are stored
#     :return: list with the mbid of the analyzed recordings. The not analyzed recordings are discarded
#     '''
#     a, n = check_list(list_mbid, recording_folder)
#     return a
#
# def remove_analyzed(list_mbid, recording_folder):
#     a, n = check_list(list_mbid, recording_folder)
#     return n
#
# # -------------------------------------------------- PLOT --------------------------------------------------
#
# def plot_single_pitch_histogram(recording_folder,list_of_rmbid):
#     ''' Plot the Pitch distribution histogram of a list of recordings.
#
#     :param recording_folder: folder where are located the recording directories
#     :param list_of_rmbid: list of recordings indicated with MusicBrainz ids
#     '''
#     # create an histogram for all the recordings in the list
#     histograms = {}
#     for mbid in list_of_rmbid:
#         vals, bins = load_pd(recording_folder + mbid + '/' + FN_PD)
#         tonic = load_tonic(recording_folder + mbid + '/' + FN_TONIC_NO_FILT)
#         histograms[mbid] = [[vals, bins], tonic]
#
#     # calculate the overall histogram from the list of histograms
#     x, y = compute_overall_histogram(histograms)
#
#     # plot the histogram
#     plt.figure(figsize=(15, 6))
#     plt.plot(x, y)
#
#     plt.xlabel("Cents")
#     plt.ylabel("Occurances")
#     plt.title('Overall Histogram')
#     plt.grid()
#     plt.xticks(np.arange(round(min(x)/100)*100-100, round(max(x)/100)*100+100 + 1, 100))
#     plt.xticks(rotation='vertical')
#     plt.show()
#
# def plot_compare_pitch_histogram(recording_folder, mbid_list1, mbid_list2):
#     '''  Plot the Pitch distribution histogram of two lists of recordings in the same plot
#
#     :param recording_folder: folder where are located the recording directories
#     :param mbid_list1: first list of recordings indicated with MusicBrainz ids
#     :param mbid_list2: second list of recordings indicated with MusicBrainz ids
#     '''
#     # create an histogram for all the recordings in the list
#     histograms1 = {}
#     for mbid in mbid_list1:
#         vals, bins = load_pd(recording_folder + mbid + '/' + FN_PD)
#         tonic = load_tonic(recording_folder + mbid + '/' + FN_TONIC_NO_FILT)
#         histograms1[mbid] = [[vals, bins], tonic]
#
#     # calculate the overall histogram from the list of histograms
#     x1, y1 = compute_overall_histogram(histograms1)
#
#     # create an histogram for all the recordings in the list
#     histograms2 = {}
#     for mbid in mbid_list2:
#         vals, bins = load_pd(recording_folder + mbid + '/' + FN_PD)
#         tonic = load_tonic(recording_folder + mbid + '/' + FN_TONIC_NO_FILT)
#         histograms2[mbid] = [[vals, bins], tonic]
#
#     # calculate the overall histogram from the list of histograms
#     x2, y2 = compute_overall_histogram(histograms2)
#
#     # plot the histogram
#     plt.figure(figsize=(15, 6))
#     plt.plot(x1, y1, label='List 1')
#     plt.plot(x2, y2, label='List 2')
#     plt.legend()
#
#     plt.xlabel("Cents")
#     plt.ylabel("Occurances")
#     plt.title('Overall Histogram')
#     plt.grid()
#     print()
#     print()
#     plt.xticks(np.arange(round(min(min(x1),min(x2)) / 100) * 100 - 100, round(max(max(x1), max(x2)) / 100) * 100 + 100, 100))
#     plt.xticks(rotation='vertical')
#     #plt.show()
#
# def plot_single_folded_pitch_histogram(recording_folder,list_of_rmbid, shift_value):
#     ''' Plot the Pitch distribution histogram of a list of recordings.
#
#     :param recording_folder: folder where are located the recording directories
#     :param list_of_rmbid: list of recordings indicated with MusicBrainz ids
#     '''
#     # create an histogram for all the recordings in the list
#     histograms = {}
#     for mbid in list_of_rmbid:
#         vals, bins = load_pd(recording_folder + mbid + '/' + FN_PD)
#         tonic = load_tonic(recording_folder + mbid + '/' + FN_TONIC_NO_FILT)
#         histograms[mbid] = [[vals, bins], tonic]
#
#     # calculate the overall histogram from the list of histograms
#     x, y = compute_overall_histogram(histograms)
#
#     x_unfolded = list(range(NUM_CENTS))
#     y_unfolded = [0] * NUM_CENTS
#
#     # unfold value
#     for element in x:
#         old_index = x.index(element)
#         new_index = element % 1200
#         y_unfolded[new_index] += y[old_index]
#
#     if shift_value == 'auto':
#         score_histogram = get_folded_score_histogram(recording_folder, list_of_rmbid)
#         max_bar_index = score_histogram.index(max(score_histogram))
#         #print("max_bar_index: " + str(max_bar_index))
#         max_index_hist = y_unfolded.index(max(y_unfolded))
#         #print("max_index_hist: " + str(max_index_hist))
#         max_index_zone = int((max_index_hist/50)/2)
#         #print("max_index_50_zone/2: " + str(max_index_zone))
#         #print("max_index_100_zone: " + str(max_index_hist/100))
#         if max_index_zone == 12:
#             shift_value = -50
#         else:
#             shift_value = ((max_bar_index + 1 - max_index_zone)%12) * 100 -50
#
#     # shift in cents
#     x_shifted = list(range(-shift_value,NUM_CENTS-shift_value))
#     y_shifted = y_unfolded[NUM_CENTS-shift_value:] + y_unfolded[0:NUM_CENTS-shift_value]
#
#     # plot the histogram
#     #plt.figure(figsize=(15, 6))
#     plt.plot(x_shifted, y_shifted)
#
#     plt.xlabel("Cents")
#     plt.ylabel("Occurances")
#     plt.title('Overall Unfolded Pitch Histogram')
#     plt.grid()
#     plt.xticks(np.arange(round(min(x_shifted)/100)*100-100, round(max(x_shifted)/100)*100+100 + 1, 100))
#     plt.xticks(rotation='vertical')
#     #plt.show()
#
# def plot_bar_folded_histogram(recording_folder,list_of_rmbid, shift_value):
#     plt.figure(figsize=(15, 6))
#     plt.subplot(121)
#     plot_single_folded_pitch_histogram(recording_folder, list_of_rmbid, shift_value)
#     plt.subplot(122)
#     plot_folded_score_histogram(recording_folder, list_of_rmbid)
#
# # -------------------------------------------------- SCORE --------------------------------------------------
#
# def plot_folded_score_histogram(recordings_folder, mbid_list):
#
#     hist_y = get_folded_score_histogram(recordings_folder, mbid_list)
#
#     x_fake = list((i) for i in range(len(list_notes)))
#
#     plt.bar(x_fake, hist_y, tick_label=list_notes)
#
#     plt.xlabel("Note")
#     plt.ylabel("Duration")
#     plt.title('Overall Score Histogram')
#
#
# def plot_overall_score_histogram_21(recordings_folder, mbid_list):
#
#     newNoteStream = stream.Stream()
#     #noteList = []
#
#     for rmbid in mbid_list:
#         rmbid_parsing = converter.parse(os.path.join(recordings_folder, rmbid, rmbid) + '-symbtrxml.xml')
#         rmbid_stream = rmbid_parsing.recurse().notes # songNotesStream
#
#         for myNote in rmbid_stream:
#             #noteList.append(myNote.name + ' ' + str(myNote.duration.quarterLength))
#             for k in range(0, int(myNote.duration.quarterLength / 0.25)):
#                 newNote = note.Note(myNote.name, type='quarter')
#                 newNote.quarterLength = 0.25
#                 newNoteStream.append(newNote)
#
#     newNoteStream.plot('histogram', 'octave', xHideUnused=False, yAxisLabel='Duration')

# # -------------------------------------------------- COMPUTATION --------------------------------------------------
#
# def compute_pitch(recordings_folder):
#     ''' Compute the json files of pitch creating a txt for the pitch and the filtered version in json and txt
#             for all the recordings of the directory in input
#
#     :param recordings_folder: directory of the recording folders
#     '''
#     tree = os.walk(recordings_folder)
#     list_dir = [x[0] for x in tree]
#
#     mbid_list = list()
#
#     for recording in list_dir:
#         mbid = recording.split(recordings_folder)[1]
#         if mbid:
#             mbid_list.append(mbid)
#
#     for recording in mbid_list:
#         create_filtered_pitch_json(recordings_folder + recording + '/')
#         pitches_json_to_text(recordings_folder + recording + '/')

# def create_andalusian_recording_directory(data_folder_input, data_folder_output, mbid):
#     ''' Create a directory for a recording as in dunya-desktop.
#             Pitch, tonic, pitch, filtered, pitch distribution will be calculated and saved as json file
#             in the directory
#
#     :param data_folder_input: folder where the recording is stored
#     :param data_folder_output: folder where the files will be saved
#     :param mbid: MusicBrainz id of the recording
#     '''
#     # create directory using data_folder_output
#     dst = os.path.join(data_folder_output, mbid)
#     if not os.path.exists(dst):
#         os.makedirs(dst)
#
#     src = os.path.join(data_folder_input, mbid + '.mp3')
#     dst = os.path.join(dst, mbid + '.mp3')
#     # copy audio file into the new directory
#     copyfile(src, dst)
#
#     # calculate pitch, tonic, pitch, filtered, pitch distribution for a recording
#     compute_recording(data_folder_output + mbid + '/', mbid)
# #
# dataframe_list = list()
# correct_tab_percentage_list = list()
# correct_scale_percentage_list = list()
#
# for std in std_list:
#
#     x_model, y_models_list = convert_folded_scores_in_models(y_avg_tab_list, std)
#     do.save_scores_models(notes_avg_tab_list, y_avg_tab_list, x_model, y_models_list, "std", std)
#     rmbid_test_list, correct_tab_list = do.get_test_dataset()
#     correct_tab_counter = 0
#     correct_scale_counter = 0
#     counter = 0
#     ATTRIBUTES_EXP_DATAFRAME = ['mbid', 'tab', 'resulting_tab', 'scale', 'resulting_scale', 'best_shift']
#     experiment_dataframe = pd.DataFrame(columns=ATTRIBUTES_EXP_DATAFRAME)
#     for rmbid_test in rmbid_test_list:
#         print(rmbid_test)
#         index_in_test_list = rmbid_test_list.index(rmbid_test)
#         correct_tab = correct_tab_list[index_in_test_list]
#         correct_scale = get_scale_set(correct_tab)
#         resulting_tab, best_shift = get_tab_by_folded_score_auc(do, rmbid_test, y_models_list, std)
#         resulting_scale = get_scale_set(resulting_tab)
#
#         # insert in dataframe
#         experiment_dataframe.loc[index_in_test_list] = [rmbid_test, correct_tab, resulting_tab, \
#                                                         correct_scale, resulting_scale, best_shift]
#
#         # counters
#         if correct_tab == resulting_tab:
#             tab_detection = True
#             correct_tab_counter += 1
#         else:
#             tab_detection = False
#
#         if correct_scale == resulting_scale:
#             correct_scale_counter += 1
#             scale_detection = True
#         else:
#             scale_detection = False
#
#         print("Shift: {}".format(best_shift))
#         print("Correct_tab: {} - Resulting: {} => {}".format(correct_tab, resulting_tab, tab_detection))
#         print("Correct_scale: {} - Resulting:{} => {}".format(correct_scale, resulting_scale, scale_detection))
#     dataframe_list.append(experiment_dataframe)
#     do.export_experiment_results_to_csv(experiment_dataframe, 'std', std)
#     print()
#     print("------------------------------------")
#     print("Tab Detection: {}".format(correct_tab_counter / len(correct_tab_list)))
#     print("Scale Detection: {}".format(correct_scale_counter / len(correct_tab_list)))
#     print("------------------------------------")
#     print()
#     correct_tab_percentage_list.append(correct_tab_counter / len(correct_tab_list))
#     correct_scale_percentage_list.append(correct_scale_counter / len(correct_tab_list))
