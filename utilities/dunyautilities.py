#__author__ = "Niccolò Pretto"
#__email__ = "niccolo.pretto_at_dei.unipd.it"
#__copyright__ = "Copyright 2018, Università degli Studi di Padova, Universitat Pompeu Fabra"
#__license__ = "GPL"
#__version__ = "0.1"

import os
import json
import logging
from compmusic import dunya
import musicbrainzngs as mb

from utilities.constants import *
from utilities.recordingcomputation import *
from utilities.generalutilities import *

mb_logger = logging.getLogger('musicbrainzngs')
mb_logger.setLevel(logging.WARNING)

# Dunya token
dunya.set_token(DUNYA_TOKEN)

# -------------------------------------------------- CHECK --------------------------------------------------

def check_dunya_metadata():
    ''' Check if all the json file relating to metadata are stored in the data directory

    :return: True if all the file exist. Otherwise, return False
    '''
    flag = True
    for fn in DF_LISTS:
        file_name = PREFIX_JSON + fn + '.json'
        if not os.path.exists(os.path.join(DATA_DIR, file_name)):
            flag = False
    return flag

def check_the_score(rmbid):
    ''' Check if the score exist in Dunya using the recording Music Brainz id

    :param rmbid:  Music Brainz id of a recording
    :return: True if the score is in Dunya
    '''
    doc = dunya.docserver.document(rmbid)
    for source in doc['sourcefiles']:
        if source == 'symbtrxml':
            return True
    return False

# -------------------------------------------------- DUNYA METADATA --------------------------------------------------

def collect_metadata():
    """ Create json files with the description of all the info required in json_lists and
            the description of the collection.
    """

    # download the recording list
    info_json = list()
    for i in range(len(DF_LISTS)-1):
        info_json.append(get_unique_json(API_PATH, DF_LISTS[i]))
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        with open(os.path.join(DATA_DIR, PREFIX_JSON + DF_LISTS[i] + '.json'), mode='w') as f:
            json.dump(info_json[i], f)

    # Grab all the recordings metadata from json file
    collectionRecordings = None
    recording_json_path = os.path.join(DATA_DIR, PREFIX_JSON + DF_LISTS[0] + '.json')
    with open(recording_json_path) as json_element:  # this list is manually compiled
        collectionRecordings = json.load(json_element)
    collectionRecordings = collectionRecordings['results']

    # Get the complete description of all the recordings from Dunya
    complete_description = list()
    for element in collectionRecordings:
        complete_description.append(get_recording(API_PATH + DF_LISTS[0] + '/', element['mbid']))

    # save the description in a json file
    with open(os.path.join(DATA_DIR, PREFIX_JSON + DF_LISTS[5] + '.json'), mode='w') as f:
        json.dump(complete_description, f)

def get_unique_json(apipath, type_list):
    ''' Get a unique json file about a specific characteristic of the collection.
            Dunya return json file with maximum 100 records.
            The file merge all the information in a single file

    :param apipath: relative path of the api
    :param type_list: type of information that
    :return: a complete list of the information requested in json format from Dunya
    '''

    info_json = dunya.conn._dunya_query_json(apipath + type_list)
    while info_json['next'] != None:
        # extract the offset of the next json file
        next_json = str(info_json['next']).split(type_list)[1]
        # download the next json file
        temp = dunya.conn._dunya_query_json(apipath + type_list + next_json)
        # append the results to the main file
        info_json['results'] += temp['results']
        # change next
        info_json['next'] = temp['next']

    return info_json

def get_recording(apipath, rmbid):
    """ Get specific information about a recording.

    :param apipath: relative path of the api
    :param rmbid: A recording mbid

    :returns: description of a recording from dunya API

    """
    return dunya.conn._dunya_query_json(apipath + rmbid)

# ---------------------------------------- DUNYA DATA ----------------------------------------

def download_list_of(recordings_dir, rmbid_list, mp3_flag, score_flag, mb_flag, ow_flag):
    ''' Download mp3 and/or score and/or Music Brainz metadata for a list of recordings

    :param recordings_dir: directory where the data will be stored
    :param list_of_rmbid: list of recording Music Brainz id
    :param mp3_flag: if True mp3 will be downloaded
    :param score_flag: if True score will be downloaded (if available)
    :param mb_flag: if True Music Brainz metadata will be downloaded
    :param ow_flag: if True the files will be overwrite
    '''

    if not mp3_flag and not score_flag and not mb_flag:
        print("No type of data selected")
    else:
        # create the main directory if it not exists
        if not os.path.exists(recordings_dir):
            os.makedirs(recordings_dir)
        for rmbid in rmbid_list:
            # create the recording directory if it not exists
            rec_dir = os.path.join(recordings_dir, rmbid)
            if not os.path.exists(rec_dir):
                os.makedirs(rec_dir)

            print()
            print("Downloading data for recording " + rmbid)

            # download the mp3 if flag is True otherwise check if file exist
            if mp3_flag:
                if not ow_flag and check_file_of_rmbid(recordings_dir, rmbid, 'mp3'):
                    print(" - mp3 already exists")
                else:
                    download_mp3_from_dunya(rec_dir, rmbid)
                    print(" - mp3 downloaded")

            if score_flag:
                if not ow_flag and check_file_of_rmbid(recordings_dir, rmbid, 'score'):
                    print(" - Score already exists")
                else:
                    if download_score_from_dunya(rec_dir, rmbid):
                        print(" - Score downloaded")
                    else:
                        print(" - Score not in Dunya")

            if mb_flag:
                if not ow_flag and check_file_of_rmbid(recordings_dir, rmbid, FN_METADATA):
                    print(" - Music Brainz Metadata already exists")
                else:
                    download_metadata_from_music_brainz(rec_dir, rmbid)
                    print(" - Music Brainz Metadata downloaded")

def download_mp3_from_dunya(single_recording_dir, rmbid):
    """Download the mp3 of a document and save it to the specified directory.

    :param single_recording_dir: Where to save the mp3
    :param rmbid: The MBID of the recording
    """
    if not os.path.exists(single_recording_dir):
        raise Exception("Location %s doesn't exist; can't save" % single_recording_dir)

    contents = dunya.docserver.get_mp3(rmbid)
    name = "%s.mp3" % (rmbid)
    path = os.path.join(single_recording_dir, name)
    open(path, "wb").write(contents)

def download_score_from_dunya (single_recording_dir, rmbid):
    ''' Download a symbtrxml of the score in the recording directory.
            It verifies if the score exist before to call the funtion to download the xml

    :param score_folder: directory where the score will be stored
    :param rmbid: recording MusicBrainz ID
    :return: False if the score doesn't exist in Dunya, True the score exist and it is downloaded
    '''
    # check if the score exists
    if not check_the_score(rmbid):
        return False

    if not os.path.exists(single_recording_dir):
        raise Exception("Location %s doesn't exist; can't save" % single_recording_dir)

    score_xml = dunya.docserver.file_for_document(rmbid, 'symbtrxml')
    name = rmbid + XML_SUFFIX
    path = os.path.join(single_recording_dir, name)
    open(path, "wb").write(score_xml)

    return True

# ---------------------------------------- MUSIC BRAINZ ----------------------------------------
def download_metadata_from_music_brainz(single_recording_dir, rmbid):
    ''' download metadata from Music Brainz of a recording

    :param single_recording_dir:
    :param rmbid:
    :return:
    '''
    if not os.path.exists(single_recording_dir):
        raise Exception("Location %s doesn't exist; can't save" % single_recording_dir)

    # save the metadata form Music Brainz in a json file
    with open(os.path.join(single_recording_dir, FN_METADATA), mode='w') as f:
        json.dump(get_mb_recording_info(rmbid), f)

def get_mb_recording_info(rmbid):
    '''  Get info related to the recording passed by input from Music Brainz

    :param rmbid: Music Brainz id of a recording
    :return: info from Music Brainz
    '''
    mb.set_useragent("andalusian_analysis_notebook", "0.1")
    json_file = mb.get_recording_by_id(rmbid, includes=["artist-rels"])
    if not json_file:
        print("Music Brainz return empty metadata")
    # replace inverted comma with inverted commas. Necessary to avoid parsing error
    json_file = replace_invertedcomma(str(json_file))
    return json_file

# def download_collection(data_folder, json_recording_list_name, audio_folder):
#     """ Download the entire collection of recordings listed in a json file
#
#     :param data_folder: directory of the json file with the list of recordings
#     :param json_recording_list_name: name of the json file with the list of recordings (collection)
#     :param audio_folder: directory where all the mp3 of the collection will be saved
#
#     """
#     with open(data_folder + json_recording_list_name) as json_element:
#         collectionRecordings = json.load(json_element)
#
#     counter = 0
#     for recording in collectionRecordings['results']:
# #        if counter == 20:
# #            break
#         counter += 1
#         print(counter)
#         mp3FileURI = download_mp3_from_dunya(recording['mbid'],audio_folder)
#
#
#
# def download_list_of_score(recordings_folder, rmbid_list):
#     ''' Download a list of score from Dunya. If the score doesn't exist, the rmbid of that recording is
#             added to the list that will be returned
#
#     :param score_folder: directory where the score are stored
#     :param rmbid_list: list of recording
#     :return: list of recording without the score
#     '''
#     recordings_without_recordings = list()
#     for rec in rmbid_list:
#         flag = download_score_from_mbid(os.path.join(recordings_folder, rec), rec)
#         if not flag:
#             recordings_without_recordings.append(rec)
#
#     return recordings_without_recordings
#     # TODO: check if the recording is just analyzed and if its directory exist










