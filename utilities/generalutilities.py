#__author__ = "Niccolò Pretto"
#__email__ = "niccolo.pretto_at_dei.unipd.it"
#__copyright__ = "Copyright 2018, Università degli Studi di Padova, Universitat Pompeu Fabra"
#__license__ = "GPL"
#__version__ = "0.1"

import os
import csv
import json
import time
import zipfile
import pandas as pd
import numpy as np

def get_interval(end, start):
    e = time.strptime(end, "%H:%M:%S")
    e_sec = e.tm_sec + e.tm_min*60 + e.tm_hour*3600
    s = time.strptime(start, "%H:%M:%S")
    s_sec = s.tm_sec + s.tm_min*60 + s.tm_hour*3600
    return e_sec-s_sec

def get_seconds(value_in_sec):
    e = time.strptime(value_in_sec, "%H:%M:%S")
    return  e.tm_sec + e.tm_min*60 + e.tm_hour*3600

def get_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "{0}:{1:02d}:{2:02d}".format(h, m, s)
    #return "%d:%02d:%02d" % (h, m, s)

def convert_column_in_time(df, column):
    for row in df.index.tolist():
        df.loc[row, column] = get_time(df.loc[row, column])
    return df

def convert_dataframe_in_time(df):
    for col in df:
        df = convert_column_in_time(df, col)
    return df

def replace_invertedcomma(json_file):
    return json_file.replace("'", '"')

def list_intersection(a,b):
    c = list()
    for e in a:
        if e in b:
            c.append(e)
    return c

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def extract_files_from_zip(recordings_dir, zip_path):
    # if the directory doesn't exist, it create the directory
    if not os.path.exists(recordings_dir):
        os.makedirs(recordings_dir)

    # check if the zip file exists
    if zipfile.is_zipfile(zip_path):
        # extract all the files in the zip file in the directory. NB: it overwrite all the existing files
        zip_ref = zipfile.ZipFile(zip_path, 'r')
        zip_ref.extractall(recordings_dir)
        print("Zip Extracted!")
        zip_ref.close()
    else:
        raise Exception("{} is not a zip file".format(zip_path))

