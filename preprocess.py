import hdf5_getters as hdf
import numpy as np
import tensorflow as tf
import matplotlib as plot
import os
import glob
import tables
import spotipy
import json
import spotipy.util as util
import spotipy.oauth2 as oauth2
import re

DATA_SIZE = 1024

def get_all_files(basedir,ext) :
    """
    From a root directory, go through all subdirectories
    and find all files with the given extension.

    param: base directory from which to begin the search, extension to find
    returns: a list of all full path names to each file with ext as extension
    """
    allfiles = []
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+ext))
        for f in files :
            allfiles.append( os.path.abspath(f) )
    return allfiles

def get_danceability(query, sp):
    """
    Makes a query to spotify to get the danceability for each song in our
    dataset

    param: the query to search Spotify, the spotify object to make calls to
    returns: 1 if the song is danceable, 0 if not
    """
    search = sp.search(query, type='track', limit =1)
    if search["tracks"]["items"]:
        trackID = search["tracks"]["items"][0]["uri"]
        features = sp.audio_features(tracks=[trackID])
        if features[0] == None:
            return -1
        danceability = features[0]["danceability"]
        if danceability >= 0.6:
            dance_track = 1
        else:
            dance_track = 0
    else:
        dance_track = -1
    return dance_track

def create_data(songs, labels, broken_labels):
    """
    Goes through all songs in list to find the MFCC timbre data to train on

    param: the list of song absolute file path names, and the danceability
    labels for each
    returns: a [num_songs, 1024, 12] array of data, and a modified labels list
    with only valid songs
    """
    print("creating data...")
    zero_fix_songs = songs[1:len(songs)]
    # unique, counts = np.unique(labels, return_counts=True)
    # num_occurences = dict(zip(unique, counts))
    # num_valid_songs = num_occurences[1] + num_occurences[0]
    valid_songs = np.delete(zero_fix_songs,broken_labels,axis=0)
    data = np.zeros([valid_songs.shape[0], DATA_SIZE, 12])
    acc = 0
    for i in range(0,len(valid_songs)):
        print(i)
        if labels[i-1] != -1:
            song_file = hdf.open_h5_file_read(valid_songs[i])
            MFCC_data = np.array(hdf.get_segments_timbre(song_file))
            if MFCC_data.shape[0] < DATA_SIZE:
                pad_amount = DATA_SIZE - MFCC_data.shape[0]
                pad_array = np.zeros([pad_amount, 12])
                MFCC_data = np.vstack((MFCC_data, pad_array))
            elif MFCC_data.shape[0] > DATA_SIZE:
                MFCC_data = MFCC_data[0:DATA_SIZE, :]
            data[acc] = MFCC_data
            acc+=1
            song_file.close()
    return data


def create_labels(songs, sp):
    # Songs now holds list of all file paths to each song as a string
    """
    Goes through all songs in list to find the danceability from spotify

    param: the list of song absolute file path names, and the Spotify object to
    use to make calls
    returns: a [num_songs] array of danceability labels (-1,0,1)
    """
    print("creating labels...")
    acc = 0
    labels = []
    broken_labels = []
    for i in range(1,len(songs)):
        print(i)
        file_object= hdf.open_h5_file_read(songs[i])
        artist_name = hdf.get_artist_name(file_object).decode("utf-8")
        title = re.sub(r"\(.*\)","",hdf.get_title(file_object).decode("utf-8"))
        query = "artist: " + artist_name + " track: " + title
        label = get_danceability(query, sp)
        if label != -1:
            labels.append(label)
        else:
            broken_labels.append(i)
            acc+=1
        file_object.close()
    print("NUMBER OF LOST SONGS = ", acc)
    return np.array(labels, dtype=np.int32), np.array(broken_labels, dtype=np.int32)

def get_data(filename):
    """
    Creates the data and labels for all valid songs in the dataset

    param: the root file name with all song data
    returns: data: [num_valid_songs, 1024, 12] matrix of MFCC content,
        labels: [num_valid_songs] array of danceability label
    """
    credentials = oauth2.SpotifyClientCredentials(
        client_id='b8aaedbcc185427d96c4f92151d23951',
        client_secret='8be53030941e409cb0e7c6196ad979fd')
    token = credentials.get_access_token()
    sp = spotipy.Spotify(auth=token)
    songs = get_all_files(filename, '.h5')
    # labels, broken_labels = create_labels(songs, sp)
    # with open("labels.txt", "wb") as f1:
    #     np.savetxt(f1, labels.astype(int), fmt='%i', delimiter=",")
    # with open("broken_labels.txt", "wb") as f2:
    #     np.savetxt(f2, broken_labels.astype(int), fmt='%i', delimiter=",")
    with open("labels.txt", "r") as f:
        labels = np.loadtxt(f,dtype=np.int32)
    with open("broken_labels.txt", "r") as f:
        broken_labels = np.loadtxt(f,dtype=np.int32)
    data = create_data(songs, labels, broken_labels)
    print("SIZE OF DATA = ", data.shape)
    print("SIZE OF LABELS = ", labels.shape)
    return data, labels

''' ----------------------------------------------------------------------------
Code to load JSON data that holds Spotify Track ID for each song in dataset if it exists

def get_spotify_json(basedir, songID, ext) :
    """
    From a root directory, go through all subdirectories
    and find all files with the given extension.
    Return all absolute paths in a list.
    """
    allfiles = []
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,songID+ext))
        for f in files :
            allfiles.append( os.path.abspath(f) )
    return allfiles

echo_ID = hdf.get_song_id(file_object)
echo_ID = echo_ID.decode("utf-8")
json_file_name = get_spotify_json("./millionsongdataset_echonest/", echo_ID, ".json")[0]
with open(json_file_name, 'r') as myfile:
  data=myfile.read()
obj = json.loads(data)
  ----------------------------------------------------------------------------
'''
