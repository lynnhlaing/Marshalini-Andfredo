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
    search = sp.search(query,type='track', limit =5)
    if search["tracks"]["items"]:
        trackID = search["tracks"]["items"][0]["uri"]
        features = sp.audio_features(tracks=[trackID])
        danceability = features[0]["danceability"]
        if danceability >= 0.6:
            dance_track = 1
        else:
            dance_track = 0
    else:
        dance_track = -1
    return dance_track

def create_data(songs, labels):
    """
    Goes through all songs in list to find the MFCC timbre data to train on

    param: the list of song absolute file path names, and the danceability
    labels for each
    returns: a [num_songs, 1024, 12] array of data, and a modified labels list
    with only valid songs
    """
    print("creating data...")
    unique, counts = np.unique(labels, return_counts=True)
    num_occurences = dict(zip(unique, counts))
    num_valid_songs = num_occurences[1] + num_occurences[0]
    data = np.zeros([num_valid_songs, DATA_SIZE, 12])
    new_labels = []
    acc = 0
    for i in range(1,len(songs)-1):
        if labels[i] != -1:
            new_labels.append(labels[i])
            song_file = hdf.open_h5_file_read(songs[i])
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
    new_labels = np.array(new_labels)
    return data, labels

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
    labels = np.zeros([len(songs) -1])
    for i in range(1,len(songs)):
        file_object= hdf.open_h5_file_read(songs[i])
        artist_name = hdf.get_artist_name(file_object).decode("utf-8")
        title = re.sub(r"\(.*\)","",hdf.get_title(file_object).decode("utf-8"))
        query = "artist: " + artist_name + " track: " + title
        label = get_danceability(query, sp)
        if label == -1:
            acc+=1
        labels[i-1] = label
        file_object.close()
        print(i)
    print("NUMBER OF LOST SONGS = ", acc)
    return labels

def get_audio_features(songID, sp):
    """
    Returns a json object containing the audio features of a given song

    param: the unique spotify TrackID, the spotify object
    returns: JSON object of features
    """
    return sp.audio_features(tracks=[songID])

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
    primary_labels = create_labels(songs, sp)
    np.savetxt('test.txt', primary_labels, delimiter=',')
    data,labels = create_data(songs, primary_labels)
    print("SIZE OF DATA = ", data.shape)
    print("SIZE OF LABELS = ", labels.shape)

    return data, labels

# if __name__ == '__main__':
#    get_data("./MillionSongSubset/")

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
