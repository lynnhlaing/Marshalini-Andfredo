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

def get_all_files(basedir,ext) :
    """
    From a root directory, go through all subdirectories
    and find all files with the given extension.
    Return all absolute paths in a list.
    """
    allfiles = []
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+ext))
        for f in files :
            allfiles.append( os.path.abspath(f) )
    return allfiles

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

def get_danceability(query, sp):
    search = sp.search(query,type='track', limit =5)
    if search["tracks"]["items"]:
        track = search["tracks"]["items"][0]["uri"]
        features = sp.audio_features(tracks=[track])
        danceability = features[0]["danceability"]
        if (danceability>=0.6):
            dance_track = 1
        else:
            dance_track = 0
    else:
        dance_track = -1

    return dance_track

def read_songs(filename,sp):
    songs = get_all_files(filename, '.h5')
    labels = []
    for i in range(1,len(songs)):
        file_object= hdf.open_h5_file_read(songs[i])
        echo_ID = hdf.get_song_id(file_object)
        echo_ID = echo_ID.decode("utf-8")
        json_file_name = get_spotify_json("./millionsongdataset_echonest/", echo_ID, ".json")[0]
        with open(json_file_name, 'r') as myfile:
            data=myfile.read()
        obj = json.loads(data)
        title = re.sub(r"\(.*\)","",hdf.get_title(file_object).decode("utf-8"))
        # print(title)
        query = "artist: " + hdf.get_artist_name(file_object).decode("utf-8") + " track: " + title
        label = get_danceability(query, sp)
        labels.append(label)
        file_object.close()
    print(labels)

def get_audio_features(songID):
    credentials = oauth2.SpotifyClientCredentials(
        client_id='b8aaedbcc185427d96c4f92151d23951',
        client_secret='8be53030941e409cb0e7c6196ad979fd')
    token = credentials.get_access_token()

    return sp.audio_features(tracks=[songID])

def main():
    credentials = oauth2.SpotifyClientCredentials(
        client_id='b8aaedbcc185427d96c4f92151d23951',
        client_secret='8be53030941e409cb0e7c6196ad979fd')
    token = credentials.get_access_token()
    sp = spotipy.Spotify(auth=token)
    read_songs("./MillionSongSubset/", sp)


if __name__ == '__main__':
   main()
