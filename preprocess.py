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


def read_songs(filename):
	songs = get_all_files(filename, '.h5')
	for i in range(1,2):
		file_object= hdf.open_h5_file_read(songs[i])
		echo_ID = hdf.get_song_id(file_object)
		echo_ID = echo_ID.decode("utf-8")
		json_file_name = get_spotify_json("./millionsongdataset_echonest/", echo_ID, ".json")[0]
		with open(json_file_name, 'r') as myfile:
			data=myfile.read()
		# parse file
		obj = json.loads(data)
		spotifyID = obj["response"]["songs"][0]["tracks"][1]["foreign_id"]
		print(get_audio_features(spotifyID))
		file_object.close()
		
def get_audio_features(songID):
	credentials = oauth2.SpotifyClientCredentials(
		client_id='b8aaedbcc185427d96c4f92151d23951',
		client_secret='8be53030941e409cb0e7c6196ad979fd')
	token = credentials.get_access_token()
	sp = spotipy.Spotify(auth=token)
	return sp.audio_features(tracks=[songID])

def main():
	read_songs("./MillionSongSubset/")


if __name__ == '__main__':
   main()