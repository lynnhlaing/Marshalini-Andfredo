import hdf5_getters as hdf
import numpy as np
import tensorflow as tf
import matplotlib as plot


def read_songs(filename):
	songs_file = hdf.open_h5_file_read(filename)

	print("Start time of Fade out = ", hdf.get_start_of_fade_out(songs_file))


def main():
	read_songs("./MillionSongSubset/data/A/A/A/TRAAAVG12903CFA543.h5")


if __name__ == '__main__':
   main()