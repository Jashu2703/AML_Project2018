import os
import glob
import hdf5_getters
from msd import fill_attributes
from msd import Song
from msd import get_headers

def main():

    #directory that contain hdf5 files.
    basedir = "E://Engineering//7th_sem//AML//Project//AML_Project2018//data//MillionSongSubset//data//"     
    ext = ".H5"

    #dont remove
    print(get_headers())

    #main loop
    for root, _ , files in os.walk(basedir):        
        files = glob.glob(os.path.join(root,'*'+ext))
        for file in files:

            #open file
            songH5File = hdf5_getters.open_h5_file_read(file)
 
            #Get the song object
            song = Song(str(hdf5_getters.get_song_id(songH5File)))

            #fill the attrobutes in song file
            song = fill_attributes(song,songH5File)

            #dont remove
            print(song.convert_tostring())
            
            songH5File.close()

main()
