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
    ouputcsv = "E://Engineering//7th_sem//AML//Project//AML_Project2018//data//msd.csv"
    count=0
    
    #clear contents of file
    with open(ouputcsv,'w') as fd:
        print("Cleared existing contents")

    #Add headers
    with open(ouputcsv,'a') as fd:
        text = get_headers()
        fd.write(text)
        fd.write("\n")

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

            #Add songs line by line
            with open(ouputcsv,'a') as fd:
                text = song.convert_tostring()
                fd.write(text)
                fd.write("\n")
            
            songH5File.close()

            count = count+1

    
    print(count)

main()
