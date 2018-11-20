import hdf5_getters
import numpy as np


class Song:
    songCount = 0

    def __init__(self, songID):
        self.id = songID
        Song.songCount += 1
        
        self.analysisSampleRate = None
        self.artistDigitalID = None
        self.artistFamiliarity = None
        self.artistHotness = None
        self.artistID = None
        self.artistLatitude = None
        self.artistLocation = None
        self.artistLongitude = None
        self.artistmbID = None
        self.artistmbTags = None
        #array
        self.artistmbTagsCount_mean = None
        self.artistmbTagsCount_var = None
        self.artistName = None
        self.artistPlayMeID = None
        self.artistTerms = None
        #array
        self.artistTermsFreq_mean = None
        self.artistTermsFreq_var = None
        #array
        self.artistTermsWeight_mean = None
        self.artistTermsWeight_var = None
        self.audioMD5 = None
        #array
        self.barsConfidence_mean = None
        self.barsConfidence_var = None
        #array
        self.barsStart_mean = None
        self.barsStart_var = None
        #array
        self.beatsConfidence_mean = None
        self.beatsConfidence_var = None
        #array
        self.beatsStart_mean = None
        self.beatsStart_var = None
        self.danceability = None
        self.duration = None
        self.endOfFadeIn = None
        self.energy = None
        self.key = None
        self.keyConfidence = None
        self.loudness = None
        self.mode = None
        self.modeConfidence = None
        self.release = None
        self.releaseDigitalID = None
        #array
        self.sectionsConfidence_mean = None
        self.sectionsConfidence_var = None
        #array
        self.sectionsStart_mean = None
        self.sectionsStart_var = None
        #array
        self.segementsConfidence_mean = None
        self.segementsConfidence_var = None
        #array
        self.segmentsLoudnessMax_mean = None
        self.segmentsLoudnessMax_var = None
        #array
        self.segmentsLoudnessMaxTime_mean = None
        self.segmentsLoudnessMaxTime_var = None
        #array
        self.segmentsLoudnessMaxStart_mean = None
        self.segmentsLoudnessMaxStart_var = None
        #array2d
        self.segmentsPitches_mean = []
        self.segmentsPitches_var = []
        
        #self.segmentsStart = None
        self.segmentsStart_mean = None
        self.segmentsStart_var = None
        #self.segmentsTimbre = None
        self.segmentsTimbre_mean = []
        self.segmentsTimbre_var = []
        
        self.similarArtists = None
        self.songHotttnesss = None
        self.startOfFadeOut = None
        
        #array
        self.tatumsConfidence_mean = None
        self.tatumsConfidence_var = None
        
        #array
        self.tatumsStart_mean = None
        self.tatumsStart_var = None
        self.tempo = None
        self.timeSignature = None
        self.timeSignatureConfidence = None
        self.title = None
        self.trackID = None
        self.trackDigitalID = None
        self.year = None

    #converts every song object to string 
    def convert_tostring(self):

            array2d_pitches=""
            array2d_timbre=""

            for i in range(0,12):
                array2d_pitches += str(self.segmentsPitches_mean[i]) +  "\t" 
                array2d_pitches += str(self.segmentsPitches_var[i]) + "\t"
                array2d_timbre += str(self.segmentsTimbre_mean[i]) + "\t"
                array2d_timbre += str(self.segmentsTimbre_var[i]) + "\t"
            
            return (
                    str(self.analysisSampleRate) + "\t" + str(self.artistDigitalID) + "\t" + str(self.artistFamiliarity) +  "\t" + str(self.artistHotness)  + "\t" + 
                    str(self.artistID)           + "\t" + str(self.artistLatitude)  + "\t" + str(self.artistLocation)    + "\t" + str(self.artistLongitude) + "\t" +
                    str(self.artistmbID)         + "\t" + str(self.artistmbTagsCount_mean) + "\t" + str(self.artistmbTagsCount_var) + "\t" + str(self.artistName) + "\t" +
                    str(self.artistPlayMeID)     + "\t" + str(self.artistTermsFreq_mean)  + "\t" + str(self.artistTermsFreq_var) + "\t" + str(self.artistTermsWeight_mean) + "\t" +
                    str(self.artistTermsWeight_var) + "\t" + str(self.audioMD5) + "\t" + str(self.barsConfidence_mean) + "\t" + str(self.barsConfidence_var) + "\t" + 
                    str(self.barsStart_mean)  + "\t" + str(self.barsStart_var)  + "\t" + str(self.beatsConfidence_mean) + "\t" + str(self.beatsConfidence_var) + "\t" +
                    str(self.beatsStart_mean) + "\t" +str(self.beatsStart_var) +  "\t" + str(self.danceability) + "\t" + str(self.duration) + "\t" + str(self.endOfFadeIn) + "\t" +
                    str(self.energy) + "\t" + str(self.key) + "\t" +str(self.keyConfidence) + "\t" + str(self.loudness) + "\t" + str(self.mode) + "\t" + str(self.modeConfidence) +  "\t" +
                    str(self.release) + "\t" +str(self.releaseDigitalID) + "\t" + str(self.sectionsConfidence_mean) + "\t" +str(self.sectionsConfidence_var) +  "\t" + str(self.sectionsStart_mean) + "\t" + 
                    str(self.sectionsStart_var) + "\t" + str(self.segementsConfidence_mean) + "\t" + str(self.segementsConfidence_var) + "\t" + str(self.segmentsLoudnessMax_mean)  + "\t" +
                    str(self.segmentsLoudnessMax_var)  + "\t" + str(self.segmentsLoudnessMaxTime_mean) + "\t" + str(self.segmentsLoudnessMaxTime_var) + "\t" + str(self.segmentsLoudnessMaxStart_mean) + "\t" +
                    str(self.segmentsLoudnessMaxStart_var) + "\t" + str(self.segmentsStart_mean) + "\t" + str(self.segmentsStart_var) + "\t" +
                    str(self.songHotttnesss) + "\t" +str(self.startOfFadeOut) +  "\t" +str(self.tatumsConfidence_mean) + "\t" +str(self.tatumsConfidence_var) + "\t" +
                    str(self.tatumsStart_mean) +   "\t" +str(self.tatumsStart_var) + "\t" +str(self.tempo) + "\t" +str(self.timeSignature) + "\t" +str(self.timeSignatureConfidence) + "\t" +
                    str(self.title) + "\t" +str(self.trackID) + "\t" + str(self.trackDigitalID) + "\t" +str(self.year) + "\t" + str(array2d_pitches) + str(array2d_timbre) + 
                    str(self.artistmbTags)  + "\t" + str(self.artistTerms) + "\t" +str(self.similarArtists)  #string array
                )


#-------------------------Utility functions---------------------------
def convert_array_to_meanvar(array):
    return np.mean(array),np.var(array,ddof=1)

def covert_2darray_to_meanvar(array):
    return array.mean(axis=0), array.var(axis=0)

def convert_array_to_string(array):
    return " ".join( map( str, array ) )

#given song object fills data from file
def fill_attributes(song,songH5File):
 
    #----------------------------non array attributes-------------------------------
    song.analysisSampleRate = str(hdf5_getters.get_analysis_sample_rate(songH5File))
    song.artistDigitalID = str(hdf5_getters.get_artist_7digitalid(songH5File))
    song.artistFamiliarity = str(hdf5_getters.get_artist_familiarity(songH5File))
    song.artistHotness = str(hdf5_getters.get_artist_hottness(songH5File))
    song.artistID = str(hdf5_getters.get_artist_id(songH5File))
    song.artistLatitude = str(hdf5_getters.get_artist_latitude(songH5File))
    song.artistLocation = str(hdf5_getters.get_artist_location(songH5File))
    song.artistLongitude = str(hdf5_getters.get_artist_longitude(songH5File))
    song.artistmbID = str(hdf5_getters.get_artist_mbid(songH5File))
    song.artistName = str(hdf5_getters.get_artist_name(songH5File))     
    song.artistPlayMeID = str(hdf5_getters.get_artist_playmeid(songH5File))
    song.audioMD5 = str(hdf5_getters.get_audio_md5(songH5File))
    song.danceability = str(hdf5_getters.get_danceability(songH5File))
    song.duration = str(hdf5_getters.get_duration(songH5File))
    song.endOfFadeIn = str(hdf5_getters.get_end_of_fade_in(songH5File))
    song.energy = str(hdf5_getters.get_energy(songH5File))
    song.key = str(hdf5_getters.get_key(songH5File))
    song.keyConfidence = str(hdf5_getters.get_key_confidence(songH5File))
    song.segementsConfidence = str(hdf5_getters.get_segments_confidence(songH5File))
    song.segementsConfidence = str(hdf5_getters.get_sections_confidence(songH5File))
    song.loudness = str(hdf5_getters.get_loudness(songH5File))
    song.mode = str(hdf5_getters.get_mode(songH5File))
    song.modeConfidence = str(hdf5_getters.get_mode_confidence(songH5File))
    song.release = str(hdf5_getters.get_release(songH5File))
    song.releaseDigitalID = str(hdf5_getters.get_release_7digitalid(songH5File))
    song.songHotttnesss = str(hdf5_getters.get_song_hotttnesss(songH5File))
    song.startOfFadeOut = str(hdf5_getters.get_start_of_fade_out(songH5File))
    song.tempo = str(hdf5_getters.get_tempo(songH5File))
    song.timeSignature = str(hdf5_getters.get_time_signature(songH5File))
    song.timeSignatureConfidence = str(hdf5_getters.get_time_signature_confidence(songH5File))
    song.title =  str(hdf5_getters.get_title(songH5File))
    song.trackID = str(hdf5_getters.get_track_id(songH5File))
    song.trackDigitalID = str(hdf5_getters.get_track_7digitalid(songH5File))
    song.year = str(hdf5_getters.get_year(songH5File))

    #-------------------------------array attributes--------------------------------------
    #array float
    song.beatsStart_mean,song.beatsStart_var = convert_array_to_meanvar(hdf5_getters.get_beats_start(songH5File))
    #array float
    song.artistTermsFreq_mean, song.artistTermsFreq_var = convert_array_to_meanvar(hdf5_getters.get_artist_terms_freq(songH5File))
    #array float
    song.artistTermsWeight_mean, song.artistTermsWeight_var = convert_array_to_meanvar(hdf5_getters.get_artist_terms_weight(songH5File))
    #array int
    song.artistmbTagsCount_mean,song.artistmbTagsCount_var = convert_array_to_meanvar(hdf5_getters.get_artist_mbtags_count(songH5File))
    #array float
    song.barsConfidence_mean,song.barsConfidence_var = convert_array_to_meanvar(hdf5_getters.get_bars_confidence(songH5File))
    #array float
    song.barsStart_mean,song.barsStart_var = convert_array_to_meanvar(hdf5_getters.get_bars_start(songH5File))
    #array float
    song.beatsConfidence_mean,song.beatsConfidence_var = convert_array_to_meanvar(hdf5_getters.get_beats_confidence(songH5File))
    #array float
    song.sectionsConfidence_mean , song.sectionsConfidence_var =  convert_array_to_meanvar(hdf5_getters.get_sections_confidence(songH5File))
    #array float
    song.sectionsStart_mean , song.sectionsStart_var =  convert_array_to_meanvar(hdf5_getters.get_sections_start(songH5File))
    #array float
    song.segmentsConfidence_mean, song.segmentsConfidence_var = convert_array_to_meanvar(hdf5_getters.get_segments_confidence(songH5File))
    #array float
    song.segmentsLoudness_mean, song.segmentsLoudness_var = convert_array_to_meanvar(hdf5_getters.get_segments_loudness_max(songH5File))
    #array float
    song.segmentsLoudnessMaxTime_mean,song.segmentsLoudnessMaxTime_var  = convert_array_to_meanvar(hdf5_getters.get_segments_loudness_max_time(songH5File))
     #array float
    song.segmentsLoudnessMaxStart_mean , song.segmentsLoudnessMaxStart_var = convert_array_to_meanvar(hdf5_getters.get_segments_loudness_start(songH5File))
    #array float
    song.segmentsStart_mean,song.segmentsStart_var  = convert_array_to_meanvar(hdf5_getters.get_segments_start(songH5File))
    #array float
    song.tatumsConfidence_mean,song.tatumsConfidence_var  = convert_array_to_meanvar(hdf5_getters.get_tatums_confidence(songH5File))
    #array float
    song.tatumsStart_mean, song.tatumsStart_var = convert_array_to_meanvar(hdf5_getters.get_tatums_start(songH5File))
    #array2d float
    song.segmentsTimbre_mean,song.segmentsTimbre_var  = covert_2darray_to_meanvar(hdf5_getters.get_segments_timbre(songH5File))
    #array2d float
    song.segmentsPitches_mean,song.segmentsPitches_var = covert_2darray_to_meanvar(hdf5_getters.get_segments_pitches(songH5File))

    #------------------------array string attributes------------------------
    song.similarArtists = convert_array_to_string(hdf5_getters.get_similar_artists(songH5File)) #array string
    song.artistTerms = convert_array_to_string(hdf5_getters.get_artist_terms(songH5File))       #array string
    song.artistmbTags = convert_array_to_string(hdf5_getters.get_artist_mbtags(songH5File))     #array string
    
    return song

#function which returns string of headers
def get_headers():
    basic_headers = ("analysisSampleRate\tartistDigitalID\tartistFamiliarity\tartistHotness\t"+
                    "artistID\tartistLatitude\tartistLocation\tartistLongitude\t"+
                    "artistmbID\tartistmbTagsCount_mean\tartistmbTagsCount_var\tartistName\t"+
                    "artistPlayMeID\tartistTermsFreq_mean\tartistTermsFreq_var\tartistTermsWeight_mean\t"+
                    "artistTermsWeight_var\taudioMD5\tbarsConfidence_mean\tbarsConfidence_var\t"+ 
                    "barsStart_mean\tbarsStart_var\tbeatsConfidence_mean\tbeatsConfidence_var\t"+
                    "beatsStart_mean\tbeatsStart_var\tdanceability\tduration\tendOfFadeIn\t"+
                    "energy\tkey\tkeyConfidence\tloudness\tmode\tmodeConfidence\t"+
                    "release\treleaseDigitalID\tsectionsConfidence_mean\tsectionsConfidence_var\tsectionsStart_mean\t"+ 
                    "sectionsStart_var\tsegementsConfidence_mean\tsegementsConfidence_var\tsegmentsLoudnessMax_mean\t"+
                    "segmentsLoudnessMax_var\tsegmentsLoudnessMaxTime_mean\tsegmentsLoudnessMaxTime_var\tsegmentsLoudnessMaxStart_mean\t"+
                    "segmentsLoudnessMaxStart_var\tsegmentsStart_mean\tsegmentsStart_var\t"+
                    "songHotttnesss\tstartOfFadeOut\ttatumsConfidence_mean\ttatumsConfidence_var\t"+
                    "tatumsStart_mean\ttatumsStart_var\ttempo\ttimeSignature\ttimeSignatureConfidence\t"+
                    "title\ttrackID\ttrackDigitalID\tyear\t")
    
    for i in range(0,12):
        basic_headers += "segmentsTimbre_mean_"+ str(i+1) + "\t"
        basic_headers += "segmentsTimbre_var_" + str(i+1) + "\t"
    
    for i in range(0,12):
        basic_headers += "segmentsPitches_mean_"+ str(i+1) + "\t"
        basic_headers += "segmentsPitches_var_" + str(i+1) + "\t"

    basic_headers += "artistmbTags\tartistTerms\tsimilarArtists"

    return basic_headers
    