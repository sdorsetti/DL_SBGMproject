import os
import sys
import pretty_midi
import logging
import pandas as pd
import argparse
import sys
from DL_SBGMproject.preprocessing.cleaning import sort_by_size
from DL_SBGMproject.preprocessing.encoding import *
from IPython.display import clear_output

class MidiFileParser():
    def __init__(self, src, max_size,instrument=None,program=None, logging=False):
        
        """_summary_

        Args:
            src (_type_): _description_
        """
        self.src = src
        self.max_size = max_size
        self.instrument = instrument
        self.program = program
    @property
    def clean_folder(self):
        return sort_by_size(self.src, self.max_size)
    @property
    def get_instrument_df(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        instrument_ary = [[]]
        instrument_ary.append(['program', 'is_drum', 'name','filepath'])
        midi_files = self.clean_folder
        for index, file in enumerate(midi_files):
            clear_output(wait=True)
            if self.logging: 
                logging.info("{}/{}: Loading and parsing {}".format(index, len(midi_files), os.path.basename(file)))
            try:
                pm = pretty_midi.PrettyMIDI(file)
                instruments = pm.instruments
                for instrument in instruments:
                    instrument_ary.append([instrument.program, instrument.is_drum, instrument.name.replace(';',''),file])
            except:
                continue
        return pd.DataFrame(data=instrument_ary, columns=["program", "is_drum", "name", "filepath"]).dropna(subset=['name'])
    
    def get_instruments_object(self, filename):
        """_summary_

        Args:
            filename (_type_): _description_

        Returns:
            _type_: _description_
        """
        pm = pretty_midi.PrettyMIDI(filename)
        instruments = pm.instruments
        return instruments
                
    def encoding(self, filename, fs):

        semi_shift = transposer(filename)
        pm = pretty_midi.PrettyMIDI(filename)
        # sampling_freq = 1/ (pm.get_beats()[1]/4)
        sampling_freq = 1/ fs
        l = []
        for j, instrument in enumerate(pm.instruments):
            if instrument.program == 0 and self.instrument in instrument.name.lower():
                for note in instrument.notes:
                    note.pitch += semi_shift

                df = encode_dummies(instrument, sampling_freq).fillna(value=0) 
                df.reset_index(inplace=True, drop=True)
                top_level_index = "{}_{}:{}".format(filename.split("/")[-1], 0, j)
                df['timestep'] = df.index
                df['piano_roll_name'] = top_level_index
                df = df.set_index(['piano_roll_name', 'timestep'])
                l.append(df)
        return pd.concat(l)

    def get_piano_roll_df(self,path_to_csv, fs,transposer_=False, chopster_=False, trim_blanks_ = False, minister_=False,arpster_=False, cutster_=False, padster_=False):
        """
        """
        if self.logging:
            logging.basicConfig(filename='midi_parser.log', level=logging.DEBUG)
            logging.info("*****parsing all files in {} of size lower than {} and with {} playing***********".format(self.src, self.max_size, self.instrument))
        instruments = self.get_instrument_df.iloc[1:,:]
        if self.program == None: 
            self.program = list(instruments["program"].unique())
        instruments = instruments[instruments['program'].isin(self.program)]

        if self.instrument == None: 
            self.instrument = ""
        if self.logging:
            logging.info("******ENCODING*********")
        for i, file in enumerate(instruments['filepath']):
            song_name = os.path.basename(file)  
            try:
                if transposer_:
                    semi_shift = transposer(file)
                pm = pretty_midi.PrettyMIDI(file)
            except Exception as e:
                logging.warning("{}/{}: {}. ENCOUNTERED EXCEPTION {e}".format(i, len(instruments), song_name,str(e)))
                continue
            for j, instrument in enumerate(pm.instruments):
                for note in instrument.notes:
                    note.pitch += semi_shift
                try:
                    df = encode_dummies(instrument, fs).fillna(value=0) 
                except Exception as e:
                    logging.warning("{}/{}: {}. ENCOUNTERED EXCEPTION {}".format(i, len(instruments), song_name,str(e)))
                    continue
                if chopster_:
                    df = chopster(df)
                if trim_blanks_:                 
                    df = trim_blanks(df)
                if df is None:
                    logging.warning("{}/{}: {}. IS AN EMPTY TRACK".format(i, len(instruments), song_name))
                    continue
                if minister_:
                    df = minister(df)   
                if arpster_:         
                    df = arpster(df)
                if padster_: 
                    df = padster(df)
                if cutster_: 
                    df = cutster(df)

                df.reset_index(inplace=True, drop=True)
                top_level_index = "{}_{}:{}".format(song_name, i, j)
                df['timestep'] = df.index
                df['piano_roll_name'] = top_level_index
                df = df.set_index(['piano_roll_name', 'timestep'])
                df.to_csv(path_to_csv, sep=';', mode='a', encoding='utf-8', header=False)
                if self.logging:
                    logging.info("{}/{}: {}. ENCODED SUCCESSFULLY".format(i, len(instruments), song_name))

                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--input_path',metavar = "-p",type = str,help='path to directory where midi files are stored')
    parser.add_argument('--output_path',metavar = "-p",type = str,help='path to directory where to store csv file')
    parser.add_argument("--max_size", metavar = "-s", type = int, help="max size of file")
    parser.add_argument("--instruments", metavar = "-i", type = str, help="list of instruments to keep")

    args = parser.parse_args()
    src = args["input_path"]
    path = args["output_path"]
    instrument = args["instruments"]
    max_size  = int(args["max_size"])

    
    
    mdp = MidiFileParser(src =src, instrument = instrument ,max_size=max_size)
    mdp.get_piano_roll_df(path)



                
