import pretty_midi
import librosa.display as display
import matplotlib.pyplot as plt
from IPython.display import Audio

def plot_midi(encoded_midi_file, title = None):
    """_summary_

    Args:
        encoded_midi_file (_type_): _description_
        title (_type_, optional): _description_. Defaults to None.
    """
    plt.figure(figsize=(10, 3))
    display.specshow(encoded_midi_file, y_axis='cqt_note', cmap=plt.cm.hot)
    plt.title(title)

def play_midi(midi_sample):
    """_summary_

    Args:
        midi_object (_type_): _description_
    """
    fs = 44100
    synth = midi_sample.synthesize(fs=fs)
    return Audio([synth], fs)
    
    