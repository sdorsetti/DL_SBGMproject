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
    

def piano_roll_to_pretty_midi(piano_roll, fs=32, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    piano_roll = piano_roll.T
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm
