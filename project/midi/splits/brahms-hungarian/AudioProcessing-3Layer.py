
# coding: utf-8

# In[1]:


######################
# Process MIDI Files #
######################

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from music21 import converter
from music21 import instrument
from music21 import chord
from music21 import note
import glob
import json
import os
import numpy
import pickle


def main():
    notes, notes_by_file = process_midis("data/*.mid")

    savefile(notes, "model/notes.json")
    savefile(notes_by_file, "model/notes-by-file.json")

    #notes = loadfile("model/notes.json")
    #notes_by_file = loadfile("model/notes-by-file.json")

    note_to_int, int_to_note = generate_mappings(notes)

    sequence_length = 100
    x, y = generate_sequences(
        notes_by_file=notes_by_file,
        note_to_int=note_to_int,
        sequence_length=sequence_length)

    n_sequences = len(x)
    n_vocab = len(notes)

    x = numpy.reshape(x, (n_sequences, sequence_length, 1)) / n_vocab
    y = np_utils.to_categorical(y)

    model = build_model(sequence_length, n_vocab)

    train_model(
        model=model,
        x=x,
        y=y,
        model_file="model/weights-improvements-{epoch:03d}-{loss:.4f}.hdf5"
    )


def process_midis(glob_dir):
    notes = set()
    notes_by_file = {}

    for input_file in glob.glob(glob_dir):
        print("Ingesting... %s" % input_file)

        midi = converter.parse(input_file)
        parts = instrument.partitionByInstrument(midi)
        if parts and len(parts) > 0:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes

        file_notes = []
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                file_notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                file_notes.append('.'.join(str(n) for n in element.normalOrder))


        notes_by_file[input_file] = file_notes
        notes.update(set(file_notes))

    return sorted(notes), notes_by_file


def mkdir(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

def savefile(content, filename):
    mkdir(filename)
    pickle.dump(content, open(filename, 'wb'))

def loadfile(filename):
    pickle.load(open(filename, 'rb'))

########################
# Build Note Sequences #
########################

def generate_mappings(notes):
    note_to_int = dict((note, number) for number, note in enumerate(notes))
    int_to_note = dict((number, note) for number, note in enumerate(notes))
    return note_to_int, int_to_note


def generate_sequences(notes_by_file, note_to_int, sequence_length = 100):
    x = []
    y = []

    for notes in notes_by_file.values():
        for i in range(0, len(notes) - sequence_length, 1):
            input_sequence = notes[i:i + sequence_length]
            output_sequence = notes[i + sequence_length]

            x.append([note_to_int[c] for c in input_sequence])
            y.append(note_to_int[output_sequence])

    return x, y

# In[3]:

#######################
# Construct the Model #
#######################

def build_model(input_width, output_width):
    model = Sequential()

    model.add(LSTM(
        256,
        input_shape=(input_width, 1),
        return_sequences=True
    ))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(Dropout(0.2))
    model.add(Dense(output_width))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

# In[4]:

###################
# Train the Model #
###################

def train_model(model, x, y, model_file, epochs=200, batch_size=128):
    checkpoint = ModelCheckpoint(model_file, monitor='loss', save_best_only=True)
    callbacks_list = [checkpoint]
    model.fit(x, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)

main()
