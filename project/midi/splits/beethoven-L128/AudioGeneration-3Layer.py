
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
from music21 import stream
import glob
import json
import os
import numpy
import pickle
import uuid


def main():
    #notes, notes_by_file = process_midis("data/sonate_*.mid")

    #savefile(notes, "model/notes.json")
    #savefile(notes_by_file, "model/notes-by-file.json")

    notes = loadfile("model/notes.json")
    notes_by_file = loadfile("model/notes-by-file.json")

    note_to_int, int_to_note = generate_mappings(notes)

    sequence_length = 100
    x, y = generate_sequences(
        notes_by_file=notes_by_file,
        note_to_int=note_to_int,
        sequence_length=sequence_length)

    n_sequences = len(x)
    n_vocab = len(notes)

    x_norm = numpy.reshape(x, (n_sequences, sequence_length, 1)) / n_vocab
    y = np_utils.to_categorical(y)

    model_file = "model/weights-improvements-197-1.6344.hdf5"
    model = build_model(sequence_length, n_vocab, model_file)

    predicted_notes = generate_notes(model, x, n_vocab, int_to_note)
    create_midi(predicted_notes, model_file)


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
    return pickle.load(open(filename, 'rb'))

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

def build_model(input_width, output_width, model_file = None):
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

    if model_file:
        model.load_weights(model_file)

    return model

# In[4]:

###################
# Train the Model #
###################

def train_model(model, x, y, model_file, epochs=200, batch_size=128):
    checkpoint = ModelCheckpoint(model_file, monitor='loss', save_best_only=True)
    callbacks_list = [checkpoint]
    model.fit(x, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)


def generate_notes(model, x, n_vocab, int_to_note):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    start = numpy.random.randint(0, len(x)-1)

    pattern = x[start]
    prediction_output = []

    # generate 500 notes
    for note_index in range(500):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output


def create_midi(prediction_output, model_file):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='%s-%s.mid' % (os.path.basename(model_file), uuid.uuid4()))

main()
