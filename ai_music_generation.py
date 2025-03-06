# ai_music_generation.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from music21 import converter, instrument, note, chord, stream
import pickle

def load_data():
    with open("notes.pkl", "rb") as f:
        notes = pickle.load(f)
    return notes

def prepare_sequences(notes, sequence_length=100):
    pitchnames = sorted(set(notes))
    note_to_int = {note: number for number, note in enumerate(pitchnames)}
    input_sequences, output_notes = [], []
    
    for i in range(len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        input_sequences.append([note_to_int[n] for n in seq_in])
        output_notes.append(note_to_int[seq_out])
    
    return np.array(input_sequences), np.array(output_notes), note_to_int

def build_model(input_shape):
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=input_shape),
        LSTM(256),
        Dense(256, activation='relu'),
        Dense(len(set(notes)), activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def generate_music(model, note_to_int, int_to_note, sequence_length=100, num_notes=500):
    start_seq = np.random.randint(0, len(note_to_int) - sequence_length - 1)
    pattern = list(note_to_int.keys())[start_seq:start_seq + sequence_length]
    output = []
    
    for _ in range(num_notes):
        input_seq = np.reshape([note_to_int[n] for n in pattern], (1, sequence_length, 1))
        prediction = model.predict(input_seq, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        output.append(result)
        pattern.append(result)
        pattern = pattern[1:]
    
    return output

def save_midi(notes, output_file="output.mid"):
    midi_stream = stream.Stream()
    for pattern in notes:
        if '.' in pattern or pattern.isdigit():
            midi_stream.append(chord.Chord(pattern.split('.')))
        else:
            midi_stream.append(note.Note(pattern))
    midi_stream.write("midi", fp=output_file)
    
if __name__ == "__main__":
    notes = load_data()
    input_sequences, output_notes, note_to_int = prepare_sequences(notes)
    int_to_note = {i: n for n, i in note_to_int.items()}
    
    model = build_model((100, 1))
    model.load_weights("music_model.h5")
    generated_notes = generate_music(model, note_to_int, int_to_note)
    save_midi(generated_notes)
    print("Music generated and saved as output.mid")
