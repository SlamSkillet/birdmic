import librosa
import math
import numpy as np
import operator
import os
import shutil
import tflite_runtime.interpreter as tflite
import time

from datetime import datetime, timezone
from dotenv import load_dotenv
from pathlib import Path
from pymongo import MongoClient

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Oakland, CA
LAT = 37.8146343
LNG = -122.2303516
INCLUDE_LIST = []
EXCLUDE_LIST = []

OVERLAP = 0.0
MIN_CONF = 0.7
BASE_DIR = '/home/sam/birdaudio'
MODEL_DIR = f'{BASE_DIR}/model'
RECORDINGS_DIR = f'{BASE_DIR}/recordings'

load_dotenv()

def main():
  global INTERPRETER
  global MONGO_CLIENT

  MONGO_DB_USER_PASS = os.getenv('MONGO_DB_USER_PASS')

  INTERPRETER = load_model()
  client = MongoClient(f'mongodb+srv://raspberrypi:{MONGO_DB_USER_PASS}@deckrecordings.hv661.mongodb.net/observations?retryWrites=true&w=majority')
  collection = client.observations.entries
  
  # Loop through /deck folder
  for audio_file in Path(f'{RECORDINGS_DIR}/deck').iterdir():
    chunks = read_audio_data(audio_file)
    detections = analyze_audio_data(chunks)
    observations = parse_detections(detections)

    # Save to mongo
    docs = []
    for entry in observations:
      date_string = audio_file.name.replace('.wav', '')
      date = datetime.strptime(date_string, '%Y-%m-%d-%H:%M:%S-%z')
      docs.append({
        "species": entry[0],
        "confidence": entry[1].item(),
        "recordedAt": date,
        "filename": audio_file.name
      })

    if len(docs) > 0:
      print('Saving entries to mongo...', docs)
      collection.insert_many(docs)

    # Randomly sample for upload to S3?

    # Move to /processed folder
    shutil.move(audio_file, f'{RECORDINGS_DIR}/processed')

def load_model():

    global INPUT_LAYER_INDEX
    global OUTPUT_LAYER_INDEX
    global MDATA_INPUT_INDEX
    global CLASSES

    print('LOADING TF LITE MODEL...', end=' ')

    # Load TFLite model and allocate tensors.
    myinterpreter = tflite.Interpreter(model_path=f'{MODEL_DIR}/BirdNET_6K_GLOBAL_MODEL.tflite',num_threads=2)
    myinterpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = myinterpreter.get_input_details()
    output_details = myinterpreter.get_output_details()

    # Get input tensor index
    INPUT_LAYER_INDEX = input_details[0]['index']
    MDATA_INPUT_INDEX = input_details[1]['index']
    OUTPUT_LAYER_INDEX = output_details[0]['index']

    # Load labels
    CLASSES = []
    with open(f'{MODEL_DIR}/labels.txt', 'r') as lfile:
        for line in lfile.readlines():
            CLASSES.append(line.replace('\n', ''))

    return myinterpreter

def split_signal(sig, rate, seconds=3.0, minlen=1.5):

    # Split signal with overlap
    sig_splits = []
    for i in range(0, len(sig), int((seconds - OVERLAP) * rate)):
        split = sig[i:i + int(seconds * rate)]

        # End of signal?
        if len(split) < int(minlen * rate):
            break
        
        # Signal chunk too short? Fill with zeros.
        if len(split) < int(rate * seconds):
            temp = np.zeros((int(rate * seconds)))
            temp[:len(split)] = split
            split = temp
        
        sig_splits.append(split)

    return sig_splits

def read_audio_data(path, sample_rate=48000):

    print('READING AUDIO DATA...', end=' ', flush=True)

    # Open file with librosa (uses ffmpeg or libav)
    sig, rate = librosa.load(path, sr=sample_rate, mono=True, res_type='kaiser_fast')

    # Split audio into 3-second chunks
    chunks = split_signal(sig, rate)

    print('DONE! READ', str(len(chunks)), 'CHUNKS.')

    return chunks

def convert_metadata(m):

    # Convert week to cosine
    if m[2] >= 1 and m[2] <= 48:
        m[2] = math.cos(math.radians(m[2] * 7.5)) + 1 
    else:
        m[2] = -1

    # Add binary mask
    mask = np.ones((3,))
    if m[0] == -1 or m[1] == -1:
        mask = np.zeros((3,))
    if m[2] == -1:
        mask[2] = 0.0

    return np.concatenate([m, mask])

def custom_sigmoid(x, sensitivity=1.0):
    return 1 / (1.0 + np.exp(-sensitivity * x))

def predict(sample, sensitivity):
    global INTERPRETER
    # Make a prediction
    INTERPRETER.set_tensor(INPUT_LAYER_INDEX, np.array(sample[0], dtype='float32'))
    INTERPRETER.set_tensor(MDATA_INPUT_INDEX, np.array(sample[1], dtype='float32'))
    INTERPRETER.invoke()
    prediction = INTERPRETER.get_tensor(OUTPUT_LAYER_INDEX)[0]

    # Apply custom sigmoid
    p_sigmoid = custom_sigmoid(prediction, sensitivity)

    # Get label and scores for pooled predictions
    p_labels = dict(zip(CLASSES, p_sigmoid))

    # Sort by score
    p_sorted = sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)

    # Remove species that are on blacklist
    for i in range(min(10, len(p_sorted))):
        if p_sorted[i][0] in ['Human_Human', 'Non-bird_Non-bird', 'Noise_Noise']:
            p_sorted[i] = (p_sorted[i][0], 0.0)

    # Only return first the top ten results
    return p_sorted[:10]

def analyze_audio_data(chunks, week=12, sensitivity=1.0):
    global INTERPRETER

    detections = {}
    start = time.time()
    print('ANALYZING AUDIO...', end=' ', flush=True)

    # Convert and prepare metadata
    mdata = convert_metadata(np.array([LAT, LNG, week]))
    mdata = np.expand_dims(mdata, 0)

    # Parse every chunk
    pred_start = 0.0
    for c in chunks:

        # Prepare as input signal
        sig = np.expand_dims(c, 0)

        # Make prediction
        p = predict([sig, mdata], sensitivity)

        # Save result and timestamp
        pred_end = pred_start + 3.0
        detections[str(pred_start) + ';' + str(pred_end)] = p
        pred_start = pred_end - OVERLAP

    print('DONE! Time', int((time.time() - start) * 10) / 10.0, 'SECONDS')

    return detections

def is_species_eligible(species_name):
  return (species_name in INCLUDE_LIST or len(INCLUDE_LIST) == 0) and (species_name not in EXCLUDE_LIST or len(EXCLUDE_LIST) == 0)

def parse_detections(detections):
  observations = []
  for d in detections:
    for entry in detections[d]:
      [species_name, confidence] = entry
      readable_name = species_name.split('_')[1]

      if confidence >= MIN_CONF and is_species_eligible(species_name):
        observations.append((readable_name, confidence))
  return observations

if __name__ == '__main__':
    main()