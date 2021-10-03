from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import os
import audiotochunks
import numpy as np
import time
import pyaudio
import wave
import pickle
from sys import byteorder
from array import array
from struct import pack
from sklearn.neural_network import MLPClassifier
from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr 
import os, glob, pickle
r = sr.Recognizer()
from pydub import AudioSegment

from utils import extract_feature
import sys
import wave,struct,math
import getemotiontext

model = pickle.load(open("result/mlp_classifier.model", "rb"))
THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000

SILENCE = 30

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.
    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


app = Flask(__name__)
UPLOAD_FOLDER="uploads"
ALLOWED_EXTENSIONS = {'wav'}


def allowed_file(filename):
    return '.' in filename and \
    filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/")
def about_page():
    return render_template('index.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
 if request.method == 'POST':
      f = request.files['file']
      if f and allowed_file(f.filename): 
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)))
        epoch = time.time()
        user_id = f.filename 
        id = "%s_%d" % (user_id, epoch)
        audiotochunks.get_large_audio_transcription("uploads/"+f.filename,id)
        

        # Path
        parent_dir="uploads//audio-chunks//"+id+"//"
        directory = "output"
        path = os.path.join(parent_dir, directory)

        # Create the directory
        # 'GeeksForGeeks' in
        # '/home / User / Documents'
        os.mkdir(path)
        ii=1
        row=[]
        arr=[[]]
        for filename in glob.glob("uploads//audio-chunks//"+id+"//chunk**.wav"):
            sound = AudioSegment.from_wav(filename)
            sound = sound.set_channels(1)
            if (ii<10):
                sound.export(f"uploads//audio-chunks//"+id+"//output//chunk0"+str(ii)+".wav", format="wav")
            if(ii>10):
                sound.export(f"uploads//audio-chunks//"+id+"//output//chunk"+str(ii)+".wav", format="wav")
            ii=ii+1
        for filename in glob.glob("uploads//audio-chunks//"+id+"//output//chunk**.wav"):
            row=[]
            print(filename)
            features = extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1, -1)
        # predict
            result = model.predict(features)[0]
        # show the result !

            with sr.AudioFile(filename) as source:
                audio_listened = r.record(source)
            # try converting it to text
            try:
                text = r.recognize_google(audio_listened,language="fr_FR")
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
                print(filename, ":", text)
                a,b,c=getemotiontext.predict_emotion(text)
                row.append(filename)
                row.append(text)
                row.append(a[0])
                
            print("result:", result)
            row.append(result)
            row.append(b)
            arr.append(row)
            print("result:", arr)
        return render_template('index2.html',arr=arr)
      return 'FILE EXTENSIONS ERROR'

if __name__=='__main__':
    app.run(debug=True)
