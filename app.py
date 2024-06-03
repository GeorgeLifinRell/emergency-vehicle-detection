from keras.models import load_model
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import LabelEncoder
import numpy as np
import librosa

# Importing Local files
from tracker import video_analysis

# Defining model path
model_path = '/models/Best Model.h5'
model = None
try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading the model:", e)

# Audio file path
audio_file_path = 'audio_files/Emergency audio.wav'

def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=80)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

def test_audio_lstm(audio_file_path, model):
    # Extract features from the audio file
    features = features_extractor(audio_file_path)
    # Reshape the features to match the input shape of the model
    features = features.reshape(1, -1, 80)
    # Make prediction using the model
    prediction = model.predict(features)
    # Decode the predicted class
    predicted_class = LabelEncoder.inverse_transform([np.argmax(prediction)])
    print(predicted_class)
    return predicted_class[0]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/audio')
def audio_simulation():
    return render_template('audio.html')

@app.route('/video')
def video_simulation():
    return render_template('video.html')

@app.route('/ambulance', methods=['GET', 'POST'])
def ambulance_generated():
    global model
    emergency_vehicle_detected = None
    if model is not None:
        emergency_vehicle_detected = test_audio_lstm(audio_file_path=audio_file_path, model=model)
    else:
        print("Model not loaded!")
    print(emergency_vehicle_detected)
    # TODO: uncomment after development
    # time_to_be_added = 10
    # speed, distance = video_analysis()
    # if speed < 30 and distance < 20:
    #     time_to_be_added = 10
    # elif speed < 50 and distance < 50:
    #     time_to_be_added = 20
    # else:
    #     time_to_be_added = 30
    return jsonify({'emergency_vehicle_detected': emergency_vehicle_detected,
                    'time_to_be_added': time_to_be_added})

if __name__ == '__main__':
    app.run(debug=True)