from keras.models import load_model
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import LabelEncoder
import numpy as np
import librosa

# Defining model path
model_path = 'models/lstm_model.h5'
try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading the model:", e)

# Audio file path
audio_file_path = './audio_files/fileName.mp3'

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
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def ambulance_generated():
    if request.method == 'POST':
        emergency_vehicle_detected = test_audio_lstm(audio_file_path=audio_file_path, model=model)
        return jsonify({'emergency_vehicle_detected': emergency_vehicle_detected})
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)