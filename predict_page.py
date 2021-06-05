import streamlit as st
import pickle
import numpy as np
import os
import librosa
from speakerfeatures import extract_features
from sound import record


def load_model():
    path = "speaker_models\\"
    gmm_files = [os.path.join(path, file_name) for file_name in os.listdir(path) if file_name.endswith('.gmm')]
    models = [pickle.load(open(file_namename, 'r+b')) for file_namename in gmm_files]
    return models


models = load_model()


def show_predict_page():
    st.title("Speaker Identification")
    st.write("### Welcome to the speaker identification demo. Speaker identification can automatically identify the "
             "person speaking in an audio file given a group of speakers. The input audio is compared against the "
             " provided group of speakers, and in the case there is a match found. The speaker's identity is returned.")
    speakers = ["Nguyen Huy Hoang",
                "Nguyen Manh Dung",
                "Nguyen Phuc Hai",
                "Nguyen Xuan Hoang",
                "Tran Manh Hieu",
                "Tran Trung Thanh"]
    uploaded_audio = st.file_uploader("Upload an audio file")
    if uploaded_audio:
        st.audio(uploaded_audio)
        signal, sr = librosa.load(uploaded_audio)
        print(signal.shape, sr)
        X = extract_features(signal, sr)
    if st.button("Record"):
        with st.spinner(f'Recording for {5} seconds ....'):
            record()
        st.success("Recording completed")
        st.audio("record.wav")
    if st.button('Predict'):
        if not uploaded_audio:
            signal, sr = librosa.load("record.wav")
            print(signal.shape, sr)
            X = extract_features(signal, sr)
        log_likelihood = np.zeros(len(models))
        for i in range(len(models)):
            gmm = models[i]  # checking with each model one by one
            scores = np.array(gmm.score(X))
            log_likelihood[i] = scores.sum()
        winner = np.argmax(log_likelihood)
        st.markdown('**' + speakers[winner] + '**' + " is the one identified speaking in the audio.")