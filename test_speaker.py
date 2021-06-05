import os
import pickle
import numpy as np
import time
import librosa
from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")

#path to training data
source = "dataset\\"
model_path = "speaker_models\\"
test_file = "test_path.txt"
file_paths = open(test_file, 'r')
num_correct_label = 0
num_test_files = 0
gmm_files = [os.path.join(model_path, fname) for fname in
             os.listdir(model_path) if fname.endswith('.gmm')]

#Load the Gaussian Mixture Models
models = [pickle.load(open(fname, 'r+b')) for fname in gmm_files]
speakers = [fname.split("\\")[-1].split(".gmm")[0] for fname in gmm_files]

# Read the test directory and get the list of test audio files 
for path in file_paths:
    path = path.strip()
    num_test_files += 1
    print(path)
    signal, sr = librosa.load(source + path)
    feature_vector = extract_features(signal, sr)
    log_likelihood = np.zeros(len(models)) 
    
    for i in range(len(models)):
        gmm = models[i]         #checking with each model one by one
        scores = np.array(gmm.score(feature_vector))
        log_likelihood[i] = scores.sum()
    
    winner = np.argmax(log_likelihood)
    if(speakers[winner] == path.split("\\")[0]):
        num_correct_label += 1
    print("\tdetected as - ", speakers[winner])
    time.sleep(1.0)

print('Accuracy:' + str(num_correct_label) + '/' + str(num_test_files))