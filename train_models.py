import pickle
import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")

#path to training data
source = "dataset\\train\\"
#path where training speakers will be saved
dest = "speaker_models\\"
train_file = "train_path.txt"
file_paths = open(train_file,'r')
count = 1
# Extracting features for each speaker (5 files per speakers)
features = np.asarray(())
for path in file_paths:    
    path = path.strip()   
    #print(path)
    # read the audio
    signal, sr = librosa.load(source + path)
    #print(signal.shape, sr)
    #extract 40 dimensional MFCC & delta MFCC features
    vector = extract_features(signal, sr)
    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
    # when features of 5 files of speaker are concatenated, then do model training
    if count == 5:
        gmm = GaussianMixture(n_components=16, max_iter=200, covariance_type='diag', n_init=3)
        gmm = GaussianMixture()
        gmm.fit(features)
        # dumping the trained gaussian model
        picklefile = path.split("\\")[0]+".gmm"
        pickle.dump(gmm, open(dest + picklefile, 'wb'))
        print('+ modeling completed for speaker:', picklefile, " with data point = ", features.shape)
        features = np.asarray(())
        count = 0
    count = count + 1