# -*- coding: utf-8 -*-
"""
@Note :  20 dim MFCC
         20 dim delta computation on MFCC features. 
@output : It returns 40 dimensional feature vectors for an audio.
"""

import numpy as np
from sklearn import preprocessing
import librosa


def extract_features(signal, sr):
    """extract 20 dim mfcc features from an audio, performs CMS and combines 
    delta to make it 40 dim feature vector"""

    mfcc_feature = librosa.feature.mfcc(signal, n_mfcc=20, sr=sr)
    mfcc_feature = preprocessing.scale(mfcc_feature, axis=1)
    mfcc_feature = np.transpose(mfcc_feature)
    #print(mfcc_feature.shape)
    delta = librosa.feature.delta(mfcc_feature)
    #print(delta.shape)
    combined = np.hstack((mfcc_feature, delta))
    #print(combined.shape)
    return combined


if __name__ == "__main__":
     print("In main, Call extract_features(audio,signal_rate) as parameters")