from __future__ import print_function
import numpy as np
import sklearn.cluster
import scipy
import os,json
from pyAudioAnalysis import ShortTermFeatures as sF
from pyAudioAnalysis import MidTermFeatures as aF
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioBasicIO
from scipy.spatial import distance
from pyAudioAnalysis import audioSegmentation as aS
import matplotlib.pyplot as plt
import sklearn.discriminant_analysis
import csv
import os.path
import sklearn
import sklearn.cluster
import hmmlearn.hmm
import pickle as cPickle
import glob
from pydub import AudioSegment
from pydub.utils import make_chunks
from datetime import datetime
import pprint
import time
import azure.cognitiveservices.speech as speechsdk
from os.path import sep, join

""" General utility functions """

from pyAudioAnalysis.audioSegmentation import (smoothMovingAvg,
                                               selfSimilarityMatrix,
                                               flags2segs,
                                               segs2flags,
                                               computePreRec,
                                               readSegmentGT,
                                               plotSegmentationResults,
                                               evaluateSpeakerDiarization,
                                               trainHMM_computeStatistics,
                                               trainHMM_fromFile,
                                               trainHMM_fromDir,
                                               hmmSegmentation,
                                               mtFileClassification,
                                               evaluateSegmentationClassificationDir,
                                               silenceRemoval,
                                               speakerDiarizationEvaluateScript,
                                               musicThumbnailing
                                              )

"""Import Greenway Diarization Functions """
from GreenwayHealth import dirGreenwaySpeakerDiarization, fileGreenwaySpeakerDiarization

#Set path separators indep of OS
def pjoin(*args, **kwargs):
  return join(*args, **kwargs).replace(sep, '/')

#Cal Greenway Health Diarization Function
#Local on my machine

file_location = "/Users/macmini/Desktop/scottish.wav"
output_folder = "/Users/macmini/Desktop/"

_,output_json=fileGreenwaySpeakerDiarization( filename=pjoin("C:\\Users\\anfrankl\\Desktop\\volume\\audio_test_min_1.wav"),\
    output_folder=pjoin("C:\\Users\\anfrankl\\Desktop\\volume\\") )