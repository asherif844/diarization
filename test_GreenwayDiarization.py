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
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('PS')

# try:
#    import matplotlib.pyplot as plt
# except RuntimeError as e:
#    if 'Python is not installed as a framework.' in e.message:
#      warnings.warn(".. some warning about disabled plotting...")

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

file_name = 'audio_test_min_1.wav'
folder_location_output = "audio_output"

file_location = os.path.join(folder_location_output, file_name)


def final_output(file_location,folder_location_output):
  # _,output_json=fileGreenwaySpeakerDiarization( filename=pjoin(file_location),output_folder=pjoin(folder_location_output) )
  _,output_json=fileGreenwaySpeakerDiarization( filename=file_location,output_folder=folder_location_output )
  return output_json


# print(final_output(file_location,folder_location_output))
print(f'filename = {file_name}')
print(f'folder_location = {folder_location_output}')
print(f'file_location = {file_location}')