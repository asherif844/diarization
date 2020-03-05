from __future__ import print_function
from GreenwayHealth import dirGreenwaySpeakerDiarization, fileGreenwaySpeakerDiarization
import numpy as np
import sklearn.cluster
import scipy
import os
import json
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

# Set path separators indep of OS


def pjoin(*args, **kwargs):
    return join(*args, **kwargs).replace(sep, '/')

# Cal Greenway Health Diarization Function
# Local on my machine
# import os
# audio_folder = 'audio_output'
# file_name = 'audio_test_min_1.wav'
# output_folder_name = os.path.join(os.getcwd(), audio_folder)
# file_location = os.path.join(output_folder_name, file_name)

# _,output_json=fileGreenwaySpeakerDiarization( filename=pjoin(file_location),\
#     output_folder=pjoin(output_folder_name) )


def output_function(input_file, input_folder):
    _, output_json = fileGreenwaySpeakerDiarization(
        filename=pjoin(input_file), output_folder=pjoin(input_folder))
    return output_json


# def transcription():
#     total_input = '/Users/macmini/Dropbox/docker/anthonyDiarization/audio_output/audio_test_min_1.wav,/Users/macmini/Dropbox/docker/anthonyDiarization/audio_output/'
#     input_file = total_input.split(',')[0].replace("'", "").replace('"', '')
#     input_folder = total_input.split(',')[1].replace("'", "").replace('"', '')
#     input_file = str(input_file)
#     input_folder = str(input_folder)
#     run_function = output_function(input_file, input_folder)
#     return 'Transcription Successful'


# def transcription_json():
#     with open('sample-test-input.json') as f:
#       total_input = json.load(f)
#       input_file = total_input.get('file_name')
#       input_folder = total_input.get('file_location')
#       run_function = output_function(input_file, input_folder)
#       return 'transcription complete'

# transcription_json()
# transcription()
