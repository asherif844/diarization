from __future__ import print_function
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
import matplotlib
matplotlib.use('PS')

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
from os.path import sep, join


def pjoin(*args, **kwargs):
    return join(*args, **kwargs).replace(sep, '/')


def dirGreenwaySpeakerDiarization(audio_folder, output_folder="./pyAudioAnalysis/data/Greenway/"):
   # Create files list, exclude "snippet" files
    files_list = []
    for files in glob.glob(pjoin(audio_folder)+"*.wav"):
        if ("snippet" not in files) & ("_min_" in files):
            files_list.append(pjoin(files))
            print(files)
        else:
            pass

    # Call Greenway Diarization Function
    for wav_file in files_list:
        fileGreenwaySpeakerDiarization(filename=pjoin(
            wav_file), output_folder=output_folder)


def fileGreenwaySpeakerDiarization(filename, output_folder, speech_key="52fe944f29784ae288482e5eb3092e2a", service_region="eastus2",
                                   n_speakers=2, mt_size=2.0, mt_step=0.2,
                                   st_win=0.05, lda_dim=35, plot_res=False, save_plot=True):
    """
    ARGUMENTS:
        - filename:        the name of the WAV file to be analyzed
                            the filename should have a suffix of the form: ..._min_3
                            this informs the service that audio file corresponds to the 3rd minute of the dialogue
        - output_folder    the folder location for saving the audio snippets generated from diarization                           
        - speech_key       mid-term window size            
        - service_region       the number of speakers (clusters) in
                           the recording (<=0 for unknown)
        - n_speakers       the number of speakers (clusters) in
                           the recording (<=0 for unknown)
        - mt_size (opt)    mid-term window size
        - mt_step (opt)    mid-term window step
        - st_win  (opt)    short-term window size
        - lda_dim (opt     LDA dimension (0 for no LDA)
        - plot_res         (opt)   0 for not plotting the results 1 for plotting
        - save_plot        (opt)   1|True for saving plot in output folder
    """
    '''
    OUTPUTS:
        - cls:             this is a vector with speaker ids in chronological sequence of speaker dialogue.
        - output:          a list of python dictionaries containing dialogue sequence information.
                            - dialogue_id
                            - sequence_id
                            - start_time
                            - end_time
                            - text
    '''

    filename_only = filename if "/" not in filename else filename.split("/")[-1]
    nameoffile = filename_only.split("_min_")[0]
    timeoffile = filename_only.split("_min_")[1]

    [fs, x] = audioBasicIO.read_audio_file(filename)
    x = audioBasicIO.stereo_to_mono(x)
    duration = len(x) / fs

    [classifier_1, MEAN1, STD1, classNames1, mtWin1, mtStep1, stWin1, stStep1, computeBEAT1] = aT.load_model_knn(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "pyAudioAnalysis/data/models", "knn_speaker_10"))
    [classifier_2, MEAN2, STD2, classNames2, mtWin2, mtStep2, stWin2, stStep2, computeBEAT2] = aT.load_model_knn(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "pyAudioAnalysis/data/models", "knn_speaker_male_female"))

    [mt_feats, st_feats, _] = aF.mid_feature_extraction(x, fs, mt_size * fs,
                                                        mt_step * fs,
                                                        round(fs * st_win),
                                                        round(fs*st_win * 0.5))

    MidTermFeatures2 = np.zeros((mt_feats.shape[0] + len(classNames1) +
                                 len(classNames2), mt_feats.shape[1]))

    for i in range(mt_feats.shape[1]):
        cur_f1 = (mt_feats[:, i] - MEAN1) / STD1
        cur_f2 = (mt_feats[:, i] - MEAN2) / STD2
        [res, P1] = aT.classifierWrapper(classifier_1, "knn", cur_f1)
        [res, P2] = aT.classifierWrapper(classifier_2, "knn", cur_f2)
        MidTermFeatures2[0:mt_feats.shape[0], i] = mt_feats[:, i]
        MidTermFeatures2[mt_feats.shape[0]:mt_feats.shape[0] +
                         len(classNames1), i] = P1 + 0.0001
        MidTermFeatures2[mt_feats.shape[0] +
                         len(classNames1)::, i] = P2 + 0.0001

    mt_feats = MidTermFeatures2    # TODO
    iFeaturesSelect = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 41,
                       42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]

    mt_feats = mt_feats[iFeaturesSelect, :]

    (mt_feats_norm, MEAN, STD) = aT.normalizeFeatures([mt_feats.T])
    mt_feats_norm = mt_feats_norm[0].T
    n_wins = mt_feats.shape[1]

    # remove outliers:
    dist_all = np.sum(distance.squareform(distance.pdist(mt_feats_norm.T)),
                      axis=0)
    m_dist_all = np.mean(dist_all)
    i_non_outliers = np.nonzero(dist_all < 1.2 * m_dist_all)[0]

    # TODO: Combine energy threshold for outlier removal:
    #EnergyMin = np.min(mt_feats[1,:])
    #EnergyMean = np.mean(mt_feats[1,:])
    #Thres = (1.5*EnergyMin + 0.5*EnergyMean) / 2.0
    #i_non_outliers = np.nonzero(mt_feats[1,:] > Thres)[0]
    # print i_non_outliers

    perOutLier = (100.0 * (n_wins - i_non_outliers.shape[0])) / n_wins
    mt_feats_norm_or = mt_feats_norm
    mt_feats_norm = mt_feats_norm[:, i_non_outliers]

    # LDA dimensionality reduction:
    if lda_dim > 0:
        # [mt_feats_to_red, _, _] = aF.mtFeatureExtraction(x, fs, mt_size * fs,
        # st_win * fs, round(fs*st_win), round(fs*st_win));
        # extract mid-term features with minimum step:
        mt_win_ratio = int(round(mt_size / st_win))
        mt_step_ratio = int(round(st_win / st_win))
        mt_feats_to_red = []
        num_of_features = len(st_feats)
        num_of_stats = 2
        # for i in range(num_of_stats * num_of_features + 1):
        for i in range(num_of_stats * num_of_features):
            mt_feats_to_red.append([])

        # for each of the short-term features:
        for i in range(num_of_features):
            curPos = 0
            N = len(st_feats[i])
            while (curPos < N):
                N1 = curPos
                N2 = curPos + mt_win_ratio
                if N2 > N:
                    N2 = N
                curStFeatures = st_feats[i][N1:N2]
                mt_feats_to_red[i].append(np.mean(curStFeatures))
                mt_feats_to_red[i +
                                num_of_features].append(np.std(curStFeatures))
                curPos += mt_step_ratio
        mt_feats_to_red = np.array(mt_feats_to_red)
        mt_feats_to_red_2 = np.zeros((mt_feats_to_red.shape[0] +
                                      len(classNames1) + len(classNames2),
                                      mt_feats_to_red.shape[1]))
        for i in range(mt_feats_to_red.shape[1]):
            cur_f1 = (mt_feats_to_red[:, i] - MEAN1) / STD1
            cur_f2 = (mt_feats_to_red[:, i] - MEAN2) / STD2
            [res, P1] = aT.classifierWrapper(classifier_1, "knn", cur_f1)
            [res, P2] = aT.classifierWrapper(classifier_2, "knn", cur_f2)
            mt_feats_to_red_2[0:mt_feats_to_red.shape[0],
                              i] = mt_feats_to_red[:, i]
            mt_feats_to_red_2[mt_feats_to_red.shape[0]                              :mt_feats_to_red.shape[0] + len(classNames1), i] = P1 + 0.0001
            mt_feats_to_red_2[mt_feats_to_red.shape[0] +
                              len(classNames1)::, i] = P2 + 0.0001
        mt_feats_to_red = mt_feats_to_red_2
        mt_feats_to_red = mt_feats_to_red[iFeaturesSelect, :]
        #mt_feats_to_red += np.random.rand(mt_feats_to_red.shape[0], mt_feats_to_red.shape[1]) * 0.0000010
        (mt_feats_to_red, MEAN, STD) = aT.normalizeFeatures(
            [mt_feats_to_red.T])
        mt_feats_to_red = mt_feats_to_red[0].T
        #dist_all = np.sum(distance.squareform(distance.pdist(mt_feats_to_red.T)), axis=0)
        #m_dist_all = np.mean(dist_all)
        #iNonOutLiers2 = np.nonzero(dist_all < 3.0*m_dist_all)[0]
        #mt_feats_to_red = mt_feats_to_red[:, iNonOutLiers2]
        Labels = np.zeros((mt_feats_to_red.shape[1], ))
        LDAstep = 1.0
        LDAstepRatio = LDAstep / st_win
        # print LDAstep, LDAstepRatio
        for i in range(Labels.shape[0]):
            Labels[i] = int(i*st_win/LDAstepRatio)
        clf = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(
            n_components=lda_dim)
        clf.fit(mt_feats_to_red.T, Labels)
        mt_feats_norm = (clf.transform(mt_feats_norm.T)).T

    if n_speakers <= 0:
        s_range = range(2, 10)
    else:
        s_range = [n_speakers]
    clsAll = []
    sil_all = []
    centersAll = []

    for iSpeakers in s_range:
        k_means = sklearn.cluster.KMeans(n_clusters=iSpeakers)
        k_means.fit(mt_feats_norm.T)
        cls = k_means.labels_
        means = k_means.cluster_centers_

        # Y = distance.squareform(distance.pdist(mt_feats_norm.T))
        clsAll.append(cls)
        centersAll.append(means)
        sil_1 = []
        sil_2 = []
        for c in range(iSpeakers):
            # for each speaker (i.e. for each extracted cluster)
            clust_per_cent = np.nonzero(cls == c)[0].shape[0] / \
                float(len(cls))
            if clust_per_cent < 0.020:
                sil_1.append(0.0)
                sil_2.append(0.0)
            else:
                # get subset of feature vectors
                mt_feats_norm_temp = mt_feats_norm[:, cls == c]
                # compute average distance between samples
                # that belong to the cluster (a values)
                Yt = distance.pdist(mt_feats_norm_temp.T)
                sil_1.append(np.mean(Yt)*clust_per_cent)
                silBs = []
                for c2 in range(iSpeakers):
                    # compute distances from samples of other clusters
                    if c2 != c:
                        clust_per_cent_2 = np.nonzero(cls == c2)[0].shape[0] /\
                            float(len(cls))
                        MidTermFeaturesNormTemp2 = mt_feats_norm[:, cls == c2]
                        Yt = distance.cdist(mt_feats_norm_temp.T,
                                            MidTermFeaturesNormTemp2.T)
                        silBs.append(np.mean(Yt)*(clust_per_cent
                                                  + clust_per_cent_2)/2.0)
                silBs = np.array(silBs)
                # ... and keep the minimum value (i.e.
                # the distance from the "nearest" cluster)
                sil_2.append(min(silBs))
        sil_1 = np.array(sil_1)
        sil_2 = np.array(sil_2)
        sil = []
        for c in range(iSpeakers):
            # for each cluster (speaker) compute silhouette
            sil.append((sil_2[c] - sil_1[c]) / (max(sil_2[c],
                                                    sil_1[c]) + 0.00001))
        # keep the AVERAGE SILLOUETTE
        sil_all.append(np.mean(sil))

    imax = np.argmax(sil_all)
    # optimal number of clusters
    nSpeakersFinal = s_range[imax]

    # generate the final set of cluster labels
    # (important: need to retrieve the outlier windows:
    # this is achieved by giving them the value of their
    # nearest non-outlier window)
    cls = np.zeros((n_wins,))
    for i in range(n_wins):
        j = np.argmin(np.abs(i-i_non_outliers))
        cls[i] = clsAll[imax][j]

    # Post-process method 1: hmm smoothing
    for i in range(1):
        # hmm training
        start_prob, transmat, means, cov = \
            trainHMM_computeStatistics(mt_feats_norm_or, cls)
        hmm = hmmlearn.hmm.GaussianHMM(start_prob.shape[0], "diag")
        hmm.startprob_ = start_prob
        hmm.transmat_ = transmat
        hmm.means_ = means
        hmm.covars_ = cov
        cls = hmm.predict(mt_feats_norm_or.T)

    # Post-process method 2: median filtering:
    cls = scipy.signal.medfilt(cls, 13)
    cls = scipy.signal.medfilt(cls, 11)

    sil = sil_all[imax]
    class_names = ["speaker{0:d}".format(c) for c in range(nSpeakersFinal)]

    # load ground-truth if available
    gt_file = filename.replace('.wav', '.segments')
    # if groundturh exists
    if os.path.isfile(gt_file):
        [seg_start, seg_end, seg_labs] = readSegmentGT(gt_file)
        flags_gt, class_names_gt = segs2flags(
            seg_start, seg_end, seg_labs, mt_step)

    if plot_res:
        fig = plt.figure()
        if n_speakers > 0:
            ax1 = fig.add_subplot(111)
        else:
            ax1 = fig.add_subplot(211)
        ax1.set_yticks(np.array(range(len(class_names))))
        ax1.axis((0, duration, -1, len(class_names)))
        ax1.set_yticklabels(class_names)
        ax1.plot(np.array(range(len(cls)))*mt_step+mt_step/2.0, cls)

    if os.path.isfile(gt_file):
        if plot_res:
            ax1.plot(np.array(range(len(flags_gt))) *
                     mt_step + mt_step / 2.0, flags_gt, 'r')
        purity_cluster_m, purity_speaker_m = \
            evaluateSpeakerDiarization(cls, flags_gt)
        print("{0:.1f}\t{1:.1f}".format(100 * purity_cluster_m,
                                        100 * purity_speaker_m))
        if plot_res:
            plt.title("Cluster purity: {0:.1f}% - "
                      "Speaker purity: {1:.1f}%".format(100 * purity_cluster_m,
                                                        100 * purity_speaker_m))
    if plot_res:
        plt.xlabel("time (seconds)")
        # print s_range, sil_all
        if n_speakers <= 0:
            plt.subplot(212)
            plt.plot(s_range, sil_all)
            plt.xlabel("number of clusters")
            plt.ylabel("average clustering's sillouette")
        if save_plot:
            plt.savefig(
                f"{output_folder}{filename_only}".replace(".wav", ".png"))
        else:
            pass
        plt.show()

    # Create Time Vector
    time_vec = np.array(range(len(cls)))*mt_step+mt_step/2.0

    # Find Change Points
    speaker_change_index = np.where(np.roll(cls, 1) != cls)[0]

    # Create List of dialogue convos
    output_list = []
    temp = {}
    for ind, sc in enumerate(speaker_change_index):
        temp['dialogue_id'] = str(datetime.now()).strip()
        temp['sequence_id'] = str(ind)
        temp['speaker'] = list(cls)[sc]
        temp['start_time'] = time_vec[sc]
        temp['end_time'] = time_vec[speaker_change_index[ind+1] -
                                    1] if ind+1 < len(speaker_change_index) else time_vec[-1]
        temp["text"] = ""
        output_list.append(temp)
        temp = {}

    def snip_transcribe(output_list, filename, output_folder=output_folder,
                        speech_key=speech_key, service_region=service_region):
        speech_config = speechsdk.SpeechConfig(
            subscription=speech_key, region=service_region)
        speech_config.enable_dictation

        def recognized_cb(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                # Do something with the recognized text
                output_list[ind]['text'] = output_list[ind]['text'] + \
                    str(evt.result.text)
                print(evt.result.text)

        for ind, diag in enumerate(output_list):
            t1 = diag['start_time']
            t2 = diag['end_time']
            newAudio = AudioSegment.from_wav(filename)
            chunk = newAudio[t1*1000:t2*1000]
            filename_out = output_folder + f"snippet_{diag['sequence_id']}.wav"
            # Exports to a wav file in the current path.
            chunk.export(filename_out, format="wav")
            done = False

            def stop_cb(evt):
                """callback that signals to stop continuous recognition upon receiving an event `evt`"""
                print('CLOSING on {}'.format(evt))
                nonlocal done
                done = True

            audio_input = speechsdk.AudioConfig(filename=filename_out)
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config, audio_config=audio_input)
            output_list[ind]['snippet_path'] = filename_out

            speech_recognizer.recognized.connect(recognized_cb)

            speech_recognizer.session_stopped.connect(stop_cb)
            speech_recognizer.canceled.connect(stop_cb)

            # Start continuous speech recognition
            speech_recognizer.start_continuous_recognition()
            while not done:
                time.sleep(.5)

            speech_recognizer.stop_continuous_recognition()

        return output_list

    output = snip_transcribe(output_list, filename,
                             output_folder=output_folder)
    output_json = {filename_only: output}

    with open(f"{output_folder}{nameoffile}_{timeoffile}.txt", "w") as outfile:
        json.dump(output_json, outfile)

    return cls, output_json
