#!/usr/bin/env python

import numpy as np, os, sys
import joblib
from get_12ECG_features import get_12ECG_features

import torch
import torch.nn as nn
import neurokit as nk
from scipy import signal
from scipy.io import loadmat
from classifier.inceptiontime import InceptionTimeV2
from classifier.resnest import resnest

def run_12ECG_classifier(data,header_data,loaded_model):

    segments = data_preprocess(data, header_data)
    model = loaded_model['model']
    classes = loaded_model['classes']
    alphas = loaded_model['alphas']
    indices = loaded_model['indices']

    num_classes = model.num_classes
    alphas_all = np.zeros((num_classes, ))
    k = 0
    for i in range(num_classes):
        if indices[i]:
            alphas_all[i] = alphas[k]
            k += 1
        else:
            alphas_all[i] = 0.5

    current_scores = predict_proba(model, segments)
    print(current_scores.shape)
    current_score = np.mean(current_scores, axis=0)
    current_label = predict2(current_score, alphas_all)

    return current_label, current_score, classes

def load_12ECG_model(input_directory):
    # load the model from disk 
    f_model = 'model_best.pth'
    f_classes = 'classes.mat'
    f_alphas = 'best_alphas.mat'
    f_indices = 'indices.mat'

    filename_model = os.path.join(input_directory,f_model)
    filename_classes = os.path.join(input_directory,f_classes)
    filename_alphas = os.path.join(input_directory, f_alphas)
    filename_indices = os.path.join(input_directory, f_indices)

    # model = InceptionTimeV2(
    #     in_channels = 12,
    #     num_classes = 108,
    #     n_blocks = 2,
    #     n_filters = 32,
    #     kernel_sizes=[9, 19, 39],
    #     bottleneck_channels=32,
    #     input_bottleneck=True,
    #     use_residual=True,
    #     attention='CBAM_Channel')

    model = resnest(
        layers=[2, 2, 1, 3],
        bottleneck_width=64,
        stem_width=16,
        num_classes=108,
        kernel_size=7
        )

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        checkpoint = torch.load(filename_model)
    else:
        checkpoint = torch.load(filename_model, map_location='cpu')

    model.load_state_dict(checkpoint['state_dict'])

    classes = list(loadmat(filename_classes)['val'])

    indices = loadmat(filename_indices)['val']
    indices = indices.reshape([indices.shape[1], ])

    alphas = loadmat(filename_alphas)['val']
    alphas = alphas.reshape([alphas.shape[1], ])

    dict = {
        'model': model,
        'classes': classes,
        'alphas': alphas,
        'indices': indices
    }

    return dict

def data_preprocess(data, header_data):
    # get information from header_data
    tmp_hea = header_data[0].split(' ')
    ptID = tmp_hea[0]
    num_leads = int(tmp_hea[1])
    sample_Fs = int(tmp_hea[2])
    sample_len = int(tmp_hea[3])
    gain_lead = np.zeros(num_leads)

    for ii in range(num_leads):
        tmp_hea = header_data[ii+1].split(' ')
        gain_lead[ii] = int(tmp_hea[2].split('/')[0])

    # divide adc_gain
    for ii in range(num_leads):
        data[ii] /= gain_lead[ii]

    # resample to 300Hz
    resample_Fs = 300
    resample_len = int(sample_len * (resample_Fs / sample_Fs))
    resample_data = signal.resample(data, resample_len, axis=1, window=None)

    # cut resampled data via sliding windows
    segments = list()
    if resample_len < 3000:
        try:
            seg = ecg_filling(resample_data, sampling_rate=resample_Fs, length=resample_len)
        except:
            seg = ecg_filling2(resample_data, length=resample_len)
        segments.append(seg)
    elif resample_len == 3000:
        segments.append(resample_data)
    else:
        segments = slide_and_cut(resample_data, length=resample_len)

    # return list of segments
    return np.array(segments)

def ecg_filling(ecg, sampling_rate, length):
    ecg_II = ecg[1]
    processed_ecg = nk.ecg_process(ecg_II, sampling_rate)
    rpeaks = processed_ecg[1]['ECG_R_Peaks']
    ecg_filled = np.zeros((ecg.shape[0], length))
    sta = rpeaks[-1]
    ecg_filled[:, :sta] = ecg[:, :sta]
    seg = ecg[:, rpeaks[0]:rpeaks[-1]]
    len = seg.shape[1]
    while True:
        if (sta + len) >= length:
            ecg_filled[:, sta: length] = seg[:, : length - sta]
            break
        else:
            ecg_filled[:, sta: sta + len] = seg[:, :]
            sta = sta + len
    return ecg_filled

def ecg_filling2(ecg, length):
    len = ecg.shape[1]
    ecg_filled = np.zeros((ecg.shape[0], length))
    ecg_filled[:, :len] = ecg
    sta = len
    while length - sta > len:
        ecg_filled[:, sta : sta + len] = ecg
        sta += len
    ecg_filled[:, sta:length] = ecg[:, :length-sta]

    return ecg_filled

def slide_and_cut(ecg, length, window_size=3000, overlap=0.5):
    segments = list()
    offset = 0
    while offset + window_size <= length:
        segments.append(ecg[:, offset : offset+window_size])
        offset = offset + int(overlap * window_size)
    return segments

def predict_proba(model, input):
    model.eval()
    input = torch.from_numpy(input).type(torch.float32)
    scores = nn.Sigmoid()(model(input))
    return scores.detach().numpy()

def predict(score, alpha=0.5):
    label = np.zeros((score.shape[0], ))
    for i in range(score.shape[0]):
        if score[i] >= alpha:
            label[i] = 1
        else:
            label[i] = 0
    return label.astype(int)

def predict2(score, alphas):
    label = np.zeros((score.shape[0], ))
    for i in range(score.shape[0]):
        if score[i] >= alphas[i]:
            label[i] = 1
        else:
            label[i] = 0
    return label.astype(int)
