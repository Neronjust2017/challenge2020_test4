import os
import json
import numpy as np
from numpy import inf
from scipy import signal
from scipy.io import loadmat, savemat
import torch
from torch.utils.data import Dataset
import logging
import neurokit as nk
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import matplotlib.pyplot as plt

# Utilty functions
# Data loading and processing

def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

# Find Challenge files.
def load_label_files(label_directory):
    label_files = list()
    for f in sorted(os.listdir(label_directory)):
        F = os.path.join(label_directory, f) # Full path for label file
        if os.path.isfile(F) and F.lower().endswith('.hea') and not f.lower().startswith('.'):
            # root, ext = os.path.splitext(f)
            label_files.append(F)
    if label_files:
        return label_files
    else:
        raise IOError('No label or output files found.')

# Load labels from header/label files.
def load_labels(label_files, normal_class, equivalent_classes_collection):
    # The labels_onehot should have the following form:
    #
    # Dx: label_1, label_2, label_3
    #
    num_recordings = len(label_files)

    # Load diagnoses.
    tmp_labels = list()
    for i in range(num_recordings):
        with open(label_files[i], 'r') as f:
            for l in f:
                if l.startswith('#Dx'):
                    dxs = set(arr.strip() for arr in l.split(': ')[1].split(','))
                    tmp_labels.append(dxs)

    # Identify classes.
    classes = set.union(*map(set, tmp_labels))
    if normal_class not in classes:
        classes.add(normal_class)
        print('- The normal class {} is not one of the label classes, so it has been automatically added, but please check that you chose the correct normal class.'.format(normal_class))
    classes = sorted(classes)
    num_classes = len(classes)

    # Use one-hot encoding for labels.
    labels_onehot = np.zeros((num_recordings, num_classes), dtype=np.bool)
    for i in range(num_recordings):
        dxs = tmp_labels[i]
        for dx in dxs:
            j = classes.index(dx)
            labels_onehot[i, j] = 1

    # For each set of equivalent class, use only one class as the representative class for the set and discard the other classes in the set.
    # The label for the representative class is positive if any of the labels_onehot in the set is positive.
    remove_classes = list()
    remove_indices = list()
    for equivalent_classes in equivalent_classes_collection:
        equivalent_classes = [x for x in equivalent_classes if x in classes]
        if len(equivalent_classes)>1:
            representative_class = equivalent_classes[0]
            other_classes = equivalent_classes[1:]
            equivalent_indices = [classes.index(x) for x in equivalent_classes]
            representative_index = equivalent_indices[0]
            other_indices = equivalent_indices[1:]

            labels_onehot[:, representative_index] = np.any(labels_onehot[:, equivalent_indices], axis=1)
            remove_classes += other_classes
            remove_indices += other_indices

    for x in remove_classes:
        classes.remove(x)
    labels_onehot = np.delete(labels_onehot, remove_indices, axis=1)

    # If the labels_onehot are negative for all classes, then change the label for the normal class to positive.
    normal_index = classes.index(normal_class)
    for i in range(num_recordings):
        num_positive_classes = np.sum(labels_onehot[i, :])
        if num_positive_classes==0:
            labels_onehot[i, normal_index] = 1

    labels = list()
    for i in range(num_recordings):
        class_list = []
        for j in range(len(classes)):
            if labels_onehot[i][j] == True:
                class_list.append(classes[j])
        class_set = set()
        class_set.update(class_list)
        labels.append(class_set)

    return classes, labels_onehot, labels

# Load challenge data.
def load_challenge_data(label_file, data_dir):
    file = os.path.basename(label_file)
    name, ext = os.path.splitext(file)
    with open(label_file, 'r') as f:
        header = f.readlines()
    mat_file = file.replace('.hea', '.mat')
    x = loadmat(os.path.join(data_dir, mat_file))
    recording = np.asarray(x['val'], dtype=np.float64)
    return recording, header, name

# Load weights.
def load_weights(weight_file, classes):
    # Load the weight matrix.
    rows, cols, values = load_table(weight_file)
    assert(rows == cols)
    num_rows = len(rows)

    # Assign the entries of the weight matrix with rows and columns corresponding to the classes.
    num_classes = len(classes)
    weights = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i, a in enumerate(rows):
        if a in classes:
            k = classes.index(a)
            for j, b in enumerate(rows):
                if b in classes:
                    l = classes.index(b)
                    weights[k, l] = values[i, j]

    return weights

# Load_table
def load_table(table_file):
    # The table should have the following form:
    #
    # ,    a,   b,   c
    # a, 1.2, 2.3, 3.4
    # b, 4.5, 5.6, 6.7
    # c, 7.8, 8.9, 9.0
    #
    table = list()
    print(os.getcwd())
    with open(table_file, 'r') as f:
        for i, l in enumerate(f):
            arrs = [arr.strip() for arr in l.split(',')]
            table.append(arrs)

    # Define the numbers of rows and columns and check for errors.
    num_rows = len(table)-1
    if num_rows<1:
        raise Exception('The table {} is empty.'.format(table_file))

    num_cols = set(len(table[i])-1 for i in range(num_rows))
    if len(num_cols)!=1:
        raise Exception('The table {} has rows with different lengths.'.format(table_file))
    num_cols = min(num_cols)
    if num_cols<1:
        raise Exception('The table {} is empty.'.format(table_file))

    # Find the row and column labels.
    rows = [table[0][j+1] for j in range(num_rows)]
    cols = [table[i+1][0] for i in range(num_cols)]

    # Find the entries of the table.
    values = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            value = table[i+1][j+1]
            if is_number(value):
                values[i, j] = float(value)
            else:
                values[i, j] = float('nan')

    return rows, cols, values

# Divide ADC_gain and resample
def resample(data, header_data, resample_Fs = 300):
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

    resample_len = int(sample_len * (resample_Fs / sample_Fs))
    resample_data = signal.resample(data, resample_len, axis=1, window=None)

    return resample_data

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

def slide_and_cut(data, n_segment=1, window_size=3000, sampling_rate=300):
    length = data.shape[1]
    print("length:", length)
    if length < window_size:
        segments = []
        try:
            ecg_filled = ecg_filling(data, sampling_rate, window_size)
        except:
            ecg_filled = ecg_filling2(data, window_size)
        segments.append(ecg_filled)
        segments = np.array(segments)
    else:
        offset = (length - window_size * n_segment) / (n_segment + 1)
        if offset >= 0:
            start = 0 + offset
        else:
            offset = (length - window_size * n_segment) / (n_segment - 1)
            start = 0
        segments = []
        for j in range(n_segment):
            ind = int(start + j * (window_size + offset))
            segment = data[:, ind:ind + window_size]
            segments.append(segment)
        segments = np.array(segments)

    return segments

# split into training and validation
def stratification(label_dir):
    print('Stratification...')

    # Define the weights, the SNOMED CT code for the normal class, and equivalent SNOMED CT codes.
    normal_class = '426783006'
    equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

    # Find the label files.
    label_files = load_label_files(label_dir)

    # Load the labels and classes.
    label_classes, labels_onehot, labels = load_labels(label_files, normal_class, equivalent_classes)

    temp = [[] for _ in range(len(labels_onehot))]
    indexes, values = np.where(np.array(labels_onehot).astype(int) == 1)
    for k, v in zip(indexes, values):
       temp[k].append(v)
    labels_int = temp

    X = np.zeros(len(labels_onehot))
    y = labels_onehot

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
    for train_index, val_index in msss.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        print('Saving split index...')
        datasets_distribution(labels_int, [train_index, val_index])
        savemat('model_training/split.mat', {'train_index': train_index, 'val_index': val_index})

    print('Stratification done.')

def datasets_distribution(labels_int, indexs):
   num_of_bins = 108
   fig, axs = plt.subplots(len(indexs), 1, sharey=True, figsize=(50, 50))
   for i in range(len(indexs)):
      subdataset = list()
      for j in indexs[i]:
         for k in labels_int[j]:
            subdataset.append(k)
      subdataset = np.array(subdataset)
      axs[i].hist(subdataset, bins=num_of_bins)
   plt.show()

# Training
def make_dirs(base_dir):

    checkpoint_dir = base_dir + '/checkpoints'
    log_dir = base_dir + '/log'
    tb_dir = base_dir + '/tb_log'
    result_dir = base_dir + '/results'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)

    return result_dir, log_dir, checkpoint_dir, tb_dir

def init_obj(hype_space, name, module, *args, **kwargs):
    """
    Finds a function handle with the name given as 'type' in config, and returns the
    instance initialized with corresponding arguments given.
    """
    module_name = hype_space[name]['type']
    module_args = dict(hype_space[name]['args'])
    assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    return getattr(module, module_name)(*args, **module_args)

def to_np(tensor, device):
    if device.type == 'cuda':
        return tensor.cpu().detach().numpy()
    else:
        return tensor.detach().numpy()

def get_mnt_mode(trainer):
    monitor = trainer.get('monitor', 'off')
    if monitor == 'off':
        mnt_mode = 'off'
        mnt_best = 0
        early_stop = 0
        mnt_metric_name = None
    else:
        mnt_mode, mnt_metric_name = monitor.split()
        assert mnt_mode in ['min', 'max']
        mnt_best = inf if mnt_mode == 'min' else -inf
        early_stop = trainer.get('early_stop', inf)

    return mnt_metric_name, mnt_mode, mnt_best, early_stop

def load_checkpoint(checkpoint_dir, use_cuda):
    best_model = checkpoint_dir + '/model_best.pth'

    if use_cuda:
        checkpoint = torch.load(best_model)
    else:
        checkpoint = torch.load(best_model, map_location='cpu')

    return checkpoint

def save_checkpoint(model, epoch, mnt_best, checkpoint_dir, save_best=False):
    arch = type(model).__name__
    state = {
        'arch': arch,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'monitor_best': mnt_best,
    }

    save_path = checkpoint_dir + '/model_' + str(epoch) + '.pth'
    torch.save(state, save_path)

    if save_best:
        best_path = checkpoint_dir + '/model_best.pth'
        torch.save(state, best_path)
        print("Saving current best: model_best.pth ...")

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def progress(data_loader, batch_idx):
    base = '[{}/{} ({:.0f}%)]'
    if hasattr(data_loader, 'n_samples'):
        current = batch_idx * data_loader.batch_size
        total = data_loader.n_samples
    else:
        current = batch_idx
        total = len(data_loader)
    return base.format(current, total, 100.0 * current / total)

# Customed TensorDataset
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, *tensors, transform=None, p=0.5):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.p = p

    def __getitem__(self, index):
        x = self.tensors[0][index]
        torch.randn(1)

        if self.transform:
            if torch.rand(1) >= self.p:
                x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

# Model calibration
# ChallengeMetric class for model calibration
class ChallengeMetric_calibration():

    def __init__(self, input_directory, alphas):

        # challengeMetric initialization
        weights_file = 'model_training/weights.csv'
        normal_class = '426783006'
        equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

        # Find the label files.
        label_files = load_label_files(input_directory)

        # Load the labels and classes.
        classes, labels_onehot, labels = load_labels(label_files, normal_class, equivalent_classes)

        num_files = len(label_files)

        # Load the weights for the Challenge metric.
        weights = load_weights(weights_file, classes)

        # Only consider classes that are scored with the Challenge metric.
        indices = np.any(weights, axis=0)  # Find indices of classes in weight matrix.
        classes = [x for i, x in enumerate(classes) if indices[i]]
        weights = weights[np.ix_(indices, indices)]

        self.weights = weights
        self.indices = indices
        self.classes = classes
        self.normal_class = normal_class

        self.alphas = alphas

    # Compute recording-wise accuracy.
    def accuracy(self, outputs, labels):
        outputs = outputs[:, self.indices]
        labels = labels[:, self.indices]
        outputs = self.get_pred(outputs)

        num_recordings, num_classes = np.shape(labels)

        num_correct_recordings = 0
        for i in range(num_recordings):
            if np.all(labels[i, :] == outputs[i, :]):
                num_correct_recordings += 1

        return float(num_correct_recordings) / float(num_recordings)

    # Compute confusion matrices.
    def confusion_matrices(self, outputs, labels, normalize=False):
        # Compute a binary confusion matrix for each class k:
        #
        #     [TN_k FN_k]
        #     [FP_k TP_k]
        #
        # If the normalize variable is set to true, then normalize the contributions
        # to the confusion matrix by the number of labels per recording.
        num_recordings, num_classes = np.shape(labels)

        if not normalize:
            A = np.zeros((num_classes, 2, 2))
            for i in range(num_recordings):
                for j in range(num_classes):
                    if labels[i, j] == 1 and outputs[i, j] == 1:  # TP
                        A[j, 1, 1] += 1
                    elif labels[i, j] == 0 and outputs[i, j] == 1:  # FP
                        A[j, 1, 0] += 1
                    elif labels[i, j] == 1 and outputs[i, j] == 0:  # FN
                        A[j, 0, 1] += 1
                    elif labels[i, j] == 0 and outputs[i, j] == 0:  # TN
                        A[j, 0, 0] += 1
                    else:  # This condition should not happen.
                        raise ValueError('Error in computing the confusion matrix.')
        else:
            A = np.zeros((num_classes, 2, 2))
            for i in range(num_recordings):
                normalization = float(max(np.sum(labels[i, :]), 1))
                for j in range(num_classes):
                    if labels[i, j] == 1 and outputs[i, j] == 1:  # TP
                        A[j, 1, 1] += 1.0 / normalization
                    elif labels[i, j] == 0 and outputs[i, j] == 1:  # FP
                        A[j, 1, 0] += 1.0 / normalization
                    elif labels[i, j] == 1 and outputs[i, j] == 0:  # FN
                        A[j, 0, 1] += 1.0 / normalization
                    elif labels[i, j] == 0 and outputs[i, j] == 0:  # TN
                        A[j, 0, 0] += 1.0 / normalization
                    else:  # This condition should not happen.
                        raise ValueError('Error in computing the confusion matrix.')

        return A

    # Compute macro F-measure.
    def f_measure(self, outputs, labels):
        outputs = outputs[:, self.indices]
        labels = labels[:, self.indices]
        outputs = self.get_pred(outputs)
        num_recordings, num_classes = np.shape(labels)

        A = self.confusion_matrices(outputs, labels)

        f_measure = np.zeros(num_classes)
        for k in range(num_classes):
            tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
            if 2 * tp + fp + fn:
                f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
            else:
                f_measure[k] = float('nan')

        macro_f_measure = np.nanmean(f_measure)

        return macro_f_measure

    def beta_measures(self, outputs, labels, beta=2):
        outputs = outputs[:, self.indices]
        labels = labels[:, self.indices]
        outputs = self.get_pred(outputs)
        num_recordings, num_classes = np.shape(labels)

        A = self.confusion_matrices(outputs, labels, normalize=True)

        f_beta_measure = np.zeros(num_classes)
        g_beta_measure = np.zeros(num_classes)
        for k in range(num_classes):
            tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
            if (1 + beta ** 2) * tp + fp + beta ** 2 * fn:
                f_beta_measure[k] = float((1 + beta ** 2) * tp) / float((1 + beta ** 2) * tp + fp + beta ** 2 * fn)
            else:
                f_beta_measure[k] = float('nan')
            if tp + fp + beta * fn:
                g_beta_measure[k] = float(tp) / float(tp + fp + beta * fn)
            else:
                g_beta_measure[k] = float('nan')

        macro_f_beta_measure = np.nanmean(f_beta_measure)
        macro_g_beta_measure = np.nanmean(g_beta_measure)

        return macro_f_beta_measure, macro_g_beta_measure

    # Compute modified confusion matrix for multi-class, multi-label tasks.
    def modified_confusion_matrix(self, outputs, labels):
        # Compute a binary multi-class, multi-label confusion matrix, where the rows
        # are the labels and the columns are the outputs.
        num_recordings, num_classes = np.shape(labels)

        A = np.zeros((num_classes, num_classes))

        # Iterate over all of the recordings.
        for i in range(num_recordings):
            # Calculate the number of positive labels and/or outputs.
            normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
            # Iterate over all of the classes.
            for j in range(num_classes):
                # Assign full and/or partial credit for each positive class.
                if labels[i, j]:
                    for k in range(num_classes):
                        if outputs[i, k]:
                            A[j, k] += 1.0 / normalization

        return A

    # Compute the evaluation metric for the Challenge.
    def challenge_metric(self, outputs, labels):
        outputs = outputs[:, self.indices]
        labels = labels[:, self.indices]
        outputs = self.get_pred(outputs)

        num_recordings, num_classes = np.shape(labels)
        normal_index = self.classes.index(self.normal_class)

        # Compute the observed score.
        A = self.modified_confusion_matrix(outputs, labels)
        observed_score = np.nansum(self.weights * A)

        # Compute the score for the model that always chooses the correct label(s).
        correct_outputs = labels
        A = self.modified_confusion_matrix(labels, correct_outputs)
        correct_score = np.nansum(self.weights * A)

        # Compute the score for the model that always chooses the normal class.
        inactive_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
        inactive_outputs[:, normal_index] = 1
        A = self.modified_confusion_matrix(labels, inactive_outputs)
        inactive_score = np.nansum(self.weights * A)

        if correct_score != inactive_score:
            normalized_score = float(observed_score - inactive_score) / float(correct_score - inactive_score)
        else:
            normalized_score = float('nan')

        return normalized_score

    def get_pred(self, output):
        num_recordings, num_classes = output.shape
        labels = np.zeros((num_recordings, num_classes))
        for i in range(num_recordings):
            for j in range(num_classes):
                if output[i, j] >= self.alphas[j]:
                    labels[i, j] = 1
                else:
                    labels[i, j] = 0
        return labels

def get_metrics(outputs, targets, challenge_metrics):
    accuracy = challenge_metrics.accuracy(outputs, targets)
    macro_f_measure = challenge_metrics.f_measure(outputs, targets)
    macro_f_beta_measure, macro_g_beta_measure = challenge_metrics.beta_measures(outputs, targets)
    challenge_metric = challenge_metrics.challenge_metric(outputs, targets)
    return accuracy, macro_f_measure, macro_f_beta_measure, macro_g_beta_measure, challenge_metric