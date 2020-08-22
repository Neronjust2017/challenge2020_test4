#!/usr/bin/env python

import json
import time
import torch.nn as nn
from hyperopt import hp, tpe, fmin, Trials
from model_training.training import *
from tensorboardX import SummaryWriter
import model_training.training as modules

import classifier.inceptiontime as module_arch_inceptiontime
import classifier.resnest as module_arch_resnest

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# model selection
files_models = {
    "inceptiontime": ['InceptionTimeV1', 'InceptionTimeV2'],
    "resnest": ['resnest50', 'resnest'],
}

log_step = 1


def train_12ECG_classifier(input_directory, output_directory):

    split_idx = 'model_training/split.mat'
    config_json = 'model_training/resnest.json'

    # Load data.
    print('Loading data...')

    # Split data into training data and validation data
    stratification(input_directory)

    # Get training configs
    with open(config_json, 'r', encoding='utf8')as fp:
        config = json.load(fp)

    # Train model.
    print('Training model...')

    train_challenge2020(config, split_idx, input_directory, output_directory)

    # Finish
    print('Finish training.')



# Load challenge data.
def load_challenge_data(header_file):
    with open(header_file, 'r') as f:
        header = f.readlines()
    mat_file = header_file.replace('.hea', '.mat')
    x = loadmat(mat_file)
    recording = np.asarray(x['val'], dtype=np.float64)
    return recording, header

# Find unique classes.
def get_classes(input_directory, filenames):
    classes = set()
    for filename in filenames:
        with open(filename, 'r') as f:
            for l in f:
                if l.startswith('#Dx'):
                    tmp = l.split(': ')[1].split(',')
                    for c in tmp:
                        classes.add(c.strip())
    return sorted(classes)

def train(model, optimizer, train_loader, criterion, metric, indices, epoch, device=None):
    sigmoid = nn.Sigmoid()
    model.train()
    cc = 0
    Loss = 0
    total = 0
    batchs = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_start = time.time()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if not indices is None:
            loss = criterion(output[:, indices], target[:, indices])
        else:
            loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        c = metric(to_np(sigmoid(output), device), to_np(target, device))
        cc += c
        Loss += loss
        total += target.size(0)
        batchs += 1

        if batch_idx % log_step == 0:
            batch_end = time.time()
            # logger.debug('Epoch: {} {} Loss: {:.6f}, 1 batch cost time {:.2f}'.format(epoch, batch_idx, loss.item(),
            #                                                                           batch_end - batch_start))
            print('Train Epoch: {} {} Loss: {:.6f}, 1 batch cost time {:.2f}'.format(epoch,
                                                                                     progress(train_loader, batch_idx),
                                                                                     loss.item(),
                                                                                     batch_end - batch_start))

    return Loss / total, cc / batchs

def valid(model, valid_loader, criterion, metric, indices, device=None):
    sigmoid = nn.Sigmoid()
    model.eval()
    cc = 0
    Loss = 0
    total = 0
    batchs = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(valid_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            if not indices is None:
                loss = criterion(output[:, indices], target[:, indices])
            else:
                loss = criterion(output, target)
            c = metric(to_np(sigmoid(output), device), to_np(target, device))
            cc += c
            Loss += loss
            total += target.size(0)
            batchs += 1

    return Loss / total, cc / batchs

def train_challenge2020(config, split_idx, input_directory, ouput_directory):
    
    # Paths to save log, checkpoint, tensorboard logs and results
    base_dir = config['base_dir'] + '/training_results'
    result_dir, log_dir, checkpoint_dir, tb_dir = make_dirs(base_dir)

    # Logger for train
    logger = get_logger(log_dir + '/info.log', name='train')

    # Tensorboard
    train_writer = SummaryWriter(tb_dir + '/train')
    val_writer = SummaryWriter(tb_dir + '/valid')

    # Setup Cuda
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    # Data_loader
    train_loader = ChallengeDataLoader(input_directory, split_idx,
                                          batch_size=config['data_loader']['batch_size'],
                                          normalization=config['data_loader']['normalization'],
                                          augmentations=config['data_loader']['augmentation']['method'],
                                          p=config['data_loader']['augmentation']['prob'])

    savemat(os.path.join(ouput_directory, 'classes.mat'), {'val':train_loader.classes})
    savemat(os.path.join(ouput_directory, 'indices.mat'), {'val':train_loader.indices})

    valid_loader = train_loader.valid_data_loader

    # Build model architecture
    global model
    for file, types in files_models.items():
        for type in types:
            if config["arch"]["type"] == type:
                model = init_obj(config, 'arch', eval("module_arch_" + file))

    model.to(device)

    # Get function handles of loss and metrics
    criterion = getattr(modules, config['loss']['type'])

    # Get function handles of metrics
    challenge_metrics = ChallengeMetric(input_directory)
    metric = challenge_metrics.challenge_metric

    # Get indices of the scored labels
    if config['trainer']['only_scored']:
        indices = challenge_metrics.indices
    else:
        indices = None

    # Build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = init_obj(config, 'optimizer', torch.optim, trainable_params)

    if config['lr_scheduler']['type'] == 'GradualWarmupScheduler':
        params = config["lr_scheduler"]["args"]
        scheduler_steplr_args = dict(params["after_scheduler"]["args"])
        scheduler_steplr = getattr(torch.optim.lr_scheduler, params["after_scheduler"]["type"])(optimizer, **scheduler_steplr_args)
        lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=params["multiplier"],
                                              total_epoch=params["total_epoch"], after_scheduler=scheduler_steplr)
    else:
        lr_scheduler = init_obj(config, 'lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # Begin training process
    trainer = config['trainer']
    epochs = trainer['epochs']

    # Full train and valid logic
    mnt_metric_name, mnt_mode, mnt_best, early_stop = get_mnt_mode(trainer)
    not_improved_count = 0

    for epoch in range(epochs):
        best = False
        train_loss, train_metric = train(model, optimizer, train_loader, criterion, metric, indices, epoch, device=device)
        val_loss, val_metric = valid(model, valid_loader, criterion, metric, indices, device=device)

        if config['lr_scheduler']['type'] == 'ReduceLROnPlateau':
            lr_scheduler.step(val_loss)
        elif config['lr_scheduler']['type'] == 'GradualWarmupScheduler':
            lr_scheduler.step(epoch, val_loss)
        else:
            lr_scheduler.step()

        logger.info(
            'Epoch:[{}/{}]\t {:10s}: {:.5f}\t {:10s}: {:.5f}'.format(epoch, epochs, 'loss', train_loss, 'metric',
                                                                     train_metric))
        logger.info(
            '             \t {:10s}: {:.5f}\t {:10s}: {:.5f}'.format('val_loss', val_loss, 'val_metric', val_metric))
        logger.info('             \t learning_rate: {}'.format(optimizer.param_groups[0]['lr']))

        # check whether model performance improved or not, according to specified metric(mnt_metric)
        if mnt_mode != 'off':
            mnt_metric = val_loss if mnt_metric_name == 'val_loss' else val_metric
            improved = (mnt_mode == 'min' and mnt_metric <= mnt_best) or \
                       (mnt_mode == 'max' and mnt_metric >= mnt_best)
            if improved:
                mnt_best = mnt_metric
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1

            if not_improved_count > early_stop:
                logger.info("Validation performance didn\'t improve for {} epochs. Training stops.".format(early_stop))
                break

        save_checkpoint(model, epoch, mnt_best, checkpoint_dir, save_best=False)

        if best == True:
            save_checkpoint(model, epoch, mnt_best, ouput_directory, save_best=True)
            logger.info("Saving current best: model_best.pth ...")

        # Tensorboard log
        train_writer.add_scalar('loss', train_loss, epoch)
        train_writer.add_scalar('metric', train_metric, epoch)
        train_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        val_writer.add_scalar('loss', val_loss, epoch)
        val_writer.add_scalar('metric', val_metric, epoch)


    # Model calibration
    print('Begin model calibration...')

    # model calibration
    checkpoint = load_checkpoint(ouput_directory, use_cuda)
    model.load_state_dict(checkpoint['state_dict'])

    model_calibration(model, valid_loader, input_directory, ouput_directory, device)


def model_calibration(model, valid_loader, input_directory, ouput_directory, device):
    space = {
        "alpha1": hp.quniform("alpha1", 0, 1, 0.1),
        "alpha2": hp.quniform("alpha2", 0, 1, 0.1),
        "alpha3": hp.quniform("alpha3", 0, 1, 0.1),
        "alpha4": hp.quniform("alpha4", 0, 1, 0.1),
        "alpha5": hp.quniform("alpha5", 0, 1, 0.1),
        "alpha6": hp.quniform("alpha6", 0, 1, 0.1),
        "alpha7": hp.quniform("alpha7", 0, 1, 0.1),
        "alpha8": hp.quniform("alpha8", 0, 1, 0.1),
        "alpha9": hp.quniform("alpha9", 0, 1, 0.1),
        "alpha10": hp.quniform("alpha10", 0, 1, 0.1),
        "alpha11": hp.quniform("alpha11", 0, 1, 0.1),
        "alpha12": hp.quniform("alpha12", 0, 1, 0.1),
        "alpha13": hp.quniform("alpha13", 0, 1, 0.1),
        "alpha14": hp.quniform("alpha14", 0, 1, 0.1),
        "alpha15": hp.quniform("alpha15", 0, 1, 0.1),
        "alpha16": hp.quniform("alpha16", 0, 1, 0.1),
        "alpha17": hp.quniform("alpha17", 0, 1, 0.1),
        "alpha18": hp.quniform("alpha18", 0, 1, 0.1),
        "alpha19": hp.quniform("alpha19", 0, 1, 0.1),
        "alpha20": hp.quniform("alpha20", 0, 1, 0.1),
        "alpha21": hp.quniform("alpha21", 0, 1, 0.1),
        "alpha22": hp.quniform("alpha22", 0, 1, 0.1),
        "alpha23": hp.quniform("alpha23", 0, 1, 0.1),
        "alpha24": hp.quniform("alpha24", 0, 1, 0.1),
    }

    model.eval()
    with torch.no_grad():
        outputs = torch.zeros((valid_loader.n_samples, model.num_classes))
        targets = torch.zeros((valid_loader.n_samples, model.num_classes))
        start = 0
        for batch_idx, (data, target) in enumerate(valid_loader):
            data = data.to(device=device, dtype=torch.float32)
            output = nn.Sigmoid()(model(data))
            end = len(data) + start
            outputs[start:end, :] = output
            targets[start:end, :] = target
            start = end
    outputs = to_np(outputs, device)
    targets = to_np(targets, device)

    def find_alpha(hp):
        alphas = [hp['alpha1'], hp['alpha2'], hp['alpha3'], hp['alpha4'], hp['alpha5'], hp['alpha6'], hp['alpha7'],
                  hp['alpha8'],
                  hp['alpha9'], hp['alpha10'], hp['alpha11'], hp['alpha12'], hp['alpha13'], hp['alpha14'],
                  hp['alpha15'],
                  hp['alpha16'],
                  hp['alpha17'], hp['alpha18'], hp['alpha19'], hp['alpha20'], hp['alpha21'], hp['alpha22'],
                  hp['alpha23'],
                  hp['alpha24']]

        challenge_metrics = ChallengeMetric_calibration(input_directory, alphas)

        accuracy, macro_f_measure, macro_f_beta_measure, macro_g_beta_measure, challenge_metric = get_metrics(
            outputs, targets, challenge_metrics=challenge_metrics)

        return -challenge_metric

    trials = Trials()
    max_evals = 500
    best = fmin(
        find_alpha,
        space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals,
    )
    print("BEST:", best)
    # for trial in trials:
    #     print(trial)
    best_alphas = list()
    paras = ['alpha1', 'alpha2', 'alpha3', 'alpha4', 'alpha5', 'alpha6', 'alpha7', 'alpha8',
            'alpha9', 'alpha10', 'alpha11', 'alpha12', 'alpha13','alpha14', 'alpha15', 'alpha16',
            'alpha17', 'alpha18', 'alpha19', 'alpha20', 'alpha21', 'alpha22', 'alpha23', 'alpha24']

    for p in paras:
        best_alphas.append(best[p])

    best_alphas = np.array(best_alphas)
    savemat(os.path.join(ouput_directory, 'best_alphas.mat'), {'val': best_alphas})



