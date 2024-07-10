import numpy as np
import wandb
import os

import logging
import torch

from datetime import datetime
import math


PAD = 0.0
STOP = 0.98
OFFSET = 0.0
OFFSET_S = 0.0

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %.3f M' % (num_params/1000000))

def glorot(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    init = (2 * init_range) * torch.rand(shape[0], shape[1]) - init_range
    # init = init / (init.norm(2, dim=1).unsqueeze(1) + 1e-8)
    return init

def prep_workspace(save_prefix, model_type, datasets, is_wandb_used=False):
    """
    prepare a workspace directory
    :param workspace:
    :param oridata:
    :return:
    """
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y-%H_%M_%S")
    if not is_wandb_used:
        path = save_prefix+'/test/%s/%s/%s' % (model_type,datasets,date_time)
    else:
        path = save_prefix+'/%s/%s/%s' % (model_type,datasets,date_time)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def get_workspace_logger(data_path):

    #data_path = '../data'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s: %(message)s")
    fh = logging.FileHandler(data_path +'/all.log' , mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


class WandbLogger():
    def __init__(self, project,is_used,name = None):

        self.is_used = is_used
        if is_used:
            wandb.init(project=project,entity="lin",name=name)

    def watch_model(self,model):
        if self.is_used:
            wandb.watch(model)

    def log_hyperparams(self, params):
        if self.is_used:
            wandb.config.update(params)

    def log_metrics(self, metrics):
        if self.is_used:
            wandb.log(metrics)

    def log(self, key, value, round_idx):
        if self.is_used:
            wandb.log({key: value, "Round": round_idx})

    def log_str(self, key, value):
        if self.is_used:
            wandb.log({key: value})


    def save_file(self, path):
        if path is not None and os.path.exists(path) and self.is_used:
            wandb.save(path)



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))

def compute_event(event, non_pad_mask):
    """ Log-likelihood of events. """

    # add 1e-9 in case some events have 0 likelihood
    event += math.pow(10, -9)
    event.masked_fill_(~non_pad_mask.bool(), 1.0)

    result = torch.log(event)
    return result

def compute_integral_biased(all_lambda, time, non_pad_mask):
    """ Log-likelihood of non-events, using linear interpolation. """

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    diff_lambda = (all_lambda[:, 1:] + all_lambda[:, :-1]) * non_pad_mask[:, 1:]

    biased_integral = diff_lambda * diff_time
    result = 0.5 * biased_integral
    return result


def compute_integral_unbiased(model, data, time, non_pad_mask, type_mask):
    """ Log-likelihood of non-events, using Monte Carlo integration. """

    num_samples = 100

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:] # [batch_size,seq_len]
    temp_time = diff_time.unsqueeze(2) * \
                torch.rand([*diff_time.size(), num_samples], device=data.device)# [batch_size,seq_len,numsamples]
    temp_time /= (time[:, :-1] + 1).unsqueeze(2) ### compute integral , how to get rid of that 

    temp_hid = model.linear(data)[:, 1:, :]
    temp_hid = torch.sum(temp_hid * type_mask[:, 1:, :], dim=2, keepdim=True)

    all_lambda = softplus( temp_time, model.beta)
    all_lambda = torch.sum(all_lambda, dim=2) / num_samples

    unbiased_integral = all_lambda * diff_time
    return unbiased_integral


def log_likelihood(model, data, time, types):
    """ Log-likelihood of sequence. """

    non_pad_mask = get_non_pad_mask(time).squeeze(2) #[batch, seq_len]

    # type_mask = torch.zeros([*types.size(), model.num_types], device=data.device)
    # for i in range(model.num_types):
    #     type_mask[:, :, i] = (types == i + 1).bool().to(data.device)

    all_hid = model.linear(data) #[batch, seq_len,10] ???
    all_lambda = softplus(all_hid, model.beta) #[batch, seq_len,10] ???
    #type_lambda = torch.sum(all_lambda * type_mask, dim=2)

    # event log-likelihood
    # event_ll = compute_event(type_lambda, non_pad_mask)
    # event_ll = torch.sum(event_ll, dim=-1)

    # non-event log-likelihood, either numerical integration or MC integration
    # non_event_ll = compute_integral_biased(type_lambda, time, non_pad_mask)
    non_event_ll = compute_integral_unbiased(model, data, time, non_pad_mask, non_pad_mask)
    non_event_ll = torch.sum(non_event_ll, dim=-1)

    return  non_event_ll



class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val