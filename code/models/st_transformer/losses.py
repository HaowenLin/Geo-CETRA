import torch
import torch.nn.functional as F
import numpy as np



def time_log_loss(prediction, event_time,mask,loss_type='mse'):
    """ Time prediction loss. """

    prediction.squeeze_(-1) #[batch, seq_len]

    prediction = prediction[:, :-1]
    event_time = event_time[:, 1:]

    #mask.squeeze_(-1)
    mask = mask[:, 1:]

    error_amplitude = (event_time - prediction).abs().float()
    zeros = torch.zeros_like(error_amplitude).float()
    


    


    #time_loss = torch.sum(F.l1_loss(prediction, event_time, reduction='none')*mask )/ torch.sum(mask)
    if loss_type == 'mse':
        time_loss = F.mse_loss(error_amplitude, zeros, reduction='none')*mask 
    elif loss_type == 'mae':
        time_loss = F.l1_loss(error_amplitude, zeros, reduction='none')*mask 
    #time_loss = torch.sum(F.mse_loss(prediction, event_time, reduction='none')*mask )/ torch.sum(mask)
    return time_loss

def time_loss(prediction, event_time,mask,loss_type='mse'):
    """ Time prediction loss. """

    prediction.squeeze_(-1) #[batch, seq_len]

    if loss_type == 'mse':
        time_loss = F.mse_loss(prediction, event_time, reduction='none')*mask 
    elif loss_type == 'mae':
        time_loss = F.l1_loss(prediction, event_time, reduction='none')*mask 
    #time_loss = torch.sum(F.mse_loss(prediction, event_time, reduction='none')*mask )/ torch.sum(mask)
    return time_loss



def spatial_loss(prediction, event_spatial,mask,loss_type='mse'):
    """ Time prediction loss. """

    prediction.squeeze_(-1) #[batch, seq_len]

    mask = mask[:,:,None].repeat(1,1,2)

    if loss_type == 'mse':
        time_loss = torch.sum(F.mse_loss(prediction, event_spatial, reduction='none') * mask)/ torch.sum(mask)
    elif loss_type == 'mae':
        time_loss = torch.sum(F.l1_loss(prediction, event_spatial, reduction='none') * mask)/ torch.sum(mask)
    return time_loss



def RMSE(v, v_, axis=None):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, RMSE averages on all elements of input.
    '''
    return np.sqrt(np.mean((v_ - v) ** 2, axis)).astype(np.float64)





    