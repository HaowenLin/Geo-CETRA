import argparse
import itertools
import math
import numpy as np
import os
import sys
#print(sys.path)
import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.utils import prep_workspace, Namespace
from code.train_st_transformer import Trainer




def inital_setting(seed:int):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def args_set():
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--training_mode", type=str, choices=["evaluate", "training"], default='evaluate')
    parser.add_argument("--best_model_cp", type=str, default='/home/users/haowenli/constraint_gen/exps/thp/baseline_60mins_remove_zero/12_05_2023-14_27_45')
    
    ## debug
    parser.add_argument("--mode", type=str, default='with_s_encode,spatial_time,none') # none #no_pretrain
    parser.add_argument('--test_small_batch', default=True, type=lambda x: (str(x).lower() == 'true')) ## true just test with 32 samples
    parser.add_argument('--pre_time', default=False, type=lambda x: (str(x).lower() == 'true')) ## true just test with 32 samples

    
    # log 
    parser.add_argument('--is_wandb_used', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--save_path', type=str, default='/home/users/haowenli/constraint_gen/exps', help='')
    parser.add_argument("--logfreq", type=int, default=3)
    parser.add_argument("--evalfreq", type=int, default=5)
    parser.add_argument("--display_name", type=str, default='stop_test')
    

    # cuda
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--is_cuda", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)

    # dataset
    #parser.add_argument("--dataset", type=str, default='imstaypoints')# imstaypoints
    parser.add_argument("--dataset", type=str, default='baseline_60mins_constraint_1127')# imstaypoints
    parser.add_argument("--time_format", type=str,choices=["time_gap", "time_increase"], default='time_gap')
    parser.add_argument("--normalization", type=str, choices=["min_max", "z_score","none"],default='z_score') #
    parser.add_argument("--max_events", type=int, default=2500) # 260
    parser.add_argument("--max_length", type=int, default=65)
    #parser.add_argument("--time_format", type=float, default=60)

    # model
    parser.add_argument("--spatial_num_mix_components", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_rnn", type=int, default=128)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--spatial_model", type=str, choices=["condGMM", "attncnf",],default='condGMM')
    

    #parser.add_argument("--mode", type=str, default='with_s_encode,spatial_time,none') # none #no_pretrain
    
    parser.add_argument("--encoder", type=str, choices=["general", "angle",],default='general')



    

    ### gausian mixture model 

    
    parser.add_argument("--num_mix_components", type=int, default=1)
    parser.add_argument("--model_choice", type=str,choices=["time_log", "regression","angle"], default='time_log') #
    parser.add_argument("--time_decoder", type=str,choices=["none", "constraint","truncate"], default='constraint') #


    # training 

    parser.add_argument("--scheduler_gamma", type=float, default=0.95)
    parser.add_argument("--stop_epochs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--gen_epoch", type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=26)
    
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--stop_lr", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--gradclip", type=float, default=10)

    parser.add_argument("--eta", type=float, default=1e-4) # for alpha update 
    parser.add_argument("--theta_steps", type=int, default=2)
    parser.add_argument("--warmup_itrs", type=int, default=0)
    parser.add_argument("--loss_type", type=str, choices=["mae", "mse"],default='mse') #
    parser.add_argument("--lr_s_std_regularization", type=float, default=0.0)



    ### generation
        
    parser.add_argument("--generation_requirment", type=str, choices=["none","single_constraint","multi_constraint"], default='single_constraint')
    parser.add_argument("--sample_location_method", type=str, choices=["fixed_statistical","time_statistical","gt_index","beam_search"], default='fixed_statistical')
    parser.add_argument("--beam_search_k", type=int, default=3)
    parser.add_argument("--top_prob_k", type=int, default=3)
    parser.add_argument("--index_k", type=int, default=1)


    parser.add_argument("--enc_type", type=str,default='l_default,t_default') #gaussian_noise,
    


    return parser.parse_args()
    





    






if __name__ == '__main__':



    args = args_set()
    torch.autograd.set_detect_anomaly(True)
    args.mode =  list(map(str, args.mode.split(",")))
    args.enc_type =  list(map(str, args.enc_type.split(",")))
    model_type = 'thp'

    if args.training_mode != 'evaluate':
        path = prep_workspace(args.save_path,model_type = model_type,datasets =args.dataset,is_wandb_used=args.is_wandb_used)
        args.save_final = path
    else:
        path = prep_workspace(args.save_path,model_type = model_type,datasets =args.dataset,is_wandb_used=args.is_wandb_used)
        args.save_final = args.best_model_cp
    inital_setting(args.seed)
    device = torch.device("cuda:{}".format(args.cuda_id) if args.is_cuda else "cpu")
    trainer = Trainer(args)

    save_constraint = True
    irregular=True
    
    
    if args.training_mode == 'evaluate': ### load pretrain and generate
        best_epoch = trainer.load_best()
        if args.pre_time == True:
                trainer.generation_loc_step(epoch=best_epoch,irregular=True)
        else:
                trainer.generation_step(epoch=best_epoch,irregular=True,save_constraint=save_constraint)
        
    else: ### training the model and generate the result
        trainer.train_epoch_time_log()
        best_epoch = trainer.load_best()
        trainer.generation_step(epoch=best_epoch,irregular=True)

        
    

