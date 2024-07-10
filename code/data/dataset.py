import numpy as np
import torch
import torch.utils.data
from copy import deepcopy
import math
from torch.utils.data import Dataset
from utils.constants import Constants # type: ignore
 



data_file_dict = {

### load pre-processed datafile

    'houston_continuous': '/home/users/constraint_gen/dataset/veraset/houston/my_process/houston_60mins_random_constraint_traj.npz',
    'houston_readable_constraint': '/home/users/constraint_gen/dataset/veraset/houston/my_process/houston_60mins_readable_constraint.npy',
    'houston_sample_loc_time_statis_constraint': '/home/users/constraint_gen/dataset/veraset/houston/my_process/houston_60mins_statistics_constraint.npy',
    'houston_discrete': '/home/users/constraint_gen/dataset/veraset/houston/my_process/houston_discrete_traj.npz', 
    'houston_discrete_600': '/home/users/constraint_gen/dataset/veraset/houston/my_process/houston_discrete_traj_600.npz', 

}



def get_loc_2d_dim(dataset):


    # horizontol: lon vertical: lat
    if dataset == 'houston_discrete_600':
        return 80,92
    else:
        raise ValueError(f'No such dataset {dataset}')




    
def de_normalized(data,args,data_loader,dim,mask=None):

    #dim = data.shape[-1]
    
    if args.normalization == 'min_max':

        if dim == 1: ### time_gap_batch:
            min_val, max_val = data_loader.T_gap_min, data_loader.T_gap_max
            scale = max_val - min_val + 1e-8
            data = data * scale + min_val
        elif dim ==2: ### spatial batch
            min_val, max_val = data_loader.S_min, data_loader.S_max
            data[:,:,0] = data[:,:,0] * (max_val[0] - min_val[0]) + min_val[0]
            data[:,:,1] = data[:,:,1] * (max_val[1] - min_val[1]) + min_val[1]

    elif args.normalization == 'z_score':
        
        if dim == 1:
            mean, std = data_loader.T_gap_mean, data_loader.T_gap_std
            data = data * std + mean
    
        elif dim == 2:
            mean, std = data_loader.S_mean, data_loader.S_std
            device = data.device
            std = std.to(device)
            mean = mean.to(device)
            data = data * std + mean
        
    if mask != None:  
        if dim == 2:
            
            mask = mask.unsqueeze(-1)
            mask = mask.repeat(1,1,2)
        data = data * mask

    return data


class TimeDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data,dtype = torch.float32)
        #self.data = data # (N, max_T)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SpatioTemporalDataset(torch.utils.data.Dataset):

    def __init__(self, train_set, test_set, train):
        self.S_mean, self.S_std = self._standardize(train_set)
        S_mean_ = torch.cat([torch.zeros(1, 1).to(self.S_mean), self.S_mean], dim=1)
        S_std_ = torch.cat([torch.ones(1, 1).to(self.S_std), self.S_std], dim=1)
        self.dataset = [(torch.tensor(seq) - S_mean_) / S_std_ for seq in (train_set if train else test_set)]

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        return self.dataset[index]




class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self,args,name,normalization='standard',test=False):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """


        self.args = args
        self.name = name
        datafile = data_file_dict[name]
        dataset = np.load(datafile)

        if test:

            data = [dataset[f] for f in dataset.files[:32]]
        else:
            data = [dataset[f] for f in dataset.files]



        self.agent_id = [[elem[0] for elem in inst] for inst in data]
        self.all_lengths = [len(seq) for seq in self.agent_id] ## store
        
        self.get_time_gap(data=data)
        self.get_spatial_data(method=normalization,data=data)

 
        self.length = len(data)
        self.max_length = max(len(x) for x in self.agent_id) 

        self.lower_bound = [[elem[-2] for elem in inst] for inst in data]
        self.upper_bound = [[elem[-1] for elem in inst] for inst in data]
        

    

    def get_constraint(self):

        
        return self.max_length



    def get_spatial_data(self,data,method):
        
        if method == 'none':
            
            self.lat = [[elem[2]   for elem in inst] for inst in data]
            self.long =[[elem[3]  for elem in inst] for inst in data]


            
            self.time = [[elem[1] for elem in inst] for inst in data]

                  
        elif method == 'z_score':

            dataset = [torch.tensor(seq) for seq in data]
            full = torch.cat(dataset, dim=0)
            S = full[:, 2:4]
            self.S_mean = S.mean(0, keepdims=True) #[1,2]
            self.S_std = S.std(0, keepdims=True)
            

            
            self.lat = [[(elem[2] -self.S_mean[0,0]) /self.S_std [0,0]  for elem in inst] for inst in data]
            self.long =[[(elem[3]  -self.S_mean[0,1]) / self.S_std [0,1]   for elem in inst] for inst in data]


            time_array_torch = full[:, 1:2]
            self.T_mean = time_array_torch.mean(0).item()
            self.T_std = time_array_torch.std(0).item()

            self.time = [[elem[1]  for elem in inst] for inst in data]
            self.time = [[(elem[1] - self.T_mean) / (self.T_std)  for elem in inst] for inst in data]


            
        
        elif method == 'min_max':

            
            dataset = [torch.tensor(seq) for seq in data]
            full = torch.cat(dataset, dim=0)
            S = full[:, 2:4]

              
            self.S_max = S.max(0, keepdims=True)[0].squeeze(0) #[1,2]
            self.S_min = S.min(0, keepdims=True)[0].squeeze(0)         
            self.S_max += Constants.OFFSET_S
            print(self.S_min,self.S_max)



            self.lat = [[(elem[2] - self.S_min[0]) / (self.S_max [0] - self.S_min [0])  for elem in inst] for inst in data]    
            self.long =[[(elem[3] - self.S_min[1]) / (self.S_max [1] - self.S_min [1])  for elem in inst] for inst in data]
            self.time = [[elem[1]  for elem in inst] for inst in data]



    
    def get_time_gap(self,data):
        
        
        time_array = [[elem[1] for elem in inst] for inst in data]
        time_gap_dif  = [[j-i+ Constants.OFFSET for i, j in zip(t[:-1], t[1:])] for t in time_array]
        time_gap_diff = [[t[0]+ Constants.OFFSET]+seq for (t,seq) in zip(time_array,time_gap_dif) ] 
        time_gap_diff_torch = torch.cat([torch.tensor(seq) for seq in time_gap_diff],dim=0)
        time_gap_diff_torch = time_gap_diff_torch
        
    
        self.mean_log_inter_time = time_gap_diff_torch.log().mean() 
        self.std_log_inter_time =  time_gap_diff_torch.log().std()
        min_val = time_gap_diff_torch.log().min().item()
        max_val = time_gap_diff_torch.log().max().item()

        self.T_gap_min = time_gap_diff_torch.min().item()
        self.T_gap_max = time_gap_diff_torch.max().item()

        print(f'get_stats time gap {self.mean_log_inter_time.item()} {self.std_log_inter_time.item()} {min_val} {max_val}')

        self.T_gap_mean = time_gap_diff_torch.mean().item()
        self.T_gap_std = time_gap_diff_torch.std().item()
        self.time_gap = time_gap_diff


    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        return self.agent_id[idx],self.time[idx], self.time_gap[idx], self.long[idx], self.lat[idx], self.lower_bound[idx], self.upper_bound[idx]


def pad_time(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)



    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32)


def pad_constraint(insts,pad_value):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)



    batch_seq = np.array([
        inst + [pad_value] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32)


def pad_type(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.long)


def collate_fn(insts):
    """ Collate function, as required by PyTorch. """

    agent_id, time, time_gap, lng, lat, lower_bound_time, upper_bound_time = list(zip(*insts))
    lengths = [len(seq) for seq in agent_id]
    max_len = max(lengths)
    num = len(lengths)
    
    mask = torch.stack([torch.cat([torch.ones(seq_len), torch.zeros(max_len - seq_len)], dim=0) for seq_len in lengths])
    time = pad_time(time)
    
    agent_id = pad_time(agent_id)
    time_gap = pad_time(time_gap)
    lat = pad_time(lat)

    lng = pad_time(lng)
    lower_bound_time = pad_constraint(lower_bound_time,0.002)
    upper_bound_time = pad_constraint(upper_bound_time,24.0)

    stop_state = torch.stack([torch.cat([torch.zeros(seq_len-1), torch.ones(max_len - seq_len+1)], dim=0) for seq_len in lengths])
    return agent_id, time, time_gap, lng, lat , mask, stop_state, lower_bound_time, upper_bound_time


def get_dataloader(args,data, batch_size,normalization, shuffle=True):
    """ Prepare dataloader. """


    ds = EventData(args,data,normalization = normalization, test=args.test_small_batch)
    
    
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=8,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )
    return dl
