import os,sys
import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from data.dataset import get_dataloader,data_file_dict
from utils.utils import *
#import models.st_transformer.losses as losses
from data.dataset import de_normalized, TimeDataset
from models.st_transformer.Time_lognorm import ST_Encoder_lognorm


def cast(tensor, device):
    return tensor.float().to(device)




class Trainer(object):
    def __init__(self, args):
        
        self.args = args
        self.save_path_final  = args.save_final
        self.device = torch.device("cuda:{}".format(args.cuda_id) if args.is_cuda else "cpu")

        
        self.prepare_dataloader()
        display_name = args.display_name

        
        
        
        self.logger = get_workspace_logger(self.save_path_final)
        self.logger.info(f"save path at {self.save_path_final}")
        self.wandb_logger = WandbLogger("constrain_gen", self.args.is_wandb_used,name=display_name)
        self.wandb_logger.log_hyperparams(self.args)
        self.load_model()
        

        now = datetime.now()
        self.date_time = now.strftime("%m_%d_%Y")
        
        

        self.space_loss = 0.
        self.time_loss = 0.
        self.max_length_batch = []
        self.logger.info(f"max length {self.trainloader.dataset.max_length} traj numbers {len(self.trainloader.dataset)}")
        self.max_length = self.trainloader.dataset.max_length

    

    def prepare_dataloader(self):
        self.trainloader = get_dataloader(self.args,self.args.dataset ,self.args.batch_size,self.args.normalization,shuffle=True)
        self.testloader = get_dataloader(self.args,self.args.dataset ,self.args.batch_size,self.args.normalization,shuffle=False)

        
    
    def load_model(self):

        self.optims = []
        self.scheds = []

        mean_log_inter_time = self.trainloader.dataset.mean_log_inter_time.to(self.device)
        std_log_inter_time = self.trainloader.dataset.std_log_inter_time.to(self.device)
        self.model = ST_Encoder_lognorm(args=self.args,device=self.device,d_model=self.args.d_model,d_rnn=self.args.d_rnn,n_head=self.args.n_head, n_layers=self.args.n_layer, mean_log_inter_time=mean_log_inter_time, std_log_inter_time = std_log_inter_time)


        self.model.distance_model.mean_log_inter_time = self.trainloader.dataset.mean_log_dis.to(self.device)
        self.model.distance_model.std_log_inter_time = self.trainloader.dataset.std_log_dis.to(self.device)
        self.model =self.model.to(self.device)
       
        
        self.optimizer = AdamW(self.model.parameters(), lr = self.args.lr, betas = (0.9, 0.99))
        self.optims.append(self.optimizer)
        scheduler = optim.lr_scheduler.ExponentialLR(self.optims[0],gamma=self.args.scheduler_gamma)
        self.scheds.append(scheduler)
        

    def init_weights(self, module):
        """Initialize the weights"""


        if isinstance(module, nn.Linear):
            initrange = 0.1
            module.weight.data.normal_(mean=0.0, std=initrange)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=initrange)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    
    
    def gaussian_sample(mean, log_std):
        mean = mean + torch.tensor(0.)
        log_std = log_std + torch.tensor(0.)
        z = torch.randn_like(mean) * torch.exp(log_std) + mean
        return z

    def adjust_learning(self):
        if self.space_loss == 0 and self.time_loss == 0.:
            return 1.
        else:
            return self.time_loss / self.space_loss
    
    
    def get_num_parameters(self):
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.log('[Info] Number of parameters: {}'.format(num_params))

    def evaluate(self,prediction,target,mask, dim):
        
        prediction = de_normalized(prediction,self.args,self.trainloader.dataset,dim)
        target = de_normalized(target,self.args,self.trainloader.dataset,dim)

        prediction = prediction.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        total = mask.sum().item() * abs(dim)

        rmse = np.sqrt(np.sum((prediction-target)**2)/ total ).astype(np.float64)
        return rmse



    def load_best(self):
        
        best_file = os.path.join(self.args.save_final,'best_model.pt') 
        checkpoint= torch.load(best_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        return epoch
      


    def load_test_from_npy(self):

        time = '02_13_2024-19_33_42'
        dataset = 'baseline_60mins_constraint_polar'
        test_option = False
        interval_option = False
        epoch_num = 249
        npy_data = load_test_data(time,epoch_num,test=test_option,interval_option=interval_option,dataset=dataset) #[n,21]
        #if time only, batch_size,length

        time_dataset = TimeDataset(npy_data)
        
        self.time_loader  = torch.utils.data.DataLoader(time_dataset,
            num_workers=8,
            batch_size=self.args.batch_size,
            shuffle=False
        )


            

    @torch.no_grad() ### this funcation only use if you want to generate the time first and generate location
    def generation_loc_step(self,epoch,irregular=False,save_constraint = False):

        self.load_test_from_npy()
        self.model.eval()
        samples = []
        gt_samples = []
        samples_ori = []
        seq_len_all = []
        constraint = [] 


        for batch_idx, (batch,pre_time_batch) in enumerate(zip(self.testloader,self.time_loader)):
            agent_id_batch, time_batch, time_gap_batch, lng_batch, lat_batch, mask_batch, stop_state_batch, lower_bound_time, upper_bound_time = map(lambda x: cast(x, self.device), batch)
            event_loc = torch.cat((lat_batch.unsqueeze(dim=2),lng_batch.unsqueeze(dim=2)),dim=-1) #[b,s,2]
            pre_time_batch = pre_time_batch.to(self.device)
            start_time = pre_time_batch[:,0].unsqueeze(-1) 
            start_loc = torch.cat((lat_batch[:,0].unsqueeze(dim=1),lng_batch[:,0].unsqueeze(dim=1)),dim=-1) #[batch,2]
            start_loc = start_loc.unsqueeze(dim=1) #[batch,1,2]
            generated_batches = self.model.sample_loc_only(num_samples=time_gap_batch.shape[0],start_batch=(start_loc,start_time),max_len=self.trainloader.dataset.max_length,generated_time=pre_time_batch)


            generated_loc, generated_time, gen_len = generated_batches

                
            
            assert generated_time.shape[0] == generated_loc.shape[0] and generated_time.shape[1] == generated_loc.shape[1], "generated time and loc does not match"
            
            
            if self.args.time_format == 'time_gap':
                generated_time = generated_time- OFFSET
                de_noramized_time = torch.cumsum(generated_time,dim=1) 
                time_gap_batch = time_gap_batch - OFFSET
                gt_de_normalized_time = torch.cumsum(time_gap_batch,dim=1)
                gt_de_normalized_time = gt_de_normalized_time * mask_batch ### this is for gt only, no wonder

                
    
            de_normalized_spatial = de_normalized(generated_loc,self.args,self.trainloader.dataset,dim=2)
            
            gt_de_normalized_spatial = de_normalized(event_loc,self.args,self.trainloader.dataset,dim=2,mask = mask_batch)  
            gt_de_normalized_time = gt_de_normalized_time.unsqueeze(dim=-1)
            de_noramized_time = de_noramized_time.unsqueeze(dim=-1)
            generated_time = generated_time.unsqueeze(dim=-1)


            gt_batch_ori = torch.concat((gt_de_normalized_time,gt_de_normalized_spatial),dim=-1)
            generated_batch_ori = torch.concat((generated_time,generated_loc),dim=-1)
            generated_batch = torch.concat((de_noramized_time,de_normalized_spatial),dim=-1)
            
            gt_batch_ori = gt_batch_ori.cpu().detach().numpy()
            generated_batch = generated_batch.cpu().detach().numpy()
            generated_batch_ori = generated_batch_ori.cpu().detach().numpy() #[2,2,3]
            gen_len = gen_len.cpu().detach().numpy()

            if irregular == True:
                pad_generated_batch_ori = np.zeros((generated_batch_ori.shape[0],self.trainloader.dataset.max_length,generated_batch_ori.shape[-1]))
                pad_generated_batch = np.zeros((generated_batch.shape[0],self.trainloader.dataset.max_length,generated_batch.shape[-1]))
                pad_generated_batch_ori[:,:generated_batch_ori.shape[1],:] = generated_batch_ori
                pad_generated_batch[:,:generated_batch.shape[1],:] = generated_batch
                samples_ori.extend(pad_generated_batch_ori)
                samples.extend(pad_generated_batch)
            else:
                samples_ori.extend(generated_batch_ori)
                samples.extend(generated_batch)


            seq_len_all.extend(gen_len)
            gt_samples.append(gt_batch_ori)
        samples = np.array(samples)
        samples_ori = np.array(samples_ori)
        seq_len_all = np.array(seq_len_all)

        if save_constraint:
            constraint = np.array(constraint)
            save_pth = os.path.join(self.args.save_final, f'time_constraint_epoch_{epoch}')
            pickle.dump(constraint, open(save_pth, 'wb'))
       
        self.save_generated_file(gt_samples,samples_ori, samples,seq_len_all, epoch=epoch)
    
    def process_beam_search_index(self,generated_batches_list,update_constraint_list,ret_prob_list):

        
        ret_prob = torch.stack(ret_prob_list) #[index_k,b]
        max_values, indices = torch.max(ret_prob, dim=0)

        generated_loc = [t[0] for t in generated_batches_list]
        generated_time = [t[1] for t in generated_batches_list]

        concatenated_generated_loc = torch.stack(generated_loc, dim=-1)
        concatenated_generated_time = torch.stack(generated_time, dim=-1)
        batch_indices_time = indices[:,None,None].expand(-1,concatenated_generated_time.size(1),-1)
        batch_indices_loc = indices[:,None,None,None].expand(-1,concatenated_generated_loc.size(1),concatenated_generated_loc.size(2),-1)

        selected_time= torch.gather(concatenated_generated_time, dim=2, index=batch_indices_time).squeeze(-1)
        selected_loc= torch.gather(concatenated_generated_loc, dim=-1, index=batch_indices_loc).squeeze(-1)

        ret_generated_batches = (generated_batches_list[0],selected_loc,selected_time)



        lower_bound = [t[0].squeeze(-1) for t in update_constraint_list]
        upper_bound = [t[1].squeeze(-1) for t in update_constraint_list]

        lower_bound = torch.stack(lower_bound, dim=-1)
        upper_bound = torch.stack(upper_bound, dim=-1)

        selected_lower_bound = torch.gather(lower_bound, dim=-1, index=batch_indices_time[:,:-1,:])
        selected_upper_bound = torch.gather(upper_bound, dim=-1, index=batch_indices_time[:,:-1,:])


        ret_constraint = (selected_lower_bound,selected_upper_bound)

        return ret_generated_batches,ret_constraint



    @torch.no_grad()
    def generation_step(self,epoch,irregular=False,save_constraint = False):

        self.model.eval()
        samples = []
        gt_samples = []
        samples_ori = []
        seq_len_all = []
        constraint = [] 
        sample_index_all = []  
        if self.args.generation_requirment != "none":
            self.load_constraint_file(method = self.args.sample_location_method)
        

        for batch_idx,batch in enumerate(self.testloader):
            #print(f'check {batch_idx}')
            agent_id_batch, time_batch, time_gap_batch, lng_batch, lat_batch, mask_batch, stop_state_batch, lower_bound_time, upper_bound_time = map(lambda x: cast(x, self.device), batch)
            event_loc = torch.cat((lat_batch.unsqueeze(dim=2),lng_batch.unsqueeze(dim=2)),dim=-1) #[b,s,2]
            
            
            
            if self.args.generation_requirment != "none":
                read_constraint_batch = self.read_constraint_dataset[batch_idx*self.args.batch_size:batch_idx*self.args.batch_size+event_loc.shape[0],:]
                constraint_index  = self.sample_constraint_loc(self.args.sample_location_method,event_loc.shape[0],batch_idx) #[batch]
                constraint_index = constraint_index.to(self.device)


            
        
            start_time = time_gap_batch[:,0].unsqueeze(-1) #[batch,1], 
            start_loc = torch.cat((lat_batch[:,0].unsqueeze(dim=1),lng_batch[:,0].unsqueeze(dim=1)),dim=-1) #[batch,2]
            start_loc = start_loc.unsqueeze(dim=1) #[batch,1,2]
            
            if self.args.generation_requirment != "none": ### has constraint adaptive
                ### default seeting 
                if self.args.time_decoder == 'constraint' or self.args.time_decoder == 'truncate':
                    if self.args.index_k > 1 and self.args.sample_location_method== 'beam_search':
                        ret_prob_list = []
                        generated_batch_list = []
                        update_constraint_list = []
                        for i in range(self.args.index_k):
                            generated_batches,update_constraint,ret_prob = self.model.sample_adaptive_constraint(num_samples=time_gap_batch.shape[0],start_batch=(start_loc,start_time),max_len=self.trainloader.dataset.max_length,readable_constraint=read_constraint_batch,constraint_loc = constraint_index[:,i])
                            generated_batch_list.append(generated_batches)
                            ret_prob_list.append(ret_prob)
                            update_constraint_list.append(update_constraint)
                        self.process_beam_search_index(generated_batch_list,update_constraint_list,ret_prob_list)
                    else:
                        generated_batches,update_constraint,ret_prob = self.model.sample_adaptive_constraint(num_samples=time_gap_batch.shape[0],start_batch=(start_loc,start_time),max_len=self.trainloader.dataset.max_length,readable_constraint=read_constraint_batch,constraint_loc = constraint_index)
                elif self.args.time_decoder == 'none':

                    generated_batches,update_constraint = self.model.sample_adaptive_constraint_wo_constraint(num_samples=time_gap_batch.shape[0],start_batch=(start_loc,start_time),max_len=self.trainloader.dataset.max_length,readable_constraint=read_constraint_batch,constraint_loc = constraint_index)

            else: ### not adaptive constraint
                
                generated_batches,update_constraint = self.model.sample_adaptive_constraint_wo_constraint(num_samples=time_gap_batch.shape[0],start_batch=(start_loc,start_time),max_len=self.trainloader.dataset.max_length,readable_constraint=read_constraint_batch,constraint_loc = constraint_index)
            
            
            generated_loc, generated_time, gen_len = generated_batches


            if save_constraint:
                if self.args.generation_requirment != "none":
                    lower_bound_update,upper_bound_update = update_constraint
                    save_final = torch.cat((lower_bound_update.unsqueeze(-1),upper_bound_update.unsqueeze(-1)),dim=-1)
                    save_final = save_final.cpu().detach().numpy()
                    constraint.extend(save_final)
                    sample_index_all.extend(constraint_index.cpu().detach().numpy())
            
            
                
            
            assert generated_time.shape[0] == generated_loc.shape[0] and generated_time.shape[1] == generated_loc.shape[1], "generated time and loc does not match"
            
            control_val = 0.0
            if self.args.time_format == 'time_gap':
                generated_time = generated_time- control_val
                generated_time[generated_time < 0.] = 0.
                de_noramized_time = torch.cumsum(generated_time,dim=1)
                gt_de_normalized_time = torch.cumsum(time_gap_batch,dim=1)
                gt_de_normalized_time = gt_de_normalized_time * mask_batch ### this is for gt only, no wonder

                                


            de_normalized_spatial = de_normalized(generated_loc,self.args,self.trainloader.dataset,dim=2)
            
            gt_de_normalized_spatial = de_normalized(event_loc,self.args,self.trainloader.dataset,dim=2,mask = mask_batch)  
            gt_de_normalized_time = gt_de_normalized_time.unsqueeze(dim=-1)
            de_noramized_time = de_noramized_time.unsqueeze(dim=-1)
            generated_time = generated_time.unsqueeze(dim=-1)
            gt_batch_ori = torch.concat((gt_de_normalized_time,gt_de_normalized_spatial),dim=-1)

            generated_batch_ori = torch.concat((generated_time,generated_loc),dim=-1)
            generated_batch = torch.concat((de_noramized_time,de_normalized_spatial),dim=-1)
            
            gt_batch_ori = gt_batch_ori.cpu().detach().numpy()
            generated_batch = generated_batch.cpu().detach().numpy()
            generated_batch_ori = generated_batch_ori.cpu().detach().numpy() #[2,2,3]
            gen_len = gen_len.cpu().detach().numpy()

            if irregular == True:
                pad_generated_batch_ori = np.zeros((generated_batch_ori.shape[0],self.trainloader.dataset.max_length,generated_batch_ori.shape[-1]))
                pad_generated_batch = np.zeros((generated_batch.shape[0],self.trainloader.dataset.max_length,generated_batch.shape[-1]))
                pad_generated_batch_ori[:,:generated_batch_ori.shape[1],:] = generated_batch_ori
                pad_generated_batch[:,:generated_batch.shape[1],:] = generated_batch
                samples_ori.extend(pad_generated_batch_ori)
                samples.extend(pad_generated_batch)
            else:
                samples_ori.extend(generated_batch_ori)
                samples.extend(generated_batch)


            seq_len_all.extend(gen_len)
            gt_samples.append(gt_batch_ori)
        samples = np.array(samples)
        samples_ori = np.array(samples_ori)
        seq_len_all = np.array(seq_len_all)

        if save_constraint:
            constraint = np.array(constraint)
            sample_index_all = np.array(sample_index_all)
            save_pth = os.path.join(self.args.save_final, f'time_constraint_epoch_{epoch}')
            np.savez(save_pth,sample_index = sample_index_all,bound = constraint)

     
        self.save_generated_file(gt_samples,samples_ori, samples,seq_len_all, epoch=epoch)
    

    
    def save_best_model(self,epoch):
        self.logger.info(f'best model saved at epoch {epoch}')
        
        save_pth = os.path.join(self.args.save_final, f'best_model.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            }, save_pth)



    def save_generated_file(self,groundtruth_samples,gen_samples_normalized, gen_samples_denormalized,seq_length, epoch):
        
        
        save_pth = os.path.join(self.args.save_final, f'generated_samples_epoch_{epoch}')
        pickle.dump(groundtruth_samples, open(f'{save_pth}_groundtruth.pkl', 'wb'))
        np.savez(save_pth, gen_samples_normalized=gen_samples_normalized,gen_samples_denormalized= gen_samples_denormalized, seq_length = seq_length)
    
    def get_lr(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    

    def train_epoch_time_log(self):
        print('start training epoch time log' )
        iter_num = 0
        time_loglik_meter = AverageMeter()
        space_loglik_meter = AverageMeter()

        self.best_loss = math.inf

        
        for epoch in range(self.args.epochs):
            self.model.train()
            for batch_idx,batch in enumerate(self.trainloader):
                
                iter_num +=1 
                agent_id_batch, time_batch, time_gap_batch, lng_batch, lat_batch, mask_batch, stop_state_batch, lower_bound_time, upper_bound_time = map(lambda x: cast(x, self.device), batch)
                
                if epoch ==0:
                    self.max_length_batch.append(time_batch.shape[1])
                non_pad_mask = mask_batch.unsqueeze(2)
                event_loc = torch.cat((lat_batch.unsqueeze(dim=2),lng_batch.unsqueeze(dim=2)),dim=-1)
                self.optimizer.zero_grad()
                if self.args.time_decoder == 'constraint' or self.args.time_decoder == 'truncate':
                    time_loglik, time_mse, space_loglik, std_regularization, space_mse = self.model.forward_angle(event_loc,time_gap_batch,non_pad_mask,lower_bound_time, upper_bound_time,ori_time = time_batch)
                else:
                    time_loglik, time_mse, space_loglik, std_regularization, space_mse = self.model.forward_angle(event_loc,time_gap_batch,non_pad_mask)
                loss = time_loglik
           
                if 'spatial_time' in self.args.mode:
                    loss += space_loglik

                

                loss = -loss.mean()
                if self.args.lr_s_std_regularization > 0:
                    loss += self.args.lr_s_std_regularization * std_regularization
                time_mse = time_mse.mean()  
                space_mse = space_mse.mean() 
                loss = loss + time_mse / 10. + space_mse / 10.
                

                loss.backward()
                self.optimizer.step()

                time_log_mean = -time_loglik.mean().item()
                space_log_mean = -space_loglik.mean().item()

                time_loglik_meter.update(time_log_mean)
                space_loglik_meter.update(space_log_mean)
                self.wandb_logger.log("time_ell", time_log_mean, iter_num)
                self.wandb_logger.log("space_ell", space_log_mean, iter_num)
                if 'spatial_predict' in self.args.mode:
                    self.wandb_logger.log("space_mse", space_mse, iter_num)
                if 'time_predict' in self.args.mode:
                    self.wandb_logger.log("time_mse", time_mse, iter_num)

            lr = self.get_lr(self.optims[0])
            self.wandb_logger.log("generator_lr", lr, epoch)
            self.time_loss = time_loglik_meter.avg
            self.space_loss = space_loglik_meter.avg
            self.wandb_logger.log("time_ell_epoch", time_loglik_meter.avg, epoch)
            self.wandb_logger.log("space_ell_epoch", space_loglik_meter.avg, epoch)
            loss_all  = self.time_loss
            if 'spatial_time' in self.args.mode:
                loss_all += self.space_loss

            if loss_all < self.best_loss:
                self.best_loss = loss_all
                self.save_best_model(epoch)
            time_loglik_meter.reset()
            space_loglik_meter.reset()
            self.scheds[0].step()
            if (epoch) % self.args.gen_epoch  ==0 and epoch >0:
                self.generation_step(epoch,irregular=True)
    

    

    
    def save_best_model(self,epoch):
        self.logger.info(f'best model saved at epoch {epoch}')
        
        save_pth = os.path.join(self.args.save_final, f'best_model.pt')

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            }, save_pth)
  
    def get_lr(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    

  
    
    def load_constraint_file(self,method):

        if 'baseline' in self.args.dataset:
            name = 'baseline_readable_constraint'
        elif 'houston' in self.args.dataset:
            name = 'houston_readable_constraint'
        else:
            raise NotImplementedError(f"This {self.args.dataset} has not been implemented yet.")
        datafile = data_file_dict[name]
        self.read_constraint_dataset = torch.from_numpy(np.load(datafile)) #[b=35525,3]

        if method == 'fixed_statistical':
            
            self.count_list = np.zeros(self.max_length)
            max_num = self.read_constraint_dataset.shape[0]
            for i in range(max_num):
        
                self.count_list[int(self.read_constraint_dataset[i,0])] +=1
            self.count_list  = self.count_list / self.count_list.sum()
        elif method == 'time_statistical' or 'beam_search':
            #name = 'sample_loc_time_statis_constraint' 
            if 'baseline' in self.args.dataset:
                name = 'baseline_sample_loc_time_statis_constraint' 
            elif 'houston' in self.args.dataset:
                name = 'houston_readable_constraint'
            else:
                raise NotImplementedError(f"This {self.args.dataset} has not been implemented yet.")
            datafile = data_file_dict[name]
            self.time_constraint = np.load(datafile) #[b,max_length]
            row_sums = self.time_constraint.sum(axis=1)[:, None]
            self.time_constraint = self.time_constraint / row_sums # get probability 

        else:
            raise NotImplementedError(f"This {method} has not been implemented yet.")

    def sample_constraint_loc(self,method,batch_size,batch_id):
        if method == 'fixed_statistical':

            sampled_indices = np.random.choice(a=len(self.count_list), size=batch_size, p=self.count_list)
            sampled_indices = torch.from_numpy(sampled_indices)
        
        elif method == 'time_statistical':
             
            
            sampled_indices = np.zeros(batch_size, dtype=int)
            max_length = self.time_constraint.shape[1]


            for i in range(batch_size):
                cur_index = self.args.batch_size * batch_id + i
                sampled_indices[i] = np.random.choice(max_length, size=1, p=self.time_constraint[cur_index])
            sampled_indices = torch.from_numpy(sampled_indices)

        elif method == 'beam_search':
            
            sampled_indices = np.zeros((batch_size,self.args.index_k), dtype=int)
            max_length = self.time_constraint.shape[1]
            for i in range(batch_size):
                cur_index = self.args.batch_size * batch_id + i
                sampled_indices[i] = np.random.choice(max_length, size=self.args.index_k, p=self.time_constraint[cur_index])
            sampled_indices = torch.from_numpy(sampled_indices) #[b,k]
 

            
        else:
            raise NotImplementedError(f"This {method} has not been implemented yet.")
        
        return sampled_indices
