import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.dpp.models.log_norm_mix import LogNormMix
from models.dpp.models.log_norm_mix_constraint import ConstraintLogNormMix
from models.st_transformer.Models import get_square_subsequent_mask, Predictor
import models.st_transformer.losses as losses
from models.dpp.models.cond_gmm import ConditionalGMM

import torch.distributions as D

from models.encoder.location_encoder import LocationEncoder



class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data, non_pad_mask): #
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        out = self.projection(out)
        return out




class AngleEncoder(nn.Module):
    
    """ A encoder model with self attention mechanism. """
    ## also with time

    def __init__(
            self,
            args,
            device,
            d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout):
        super().__init__()

        self.args = args
       
        self.d_model = d_model
        self.loc_dim = 2
        self.device = device 

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=device)
        
        
        if 'gaussian_noise'in self.args.enc_type:
            self.gps_encoder = LocationEncoder(embed_dim=d_model, sigma=[2**0, 2**4, 2**8])
        
        if 'l_default' in self.args.enc_type:
            self.event_emb = nn.Sequential(
            nn.Linear(self.loc_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
        



        self.angle_emb = nn.Sequential(
          nn.Linear(1, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
        )

        self.distance_emb = nn.Sequential(
          nn.Linear(1, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
        )






        ### the transformer encoder for all features 
        encoder_layers = TransformerEncoderLayer(self.d_model*2, n_head, 64, 0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)


        loc_encoder_layers = TransformerEncoderLayer(self.d_model, n_head, 64, 0.1)
        self.transformer_loc = TransformerEncoder(loc_encoder_layers, 3)

        time_encoder_layers = TransformerEncoderLayer(self.d_model, n_head, 64, 0.1)
        self.transformer_time = TransformerEncoder(time_encoder_layers, 3)
        

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result 
        #return result * non_pad_mask
    

    def loc_enc(self, event_loc):

        '''
        give a batch of location encoding
        '''

        if 'l_default' in self.args.enc_type:
            return self.event_emb(event_loc)
        elif 'gaussian_noise'in self.args.enc_type:

            return self.gps_encoder(event_loc)
        else:
            raise NotImplementedError
    

        
    def forward(self, event_loc, event_time, non_pad_mask):
                # event_loc: [b,s,2] event_time[b,s]
        """ Encode event sequences via masked self-attention. """

        square_mask = get_square_subsequent_mask(event_time) #[seq_len,seq_len]
        square_mask = square_mask.to(self.device)
        tem_enc = self.temporal_enc(event_time, non_pad_mask) # [batch, length, hdim=256]
        
        spatial_enc = self.loc_enc(event_loc) # [batch, length, hdim=256]


        encode_all = torch.cat((tem_enc, spatial_enc),dim=-1)   # [batch, length, hdim=512]
        encoder_output = self.transformer_encoder(encode_all.transpose(1, 0),mask=square_mask,src_key_padding_mask=torch.squeeze(non_pad_mask, -1))

        encoder_time = self.transformer_time(tem_enc.transpose(1, 0),mask=square_mask,src_key_padding_mask=torch.squeeze(non_pad_mask, -1))#[seqlen,batch,dim=256]
        encoder_loc = self.transformer_loc(spatial_enc.transpose(1, 0))

        encode_transform = torch.cat((encoder_output, encoder_loc, encoder_time),dim=-1)
        return encode_transform.transpose(1, 0), (tem_enc,spatial_enc) #[ batch,seq_len, hdim= 512+256+256]
 

class ST_Encoder_lognorm(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,args,
            device,
            d_model=256, d_rnn=128, d_inner=1024,
            n_layers=3, n_head=4, d_k=64, d_v=64, dropout=0.1,
            mean_log_inter_time=0. , 
            std_log_inter_time =1.
            ):
        super().__init__()
        self.args = args
        self.loc_dim =2
        self.device = device
        num_types  =10
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        self.context_init = nn.Parameter(torch.zeros(d_model))

        self.n_head = n_head

        self.encoder = AngleEncoder( # 256*3
                self.args,
                self.device,
                d_model=d_model,
                d_inner=d_inner,
                n_layers=n_layers,
                n_head=n_head,
                d_k=d_k,
                d_v=d_v,
                dropout=dropout,
        )

 
  
        if 'encode_default' in self.args.enc_type:
            dimension = d_model *2
            #print(dimension)
        elif 'encode_embed'  in self.args.enc_type:
            dimension = d_model *3
            #print(dimension)
        elif 'encode_att' in self.args.enc_type:
            dimension = d_model *3
            #print(dimension)
        else:
            raise NotImplementedError
        
        if self.args.model_choice != 'angle':

           
            if self.args.spatial_model == 'condGMM':
                if 'encode_default' in self.args.enc_type:
                    dimension = d_model *2
                elif 'encode_embed'  in self.args.enc_type:
                    dimension = d_model *3
                elif 'encode_att' in self.args.enc_type:
                    dimension = d_model *3
                else:
                    raise NotImplementedError
                self.spatial_distribution = ConditionalGMM(dim=2, hidden_dims=[64, 64, 64], aux_dim=dimension, n_mixtures=self.args.spatial_num_mix_components, actfn="softplus")


            else:
                raise NotImplementedError
        

        self.time_predictor = Predictor(dimension, 1)
        self.spatial_predictor = Predictor(dimension, 2)
        

        self.d_model = d_model
        self.num_types = num_types

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, num_types)

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))


        if self.args.time_decoder != 'none':
            self.log_norm =  ConstraintLogNormMix(args=self.args, context_size=dimension, mean_log_inter_time=mean_log_inter_time, std_log_inter_time=std_log_inter_time,num_mix_components=self.args.num_mix_components)
        else:
            self.log_norm =  LogNormMix(context_size=dimension, mean_log_inter_time=mean_log_inter_time, std_log_inter_time=std_log_inter_time,num_mix_components=self.args.num_mix_components)





    def log_time(self, event_time):

        event_time = torch.log(event_time + 1e-8)
        event_time = (event_time - self.mean_log_inter_time) / self.std_log_inter_time
        return event_time
    


    def get_encoder_embedding(self, event_type, event_time,non_pad_mask):

        event_time = self.log_time(event_time) #[batch_size,seq_length]
        enc_output = self.encoder(event_type, event_time, non_pad_mask)
        
        return enc_output
    

    def forward_angle(self, spatial_locations, event_time,non_pad_mask,lower_bound =None,upper_bound =None,ori_time=None):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """

        #non_pad_mask =non_pad_mask.detach()
        min_val = 0.002
   
        
        
        enc_output_all, feature_embed = self.get_encoder_embedding(spatial_locations, event_time,non_pad_mask)[:self.args.d_model] #[seq_len, batch, hdim]
        
        tem_enc,spatial_enc = feature_embed
        tem_enc = tem_enc[:,:-1,:]
        spatial_enc = spatial_enc[:,:-1,:]
   

        enc_output, enc_output_loc, enc_output_time = enc_output_all[:,:,:self.args.d_model*2], enc_output_all[:,:,self.args.d_model*2:3*self.args.d_model],enc_output_all[:,:,3*self.args.d_model:]


        
        # for the sake of convinence , just shift1 
        enc_output = enc_output[:,:-1,:] #[batch, seq_len-1, hdim]
        enc_output_loc = enc_output_loc[:,:-1,:]
        enc_output_time =enc_output_time[:,:-1,:]


        event_time = event_time[:,1:] #[batch, seq_len-1]
        non_pad_mask = non_pad_mask[:,1:] #[batch, seq_len-1, 1]


        time_encode = enc_output
        dis_encode = enc_output
        if 'encode_embed' in self.args.enc_type:
            dis_encode = torch.cat((enc_output,enc_output_loc),dim=-1)
            time_encode = torch.cat((enc_output,enc_output_time),dim=-1)
        elif 'encode_att' in self.args.enc_type:
            dis_encode = torch.cat((enc_output,spatial_enc),dim=-1)
            time_encode = torch.cat((enc_output,tem_enc),dim=-1)
        
        #print(f'encode {dis_encode.shape} {time_encode.shape}')

        
        if self.args.model_choice == 'time_log':
            spatial_log_prob, std_regularization = self.spatial_distribution.logprob(event_time, spatial_locations[:,1:],input_mask=non_pad_mask.squeeze(-1), aux_state=dis_encode) # [batch, seq_len-1]
        else:
            raise NotImplementedError
        if self.args.time_decoder != 'none':
            inter_time_dist = self.log_norm.get_inter_time_dist(time_encode,lower_bound[:,1:],upper_bound[:,1:]) 
        else:
            inter_time_dist = self.log_norm.get_inter_time_dist(time_encode)
            

        inter_times = event_time.clamp(1e-8 +min_val) # (batch_size, seq_len-1)
        

        log_p = inter_time_dist.log_prob(inter_times)  # (batch_size, seq_len-1)
        log_p = log_p * non_pad_mask.clone().squeeze(-1)  # (batch_size, seq_len-1)

        

        if 'time_predict' in self.args.mode:
            time_prediction = self.time_predictor(time_encode,non_pad_mask) #[batch, seq_len-1, 1]
            log_batch_time = self.log_time(event_time) #[batch, seq_len-1]
            temporal = losses.time_log_loss(time_prediction, log_batch_time,non_pad_mask.squeeze_(-1),loss_type=self.args.loss_type) 
        else:
            temporal = torch.zeros_like((log_p)).to(self.device) 


        if 'spatial_predict' in self.args.mode:
            space_prediction = self.spatial_predictor(dis_encode,non_pad_mask) #[batch, seq_len-1, 1]
            
            spatial_mse = losses.spatial_loss(space_prediction, spatial_locations[:,1:],non_pad_mask.squeeze(-1))
        else:
            spatial_mse = torch.zeros_like((log_p)).to(self.device) 
        
        

        
        return log_p.sum(-1) , temporal.sum(-1), spatial_log_prob.sum(-1) ,std_regularization.sum(), spatial_mse.sum(-1)
    
        
           
   
    def sample(self,num_samples,start_batch = None,max_len=24):
    
        mask = torch.ones(num_samples,max_len,1).to(self.device)
        if start_batch is not None:
            max_len -= 1

    
        
        event_loc,event_time = start_batch #[b,s,2],[b,s]
         #[batch,seq_len,1]
        
        seqlen = torch.zeros(num_samples, device=self.device, dtype=torch.long)

        
        
        generated = False
        for i in range(max_len):

            
            enc_output = self.get_encoder_embedding(event_loc,event_time, mask[:,:i+1,:])
 
            inter_time_dist = self.log_norm.get_inter_time_dist(enc_output[:,-1,:]) 
            next_inter_times = inter_time_dist.sample() # (batch_size)
            next_inter_times = next_inter_times.unsqueeze(-1) # (batch_size, 1)
            

            generated_loc = self.spatial_distribution.sample_spatial_single(1,event_time[:,-1],event_loc, mask[:,:i+1,:],enc_output[:,-1,:])
            event_time = torch.cat([event_time, next_inter_times], dim=1)
            
            event_loc = torch.concat([event_loc, generated_loc],dim=1)


        return (event_loc, event_time, seqlen)
    



    def sample_constraint(self,num_samples,start_batch = None,max_len=24,lower_bound = None,upper_bound=None,min_val = 0.002,max_val = 18.2):
    
        mask = torch.ones(num_samples,max_len,1).to(self.device)
        if start_batch is not None:
            max_len -= 1
            lower_bound = lower_bound[:,1:]
            upper_bound = upper_bound[:,1:]
        
        
        

        lower_bound_update = torch.ones((num_samples,max_len),device =self.device ) * min_val
        upper_bound_update = torch.ones((num_samples,max_len),device =self.device) * max_val
        lower_bound_update[:,0] = lower_bound[:,0]
        upper_bound_update[:,0] = upper_bound[:,0]

        
        event_loc,event_time = start_batch #[b,s,2],[b,s]
        
        seqlen = torch.ones(num_samples, device=self.device, dtype=torch.long) *-1
        cumsum = torch.zeros((num_samples,max_len+1), device=self.device, dtype=torch.float)
        cumsum[:,0] = event_time.squeeze(-1)
               
        
        generated = False
        for i in range(max_len):

            
            enc_output = self.get_encoder_embedding(event_loc,event_time, mask[:,:i+1,:])
            inter_time_dist = self.log_norm.get_inter_time_dist(enc_output[:,-1,:],lower_bound_update[:,i],upper_bound_update[:,i])
            next_inter_times = inter_time_dist.sample() # (batch_size)
            cumsum[:,i+1] = cumsum[:,i] + next_inter_times


            next_inter_times = next_inter_times.unsqueeze(-1) # (batch_size, 1)
            
            
            

            if self.args.spatial_model =='attncnf':
                #event_time = torch.cat([event_time, next_inter_times], dim=1)
                event_time_cmu = torch.cumsum(event_time,dim=1)
                generated_loc = self.spatial_distribution.sample_spatial_single(event_time_cmu,event_loc, mask[:,:i+1,:],enc_output)
                event_time = torch.cat([event_time, next_inter_times], dim=1)
            else:
    
                generated_loc = self.spatial_distribution.sample_spatial_single(1,event_time[:,-1],event_loc, mask[:,:i+1,:],enc_output[:,-1,:])
                event_time = torch.cat([event_time, next_inter_times], dim=1)

            
            event_loc = torch.concat([event_loc, generated_loc],dim=1)
        return (event_loc, event_time, seqlen)


    #after we done one add more 

   
    def update_time_bound(self,i,lower_bound_update,upper_bound_update,generated_time,readable_constraint=None,constraint_loc = None,min_val = 0.002,max_val = 18.2):
        lower_value = readable_constraint[:,1]
        upper_value = readable_constraint[:,2]

        
        prev_generate_time = generated_time[:,-1]
        cumsum = torch.cumsum(generated_time, dim=1)
        mask =  (i +1)  < constraint_loc
        indices = mask.nonzero(as_tuple=True)
        row_indices = indices[0] #[b]
        
        if i ==0 :
            upper_bound_update[row_indices,i] = (lower_value[row_indices] - prev_generate_time[row_indices])
        else:
            upper_bound_update[row_indices,i] = (upper_bound_update[row_indices,i-1] - prev_generate_time[row_indices])

        mask =  (i +1)  == constraint_loc
        indices = mask.nonzero(as_tuple=True)
        row_indices = indices[0] #[b]

        lower_bound_update[row_indices ,i] = (lower_value[row_indices] - cumsum[row_indices,-1]) ###
        upper_bound_update[row_indices ,i] = (upper_value[row_indices]- cumsum[row_indices,-1])


        return lower_bound_update, upper_bound_update
    

    def update_time_bound_beam(self,i,lower_bound_update,upper_bound_update,generated_time,readable_constraint=None,constraint_loc = None,min_val = 0.002,max_val = 18.2):
        '''
        generated_time [b,s,k]

        '''

        
        #i = i+1
        beam_width = generated_time.shape[-1]
        lower_value = readable_constraint[:,1]
        upper_value = readable_constraint[:,2]

        
        prev_generate_time = generated_time[:,-1,:] #[b,k]  
        cumsum = torch.cumsum(generated_time, dim=1)


        for j in range(beam_width):
            mask =  (i +1)  < constraint_loc
            indices = mask.nonzero(as_tuple=True)
            row_indices = indices[0]


            offset = (min_val *(constraint_loc[row_indices] - i).to(upper_bound_update))
            
            if i ==0 :
                
                upper_bound_update[row_indices,i,j] =  torch.max((lower_value[row_indices] - prev_generate_time[row_indices,j]) - offset,(lower_value[row_indices] - prev_generate_time[row_indices,j]) *1.0 /(constraint_loc[row_indices] - i)).to(upper_bound_update.dtype)
                # upper_zero = (lower_value[row_indices] - prev_generate_time[row_indices,j]) / (constraint_loc[row_indices] - i).to(upper_bound_update)
                # upper_bound_update[row_indices,i,j] = max(upper_bound_update[row_indices,i,j],upper_zero)
            else:
                upper_bound_update[row_indices,i,j] = torch.max((lower_value[row_indices] - cumsum[row_indices,i,j]) - offset,(lower_value[row_indices] - cumsum[row_indices,i,j])*1.0 /(constraint_loc[row_indices] - i)).to(upper_bound_update.dtype)
                
            

            mask =  (i +1)  == constraint_loc
            indices = mask.nonzero(as_tuple=True)
            row_indices = indices[0] #[b]

            lower_bound_update[row_indices ,i,j] = (lower_value[row_indices] - cumsum[row_indices,-1,j]) ###
            upper_bound_update[row_indices ,i,j] = (upper_value[row_indices]- cumsum[row_indices,-1,j])



        return lower_bound_update, upper_bound_update
    

    def update_unconstraint_sampling(self,i,k,sampled_time,readable_constraint,lower_bound_update,upper_bound_update):
        '''
        sampled_time [b]
        lower_bound_update [b,s,k]
        '''
        

        mask =  ( ((i+1) <= readable_constraint[:,0]) & (sampled_time  >=upper_bound_update[:,i,k] *0.9) )
        indices = mask.nonzero(as_tuple=True)
        row_indices = indices[0]
        #print(row_indices)

        sampled_time[row_indices] = upper_bound_update[row_indices,i,k] *0.5


        mask =   (sampled_time  < lower_bound_update[:,i,k])
        indices = mask.nonzero(as_tuple=True)
        row_indices = indices[0]
        sampled_time[row_indices] = lower_bound_update[row_indices,i,k]

        return sampled_time
        









    
      
    def sample_adaptive_constraint_wo_constraint(self,num_samples,start_batch = None,max_len=24,readable_constraint=None,constraint_loc = None,min_val = 0.002, max_val = 18.2):
        ### this is the original version of sampling without constraint
        #print('beam_search no constriant')

        
        mask = torch.ones(num_samples,max_len,1).to(self.device)
        if start_batch is not None:
            max_len -= 1

        
        beam_width = self.args.beam_search_k

        lower_bound_update = torch.ones((num_samples,max_len,beam_width),device =self.device ) * min_val
        upper_bound_update = torch.ones((num_samples,max_len,beam_width),device =self.device) * max_val
        probabilities = torch.zeros((num_samples,max_len,3,beam_width),device =self.device) # prob [b,s,3,k]
        
       

    
        event_loc,event_time = start_batch #[b,s,2],[b,s]

        event_loc = event_loc[:,:,:,None].repeat(1,1,1, beam_width) #[b,s,2,k]
        event_time = event_time[:,:,None].repeat(1,1, beam_width) # [b,s,k] even though it inplace change event_time


        seqlen = torch.ones(num_samples, device=self.device, dtype=torch.long) *-1

        readable_constraint = readable_constraint.to(event_time)
        for i in range(max_len):

            lower_bound_update,upper_bound_update = self.update_time_bound_beam(i,lower_bound_update,upper_bound_update,event_time,readable_constraint,constraint_loc) #[b,s,k]
            expand_probabilities = torch.zeros((num_samples,3,beam_width*beam_width),device =self.device) #[b,3,k*k]
            expand_real_value = torch.zeros((num_samples,3,beam_width*beam_width),device =self.device) #[b,3,k*k]
            #print(f' EEE {expand_real_value.shape}')
            for j in range(beam_width):
                

                enc_output_all,feature_embed = self.get_encoder_embedding(event_loc[:,:,:,j],event_time[:,:,j], mask[:,:i+1,:]) #[seq_len, batch, hdim]
                tem_enc,spatial_enc = feature_embed
                enc_output, enc_output_loc, enc_output_time = enc_output_all[:,:,:self.args.d_model*2], enc_output_all[:,:,self.args.d_model*2:3*self.args.d_model],enc_output_all[:,:,3*self.args.d_model:]
                enc_output = enc_output[:,-1,:]
                time_encode = enc_output
                dis_encode = enc_output
                enc_output_loc = enc_output_loc[:,-1,:]
                enc_output_time = enc_output_time[:,-1,:]
                tem_enc = tem_enc[:,-1,:]
                spatial_enc = spatial_enc[:,-1,:]
                if 'encode_embed' in self.args.enc_type:

                    dis_encode = torch.cat((enc_output,enc_output_loc),dim=-1)
                    time_encode = torch.cat((enc_output,enc_output_time),dim=-1)
                elif 'encode_att' in self.args.enc_type:
                    dis_encode = torch.cat((enc_output,spatial_enc),dim=-1)
                    time_encode = torch.cat((enc_output,tem_enc),dim=-1)
        
  
                if self.args.time_decoder != 'none':
                    inter_time_dist = self.log_norm.get_inter_time_dist(time_encode,lower_bound_update[:,i,j],upper_bound_update[:,i,j]) 
                else:
                    inter_time_dist = self.log_norm.get_inter_time_dist(time_encode)




                for k in range(beam_width):
                
                    next_inter_times = inter_time_dist.sample() # (batch_size)
                    if self.args.time_decoder ==  'none':
                        next_inter_times = self.update_unconstraint_sampling(i,j,next_inter_times,readable_constraint,lower_bound_update,upper_bound_update)
                    next_inter_times = next_inter_times.clamp(1e-8)
                    
                    sampled_loc = self.spatial_distribution.sample_spatial_single(event_time.shape[0],event_time[:,-1,k],event_loc[:,-1,:,k],dis_encode) #[N,2]
                
                    time_log_prob = inter_time_dist.log_prob(next_inter_times)
                    spatial_log_prob, std_regularization = self.spatial_distribution.logprob(event_time[:,-1,k], sampled_loc,input_mask=mask[:,:i+1,:], aux_state=dis_encode) # [batch, seq_len-1]
                    

                    expand_probabilities[:,0,j*beam_width+k] = time_log_prob
                    expand_probabilities[:,1:,j*beam_width+k] = spatial_log_prob


                    expand_real_value[:,0,j*beam_width+k] = next_inter_times

                    
                    if self.args.model_choice == 'time_log':
                        expand_real_value[:,1,j*beam_width+k] = sampled_loc[:,0]
                        expand_real_value[:,2,j*beam_width+k] = sampled_loc[:,1]
                    else:
                        raise NotImplementedError



            if i==0 :
                prev_probabilites = probabilities[:,i,:,:] #[b,k,3]
            else:
                prev_probabilites = probabilities[:,i-1,:,:]



            
            prev_probabilites = prev_probabilites[:,:,:,None].repeat(1,1,1,beam_width).reshape(num_samples, 3, -1)#[b,3,k*k=9]
            
            final_prob_detail = expand_probabilities

            final_prob = final_prob_detail.sum(dim=1) #[b,k*k=9]
            final_prob, idx = final_prob.topk(k = beam_width, axis = -1) #idx[b,k=3]



            best_candidates = (idx // beam_width).long() #[b,k=3]
            best_candidates = best_candidates[:,None,:].repeat(1,event_time.shape[1],1) #[b,s,k=3]

                

            event_time = torch.gather(event_time, -1, best_candidates.long() )
            best_candidates = best_candidates[:,:,None,:].tile(1,1,2,1) #  #[b,s,2,k]
            event_loc = torch.gather(event_loc, -1, best_candidates.long() )


            idx = idx[:,None,:].tile(1,3,1)
            expand_real_value = torch.gather(expand_real_value,-1,idx) #[b,3,k=3,]

            probabilities[:,i,:,:] = torch.gather(final_prob_detail,-1,idx) #[b,3,k=3]


            event_time = torch.cat([event_time, expand_real_value[:,None,0,:]], dim=1)
            event_loc = torch.concat([event_loc, expand_real_value[:,None,1:,:]],dim=1)

        seq_prob,idx = probabilities[:,-1,:,:].sum(1).topk(k = 1, axis = -1)
        idx = idx[:,None,:].repeat(1,event_time.shape[1],1) #[b,s,1]

        final_seq_time = torch.gather(event_time,-1,idx)
        idx = idx[:,:,None,:].tile(1,1,2,1)
        final_seq_loc = torch.gather(event_loc,-1,idx)
        
        

        

        return (final_seq_loc.squeeze(),final_seq_time.squeeze(), seqlen),(lower_bound_update,upper_bound_update)  #lower_bound_update,upper_bound_update



    def sample_adaptive_constraint(self,num_samples,start_batch = None,max_len=24,readable_constraint=None,constraint_loc = None):
        ### update geo 
        #print('beam_search')
        
        now = datetime.now()
        mask = torch.ones(num_samples,max_len,1).to(self.device)
        if start_batch is not None:
            max_len -= 1

        min_val = 1e-8
        max_val = 18.2
        beam_width = self.args.beam_search_k

        lower_bound_update = torch.ones((num_samples,max_len,beam_width),device =self.device ) * min_val
        upper_bound_update = torch.ones((num_samples,max_len,beam_width),device =self.device) * max_val
        probabilities = torch.zeros((num_samples,max_len,3,beam_width),device =self.device) # prob [b,s,3,k]
        
       

    
        event_loc,event_time = start_batch #[b,s,2],[b,s]
        event_loc = event_loc[:,:,:,None].repeat(1,1,1, beam_width) #[b,s,2,k]
        event_time = event_time[:,:,None].repeat(1,1, beam_width) # [b,s,k] even though it inplace change event_time

        
        seqlen = torch.ones(num_samples, device=self.device, dtype=torch.long) *-1

        readable_constraint = readable_constraint.to(event_time)
        for i in range(max_len):
            

            lower_bound_update,upper_bound_update = self.update_time_bound_beam(i,lower_bound_update,upper_bound_update,event_time,readable_constraint,constraint_loc) #[b,s,k]


        
            expand_probabilities = torch.zeros((num_samples,3,beam_width*beam_width),device =self.device) #[b,3,k*k]
            expand_real_value = torch.zeros((num_samples,3,beam_width*beam_width),device =self.device) #[b,3,k*k]
            #print(f' EEE {expand_real_value.shape}')
            for j in range(beam_width):

                enc_output_all,feature_embed = self.get_encoder_embedding(event_loc[:,:,:,j],event_time[:,:,j], mask[:,:i+1,:]) #[seq_len, batch, hdim]
                tem_enc,spatial_enc = feature_embed
                enc_output, enc_output_loc, enc_output_time = enc_output_all[:,:,:self.args.d_model*2], enc_output_all[:,:,self.args.d_model*2:3*self.args.d_model],enc_output_all[:,:,3*self.args.d_model:]
                enc_output = enc_output[:,-1,:]
                time_encode = enc_output
                dis_encode = enc_output
                enc_output_loc = enc_output_loc[:,-1,:]
                enc_output_time = enc_output_time[:,-1,:]
                tem_enc = tem_enc[:,-1,:]
                spatial_enc = spatial_enc[:,-1,:]
                if 'encode_embed' in self.args.enc_type:
                
                    dis_encode = torch.cat((enc_output,enc_output_loc),dim=-1)
                    time_encode = torch.cat((enc_output,enc_output_time),dim=-1)
                elif 'encode_att' in self.args.enc_type:
                    dis_encode = torch.cat((enc_output,spatial_enc),dim=-1)
                    time_encode = torch.cat((enc_output,tem_enc),dim=-1)
                
                
                if self.args.model_choice == 'angle':
                    distance_dist = self.distance_model.get_distance_dist(dis_encode)
                    angle_dist = self.angle_model.get_distribution(dis_encode.unsqueeze(1))

                inter_time_dist = self.log_norm.get_inter_time_dist(time_encode,lower_bound_update[:,i,j],upper_bound_update[:,i,j]) 
                

                for k in range(beam_width):
                
                    next_inter_times = inter_time_dist.sample() # (batch_size)

                    next_inter_times = torch.clamp(next_inter_times, min=1e-8+lower_bound_update[:,i,j], max=upper_bound_update[:,i,j]-1e-8)
                    
                    if self.args.model_choice == 'angle':
                        distance  = distance_dist.sample()
                        angle = angle_dist.sample()
                    elif self.args.model_choice == 'time_log':             
                        #print(f' {event_time.shape} {event_loc.shape} {dis_encode.shape}')
                        sampled_lat_long = self.spatial_distribution.sample_spatial_single(event_time.shape[0],event_time[:,-1,k],event_loc[:,-1,:,k],dis_encode) #[N,2]

                    


                    inf_mask = torch.isnan(next_inter_times)
                    inf_indices = torch.nonzero(inf_mask)
                    is_empty = inf_indices.numel() == 0
                    if not is_empty:
                        #print(inf_indices)
                        total = inf_indices.shape[0]
                        random_values = torch.rand(total).to(lower_bound_update)
                        
                        low = lower_bound_update[inf_indices,i,j]
                        upper = upper_bound_update[inf_indices,i,j]


                        if low.dim() > 1:
                            random_values = random_values.unsqueeze(-1)
                        samples = low + (upper - low) * random_values


                        next_inter_times[inf_indices] = samples
                        


                    inf_mask = torch.isinf(next_inter_times)
                    inf_indices = torch.nonzero(inf_mask)
                    is_empty = inf_indices.numel() == 0
                    if not is_empty:

                        
                        total = inf_indices.shape[0]
                        random_values = torch.rand(total)
                        

                        low = lower_bound_update[inf_indices,i,j]
                        upper = upper_bound_update[inf_indices,i,j]
                        if low.dim() > 1:
                            random_values = random_values.unsqueeze(-1)
                        samples = low + (upper - low) * random_values

                        next_inter_times[inf_indices] = samples



                    time_log_prob = inter_time_dist.log_prob(next_inter_times)

                    if self.args.model_choice == 'angle':
                        distance_log_prob = distance_dist.log_prob(distance) # [batch,seq]
                        angle_log_prob = angle_dist.log_prob(angle)
                    elif self.args.model_choice == 'time_log':
                        
            
                        loc_distribution, std_regularization = self.spatial_distribution.logprob(next_inter_times.unsqueeze(-1), sampled_lat_long,aux_state=dis_encode.unsqueeze(1))


                    expand_probabilities[:,0,j*beam_width+k] = time_log_prob
                    expand_probabilities[:,1,j*beam_width+k] = distance_log_prob
                    expand_probabilities[:,2,j*beam_width+k] = angle_log_prob.squeeze()

          


                    expand_real_value[:,0,j*beam_width+k] = next_inter_times

                    if self.args.model_choice == 'time_log':
                        expand_real_value[:,1,j*beam_width+k] = sampled_lat_long[:,0]
                        expand_real_value[:,2,j*beam_width+k] = sampled_lat_long[:,1]
  

            final_prob_detail = expand_probabilities
            final_prob = final_prob_detail.sum(dim=1) #[b,k*k=9]
            final_prob, idx = final_prob.topk(k = beam_width, axis = -1) #idx[b,k=3]
            #idx = torch.randint(low=0, high=max(beam_width*beam_width,0), size=(num_samples, beam_width))

            best_candidates = (idx // beam_width).long() #[b,k=3]
            best_candidates = best_candidates[:,None,:].repeat(1,event_time.shape[1],1) #[b,s,k=3]

                
            ### this is choose the best candidate for past traj
            event_time = torch.gather(event_time, -1, best_candidates.long() )
            best_candidates = best_candidates[:,:,None,:].tile(1,1,2,1) #  #[b,s,2,k]
            event_loc = torch.gather(event_loc, -1, best_candidates.long() )
            
            #[b,k=3] ->[b,3,k=3]



            

            idx = idx[:,None,:].tile(1,3,1)
            expand_real_value = torch.gather(expand_real_value,-1,idx) #[b,3,k=3,]

            probabilities[:,i,:,:] = torch.gather(final_prob_detail,-1,idx) #[b,3,k=3]


            event_time = torch.cat([event_time, expand_real_value[:,None,0,:]], dim=1)
            event_loc = torch.concat([event_loc, expand_real_value[:,None,1:,:]],dim=1)



            
            
        
        seq_prob,idx = probabilities[:,-1,:,:].sum(1).topk(k = 1, axis = -1)

        idx = idx[:,None,:].repeat(1,event_time.shape[1],1) #[b,s,1]

        

        final_seq_time = torch.gather(event_time,-1,idx)
        
        
        ret_prob = probabilities[:,:,:,:].sum(2)
        ret_prob = torch.gather(ret_prob,-1,idx[:,:-1,:])
        ret_prob = torch.mean(ret_prob.squeeze(),axis=1)
        
        idx = idx[:,:,None,:].tile(1,1,2,1)
        final_seq_loc = torch.gather(event_loc,-1,idx)
        
        return (final_seq_loc.squeeze(),final_seq_time.squeeze(), seqlen),(lower_bound_update,upper_bound_update),ret_prob  #lower_bound_update,upper_bound_update



    


    @torch.no_grad()
    def sample_loc_only(self,num_samples,start_batch = None,max_len=24,generated_time=None):
    
        mask = torch.ones(num_samples,max_len,1).to(self.device)
        if start_batch is not None:
            max_len -= 1
        
        event_loc,event_time = start_batch #[b,s,2],[b,s]
        

        
        seqlen = torch.ones(num_samples, device=self.device, dtype=torch.long) *-1

               
        for i in range(max_len):

            
            enc_output_all = self.get_encoder_embedding(event_loc,event_time, mask[:,:i+1,:]) #[seq_len, batch, hdim]
            enc_output, enc_output_dis, enc_output_angle = enc_output_all[:,:,:self.args.d_model], enc_output_all[:,:,self.args.d_model:2*self.args.d_model],enc_output_all[:,:,2*self.args.d_model:]
            generated_loc = self.spatial_distribution.sample_spatial_single(1,event_time[:,-1],event_loc, mask[:,:i+1,:],enc_output[:,-1,:])
            
        
            event_time = torch.cat([event_time, generated_time[:,i+1].unsqueeze(-1)], dim=1) # +1 for compensate start batch
            event_loc = torch.concat([event_loc, generated_loc],dim=1)
            
        return (event_loc, event_time, seqlen)