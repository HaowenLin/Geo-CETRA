import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalGMM(nn.Module):

   

    def __init__(self, dim=2, hidden_dims=[64, 64, 64], aux_dim=0, n_mixtures=5, actfn="softplus"):
        super().__init__()
        assert aux_dim, "ConditionalGMM requires aux_dim > 0"
        self.dim = dim
        self.n_mixtures = n_mixtures
        #self.aux_dim = aux_dim * 2  # Since SharedHiddenStateSpatiotemporalModel splits the hidden state.
        self.aux_dim = aux_dim
        #self.gmm_params = mlp(aux_dim * 2, hidden_dims, out_dim=dim * n_mixtures * 3, actfn=actfn)
        self.gmm_params = mlp(aux_dim, hidden_dims, out_dim=dim * n_mixtures * 3, actfn=actfn)

    def logprob(self, event_times, spatial_locations, input_mask=None, aux_state=None):
        return self._cond_logliks(event_times, spatial_locations, input_mask, aux_state)

    def _cond_logliks(self, event_times, spatial_locations, input_mask=None, aux_state=None):
        """
        Args:
            event_times: (N, T)
            spatial_locations: (N, T, D)
            input_mask: (N, T) or None
            aux_state: (N, T, D_a)

        Returns:
            A tensor of shape (N, T) containing the conditional log probabilities.
        """

        if input_mask is None:
            input_mask = torch.ones_like(event_times)

        N, T = event_times.shape

        #print(f' test {event_times.shape} {spatial_locations.shape}  {aux_state.shape}')
        #aux_state = aux_state[:, :, -self.aux_dim:].reshape(N * T, self.aux_dim)
        aux_state = aux_state.reshape(N * T, self.aux_dim)
        params = self.gmm_params(aux_state) # [N * T, 20]
        logpx,std_regularization = gmm_loglik(spatial_locations, params)
        logpx = logpx.sum(-1)  # (N, T)
        return torch.where(input_mask.bool(), logpx, torch.zeros_like(logpx)), torch.where(input_mask.bool(), std_regularization, torch.zeros_like(std_regularization))

    def plot_spatial_graph(self, event_times, spatial_locations, input_mask=None, aux_state=None):
        

        if input_mask is None:
            input_mask = torch.ones_like(event_times)
        

        N, T = event_times.shape

        aux_state = aux_state[:, :, -self.aux_dim:].reshape(N * T, self.aux_dim)
        params = self.gmm_params(aux_state) # [N * T, 20]
        params = params.reshape(*spatial_locations.shape, 3, -1)
        mix_logits, means, logstds = params[..., 0, :], params[..., 1, :], params[..., 2, :] #[batch,18,2,5]
        
        #input_mask.expand_as(means)
        input_mask = input_mask[:,:,None,None]
        input_mask = input_mask.repeat(1,1,*means.shape[2:])
        #.repeat(1, *means.shape[1:])
       
        return torch.where(input_mask.bool(), means, torch.zeros_like(means)), torch.where(input_mask.bool(), logstds, torch.zeros_like(logstds))    


    def sample_spatial_single(self, nsamples, event_times, spatial_locations, aux_state=None):
        """
        Args:
            nsamples: int
            event_times: (N,)
            spatial_locations: (N, D)
            aux_state: (N,  D_a)
            self.dim = 2: lat,long

        Returns:
            Samples from the spatial distribution at event times, of shape (nsamples, N, T, D).
        """

        # if input_mask is None:
        #     input_mask = torch.ones_like(event_times)
        ## T =1 
        #print(f'okk {event_times.shape} {spatial_locations.shape}  {aux_state.shape} {self.aux_dim}')
        N = event_times.shape[0]
        T = 1
        D = spatial_locations.shape[-1]
       

        #aux_state = aux_state[:,  -self.aux_dim:].reshape(N * T, self.aux_dim) #[N,1,DIM] ->
        aux_state = aux_state.reshape(N * T, self.aux_dim)
        #print(f'params {params.shape}')
        params = self.gmm_params(aux_state).reshape(-1, self.dim, 3, self.n_mixtures) #[N*1,2,3,n_mixtures]
        #print(f'params {params.shape}')
        #params = params[None].expand(nsamples, *params.shape)
        #print(f'params2 {params.shape}')
        #samples = gmm_sample(params).reshape(nsamples, N, T, D)
        samples = gmm_sample(params).reshape(N, D)
        #print(f'samples {samples.shape}')
        #ret_samples = samples.squeeze(0)

        return samples

    def spatial_conditional_logprob_fn(self, t, event_times, spatial_locations, aux_state=None):
        """
        Args:
            t: scalar
            event_times: (T,)
            spatial_locations: (T, D)
            aux_state: (T + 1, D_a)

        Returns a function that takes locations (N, D) and returns (N,) the logprob at time t.
        """
        T, D = spatial_locations.shape

        def loglikelihood_fn(s):
            bsz = s.shape[0]
            bsz_event_times = event_times[None].expand(bsz, T)
            bsz_event_times = torch.cat([bsz_event_times, torch.ones(bsz, 1).to(bsz_event_times) * t], dim=1)
            bsz_spatial_locations = spatial_locations[None].expand(bsz, T, D)
            bsz_spatial_locations = torch.cat([bsz_spatial_locations, s.reshape(bsz, 1, D)], dim=1)

            if aux_state is not None:
                bsz_aux_state = aux_state.reshape(1, T + 1, -1).expand(bsz, -1, -1)
            else:
                bsz_aux_state = None

            return self.logprob(bsz_event_times, bsz_spatial_locations, input_mask=None, aux_state=bsz_aux_state).sum(1)

        return loglikelihood_fn


def gmm_loglik(z, params):
    params = params.reshape(*z.shape, 3, -1)
    mix_logits, means, logstds = params[..., 0, :], params[..., 1, :], params[..., 2, :] #[2,18,2,5]
    mix_logprobs = mix_logits - torch.logsumexp(mix_logits, dim=-1, keepdim=True)
    logprobs = gaussian_loglik(z[..., None], means, logstds)
    std_regularization  = logstds.abs().sum(-1).sum(-1)
    return torch.logsumexp(mix_logprobs + logprobs, dim=-1) , std_regularization


def gmm_sample(params):
    """ params is (-1, 3, n_mixtures) """
    n_mixtures = params.shape[-1]
    params = params.reshape(-1, 3, n_mixtures) 
    mix_logits, means, logstds = params[..., 0, :], params[..., 1, :], params[..., 2, :]
    mix_logprobs = mix_logits - torch.logsumexp(mix_logits, dim=-1, keepdim=True)
    samples_for_all_clusters = gaussian_sample(means, logstds)    # (-1, n_mixtures)
    cluster_idx = torch.multinomial(torch.exp(mix_logprobs), 1).reshape(-1)  # (-1,)
    cluster_idx = F.one_hot(cluster_idx, num_classes=n_mixtures)  # (-1, n_mixtures)
    select_sample = torch.sum(samples_for_all_clusters * cluster_idx.to(samples_for_all_clusters), dim=-1)
    return select_sample


def gaussian_loglik(z, mean, log_std):
    mean = mean + torch.tensor(0.)
    log_std = log_std + torch.tensor(0.)
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)


def gaussian_sample(mean, log_std):
    mean = mean + torch.tensor(0.)
    log_std = log_std + torch.tensor(0.)
    z = torch.randn_like(mean) * torch.exp(log_std) + mean
    return z


ACTFNS = {
    "softplus": nn.Softplus,
    "relu": nn.ReLU,
    "elu": nn.ELU,
}


def mlp(dim=2, hidden_dims=[64, 64, 64], out_dim=None, actfn="softplus"):
    out_dim = out_dim or dim
    if hidden_dims:
        dims = [dim] + list(hidden_dims)
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d_in, d_out))
            layers.append(ACTFNS[actfn]())
        layers.append(nn.Linear(hidden_dims[-1], out_dim))
    else:
        layers = [nn.Linear(dim, out_dim)]

    return nn.Sequential(*layers)