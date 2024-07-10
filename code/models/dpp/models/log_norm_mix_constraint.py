import torch
import torch.nn as nn

import torch.distributions as D
from models.dpp.distributions import Normal, MixtureSameFamily, TransformedDistribution,TruncatedNormal
from models.dpp.utils import clamp_preserve_gradients

#from torchrl.modules import TruncatedNormal

# from .recurrent_tpp import RecurrentTPP




class TruncatedLogNormalMixtureDistribution(TransformedDistribution):
    """
    Mixture of log-normal distributions.

    We model it in the following way (see Appendix D.2 in the paper):

    x ~ GaussianMixtureModel(locs, log_scales, log_weights)
    y = std_log_inter_time * x + mean_log_inter_time
    z = exp(y)

    Args:
        locs: Location parameters of the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        log_scales: Logarithms of scale parameters of the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        log_weights: Logarithms of mixing probabilities for the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        mean_log_inter_time: Average log-inter-event-time, see dpp.data.dataset.get_inter_time_statistics
        std_log_inter_time: Std of log-inter-event-times, see dpp.data.dataset.get_inter_time_statistics
    """
    def __init__(
        self,
        locs: torch.Tensor,
        log_scales: torch.Tensor,
        log_weights: torch.Tensor,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0,
        constraint_lower_bound: torch.Tensor = None,
        constraint_upper_bound: torch.Tensor = None
    ):
        mixture_dist = D.Categorical(logits=log_weights)
        #component_dist = Normal(loc=locs, scale=log_scales.exp())

        # constraint_lower_bound = constraint_lower_bound.unsqueeze(-1).repeat(1,locs.shape[-1])
        # constraint_upper_bound = constraint_upper_bound.unsqueeze(-1).repeat(1,locs.shape[-1])
        #print(constraint_lower_bound[18])
        #print(constraint_upper_bound[18])
        #print(constraint_lower_bound.shape,locs.shape)
        # print("b qqq")
        # print(constraint_lower_bound[3])
        if constraint_lower_bound.dim() == 1:

            constraint_lower_bound = torch.log(constraint_lower_bound).unsqueeze(-1).repeat(1,locs.shape[-1])
            constraint_upper_bound = torch.log(constraint_upper_bound).unsqueeze(-1).repeat(1,locs.shape[-1])
        elif constraint_lower_bound.dim() == 2:
            constraint_lower_bound = torch.log(constraint_lower_bound).unsqueeze(-1).repeat(1,1,locs.shape[-1])
            constraint_upper_bound = torch.log(constraint_upper_bound).unsqueeze(-1).repeat(1,1,locs.shape[-1])
        #print(constraint_lower_bound[3])
        #print(constraint_lower_bound[18])
        #print(constraint_upper_bound[18])
        # print(locs[18])
        # print(log_scales[18])
        
        #print('fine')
        # test = constraint_lower_bound.clone()


        # inf_mask = torch.isinf(constraint_lower_bound)
        # inf_indices = torch.nonzero(inf_mask)
        #print(value)
        #print('ifn here11111?')
        # print(constraint_lower_bound[inf_indices])
        # print(test[inf_indices])
        # print(constraint_lower_bound[29,0])
        #exit()
        
        
        #print("??!!")
        #print(locs.shape,log_scales.shape,constraint_lower_bound.shape,constraint_upper_bound.shape)
        component_dist = TruncatedNormal(loc=locs, scale=log_scales.exp(),a = constraint_lower_bound, b = constraint_upper_bound)
        GMM = MixtureSameFamily(mixture_dist, component_dist)

        transforms = []
        # if mean_log_inter_time == 0.0 and std_log_inter_time == 1.0:
        #     transforms = []
        # else:
        #     transforms = [D.AffineTransform(loc=mean_log_inter_time, scale=std_log_inter_time)]
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        transforms.append(D.ExpTransform())
        #if constraint_lower_bound is not None or constraint_upper_bound is not None:
        super().__init__(GMM, transforms)

    @property
    def mean(self) -> torch.Tensor:
        """
        Compute the expected value of the distribution.

        See https://github.com/shchur/ifl-tpp/issues/3#issuecomment-623720667

        Returns:
            mean: Expected value, shape (batch_size, seq_len)
        """
        a = self.std_log_inter_time
        b = self.mean_log_inter_time
        loc = self.base_dist._component_distribution.loc
        variance = self.base_dist._component_distribution.variance
        log_weights = self.base_dist._mixture_distribution.logits
        return (log_weights + a * loc + b + 0.5 * a**2 * variance).logsumexp(-1).exp()



class ConstraintLogNormalMixtureDistribution(TransformedDistribution):
    """
    Mixture of log-normal distributions.

    We model it in the following way (see Appendix D.2 in the paper):

    x ~ GaussianMixtureModel(locs, log_scales, log_weights)
    y = std_log_inter_time * x + mean_log_inter_time
    z = exp(y)

    Args:
        locs: Location parameters of the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        log_scales: Logarithms of scale parameters of the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        log_weights: Logarithms of mixing probabilities for the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        mean_log_inter_time: Average log-inter-event-time, see dpp.data.dataset.get_inter_time_statistics
        std_log_inter_time: Std of log-inter-event-times, see dpp.data.dataset.get_inter_time_statistics
    """
    def __init__(
        self,
        locs: torch.Tensor,
        log_scales: torch.Tensor,
        log_weights: torch.Tensor,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0,
        constraint_lower_bound: torch.Tensor = None,
        constraint_upper_bound: torch.Tensor = None
    ):
        mixture_dist = D.Categorical(logits=log_weights)
        component_dist = Normal(loc=locs, scale=log_scales.exp())
        GMM = MixtureSameFamily(mixture_dist, component_dist)
        if mean_log_inter_time == 0.0 and std_log_inter_time == 1.0:
            transforms = []
        else:
            transforms = [D.AffineTransform(loc=mean_log_inter_time, scale=std_log_inter_time)]
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        transforms.append(D.ExpTransform())
        if constraint_lower_bound is not None or constraint_upper_bound is not None:
            transforms.append(D.SigmoidTransform())
            transforms.append(D.transforms.AffineTransform(loc=2* constraint_lower_bound-constraint_upper_bound , scale=(constraint_upper_bound-constraint_lower_bound)*2.0))
        super().__init__(GMM, transforms)

    @property
    def mean(self) -> torch.Tensor:
        """
        Compute the expected value of the distribution.

        See https://github.com/shchur/ifl-tpp/issues/3#issuecomment-623720667

        Returns:
            mean: Expected value, shape (batch_size, seq_len)
        """
        a = self.std_log_inter_time
        b = self.mean_log_inter_time
        loc = self.base_dist._component_distribution.loc
        variance = self.base_dist._component_distribution.variance
        log_weights = self.base_dist._mixture_distribution.logits
        return (log_weights + a * loc + b + 0.5 * a**2 * variance).logsumexp(-1).exp()


class ConstraintLogNormMix(nn.Module):
    """
    RNN-based TPP model for marked and unmarked event sequences.

    The marks are assumed to be conditionally independent of the inter-event times.

    The distribution of the inter-event times given the history is modeled with a LogNormal mixture distribution.

    Args:
        num_marks: Number of marks (i.e. classes / event types)
        mean_log_inter_time: Average log-inter-event-time, see dpp.data.dataset.get_inter_time_statistics
        std_log_inter_time: Std of log-inter-event-times, see dpp.data.dataset.get_inter_time_statistics
        context_size: Size of the context embedding (history embedding)
        mark_embedding_size: Size of the mark embedding (used as RNN input)
        num_mix_components: Number of mixture components in the inter-event time distribution.
        rnn_type: Which RNN to use, possible choices {"RNN", "GRU", "LSTM"}

    """

    def __init__(
        self,
        args,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0,
        context_size: int = 256,
        num_mix_components: int = 32,
    ):
        super().__init__()
        self.args = args

        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        self.context_size = context_size
        self.num_mix_components = num_mix_components
        self.linear = nn.Linear(self.context_size, 3 * self.num_mix_components)

    def get_inter_time_dist(self, context: torch.Tensor,constraint_lower_bound: torch.Tensor,constraint_upper_bound: torch.Tensor) -> torch.distributions.Distribution:
        """
        Get the distribution over inter-event times given the context.

        Args:
            context: Context vector used to condition the distribution of each event,
                shape (batch_size, seq_len, context_size)

        Returns:
            dist: Distribution over inter-event times, has batch_shape (batch_size, seq_len)

        """
        raw_params = self.linear(context)  # (batch_size, seq_len, 3 * num_mix_components)
        # Slice the tensor to get the parameters of the mixture
        locs = raw_params[..., :self.num_mix_components] # (batch_size, seq_len, num_mix_components)
        log_scales = raw_params[..., self.num_mix_components: (2 * self.num_mix_components)]
        log_weights = raw_params[..., (2 * self.num_mix_components):]

        log_scales = clamp_preserve_gradients(log_scales, -5.0, 3.0)
        log_weights = torch.log_softmax(log_weights, dim=-1)
        # print("???")
        # print(constraint_lower_bound.shape,locs.shape)
        #print(constraint_lower_bound[3])

        
        # inf_mask = torch.isnan(log_weights)
        # inf_indices = torch.nonzero(inf_mask)
        # is_empty = inf_indices.numel() == 0
        # if not is_empty:
        #     print('inf weights')
        #     print(log_weights[inf_indices])
        #     print(inf_indices)
        #     print('inf weights')
        #     #exit()

        
        if self.args.time_decoder == 'truncate':

            return TruncatedLogNormalMixtureDistribution(
                locs=locs,
                log_scales=log_scales,
                log_weights=log_weights,
                mean_log_inter_time=self.mean_log_inter_time,
                std_log_inter_time=self.std_log_inter_time,
                constraint_lower_bound = constraint_lower_bound,
                constraint_upper_bound = constraint_upper_bound
            )
        elif self.args.time_decoder == 'constraint':
            return ConstraintLogNormalMixtureDistribution(
                locs=locs,
                log_scales=log_scales,
                log_weights=log_weights,
                mean_log_inter_time=self.mean_log_inter_time,
                std_log_inter_time=self.std_log_inter_time,
                constraint_lower_bound = constraint_lower_bound,
                constraint_upper_bound = constraint_upper_bound
            )
        else:
            raise ValueError("time_decoder not supported")
        

        