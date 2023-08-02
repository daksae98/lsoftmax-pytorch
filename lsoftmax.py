# implementation of Large margin Softmax loss by Amir H. Farzaneh.
# https://github.com/amirhfarzaneh/lsoftmax-pytorch

import math
import torch
from torch import nn
from scipy.special import binom


###############################

# psi(x) = (-1)^k * cos(mx) - 2k
# logit_target = ||W||*||x||*psi(x)
# Logit = ||w||*||x||*cos(x)

###############################

class LSoftmaxLinear(nn.Module):

    def __init__(self, input_features, output_features, margin, device):
        super().__init__()
        self.input_dim = input_features # number of input feature i.e. output of the last fc layer
        self.output_dim = output_features  # number of ouput = class numbers
        self.margin = margin # margin
        self.beta = 100 # ?
        self.beta_min = 0 # ?
        self.scale = 0.99 # ?
        self.device = device # cpu or gpu



        # Initialize L-Softmax parameters
        self.weight = nn.Parameter(torch.FloatTensor(input_features, output_features))
        self.divisor = math.pi / self.margin # pi/m
        self.C_m_2n = torch.Tensor(binom(margin, range(0, margin + 1, 2))).to(device) # C_m{2n}
        self.cos_powers = torch.Tensor(range(self.margin, -1, -2)).to(device) # m - 2n
        self.sin2_powers = torch.Tensor(range(len(self.cos_powers))).to(device) # n
        self.signs = torch.ones(margin // 2 + 1).to(device) # (-1)^k
        self.signs[1::2] = -1 # signs[1::2] -> idx 1부터 2간격으로 띄워서 (1, 3, 5, 7 ...)
        
    def calculate_cos_m_theta(self, cos_theta):
        sin2_theta = 1 - cos_theta**2
        cos_terms = cos_theta.unsqueeze(1) ** self.cos_powers.unsqueeze(0) # cos^{m-2n}
        sin2_terms = (sin2_theta.unsqueeze(1) ** self.sin2_powers.unsqueeze(0)) # sin^{n}

        cos_m_theta = (self.signs.unsqueeze(0) * # cos(mx) = -1^n C_m(2n) * cos^{m - 2n}(x) * sin2^{n}(x)
                       self.C_m_2n.unsqueeze(0) * 
                       cos_terms * 
                       sin2_terms).sum(1) # summation of all terms

        return cos_m_theta
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight.data.t()) # 

    def find_k(self,cos):
        # to account for acos numerical errors
        eps = 1e-7
        cos = torch.clamp(cos, -1 + eps, 1 - eps)
        theta = cos.acos()  # -> cos^-1
        k = (theta / self.divisor).floor().detach() #  k < (theta * m / pi) < k + 1 
        return k
    
    def forward(self, input, target=None):
        if self.training:
            assert target is not None
            x, w = input, self.weight
            beta = max(self.beta, self.beta_min)
            logit = x.mm(w) # W^T * X
            indexes = range(logit.size(0))
            logit_target = logit[indexes, target] # Wyi * X

            # cos = W * X / ||W| ||X||
            w_target_norm = w[:,target].norm(p=2, dim=0)
            x_norm = x.norm(p=2, dim=1)
            cos_theta_target = logit_target / (w_target_norm * x_norm + 1e-10)

            # eq 7. cos(m*theta) =  cos(mx) = -1^n C_m(2n) * cos^{m - 2n}(x) * sin2^{n}(x)
            cos_m_theta_target = self.calculate_cos_m_theta(cos_theta_target)

            # find k; k = floor(theta*m/pi)
            k = self.find_k(cos_theta_target)

            # f_y_i = 
            logit_target_updated = (w_target_norm * x_norm * (((-1)**k * cos_theta_target) -2* k ))

            # psi(x) + b * (W^T * X) / (1 + b) -> 처음에는 margin사용하지 않고 기존 방식대로 진행되다가, 약 60~70 iter부터 반반 (beta = 100)
            logit_target_updated_beta = (logit_target_updated + beta * logit[indexes, target]) / (1 + beta) 

            logit[indexes,target] = logit_target_updated_beta
            self.beta *= self.scale

            return logit
        else:
            assert target is None
            return input.mm(self.weight)


