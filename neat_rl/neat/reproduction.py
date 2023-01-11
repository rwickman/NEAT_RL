import numpy as np
from neat.reproduction import Reproduction
import torch


class GradientReproduction(Reproduction):
    def __init__(self, args):
        super().__init__(args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reproduce(self, net_1, net_2, sig_1=0.01, sig_2=0.1):
        child_net = net_1.copy()
        for net_1_param, net_2_param, child_param in zip(net_1.parameters(), net_2.parameters(), child_net.parameters()):
            rand_noise = torch.tensor(sig_1 * np.random.normal(size=net_1_param.data.shape)).to(self.device)
            rand_inter = sig_2 * torch.tensor(np.random.normal(scale=1.0, size=net_1_param.data.shape)).to(self.device) * (net_2_param - net_1_param)
            param = net_1_param + rand_noise + rand_inter 
            child_param.data.copy_(param)
        
        return child_net