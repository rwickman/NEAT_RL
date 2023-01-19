import numpy as np
import torch
import copy

class GradientReproduction:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reproduce(self, net_1, net_2):
        net_1_state_dict = net_1.state_dict()
        net_2_state_dict = net_2.state_dict()
        child_net_state_dict = copy.deepcopy(net_1_state_dict)
        for tensor in net_1_state_dict:
            if "weight" or "bias" in tensor:
                child_net_state_dict[tensor] = self.mutate(net_1_state_dict[tensor], net_2_state_dict[tensor])

        child_net = net_1.copy(transfer_weights=False)
        result = child_net.load_state_dict(child_net_state_dict)

        return child_net
    
    def mutate(self, x, y):
        a = torch.zeros_like(x, device=self.device).normal_(mean=0, std=self.args.iso_sigma)
        b = np.random.normal(0, self.args.line_sigma)
        z = x.clone() + a + b * (y - x)

        return z

    # def mutate(self, net_1, net_2, iso_sigma=0.005, iso_sigma=0.05):

    #     child_net = net_1.copy(transfer_weights=True)
    #     for net_1_param, net_2_param, child_param in zip(net_1.parameters(), net_2.parameters(), child_net.parameters()):
    #         rand_noise = torch.tensor(np.random.normal(0, sig_1, size=net_1_param.data.shape)).to(self.device)
    #         rand_inter = torch.tensor(np.random.normal(0, sig_2, size=net_1_param.data.shape)).to(self.device) * (net_2_param - net_1_param)
    #         param = net_1_param + rand_noise + rand_inter
    #         child_param.data.copy_(param)

    #     return child_net
    