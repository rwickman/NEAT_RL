import numpy as np


def reproduce(net_1, net_2, sig_1=0.01, sig_2=0.1):
    child_net = net_1.copy()
    for net_1_param, net_2_param, child_param in zip(net_1.parameters(), net_2.parameters(), child_net.parameters()):
        param = child_param.data + sig_1 * np.random.normal(size=net_1_param.data.shape) + sig_2 * np.random.normal(scale=1.0, size=net_1_param.data.shape)
        child_param.data.copy_(param)
    
    return child_net