import argparse
import time
import torch
import torch.nn.functional as F
from quant import *


torch.manual_seed(23)
# data = (torch.randn(9)*1)
# sign = torch.sign(data) 
# data = (data**2)*sign

# # add some examples to test saturation limits
# data = torch.cat([data, torch.tensor([7.666, 6.98, 7.01, 0.00879, 0.0142, 0.0158])])
# print(data)
# M_data = [[0.2, 0.5], [0.7, 0.3]]
# data = torch.tensor(M_data)
data = torch.tensor([7.666, 6.98, 7.01, 0.00879, 0.0142, 0.0158])
# data = torch.reshape(data, (3,3))
# print(data)

torch.manual_seed(time.time())


num = BlockMinifloat(exp=3, man=2, tile=-1, k_exp=5)
quant_func = quantizer(forward_number=num, forward_rounding="stochastic")

qdata = quant_func(data)


print("Input:", data, "\n-----------------------------")
print("Quant:", qdata, "\n-----------------------------")
print("--------------------------------------")
print("Error:", data-qdata, torch.sum((data-qdata)**2)/(len(data)))


