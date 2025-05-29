# This file is used for putting model weights from the stored state_dict to binaries on disk
import sys
import os
import random
import torch
import numpy as np

np.random.seed(7)  # Initialize seed to get reproducible results
random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed(7)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

state_dict = torch.load("temp_model")

def to_file(name):
    tensor = state_dict[name]
    arr = tensor.cpu().numpy()
    arr.tofile(name + ".bin")
    # with open('model.bin', 'wb') as f:
    #     f.write(arr.astype(np.float32).tobytes())

stdout = sys.stdout
log_file = open("state_list.log", "w")
sys.stdout = log_file

for name, tensor in state_dict.items():
    if "num_batches_tracked" in name:
        continue
    # to_file(name)
    print(name, "\t", tensor.size())
    # print(name, "\t", tensor)

log_file.close()

log_file = open("data_list.log", "w")
sys.stdout = log_file
for name, tensor in state_dict.items():
    if "num_batches_tracked" in name:
        continue
    # to_file(name)
    # print(name, "\t", tensor.size())
    print(name, "\t", tensor)

sys.stdout = stdout


