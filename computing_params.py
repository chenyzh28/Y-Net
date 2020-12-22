from networks.ynet import Res_YNet
import torch
from thop import profile
import os
import statistics as stat
import datetime
os.environ["CUDA_VISIBLE_DEVICES"]='0'

model = Res_YNet(num_classes=21, os=16)
model.float()
model.eval()
model.cuda()
input = torch.randn(1,3,600,600)
input = input.float()
input = input.cuda()
times = []
for _ in range(1000):
    start = datetime.datetime.now()
    outputs = model(input)
    torch.cuda.synchronize()
    end = datetime.datetime.now()
    delta = end - start
    times.append(int(delta.total_seconds() * 1000))
print('Average Time: {} ms'.format(stat.mean(times)))
flops, params = profile(model, inputs=(input,))
print(flops/1e9,'G')
print(params/1e6,'M')

