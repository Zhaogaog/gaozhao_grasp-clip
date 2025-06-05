from b import test
x=test(1,2)
import torch.nn as nn

for i in range(5):
    locals()[f'a_{i}'] = nn.BatchNorm2d(3)
print(a_1)
print(locals()[f'a_{1}'])
