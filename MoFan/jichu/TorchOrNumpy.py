import  torch

import numpy as np

np_data = np.arange(9).reshape((3,3))
torch_data = torch.from_numpy(np_data)

#将torch转化成numpy
tensor2numpy = torch_data.numpy()

# print(np_data)
# print(torch_data)
# print(tensor2numpy)
'''
输出：
[[0 1 2]
 [3 4 5]
 [6 7 8]]
 
tensor([[0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]], dtype=torch.int32)
        
[[0 1 2]
 [3 4 5]
 [6 7 8]]
'''


