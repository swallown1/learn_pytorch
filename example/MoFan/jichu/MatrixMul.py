import torch
import numpy as np

data = [[1,2], [3,4]]

tensor =  torch.FloatTensor(data)


print(
    '\nmatrix multiplication (matmul)',
    '\nnumpy: ', np.matmul(data,data),
    '\npytorch:', torch.matmul(tensor,tensor)
)

data = np.array(data)

print(
    '\nmatrix multiplication (dot)',
    '\nnumpy: ', data.dot(data),        # [[7, 10], [15, 22]] 在numpy 中可行
    # '\ntorch: ', torch.dot(tensor.dot(tensor))    # torch 会转换成 [1,2,3,4].dot([1,2,3,4) = 30.0
)

'''
matrix multiplication (matmul) 
numpy:  [[ 7 10]
 [15 22]]

pytorch: tensor([[ 7., 10.],
        [15., 22.]])
'''