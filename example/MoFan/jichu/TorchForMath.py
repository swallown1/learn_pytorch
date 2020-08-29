import  torch
import  numpy as np


data = [1,-2,3,-4]
tensor = torch.FloatTensor(data)
print(
    '\nabs',
    '\nnumpy: ', np.abs(data),
    '\ntorch: ', torch.abs(tensor)
)

# sin   三角函数 sin
print(
    '\nsin',
    '\nnumpy: ', np.sin(data),      # [-0.84147098 -0.90929743  0.84147098  0.90929743]
    '\ntorch: ', torch.sin(tensor)  # [-0.8415 -0.9093  0.8415  0.9093]
)


# cos   三角函数 cos
print(
    '\ncos',
    '\nnumpy: ', np.cos(data),      # [-0.84147098 -0.90929743  0.84147098  0.90929743]
    '\ntorch: ', torch.cos(tensor)  # [-0.8415 -0.9093  0.8415  0.9093]
)
# mean   均值
print(
    '\ncos',
    '\nnumpy: ', np.mean(data),      # [-0.84147098 -0.90929743  0.84147098  0.90929743]
    '\ntorch: ', torch.mean(tensor)  # [-0.8415 -0.9093  0.8415  0.9093]
)


