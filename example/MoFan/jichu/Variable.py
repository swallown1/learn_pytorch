import torch
from torch.autograd import Variable

tensor  =  torch.FloatTensor([[1,2],[3,4]])

varible = Variable(tensor,requires_grad=True)

# print(tensor)
# print(varible)

t_out = torch.mean(tensor*tensor)# x^2
v_out = torch.mean(varible*varible)


# print(t_out)
# print(v_out)

#误差的反向传递

v_out.backward()


# v_out = 1/4 * sum(variable*variable) 这是计算图中的 v_out 计算步骤
# 针对于 v_out 的梯度就是, d(v_out)/d(variable) = 1/4*2*variable = variable/2

print(varible.grad) # 初始 Variable 的梯度 也就是更新的梯度


print(varible.data.numpy())

print(varible)
