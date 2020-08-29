import torch
import torch.nn as nn    # 激励函数都在这
from torch.autograd import Variable

x = torch.linspace(-6,6,300)

x = Variable(x)

x_np = x.data.numpy() # 换成 numpy array, 出图时用

#常用的激活函数

y_relu = torch.relu(x).data.numpy()
y_sigmoid = torch.sigmoid(x).data.numpy()
y_tanh = torch.tanh(x).data.numpy()
y_softplus = nn.Softplus()(x).data.numpy()

import matplotlib.pyplot as plt

plt.figure(1,figsize=(8,6))

plt.subplot(221)
plt.plot(x_np,y_relu,c='red',label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np,y_tanh,c='red',label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np,y_sigmoid,c='red',label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np,y_softplus,c='red',label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()