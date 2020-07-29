import torch
import matplotlib.pyplot as plt

n_data = torch.ones(100,2)
x0 = torch.normal(2*n_data,1) # 类型0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)  # 类型0 y data (tensor), shape=(100, )
x1 = torch.normal(-2*n_data,1)   # 类型1 x data (tensor), shape=(100, 1)
y1 = torch.zeros(100)    # 类型1 y data (tensor), shape=(100, )

x = torch.cat((x0,x1),0).type(torch.FloatTensor)  #(tensor), shape=(200, 2)
y = torch.cat((y0,y1)).type(torch.LongTensor) #(tensor), shape=(200, )

# 画图
# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()


# 建立神经网络 ¶

class classicNet(torch.nn.Module):
    def __init__(self,in_size,hidd_size,out_size):
        super(classicNet, self).__init__()
        self.layer1 = torch.nn.Linear(in_size,hidd_size)  # 隐藏层线性输出
        self.layer2 = torch.nn.Linear(hidd_size,out_size)     # 输出层线性输出

    def forward(self,x):
        # 正向传播输入值, 神经网络分析出输出值
        x = torch.relu(self.layer1(x))    # 激励函数(隐藏层的线性值)
        x = self.layer2(x)   # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
        return x

net = classicNet(2,10,2)   #两个类别   映射成两个维度
print(net)
"""
classicNet(
  (layer1): Linear(in_features=2, out_features=10, bias=True)
  (layer2): Linear(in_features=10, out_features=2, bias=True)
)
"""


#训练

optimizer = torch.optim.SGD(net.parameters(),lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()


plt.ion()   # 画图

for i in range(100):
    out = net(x)

    loss = loss_func(out,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 2 == 0:
        # plot and show learning process
        plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()

#
# print(x1)
#
#
# print(x)

