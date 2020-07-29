import torch
import matplotlib.pyplot as plt

#构建数据

x = torch.unsqueeze(torch.linspace(-2,2,100),dim=1) # x data (tensor), shape=(100, 1)
y = x.pow(2) + torch.rand(x.size())*0.2

plt.scatter(x.data.numpy(),y.data.numpy())
plt.show()

#构建网络

class Net(torch.nn.Module):
    def __init__(self,in_size,hidd_size,out_size):
        super(Net, self).__init__()
        self.layer1 = torch.nn.Linear(in_size,hidd_size)
        self.layer2 = torch.nn.Linear(hidd_size,out_size)

    def forward(self,x):
        x = torch.sigmoid(self.layer1(x))
        y = self.layer2(x)
        return y

net = Net(1,10,1)

# print(net)
"""

Net(
  (layer1): Linear(in_features=1, out_features=10, bias=True)
  (layer2): Linear(in_features=10, out_features=1, bias=True)
)

"""

#训练网络
optimizer = torch.optim.SGD(net.parameters(),lr=0.2) # 传入 net 的所有参数, 学习率
loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)


plt.ion()   # 画图
plt.show()

for e in range(100):
    y_hat = net(x)

    loss = loss_func(y_hat,y)   #计算误差

    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    loss.backward() # 误差反向传播, 计算参数更新值
    optimizer.step()  # 将参数更新值施加到 net 的 parameters 上

    # 接着上面来
    if e % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), y_hat.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)


