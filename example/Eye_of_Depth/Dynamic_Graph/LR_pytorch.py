import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.layer = nn.Linear(2,1)

    def forward(self,x):
        x = torch.sigmoid(self.layer(x))
        return x

if __name__ == '__main__':
    batch_num = 100
    mean_value = 1.7
    bias = 1

    ## 构造二分类数据  两类数据的均值不同  即聚类的中心点不在一块
    data = torch.ones(batch_num,2)
    x0 = torch.normal(mean_value*data,1) + bias
    y0 = torch.zeros(batch_num)

    x1 = torch.normal(-mean_value*data,1) + bias
    y1 = torch.ones(batch_num)


    train_x = torch.cat([x0,x1],dim=0)
    train_y = torch.cat([y0,y1],dim=0)

    ## LR模型
    model = LR()

    ## Loss
    loss_fn = nn.BCELoss()  #二进制交叉熵损失

    ## optim
    optim = torch.optim.SGD(model.parameters(),lr = 0.01,momentum=0.9)

    ##train
    for epoch in range(100):
        y_pred = model(train_x)

        loss = loss_fn(y_pred.squeeze(),train_y)

        loss.backward()

        optim.step()  # 更新参数

        optim.zero_grad() # 清空梯度

        ##画图
        if epoch % 20 == 0:
            mask = y_pred.ge(0.5).float().squeeze()
            correct = (mask == train_y).sum()
            acc = correct.item() / train_y.size(0)
            plt.scatter(x0.data.numpy()[:, 0], x0.data.numpy()[:, 1], c='r', label='class 0')
            plt.scatter(x1.data.numpy()[:, 0], x1.data.numpy()[:, 1], c='b', label='class 1')

            w0, w1 = model.layer.weight[0]
            w0, w1 = float(w0.item()), float(w1.item())
            plot_b = float(model.layer.bias[0].item())

            plot_x = np.arange(-6, 6, 0.1)
            plot_y = (-w0 * plot_x - plot_b) / w1

            plt.xlim(-5, 7)
            plt.ylim(-7, 7)
            plt.plot(plot_x, plot_y)

            plt.text(-5, 5, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
            plt.title(
                "Iteration: {}\nw0:{:.2f} w1:{:.2f} b: {:.2f} accuracy:{:.2%}".format(epoch, w0, w1, plot_b, acc))
            plt.legend()

            plt.show()
            plt.pause(0.5)

            if acc > 0.99:
                break