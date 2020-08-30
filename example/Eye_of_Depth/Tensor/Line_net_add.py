import torch
import matplotlib.pyplot as plt

def show(x,y_pre,y):
    plt.scatter(x,y,marker="o",color="r")
    plt.plot(x,y_pre,marker="X",color='g')
    plt.show()

if __name__ == '__main__':
    #使用torch.add 训练线性回归模型
    #

    lr = 0.01
    x = torch.rand(20,1)*10
    y = 2*x + (5+torch.randn(20,1))

    w = torch.randn((1),requires_grad=True)
    b = torch.randn((1),requires_grad=True)

    for iteration in range(100):
        y_pred = torch.add(b,torch.mul(w,x))

        # 计算loss
        loss = (0.5 * (y - y_pred) ** 2).mean()

        # 反向传播
        loss.backward()

        # 更新参数
        b.data.sub_(lr * b.grad)  # 这种_的加法操作时从自身减，相当于-=
        w.data.sub_(lr * w.grad)

        # 梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()

    show(x,(w*x+b).detach().numpy(),y)


