import torch

def one_param():
    x = torch.tensor([3.], requires_grad=True)
    y = torch.pow(x, 2)  # y=x^2

    # 一次求导
    grad_1 = torch.autograd.grad(y, x, create_graph=True)  # 这里必须创建导数的计算图， grad_1 = dy/dx = 2x
    print(grad_1)  # (tensor([6.], grad_fn=<MulBackward0>),) 这是个元组，二次求导的时候我们需要第一部分

    # 二次求导
    grad_2 = torch.autograd.grad(grad_1[0], x)  # grad_2 = d(dy/dx) /dx = 2
    print(grad_2)  # (tensor([2.]),)

def mul_param():
    x1 = torch.tensor(1.0, requires_grad=True)  # x需要被求导
    x2 = torch.tensor(2.0, requires_grad=True)

    y1 = x1 * x2
    y2 = x1 + x2

    # 允许同时对多个自变量求导数
    (dy1_dx1, dy1_dx2) = torch.autograd.grad(outputs=y1, inputs=[x1, x2], retain_graph=True)
    print(dy1_dx1, dy1_dx2)  # tensor(2.) tensor(1.)

    # 如果有多个因变量，相当于把多个因变量的梯度结果求和
    (dy12_dx1, dy12_dx2) = torch.autograd.grad(outputs=[y1, y2], inputs=[x1, x2])
    print(dy12_dx1, dy12_dx2)  # tensor(3.) tensor(2.)



if __name__ == '__main__':
    # one_param()

    mul_param()