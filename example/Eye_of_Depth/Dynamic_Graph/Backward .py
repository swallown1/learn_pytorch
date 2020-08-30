import torch


def backward():
    w = torch.tensor([1.], requires_grad=True)  # 创建叶子张量，并设定requires_grad为True，因为需要计算梯度；
    x = torch.tensor([2.], requires_grad=True)  # 创建叶子张量，并设定requires_grad为True，因为需要计算梯度；

    a = torch.add(w, x)  # 执行运算并搭建动态计算图
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    y.backward(retain_graph=True)  # 对y执行backward方法就可以得到x和w两个叶子节点
    ## 设置retain_graph  可以保留计算图  可以再一次进行 backward
    print(w.grad,x.grad)

def grad_tensors():
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)  # retain_grad()
    b = torch.add(w, 1)

    y0 = torch.mul(a, b)  # y0 = (x+w) * (w+1)    dy0/dw = 5   dy0/dx = 2
    y1 = torch.add(a, b)  # y1 = (x+w) + (w+1)    dy1/dw = 2   dy1/dx = 1

    loss = torch.cat([y0, y1], dim=0)  # [y0, y1]
    grad_tensors = torch.tensor([1., 2.])

    loss.backward(gradient=grad_tensors)  # gradient 传入 torch.autograd.backward()中的grad_tensors

    print(w.grad, x.grad)   ## dloss/dw = 5*1 + 2*2 =9    dloss/dx = 2*1 + 1*2=4


if __name__ == '__main__':
    # backward()

    grad_tensors()