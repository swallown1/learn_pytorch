import torch

def grad_zero():
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    for i in range(4):
        a = torch.add(w, x)
        b = torch.add(w, 1)
        y = torch.mul(a, b)

        y.backward()
        print(w.grad)

        # w.grad.zero_()  # 将梯度去除  不会将梯度累加

def requires_grad():
    #与叶子节点相关联requires_grad默认为True
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    print(a.requires_grad, b.requires_grad, y.requires_grad)

def in_place():
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    w.add_(1)#原位操作

    y.backward()  #报错  RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.

if __name__ == '__main__':
    ###  自动求导需要注意的问题
    # grad_zero()

    requires_grad()
    # in_place()