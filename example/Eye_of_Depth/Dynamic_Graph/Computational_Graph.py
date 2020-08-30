import torch

def retain_grad():
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    a.retain_grad()
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    y.backward()
    # 查看梯度， 默认是只保留叶子节点的梯度的
    print("gradient:\n", w.grad, x.grad, a.grad, b.grad, y.grad)


if __name__ == '__main__':
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)
    a = torch.add(w, x)  # retain_grad()
    b = torch.add(w, 1)
    y = torch.mul(a, b)
    y.backward()
    print(w.grad)

    print("is_leaf:\n", w.is_leaf, x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)
    # 查看梯度
    print("gradient:\n", w.grad, x.grad, a.grad, b.grad, y.grad)
    # gradient:
    #  tensor([5.]) tensor([2.]) None None None
    # 查看 节点处执行的函数
    print("grad_fn:\n", w.grad_fn, x.grad_fn, a.grad_fn, b.grad_fn, y.grad_fn)
    # grad_fn:

    retain_grad()  ## a通过retain_grad将梯度保留下来了
