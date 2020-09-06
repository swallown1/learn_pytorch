import torch
import numpy as np

def build1():
    arr = np.ones((3,3))
    print("ndarray的数据类型:", arr.dtype)
    t = torch.tensor(arr,device='cuda',dtype=torch.int8)
    print(t)

def build_from_numpy():
    array = np.arange(1,10).reshape((3,3))
    t = torch.from_numpy(array)
    print("numpy array:", array)
    print("tensor:", t)

    print("\n修改arr")
    array[0, 0] = 0
    print("numpy array:", array)
    print("tensor:", t)

    print("\n修改tensor")
    t[2, 2] = -1
    print("numpy array:", array)
    print("tensor:", t)

def build_zero():
    out_t = torch.tensor([1])
    print(out_t,'\n')
    t = torch.zeros((2,2),out=out_t)
    print(t, '\n',out_t)
    print(id(t), id(out_t), id(t) == id(out_t))  # 这个看内存地址

def build_zeros_like():
    input = torch.ones((2,2))
    print("input : ",input, '\n')

    out = torch.zeros_like(input)
    print("out : " , out, '\n')

def build_full():
    t = torch.full((2,2),fill_value=4,dtype=torch.float64)
    print(t)

def build_linspace():
    t = torch.linspace(2, 10, 6)
    print(t)

def build_eye():
    t = torch.eye(3, 4)
    print(t)

def build_normal():
    # 模式：
    # 1）mean为标量，std为标量
    t_normal = torch.normal(0,1,size=(4,))
    print(t_normal)

    # 2）mean为标量，std为张量
    # mean为标量, std为张量
    mean = 0
    std = torch.arange(1, 5, dtype=torch.float)
    t_normal = torch.normal(mean, std)
    print("mean:{}\nstd:{}".format(mean, std))
    print(t_normal)

    # 3） mean为张量，std为标量
    # mean为张量, std为标量
    mean = torch.arange(1, 5, dtype=torch.float)
    std = 1
    t_normal = torch.normal(mean, std)
    print("mean:{}\nstd:{}".format(mean, std))
    print(t_normal)

    # 4） mean为张量，std为张量
    # mean为张量, std为张量
    mean = torch.arange(1, 5, dtype=torch.float)
    std = torch.arange(1, 5, dtype=torch.float)
    t_normal = torch.normal(mean, std)
    print("mean:{}\nstd:{}".format(mean, std))
    print(t_normal)



if __name__ == '__main__':
    ## 直接创建
    build1()

    ## 从numpy中构建
    build_from_numpy()

    ## 依据数值创建
    build_zero()
    build_zeros_like()
    build_full()
    build_linspace()
    build_eye()

    ## 依据概率创建
    build_normal()