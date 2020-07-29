## PyTorch 神经网络基础
------

1 Torch 或 Numpy

**基本使用**
```
Torch 自称为神经网络界的 Numpy, 因为他能将 torch 产生的 tensor 
放在 GPU 中加速运算 (前提是你有合适的 GPU), 就像 Numpy 会把
array 放在 CPU 中加速运算. 所以神经网络的话, 当然是用 
Torch 的 tensor 形式数据最好咯. 就像 Tensorflow 当中的 
tensor 一样.
```

**Torch 中的数学运算** 

```
其实 torch 中 tensor 的运算和 numpy array 的如出一辙, 我们就以对比的形式来看. 
```

[更多API请查找此处](https://pytorch.apachecn.org/docs/1.4/74.html)


2 变量 (Variable)

**什么是变量**
'''
在 Torch 中的 Variable 就是一个存放会变化的值的地理位置. 里面的值会不停的变化. 就像一个裝鸡蛋的篮子, 鸡蛋数会不停变动. 
那谁是里面的鸡蛋呢, 自然就是 Torch 的 Tensor 咯. 如果用一个 Variable 进行计算, 那返回的也是一个同类型的 Variable.
''

**Variable 计算, 梯度** 
'''
时刻记住, Variable 计算时, 它在背景幕布后面一步步默默地搭建着一个庞大的系统, 叫做计算图, computational graph. 这个图是用来干嘛的? 
原来是将所有的计算步骤 (节点) 都连接起来, 最后进行误差反向传递的时候, 一次性将所有 variable 里面的修改幅度 (梯度) 都计算出来, 
而 tensor 就没有这个能力啦.
'''

3 激励函数 (Activation)

**Torch 中的激励函数** 

Torch 中的激励函数有很多, 不过我们平时要用到的就这几个. relu, sigmoid, tanh, softplus.


