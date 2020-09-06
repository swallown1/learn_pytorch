import numpy as np
import torch



if __name__ == '__main__':
    t1 = torch.tensor(np.arange(1,13))
    print("t1矩阵：",t1,"\n")

    list_tensor = torch.chunk(t1,dim=0
                              ,chunks=2)

    for idx, t in enumerate(list_tensor):
        print("第{}个张量：{}, shape is {}".format(idx + 1, t, t.shape))

    t2 = torch.tensor(np.arange(1, 13).reshape(4,3))
    print("t2矩阵：", t2, "\n")

    list_tensor = torch.chunk(t2, dim=1
                              , chunks=2)
    for idx, t in enumerate(list_tensor):
        print("第{}个张量：{}, shape is {}".format(idx + 1, t, t.shape))
