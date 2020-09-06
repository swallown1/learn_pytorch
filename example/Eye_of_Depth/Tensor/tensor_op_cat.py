import numpy as np
import torch



if __name__ == '__main__':
    t1 = torch.tensor(np.arange(1,7).reshape(3,2))
    t2 = torch.tensor(np.arange(7,13).reshape(3,2))
    print("t1矩阵：",t1,"\n")
    print("t2矩阵：",t2,"\n")

    cat1 = torch.cat([t1,t2],dim=0)
    print("dim = 0: t1和t2拼接",cat1,"\n")

    cat2 = torch.cat([t1,t2],dim=1)
    print("dim = 1: t1和t2拼接",cat2,"\n")



    cat3 = torch.cat([t1,t2,t1,t2],dim=1)
    print("dim = 1: 2个t1和2个t2拼接",cat3,"\n")