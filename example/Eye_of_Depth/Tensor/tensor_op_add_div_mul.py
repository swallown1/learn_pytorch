import numpy as np
import torch



if __name__ == '__main__':
    t_0 = torch.randn((3, 3),dtype=torch.float64)
    t_1 = torch.ones_like(t_0,dtype=torch.float64)
    t_add = torch.add(t_0,10, t_1)

    print("t_0:\n{}\nt_1:\n{}\nt_add_10:\n{}".format(t_0, t_1, t_add))

