import torch
import numpy as np



if __name__ == '__main__':
    t = torch.randint(0, 9, size=(3, 3))
    idx = torch.tensor([0, 2], dtype=torch.long)
    t_select = torch.index_select(t, dim=0, index=idx)
    print("按照dim=0进行操作 \n t:\n{}\nt_select:\n{}".format(t, t_select))

    t_select = torch.index_select(t, dim=1, index=idx)
    print("按照dim=1进行操作 \n t:\n{}\nt_select:\n{}".format(t, t_select))