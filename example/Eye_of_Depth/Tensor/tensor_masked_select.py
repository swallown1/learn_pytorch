import torch
import numpy as np



if __name__ == '__main__':
    t = torch.randint(0, 9, size=(3, 3))
    mask = t.le(5)
    print(" mask:\n{}".format(mask))
    t_select = torch.masked_select(t, mask)
    print(" \n t:\n{}\nt_select:\n{}".format(t, t_select))
