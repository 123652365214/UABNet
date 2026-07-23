import mobula
import torch
import torch.nn.functional as F
from . import mobula_hack_for_pytorch
from .attention_sampler import AttSamplerGrid

# Hack MobulaOP for the compatible functions
torch._mobula_hack = mobula_hack_for_pytorch


class AttSampler(torch.nn.Module):
    def __init__(self, scale=1.0, dense=4, iters=5):
        super(AttSampler, self).__init__()
        self.scale = scale
        self.dense = dense
        self.iters = iters

    def forward(self, data, attx, atty):
        grid = AttSamplerGrid(data.detach(),
                                        attx.detach(),
                                        atty.detach(),
                                     )
        list=[]
        list.append(grid.dense)
        list.append(grid.iters)
        grid_list = torch.stack(list, dim=3)
        #print('data_data',data.shape)
        #print('grid_grid',grid.shape)
        #print('grid',grid)
        print(torch.equal(data, grid.scale))
        print(torch.equal(attx, grid.iters))
        print(torch.equal(atty, grid.dense))
        return F.grid_sample(data, grid_list),grid
