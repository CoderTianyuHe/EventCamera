import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from typing import Any
import os

from matterhorn_pytorch.util.plotter import event_tensor_plot_tyx



def extract(data:np.ndarray,mask:int,shift:int)->np.ndarray:
    return (data>>shift) & mask


def filename_2_data(filename:str,endian:str=">",datatype:str="u4")->np.ndarray:
    data_str
    
class MyData(DataSet):
    labels = ("airplane",)
    def __init__(self,root:)