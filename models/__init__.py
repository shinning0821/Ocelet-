import torch
import tqdm
import argparse
import pandas as pd
import pickle, os
import numpy as np
from haven import haven_results as hr 
from . import SegNet


def get_model(model_dict, exp_dict=None, train_set=None):
    if model_dict["name"] in ["segnet"]:
        model = SegNet.SegNet(exp_dict['model'], train_set =train_set)
    else:
        raise KeyError("Wrong model parameters!")
    return model