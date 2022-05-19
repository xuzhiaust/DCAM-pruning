from pytorchyolo.models import load_model
from pytorchyolo.utils.prune_utils import parse_module_defs
import torch
from terminaltables import AsciiTable
import numpy as np
from copy import deepcopy
import torch.nn.functional as F

class opt():
    model_def = "config/yolov3.cfg"
    data_config = "config/oxfordhand.data"
    model = 'checkpoints/yolov3_ckpt.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(opt.model_def).to(device)

CBL_idx, Conv_idx, prune_idx= parse_module_defs(model.module_defs)









