from __future__ import division
from itertools import chain
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pytorchyolo.utils.parse_config import parse_model_config
from pytorchyolo.utils.utils import weights_init_normal

from pytorchyolo.models import load_model
from pytorchyolo.utils.prune_utils import parse_module_defs

# 通过model_list[0][0].depth_conv获取权重
record_list=[]
class SELayer(nn.Module):
    def __init__(self,channel):
        super(SELayer,self).__init__()
        self.conv_channel3_1=nn.Conv2d(in_channels=channel,out_channels=channel,kernel_size=3,groups=channel,padding=1)
        self.max_pool2=nn.MaxPool2d(kernel_size=2,stride=2)
        self.avg_pool2=nn.AvgPool2d(kernel_size=2,stride=2)

        self.conv_channel1_1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, groups=channel)
        self.conv_channel1_2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, groups=channel)

        self.gn_layer_1=nn.GroupNorm(num_groups=4,num_channels=channel)
        self.conv_channel3_2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, groups=channel,padding=1)

        self.max_pool=nn.AdaptiveMaxPool2d(1)
        self.avg_pool=nn.AdaptiveAvgPool2d(1)

        self.gn_layer_2 = nn.GroupNorm(num_groups=4, num_channels=channel)
        self.gn_layer_3 = nn.GroupNorm(num_groups=4, num_channels=channel)

        self.conv_channel1_4 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, groups=channel)
        self.gn_layer_4 = nn.GroupNorm(num_groups=4, num_channels=channel)

        self.sigmoid_gn=nn.Sigmoid()

    def forward(self,x):

        yy=x.clone()
        yy=self.conv_channel3_1(yy)
        yy_max2=self.max_pool2(yy)
        yy_avg2=self.avg_pool2(yy)
        yy_max2=self.conv_channel1_1(yy_max2)
        yy_avg2=self.conv_channel1_2(yy_avg2)

        yy=yy_max2+yy_avg2
        yy=self.gn_layer_1(yy)
        yy=self.conv_channel3_2(yy)

        yy_max1=self.max_pool(yy)
        yy_avg1=self.avg_pool(yy)
        yy_max1=self.gn_layer_2(yy_max1)
        yy_avg1=self.gn_layer_3(yy_avg1)

        yy=yy_avg1+yy_max1
        yy=self.conv_channel1_4(yy)
        yy=self.gn_layer_4(yy)
        yy=self.sigmoid_gn(yy)

        record_tmp = yy.clone()
        record_tmp = record_tmp.detach().cpu()
        if self.training:
            pass
        else:
            if yy.shape[0]==1:
                record_list.append(record_tmp.numpy())
        return x+x*yy.expand_as(x)

def create_modules_attention(module_defs,prune_idx):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    hyperparams.update({
        'batch': int(hyperparams['batch']),
        'subdivisions': int(hyperparams['subdivisions']),
        'width': int(hyperparams['width']),
        'height': int(hyperparams['height']),
        'channels': int(hyperparams['channels']),
        'optimizer': hyperparams.get('optimizer'),
        'momentum': float(hyperparams['momentum']),
        'decay': float(hyperparams['decay']),
        'learning_rate': float(hyperparams['learning_rate']),
        'burn_in': int(hyperparams['burn_in']),
        'max_batches': int(hyperparams['max_batches']),
        'policy': hyperparams['policy'],
        'lr_steps': list(zip(map(int,   hyperparams["steps"].split(",")),
                             map(float, hyperparams["scales"].split(","))))
    })
    assert hyperparams["height"] == hyperparams["width"], \
        "Height and width should be equal! Non square images are padded with zeros."
    output_filters = [hyperparams["channels"]]
    module_list = nn.ModuleList()
    prune_index=0
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()
        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            stride=int(module_def["stride"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}",
                                   nn.BatchNorm2d(filters, momentum=0.1, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))
            if module_def["activation"] == "mish":
                modules.add_module(f"mish_{module_i}", Mish())

            if module_i==prune_idx[prune_index]:
                modules.add_module(f"attention",SELayer(filters))
                prune_index+=1
                if prune_index==len(prune_idx):
                    prune_index=0

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
                                   padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers]) // int(module_def.get("groups", 1))
            modules.add_module(f"route_{module_i}", nn.Sequential())

        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", nn.Sequential())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            # Define detection layer
            yolo_layer = YOLOLayer_attention(anchors, num_classes)
            modules.add_module(f"yolo_{module_i}", yolo_layer)

        # elif module_def["type"]=="attention":
        #     channels=int(module_def["channels"])
        #     modules.add_module(f"attention",SELayer(channels))

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class Mish(nn.Module):
    """ The MISH activation function (https://github.com/digantamisra98/Mish) """

    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class YOLOLayer_attention(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes):
        super(YOLOLayer_attention, self).__init__()
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.no = num_classes + 5  # number of outputs per anchor
        self.grid = torch.zeros(1)  # TODO

        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2)
        self.register_buffer('anchors', anchors)
        self.register_buffer(
            'anchor_grid', anchors.clone().view(1, -1, 1, 1, 2))
        self.stride = None

    def forward(self, x, img_size):
        stride = img_size // x.size(2)
        self.stride = stride
        bs, _, ny, nx = x.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x = x.view(bs, self.num_anchors, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        if not self.training:  # inference
            if self.grid.shape[2:4] != x.shape[2:4]:
                self.grid = self._make_grid(nx, ny).to(x.device)

            x[..., 0:2] = (x[..., 0:2].sigmoid() + self.grid) * stride  # xy
            x[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchor_grid # wh
            x[..., 4:] = x[..., 4:].sigmoid()
            x = x.view(bs, -1, self.no)

        return x

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Darknet_attention(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path,prune_idx):
        super(Darknet_attention, self).__init__()

        if isinstance(config_path, str):
            self.module_defs = parse_model_config(config_path)
        elif isinstance(config_path, list):
            self.module_defs = config_path
        self.prune_idx=prune_idx
        self.hyperparams, self.module_list = create_modules_attention(self.module_defs,self.prune_idx)
        self.yolo_layers = [layer[0]
                            for layer in self.module_list if isinstance(layer[0], YOLOLayer_attention)]
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x):
        img_size = x.size(2)
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                combined_outputs = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
                group_size = combined_outputs.shape[1] // int(module_def.get("groups", 1))
                group_id = int(module_def.get("group_id", 0))
                x = combined_outputs[:, group_size * group_id : group_size * (group_id + 1)] # Slice groupings used by yolo v4
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x = module[0](x, img_size)
                yolo_outputs.append(x)
            layer_outputs.append(x)
        return yolo_outputs if self.training else torch.cat(yolo_outputs, 1)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            # First five are header values
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        # If the weights file has a cutoff, we can find out about it by looking at the filename
        # examples: darknet53.conv.74 -> cutoff is 74
        filename = os.path.basename(weights_path)
        if ".conv." in filename:
            try:
                cutoff = int(filename.split(".")[-1])  # use last part of filename
            except ValueError:
                pass

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(
                    weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()


def load_model_attention(model_path, weights_path=None):
    """Loads the yolo model from file.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :return: Returns model
    :rtype: Darknet
    """
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")  # Select device for inference
    model = load_model(model_path,weights_path)
    _, _, prune_idx = parse_module_defs(model.module_defs)
    # print(prune_idx)

    model_attention = Darknet_attention(model_path,prune_idx).to(device)

    model_attention.apply(weights_init_normal)

    # If pretrained weights are specified, start from checkpoint or weight file
    # if weights_path:
    #     if weights_path.endswith(".pth"):
    #         # Load checkpoint weights
    #         model.load_state_dict(torch.load(weights_path, map_location=device))
    #     else:
    #         # Load darknet weights
    #         model.load_darknet_weights(weights_path)

    for i,(module_def, module,module_attention) in enumerate(zip(model.module_defs, model.module_list,model_attention.module_list)):
        if module_def["type"]=="convolutional":
            model_module_attention=model_attention.module_list[i]
            model_module=model.module_list[i]

            model_module_attention_conv,model_module_conv=model_module_attention[0],model_module[0]
            model_module_attention_conv.weight.data=model_module_conv.weight.data.clone()

            model_module_attention_conv.weight.requires_grad=False

            if module_def["batch_normalize"]==1:
                model_module_attention_bn,model_module_bn=model_module_attention[1],model_module[1]

                model_module_attention_bn.weight.data=model_module_bn.weight.data.clone()
                model_module_attention_bn.bias.data = model_module_bn.bias.data.clone()
                model_module_attention_bn.running_mean.data = model_module_bn.running_mean.data.clone()
                model_module_attention_bn.running_var.data = model_module_bn.running_var.data.clone()

                model_module_attention_bn.weight.requires_grad=False
                model_module_attention_bn.bias.requires_grad = False
            if module_def["batch_normalize"] == 0:
                model_module_attention_conv.bias.data = model_module_conv.bias.data.clone()
                model_module_attention_conv.bias.requires_grad = False
    return model_attention

def load_model_attention_pretrained(model_path, weights_path=None):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")  # Select device for inference
    model = load_model(model_path)
    _, _, prune_idx = parse_module_defs(model.module_defs)
    # print(prune_idx)

    model_attention = Darknet_attention(model_path, prune_idx).to(device)

    # model_attention.apply(weights_init_normal)
    model_attention.load_state_dict(torch.load(weights_path, map_location=device))
    return model_attention


# yolo_se=load_model_attention("G:\pytorchyolov3\pytorchyolo3_2\PyTorch-YOLOv3-master\config\yolov3.cfg")
# from torchsummary import summary
# summary(yolo_se, input_size=(3, yolo_se.hyperparams['height'], yolo_se.hyperparams['height']))