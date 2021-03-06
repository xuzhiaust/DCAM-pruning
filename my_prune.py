from pytorchyolo.models import *
from pytorchyolo.utils.utils import *
import torch
import numpy as np
from copy import deepcopy
from pytorchyolo.test import _evaluate,_create_validation_data_loader
import time
from pytorchyolo.utils.prune_utils import *
from pytorchyolo.utils.parse_config import *
from pytorchyolo.utils.augmentations import AUGMENTATION_TRANSFORMS
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#计算flops
from torchsummaryX import summary
#
# tensor=torch.rand(1,3,416,416).cuda()
# flops=summary(model,tensor)

class opt():
    model_def = "config/prune_0.4_yolov3.cfg"
    data_config = "mydata/person.data"
    model = 'checkpoints/yolov3_ckpt.pth'

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=load_model(opt.model_def).to(device)
# model.load_state_dict(torch.load(opt.model))

tensor=torch.rand(1,3,416,416).cuda()
flops=summary(model,tensor)

# data_config=parse_data_config(opt.data_config)
# valid_path=data_config["valid"]
# class_names=load_classes(data_config["names"])
#
# validation_dataloader = _create_validation_data_loader(
#     valid_path,
#     batch_size=16,
#     img_size=416,
#     n_cpu=0)
#
# eval_model=lambda model:_evaluate(model,validation_dataloader,class_names,img_size=416,iou_thres=0.5,conf_thres=0.1, nms_thres=0.5, verbose=True)
#
# obtain_num_parameters=lambda model:sum([param.nelement() for param in model.parameters()])
# # origin_model_metric = eval_model(model)
# # origin_nparameters = obtain_num_parameters(model)
#
# CBL_idx, Conv_idx, prune_idx=parse_module_defs(model.module_defs)
#
# #选择剪枝的编号
# bn_weights=gather_bn_weights(model.module_list,prune_idx)
# sorted_bn=torch.sort(bn_weights)[0]
#
# highest_thre=[]
# for idx in prune_idx:
#     highest_thre.append(model.module_list[idx][1].weight.data.abs().max().item())
#
# # with open('bn_weights_layers.txt','a') as f:
# #     for i in range(len(highest_thre)):
# #         xx=highest_thre[i]
# #         f.write(str(xx)+',')
#
# highest_thre=min(highest_thre)
#
# percent_limit=(sorted_bn==highest_thre).nonzero().item()/len(bn_weights)

# def prune_and_eval(model, sorted_bn, percent=.0):
#     model_copy = deepcopy(model)
#     thre_index = int(len(sorted_bn) * percent)
#     thre = sorted_bn[thre_index]
#
#     print(f'Channels with Gamma value less than {thre:.4f} are pruned!')
#
#     remain_num = 0
#     for idx in prune_idx:
#
#         bn_module = model_copy.module_list[idx][1]
#
#         mask = obtain_bn_mask(bn_module, thre)
#         print(mask.shape)
#
#         remain_num += int(mask.sum())
#         #剪枝,  其中.mul_()函数是in-place操作，将操作的结果直接存储在bn_module.weight.data中，因此就相当于直接完成了剪枝操作
#         #这里的剪枝操作仅仅是将被剪枝层的BN参数置为0
#         bn_module.weight.data.mul_(mask)
#
#     mAP = eval_model(model_copy)[2].mean()
#
#     print(f'Number of channels has been reduced from {len(sorted_bn)} to {remain_num}')
#     print(f'Prune ratio: {1-remain_num/len(sorted_bn):.3f}')
#     print(f'mAP of the pruned model is {mAP:.4f}')
#
#     return thre
#
# percent = 0.15
# threshold = prune_and_eval(model, sorted_bn, percent)
#
# def obtain_filters_mask(model, thre, CBL_idx, prune_idx):
#
#     pruned = 0
#     total = 0
#     num_filters = []
#     filters_mask = []
#     for idx in CBL_idx:
#         bn_module = model.module_list[idx][1]
#         if idx in prune_idx:
#             mask = obtain_bn_mask(bn_module, thre).cpu().numpy()
#             remain = int(mask.sum())
#             pruned = pruned + mask.shape[0] - remain
#
#             if remain == 0:
#                 print("Channels would be all pruned!")
#                 raise Exception
#
#             print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
#                   f'remaining channel: {remain:>4d}')
#         else:
#             mask = np.ones(bn_module.weight.data.shape)
#             remain = mask.shape[0]
#
#         total += mask.shape[0]
#         num_filters.append(remain)
#         filters_mask.append(mask.copy())
#
#     prune_ratio = pruned / total
#     print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')
#
#     return num_filters, filters_mask
#
# num_filters, filters_mask = obtain_filters_mask(model, threshold, CBL_idx, prune_idx)
#
# CBLidx2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)}
#
#
# pruned_model = prune_model_keep_size(model, prune_idx, CBL_idx, CBLidx2mask)
#
# eval_model(pruned_model)
#
# compact_module_defs = deepcopy(model.module_defs)
# for idx, num in zip(CBL_idx, num_filters):
#     assert compact_module_defs[idx]['type'] == 'convolutional'
#     compact_module_defs[idx]['filters'] = str(num)
#
# compact_model = load_model([model.hyperparams.copy()] + compact_module_defs).to(device)
# compact_nparameters = obtain_num_parameters(compact_model)
# #
# init_weights_from_loose_model(compact_model, pruned_model, CBL_idx, Conv_idx, CBLidx2mask)
#
# #%%
# random_input = torch.rand((1, 3, model.img_size, model.img_size)).to(device)
#
# def obtain_avg_forward_time(input, model, repeat=200):
#
#     model.eval()
#     start = time.time()
#     with torch.no_grad():
#         for i in range(repeat):
#             output = model(input)
#     avg_infer_time = (time.time() - start) / repeat
#
#     return avg_infer_time, output
#
# pruned_forward_time, pruned_output = obtain_avg_forward_time(random_input, pruned_model)
# compact_forward_time, compact_output = obtain_avg_forward_time(random_input, compact_model)
#
# diff = (pruned_output-compact_output).abs().gt(0.001).sum().item()
# if diff > 0:
#     print('Something wrong with the pruned model!')
#
# #%%
# # 在测试集上测试剪枝后的模型, 并统计模型的参数数量
# compact_model_metric = eval_model(compact_model)
#
# #%%
# # 比较剪枝前后参数数量的变化、指标性能的变化
# metric_table = [
#     ["Metric", "Before", "After"],
#     ["mAP", f'{origin_model_metric[2].mean():.6f}', f'{compact_model_metric[2].mean():.6f}'],
#     ["Parameters", f"{origin_nparameters}", f"{compact_nparameters}"],
#     ["Inference", f'{pruned_forward_time:.4f}', f'{compact_forward_time:.4f}']
# ]
# print(AsciiTable(metric_table).table)
#
# #%%
# # 生成剪枝后的cfg文件并保存模型
# pruned_cfg_name = opt.model_def.replace('/', f'/prune_{percent}_')
# pruned_cfg_file = write_cfg(pruned_cfg_name, [model.hyperparams.copy()] + compact_module_defs)
# print(f'Config file has been saved: {pruned_cfg_file}')
#
# compact_model_name = opt.model.replace('/', f'/prune_{percent}_')
# torch.save(compact_model.state_dict(), compact_model_name)
# print(f'Compact model has been saved: {compact_model_name}')