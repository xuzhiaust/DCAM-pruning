from pytorchyolo.model_attention import load_model_attention,record_list,load_model_attention_pretrained
from pytorchyolo.utils.utils import *
from pytorchyolo.models import load_model
import torch
import numpy as np
from copy import deepcopy
from pytorchyolo.test import _evaluate,_create_validation_data_loader
import time
from pytorchyolo.utils.prune_utils import *
from pytorchyolo.utils.parse_config import *
from pytorchyolo.utils.augmentations import AUGMENTATION_TRANSFORMS
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import os

import time
from torch.cuda.amp import autocast as autocast
img_path="mydata/images"
list_img=os.listdir(img_path)
class opt():
    model_def = "config/yolov3.cfg"
    data_config = "mydata/person.data"
    model = 'checkpoints/yolov3_ckpt.pth'
    model_attention='checkpoints/yolov3_attention_ckpt_23500.pth'

    model_pp='config/prune_0.88_yolov3.cfg'
    yolo_tiny='config/yolov3-tiny.cfg'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_attention=load_model_attention_pretrained(opt.model_def,opt.model_attention)
yolo=load_model(opt.model_def,opt.model).to(device)

yolo_pp=load_model(opt.model_pp)
yolo_tiny=load_model(opt.yolo_tiny)

# from torchsummaryX import summary
#
# tensor=torch.rand(1,3,416,416).cuda()
# flops=summary(yolo_tiny,tensor)

# data_config=parse_data_config(opt.data_config)
# valid_path=data_config["valid"]
# class_names=load_classes(data_config["names"])
# validation_dataloader = _create_validation_data_loader(
#     valid_path,
#     batch_size=16,
#     img_size=416,
#     n_cpu=0)
#
# eval_model=lambda model:_evaluate(model,validation_dataloader,class_names,img_size=416,iou_thres=0.5,conf_thres=0.1, nms_thres=0.5, verbose=True)
# obtain_num_parameters=lambda model:sum([param.nelement() for param in model.parameters()])
#
#
# origin_model_metric = eval_model(yolo)
# origin_nparameters = obtain_num_parameters(yolo)
# #为剪枝做准备
# CBL_idx, Conv_idx, prune_idx=parse_module_defs(yolo_attention.module_defs)
# # count=0
# t1=time.time()
# for img_name in list_img:
#     img=cv2.imread(img_path+'/'+img_name)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = img.astype(np.float32)
#     img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
#     yolo_pp.eval()
#     out = yolo_pp(img)
#     # count+=1
#     # print(count)
# t2=time.time()
# print("time:",t2-t1)

# def record_sum(record_list,prune_idx):
#     max_index=len(prune_idx)-1
#     record_sum_list=[]
#     for i in range(len(record_list)):
#         if i<=max_index:
#             record_sum_list.append(record_list[i])
#         else:
#             record_sum_list[i%len(prune_idx)]=record_sum_list[i%len(prune_idx)]+record_list[i]
#     return record_sum_list
# record_sum_list=record_sum(record_list,prune_idx)
# def attention_record_list(record_sum_list):
#     size_list=[record_sum_list[idx].shape[1] for idx in range(len(record_sum_list))]
#     attention_weights=torch.zeros(sum(size_list),dtype=torch.float32)
#     index=0
#     for i in range(len(record_sum_list)):
#         for idx in range(record_sum_list[i].shape[1]):
#             attention_weights[index]=torch.from_numpy(record_sum_list[i][0][idx][0])
#             index += 1
#     return attention_weights
# attention_weights=attention_record_list(record_sum_list)
#
# # with open('attention_weights_single.txt','a') as f:
# #     for i in range(len(attention_weights)):
# #         xx=attention_weights[i].clone().numpy()
# #         f.write(str(xx)+',')
#
#
# sorted_attention=torch.sort(attention_weights)[0]
# highest_thre=[]
#
# for idx in range(len(record_sum_list)):
#     highest_thre.append(record_sum_list[idx].max())
#
# # with open('attention_layers.txt','a') as f:
# #     for i in range(len(highest_thre)):
# #         xx=highest_thre[i]
# #         f.write(str(xx)+',')
#
#
# highest_thre=min(highest_thre).tolist()
# # highest_thre_torch=torch.from_numpy(np.array(highest_thre))
# percent_limit=(sorted_attention==highest_thre).nonzero()[0].item()/len(sorted_attention)
#
# def get_threshold(sorted_attention, percent=.0):
#     thre_index = int(len(sorted_attention) * percent)
#     thre = sorted_attention[thre_index]
#     print(f'Channels with Gamma value less than {thre:.4f} are pruned!')
#     return thre
#
# percent = 0.4
# threshold = get_threshold( sorted_attention, percent)
#
# def obtain_filters_mask(model,thre, CBL_idx, prune_idx):
#     pruned = 0
#     total = 0
#     num_filters = []
#     filters_mask = []
#     index=0
#     for idx in CBL_idx:
#         record_sum_weight = record_sum_list[index]
#         bn_module = model.module_list[idx][1]
#         if idx in prune_idx:
#             mask = obtatin_attention_mask(record_sum_weight, thre).cpu().numpy()
#             remain = int(mask.sum())
#             pruned = pruned + mask.shape[0] - remain
#
#             if remain == 0:
#                 print("Channels would be all pruned!")
#                 raise Exception
#
#             print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
#                   f'remaining channel: {remain:>4d}')
#             index+=1
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
# num_filters, filters_mask = obtain_filters_mask(yolo,threshold, CBL_idx, prune_idx)
#
# CBLidx2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)}
#
# pruned_model = prune_model_keep_size(yolo, prune_idx, CBL_idx, CBLidx2mask)
# eval_model(pruned_model)
#
# compact_module_defs = deepcopy(yolo_attention.module_defs)
# for idx, num in zip(CBL_idx, num_filters):
#     assert compact_module_defs[idx]['type'] == 'convolutional'
#     compact_module_defs[idx]['filters'] = str(num)
#
# compact_model = load_model([yolo_attention.hyperparams.copy()] + compact_module_defs).to(device)
# compact_nparameters = obtain_num_parameters(compact_model)
#
# init_weights_from_loose_model(compact_model, pruned_model, CBL_idx, Conv_idx, CBLidx2mask)
#
# random_input = torch.rand((1, 3, yolo.hyperparams['height'], yolo.hyperparams['height'])).to(device)
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
# pruned_cfg_file = write_cfg(pruned_cfg_name, [yolo.hyperparams.copy()] + compact_module_defs)
# print(f'Config file has been saved: {pruned_cfg_file}')
#
# compact_model_name = opt.model.replace('/', f'/prune_{percent}_')
# torch.save(compact_model.state_dict(), compact_model_name)
# print(f'Compact model has been saved: {compact_model_name}')

