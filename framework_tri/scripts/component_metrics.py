from ultralytics.utils.metrics import ap_per_class
import numpy as np
import json
import os
from ultralytics.utils.metrics import box_iou
import torch
from ultralytics.engine.validator import BaseValidator


class TripletDisentangle:
    def __init__(self, map_path="/ssd/prostate/dataset_triplet/triplet_maps.txt"):
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"映射文件 {map_path} 不存在")
        self.mapping = {}
        with open(map_path, 'r') as f:
            next(f)
            for line in f:
                if line.strip():
                    values = line.strip().split(',')
                    ivt, i, v, t, iv, it = map(int, values)
                    self.mapping[ivt] = {
                        'ivt': ivt,
                        'i': i,
                        'v': v, 
                        't': t,
                        'iv': iv,
                        'it': it
                    }
        self.num_triplets = len(self.mapping)
        i_set = {info['i'] for info in self.mapping.values()}
        v_set = {info['v'] for info in self.mapping.values()}
        t_set = {info['t'] for info in self.mapping.values()}
        
        self.num_tools = max(i_set) + 1
        self.num_actions = max(v_set) + 1
        self.num_targets = max(t_set) + 1
        
    def extract(self, inputs, component="i"):
        if component == "ivt":
            return inputs
        outputs = np.zeros(len(inputs), dtype=np.int64)
        
        for i, triplet_idx in enumerate(inputs):
            if not (0 <= triplet_idx < self.num_triplets):
                raise ValueError(f"invalid triplet index: {triplet_idx}")
            
            outputs[i] = self.mapping[int(triplet_idx)][component]
        
        return outputs

def compute_component_maps(triplet_stats, component="i"):
    disentangler = TripletDisentangle()
    tp = triplet_stats["tp"]
    conf = triplet_stats["conf"]
    pred_cls = triplet_stats["pred_cls"]
    target_cls = triplet_stats["target_cls"]
    pred_boxes = triplet_stats["pred_boxes"]
    target_boxes = triplet_stats["target_boxes"]
    component_pred_cls = torch.tensor(disentangler.extract(pred_cls.cpu().numpy(), component), 
                                      device=pred_cls.device)
    component_target_cls = torch.tensor(disentangler.extract(target_cls.cpu().numpy(), component), 
                                        device=target_cls.device)
    component_tp = torch.zeros_like(tp)
    if len(component_pred_cls) == 0 or len(component_target_cls) == 0:
        return {
            f"mAP50_{component}": 0.0,
            f"mAP50-95_{component}": 0.0
        }
    iou_matrix = box_iou(target_boxes, pred_boxes)
    correct_class = component_target_cls[:, None] == component_pred_cls
    iou = iou_matrix * correct_class
    iou = iou.cpu().numpy()
    iou_thresholds = torch.linspace(0.5, 0.95, 10, device=component_pred_cls.device)
    for i, threshold in enumerate(iou_thresholds.cpu().tolist()):
        matches = np.nonzero(iou >= threshold)
        matches = np.array(matches).T
        
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            component_tp[matches[:, 1].astype(int), i] = 1
    component_tp_np = component_tp.cpu().numpy()
    conf_np = conf.cpu().numpy()
    component_pred_cls_np = component_pred_cls.cpu().numpy()
    component_target_cls_np = component_target_cls.cpu().numpy()
    results = ap_per_class(
        component_tp_np, 
        conf_np, 
        component_pred_cls_np, 
        component_target_cls_np,
        plot=False
    )
    ap = results[5]
    
    if len(ap) > 0:
        mAP50 = ap[:, 0].mean()
        mAP5095 = ap.mean()
        return {
            f"mAP50_{component}": float(mAP50),
            f"mAP50-95_{component}": float(mAP5095)
        }
    else:
        return {
            f"mAP50_{component}": 0.0,
            f"mAP50-95_{component}": 0.0
        }

def compute_frame_component_tp(pred_boxes, target_boxes, comp_pred_cls_dict, comp_target_cls_dict, iouv):
    n_preds = pred_boxes.shape[0]
    n_iou = len(iouv)
    tp_dict = {}
    for component in comp_pred_cls_dict.keys():
        tp_dict[f'tp_{component}'] = torch.zeros((n_preds, n_iou), dtype=torch.bool, device=pred_boxes.device)
    if n_preds == 0 or target_boxes.shape[0] == 0:
        return tp_dict
    iou_matrix = box_iou(target_boxes, pred_boxes)
    for component in comp_pred_cls_dict.keys():
        component_pred_cls = comp_pred_cls_dict[component]
        component_target_cls = comp_target_cls_dict[component]
        correct_class = component_target_cls[:, None] == component_pred_cls
        filtered_iou = iou_matrix * correct_class.float()
        filtered_iou_np = filtered_iou.cpu().numpy()
        for t, threshold in enumerate(iouv.cpu().tolist()):
            matches = np.nonzero(filtered_iou_np >= threshold)
            matches = np.array(matches).T
            
            if matches.shape[0]:
                matches = matches[filtered_iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                tp_dict[f'tp_{component}'][matches[:, 1], t] = True
    
    return tp_dict

def compute_frame_component_tp_V2(pred_boxes, target_boxes, comp_pred_cls_dict, comp_target_cls_dict, iouv):
    n_preds = pred_boxes.shape[0]
    n_iou = len(iouv)
    tp_dict = {}
    for component in comp_pred_cls_dict.keys():
        tp_dict[f'tp_{component}'] = torch.zeros((n_preds, n_iou), dtype=torch.bool, device=pred_boxes.device)
    if n_preds == 0 or target_boxes.shape[0] == 0:
        return tp_dict
    iou_matrix = box_iou(target_boxes, pred_boxes)
    for component in comp_pred_cls_dict.keys():
        component_pred_cls = comp_pred_cls_dict[component]
        component_target_cls = comp_target_cls_dict[component]
        correct_class = component_target_cls[:, None] == component_pred_cls
        filtered_iou = iou_matrix * correct_class
        filtered_iou_np = filtered_iou.cpu().numpy()
        for t, threshold in enumerate(iouv.cpu().tolist()):
            matches = np.nonzero(filtered_iou_np >= threshold)
            matches = np.array(matches).T
            
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[filtered_iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                tp_dict[f'tp_{component}'][matches[:, 1], t] = True
    
    return tp_dict