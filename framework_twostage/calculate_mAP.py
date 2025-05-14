import torch
import numpy as np
from collections import defaultdict
from ultralytics.utils.metrics import ap_per_class, box_iou
from component_metrics import compute_frame_component_tp_V2

def calculate_mAP(all_results):
    iouv = torch.linspace(0.5, 0.95, 10)
    video_stats = defaultdict(lambda: {
        'tp_ivt': [], 'conf': [], 'pred_cls_ivt': [], 'target_cls_ivt': [],
        'pred_boxes': [], 'target_boxes': [],
        'tp_i': [], 'tp_v': [], 'tp_t': [],
        'pred_cls_i': [], 'pred_cls_v': [], 'pred_cls_t': [],
        'target_cls_i': [], 'target_cls_v': [], 'target_cls_t': []
    })
    
    components = ['i', 'v', 't', 'ivt']
    for frame_key, frame_data in all_results.items():
        video_name = frame_key.split('_')[0]
        preds = frame_data['preds']
        gts = frame_data['gts']
        if not preds and not gts:
            continue
        if preds:
            pred_boxes = torch.from_numpy(np.array([p['bbox'] for p in preds]))
            pred_scores = torch.tensor([p['score'] for p in preds])
            pred_triplet_ids = torch.tensor([p['triplet_id'] for p in preds])
            pred_tool_ids = torch.tensor([p['tool_id'] for p in preds])
            pred_action_ids = torch.tensor([p['action_id'] for p in preds])
            pred_target_ids = torch.tensor([p['target_id'] for p in preds])
        else:
            pred_boxes = torch.zeros((0, 4))
            pred_scores = torch.zeros(0)
            pred_triplet_ids = torch.zeros(0, dtype=torch.long)
            pred_tool_ids = torch.zeros(0, dtype=torch.long)
            pred_action_ids = torch.zeros(0, dtype=torch.long)
            pred_target_ids = torch.zeros(0, dtype=torch.long)
        if gts:
            gt_boxes = torch.tensor([g['bbox'] for g in gts])
            gt_triplet_ids = torch.tensor([g['triplet_id'] for g in gts])
            gt_tool_ids = torch.tensor([g['tool_id'] for g in gts])
            gt_action_ids = torch.tensor([g['action_id'] for g in gts])
            gt_target_ids = torch.tensor([g['target_id'] for g in gts])
        else:
            gt_boxes = torch.zeros((0, 4))
            gt_triplet_ids = torch.zeros(0, dtype=torch.long)
            gt_tool_ids = torch.zeros(0, dtype=torch.long)
            gt_action_ids = torch.zeros(0, dtype=torch.long)
            gt_target_ids = torch.zeros(0, dtype=torch.long)
        comp_pred_cls_dict = {
            'i': pred_tool_ids,
            'v': pred_action_ids,
            't': pred_target_ids,
            'ivt': pred_triplet_ids
        }
        comp_target_cls_dict = {
            'i': gt_tool_ids,
            'v': gt_action_ids,
            't': gt_target_ids,
            'ivt': gt_triplet_ids
        }
        comp_tp_dict = compute_frame_component_tp_V2(
            pred_boxes, gt_boxes, comp_pred_cls_dict, comp_target_cls_dict, iouv
        )
        video_stats[video_name]['pred_boxes'].append(pred_boxes)
        video_stats[video_name]['target_boxes'].append(gt_boxes)
        video_stats[video_name]['conf'].append(pred_scores)
        video_stats[video_name]['pred_cls_ivt'].append(pred_triplet_ids)
        video_stats[video_name]['target_cls_ivt'].append(gt_triplet_ids)
        video_stats[video_name]['tp_ivt'].append(comp_tp_dict['tp_ivt'])
        for component in ['i', 'v', 't']:
            video_stats[video_name][f'tp_{component}'].append(comp_tp_dict[f'tp_{component}'])
            video_stats[video_name][f'pred_cls_{component}'].append(comp_pred_cls_dict[component])
            video_stats[video_name][f'target_cls_{component}'].append(comp_target_cls_dict[component])
    results = {}
    for video_name, stats in video_stats.items():
        processed_stats = {}
        for k in ['tp_ivt', 'conf', 'pred_cls_ivt', 'target_cls_ivt', 'pred_boxes', 'target_boxes']:
            if len(stats[k]) > 0:
                processed_stats[k] = torch.cat(stats[k], 0)
            else:
                if k in ['conf', 'pred_cls_ivt', 'target_cls_ivt']:
                    processed_stats[k] = torch.zeros(0)
                elif k.startswith('tp'):
                    processed_stats[k] = torch.zeros((0, len(iouv)), dtype=torch.bool)
                else:
                    processed_stats[k] = torch.zeros((0, 4))
        for component in components:
            tp_key = f'tp_{component}'
            pred_cls_key = f'pred_cls_{component}'
            target_cls_key = f'target_cls_{component}'
            if len(stats[tp_key]) > 0:
                processed_stats[tp_key] = torch.cat(stats[tp_key], 0)
            else:
                processed_stats[tp_key] = torch.zeros((0, len(iouv)), dtype=torch.bool)
            if len(stats[pred_cls_key]) > 0:
                processed_stats[pred_cls_key] = torch.cat(stats[pred_cls_key], 0)
            else:
                processed_stats[pred_cls_key] = torch.zeros(0)
            if len(stats[target_cls_key]) > 0:
                processed_stats[target_cls_key] = torch.cat(stats[target_cls_key], 0)
            else:
                processed_stats[target_cls_key] = torch.zeros(0)
        results[video_name] = {}
        if processed_stats['conf'].numel() == 0 or processed_stats['target_cls_ivt'].numel() == 0:
            for component in components:
                results[video_name][f'mAP50_{component}'] = 0.0
                results[video_name][f'mAP50-95_{component}'] = 0.0
            continue
        for component in components:
            tp_key = f'tp_{component}'
            pred_cls_key = f'pred_cls_{component}'
            target_cls_key = f'target_cls_{component}'
            if all(len(processed_stats.get(k, [])) > 0 for k in [tp_key, pred_cls_key, target_cls_key]):
                comp_tp = processed_stats[tp_key]
                comp_pred_cls = processed_stats[pred_cls_key]
                comp_target_cls = processed_stats[target_cls_key]
                comp_tp_arr, comp_fp_arr, comp_p, comp_r, comp_f1, comp_ap, _, _, _, _, _, _ = ap_per_class(
                    comp_tp.cpu().numpy(),
                    processed_stats['conf'].cpu().numpy(),
                    comp_pred_cls.cpu().numpy(),
                    comp_target_cls.cpu().numpy(),
                    plot=False
                )
                if len(comp_ap) > 0:
                    results[video_name][f'mAP50_{component}'] = float(comp_ap[:, 0].mean())
                    results[video_name][f'mAP50-95_{component}'] = float(comp_ap.mean())
                else:
                    results[video_name][f'mAP50_{component}'] = 0.0
                    results[video_name][f'mAP50-95_{component}'] = 0.0
                results[video_name][f'tp_{component}'] = comp_tp_arr.tolist()
                results[video_name][f'fp_{component}'] = comp_fp_arr.tolist()
                results[video_name][f'p_{component}'] = comp_p.tolist()
                results[video_name][f'r_{component}'] = comp_r.tolist()
                results[video_name][f'f1_{component}'] = comp_f1.tolist()
                results[video_name][f'ap_{component}'] = comp_ap.tolist()
            else:
                results[video_name][f'mAP50_{component}'] = 0.0
                results[video_name][f'mAP50-95_{component}'] = 0.0
                results[video_name][f'tp_{component}'] = []
                results[video_name][f'fp_{component}'] = []
                results[video_name][f'p_{component}'] = []
                results[video_name][f'r_{component}'] = []
                results[video_name][f'f1_{component}'] = []
                results[video_name][f'ap_{component}'] = []
    if results:
        results['average'] = {}
        metrics = []
        for component in components:
            metrics.append(f'mAP50_{component}')
            metrics.append(f'mAP50-95_{component}')
        
        for metric in metrics:
            values = [v[metric] for v in results.values() if metric in v and v != results['average']]
            if values:
                results['average'][metric] = float(np.mean(values))
            else:
                results['average'][metric] = 0.0
    
    return results 