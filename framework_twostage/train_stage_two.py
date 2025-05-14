import os
import argparse
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import models
import torchshow as ts
from collections import defaultdict
import sys
from calculate_mAP import calculate_mAP
from network_stage2 import StageTwo, StageTwoWithPaired
from dataloader_stage2 import create_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description='train second stage model')
    parser.add_argument('--data-dir', type=str, default='runs/dataset_stage_two', help='dataset directory')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='number of data loading threads')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--output-dir', type=str, default='runs/stage2_model', help='output directory')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint for resume training')
    parser.add_argument('--device', type=str, default='cuda', help='training device')
    parser.add_argument('--model', type=str, default='resnet50', help='base model')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], 
                       help='running mode: train(including training and testing) or test(only testing)')
    parser.add_argument('--weights', type=str, default=None, 
                       help='weight file path for test mode, if not specified, the output-dir/best_model.pth will be used')
    parser.add_argument('--paired', action='store_true', default=True, 
                       help='whether to use paired dataset (crop and original input simultaneously)')
    parser.add_argument('--warmup-epochs', type=int, default=3, help='number of epochs for learning rate warmup')
    parser.add_argument('--warmup-factor', type=float, default=0.1, help='start factor for learning rate warmup')
    parser.add_argument('--step-size', type=int, default=20, help='step size for learning rate decay')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay factor')
    
    return parser.parse_args()
class WarmupLRScheduler:
    def __init__(self, optimizer, warmup_epochs, warmup_factor, after_scheduler=None):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.after_scheduler = after_scheduler
        self.finished_warmup = False
        self.epoch = 1
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self._set_warmup_lr()
    
    def _set_warmup_lr(self):
        if self.epoch > self.warmup_epochs:
            if not self.finished_warmup:
                self.finished_warmup = True
                for param_group, lr in zip(self.optimizer.param_groups, self.base_lrs):
                    param_group['lr'] = lr
        else:
            warmup_factor = self.warmup_factor + (1 - self.warmup_factor) * (self.epoch / self.warmup_epochs)
            for param_group, lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = lr * warmup_factor
    
    def step(self):
        if self.finished_warmup and self.after_scheduler:
            self.after_scheduler.step()
        else:
            self.epoch += 1
            self._set_warmup_lr()
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

def evaluate_with_metrics(model, dataloader, device, paired=True, data_dir=None, split='val'):
    model.eval()
    data_dir = Path(data_dir) if data_dir else Path("runs/dataset_stage_two")
    error_log_path = Path(f"class3_errors_{split}.txt")
    with open(error_log_path, "w") as error_log:
        error_log.write("# Record of class 3 detection errors\n")
        error_log.write("# Format: [video name_frame ID] [error type] [predicted tool ID] [GT tool ID] [IoU]\n\n")
    all_results = defaultdict(lambda: {'preds': [], 'gts': []})
    video_frames = defaultdict(set)
    total_frames = 0
    total_inference_time = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if paired:
                inputs = {
                    'crop_img': batch['crop_img'].to(device),
                    'original_img': batch['original_img'].to(device)
                }
                crop_labels = batch['crop_label']
                batch_tool_probs = []
                for label in crop_labels:
                    if 'tool_probs' in label:
                        batch_tool_probs.append(label['tool_probs'])
                
                if batch_tool_probs:
                    inputs['tool_probs'] = torch.stack(batch_tool_probs).to(device)
                start_time = time.time()
                outputs = model(inputs)
                torch.cuda.synchronize()
                end_time = time.time()
                inference_time = end_time - start_time
                batch_size = inputs['crop_img'].size(0)
                total_frames += batch_size
                total_inference_time += inference_time
                tool_probs = outputs['tool_logits']
                _, tool_ids = torch.max(tool_probs, 1)
                _, action_ids = torch.max(outputs['action_logits'], 1)
                _, target_ids = torch.max(outputs['target_logits'], 1)
                _, triplet_ids = torch.max(outputs['triplet_logits'], 1)
                batch_size = len(crop_labels)
                for i in range(batch_size):
                    label = crop_labels[i]
                    video_name = label['video_name']
                    frame_id = label['frame_id']
                    image_type = label['image_type']
                    frame_key = f"{video_name}_{frame_id}"
                    video_frames[video_name].add(frame_key)
                    pred_tool_id = tool_ids[i].item()
                    pred_triplet_id = triplet_ids[i].item()
                    if image_type == 'pred_patch' and 'pred_bbox' in label:
                        pred_bbox = label['pred_bbox'].cpu().numpy()
                        if len(pred_bbox) > 0:
                            pred_score = label['pred_score'].item()
                            all_results[frame_key]['preds'].append({
                                'tool_id': pred_tool_id,
                                'action_id': action_ids[i].item(),
                                'target_id': target_ids[i].item(),
                                'triplet_id': pred_triplet_id,
                                'bbox': pred_bbox,
                                'score': pred_score,
                                'im_file': label.get('im_file', '')
                            })
                    if 'gt_bbox' in label and len(label['gt_bbox']) > 0:
                        gt_bbox = label['gt_bbox'].cpu().numpy()
                        if 'gt_tool_id' in label and len(label['gt_tool_id']) > 0:
                            gt_tool_id = label['gt_tool_id'][0].item()
                            gt_action_id = label['gt_action_id'][0].item() if 'gt_action_id' in label and len(label['gt_action_id']) > 0 else None
                            gt_target_id = label['gt_target_id'][0].item() if 'gt_target_id' in label and len(label['gt_target_id']) > 0 else None
                            gt_triplet_id = label['gt_triplet_id'][0].item() if 'gt_triplet_id' in label and len(label['gt_triplet_id']) > 0 else None
                            gt_data = {
                                'bbox': gt_bbox,
                                'im_file': label.get('im_file', '')
                            }
                            if gt_tool_id is not None:
                                gt_data['tool_id'] = gt_tool_id
                            else:
                                gt_data['tool_id'] = []
                            if gt_action_id is not None:
                                gt_data['action_id'] = gt_action_id
                            else:
                                gt_data['action_id'] = []
                            if gt_target_id is not None:
                                gt_data['target_id'] = gt_target_id
                            else:
                                gt_data['target_id'] = []
                            if gt_triplet_id is not None:
                                gt_data['triplet_id'] = gt_triplet_id
                            else:
                                gt_data['triplet_id'] = []
                            all_results[frame_key]['gts'].append(gt_data)
                

            else:
                raise NotImplementedError("Only paired data mode is supported")
    unpaired_file = data_dir / f"unpaired_originals_{split}.json"
    
    unpaired_originals = []
    if unpaired_file.exists():
        from find_unpaired_originals import load_unpaired_originals
        unpaired_originals = load_unpaired_originals(unpaired_file)
        
    else:
        print(f"Warning: Unpaired original image file not found: {unpaired_file}")
    for item in unpaired_originals:
        frame_key = item['key']
        video_name = item['video_name']
        video_frames[video_name].add(frame_key)
        all_results[frame_key]['preds'] = []
        if 'ground_truths' in item and item['ground_truths']:
            for gt in item['ground_truths']:
                if 'bbox' in gt:
                    gt_data = {'bbox': gt['bbox']}
                    gt_data['tool_id'] = gt['tool_id']
                    gt_data['action_id'] = gt['action_id']
                    gt_data['target_id'] = gt['target_id']
                    gt_data['triplet_id'] = gt['triplet_id']
                    all_results[frame_key]['gts'].append(gt_data)
    results = calculate_mAP(all_results)
    if total_frames > 0 and total_inference_time > 0:
        fps = total_frames / total_inference_time
        
        if split == 'test':
            print("\nPerformance test results:")
            print(f"Total frames: {total_frames}")
            print(f"Total inference time: {total_inference_time:.4f} seconds")
            print(f"FPS: {fps:.2f} frames/second")
            results['fps'] = fps
    
    return results

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs, output_dir, num_triplets, num_tools, paired=True, data_dir=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    loss_file = output_dir / 'loss.json'
    loss_history = {
        'train': {'action': [], 'target': [], 'triplet': [], 'total': []},
        'val': {'action': [], 'target': [], 'triplet': [], 'total': []}
    }
    metrics_file = output_dir / 'metrics.json'
    metrics_history = {
        'val': {
            'action_acc': [], 
            'target_acc': [], 
            'triplet_acc': [],
            'mAP_i': [],
            'mAP_v': [],
            'mAP_t': [],
            'mAP_iv': [],
            'mAP_it': [],
            'mAP_ivt': []
        }
    }
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        model.train()
        
        running_loss = 0.0
        running_action_loss = 0.0
        running_target_loss = 0.0
        running_triplet_loss = 0.0
        for batch in tqdm(dataloaders['train']):
            if paired:
                inputs = {
                    'crop_img': batch['crop_img'].to(device),
                    'original_img': batch['original_img'].to(device)
                }
                labels = batch['crop_label']
                if 'tool_probs' in labels:
                    inputs['tool_probs'] = labels['tool_probs'].to(device)
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
            for k, v in labels.items():
                if isinstance(v, torch.Tensor):
                    labels[k] = v.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                action_loss = criterion(outputs['action_logits'], labels['action_id'])
                target_loss = criterion(outputs['target_logits'], labels['target_id'])
                triplet_loss = criterion(outputs['triplet_logits'], labels['triplet_id'])
                loss = action_loss + target_loss + triplet_loss
                loss.backward()
                optimizer.step()
            batch_size = labels['action_id'].size(0)
            running_loss += loss.item() * batch_size
            running_action_loss += action_loss.item() * batch_size
            running_target_loss += target_loss.item() * batch_size
            running_triplet_loss += triplet_loss.item() * batch_size
        scheduler.step()
        dataset_size = len(dataloaders['train'].dataset)
        epoch_loss = running_loss / dataset_size
        epoch_action_loss = running_action_loss / dataset_size
        epoch_target_loss = running_target_loss / dataset_size
        epoch_triplet_loss = running_triplet_loss / dataset_size
        loss_history['train']['total'].append(epoch_loss)
        loss_history['train']['action'].append(epoch_action_loss)
        loss_history['train']['target'].append(epoch_target_loss)
        loss_history['train']['triplet'].append(epoch_triplet_loss)
        
        print(f'训练阶段 Loss: {epoch_loss:.4f} - '
              f'Action: {epoch_action_loss:.4f}, Target: {epoch_target_loss:.4f}, '
              f'Triplet: {epoch_triplet_loss:.4f}')
        model.eval()
        
        detect_results = evaluate_with_metrics(
            model, 
            dataloaders['val'], 
            device, 
            paired=paired,
            data_dir=data_dir,
            split='val'
        )
        avg = detect_results['average']
        line = (
            f"i: {avg['mAP50_i']*100:.2f}%/{avg['mAP50-95_i']*100:.2f}% | "
            f"v: {avg['mAP50_v']*100:.2f}%/{avg['mAP50-95_v']*100:.2f}% | "
            f"t: {avg['mAP50_t']*100:.2f}%/{avg['mAP50-95_t']*100:.2f}% | "
            f"ivt: {avg['mAP50_ivt']*100:.2f}%/{avg['mAP50-95_ivt']*100:.2f}%"
        )
        print(f'Val Average mAP: {line}')
        metrics_history['val']['mAP_i'].append(avg['mAP50_i'])
        metrics_history['val']['mAP_v'].append(avg['mAP50_v'])
        metrics_history['val']['mAP_t'].append(avg['mAP50_t'])
        metrics_history['val']['mAP_ivt'].append(avg['mAP50_ivt'])
        curr_acc = avg['mAP50_ivt']
        if curr_acc > best_acc:
            best_acc = curr_acc
            torch.save(model.state_dict(), output_dir / 'best_yolotriplet_model.pth')
            print(f'New best mAP50_ivt: {best_acc:.4f}, save model')
        with open(loss_file, 'w') as f:
            json.dump(loss_history, f, indent=2)
        with open(metrics_file, 'w') as f:
            json.dump(metrics_history, f, indent=2)
    
    print(f"Epoch {epoch} finished")

    return model


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Dataset directory not found: {data_dir}")
        return
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    mappings = {}
    mappings_file = data_dir / 'class_mappings.json'
    if mappings_file.exists():
        with open(mappings_file, 'r') as f:
            mappings = json.load(f)
    else:
        print(f"Warning: Class mapping file not found: {mappings_file}")
    num_tools = len(mappings.get('tool_names', {}))
    num_actions = len(mappings.get('action_names', {}))
    num_targets = len(mappings.get('target_names', {}))
    num_triplets = len(mappings.get('triplet_names', {}))
    
    print(f"Number of classes: tool={num_tools}, action={num_actions}, target={num_targets}, triplet={num_triplets}")
    dataloaders = create_dataloaders(
        data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        paired=args.paired
    )
    model = StageTwoWithPaired(num_tools, num_actions, num_targets, num_triplets, base_model=args.model)
    print("Using paired data mode (crop+original)")
  
    
    model = model.to(device)
    if args.mode == 'train':
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        after_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        scheduler = WarmupLRScheduler(
            optimizer, 
            warmup_epochs=args.warmup_epochs, 
            warmup_factor=args.warmup_factor, 
            after_scheduler=after_scheduler
        )
        start_epoch = 0
        if args.resume:
            if os.path.exists(args.resume):
                print(f"Loading checkpoint: {args.resume}")
                checkpoint = torch.load(args.resume)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
                scheduler.epoch = start_epoch
                if start_epoch > args.warmup_epochs:
                    scheduler.finished_warmup = True
                    for _ in range(start_epoch - args.warmup_epochs):
                        after_scheduler.step()
                
                print(f"Resuming training, starting from epoch {start_epoch}")
            else:
                print(f"Warning: Checkpoint not found: {args.resume}")
        print("Starting training...")
        model = train_model(
            model,
            dataloaders,
            criterion,
            optimizer,
            scheduler,
            device,
            args.epochs - start_epoch,
            args.output_dir,
            num_triplets,
            num_tools,
            paired=args.paired,
            data_dir=args.data_dir
        )
        if 'test' in dataloaders:
            print("Using the best model to evaluate on the test set...")
            best_model_path = Path(args.output_dir) / 'best_model.pth'
            if os.path.exists(best_model_path):
                model.load_state_dict(torch.load(best_model_path))
                detect_results = evaluate_with_metrics(
                    model, 
                    dataloaders['test'], 
                    device, 
                    paired=args.paired,
                    data_dir=args.data_dir,
                    split='test'
                )
                if 'average' in detect_results:
                    avg = detect_results['average']
                    line = (
                        f"i: {avg['mAP50_i']*100:.2f}%/{avg['mAP50-95_i']*100:.2f}% | "
                        f"v: {avg['mAP50_v']*100:.2f}%/{avg['mAP50-95_v']*100:.2f}% | "
                        f"t: {avg['mAP50_t']*100:.2f}%/{avg['mAP50-95_t']*100:.2f}% | "
                        f"ivt: {avg['mAP50_ivt']*100:.2f}%/{avg['mAP50-95_ivt']*100:.2f}%"
                    )
                    print(f'Test Average mAP: {line}')
                    for video_name, video_results in detect_results.items():
                        if video_name == 'average' or video_name == 'fps':
                            continue
                        if isinstance(video_results, dict) and 'mAP50_i' in video_results:
                            video_line = (
                                f"i: {video_results['mAP50_i']*100:.2f}%/{video_results['mAP50-95_i']*100:.2f}% | "
                                f"v: {video_results['mAP50_v']*100:.2f}%/{video_results['mAP50-95_v']*100:.2f}% | "
                                f"t: {video_results['mAP50_t']*100:.2f}%/{video_results['mAP50-95_t']*100:.2f}% | "
                                f"ivt: {video_results['mAP50_ivt']*100:.2f}%/{video_results['mAP50-95_ivt']*100:.2f}%"
                            )
                            print(f'{video_name}: {video_line}')
                else:
                    print("Warning: No average key in test results, cannot print mAP results.")
            else:
                print(f"Warning: Best model file not found: {best_model_path}")
        else:
            print("Warning: No test set available, skipping final test evaluation")
        
        print(f"Training completed, model saved in: {args.output_dir}")
        
    else:
        weights_path = args.weights
        if weights_path is None:
            weights_path = os.path.join(args.output_dir, 'best_model.pth')
        
        if not os.path.exists(weights_path):
            print(f"Error: Weight file not found: {weights_path}")
            return
        print(f"Loading weights: {weights_path}")
        model.load_state_dict(torch.load(weights_path))
        if 'test' not in dataloaders:
            print("Error: No available test dataset")
            return
        
        detect_results = evaluate_with_metrics(
            model, 
            dataloaders['test'], 
            device, 
            paired=args.paired,
            data_dir=args.data_dir,
            split='test'
        )
    
        if 'average' in detect_results:
            avg = detect_results['average']
            line = (
                f"i: {avg['mAP50_i']*100:.2f}%/{avg['mAP50-95_i']*100:.2f}% | "
                f"v: {avg['mAP50_v']*100:.2f}%/{avg['mAP50-95_v']*100:.2f}% | "
                f"t: {avg['mAP50_t']*100:.2f}%/{avg['mAP50-95_t']*100:.2f}% | "
                f"ivt: {avg['mAP50_ivt']*100:.2f}%/{avg['mAP50-95_ivt']*100:.2f}%"
            )
            print(f'Test Average mAP: {line}')
            for video_name, video_results in detect_results.items():
                if video_name == 'average' or video_name == 'fps':
                    continue
                if isinstance(video_results, dict) and 'mAP50_i' in video_results:
                    video_line = (
                        f"i: {video_results['mAP50_i']*100:.2f}%/{video_results['mAP50-95_i']*100:.2f}% | "
                        f"v: {video_results['mAP50_v']*100:.2f}%/{video_results['mAP50-95_v']*100:.2f}% | "
                        f"t: {video_results['mAP50_t']*100:.2f}%/{video_results['mAP50-95_t']*100:.2f}% | "
                        f"ivt: {video_results['mAP50_ivt']*100:.2f}%/{video_results['mAP50-95_ivt']*100:.2f}%"
                    )
                    print(f'{video_name}: {video_line}')
        else:
            print("Warning: No average key in test results, cannot print mAP results.")
        combined_results = {
            'detection': detect_results
        }
        results_filename = f'test_results_{Path(weights_path).stem}.json'
        with open(Path(args.output_dir) / results_filename, 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        print(f"Test completed, results saved in: {os.path.join(args.output_dir, results_filename)}")

if __name__ == '__main__':
    main()
