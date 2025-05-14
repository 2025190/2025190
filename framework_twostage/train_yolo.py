import os
import argparse
import yaml
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics
import json
import shutil
from tqdm import tqdm
import cv2
import torch
import torch.nn.functional as F
from ultralytics.utils import ops


def parse_args():
    parser = argparse.ArgumentParser(description='use dataset_tool_only dataset to train and test YOLO12 model')
    parser.add_argument('--data', type=str, default='dataset_tool_only.yaml', help='dataset config file path')
    parser.add_argument('--model', type=str, default='yolov12l.pt', help='model file or config file path')
    parser.add_argument('--epochs', type=int, default=200, help='training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--device', type=str, default='0', help='device selection, e.g. 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--name', type=str, default='yolov12_train', help='save result folder name')
    parser.add_argument('--exist-ok', action='store_true', help='whether to overwrite existing experiment folder')
    parser.add_argument('--patience', type=int, default=50, help='early stopping epochs')
    parser.add_argument('--test-only', action='store_true', help='only test model')
    parser.add_argument('--weights', type=str, default=None, help='weight file path for testing')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold for testing')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold for testing')
    parser.add_argument('--save-txt', action='store_true', help='save prediction results as txt files')
    parser.add_argument('--save-conf', action='store_true', help='save confidence scores')
    parser.add_argument('--project', type=str, default='runs', help='save result project name')
    parser.add_argument('--optimizer', type=str, default='auto', help='optimizer selection (SGD, Adam, AdamW, etc.)')
    parser.add_argument('--lr0', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='final learning rate = initial learning rate * lrf')
    parser.add_argument('--cos-lr', action='store_true', help='use cosine learning rate scheduling')
    parser.add_argument('--warmup-epochs', type=float, default=3.0, help='warmup epochs')
    parser.add_argument('--warmup-momentum', type=float, default=0.8, help='warmup momentum')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout ratio')
    parser.add_argument('--split', type=str, default='val', help='test set selection')
    parser.add_argument('--two-stage-prepare', action='store_true', help='enable two-stage preparation mode, save detection results for the second stage')
    parser.add_argument('--save-dir', type=str, default='stage_two_data', help='save two-stage data directory name')
    parser.add_argument('--triplet-data', type=str, default=None, help='triplet dataset config file path, for two-stage preparation mode')
    return parser.parse_args()


def validate_dataset(data_path):
    try:
        data_dir = Path(data_path).parent 
        with open(data_path, 'r') as f:
            data_config = yaml.safe_load(f)
        required_fields = ['train', 'val', 'nc', 'names']
        for field in required_fields:
            if field not in data_config:
                print(f"Warning: dataset config file missing required field '{field}'")
                return False
        if len(data_config['names']) != data_config['nc']:
            print(f"Warning: class number ({data_config['nc']}) does not match the length of class name list ({len(data_config['names'])})")
            return False
        train_path = data_config['train']
        val_path = data_config['val']
        train_dir = Path(train_path) if train_path.startswith('/') else data_dir / train_path
        val_dir = Path(val_path) if val_path.startswith('/') else data_dir / val_path
        
        if not train_dir.exists():
            print(f"Warning: training image directory does not exist: {train_dir}")
            return False
        if not val_dir.exists():
            print(f"Warning: validation image directory does not exist: {val_dir}")
            return False
        train_labels_dir = Path(str(train_dir).replace('images', 'labels'))
        val_labels_dir = Path(str(val_dir).replace('images', 'labels'))
        if not train_labels_dir.exists():
            print(f"Warning: training label directory does not exist: {train_labels_dir}")
            return False
        if not val_labels_dir.exists():
            print(f"Warning: validation label directory does not exist: {val_labels_dir}")
            return False
        
        print("Dataset validation passed!")
        return True
    except Exception as e:
        print(f"Error validating dataset: {e}")
        return False


def print_metrics(metrics, class_names=None):
    print("\n--- Overall evaluation metrics ---")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision (P): {metrics.box.mp:.4f}")
    print(f"Recall (R): {metrics.box.mr:.4f}")
    print(f"F1 score: {(2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-16)):.4f}")
    if class_names is not None and len(class_names) > 0 and hasattr(metrics, 'ap_class_index'):
        print("\n--- Evaluation metrics for each class ---")
        for i, idx in enumerate(metrics.ap_class_index):
            if idx < len(class_names) and i < len(metrics.box.ap50):
                cls_name = class_names[idx]
                ap50 = metrics.box.ap50[i]
                ap = metrics.box.ap[i]
                precision = metrics.box.p[i]
                recall = metrics.box.r[i]
                f1 = 2 * precision * recall / (precision + recall + 1e-16)
                
                print(f"Class {idx} ({cls_name}): AP50={ap50:.4f}, AP50-95={ap:.4f}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")


def train(args):
    model = YOLO(args.model)
    print(f"start training YOLO12 model, will train {args.epochs} epochs...")
    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'workers': args.workers,
        'name': args.name,
        'exist_ok': args.exist_ok,
        'patience': args.patience,
        'verbose': True,
        'project': args.project,
        'optimizer': args.optimizer,
        'lr0': args.lr0,
        'lrf': args.lrf,
        'cos_lr': args.cos_lr,
        'warmup_epochs': args.warmup_epochs,
        'warmup_momentum': args.warmup_momentum,
        'dropout': args.dropout
    }
    results = model.train(**train_args)
    print(f"training completed! model saved to {model.trainer.save_dir}")
    
    return model, results


def test(model, args):
    if args.weights:
        if os.path.exists(args.weights):
            model = YOLO(args.weights)
        else:
            print(f"Warning: weight file {args.weights} not found, using loaded model")
    print("start testing YOLO12 model...")
    test_args = {
        'data': args.data,
        'split': 'val',
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'conf': args.conf,
        'iou': args.iou,
        'verbose': True,
        'save_txt': args.save_txt,
        'save_conf': args.save_conf,
        'project': args.project,
        'name': args.name + '_test',
    }
    results = model.val(**test_args)
    class_names = []
    try:
        with open(args.data, 'r') as f:
            data_config = yaml.safe_load(f)
            if 'names' in data_config:
                class_names = data_config['names']
    except Exception as e:
        print(f"Warning: cannot read class names: {e}")
    print_metrics(results, class_names)
    
    return results


def prepare_two_stage_data(model, args):
    print(f"preparing two-stage detection data...")
    
    import patch
    save_dir = os.path.join(args.project, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(save_dir, split)
        os.makedirs(split_dir, exist_ok=True)
    try:
        with open(args.data, 'r') as f:
            tool_data_config = yaml.safe_load(f)
        triplet_data_path = args.triplet_data if args.triplet_data else args.data
        with open(triplet_data_path, 'r') as f:
            triplet_data_config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error: cannot load dataset config: {e}")
        return
    triplet_names = triplet_data_config.get('names', {})
    tool_names = triplet_data_config.get('tools', {})
    action_names = triplet_data_config.get('actions', {})
    target_names = triplet_data_config.get('targets', {})
    
    if not all([triplet_names, tool_names, action_names, target_names]):
        print(f"Warning: triplet dataset config missing necessary class mappings, please check {triplet_data_path}")
        if not args.triplet_data:
            print(f"Warning: please use --triplet-data parameter to specify the triplet dataset config file")
        return
    mappings = {
        'triplet_names': triplet_names,
        'tool_names': tool_names,
        'action_names': action_names,
        'target_names': target_names
    }
    
    with open(os.path.join(save_dir, 'class_mappings.json'), 'w') as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)
    for split in ['train', 'val', 'test']:
        if split in tool_data_config:
            data_config = tool_data_config
            print(f"using tool detection dataset config to process {split} data")
        elif split in triplet_data_config:
            data_config = triplet_data_config
            print(f"using triplet dataset config to process {split} data")
        else:
            print(f"Warning: {split} split not found in dataset config")
            continue
        
        split_dir = os.path.join(save_dir, split)
        images_path = os.path.join(Path(args.data).parent, data_config[split])
        if args.triplet_data and split in triplet_data_config:
            triplet_images_path = os.path.join(Path(args.triplet_data).parent, triplet_data_config[split])
            triplet_labels_path = triplet_images_path.replace('images', 'labels')
            labels_path = triplet_labels_path
            print(f"using triplet data set label path: {labels_path}")
        else:
            labels_path = images_path.replace('images', 'labels')
            print(f"using tool detection data set label path: {labels_path}")
        if not os.path.exists(images_path):
            print(f"Warning: {split} image directory not found: {images_path}")
            continue
        
        if not os.path.exists(labels_path):
            print(f"Warning: {split} label directory not found: {labels_path}")
            continue
        image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"Warning: {split} image directory not found: {images_path}")
            continue
        
        print(f"processing {len(image_files)} images in {split} set...")
        results_dir = os.path.join(split_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        for img_file in tqdm(image_files):
            img_path = os.path.join(images_path, img_file)
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(labels_path, label_file)
            
            if not os.path.exists(label_path):
                continue
            parts = os.path.splitext(img_file)[0].split('_')
            if len(parts) >= 2:
                video_name = '_'.join(parts[:-1])
                frame_id = parts[-1]
            else:
                video_name = os.path.splitext(img_file)[0]
                frame_id = "0"
            gt_labels = []
            with open(label_path, 'r') as f:
                for line in f:
                    values = line.strip().split()
                    if len(values) >= 8:
                        triplet_id = int(values[0])
                        tool_id = int(values[1])
                        action_id = int(values[2])
                        target_id = int(values[3])
                        cx, cy, w, h = map(float, values[4:8])
                        x1 = cx - w/2
                        y1 = cy - h/2
                        x2 = cx + w/2
                        y2 = cy + h/2
                        
                        gt_labels.append({
                            'triplet_id': triplet_id,
                            'tool_id': tool_id,
                            'action_id': action_id,
                            'target_id': target_id,
                            'bbox': [x1, y1, x2, y2]
                        })
                    else:
                        print(f"Warning: skipping incorrect label line: {line.strip()}")
            results = model.predict(img_path, conf=args.conf, iou=args.iou, verbose=False)[0]
            pred_boxes = results.boxes.xyxyn.cpu().numpy()
            pred_classes = results.boxes.cls.cpu().numpy()
            pred_confs = results.boxes.conf.cpu().numpy()
            all_class_probs = None
            if results.boxes.data.shape[1] > 6:
                all_class_probs = results.boxes.data[:, 6:].cpu().numpy()
            image_data = {
                'image_file': img_file,
                'video_name': video_name,
                'frame_id': frame_id,
                'predictions': [],
                'ground_truths': []
            }
            if len(pred_boxes) > 0:
                for i, (box, cls, conf) in enumerate(zip(pred_boxes, pred_classes, pred_confs)):
                    bbox = [float(coord) for coord in box]
                    pred = {
                        'bbox': bbox,
                        'score': float(conf),
                        'tool_id': int(cls)
                    }
                    if all_class_probs is not None:
                        pred['all_class_probs'] = [float(p) for p in all_class_probs[i]]
                    
                    image_data['predictions'].append(pred)
            for gt in gt_labels:
                bbox = [float(coord) for coord in gt['bbox']]
                gt_info = {
                    'bbox': bbox,
                    'tool_id': int(gt['tool_id']),
                    'action_id': int(gt['action_id']),
                    'target_id': int(gt['target_id']),
                    'triplet_id': int(gt['triplet_id'])
                }
                image_data['ground_truths'].append(gt_info)
            base_name = os.path.splitext(img_file)[0]
            result_file = os.path.join(results_dir, f'{base_name}.json')
            with open(result_file, 'w') as f:
                json.dump(image_data, f, indent=2)
        
        print(f"saved {split} results to {results_dir}, processed {len(image_files)} images")
    for split in ['train', 'val', 'test']:
        split_results_dir = os.path.join(save_dir, split, 'results')
        if os.path.exists(split_results_dir):
            json_files = [f for f in os.listdir(split_results_dir) if f.lower().endswith('.json')]
            index = {
                'total_images': len(json_files),
                'files': json_files
            }
            index_file = os.path.join(save_dir, split, f'{split}_index.json')
            with open(index_file, 'w') as f:
                json.dump(index, f, indent=2)
    
    print(f"two-stage detection data preparation completed, saved to {save_dir}")


def calculate_iou_numpy(box1, box2):
    box1 = np.array(box1)
    box2 = np.array(box2)
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def main():
    args = parse_args()
    if not validate_dataset(args.data):
        print("dataset validation failed, please check dataset format")
        return
    if args.two_stage_prepare:
        if args.weights is None:
            args.weights = os.path.join(args.project, args.name, 'weights/best.pt')
            if not os.path.exists(args.weights):
                print(f"Error: weight file {args.weights} not found, please provide a valid weight file path")
                return
        if args.triplet_data is None:
            possible_triplet_path = Path(args.data).parent / 'dataset_triplet.yaml'
            if possible_triplet_path.exists():
                args.triplet_data = str(possible_triplet_path)
                print(f"automatically detected triplet dataset config file: {args.triplet_data}")
            else:
                print(f"Warning: triplet dataset config file not provided, will try to use tool detection dataset config file({args.data}).")
                print(f"if you need complete triplet information, please use --triplet-data parameter to specify the config file path")
        
        model = YOLO(args.weights)
        prepare_two_stage_data(model, args)
        return
    if args.test_only:
        if args.weights is None:
            args.weights = os.path.join(args.project, args.name, 'weights/best.pt')
            if not os.path.exists(args.weights):
                print(f"Error: weight file {args.weights} not found, please provide a valid weight file path")
                return
        
        model = YOLO(args.weights)
        test(model, args)
        return
    model, train_results = train(args)
    best_weights = Path(model.trainer.save_dir) / 'weights' / 'best.pt'
    args.weights = str(best_weights)
    test_results = test(model, args)


if __name__ == '__main__':
    main()
