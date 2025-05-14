import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import traceback

class PairedDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.images_dir = self.data_dir / split / 'images'
        self.labels_dir = self.data_dir / split / 'labels'
        self.mappings = {}
        mappings_file = self.data_dir / 'class_mappings.json'
        if mappings_file.exists():
            with open(mappings_file, 'r') as f:
                self.mappings = json.load(f)
        self.original_images = {}
        self._load_original_images()
        self.data_records = []
        self._load_data_records()
        if transform is None:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
    
    def _load_original_images(self):
        json_files = list(self.labels_dir.glob('*_original.json'))
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                label_data = json.load(f)
            img_filename = json_file.stem + '.jpg'
            img_path = self.images_dir / img_filename
            
            if not img_path.exists():
                continue
            video_name = label_data.get('video_name', '')
            frame_id = label_data.get('frame_id', '')
            key = f"{video_name}_{frame_id}"
            self.original_images[key] = {
                'image_path': str(img_path),
                'label_path': str(json_file),
                'data': label_data,
                'im_file': img_filename
            }
    
    def _load_data_records(self):
        json_files = [f for f in self.labels_dir.glob('*.json') 
                     if not f.stem.endswith('_original')]
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                label_data = json.load(f)
            img_filename = json_file.stem + '.jpg'
            img_path = self.images_dir / img_filename
            
            if not img_path.exists():
                continue
            video_name = label_data.get('video_name', '')
            frame_id = label_data.get('frame_id', '')
            key = f"{video_name}_{frame_id}"
            if key in self.original_images:
                self.data_records.append({
                    'crop': {
                        'image_path': str(img_path),
                        'label_path': str(json_file),
                        'data': label_data,
                        'im_file': img_filename
                    },
                    'original': self.original_images[key]
                })
    
    def __len__(self):
        return len(self.data_records)
    
    def __getitem__(self, idx):
        record = self.data_records[idx]
        crop_img = Image.open(record['crop']['image_path']).convert('RGB')
        original_img = Image.open(record['original']['image_path']).convert('RGB')
        if self.transform:
            seed = random.randint(0, 2**32)
            random.seed(seed)
            torch.manual_seed(seed)
            crop_img_tensor = self.transform(crop_img)
            random.seed(seed)
            torch.manual_seed(seed)
            original_img_tensor = self.transform(original_img)
        else:
            to_tensor = transforms.ToTensor()
            crop_img_tensor = to_tensor(crop_img)
            original_img_tensor = to_tensor(original_img)
        if self.split == 'train':
            crop_label = self._process_train_item(record['crop']['data'], record['crop']['im_file'])
            original_label = {'im_file': record['original']['im_file']}
        else:
            crop_label = self._process_val_test_item(record['crop']['data'], record['crop']['im_file'])
            original_label = {'im_file': record['original']['im_file']}
        
        return {
            'crop_img': crop_img_tensor,
            'original_img': original_img_tensor,
            'crop_label': crop_label,
            'original_label': original_label
        }
    
    def _process_train_item(self, label_data, im_file):
        tool_id = label_data.get('tool_id', -1)
        action_id = label_data.get('action_id', -1)
        target_id = label_data.get('target_id', -1)
        triplet_id = label_data.get('triplet_id', -1)

        try:
            if tool_id == -1 or action_id == -1 or target_id == -1 or triplet_id == -1:
                assert False, "tool_id, action_id, target_id, triplet_id 不能为-1"
        except:
            print(f"tool_id: {tool_id}, action_id: {action_id}, target_id: {target_id}, triplet_id: {triplet_id}")
        all_class_probs = label_data.get('all_class_probs', [])
        if all_class_probs:
            tool_probs = torch.tensor(all_class_probs, dtype=torch.float32)
        else:
            num_tools = len(self.mappings.get('tool_names', {}))
            tool_probs = torch.zeros(num_tools, dtype=torch.float32)
            if tool_id < num_tools:
                tool_probs[tool_id] = 1.0
        label = {
            'tool_id': torch.tensor(tool_id, dtype=torch.long),
            'action_id': torch.tensor(action_id, dtype=torch.long),
            'target_id': torch.tensor(target_id, dtype=torch.long),
            'triplet_id': torch.tensor(triplet_id, dtype=torch.long),
            'tool_probs': tool_probs,
            'video_name': label_data.get('video_name', ''),
            'frame_id': label_data.get('frame_id', ''),
            'im_file': im_file,
            'image_type': label_data.get('image_type', '')
        }
        
        return label
    
    def _process_val_test_item(self, label_data, im_file):
        pred_bbox = label_data.get('pred_bbox', [0, 0, 1, 1])
        pred_score = label_data.get('pred_score', 0.0)
        pred_tool_id = label_data.get('pred_tool_id', 0)
        all_class_probs = label_data.get('all_class_probs', [])
        gt_bbox = label_data.get('gt_bbox', None)
        gt_tool_id = label_data.get('gt_tool_id', None)
        gt_action_id = label_data.get('gt_action_id', None)
        gt_target_id = label_data.get('gt_target_id', None)
        gt_triplet_id = label_data.get('gt_triplet_id', None)
        if all_class_probs:
            tool_probs = torch.tensor(all_class_probs, dtype=torch.float32)
        else:
            num_tools = len(self.mappings.get('tool_names', {}))
            tool_probs = torch.zeros(num_tools, dtype=torch.float32)
            if pred_tool_id < num_tools:
                tool_probs[pred_tool_id] = 1.0
        try:
            label = {
                'pred_bbox': torch.tensor(pred_bbox, dtype=torch.float32) if pred_bbox else torch.tensor([], dtype=torch.float32),
                'pred_score': torch.tensor(pred_score, dtype=torch.float32),
                'pred_tool_id': torch.tensor(pred_tool_id, dtype=torch.long),
                'tool_probs': tool_probs,
                'gt_bbox': torch.tensor(gt_bbox, dtype=torch.float32) if gt_bbox else torch.tensor([], dtype=torch.float32),
                'gt_tool_id': torch.tensor([gt_tool_id], dtype=torch.long) if gt_tool_id is not None else torch.tensor([], dtype=torch.long),
                'gt_action_id': torch.tensor([gt_action_id], dtype=torch.long) if gt_action_id is not None else torch.tensor([], dtype=torch.long),
                'gt_target_id': torch.tensor([gt_target_id], dtype=torch.long) if gt_target_id is not None else torch.tensor([], dtype=torch.long),
                'gt_triplet_id': torch.tensor([gt_triplet_id], dtype=torch.long) if gt_triplet_id is not None else torch.tensor([], dtype=torch.long),
                'video_name': label_data.get('video_name', ''),
                'frame_id': label_data.get('frame_id', ''),
                'im_file': im_file,
                'image_type': label_data.get('image_type', '')
            }
        except Exception as e:
            print(f"处理验证集标签时出错: {e}")
            print(f"GT信息: bbox={gt_bbox}, tool_id={gt_tool_id}, action_id={gt_action_id}, target_id={gt_target_id}, triplet_id={gt_triplet_id}")
            traceback.print_exc()
            raise
        
        return label

class OriginalImageDataset(Dataset):
    def __init__(self, data_dir, split='val', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.images_dir = self.data_dir / split / 'images'
        self.labels_dir = self.data_dir / split / 'labels'
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.original_images = []
        self._load_original_images()
    
    def _load_original_images(self):
        json_files = list(self.labels_dir.glob('*_original.json'))
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                label_data = json.load(f)
            img_filename = json_file.stem + '.jpg'
            img_path = self.images_dir / img_filename
            
            if not img_path.exists():
                continue
            self.original_images.append({
                'image_path': str(img_path),
                'label_path': str(json_file),
                'data': label_data,
                'im_file': img_filename
            })
    
    def __len__(self):
        return len(self.original_images)
    
    def __getitem__(self, idx):
        record = self.original_images[idx]
        img = Image.open(record['image_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = {
            'video_name': record['data'].get('video_name', ''),
            'frame_id': record['data'].get('frame_id', ''),
            'im_file': record['im_file'],
            'image_type': 'original'
        }
        
        return img, label
def val_test_collate_fn(batch):
    result = {}
    result['crop_img'] = torch.stack([item['crop_img'] for item in batch])
    result['original_img'] = torch.stack([item['original_img'] for item in batch])
    result['crop_label'] = [item['crop_label'] for item in batch]
    result['original_label'] = [item['original_label'] for item in batch]
    
    return result
def debug_collate_fn(batch):
    try:
        from torch.utils.data._utils.collate import default_collate
        return default_collate(batch)
    except Exception as e:
        raise

def create_dataloaders(data_dir, batch_size=32, num_workers=4, paired=True):
    dataloaders = {}
    train_dataset = PairedDataset(data_dir, split='train')
    dataloaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    for split in ['val', 'test']:
        if (Path(data_dir) / split).exists():
            dataset = PairedDataset(data_dir, split=split)
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=val_test_collate_fn
            )
    for split in ['val', 'test']:
        if (Path(data_dir) / split).exists():
            dataset = OriginalImageDataset(data_dir, split=split)
            dataloaders[f'{split}_original'] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
    
    return dataloaders
