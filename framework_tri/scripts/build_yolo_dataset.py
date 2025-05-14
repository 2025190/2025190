import os
import json
import shutil
import argparse
from pathlib import Path
import numpy as np
TOOLS = {
    "monopolar scissors": 0,
    "bipolar forceps": 1,
    "aspirator": 2, 
    "needle driver": 3,
    "ProGrasp": 4,
    "clip applier": 5,
    "Endobag": 6
}
DATASET_SPLIT = {
    'train': ['easdv1', 'easdv3', 'easdv4'],
    'val': ['easdv2'],
    'test': ['easdv2']
}

def parse_args():
    parser = argparse.ArgumentParser(description="Convert labelme annotations to YOLO format")
    parser.add_argument('--input-dir', type=str, default='/ssd/prostate/framework_dataset/prostate_640',
                      help='Directory containing video folders with labelme annotations')
    parser.add_argument('--output-dir', type=str, default='/ssd/prostate/framework_yolo/dataset',
                      help='Output directory for YOLO format dataset')
    return parser.parse_args()

def convert_bbox_to_yolo(bbox, img_width, img_height):
    x1, y1 = min(bbox[0][0], bbox[1][0]), min(bbox[0][1], bbox[1][1])
    x2, y2 = max(bbox[0][0], bbox[1][0]), max(bbox[0][1], bbox[1][1])
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    return [x_center, y_center, width, height]

def process_json_file(json_path, video_name, output_image_dir, output_label_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    img_width = data['imageWidth']
    img_height = data['imageHeight']
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    output_img_path = os.path.join(output_image_dir, f"{video_name}_{base_name}.jpg")
    output_txt_path = os.path.join(output_label_dir, f"{video_name}_{base_name}.txt")
    input_img_path = os.path.join(os.path.dirname(json_path), f"{base_name}.jpg")
    if os.path.exists(input_img_path):
        shutil.copy(input_img_path, output_img_path)
    with open(output_txt_path, 'w') as f:
        for shape in data['shapes']:
            label = shape['label']
            if label in TOOLS:
                class_id = TOOLS[label]
                bbox = shape['points']
                yolo_bbox = convert_bbox_to_yolo(bbox, img_width, img_height)
                f.write(f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")

def create_dataset_yaml(output_dir):
    yaml_content = f"""# YOLOv8 dataset configuration
path: {output_dir}
train: train/images
val: val/images
test: test/images
nc: {len(TOOLS)}
names:
  0: monopolar scissors
  1: bipolar forceps
  2: aspirator
  3: needle driver
  4: ProGrasp
  5: clip applier
  6: Endobag
"""
    
    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)

def main():
    args = parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    for video_dir in os.listdir(input_dir):
        video_path = os.path.join(input_dir, video_dir)
        if not os.path.isdir(video_path):
            continue
        assigned_splits = []
        for split, videos in DATASET_SPLIT.items():
            if video_dir in videos:
                assigned_splits.append(split)
                
        if not assigned_splits:
            print(f"Warning: {video_dir} not assigned to any split, skipping.")
            continue
        for filename in os.listdir(video_path):
            if filename.endswith('.json'):
                json_path = os.path.join(video_path, filename)
                for split in assigned_splits:
                    output_image_dir = os.path.join(output_dir, split, 'images')
                    output_label_dir = os.path.join(output_dir, split, 'labels')
                    process_json_file(json_path, video_dir, output_image_dir, output_label_dir)
    create_dataset_yaml(output_dir)
    
    print(f"Conversion complete. Dataset saved to {output_dir}")
    print(f"Dataset statistics:")
    for split in ['train', 'val', 'test']:
        img_count = len(os.listdir(os.path.join(output_dir, split, 'images')))
        print(f"  {split}: {img_count} images")

if __name__ == "__main__":
    main()
