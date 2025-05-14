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
ACTIONS = {
    "retract": 0,
    "coagulate": 1,
    "cut": 2,
    "dissect": 3,
    "grasp": 4,
    "bag": 5,
    "suture": 6,
    "suck": 7,
    "clip": 8,
    "null": 9
}
TARGETS = {
    "bladder Neck": 0,
    "catheter": 1,
    "seminal vesicle": 2,
    "prostate": 3,
    "fascias": 4,
    "gauze": 5,
    "Endobag": 6,
    "thread": 7,
    "fluid": 8,
    "null": 9
}
TARGET_NORMALIZATION = {
    "bladder": "bladder Neck",
    "vas deferens": "seminal vesicle"
}
TRIPLETS = {
    0: ["monopolar scissors", "retract", "bladder Neck"],
    1: ["monopolar scissors", "retract", "catheter"],
    2: ["monopolar scissors", "retract", "seminal vesicle"],
    3: ["monopolar scissors", "retract", "prostate"],
    4: ["monopolar scissors", "retract", "fascias"],
    5: ["monopolar scissors", "coagulate", "bladder Neck"],
    6: ["monopolar scissors", "coagulate", "seminal vesicle"],
    7: ["monopolar scissors", "coagulate", "prostate"],
    8: ["monopolar scissors", "coagulate", "fascias"],
    9: ["monopolar scissors", "cut", "bladder Neck"],
    10: ["monopolar scissors", "cut", "seminal vesicle"],
    11: ["monopolar scissors", "cut", "prostate"],
    12: ["monopolar scissors", "cut", "fascias"],
    13: ["monopolar scissors", "cut", "thread"],
    14: ["monopolar scissors", "dissect", "bladder Neck"],
    15: ["monopolar scissors", "dissect", "seminal vesicle"],
    16: ["monopolar scissors", "dissect", "prostate"],
    17: ["monopolar scissors", "dissect", "fascias"],
    18: ["monopolar scissors", "null", "null"],
    19: ["bipolar forceps", "retract", "bladder Neck"],
    20: ["bipolar forceps", "retract", "seminal vesicle"],
    21: ["bipolar forceps", "retract", "prostate"],
    22: ["bipolar forceps", "retract", "fascias"],
    23: ["bipolar forceps", "coagulate", "prostate"],
    24: ["bipolar forceps", "coagulate", "fascias"],
    25: ["bipolar forceps", "dissect", "fascias"],
    26: ["bipolar forceps", "grasp", "catheter"],
    27: ["bipolar forceps", "grasp", "seminal vesicle"],
    28: ["bipolar forceps", "grasp", "prostate"],
    29: ["bipolar forceps", "grasp", "fascias"],
    30: ["bipolar forceps", "grasp", "thread"],
    31: ["bipolar forceps", "suture", "bladder Neck"],
    32: ["bipolar forceps", "suture", "prostate"],
    33: ["bipolar forceps", "suture", "fascias"],
    34: ["bipolar forceps", "null", "null"],
    35: ["aspirator", "retract", "bladder Neck"],
    36: ["aspirator", "retract", "seminal vesicle"],
    37: ["aspirator", "retract", "prostate"],
    38: ["aspirator", "retract", "fascias"],
    39: ["aspirator", "suck", "fluid"],
    40: ["aspirator", "null", "null"],
    41: ["needle driver", "retract", "bladder Neck"],
    42: ["needle driver", "retract", "fascias"],
    43: ["needle driver", "grasp", "fascias"],
    44: ["needle driver", "grasp", "thread"],
    45: ["needle driver", "suture", "bladder Neck"],
    46: ["needle driver", "suture", "prostate"],
    47: ["needle driver", "suture", "fascias"],
    48: ["needle driver", "null", "null"],
    49: ["ProGrasp", "retract", "bladder Neck"],
    50: ["ProGrasp", "retract", "seminal vesicle"],
    51: ["ProGrasp", "retract", "prostate"],
    52: ["ProGrasp", "retract", "fascias"],
    53: ["ProGrasp", "grasp", "catheter"],
    54: ["ProGrasp", "grasp", "seminal vesicle"],
    55: ["ProGrasp", "grasp", "prostate"],
    56: ["ProGrasp", "grasp", "fascias"],
    57: ["ProGrasp", "null", "null"],
    58: ["clip applier", "clip", "bladder Neck"],
    59: ["clip applier", "clip", "seminal vesicle"],
    60: ["clip applier", "clip", "prostate"],
    61: ["clip applier", "clip", "fascias"],
    62: ["clip applier", "null", "null"],
    63: ["Endobag", "bag", "prostate"],
    64: ["Endobag", "null", "null"]
}
TRIPLET_MAP = {}
for triplet_id, triplet in TRIPLETS.items():
    TRIPLET_MAP[tuple(triplet)] = triplet_id
DATASET_SPLIT = {
    'train': ['easdv1', 'easdv3', 'easdv4'],
    'val': ['easdv2'],
    'test': ['easdv2','easdv3']
}

def parse_args():
    parser = argparse.ArgumentParser(description="Convert labelme annotations to YOLO format with triplets")
    parser.add_argument('--input-dir', type=str, default='/ssd/prostate/framework_dataset/prostate_640',
                      help='Directory containing video folders with labelme annotations')
    parser.add_argument('--output-dir', type=str, default='/ssd/prostate/dataset_triplet2',
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

def normalize_target(target):
    return TARGET_NORMALIZATION.get(target, target)

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
            tool = shape['label']
            if tool not in TOOLS:
                continue
            attributes = shape.get('attributes', {})
            action = attributes.get('Action', 'null')
            target = attributes.get('Target', 'null')
            target = normalize_target(target)
            tool_id = TOOLS[tool]
            action_id = ACTIONS.get(action, ACTIONS['null'])
            target_id = TARGETS.get(target, TARGETS['null'])
            triplet = (tool, action, target)
            triplet_id = TRIPLET_MAP.get(triplet, -1)
            if triplet_id == -1:
                print(f"warning: triplet {triplet} not found, image path: {json_path}")
                null_triplet = (tool, 'null', 'null')
                triplet_id = TRIPLET_MAP.get(null_triplet, 0)
            bbox = shape['points']
            yolo_bbox = convert_bbox_to_yolo(bbox, img_width, img_height)
            f.write(f"{triplet_id} {tool_id} {action_id} {target_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")

def create_dataset_yaml(output_dir):
    yaml_content = f"""# YOLOv8 dataset configuration with triplets
path: {output_dir}
train: train/images
val: val/images
test: test/images
nc: {len(TRIPLETS)}
nc_tool: {len(TOOLS)}
nc_action: {len(ACTIONS)}
nc_target: {len(TARGETS)}
names:
tools:
  0: monopolar scissors
  1: bipolar forceps
  2: aspirator
  3: needle driver
  4: ProGrasp
  5: clip applier
  6: Endobag
actions:
  0: retract
  1: coagulate
  2: cut
  3: dissect
  4: grasp
  5: bag
  6: suture
  7: suck
  8: clip
  9: null
targets:
  0: bladder Neck
  1: catheter
  2: seminal vesicle
  3: prostate
  4: fascias
  5: gauze
  6: Endobag
  7: thread
  8: fluid
  9: null
"""
    
    with open(os.path.join(output_dir, 'dataset_triplet.yaml'), 'w') as f:
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
            print(f"warning: {video_dir} not assigned to any dataset split, skipping processing.")
            continue
        for filename in os.listdir(video_path):
            if filename.endswith('.json'):
                json_path = os.path.join(video_path, filename)
                for split in assigned_splits:
                    output_image_dir = os.path.join(output_dir, split, 'images')
                    output_label_dir = os.path.join(output_dir, split, 'labels')
                    process_json_file(json_path, video_dir, output_image_dir, output_label_dir)
    create_dataset_yaml(output_dir)
    
    print(f"conversion completed. dataset saved to {output_dir}")
    print(f"dataset statistics:")
    for split in ['train', 'val', 'test']:
        img_count = len(os.listdir(os.path.join(output_dir, split, 'images')))
        print(f"  {split}: {img_count} images")

if __name__ == "__main__":
    main()
