import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='Find original images without corresponding crop images')
    parser.add_argument('--data-dir', type=str, default='runs/dataset_stage_two', help='Dataset directory')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory, default same as data directory')
    return parser.parse_args()

def find_unpaired_originals(data_dir, split):
    data_dir = Path(data_dir)
    images_dir = data_dir / split / 'images'
    labels_dir = data_dir / split / 'labels'
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"Warning: {split} directory does not exist")
        return [], []
    unpaired_originals = []
    missed_detections_list = []
    total_originals = 0
    no_preds_count = 0
    missed_detections = 0
    for json_file in labels_dir.glob('*_original.json'):
        try:
            with open(json_file, 'r') as f:
                label_data = json.load(f)
            
            total_originals += 1
            if label_data.get('pred_patches', -1) == 0:
                no_preds_count += 1
                img_filename = json_file.stem + '.jpg'
                img_path = images_dir / img_filename
                
                if not img_path.exists():
                    continue
                video_name = label_data.get('video_name', '')
                frame_id = label_data.get('frame_id', '')
                key = f"{video_name}_{frame_id}"
                ground_truths = label_data.get('ground_truths', [])
                item_data = {
                    'key': key,
                    'video_name': video_name,
                    'frame_id': frame_id,
                    'filename': img_filename,
                    'json_file': str(json_file),
                    'ground_truths': ground_truths
                }
                
                unpaired_originals.append(item_data)
                if ground_truths and len(ground_truths) > 0:
                    missed_detections += 1
                    missed_detections_list.append(item_data)
                
        except Exception as e:
            print(f"Error processing file {json_file}: {e}")
    
    print(f"{split}:")
    print(f"- Total original images: {total_originals}")
    print(f"- Images with no predictions (pred_patches=0): {no_preds_count}")
    print(f"- Missed detections (pred_patches=0 but has GT): {missed_detections}")
    
    return unpaired_originals, missed_detections_list

def save_unpaired_originals(unpaired_originals, output_path):
    with open(output_path, 'w') as f:
        for item in unpaired_originals:
            f.write(json.dumps(item) + '\n')
    
    print(f"Saved {len(unpaired_originals)} unpaired original images to {output_path}")

def load_unpaired_originals(input_path):
    unpaired_originals = []
    
    if not os.path.exists(input_path):
        print(f"Warning: File does not exist: {input_path}")
        return unpaired_originals
    
    with open(input_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            unpaired_originals.append(item)
    return unpaired_originals

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Dataset directory does not exist: {data_dir}")
        return
    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ['val', 'test']:
        print(f"Processing {split} set...")
        unpaired_originals, missed_detections_list = find_unpaired_originals(data_dir, split)
        
        if unpaired_originals:
            output_path = output_dir / f"unpaired_originals_{split}.json"
            save_unpaired_originals(unpaired_originals, output_path)
            print(f"Found {len(unpaired_originals)} unpaired original images")
        else:
            print(f"No unpaired original images found in {split} set")
        if missed_detections_list:
            missed_output_path = output_dir / f"missed_detections_{split}.json"
            save_unpaired_originals(missed_detections_list, missed_output_path)
            print(f"Found {len(missed_detections_list)} missed original images")
        else:
            print(f"No missed original images found in {split} set")

if __name__ == '__main__':
    main() 