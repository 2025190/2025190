import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json


def visualize_roi_features(features_dir, image_names=None, max_images=3, max_boxes=3):
    if image_names is None:
        image_names = [os.path.basename(f).replace('.pt', '') 
                      for f in os.listdir(features_dir) 
                      if f.endswith('.pt')][:max_images]
    else:
        image_names = image_names[:max_images]
    
    if not image_names:
        print(f"No feature files found in directory {features_dir}")
        return
    
    print(f"start to visualize {len(image_names)} ROI features...")
    
    for img_name in image_names:
        pt_path = os.path.join(features_dir, f"{img_name}.pt")
        if not os.path.exists(pt_path):
            print(f"feature file not found: {pt_path}")
            continue
            
        features_data = torch.load(pt_path)
        layer_features = features_data["features"]
        json_path = os.path.join(features_dir, f"{img_name}.json")
        if not os.path.exists(json_path):
            print(f"detection result file not found: {json_path}")
            continue
            
        with open(json_path, 'r') as f:
            detections = json.load(f)
        layers = list(layer_features.keys())
        if not layers:
            print(f"image {img_name} has no feature layers")
            continue
            
        num_boxes = min(len(layer_features[layers[0]]), max_boxes)
        if num_boxes == 0:
            print(f"image {img_name} has no bounding boxes")
            continue
        fig, axes = plt.subplots(num_boxes, len(layers), figsize=(len(layers)*4, num_boxes*4))
        fig.suptitle(f"ROI features: {img_name}", fontsize=16)
        if num_boxes == 1 and len(layers) == 1:
            axes = np.array([[axes]])
        elif num_boxes == 1:
            axes = np.array([axes])
        elif len(layers) == 1:
            axes = np.array([[ax] for ax in axes])
        for box_idx in range(num_boxes):
            if box_idx < len(detections["boxes"]):
                box = detections["boxes"][box_idx]
                score = detections["scores"][box_idx]
                class_id = detections["class_ids"][box_idx]
                box_info = f"Box {box_idx+1}: class={class_id}, conf={score:.2f}"
            else:
                box_info = f"Box {box_idx+1}"
            for layer_idx, layer_name in enumerate(layers):
                feature = layer_features[layer_name][box_idx]
                feature_avg = feature.mean(dim=0).cpu().numpy()
                if feature_avg.max() > feature_avg.min():
                    feature_norm = (feature_avg - feature_avg.min()) / (feature_avg.max() - feature_avg.min())
                else:
                    feature_norm = feature_avg
                ax = axes[box_idx, layer_idx]
                im = ax.imshow(feature_norm, cmap='viridis')
                ax.set_title(f"{layer_name}\n{box_info}")
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        save_path = os.path.join(features_dir, f"{img_name}_roi_viz.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"save ROI features visualization: {save_path}")


def visualize_features_with_original_image(features_dir, images_dir, image_names=None, max_images=3, max_boxes=3):
    from PIL import Image, ImageDraw
    if image_names is None:
        image_names = [os.path.basename(f).replace('.pt', '') 
                      for f in os.listdir(features_dir) 
                      if f.endswith('.pt')][:max_images]
    else:
        image_names = image_names[:max_images]
    
    if not image_names:
        print(f"No feature files found in directory {features_dir}")
        return
    
    print(f"start to visualize {len(image_names)} ROI features...")
    
    for img_name in image_names:
        pt_path = os.path.join(features_dir, f"{img_name}.pt")
        if not os.path.exists(pt_path):
            print(f"feature file not found: {pt_path}")
            continue
            
        features_data = torch.load(pt_path)
        layer_features = features_data["features"]
        json_path = os.path.join(features_dir, f"{img_name}.json")
        if not os.path.exists(json_path):
            print(f"detection result file not found: {json_path}")
            continue
            
        with open(json_path, 'r') as f:
            detections = json.load(f)
        image_path = os.path.join(images_dir, f"{img_name}.jpg")
        if not os.path.exists(image_path):
            print(f"original image not found: {image_path}")
            continue
            
        original_image = Image.open(image_path)
        draw = ImageDraw.Draw(original_image)
        for i, box in enumerate(detections["boxes"]):
            if i >= max_boxes:
                break
                
            x1, y1, x2, y2 = box
            score = detections["scores"][i]
            class_id = detections["class_ids"][i]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            label = f"{i+1}: class={class_id}, conf={score:.2f}"
            draw.text((x1, y1-15), label, fill="red")
        viz_dir = os.path.join(features_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        image_save_path = os.path.join(viz_dir, f"{img_name}_boxes.jpg")
        original_image.save(image_save_path)
        print(f"save image with bounding boxes: {image_save_path}")
        layers = list(layer_features.keys())
        if not layers:
            print(f"image {img_name} has no feature layers")
            continue
            
        num_boxes = min(len(layer_features[layers[0]]), max_boxes)
        if num_boxes == 0:
            print(f"image {img_name} has no bounding boxes")
            continue
        for box_idx in range(num_boxes):
            fig, axes = plt.subplots(1, len(layers), figsize=(len(layers)*4, 4))
            if len(layers) == 1:
                axes = [axes]
            if box_idx < len(detections["boxes"]):
                box = detections["boxes"][box_idx]
                score = detections["scores"][box_idx]
                class_id = detections["class_ids"][box_idx]
                box_title = f"Box {box_idx+1}: class={class_id}, conf={score:.2f}"
            else:
                box_title = f"Box {box_idx+1}"
            
            fig.suptitle(f"{img_name} - {box_title}", fontsize=16)
            for layer_idx, layer_name in enumerate(layers):
                feature = layer_features[layer_name][box_idx]
                feature_avg = feature.mean(dim=0).cpu().numpy()
                if feature_avg.max() > feature_avg.min():
                    feature_norm = (feature_avg - feature_avg.min()) / (feature_avg.max() - feature_avg.min())
                else:
                    feature_norm = feature_avg
                ax = axes[layer_idx]
                im = ax.imshow(feature_norm, cmap='viridis')
                ax.set_title(f"{layer_name}")
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            save_path = os.path.join(viz_dir, f"{img_name}_box{box_idx+1}_features.png")
            plt.savefig(save_path, dpi=150)
            plt.close(fig)
            print(f"save ROI features visualization: {save_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='visualize ROI features')
    parser.add_argument('--features_dir', type=str, default='../features/', help='directory to save features')
    parser.add_argument('--images_dir', type=str, default='/ssd/prostate/dataset_triplet/test/images/', help='original image directory, used to generate images with bounding boxes')
    parser.add_argument('--max_images', type=int, default=3, help='maximum number of images to visualize')
    parser.add_argument('--max_boxes', type=int, default=3, help='maximum number of bounding boxes to visualize for each image')
    
    args = parser.parse_args()
    visualize_roi_features(args.features_dir, max_images=args.max_images, max_boxes=args.max_boxes)
    if args.images_dir:
        visualize_features_with_original_image(
            args.features_dir, 
            args.images_dir, 
            max_images=args.max_images, 
            max_boxes=args.max_boxes
        ) 