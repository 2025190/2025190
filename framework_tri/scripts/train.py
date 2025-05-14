from ultralytics import YOLO
import os
import argparse
import yaml
from ultralytics.utils import yaml_load, checks

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv11 model training and testing')
    parser.add_argument('--mode', type=str, default='train_test', choices=['train_test', 'train_only', 'test_only', 'predict_only', 'yolotriplet_prepare', 'yolotriplet'],
                        help='running mode: train_test(train and test)、train_only(only train)、test_only(only evaluate metrics)、predict_only(only predict)、yolotriplet_prepare(triplet prepare)、yolotriplet(triplet recognition)')
    parser.add_argument('--model', type=str, default='yolo11l',
                        help='model name, such as yolo11n, yolo11s, yolo11m, yolo11l, yolo11x or model path')
    parser.add_argument('--best_model', type=str, default=None,
                        help='the best model path used in test, if not specified, use the best.pt in training output')
    parser.add_argument('--data', type=str, default='/ssd/prostate/dataset_triplet/dataset_triplet.yaml',
                        help='dataset config file path')
    parser.add_argument('--is_triplet', type=int, default=1,
                   help='use triplet format label, 0=no, 1=yes')
    parser.add_argument('--cls_type', type=str, default='tool', choices=['triplet', 'tool'],
                        help='classification type, yolo backbone use triplet_id or tool_id for classification')
    parser.add_argument('--img_size', type=int, default=640,
                        help='image size')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='training batch size')
    parser.add_argument('--epochs', type=int, default=250,
                        help='training epochs')
    parser.add_argument('--workers', type=int, default=16,
                        help='data loading threads')
    parser.add_argument('--device', type=str, default='',
                        help='device, such as cuda:0 or 0,1,2,3')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='test confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU threshold')
    parser.add_argument('--max_det', type=int, default=300,
                        help='max detections per image')
    parser.add_argument('--save_txt', type=int, default=1,
                        help='save test results as txt file, 0=no, 1=yes')
    parser.add_argument('--save_conf', type=int, default=0,
                        help='save confidence to txt file, 0=no, 1=yes')
    parser.add_argument('--save_crop', type=int, default=0,
                        help='save cropped prediction box, 0=no, 1=yes')
    parser.add_argument('--patience', type=int, default=80,
                        help='early stopping epochs')
    parser.add_argument('--project', type=str, default='runs_triplet',
                        help='project name')
    parser.add_argument('--use_tp_metric_v2', type=int, default=1,
                        help='use V2 version of component matching algorithm (consistent with YOLO original matching logic), 0=no, 1=yes')
    parser.add_argument('--roi_size', type=int, default=7,
                        help='ROI Align output feature size')
    parser.add_argument('--features_dir', type=str, default='./features',
                        help='ROI feature save/load directory')
    return parser.parse_args()

def train_model(args):
    print("\n============ Start training mode ============")
    model_base_name = os.path.basename(args.model)
    train_name = f"{model_base_name}_train"
    model_path = args.model
    data_dict = yaml_load(checks.check_yaml(args.data))
    if args.cls_type == 'tool':
        nc_value = data_dict.get("nc_tool", 0)
        print(f"Use tool classification mode, number of classes: {nc_value}")
    else:
        nc_value = data_dict.get("nc", 0)
        print(f"Use triplet classification mode, number of classes: {nc_value}")
    model = YOLO(model_path)
    print("Model type:", type(model.model))
    for i, layer in enumerate(model.model.model):
        print(f"Layer {i}: {layer.__class__.__name__}, output shape: {getattr(layer, 'out_channels', None)}")
    train_params = {
        'data': args.data,
        'epochs': args.epochs,
        'patience': args.patience,
        'batch': args.batch_size,
        'imgsz': args.img_size,
        'workers': args.workers,
        'device': args.device,
        'is_triplet': bool(args.is_triplet), 
        'cls_type': args.cls_type,
        'project': os.path.join(args.project, 'train'),
        'name': train_name,
        'use_tp_v2': bool(args.use_tp_metric_v2), 
        'nc': nc_value,
    }
    print(f"Training parameters: {train_params}")
    print(f"Classification type: {args.cls_type}, number of classes: {nc_value}")
    results = model.train(**train_params)
    save_dir = model.trainer.save_dir
    print(f"Model actual save directory: {save_dir}")
    best_model_path = os.path.join(save_dir, 'weights', 'best.pt')
    
    if os.path.exists(best_model_path):
        print(f"\nTraining completed! The best model is saved in: {best_model_path}")
        return best_model_path
    else:
        print(f"\nTraining completed, but the best model {best_model_path} was not found. Trying to use the last model.")
        last_model_path = os.path.join(save_dir, 'weights', 'last.pt')
        if os.path.exists(last_model_path):
            print(f"Last model found: {last_model_path}")
            return last_model_path

def test_model(args, model_path=None):
    print("\n============ Start test mode (calculate metrics) ============")
    if args.best_model is not None and os.path.exists(args.best_model):
        model_path = args.best_model
        print(f"Use specified model: {model_path}")
    elif model_path is not None and args.best_model is None :
        print(f"Use specified model: {model_path}")
    else:
        print(f"Warning: No trained weight file found")
        assert False
    model = YOLO(model_path)
    print(f"Successfully loaded model: {model_path}")
    model_base = os.path.basename(model_path).split('.')[0]
    test_name = f"{model_base}_test"
    print(f"\nCalculate performance metrics on test set")
    
    results = model.val(
        data=args.data,
        split='test', 
        imgsz=args.img_size,
        batch=args.batch_size,
        device=args.device,
        iou=args.iou,
        conf=args.conf,
        max_det=args.max_det,
        project=os.path.join(args.project, 'test'),
        name=test_name,
        save_json=True,
        save_txt=bool(args.save_txt),
        is_triplet=bool(args.is_triplet),
        cls_type=args.cls_type,
        use_tp_v2=bool(args.use_tp_metric_v2),  
    )
    
    print(f"\nTest completed! Results saved in {os.path.join(args.project, 'test', test_name)}/")
    if not args.is_triplet:
        metrics = results.box
        print(f"\nPerformance metrics:")
        print(f"mAP@0.5: {metrics.map50:.4f}")
        print(f"mAP@0.5:0.95: {metrics.map:.4f}")
        print(f"Precision: {metrics.p.mean():.4f}")
        print(f"Recall: {metrics.r.mean():.4f}")
        print(f"F1 score: {metrics.f1.mean():.4f}")
        if hasattr(results, "speed"):
            speed_dict = {k: getattr(results.speed, k) for k in dir(results.speed) if not k.startswith('_')}
            if "metrics/FPS" in speed_dict:
                print(f"FPS: {speed_dict['metrics/FPS']:.2f}")
            elif all(k in speed_dict for k in ["preprocess", "inference", "postprocess"]):
                total_time = sum(speed_dict.values())
                fps = 1000 / total_time if total_time > 0 else 0
                print(f"FPS: {fps:.2f}")
            print(f"Processing time: preprocess {speed_dict.get('preprocess', 0):.1f}ms, inference {speed_dict.get('inference', 0):.1f}ms, postprocess {speed_dict.get('postprocess', 0):.1f}ms per image")


def predict_model(args, model_path=None):
    print("\n============ Start prediction mode ============")
    if model_path is None:
        if args.best_model is not None and os.path.exists(args.best_model):
            model_path = args.best_model
            print(f"Use specified model: {model_path}")
        elif args.model.endswith('.pt'):
            model_path = args.model
        else:
            print(f"Warning: No trained weight file found, using default config file: {args.model}")
            model_path = args.model
    model = YOLO(model_path)
    print(f"Successfully loaded model: {model_path}")
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    dataset_path = data_config.get('path', '')
    test_dir = data_config.get('test', 'images/test')
    test_path = os.path.join(dataset_path, test_dir)
    model_base = os.path.basename(model_path).split('.')[0]
    test_name = f"{model_base}_predict"
    print(f"\nRun prediction on test set: {test_path}")
    
    results = model.predict(
        source=test_path,
        imgsz=args.img_size,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        save_txt=bool(args.save_txt),  
        save_conf=bool(args.save_conf),  
        save_crop=bool(args.save_crop),  
        device=args.device,
        project=os.path.join(args.project, 'predict'),
        name=test_name,
    )
    
    print(f"\nPrediction completed! Results saved in {os.path.join(args.project, 'predict', test_name)}/")
    
    if args.save_txt:
        labels_dir = os.path.join(args.project, 'predict', test_name, 'labels')
        print(f"Detection result annotation files saved in: {labels_dir}")
        if os.path.exists(labels_dir):
            txt_files = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
            print(f"Total {txt_files} annotation files generated")

def yolotriplet_prepare(args):
    if args.best_model is not None and os.path.exists(args.best_model):
        model_path = args.best_model
        print(f"Use specified model: {model_path}")
    else:
        print(f"Warning: No trained weight file found")
        assert False
    
    model = YOLO(model_path)
    print(f"Successfully loaded model: {model_path}")
    yolotriplet_prepare = True
    model_base = os.path.basename(model_path).split('.')[0]
    test_name = f"{model_base}_prepare"
    print(f"\nCalculate performance metrics on test set")
    
    results = model.val(
        data=args.data,
        split='test', 
        imgsz=args.img_size,
        batch=args.batch_size,
        device=args.device,
        iou=args.iou,
        conf=args.conf,
        max_det=args.max_det,
        project=os.path.join(args.project, 'test'),
        name=test_name,
        save_json=False,
        save_txt=bool(args.save_txt),
        is_triplet=bool(args.is_triplet),
        cls_type=args.cls_type,
        use_tp_v2=bool(args.use_tp_metric_v2),
        yolotriplet_prepare = True,
        features_dir = args.features_dir,
        roi_size = args.roi_size,
    )
    if not args.is_triplet:
        metrics = results.box
        print(f"\nPerformance metrics:")
        print(f"mAP@0.5: {metrics.map50:.4f}")
        print(f"mAP@0.5:0.95: {metrics.map:.4f}")
        print(f"Precision: {metrics.p.mean():.4f}")
        print(f"Recall: {metrics.r.mean():.4f}")
        print(f"F1分数: {metrics.f1.mean():.4f}")
        if hasattr(results, "speed"):
            speed_dict = {k: getattr(results.speed, k) for k in dir(results.speed) if not k.startswith('_')}
            if "metrics/FPS" in speed_dict:
                print(f"FPS: {speed_dict['metrics/FPS']:.2f}")
            elif all(k in speed_dict for k in ["preprocess", "inference", "postprocess"]):
                total_time = sum(speed_dict.values())
                fps = 1000 / total_time if total_time > 0 else 0
                print(f"FPS: {fps:.2f}")
            print(f"Processing time: preprocess {speed_dict.get('preprocess', 0):.1f}ms, inference {speed_dict.get('inference', 0):.1f}ms, postprocess {speed_dict.get('postprocess', 0):.1f}ms per image")



def yolotriplet(args):
    print("\n============ Start YOLO triplet recognition mode ============")
    pass

def main():
    args = parse_args()
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Running mode: {args.mode}")
    if args.mode == 'train_test':
        model_path = train_model(args)
        if model_path:
            test_model(args, model_path)
        else:
            print("Training failed, cannot perform test.")
    elif args.mode == 'train_only':
        train_model(args) 
    elif args.mode == 'test_only':
        test_model(args)
    elif args.mode == 'predict_only':
        predict_model(args)
    elif args.mode == 'yolotriplet_prepare':
        yolotriplet_prepare(args)
    elif args.mode == 'yolotriplet':
        yolotriplet(args)
    else:
        print(f"Invalid mode: {args.mode}")

if __name__ == "__main__":
    main() 