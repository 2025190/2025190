
import os
from pathlib import Path
import json

import numpy as np
import torch

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import output_to_target, plot_images
from ultralytics.utils.metrics import ap_per_class

from component_metrics import TripletDisentangle, compute_component_maps, compute_frame_component_tp, compute_frame_component_tp_V2
import sys
from yolo_roi_align import FeatureExtractor
from torchvision.ops import RoIAlign

class DetectionValidator(BaseValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.nt_per_class = None
        self.nt_per_image = None
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.metrics = DetMetrics(save_dir=self.save_dir)
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.niou = self.iouv.numel()
        self.validator = self

    def preprocess(self, batch, is_triplet=False):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        if not is_triplet:
            for k in ["batch_idx", "cls", "bboxes"]:
                batch[k] = batch[k].to(self.device)
        else:
            batch["video_name"] = []
            for k in ["batch_idx", "cls", "bboxes", "tool_id", "action_id", "target_id"]:
                batch[k] = batch[k].to(self.device)
            for im_file in batch["im_file"]:
                jpg_name = im_file.split("/")[-1].replace(".jpg", "")
                video_name = jpg_name.split("_")[-2]
                batch["video_name"].append(video_name)
        return batch

    def init_metrics(self, model):
        val = self.data.get(self.args.split, "")
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1, len(model.names) + 1))
        self.args.save_json |= self.args.val and (self.is_coco or self.is_lvis) and not self.training
        self.names = model.names
        self.nc = len(model.names)
        self.end2end = getattr(model, "end2end", False)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        if self.args.is_triplet:
            self.video_stats = {}
            self.disentangler = TripletDisentangle()
            

    def get_desc(self):
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    def postprocess(self, preds):
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            nc=self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            rotated=self.args.task == "obb",
        )

    def _prepare_batch(self, si, batch):
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]

        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)
            
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        predn = pred.clone()
        
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )
        
        
        return predn

    def update_metrics(self, preds, batch, is_triplet=False):
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
                
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(batch['im_file'][si]).stem}.txt",
                )
    
    def update_metrics_triplet(self, preds, batch):
        
        disentangler = self.disentangler
        components = ['i', 'v', 't']
        
        yolotriplet_prepare = getattr(self.args, 'yolotriplet_prepare', False)
        if yolotriplet_prepare:
            features_dir = getattr(self.args, 'features_dir', './features')
            os.makedirs(features_dir, exist_ok=True)
        for si, pred in enumerate(preds):
            video_name = batch["video_name"][si]
            image_name = os.path.basename(batch["im_file"][si]).replace(".jpg", "")
            roi_features_dict = {}
            if yolotriplet_prepare and len(pred) > 0 and hasattr(self, 'validator') and hasattr(self.validator, 'feature_extractor') and self.validator.feature_extractor is not None:
                current_features = self.validator.feature_extractor.features
                if current_features:
                    pbatch = self._prepare_batch(si, batch) 
                    predn = self._prepare_pred(pred, pbatch) if len(pred) > 0 else None
                    
                    if predn is not None and len(predn) > 0:
                        batch_indices = torch.full((len(predn), 1), 0, dtype=torch.float, device=predn.device)
                        roi_boxes = torch.cat([batch_indices, predn[:, :4]], dim=1)
                        for layer_name, feature in current_features.items():
                            if layer_name in self.validator.roi_aligns:
                                try:
                                    roi_features = self.validator.roi_aligns[layer_name](feature, roi_boxes)
                                    roi_features_dict[layer_name] = roi_features
                                except Exception as e:
                                    assert False, f"提取{layer_name}的ROI特征时出错: {e}"
            if video_name not in self.video_stats:
                self.video_stats[video_name] = {
                    "tp": [], "conf": [], "pred_cls": [], "target_cls": [], 
                    "pred_boxes": [], "target_boxes": []
                }
                for comp in components:
                    self.video_stats[video_name][f"tp_{comp}"] = []
                for comp in components:
                    self.video_stats[video_name][f"pred_cls_{comp}"] = []
                    self.video_stats[video_name][f"target_cls_{comp}"] = []
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            cur_stat = {
                "conf": torch.zeros(0, device=self.device),
                "pred_cls": torch.zeros(0, device=self.device),
                "tp": torch.zeros(len(pred), self.niou, dtype=torch.bool, device=self.device),
                "pred_boxes": torch.zeros(0, 4, device=self.device),
                "target_boxes": bbox,
                "target_cls": cls
            }
            for comp in components:
                cur_stat[f"tp_{comp}"] = torch.zeros(len(pred), self.niou, dtype=torch.bool, device=self.device)
            if len(pred) == 0:
                if nl:
                    for comp in components:
                        if comp == 'ivt':
                            cur_stat[f"pred_cls_{comp}"] = torch.zeros(0, device=self.device)
                            cur_stat[f"target_cls_{comp}"] = cls
                        else:
                            cur_stat[f"pred_cls_{comp}"] = torch.zeros(0, device=self.device)
                            cur_stat[f"target_cls_{comp}"] = torch.tensor(
                                disentangler.extract(cls.cpu().numpy(), comp),
                                device=cls.device
                            )
                    for k in cur_stat:
                        self.video_stats[video_name][k].append(cur_stat[k])
                if yolotriplet_prepare:
                    result = {
                        "boxes": [],
                        "scores": [],
                        "class_ids": []
                    }
                    result_path = os.path.join(features_dir, f"{image_name}.json")
                    with open(result_path, 'w') as f:
                        json.dump(result, f)
                        
                continue
            predn = self._prepare_pred(pred, pbatch)
            cur_stat["pred_boxes"] = predn[:, :4]
            cur_stat["conf"] = predn[:, 4]
            cur_stat["pred_cls"] = predn[:, 5]
            if yolotriplet_prepare:
                result = {
                    "boxes": predn[:, :4].cpu().numpy().tolist(),
                    "scores": predn[:, 4].cpu().numpy().tolist(),
                    "class_ids": predn[:, 5].cpu().numpy().tolist()
                }
                result_path = os.path.join(features_dir, f"{image_name}.json")
                with open(result_path, 'w') as f:
                    json.dump(result, f)
                if roi_features_dict:
                    features_by_layer = {
                        layer_name: [] for layer_name in roi_features_dict.keys()
                    }
                    num_boxes = min([len(features) for features in roi_features_dict.values()])
                    for i in range(num_boxes):
                        for layer_name, features in roi_features_dict.items():
                            if i < len(features):
                                features_by_layer[layer_name].append(features[i])
                    if all(len(tensors) > 0 for tensors in features_by_layer.values()):
                        features_data = {"features": features_by_layer}
                        pt_path = os.path.join(features_dir, f"{image_name}.pt")
                        try:
                            torch.save(features_data, pt_path)
                        except Exception as e:
                            assert False, f"error saving ROI features using PyTorch: {e}"
                    else:
                        assert False, f"not enough ROI features collected to save {image_name}"
            if nl:
                cur_stat["tp"] = self._process_batch(predn, bbox, cls)
                comp_pred_cls_dict = {}
                comp_target_cls_dict = {}
                
                for component in components:
                    if component == 'ivt':
                        comp_pred_cls = predn[:, 5]
                        comp_target_cls = cls
                    else:
                        comp_pred_cls = torch.tensor(
                            disentangler.extract(predn[:, 5].cpu().numpy(), component),
                            device=predn.device
                        )
                        comp_target_cls = torch.tensor(
                            disentangler.extract(cls.cpu().numpy(), component),
                            device=cls.device
                        )
                    comp_pred_cls_dict[component] = comp_pred_cls
                    comp_target_cls_dict[component] = comp_target_cls
                    cur_stat[f'pred_cls_{component}'] = comp_pred_cls
                    cur_stat[f'target_cls_{component}'] = comp_target_cls
                if self.args.use_tp_v2:
                    comp_tp_dict = compute_frame_component_tp_V2(
                        predn[:, :4], bbox, comp_pred_cls_dict, comp_target_cls_dict, self.iouv
                    )
                else:
                    comp_tp_dict = compute_frame_component_tp(
                        predn[:, :4], bbox, comp_pred_cls_dict, comp_target_cls_dict, self.iouv
                    )
                for k, v in comp_tp_dict.items():
                    if k in cur_stat:
                        cur_stat[k] = v
            else:
                for comp in components:
                    if comp == 'ivt':
                        cur_stat[f"pred_cls_{comp}"] = predn[:, 5]
                        cur_stat[f"target_cls_{comp}"] = torch.zeros(0, device=self.device)
                    else:
                        cur_stat[f"pred_cls_{comp}"] = torch.tensor(
                            disentangler.extract(predn[:, 5].cpu().numpy(), comp),
                            device=predn.device
                        )
                        cur_stat[f"target_cls_{comp}"] = torch.zeros(0, device=self.device)
            for k in cur_stat:
                if k in self.video_stats[video_name]:
                    self.video_stats[video_name][k].append(cur_stat[k])
        
    def finalize_metrics(self, *args, **kwargs):
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix
        


    def get_stats(self):
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
        stats.pop("target_img", None)
        if len(stats):
            self.metrics.process(**stats, on_plot=self.on_plot)
        return self.metrics.results_dict
    
    def get_triplet_stats(self):
        video_results = {}
        for video_name, video_stat in self.video_stats.items():
            if not all(len(v) > 0 for v in [video_stat["conf"], video_stat["target_cls"]]):
                video_results[video_name] = {"mAP50": 0.0, "mAP50-95": 0.0}
                continue
            processed_video_stats = {}
            for k in ["tp", "conf", "pred_cls", "target_cls", "pred_boxes", "target_boxes"]:
                if len(video_stat.get(k, [])) > 0:
                    processed_video_stats[k] = torch.cat(video_stat[k], 0)
                else:
                    if k in ["conf", "pred_cls", "target_cls"]:
                        processed_video_stats[k] = torch.zeros(0, device=self.device)
                    elif k.startswith("tp"):
                        processed_video_stats[k] = torch.zeros((0, self.niou), dtype=torch.bool, device=self.device)
                    elif k in ["pred_boxes", "target_boxes"]:
                        processed_video_stats[k] = torch.zeros((0, 4), device=self.device)
            if processed_video_stats["conf"].shape[0] == 0 or processed_video_stats["target_cls"].shape[0] == 0:
                video_results[video_name] = {"mAP50": 0.0, "mAP50-95": 0.0}
                continue
            tp, fp, p, r, f1, ap, _, _, _, _, _, _ = ap_per_class(
                processed_video_stats["tp"].cpu().numpy(),
                processed_video_stats["conf"].cpu().numpy(),
                processed_video_stats["pred_cls"].cpu().numpy(),
                processed_video_stats["target_cls"].cpu().numpy(),
                plot=False,
                save_dir=self.save_dir,
                names=self.names,
                prefix=f"video_{video_name}"
            )
            video_results[video_name] = {
                "mAP50": ap[:, 0].mean() if len(ap) else 0.0,
                "mAP50-95": ap.mean() if len(ap) else 0.0,
                "precision": p.mean() if len(p) else 0.0,
                "AR@300": r.mean() if len(r) else 0.0,
                "f1": f1.mean() if len(f1) else 0.0
            }
            components = ['i', 'v', 't']
            for component in components:
                tp_key = f"tp_{component}"
                pred_cls_key = f"pred_cls_{component}"
                target_cls_key = f"target_cls_{component}"
                if all(len(video_stat.get(k, [])) > 0 for k in [tp_key, pred_cls_key, target_cls_key]):
                    comp_tp = torch.cat(video_stat[tp_key], 0)
                    comp_pred_cls = torch.cat(video_stat[pred_cls_key], 0)
                    comp_target_cls = torch.cat(video_stat[target_cls_key], 0)
                    _, _, _, _, _, comp_ap, _, _, _, _, _, _ = ap_per_class(
                        comp_tp.cpu().numpy(),
                        processed_video_stats["conf"].cpu().numpy(),
                        comp_pred_cls.cpu().numpy(),
                        comp_target_cls.cpu().numpy(),
                        plot=False
                    )
                    
                    if len(comp_ap) > 0:
                        video_results[video_name][f"mAP50_{component}"] = float(comp_ap[:, 0].mean())
                        video_results[video_name][f"mAP50-95_{component}"] = float(comp_ap.mean())
                    else:
                        video_results[video_name][f"mAP50_{component}"] = 0.0
                        video_results[video_name][f"mAP50-95_{component}"] = 0.0
                else:
                    video_results[video_name][f"mAP50_{component}"] = 0.0
                    video_results[video_name][f"mAP50-95_{component}"] = 0.0
        if video_results:
            video_results["average"] = {"mAP50": 0.0, "mAP50-95": 0.0}
            metrics = ["mAP50", "mAP50-95", "precision", "AR@300", "f1"]
            for component in components:
                metrics.append(f"mAP50_{component}")
                metrics.append(f"mAP50-95_{component}")
            for metric in metrics:
                values = [v[metric] for v in video_results.values() if metric in v and v != video_results["average"] and not np.isnan(v[metric])]
                if values:
                    video_results["average"][metric] = float(np.mean(values))
                else:
                    video_results["average"][metric] = 0.0
        
        return video_results
        

    def print_results(self):
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)
        LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(f"no labels found in {self.args.task} set, can not compute metrics without labels")
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf % (self.names[c], self.nt_per_image[c], self.nt_per_class[c], *self.metrics.class_result(i))
                )

        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )
    
    def print_triplet_stats(self, video_stats):
        if not video_stats:
            LOGGER.warning("no video evaluation results to display")
            return

        if not self.training:
            results_json_path = os.path.join(self.save_dir, "triplet_results.json")
            with open(results_json_path, 'w', encoding='utf-8') as f:
                json.dump(video_stats, f, indent=4, ensure_ascii=False)
            results_txt_path = os.path.join(self.save_dir, "triplet_results.txt")
            with open(results_txt_path, 'w', encoding='utf-8') as f:
                for video_name, metrics in video_stats.items():
                    f.write(f"\n{video_name}:\n")
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, float):
                            f.write(f"  {metric_name}: {metric_value:.4f}\n")
                        else:
                            f.write(f"  {metric_name}: {metric_value}\n")
        if not self.training:
            LOGGER.info("\nvideo-level detection evaluation results:")
            LOGGER.info(f"results saved to: {results_json_path}")
            LOGGER.info(f"results saved to: {results_txt_path}")
            for video_name, metrics in video_stats.items():
                metric_str = f"***{video_name}***: "
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, float):
                        metric_str += f"{metric_name}: {metric_value:.4f} | "
                    else:
                        metric_str += f"{metric_name}: {metric_value} | "
                LOGGER.info(metric_str.rstrip(" | "))
        else:
            for video_name, metrics in video_stats.items(): 
                if video_name == "average":
                    metric_str = f"***{video_name}***: "
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, float):
                            metric_str += f"{metric_name}: {metric_value:.4f} | "
                        else:
                            metric_str += f"{metric_name}: {metric_value} | "
                    LOGGER.info(metric_str.rstrip(" | "))
        
    def _process_batch(self, detections, gt_bboxes, gt_cls):
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def build_dataset(self, img_path, mode="val", batch=None):
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path, batch_size):
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)

    def plot_val_samples(self, batch, ni):
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def save_one_txt(self, predn, save_conf, shape, file):
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=predn[:, :6],
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename):
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])
        box[:, :2] -= box[:, 2:] / 2
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )

    def eval_json(self, stats):
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            pred_json = self.save_dir / "predictions.json"
            anno_json = (
                self.data["path"]
                / "annotations"
                / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
            )
            pkg = "pycocotools" if self.is_coco else "lvis"
            LOGGER.info(f"\nEvaluating {pkg} mAP using {pred_json} and {anno_json}...")
            try:
                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} file not found"
                check_requirements("pycocotools>=2.0.6" if self.is_coco else "lvis>=0.5.3")
                if self.is_coco:
                    from pycocotools.coco import COCO
                    from pycocotools.cocoeval import COCOeval

                    anno = COCO(str(anno_json))
                    pred = anno.loadRes(str(pred_json))
                    val = COCOeval(anno, pred, "bbox")
                else:
                    from lvis import LVIS, LVISEval

                    anno = LVIS(str(anno_json))
                    pred = anno._load_json(str(pred_json))
                    val = LVISEval(anno, pred, "bbox")
                val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]
                val.evaluate()
                val.accumulate()
                val.summarize()
                if self.is_lvis:
                    val.print_results()
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = (
                    val.stats[:2] if self.is_coco else [val.results["AP"], val.results["AP50"]]
                )
                if self.is_lvis:
                    stats["metrics/APr(B)"] = val.results["APr"]
                    stats["metrics/APc(B)"] = val.results["APc"]
                    stats["metrics/APf(B)"] = val.results["APf"]
                    stats["fitness"] = val.results["AP"]
            except Exception as e:
                LOGGER.warning(f"{pkg} unable to run: {e}")
        return stats



