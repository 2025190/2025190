import ultralytics.engine.results
import ultralytics.utils.ops

def init(self, boxes, orig_shape) -> None:
    if boxes.ndim == 1:
        boxes = boxes[None, :]
    n = boxes.shape[-1]
    super(ultralytics.engine.results.Boxes, self).__init__(boxes, orig_shape)
    self.orig_shape = orig_shape
    self.num_classes = 0

    if n == 6:
        self.format = 'xyxy_conf_cls'
    elif n == 7:
        self.format = 'xyxy_conf_cls_track'
        self.is_track = True
    else:
        self.format = 'xyxy_conf_cls_classconf'
        self.num_classes = n - 6

ultralytics.engine.results.Boxes.__init__ = init

from ultralytics.utils.ops import xywh2xyxy, LOGGER, nms_rotated
import torch
import time

def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
    rotated=False,
    end2end=False,
    return_idxs=False,
):
    
    import torchvision
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    if prediction.shape[-1] == 6 or end2end:
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output
    
    bs = prediction.shape[0]
    nc = nc or (prediction.shape[1] - 4)
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc
    xc = prediction[:, 4:mi].amax(1) > conf_thres
    xinds = torch.stack([torch.arange(len(i), device=prediction.device) for i in xc])[..., None]
    time_limit = 2.0 + max_time_img * bs
    multi_label &= nc > 1

    prediction = prediction.transpose(-1, -2)
    if not rotated:
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])
        else:
            prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)

    t = time.time()
    output = [torch.zeros((0, 6 + nc + nm), device=prediction.device)] * bs
    keepi = [torch.zeros((0, 1), device=prediction.device)] * bs
    for xi, (x, xk) in enumerate(zip(prediction, xinds)):
        filt = xc[xi]
        x, xk = x[filt], xk[filt]
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0
            x = torch.cat((x, v), 0)
        if not x.shape[0]:
            continue
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            xk = xk[i]
        else:
            conf, j = cls.max(1, keepdim=True)
            filt = conf.view(-1) > conf_thres
            x = torch.cat((box, conf, j.float(), cls, mask), 1)[filt]
            xk = xk[filt]
        if classes is not None:
            filt = (x[:, 5:6] == classes).any(1)
            x, xk = x[filt], xk[filt]
        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            filt = x[:, 4].argsort(descending=True)[:max_nms]
            x, xk = x[filt], xk[filt]
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        scores = x[:, 4]
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)
            i = nms_rotated(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c
            i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_det]
        output[xi], keepi[xi] = x[i], xk[i].reshape(-1)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break
    return (output, keepi) if return_idxs else output
ultralytics.utils.ops.non_max_suppression = non_max_suppression
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import ops
from ultralytics.engine.results import Results
original_construct_result = DetectionPredictor.construct_result
def patched_construct_result(self, pred, img, orig_img, img_path):
    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
    return Results(orig_img, path=img_path, names=self.model.names, boxes=pred)
DetectionPredictor.construct_result = patched_construct_result



@property
def conf(self):
    return self.data[:, 4]

@property
def cls(self):
    return self.data[:, 5]
ultralytics.engine.results.Boxes.conf = conf
ultralytics.engine.results.Boxes.cls = cls