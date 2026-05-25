import cv2
import torch
from torchvision.ops import nms

# Hardcoded routing for yolo26 models that don't carry layer.f attributes
_SAVE_YOLO26 = {4, 6, 10, 13, 16, 19, 22}
_INPUT_YOLO26 = [
    None,None,None,None,None,None,None,None,None,None,None,None,
    [-1,6],None,None,[-1,4],None,None,[-1,13],None,None,[-1,10],None,[16,19,22]
]


def get_save_set(full_layers):
    """
    Derive which global layer indices must be saved for skip connections.
    Returns a set of indices, or None if the model lacks layer.f attributes
    (in which case the hardcoded yolo26 fallback will be used).
    """
    save = set()
    for layer in full_layers:
        f = getattr(layer, 'f', None)
        if f is None:
            return None  # model doesn't carry f — caller should use yolo26 fallback
        if isinstance(f, (list, tuple)):
            for fi in f:
                if fi != -1:
                    save.add(fi)
        elif isinstance(f, int) and f != -1:
            save.add(f)
    return save


def inference(model, x, y, cut, save_set=None):
    """
    Run model layers starting at global index `cut`.
    save_set: set of global indices to save (from get_save_set).
              None → fall back to hardcoded yolo26 routing.
    """
    for i, layer in enumerate(model):
        idx = i + cut
        f = getattr(layer, 'f', None)

        if f is not None:
            # Ultralytics-style routing (yolo11, yolo26 built with ultralytics)
            if isinstance(f, (list, tuple)):
                x = [x if fi == -1 else y[fi] for fi in f]
            elif isinstance(f, int) and f != -1:
                x = y[f]
        elif idx < len(_INPUT_YOLO26) and _INPUT_YOLO26[idx] is not None:
            # Hardcoded yolo26 fallback
            r = _INPUT_YOLO26[idx]
            inputs = [x if r[0] == -1 else y[r[0]]]
            inputs += [y[r[j]] for j in range(1, len(r))]
            x = inputs

        x = layer(x)
        if isinstance(x, tuple):
            x = x[0]

        effective_save = save_set if save_set is not None else _SAVE_YOLO26
        y.append(x if idx in effective_save else None)

    return x, y

def postprocess_yolo(output, conf_thres=0.25, iou_thres=0.5):
    # yolo11/ultralytics: Detect trả về (pred[B,4+nc,N], features)
    # yolo26/custom:      Detect trả về tensor [B,N,6]
    if isinstance(output, (tuple, list)):
        output = output[0]  # lấy pred tensor từ tuple

    batch_results = []
    B = output.shape[0]

    # Phân biệt format: yolo26=[B,N,6], yolo11=[B,4+nc,N] (dim 1 > dim 2 usually)
    if output.dim() == 3 and output.shape[1] != output.shape[2] and output.shape[2] == 6:
        # yolo26 format: [B, N, 6] — (x1,y1,x2,y2, conf, cls_id) xyxy pixel
        for b in range(B):
            pred = output[b]           # [N, 6]
            boxes  = pred[:, :4]
            scores = pred[:, 4]
            classes = pred[:, 5].long()

            mask = scores > conf_thres
            boxes, scores, classes = boxes[mask], scores[mask], classes[mask]

            if len(boxes):
                keep = nms(boxes, scores, iou_thres)
                boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

            batch_results.append({"boxes": boxes, "scores": scores, "classes": classes})
    else:
        # yolo11 format: [B, 4+nc, N] — box=(cx,cy,w,h) pixel, rest=class probs
        pred_t = output.permute(0, 2, 1)   # [B, N, 4+nc]
        for b in range(B):
            pred = pred_t[b]               # [N, 4+nc]
            boxes_xywh  = pred[:, :4]
            class_probs = pred[:, 4:]
            scores, classes = class_probs.max(dim=1)

            mask = scores > conf_thres
            boxes_xywh, scores, classes = boxes_xywh[mask], scores[mask], classes[mask]

            # xywh → xyxy
            x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
            y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
            x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
            y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
            boxes = torch.stack([x1, y1, x2, y2], dim=1)

            if len(boxes):
                keep = nms(boxes, scores, iou_thres)
                boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

            batch_results.append({"boxes": boxes, "scores": scores, "classes": classes})

    return batch_results

def draw_img(img, r):
    for box, score, cls in zip(r["boxes"], r["scores"], r["classes"]):
        x1, y1, x2, y2 = box.int().tolist()

        conf = score.item()
        cls_id = cls.item()

        label = f"{cls_id}:{conf:.2f}"

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            img,
            label,
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return img

