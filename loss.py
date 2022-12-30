import torch
import torch.nn as nn

def mean_average_precision(predictions, targets, min_iou = 0.5):
    """Calculates mean average precision for evaluating the YOLO model's accuracy.
    Arguments are the lists of predicted and target boxes in dict format.

    """

    evaluations = []
    num_target_boxes = 0
    for prediction, target in zip(predictions, targets):
        num_target_boxes += len(target['bboxes'])
        for bbox, confidence in zip(prediction['bboxes'], prediction['confidences']):
            max_iou = max([intersection_over_union(bbox, target_bbox) for target_bbox in target['bboxes']])
            evaluations.append({
                "confidence": confidence,
                "positive": max_iou >= min_iou
            })

    evaluations = sorted(evaluations, key = lambda item: item['confidence'])


def intersection_over_union(box1, box2):
    box1x1 = box1[..., 0] - box1[..., 2] / 2
    box1y1 = box1[..., 1] - box1[..., 3] / 2
    box1x2 = box1[..., 0] + box1[..., 2] / 2
    box1y2 = box1[..., 1] + box1[..., 3] / 2
    box2x1 = box2[..., 0] - box2[..., 2] / 2
    box2y1 = box2[..., 1] - box2[..., 3] / 2
    box2x2 = box2[..., 0] + box2[..., 2] / 2
    box2y2 = box2[..., 1] + box2[..., 3] / 2

    box1area = torch.abs((box1x1 - box1x2) * (box1y1 - box1y2))
    box2area = torch.abs((box2x1 - box2x2) * (box2y1 - box2y2))

    x1 = torch.max(box1x1, box2x1)
    y1 = torch.max(box1y1, box2y1)
    x2 = torch.min(box1x2, box2x2)
    y2 = torch.min(box1y2, box2y2)

    intersection_area = torch.clamp(x2 - x1, min = 0) * torch.clamp(y2 - y1, min = 0)

    iou = intersection_area / (box1area + box2area - intersection_area + 1e-6)
    return iou.unsqueeze(3)

class YoloLoss(nn.Module):
    def __init__(self, num_classes=20, num_boxes = 2):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
    
    def forward(self, predictions, target):
        batch_size = target.size()[0]

        predictions = predictions.reshape(-1, 7, 7, self.num_classes + 5 * self.num_boxes)
        class_predictions = predictions[..., :self.num_classes]

        class_target = target[..., :self.num_classes]
        indicator_i = target[..., self.num_classes].unsqueeze(3)

        # class loss
        class_loss = self.mse(
            indicator_i * class_predictions, 
            indicator_i * class_target
        )

        box_predictions = predictions[..., self.num_classes:].reshape(-1, 7, 7, self.num_boxes, 5)

        box_target = target[..., self.num_classes:]
        box_target = torch.cat((box_target, box_target), dim=3).reshape(-1, 7, 7, self.num_boxes, 5)
        
        iou = torch.cat(
            [
                intersection_over_union(
                    box_predictions[..., i, 1:], 
                    box_target[..., i, 1:]
                    ).unsqueeze(0)
                for i in range(self.num_boxes)
            ],
            dim = 0
            )

        best_iou, best_box = torch.max(iou, dim = 0)

        first_box_mask = torch.cat((torch.ones_like(indicator_i), torch.zeros_like(indicator_i)), dim=3)
        second_box_mask = torch.cat((torch.zeros_like(indicator_i), torch.ones_like(indicator_i)), dim=3)

        indicator_ij = (indicator_i * ((1-best_box) * first_box_mask + best_box * second_box_mask)).unsqueeze(4)

        box_target[..., 0] = torch.cat((best_iou, best_iou), dim=3)
        box_target = indicator_ij * box_target
        
        # localization loss
        xy_loss = self.lambda_coord * self.mse(
            indicator_ij * box_predictions[..., 1:3],
            indicator_ij * box_target[..., 1:3]
        )

        wh_loss = self.lambda_coord * self.mse(
            indicator_ij * torch.sign(box_predictions[..., 3:5]) * torch.sqrt(torch.abs(box_predictions[..., 3:5]) + 1e-6),
            indicator_ij * torch.sign(box_target[..., 3:5]) * torch.sqrt(torch.abs(box_target[..., 3:5]) + 1e-6)
        )

        # object loss
        object_loss = self.mse(
            indicator_ij * box_predictions[..., 0:1],
            indicator_ij * box_target[..., 0:1]
        )

        # no object loss
        no_object_loss = self.lambda_noobj * self.mse(
            (1-indicator_ij) * box_predictions[..., 0:1],
            (1-indicator_ij) * box_target[..., 0:1]
        )

        return (xy_loss + wh_loss + object_loss + no_object_loss + class_loss) / float(batch_size)