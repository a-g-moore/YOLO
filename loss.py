import torch
import torch.nn as nn

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
    x2 = torch.max(box1x2, box2x2)
    y2 = torch.max(box1y2, box2y2)

    intersection_area = torch.clamp(x2 - x1, min = 0) * torch.clamp(y2 - y1, min = 0)

    iou = intersection_area / (box1area + box2area - intersection_area + 1e-6)
    return iou.unsqeeze(3)

class YoloLoss(nn.Module):
  def __init__(self, num_classes=20, num_boxes = 2):
    super(YoloLoss, self).__init__()
    self.mse = nn.MSELoss(reduction="sum")
    self.num_classes = num_classes
    self.num_boxes = num_boxes
    self.lambda_noobj = 0.5
    self.lambda_coord = 5

  def forward(self, predictions, target):
    predictions = predictions.reshape(-1, 7, 7, self.num_classes + 5 * self.num_boxes)
    class_predictions = predictions[..., :self.num_classes]
    box_predictions = predictions[..., self.num_classes:]

    class_target = target[..., :self.num_classes]
    box_target = target[..., self.num_classes:]
    box_exists = box_target[..., 0].unsqueeze(3) 

    iou = torch.cat(
        [
            intersection_over_union(
                box_predictions[..., (i*5 + 1):(i*5 + 5)], 
                box_target[..., 1:]
                ).unsqueeze(0)
            for i in range(self.num_boxes)
        ],
        dim = 0
        )

    best_iou, best_box = torch.max(iou, dim = 0)
    
    ## TODO: Make this work for something other than B=2
    predicted_box = (
        (1-best_box) * box_predictions[1:5]
        + best_box * box_predictions[6:10]
    )
    target_box = box_target[..., 1:]
    predicted_confidence = (
        (1-best_box) * box_predictions[0:1]
        + best_box * box_predictions[5:6]
    )
    target_confidence = best_iou


    box_loc_loss = self.lambda_coord * self.mse(
        torch.flatten(box_exists * predicted_box[..., 0:2]),
        torch.flatten(box_exists * target_box[..., 0:2])
    )

    predicted_box = torch.sign(predicted_box[..., 2:4]) * torch.sqrt(torch.abs(predicted_box[..., 2:4]) + 1e-6)
    target_box = torch.sqrt(torch.abs(target_box[..., 2:4] + 1e-6))

    box_size_loss = self.lambda_coord * self.mse(
        torch.flatten(box_exists * predicted_box),
        torch.flatten(box_exists * target_box)
    )

    # object loss
    object_loss = self.mse(
        torch.flatten(box_exists * predicted_confidence),
        torch.flatten(box_exists * target_confidence)
    )

    # no object loss
    no_object_loss = self.lambda_noobj * self.mse(
        torch.flatten((1-box_exists) * predicted_confidence),
        torch.zeros_like(box_exists)
    )

    # class loss
    class_loss = self.mse(
        torch.flatten(box_exists * class_predictions),
        torch.flatten(box_exists * class_target)
    )

    loss = box_loc_loss + box_size_loss + object_loss + no_object_loss + class_loss

    return loss


