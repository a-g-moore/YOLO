import torch
import torch.nn as nn

class YoloLoss(nn.Module):
  def __init__(self, num_classes=20):
    super(YoloLoss, self).__init__()
    self.mse = nn.MSELoss(reduction="sum")
    self.num_classes = num_classes
    self.lambda_noobj = 0.5
    self.lambda_coord = 5

  def forward(self, predictions, target):
    predictions = predictions.reshape(-1, 7, 7, self.num_classes + 5)
    box_exists = target[..., self.num_classes].unsqueeze(3) 


    # box loss
    box_pred = box_exists * predictions[..., (self.num_classes+1):(self.num_classes+5)]
    box_targ = box_exists * target[..., (self.num_classes+1):(self.num_classes+5)]


    box_loc_loss = self.mse(
        torch.flatten(box_pred[..., 0:2], end_dim = -2),
        torch.flatten(box_targ[..., 0:2], end_dim = -2)
    )

    box_pred = torch.sign(box_pred[..., 2:4]) * torch.sqrt(torch.abs(box_pred[..., 2:4]) + 1e-6)
    box_targ = torch.sqrt(torch.abs(box_targ[..., 2:4] + 1e-6))

    box_size_loss = self.mse(
        torch.flatten(box_pred),
        torch.flatten(box_targ)
    )

    # object loss
    object_loss = self.mse(
        torch.flatten(box_exists * predictions[..., self.num_classes:(self.num_classes+1)]),
        torch.flatten(box_exists * target[..., self.num_classes:(self.num_classes+1)])
    )

    # no object loss
    no_object_loss = self.lambda_noobj * self.mse(
        torch.flatten((1-box_exists) * predictions[..., self.num_classes:(self.num_classes+1)]),
        torch.flatten((1-box_exists) * target[..., self.num_classes:(self.num_classes+1)])
    )

    # class loss
    class_loss = self.mse(
        torch.flatten(predictions[..., :self.num_classes]),
        torch.flatten(target[..., :self.num_classes])
    )

    loss = self.lambda_coord * (box_loc_loss + box_size_loss) + object_loss + self.lambda_noobj * no_object_loss + class_loss

    return loss


