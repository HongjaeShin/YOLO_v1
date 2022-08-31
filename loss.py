import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # 7*7*30의 크기로 변환하는 목적인 것 같은데 -1은 왜 있는거지? 더 큰게 와도 맞춰서 한다는 의미인가?
        predictions = predictions.reshape(-1,
                                          self.S, self.S, self.C + self.B*5)
        # 0 ~ 19 Class probabilities, 20 Class score, 21~24 first box, 25 Class Score, 26~29 second box
        iou_b1 = intersection_over_union(
            predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(
            predictions[..., 26:30], target[..., 21:25])
        # 두 개의 바운딩 박스의 좌표의 IoU를 계산하여 IoU가 더 큰 박스를 구한다.
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, bestbox = torch.max(ious, dim=0)
        # target 의 shape은 어떻게 되지?
        exists_box = target[..., 20].unsqueeze(3)

        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 26:30]
                + (1 - bestbox) * predictions[..., 21:25]
            )
        )

        box_targets = exists_box * target[..., 21:25]
