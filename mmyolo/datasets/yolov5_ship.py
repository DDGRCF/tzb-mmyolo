# Copyright (c) OpenMMLab. All rights reserved.
from .tzb_ship import TzbShipDataset

from .yolov5_coco import BatchShapePolicyDataset
from ..registry import DATASETS


@DATASETS.register_module()
class YOLOv5ShipDataset(BatchShapePolicyDataset, TzbShipDataset):
    """Dataset for YOLOv5 tzb-ship Dataset.

    We only add `BatchShapePolicy` function compared with CocoDataset. See
    `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """
    pass
